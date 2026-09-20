#!/usr/bin/env python3
"""Build the private Split-B SkyRL inputs and a sanitized staging packet.

This command is deliberately local and GET-only.  Fleet task prompts remain in
``--private-root`` outside the repository.  The only repository output is a
digest inventory which cannot be used to recover those prompts.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx

from cyber_post_train.jobs import digest
from evals.fleet import opencode_self_hosted as fleet
from training import rl_data
from training.corpus import local_tokenizer
from training.sft import read_mapping

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SKYRL_REVISION = "f5bc3b78dfddfb352870d5d7430cd226e5785838"
EXPECTED_DATASET_SHA256 = "ff041e24a24e7d99c9c20052ae643b137acb0f1777016ddba79260a42226466c"
EXPECTED_MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
DATASET_MODULE = Path("skyrl/train/dataset/dataset.py")
PACKET = ROOT / "configs/qualification/qwen38-skyrl-production-data-staging-v1.json"
CONFIGS = (
    ROOT / "configs/qualification/qwen38-skyrl-production-data-a1-v1.json",
    ROOT / "configs/qualification/qwen38-skyrl-production-data-lr3e7-v1.json",
    ROOT / "configs/qualification/qwen38-skyrl-production-data-lr3e6-v1.json",
    ROOT / "configs/qualification/qwen38-skyrl-production-data-seed43-v1.json",
    ROOT / "configs/qualification/qwen38-skyrl-production-data-dose50-v1.json",
)
PRIVATE_FILES = {"train.jsonl", "dev.jsonl", "split.json", "task-set.json", "manifest.json"}


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return "sha256:" + value.hexdigest()


def _git_revision(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def validate_native_inputs(skyrl_source: Path, tokenizer_root: Path, lock: dict) -> None:
    source = skyrl_source / DATASET_MODULE
    if (
        _git_revision(skyrl_source) != EXPECTED_SKYRL_REVISION
        or file_sha256(source) != "sha256:" + EXPECTED_DATASET_SHA256
    ):
        raise ValueError("local SkyRL source differs from the reviewed production revision")
    if lock.get("revision") != EXPECTED_MODEL_REVISION:
        raise ValueError("model lock differs from the production queue")
    local_tokenizer(lock, tokenizer_root)


class CachedGetClient:
    """One authenticated upstream GET per unique URL, then in-memory replay."""

    follow_redirects = False

    def __init__(self, token: str) -> None:
        if not token:
            raise ValueError("FLEET_API_KEY is required")
        self._client = httpx.Client(
            headers={"Authorization": "Bearer " + token},
            timeout=60,
            follow_redirects=False,
            transport=httpx.HTTPTransport(retries=0),
        )
        self._cache: dict[str, tuple[int, str, bytes]] = {}
        self.upstream_gets: list[str] = []

    def __enter__(self) -> CachedGetClient:
        return self

    def __exit__(self, *args: object) -> None:
        self._client.close()

    def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        if method != "GET" or set(kwargs) - {"params"}:
            raise ValueError("private data preparation permits Fleet GET requests only")
        request = self._client.build_request(method, url, params=kwargs.get("params"))
        key = str(request.url)
        cached = self._cache.get(key)
        if cached is None:
            response = self._client.send(request)
            body = response.read()
            cached = (response.status_code, response.headers.get("content-type", ""), body)
            if response.status_code < 400:
                self._cache[key] = cached
            self.upstream_gets.append(request.url.path)
        status, content_type, body = cached
        headers = {"content-type": content_type} if content_type else {}
        return httpx.Response(status, headers=headers, content=body, request=request)


@contextlib.contextmanager
def native_skyrl_from(skyrl_source: Path, tokenizer_root: Path):
    """Use the exact reviewed loader source while keeping model bytes local."""
    original = rl_data._native_skyrl
    sys.path.insert(0, str(skyrl_source))

    def load(lock: dict, _canonical_model_root: str):
        return original(lock, tokenizer_root)

    rl_data._native_skyrl = load
    try:
        yield
    finally:
        rl_data._native_skyrl = original
        sys.path.remove(str(skyrl_source))


def _rows(path: Path) -> list[dict]:
    values = []
    with path.open() as handle:
        for line in handle:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("private SkyRL row is not an object")
            values.append(value)
    return values


def _safe_binding_digest(rows: list[dict], expected_split: str) -> str:
    summaries = []
    for row in rows:
        if row.get("split") != expected_split:
            raise ValueError("private row crossed its frozen split")
        config = json.loads(row["cyber_config_json"])
        if config.get("config_sha256") != fleet.digest_without(config, "config_sha256"):
            raise ValueError("private row binding digest differs")
        if config["execution"] != {
            "required_task_tools": ["bash", "submit_report"],
            "required_task_tool_catalog_sha256": config["execution"][
                "required_task_tool_catalog_sha256"
            ],
        }:
            raise ValueError("private row tool surface differs")
        summaries.append(
            {
                "task": config["task"],
                "environment": config["environment"],
                "verifier": config["verifier"],
                "tools": config["execution"],
                "config_sha256": config["config_sha256"],
            }
        )
    return fleet.sha256(canonical(summaries))


def inspect_private_package(
    root: Path, config_path: Path, config: dict, task_set: dict, split: dict
) -> dict:
    if {path.name for path in root.iterdir()} != PRIVATE_FILES:
        raise ValueError("private package file set differs")
    if root.stat().st_mode & 0o777 != 0o700:
        raise ValueError("private package directory must be mode 0700")
    for path in root.iterdir():
        if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o777 != 0o600:
            raise ValueError("private package files must be regular mode-0600 files")
    manifest = read_mapping(root / "manifest.json")
    if (
        manifest.get("schema") != "cyber_skyrl_data_v1"
        or manifest.get("sha256") != "sha256:" + digest(
            {key: value for key, value in manifest.items() if key != "sha256"}
        )
        or manifest.get("selection_sha256") != task_set["sha256"]
        or manifest.get("split_sha256") != split["sha256"]
        or manifest.get("name") != config["name"]
        or manifest.get("limits") != config["limits"]
        or (manifest.get("tokenizer") or {}).get("revision") != EXPECTED_MODEL_REVISION
        or read_mapping(root / "task-set.json") != task_set
        or read_mapping(root / "split.json") != split
    ):
        raise ValueError("private data manifest does not match the sealed queue")
    train, dev = _rows(root / "train.jsonl"), _rows(root / "dev.jsonl")
    if len(train) != 59 or len(dev) != 20:
        raise ValueError("private data must contain exactly 59 train and 20 development rows")
    for group, rows in (("train", train), ("dev", dev)):
        payload = (root / f"{group}.jsonl").read_bytes()
        spec = (manifest.get("files") or {}).get(group)
        if (
            spec is None
            or spec.get("path") != f"{group}.jsonl"
            or spec.get("rows") != len(rows)
            or spec.get("sha256") != fleet.sha256(payload)
            or not isinstance(spec.get("max_prompt_tokens"), int)
            or spec["max_prompt_tokens"] <= 0
        ):
            raise ValueError("private payload does not match its data manifest")
    tests = {
        (row["task_key"], row["task_version_id"])
        for row in split["tasks"]
        if row["split"] == "test"
    }
    expected = {
        (row["task_key"], row["task_version_id"]): row["split"]
        for row in split["tasks"]
        if row["split"] in {"train", "dev"}
    }
    exact_runtime = {
        (row["task_key"], row["task_version_id"]): row for row in task_set["tasks"]
    }
    payload_tasks = set()
    for row in train + dev:
        config_json = json.loads(row["cyber_config_json"])
        bound, environment = config_json["task"], config_json["environment"]
        key = (bound["key"], bound["version_id"])
        source = exact_runtime.get(key)
        if (
            source is None
            or expected.get(key) != row["split"]
            or environment
            != {
                "id": source["env_key"],
                "version": source["env_version"],
                "version_id": source["environment_version_id"],
                "data_id": source["data_key"],
                "data_version": source["data_version"],
                "runtime_seed_content_sha256": environment["runtime_seed_content_sha256"],
                "ttl_seconds": 32400,
            }
        ):
            raise ValueError("private row no longer matches its exact Fleet runtime tuple")
        payload_tasks.add(key)
    if len(tests) != 10 or tests & payload_tasks or payload_tasks != set(expected):
        raise ValueError("the ten frozen final-test tasks must remain absent from RL inputs")
    files = []
    for name in sorted(PRIVATE_FILES):
        path = root / name
        item = {
            "path": name,
            "bytes": path.stat().st_size,
            "mode": "0600",
            "sha256": file_sha256(path),
        }
        if name == "train.jsonl":
            item["rows"] = 59
        elif name == "dev.jsonl":
            item["rows"] = 20
        files.append(item)
    return {
        "name": config["name"],
        "source_config": str(config_path.relative_to(ROOT)),
        "source_config_sha256": file_sha256(config_path),
        "private_source": str(root),
        "create_once_target": config["output"],
        "data_manifest_sha256": manifest["sha256"],
        "data_manifest_file_sha256": file_sha256(root / "manifest.json"),
        # The SkyRL manifest contains only public model identity, limits, row
        # counts, and payload digests.  Retaining it in the sanitized packet
        # lets the offline compiler bind an exact candidate without copying a
        # prompt or any other private row bytes into Git.
        "sanitized_manifest": manifest,
        "binding_aggregate_sha256": {
            "train": _safe_binding_digest(train, "train"),
            "dev": _safe_binding_digest(dev, "dev"),
        },
        "rows": {"train": 59, "dev": 20, "final_test": 0},
        "files": files,
    }


def staging_packet(
    arms: list[dict], task_set: dict, split: dict, upstream_gets: list[str]
) -> dict:
    account_gets = sum(path == "/v1/account" for path in upstream_gets)
    task_gets = sum(path.startswith("/v1/tasks/") for path in upstream_gets)
    if account_gets != 1 or task_gets != 79 or len(upstream_gets) != 80:
        raise ValueError("expected one account GET and 79 exact-version task GETs")
    packet = {
        "schema": "cyber_qwen38_skyrl_production_data_staging_v1",
        "state": "locally_built_not_staged",
        "external_mutations": 0,
        "bindings": {
            "model_revision": EXPECTED_MODEL_REVISION,
            "task_set_sha256": task_set["sha256"],
            "split_sha256": split["sha256"],
            "tool_catalog_sha256": task_set["tool_catalog_sha256"],
            "rows": {"train": 59, "dev": 20, "final_test_untouched": 10},
        },
        "fleet_reads": {
            "methods": ["GET"],
            "account_gets": account_gets,
            "unique_exact_version_task_gets": task_gets,
            "mutating_requests": 0,
        },
        "runtime_ownership": {
            "uid": 1000,
            "gid": 100,
            "directory_mode": "0700",
            "file_mode": "0600",
            "required_checks": [
                "copy_to_a_new_absent_target_without_replace",
                "set_owner_before_runtime_readback",
                "rehash_every_destination_file_as_uid_1000_gid_100",
                "reject_any_extra_or_missing_file",
            ],
        },
        "privacy": {
            "private_bytes_location": "outside_git",
            "repository_payload_bytes": 0,
            "repository_contains_only_counts_digests_and_public_bindings": True,
        },
        "arms": arms,
        "remaining_gates": [
            "observe_each_create_once_target_absent_immediately_before_copy",
            "perform_separately_authorized_cpu_only_create_once_copy",
            "verify_destination_owner_modes_file_set_and_digests_as_uid_1000_gid_100",
            "accept_prod4_canary_before_compiling_or_submitting_any_production_arm",
        ],
    }
    packet["sha256"] = "sha256:" + digest(packet)
    return packet


def _write_packet(path: Path, packet: dict) -> None:
    payload = canonical(packet) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def verify_existing(private_root: Path, packet_path: Path) -> dict:
    packet = read_mapping(packet_path)
    if (
        packet.get("schema") != "cyber_qwen38_skyrl_production_data_staging_v1"
        or packet.get("sha256")
        != "sha256:" + digest({key: value for key, value in packet.items() if key != "sha256"})
    ):
        raise ValueError("staging packet digest differs")
    configs = [(path, read_mapping(path)) for path in CONFIGS]
    task_set = read_mapping(configs[0][0].parent / configs[0][1]["task_set"])
    split = read_mapping(configs[0][0].parent / configs[0][1]["split"])
    arms = [
        inspect_private_package(
            private_root / config["name"] / "data",
            config_path,
            config,
            task_set,
            split,
        )
        for config_path, config in configs
    ]
    if packet.get("arms") != arms or packet.get("external_mutations") != 0:
        raise ValueError("private package inventory differs from the staging packet")
    return packet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--skyrl-source", type=Path)
    parser.add_argument("--tokenizer-root", type=Path)
    parser.add_argument("--packet", type=Path, default=PACKET)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    private_root = args.private_root.resolve()
    if private_root == ROOT or ROOT in private_root.parents:
        raise ValueError("private RL payloads must be written outside Git")
    if args.verify_existing:
        packet = verify_existing(private_root, args.packet.resolve())
        print(json.dumps({"arms": len(packet["arms"]), "sha256": packet["sha256"]}, sort_keys=True))
        return
    if private_root.exists() or private_root.is_symlink():
        raise FileExistsError("private data root is create-once and already exists")
    if args.skyrl_source is None or args.tokenizer_root is None:
        raise ValueError("new builds require --skyrl-source and --tokenizer-root")
    configs = [(path, read_mapping(path)) for path in CONFIGS]
    task_set = read_mapping(configs[0][0].parent / configs[0][1]["task_set"])
    split = read_mapping(configs[0][0].parent / configs[0][1]["split"])
    selected = rl_data.selection(task_set, split)
    if sum(row["split"] == "train" for row in selected) != 59 or sum(
        row["split"] == "dev" for row in selected
    ) != 20:
        raise ValueError("production queue no longer selects Split B's 59/20 rows")
    lock = read_mapping(configs[0][0].parent / configs[0][1]["model_lock"])
    validate_native_inputs(args.skyrl_source.resolve(), args.tokenizer_root.resolve(), lock)
    private_root.mkdir(parents=True, mode=0o700)
    token = os.environ.get("FLEET_API_KEY", "")
    arms = []
    try:
        with (
            CachedGetClient(token) as client,
            native_skyrl_from(args.skyrl_source.resolve(), args.tokenizer_root.resolve()),
        ):
            for config_path, original in configs:
                config = dict(original)
                destination = private_root / config["name"] / "data"
                config["output"] = str(destination)
                result = rl_data.build(config, relative_to=config_path.parent, client=client)
                if result.get("submitted") is not False:
                    raise ValueError("local data builder unexpectedly submitted work")
                arms.append(
                    inspect_private_package(destination, config_path, original, task_set, split)
                )
            packet = staging_packet(arms, task_set, split, client.upstream_gets)
    except Exception:
        # Preserve private failure bytes for diagnosis.  Never publish a partial packet.
        raise
    _write_packet(args.packet.resolve(), packet)
    print(
        json.dumps(
            {
                "arms": len(arms),
                "train_rows_per_arm": 59,
                "dev_rows_per_arm": 20,
                "final_test_rows": 0,
                "mutating_requests": 0,
                "packet_sha256": packet["sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
