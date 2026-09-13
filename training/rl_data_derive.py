"""Create a policy-specific Miles dataset without rebuilding task prompts.

The source is an already accepted ``cyber_miles_data_v1`` artifact.  This
module reopens every byte and every sealed task/config identity, changes only
the run id and, when explicitly requested, policy-root metadata, and publishes
a new create-once artifact.  It performs no network requests and never returns
or logs row payloads.
"""

from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path

from cyber_post_train.jobs import digest
from evals.fleet import opencode_self_hosted as fleet

from . import miles, rl_episode
from .rl_data import AUTHORITY, _sealed, selection
from .sft import _known, _sfs_root

DERIVATION_SCHEMA = "cyber_miles_data_derivation_v1"
_MANIFEST_FIELDS = {
    "schema",
    "name",
    "selection_sha256",
    "split_sha256",
    "tokenizer",
    "template_sha256",
    "tool_catalog_sha256",
    "limits",
    "files",
    "gpus",
    "environment_creates",
    "sha256",
}
_POLICY_TRANSFORMED_FIELDS = [
    "metadata.cyber_config.run_id",
    "metadata.cyber_config.model.root",
    "metadata.cyber_config.config_sha256",
]
_RUN_TRANSFORMED_FIELDS = [
    "metadata.cyber_config.run_id",
    "metadata.cyber_config.config_sha256",
]


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"sha256:[a-f0-9]{64}", value) is None:
        raise ValueError(f"{label} must be a prefixed SHA-256 digest")
    return value


def _stable_bytes(path: Path, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is missing or indirect")
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    if any(
        getattr(before, key) != getattr(after, key)
        for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    ):
        raise ValueError(f"{label} changed while it was read")
    return payload


def _json_object(payload: bytes, label: str) -> dict:
    try:
        value = json.loads(payload)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _write_once(path: Path, payload: bytes) -> None:
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _rows(payload: bytes, *, split: str, manifest: dict, source_root: str) -> list[dict]:
    if not payload.endswith(b"\n"):
        raise ValueError("source Miles JSONL must end with one complete row")
    lines = payload.splitlines()
    if not lines:
        raise ValueError("source Miles split is empty")
    values = []
    expected_limits = {k: v for k, v in manifest["limits"].items() if k != "response_tokens"}
    for line in lines:
        row = _json_object(line, "source Miles row")
        if fleet.canonical_json(row) != line or set(row) != {"input", "metadata"}:
            raise ValueError("source Miles row is noncanonical or has unknown fields")
        metadata = row["metadata"]
        if not isinstance(metadata, dict) or set(metadata) != {
            "split",
            "lineage",
            "cyber_config",
        }:
            raise ValueError("source Miles row metadata changed")
        prompt, config = row["input"], metadata["cyber_config"]
        if not isinstance(prompt, str) or not prompt or not isinstance(config, dict):
            raise ValueError("source Miles prompt/config is invalid")
        rl_episode._validate(config)
        model = config.get("model")
        execution = config.get("execution")
        if (
            metadata["split"] != split
            or config.get("run_id") != manifest["name"]
            or not isinstance(model, dict)
            or set(model)
            != {
                "repo",
                "revision",
                "root",
                "tito_family",
                "runtime_chat_template_sha256",
            }
            or model["repo"] != manifest["tokenizer"]["repo"]
            or model["revision"] != manifest["tokenizer"]["revision"]
            or model["root"] != source_root
            or model["tito_family"] != "qwen35"
            or model["runtime_chat_template_sha256"] != manifest["template_sha256"]
            or config.get("authority") != AUTHORITY
            or execution
            != {
                "required_task_tools": ["bash", "submit_report"],
                "required_task_tool_catalog_sha256": manifest["tool_catalog_sha256"],
            }
            or config.get("rl") != expected_limits
            or config.get("initial_prompt_sha256") != fleet.sha256(prompt.encode())
        ):
            raise ValueError("source Miles row identity changed")
        values.append(row)
    return values


def derive(config: dict, *, relative_to: Path) -> dict:
    """Derive one run/policy-specific data artifact; return sanitized digests only."""
    _known(
        config,
        {
            "source_manifest",
            "source_manifest_file_sha256",
            "source_manifest_sha256",
            "source_policy_identity_root",
            "expected_limits",
            "mode",
            "name",
            "policy_identity_root",
            "output",
        },
        "RL data derivation",
    )
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,30}", config["name"]):
        raise ValueError("invalid derived RL run name")
    source_policy_root = _sfs_root(
        config["source_policy_identity_root"], "source policy identity root"
    )
    policy_root = _sfs_root(config["policy_identity_root"], "policy identity root")
    mode = config.get("mode", "policy_identity_rebind")
    if mode not in {"policy_identity_rebind", "run_id_rebind"}:
        raise ValueError("invalid RL data derivation mode")
    if mode == "policy_identity_rebind" and policy_root == source_policy_root:
        raise ValueError("derived policy identity must differ from the base policy")
    if mode == "run_id_rebind" and policy_root != source_policy_root:
        raise ValueError("run-id-only derivation must preserve policy identity")

    declared = Path(config["source_manifest"])
    source_path = declared if declared.is_absolute() else relative_to / declared
    source_path = Path(os.path.abspath(source_path))
    if source_path.name != "manifest.json" or source_path.parent.is_symlink():
        raise ValueError("source must be a direct, non-symlinked manifest.json")
    source_payload = _stable_bytes(source_path, "source manifest")
    if fleet.sha256(source_payload) != _sha(
        config["source_manifest_file_sha256"], "source manifest file digest"
    ):
        raise ValueError("source manifest file digest mismatch")
    manifest = _json_object(source_payload, "source manifest")
    if set(manifest) != _MANIFEST_FIELDS:
        raise ValueError("source must be an original accepted Miles data manifest")
    _sealed(manifest, "cyber_miles_data_v1")
    if manifest["sha256"] != _sha(config["source_manifest_sha256"], "source manifest self digest"):
        raise ValueError("source manifest self digest mismatch")
    if (
        manifest["name"] == config["name"]
        or manifest.get("gpus") != 0
        or manifest.get("environment_creates") != 0
        or set(manifest.get("files", {})) != {"train", "dev"}
        or manifest.get("template_sha256") != "sha256:" + miles.TEMPLATE_SHA256
        or manifest.get("tool_catalog_sha256") is None
    ):
        raise ValueError("source Miles manifest identity changed")
    _sha(manifest["tool_catalog_sha256"], "source tool catalog digest")
    limits = manifest.get("limits")
    required_limits = {
        "context_tokens",
        "response_tokens",
        "max_tokens_per_turn",
        "max_turns",
        "episode_seconds",
        "tool_seconds",
        "tool_result_chars",
    }
    if (
        not isinstance(limits, dict)
        or set(limits) != required_limits
        or any(type(value) is not int or value <= 0 for value in limits.values())
        or not limits["max_tokens_per_turn"]
        <= limits["response_tokens"]
        < limits["context_tokens"]
        <= 98304
    ):
        raise ValueError("source Miles limits changed")
    if config["expected_limits"] != limits:
        raise ValueError("source Miles limits differ from the target arm contract")
    tokenizer = manifest.get("tokenizer")
    if (
        not isinstance(tokenizer, dict)
        or set(tokenizer) != {"repo", "revision", "files", "chat_template_sha256", "backend_sha256"}
        or tokenizer.get("repo") != "Qwen/Qwen3.8-27B"
        or re.fullmatch(r"[a-f0-9]{40}", str(tokenizer.get("revision", ""))) is None
        or re.fullmatch(r"[a-f0-9]{64}", str(tokenizer.get("chat_template_sha256", ""))) is None
        or re.fullmatch(r"[a-f0-9]{64}", str(tokenizer.get("backend_sha256", ""))) is None
        or not isinstance(tokenizer.get("files"), list)
        or not tokenizer["files"]
        or any(
            not isinstance(item, dict)
            or set(item) != {"path", "sha256"}
            or not isinstance(item["path"], str)
            or not item["path"]
            or Path(item["path"]).name != item["path"]
            or re.fullmatch(r"[a-f0-9]{64}", str(item["sha256"])) is None
            for item in tokenizer["files"]
        )
    ):
        raise ValueError("source Miles tokenizer identity changed")

    root = source_path.parent
    payloads = {"manifest.json": source_payload}
    documents = {}
    for name, schema, digest_field in (
        ("task-set.json", "cyber_rl_task_set_v1", "selection_sha256"),
        ("split.json", "cyber_task_split_v1", "split_sha256"),
    ):
        payload = _stable_bytes(root / name, "source " + name)
        value = _json_object(payload, "source " + name)
        _sealed(value, schema)
        if value["sha256"] != manifest[digest_field]:
            raise ValueError("source task selection/split differs from its manifest")
        payloads[name] = payload
        documents[name] = value
    if documents["task-set.json"].get("tool_catalog_sha256") != manifest["tool_catalog_sha256"]:
        raise ValueError("source task set tool catalog differs from its manifest")
    expected = {
        (row["split"], row["task_key"], row["task_version_id"]): row
        for row in selection(documents["task-set.json"], documents["split.json"])
    }

    source_rows = {}
    actual = {}
    for split in ("train", "dev"):
        item = manifest["files"][split]
        if (
            not isinstance(item, dict)
            or set(item) != {"path", "sha256", "rows", "max_prompt_tokens"}
            or item["path"] != split + ".jsonl"
            or type(item["rows"]) is not int
            or item["rows"] < 1
            or type(item["max_prompt_tokens"]) is not int
            or item["max_prompt_tokens"] < 1
        ):
            raise ValueError("source Miles file inventory changed")
        payload = _stable_bytes(root / item["path"], "source " + item["path"])
        if fleet.sha256(payload) != _sha(item["sha256"], "source split digest"):
            raise ValueError("source Miles split digest mismatch")
        rows = _rows(payload, split=split, manifest=manifest, source_root=source_policy_root)
        if len(rows) != item["rows"]:
            raise ValueError("source Miles row count changed")
        for row in rows:
            metadata = row["metadata"]
            task = metadata["cyber_config"].get("task")
            if not isinstance(task, dict):
                raise ValueError("source Miles task binding is absent")
            key = (split, task.get("key"), task.get("version_id"))
            if key in actual:
                raise ValueError("source Miles task version is duplicated")
            selected = expected.get(key)
            environment = metadata["cyber_config"].get("environment")
            if (
                selected is None
                or metadata["lineage"] != selected["lineage"]
                or not isinstance(environment, dict)
                or {
                    "env_key": environment.get("id"),
                    "env_version": environment.get("version"),
                    "environment_version_id": environment.get("version_id"),
                    "data_key": environment.get("data_id"),
                    "data_version": environment.get("data_version"),
                }
                != {
                    key: selected[key]
                    for key in (
                        "env_key",
                        "env_version",
                        "environment_version_id",
                        "data_key",
                        "data_version",
                    )
                }
            ):
                raise ValueError("source Miles row runtime binding changed")
            actual[key] = selected
        payloads[item["path"]] = payload
        source_rows[split] = rows
    if actual.keys() != expected.keys():
        raise ValueError("source Miles rows differ from the frozen task split")

    # Reopen every byte immediately before the create-once boundary.  A source
    # changed during validation cannot seed a derived artifact.
    if any(
        _stable_bytes(root / name, "source " + name) != payload
        for name, payload in payloads.items()
    ):
        raise ValueError("source Miles artifact changed during validation")

    output = Path(config["output"])
    output = output if output.is_absolute() else relative_to / output
    output = Path(os.path.abspath(output))
    if output.exists() or output.is_symlink():
        raise FileExistsError("create-once derived RL data destination exists")
    if output == root or output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("source and derived RL data roots must not overlap")

    transformed = {}
    files = {}
    for split, rows in source_rows.items():
        values = []
        for source_row in rows:
            row = copy.deepcopy(source_row)
            episode = row["metadata"]["cyber_config"]
            episode["run_id"] = config["name"]
            if mode == "policy_identity_rebind":
                episode["model"]["root"] = policy_root
            episode["config_sha256"] = fleet.digest_without(episode, "config_sha256")
            check = copy.deepcopy(row)
            check_config = check["metadata"]["cyber_config"]
            check_config["run_id"] = manifest["name"]
            check_config["model"]["root"] = source_policy_root
            check_config["config_sha256"] = source_row["metadata"]["cyber_config"]["config_sha256"]
            if check != source_row:
                raise ValueError("metadata-only derivation changed an unapproved field")
            rl_episode._validate(episode)
            values.append(row)
        payload = b"".join(fleet.canonical_json(row) + b"\n" for row in values)
        transformed[split + ".jsonl"] = payload
        files[split] = {**manifest["files"][split], "sha256": fleet.sha256(payload)}

    provenance = {
        "schema": DERIVATION_SCHEMA,
        "mode": "metadata_only_" + mode,
        "source_manifest_sha256": manifest["sha256"],
        "source_manifest_file_sha256": fleet.sha256(source_payload),
        "source_policy_identity_sha256": fleet.sha256(source_policy_root.encode()),
        "limits_sha256": fleet.sha256(fleet.canonical_json(limits)),
        "source_artifact_files": {
            name: fleet.sha256(payload) for name, payload in sorted(payloads.items())
        },
        "transformed_fields": (
            _POLICY_TRANSFORMED_FIELDS
            if mode == "policy_identity_rebind"
            else _RUN_TRANSFORMED_FIELDS
        ),
    }
    provenance["sha256"] = "sha256:" + digest(provenance)
    derived = {
        **{key: copy.deepcopy(value) for key, value in manifest.items() if key != "sha256"},
        "name": config["name"],
        "files": files,
        "derivation": provenance,
    }
    derived["sha256"] = "sha256:" + digest(derived)

    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    for name, payload in transformed.items():
        _write_once(output / name, payload)
    for name in ("task-set.json", "split.json"):
        _write_once(output / name, payloads[name])
    _write_once(output / "manifest.json", fleet.canonical_json(derived) + b"\n")
    directory = os.open(output, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return {
        "manifest_sha256": derived["sha256"],
        "source_manifest_sha256": manifest["sha256"],
        "files": files,
        "submitted": False,
    }
