"""Sanitized contract for the private Split-B SkyRL data build."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cyber_post_train.jobs import digest
from evals.fleet import opencode_self_hosted as fleet
from scripts import build_qwen38_skyrl_production_data as private
from scripts import prepare_qwen38_skyrl_production_queue as queue


def read(path: Path) -> dict:
    return json.loads(path.read_bytes())


def test_committed_packet_is_sanitized_sealed_and_bound_to_queue() -> None:
    packet = read(private.PACKET)
    qualification, evidence = read(queue.QUALIFICATION), read(queue.EVIDENCE)
    assert packet["sha256"] == "sha256:" + digest(
        {key: value for key, value in packet.items() if key != "sha256"}
    )
    assert packet["state"] == "locally_built_not_staged"
    assert packet["external_mutations"] == 0
    assert packet["fleet_reads"] == {
        "methods": ["GET"],
        "account_gets": 1,
        "unique_exact_version_task_gets": 79,
        "mutating_requests": 0,
    }
    assert packet["bindings"]["rows"] == {
        "train": 59,
        "dev": 20,
        "final_test_untouched": 10,
    }
    assert packet["runtime_ownership"]["uid"] == 1000
    assert packet["runtime_ownership"]["gid"] == 100
    assert packet["runtime_ownership"]["directory_mode"] == "0700"
    assert packet["runtime_ownership"]["file_mode"] == "0600"
    assert packet["privacy"]["repository_payload_bytes"] == 0
    assert [arm["name"] for arm in packet["arms"]] == [arm["name"] for arm in queue.ARMS]
    for arm in packet["arms"]:
        assert Path(arm["private_source"]).is_absolute()
        assert queue.ROOT not in Path(arm["private_source"]).parents
        assert arm["rows"] == {"train": 59, "dev": 20, "final_test": 0}
        assert {item["path"] for item in arm["files"]} == private.PRIVATE_FILES
        assert all(item["mode"] == "0600" for item in arm["files"])
    serialized = private.canonical(packet).decode()
    # Private row bodies and their sensitive field names never enter the packet.
    assert all(f'"{key}"' not in serialized for key in ("prompt", "input", "flag", "answer"))
    packet_ref = qualification["private_data"]["staging_packet"]
    assert packet_ref["self_sha256"] == packet["sha256"]
    assert evidence["data"]["local_private_build"]["packet"] == packet_ref
    assert all(
        arm["local_data_manifest_sha256"]
        == next(item for item in packet["arms"] if item["name"] == arm["name"])[
            "data_manifest_sha256"
        ]
        for arm in evidence["arms"]
    )


def _synthetic_private_package(tmp_path: Path) -> tuple[Path, Path, dict, dict, dict]:
    config_path = private.CONFIGS[0]
    config = read(config_path)
    task_set = read(config_path.parent / config["task_set"])
    split = read(config_path.parent / config["split"])
    split_by_key = {
        (row["task_key"], row["task_version_id"]): row["split"] for row in split["tasks"]
    }
    selected = [
        row
        for row in task_set["tasks"]
        if split_by_key[(row["task_key"], row["task_version_id"])] in {"train", "dev"}
    ]
    rows = {"train": [], "dev": []}
    for source in selected:
        group = split_by_key[(source["task_key"], source["task_version_id"])]
        frozen = {
            "task": {"key": source["task_key"], "version_id": source["task_version_id"]},
            "environment": {
                "id": source["env_key"],
                "version": source["env_version"],
                "version_id": source["environment_version_id"],
                "data_id": source["data_key"],
                "data_version": source["data_version"],
                "runtime_seed_content_sha256": "sha256:" + "a" * 64,
                "ttl_seconds": 32400,
            },
            "verifier": {
                "id": "synthetic-verifier",
                "version_id": "11111111-1111-4111-8111-111111111111",
                "version": "v1",
                "sha256": "sha256:" + "b" * 64,
                "function_name": "verify",
            },
            "execution": {
                "required_task_tools": ["bash", "submit_report"],
                "required_task_tool_catalog_sha256": task_set["tool_catalog_sha256"],
            },
        }
        frozen["config_sha256"] = fleet.digest_without(frozen, "config_sha256")
        rows[group].append(
            {
                "prompt": [{"role": "user", "content": "synthetic private row"}],
                "env_class": source["env_key"],
                "split": group,
                "cyber_config_json": fleet.canonical_json(frozen).decode(),
            }
        )
    root = tmp_path / "data"
    root.mkdir(mode=0o700)
    files = {}
    for group in ("train", "dev"):
        payload = b"".join(fleet.canonical_json(row) + b"\n" for row in rows[group])
        path = root / f"{group}.jsonl"
        path.write_bytes(payload)
        path.chmod(0o600)
        files[group] = {
            "path": path.name,
            "sha256": fleet.sha256(payload),
            "rows": len(rows[group]),
            "max_prompt_tokens": 8,
        }
    for name, value in (("split.json", split), ("task-set.json", task_set)):
        path = root / name
        path.write_bytes(fleet.canonical_json(value) + b"\n")
        path.chmod(0o600)
    manifest = {
        "schema": "cyber_skyrl_data_v1",
        "name": config["name"],
        "selection_sha256": task_set["sha256"],
        "split_sha256": split["sha256"],
        "tokenizer": {"revision": private.EXPECTED_MODEL_REVISION},
        "template_sha256": "sha256:" + "c" * 64,
        "tool_catalog_sha256": task_set["tool_catalog_sha256"],
        "limits": config["limits"],
        "files": files,
        "gpus": 0,
        "environment_creates": 0,
    }
    manifest["sha256"] = "sha256:" + digest(manifest)
    path = root / "manifest.json"
    path.write_bytes(fleet.canonical_json(manifest) + b"\n")
    path.chmod(0o600)
    return root, config_path, config, task_set, split


def test_post_build_verifier_accepts_exact_59_20_and_rejects_final_test(tmp_path: Path) -> None:
    root, config_path, config, task_set, split = _synthetic_private_package(tmp_path)
    result = private.inspect_private_package(root, config_path, config, task_set, split)
    assert result["rows"] == {"train": 59, "dev": 20, "final_test": 0}
    final = next(row for row in split["tasks"] if row["split"] == "test")
    path = root / "train.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    frozen = json.loads(rows[0]["cyber_config_json"])
    frozen["task"] = {"key": final["task_key"], "version_id": final["task_version_id"]}
    frozen["config_sha256"] = fleet.digest_without(frozen, "config_sha256")
    rows[0]["cyber_config_json"] = fleet.canonical_json(frozen).decode()
    payload = b"".join(fleet.canonical_json(row) + b"\n" for row in rows)
    path.write_bytes(payload)
    path.chmod(0o600)
    manifest = read(root / "manifest.json")
    manifest["files"]["train"]["sha256"] = fleet.sha256(payload)
    manifest["sha256"] = "sha256:" + digest(
        {key: value for key, value in manifest.items() if key != "sha256"}
    )
    (root / "manifest.json").write_bytes(fleet.canonical_json(manifest) + b"\n")
    (root / "manifest.json").chmod(0o600)
    with pytest.raises(ValueError, match="exact Fleet runtime tuple|final-test"):
        private.inspect_private_package(root, config_path, config, task_set, split)


def test_read_client_rejects_any_mutating_method() -> None:
    with private.CachedGetClient("synthetic-token") as client:
        with pytest.raises(ValueError, match="GET"):
            client.request("POST", "https://orchestrator.fleetai.com/v1/jobs")
        assert client.upstream_gets == []
