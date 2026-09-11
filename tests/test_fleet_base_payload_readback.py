"""Offline tests for the read-only shared-base helper Pod."""

import hashlib
from pathlib import Path

from evals.fleet import base_payload_readback as readback


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _config(root: Path) -> dict:
    weights = [{"path": "model-1.safetensors", "size": 7, "sha256": _sha(b"weights")}]
    tokenizer = [
        {"path": "chat_template.jinja", "sha256": _sha(b"template")},
        {"path": "tokenizer.json", "sha256": _sha(b"tokenizer")},
    ]
    return {
        "plan_sha256": "sha256:" + "a" * 64,
        "root": str(root),
        "repository": "Qwen/synthetic",
        "revision": "b" * 40,
        "weights": weights,
        "tokenizer": tokenizer,
        "configuration": {"config.json": _sha(b"config")},
        "index_sha256": _sha(b"index"),
        "source_serving_pod": "serving-pod",
        "source_serving_pod_uid": "serving-uid",
        "source_node": "node-1",
    }


def test_probe_rehashes_complete_payload_without_exposing_bytes(tmp_path: Path) -> None:
    payloads = {
        "model-1.safetensors": b"weights",
        "chat_template.jinja": b"template",
        "tokenizer.json": b"tokenizer",
        "config.json": b"config",
        "model.safetensors.index.json": b"index",
        "extra-runtime.py": b"runtime",
    }
    for name, content in payloads.items():
        (tmp_path / name).write_bytes(content)
    receipt = readback.probe(
        _config(tmp_path),
        environ={"POD_NAME": "probe", "POD_UID": "probe-uid", "NODE_NAME": "node-1"},
    )
    assert receipt["status"] == "passed"
    assert receipt["payload"]["file_count"] == len(payloads)
    assert receipt["payload"]["total_bytes"] == sum(map(len, payloads.values()))
    assert receipt["scope"]["gpu_requests"] == 0
    assert all(receipt["scope"][key] == 0 for key in (
        "prompt_requests", "completion_requests", "scoring_requests", "task_or_grading_requests"
    ))
    assert receipt["sha256"] == readback.digest_json(
        {key: value for key, value in receipt.items() if key != "sha256"}
    )
    assert "files" not in receipt["payload"]


def test_probe_rejects_changed_locked_bytes(tmp_path: Path) -> None:
    for name, content in {
        "model-1.safetensors": b"changed",
        "chat_template.jinja": b"template",
        "tokenizer.json": b"tokenizer",
        "config.json": b"config",
        "model.safetensors.index.json": b"index",
    }.items():
        (tmp_path / name).write_bytes(content)
    try:
        readback.probe(_config(tmp_path))
    except ValueError as error:
        assert "weight bytes differ" in str(error)
    else:
        raise AssertionError("changed weight bytes were accepted")


def test_helper_pod_is_digest_pinned_read_only_and_zero_gpu(tmp_path: Path) -> None:
    config = _config(tmp_path)
    pod = readback.build_pod(
        config,
        name="readback-v1",
        namespace="inference",
        image="registry/image@sha256:" + "c" * 64,
    )
    spec = pod["spec"]
    container = spec["containers"][0]
    assert spec["restartPolicy"] == "Never"
    assert spec["automountServiceAccountToken"] is False
    assert spec["nodeName"] == "node-1"
    assert container["volumeMounts"] == [
        {"name": "served-scratch", "mountPath": "/scratch", "readOnly": True}
    ]
    assert "nvidia.com/gpu" not in container["resources"]["requests"]
    assert "nvidia.com/gpu" not in container["resources"]["limits"]
    assert not any("TOKEN" in row["name"] or "KEY" in row["name"] for row in container["env"])
