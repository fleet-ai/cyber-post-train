from __future__ import annotations

import json
from pathlib import Path

from training import miles96_model_stage as stage


def _sources(root: Path) -> tuple[Path, Path]:
    hf = root / "hf"
    hf.mkdir()
    (hf / "config.json").write_text("{}\n")
    (hf / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.embed_tokens.weight": "model-00001.safetensors"}})
    )
    (hf / "model-00001.safetensors").write_bytes(b"weights")
    (hf / "tokenizer.json").write_text("{}\n")
    megatron = root / "megatron"
    megatron.mkdir()
    (megatron / "latest_checkpointed_iteration.txt").write_text("release\n")
    (megatron / "model.pt").write_bytes(b"checkpoint")
    return hf, megatron


def test_stage_copies_exact_sources_and_atomically_completes(tmp_path: Path, monkeypatch) -> None:
    hf, megatron = _sources(tmp_path)
    destination = tmp_path / "prepared"
    partial = tmp_path / ".prepared.partial"
    monkeypatch.setattr(stage, "HF_SOURCE", hf)
    monkeypatch.setattr(stage, "MEGATRON_SOURCE", megatron)
    monkeypatch.setattr(stage, "DESTINATION", destination)
    monkeypatch.setattr(stage, "PARTIAL", partial)
    monkeypatch.setattr(
        stage.mechanics,
        "_hf_index",
        lambda _root: {"model.embed_tokens.weight": "model-00001.safetensors"},
    )
    manifest = stage.stage()
    assert destination.is_dir()
    assert not partial.exists()
    assert (destination / stage.COMPLETE).read_text().strip() == manifest["sha256"]
    assert (
        stage.mechanics.prepared_model_inventory(destination)["sha256"]
        == manifest["prepared_model_binding_sha256"]
    )


def test_packet_is_zero_gpu_create_once_alerts_off_and_source_read_only() -> None:
    packet = stage.build_packet()
    proof = stage.validate_packet(packet)
    assert proof["gpus"] == 0
    config_map, job = packet["bundle"]["items"]
    assert config_map["immutable"] is True
    assert job["metadata"]["annotations"]["fleet.ai/failure-alerts"] == "off"
    assert job["metadata"]["labels"]["kueue.x-k8s.io/priority-class"] == "q1"
    assert job["spec"]["backoffLimit"] == 0
    sequence = packet["execution_sequence"]
    assert sequence["config_map_create_request_count"] == 1
    assert sequence["config_map_create_retry_allowed"] is False
    assert sequence["job_create_request_count"] == 1
    assert sequence["job_create_retry_allowed"] is False
    assert sequence["job_created_suspended"] is True
    assert sequence["unsuspend"]["request_count"] == 1
    assert sequence["unsuspend"]["retry_allowed"] is False
    assert sequence["unsuspend"]["operations_template"] == [
        {"op": "test", "path": "/metadata/uid", "value": "$JOB_UID"},
        {"op": "test", "path": "/spec/suspend", "value": True},
        {"op": "replace", "path": "/spec/suspend", "value": False},
    ]
    assert sequence["post_unsuspend_create_or_patch_requests_allowed"] is False
    pod = job["spec"]["template"]["spec"]
    assert pod["priorityClassName"] == "c1"
    assert job["spec"]["suspend"] is True
    container = pod["containers"][0]
    security = container["securityContext"]
    assert security["runAsUser"] == 0
    assert security["readOnlyRootFilesystem"] is True
    assert security["privileged"] is False
    assert security["capabilities"] == {"drop": ["ALL"]}
    assert packet["expected"]["root_access_justification"] == stage.ROOT_ACCESS_JUSTIFICATION
    mounts = {item["name"]: item for item in container["volumeMounts"]}
    assert mounts["hf"]["readOnly"] is True
    assert mounts["megatron"]["readOnly"] is True
    assert mounts["jobs"].get("readOnly") is not True
    assert all(
        "nvidia.com/gpu" not in values
        for container in pod.get("initContainers", []) + pod["containers"]
        for values in container.get("resources", {}).values()
    )


def test_embedded_stager_is_exact() -> None:
    config_map = stage.build_packet()["bundle"]["items"][0]
    assert config_map["data"]["stage_module.py"] == Path(stage.__file__).read_text()
    assert config_map["data"]["driver.py"] == stage.DRIVER
