from __future__ import annotations

import copy
import json
from pathlib import Path

from training.model_stage import digest_json

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT / "configs/evaluation/qwen38-fresh75-step230-base-runtime-clone-v1.json"
)


def _replace_flag(args: list[str], flag: str, value: str) -> None:
    assert args.count(flag) == 1
    args[args.index(flag) + 1] = value


def _fixed_control(spec: dict) -> dict:
    value = copy.deepcopy(spec)
    value["displayName"] = "__display_name__"
    value["desiredState"] = "__operational_state__"
    value["placement"]["priorityClassName"] = "__priority__"
    value["scaling"] = {"normalized": True}
    for field in ("path", "revision", "sourcePath"):
        value["model"][field] = "__checkpoint_identity__"
    _replace_flag(value["runtime"]["args"], "--model-path", "__checkpoint_identity__")
    _replace_flag(value["runtime"]["args"], "--served-model-name", "__served_identity__")
    return value


def test_fresh75_manifest_is_self_digesting_and_runtime_matched() -> None:
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["schema"] == "cyber_qwen38_checkpoint_base_runtime_clone_v1"
    assert manifest["status"] == "live_route_observed_runtime_matched"
    unsigned = {key: value for key, value in manifest.items() if key != "sha256"}
    assert manifest["sha256"] == digest_json(unsigned)

    base = manifest["base"]
    candidate = manifest["candidate"]
    comparison = manifest["fixed_control_comparison"]
    assert base["spec_sha256"] == digest_json(base["spec"])
    assert candidate["spec_sha256"] == digest_json(candidate["spec"])
    assert candidate["registration_sha256"] == digest_json(
        {"id": candidate["id"], "spec": candidate["spec"]}
    )
    assert comparison["base_normalized_spec_sha256"] == digest_json(
        _fixed_control(base["spec"])
    )
    assert comparison["candidate_normalized_spec_sha256"] == digest_json(
        _fixed_control(candidate["spec"])
    )
    assert comparison["base_normalized_spec_sha256"] == comparison[
        "candidate_normalized_spec_sha256"
    ]
    assert comparison["equal"] is True


def test_fresh75_candidate_is_paused_c1_and_binds_accepted_payload() -> None:
    manifest = json.loads(MANIFEST.read_text())
    candidate = manifest["candidate"]
    spec = candidate["spec"]
    args = spec["runtime"]["args"]
    assert candidate["id"] == "chris-q38-fresh75-step230-web-v2"
    assert candidate["phase"] == "ready"
    assert spec["desiredState"] == "serving"
    assert spec["scaling"] == {"minReplicas": 1, "replicas": 1}
    assert spec["placement"]["priorityClassName"] == "c1"
    assert spec["model"]["tensorParallelSize"] == 1
    assert spec["model"]["dataParallelSize"] == 8
    assert spec["model"]["sourcePath"] == candidate["staged_path"]
    assert spec["model"]["revision"] == candidate["staged_payload_manifest_sha256"]
    assert args[args.index("--context-length") + 1] == "262144"
    assert args[args.index("--kv-cache-dtype") + 1] == "fp8_e4m3"
    assert args[args.index("--reasoning-parser") + 1] == "qwen3"
    assert args[args.index("--tool-call-parser") + 1] == "qwen3_coder"
    assert manifest["operation"] == {
        "duplicate_route_proposed": False,
        "resume_or_gpu_allocation_performed_by_this_audit": False,
        "route_created_by_this_audit": False,
        "route_modified_by_this_audit": False,
        "route_owner": "checkpoint_promotion",
    }
