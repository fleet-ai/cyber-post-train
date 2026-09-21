import copy
import hashlib
import json
from pathlib import Path

import pytest

from training import sft, sft_runtime

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "configs" / "runs"
A1_CONFIG = RUNS / "qwen38-27b-lora-sft-r64-a32-anchor-v1.json"
A2_CONFIG = RUNS / "qwen38-27b-lora-sft-r64-a32-anchor-a2-v1.json"
A2_PACKET = ROOT / "configs" / "qualification" / "qwen38-lora-anchor-parallel-candidate-a2-v1.json"
A1_COLLISION = (
    ROOT
    / "docs"
    / "evidence"
    / "qwen38-lora-anchor-a1-cpu-preflight-output-collision-20260921.json"
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _without_identity(value: dict) -> dict:
    result = copy.deepcopy(value)
    result.pop("name", None)
    result.pop("run_name", None)
    result.pop("output_root", None)
    wandb = result.get("wandb")
    if isinstance(wandb, dict):
        wandb.pop("run_id", None)
        wandb.pop("name", None)
    return result


def test_a2_is_an_identity_only_successor_of_the_broad_lora_anchor():
    a1 = _read(A1_CONFIG)
    a2 = _read(A2_CONFIG)

    assert _without_identity(a2) == _without_identity(a1)
    assert a2["name"] == "chris-q38-lora-sft-a2-v1"
    assert a2["output_root"] == "/mnt/sfs/jobs/chris-q38-lora-sft-a2-v1"
    assert a2["name"] == a2["wandb"]["run_id"] == a2["wandb"]["name"]

    plan = sft.compile_sft(a2, relative_to=RUNS)
    request = sft.job_request(plan)
    assert _without_identity(plan) == _without_identity(sft.compile_sft(a1, relative_to=RUNS))
    assert plan["qualification_gate"] == sft_runtime.QWEN38_LORA_PRODUCTION_QUALIFICATION
    assert sft_runtime._qwen38_lora_one_step_identity(plan) == (
        sft_runtime.qwen38_lora_broad_full_plan_binding(a2["name"])
    )
    assert request["name"] == request["title"] == a2["name"]
    assert request["run_dir"] == a2["output_root"]
    assert request["env"]["WANDB_RUN_ID"] == request["env"]["WANDB_NAME"] == a2["name"]
    assert request["priority_class"] == "c1"
    assert request["failureAlerts"] is False


@pytest.mark.parametrize(
    "path",
    [
        ("name",),
        ("output_root",),
        ("wandb", "run_id"),
        ("wandb", "name"),
    ],
)
def test_a2_rejects_a_partial_identity_reversion(path):
    a1 = _read(A1_CONFIG)
    a2 = _read(A2_CONFIG)
    target = a2
    source = a1
    for key in path[:-1]:
        target = target[key]
        source = source[key]
    target[path[-1]] = source[path[-1]]

    with pytest.raises(ValueError, match="Qwen3.8 LoRA requires"):
        sft.compile_sft(a2, relative_to=RUNS)


def test_a1_output_collision_is_sanitized_and_a2_packet_preserves_its_lesson():
    evidence = _read(A1_COLLISION)
    expected_evidence_digest = evidence.pop("sha256")
    assert expected_evidence_digest == _digest(evidence)
    assert evidence["status"] == "terminal_failure_released_no_gpu_allocation"
    assert evidence["attempt"]["job"]["uid"] == "b0fcf2e3-3901-4f5d-8dc8-ff2016b57230"
    assert evidence["attempt"]["pod"]["uid"] == "6461aa33-8472-4256-9aa9-ff4c8c120b7d"
    assert evidence["precreate_contract_observed"]["root_failure_alert_annotation"] == {
        "fleet.ai/failure-alerts": "off"
    }
    assert evidence["precreate_contract_observed"]["gpus"] == 0
    assert evidence["sanitized_terminal_receipt"] == {
        "schema": "cyber_sft_cpu_preflight_observation_v1",
        "status": "failed",
        "error_class": "FileExistsError",
    }
    assert evidence["cleanup"]["post_delete_read_only_recheck"] == {
        "job_present": False,
        "pods_present": False,
    }
    assert not {"logs", "prompts", "traces"} & set(evidence)

    packet = _read(A2_PACKET)
    expected_packet_digest = packet.pop("packet_sha256")
    assert expected_packet_digest == _digest(packet)
    assert packet["candidate"]["config_path"] == str(A2_CONFIG.relative_to(ROOT))
    assert (
        packet["candidate"]["config_file_sha256"]
        == "sha256:" + hashlib.sha256(A2_CONFIG.read_bytes()).hexdigest()
    )
    assert packet["identity_successor_of"]["terminal_preflight_evidence_path"] == str(
        A1_COLLISION.relative_to(ROOT)
    )
    assert packet["identity_successor_of"]["only_allowed_config_leaf_differences"] == [
        "name",
        "output_root",
        "wandb.run_id",
        "wandb.name",
    ]
    assert packet["launch_rail"]["remote_preparation_boundary"]["status"] == (
        "supported_sft_only_zero_gpu_preflight_job"
    )
    assert any("A1's output" in item for item in packet["pre_create_checklist"])
