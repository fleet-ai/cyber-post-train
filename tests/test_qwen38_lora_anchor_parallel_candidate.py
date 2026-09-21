import hashlib
import json
from pathlib import Path

from training import sft

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "configs" / "runs"
PACKET = ROOT / "configs" / "qualification" / "qwen38-lora-anchor-parallel-candidate-v1.json"


def _digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def test_lora_anchor_parallel_candidate_is_sealed_to_the_qualified_broad_config():
    packet = json.loads(PACKET.read_text())
    expected = packet.pop("packet_sha256")
    assert expected == _digest(packet)
    assert packet["schema"] == "cyber_qwen38_sft_parallel_candidate_v1"
    assert packet["status"] == "local_only_not_prepared_or_submitted"

    candidate = packet["candidate"]
    config_path = ROOT / candidate["config_path"]
    config = json.loads(config_path.read_text())
    assert candidate["config_file_sha256"] == "sha256:" + hashlib.sha256(
        config_path.read_bytes()
    ).hexdigest()

    plan = sft.compile_sft(config, relative_to=RUNS)
    request = sft.job_request(plan)
    assert plan["run_name"] == candidate["name"]
    assert plan["output_root"] == candidate["output_root"]
    assert plan["model"]["repo"] == candidate["model"]["repo"]
    assert plan["model"]["revision"] == candidate["model"]["revision"]
    assert plan["lora"] == candidate["adapter"]
    assert plan["recipe"]["epochs"] == candidate["recipe"]["epochs"]
    assert plan["recipe"]["batch_size"] == candidate["recipe"]["global_batch"]
    assert plan["recipe"]["microbatch_per_gpu"] == candidate["recipe"]["microbatch_per_gpu"]
    assert plan["recipe"]["lr"] == candidate["recipe"]["learning_rate"]
    assert plan["recipe"]["max_length"] == candidate["recipe"]["max_length"]
    assert plan["recipe"]["max_steps"] == candidate["recipe"]["optimizer_steps"]
    assert plan["recipe"]["checkpoint_interval"] == candidate["recipe"]["checkpoint_interval"]
    assert plan["recipe"]["keep_checkpoints"] == candidate["recipe"]["keep_checkpoints"]
    assert plan["validation_mode"] == candidate["recipe"]["validation_mode"]
    assert plan["execution"]["priority"] == candidate["resources"]["priority_class"]
    assert request["workers"] == candidate["resources"]["nodes"]
    assert request["gpus_per_worker"] == candidate["resources"]["gpus_per_node"]
    assert request["failureAlerts"] is candidate["resources"]["failure_alert_request"]


def test_lora_anchor_parallel_candidate_requires_a_proven_root_alert_annotation_before_create():
    packet = json.loads(PACKET.read_text())
    rail = packet["launch_rail"]
    normal = rail["normal_path"]
    fallback = rail["sft_only_fallback_when_normal_preview_omits_only_the_root_annotation"]

    assert "preview" in normal["command"]
    assert "does not need an SFS mount" in rail["preparation_location"]["supported"]
    assert rail["remote_preparation_boundary"]["status"] == (
        "no_complete_remote_preparation_cli_rail_in_this_source"
    )
    assert "neither creates a LoRA CPU preflight Pod" in rail["remote_preparation_boundary"][
        "reason"
    ]
    assert any("root RayJob" in gate for gate in normal["allowed_only_if"])
    assert "direct-submit-sft" in fallback["command_with_local_sfs_mount"]
    assert "direct-submit-sft" in fallback["command_without_local_sfs_mount"]
    assert any("API-server dry-run" in gate for gate in fallback["proofs_the_command_requires"])
    assert "RL" in fallback["scope_limit"]
    assert packet["candidate"]["resources"]["required_root_annotation"] == {
        "fleet.ai/failure-alerts": "off"
    }
