from pathlib import Path

from evals.fleet import qwen38_dedicated_v2 as v2
from evals.fleet import qwen38_dedicated_v2_live as live
from evals.fleet import self_hosted

ROOT = Path.cwd()


def test_v2_is_fresh_one_gpu_server_then_held_rank2_remainder() -> None:
    value = v2.spec(ROOT)
    request = v2.payload(value, ROOT)
    assert request["workers"] == request["gpus_per_worker"] == 1
    assert request["title"].endswith("-v2")
    assert request["run_dir"].endswith("-v2")
    assert request["priority_class"] == "fleet-infra-quiet"
    assert value["resources"]["preemption_policy"] == "Never"
    assert value["evaluation"]["rank2_attempts_allowed_after_parity"] == [2, 3, 4]
    assert value["evaluation"]["scored_tasks_allowed_before_parity"] is False


def test_v2_preserves_exact_model_runtime_and_context() -> None:
    value = v2.spec(ROOT)
    command = v2.payload(value, ROOT)["command"]
    assert value["model"]["revision"] == v2.MODEL_REVISION
    assert value["runtime"]["image"] == v2.IMAGE
    assert "--context-length 262144" in command
    assert "--reasoning-parser qwen3" in command
    assert "--tool-call-parser qwen3_coder" in command
    assert "IDLE_SECONDS=600" in command


def test_v2_live_rail_requires_fresh_runtime_and_fresh_parity() -> None:
    source = (ROOT / "evals/fleet/qwen38_dedicated_v2_live.py").read_text()
    assert "old_runtime_uid_matches" in source
    assert '"fresh_parity_required_before_scoring": True' in source
    assert 'client.post("/v1/runs", json=payload)' in source
    assert '"scored_tasks_launched": 0' in source
    assert live.live_gate


def test_v2_authorization_receipt_is_sealed_and_binds_files() -> None:
    receipt = v2.load(
        ROOT / "docs/evidence/qwen38-study/2026-09-05-qwen38-dedicated-serving-v2-authorized.json"
    )
    assert receipt["receipt_sha256"] == self_hosted.digest_without(receipt, "receipt_sha256")
    for relative, expected in receipt["file_sha256"].items():
        assert self_hosted.sha256((ROOT / relative).read_bytes()) == expected
