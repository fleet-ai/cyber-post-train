from pathlib import Path

from evals.fleet import qwen38_dedicated_v1 as v1
from evals.fleet import qwen38_dedicated_v1_live as live
from evals.fleet import self_hosted

ROOT = Path.cwd()


def test_qwen_v1_is_one_non_scored_tp1_canary() -> None:
    value = v1.spec(ROOT)
    request = v1.payload(value, ROOT)
    assert value["scope"] == "one_non_scored_parity_canary_only"
    assert value["evaluation"]["scored_tasks_allowed"] is False
    assert request["workers"] == 1
    assert request["gpus_per_worker"] == 1
    assert request["priority_class"] == "fleet-infra-quiet"
    assert request["privileged"] is False


def test_qwen_v1_matches_live_tp1_context_and_tool_runtime() -> None:
    value = v1.spec(ROOT)
    command = v1.payload(value, ROOT)["command"]
    assert value["shape_basis"]["exact_live_tensor_parallel_size"] == 1
    assert value["shape_basis"]["exact_live_context_length"] == 262144
    assert value["shape_basis"]["exact_live_privileged"] is False
    assert "--tp-size 1" in command
    assert "--context-length 262144" in command
    assert "--kv-cache-dtype fp8_e4m3" in command
    assert "--reasoning-parser qwen3" in command
    assert "--tool-call-parser qwen3_coder" in command


def test_qwen_v1_rejects_partial_b300_and_embeds_evidence() -> None:
    command = v1.payload(v1.spec(ROOT), ROOT)["command"]
    assert "HARDWARE-PREFLIGHT.json" in command
    assert "FAILED_BEFORE_READINESS" in command
    assert "275040" in command
    assert "IDLE_SECONDS=600" in command


def test_qwen_v1_binds_exact_staging() -> None:
    assert v1.STAGING_RECEIPT_SHA.removeprefix("sha256:") == live.STAGING_RECEIPT_SHA
    assert live.MODEL_METADATA["indexed_shard_count"] == 18
    assert live.MODEL_METADATA["indexed_weight_bytes"] == 55563006776
    assert live.MODEL_METADATA["config_sha256"] == (
        "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab"
    )


def test_qwen_v1_live_rail_is_create_once_and_releasable() -> None:
    source = (ROOT / "evals/fleet/qwen38_dedicated_v1_live.py").read_text()
    assert 'client.post("/v1/runs", json=payload)' in source
    assert '"DELETE /v1/runs/{name}"' in source
    assert '"scored_tasks_launched": 0' in source


def test_qwen_v1_authorization_receipt_is_digest_and_file_valid() -> None:
    receipt = v1.load(
        ROOT / "docs/evidence/qwen38-study/2026-09-05-qwen38-dedicated-serving-v1-authorized.json"
    )
    assert receipt["receipt_sha256"] == self_hosted.digest_without(receipt, "receipt_sha256")
    for relative, digest in receipt["file_sha256"].items():
        assert self_hosted.sha256((ROOT / relative).read_bytes()) == digest
