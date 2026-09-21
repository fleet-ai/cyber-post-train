"""Offline regressions for the seed-44 Base/Fresh75 packet renderer."""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

from evals.fleet import heldout_launch
from scripts import prepare_qwen38_fleet_seed44_two_arm_packets as packets

ROOT = Path(__file__).resolve().parents[1]


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _uid(character: str) -> str:
    return f"{character * 8}-{character * 4}-4{character * 3}-8{character * 3}-{character * 12}"


def _next_hex(character: str, offset: int) -> str:
    return format((int(character, 16) + offset) % 16, "x")


def _probe(served_model: str, character: str) -> dict:
    return {
        "served_model": served_model,
        "tool_call_passed": True,
        "tool_name": "identity",
        "tool_argument_keys": ["value"],
        "tool_response_sha256": _digest(character),
        "logits_finite": True,
        "logit_token_identity_deterministic": True,
        "logit_values_exactly_equal": True,
        "max_abs_logprob_delta": 0.0,
        "logit_token_count": 1,
        "logit_response_sha256": [_digest(character), _digest(character)],
        "logit_projection_sha256": _digest(character),
        "logit_token_identity_sha256": _digest(character),
    }


def _arm(config: dict, source_path: str, *, replicas: int, character: str) -> dict:
    route = next(iter(config["routes"].values()))
    revision = next(iter(config["models"].values()))["revision"]
    served_model = route["served_id"]
    contract = _digest("a")
    return {
        "served_model": served_model,
        "resource_version": "12345",
        "kubernetes_resource_version": "12346",
        "model_revision": revision,
        "source_path": source_path,
        "serving_path": route["model_info"]["model_path"],
        "normalized_contract_sha256": contract,
        "registration_spec_sha256": _digest(character),
        "catalog": {
            "id": served_model,
            "model_revision": revision,
            "status": "ready",
            "routed": True,
            "ready_replicas": replicas,
            "engine": route["catalog"]["engine"],
            "precision": route["catalog"]["precision"],
            "tensor_parallel_size": route["catalog"]["tensor_parallel_size"],
            "data_parallel_size": 8,
            "data_parallel_attention": False,
            "capabilities": ["chat_completions", "tool_calling"],
        },
        "model_info": {
            "architectures": route["model_info"]["architectures"],
            "model_type": route["model_info"]["model_type"],
            "load_format": "auto",
            "reasoning_parser": route["server_info"]["reasoning_parser"],
            "tool_call_parser": route["server_info"]["tool_call_parser"],
            "weight_version": revision,
        },
        "server_info": {
            "version": "0.5.2",
            "context_length": route["server_info"]["context_length"],
            "tp_size": route["server_info"]["tp_size"],
            "dp_size": route["server_info"]["dp_size"],
            "dtype": "bfloat16",
            "quantization": route["server_info"]["quantization"],
            "kv_cache_dtype": route["server_info"]["kv_cache_dtype"],
            "attention_backend": "trtllm_mha",
            "decode_attention_backend": "trtllm_mha",
            "prefill_attention_backend": "trtllm_mha",
            "chunked_prefill_size": 32768,
            "max_prefill_tokens": 32768,
            "reasoning_parser": route["server_info"]["reasoning_parser"],
            "tool_call_parser": route["server_info"]["tool_call_parser"],
            "speculative_algorithm": "NEXTN",
            "speculative_num_steps": 3,
            "speculative_eagle_topk": 1,
            "speculative_num_draft_tokens": 4,
            "load_format": "auto",
            "weight_version": revision,
        },
        "kubernetes": {
            "inference_model_uid": _uid(character),
            "inference_model_resource_version": "12346",
            "deployment_uid": _uid(_next_hex(character, 1)),
            "replicaset_uids": [_uid(_next_hex(character, 2))],
            "pod_uids": [_uid(_next_hex(character, 3 + index)) for index in range(replicas)],
            "service_uid": _uid(_next_hex(character, 5)),
            "endpoint_slice_uids": [_uid(_next_hex(character, 6))],
            "node_names": [f"node-{index}" for index in range(replicas)],
            "ready_replicas": replicas,
            "restart_counts": [0] * replicas,
            "image_digest": packets.EXPECTED_SERVING_IMAGE,
            "ready_endpoint_count": replicas,
            "normalized_inference_spec_sha256": contract,
        },
        "probes": _probe(served_model, character),
    }


def _live_parity(path: Path, observed_at: datetime) -> dict:
    readiness = json.loads(packets.READINESS.read_text())
    configs = {
        arm_id: json.loads((ROOT / arm["config_path"]).read_text())
        for arm_id, arm in readiness["arms"].items()
    }
    base = _arm(
        configs["base"],
        readiness["arms"]["base"]["expected_source_path"],
        replicas=2,
        character="1",
    )
    candidate = _arm(
        configs["fresh75"],
        readiness["arms"]["fresh75"]["expected_source_path"],
        replicas=1,
        character="8",
    )
    value = {
        "schema": packets.LIVE_PARITY_SCHEMA,
        "status": "passed",
        "observed_at": observed_at.isoformat().replace("+00:00", "Z"),
        "endpoint_origin": "https://inference.flt.build",
        "arms": {"base": base, "candidate": candidate},
        "held_constant": {
            "normalized_contract_sha256": base["normalized_contract_sha256"],
            "model_info_sha256": _digest("b"),
            "server_info_sha256": _digest("c"),
            "inference_precision": "bf16",
            "max_context_size": 262144,
            "weight_quantization": "none",
        },
        "benchmark_content_included": False,
        "response_content_recorded": False,
        "scores_observed": False,
        "task_content_included": False,
        "external_mutations_performed": 0,
        "resource_versions_are_non_atomic_live_observations": True,
        "fixed_probe_logit_projection_differs_between_weights": True,
    }
    value["receipt_sha256"] = packets._canonical_digest(value)  # noqa: SLF001
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return value


def test_renderer_builds_two_offline_launch_packets_with_one_fresh_receipt(tmp_path) -> None:
    now = datetime(2026, 9, 21, 20, tzinfo=UTC)
    parity_path = tmp_path / "live-parity.json"
    parity = _live_parity(parity_path, now)
    output = tmp_path / "packets"

    receipt = packets.prepare(output=output, live_parity=parity_path, now=now)

    assert receipt["launch_performed"] is False
    assert receipt["external_mutations"] == 0
    assert receipt["live_parity_receipt_sha256"] == parity["receipt_sha256"]
    assert [arm["arm_id"] for arm in receipt["arms"]] == ["base", "fresh75"]
    for arm_id in ("base", "fresh75"):
        packet_path = output / arm_id / "LAUNCH_PACKET.json"
        package = heldout_launch.build_package(packet_path)
        assert package.job["metadata"]["annotations"] == {
            "fleet.ai/failure-alerts": "off",
            "cyber-post-train.fleet.ai/create-once": "true",
        }
        assert "nvidia.com/gpu" not in json.dumps(package.job)
        assert package.packet.files["serving_route_proof"].read_bytes() == parity_path.read_bytes()
        data = package.config_map["data"]
        assert ("model-artifact.json" in data) is (arm_id == "fresh75")
        assert ("model-artifact-acceptance.json" in data) is (arm_id == "fresh75")
        if arm_id == "fresh75":
            assert package.packet.job_name == "chris-q38-dev17-s44-fresh75-p1-v2"
            assert package.packet.config_map_name == "chris-q38-dev17-s44-fresh75-code-v2"
            assert package.packet.output_root.endswith("-fresh75-p1-v2")
            assert package.packet.database == "q38_dev17_s44_fresh75_p1_v2"
            assert package.evaluation_config["name"] == "q38-dev17-s44-fresh75-p1-v2"
            assert (
                'install -m 0600 /bootstrap/model-artifact.json "$artifact_packet"'
                in data["run.sh"]
            )
        else:
            assert package.packet.job_name == "chris-q38-dev17-s44-base-p1-v1"


def test_renderer_rejects_stale_or_route_drifted_receipts(tmp_path) -> None:
    now = datetime(2026, 9, 21, 20, tzinfo=UTC)
    stale_path = tmp_path / "stale.json"
    _live_parity(stale_path, now - timedelta(seconds=packets.MAX_PROOF_AGE_SECONDS + 1))
    with pytest.raises(ValueError, match="stale or unsafe"):
        packets.prepare(output=tmp_path / "stale-output", live_parity=stale_path, now=now)

    drifted_path = tmp_path / "drifted.json"
    drifted = _live_parity(drifted_path, now)
    drifted["arms"]["candidate"]["kubernetes"]["image_digest"] = _digest("f")
    drifted["receipt_sha256"] = packets._canonical_digest(  # noqa: SLF001
        {key: item for key, item in drifted.items() if key != "receipt_sha256"}
    )
    drifted_path.write_text(json.dumps(drifted, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not exact and ready"):
        packets.prepare(output=tmp_path / "drift-output", live_parity=drifted_path, now=now)


def test_renderer_rejects_setup_successor_without_zero_claim_evidence(
    tmp_path, monkeypatch
) -> None:
    now = datetime(2026, 9, 21, 20, tzinfo=UTC)
    parity_path = tmp_path / "live-parity.json"
    _live_parity(parity_path, now)
    readiness = json.loads(packets.READINESS.read_text())
    readiness["reviewed_setup_successor"]["primary"]["output_root_absent"] = False
    readiness["sha256"] = packets._canonical_digest(  # noqa: SLF001
        {key: item for key, item in readiness.items() if key != "sha256"}
    )
    path = tmp_path / "readiness.json"
    path.write_text(json.dumps(readiness), encoding="utf-8")
    monkeypatch.setattr(packets, "READINESS", path)
    with pytest.raises(ValueError, match="setup-successor evidence"):
        packets.prepare(output=tmp_path / "packets", live_parity=parity_path, now=now)


def test_rendered_root_job_alert_gate_is_not_only_a_pod_annotation(tmp_path) -> None:
    now = datetime(2026, 9, 21, 20, tzinfo=UTC)
    parity_path = tmp_path / "live-parity.json"
    _live_parity(parity_path, now)
    output = tmp_path / "packets"
    packets.prepare(output=output, live_parity=parity_path, now=now)
    job_path = output / "base" / "job.yaml"
    job = yaml.safe_load(job_path.read_text())
    assert job["metadata"]["annotations"]["fleet.ai/failure-alerts"] == "off"
    assert job["spec"]["template"].get("metadata", {}).get("annotations") is None


def test_live_parity_self_digest_covers_both_arm_identities(tmp_path) -> None:
    now = datetime(2026, 9, 21, 20, tzinfo=UTC)
    path = tmp_path / "live-parity.json"
    value = _live_parity(path, now)
    changed = copy.deepcopy(value)
    changed["arms"]["candidate"]["model_revision"] = _digest("e")
    path.write_text(json.dumps(changed, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="self digest differs"):
        packets.prepare(output=tmp_path / "packets", live_parity=path, now=now)
