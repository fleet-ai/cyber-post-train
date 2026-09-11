"""Offline tests for the split-A matched-base certification gate."""

import copy
import json
from pathlib import Path

import pytest

from evals import study_sealing
from evals.fleet import base_control_certification as certification
from evals.fleet import dev_outcome_protocol as fleet
from training.io import digest_json

ROOT = Path(__file__).parents[1]
PLAN_PATH = (
    ROOT / "configs" / "evaluation" / "qwen38-blackbox-fleet-dev-a-base-certification-v1.json"
)


def seal(value: dict) -> dict:
    return {**value, "sha256": digest_json(value)}


def inputs() -> tuple[dict, dict, dict, dict]:
    plan = json.loads(PLAN_PATH.read_text())
    lock = json.loads((ROOT / plan["references"]["model_lock"]["path"]).read_text())
    candidate = plan["candidate_base_route"]
    registration = {
        key: copy.deepcopy(candidate[key])
        for key in (
            "endpoint_origin",
            "served_model_id",
            "serving_block_kind",
            "object_identity",
            "runtime",
        )
    }
    registration["serving_route_profile_sha256"] = digest_json(registration)
    registration["serving_registration_receipt_sha256"] = "sha256:" + "a" * 64
    base = seal(
        {
            "schema": certification.BASE_SCHEMA,
            "status": "passed",
            "plan_sha256": plan["sha256"],
            "observed_at": "2026-09-11T23:00:00Z",
            "model_artifact": {
                "repository": lock["repo"],
                "revision": lock["revision"],
                "base_snapshot_manifest_sha256": "sha256:" + "b" * 64,
                "weights_manifest_sha256": lock["weights"]["manifest_sha256"],
                "tokenizer_manifest_sha256": lock["tokenizer"]["manifest_sha256"],
                "chat_template_sha256": "sha256:"
                + next(
                    row["sha256"]
                    for row in lock["tokenizer"]["files"]
                    if row["path"] == "chat_template.jinja"
                ),
                "staging_receipt_sha256": "sha256:" + "c" * 64,
                "payload_rehashed": True,
                "symlinks_absent": True,
            },
            "registration": registration,
            "readiness": {
                "catalog_ready": True,
                "routed": True,
                "ready_endpoint_count": 1,
                "model_info_http_status": 200,
                "server_info_http_status": 200,
                "prompt_requests": 0,
                "completion_requests": 0,
                "scoring_requests": 0,
            },
        }
    )
    harness = seal(
        {
            "schema": certification.HARNESS_SCHEMA,
            "status": "passed",
            "plan_sha256": plan["sha256"],
            "qualified_at": "2026-09-11T23:05:00+00:00",
            "platform": "linux/amd64",
            "agent_image_digest": "registry/agent@sha256:" + "d" * 64,
            "proxy_image_digest": "registry/proxy@sha256:" + "e" * 64,
            "harness_sha256": plan["harness_runtime"]["harness_sha256"],
            "sampling_sha256": plan["harness_runtime"]["sampling_sha256"],
            "runtime_files": plan["harness_runtime"]["runtime_files"],
            "runtime_files_sha256": plan["harness_runtime"]["runtime_files_sha256"],
            "checks": {
                "fresh_digest_pull": True,
                "opencode_release_and_label": True,
                "private_home_startup": True,
                "ordered_bash_submit_report_tools": True,
                "native_compaction_trigger": True,
                "automatic_continuation_after_compaction": True,
                "fixed_sampling_seed_and_32768_output": True,
                "proxy_path_auth_size_and_request_limits": True,
                "zero_prompt_completion_and_scoring_requests": True,
            },
        }
    )
    runtime = {
        "endpoint_origin": candidate["endpoint_origin"],
        "serving_block_kind": candidate["serving_block_kind"],
        **candidate["runtime"],
        "tokenizer_manifest_sha256": base["model_artifact"]["tokenizer_manifest_sha256"],
        "chat_template_sha256": base["model_artifact"]["chat_template_sha256"],
        "agent_image_digest": harness["agent_image_digest"],
        "proxy_image_digest": harness["proxy_image_digest"],
    }
    post = {
        "checkpoint_seal_sha256": "sha256:" + "1" * 64,
        "export_receipt_sha256": "sha256:" + "2" * 64,
        "weights_manifest_sha256": "sha256:" + "3" * 64,
        "tokenizer_manifest_sha256": base["model_artifact"]["tokenizer_manifest_sha256"],
        "chat_template_sha256": base["model_artifact"]["chat_template_sha256"],
        "staging_receipt_sha256": "sha256:" + "4" * 64,
        "serving_registration_receipt_sha256": "sha256:" + "5" * 64,
        "served_model_id": "qwen38-sft-arm-a",
        "serving_route_profile_sha256": "sha256:" + "6" * 64,
        "agent_image_digest": harness["agent_image_digest"],
        "proxy_image_digest": harness["proxy_image_digest"],
    }
    pair = seal(
        {
            "schema": certification.PAIR_SCHEMA,
            "status": "passed",
            "plan_sha256": plan["sha256"],
            "observed_at": "2026-09-11T23:10:00Z",
            "base_route_receipt_sha256": base["sha256"],
            "harness_receipt_sha256": harness["sha256"],
            "post_sft": post,
            "base_runtime": runtime,
            "post_runtime": copy.deepcopy(runtime),
            "checks": {
                "post_checkpoint_export_accepted": True,
                "post_staging_create_once": True,
                "base_and_post_registration_current": True,
                "tokenizer_encode_decode_match": True,
                "structured_tool_call_both": True,
                "fixed_logit_probe_finite_both": True,
                "fixed_logit_probe_deterministic_within_arm": True,
                "context_compaction_and_autocontinue_both": True,
                "same_runtime_except_weights": True,
            },
            "observations": {
                "normalized_tool_request_sha256": "sha256:" + "7" * 64,
                "normalized_logit_request_sha256": "sha256:" + "8" * 64,
                "tokenizer_probe_sha256": "sha256:" + "9" * 64,
                "base_response_manifest_sha256": "sha256:" + "0" * 64,
                "post_response_manifest_sha256": "sha256:" + "f" * 64,
                "window_seconds": 300,
                "non_scored_completion_requests": 6,
                "task_instance_creates": 0,
                "grading_requests": 0,
            },
        }
    )
    return plan, base, harness, pair


def reseal(value: dict) -> None:
    value["sha256"] = digest_json({key: item for key, item in value.items() if key != "sha256"})


def test_frozen_plan_is_exact_nonlaunchable_and_offline():
    plan, _, _, _ = inputs()
    certification.validate_plan(plan, root=ROOT)
    assert plan["launchable"] is False
    assert plan["paid_or_scored_work_authorized"] is False
    assert [row["availability"] for row in plan["workflow"]["order"]] == [
        "available_now",
        "available_now",
        "wait_for_accepted_post_sft_export",
        "wait_for_both_routes_live",
        "wait_for_all_receipts",
    ]
    assert all(row["new_gpu_count"] == 0 for row in plan["workflow"]["order"])
    assert all(
        template["status"] == "template_not_evidence" and template["sha256"] is None
        for template in plan["receipt_templates"].values()
    )


def test_complete_receipts_project_exact_study_child_bindings():
    plan, base, harness, pair = inputs()
    certificate = certification.assemble_certificate(plan, base, harness, pair, root=ROOT)
    assert certificate["base_serving_binding"]["launchable"] is True
    assert set(certificate["base_serving_binding"]["fields"]) == set(
        fleet.UNBOUND_BASE_SERVING_FIELDS
    )
    assert set(certificate["post_sft_binding"]) == set(fleet.UNBOUND_CHECKPOINT_FIELDS)
    assert set(certificate["matched_runtime"]) == set(study_sealing.MATCHED_RUNTIME_FIELDS)
    assert (
        certificate["base_serving_binding"]["fields"]["weights_manifest_sha256"]
        != certificate["post_sft_binding"]["weights_manifest_sha256"]
    )
    assert certificate["paid_or_scored_work_authorized"] is False
    assert certificate["sha256"] == digest_json(
        {key: value for key, value in certificate.items() if key != "sha256"}
    )


def test_plan_rejects_a_different_live_object_even_when_resealed():
    plan, _, _, _ = inputs()
    plan["candidate_base_route"]["object_identity"]["pod_uid"] = (
        "00000000-0000-0000-0000-000000000000"
    )
    reseal(plan)
    with pytest.raises(ValueError, match="differs from the read-only audit"):
        certification.validate_plan(plan, root=ROOT)


@pytest.mark.parametrize(
    ("target", "mutation", "message"),
    [
        (
            "base",
            lambda value: value["model_artifact"].update(payload_rehashed=False),
            "safely and completely rehashed",
        ),
        (
            "base",
            lambda value: value["readiness"].update(completion_requests=1),
            "contains model/scoring requests",
        ),
        (
            "harness",
            lambda value: value["checks"].update(automatic_continuation_after_compaction=False),
            "offline v2 checks",
        ),
        (
            "pair",
            lambda value: value["post_sft"].update(
                weights_manifest_sha256=inputs()[1]["model_artifact"]["weights_manifest_sha256"]
            ),
            "weights must differ",
        ),
        (
            "pair",
            lambda value: value["post_runtime"].update(data_parallel_size=1),
            "runtime differs",
        ),
        (
            "pair",
            lambda value: value["observations"].update(task_instance_creates=1),
            "task-free",
        ),
    ],
)
def test_certificate_rejects_partial_or_unmatched_evidence(target, mutation, message):
    plan, base, harness, pair = inputs()
    values = {"base": base, "harness": harness, "pair": pair}
    mutation(values[target])
    reseal(values[target])
    if target in {"base", "harness"}:
        reference = "base_route_receipt_sha256" if target == "base" else "harness_receipt_sha256"
        pair[reference] = values[target]["sha256"]
        reseal(pair)
    with pytest.raises(ValueError, match=message):
        certification.assemble_certificate(plan, base, harness, pair, root=ROOT)


def test_receipts_reject_unknown_fields_that_could_leak_private_content():
    plan, base, harness, pair = inputs()
    pair["raw_probe_output"] = "forbidden"
    reseal(pair)
    with pytest.raises(ValueError, match="unknown or missing"):
        certification.assemble_certificate(plan, base, harness, pair, root=ROOT)
