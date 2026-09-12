"""Offline tests for early split-A base acquisition and later parity reuse."""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from cyber_post_train.jobs import digest as raw_digest
from evals import study_sealing
from evals.fleet import base_control_certification as certification_v1
from evals.fleet import base_outcome_acquisition as acquisition
from evals.fleet import dev_outcome_protocol as fleet
from training.io import digest_json

ROOT = Path(__file__).parents[1]
PLAN_PATH = ROOT / "configs/evaluation/qwen38-blackbox-fleet-dev-a-base-acquisition-v1.json"
PAYLOAD_PATH = (
    ROOT / "docs/evidence/inference/2026-09-11-qwen38-shared-base-payload-readback-v1.json"
)
ROUTE_PATH = ROOT / "docs/evidence/inference/2026-09-11-qwen38-shared-base-route-component-v2.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def seal(value: dict) -> dict:
    value = {key: item for key, item in value.items() if key != "sha256"}
    return {**value, "sha256": digest_json(value)}


def raw_seal(value: dict) -> dict:
    value = {key: item for key, item in value.items() if key != "sha256"}
    return {**value, "sha256": raw_digest(value)}


def agent_image_qualification(plan: dict) -> dict:
    frozen = plan["frozen_design"]["agent_image_publication"]
    image = f"{frozen['registry_repository']}@sha256:" + "a" * 64
    return seal(
        {
            "schema": acquisition.AGENT_IMAGE_QUALIFICATION_SCHEMA,
            "status": "passed",
            "publication_plan_sha256": frozen["plan_sha256"],
            "qualified_at": "2026-09-12T12:00:00Z",
            "agent_image_digest": image,
            "publication": {
                "repository_commit": frozen["repository_commit"],
                "dockerfile_sha256": frozen["dockerfile_sha256"],
                "release_asset_sha256": frozen["release_asset_sha256"],
                "opencode_binary_sha256": frozen["opencode_binary_sha256"],
                "source_date_epoch": frozen["source_date_epoch"],
                "context_archive_sha256": frozen["context_archive_sha256"],
                "context_receipt_sha256": frozen["context_receipt_sha256"],
                "registry_repository": frozen["registry_repository"],
                "tag": frozen["tag"],
                "rewrite_timestamp": True,
                "terminal_status": "succeeded",
                "image_digest": image,
            },
            "clean_pull": {
                "cluster_context": "nebius-mk8s-fleetai-training-dev-e04p03enwk5c0va9tb",
                "namespace": "default",
                "pod": "opencode-qualification",
                "pod_uid": "synthetic-pod-uid",
                "node": "synthetic-dev-node",
                "server_dry_run_passed": True,
                "image_pull_policy": "Always",
                "observed_image_id": image,
                "platform": "linux/amd64",
                "terminal_phase": "Succeeded",
                "exit_code": 0,
                "restart_count": 0,
                "gpu_requests": 0,
                "deleted": True,
                "absence_confirmed": True,
            },
            "checks": {
                "fresh_digest_pull": True,
                "exact_source_labels": True,
                "opencode_1_18_27": True,
                "private_home": True,
                "non_root_user": True,
                "working_directory": True,
                "ordered_bash_submit_report_tools": True,
                "native_compaction_autocontinue_v1": True,
                "native_compaction_autocontinue_v2": True,
            },
            "scope": {
                "gpu_requests": 0,
                "model_requests": 0,
                "prompt_requests": 0,
                "completion_requests": 0,
                "scoring_requests": 0,
                "task_or_grading_requests": 0,
                "evaluation_submissions": 0,
            },
            "paid_or_scored_work_authorized": False,
        }
    )


def harness_receipt(plan: dict, qualification: dict) -> dict:
    v2 = load(ROOT / plan["references"]["base_certification"]["path"])
    v1 = load(ROOT / v2["supersedes"]["path"])
    runtime = v1["harness_runtime"]
    return seal(
        {
            "schema": certification_v1.HARNESS_SCHEMA,
            "status": "passed",
            "plan_sha256": v1["sha256"],
            "qualified_at": "2026-09-12T12:00:00Z",
            "platform": "linux/amd64",
            "agent_image_digest": qualification["agent_image_digest"],
            "proxy_image_digest": plan["frozen_design"]["fixed_proxy_image_digest"],
            "harness_sha256": runtime["harness_sha256"],
            "sampling_sha256": runtime["sampling_sha256"],
            "runtime_files": runtime["runtime_files"],
            "runtime_files_sha256": runtime["runtime_files_sha256"],
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


def certificate(plan: dict) -> dict:
    qualification = agent_image_qualification(plan)
    return acquisition.assemble_acquisition_certificate(
        plan,
        load(PAYLOAD_PATH),
        load(ROUTE_PATH),
        harness_receipt(plan, qualification),
        qualification,
        root=ROOT,
    )


def evaluator_plan(plan: dict, cert: dict) -> dict:
    task_set = load(ROOT / plan["references"]["task_set"]["path"])
    runtime = cert["runtime"]
    tasks = [
        {
            key: row[key]
            for key in sorted(
                {
                    "task_key",
                    "task_version_id",
                    "env_key",
                    "env_version",
                    "environment_version_id",
                    "data_key",
                    "data_version",
                }
            )
        }
        for row in task_set["tasks"]
    ]
    tasks.sort(key=lambda row: row["task_version_id"])
    versions = [row["task_version_id"] for row in tasks]
    model_path = f"/scratch/models/qwen3.8-27b/{fleet.MODEL_REVISION}"
    route = {
        "model": "qwen-base",
        "served_id": cert["base_binding"]["served_model_id"],
        "task_versions": versions,
        "catalog": {
            "engine": runtime["engine"],
            "precision": runtime["precision"],
            "tensor_parallel_size": runtime["tensor_parallel_size"],
        },
        "model_info": {
            "model_path": model_path,
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForConditionalGeneration"],
        },
        "server_info": {
            "model_path": model_path,
            "context_length": runtime["context_length"],
            "tp_size": runtime["tensor_parallel_size"],
            "dp_size": runtime["data_parallel_size"],
            "load_balance_method": "total_tokens",
            "quantization": None,
            "kv_cache_dtype": runtime["kv_cache_dtype"],
            "reasoning_parser": runtime["reasoning_parser"],
            "tool_call_parser": runtime["tool_call_parser"],
        },
        "endpoint_origin": runtime["endpoint_origin"],
    }
    return raw_seal(
        {
            "schema": "cyber_fleet_eval_v1",
            "campaign_id": "chris-q38-split-a-base-v1",
            "run_prefix": "chris-q38-split-a-base-v1",
            "selection": {"source_job_id": task_set["source_job_id"]},
            "tasks": tasks,
            "models": {
                "qwen-base": {
                    "repository": fleet.MODEL_REPOSITORY,
                    "revision": fleet.MODEL_REVISION,
                    "session_model": cert["base_binding"]["served_model_id"],
                }
            },
            "routes": {"base-shared": route},
            "treatment": plan["frozen_design"]["harness"],
            "images": {
                "agent": cert["base_binding"]["agent_image_digest"],
                "proxy": cert["base_binding"]["proxy_image_digest"],
            },
            "pass_k": 4,
            "concurrency": plan["frozen_design"]["concurrency_per_worker"],
            "training_data_eligible": False,
            "automatic_retry": False,
            "runtime_files": plan["frozen_design"]["transport_runtime_files"],
            "sampling": {"temperature": 0.6, "top_p": 0.95, "seed": 42},
            "interpretation": "serving-block descriptive evaluation",
        }
    )


def evaluator_preflight(plan: dict, evaluator: dict) -> dict:
    task_set = load(ROOT / plan["references"]["task_set"]["path"])
    bindings = {
        row["task_version_id"]: [
            row["exact_binding"]["task"],
            row["exact_binding"]["environment"],
            row["exact_binding"]["verifier"],
        ]
        for row in task_set["tasks"]
    }
    routes = {}
    for name, route in evaluator["routes"].items():
        model = evaluator["models"][route["model"]]
        routes[name] = {
            "served_id": route["served_id"],
            "revision": model["revision"],
            "profile_sha256": raw_digest(route),
            "ready": True,
        }
    return raw_seal(
        {
            "schema": "cyber_fleet_eval_preflight_v1",
            "plan_sha256": evaluator["sha256"],
            "task_bindings": bindings,
            "routes": routes,
            "images": evaluator["images"],
        }
    )


def transport(plan: dict, cert: dict) -> dict:
    evaluator = evaluator_plan(plan, cert)
    return acquisition.assemble_transport(
        plan,
        cert,
        evaluator,
        evaluator_preflight(plan, evaluator),
        root=ROOT,
    )


def preflight(plan: dict, cert: dict, guard: dict) -> tuple[dict, datetime]:
    now = datetime(2026, 9, 12, 12, 5, tzinfo=UTC)
    value = acquisition.assemble_preflight(
        plan,
        cert,
        guard,
        observed_at=(now - timedelta(minutes=1)).isoformat(),
        expires_at=(now + timedelta(minutes=10)).isoformat(),
        root=ROOT,
    )
    return value, now


def child_bundle(tmp_path: Path) -> tuple[dict, dict, dict, dict, datetime]:
    plan = load(PLAN_PATH)
    cert = certificate(plan)
    guard = transport(plan, cert)
    check, now = preflight(plan, cert, guard)
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    child = acquisition.assemble_child(
        plan,
        cert,
        guard,
        check,
        campaign_id=guard["evaluator_plan"]["campaign_id"],
        private_results_root=private,
        claim_journal=private / "claims.jsonl",
        root=ROOT,
        now=now,
    )
    return plan, cert, guard, check, child


def private_completion(child: dict, *, valid: int = 80) -> dict:
    return seal(
        {
            "schema": acquisition.PRIVATE_COMPLETION_SCHEMA,
            "child_sha256": child["sha256"],
            "terminal_classification": "valid_outcome_set",
            "terminal_evidence_sha256": "sha256:" + "c" * 64,
            "raw_result_manifest_sha256": "sha256:" + "d" * 64,
            "private_results_root": child["execution"]["private_results_root"],
            "task_seed_pair_count": 80,
            "valid_outcome_count": valid,
            "infrastructure_invalid_count": 0,
            "interrupted_or_unknown_count": 0,
            "resources_released": True,
            "outcomes_exposed": False,
            "wandb_exported": False,
        }
    )


def pair_receipt(plan: dict, cert: dict, outcome_seal: dict, now: datetime) -> dict:
    base = copy.deepcopy(cert["base_binding"])
    base.update(
        {
            "staging_receipt_sha256": "sha256:" + "1" * 64,
            "serving_registration_receipt_sha256": "sha256:" + "2" * 64,
        }
    )
    post = {
        "checkpoint_seal_sha256": "sha256:" + "3" * 64,
        "export_receipt_sha256": "sha256:" + "4" * 64,
        "weights_manifest_sha256": "sha256:" + "5" * 64,
        "tokenizer_manifest_sha256": base["tokenizer_manifest_sha256"],
        "chat_template_sha256": base["chat_template_sha256"],
        "staging_receipt_sha256": "sha256:" + "6" * 64,
        "serving_registration_receipt_sha256": "sha256:" + "7" * 64,
        "served_model_id": "qwen38-post-sft",
        "serving_route_profile_sha256": "sha256:" + "8" * 64,
        "agent_image_digest": base["agent_image_digest"],
        "proxy_image_digest": base["proxy_image_digest"],
    }
    observations = {
        key: "sha256:" + f"{index:064x}"[-64:]
        for index, key in enumerate(
            (
                "normalized_tool_request_sha256",
                "normalized_logit_request_sha256",
                "tokenizer_probe_sha256",
                "base_response_manifest_sha256",
                "post_response_manifest_sha256",
            ),
            10,
        )
    }
    observations.update(
        {
            "window_seconds": 120,
            "non_scored_completion_requests": 2,
            "task_instance_creates": 0,
            "grading_requests": 0,
        }
    )
    return seal(
        {
            "schema": acquisition.PAIR_RECEIPT_SCHEMA,
            "status": "passed",
            "plan_sha256": plan["sha256"],
            "acquisition_certificate_sha256": cert["sha256"],
            "base_outcome_seal_sha256": outcome_seal["sha256"],
            "observed_at": now.isoformat(),
            "base_serving": base,
            "post_sft": post,
            "base_runtime": cert["runtime"],
            "post_runtime": cert["runtime"],
            "checks": {
                "post_checkpoint_export_accepted": True,
                "post_staging_create_once": True,
                "base_and_post_registration_current": True,
                "base_acquisition_runtime_revalidated": True,
                "tokenizer_encode_decode_match": True,
                "structured_tool_call_both": True,
                "fixed_logit_probe_finite_both": True,
                "fixed_logit_probe_deterministic_within_arm": True,
                "context_compaction_and_autocontinue_both": True,
                "same_runtime_except_weights": True,
                "all_80_base_outcomes_still_sealed": True,
            },
            "observations": observations,
        }
    )


def candidate_child(plan: dict, pair: dict, private: Path) -> dict:
    private.mkdir(mode=0o700)
    pair_sha256 = pair["sha256"]
    return seal(
        {
            "schema": study_sealing.CHILD_SCHEMA,
            "study_id": "qwen-split-a-study",
            "arm_id": "teacher-lr3e6-v1",
            "blinded_evaluation_id": "blind-fleet-dev-001",
            "surface": "fleet_dev",
            "parent_protocol": plan["references"]["parent_protocol"],
            "base_control": plan["references"]["base_control"],
            "bindings": {
                "base": {
                    **pair["base_serving"],
                    "live_pair_parity_receipt_sha256": pair_sha256,
                },
                "post_sft": {
                    **pair["post_sft"],
                    "live_parity_receipt_sha256": pair_sha256,
                },
                "matched_runtime": {
                    **pair["base_runtime"],
                    "live_pair_parity_receipt_sha256": pair_sha256,
                },
            },
            "execution": {
                "campaign_id": "chris-q38-split-a-paired-v1",
                "private_results_root": str(private),
                "claim_journal": str(private / "claims.jsonl"),
            },
            "result_policy": copy.deepcopy(study_sealing.RESULT_POLICY),
            "launchable": True,
        }
    )


def test_plan_freezes_exact_80_cell_opencode_treatment() -> None:
    plan = load(PLAN_PATH)
    acquisition.validate_plan(plan, root=ROOT)
    frozen = plan["frozen_design"]
    assert frozen["harness"]["harness_version"] == "1.18.27"
    assert frozen["attempt_seeds"] == [42, 43, 44, 45]
    assert len(frozen["task_bindings"]) == 20
    assert frozen["task_seed_pair_count"] == 80
    assert all(
        {
            "prompt_sha256",
            "verifier_sha256",
            "runtime_seed_content_sha256",
        }
        <= row.keys()
        for row in frozen["task_bindings"]
    )


def test_base_certificate_does_not_require_or_authorize_a_post_arm() -> None:
    plan = load(PLAN_PATH)
    cert = certificate(plan)
    assert cert["post_sft_or_live_pair_evidence_included"] is False
    assert cert["paid_or_scored_work_authorized"] is False
    assert cert["task_seed_pair_count"] == 80


def test_base_certificate_rejects_unrelated_or_unqualified_agent_image() -> None:
    plan = load(PLAN_PATH)
    qualification = agent_image_qualification(plan)
    harness = harness_receipt(plan, qualification)
    harness["agent_image_digest"] = "registry.test/opencode@sha256:" + "c" * 64
    harness = seal(harness)
    with pytest.raises(ValueError, match="frozen acquisition treatment"):
        acquisition.assemble_acquisition_certificate(
            plan,
            load(PAYLOAD_PATH),
            load(ROUTE_PATH),
            harness,
            qualification,
            root=ROOT,
        )

    tampered = copy.deepcopy(qualification)
    tampered["publication"]["rewrite_timestamp"] = False
    tampered = seal(tampered)
    with pytest.raises(ValueError, match="frozen build"):
        acquisition.assemble_acquisition_certificate(
            plan,
            load(PAYLOAD_PATH),
            load(ROUTE_PATH),
            harness_receipt(plan, tampered),
            tampered,
            root=ROOT,
        )


def test_transport_restores_every_hash_the_generic_plan_drops() -> None:
    plan = load(PLAN_PATH)
    cert = certificate(plan)
    guard = transport(plan, cert)
    acquisition.validate_transport(plan, cert, guard, root=ROOT)

    tampered = copy.deepcopy(guard)
    version = next(iter(tampered["evaluator_preflight"]["task_bindings"]))
    tampered["evaluator_preflight"]["task_bindings"][version][0]["prompt_sha256"] = (
        "sha256:" + "f" * 64
    )
    tampered["evaluator_preflight"] = raw_seal(tampered["evaluator_preflight"])
    tampered = seal(tampered)
    with pytest.raises(ValueError, match="dropped a frozen"):
        acquisition.validate_transport(plan, cert, tampered, root=ROOT)


def test_child_is_base_only_private_and_exactly_80_cells(tmp_path: Path) -> None:
    plan, cert, guard, check, child = child_bundle(tmp_path)
    acquisition.validate_child(plan, cert, guard, check, child, root=ROOT)
    assert child["task_seed_pair_count"] == 80
    assert child["bindings"]["post_sft"] is None
    assert child["execution_transport"]["candidate_sessions"] == 0
    assert child["paid_or_scored_work_authorized_by_certificate"] is False


def test_base_outcome_seal_requires_all_80_and_discloses_only_digests(
    tmp_path: Path,
) -> None:
    plan, cert, guard, check, child = child_bundle(tmp_path)
    with pytest.raises(ValueError, match="complete frozen 80"):
        acquisition.seal_base_outcomes(
            plan,
            cert,
            guard,
            check,
            child,
            private_completion(child, valid=79),
            root=ROOT,
        )
    outcome = acquisition.seal_base_outcomes(
        plan,
        cert,
        guard,
        check,
        child,
        private_completion(child),
        root=ROOT,
    )
    assert outcome["all_frozen_pairs_valid"] is True
    assert outcome["outcomes_remaining_private"] is True
    assert "score" not in json.dumps(outcome).lower()


def test_later_reuse_reproves_full_pair_parity(tmp_path: Path) -> None:
    plan, cert, guard, check, child = child_bundle(tmp_path)
    outcome = acquisition.seal_base_outcomes(
        plan,
        cert,
        guard,
        check,
        child,
        private_completion(child),
        root=ROOT,
    )
    now = datetime(2026, 9, 12, 13, 0, tzinfo=UTC)
    pair = pair_receipt(plan, cert, outcome, now)
    candidate = candidate_child(plan, pair, tmp_path / "candidate-private")
    final = acquisition.assemble_matched_pair_certificate(
        plan,
        cert,
        guard,
        check,
        child,
        outcome,
        pair,
        candidate,
        root=ROOT,
        now=now + timedelta(minutes=1),
    )
    assert final["only_scientific_difference"] == "weights_manifest_sha256"
    assert final["paid_or_scored_work_authorized"] is False

    drifted = copy.deepcopy(pair)
    drifted["post_runtime"]["tool_call_parser"] = "changed"
    drifted = seal(drifted)
    with pytest.raises(ValueError, match="runtime differs"):
        acquisition.assemble_matched_pair_certificate(
            plan,
            cert,
            guard,
            check,
            child,
            outcome,
            drifted,
            candidate,
            root=ROOT,
            now=now + timedelta(minutes=1),
        )


def test_fresh_pair_receipt_is_mandatory(tmp_path: Path) -> None:
    plan, cert, guard, check, child = child_bundle(tmp_path)
    outcome = acquisition.seal_base_outcomes(
        plan,
        cert,
        guard,
        check,
        child,
        private_completion(child),
        root=ROOT,
    )
    now = datetime(2026, 9, 12, 13, 0, tzinfo=UTC)
    pair = pair_receipt(plan, cert, outcome, now)
    candidate = candidate_child(plan, pair, tmp_path / "candidate-private")
    with pytest.raises(ValueError, match="not fresh"):
        acquisition.assemble_matched_pair_certificate(
            plan,
            cert,
            guard,
            check,
            child,
            outcome,
            pair,
            candidate,
            root=ROOT,
            now=now + timedelta(minutes=16),
        )
