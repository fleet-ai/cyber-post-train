from __future__ import annotations

from pathlib import Path
from typing import Any

from evals.fleet import autocontinue_canary_controller as canary
from evals.fleet import self_hosted

ROUTE_PATH = "docs/evidence/qwen38-study/2026-09-04-opencode-autocontinue-hosted-only-route-v1.json"
ROUTE_SHA = "sha256:d3a285c09f0abb1f66d6ab692193f5ad16c3cd1e432be87f134ae89977d5d00c"
PACKAGE_COMMIT = "9c93095f60d627c239fe2d01f35efd918f328fe6"

BUNDLES = {
    "qwen38_autocontinue_canary": {
        "preauth_path": (
            "docs/evidence/qwen38-study/"
            "2026-09-04-qwen38-autocontinue-canary-preflight-authorization-v2.json"
        ),
        "preauth_sha": "sha256:d1b764cd8f96bbe5d36a62956ea32931066740c8c77b8ac60c5474c7209f3720",
        "preflight_path": (
            "docs/evidence/qwen38-study/"
            "2026-09-04-qwen38-autocontinue-canary-preflight-v2-pass.json"
        ),
        "preflight_sha": "sha256:21e459b1a1773266fb3e39b12d5cdaa727e0ecd6848173ac4f0840d339579790",
        "preflight_sfs_file_sha": (
            "sha256:1d3f54e862e8891447a82c685b84ddfb508e4998f2133e5fa8873c02d236d884"
        ),
        "post_exit_path": (
            "docs/evidence/qwen38-study/"
            "2026-09-04-qwen38-autocontinue-canary-preflight-v2-post-exit.json"
        ),
        "post_exit_sha": "sha256:6e3cbbc1ea2a7532c06e3d7fc521976ed08c4c64b69f829da129c8bb437d4b09",
        "duplicate_path": (
            "docs/evidence/qwen38-study/"
            "2026-09-04-qwen38-autocontinue-canary-duplicate-inventory-v2.json"
        ),
        "duplicate_sha": "sha256:bcbe6e19fd0e2c5fbe6aaec995ac59afad114abdc8fa04133b1ad63251259345",
        "job_uid": "4a90d77e-c89c-4c33-970b-5dd2e4e9696e",
        "pod_uid": "ac946812-36a6-43b5-81e2-fe7638daa4d5",
        "pod_name": "chris-q38-ac-canary1-v2-preflight-nhqcq",
        "configmap_uid": "c477496a-424b-48b4-8fbf-f093295730f0",
        "job_created_at": "2026-09-04T21:36:59Z",
        "pod_started_at": "2026-09-04T21:37:00Z",
        "configmap_created_at": "2026-09-04T21:36:55Z",
    },
    "glm53_autocontinue_canary": {
        "preauth_path": (
            "docs/evidence/qwen38-study/"
            "2026-09-04-glm53-autocontinue-canary-preflight-authorization-v2.json"
        ),
        "preauth_sha": "sha256:a7cef96859a38ace334d5edbef22ffa1d87ed74d5c1d216857945035bf07f530",
        "preflight_path": (
            "docs/evidence/qwen38-study/2026-09-04-glm53-autocontinue-canary-preflight-v2-pass.json"
        ),
        "preflight_sha": "sha256:e66c09c09953c1220d8752f5ba43419e2ed0d934c075eea1bf156c47911879ba",
        "preflight_sfs_file_sha": (
            "sha256:219c7ae02c461af77a588de03ddc263b7f6caecdcd74f7763aa2083df16fc432"
        ),
        "post_exit_path": (
            "docs/evidence/qwen38-study/"
            "2026-09-04-glm53-autocontinue-canary-preflight-v2-post-exit.json"
        ),
        "post_exit_sha": "sha256:b22d011cf592786382e0ae2f729fe709673ab211b970ce9ff19b25d79bfa80a5",
        "duplicate_path": (
            "docs/evidence/qwen38-study/"
            "2026-09-04-glm53-autocontinue-canary-duplicate-inventory-v2.json"
        ),
        "duplicate_sha": "sha256:57f84d6bf2ec923d825322038017241a6633e9a4932a5ebf904a9f0c9c1627de",
        "job_uid": "a50d1348-2c66-4b7b-bccc-210d98ad3721",
        "pod_uid": "e15d615f-099f-4cfe-89f7-a75aab32b4b9",
        "pod_name": "chris-glm53-ac-canary1-v2-preflight-wvrxr",
        "configmap_uid": "f7e47004-2077-4721-9202-88255c2cbdfd",
        "job_created_at": "2026-09-04T21:37:00Z",
        "pod_started_at": "2026-09-04T21:37:00Z",
        "configmap_created_at": "2026-09-04T21:36:58Z",
    },
}

OBSERVED_AT = "2026-09-04T21:39:55Z"
INTENT_CONFIGMAP = {
    "name": "chris-ac-canary1-preflight-submit-v2",
    "uid": "9d20fa32-10c0-4e31-a0e8-18113d869e45",
    "created_at_utc": "2026-09-04T21:36:52Z",
    "immutable": True,
}
UV_REQUESTED_IMAGE = (
    "ghcr.io/astral-sh/uv:python3.12-bookworm@"
    "sha256:9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7"
)
UV_RESOLVED_IMAGE = (
    "ghcr.io/astral-sh/uv@sha256:9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7"
)
PRIVACY = {
    "credentials_included": False,
    "prompts_or_traces_included": False,
    "scores_included": False,
}


def _load_exact(root: Path, path: str, receipt_sha: str) -> dict[str, Any]:
    value = canary.load_object(root / path)
    if value.get("receipt_sha256") != receipt_sha or value.get(
        "receipt_sha256"
    ) != canary.digest_without(value, "receipt_sha256"):
        raise ValueError("hosted-only canary evidence digest drifted")
    return value


def validate_hosted_route(receipt: dict[str, Any]) -> None:
    expected = {
        "schema_version": "fleet-opencode-autocontinue-hosted-only-route-v1",
        "append_only": True,
        "status": "PASSED",
        "observed_at_utc": OBSERVED_AT,
        "campaign_sha256": canary.CAMPAIGN_SHA,
        "fleet_team": {
            "team_id": self_hosted.FLEET_TEAM_ID,
            "qwen_preflight_receipt_sha256": BUNDLES["qwen38_autocontinue_canary"]["preflight_sha"],
            "glm_preflight_receipt_sha256": BUNDLES["glm53_autocontinue_canary"]["preflight_sha"],
            "both_preflights_authenticated_as_fleet": True,
        },
        "hosted_routes": {
            "health_receipt_sha256": canary.HOSTED_HEALTH_SHA,
            "origin": "https://inference.flt.build",
            "qwen_served_id": "qwen3.8-27b",
            "glm_served_id": "glm-5.3",
            "qwen_route_available_in_bound_health_receipt": True,
            "glm_route_available_in_bound_health_receipt": True,
            "generation_calls": 0,
        },
        "dedicated_serving": {
            "state": "USER_STOPPED_UNAVAILABLE",
            "may_be_recreated_without_ready_scored_consumer_and_new_parity": False,
            "rayjobs_present": 0,
            "head_pods_present": 0,
            "services_present": 0,
            "replica_a": {
                "rayjob_name": "ft-run-98c32208",
                "head_pod_name": "ft-run-98c32208-5gpb2-head-7qxwc",
                "service_name": "ft-run-98c32208-5gpb2-head-svc",
            },
            "replica_b": {
                "rayjob_name": "ft-run-9e92209d",
                "head_pod_name": "ft-run-9e92209d-pppzg-head-rc9nd",
                "service_name": "ft-run-9e92209d-pppzg-head-svc",
            },
            "historical_parity_receipt_is_not_current_live_route_authority": True,
        },
        "candidate_absence": {
            "qwen_scored_job_name": "chris-q38-ac-canary1-v1",
            "glm_scored_job_name": "chris-glm53-ac-canary1-v1",
            "qwen_scored_configmap_name": "chris-q38-ac-canary1-run-v2",
            "glm_scored_configmap_name": "chris-glm53-ac-canary1-run-v2",
            "exact_scored_jobs_present": 0,
            "exact_scored_pods_present": 0,
            "exact_scored_configmaps_present": 0,
            "qwen_scored_sfs_root_absent": True,
            "glm_scored_sfs_root_absent": True,
            "qwen_global_cell_claim_absent": True,
            "glm_global_cell_claim_absent": True,
            "sfs_observer_pod_name": "allie-dev",
            "sfs_observer_pod_uid": "73dabe56-60f8-4879-be9f-365196c502e3",
        },
        "authorization": {
            "route": "HOSTED_ONLY",
            "scored_launch_authorized": False,
            "dedicated_recreation_authorized": False,
        },
        "privacy": PRIVACY,
    }
    if (
        receipt.get("receipt_sha256") != ROUTE_SHA
        or receipt.get("receipt_sha256") != canary.digest_without(receipt, "receipt_sha256")
        or {key: value for key, value in receipt.items() if key != "receipt_sha256"} != expected
    ):
        raise ValueError("hosted-only canary route evidence is not authoritative")


def validate_preflight_bundle(plan: dict[str, Any], root: Path) -> dict[str, Any]:
    canary.validate_plan(plan)
    expected = canary.EXPECTED[plan["shard_key"]]
    bundle = BUNDLES[plan["shard_key"]]
    preauth = _load_exact(root, bundle["preauth_path"], bundle["preauth_sha"])
    canary.validate_preflight_authorization(
        preauth,
        plan,
        root,
        PACKAGE_COMMIT,
        expected["plan_file_sha256"],
        preauth["authorized_at_utc"],
        preauth["authorization"]["statement"],
    )
    preflight = _load_exact(root, bundle["preflight_path"], bundle["preflight_sha"])
    post_exit = _load_exact(root, bundle["post_exit_path"], bundle["post_exit_sha"])
    duplicate = _load_exact(root, bundle["duplicate_path"], bundle["duplicate_sha"])
    route = _load_exact(root, ROUTE_PATH, ROUTE_SHA)
    validate_hosted_route(route)
    expected_cell = {
        "source_rank": expected["source_rank"],
        "attempt": 1,
        "task_version_id": expected["task_version_id"],
    }
    expected_preflight = {
        "schema_version": "fleet-opencode-autocontinue-canary-preflight-v1",
        "status": "PASSED",
        "plan_sha256": plan["plan_sha256"],
        "release_receipt_sha256": bundle["preauth_sha"],
        "fleet_team_id": self_hosted.FLEET_TEAM_ID,
        "current_plan_run_and_claim_identities_absent": True,
        "output_root_absent": True,
        "sfs_job_roots_reconciled": 99,
        "exact_treatment_sessions_reconciled": 0,
        "job_uid": bundle["job_uid"],
        "pod_uid": bundle["pod_uid"],
        "scores_read": False,
        "prompts_or_traces_read": False,
    }
    expected_post_exit = {
        "schema_version": "fleet-opencode-autocontinue-canary-preflight-post-exit-v1",
        "append_only": True,
        "status": "PASSED",
        "observed_at_utc": OBSERVED_AT,
        "campaign_sha256": canary.CAMPAIGN_SHA,
        "plan_sha256": plan["plan_sha256"],
        "preflight_authorization_receipt_sha256": bundle["preauth_sha"],
        "preflight_receipt_sha256": bundle["preflight_sha"],
        "preflight_sfs_file_sha256": bundle["preflight_sfs_file_sha"],
        "hosted_only_route_receipt_sha256": ROUTE_SHA,
        "job": {
            "name": expected["preflight_job_v2"],
            "uid": bundle["job_uid"],
            "created_at_utc": bundle["job_created_at"],
            "completion_time": "2026-09-04T21:37:20Z",
            "completion_reason": "CompletionsReached",
            "succeeded": 1,
            "failed": 0,
            "priority_class": canary.PRIORITY,
            "priority_value": 10000,
            "preemption_policy": "PreemptLowerPriority",
            "queue": "training-lq",
        },
        "pod": {
            "name": bundle["pod_name"],
            "uid": bundle["pod_uid"],
            "owner_job_name": expected["preflight_job_v2"],
            "owner_job_uid": bundle["job_uid"],
            "owner_is_controller": True,
            "phase": "Succeeded",
            "exit_code": 0,
            "restart_count": 0,
            "started_at": bundle["pod_started_at"],
            "finished_at": "2026-09-04T21:37:18Z",
            "requested_image": UV_REQUESTED_IMAGE,
            "resolved_image": UV_RESOLVED_IMAGE,
        },
        "configmap": {
            "name": expected["preflight_configmap_v2"],
            "uid": bundle["configmap_uid"],
            "created_at_utc": bundle["configmap_created_at"],
            "immutable": True,
        },
        "intent_configmap": INTENT_CONFIGMAP,
        "reconciliation": {
            "fleet_team_id": self_hosted.FLEET_TEAM_ID,
            "current_plan_run_and_claim_identities_absent": True,
            "exact_treatment_sessions_reconciled": 0,
            "sfs_job_roots_reconciled": 99,
            "scored_job_created": False,
            "scored_configmap_created": False,
            "scored_sfs_root_absent": True,
            "global_cell_claim_absent": True,
        },
        "privacy": PRIVACY,
    }
    expected_duplicate = {
        "schema_version": "fleet-opencode-autocontinue-canary-duplicate-inventory-v1",
        "append_only": True,
        "status": "PASSED",
        "observed_at_utc": OBSERVED_AT,
        "fleet_team_id": self_hosted.FLEET_TEAM_ID,
        "pagination_exhausted": True,
        "plan_sha256": plan["plan_sha256"],
        "cell": expected_cell,
        "source_preflight_receipt_sha256": bundle["preflight_sha"],
        "post_exit_receipt_sha256": bundle["post_exit_sha"],
        "exact_treatment_session_rows": 0,
        "current_plan_run_rows": 0,
        "active_attempts": 0,
        "global_cell_claim_absent": True,
        "job_pod_and_sfs_identities_absent": True,
        "privacy": PRIVACY,
    }
    if (
        {key: value for key, value in preflight.items() if key != "receipt_sha256"}
        != expected_preflight
        or {key: value for key, value in post_exit.items() if key != "receipt_sha256"}
        != expected_post_exit
        or {key: value for key, value in duplicate.items() if key != "receipt_sha256"}
        != expected_duplicate
    ):
        raise ValueError("hosted-only canary preflight bundle is not authoritative")
    return {
        "preauth": preauth,
        "preflight": preflight,
        "post_exit": post_exit,
        "duplicate": duplicate,
        "route": route,
    }
