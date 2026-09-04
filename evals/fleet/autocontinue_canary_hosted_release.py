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
    "qwen38_autocontinue_canary_v2": {
        "successor": True,
        "package_commit": "f2c32e4571fddfa43770249f323e1843fc205f95",
        "observed_at": "2026-09-04T22:57:09Z",
        "preauth_path": (
            "docs/evidence/qwen38-study/"
            "2026-09-04-qwen38-autocontinue-canary-preflight-authorization-v3.json"
        ),
        "preauth_sha": "sha256:2ccba714a6885ff8de4cd13f7a7857df2ccdf75bd8a258be04969e502d1ccf10",
        "preauth_file_sha": (
            "sha256:1059c755a352b5f2320b540094ba7f4e5cc575b7b6ff5af4ca68a601b0f4f80c"
        ),
        "compatibility_sha": (
            "sha256:b3920547030db509de25f3a04559d9f4173866b81b02efa0957812b35cf06ba2"
        ),
        "preflight_path": (
            "docs/evidence/qwen38-study/"
            "2026-09-04-qwen38-autocontinue-canary-preflight-v3-pass.json"
        ),
        "preflight_sha": "sha256:8feaa88ebe0aadf11ff08346023b9d756b89ebc2701b08a9f6d709fbf23c4fe3",
        "preflight_sfs_file_sha": (
            "sha256:aad9ff608f860ecccebb64ea7c3619050d13c99a7b42a34060268e1b932e508f"
        ),
        "post_exit_path": (
            "docs/evidence/qwen38-study/"
            "2026-09-04-qwen38-autocontinue-canary-preflight-v3-post-exit.json"
        ),
        "post_exit_sha": "sha256:42d9e194b2934f6705d68d498b703a4daa9087e6f05671e058ad575726a420d2",
        "duplicate_path": (
            "docs/evidence/qwen38-study/"
            "2026-09-04-qwen38-autocontinue-canary-duplicate-inventory-v3.json"
        ),
        "duplicate_sha": "sha256:662c822282d33d376a0ab1755c6970dc27801caa5e6e92d1d438732a50276be2",
        "job_uid": "97368279-f17a-415f-8533-31f2627e9ae7",
        "pod_uid": "6e29a013-e4de-44da-8724-18631dfe9837",
        "pod_name": "chris-q38-ac-canary1-v3-preflight-dc4sp",
        "configmap_uid": "253e168a-5ae6-49f4-b7f1-16be5c5cb14a",
        "job_created_at": "2026-09-04T22:55:01Z",
        "pod_started_at": "2026-09-04T22:55:02Z",
        "configmap_created_at": "2026-09-04T22:54:57Z",
        "intent_configmap": {
            "name": "chris-ac-canary1-preflight-submit-v3",
            "uid": "3fbc419d-f55e-4117-9a44-38ceb6fb724e",
            "created_at_utc": "2026-09-04T22:54:54Z",
            "immutable": True,
        },
    },
    "glm53_autocontinue_canary_v2": {
        "successor": True,
        "package_commit": "f2c32e4571fddfa43770249f323e1843fc205f95",
        "observed_at": "2026-09-04T22:57:09Z",
        "preauth_path": (
            "docs/evidence/qwen38-study/"
            "2026-09-04-glm53-autocontinue-canary-preflight-authorization-v3.json"
        ),
        "preauth_sha": "sha256:f5766496049bfe8e649e95b2ef450263b461c03b803a7b39b0da8bca63b3ea5c",
        "preauth_file_sha": (
            "sha256:de0d7aec69f8471c5e3b35797de5f7dda723a70b90c97407c8aeb9c21c31f69e"
        ),
        "compatibility_sha": (
            "sha256:b3920547030db509de25f3a04559d9f4173866b81b02efa0957812b35cf06ba2"
        ),
        "preflight_path": (
            "docs/evidence/qwen38-study/"
            "2026-09-04-glm53-autocontinue-canary-preflight-v3-pass.json"
        ),
        "preflight_sha": "sha256:b32da5b2ce836440e2b845f3f960cb77cceaf19a5c6e555ed6418e12c7889603",
        "preflight_sfs_file_sha": (
            "sha256:1a515e446f1c40385b2fd71d2f5e8277024d037315df47da57ce0b7c5cadee8b"
        ),
        "post_exit_path": (
            "docs/evidence/qwen38-study/"
            "2026-09-04-glm53-autocontinue-canary-preflight-v3-post-exit.json"
        ),
        "post_exit_sha": "sha256:858aae67031f22ef42628891ff32224e14ed99d0303a4ed59207d24d869bfe26",
        "duplicate_path": (
            "docs/evidence/qwen38-study/"
            "2026-09-04-glm53-autocontinue-canary-duplicate-inventory-v3.json"
        ),
        "duplicate_sha": "sha256:e4d215278d56b070c59c3eb7af946747280eb93ae93898b5b1e1b7f0a552d9d1",
        "job_uid": "978d013d-95b6-4455-9aac-e406a306f02a",
        "pod_uid": "0bb25778-5bbe-4c59-aafc-9b2f9e039e5e",
        "pod_name": "chris-glm53-ac-canary1-v3-preflight-nmxz8",
        "configmap_uid": "cba581f2-45ab-4985-8802-22dab5eea894",
        "job_created_at": "2026-09-04T22:55:01Z",
        "pod_started_at": "2026-09-04T22:55:02Z",
        "configmap_created_at": "2026-09-04T22:55:00Z",
        "intent_configmap": {
            "name": "chris-ac-canary1-preflight-submit-v3",
            "uid": "3fbc419d-f55e-4117-9a44-38ceb6fb724e",
            "created_at_utc": "2026-09-04T22:54:54Z",
            "immutable": True,
        },
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
SUCCESSOR_HELD_SCHEMA = "fleet-opencode-autocontinue-canary-successor-held-release-v1"
SUCCESSOR_INCIDENT_SHA = "sha256:b30930aeb12cbe4d8ab5ef91a1608ec672e2536576168b50639318b866e30423"
SUCCESSOR_LAUNCHER = "evals/fleet/scripts/submit_opencode_autocontinue_canaries_v2.sh"


def _load_exact(root: Path, path: str, receipt_sha: str) -> dict[str, Any]:
    value = canary.load_object(root / path)
    if value.get("receipt_sha256") != receipt_sha or value.get(
        "receipt_sha256"
    ) != canary.digest_without(value, "receipt_sha256"):
        raise ValueError("hosted-only canary evidence digest drifted")
    return value


def _validate_successor_preauth(
    preauth: dict[str, Any], plan: dict[str, Any], root: Path, bundle: dict[str, Any]
) -> None:
    """Bind the immutable Phase-A authorization without reinterpreting later bytes."""
    expected = canary.EXPECTED[plan["shard_key"]]
    if (
        canary._sha(root / bundle["preauth_path"]) != bundle["preauth_file_sha"]
        or preauth.get("schema_version")
        != "fleet-opencode-autocontinue-canary-preflight-authorization-v3"
        or preauth.get("append_only") is not True
        or preauth.get("status") != "PREFLIGHT_AUTHORIZED"
        or preauth.get("package_commit") != bundle["package_commit"]
        or preauth.get("campaign_sha256") != canary.CAMPAIGN_SHA
        or preauth.get("plan_sha256") != plan["plan_sha256"]
        or preauth.get("cell")
        != {
            "source_rank": expected["source_rank"],
            "attempt": 1,
            "task_version_id": expected["task_version_id"],
        }
        or preauth.get("preflight_identity")
        != {
            "configmap_name": expected["preflight_configmap_v2"],
            "job_name": expected["preflight_job_v2"],
            "sfs_root": f"/mnt/sfs/jobs/{expected['preflight_job_v2']}",
            "scored_configmap_name": expected["scored_configmap"],
            "scored_job_name": plan["scored_job_name"],
            "scored_job_created": False,
        }
        or preauth.get("authorization", {}).get("preflight_authorized") is not True
        or preauth.get("authorization", {}).get("launch_authorized") is not False
        or preauth.get("privacy") != PRIVACY
    ):
        raise ValueError("successor preflight authorization is not authoritative")


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
    successor = bundle.get("successor") is True
    preauth = _load_exact(root, bundle["preauth_path"], bundle["preauth_sha"])
    if successor:
        _validate_successor_preauth(preauth, plan, root, bundle)
    else:
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
    route = None
    if not successor:
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
        "sfs_job_roots_reconciled": 101 if successor else 99,
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
    if successor:
        expected_post_exit = {
            "schema_version": (
                "fleet-opencode-autocontinue-canary-successor-preflight-post-exit-v1"
            ),
            "append_only": True,
            "status": "PASSED",
            "observed_at_utc": bundle["observed_at"],
            "package_commit": bundle["package_commit"],
            "campaign_sha256": canary.CAMPAIGN_SHA,
            "plan_sha256": plan["plan_sha256"],
            "preflight_authorization_receipt_sha256": bundle["preauth_sha"],
            "preflight_receipt_sha256": bundle["preflight_sha"],
            "preflight_sfs_file_sha256": bundle["preflight_sfs_file_sha"],
            "job": {
                "name": expected["preflight_job_v2"],
                "uid": bundle["job_uid"],
                "created_at_utc": bundle["job_created_at"],
                "completion_time": "2026-09-04T22:55:19Z",
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
                "finished_at": "2026-09-04T22:55:17Z",
                "requested_image": UV_REQUESTED_IMAGE,
                "resolved_image": UV_RESOLVED_IMAGE,
            },
            "configmap": {
                "name": expected["preflight_configmap_v2"],
                "uid": bundle["configmap_uid"],
                "created_at_utc": bundle["configmap_created_at"],
                "immutable": True,
            },
            "intent_configmap": bundle["intent_configmap"],
            "reconciliation": {
                "fleet_team_id": self_hosted.FLEET_TEAM_ID,
                "current_plan_run_and_claim_identities_absent": True,
                "exact_treatment_sessions_reconciled": 0,
                "sfs_job_roots_reconciled": 101,
                "scored_job_created": False,
                "scored_configmap_created": False,
                "scored_sfs_root_absent": True,
                "global_cell_claim_absent": True,
                "global_claim_json_count": 0,
                "sfs_observer_pod_name": "allie-dev",
                "sfs_observer_pod_uid": "73dabe56-60f8-4879-be9f-365196c502e3",
                "sfs_mount_path": "/shared",
            },
            "hosted_only": {
                "dedicated_serving_state": "USER_STOPPED_UNAVAILABLE",
                "dedicated_recreation_authorized": False,
                "fresh_authenticated_models_get_required_before_scored_create": True,
                "fresh_authenticated_models_get_required_before_global_claim": True,
            },
            "privacy": {**PRIVACY, "logs_read": False},
        }
        expected_duplicate = {
            "schema_version": (
                "fleet-opencode-autocontinue-canary-successor-duplicate-inventory-v1"
            ),
            "append_only": True,
            "status": "PASSED",
            "observed_at_utc": bundle["observed_at"],
            "fleet_team_id": self_hosted.FLEET_TEAM_ID,
            "pagination_exhausted_by_uid_bound_preflight": True,
            "plan_sha256": plan["plan_sha256"],
            "cell": expected_cell,
            "source_preflight_receipt_sha256": bundle["preflight_sha"],
            "post_exit_receipt_sha256": bundle["post_exit_sha"],
            "exact_treatment_session_rows": 0,
            "current_plan_run_rows": 0,
            "active_attempts": 0,
            "global_cell_claim_absent": True,
            "global_claim_json_count": 0,
            "scored_job_name": plan["scored_job_name"],
            "scored_configmap_name": expected["scored_configmap"],
            "scored_sfs_root": f"/shared/jobs/{plan['sfs_root']}",
            "job_pod_configmap_and_sfs_identities_absent": True,
            "dedicated_serving_state": "USER_STOPPED_UNAVAILABLE",
            "privacy": {**PRIVACY, "logs_read": False},
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


def validate_successor_held_release(
    release: dict[str, Any], plan: dict[str, Any], root: Path
) -> None:
    """Validate a preview-only successor package; this never grants scoring."""
    validate_preflight_bundle(plan, root)
    bundle = BUNDLES[plan["shard_key"]]
    if bundle.get("successor") is not True:
        raise ValueError("held successor release received a legacy plan")
    expected = canary.EXPECTED[plan["shard_key"]]
    expected_value = {
        "schema_version": SUCCESSOR_HELD_SCHEMA,
        "append_only": True,
        "status": "HELD",
        "observed_at_utc": bundle["observed_at"],
        "campaign_sha256": canary.CAMPAIGN_SHA,
        "plan_sha256": plan["plan_sha256"],
        "cell": {
            "source_rank": expected["source_rank"],
            "attempt": 1,
            "task_version_id": expected["task_version_id"],
        },
        "evidence": {
            "preflight_package_commit": bundle["package_commit"],
            "preflight_authorization_path": bundle["preauth_path"],
            "preflight_authorization_sha256": bundle["preauth_sha"],
            "preflight_receipt_path": bundle["preflight_path"],
            "preflight_receipt_sha256": bundle["preflight_sha"],
            "preflight_sfs_file_sha256": bundle["preflight_sfs_file_sha"],
            "preflight_post_exit_path": bundle["post_exit_path"],
            "preflight_post_exit_sha256": bundle["post_exit_sha"],
            "duplicate_inventory_path": bundle["duplicate_path"],
            "duplicate_inventory_sha256": bundle["duplicate_sha"],
            "controller_compatibility_receipt_sha256": bundle["compatibility_sha"],
            "predecessor_scored_bootstrap_failure_receipt_sha256": SUCCESSOR_INCIDENT_SHA,
            "task_inventory_receipt_sha256": canary.TASK_INVENTORY_SHA,
            "task_inventory_execution_sha256": canary.TASK_INVENTORY_EXECUTION_SHA,
            "hosted_health_receipt_sha256": canary.HOSTED_HEALTH_SHA,
            "shared_pvc_flock_receipt_sha256": canary.FLOCK_GATE_SHA,
            "dedicated_parity_receipt_sha256": canary.DEDICATED_PARITY_SHA,
            "dedicated_parity_role": "historical_only_not_current_route_authority",
        },
        "implementation": {
            "executable_package_commit": None,
            "executable_package_commit_required_after_audit": True,
            "plan_file_sha256": canary._sha(root / expected["plan_path"]),
            "controller_sha256": canary._sha(
                root / "evals/fleet/autocontinue_canary_controller.py"
            ),
            "frozen_controller_sha256": canary._sha(
                root / "evals/fleet/hosted_sweep_controller.py"
            ),
            "phase_c_validator_sha256": canary._sha(
                root / "evals/fleet/autocontinue_canary_hosted_release.py"
            ),
            "hosted_runtime_sha256": canary._sha(
                root / "evals/fleet/autocontinue_canary_hosted_runtime.py"
            ),
            "hosted_health_sha256": canary._sha(
                root / "evals/fleet/autocontinue_hosted_health.py"
            ),
            "manifest_authorization_sha256": canary._sha(
                root / "evals/fleet/autocontinue_successor_manifest_authorization.py"
            ),
            "self_hosted_sha256": canary._sha(root / "evals/fleet/self_hosted.py"),
            "runner_sha256": canary._sha(
                root / "evals/fleet/opencode_train_sweep_runner.py"
            ),
            "endpoint_lease_sha256": canary._sha(root / "evals/fleet/endpoint_lease.py"),
            "fixed_proxy_sha256": canary._sha(root / "evals/fleet/fixed_proxy.py"),
            "dockerfile_sha256": canary._sha(root / "evals/fleet/Dockerfile.opencode"),
            "campaign_file_sha256": canary._sha(
                root
                / "evals/fleet/configs/q38-glm53-opencode-autocontinue-primary-campaign-v1.json"
            ),
            "compatibility_file_sha256": canary._sha(
                root
                / "docs/evidence/qwen38-study/"
                "2026-09-04-opencode-autocontinue-canary-controller-compatibility-v3.json"
            ),
            "preflight_manifest_sha256": canary._sha(
                root / "evals/fleet/cluster/opencode-autocontinue-canary-preflights-v3.yaml"
            ),
            "scored_manifest_sha256": canary._sha(
                root
                / "evals/fleet/cluster/opencode-autocontinue-canary-successor-scored-v4.yaml"
            ),
            "run_script_sha256": canary._sha(
                root / "evals/fleet/scripts/run_opencode_autocontinue_canary.sh"
            ),
            "launcher_sha256": canary._sha(root / SUCCESSOR_LAUNCHER),
        },
        "scored_identity": {
            "configmap_name": expected["scored_configmap"],
            "job_name": plan["scored_job_name"],
            "sfs_root": f"/mnt/sfs/jobs/{plan['sfs_root']}",
            "priority_class": canary.PRIORITY,
            "priority_class_is_not_preemption_immunity": True,
            "create_once_required": True,
        },
        "route": {
            "serving_route": "HOSTED_ONLY",
            "required_served_ids": ["qwen3.8-27b", "glm-5.3"],
            "dedicated_serving_state": "USER_STOPPED_UNAVAILABLE",
            "dedicated_recreation_authorized": False,
            "fresh_authenticated_models_get_required_before_scored_create": True,
            "fresh_authenticated_models_get_required_before_global_claim": True,
        },
        "authorization": {
            "launch_authorized": False,
            "bulk_release_authorized": False,
            "root_scoring_authorization_required": True,
        },
        "remaining_gates": [
            "immutable_executable_package_commit",
            "fresh_authenticated_hosted_route_within_120_seconds_of_create",
            "fresh_api_kubernetes_sfs_and_global_claim_inventory",
            "model_specific_released_receipt_with_root_authorization",
            "runtime_hosted_route_recheck_before_any_permanent_claim",
            "post_exit_uid_bound_acceptance_observer",
        ],
        "privacy": {**PRIVACY, "logs_read": False},
    }
    if (
        release.get("receipt_sha256") != canary.digest_without(release, "receipt_sha256")
        or {key: value for key, value in release.items() if key != "receipt_sha256"}
        != expected_value
    ):
        raise ValueError("successor held release is not authoritative")
