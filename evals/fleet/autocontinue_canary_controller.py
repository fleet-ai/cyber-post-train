"""Fail-closed one-cell controller for the corrected OpenCode canaries.

This module deliberately leaves the immutable primary campaign and its frozen
hosted controller untouched.  It reuses only the audited attempt/inventory
primitives and adds canary-specific plan, release, and terminal accounting.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import stat
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.fleet import endpoint_lease, self_hosted
from evals.fleet import hosted_sweep_controller as hosted

CAMPAIGN_SHA = "sha256:63946f224a33eb0d2c2a6fdba34358ebf9ca379e5f0f156cd137c95a9b98097e"
CAMPAIGN_FILE_SHA = "sha256:1dfcc6d9ddbd19ef5ff46b5d6308d62526db4d1a0ccf05c5fa3b10b176910f34"
FLOCK_GATE_SHA = "sha256:e9f34d35ac3c9e60d2698ebc1685374b98f13e41ac305df0609fb051f35c416c"
HOSTED_HEALTH_SHA = "sha256:d81a01ffe1087d7d85fe511bc9c978461ba3e24a2c15675871cfb8668c49bd85"
DEDICATED_PARITY_SHA = "sha256:07e733b4b161ed7c36113fc3688f04a42938d93a499b1bcc10c7cd21342900df"
TASK_INVENTORY_SHA = "sha256:c71604d22b1d52a7091727b957e0f56f2e8e270e461c844fb57d48b392d831c5"
TASK_INVENTORY_EXECUTION_SHA = (
    "sha256:b4c175db0d04746e16fa9314edb093f7306e859545cd95972694e461b7f82fcb"
)
V1_PREFLIGHT_FAILURE_SHA = "sha256:c07804a36065a68cc821b0c57e7989f7e9caa077142969046b4bff913a592ee4"
PRE_MANIFEST_SHA = "sha256:979071556343cfa23373a94742fefa684a9d9e0e39ecbe1f2bea05c58dce998a"
SCORED_MANIFEST_SHA = "sha256:1217666a0bfe5ab4d9aec0191213a3f1af8967f053b046922de656ed4a953be2"
PRE_MANIFEST_V3_SHA = "sha256:44262e7dad2329219480f9e52b19064fab3c59d67f9c174fd11e7e058beb4f12"
SCORED_MANIFEST_V3_SHA = "sha256:f7402d99d575a1f1ce8f4c6073de0fe6decd58a289d013f375bb27519ea2fb22"
SCORED_V1_FAILURE_SHA = "sha256:b30930aeb12cbe4d8ab5ef91a1608ec672e2536576168b50639318b866e30423"
SELF_HOSTED_SHA = "sha256:16df432b5fde55112924d6106e6c03f09817846f1344ba0fe100dcf785c33d8b"
RUNNER_SHA = "sha256:b1f9c5028f65b0d7772538e3ce075310dc6c7a46b3de58d0196bc474e74e9e9d"
ENDPOINT_LEASE_SHA = "sha256:1df60ee13be8c6057113dbebadf9020343649e175b5de38aea41706748987019"
LEASE_ROOT = "/mnt/sfs/endpoint-leases/opencode11827-autocontinue-primary-v1"
CONTEXT = "opencode_1.18.27_native_compaction_autocontinue_v1"
PRIORITY = "fleet-train-high"
PLAN_SCHEMA = "fleet-hosted-opencode-task-boundary-shard-v1"
RELEASE_SCHEMA = "fleet-opencode-autocontinue-canary-scoring-release-v2"
TERMINAL_SCHEMA = "fleet-opencode-autocontinue-canary-terminal-v2"
POST_EXIT_SCHEMA = "fleet-opencode-autocontinue-canary-post-exit-v1"
CELL_CLAIM_ROOT = "/mnt/sfs/cell-claims/opencode11827-autocontinue-primary-v1"

EXPECTED = {
    "qwen38_autocontinue_canary": {
        "plan_sha256": "sha256:5b2c5c2792de233d70cc4cb5d1c2b5482807c3745090939bcb426227089ff3b3",
        "campaign_id": "chris-q38-ac-canary1-v1",
        "source_rank": 4,
        "attempt": 1,
        "task_version_id": "02dd4e3f-d85d-4bf8-9976-eae2f102384d",
        "served_id": "qwen3.8-27b",
        "session_model": "fleet-cluster-opencode-1.18.27/qwen3.8-27b-opencode11827-autocontinue-v1",
        "serving_block": "qwen-hosted-autocontinue-v1",
        "settings_file_sha256": (
            "sha256:1328f6eb97861443b712625d9038a924de9c220c146667565dd4c5bb73a6d8f8"
        ),
        "preflight_configmap": "chris-q38-ac-canary1-pre-v1",
        "preflight_configmap_v2": "chris-q38-ac-canary1-pre-v2",
        "preflight_job_v2": "chris-q38-ac-canary1-v2-preflight",
        "scored_configmap": "chris-q38-ac-canary1-run-v2",
        "plan_path": "evals/fleet/configs/qwen38-opencode-autocontinue-canary1-v1.json",
        "plan_file_sha256": (
            "sha256:28d3e45ae8ef317506d5414af89f4d3f7f38e3ed3762eccbe57cca3a45728ebc"
        ),
        "preflight_statement": (
            "I authorize the create-once read-only Qwen corrected-treatment canary preflight "
            "only; no scored Job or scored ConfigMap may be created."
        ),
        "model_sha256": "sha256:bdda482324e070f7051533d2a2f3e7e6c2883a02f70ba0212bf6d390ee61a584",
        "harness_sha256": "sha256:a8b47884934b2ffc24af836f0f5d8d889992ce68066da6c0774196f56b92508a",
        "treatment_sha256": (
            "sha256:5093fcd0cfc92c904146b47fd65682c1de6d38efc4699e6099a1f735c4fba685"
        ),
        "tasks_sha256": "sha256:b8e077dc94ca5c3b214f82e12c8b8172cd2c67e6089adbb6ff5e61468c5495d9",
        "attempts_sha256": (
            "sha256:75980a8914aed26b132a35597b6fa86798220599dec78577466145911e0d1603"
        ),
        "execution_sha256": (
            "sha256:f6da8c2bc09a5847d67d20470b225dca7b8e1d4900995c9ff484189d8b38fe59"
        ),
    },
    "glm53_autocontinue_canary": {
        "plan_sha256": "sha256:6bfc06e5ede3a572f03d74e810dc26006bdbf5134b841ca12565cdf13c02ca93",
        "campaign_id": "chris-glm53-ac-canary1-v1",
        "source_rank": 13,
        "attempt": 1,
        "task_version_id": "9375a9b9-04e5-4f6f-ad47-286121278992",
        "served_id": "glm-5.3",
        "session_model": "fleet-cluster-opencode-1.18.27/glm-5.3-opencode11827-autocontinue-v1",
        "serving_block": "glm-hosted-autocontinue-v1",
        "settings_file_sha256": (
            "sha256:84a1ca763a297bd8badb184391a02c82d57069b9a211ecef4f5e53dd79e20e20"
        ),
        "preflight_configmap": "chris-glm53-ac-canary1-pre-v1",
        "preflight_configmap_v2": "chris-glm53-ac-canary1-pre-v2",
        "preflight_job_v2": "chris-glm53-ac-canary1-v2-preflight",
        "scored_configmap": "chris-glm53-ac-canary1-run-v2",
        "plan_path": "evals/fleet/configs/glm53-opencode-autocontinue-canary1-v1.json",
        "plan_file_sha256": (
            "sha256:2eada8438613465f8d9422836e9a6cce76f5a6eefbdc31635939b6ff3fd958c8"
        ),
        "preflight_statement": (
            "I authorize the create-once read-only GLM corrected-treatment canary preflight "
            "only; no scored Job or scored ConfigMap may be created."
        ),
        "model_sha256": "sha256:c7df17e25b5a04484012a02b9adf219f7e2c5abe6a6995eba68fedda687cd2e6",
        "harness_sha256": "sha256:807859e731b15f7c7e977eb45a0679c6a56f1c4c3dcb6ecae43f1e0c34c71099",
        "treatment_sha256": (
            "sha256:a8918985f69f92048c9f66f514d9ef89f26b1519286b9787421617db92aaa75a"
        ),
        "tasks_sha256": "sha256:c29097b7880787dd3eac5f700322af81fdd909a5e070083f84ec65bc90e5c465",
        "attempts_sha256": (
            "sha256:200e15d235310ca6c4fed2e5c21d7d7e42cec994e574a84bf414f355f06f15e0"
        ),
        "execution_sha256": (
            "sha256:fc320747aa47228ae4906e534ef6a8102e66655228fa7d2915e4047d75a19e60"
        ),
    },
}

EXPECTED["qwen38_autocontinue_canary_v2"] = {
    **EXPECTED["qwen38_autocontinue_canary"],
    "plan_sha256": "sha256:726320dd5e161e28becd22d7bb6e31f1627f36a3c7291c0dabb53f94626b3c2d",
    "campaign_id": "chris-q38-ac-canary1-v2",
    "plan_preflight_job_name": "chris-q38-ac-canary1-v3-preflight",
    "preflight_configmap_v2": "chris-q38-ac-canary1-pre-v3",
    "preflight_job_v2": "chris-q38-ac-canary1-v3-preflight",
    "scored_configmap": "chris-q38-ac-canary1-run-v3",
    "plan_path": "evals/fleet/configs/qwen38-opencode-autocontinue-canary1-v2.json",
    "plan_file_sha256": "sha256:fa0f5f0f7b762d3050fab6b1c9335c6af4737a56b77c3bc96ea00c77f84d754c",
    "attempts_sha256": "sha256:63b32fe929b2bcb132f6e29621716639ac923499aac716849da0bef41bb5f9b0",
    "source_sha256": "sha256:40d072dd47f3cf195a7afd76ec4aac5fff4e5ae04ffd90f7258dc31a0d55632f",
    "preflight_manifest_sha256": PRE_MANIFEST_V3_SHA,
    "scored_manifest_sha256": SCORED_MANIFEST_V3_SHA,
}
EXPECTED["glm53_autocontinue_canary_v2"] = {
    **EXPECTED["glm53_autocontinue_canary"],
    "plan_sha256": "sha256:9654b0f2e9cfbe5690a77ce83bc63e7e537a43597479686fc093db7d6533959a",
    "campaign_id": "chris-glm53-ac-canary1-v2",
    "plan_preflight_job_name": "chris-glm53-ac-canary1-v3-preflight",
    "preflight_configmap_v2": "chris-glm53-ac-canary1-pre-v3",
    "preflight_job_v2": "chris-glm53-ac-canary1-v3-preflight",
    "scored_configmap": "chris-glm53-ac-canary1-run-v3",
    "plan_path": "evals/fleet/configs/glm53-opencode-autocontinue-canary1-v2.json",
    "plan_file_sha256": "sha256:7096622e3b8dcaa04b8b8f286822db6564d89866a0df7e06c30ec3998b3d28bf",
    "attempts_sha256": "sha256:83d511e5de5d66e387829b0b4935e284b9ad0ab4b620e9ca71c06edf17c41fff",
    "source_sha256": "sha256:9196626fb679dbc2e768394a42044a789270e1350e0560d9a48268fdddcc509f",
    "preflight_manifest_sha256": PRE_MANIFEST_V3_SHA,
    "scored_manifest_sha256": SCORED_MANIFEST_V3_SHA,
}


def load_object(path: Path) -> dict[str, Any]:
    return hosted.load_object(path)


def digest_without(value: dict[str, Any], field: str) -> str:
    return hosted.digest_without(value, field)


def _sha(path: Path) -> str:
    return self_hosted.sha256(path.read_bytes())


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
        return False
    return all(character in "0123456789abcdef" for character in value[7:])


def _is_git_commit(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_uuid(value: object) -> bool:
    try:
        uuid.UUID(str(value))
    except (TypeError, ValueError, AttributeError):
        return False
    return True


def _is_utc_timestamp(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() == UTC.utcoffset(parsed)


def validate_v1_preflight_failure(receipt: dict[str, Any]) -> None:
    expected_preflights = [
        {
            "model": "qwen3.8-27b",
            "configmap": {
                "name": "chris-q38-ac-canary1-pre-v1",
                "uid": "41a2555c-fa9c-4bea-be18-2eb55aff21ef",
                "created_at_utc": "2026-09-04T21:12:03Z",
                "immutable": True,
            },
            "job": {
                "name": "chris-q38-ac-canary1-v1-preflight",
                "uid": "0602fd25-840c-4c4a-9dbd-b62ce5ceea75",
                "created_at_utc": "2026-09-04T21:12:07Z",
                "started_at_utc": "2026-09-04T21:12:08Z",
                "failed_at_utc": "2026-09-04T21:12:14Z",
                "reason": "BackoffLimitExceeded",
                "succeeded": 0,
                "failed": 1,
            },
            "pod": {
                "name": "chris-q38-ac-canary1-v1-preflight-9t74n",
                "uid": "b949d555-730f-41be-b6d6-c5f00259e086",
                "owner_job_name": "chris-q38-ac-canary1-v1-preflight",
                "owner_job_uid": "0602fd25-840c-4c4a-9dbd-b62ce5ceea75",
                "owner_is_controller": True,
                "phase": "Failed",
                "exit_code": 1,
                "reason": "Error",
                "restart_count": 0,
                "started_at_utc": "2026-09-04T21:12:08Z",
                "finished_at_utc": "2026-09-04T21:12:12Z",
            },
        },
        {
            "model": "glm-5.3",
            "configmap": {
                "name": "chris-glm53-ac-canary1-pre-v1",
                "uid": "7b23aae0-dff8-4e35-b794-f99df68e0ed7",
                "created_at_utc": "2026-09-04T21:12:06Z",
                "immutable": True,
            },
            "job": {
                "name": "chris-glm53-ac-canary1-v1-preflight",
                "uid": "bb909f75-e7d3-4746-9689-0e81af9bd306",
                "created_at_utc": "2026-09-04T21:12:08Z",
                "started_at_utc": "2026-09-04T21:12:08Z",
                "failed_at_utc": "2026-09-04T21:12:15Z",
                "reason": "BackoffLimitExceeded",
                "succeeded": 0,
                "failed": 1,
            },
            "pod": {
                "name": "chris-glm53-ac-canary1-v1-preflight-tgkbb",
                "uid": "999eb81f-d3d1-435a-850d-a6c15cc81b51",
                "owner_job_name": "chris-glm53-ac-canary1-v1-preflight",
                "owner_job_uid": "bb909f75-e7d3-4746-9689-0e81af9bd306",
                "owner_is_controller": True,
                "phase": "Failed",
                "exit_code": 1,
                "reason": "Error",
                "restart_count": 0,
                "started_at_utc": "2026-09-04T21:12:09Z",
                "finished_at_utc": "2026-09-04T21:12:12Z",
            },
        },
    ]
    expected_absence = {
        "observed_at_utc": "2026-09-04T21:24:30Z",
        "observer_pod_name": "ft-run-98c32208-5gpb2-head-7qxwc",
        "observer_pod_uid": "3f37ab91-4afa-4e4d-8f29-a3eeda70a774",
        "exact_scored_jobs_present": 0,
        "exact_scored_configmaps_present": 0,
        "exact_v2_preflight_jobs_present": 0,
        "exact_v2_preflight_configmaps_present": 0,
        "sfs_roots_present": 0,
        "global_cell_claims_present": 0,
        "checked_sfs_roots": [
            "/mnt/sfs/jobs/chris-q38-ac-canary1-v1-preflight",
            "/mnt/sfs/jobs/chris-glm53-ac-canary1-v1-preflight",
            "/mnt/sfs/jobs/chris-q38-ac-canary1-v2-preflight",
            "/mnt/sfs/jobs/chris-glm53-ac-canary1-v2-preflight",
            "/mnt/sfs/jobs/chris-q38-ac-canary1-v1",
            "/mnt/sfs/jobs/chris-glm53-ac-canary1-v1",
        ],
        "checked_global_cell_claims": [
            "/mnt/sfs/cell-claims/opencode11827-autocontinue-primary-v1/"
            "d9a8b7af84dde45ed8ef0a4ad6b744827608427c5ced6528c04544dd7f8c813d.json",
            "/mnt/sfs/cell-claims/opencode11827-autocontinue-primary-v1/"
            "29b5a7875caa52f4d90544a6ede30dbefd5f723659674816a704cf994153872c.json",
        ],
    }
    if (
        set(receipt)
        != {
            "schema_version",
            "append_only",
            "status",
            "recorded_at_utc",
            "phase_b_commit",
            "package_commit",
            "campaign_sha256",
            "preflight_manifest_sha256",
            "controller_sha256",
            "intent_configmap",
            "scheduling",
            "image",
            "terminal_preflights",
            "sanitized_diagnosis",
            "required_successor",
            "absence_observation",
            "privacy",
            "receipt_sha256",
        }
        or receipt.get("schema_version")
        != "fleet-opencode-autocontinue-canary-preflight-bootstrap-failure-v1"
        or receipt.get("append_only") is not True
        or receipt.get("status") != "TERMINAL_INFRASTRUCTURE_FAILURE"
        or receipt.get("recorded_at_utc") != "2026-09-04T21:14:37Z"
        or receipt.get("phase_b_commit") != "c5eee8f6e9d500d41e0718c4b7c631034c3334ad"
        or receipt.get("package_commit") != "fdad6c80bcbfa999cd98e6f3194040bd1d85d679"
        or receipt.get("campaign_sha256") != CAMPAIGN_SHA
        or receipt.get("preflight_manifest_sha256")
        != "sha256:45f4d0942e6d4485b35be4d0e3ebccdaa825bef5ceec377ba5f983cdd7e499d6"
        or receipt.get("controller_sha256")
        != "sha256:f023dffd60f4594ff60a6388b6ca653c24e56ebb760dc8ecdf8afcc6d7ee04f9"
        or receipt.get("intent_configmap")
        != {
            "name": "chris-ac-canary1-preflight-submit-v1",
            "uid": "b0701452-d347-4581-9498-cd96d7f2cd0e",
            "created_at_utc": "2026-09-04T21:12:00Z",
            "immutable": True,
        }
        or receipt.get("scheduling")
        != {
            "priority_class": PRIORITY,
            "priority_value": 10000,
            "preemption_policy": "PreemptLowerPriority",
            "queue": "training-lq",
            "priority_class_is_not_preemption_immunity": True,
        }
        or receipt.get("image")
        != {
            "requested": (
                "ghcr.io/astral-sh/uv:python3.12-bookworm@sha256:"
                "9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7"
            ),
            "resolved": (
                "ghcr.io/astral-sh/uv@sha256:"
                "9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7"
            ),
        }
        or receipt.get("terminal_preflights") != expected_preflights
        or receipt.get("sanitized_diagnosis")
        != {
            "classification": "deterministic_bootstrap_validation_failure_before_api",
            "logs_read": False,
            "source_ordering_proof": True,
            "bootstrap_installed_campaign_json": False,
            "bootstrap_installed_plan_at_repo_expected_path": False,
            "bootstrap_installed_scored_manifest": False,
            "validator_first_missing_dependency": "immutable_campaign_json",
            "validator_would_also_require_unshipped_plan_and_scored_manifest": True,
            "fleet_api_called": False,
            "model_called": False,
            "verifier_called": False,
            "session_created": False,
            "scored_job_created": False,
            "scored_configmap_created": False,
            "cell_claim_created": False,
            "retry_same_identity_allowed": False,
        }
        or receipt.get("required_successor")
        != {
            "fresh_create_once_configmaps_jobs_and_sfs_roots": True,
            "read_only_preflight_only": True,
            "must_not_package_scored_execution_payload": True,
            "must_validate_only_shipped_bytes_or_exact_caller_bound_constants": True,
            "v1_objects_are_terminal_and_must_not_be_mutated": True,
        }
        or receipt.get("absence_observation") != expected_absence
        or receipt.get("privacy")
        != {
            "credentials_included": False,
            "logs_read": False,
            "prompts_or_traces_included": False,
            "scores_included": False,
        }
        or receipt.get("receipt_sha256") != V1_PREFLIGHT_FAILURE_SHA
        or receipt.get("receipt_sha256") != digest_without(receipt, "receipt_sha256")
    ):
        raise ValueError("v1 canary preflight failure receipt is not authoritative")


def validate_scored_v1_failure(receipt: dict[str, Any]) -> None:
    expected_top = {
        "schema_version",
        "append_only",
        "status",
        "recorded_at_utc",
        "executable_package_commit",
        "release_commit",
        "intent_configmap",
        "terminal_canaries",
        "sfs_observer",
        "api_observer",
        "sanitized_diagnosis",
        "required_successor",
        "privacy",
        "receipt_sha256",
    }
    expected_ids = {
        "qwen38-hosted-canary1": {
            "plan": EXPECTED["qwen38_autocontinue_canary"]["plan_sha256"],
            "release": "sha256:953e78366fb7634356e8ddef276184359fa6f93fd53a6536632ec9e48bbb43ba",
            "cm": (
                "chris-q38-ac-canary1-run-v2",
                "6315320f-3768-4e4c-9663-d97f98c49f0b",
                "2026-09-04T22:18:53Z",
            ),
            "job": (
                "chris-q38-ac-canary1-v1",
                "b4b1c5ff-47f8-4866-8734-adec995ea5af",
                "2026-09-04T22:20:45Z",
            ),
            "pod": (
                "chris-q38-ac-canary1-v1-j5gh8",
                "9cafd38d-9be4-4272-9786-88529a7ed5b6",
                "2026-09-04T22:19:01Z",
                "2026-09-04T22:20:36Z",
            ),
            "claim": "d9a8b7af84dde45ed8ef0a4ad6b744827608427c5ced6528c04544dd7f8c813d.json",
        },
        "glm53-hosted-canary1": {
            "plan": EXPECTED["glm53_autocontinue_canary"]["plan_sha256"],
            "release": "sha256:272fffb49130fec19f3afbcbeba4addfb95079d2afa3f3ba6460254045c7a87a",
            "cm": (
                "chris-glm53-ac-canary1-run-v2",
                "b9ef6ce4-43a5-4440-a505-2d35150042e4",
                "2026-09-04T22:18:57Z",
            ),
            "job": (
                "chris-glm53-ac-canary1-v1",
                "9a60ae83-ef8c-4f55-b580-d0ef67a1df8b",
                "2026-09-04T22:19:51Z",
            ),
            "pod": (
                "chris-glm53-ac-canary1-v1-d94qj",
                "d0fb406d-a48b-4454-89b3-63e98983606d",
                "2026-09-04T22:19:00Z",
                "2026-09-04T22:19:41Z",
            ),
            "claim": "29b5a7875caa52f4d90544a6ede30dbefd5f723659674816a704cf994153872c.json",
        },
    }
    rows = receipt.get("terminal_canaries")
    if (
        set(receipt) != expected_top
        or receipt.get("schema_version")
        != "fleet-opencode-autocontinue-canary-scored-bootstrap-failure-v1"
        or receipt.get("append_only") is not True
        or receipt.get("status") != "TERMINAL_INFRASTRUCTURE_FAILURE_BEFORE_CLAIM"
        or receipt.get("recorded_at_utc") != "2026-09-04T22:22:50Z"
        or receipt.get("executable_package_commit") != "fa2ec783e02fc152f4fc198c1a73badf9ceabb1b"
        or receipt.get("release_commit") != "a2a481f6a9007468f75e015db41aaf952348b0a5"
        or receipt.get("intent_configmap")
        != {
            "name": "chris-ac-canary1-hosted-scored-submit-v1",
            "uid": "be93b29f-249f-475b-b4a7-ca7efa47fff0",
            "created_at_utc": "2026-09-04T22:18:50Z",
            "immutable": True,
        }
        or not isinstance(rows, list)
        or len(rows) != 2
        or {row.get("shard_key") for row in rows} != set(expected_ids)
        or receipt.get("receipt_sha256") != SCORED_V1_FAILURE_SHA
        or receipt.get("receipt_sha256") != digest_without(receipt, "receipt_sha256")
    ):
        raise ValueError("scored v1 failure receipt is not authoritative")
    image = "sha256:ad1dae1e1b3cd770b34a868304c2eb72c6e8c44f807417454ef92e1fb808cf7d"
    image_id = (
        "ghcr.io/astral-sh/uv@sha256:"
        "9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7"
    )
    dind = "sha256:d2dc198f7d839eae26b5a9cb0e7cdc4e2c97d9cb4ea66dbeb0a4c0c7f0b165f8"
    dind_id = (
        "docker.io/library/docker@sha256:"
        "f649ef046008ca7f926a2571c32b0ac22e5c59eb61b959617f9acc2a4c638cf5"
    )
    for row in rows:
        expected = expected_ids[row["shard_key"]]
        cm_name, cm_uid, cm_created = expected["cm"]
        job_name, job_uid, terminal_at = expected["job"]
        pod_name, pod_uid, started_at, finished_at = expected["pod"]
        if (
            row.get("plan_sha256") != expected["plan"]
            or row.get("release_receipt_sha256") != expected["release"]
            or row.get("configmap")
            != {"name": cm_name, "uid": cm_uid, "created_at_utc": cm_created, "immutable": True}
            or row.get("job")
            != {
                "name": job_name,
                "uid": job_uid,
                "created_at_utc": "2026-09-04T22:18:58Z",
                "terminal_condition_at_utc": terminal_at,
                "terminal_condition_type": "Failed",
                "terminal_condition_status": "True",
                "active": 0,
                "succeeded": 0,
                "failed": 1,
                "reason": "BackoffLimitExceeded",
            }
            or row.get("pod")
            != {
                "name": pod_name,
                "uid": pod_uid,
                "owner_job_uid": job_uid,
                "phase": "Failed",
                "evaluator_exit_code": 1,
                "evaluator_started_at_utc": started_at,
                "evaluator_finished_at_utc": finished_at,
                "evaluator_restart_count": 0,
                "evaluator_requested_image": image,
                "evaluator_resolved_image_id": image_id,
                "dind_exit_code": 0,
                "dind_restart_count": 0,
                "dind_requested_image": dind,
                "dind_resolved_image_id": dind_id,
            }
            or row.get("exact_treatment_sessions") != 0
            or row.get("exact_run_sessions") != 0
            or row.get("sfs_root") != f"/mnt/sfs/jobs/{job_name}"
            or row.get("sfs_root_absent") is not True
            or row.get("global_claim_path") != f"{CELL_CLAIM_ROOT}/{expected['claim']}"
            or row.get("global_claim_absent") is not True
        ):
            raise ValueError("scored v1 terminal canary evidence drifted")
    if (
        receipt.get("sfs_observer")
        != {
            "namespace": "fleet-train-jobs",
            "pod_name": "allie-dev",
            "pod_uid": "73dabe56-60f8-4879-be9f-365196c502e3",
            "phase": "Running",
            "ready": True,
            "restart_count": 0,
            "pvc": "sfs-shared",
            "observed_at_utc": "2026-09-04T22:22:50Z",
            "global_claim_json_count": 0,
        }
        or receipt.get("api_observer")
        != {
            "observed_at_utc": "2026-09-04T22:26:55.765087+00:00",
            "authority": "https://orchestrator.fleetai.com",
            "fleet_team_id": self_hosted.FLEET_TEAM_ID,
            "fleet_team_name": "fleet",
            "method": "credentialed_read_only_exhaustive_task_session_pagination",
            "exact_treatment_and_planned_run_counts_recorded_per_canary": True,
            "credentials_included": False,
            "session_content_included": False,
        }
        or receipt.get("sanitized_diagnosis")
        != {
            "classification": (
                "deterministic_bootstrap_canonical_path_failure_before_release_validation"
            ),
            "bootstrap_installed_run_path": (
                "/workspace/cyber-post-train/evals/fleet/scripts/run.sh"
            ),
            "validator_required_run_path": (
                "/workspace/cyber-post-train/evals/fleet/scripts/"
                "run_opencode_autocontinue_canary.sh"
            ),
            "canonical_path_mismatch": True,
            "source_ordering_proves_before_fleet_api": True,
            "launcher_missing_yq_detected_after_creation": True,
            "launcher_missing_yq_did_not_invalidate_independently_audited_exact_objects": True,
            "logs_read": False,
            "prompts_read": False,
            "traces_read": False,
            "scores_read": False,
            "model_called": False,
            "verifier_called": False,
            "session_created": False,
            "global_claim_created": False,
            "task_root_created": False,
            "retry_same_job_identity_allowed": False,
        }
        or receipt.get("required_successor")
        != {
            "fresh_configmap_job_pod_sfs_and_run_identities": True,
            "canonical_packaged_paths_required": True,
            "rendered_bootstrap_validation_required": True,
            "fresh_uid_bound_sfs_and_api_duplicate_preflight_required": True,
            "new_scored_authorization_required": True,
            "launch_authorized": False,
        }
        or receipt.get("privacy")
        != {
            "prompts_included": False,
            "traces_included": False,
            "scores_included": False,
            "logs_included": False,
            "secrets_included": False,
        }
    ):
        raise ValueError("scored v1 failure observer or diagnosis drifted")


def validate_compatibility(
    receipt: dict[str, Any], root: Path, *, allow_missing_campaign: bool = False
) -> None:
    failure_path = (
        root / "docs/evidence/qwen38-study/"
        "2026-09-04-opencode-autocontinue-canary-preflight-v1-bootstrap-failure.json"
    )
    if not failure_path.is_file():
        raise ValueError("v1 canary preflight failure receipt is missing")
    validate_v1_preflight_failure(load_object(failure_path))
    compatibility = receipt.get("compatibility") or {}
    gates = receipt.get("gates") or {}
    overlay = receipt.get("allowed_append_only_overlay") or {}
    campaign_path = (
        root / "evals/fleet/configs/q38-glm53-opencode-autocontinue-primary-campaign-v1.json"
    )
    campaign_file_matches = campaign_path.exists() and _sha(campaign_path) == CAMPAIGN_FILE_SHA
    if (
        receipt.get("schema_version")
        != "fleet-opencode-autocontinue-canary-controller-compatibility-v2"
        or receipt.get("append_only") is not True
        or receipt.get("status") != "HELD_COMPATIBLE"
        or receipt.get("supersedes")
        != {
            "path": (
                "docs/evidence/qwen38-study/"
                "2026-09-04-opencode-autocontinue-canary-controller-compatibility-v1.json"
            ),
            "receipt_sha256": (
                "sha256:a0ded706002accbbf482dfcba58d840b6ba7666f6af225a2207cfbcdbe5adc3c"
            ),
        }
        or receipt.get("campaign_sha256") != CAMPAIGN_SHA
        or receipt.get("immutable_campaign_file_sha256") != CAMPAIGN_FILE_SHA
        or (
            not campaign_file_matches
            and not (allow_missing_campaign and not campaign_path.exists())
        )
        or receipt.get("bootstrap_contract")
        != {
            "campaign_file_may_be_absent_during_read_only_preflight": True,
            "plan_raw_sha256_is_caller_bound": True,
            "scored_manifest_is_not_packaged_or_required": True,
            "compatibility_and_all_executable_dependencies_are_hash_validated": True,
            "v1_bootstrap_failure_receipt_sha256": V1_PREFLIGHT_FAILURE_SHA,
        }
        or receipt.get("receipt_sha256") != digest_without(receipt, "receipt_sha256")
        or receipt.get("frozen_controller")
        != {
            "path": "evals/fleet/hosted_sweep_controller.py",
            "sha256": _sha(root / "evals/fleet/hosted_sweep_controller.py"),
            "modified": False,
        }
        or receipt.get("canary_controller", {}).get("path")
        != "evals/fleet/autocontinue_canary_controller.py"
        or receipt.get("canary_controller", {}).get("sha256")
        != "sha256:412bf8c0dd33d23e50a56c4597e5e0dfc90b122a8b06af0987059afdca53f7ef"
        or receipt.get("canary_controller", {}).get("scope") != "two_exact_one_cell_canaries_only"
        or compatibility.get("campaign_bytes_unchanged") is not True
        or compatibility.get("legacy_credit") != 0
        or compatibility.get("exact_context_policy") != CONTEXT
        or compatibility.get("release_required_before_preflight_or_run") is not True
        or compatibility.get("endpoint_lease_required") is not True
        or compatibility.get("downward_job_and_pod_uid_required") is not True
        or compatibility.get("post_exit_k8s_observer_required") is not True
        or compatibility.get("bulk_release_granted") is not False
        or compatibility.get(
            "supersedes_only_campaign_held_runtime_gate_booleans_via_append_only_evidence"
        )
        is not True
        or compatibility.get(
            "campaign_embedded_canary_and_parity_false_fields_remain_historical_preview_state"
        )
        is not True
        or gates.get("dedicated_parity_receipt_sha256") != DEDICATED_PARITY_SHA
        or gates.get("hosted_health_receipt_sha256") != HOSTED_HEALTH_SHA
        or gates.get("shared_pvc_flock_receipt_sha256") != FLOCK_GATE_SHA
        or gates.get("scored_canary_launch_authorized") is not False
        or overlay
        != {
            "campaign_mutation_allowed": False,
            "satisfied_preview_gate_evidence": {
                "r114_immutable_hydration_and_execution_binding": TASK_INVENTORY_SHA,
                "fresh_fleet_task_version_environment_verifier_inventory": TASK_INVENTORY_SHA,
                "dedicated_a_and_b_uid_bound_autocontinue_parity": DEDICATED_PARITY_SHA,
                "shared_pvc_cross_pod_flock_preflight": FLOCK_GATE_SHA,
            },
            "still_held": [
                "fresh_exact_cell_duplicate_preflights",
                "qwen_and_glm_one_cell_canary_authorization_and_acceptance",
                "append_only_bulk_release_authorization",
            ],
            "forbidden_overrides": [
                "campaign_id",
                "scientific_mapping",
                "models",
                "partitions",
                "task_or_cell_identity",
                "treatment_or_session_model",
                "counts",
                "job_or_sfs_identity",
                "maximum_streams_per_endpoint",
                "retry_policy",
                "canary_launch_authorized",
                "bulk_launch_authorized",
            ],
        }
    ):
        raise ValueError("canary controller compatibility receipt is not authoritative")


def _compatibility(root: Path, *, allow_missing_campaign: bool = False) -> dict[str, Any]:
    receipt = load_object(
        root / "docs/evidence/qwen38-study/"
        "2026-09-04-opencode-autocontinue-canary-controller-compatibility-v2.json"
    )
    validate_compatibility(receipt, root, allow_missing_campaign=allow_missing_campaign)
    return receipt


def validate_successor_compatibility(
    receipt: dict[str, Any], root: Path, *, preflight_workspace: bool = False
) -> None:
    incident_path = (
        root / "docs/evidence/qwen38-study/"
        "2026-09-04-opencode-autocontinue-canary-scored-v1-bootstrap-failure.json"
    )
    if not incident_path.is_file():
        raise ValueError("scored v1 failure receipt is missing")
    validate_scored_v1_failure(load_object(incident_path))
    expected_plans = {
        "qwen": {
            "path": EXPECTED["qwen38_autocontinue_canary_v2"]["plan_path"],
            "plan_sha256": EXPECTED["qwen38_autocontinue_canary_v2"]["plan_sha256"],
            "file_sha256": EXPECTED["qwen38_autocontinue_canary_v2"]["plan_file_sha256"],
        },
        "glm": {
            "path": EXPECTED["glm53_autocontinue_canary_v2"]["plan_path"],
            "plan_sha256": EXPECTED["glm53_autocontinue_canary_v2"]["plan_sha256"],
            "file_sha256": EXPECTED["glm53_autocontinue_canary_v2"]["plan_file_sha256"],
        },
    }
    campaign_path = (
        root / "evals/fleet/configs/q38-glm53-opencode-autocontinue-primary-campaign-v1.json"
    )
    scored_manifest_path = (
        root / "evals/fleet/cluster/opencode-autocontinue-canary-scored-v3.yaml"
    )
    manifest_authorization_path = root / "evals/fleet/scored_manifest_authorization.py"
    if preflight_workspace:
        if scored_manifest_path.exists() or manifest_authorization_path.exists():
            raise ValueError("preflight workspace contains scored execution payload")
        scored_manifest_sha256 = SCORED_MANIFEST_V3_SHA
        manifest_authorization_sha256 = (
            "sha256:8a576d38be80bd9d77e7ef587a5e2b13f7f8ab00940ed96fa1f2d4b470bf8b7e"
        )
    else:
        scored_manifest_sha256 = _sha(scored_manifest_path)
        manifest_authorization_sha256 = _sha(manifest_authorization_path)
    if (
        set(receipt)
        != {
            "schema_version",
            "append_only",
            "status",
            "supersedes",
            "campaign_sha256",
            "immutable_campaign_file_sha256",
            "scored_v1_failure",
            "implementation",
            "successor_plans",
            "bootstrap_contract",
            "authorization",
            "privacy",
            "receipt_sha256",
        }
        or receipt.get("schema_version")
        != "fleet-opencode-autocontinue-canary-controller-compatibility-v3"
        or receipt.get("append_only") is not True
        or receipt.get("status") != "HELD_COMPATIBLE"
        or receipt.get("supersedes")
        != {
            "path": (
                "docs/evidence/qwen38-study/"
                "2026-09-04-opencode-autocontinue-canary-controller-compatibility-v2.json"
            ),
            "receipt_sha256": (
                "sha256:7851d18a8176abb08aebcb857960380f7f8a59bb854c51437e4010f169c82b6a"
            ),
        }
        or receipt.get("campaign_sha256") != CAMPAIGN_SHA
        or receipt.get("immutable_campaign_file_sha256") != CAMPAIGN_FILE_SHA
        or _sha(campaign_path) != CAMPAIGN_FILE_SHA
        or receipt.get("scored_v1_failure")
        != {
            "path": str(incident_path.relative_to(root)),
            "receipt_sha256": SCORED_V1_FAILURE_SHA,
        }
        or receipt.get("implementation")
        != {
            "frozen_controller_path": "evals/fleet/hosted_sweep_controller.py",
            "frozen_controller_sha256": _sha(root / "evals/fleet/hosted_sweep_controller.py"),
            "canary_controller_path": "evals/fleet/autocontinue_canary_controller.py",
            "canary_controller_sha256": _sha(
                root / "evals/fleet/autocontinue_canary_controller.py"
            ),
            "scored_manifest_authorization_path": ("evals/fleet/scored_manifest_authorization.py"),
            "scored_manifest_authorization_sha256": manifest_authorization_sha256,
        }
        or receipt.get("successor_plans") != expected_plans
        or receipt.get("bootstrap_contract")
        != {
            "preflight_manifest_path": (
                "evals/fleet/cluster/opencode-autocontinue-canary-preflights-v3.yaml"
            ),
            "preflight_manifest_sha256": _sha(
                root / "evals/fleet/cluster/opencode-autocontinue-canary-preflights-v3.yaml"
            ),
            "scored_manifest_path": (
                "evals/fleet/cluster/opencode-autocontinue-canary-scored-v3.yaml"
            ),
            "scored_manifest_sha256": scored_manifest_sha256,
            "canonical_run_path": (
                "/workspace/cyber-post-train/evals/fleet/scripts/"
                "run_opencode_autocontinue_canary.sh"
            ),
            "canonical_submit_path": (
                "/workspace/cyber-post-train/evals/fleet/scripts/"
                "submit_opencode_autocontinue_canaries_v2.sh"
            ),
            "rendered_reconstruction_tested": True,
            "preflight_packages_no_scored_execution_payload": True,
        }
        or receipt.get("authorization")
        != {
            "preflight_authorized": False,
            "scored_launch_authorized": False,
            "bulk_launch_authorized": False,
        }
        or receipt.get("privacy")
        != {
            "credentials_included": False,
            "prompts_or_traces_included": False,
            "scores_included": False,
        }
        or receipt.get("receipt_sha256") != digest_without(receipt, "receipt_sha256")
    ):
        raise ValueError("successor controller compatibility receipt is not authoritative")


def _successor_compatibility(
    root: Path, *, preflight_workspace: bool = False
) -> dict[str, Any]:
    receipt = load_object(
        root / "docs/evidence/qwen38-study/"
        "2026-09-04-opencode-autocontinue-canary-controller-compatibility-v3.json"
    )
    validate_successor_compatibility(
        receipt, root, preflight_workspace=preflight_workspace
    )
    return receipt


def _open_directory_nofollow(path: Path) -> int:
    """Create/open every directory component without following a symlink."""
    if any(part == ".." for part in path.parts):
        raise RuntimeError("global canary cell-claim root is unsafe")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    current_fd = os.open("/" if path.is_absolute() else ".", directory_flags)
    parts = path.parts[1:] if path.is_absolute() else path.parts
    try:
        for part in parts:
            if part in {"", "."}:
                continue
            created = False
            try:
                os.mkdir(part, mode=0o700, dir_fd=current_fd)
                created = True
            except FileExistsError:
                pass
            if created:
                os.fsync(current_fd)
            try:
                next_fd = os.open(part, directory_flags, dir_fd=current_fd)
            except OSError as exc:
                raise RuntimeError("global canary cell-claim root is unsafe") from exc
            if not stat.S_ISDIR(os.fstat(next_fd).st_mode):
                os.close(next_fd)
                raise RuntimeError("global canary cell-claim root is unsafe")
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except Exception:
        os.close(current_fd)
        raise


def _open_claim_lock(root_fd: int) -> int:
    """Open the permanent root lock, tolerating only transient create visibility."""
    lock_flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        lock_flags |= os.O_NOFOLLOW
    last_error: FileNotFoundError | None = None
    for _ in range(32):
        try:
            return os.open(".claim.lock", lock_flags, 0o600, dir_fd=root_fd)
        except FileNotFoundError as exc:
            last_error = exc
            if not stat.S_ISDIR(os.fstat(root_fd).st_mode):
                raise RuntimeError("global canary cell-claim root is unsafe") from exc
            time.sleep(0.001)
    raise RuntimeError("global canary cell-claim lock could not be created") from last_error


def claim_global_cell(plan: dict[str, Any], claim_root: Path | None = None) -> dict[str, Any]:
    """Permanently claim one corrected-treatment cell before any paid side effect."""
    validate_plan(plan)
    if not _is_uuid(os.environ.get("JOB_UID")) or not _is_uuid(os.environ.get("POD_UID")):
        raise RuntimeError("global canary cell claim requires downward API UIDs")
    root = claim_root or Path(CELL_CLAIM_ROOT)
    task = plan["tasks"][0]
    item = plan["attempts"][0]
    identity = {
        "context_management": CONTEXT,
        "served_id": plan["model"]["served_id"],
        "session_model": plan["model"]["session_model"],
        "task_version_id": task["task"]["version_id"],
        "attempt": int(item["attempt"]),
    }
    identity_sha = self_hosted.sha256(self_hosted.canonical_json(identity))
    claim_name = f"{identity_sha.removeprefix('sha256:')}.json"
    root_fd = _open_directory_nofollow(root)
    try:
        lock_fd = _open_claim_lock(root_fd)
        with os.fdopen(lock_fd, "a+b") as lock:
            if not stat.S_ISREG(os.fstat(lock.fileno()).st_mode):
                raise RuntimeError("global canary cell-claim lock is not regular")
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                existing = os.stat(claim_name, dir_fd=root_fd, follow_symlinks=False)
            except FileNotFoundError:
                existing = None
            if existing is not None:
                if not stat.S_ISREG(existing.st_mode):
                    raise RuntimeError("global canary cell-claim path is unsafe")
                raise RuntimeError("corrected-treatment canary cell is already claimed")
            receipt = {
                "schema_version": "fleet-opencode-autocontinue-global-cell-claim-v1",
                "identity": identity,
                "identity_sha256": identity_sha,
                "campaign_sha256": CAMPAIGN_SHA,
                "plan_sha256": plan["plan_sha256"],
                "run_id": item["run_id"],
                "job_uid": os.environ["JOB_UID"],
                "pod_uid": os.environ["POD_UID"],
                "claimed_at_utc": datetime.now(UTC).isoformat(),
                "immutable": True,
                "retry_allowed": False,
                "scores_included": False,
                "prompts_or_traces_included": False,
            }
            receipt["receipt_sha256"] = digest_without(receipt, "receipt_sha256")
            claim_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                claim_flags |= os.O_NOFOLLOW
            claim_fd = os.open(claim_name, claim_flags, 0o600, dir_fd=root_fd)
            try:
                if not stat.S_ISREG(os.fstat(claim_fd).st_mode):
                    raise RuntimeError("global canary cell-claim path is unsafe")
                payload = self_hosted.canonical_json(receipt) + b"\n"
                with os.fdopen(claim_fd, "wb") as stored:
                    claim_fd = -1
                    stored.write(payload)
                    stored.flush()
                    os.fsync(stored.fileno())
            finally:
                if claim_fd >= 0:
                    os.close(claim_fd)
            os.fsync(root_fd)
            return receipt
    finally:
        os.close(root_fd)


def _bound_receipt(
    root: Path, evidence: dict[str, Any], path_field: str, digest_field: str
) -> dict[str, Any]:
    relative = evidence.get(path_field)
    if not isinstance(relative, str) or relative.startswith("/") or ".." in Path(relative).parts:
        raise ValueError("canary evidence path is unsafe")
    value = load_object(root / relative)
    if value.get("receipt_sha256") != evidence.get(digest_field) or value.get(
        "receipt_sha256"
    ) != digest_without(value, "receipt_sha256"):
        raise ValueError("canary evidence digest drifted")
    return value


def _validate_final_preflight_evidence(
    evidence: dict[str, Any], plan: dict[str, Any], root: Path, package_commit: str
) -> None:
    duplicate = _bound_receipt(
        root,
        evidence,
        "fresh_duplicate_inventory_receipt_path",
        "fresh_duplicate_inventory_receipt_sha256",
    )
    preauth = _bound_receipt(
        root,
        evidence,
        "preflight_authorization_receipt_path",
        "preflight_authorization_receipt_sha256",
    )
    validate_preflight_authorization(
        preauth,
        plan,
        root,
        package_commit,
        preauth.get("implementation", {}).get("plan_file_sha256"),
        preauth.get("authorized_at_utc"),
        preauth.get("authorization", {}).get("statement"),
    )
    preflight = _bound_receipt(root, evidence, "preflight_receipt_path", "preflight_receipt_sha256")
    observed = _bound_receipt(
        root,
        evidence,
        "preflight_post_exit_receipt_path",
        "preflight_post_exit_receipt_sha256",
    )
    expected = EXPECTED[plan["shard_key"]]
    observed_job = observed.get("job") or {}
    observed_pod = observed.get("pod") or {}
    observed_configmap = observed.get("configmap") or {}
    if (
        duplicate.get("schema_version")
        != "fleet-opencode-autocontinue-canary-duplicate-inventory-v1"
        or duplicate.get("status") != "PASSED"
        or duplicate.get("fleet_team_id") != self_hosted.FLEET_TEAM_ID
        or duplicate.get("pagination_exhausted") is not True
        or duplicate.get("plan_sha256") != plan["plan_sha256"]
        or duplicate.get("cell")
        != {
            "source_rank": expected["source_rank"],
            "attempt": 1,
            "task_version_id": expected["task_version_id"],
        }
        or duplicate.get("exact_treatment_session_rows") != 0
        or duplicate.get("current_plan_run_rows") != 0
        or duplicate.get("active_attempts") != 0
        or duplicate.get("global_cell_claim_absent") is not True
        or duplicate.get("job_pod_and_sfs_identities_absent") is not True
        or not isinstance(duplicate.get("observed_at_utc"), str)
        or not duplicate.get("observed_at_utc")
        or duplicate.get("privacy")
        != {
            "credentials_included": False,
            "prompts_or_traces_included": False,
            "scores_included": False,
        }
        or preflight.get("schema_version") != "fleet-opencode-autocontinue-canary-preflight-v1"
        or preflight.get("status") != "PASSED"
        or preflight.get("plan_sha256") != plan["plan_sha256"]
        or preflight.get("release_receipt_sha256") != preauth["receipt_sha256"]
        or preflight.get("fleet_team_id") != self_hosted.FLEET_TEAM_ID
        or preflight.get("current_plan_run_and_claim_identities_absent") is not True
        or preflight.get("output_root_absent") is not True
        or not isinstance(preflight.get("sfs_job_roots_reconciled"), int)
        or isinstance(preflight.get("sfs_job_roots_reconciled"), bool)
        or preflight.get("sfs_job_roots_reconciled") < 0
        or preflight.get("exact_treatment_sessions_reconciled") != 0
        or not _is_uuid(preflight.get("job_uid"))
        or not _is_uuid(preflight.get("pod_uid"))
        or preflight.get("scores_read") is not False
        or preflight.get("prompts_or_traces_read") is not False
        or observed.get("schema_version")
        != "fleet-opencode-autocontinue-canary-preflight-post-exit-v1"
        or observed.get("status") != "PASSED"
        or observed.get("plan_sha256") != plan["plan_sha256"]
        or observed.get("preflight_receipt_sha256") != preflight["receipt_sha256"]
        or observed_job.get("name") != expected["preflight_job_v2"]
        or observed_job.get("uid") != preflight.get("job_uid")
        or observed_job.get("succeeded") != 1
        or observed_job.get("failed") != 0
        or observed_job.get("priority_class") != PRIORITY
        or observed_job.get("preemption_policy") != "PreemptLowerPriority"
        or not isinstance(observed_job.get("completion_time"), str)
        or not observed_job.get("completion_time")
        or not isinstance(observed_pod.get("name"), str)
        or not observed_pod.get("name")
        or observed_pod.get("uid") != preflight.get("pod_uid")
        or observed_pod.get("phase") != "Succeeded"
        or observed_pod.get("exit_code") != 0
        or observed_pod.get("restart_count") != 0
        or not isinstance(observed_pod.get("finished_at"), str)
        or not observed_pod.get("finished_at")
        or observed_configmap.get("name") != expected["preflight_configmap_v2"]
        or not _is_uuid(observed_configmap.get("uid"))
        or observed_configmap.get("immutable") is not True
        or observed.get("scored_job_created") is not False
        or observed.get("scored_sfs_root_absent") is not True
        or observed.get("privacy")
        != {
            "credentials_included": False,
            "prompts_or_traces_included": False,
            "scores_included": False,
        }
    ):
        raise ValueError("canary final preflight evidence is not authoritative")


def validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("canary plan schema drifted")
    if plan.get("plan_sha256") != digest_without(plan, "plan_sha256"):
        raise ValueError("canary plan digest mismatch")
    expected = EXPECTED.get(plan.get("shard_key"))
    if expected is None:
        raise ValueError("canary shard identity drifted")
    if plan.get("plan_sha256") != expected["plan_sha256"]:
        raise ValueError("canary approved plan digest drifted")
    tasks = plan.get("tasks") or []
    attempts = plan.get("attempts") or []
    model = plan.get("model") or {}
    harness = plan.get("harness") or {}
    execution = plan.get("execution") or {}
    source = plan.get("source") or {}
    canonical = self_hosted.canonical_json
    if (
        plan.get("campaign_id") != expected["campaign_id"]
        or plan.get("source_job_id") != expected["campaign_id"]
        or plan.get("preflight_job_name")
        != expected.get("plan_preflight_job_name", expected["campaign_id"] + "-preflight")
        or plan.get("scored_job_name") != expected["campaign_id"]
        or plan.get("sfs_root") != expected["campaign_id"]
        or plan.get("serving_block") != expected["serving_block"]
        or plan.get("task_count") != 1
        or plan.get("pass_k") != 4
        or plan.get("total_session_count") != 1
        or plan.get("new_session_count") != 1
        or plan.get("credited_sessions") != []
        or plan.get("legacy_credited_sessions") != 0
        or len(tasks) != 1
        or len(attempts) != 1
        or int(tasks[0].get("source_rank") or 0) != expected["source_rank"]
        or tasks[0].get("task", {}).get("version_id") != expected["task_version_id"]
        or int(attempts[0].get("source_rank") or 0) != expected["source_rank"]
        or int(attempts[0].get("attempt") or 0) != expected["attempt"]
        or model.get("served_id") != expected["served_id"]
        or model.get("session_model") != expected["session_model"]
        or harness.get("context_management") != CONTEXT
        or harness.get("settings_file_sha256") != expected["settings_file_sha256"]
        or execution.get("required_priority_class") != PRIORITY
        or execution.get("launch_authorized") is not False
        or execution.get("inventory_policy") != "conservative_no_same_model_session_for_task_key_v1"
        or execution.get("endpoint_lease")
        != {
            "lease_root": LEASE_ROOT,
            "endpoint_key": expected["serving_block"],
            "maximum_streams": 2,
        }
        or source.get("campaign_sha256") != CAMPAIGN_SHA
        or source.get("flock_release_gate_receipt_sha256") != FLOCK_GATE_SHA
        or source.get("hosted_health_receipt_sha256") != HOSTED_HEALTH_SHA
        or source.get("fresh_inventory_receipt_sha256") is not None
        or source.get("dedicated_parity_receipt_sha256") is not None
        or self_hosted.sha256(canonical(model)) != expected["model_sha256"]
        or self_hosted.sha256(canonical(harness)) != expected["harness_sha256"]
        or self_hosted.sha256(canonical(plan.get("authority")))
        != "sha256:cfea8580278e65a337958dd39c431047a22a83cf32f393dab57966b5fe21746d"
        or self_hosted.sha256(canonical(plan.get("treatment_block")))
        != expected["treatment_sha256"]
        or self_hosted.sha256(canonical(tasks)) != expected["tasks_sha256"]
        or self_hosted.sha256(canonical(attempts)) != expected["attempts_sha256"]
        or self_hosted.sha256(canonical(execution)) != expected["execution_sha256"]
        or self_hosted.sha256(canonical(source))
        != expected.get(
            "source_sha256",
            "sha256:4da18b77d108242475c364cbb64226f565137217a047dc33490e11126dff6b6d",
        )
    ):
        raise ValueError("canary plan semantic identity drifted")
    expected_run_id = (
        f"{expected['campaign_id']}-sr{expected['source_rank']:03d}"
        f"-a1-{expected['task_version_id'][:8]}"
    )
    if attempts[0].get("run_id") != expected_run_id:
        raise ValueError("canary run identity drifted")


def validate_release(release: dict[str, Any], plan: dict[str, Any], root: Path) -> None:
    """Accept only an append-only final release; HELD templates always fail."""
    validate_plan(plan)
    if (
        release.get("schema_version") != RELEASE_SCHEMA
        or release.get("append_only") is not True
        or release.get("status") != "RELEASED"
        or not isinstance(release.get("released_at_utc"), str)
        or not release.get("released_at_utc")
        or release.get("receipt_sha256") != digest_without(release, "receipt_sha256")
    ):
        raise ValueError("canary scoring release is not authoritative")
    expected = EXPECTED[plan["shard_key"]]
    evidence = release.get("evidence") or {}
    implementation = release.get("implementation") or {}
    authorization = release.get("authorization") or {}
    terminal = release.get("terminal_contract") or {}
    compatibility = _compatibility(root)
    compatibility_sha = compatibility["receipt_sha256"]
    package_commit = implementation.get("package_commit")
    if not _is_git_commit(package_commit):
        raise ValueError("canary scoring release package commit is not authoritative")
    _validate_final_preflight_evidence(evidence, plan, root, package_commit)
    if (
        release.get("campaign_sha256") != CAMPAIGN_SHA
        or release.get("plan_sha256") != plan["plan_sha256"]
        or release.get("cell")
        != {
            "source_rank": expected["source_rank"],
            "attempt": 1,
            "task_version_id": expected["task_version_id"],
        }
        or evidence.get("fresh_inventory_receipt_sha256") != TASK_INVENTORY_SHA
        or evidence.get("fresh_inventory_execution_sha256") != TASK_INVENTORY_EXECUTION_SHA
        or not _is_sha256(evidence.get("fresh_duplicate_inventory_receipt_sha256"))
        or not _is_sha256(evidence.get("preflight_receipt_sha256"))
        or not _is_sha256(evidence.get("preflight_post_exit_receipt_sha256"))
        or not _is_sha256(evidence.get("preflight_authorization_receipt_sha256"))
        or evidence.get("dedicated_parity_receipt_sha256") != DEDICATED_PARITY_SHA
        or evidence.get("hosted_health_receipt_sha256") != HOSTED_HEALTH_SHA
        or evidence.get("shared_pvc_flock_receipt_sha256") != FLOCK_GATE_SHA
        or evidence.get("controller_compatibility_receipt_sha256") != compatibility_sha
        or implementation.get("plan_sha256") != plan["plan_sha256"]
        or implementation.get("controller_sha256")
        != _sha(root / "evals/fleet/autocontinue_canary_controller.py")
        or implementation.get("frozen_controller_sha256")
        != _sha(root / "evals/fleet/hosted_sweep_controller.py")
        or implementation.get("preflight_manifest_sha256")
        != _sha(root / "evals/fleet/cluster/opencode-autocontinue-canary-preflights-v2.yaml")
        or implementation.get("scored_manifest_sha256")
        != _sha(root / "evals/fleet/cluster/opencode-autocontinue-canary-scored-v2.yaml")
        or implementation.get("launcher_sha256")
        != _sha(root / "evals/fleet/scripts/submit_opencode_autocontinue_canaries_v1.sh")
        or authorization.get("launch_authorized") is not True
        or authorization.get("create_once") is not True
        or authorization.get("required_priority_class") != PRIORITY
        or not isinstance(authorization.get("author"), str)
        or not authorization.get("author")
        or not isinstance(authorization.get("statement"), str)
        or not authorization.get("statement")
        or terminal.get("downward_job_uid_required") is not True
        or terminal.get("downward_pod_uid_required") is not True
        or terminal.get("post_exit_k8s_observer_required") is not True
        or terminal.get("terminal_schema_version") != TERMINAL_SCHEMA
        or terminal.get("post_exit_schema_version") != POST_EXIT_SCHEMA
        or release.get("privacy")
        != {
            "credentials_included": False,
            "prompts_or_traces_included": False,
            "scores_included": False,
        }
    ):
        raise ValueError("canary scoring release is not authoritative")


def validate_preflight_authorization(
    release: dict[str, Any],
    plan: dict[str, Any],
    root: Path,
    package_commit: str,
    plan_file_sha256: str,
    authorized_at_utc: str,
    authorization_statement: str,
    *,
    preflight_workspace: bool = False,
) -> None:
    validate_plan(plan)
    successor = plan["shard_key"].endswith("_v2")
    if not _is_git_commit(package_commit):
        raise ValueError("canary preflight authorization package commit is invalid")
    if plan_file_sha256 != EXPECTED[plan["shard_key"]]["plan_file_sha256"]:
        raise ValueError("canary preflight plan file digest is invalid")
    if not _is_utc_timestamp(authorized_at_utc) or not authorization_statement:
        raise ValueError("canary preflight root authorization is invalid")
    expected = EXPECTED[plan["shard_key"]]
    compatibility = (
        _successor_compatibility(root, preflight_workspace=preflight_workspace)
        if successor
        else _compatibility(root, allow_missing_campaign=True)
    )
    expected_evidence = {
        "task_inventory_receipt_sha256": TASK_INVENTORY_SHA,
        "task_inventory_execution_sha256": TASK_INVENTORY_EXECUTION_SHA,
        "dedicated_parity_receipt_sha256": DEDICATED_PARITY_SHA,
        "hosted_health_receipt_sha256": HOSTED_HEALTH_SHA,
        "shared_pvc_flock_receipt_sha256": FLOCK_GATE_SHA,
        "controller_compatibility_receipt_sha256": compatibility["receipt_sha256"],
        "v1_bootstrap_failure_receipt_sha256": V1_PREFLIGHT_FAILURE_SHA,
        "fresh_duplicate_inventory_receipt_sha256": None,
    }
    if successor:
        expected_evidence["scored_v1_failure_receipt_sha256"] = SCORED_V1_FAILURE_SHA
    expected_implementation = {
        "package_commit": package_commit,
        "plan_sha256": plan["plan_sha256"],
        "plan_file_sha256": plan_file_sha256,
        "controller_sha256": (
            _sha(root / "evals/fleet/autocontinue_canary_controller.py")
            if successor
            else "sha256:412bf8c0dd33d23e50a56c4597e5e0dfc90b122a8b06af0987059afdca53f7ef"
        ),
        "frozen_controller_sha256": _sha(root / "evals/fleet/hosted_sweep_controller.py"),
        "self_hosted_sha256": SELF_HOSTED_SHA,
        "runner_sha256": RUNNER_SHA,
        "endpoint_lease_sha256": ENDPOINT_LEASE_SHA,
        "compatibility_file_sha256": _sha(
            root
            / "docs/evidence/qwen38-study/"
            / (
                "2026-09-04-opencode-autocontinue-canary-controller-compatibility-v3.json"
                if successor
                else "2026-09-04-opencode-autocontinue-canary-controller-compatibility-v2.json"
            )
        ),
        "preflight_manifest_sha256": expected.get("preflight_manifest_sha256", PRE_MANIFEST_SHA),
    }
    if (
        set(release)
        != {
            "schema_version",
            "append_only",
            "status",
            "authorized_at_utc",
            "package_commit",
            "campaign_sha256",
            "plan_sha256",
            "cell",
            "preflight_identity",
            "evidence",
            "implementation",
            "authorization",
            "privacy",
            "receipt_sha256",
        }
        or release.get("receipt_sha256") != digest_without(release, "receipt_sha256")
        or release.get("schema_version")
        != (
            "fleet-opencode-autocontinue-canary-preflight-authorization-v3"
            if successor
            else "fleet-opencode-autocontinue-canary-preflight-authorization-v2"
        )
        or release.get("append_only") is not True
        or release.get("status") != "PREFLIGHT_AUTHORIZED"
        or release.get("authorized_at_utc") != authorized_at_utc
        or not _is_utc_timestamp(release.get("authorized_at_utc"))
        or release.get("package_commit") != package_commit
        or release.get("campaign_sha256") != CAMPAIGN_SHA
        or release.get("plan_sha256") != plan["plan_sha256"]
        or release.get("cell")
        != {
            "source_rank": plan["tasks"][0]["source_rank"],
            "attempt": 1,
            "task_version_id": plan["tasks"][0]["task"]["version_id"],
        }
        or release.get("preflight_identity")
        != {
            "configmap_name": expected["preflight_configmap_v2"],
            "job_name": expected["preflight_job_v2"],
            "sfs_root": f"/mnt/sfs/jobs/{expected['preflight_job_v2']}",
            "scored_configmap_name": expected["scored_configmap"],
            "scored_job_name": plan["scored_job_name"],
            "scored_job_created": False,
        }
        or release.get("evidence") != expected_evidence
        or release.get("implementation") != expected_implementation
        or _sha(root / "evals/fleet/self_hosted.py") != SELF_HOSTED_SHA
        or _sha(root / "evals/fleet/opencode_train_sweep_runner.py") != RUNNER_SHA
        or _sha(root / "evals/fleet/endpoint_lease.py") != ENDPOINT_LEASE_SHA
        or _sha(
            root
            / (
                "evals/fleet/cluster/opencode-autocontinue-canary-preflights-v3.yaml"
                if successor
                else "evals/fleet/cluster/opencode-autocontinue-canary-preflights-v2.yaml"
            )
        )
        != expected.get("preflight_manifest_sha256", PRE_MANIFEST_SHA)
        or release.get("authorization")
        != {
            "preflight_authorized": True,
            "launch_authorized": False,
            "author": "/root",
            "statement": authorization_statement,
        }
        or release.get("privacy")
        != {
            "credentials_included": False,
            "prompts_or_traces_included": False,
            "scores_included": False,
        }
    ):
        raise ValueError("canary preflight authorization drifted")


def validate_held_release(
    release: dict[str, Any], plan: dict[str, Any], root: Path | None = None
) -> None:
    validate_plan(plan)
    if plan["shard_key"].endswith("_v2"):
        if root is None:
            raise ValueError("successor canary held release requires repository root")
        expected = EXPECTED[plan["shard_key"]]
        compatibility = _successor_compatibility(root)
        predecessor_by_shard = {
            "qwen38_autocontinue_canary_v2": {
                "intent_configmap": {
                    "name": "chris-ac-canary1-hosted-scored-submit-v1",
                    "uid": "be93b29f-249f-475b-b4a7-ca7efa47fff0",
                },
                "scored_configmap": {
                    "name": "chris-q38-ac-canary1-run-v2",
                    "uid": "6315320f-3768-4e4c-9663-d97f98c49f0b",
                },
                "scored_job": {
                    "name": "chris-q38-ac-canary1-v1",
                    "uid": "b4b1c5ff-47f8-4866-8734-adec995ea5af",
                },
                "scored_pod": {
                    "name": "chris-q38-ac-canary1-v1-j5gh8",
                    "uid": "9cafd38d-9be4-4272-9786-88529a7ed5b6",
                },
            },
            "glm53_autocontinue_canary_v2": {
                "intent_configmap": {
                    "name": "chris-ac-canary1-hosted-scored-submit-v1",
                    "uid": "be93b29f-249f-475b-b4a7-ca7efa47fff0",
                },
                "scored_configmap": {
                    "name": "chris-glm53-ac-canary1-run-v2",
                    "uid": "b9ef6ce4-43a5-4440-a505-2d35150042e4",
                },
                "scored_job": {
                    "name": "chris-glm53-ac-canary1-v1",
                    "uid": "9a60ae83-ef8c-4f55-b580-d0ef67a1df8b",
                },
                "scored_pod": {
                    "name": "chris-glm53-ac-canary1-v1-d94qj",
                    "uid": "d0fb406d-a48b-4454-89b3-63e98983606d",
                },
            },
        }
        expected_predecessor = {
            "scored_bootstrap_failure_receipt_sha256": SCORED_V1_FAILURE_SHA,
            **predecessor_by_shard[plan["shard_key"]],
            "cell_unclaimed": True,
            "exact_treatment_sessions": 0,
            "sfs_root_absent": True,
            "global_claim_absent": True,
        }
        expected_evidence = {
            "task_inventory_receipt_sha256": TASK_INVENTORY_SHA,
            "hosted_health_receipt_sha256": HOSTED_HEALTH_SHA,
            "shared_pvc_flock_receipt_sha256": FLOCK_GATE_SHA,
            "controller_compatibility_receipt_sha256": compatibility["receipt_sha256"],
            "dedicated_parity_is_historical_only": True,
            "fresh_duplicate_inventory_receipt_sha256": None,
        }
        expected_implementation = {
            "package_commit": None,
            "package_commit_bound_after_phase_a_audit": True,
            "controller_sha256": _sha(root / "evals/fleet/autocontinue_canary_controller.py"),
            "frozen_controller_sha256": _sha(root / "evals/fleet/hosted_sweep_controller.py"),
            "preflight_manifest_sha256": PRE_MANIFEST_V3_SHA,
            "scored_manifest_sha256": SCORED_MANIFEST_V3_SHA,
            "scored_manifest_authorization_sha256": _sha(
                root / "evals/fleet/scored_manifest_authorization.py"
            ),
            "preflight_launcher_sha256": None,
            "scored_launcher_sha256": None,
        }
        expected_remaining_gates = [
            "phase_a_immutable_commit",
            "read_only_v3_preflight_authorization",
            "successful_uid_bound_v3_preflight",
            "fresh_exact_cell_duplicate_inventory",
            "new_root_scored_launch_authorization",
            "final_hosted_released_receipt",
        ]
        if (
            set(release)
            != {
                "schema_version",
                "append_only",
                "status",
                "campaign_sha256",
                "plan_sha256",
                "cell",
                "predecessor",
                "evidence",
                "implementation",
                "fresh_identities",
                "remaining_gates",
                "authorization",
                "privacy",
                "receipt_sha256",
            }
            or release.get("schema_version") != "fleet-opencode-autocontinue-canary-held-release-v2"
            or release.get("append_only") is not True
            or release.get("status") != "HELD"
            or release.get("campaign_sha256") != CAMPAIGN_SHA
            or release.get("plan_sha256") != plan["plan_sha256"]
            or release.get("cell")
            != {
                "source_rank": expected["source_rank"],
                "attempt": 1,
                "task_version_id": expected["task_version_id"],
            }
            or release.get("predecessor") != expected_predecessor
            or release.get("evidence") != expected_evidence
            or release.get("implementation") != expected_implementation
            or release.get("fresh_identities")
            != {
                "preflight_configmap": expected["preflight_configmap_v2"],
                "preflight_job": expected["preflight_job_v2"],
                "preflight_sfs_root": f"/mnt/sfs/jobs/{expected['preflight_job_v2']}",
                "scored_configmap": expected["scored_configmap"],
                "scored_job": plan["scored_job_name"],
                "scored_sfs_root": f"/mnt/sfs/jobs/{plan['sfs_root']}",
            }
            or release.get("authorization")
            != {
                "author": None,
                "preflight_authorized": False,
                "launch_authorized": False,
                "create_once": True,
                "required_priority_class": PRIORITY,
                "statement": None,
            }
            or release.get("privacy")
            != {
                "credentials_included": False,
                "prompts_or_traces_included": False,
                "scores_included": False,
            }
            or release.get("remaining_gates") != expected_remaining_gates
            or release.get("receipt_sha256") != digest_without(release, "receipt_sha256")
        ):
            raise ValueError("successor canary held release drifted")
        return
    if (
        release.get("schema_version") != "fleet-opencode-autocontinue-canary-held-release-v1"
        or release.get("append_only") is not True
        or release.get("status") != "HELD"
        or release.get("campaign_sha256") != CAMPAIGN_SHA
        or release.get("plan_sha256") != plan["plan_sha256"]
        or release.get("authorization", {}).get("preflight_authorized") is not False
        or release.get("authorization", {}).get("launch_authorized") is not False
        or release.get("receipt_sha256") != digest_without(release, "receipt_sha256")
    ):
        raise ValueError("canary held package release drifted")


def validate_terminal(
    terminal: dict[str, Any], plan: dict[str, Any], release: dict[str, Any]
) -> None:
    validate_plan(plan)
    accepted = terminal.get("accepted") is True
    quarantined = terminal.get("quarantined") is True
    compatibility_sha = release.get("evidence", {}).get("controller_compatibility_receipt_sha256")
    if (
        terminal.get("schema_version") != TERMINAL_SCHEMA
        or terminal.get("receipt_sha256") != digest_without(terminal, "receipt_sha256")
        or terminal.get("plan_sha256") != plan["plan_sha256"]
        or terminal.get("campaign_sha256") != CAMPAIGN_SHA
        or terminal.get("controller_compatibility_receipt_sha256") != compatibility_sha
        or terminal.get("release_receipt_sha256") != release.get("receipt_sha256")
        or not _is_uuid(terminal.get("job_uid"))
        or not _is_uuid(terminal.get("pod_uid"))
        or terminal.get("source_rank") != plan["tasks"][0]["source_rank"]
        or terminal.get("attempt") != 1
        or terminal.get("task_version_id") != plan["tasks"][0]["task"]["version_id"]
        or terminal.get("run_id") != plan["attempts"][0]["run_id"]
        or not _is_sha256(terminal.get("global_cell_claim_receipt_sha256"))
        or terminal.get("model_revision") != plan["model"]["revision"]
        or terminal.get("served_id") != plan["model"]["served_id"]
        or terminal.get("session_model") != plan["model"]["session_model"]
        or terminal.get("model_sha256")
        != self_hosted.sha256(self_hosted.canonical_json(plan["model"]))
        or terminal.get("harness_sha256")
        != self_hosted.sha256(self_hosted.canonical_json(plan["harness"]))
        or terminal.get("treatment_block_sha256")
        != self_hosted.sha256(self_hosted.canonical_json(plan["treatment_block"]))
        or terminal.get("context_management") != CONTEXT
        or terminal.get("settings_file_sha256") != plan["harness"]["settings_file_sha256"]
        or terminal.get("required_task_tools") != ["bash", "submit_report"]
        or terminal.get("required_task_tool_catalog_sha256")
        != plan["execution"]["required_task_tool_catalog_sha256"]
        or terminal.get("endpoint_lease") != plan["execution"]["endpoint_lease"]
        or not _is_sha256(terminal.get("attempt_config_sha256"))
        or not _is_sha256(terminal.get("claim_sha256"))
        or not _is_utc_timestamp(terminal.get("terminal_at_utc"))
        or accepted == quarantined
        or terminal.get("credited") is not accepted
        or terminal.get("exact_cell_count") != 1
        or terminal.get("legacy_credited_sessions") != 0
        or terminal.get("retry_allowed") is not False
        or terminal.get("bulk_release_granted") is not False
        or terminal.get("post_exit_k8s_observer_required") is not True
        or terminal.get("scores_included") is not False
        or terminal.get("prompts_or_traces_included") is not False
    ):
        raise ValueError("canary terminal is not authoritative")
    if accepted and (
        not _is_sha256(terminal.get("acceptance_receipt_sha256"))
        or not isinstance(terminal.get("session_id"), str)
        or not isinstance(terminal.get("verifier_execution_id"), str)
        or terminal.get("session_ingest_completed") is not True
        or terminal.get("cleanup_completed") is not True
    ):
        raise ValueError("accepted canary terminal is incomplete")


def validate_post_exit(
    observer: dict[str, Any],
    terminal: dict[str, Any],
    plan: dict[str, Any],
    release: dict[str, Any],
) -> None:
    validate_terminal(terminal, plan, release)
    job = observer.get("job") or {}
    pod = observer.get("pod") or {}
    if (
        observer.get("schema_version") != POST_EXIT_SCHEMA
        or observer.get("status") != "PASSED"
        or observer.get("receipt_sha256") != digest_without(observer, "receipt_sha256")
        or observer.get("campaign_sha256") != CAMPAIGN_SHA
        or observer.get("plan_sha256") != plan["plan_sha256"]
        or observer.get("final_release_receipt_sha256") != release.get("receipt_sha256")
        or observer.get("canary_terminal_receipt_sha256") != terminal["receipt_sha256"]
        or observer.get("release_receipt_sha256") != release.get("receipt_sha256")
        or observer.get("terminal_receipt_sha256") != terminal["receipt_sha256"]
        or observer.get("acceptance_receipt_sha256") != terminal.get("acceptance_receipt_sha256")
        or observer.get("cell")
        != {
            "source_rank": plan["tasks"][0]["source_rank"],
            "attempt": 1,
            "task_version_id": plan["tasks"][0]["task"]["version_id"],
            "run_id": plan["attempts"][0]["run_id"],
            "claim_sha256": terminal["claim_sha256"],
            "acceptance_receipt_sha256": terminal["acceptance_receipt_sha256"],
            "session_id": terminal["session_id"],
            "verifier_execution_id": terminal["verifier_execution_id"],
            "accepted": True,
            "credited": True,
            "retry_allowed": False,
        }
        or job
        != {
            "name": plan["scored_job_name"],
            "uid": terminal["job_uid"],
            "succeeded": 1,
            "failed": 0,
            "priority_class": PRIORITY,
            "preemption_policy": "PreemptLowerPriority",
            "completion_time": job.get("completion_time"),
        }
        or not isinstance(job.get("completion_time"), str)
        or not isinstance(pod.get("name"), str)
        or pod.get("uid") != terminal["pod_uid"]
        or pod.get("phase") != "Succeeded"
        or pod.get("exit_code") != 0
        or pod.get("restart_count") != 0
        or not isinstance(pod.get("finished_at"), str)
        or observer.get("endpoint_lease") != plan["execution"]["endpoint_lease"]
        or observer.get("endpoint_lease_reacquired_after_job_exit") is not True
        or observer.get("endpoint_lease_released") is not True
        or observer.get("exact_claim_count") != 1
        or observer.get("exact_accepted_cell_count") != 1
        or observer.get("quarantine_count") != 0
        or observer.get("bulk_release_eligible") is not True
        or terminal.get("accepted") is not True
        or observer.get("evidence")
        != {
            "exact_claim_count": 1,
            "exact_accepted_count": 1,
            "exact_session_count": 1,
            "exact_verifier_execution_count": 1,
            "global_cell_claim_receipt_sha256": terminal["global_cell_claim_receipt_sha256"],
            "claim_sha256": terminal["claim_sha256"],
            "acceptance_receipt_sha256": terminal["acceptance_receipt_sha256"],
            "session_id": terminal["session_id"],
            "verifier_execution_id": terminal["verifier_execution_id"],
            "final_release_receipt_sha256": release["receipt_sha256"],
            "canary_terminal_receipt_sha256": terminal["receipt_sha256"],
            "quarantine_count": 0,
            "session_ingest_completed": True,
            "cleanup_completed": True,
        }
        or observer.get("privacy")
        != {
            "credentials_included": False,
            "prompts_or_traces_included": False,
            "scores_included": False,
        }
    ):
        raise ValueError("canary post-exit evidence is not authoritative")


def preflight(
    plan: dict[str, Any],
    release: dict[str, Any],
    root: Path,
    key: str,
    repo: Path,
    package_commit: str,
    plan_file_sha256: str,
    authorized_at_utc: str,
    authorization_statement: str,
) -> dict[str, Any]:
    validate_preflight_authorization(
        release,
        plan,
        repo,
        package_commit,
        plan_file_sha256,
        authorized_at_utc,
        authorization_statement,
        preflight_workspace=True,
    )
    roots = hosted._validate_plan_identity_absence(plan, root)
    with hosted._client(key) as client:
        account = self_hosted._request(client, "GET", "/v1/account")
    if account.get("team_name") != "fleet" or account.get("team_id") != self_hosted.FLEET_TEAM_ID:
        raise RuntimeError("FLEET_API_KEY is not scoped to Fleet")
    scratch = Path("/tmp/autocontinue-canary-preflight-empty")
    if scratch.exists():
        raise RuntimeError("canary preflight scratch exists")
    (scratch / "attempts").mkdir(parents=True)
    try:
        sessions = hosted._validate_inventory_for_task(plan, scratch, plan["tasks"][0], key)
    finally:
        (scratch / "attempts").rmdir()
        scratch.rmdir()
    receipt = {
        "schema_version": "fleet-opencode-autocontinue-canary-preflight-v1",
        "status": "PASSED",
        "plan_sha256": plan["plan_sha256"],
        "release_receipt_sha256": release["receipt_sha256"],
        "fleet_team_id": self_hosted.FLEET_TEAM_ID,
        "current_plan_run_and_claim_identities_absent": True,
        "output_root_absent": True,
        "sfs_job_roots_reconciled": roots,
        "exact_treatment_sessions_reconciled": sessions,
        "job_uid": os.environ["JOB_UID"],
        "pod_uid": os.environ["POD_UID"],
        "scores_read": False,
        "prompts_or_traces_read": False,
    }
    receipt["receipt_sha256"] = digest_without(receipt, "receipt_sha256")
    return receipt


def _run_cell(plan: dict[str, Any], root: Path, proxy: Path, key: str) -> dict[str, Any]:
    task = plan["tasks"][0]
    item = plan["attempts"][0]
    rank = int(task["rank"])
    config = hosted._attempt_config(plan, task, item)
    task_claim = {
        "schema_version": "fleet-hosted-opencode-task-claim-v1",
        "plan_sha256": plan["plan_sha256"],
        "rank": rank,
        "source_rank": int(task["source_rank"]),
        "task_key": task["task"]["key"],
        "task_version_id": task["task"]["version_id"],
        "run_ids": [item["run_id"]],
    }
    task_claim["claim_sha256"] = digest_without(task_claim, "claim_sha256")
    claim = {
        "schema_version": "fleet-hosted-opencode-attempt-claim-v1",
        "plan_sha256": plan["plan_sha256"],
        "task_claim_sha256": task_claim["claim_sha256"],
        "run_id": item["run_id"],
        "rank": rank,
        "source_rank": int(item["source_rank"]),
        "attempt": 1,
        "network": item["network"],
        "task_key": task["task"]["key"],
        "task_version_id": task["task"]["version_id"],
        "config_sha256": config["config_sha256"],
    }
    claim["claim_sha256"] = digest_without(claim, "claim_sha256")
    hosted._validate_inventory_for_task(plan, root, task, key)
    hosted._claim_task_and_first_attempt_or_raise_drained(
        plan,
        root,
        root / "task-claims/rank-001.json",
        task_claim,
        root / "claims" / f"{item['run_id']}.json",
        claim,
    )
    out_dir = root / "attempts" / item["run_id"]
    try:
        self_hosted.run(config, out_dir, proxy)
        outcome = hosted._classify_result(out_dir, config, item, claim["claim_sha256"], key)
    except Exception as exc:
        if not (
            isinstance(exc, (self_hosted.FleetRequestError, self_hosted.SessionIngestError))
            or hosted._has_sanitized_infrastructure_failure(out_dir)
        ):
            raise
        quarantined = hosted._quarantine_infrastructure_cell(
            plan=plan,
            task=task,
            item=item,
            claim_sha256=claim["claim_sha256"],
            root=root,
            error_type=type(exc).__name__,
            accepted_count=0,
            remaining_attempts=[],
        )
        return {
            "accepted": False,
            "quarantined": bool(quarantined["quarantined"]),
            "claim_sha256": claim["claim_sha256"],
            "attempt_config_sha256": config["config_sha256"],
        }
    return {
        "accepted": bool(outcome["accepted"]),
        "quarantined": False,
        "claim_sha256": claim["claim_sha256"],
        "attempt_config_sha256": config["config_sha256"],
        "acceptance_receipt_sha256": outcome["receipt_sha256"],
        "session_id": outcome["session_id"],
        "verifier_execution_id": outcome["verifier_execution_id"],
        "session_ingest_completed": outcome["session_ingest_completed"],
        "cleanup_completed": outcome["cleanup_completed"],
    }


def run(
    plan: dict[str, Any], release: dict[str, Any], root: Path, proxy: Path, repo: Path
) -> dict[str, Any]:
    validate_release(release, plan, repo)
    key = os.environ.get("FLEET_API_KEY")
    if not key:
        raise RuntimeError("FLEET_API_KEY is required")
    lease = plan["execution"]["endpoint_lease"]
    with endpoint_lease.acquire_endpoint_lease(
        lease_root=Path(lease["lease_root"]),
        endpoint_key=lease["endpoint_key"],
        maximum_streams=lease["maximum_streams"],
    ):
        hosted._validate_plan_identity_absence(plan, root)
        with hosted._client(key) as client:
            account = self_hosted._request(client, "GET", "/v1/account")
        if (
            account.get("team_name") != "fleet"
            or account.get("team_id") != self_hosted.FLEET_TEAM_ID
        ):
            raise RuntimeError("FLEET_API_KEY is not scoped to Fleet")
        hosted._validate_inventory_for_task(plan, root, plan["tasks"][0], key)
        global_claim = claim_global_cell(plan)
        root.mkdir(mode=0o700)
        for name in ("attempts", "claims", "task-claims", "task-results", "quarantine", "ramps"):
            (root / name).mkdir(mode=0o700)
        self_hosted.write_json_once(root / "PLAN.json", plan)
        self_hosted.write_json_once(root / "SCORING-RELEASE.json", release)
        result = _run_cell(plan, root, proxy, key)
        terminal = {
            "schema_version": TERMINAL_SCHEMA,
            "plan_sha256": plan["plan_sha256"],
            "campaign_sha256": CAMPAIGN_SHA,
            "controller_compatibility_receipt_sha256": release["evidence"][
                "controller_compatibility_receipt_sha256"
            ],
            "release_receipt_sha256": release["receipt_sha256"],
            "job_uid": os.environ["JOB_UID"],
            "pod_uid": os.environ["POD_UID"],
            "source_rank": int(plan["tasks"][0]["source_rank"]),
            "attempt": 1,
            "task_version_id": plan["tasks"][0]["task"]["version_id"],
            "run_id": plan["attempts"][0]["run_id"],
            "global_cell_claim_receipt_sha256": global_claim["receipt_sha256"],
            "model_revision": plan["model"]["revision"],
            "served_id": plan["model"]["served_id"],
            "session_model": plan["model"]["session_model"],
            "model_sha256": self_hosted.sha256(self_hosted.canonical_json(plan["model"])),
            "harness_sha256": self_hosted.sha256(self_hosted.canonical_json(plan["harness"])),
            "treatment_block_sha256": self_hosted.sha256(
                self_hosted.canonical_json(plan["treatment_block"])
            ),
            "context_management": CONTEXT,
            "settings_file_sha256": plan["harness"]["settings_file_sha256"],
            "required_task_tools": ["bash", "submit_report"],
            "required_task_tool_catalog_sha256": plan["execution"][
                "required_task_tool_catalog_sha256"
            ],
            "endpoint_lease": plan["execution"]["endpoint_lease"],
            "attempt_config_sha256": result["attempt_config_sha256"],
            "claim_sha256": result["claim_sha256"],
            "terminal_at_utc": datetime.now(UTC).isoformat(),
            "accepted": result["accepted"],
            "credited": result["accepted"],
            "quarantined": result["quarantined"],
            "exact_cell_count": 1,
            "legacy_credited_sessions": 0,
            "retry_allowed": False,
            "bulk_release_granted": False,
            "post_exit_k8s_observer_required": True,
            "scores_included": False,
            "prompts_or_traces_included": False,
        }
        if result.get("acceptance_receipt_sha256"):
            terminal["acceptance_receipt_sha256"] = result["acceptance_receipt_sha256"]
            terminal["session_id"] = result["session_id"]
            terminal["verifier_execution_id"] = result["verifier_execution_id"]
            terminal["session_ingest_completed"] = result["session_ingest_completed"]
            terminal["cleanup_completed"] = result["cleanup_completed"]
        terminal["receipt_sha256"] = digest_without(terminal, "receipt_sha256")
        validate_terminal(terminal, plan, release)
        self_hosted.write_json_once(root / "CANARY-TERMINAL.json", terminal)
        return terminal


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    held = sub.add_parser("validate-held")
    held.add_argument("--plan", type=Path, required=True)
    held.add_argument("--release", type=Path, required=True)
    held.add_argument("--repo", type=Path, required=True)
    final = sub.add_parser("validate-release")
    final.add_argument("--plan", type=Path, required=True)
    final.add_argument("--release", type=Path, required=True)
    final.add_argument("--repo", type=Path, required=True)
    post = sub.add_parser("validate-post-exit")
    post.add_argument("--plan", type=Path, required=True)
    post.add_argument("--release", type=Path, required=True)
    post.add_argument("--terminal", type=Path, required=True)
    post.add_argument("--observer", type=Path, required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--plan", type=Path, required=True)
    pre.add_argument("--release", type=Path, required=True)
    pre.add_argument("--out", type=Path, required=True)
    pre.add_argument("--out-dir", type=Path, required=True)
    pre.add_argument("--repo", type=Path, required=True)
    pre.add_argument("--package-commit", required=True)
    pre.add_argument("--plan-file-sha256", required=True)
    pre.add_argument("--authorized-at-utc", required=True)
    pre.add_argument("--authorization-statement", required=True)
    execute = sub.add_parser("run")
    execute.add_argument("--plan", type=Path, required=True)
    execute.add_argument("--release", type=Path, required=True)
    execute.add_argument("--out-dir", type=Path, required=True)
    execute.add_argument("--proxy", type=Path, required=True)
    execute.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    plan = load_object(args.plan)
    release = load_object(args.release)
    if args.command == "validate-held":
        validate_held_release(release, plan, args.repo)
        return 0
    if args.command == "validate-release":
        validate_release(release, plan, args.repo)
        return 0
    if args.command == "validate-post-exit":
        validate_post_exit(load_object(args.observer), load_object(args.terminal), plan, release)
        return 0
    if args.command == "preflight":
        key = os.environ.get("FLEET_API_KEY")
        if not key:
            raise RuntimeError("FLEET_API_KEY is required")
        if _sha(args.plan) != args.plan_file_sha256:
            raise RuntimeError("canary preflight loaded plan digest drifted")
        receipt = preflight(
            plan,
            release,
            args.out_dir,
            key,
            args.repo,
            args.package_commit,
            args.plan_file_sha256,
            args.authorized_at_utc,
            args.authorization_statement,
        )
        args.out.parent.mkdir(mode=0o700, parents=True, exist_ok=False)
        self_hosted.write_json_once(args.out, receipt)
        return 0
    terminal = run(plan, release, args.out_dir, args.proxy, args.repo)
    print(json.dumps({"ok": True, "receipt_sha256": terminal["receipt_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
