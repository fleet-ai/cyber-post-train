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
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.fleet import endpoint_lease, self_hosted
from evals.fleet import hosted_sweep_controller as hosted

CAMPAIGN_SHA = "sha256:63946f224a33eb0d2c2a6fdba34358ebf9ca379e5f0f156cd137c95a9b98097e"
FLOCK_GATE_SHA = "sha256:e9f34d35ac3c9e60d2698ebc1685374b98f13e41ac305df0609fb051f35c416c"
HOSTED_HEALTH_SHA = "sha256:d81a01ffe1087d7d85fe511bc9c978461ba3e24a2c15675871cfb8668c49bd85"
DEDICATED_PARITY_SHA = "sha256:07e733b4b161ed7c36113fc3688f04a42938d93a499b1bcc10c7cd21342900df"
TASK_INVENTORY_SHA = "sha256:c71604d22b1d52a7091727b957e0f56f2e8e270e461c844fb57d48b392d831c5"
TASK_INVENTORY_EXECUTION_SHA = (
    "sha256:b4c175db0d04746e16fa9314edb093f7306e859545cd95972694e461b7f82fcb"
)
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
        "scored_configmap": "chris-q38-ac-canary1-run-v2",
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
        "scored_configmap": "chris-glm53-ac-canary1-run-v2",
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


def _is_uuid(value: object) -> bool:
    try:
        uuid.UUID(str(value))
    except (TypeError, ValueError, AttributeError):
        return False
    return True


def validate_compatibility(receipt: dict[str, Any], root: Path) -> None:
    compatibility = receipt.get("compatibility") or {}
    gates = receipt.get("gates") or {}
    overlay = receipt.get("allowed_append_only_overlay") or {}
    if (
        receipt.get("schema_version")
        != "fleet-opencode-autocontinue-canary-controller-compatibility-v1"
        or receipt.get("append_only") is not True
        or receipt.get("status") != "HELD_COMPATIBLE"
        or receipt.get("campaign_sha256") != CAMPAIGN_SHA
        or receipt.get("immutable_campaign_file_sha256")
        != _sha(
            root / "evals/fleet/configs/q38-glm53-opencode-autocontinue-primary-campaign-v1.json"
        )
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
        != _sha(root / "evals/fleet/autocontinue_canary_controller.py")
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


def _compatibility(root: Path) -> dict[str, Any]:
    receipt = load_object(
        root / "docs/evidence/qwen38-study/"
        "2026-09-04-opencode-autocontinue-canary-controller-compatibility-v1.json"
    )
    validate_compatibility(receipt, root)
    return receipt


def claim_global_cell(plan: dict[str, Any], claim_root: Path | None = None) -> dict[str, Any]:
    """Permanently claim one corrected-treatment cell before any paid side effect."""
    validate_plan(plan)
    if not _is_uuid(os.environ.get("JOB_UID")) or not _is_uuid(os.environ.get("POD_UID")):
        raise RuntimeError("global canary cell claim requires downward API UIDs")
    root = claim_root or Path(CELL_CLAIM_ROOT)
    if root.exists() and (root.is_symlink() or not root.is_dir()):
        raise RuntimeError("global canary cell-claim root is unsafe")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
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
    path = root / f"{identity_sha.removeprefix('sha256:')}.json"
    lock_path = root / ".claim.lock"
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(lock_path, flags, 0o600)
    with os.fdopen(fd, "a+b") as lock:
        if not stat.S_ISREG(os.fstat(lock.fileno()).st_mode):
            raise RuntimeError("global canary cell-claim lock is not regular")
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if os.path.lexists(path):
            if path.is_symlink() or not path.is_file():
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
        self_hosted.write_json_once(path, receipt)
        with path.open("rb") as stored:
            os.fsync(stored.fileno())
        directory_fd = os.open(root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return receipt


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
    evidence: dict[str, Any], plan: dict[str, Any], root: Path
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
    validate_preflight_authorization(preauth, plan, root)
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
        or not _is_uuid(preflight.get("job_uid"))
        or not _is_uuid(preflight.get("pod_uid"))
        or preflight.get("scores_read") is not False
        or preflight.get("prompts_or_traces_read") is not False
        or observed.get("schema_version")
        != "fleet-opencode-autocontinue-canary-preflight-post-exit-v1"
        or observed.get("status") != "PASSED"
        or observed.get("plan_sha256") != plan["plan_sha256"]
        or observed.get("preflight_receipt_sha256") != preflight["receipt_sha256"]
        or observed_job.get("name") != plan["preflight_job_name"]
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
        or observed_configmap.get("name") != expected["preflight_configmap"]
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
        or plan.get("preflight_job_name") != expected["campaign_id"] + "-preflight"
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
        != "sha256:4da18b77d108242475c364cbb64226f565137217a047dc33490e11126dff6b6d"
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
    _validate_final_preflight_evidence(evidence, plan, root)
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
        != _sha(root / "evals/fleet/cluster/opencode-autocontinue-canary-preflights-v1.yaml")
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
    release: dict[str, Any], plan: dict[str, Any], root: Path
) -> None:
    validate_plan(plan)
    evidence = release.get("evidence") or {}
    implementation = release.get("implementation") or {}
    compatibility = _compatibility(root)
    if (
        release.get("schema_version")
        != "fleet-opencode-autocontinue-canary-preflight-authorization-v1"
        or release.get("append_only") is not True
        or release.get("status") != "PREFLIGHT_AUTHORIZED"
        or not isinstance(release.get("authorized_at_utc"), str)
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
            "configmap_name": EXPECTED[plan["shard_key"]]["preflight_configmap"],
            "job_name": plan["preflight_job_name"],
            "sfs_root": f"/mnt/sfs/jobs/{plan['preflight_job_name']}",
            "scored_configmap_name": EXPECTED[plan["shard_key"]]["scored_configmap"],
            "scored_job_name": plan["scored_job_name"],
            "scored_job_created": False,
        }
        or evidence.get("task_inventory_receipt_sha256") != TASK_INVENTORY_SHA
        or evidence.get("task_inventory_execution_sha256") != TASK_INVENTORY_EXECUTION_SHA
        or evidence.get("dedicated_parity_receipt_sha256") != DEDICATED_PARITY_SHA
        or evidence.get("hosted_health_receipt_sha256") != HOSTED_HEALTH_SHA
        or evidence.get("shared_pvc_flock_receipt_sha256") != FLOCK_GATE_SHA
        or evidence.get("controller_compatibility_receipt_sha256")
        != compatibility["receipt_sha256"]
        or evidence.get("fresh_duplicate_inventory_receipt_sha256") is not None
        or implementation.get("plan_sha256") != plan["plan_sha256"]
        or implementation.get("controller_sha256")
        != _sha(root / "evals/fleet/autocontinue_canary_controller.py")
        or implementation.get("frozen_controller_sha256")
        != _sha(root / "evals/fleet/hosted_sweep_controller.py")
        or implementation.get("preflight_manifest_sha256")
        != _sha(root / "evals/fleet/cluster/opencode-autocontinue-canary-preflights-v1.yaml")
        or release.get("authorization", {}).get("preflight_authorized") is not True
        or release.get("authorization", {}).get("launch_authorized") is not False
        or release.get("authorization", {}).get("author") != "/root"
        or not isinstance(release.get("authorization", {}).get("statement"), str)
        or not release.get("authorization", {}).get("statement")
        or release.get("privacy")
        != {
            "credentials_included": False,
            "prompts_or_traces_included": False,
            "scores_included": False,
        }
        or release.get("receipt_sha256") != digest_without(release, "receipt_sha256")
    ):
        raise ValueError("canary preflight authorization drifted")


def validate_held_release(release: dict[str, Any], plan: dict[str, Any]) -> None:
    validate_plan(plan)
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
        or not isinstance(terminal.get("job_uid"), str)
        or not isinstance(terminal.get("pod_uid"), str)
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
        or not isinstance(terminal.get("terminal_at_utc"), str)
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
    plan: dict[str, Any], release: dict[str, Any], root: Path, key: str, repo: Path
) -> dict[str, Any]:
    validate_preflight_authorization(release, plan, repo)
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
        validate_held_release(release, plan)
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
        receipt = preflight(plan, release, args.out_dir, key, args.repo)
        args.out.parent.mkdir(mode=0o700, parents=True, exist_ok=False)
        self_hosted.write_json_once(args.out, receipt)
        return 0
    terminal = run(plan, release, args.out_dir, args.proxy, args.repo)
    print(json.dumps({"ok": True, "receipt_sha256": terminal["receipt_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
