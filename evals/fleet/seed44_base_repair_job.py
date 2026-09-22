"""Render the two create-once CPU Jobs for the seed-44 Base repair."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evals.fleet import evaluate, reviewed_recovery_v2
from evals.fleet import stored_session_reconciliation_v2 as stored_v2

SCHEMA = "qwen38-fleet-dev17-seed44-base-narrow-repair-v1"
SUCCESSOR_SCHEMA = "qwen38-fleet-dev17-seed44-base-stageb-successor-v1"
CORRECTED_SUCCESSOR_SCHEMA = "qwen38-fleet-dev17-seed44-base-stageb-successor-v2"
FAILURE_ALERT_ANNOTATION = "fleet.ai/failure-alerts"
NAMESPACE = "fleet-train-jobs"
EXPECTED_CONTEXT = "nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6"
EVALUATOR_IMAGE = (
    "ghcr.io/astral-sh/uv:python3.12-bookworm@"
    "sha256:9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7"
)
DIND_IMAGE = (
    "docker.io/library/docker@"
    "sha256:f649ef046008ca7f926a2571c32b0ac22e5c59eb61b959617f9acc2a4c638cf5"
)
COMMON_CODE_FILES = {
    "jobs.py": "cyber_post_train/jobs.py",
    "evaluate.py": "evals/fleet/evaluate.py",
    "exact_pass4_crypto.py": "evals/fleet/exact_pass4_crypto.py",
    "exact_pass4_universe.py": "evals/fleet/exact_pass4_universe.py",
    "fixed_proxy.py": "evals/fleet/fixed_proxy.py",
    "opencode_self_hosted.py": "evals/fleet/opencode_self_hosted.py",
    "retry_review_policy.py": "evals/fleet/retry_review_policy.py",
    "rollout_campaign.py": "evals/fleet/rollout_campaign.py",
    "rollout_ledger.py": "evals/fleet/rollout_ledger.py",
    "rollout_postgres.py": "evals/fleet/rollout_postgres.py",
    "rollout_worker.py": "evals/fleet/rollout_worker.py",
    "stored_session_reconciliation.py": "evals/fleet/stored_session_reconciliation.py",
}
STORED_CODE_FILES = {
    **COMMON_CODE_FILES,
    "stored_session_reconciliation_v2.py": ("evals/fleet/stored_session_reconciliation_v2.py"),
}
ROLLOUT_CODE_FILES = {
    **COMMON_CODE_FILES,
    "cluster_entry.py": "evals/fleet/cluster_entry.py",
    "model_artifact.py": "evals/fleet/model_artifact.py",
    "repair_lineage.py": "evals/fleet/repair_lineage.py",
    "reviewed_recovery_v2.py": "evals/fleet/reviewed_recovery_v2.py",
    "reviewed_recovery_worker_v2.py": "evals/fleet/reviewed_recovery_worker_v2.py",
}


class PackageError(ValueError):
    """The narrow repair package differs from its reviewed public plan."""


def _worker_name(value: Any) -> str:
    """Apply the runtime worker-name grammar before a package can render."""

    if not isinstance(value, str):
        raise PackageError("rollout repair worker identity is invalid")
    try:
        evaluate._name(value)  # noqa: SLF001
    except ValueError as exc:
        raise PackageError("rollout repair worker identity is invalid") from exc
    return value


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_digest(value: Any) -> str:
    return "sha256:" + _sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )


def _load(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PackageError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise PackageError(f"{label} must be one JSON object")
    return value


def _safe_repo_path(repo_root: Path, value: Any, label: str) -> Path:
    if (
        not isinstance(value, str)
        or not value
        or Path(value).is_absolute()
        or ".." in Path(value).parts
    ):
        raise PackageError(f"{label} path is invalid")
    path = (repo_root / value).resolve()
    if repo_root.resolve() not in path.parents or path.is_symlink() or not path.is_file():
        raise PackageError(f"{label} path is invalid")
    return path


def _stored_script() -> str:
    modules = " ".join(name for name in STORED_CODE_FILES if name != "jobs.py")
    return f"""#!/usr/bin/env bash
set -euo pipefail
umask 077
: "${{FLEET_API_KEY:?FLEET_API_KEY is required}}"
: "${{ROLLOUT_DATABASE_URL:?ROLLOUT_DATABASE_URL is required}}"
: "${{EVALUATION_DATABASE:?EVALUATION_DATABASE is required}}"
: "${{EVALUATION_DIRECTORY:?EVALUATION_DIRECTORY is required}}"
: "${{REPAIR_OUTPUT:?REPAIR_OUTPUT is required}}"
root=/workspace/cyber-post-train
mkdir -p "$root/cyber_post_train" "$root/evals/fleet"
touch "$root/cyber_post_train/__init__.py" "$root/evals/__init__.py" "$root/evals/fleet/__init__.py"
install -m 0644 /bootstrap/jobs.py "$root/cyber_post_train/jobs.py"
for name in {modules}; do
  install -m 0644 "/bootstrap/$name" "$root/evals/fleet/$name"
done
cd "$root"
exec uv run --no-project --with httpx==0.28.1 --with pyyaml==6.0.3 \
  --with 'psycopg[binary]==3.3.5' python -m evals.fleet.stored_session_reconciliation_v2 \
  --evaluation-directory "$EVALUATION_DIRECTORY" \
  --output-root "$REPAIR_OUTPUT" \
  --postgres-admin-dsn-env ROLLOUT_DATABASE_URL \
  --postgres-database "$EVALUATION_DATABASE" \
  --intent /intent/intent.json
"""


def _rollout_script(worker_id: str = "q38_s44_base_repair") -> str:
    modules = " ".join(name for name in ROLLOUT_CODE_FILES if name != "jobs.py")
    return f"""#!/usr/bin/env bash
set -euo pipefail
umask 077
: "${{FLEET_API_KEY:?FLEET_API_KEY is required}}"
: "${{ROLLOUT_DATABASE_URL:?ROLLOUT_DATABASE_URL is required}}"
: "${{DOCKER_BIND_ROOT:?DOCKER_BIND_ROOT is required}}"
: "${{EVALUATION_DATABASE:?EVALUATION_DATABASE is required}}"
: "${{EVALUATION_DIRECTORY:?EVALUATION_DIRECTORY is required}}"
: "${{HARNESS_RECEIPT:?HARNESS_RECEIPT is required}}"
: "${{HARNESS_RECEIPT_SHA256:?HARNESS_RECEIPT_SHA256 is required}}"
: "${{HARNESS_TAR:?HARNESS_TAR is required}}"
: "${{HARNESS_TAR_SHA256:?HARNESS_TAR_SHA256 is required}}"
: "${{REPAIR_OUTPUT:?REPAIR_OUTPUT is required}}"
root=/workspace/cyber-post-train
mkdir -p "$root/cyber_post_train" "$root/evals/fleet"
touch "$root/cyber_post_train/__init__.py" "$root/evals/__init__.py" "$root/evals/fleet/__init__.py"
install -m 0644 /bootstrap/jobs.py "$root/cyber_post_train/jobs.py"
for name in {modules}; do
  install -m 0644 "/bootstrap/$name" "$root/evals/fleet/$name"
done
for _ in $(seq 1 120); do
  docker info >/dev/null 2>&1 && break
  sleep 1
done
docker info >/dev/null
cd "$root"
exec uv run --no-project --with httpx==0.28.1 --with pyyaml==6.0.3 \
  --with 'psycopg[binary]==3.3.5' python -m evals.fleet.reviewed_recovery_worker_v2 \
  --evaluation-directory "$EVALUATION_DIRECTORY" \
  --output-root "$REPAIR_OUTPUT" \
  --postgres-admin-dsn-env ROLLOUT_DATABASE_URL \
  --postgres-database "$EVALUATION_DATABASE" \
  --harness-tar "$HARNESS_TAR" \
  --harness-tar-sha256 "$HARNESS_TAR_SHA256" \
  --harness-receipt "$HARNESS_RECEIPT" \
  --harness-receipt-sha256 "$HARNESS_RECEIPT_SHA256" \
  --reviewed-recovery-intent /intent/intent.json \
  --worker-id {worker_id}
"""


@dataclass(frozen=True)
class StagePackage:
    stage: str
    config_map: dict[str, Any]
    secret: dict[str, Any]
    job: dict[str, Any]
    proof: dict[str, Any]

    @property
    def bundle(self) -> dict[str, Any]:
        return {
            "apiVersion": "v1",
            "kind": "List",
            "items": [self.config_map, self.secret, self.job],
        }


def _code(repo_root: Path, files: dict[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    values = {}
    digests = {}
    for name, relative in files.items():
        path = _safe_repo_path(repo_root, relative, f"runtime source {relative}")
        values[name] = path.read_text(encoding="utf-8")
        digests[relative] = _sha256(values[name].encode())
    return values, digests


def _terminal_semantics(value: dict[str, Any], source: dict[str, Any]) -> None:
    try:
        frozen = value["frozen_evaluation"]
        job = value["source_controller"]["job"]
        census = value["score_blind_ledger_census"]
        classification = value["metadata_only_classification"]
        existing = classification["accept_existing_scored_session"]
        retry = classification["single_rollout_repair"]
    except (KeyError, TypeError) as exc:
        raise PackageError("source terminal evidence shape is invalid") from exc
    if any(
        (
            value.get("schema") != "qwen38_base_fleet_dev17_seed44_terminal_census_v1",
            value.get("sha256")
            != _sha256(
                json.dumps(
                    {key: item for key, item in value.items() if key != "sha256"},
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode()
            ),
            frozen.get("evaluation_plan_sha256") != source.get("evaluation_plan_sha256"),
            frozen.get("ledger_plan_sha256") != source.get("ledger_plan_sha256"),
            frozen.get("database") != source.get("database"),
            frozen.get("output_root") != source.get("evaluation_directory"),
            frozen.get("task_count") != 17,
            frozen.get("pass_k") != 1,
            frozen.get("sampling_seed") != 44,
            frozen.get("harness") != "opencode",
            frozen.get("harness_version") != "1.18.27",
            frozen.get("final_eight_task_set_accessed") is not False,
            job.get("name") != source.get("job_name"),
            job.get("uid") != source.get("job_uid"),
            job.get("condition") != "Failed",
            job.get("reason") != "BackoffLimitExceeded",
            job.get("root_failure_alert_annotation") != "off",
            census.get("total_cells") != 17,
            census.get("accepted") != 10,
            census.get("retry_review") != 7,
            census.get("active") != 0,
            census.get("pending") != 0,
            census.get("local_results") != 15,
            existing.get("count") != 5,
            existing.get("source_agent_termination") != "output_limit",
            existing.get("source_agent_exit_code") != 0,
            existing.get("one_exact_completed_authoritative_session_count") != 5,
            existing.get("model_generation_allowed") is not False,
            existing.get("scoring_call_allowed") is not False,
            retry.get("count") != 2,
            retry.get("source_agent_session_count") != 0,
            retry.get("local_result_count") != 0,
            retry.get("trace_manifest_count") != 0,
            retry.get("scoring_intent_count") != 0,
            retry.get("maximum_new_rollouts_each") != 1,
            retry.get("producer_failure", {}).get("method") != "POST",
            retry.get("producer_failure", {}).get("http_status") != 504,
            value.get("decision", {}).get("preserve_original_accepted_cells") != 10,
            any(value is not False for value in value.get("privacy", {}).values()),
        )
    ):
        raise PackageError("source terminal evidence semantics differ from narrow repair")


def _successor_plan(
    repo_root: Path,
    value: dict[str, Any],
    *,
    allow_historical_invalid_worker: bool = False,
) -> dict[str, Any]:
    """Materialize the sole reviewed Stage B successor over the immutable v1 plan."""

    if set(value) != {
        "schema",
        "status",
        "launchable",
        "base_repair_plan",
        "predecessor_evidence",
        "stage_overrides",
        "contract",
        "live_evidence_binding",
        "operation",
        "sha256",
    } or value.get("sha256") != _canonical_digest(
        {key: item for key, item in value.items() if key != "sha256"}
    ):
        raise PackageError("Stage B successor plan identity differs")
    if (
        value.get("status") != "source_prepared_not_launchable"
        or value.get("launchable") is not False
    ):
        raise PackageError("Stage B successor plan must remain launch inert")
    if value.get("live_evidence_binding") != {
        "kubernetes_context": EXPECTED_CONTEXT,
        "namespace": NAMESPACE,
        "assert_before_every_live_read": True,
        "object_absence_valid_only_after_exact_binding_assertion": True,
    }:
        raise PackageError("Stage B successor live evidence binding differs")
    if set(value.get("operation", {}).values()) != {0}:
        raise PackageError("Stage B successor source claims a live operation")

    base_reference = value.get("base_repair_plan")
    if not isinstance(base_reference, dict) or set(base_reference) != {
        "path",
        "file_sha256",
        "sha256",
    }:
        raise PackageError("Stage B successor base plan reference is invalid")
    base_path = _safe_repo_path(repo_root, base_reference["path"], "base repair plan")
    base = _load(base_path, "base repair plan")
    if any(
        (
            base.get("schema") != SCHEMA,
            base.get("sha256") != base_reference["sha256"],
            "sha256:" + _sha256(base_path.read_bytes()) != base_reference["file_sha256"],
            base.get("sha256")
            != _canonical_digest({key: item for key, item in base.items() if key != "sha256"}),
        )
    ):
        raise PackageError("Stage B successor base plan differs")

    evidence_reference = value.get("predecessor_evidence")
    if not isinstance(evidence_reference, dict) or set(evidence_reference) != {
        "path",
        "file_sha256",
        "receipt_sha256",
    }:
        raise PackageError("Stage B predecessor evidence reference is invalid")
    evidence_path = _safe_repo_path(
        repo_root, evidence_reference["path"], "Stage B predecessor evidence"
    )
    evidence = _load(evidence_path, "Stage B predecessor evidence")
    if any(
        (
            "sha256:" + _sha256(evidence_path.read_bytes()) != evidence_reference["file_sha256"],
            evidence.get("receipt_sha256") != evidence_reference["receipt_sha256"],
            evidence.get("receipt_sha256")
            != _canonical_digest(
                {key: item for key, item in evidence.items() if key != "receipt_sha256"}
            ),
        )
    ):
        raise PackageError("Stage B predecessor evidence digest differs")

    predecessor = evidence.get("predecessor", {})
    diagnostic = evidence.get("startup_diagnostic", {})
    preflight = evidence.get("successor_preflight", {})
    selected = preflight.get("selected_roster", {})
    classification = evidence.get("classification", {})
    contract = value.get("contract", {})
    expected_stages = [
        "load_intent",
        "database_and_evaluation_preflight",
        "harness_archive_digest",
        "docker_load_harness",
        "docker_inspect_harness",
        "docker_pull_proxy",
        "evaluate_check_images",
        "fleet_team_and_route",
        "eligibility_observation_one",
        "eligibility_observation_two",
    ]
    try:
        predecessor_uid = str(uuid.UUID(predecessor["job"]["uid"]))
        diagnostic_uid = str(uuid.UUID(diagnostic["pod_uid"]))
        preflight_uid = str(uuid.UUID(preflight["pod_uid"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise PackageError("Stage B predecessor evidence UID is invalid") from exc
    if any(
        (
            predecessor_uid != predecessor["job"]["uid"],
            diagnostic_uid != diagnostic["pod_uid"],
            preflight_uid != preflight["pod_uid"],
            predecessor["job"].get("name") != base["stages"]["single_rollout_repair"]["job_name"],
            predecessor["job"].get("condition") != "Failed",
            predecessor["job"].get("reason") != "BackoffLimitExceeded",
            predecessor["job"].get("root_failure_alert_annotation") != "off",
            predecessor.get("controller_terminal", {}).get("accepted") is not False,
            predecessor.get("controller_terminal", {}).get("controller_failure_code")
            != "reviewed_recovery_v2_startup_failed",
            diagnostic.get("phase") != "Succeeded",
            diagnostic.get("exit_code") != 0,
            diagnostic.get("passed_stages") != expected_stages,
            diagnostic.get("ledger_mutations") != 0,
            diagnostic.get("model_calls") != 0,
            diagnostic.get("scoring_calls") != 0,
            diagnostic.get("objects_released_after_terminal_evidence") is not True,
            preflight.get("phase") != "Succeeded",
            preflight.get("exit_code") != 0,
            preflight.get("receipt_sha256")
            != "sha256:e012b8fe162f378c85bbf4976629f20344cc056c4c121be51bc05e1da842aed7",
            preflight.get("ledger_census")
            != {
                "total": 17,
                "accepted": 15,
                "retry_review": 2,
                "local_results": 15,
                "active": 0,
                "stale_active": 0,
            },
            selected.get("count") != 2,
            selected.get("retry_review") != 2,
            selected.get("retry_count_zero") != 2,
            selected.get("reconciliation_digest_null") != 2,
            selected.get("session_id_null") != 2,
            selected.get("receipt_digest_null") != 2,
            selected.get("local_result_count") != 0,
            selected.get("matching_recovery_apply_receipt_count") != 0,
            selected.get("exact_authoritative_session_count") != 0,
            selected.get("model_execution_artifact_count") != 0,
            selected.get("scoring_artifact_count") != 0,
            preflight.get("predecessor_output_absent") is not True,
            preflight.get("successor_output_absent") is not True,
            preflight.get("two_identical_eligibility_observations") is not True,
            preflight.get("ledger_mutations") != 0,
            preflight.get("model_generation_calls") != 0,
            preflight.get("scoring_calls") != 0,
            preflight.get("accepted_cell_replays") != 0,
            preflight.get("objects_released_after_terminal_evidence") is not True,
            classification.get("failure_stage") != "before_predecessor_output_root_creation",
            classification.get("predecessor_apply_intent_executed") is not False,
            classification.get("predecessor_model_generation_started") is not False,
            classification.get("predecessor_scoring_started") is not False,
            classification.get("persistent_source_or_identity_defect_observed") is not False,
            classification.get("unique_successor_allowed") is not True,
            classification.get("same_name_replay_allowed") is not False,
            classification.get("third_successor_on_unchanged_signature_allowed") is not False,
            contract
            != {
                "selected_cell_count": 2,
                "execution_generation": 2,
                "accepted_cell_replay_count": 0,
                "stored_session_rescore_count": 0,
                "stored_session_regeneration_count": 0,
                "final_eight_task_set_accessed": False,
                "identical_private_intent_required": True,
                "third_successor_on_unchanged_signature_allowed": False,
            },
        )
    ):
        raise PackageError("Stage B predecessor evidence semantics differ")

    overrides = value.get("stage_overrides")
    if not isinstance(overrides, dict) or set(overrides) != {
        "job_name",
        "config_map_name",
        "secret_name",
        "output_root",
        "worker_id",
        "run_script_sha256",
    }:
        raise PackageError("Stage B successor overrides are invalid")
    old = base["stages"]["single_rollout_repair"]
    for field in ("job_name", "config_map_name", "secret_name"):
        if (
            not isinstance(overrides[field], str)
            or re.fullmatch(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?", overrides[field]) is None
            or overrides[field] == old[field]
        ):
            raise PackageError("Stage B successor Kubernetes identity is invalid")
    if (
        not isinstance(overrides["output_root"], str)
        or not overrides["output_root"].startswith("/mnt/sfs/jobs/")
        or overrides["output_root"] == old["output_root"]
        or ".." in Path(overrides["output_root"]).parts
        or overrides["output_root"] != preflight.get("successor_output_root")
        or old["output_root"] != preflight.get("predecessor_output_root")
    ):
        raise PackageError("Stage B successor output identity is invalid")
    if allow_historical_invalid_worker:
        if not isinstance(overrides["worker_id"], str) or not overrides["worker_id"].isidentifier():
            raise PackageError("Stage B successor worker identity is invalid")
    else:
        _worker_name(overrides["worker_id"])

    effective = copy.deepcopy(base)
    effective["stages"]["single_rollout_repair"].update(overrides)
    effective["_successor_binding"] = {
        "successor_plan_sha256": value["sha256"],
        "predecessor_evidence_receipt_sha256": evidence["receipt_sha256"],
        "predecessor_job_uid": predecessor["job"]["uid"],
        "successor_preflight_receipt_sha256": preflight["receipt_sha256"],
    }
    return effective


def _corrected_successor_plan(repo_root: Path, value: dict[str, Any]) -> dict[str, Any]:
    """Bind the v3 infrastructure successor to the exact v2 naming defect."""

    if set(value) != {
        "schema",
        "status",
        "launchable",
        "prior_successor_plan",
        "failure_evidence",
        "stage_overrides",
        "contract",
        "live_evidence_binding",
        "operation",
        "sha256",
    } or value.get("sha256") != _canonical_digest(
        {key: item for key, item in value.items() if key != "sha256"}
    ):
        raise PackageError("corrected Stage B successor plan identity differs")
    if (
        value.get("status") != "source_prepared_not_launchable"
        or value.get("launchable") is not False
        or value.get("live_evidence_binding")
        != {
            "kubernetes_context": EXPECTED_CONTEXT,
            "namespace": NAMESPACE,
            "assert_before_every_live_read": True,
            "object_absence_valid_only_after_exact_binding_assertion": True,
        }
        or set(value.get("operation", {}).values()) != {0}
    ):
        raise PackageError("corrected Stage B successor must remain launch inert")

    prior_reference = value.get("prior_successor_plan")
    if not isinstance(prior_reference, dict) or set(prior_reference) != {
        "path",
        "file_sha256",
        "sha256",
    }:
        raise PackageError("prior Stage B successor reference is invalid")
    prior_path = _safe_repo_path(repo_root, prior_reference["path"], "prior Stage B successor plan")
    prior = _load(prior_path, "prior Stage B successor plan")
    if any(
        (
            prior.get("schema") != SUCCESSOR_SCHEMA,
            prior.get("sha256") != prior_reference["sha256"],
            "sha256:" + _sha256(prior_path.read_bytes()) != prior_reference["file_sha256"],
            prior.get("sha256")
            != _canonical_digest({key: item for key, item in prior.items() if key != "sha256"}),
        )
    ):
        raise PackageError("prior Stage B successor plan differs")
    effective = _successor_plan(repo_root, prior, allow_historical_invalid_worker=True)
    prior_execution = effective["stages"]["single_rollout_repair"]

    evidence_reference = value.get("failure_evidence")
    if not isinstance(evidence_reference, dict) or set(evidence_reference) != {
        "path",
        "file_sha256",
        "receipt_sha256",
    }:
        raise PackageError("corrected Stage B failure evidence reference is invalid")
    evidence_path = _safe_repo_path(
        repo_root, evidence_reference["path"], "corrected Stage B failure evidence"
    )
    evidence = _load(evidence_path, "corrected Stage B failure evidence")
    if any(
        (
            "sha256:" + _sha256(evidence_path.read_bytes()) != evidence_reference["file_sha256"],
            evidence.get("receipt_sha256") != evidence_reference["receipt_sha256"],
            evidence.get("receipt_sha256")
            != _canonical_digest(
                {key: item for key, item in evidence.items() if key != "receipt_sha256"}
            ),
        )
    ):
        raise PackageError("corrected Stage B failure evidence digest differs")

    try:
        failed = evidence["failed_successor"]
        failed_job = failed["job"]
        failed_pod = failed["pod"]
        failed_workload = failed["workload"]
        signature = evidence["failure_signature"]
        diagnostic = evidence["diagnostic"]
        preflight = evidence["postfailure_preflight"]
        selected = preflight["selected_roster"]
        resources = evidence["resources"]
        correction = evidence["corrected_successor_contract"]
        classification = evidence["classification"]
        for identity in (
            failed_job["uid"],
            failed_pod["uid"],
            failed["config_map"]["uid"],
            failed["secret"]["uid"],
            failed_workload["uid"],
            diagnostic["pod_uid"],
            diagnostic["config_map_uid"],
            preflight["pod_uid"],
            preflight["config_map_uid"],
        ):
            if str(uuid.UUID(identity)) != identity:
                raise ValueError("non-canonical UUID")
    except (KeyError, TypeError, ValueError) as exc:
        raise PackageError("corrected Stage B failure evidence shape is invalid") from exc

    expected_error = "expected an owner-specific lowercase name"
    expected_error_sha256 = "sha256:" + _sha256(expected_error.encode())
    prior_worker = prior_execution.get("worker_id")
    try:
        evaluate._name(prior_worker)  # noqa: SLF001
    except ValueError:
        prior_worker_invalid = True
    else:
        prior_worker_invalid = False
    if any(
        (
            evidence.get("schema") != "qwen38_fleet_dev17_seed44_stageb_v2_worker_id_failure_v1",
            evidence.get("live_evidence_binding")
            != {
                "kubernetes_context": EXPECTED_CONTEXT,
                "namespace": NAMESPACE,
                "context_asserted_before_every_live_read": True,
            },
            failed.get("plan_sha256") != prior["sha256"],
            failed_job.get("name") != prior_execution["job_name"],
            failed_job.get("condition") != "Failed",
            failed_job.get("reason") != "BackoffLimitExceeded",
            failed_job.get("root_failure_alert_annotation") != "off",
            failed_job.get("backoff_limit") != 0,
            failed_pod.get("phase") != "Failed",
            failed_pod.get("repair_restart_count") != 0,
            failed_pod.get("repair_exit_code") != 1,
            failed_pod.get("dind_restart_count") != 0,
            failed_pod.get("dind_exit_code") != 0,
            failed["config_map"].get("name") != prior_execution["config_map_name"],
            failed["secret"].get("name") != prior_execution["secret_name"],
            failed_workload.get("finished") is not True,
            failed_workload.get("reason") != "Failed",
            failed.get("controller_terminal", {}).get("accepted") is not False,
            failed.get("controller_terminal", {}).get("controller_failure_code")
            != "reviewed_recovery_v2_startup_failed",
            signature.get("stage") != "evaluate._name_before_output_root_or_intent_apply",
            signature.get("error_type") != "valueerror",
            signature.get("error_message_plaintext") != expected_error,
            signature.get("error_message_sha256") != expected_error_sha256,
            signature.get("worker_id") != prior_worker,
            signature.get("worker_id_contract") != r"[a-z0-9][a-z0-9-]{0,40}",
            signature.get("worker_id_contract_satisfied") is not False,
            signature.get("same_as_predecessor_v1") is not True,
            signature.get("docker_image_staging_started") is not False,
            prior_worker_invalid is not True,
            diagnostic.get("phase") != "Failed",
            diagnostic.get("restart_count") != 0,
            diagnostic.get("exit_code") != 1,
            diagnostic.get("failure_stage") != "entry",
            diagnostic.get("error_type") != "valueerror",
            diagnostic.get("error_message_sha256") != expected_error_sha256,
            diagnostic.get("output_root_created") is not False,
            diagnostic.get("ledger_mutations") != 0,
            diagnostic.get("model_generation_calls") != 0,
            diagnostic.get("scoring_calls") != 0,
            diagnostic.get("objects_released_after_terminal_evidence") is not True,
            preflight.get("phase") != "Succeeded",
            preflight.get("restart_count") != 0,
            preflight.get("exit_code") != 0,
            preflight.get("receipt_sha256")
            != "sha256:e012b8fe162f378c85bbf4976629f20344cc056c4c121be51bc05e1da842aed7",
            preflight.get("ledger_census")
            != {
                "total": 17,
                "accepted": 15,
                "retry_review": 2,
                "local_results": 15,
                "active": 0,
                "stale_active": 0,
            },
            selected
            != {
                "count": 2,
                "retry_review": 2,
                "retry_count_zero": 2,
                "reconciliation_digest_null": 2,
                "session_id_null": 2,
                "receipt_digest_null": 2,
                "local_result_count": 0,
                "matching_recovery_apply_receipt_count": 0,
                "exact_authoritative_session_count": 0,
                "model_execution_artifact_count": 0,
                "scoring_artifact_count": 0,
            },
            preflight.get("predecessor_v1_output_absent") is not True,
            preflight.get("failed_successor_v2_output_absent") is not True,
            preflight.get("route_ready") is not True,
            preflight.get("ledger_mutations") != 0,
            preflight.get("model_generation_calls") != 0,
            preflight.get("scoring_calls") != 0,
            preflight.get("accepted_cell_replays") != 0,
            preflight.get("objects_released_after_terminal_evidence") is not True,
            resources
            != {
                "evaluator_gpu_request": 0,
                "evaluator_pod_terminal": True,
                "workload_finished": True,
                "active_gpu_allocation": 0,
                "diagnostic_objects_released": True,
            },
            classification
            != {
                "deterministic_repository_owned_source_defect": True,
                "prior_attempts_crossed_mutation_boundary": False,
                "prior_attempts_applied_intent": False,
                "prior_attempts_started_model_generation": False,
                "prior_attempts_started_scoring": False,
                "corrected_successor_allowed_after_review": True,
                "final_eight_task_set_accessed": False,
            },
            any(item is not False for item in evidence.get("privacy", {}).values()),
        )
    ):
        raise PackageError("corrected Stage B failure evidence semantics differ")

    contract = value.get("contract")
    if contract != {
        "selected_cell_count": 2,
        "infrastructure_successor_generation": 3,
        "scientific_execution_generation": 2,
        "scientific_generation_two_was_never_opened": True,
        "accepted_cell_replay_count": 0,
        "stored_session_rescore_count": 0,
        "stored_session_regeneration_count": 0,
        "final_eight_task_set_accessed": False,
        "identical_private_intent_required": True,
        "distinct_deterministic_fix": "runtime_worker_name_grammar",
        "same_signature_successor_allowed": False,
    }:
        raise PackageError("corrected Stage B successor contract differs")
    if correction.get("infrastructure_successor_generation") != 3 or correction.get(
        "scientific_execution_generation"
    ) != prior_execution.get("execution_generation"):
        raise PackageError("corrected Stage B generation binding differs")
    if (
        correction.get("scientific_generation_two_was_never_opened") is not True
        or correction.get("worker_id_contract_satisfied") is not True
        or correction.get("same_signature_replay_allowed") is not False
        or correction.get("distinct_deterministic_fix_required") is not True
    ):
        raise PackageError("corrected Stage B repair classification differs")

    overrides = value.get("stage_overrides")
    if not isinstance(overrides, dict) or set(overrides) != {
        "job_name",
        "config_map_name",
        "secret_name",
        "output_root",
        "worker_id",
        "run_script_sha256",
    }:
        raise PackageError("corrected Stage B successor overrides are invalid")
    for field in ("job_name", "config_map_name", "secret_name"):
        if (
            not isinstance(overrides[field], str)
            or re.fullmatch(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?", overrides[field]) is None
            or overrides[field] == prior_execution[field]
            or overrides[field] != correction[field]
        ):
            raise PackageError("corrected Stage B Kubernetes identity is invalid")
    if any(
        (
            not isinstance(overrides["output_root"], str),
            not overrides["output_root"].startswith("/mnt/sfs/jobs/"),
            ".." in Path(overrides["output_root"]).parts,
            overrides["output_root"] == prior_execution["output_root"],
            overrides["output_root"] != correction["output_root"],
        )
    ):
        raise PackageError("corrected Stage B output identity is invalid")
    worker_id = _worker_name(overrides["worker_id"])
    if worker_id == prior_worker or worker_id != correction["worker_id"]:
        raise PackageError("corrected Stage B worker identity is not a distinct fix")
    if (
        not isinstance(overrides["run_script_sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", overrides["run_script_sha256"]) is None
    ):
        raise PackageError("corrected Stage B run script digest is invalid")

    effective["stages"]["single_rollout_repair"].update(overrides)
    effective["_successor_binding"] = {
        "successor_plan_sha256": value["sha256"],
        "predecessor_successor_plan_sha256": prior["sha256"],
        "predecessor_evidence_receipt_sha256": evidence["receipt_sha256"],
        "predecessor_job_uid": failed_job["uid"],
        "successor_preflight_receipt_sha256": preflight["receipt_sha256"],
        "infrastructure_successor_generation": 3,
        "scientific_execution_generation": prior_execution["execution_generation"],
    }
    return effective


def _base_job(
    *,
    execution: dict[str, Any],
    source: dict[str, Any],
    config_map_name: str,
    secret_name: str,
    dind: bool,
) -> dict[str, Any]:
    job_name = execution["job_name"]
    labels = {
        "cyber-post-train.fleet.ai/experiment": job_name,
        "cyber-post-train.fleet.ai/owner": "chris",
        "kueue.x-k8s.io/queue-name": "training-lq",
    }
    volumes = [
        {"name": "bootstrap", "configMap": {"name": config_map_name, "defaultMode": 0o444}},
        {"name": "intent", "secret": {"secretName": secret_name, "defaultMode": 0o400}},
        {"name": "workspace", "emptyDir": {"sizeLimit": "5Gi"}},
        {"name": "sfs", "persistentVolumeClaim": {"claimName": "sfs-shared"}},
    ]
    mounts = [
        {"name": "bootstrap", "mountPath": "/bootstrap", "readOnly": True},
        {"name": "intent", "mountPath": "/intent", "readOnly": True},
        {"name": "workspace", "mountPath": "/workspace"},
        {"name": "sfs", "mountPath": "/mnt/sfs"},
    ]
    env = [
        {"name": "EVALUATION_DIRECTORY", "value": source["evaluation_directory"]},
        {"name": "EVALUATION_DATABASE", "value": source["database"]},
        {"name": "REPAIR_OUTPUT", "value": execution["output_root"]},
        {
            "name": "FLEET_API_KEY",
            "valueFrom": {
                "secretKeyRef": {
                    "name": "chris-cyber-opencode-evals-v2",
                    "key": "FLEET_API_KEY",
                }
            },
        },
        {
            "name": "ROLLOUT_DATABASE_URL",
            "valueFrom": {
                "secretKeyRef": {
                    "name": "chris-cyber-rollout-postgres-v1",
                    "key": "ROLLOUT_DATABASE_URL",
                }
            },
        },
    ]
    container = {
        "name": "repair",
        "image": EVALUATOR_IMAGE,
        "command": ["/bin/bash", "-ceu", "--"],
        "args": [
            (
                "apt-get update\n"
                "apt-get install --yes --no-install-recommends docker.io\n"
                "exec /bin/bash /bootstrap/run.sh\n"
                if dind
                else "exec /bin/bash /bootstrap/run.sh\n"
            )
        ],
        "env": env,
        "resources": execution["resources"],
        "volumeMounts": mounts,
    }
    pod_spec: dict[str, Any] = {
        "priorityClassName": "c1",
        "restartPolicy": "Never",
        "nodeSelector": {
            "kubernetes.io/arch": "amd64",
            "workload": "fleetai-training-ng-cpu",
        },
        "tolerations": [
            {
                "key": "workload",
                "operator": "Equal",
                "value": "fleetai-training-ng-cpu",
                "effect": "NoSchedule",
            }
        ],
        "containers": [container],
        "volumes": volumes,
    }
    if dind:
        staging = execution["image_staging"]
        env[:0] = [
            {"name": "DOCKER_HOST", "value": "unix:///var/run/docker.sock"},
            {"name": "DOCKER_TLS_CERTDIR", "value": ""},
            {"name": "DOCKER_BIND_ROOT", "value": "/docker-bind"},
            {"name": "HARNESS_TAR", "value": staging["harness_tar"]},
            {"name": "HARNESS_TAR_SHA256", "value": staging["harness_tar_sha256"]},
            {"name": "HARNESS_RECEIPT", "value": staging["harness_receipt"]},
            {
                "name": "HARNESS_RECEIPT_SHA256",
                "value": staging["harness_receipt_sha256"],
            },
        ]
        volumes[2:2] = [
            {"name": "docker-data", "emptyDir": {"sizeLimit": "80Gi"}},
            {"name": "docker-socket", "emptyDir": {}},
            {"name": "docker-bind", "emptyDir": {"sizeLimit": "2Gi"}},
        ]
        mounts[:0] = [
            {"name": "docker-socket", "mountPath": "/var/run"},
            {"name": "docker-bind", "mountPath": "/docker-bind"},
        ]
        pod_spec["initContainers"] = [
            {
                "name": "dind",
                "image": DIND_IMAGE,
                "restartPolicy": "Always",
                "args": ["--host=unix:///var/run/docker.sock", "--tls=false"],
                "securityContext": {"privileged": True},
                "resources": {
                    "requests": {
                        "cpu": "2",
                        "memory": "2Gi",
                        "ephemeral-storage": "10Gi",
                    },
                    "limits": {
                        "cpu": "4",
                        "memory": "4Gi",
                        "ephemeral-storage": "40Gi",
                    },
                },
                "volumeMounts": [
                    {"name": "docker-socket", "mountPath": "/var/run"},
                    {"name": "docker-data", "mountPath": "/var/lib/docker"},
                    {"name": "docker-bind", "mountPath": "/docker-bind"},
                    {"name": "workspace", "mountPath": "/workspace"},
                    {"name": "sfs", "mountPath": "/mnt/sfs"},
                ],
            }
        ]
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job_name,
            "namespace": NAMESPACE,
            "labels": labels,
            "annotations": {
                FAILURE_ALERT_ANNOTATION: "off",
                "cyber-post-train.fleet.ai/create-once": "true",
            },
        },
        "spec": {
            "backoffLimit": 0,
            "activeDeadlineSeconds": execution["active_deadline_seconds"],
            "template": {
                "metadata": {
                    "labels": {
                        "cyber-post-train.fleet.ai/experiment": job_name,
                        "cyber-post-train.fleet.ai/owner": "chris",
                        "cyber-post-train.fleet.ai/postgres-client": "true",
                    }
                },
                "spec": pod_spec,
            },
        },
    }


def _stage(
    *,
    repo_root: Path,
    stage: str,
    plan: dict[str, Any],
    intent_path: Path,
) -> StagePackage:
    source = plan["source"]
    execution = plan["stages"][stage]
    if stage == "accept_existing_scored_session":
        intent = stored_v2.load_intent(intent_path)
        files = STORED_CODE_FILES
        run_script = _stored_script()
        expected_count = 5
        dind = False
    elif stage == "single_rollout_repair":
        intent = reviewed_recovery_v2.load_intent(intent_path)
        files = ROLLOUT_CODE_FILES
        worker_id = _worker_name(execution.get("worker_id", "q38_s44_base_repair"))
        run_script = _rollout_script(worker_id)
        expected_count = 2
        dind = True
        if execution.get("execution_generation") != 2:
            raise PackageError("rollout repair must use fresh execution generation two")
        staging = execution.get("image_staging")
        if not isinstance(staging, dict) or set(staging) != {
            "harness_tar",
            "harness_tar_sha256",
            "harness_receipt",
            "harness_receipt_sha256",
        }:
            raise PackageError("rollout repair image staging binding is invalid")
        if any(
            not isinstance(staging[key], str)
            or not staging[key].startswith("/mnt/sfs/jobs/")
            or ".." in Path(staging[key]).parts
            for key in ("harness_tar", "harness_receipt")
        ) or any(
            not isinstance(staging[key], str)
            or len(staging[key]) != 71
            or not staging[key].startswith("sha256:")
            or any(character not in "0123456789abcdef" for character in staging[key][7:])
            for key in ("harness_tar_sha256", "harness_receipt_sha256")
        ):
            raise PackageError("rollout repair image staging binding is invalid")
    else:
        raise PackageError("unknown narrow repair stage")
    code, code_sha256 = _code(repo_root, files)
    expected_code = plan["code_sha256"][stage]
    if code_sha256 != expected_code or _sha256(run_script.encode()) != execution.get(
        "run_script_sha256"
    ):
        raise PackageError("narrow repair runtime bytes differ from reviewed plan")
    if any(
        (
            intent.evaluation_plan_sha256
            != source["evaluation_plan_sha256"].removeprefix("sha256:"),
            intent.source_output_root != source["evaluation_directory"],
            intent.source_database != source["database"],
            intent.source_job_uid != source["job_uid"],
            intent.source_job_terminal_receipt_sha256
            != source["terminal_evidence_sha256"].removeprefix("sha256:"),
            len(intent.selected_cell_ids) != expected_count,
        )
    ):
        raise PackageError("private intent differs from narrow repair public plan")
    code["run.sh"] = run_script
    config_map_name = execution["config_map_name"]
    secret_name = execution["secret_name"]
    config_map = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": config_map_name, "namespace": NAMESPACE},
        "immutable": True,
        "data": code,
    }
    secret = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {"name": secret_name, "namespace": NAMESPACE},
        "immutable": True,
        "type": "Opaque",
        "stringData": {"intent.json": intent_path.read_text(encoding="utf-8")},
    }
    job = _base_job(
        execution=execution,
        source=source,
        config_map_name=config_map_name,
        secret_name=secret_name,
        dind=dind,
    )
    proof_body = {
        "schema": "qwen38-fleet-seed44-base-repair-stage-proof-v1",
        "stage": stage,
        "intent_sha256": "sha256:" + intent.sha256,
        "selected_cell_count": expected_count,
        "job_name": execution["job_name"],
        "config_map_name": config_map_name,
        "secret_name": secret_name,
        "output_root": execution["output_root"],
        "execution_generation": execution.get("execution_generation"),
        "root_failure_alert_annotation": job["metadata"]["annotations"][FAILURE_ALERT_ANNOTATION],
        "priority_class": "c1",
        "gpu_request": 0,
        "private_intent_content_included": False,
        "image_staging": execution.get("image_staging"),
    }
    if successor := plan.get("_successor_binding"):
        proof_body["successor_binding"] = successor
    proof = {**proof_body, "sha256": _canonical_digest(proof_body)}
    package = StagePackage(
        stage=stage,
        config_map=config_map,
        secret=secret,
        job=job,
        proof=proof,
    )
    validate(package)
    return package


def render(
    *,
    repo_root: Path,
    plan_path: Path,
    stored_intent_path: Path,
    recovery_intent_path: Path,
) -> dict[str, StagePackage]:
    source_plan = _load(plan_path, "narrow repair plan")
    is_successor = source_plan.get("schema") in {
        SUCCESSOR_SCHEMA,
        CORRECTED_SUCCESSOR_SCHEMA,
    }
    if source_plan.get("schema") == SUCCESSOR_SCHEMA:
        plan = _successor_plan(repo_root, source_plan)
    elif source_plan.get("schema") == CORRECTED_SUCCESSOR_SCHEMA:
        plan = _corrected_successor_plan(repo_root, source_plan)
    else:
        plan = source_plan
        if plan.get("schema") != SCHEMA:
            raise PackageError("narrow repair plan schema is unsupported")
        if plan.get("sha256") != _canonical_digest(
            {key: item for key, item in plan.items() if key != "sha256"}
        ):
            raise PackageError("narrow repair plan self digest differs")
    if plan.get("live_evidence_binding") != {
        "kubernetes_context": EXPECTED_CONTEXT,
        "namespace": NAMESPACE,
        "assert_before_every_live_read": True,
        "object_absence_valid_only_after_exact_binding_assertion": True,
    }:
        raise PackageError("narrow repair live evidence binding differs")
    source = plan.get("source")
    if not isinstance(source, dict):
        raise PackageError("narrow repair source is invalid")
    evidence_path = _safe_repo_path(
        repo_root, source.get("terminal_evidence_path"), "source terminal evidence"
    )
    evidence = _load(evidence_path, "source terminal evidence")
    if _sha256(evidence_path.read_bytes()) != source.get(
        "terminal_evidence_file_sha256", ""
    ).removeprefix("sha256:") or evidence.get("sha256") != source.get(
        "terminal_evidence_sha256", ""
    ).removeprefix("sha256:"):
        raise PackageError("source terminal evidence digest differs")
    _terminal_semantics(evidence, source)
    stored_intent = stored_v2.load_intent(stored_intent_path)
    stored = None
    if not is_successor:
        stored = _stage(
            repo_root=repo_root,
            stage="accept_existing_scored_session",
            plan=plan,
            intent_path=stored_intent_path,
        )
    recovery = _stage(
        repo_root=repo_root,
        stage="single_rollout_repair",
        plan=plan,
        intent_path=recovery_intent_path,
    )
    recovery_intent = reviewed_recovery_v2.load_intent(recovery_intent_path)
    if recovery_intent.prior_stored_session_intent_sha256 != stored_intent.sha256:
        raise PackageError("rollout repair is not chained to stored-session acceptance")
    if is_successor:
        return {"single_rollout_repair": recovery}
    return {"accept_existing_scored_session": stored, "single_rollout_repair": recovery}


def validate(package: StagePackage) -> None:
    job = package.job
    pod = job.get("spec", {}).get("template", {}).get("spec", {})
    containers = pod.get("containers") or []
    if any(
        (
            job.get("metadata", {}).get("namespace") != NAMESPACE,
            job.get("metadata", {}).get("annotations", {}).get(FAILURE_ALERT_ANNOTATION) != "off",
            job.get("spec", {}).get("backoffLimit") != 0,
            pod.get("priorityClassName") != "c1",
            len(containers) != 1,
            "nvidia.com/gpu" in json.dumps(job),
            package.config_map.get("immutable") is not True,
            package.secret.get("immutable") is not True,
            package.proof.get("gpu_request") != 0,
            package.proof.get("private_intent_content_included") is not False,
            package.proof.get("output_root") != f"/mnt/sfs/jobs/{job['metadata']['name']}",
        )
    ):
        raise PackageError("narrow repair package violates the execution contract")


def _write_private_json(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if fd >= 0:
            os.close(fd)


def write_private_packages(
    *,
    repo_root: Path,
    plan_path: Path,
    stored_intent_path: Path,
    recovery_intent_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Write both secret-bearing Kubernetes List manifests without printing them."""

    if output_root.exists():
        raise FileExistsError("create-once private repair package output already exists")
    packages = render(
        repo_root=repo_root,
        plan_path=plan_path,
        stored_intent_path=stored_intent_path,
        recovery_intent_path=recovery_intent_path,
    )
    output_root.mkdir(parents=True, mode=0o700)
    stages = {}
    for stage, package in packages.items():
        stage_root = output_root / stage
        stage_root.mkdir(mode=0o700)
        _write_private_json(stage_root / "bundle.json", package.bundle)
        _write_private_json(stage_root / "proof.json", package.proof)
        stages[stage] = {
            "job_name": package.job["metadata"]["name"],
            "root_failure_alert_annotation": package.job["metadata"]["annotations"][
                FAILURE_ALERT_ANNOTATION
            ],
            "selected_cell_count": package.proof["selected_cell_count"],
            "execution_generation": package.proof["execution_generation"],
            "bundle_file_sha256": "sha256:" + _sha256((stage_root / "bundle.json").read_bytes()),
            "proof_sha256": package.proof["sha256"],
            "private_intent_content_included": False,
        }
    body = {
        "schema": "qwen38-fleet-seed44-base-private-repair-packages-v1",
        "repair_plan_sha256": _load(plan_path, "narrow repair plan")["sha256"],
        "stages": stages,
        "jobs_created": 0,
        "config_maps_created": 0,
        "secrets_created": 0,
        "database_cells_changed": 0,
        "model_calls": 0,
        "scoring_calls": 0,
        "private_cell_task_session_or_trace_identifiers_included": False,
    }
    receipt = {**body, "sha256": _canonical_digest(body)}
    _write_private_json(output_root / "PREPARED.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--stored-intent", type=Path, required=True)
    parser.add_argument("--recovery-intent", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = write_private_packages(
            repo_root=args.repo_root,
            plan_path=args.plan,
            stored_intent_path=args.stored_intent,
            recovery_intent_path=args.recovery_intent,
            output_root=args.output_root,
        )
    except BaseException as exc:  # noqa: BLE001
        print(
            json.dumps(
                {
                    "schema": "qwen38-fleet-seed44-base-private-repair-packages-v1",
                    "status": "failed",
                    "controller_failure_code": type(exc).__name__.lower(),
                    "private_content_included": False,
                    "jobs_created": 0,
                },
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
