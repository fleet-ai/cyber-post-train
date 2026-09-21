"""Render one immutable CPU-only stored-session reconciliation Job package."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evals.fleet import stored_session_reconciliation as reconciliation

SCHEMA = "fleet-stored-session-reconciliation-job-v1"
FAILURE_ALERT_ANNOTATION = "fleet.ai/failure-alerts"
CODE_FILES = {
    "jobs.py": "cyber_post_train/jobs.py",
    "evaluate.py": "evals/fleet/evaluate.py",
    "exact_pass4_crypto.py": "evals/fleet/exact_pass4_crypto.py",
    "exact_pass4_universe.py": "evals/fleet/exact_pass4_universe.py",
    "fixed_proxy.py": "evals/fleet/fixed_proxy.py",
    "opencode_self_hosted.py": "evals/fleet/opencode_self_hosted.py",
    "retry_review_policy.py": "evals/fleet/retry_review_policy.py",
    "rollout_ledger.py": "evals/fleet/rollout_ledger.py",
    "rollout_campaign.py": "evals/fleet/rollout_campaign.py",
    "rollout_postgres.py": "evals/fleet/rollout_postgres.py",
    "rollout_worker.py": "evals/fleet/rollout_worker.py",
    "stored_session_reconciliation.py": "evals/fleet/stored_session_reconciliation.py",
}
TERMINAL_RECEIPT_FIELDS = {
    "schema",
    "observed_at",
    "scope",
    "frozen_evaluation",
    "source_controller",
    "score_blind_ledger_census",
    "serving_release",
    "decision",
    "privacy",
    "sha256",
}
FROZEN_EVALUATION_FIELDS = {
    "evaluation_plan_sha256",
    "database",
    "output_root",
    "task_count",
    "pass_k",
    "final_eight_task_set_accessed",
}
SOURCE_CONTROLLER_FIELDS = {"job", "pod", "workload"}
SOURCE_JOB_FIELDS = {
    "name",
    "uid",
    "resource_version",
    "condition",
    "reason",
    "terminal_at",
    "active",
    "failed",
    "root_failure_alert_annotation",
    "backoff_limit",
}
SOURCE_POD_FIELDS = {
    "name",
    "uid",
    "resource_version",
    "phase",
    "evaluator_exit_code",
    "evaluator_restart_count",
    "finished_at",
}
SOURCE_WORKLOAD_FIELDS = {
    "name",
    "uid",
    "resource_version",
    "finished",
    "reason",
    "finished_at",
}
LEDGER_CENSUS_FIELDS = {
    "total_cells",
    "accepted",
    "retry_review",
    "claimed",
    "stale_active",
    "pending",
    "local_results",
    "unresolved_cells",
    "unresolved_exact_completed_sessions",
    "unresolved_exact_model_matches",
    "unresolved_exact_pinned_task_version_metadata_matches",
    "unresolved_reference_trace_payloads_empty",
    "unresolved_exact_verifier_matches",
    "unresolved_authoritative_scores_finite_and_equal_to_private_local_results",
    "unresolved_completed_ingests",
    "unresolved_process_error_exit_one",
    "rollout_regeneration_eligible",
    "stored_session_only_reconciliation_eligible",
    "score_values_read_for_branching",
    "score_values_included",
}
SERVING_RELEASE_FIELDS = {
    "model_id",
    "inference_model_uid",
    "resource_version",
    "generation",
    "phase",
    "active_pods",
    "ready_pods",
    "deployment_uid",
    "deployment_replicas",
    "last_useful_request_at",
    "stored_session_reconciliation_requires_model_route",
}
DECISION_FIELDS = {
    "capability_result_status",
    "never_regenerate_unresolved_rollouts",
    "never_repeat_scoring_for_already_scored_sessions",
    "accept_only_after_two_identical_metadata_observations_and_one_atomic_database_transaction",
    "model_generation_performed_by_reconciliation",
    "scoring_call_performed_by_reconciliation",
}
PRIVACY_FIELDS = {
    "score_values_included",
    "prompts_responses_flags_rewards_or_trace_content_included",
    "private_cell_task_session_or_trace_identifiers_included",
    "credentials_included",
}


class PackageError(ValueError):
    """The reviewed reconciliation package is incomplete or drifted."""


def _exact_object(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise PackageError(f"source terminal receipt {name} shape is invalid")
    return value


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise PackageError(f"source terminal receipt {name} digest is invalid")
    normalized = value.removeprefix("sha256:")
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise PackageError(f"source terminal receipt {name} digest is invalid")
    return normalized


def _terminal_receipt_semantics(
    receipt: dict[str, Any],
    *,
    source: dict[str, Any],
    intent: reconciliation.StoredSessionIntent,
) -> None:
    _exact_object(receipt, TERMINAL_RECEIPT_FIELDS, "root")
    frozen = _exact_object(receipt["frozen_evaluation"], FROZEN_EVALUATION_FIELDS, "evaluation")
    controller = _exact_object(receipt["source_controller"], SOURCE_CONTROLLER_FIELDS, "controller")
    job = _exact_object(controller["job"], SOURCE_JOB_FIELDS, "Job")
    pod = _exact_object(controller["pod"], SOURCE_POD_FIELDS, "Pod")
    workload = _exact_object(controller["workload"], SOURCE_WORKLOAD_FIELDS, "Workload")
    census = _exact_object(
        receipt["score_blind_ledger_census"], LEDGER_CENSUS_FIELDS, "ledger census"
    )
    serving = _exact_object(receipt["serving_release"], SERVING_RELEASE_FIELDS, "serving release")
    decision = _exact_object(receipt["decision"], DECISION_FIELDS, "decision")
    privacy = _exact_object(receipt["privacy"], PRIVACY_FIELDS, "privacy")
    for name, value in {
        "Job UID": job["uid"],
        "Pod UID": pod["uid"],
        "Workload UID": workload["uid"],
        "inference model UID": serving["inference_model_uid"],
        "deployment UID": serving["deployment_uid"],
    }.items():
        try:
            uuid.UUID(value)
        except (AttributeError, TypeError, ValueError) as exc:
            raise PackageError(f"source terminal receipt {name} is invalid") from exc
    integer_values = {
        "task count": frozen["task_count"],
        "pass k": frozen["pass_k"],
        "Job active count": job["active"],
        "Job failed count": job["failed"],
        "Job backoff limit": job["backoff_limit"],
        "Pod exit code": pod["evaluator_exit_code"],
        "Pod restart count": pod["evaluator_restart_count"],
        "serving generation": serving["generation"],
        "active serving Pod count": serving["active_pods"],
        "ready serving Pod count": serving["ready_pods"],
        "serving deployment replicas": serving["deployment_replicas"],
        **{
            name: census[name]
            for name in LEDGER_CENSUS_FIELDS
            if name
            not in {
                "score_values_read_for_branching",
                "score_values_included",
            }
        },
    }
    if any(type(value) is not int or value < 0 for value in integer_values.values()):
        raise PackageError("source terminal receipt count is invalid")
    unresolved = census["retry_review"] + census["stale_active"]
    unresolved_facts = (
        "unresolved_cells",
        "unresolved_exact_completed_sessions",
        "unresolved_exact_model_matches",
        "unresolved_exact_pinned_task_version_metadata_matches",
        "unresolved_reference_trace_payloads_empty",
        "unresolved_exact_verifier_matches",
        "unresolved_authoritative_scores_finite_and_equal_to_private_local_results",
        "unresolved_completed_ingests",
        "unresolved_process_error_exit_one",
        "stored_session_only_reconciliation_eligible",
    )
    if any(
        (
            receipt["schema"] != "qwen38_lr30_step76_fleet_dev17_seed43_terminal_census_v1",
            not isinstance(receipt["observed_at"], str),
            not isinstance(receipt["scope"], str),
            _digest(frozen["evaluation_plan_sha256"], "evaluation plan")
            != intent.evaluation_plan_sha256,
            _digest(source.get("evaluation_plan_sha256"), "public evaluation plan")
            != intent.evaluation_plan_sha256,
            frozen["database"] != source.get("database"),
            frozen["output_root"] != source.get("evaluation_directory"),
            frozen["output_root"] != intent.source_output_root,
            frozen["pass_k"] != 1,
            frozen["final_eight_task_set_accessed"] is not False,
            job["name"] != source.get("job_name"),
            job["uid"] != source.get("job_uid"),
            job["uid"] != intent.source_job_uid,
            job["condition"] != "Failed",
            job["reason"] != "BackoffLimitExceeded",
            job["active"] != 0,
            job["failed"] != 1,
            job["root_failure_alert_annotation"] != "off",
            job["backoff_limit"] != 0,
            pod["phase"] != "Failed",
            pod["evaluator_exit_code"] != 1,
            pod["evaluator_restart_count"] != 0,
            workload["finished"] is not True,
            workload["reason"] != "Failed",
            census["total_cells"]
            != census["accepted"] + census["retry_review"] + census["claimed"] + census["pending"],
            census["total_cells"] != frozen["task_count"],
            census["local_results"] != census["total_cells"],
            census["claimed"] != census["stale_active"],
            unresolved != len(intent.selected_cell_ids),
            any(census[name] != unresolved for name in unresolved_facts),
            census["rollout_regeneration_eligible"] != 0,
            census["score_values_read_for_branching"] is not False,
            census["score_values_included"] is not False,
            serving["phase"] != "paused",
            serving["active_pods"] != 0,
            serving["ready_pods"] != 0,
            serving["deployment_replicas"] != 0,
            serving["stored_session_reconciliation_requires_model_route"] is not False,
            decision["capability_result_status"]
            != "infrastructure_incomplete_pending_stored_session_reconciliation",
            decision["never_regenerate_unresolved_rollouts"] is not True,
            decision["never_repeat_scoring_for_already_scored_sessions"] is not True,
            decision[
                "accept_only_after_two_identical_metadata_observations_and_one_atomic_database_transaction"
            ]
            is not True,
            decision["model_generation_performed_by_reconciliation"] is not False,
            decision["scoring_call_performed_by_reconciliation"] is not False,
            any(value is not False for value in privacy.values()),
        )
    ):
        raise PackageError("source terminal receipt semantics differ from reviewed recovery")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PackageError(f"{path.name} is unreadable") from exc
    if not isinstance(value, dict):
        raise PackageError(f"{path.name} must contain an object")
    return value


@dataclass(frozen=True)
class Package:
    config_map: dict[str, Any]
    secret: dict[str, Any]
    job: dict[str, Any]
    proof: dict[str, Any]


def _run_script() -> str:
    modules = " ".join(
        name for name in CODE_FILES if name not in {"jobs.py", "stored_session_reconciliation.py"}
    )
    return f"""#!/usr/bin/env bash
set -euo pipefail
umask 077
: "${{FLEET_API_KEY:?FLEET_API_KEY is required}}"
: "${{ROLLOUT_DATABASE_URL:?ROLLOUT_DATABASE_URL is required}}"
: "${{EVALUATION_DATABASE:?EVALUATION_DATABASE is required}}"
: "${{EVALUATION_DIRECTORY:?EVALUATION_DIRECTORY is required}}"
: "${{RECONCILIATION_OUTPUT:?RECONCILIATION_OUTPUT is required}}"
root=/workspace/cyber-post-train
mkdir -p "$root/cyber_post_train" "$root/evals/fleet"
touch "$root/cyber_post_train/__init__.py" "$root/evals/__init__.py" "$root/evals/fleet/__init__.py"
install -m 0644 /bootstrap/jobs.py "$root/cyber_post_train/jobs.py"
for name in {modules}; do
  install -m 0644 "/bootstrap/$name" "$root/evals/fleet/$name"
done
install -m 0644 /bootstrap/stored_session_reconciliation.py \
  "$root/evals/fleet/stored_session_reconciliation.py"
cd "$root"
exec uv run --no-project --with httpx==0.28.1 --with pyyaml==6.0.3 \
  --with 'psycopg[binary]==3.3.5' python -m evals.fleet.stored_session_reconciliation \
  --evaluation-directory "$EVALUATION_DIRECTORY" \
  --output-root "$RECONCILIATION_OUTPUT" \
  --postgres-admin-dsn-env ROLLOUT_DATABASE_URL \
  --postgres-database "$EVALUATION_DATABASE" \
  --intent /intent/intent.json
"""


def render(*, repo_root: Path, plan_path: Path, intent_path: Path) -> Package:
    plan = _load(plan_path)
    intent = reconciliation.load_intent(intent_path)
    if plan.get("schema") != SCHEMA:
        raise PackageError("stored-session Job plan schema is unsupported")
    source = plan.get("source")
    runtime = plan.get("runtime")
    execution = plan.get("execution")
    if not all(isinstance(value, dict) for value in (source, runtime, execution)):
        raise PackageError("stored-session Job plan sections are malformed")
    terminal_receipt_relative = source.get("terminal_receipt_path")
    if (
        not isinstance(terminal_receipt_relative, str)
        or not terminal_receipt_relative
        or Path(terminal_receipt_relative).is_absolute()
        or ".." in Path(terminal_receipt_relative).parts
    ):
        raise PackageError("source terminal receipt path is invalid")
    terminal_receipt_path = repo_root / terminal_receipt_relative
    if any(
        (
            terminal_receipt_path.is_symlink(),
            not terminal_receipt_path.is_file(),
            repo_root.resolve() not in terminal_receipt_path.resolve().parents,
        )
    ):
        raise PackageError("source terminal receipt path is invalid")
    try:
        terminal_receipt = _load(terminal_receipt_path)
        terminal_receipt_sha256 = reconciliation._body_digest(  # noqa: SLF001
            {key: value for key, value in terminal_receipt.items() if key != "sha256"}
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PackageError("source terminal receipt is invalid") from exc
    if any(
        (
            terminal_receipt.get("sha256") != terminal_receipt_sha256,
            str(source.get("terminal_receipt_sha256", "")).removeprefix("sha256:")
            != terminal_receipt_sha256,
        )
    ):
        raise PackageError("source terminal receipt digest differs from reviewed plan")
    _terminal_receipt_semantics(terminal_receipt, source=source, intent=intent)
    if any(
        (
            intent.evaluation_plan_sha256
            != str(source.get("evaluation_plan_sha256", "")).removeprefix("sha256:"),
            intent.source_output_root != source.get("evaluation_directory"),
            intent.source_database != source.get("database"),
            intent.source_job_uid != source.get("job_uid"),
            intent.source_job_terminal_receipt_sha256
            != str(source.get("terminal_receipt_sha256", "")).removeprefix("sha256:"),
            len(intent.selected_cell_ids) != source.get("selected_cell_count"),
        )
    ):
        raise PackageError("private intent differs from reviewed public plan")
    code: dict[str, str] = {}
    observed_sha256: dict[str, str] = {}
    for name, relative in CODE_FILES.items():
        path = repo_root / relative
        value = path.read_text(encoding="utf-8")
        code[name] = value
        observed_sha256[relative] = _sha256(value.encode())
    if observed_sha256 != plan.get("code_sha256"):
        raise PackageError("stored-session source bytes differ from reviewed plan")
    run_script = _run_script()
    if _sha256(run_script.encode()) != runtime.get("run_script_sha256"):
        raise PackageError("stored-session bootstrap bytes differ from reviewed plan")
    code["run.sh"] = run_script
    namespace = execution["namespace"]
    config_map_name = execution["config_map_name"]
    secret_name = execution["secret_name"]
    job_name = execution["job_name"]
    config_map = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": config_map_name, "namespace": namespace},
        "immutable": True,
        "data": code,
    }
    secret = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {"name": secret_name, "namespace": namespace},
        "immutable": True,
        "type": "Opaque",
        "stringData": {"intent.json": intent_path.read_text(encoding="utf-8")},
    }
    labels = {
        "cyber-post-train.fleet.ai/experiment": job_name,
        "cyber-post-train.fleet.ai/owner": "chris",
        "kueue.x-k8s.io/queue-name": "training-lq",
    }
    job = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job_name,
            "namespace": namespace,
            "labels": labels,
            "annotations": {
                FAILURE_ALERT_ANNOTATION: "off",
                "cyber-post-train.fleet.ai/create-once": "true",
            },
        },
        "spec": {
            "backoffLimit": 0,
            "activeDeadlineSeconds": runtime["active_deadline_seconds"],
            "template": {
                "metadata": {
                    "labels": {
                        "cyber-post-train.fleet.ai/experiment": job_name,
                        "cyber-post-train.fleet.ai/owner": "chris",
                        "cyber-post-train.fleet.ai/postgres-client": "true",
                    }
                },
                "spec": {
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
                    "containers": [
                        {
                            "name": "reconciler",
                            "image": runtime["image"],
                            "command": ["/bin/bash", "/bootstrap/run.sh"],
                            "env": [
                                {
                                    "name": "EVALUATION_DIRECTORY",
                                    "value": source["evaluation_directory"],
                                },
                                {
                                    "name": "RECONCILIATION_OUTPUT",
                                    "value": execution["output_root"],
                                },
                                {
                                    "name": "EVALUATION_DATABASE",
                                    "value": intent.source_database,
                                },
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
                            ],
                            "resources": runtime["resources"],
                            "volumeMounts": [
                                {"name": "bootstrap", "mountPath": "/bootstrap", "readOnly": True},
                                {"name": "intent", "mountPath": "/intent", "readOnly": True},
                                {"name": "sfs", "mountPath": "/mnt/sfs"},
                            ],
                        }
                    ],
                    "volumes": [
                        {
                            "name": "bootstrap",
                            "configMap": {"name": config_map_name, "defaultMode": 0o444},
                        },
                        {
                            "name": "intent",
                            "secret": {"secretName": secret_name, "defaultMode": 0o400},
                        },
                        {
                            "name": "sfs",
                            "persistentVolumeClaim": {"claimName": "sfs-shared"},
                        },
                    ],
                },
            },
        },
    }
    proof_body = {
        "schema": "fleet-stored-session-reconciliation-package-proof-v1",
        "plan_sha256": _sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()),
        "intent_sha256": intent.sha256,
        "code_sha256": observed_sha256,
        "run_script_sha256": _sha256(run_script.encode()),
        "config_map_name": config_map_name,
        "secret_name": secret_name,
        "job_name": job_name,
        "source_database": intent.source_database,
        "root_failure_alert_annotation": job["metadata"]["annotations"][FAILURE_ALERT_ANNOTATION],
        "priority_class": "c1",
        "gpu_request": 0,
        "selected_cell_count": len(intent.selected_cell_ids),
        "model_generation_performed": False,
        "scoring_call_performed": False,
        "private_intent_content_included": False,
    }
    proof = {
        **proof_body,
        "sha256": _sha256(json.dumps(proof_body, sort_keys=True, separators=(",", ":")).encode()),
    }
    package = Package(config_map=config_map, secret=secret, job=job, proof=proof)
    validate(package, plan=plan)
    return package


def validate(package: Package, *, plan: dict[str, Any]) -> None:
    job = package.job
    metadata = job.get("metadata") or {}
    pod = ((job.get("spec") or {}).get("template") or {}).get("spec") or {}
    containers = pod.get("containers") or []
    environment = {
        item.get("name"): item
        for item in (containers[0].get("env") if len(containers) == 1 else []) or []
        if isinstance(item, dict)
    }
    if any(
        (
            metadata.get("annotations", {}).get(FAILURE_ALERT_ANNOTATION) != "off",
            pod.get("priorityClassName") != "c1",
            job.get("spec", {}).get("backoffLimit") != 0,
            len(containers) != 1,
            "nvidia.com/gpu" in json.dumps(containers[0].get("resources") or {}),
            package.config_map.get("immutable") is not True,
            package.secret.get("immutable") is not True,
            package.secret.get("metadata", {}).get("name") != plan["execution"]["secret_name"],
            environment.get("EVALUATION_DATABASE", {}).get("value") != plan["source"]["database"],
            package.proof.get("source_database") != plan["source"]["database"],
        )
    ):
        raise PackageError("stored-session package violates the reviewed execution contract")
