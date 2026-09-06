"""Fresh-generation held successors for the untouched hosted-Qwen ranks 15/16.

The statistical cells are unchanged from the failed pre-claim generation-19
objects.  Every execution, Job, ConfigMap, run and output identity is new.  A
separate score-blind observer must clear the release gate before these plans can
be rendered as launch-authorized workloads.
"""

from __future__ import annotations

import copy
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_bulk_runtime_v3 as engine
from evals.fleet import exact_pass4_universe as exact
from evals.fleet import qwen_hosted_whole_task_release_gate_contract_v1 as release_gate_contract
from evals.fleet import qwen_hosted_whole_task_successor_v2 as prior
from evals.fleet import self_hosted

SCHEMA = "fleet-qwen38-hosted-atomic-whole-task-plan-v4"
HELD_SCHEMA = "fleet-qwen38-hosted-atomic-whole-task-held-v10"
RELEASE_SCHEMA = "fleet-qwen38-hosted-atomic-whole-task-release-v6"
PACKAGE_SOURCE_SCHEMA = "fleet-qwen38-hosted-package-source-v3"
RUNTIME_GATE_V3_CANARY_SCHEMA = "fleet-qwen38-hosted-whole-task-runtime-gate-canary-v3"
EXECUTION_GENERATION = 21
RUNTIME_GATE_V3_CANARY_JOB = "chris-q38-hosted-whole-task-runtime-gate-canary-v7"
RELEASE_GATE_MAX_AGE_SECONDS = 3600
RELEASE_GATE_MAX_FUTURE_SKEW_SECONDS = 300
RELEASE_GATE_BINDING = {
    "binding_sha256": "sha256:fef9ebe1dd81e31e98ce61fb8ecaca19d059997ff83b8a034e96afbe3851d006",
    "predecessor_object_set_sha256": (
        "sha256:8d96b0dfdfa4e0bdd5c4c8c5d8e201b14def0b3aaf705b490d09768e67e4cff3"
    ),
    "fresh_object_set_sha256": (
        "sha256:2d9f4d00ffcd4043fbce7f2bf5b7ee2131a2adeff8eee1da46b1fee138a39218"
    ),
    "plan_set_sha256": "sha256:46d5fb9efb3a933a6791f195a7f6a8ae1effa91d051322a7c7d157465cb669df",
}
HELD_PATH = (
    "docs/evidence/qwen38-study/2026-09-06-qwen38-hosted-rank15-rank16-whole-task-held-v10.json"
)
PRIOR_CANARY_PASS = {
    "path": (
        "docs/evidence/qwen38-study/2026-09-06-qwen38-hosted-runtime-gate-canary-v5-pass.json"
    ),
    "receipt_sha256": "sha256:0d06e9bbb97c08bcb34588ddb093549a2d02d7508b53c42b11362502e4580902",
}
SUPERSEDED_HELD = {
    "path": prior.HELD_PATH,
    "receipt_sha256": "sha256:7223357495fbabcd7ee7bb07730c98576e82b1fa2b1b854ae40411d85733c3ba",
    "file_sha256": "sha256:a262e757a9f981ac6e5560ead543221ec023555b24f47a608545658ac6c562ed",
}
INCIDENT_EVIDENCE = {
    "path": "docs/evidence/qwen38-study/2026-09-06-qwen38-hosted-g20v2-adapter-failure-v1.json",
    "receipt_sha256": "sha256:6705c63327e3c9829171a4708fee58eca356a65f1a3004dd5de8966dbc1c327f",
    "file_sha256": "sha256:ec9c915bdec6e8dce387c8d9e373cf2d88bca5c0f541a4455cecfe249e53bbdb",
}
CONTROLLERS = {
    "qwen-a": {
        "rank": 15,
        "task_version_id": "fb8f2178-7dd8-429a-8d3c-f14b21a51e02",
        "job_name": "chris-q38-hosted-r015-whole-task-g21-v1",
        "configmap_name": "chris-q38-hosted-r015-whole-task-g21-package-v1",
    },
    "qwen-b": {
        "rank": 16,
        "task_version_id": "fd07b96f-7aa4-4041-862b-5aa7411eff4b",
        "job_name": "chris-q38-hosted-r016-whole-task-g21-v1",
        "configmap_name": "chris-q38-hosted-r016-whole-task-g21-package-v1",
    },
}

CLAIM_ROOT = prior.CLAIM_ROOT
RESERVATION_ROOT = prior.RESERVATION_ROOT
JOBS_ROOT = prior.JOBS_ROOT
ENDPOINT_LEASE_ROOT = prior.ENDPOINT_LEASE_ROOT
ENDPOINT_KEY = prior.ENDPOINT_KEY
CANARY_GATE_SCHEMA = prior.prior.CANARY_GATE_SCHEMA
COMMIT_RE = prior.prior.COMMIT_RE
RECONCILIATION_GATE_SCHEMA = prior.prior.RECONCILIATION_GATE_SCHEMA
SHA256_RE = prior.SHA256_RE
load = prior.load
_seal = prior._seal
validate_inventory_gate = prior.prior.validate_inventory_gate


def _rebind_attempt(controller: str, row: dict[str, Any]) -> dict[str, Any]:
    item = copy.deepcopy(row)
    execution = exact.execution_for(item["cell_id"], EXECUTION_GENERATION)
    run_id = (
        f"chris-q38-ac-g21v1-{controller[-1]}-r{item['selection_rank']:03d}-"
        f"a{item['attempt']}-{execution['execution_id'][7:15]}"
    )
    item.update(
        execution_generation=EXECUTION_GENERATION,
        execution_id=execution["execution_id"],
        run_id=run_id,
        network=run_id.removeprefix("chris-")[:63],
    )
    return item


def build_plans(root: Path) -> dict[str, dict[str, Any]]:
    predecessors = prior.build_plans(root)
    plans: dict[str, dict[str, Any]] = {}
    for controller, predecessor in predecessors.items():
        authority = CONTROLLERS[controller]
        body = copy.deepcopy(predecessor)
        body.pop("plan_sha256")
        body.update(
            schema_version=SCHEMA,
            campaign_id=authority["job_name"],
            source_job_id=authority["job_name"],
            sfs_root=f"/mnt/sfs/jobs/{authority['job_name']}",
            attempts=[_rebind_attempt(controller, row) for row in predecessor["attempts"]],
            predecessor_plan_sha256=predecessor["plan_sha256"],
            launch_authorized=False,
            release_required=True,
            execution_rollforward={
                "statistical_cell_identity_unchanged": True,
                "predecessor_execution_generation": 20,
                "execution_generation": EXECUTION_GENERATION,
                "predecessor_objects_retry_forbidden": True,
                "fresh_object_and_execution_identities_required": True,
            },
        )
        plans[controller] = {
            **body,
            "plan_sha256": self_hosted.digest_without(body, "plan_sha256"),
        }
    validate(plans, predecessors)
    return plans


def validate(
    plans: dict[str, dict[str, Any]],
    predecessors: dict[str, dict[str, Any]] | None = None,
) -> None:
    predecessors = predecessors or prior.build_plans(Path.cwd())
    if set(plans) != set(CONTROLLERS) or set(predecessors) != set(CONTROLLERS):
        raise ValueError("fresh hosted whole-task controller set drifted")
    identities: set[tuple[int, int]] = set()
    for controller, plan in plans.items():
        authority = CONTROLLERS[controller]
        predecessor = predecessors[controller]
        attempts = plan.get("attempts") or []
        prior_attempts = predecessor["attempts"]
        immutable_fields = ("model", "harness", "authority", "tasks", "execution", "treatment")
        if any(plan.get(field) != predecessor.get(field) for field in immutable_fields):
            raise ValueError("fresh hosted whole-task scientific binding drifted")
        if any(
            (
                plan.get("schema_version") != SCHEMA,
                plan.get("controller") != controller,
                plan.get("campaign_id") != authority["job_name"],
                plan.get("source_job_id") != authority["job_name"],
                plan.get("sfs_root") != f"/mnt/sfs/jobs/{authority['job_name']}",
                plan.get("predecessor_plan_sha256") != predecessor["plan_sha256"],
                plan.get("launch_authorized") is not False,
                plan.get("release_required") is not True,
                plan.get("serving_block") != "qwen-hosted-autocontinue-v1",
                plan.get("model", {}).get("revision") != "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
                plan.get("harness", {}).get("version") != "1.18.27",
                plan.get("treatment", {}).get("tools") != ["bash", "submit_report"],
                plan.get("treatment", {}).get("context_management")
                != "opencode_1.18.27_native_compaction_autocontinue_v1",
                plan.get("treatment", {}).get("context_window_size") != 262144,
                len(plan.get("tasks") or []) != 1,
                plan["tasks"][0].get("rank") != authority["rank"],
                len(attempts) != 4,
                plan.get("execution_rollforward")
                != {
                    "statistical_cell_identity_unchanged": True,
                    "predecessor_execution_generation": 20,
                    "execution_generation": EXECUTION_GENERATION,
                    "predecessor_objects_retry_forbidden": True,
                    "fresh_object_and_execution_identities_required": True,
                },
                plan.get("plan_sha256") != self_hosted.digest_without(plan, "plan_sha256"),
            )
        ):
            raise ValueError("fresh hosted whole-task plan drifted")
        for row, old in zip(attempts, prior_attempts, strict=True):
            stable = {
                key: value
                for key, value in row.items()
                if key not in {"execution_generation", "execution_id", "run_id", "network"}
            }
            old_stable = {
                key: value
                for key, value in old.items()
                if key not in {"execution_generation", "execution_id", "run_id", "network"}
            }
            expected_execution = exact.execution_for(row["cell_id"], EXECUTION_GENERATION)
            if any(
                (
                    stable != old_stable,
                    row.get("execution_generation") != EXECUTION_GENERATION,
                    row.get("execution_id") != expected_execution["execution_id"],
                    row.get("execution_id") == old.get("execution_id"),
                    row.get("run_id") == old.get("run_id"),
                    row.get("selection_rank") != authority["rank"],
                    row.get("task_version_id") != authority["task_version_id"],
                )
            ):
                raise ValueError("fresh hosted whole-task execution rollforward drifted")
            identities.add((row["selection_rank"], row["attempt"]))
    if identities != {(rank, attempt) for rank in (15, 16) for attempt in range(1, 5)}:
        raise ValueError("fresh hosted whole-task statistical cell set drifted")


def build_runtime_plan(
    controller: str, inventory_receipt: dict[str, Any], root: Path
) -> dict[str, Any]:
    plan = build_plans(root)[controller]
    if plan["inventory_receipt"] != inventory_receipt:
        raise ValueError("fresh hosted whole-task inventory projection drifted")
    return plan


def validate_all(root: Path) -> dict[str, dict[str, Any]]:
    """Expose the complete bulk-runtime adapter contract and validate every plan."""
    return build_plans(root)


def package_source_receipt(
    controller: str, plan: dict[str, Any], data: dict[str, str]
) -> dict[str, Any]:
    if controller not in CONTROLLERS or plan.get("controller") != controller:
        raise ValueError("fresh hosted whole-task package controller drifted")
    if {"release.json", "package-source.json"}.intersection(data):
        raise ValueError("fresh hosted whole-task package includes mutable binding")
    files = {name: self_hosted.sha256(value.encode()) for name, value in sorted(data.items())}
    return _seal(
        {
            "schema_version": PACKAGE_SOURCE_SCHEMA,
            "controller": controller,
            "plan_sha256": plan["plan_sha256"],
            "configmap_name": CONTROLLERS[controller]["configmap_name"],
            "files": files,
            "file_count": len(files),
            "release_excluded": True,
            "scores_included": False,
            "prompts_or_traces_included": False,
        }
    )


def validate_package_source_receipt(
    receipt: dict[str, Any], controller: str, plan: dict[str, Any]
) -> None:
    files = receipt.get("files") or {}
    if any(
        (
            receipt.get("schema_version") != PACKAGE_SOURCE_SCHEMA,
            receipt.get("controller") != controller,
            receipt.get("plan_sha256") != plan["plan_sha256"],
            receipt.get("configmap_name") != CONTROLLERS[controller]["configmap_name"],
            not isinstance(files, dict),
            not files,
            any(SHA256_RE.fullmatch(str(value)) is None for value in files.values()),
            receipt.get("file_count") != len(files),
            receipt.get("release_excluded") is not True,
            receipt.get("scores_included") is not False,
            receipt.get("prompts_or_traces_included") is not False,
            receipt.get("receipt_sha256") != self_hosted.digest_without(receipt, "receipt_sha256"),
        )
    ):
        raise RuntimeError("fresh hosted whole-task package source receipt drifted")


def release_projection(
    plans: dict[str, dict[str, Any]], package_sources: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    return [
        {
            "controller": controller,
            "selection_rank": CONTROLLERS[controller]["rank"],
            "task_version_id": CONTROLLERS[controller]["task_version_id"],
            "job_name": CONTROLLERS[controller]["job_name"],
            "configmap_name": CONTROLLERS[controller]["configmap_name"],
            "sfs_root": plan["sfs_root"],
            "plan_sha256": plan["plan_sha256"],
            "package_source_receipt_sha256": package_sources[controller]["receipt_sha256"],
            "cells": [
                {
                    "attempt": row["attempt"],
                    "cell_id": row["cell_id"],
                    "execution_generation": row["execution_generation"],
                    "execution_id": row["execution_id"],
                    "run_id": row["run_id"],
                }
                for row in plan["attempts"]
            ],
        }
        for controller, plan in sorted(plans.items())
    ]


def validate_held(
    held: dict[str, Any],
    plans: dict[str, dict[str, Any]],
    package_sources: dict[str, dict[str, Any]],
) -> None:
    if held != expected_held(plans, package_sources):
        raise RuntimeError("fresh hosted whole-task held evidence drifted")


def expected_held(
    plans: dict[str, dict[str, Any]], package_sources: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    return _seal(
        {
            "schema_version": HELD_SCHEMA,
            "status": "HELD_PENDING_FRESH_SCORE_BLIND_GATES",
            "launch_authorized": False,
            "scoring_authorized": False,
            "controllers": release_projection(plans, package_sources),
            "supersedes": SUPERSEDED_HELD,
            "preclaim_failure": prior.prior.PRECLAIM_FAILURE,
            "prior_runtime_gate_canary_pass": PRIOR_CANARY_PASS,
            "incident_evidence": INCIDENT_EVIDENCE,
            "required_release_gate": {
                "fresh_uid_bound_score_blind_observer": True,
                "fresh_v3_runtime_gate_canary_pass": True,
                "observer_job_must_succeed": True,
                "independent_terminal_validation_required": True,
                "all_eight_statistical_cells_clear": True,
                "new_job_pod_configmap_and_sfs_identities_absent": True,
                "old_execution_identities_reconciled": True,
                "canonical_claims_clear": True,
                "accepted_receipts_clear": True,
                "authoritative_sessions_clear": True,
                "both_endpoint_lease_slots_simultaneously_free": True,
            },
            "privacy": {
                "scores_read": False,
                "prompts_traces_flags_read": False,
                "credentials_included": False,
            },
        }
    )


def validate_release(
    release: dict[str, Any],
    plans: dict[str, dict[str, Any]],
    package_sources: dict[str, dict[str, Any]],
    *,
    now: datetime | None = None,
) -> None:
    observation = release.get("release_gate_observation")
    if not isinstance(observation, dict):
        raise RuntimeError("fresh hosted whole-task release-gate observation drifted")
    try:
        release_gate_contract.validate_observation(observation)
    except release_gate_contract.GateContractError as exc:
        raise RuntimeError("fresh hosted whole-task release-gate observation drifted") from exc
    observed_at = datetime.strptime(observation["observed_at_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=UTC
    )
    age_seconds = ((now or datetime.now(UTC)) - observed_at).total_seconds()
    if not -RELEASE_GATE_MAX_FUTURE_SKEW_SECONDS <= age_seconds <= RELEASE_GATE_MAX_AGE_SECONDS:
        raise RuntimeError("fresh hosted whole-task release-gate observation is stale")
    if any(
        (
            observation.get("binding_sha256") != RELEASE_GATE_BINDING["binding_sha256"],
            observation.get("predecessor_object_set_sha256")
            != RELEASE_GATE_BINDING["predecessor_object_set_sha256"],
            observation.get("fresh_object_set_sha256")
            != RELEASE_GATE_BINDING["fresh_object_set_sha256"],
            observation.get("plan_set_sha256") != RELEASE_GATE_BINDING["plan_set_sha256"],
        )
    ):
        raise RuntimeError("fresh hosted whole-task release-gate observation drifted")
    if any(
        (
            release.get("schema_version") != RELEASE_SCHEMA,
            release.get("status") != "CLEAR",
            release.get("launch_authorized") is not True,
            release.get("scoring_authorized") is not True,
            release.get("controllers") != release_projection(plans, package_sources),
            release.get("held_receipt_sha256")
            != expected_held(plans, package_sources)["receipt_sha256"],
            release.get("observer_job_succeeded") is not True,
            release.get("observer_job_uid") != observation.get("runtime", {}).get("job_uid"),
            release.get("observer_pod_uid") != observation.get("runtime", {}).get("pod_uid"),
            release.get("observer_pod_restarts") != 0,
            release.get("privacy")
            != {
                "scores_read": False,
                "prompts_traces_flags_read": False,
                "credentials_included": False,
            },
            release.get("receipt_sha256") != self_hosted.digest_without(release, "receipt_sha256"),
        )
    ):
        raise RuntimeError("fresh hosted whole-task release evidence drifted")
    validate_runtime_gate_v3_canary(
        release.get("runtime_gate_v3_canary"),
        plans,
        package_sources,
        held_receipt_sha256=release["held_receipt_sha256"],
    )


def validate_runtime_gate_v3_canary(
    value: Any,
    plans: dict[str, dict[str, Any]],
    package_sources: dict[str, dict[str, Any]],
    *,
    held_receipt_sha256: str,
) -> None:
    if not isinstance(value, dict):
        raise RuntimeError("fresh hosted whole-task v3 runtime canary evidence is absent")
    runtime = value.get("sanitized_runtime_receipt")
    if not isinstance(runtime, dict):
        raise RuntimeError("fresh hosted whole-task v3 runtime canary evidence drifted")
    validate_runtime_gate_v3_receipt(
        runtime,
        plans,
        package_sources,
        held_receipt_sha256=held_receipt_sha256,
    )
    if any(
        (
            value.get("schema_version")
            != "fleet-qwen38-hosted-whole-task-v3-runtime-gate-canary-terminal-v1",
            value.get("status") != "PASS",
            value.get("job_name") != RUNTIME_GATE_V3_CANARY_JOB,
            value.get("job_complete") is not True,
            value.get("pod_restarts") != 0,
            engine_uuid(value.get("job_uid")) is None,
            engine_uuid(value.get("pod_uid")) is None,
            runtime.get("job_uid") != value.get("job_uid"),
            runtime.get("pod_uid") != value.get("pod_uid"),
            value.get("source_logs_read") is not False,
            value.get("scores_included") is not False,
            value.get("prompts_or_traces_included") is not False,
            value.get("credentials_included") is not False,
            value.get("receipt_sha256") != self_hosted.digest_without(value, "receipt_sha256"),
        )
    ):
        raise RuntimeError("fresh hosted whole-task v3 runtime canary evidence drifted")


def validate_runtime_gate_v3_receipt(
    runtime: dict[str, Any],
    plans: dict[str, dict[str, Any]],
    package_sources: dict[str, dict[str, Any]],
    *,
    held_receipt_sha256: str,
) -> None:
    if any(
        (
            engine_uuid(runtime.get("job_uid")) is None,
            engine_uuid(runtime.get("pod_uid")) is None,
        )
    ):
        raise RuntimeError("fresh hosted whole-task v3 runtime canary evidence drifted")
    expected = runtime_gate_v3_receipt(
        plans["qwen-a"],
        {"schema_version": HELD_SCHEMA, "receipt_sha256": held_receipt_sha256},
        package_sources["qwen-a"],
        job_uid=runtime["job_uid"],
        pod_uid=runtime["pod_uid"],
    )
    if runtime != expected:
        raise RuntimeError("fresh hosted whole-task v3 runtime canary evidence drifted")


def runtime_gate_v3_receipt(
    plan: dict[str, Any],
    authority: dict[str, Any],
    package_source: dict[str, Any],
    *,
    job_uid: Any,
    pod_uid: Any,
) -> dict[str, Any]:
    zeros = {
        field: 0
        for field in (
            "output_roots_created",
            "endpoint_leases_acquired",
            "canonical_claims_created",
            "model_calls",
            "task_calls",
            "session_calls",
            "verifier_calls",
            "scoring_calls",
            "api_mutations",
        )
    }
    return _seal(
        {
            "schema_version": RUNTIME_GATE_V3_CANARY_SCHEMA,
            "status": "PASS",
            "controller": plan["controller"],
            "plan_sha256": plan["plan_sha256"],
            "job_uid": job_uid,
            "pod_uid": pod_uid,
            "authority_schema_version": authority["schema_version"],
            "authority_receipt_sha256": authority["receipt_sha256"],
            "package_source_receipt_sha256": package_source["receipt_sha256"],
            "bulk_adapter_validated": True,
            "bulk_adapter_members": list(engine.BULK_ADAPTER_MEMBERS),
            "bootstrap_stage_reached": "06-runtime-exec",
            **zeros,
            "scores_included": False,
            "prompts_or_traces_included": False,
            "credentials_included": False,
        }
    )


def engine_uuid(value: Any) -> str | None:
    try:
        parsed = uuid.UUID(str(value))
    except (ValueError, AttributeError):
        return None
    return str(parsed) if parsed.int else None


def load_runtime_release(
    plans: dict[str, dict[str, Any]], package_source: dict[str, Any]
) -> dict[str, Any]:
    raw_path = os.environ.get("QWEN_HOSTED_WHOLE_TASK_RELEASE_PATH")
    expected = os.environ.get("QWEN_HOSTED_WHOLE_TASK_RELEASE_SHA256")
    if not raw_path or not expected or not Path(raw_path).is_absolute():
        raise RuntimeError("fresh hosted whole-task release binding is required")
    release = load(Path(raw_path))
    if release.get("receipt_sha256") != expected:
        raise RuntimeError("fresh hosted whole-task release digest binding drifted")
    controller = str(package_source.get("controller"))
    validate_package_source_receipt(package_source, controller, plans[controller])
    projected = {row["controller"]: row for row in release.get("controllers") or []}
    validate_release(
        release,
        plans,
        {
            name: package_source
            if name == controller
            else {"receipt_sha256": projected.get(name, {}).get("package_source_receipt_sha256")}
            for name in plans
        },
    )
    return release


class AtomicWholeTaskClaims(prior.AtomicWholeTaskClaims):
    """Use the reviewed atomic transaction with the fresh task-version map."""

    @property
    def task_version(self) -> str:
        return CONTROLLERS[self.plan["controller"]]["task_version_id"]

    def _fresh_checks(self, claim_root: Path) -> None:
        """Recheck statistical-cell claims at the last pre-model transaction boundary."""
        super()._fresh_checks(claim_root)
        cell_ids = {row["cell_id"] for row in self.plan["attempts"]}
        collisions = 0
        for path in claim_root.rglob("*.json"):
            if path.is_symlink() or not path.is_file() or path.stat().st_size > 2_000_000:
                raise RuntimeError("fresh hosted whole-task claim scan path is unsafe")
            value = load(path)
            collisions += int(any(row in cell_ids for row in _nested_cell_ids(value)))
        if collisions:
            raise RuntimeError("fresh hosted whole-task statistical cell claim collision")


def _nested_cell_ids(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "cell_id" and isinstance(item, str):
                found.add(item)
            found.update(_nested_cell_ids(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_nested_cell_ids(item))
    return found
