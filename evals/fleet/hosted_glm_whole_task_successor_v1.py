"""Held atomic whole-task hosted GLM successors for untouched ranks 30 and 31."""

from __future__ import annotations

import copy
import fcntl
import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.fleet import hosted_glm_exact_bulk_v1 as source
from evals.fleet import hosted_glm_whole_task_engine_v1 as engine
from evals.fleet import self_hosted

SCHEMA = "fleet-hosted-glm-atomic-whole-task-plan-v1"
HELD_SCHEMA = "fleet-hosted-glm-atomic-whole-task-held-v1"
RELEASE_SCHEMA = "fleet-hosted-glm-atomic-whole-task-release-v1"
RESERVATION_SCHEMA = "fleet-hosted-glm-four-claim-reservation-v1"
PREPARING_SCHEMA = "fleet-hosted-glm-four-claim-preparing-v1"
MODEL_BOUNDARY_SCHEMA = "fleet-hosted-glm-model-boundary-v1"
SOURCE_CONTROLLER = "glm-hosted-s2"
SOURCE_PLAN_SHA256 = "sha256:c42551064b3c384399405ffbb374832b6d1a1d623ed209d6a0b51009421a0320"
CLAIM_ROOT = Path(source.CLAIM_ROOT)
RESERVATION_ROOT = Path("/mnt/sfs/cell-execution-reservations/opencode11827-autocontinue-v1")
JOBS_ROOT = Path("/mnt/sfs/jobs")
LEASE_ROOT = Path(source.LEASE_ROOT)
LEASE_ENDPOINT_KEY = "glm-hosted-autocontinue-v1"
RELEASE_PATH = Path("/bootstrap/release.json")
RELEASE_MAX_AGE_SECONDS = 600
LEDGER_AUTHORITY = {
    "path": (
        "docs/evidence/qwen38-study/"
        "2026-09-05-exact-pass4-ledger-evidence-snapshot-v47.json"
    ),
    "receipt_sha256": (
        "sha256:bf0b9086dd97eecafe20fa9a4cf3b5d643f0ce8f6abad60fae6e3cba3e3e2e29"
    ),
    "file_sha256": (
        "sha256:0a2baba7c16745a4d69f0f5aafc04010734eacbace8f6d712d44011c4a36d0dd"
    ),
}
SELECTION_AUTHORITY = {
    "path": "evals/fleet/configs/opencode-easiest-train100-selection-v2.json",
    "file_sha256": (
        "sha256:9f5e83123cfe7d8a3f4cc0b791ac1dc9032a3d5faf5eba2a431c7246069255df"
    ),
    "semantic_sha256": (
        "sha256:38bd544c74f4e45cb67b271849d657f49e356ccf41f97acca9fb6cd7eb7f56b8"
    ),
}
CONTROLLERS = {
    "glm-hosted-r30-whole-task": {
        "rank": 30,
        "task_version_id": "a0cacaaf-480b-4a4c-9ed7-6b6192bb6783",
        "job_name": "chris-glm53-exact100-hosted-r030-whole-task-g1-v1",
        "configmap_name": "chris-glm53-exact100-hosted-r030-whole-task-g1-package-v1",
    },
    "glm-hosted-r31-whole-task": {
        "rank": 31,
        "task_version_id": "51b23680-27fa-4bdd-a077-16669f3f0e18",
        "job_name": "chris-glm53-exact100-hosted-r031-whole-task-g1-v1",
        "configmap_name": "chris-glm53-exact100-hosted-r031-whole-task-g1-package-v1",
    },
}
SHA256_RE = source.SHA256_RE
COMMIT_RE = source.COMMIT_RE
CANARY_GATE_SCHEMA = source.predecessor.CANARY_GATE_SCHEMA
RECONCILIATION_GATE_SCHEMA = source.predecessor.RECONCILIATION_GATE_SCHEMA
validate_inventory_gate = source.validate_inventory_gate
load = source.load


def _seal(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "receipt_sha256": self_hosted.digest_without(body, "receipt_sha256")}


def _transform(plan: dict[str, Any], controller: str, *, runtime: bool) -> dict[str, Any]:
    authority = CONTROLLERS[controller]
    rank = authority["rank"]
    if plan.get("plan_sha256") != SOURCE_PLAN_SHA256:
        raise ValueError("hosted GLM whole-task source plan drifted")
    attempts = [copy.deepcopy(row) for row in plan["attempts"] if row["selection_rank"] == rank]
    if [row["attempt"] for row in attempts] != [1, 2, 3, 4]:
        raise ValueError("hosted GLM whole-task source attempts drifted")
    if any(row["task_version_id"] != authority["task_version_id"] for row in attempts):
        raise ValueError("hosted GLM whole-task version drifted")
    body = copy.deepcopy(plan)
    body.pop("plan_sha256")
    body.update(
        schema_version=SCHEMA,
        controller=controller,
        campaign_id=authority["job_name"],
        source_job_id=authority["job_name"],
        job_name=authority["job_name"],
        configmap_name=authority["configmap_name"],
        sfs_root=str(JOBS_ROOT / authority["job_name"]),
        task_count=1,
        new_session_count=4,
        attempts=attempts,
        serving_block=LEASE_ENDPOINT_KEY,
        launch_authorized=runtime,
        release_required=True,
        predecessor_plan_sha256=SOURCE_PLAN_SHA256,
    )
    if "tasks" in body:
        body["tasks"] = [copy.deepcopy(row) for row in body["tasks"] if row["rank"] == rank]
        if len(body["tasks"]) != 1:
            raise ValueError("hosted GLM whole-task runtime task drifted")
    body["execution"].update(
        workers=1,
        attempts_per_task_sequential=True,
        same_task_max_inflight=1,
        global_execution_claim_before_model_call=True,
        claim_root=str(CLAIM_ROOT),
        endpoint_lease={
            "lease_root": str(LEASE_ROOT),
            "endpoint_key": LEASE_ENDPOINT_KEY,
            "maximum_streams": 2,
        },
        priority_class="fleet-serve-low",
        preemption_policy="Never",
    )
    body["partition"] = {
        "whole_task_rank": rank,
        "attempts": [1, 2, 3, 4],
        "all_cells_previously_unstarted_required": True,
        "rank29_cells_excluded_and_preserved": True,
        "other_ranks_excluded": True,
    }
    body["atomic_whole_task_reservation"] = {
        "required": True,
        "claim_count": 4,
        "all_claims_validated_before_model_call": True,
        "attempt_order": [1, 2, 3, 4],
        "durable_preparing_before_claims": True,
        "restart_recovers_partial_publication": True,
        "model_boundary_is_durable_and_nonrepeatable": True,
    }
    body["plan_sha256"] = self_hosted.digest_without(body, "plan_sha256")
    return body


def build_plan(controller: str, root: Path) -> dict[str, Any]:
    if controller not in CONTROLLERS:
        raise ValueError("unknown hosted GLM whole-task controller")
    return _transform(source.validate_all(root)[SOURCE_CONTROLLER], controller, runtime=False)


def build_runtime_plan(
    controller: str, inventory_receipt: dict[str, Any], root: Path
) -> dict[str, Any]:
    if controller not in CONTROLLERS:
        raise ValueError("unknown hosted GLM whole-task controller")
    return _transform(
        source.build_runtime_plan(SOURCE_CONTROLLER, inventory_receipt, root),
        controller,
        runtime=True,
    )


def validate_all(root: Path) -> dict[str, dict[str, Any]]:
    plans = {controller: build_plan(controller, root) for controller in CONTROLLERS}
    seen: set[tuple[int, int]] = set()
    for controller, plan in plans.items():
        authority = CONTROLLERS[controller]
        reservation = plan.get("atomic_whole_task_reservation") or {}
        if any(
            (
                plan.get("launch_authorized") is not False,
                plan.get("release_required") is not True,
                plan.get("model", {}).get("served_id") != "glm-5.3",
                plan.get("model", {}).get("revision")
                != "30333038ada1f1dacb294a93270305a890b50c14",
                plan.get("harness", {}).get("version") != "1.18.27",
                plan.get("harness", {}).get("context_management")
                != "opencode_1.18.27_native_compaction_autocontinue_v1",
                plan.get("execution", {}).get("required_task_tools")
                != ["bash", "submit_report"],
                plan.get("execution", {}).get("endpoint_lease", {}).get("maximum_streams")
                != 2,
                plan.get("execution", {}).get("workers") != 1,
                [row.get("attempt") for row in plan["attempts"]] != [1, 2, 3, 4],
                any(row.get("selection_rank") != authority["rank"] for row in plan["attempts"]),
                any(
                    row.get("task_version_id") != authority["task_version_id"]
                    for row in plan["attempts"]
                ),
                reservation.get("claim_count") != 4,
                reservation.get("all_claims_validated_before_model_call") is not True,
                plan.get("plan_sha256")
                != self_hosted.digest_without(plan, "plan_sha256"),
            )
        ):
            raise ValueError("hosted GLM whole-task plan drifted")
        for row in plan["attempts"]:
            identity = (row["selection_rank"], row["attempt"])
            if identity in seen:
                raise ValueError("hosted GLM whole-task overlap")
            seen.add(identity)
    expected = {(rank, attempt) for rank in (30, 31) for attempt in range(1, 5)}
    if seen != expected:
        raise ValueError("hosted GLM whole-task partition drifted")
    return plans


def release_projection(plans: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "controller": controller,
            "selection_rank": CONTROLLERS[controller]["rank"],
            "task_version_id": CONTROLLERS[controller]["task_version_id"],
            "job_name": CONTROLLERS[controller]["job_name"],
            "configmap_name": CONTROLLERS[controller]["configmap_name"],
            "sfs_root": plan["sfs_root"],
            "plan_sha256": plan["plan_sha256"],
            "cells": [
                {
                    "attempt": row["attempt"],
                    "cell_id": row["cell_id"],
                    "execution_id": row["execution_id"],
                    "run_id": row["run_id"],
                }
                for row in plan["attempts"]
            ],
        }
        for controller, plan in sorted(plans.items())
    ]


def validate_release(
    release: dict[str, Any], plans: dict[str, dict[str, Any]], source_package_sha256: str
) -> None:
    collision = release.get("fresh_collision_reconciliation") or {}
    if any(
        (
            set(release)
            != {
                "schema_version", "status", "checked_at_utc", "launch_authorized",
                "scoring_authorized", "controller_cap", "controllers",
                "source_package_sha256", "fresh_collision_reconciliation",
                "ledger_authority", "selection_authority", "rank29_disposition",
                "privacy", "receipt_sha256",
            },
            release.get("schema_version") != RELEASE_SCHEMA,
            release.get("status") != "CLEAR",
            release.get("launch_authorized") is not True,
            release.get("scoring_authorized") is not True,
            release.get("controller_cap") != 2,
            release.get("controllers") != release_projection(plans),
            release.get("source_package_sha256") != source_package_sha256,
            SHA256_RE.fullmatch(str(source_package_sha256)) is None,
            release.get("ledger_authority") != LEDGER_AUTHORITY,
            release.get("selection_authority") != SELECTION_AUTHORITY,
            set(collision)
            != {
                "checked_immediately_before_create", "observer_job_uid", "observer_pod_uid",
                "observed_cells", "canonical_claim_collisions",
                "authoritative_session_collisions", "accepted_evidence_collisions",
                "output_root_collisions", "new_job_collisions", "new_configmap_collisions",
                "endpoint_lease_slots_available", "api_mutations",
            },
            collision.get("checked_immediately_before_create") is not True,
            engine.UUID_RE.fullmatch(str(collision.get("observer_job_uid"))) is None,
            engine.UUID_RE.fullmatch(str(collision.get("observer_pod_uid"))) is None,
            collision.get("observed_cells") != 8,
            any(
                collision.get(key) != 0
                for key in (
                    "canonical_claim_collisions", "authoritative_session_collisions",
                    "accepted_evidence_collisions", "output_root_collisions",
                    "new_job_collisions", "new_configmap_collisions", "api_mutations",
                )
            ),
            collision.get("endpoint_lease_slots_available") != 2,
            release.get("rank29_disposition")
            != {"preserved": True, "selected": False, "retry_or_supersession_authorized": False},
            release.get("privacy")
            != {
                "scores_read": False,
                "prompts_traces_flags_read": False,
                "credentials_included": False,
            },
            release.get("receipt_sha256")
            != self_hosted.digest_without(release, "receipt_sha256"),
        )
    ):
        raise RuntimeError("hosted GLM whole-task release drifted")
    try:
        checked_at = datetime.fromisoformat(
            str(release["checked_at_utc"]).replace("Z", "+00:00")
        )
        age = (datetime.now(UTC) - checked_at).total_seconds()
    except (KeyError, TypeError, ValueError):
        raise RuntimeError("hosted GLM whole-task release timestamp drifted") from None
    if age < -60 or age > RELEASE_MAX_AGE_SECONDS:
        raise RuntimeError("hosted GLM whole-task release is stale")


def load_runtime_release(
    plans: dict[str, dict[str, Any]], source_package_sha256: str
) -> dict[str, Any]:
    if SHA256_RE.fullmatch(source_package_sha256) is None:
        raise RuntimeError("hosted GLM whole-task source package binding is invalid")
    release = load(RELEASE_PATH)
    validate_release(release, plans, source_package_sha256)
    return release


def _accepted_collision_count(plan: dict[str, Any], *, jobs_root: Path = JOBS_ROOT) -> int:
    identities = {
        value
        for row in plan["attempts"]
        for value in (row["cell_id"], row["execution_id"], row["run_id"])
    }
    paths = set(jobs_root.rglob("ACCEPTED.json"))
    paths.update(jobs_root.rglob("accepted/*.json"))
    matches = 0
    for path in paths:
        try:
            if path.is_symlink() or not path.is_file() or path.stat().st_size > 2_000_000:
                continue
            value = load(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        matches += int(
            any(value.get(field) in identities for field in ("cell_id", "execution_id", "run_id"))
        )
    return matches


def _output_collision_count(plan: dict[str, Any], *, jobs_root: Path = JOBS_ROOT) -> int:
    run_ids = {row["run_id"] for row in plan["attempts"]}
    return sum(1 for path in jobs_root.rglob("*") if path.name in run_ids)


class AtomicWholeTaskClaims:
    """Crash-recoverable four-claim transaction at the engine claim boundary."""

    def __init__(
        self,
        plan: dict[str, Any],
        out: Path,
        *,
        key: str,
        jobs_root: Path = JOBS_ROOT,
        reservation_root: Path = RESERVATION_ROOT,
        session_check: Callable[[dict[str, Any], str], None] = engine._assert_run_absent,
        fault_hook: Callable[[str], None] | None = None,
    ) -> None:
        self.plan = plan
        self.out = out
        self.key = key
        self.jobs_root = jobs_root
        self.reservation_root = reservation_root
        self.session_check = session_check
        self.fault_hook = fault_hook or (lambda _stage: None)
        self.claims: dict[str, dict[str, Any]] = {}

    @property
    def task_version(self) -> str:
        return CONTROLLERS[self.plan["controller"]]["task_version_id"]

    @property
    def preparing_root(self) -> Path:
        return self.reservation_root / "preparing" / self.task_version

    def _preparing_receipt(self, job_uid: str, pod_uid: str) -> dict[str, Any]:
        return _seal(
            {
                "schema_version": PREPARING_SCHEMA,
                "status": "PREPARING",
                "controller": self.plan["controller"],
                "selection_rank": self.plan["attempts"][0]["selection_rank"],
                "task_version_id": self.task_version,
                "plan_sha256": self.plan["plan_sha256"],
                "job_uid": job_uid,
                "pod_uid": pod_uid,
                "claim_filenames": [
                    engine.claim_filename(row["execution_id"]) for row in self.plan["attempts"]
                ],
                "reservation_path": str(self.out / "RESERVATION.json"),
                "model_call_started": False,
                "scores_included": False,
                "prompts_or_traces_included": False,
            }
        )

    def _preparing_path(self, job_uid: str, pod_uid: str) -> Path:
        return self.preparing_root / f"{job_uid}-{pod_uid}.json"

    def _validate_preparing(self, value: dict[str, Any]) -> None:
        expected_names = [
            engine.claim_filename(row["execution_id"]) for row in self.plan["attempts"]
        ]
        if any(
            (
                set(value)
                != {
                    "schema_version",
                    "status",
                    "controller",
                    "selection_rank",
                    "task_version_id",
                    "plan_sha256",
                    "job_uid",
                    "pod_uid",
                    "claim_filenames",
                    "reservation_path",
                    "model_call_started",
                    "scores_included",
                    "prompts_or_traces_included",
                    "receipt_sha256",
                },
                value.get("schema_version") != PREPARING_SCHEMA,
                value.get("status") != "PREPARING",
                value.get("controller") != self.plan["controller"],
                value.get("selection_rank") != self.plan["attempts"][0]["selection_rank"],
                value.get("task_version_id") != self.task_version,
                value.get("plan_sha256") != self.plan["plan_sha256"],
                engine.UUID_RE.fullmatch(str(value.get("job_uid"))) is None,
                engine.UUID_RE.fullmatch(str(value.get("pod_uid"))) is None,
                value.get("claim_filenames") != expected_names,
                value.get("reservation_path") != str(self.out / "RESERVATION.json"),
                value.get("model_call_started") is not False,
                value.get("scores_included") is not False,
                value.get("prompts_or_traces_included") is not False,
                value.get("receipt_sha256") != self_hosted.digest_without(value, "receipt_sha256"),
            )
        ):
            raise RuntimeError("hosted GLM whole-task PREPARING transaction drifted")

    def _load_preparings(self) -> list[dict[str, Any]]:
        if not self.preparing_root.exists():
            return []
        if self.preparing_root.is_symlink() or not self.preparing_root.is_dir():
            raise RuntimeError("hosted GLM whole-task PREPARING root is unsafe")
        values: list[dict[str, Any]] = []
        for path in self.preparing_root.iterdir():
            if path.is_symlink() or not path.is_file() or path.suffix != ".json":
                raise RuntimeError("hosted GLM whole-task PREPARING entry is unsafe")
            value = load(path)
            self._validate_preparing(value)
            values.append(value)
        return values

    def _ensure_preparing(self, job_uid: str, pod_uid: str) -> dict[str, Any]:
        value = self._preparing_receipt(job_uid, pod_uid)
        path = self._preparing_path(job_uid, pod_uid)
        self.preparing_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            engine._write_once(path, value)
        except FileExistsError:
            existing = load(path)
            if existing != value:
                raise RuntimeError(
                    "hosted GLM whole-task existing PREPARING bytes drifted"
                ) from None
        self._validate_preparing(value)
        return value

    def _fresh_checks(self, claim_root: Path) -> None:
        for label, root in (("jobs", self.jobs_root), ("claim", claim_root)):
            if not root.exists() or root.is_symlink() or not root.is_dir():
                raise RuntimeError(f"hosted GLM whole-task canonical {label} root is unavailable")
        if _accepted_collision_count(self.plan, jobs_root=self.jobs_root):
            raise RuntimeError("hosted GLM whole-task accepted evidence collision")
        if _output_collision_count(self.plan, jobs_root=self.jobs_root):
            raise RuntimeError("hosted GLM whole-task output collision")
        for item in self.plan["attempts"]:
            path = claim_root / engine.claim_filename(item["execution_id"])
            if path.exists() or path.is_symlink():
                raise RuntimeError("hosted GLM whole-task canonical claim collision")
            config = engine._attempt_config(self.plan, engine._task_for_item(self.plan, item), item)
            self.session_check(config, self.key)

    def _reservation_receipt(
        self,
        claims: dict[str, dict[str, Any]],
        preparing: dict[str, Any],
        job_uid: str,
        pod_uid: str,
    ) -> dict[str, Any]:
        return _seal(
            {
                "schema_version": RESERVATION_SCHEMA,
                "status": "RESERVED",
                "controller": self.plan["controller"],
                "selection_rank": self.plan["attempts"][0]["selection_rank"],
                "task_version_id": self.task_version,
                "plan_sha256": self.plan["plan_sha256"],
                "job_uid": job_uid,
                "pod_uid": pod_uid,
                "preparing_receipt_sha256": preparing["receipt_sha256"],
                "claim_sha256s": [
                    claims[row["run_id"]]["receipt_sha256"] for row in self.plan["attempts"]
                ],
                "all_claims_before_model_call": True,
                "all_claims_byte_validated": True,
                "attempt_order": [1, 2, 3, 4],
                "scores_included": False,
                "prompts_or_traces_included": False,
            }
        )

    def _validate_reservation(
        self, value: dict[str, Any], claim_root: Path, preparings: list[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        owner = (value.get("job_uid"), value.get("pod_uid"))
        preparing_by_owner = {(row["job_uid"], row["pod_uid"]): row for row in preparings}
        preparing = preparing_by_owner.get(owner)
        claims = {
            item["run_id"]: engine._validate_preserved_claim(self.plan, item, claim_root)
            for item in self.plan["attempts"]
        }
        expected = (
            self._reservation_receipt(claims, preparing, *owner) if preparing is not None else None
        )
        if expected is None or value != expected:
            raise RuntimeError("hosted GLM whole-task reservation bytes drifted")
        if any((claim["job_uid"], claim["pod_uid"]) != owner for claim in claims.values()):
            raise RuntimeError("hosted GLM whole-task reservation claim owner drifted")
        return claims

    def _recover(self, claim_root: Path) -> bool:
        """Commit a complete transaction or roll back only proven pre-model bytes."""
        preparings = self._load_preparings()
        owners = {(row["job_uid"], row["pod_uid"]) for row in preparings}
        boundary_root = self.out / "model-boundaries"
        if boundary_root.exists() and (boundary_root.is_symlink() or not boundary_root.is_dir()):
            raise RuntimeError("hosted GLM whole-task model boundary root is unsafe")
        model_boundary_exists = boundary_root.exists() and any(boundary_root.iterdir())
        reservation_path = self.out / "RESERVATION.json"
        if reservation_path.exists() or reservation_path.is_symlink():
            if reservation_path.is_symlink() or not reservation_path.is_file():
                raise RuntimeError("hosted GLM whole-task reservation path is unsafe")
            try:
                value = load(reservation_path)
                self.claims = self._validate_reservation(value, claim_root, preparings)
                return True
            except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError):
                # The provider cannot return until the complete reservation is
                # fsynced.  A malformed file therefore proves a pre-model torn
                # write only while no model boundary exists.  Once any attempt
                # crossed that boundary, all drift fails closed.
                if model_boundary_exists:
                    raise RuntimeError(
                        "hosted GLM whole-task committed reservation drifted after model boundary"
                    ) from None
                reservation_path.unlink()
        if model_boundary_exists:
            raise RuntimeError("hosted GLM whole-task claims drifted after model boundary")
        for item in reversed(self.plan["attempts"]):
            path = claim_root / engine.claim_filename(item["execution_id"])
            if not path.exists() and not path.is_symlink():
                continue
            if path.is_symlink() or not path.is_file():
                raise RuntimeError("hosted GLM whole-task partial claim path is unsafe")
            claim = engine._validate_preserved_claim(self.plan, item, claim_root)
            if (claim["job_uid"], claim["pod_uid"]) not in owners:
                raise RuntimeError("hosted GLM whole-task claim lacks PREPARING ownership")
            if claim.get("model_call_started_when_claim_written") is not False:
                raise RuntimeError("hosted GLM whole-task recovery crossed model boundary")
            path.unlink()
        self.claims = {}
        return False

    def _reserve(self, claim_root: Path, job_uid: str, pod_uid: str) -> None:
        self.reservation_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        lock_path = self.reservation_root / f"{self.task_version}.lock"
        lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            if self._recover(claim_root):
                return
            self._fresh_checks(claim_root)
            preparing = self._ensure_preparing(job_uid, pod_uid)
            self.fault_hook("after_preparing")
            created: dict[str, dict[str, Any]] = {}
            try:
                for count, item in enumerate(reversed(self.plan["attempts"]), start=1):
                    claim = engine.claim_cell(
                        self.plan,
                        item,
                        claim_root=claim_root,
                        job_uid=job_uid,
                        pod_uid=pod_uid,
                    )
                    if claim is None:
                        raise RuntimeError("hosted GLM whole-task claim transaction collided")
                    created[item["run_id"]] = claim
                    self.fault_hook(f"after_claim_{count}")
                for count, item in enumerate(reversed(self.plan["attempts"]), start=1):
                    claim = created[item["run_id"]]
                    validated = engine._validate_preserved_claim(self.plan, item, claim_root)
                    if validated != claim:
                        raise RuntimeError("hosted GLM whole-task claim validation drifted")
                    self.fault_hook(f"after_validation_{count}")
                self.claims = created
                self.fault_hook("before_reservation_write")
                reservation = self._reservation_receipt(self.claims, preparing, job_uid, pod_uid)
                engine._write_once(self.out / "RESERVATION.json", reservation)
                self.fault_hook("after_reservation_write")
            except BaseException:
                self._recover(claim_root)
                raise
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)

    def mark_model_boundary(self, item: dict[str, Any]) -> dict[str, Any]:
        """Persist the conservative nonrepeatable boundary immediately before model entry."""
        claim = self.claims.get(item["run_id"])
        if claim is None:
            raise RuntimeError("hosted GLM whole-task model boundary lacks reserved claim")
        path = self.out / "model-boundaries" / f"{item['run_id']}.json"
        marker = _seal(
            {
                "schema_version": MODEL_BOUNDARY_SCHEMA,
                "status": "MODEL_BOUNDARY_ENTERED",
                "controller": self.plan["controller"],
                "plan_sha256": self.plan["plan_sha256"],
                "cell_id": item["cell_id"],
                "execution_id": item["execution_id"],
                "run_id": item["run_id"],
                "claim_sha256": claim["receipt_sha256"],
                "retry_allowed": False,
                "scores_included": False,
                "prompts_or_traces_included": False,
            }
        )
        if path.exists() or path.is_symlink():
            if path.is_symlink() or load(path) != marker:
                raise RuntimeError("hosted GLM whole-task model boundary marker drifted")
            raise RuntimeError(
                "hosted GLM whole-task model boundary already crossed; retry prohibited"
            )
        engine._write_once(path, marker)
        return marker

    def __call__(
        self,
        plan: dict[str, Any],
        item: dict[str, Any],
        claim_root: Path,
        job_uid: str,
        pod_uid: str,
    ) -> dict[str, Any]:
        if plan != self.plan:
            raise RuntimeError("hosted GLM whole-task claim provider plan drifted")
        if not self.claims:
            self._reserve(claim_root, job_uid, pod_uid)
        try:
            return self.claims[item["run_id"]]
        except KeyError as exc:
            raise RuntimeError("hosted GLM whole-task item is outside the reservation") from exc
