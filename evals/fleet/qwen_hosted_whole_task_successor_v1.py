"""Held atomic whole-task successors for hosted Qwen ranks 15 and 16.

Each controller owns exactly one previously untouched statistical task.  A
controller publishes and byte-validates all four canonical execution claims
under one task reservation lock before the first model call.  The four attempts
then execute sequentially while one of the two qualified hosted endpoint leases
is held.
"""

from __future__ import annotations

import copy
import fcntl
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_bulk_runtime_v3 as engine
from evals.fleet import qwen_hosted_generation19_bulk as original
from evals.fleet import qwen_hosted_generation19_v4 as source
from evals.fleet import self_hosted

SCHEMA = "fleet-qwen38-hosted-atomic-whole-task-plan-v1"
HELD_SCHEMA = "fleet-qwen38-hosted-atomic-whole-task-held-v1"
RELEASE_SCHEMA = "fleet-qwen38-hosted-atomic-whole-task-release-v1"
RESERVATION_SCHEMA = "fleet-qwen38-hosted-four-claim-reservation-v1"
LEDGER_PATH = "docs/evidence/qwen38-study/2026-09-05-exact-pass4-ledger-evidence-snapshot-v47.json"
LEDGER_SELF_SHA256 = "sha256:bf0b9086dd97eecafe20fa9a4cf3b5d643f0ce8f6abad60fae6e3cba3e3e2e29"
LEDGER_FILE_SHA256 = "sha256:0a2baba7c16745a4d69f0f5aafc04010734eacbace8f6d712d44011c4a36d0dd"
HELD_PATH = (
    "docs/evidence/qwen38-study/2026-09-06-qwen38-hosted-rank15-rank16-whole-task-held-v1.json"
)
CLAIM_ROOT = Path("/mnt/sfs/cell-execution-claims/opencode11827-autocontinue-v1")
RESERVATION_ROOT = Path("/mnt/sfs/cell-execution-reservations/opencode11827-autocontinue-v1")
JOBS_ROOT = Path("/mnt/sfs/jobs")
PREDECESSOR_TOMBSTONES = [
    {
        "selection_rank": 13,
        "path": (
            "docs/evidence/qwen38-study/"
            "2026-09-06-qwen38-hosted-a-rank13-a1-deadline-terminal-reconciliation-v1.json"
        ),
        "receipt_sha256": (
            "sha256:f8b374c05ec106555a5fa2ae522231413b685d64e53c47526efb58b2e98cfbe7"
        ),
        "file_sha256": ("sha256:9ce8e6f797488adf0b0b101c1795646aee3ad4c424b11f677e3161b460ca4a66"),
    },
    {
        "selection_rank": 14,
        "path": (
            "docs/evidence/qwen38-study/"
            "2026-09-06-qwen38-hosted-b-rank14-a1-deadline-terminal-reconciliation-v1.json"
        ),
        "receipt_sha256": (
            "sha256:3e388bb7611c3a197539128e72dcf155046fd7589fed7c443de23246de5f8a15"
        ),
        "file_sha256": ("sha256:44e7ac1a83deb518f0f28acc0155d5ec46312de5491a8c80ae81c7fc0ae80c4c"),
    },
]
CONTROLLERS = {
    "qwen-a": {
        "rank": 15,
        "task_version_id": "fb8f2178-7dd8-429a-8d3c-f14b21a51e02",
        "job_name": "chris-q38-hosted-r015-whole-task-g19-v1",
        "configmap_name": "chris-q38-hosted-r015-whole-task-g19-package-v1",
        "source_plan_sha256": (
            "sha256:5fcc7e90a4ffbb093c09ca3cbb9cf907ab755ad873e68853629b283dad57542e"
        ),
    },
    "qwen-b": {
        "rank": 16,
        "task_version_id": "fd07b96f-7aa4-4041-862b-5aa7411eff4b",
        "job_name": "chris-q38-hosted-r016-whole-task-g19-v1",
        "configmap_name": "chris-q38-hosted-r016-whole-task-g19-package-v1",
        "source_plan_sha256": (
            "sha256:c4dc5c95bb5ccc9e5d4e0ae14745d5533bb4c1d6eb64fa50d0cfe171007f93c5"
        ),
    },
}
CANARY_GATE_SCHEMA = source.CANARY_GATE_SCHEMA
COMMIT_RE = source.COMMIT_RE
RECONCILIATION_GATE_SCHEMA = source.RECONCILIATION_GATE_SCHEMA
SHA256_RE = source.SHA256_RE
load = original.g17.load
validate_inventory_gate = source.validate_inventory_gate


def _seal(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "receipt_sha256": self_hosted.digest_without(body, "receipt_sha256")}


def build_plans(root: Path) -> dict[str, dict[str, Any]]:
    sources = source.validate_all(root)
    plans: dict[str, dict[str, Any]] = {}
    for controller, authority in CONTROLLERS.items():
        predecessor = sources[controller]
        if predecessor["plan_sha256"] != authority["source_plan_sha256"]:
            raise ValueError("hosted whole-task source plan drifted")
        rank = authority["rank"]
        tasks = [copy.deepcopy(row) for row in predecessor["tasks"] if row["rank"] == rank]
        attempts = [
            copy.deepcopy(row) for row in predecessor["attempts"] if row["selection_rank"] == rank
        ]
        body = copy.deepcopy(predecessor)
        body.pop("plan_sha256")
        body.update(
            schema_version=SCHEMA,
            controller=controller,
            campaign_id=authority["job_name"],
            source_job_id=authority["job_name"],
            sfs_root=f"/mnt/sfs/jobs/{authority['job_name']}",
            tasks=tasks,
            attempts=attempts,
            launch_authorized=False,
            release_required=True,
            predecessor_plan_sha256=predecessor["plan_sha256"],
            atomic_whole_task_reservation={
                "required": True,
                "claim_count": 4,
                "all_claims_validated_before_model_call": True,
                "attempt_order": [1, 2, 3, 4],
                "partial_publication_rollback_before_model_only": True,
            },
        )
        plans[controller] = {
            **body,
            "plan_sha256": self_hosted.digest_without(body, "plan_sha256"),
        }
    validate(plans)
    return plans


def validate(plans: dict[str, dict[str, Any]]) -> None:
    if set(plans) != set(CONTROLLERS):
        raise ValueError("hosted whole-task controller set drifted")
    seen: set[tuple[int, int]] = set()
    for controller, plan in plans.items():
        authority = CONTROLLERS[controller]
        attempts = plan.get("attempts") or []
        tasks = plan.get("tasks") or []
        reservation = plan.get("atomic_whole_task_reservation") or {}
        if any(
            (
                plan.get("schema_version") != SCHEMA,
                plan.get("controller") != controller,
                plan.get("campaign_id") != authority["job_name"],
                plan.get("source_job_id") != authority["job_name"],
                plan.get("sfs_root") != f"/mnt/sfs/jobs/{authority['job_name']}",
                plan.get("predecessor_plan_sha256") != authority["source_plan_sha256"],
                plan.get("launch_authorized") is not False,
                plan.get("release_required") is not True,
                plan.get("serving_block") != "qwen-hosted-autocontinue-v1",
                plan.get("model", {}).get("revision") != "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
                plan.get("harness", {}).get("version") != "1.18.27",
                plan.get("execution", {}).get("endpoint_lease", {}).get("maximum_streams") != 2,
                plan.get("execution", {}).get("workers") != 1,
                len(tasks) != 1,
                tasks[0].get("rank") != authority["rank"],
                len(attempts) != 4,
                [row.get("attempt") for row in attempts] != [1, 2, 3, 4],
                any(row.get("selection_rank") != authority["rank"] for row in attempts),
                any(row.get("task_version_id") != authority["task_version_id"] for row in attempts),
                reservation
                != {
                    "required": True,
                    "claim_count": 4,
                    "all_claims_validated_before_model_call": True,
                    "attempt_order": [1, 2, 3, 4],
                    "partial_publication_rollback_before_model_only": True,
                },
                plan.get("plan_sha256") != self_hosted.digest_without(plan, "plan_sha256"),
            )
        ):
            raise ValueError("hosted whole-task plan drifted")
        for row in attempts:
            identity = (row["selection_rank"], row["attempt"])
            if identity in seen:
                raise ValueError("hosted whole-task plans overlap")
            seen.add(identity)
    if seen != {(15, attempt) for attempt in range(1, 5)} | {
        (16, attempt) for attempt in range(1, 5)
    }:
        raise ValueError("hosted whole-task plans include a forbidden rank")


def validate_all(root: Path) -> dict[str, dict[str, Any]]:
    return build_plans(root)


def build_runtime_plan(
    controller: str, inventory_receipt: dict[str, Any], root: Path
) -> dict[str, Any]:
    plan = build_plans(root)[controller]
    if plan["inventory_receipt"] != inventory_receipt:
        raise ValueError("hosted whole-task inventory projection drifted")
    return plan


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


def validate_held(held: dict[str, Any], plans: dict[str, dict[str, Any]]) -> None:
    if any(
        (
            set(held)
            != {
                "schema_version",
                "status",
                "launch_authorized",
                "scoring_authorized",
                "mutation_calls",
                "controller_cap",
                "endpoint_maximum_streams",
                "controllers",
                "ledger_snapshot_path",
                "ledger_snapshot_receipt_sha256",
                "ledger_snapshot_file_sha256",
                "retry_forbidden_selection_ranks",
                "predecessor_tombstones",
                "required_fresh_release_gates",
                "privacy",
                "receipt_sha256",
            },
            held.get("schema_version") != HELD_SCHEMA,
            held.get("status") != "HELD_FOR_REVIEW",
            held.get("launch_authorized") is not False,
            held.get("scoring_authorized") is not False,
            held.get("mutation_calls") != 0,
            held.get("controller_cap") != 2,
            held.get("endpoint_maximum_streams") != 2,
            held.get("controllers") != release_projection(plans),
            held.get("ledger_snapshot_path") != LEDGER_PATH,
            held.get("ledger_snapshot_receipt_sha256") != LEDGER_SELF_SHA256,
            held.get("ledger_snapshot_file_sha256") != LEDGER_FILE_SHA256,
            held.get("retry_forbidden_selection_ranks") != [13, 14],
            held.get("predecessor_tombstones") != PREDECESSOR_TOMBSTONES,
            held.get("required_fresh_release_gates")
            != [
                "prior_jobs_terminal_and_pods_absent",
                "endpoint_lease_files_absent",
                "canonical_claim_collisions_zero",
                "authoritative_session_collisions_zero",
                "accepted_evidence_collisions_zero",
                "output_root_collisions_zero",
                "atomic_four_claim_publication_and_validation",
            ],
            held.get("privacy")
            != {
                "scores_read": False,
                "prompts_traces_flags_read": False,
                "credentials_included": False,
            },
            held.get("receipt_sha256") != self_hosted.digest_without(held, "receipt_sha256"),
        )
    ):
        raise RuntimeError("hosted whole-task held evidence drifted")


def validate_release(release: dict[str, Any], plans: dict[str, dict[str, Any]]) -> None:
    privacy = release.get("privacy") or {}
    collision = release.get("fresh_collision_reconciliation") or {}
    predecessor = release.get("predecessor_disposition") or {}
    if any(
        (
            set(release)
            != {
                "schema_version",
                "status",
                "checked_at_utc",
                "launch_authorized",
                "scoring_authorized",
                "controller_cap",
                "endpoint_maximum_streams",
                "controllers",
                "ledger_snapshot_path",
                "ledger_snapshot_receipt_sha256",
                "ledger_snapshot_file_sha256",
                "predecessor_disposition",
                "predecessor_tombstones",
                "fresh_collision_reconciliation",
                "privacy",
                "receipt_sha256",
            },
            release.get("schema_version") != RELEASE_SCHEMA,
            release.get("status") != "CLEAR",
            engine.ISO_UTC_RE.fullmatch(str(release.get("checked_at_utc"))) is None,
            release.get("launch_authorized") is not True,
            release.get("scoring_authorized") is not True,
            release.get("controller_cap") != 2,
            release.get("endpoint_maximum_streams") != 2,
            release.get("controllers") != release_projection(plans),
            release.get("ledger_snapshot_path") != LEDGER_PATH,
            release.get("ledger_snapshot_receipt_sha256") != LEDGER_SELF_SHA256,
            release.get("ledger_snapshot_file_sha256") != LEDGER_FILE_SHA256,
            release.get("predecessor_tombstones") != PREDECESSOR_TOMBSTONES,
            predecessor.get("retry_forbidden_selection_ranks") != [13, 14],
            set(predecessor)
            != {
                "retry_forbidden_selection_ranks",
                "prior_job_uids",
                "prior_pod_uids",
                "prior_jobs_terminal",
                "prior_pods_absent",
                "endpoint_lease_files_absent",
            },
            predecessor.get("prior_job_uids")
            != [
                "515c370a-ecb4-4c41-a349-46a328c8fa68",
                "f23805b5-516b-4828-a3b5-82673a1b3e2f",
            ],
            predecessor.get("prior_pod_uids")
            != [
                "4d86f0c5-7cef-4f8a-a798-ea5c725e4955",
                "d5d7b7cf-d639-4ad6-b012-3001c4935b2d",
            ],
            predecessor.get("prior_jobs_terminal") is not True,
            predecessor.get("prior_pods_absent") is not True,
            predecessor.get("endpoint_lease_files_absent") is not True,
            collision.get("checked_immediately_before_release") is not True,
            set(collision)
            != {
                "checked_immediately_before_release",
                "checked_from_uid_bound_sfs_pod",
                "observer_pod_uid",
                "observed_cells",
                "canonical_claim_collisions",
                "authoritative_session_collisions",
                "accepted_evidence_collisions",
                "output_root_collisions",
                "api_mutations",
            },
            collision.get("checked_from_uid_bound_sfs_pod") is not True,
            engine.UUID_RE.fullmatch(str(collision.get("observer_pod_uid"))) is None,
            collision.get("observed_cells") != 8,
            collision.get("canonical_claim_collisions") != 0,
            collision.get("authoritative_session_collisions") != 0,
            collision.get("accepted_evidence_collisions") != 0,
            collision.get("output_root_collisions") != 0,
            collision.get("api_mutations") != 0,
            privacy
            != {
                "scores_read": False,
                "prompts_traces_flags_read": False,
                "credentials_included": False,
            },
            release.get("receipt_sha256") != self_hosted.digest_without(release, "receipt_sha256"),
        )
    ):
        raise RuntimeError("hosted whole-task release drifted")


def load_runtime_release(plans: dict[str, dict[str, Any]]) -> dict[str, Any]:
    raw_path = os.environ.get("QWEN_HOSTED_WHOLE_TASK_RELEASE_PATH")
    expected = os.environ.get("QWEN_HOSTED_WHOLE_TASK_RELEASE_SHA256")
    if not raw_path or not expected:
        raise RuntimeError("hosted whole-task release is required")
    path = Path(raw_path)
    if not path.is_absolute():
        raise RuntimeError("hosted whole-task release path must be absolute")
    release = load(path)
    if release.get("receipt_sha256") != expected:
        raise RuntimeError("hosted whole-task release digest binding drifted")
    validate_release(release, plans)
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
    """Lazily reserve one plan's four claims at the engine's claim boundary."""

    def __init__(
        self,
        plan: dict[str, Any],
        out: Path,
        *,
        key: str,
        jobs_root: Path = JOBS_ROOT,
        reservation_root: Path = RESERVATION_ROOT,
        session_check: Callable[[dict[str, Any], str], None] = engine._assert_run_absent,
    ) -> None:
        self.plan = plan
        self.out = out
        self.key = key
        self.jobs_root = jobs_root
        self.reservation_root = reservation_root
        self.session_check = session_check
        self.claims: dict[str, dict[str, Any]] = {}

    def _fresh_checks(self, claim_root: Path) -> None:
        for label, root in (("jobs", self.jobs_root), ("claim", claim_root)):
            if not root.exists() or root.is_symlink() or not root.is_dir():
                raise RuntimeError(f"hosted whole-task canonical {label} root is unavailable")
        if _accepted_collision_count(self.plan, jobs_root=self.jobs_root):
            raise RuntimeError("hosted whole-task accepted evidence collision")
        if _output_collision_count(self.plan, jobs_root=self.jobs_root):
            raise RuntimeError("hosted whole-task output collision")
        for item in self.plan["attempts"]:
            path = claim_root / engine.claim_filename(item["execution_id"])
            if path.exists() or path.is_symlink():
                raise RuntimeError("hosted whole-task canonical claim collision")
            config = engine._attempt_config(self.plan, engine._task_for_item(self.plan, item), item)
            self.session_check(config, self.key)

    def _rollback(
        self,
        created: list[tuple[dict[str, Any], dict[str, Any]]],
        claim_root: Path,
    ) -> None:
        if self.out.joinpath("RESERVATION.json").exists():
            raise RuntimeError("hosted whole-task committed reservation cannot roll back")
        for item, claim in reversed(created):
            path = claim_root / engine.claim_filename(item["execution_id"])
            if path.is_symlink() or load(path) != claim:
                raise RuntimeError("hosted whole-task partial claim bytes drifted")
            if claim.get("model_call_started_when_claim_written") is not False:
                raise RuntimeError("hosted whole-task rollback crossed model boundary")
            path.unlink()

    def _reserve(self, claim_root: Path, job_uid: str, pod_uid: str) -> None:
        self.reservation_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        task_version = CONTROLLERS[self.plan["controller"]]["task_version_id"]
        lock_path = self.reservation_root / f"{task_version}.lock"
        lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        created: list[tuple[dict[str, Any], dict[str, Any]]] = []
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            self._fresh_checks(claim_root)
            try:
                for item in reversed(self.plan["attempts"]):
                    claim = engine.claim_cell(
                        self.plan,
                        item,
                        claim_root=claim_root,
                        job_uid=job_uid,
                        pod_uid=pod_uid,
                    )
                    if claim is None:
                        raise RuntimeError("hosted whole-task claim transaction collided")
                    created.append((item, claim))
                for item, claim in created:
                    validated = engine._validate_preserved_claim(self.plan, item, claim_root)
                    if validated != claim:
                        raise RuntimeError("hosted whole-task claim validation drifted")
            except Exception:
                self._rollback(created, claim_root)
                raise
            self.claims = {item["run_id"]: claim for item, claim in created}
            reservation = _seal(
                {
                    "schema_version": RESERVATION_SCHEMA,
                    "status": "RESERVED",
                    "controller": self.plan["controller"],
                    "selection_rank": self.plan["attempts"][0]["selection_rank"],
                    "task_version_id": task_version,
                    "plan_sha256": self.plan["plan_sha256"],
                    "job_uid": job_uid,
                    "pod_uid": pod_uid,
                    "claim_sha256s": [
                        self.claims[row["run_id"]]["receipt_sha256"]
                        for row in self.plan["attempts"]
                    ],
                    "all_claims_before_model_call": True,
                    "all_claims_byte_validated": True,
                    "attempt_order": [1, 2, 3, 4],
                    "scores_included": False,
                    "prompts_or_traces_included": False,
                }
            )
            engine._write_once(self.out / "RESERVATION.json", reservation)
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)

    def __call__(
        self,
        plan: dict[str, Any],
        item: dict[str, Any],
        claim_root: Path,
        job_uid: str,
        pod_uid: str,
    ) -> dict[str, Any]:
        if plan != self.plan:
            raise RuntimeError("hosted whole-task claim provider plan drifted")
        if not self.claims:
            self._reserve(claim_root, job_uid, pod_uid)
        try:
            return self.claims[item["run_id"]]
        except KeyError as exc:
            raise RuntimeError("hosted whole-task item is outside the reservation") from exc
