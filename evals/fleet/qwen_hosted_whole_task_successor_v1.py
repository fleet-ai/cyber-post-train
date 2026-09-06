"""Held atomic whole-task successors for hosted Qwen ranks 15 and 16.

Each controller owns exactly one previously untouched statistical task.  A
controller publishes and byte-validates all four canonical execution claims
under one task reservation lock before the first model call.  The four attempts
then execute sequentially while one of the two qualified hosted endpoint leases
is held.
"""

from __future__ import annotations

import contextlib
import copy
import fcntl
import json
import os
import stat
from collections.abc import Callable
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_bulk_runtime_v3 as engine
from evals.fleet import qwen_hosted_generation19_bulk as original
from evals.fleet import qwen_hosted_generation19_v4 as source
from evals.fleet import self_hosted

SCHEMA = "fleet-qwen38-hosted-atomic-whole-task-plan-v2"
HELD_SCHEMA = "fleet-qwen38-hosted-atomic-whole-task-held-v8"
RELEASE_SCHEMA = "fleet-qwen38-hosted-atomic-whole-task-release-v4"
RESERVATION_SCHEMA = "fleet-qwen38-hosted-four-claim-reservation-v2"
PREPARING_SCHEMA = "fleet-qwen38-hosted-four-claim-preparing-v1"
MODEL_BOUNDARY_SCHEMA = "fleet-qwen38-hosted-model-boundary-v1"
PACKAGE_SOURCE_SCHEMA = "fleet-qwen38-hosted-package-source-v1"
LEASE_OBSERVER_SCHEMA = "fleet-qwen38-hosted-endpoint-lease-observer-v1"
RUNTIME_GATE_CANARY_SCHEMA = "fleet-qwen38-hosted-whole-task-runtime-gate-canary-v1"
LEDGER_PATH = "docs/evidence/qwen38-study/2026-09-05-exact-pass4-ledger-evidence-snapshot-v47.json"
LEDGER_SELF_SHA256 = "sha256:bf0b9086dd97eecafe20fa9a4cf3b5d643f0ce8f6abad60fae6e3cba3e3e2e29"
LEDGER_FILE_SHA256 = "sha256:0a2baba7c16745a4d69f0f5aafc04010734eacbace8f6d712d44011c4a36d0dd"
HELD_PATH = (
    "docs/evidence/qwen38-study/2026-09-06-qwen38-hosted-rank15-rank16-whole-task-held-v8.json"
)
SUPERSEDED_HELD = {
    "path": (
        "docs/evidence/qwen38-study/2026-09-06-qwen38-hosted-rank15-rank16-whole-task-held-v7.json"
    ),
    "receipt_sha256": "sha256:41b9a5146e57e3b577cf7bc64db9a1003fe14b0bd59f298fcc7b65c47083cca0",
    "file_sha256": "sha256:f00b4ab6cb52733ae06609c43c6bb8ac58da425f8372b0ffeb6c18eefe009880",
}
CANARY_FAILURE = {
    "path": (
        "docs/evidence/qwen38-study/2026-09-06-qwen38-hosted-runtime-gate-canary-v4-failure.json"
    ),
    "receipt_sha256": "sha256:2d99e969a6ff9911a2385747107302a4cff099795fdc3644baedab39e3626e20",
    "file_sha256": "sha256:c09fa52f4e280f1a58904260751f8da7ae618dcc6f40b2422bef2419c66f8519",
}
PRECLAIM_FAILURE = {
    "path": (
        "docs/evidence/qwen38-study/2026-09-06-qwen38-hosted-rank15-rank16-preclaim-failure-v2.json"
    ),
    "receipt_sha256": "sha256:5d51c5e8bd0009c8f01462087c11e6cf1531e87588517fea251e96d7c4030f60",
    "file_sha256": "sha256:e423007ca58b69462d7fdd3a13a9f325a34a1a0f828473882c5b72d574dc560d",
}
CLAIM_ROOT = Path("/mnt/sfs/cell-execution-claims/opencode11827-autocontinue-v1")
RESERVATION_ROOT = Path("/mnt/sfs/cell-execution-reservations/opencode11827-autocontinue-v1")
JOBS_ROOT = Path("/mnt/sfs/jobs")
ENDPOINT_LEASE_ROOT = Path("/mnt/sfs/endpoint-leases/opencode11827-autocontinue-primary-v1")
ENDPOINT_KEY = "qwen-hosted-autocontinue-v1"
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
                "durable_preparing_before_claims": True,
                "restart_recovers_partial_publication": True,
                "model_boundary_is_durable_and_nonrepeatable": True,
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
                    "durable_preparing_before_claims": True,
                    "restart_recovers_partial_publication": True,
                    "model_boundary_is_durable_and_nonrepeatable": True,
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


def observe_endpoint_lease_slots(
    observer_pod_uid: str,
    *,
    lease_root: Path = ENDPOINT_LEASE_ROOT,
    endpoint_key: str = ENDPOINT_KEY,
) -> dict[str, Any]:
    """Prove both persistent slot inodes are simultaneously free without deletion."""
    if engine.UUID_RE.fullmatch(observer_pod_uid) is None:
        raise ValueError("hosted whole-task lease observer requires a Pod UID")
    endpoint_root = lease_root / endpoint_key
    if endpoint_root.is_symlink() or not endpoint_root.is_dir():
        raise RuntimeError("hosted whole-task endpoint lease directory is unsafe")
    handles: list[tuple[Any, Path, os.stat_result]] = []
    slots: list[dict[str, Any]] = []
    try:
        for slot in (1, 2):
            path = endpoint_root / f"slot-{slot}.lock"
            flags = os.O_RDWR
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            try:
                fd = os.open(path, flags)
            except OSError as exc:
                raise RuntimeError("hosted whole-task endpoint lease inode drifted") from exc
            handle = os.fdopen(fd, "a+b")
            info = os.fstat(handle.fileno())
            handles.append((handle, path, info))
            if not stat.S_ISREG(info.st_mode) or info.st_size != 0:
                raise RuntimeError("hosted whole-task endpoint lease inode drifted")
            try:
                path_info = path.stat(follow_symlinks=False)
            except OSError as exc:
                raise RuntimeError("hosted whole-task endpoint lease inode drifted") from exc
            if (path_info.st_dev, path_info.st_ino) != (info.st_dev, info.st_ino):
                raise RuntimeError("hosted whole-task endpoint lease inode drifted")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("hosted whole-task endpoint lease slot is held") from exc
            slots.append(
                {
                    "slot": slot,
                    "device": info.st_dev,
                    "inode": info.st_ino,
                    "size": info.st_size,
                    "path_sha256": self_hosted.sha256(str(path).encode()),
                    "file_sha256": self_hosted.sha256(b""),
                }
            )
        for _handle, path, info in handles:
            try:
                path_info = path.stat(follow_symlinks=False)
            except OSError as exc:
                raise RuntimeError("hosted whole-task endpoint lease inode drifted") from exc
            if (path_info.st_dev, path_info.st_ino) != (info.st_dev, info.st_ino):
                raise RuntimeError("hosted whole-task endpoint lease inode drifted")
    finally:
        for handle, _path, _info in reversed(handles):
            if not handle.closed:
                with contextlib.suppress(OSError):
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                handle.close()
    if len(slots) != 2:
        raise RuntimeError("hosted whole-task endpoint lease probe was incomplete")
    return _seal(
        {
            "schema_version": LEASE_OBSERVER_SCHEMA,
            "status": "BOTH_SLOTS_FREE",
            "observer_pod_uid": observer_pod_uid,
            "lease_root_sha256": self_hosted.sha256(str(lease_root).encode()),
            "endpoint_key_sha256": self_hosted.sha256(endpoint_key.encode()),
            "slots": slots,
            "simultaneous_nonblocking_exclusive_acquisition": True,
            "all_locks_released": True,
            "files_created": 0,
            "files_deleted": 0,
            "api_mutations": 0,
            "scores_included": False,
            "prompts_or_traces_included": False,
        }
    )


def validate_endpoint_lease_observer(value: dict[str, Any]) -> None:
    slots = value.get("slots") or []
    if any(
        (
            set(value)
            != {
                "schema_version",
                "status",
                "observer_pod_uid",
                "lease_root_sha256",
                "endpoint_key_sha256",
                "slots",
                "simultaneous_nonblocking_exclusive_acquisition",
                "all_locks_released",
                "files_created",
                "files_deleted",
                "api_mutations",
                "scores_included",
                "prompts_or_traces_included",
                "receipt_sha256",
            },
            value.get("schema_version") != LEASE_OBSERVER_SCHEMA,
            value.get("status") != "BOTH_SLOTS_FREE",
            engine.UUID_RE.fullmatch(str(value.get("observer_pod_uid"))) is None,
            value.get("lease_root_sha256") != self_hosted.sha256(str(ENDPOINT_LEASE_ROOT).encode()),
            value.get("endpoint_key_sha256") != self_hosted.sha256(ENDPOINT_KEY.encode()),
            not isinstance(slots, list),
            len(slots) != 2,
            [row.get("slot") for row in slots] != [1, 2],
            any(
                set(row) != {"slot", "device", "inode", "size", "path_sha256", "file_sha256"}
                or type(row.get("device")) is not int
                or row["device"] < 1
                or type(row.get("inode")) is not int
                or row["inode"] < 1
                or row.get("size") != 0
                or row.get("path_sha256")
                != self_hosted.sha256(
                    str(
                        ENDPOINT_LEASE_ROOT / ENDPOINT_KEY / f"slot-{row.get('slot')}.lock"
                    ).encode()
                )
                or row.get("file_sha256") != self_hosted.sha256(b"")
                for row in slots
            ),
            len({(row.get("device"), row.get("inode")) for row in slots}) != 2,
            value.get("simultaneous_nonblocking_exclusive_acquisition") is not True,
            value.get("all_locks_released") is not True,
            value.get("files_created") != 0,
            value.get("files_deleted") != 0,
            value.get("api_mutations") != 0,
            value.get("scores_included") is not False,
            value.get("prompts_or_traces_included") is not False,
            value.get("receipt_sha256") != self_hosted.digest_without(value, "receipt_sha256"),
        )
    ):
        raise RuntimeError("hosted whole-task endpoint lease observer drifted")


def validate_runtime_gate_canary(value: dict[str, Any]) -> None:
    if any(
        (
            set(value)
            != {
                "schema_version",
                "status",
                "controller",
                "plan_sha256",
                "job_uid",
                "pod_uid",
                "authority_schema_version",
                "authority_receipt_sha256",
                "package_source_receipt_sha256",
                "bootstrap_stage_reached",
                "output_roots_created",
                "endpoint_leases_acquired",
                "canonical_claims_created",
                "model_calls",
                "task_calls",
                "session_calls",
                "verifier_calls",
                "scoring_calls",
                "api_mutations",
                "scores_included",
                "prompts_or_traces_included",
                "credentials_included",
                "receipt_sha256",
            },
            value.get("schema_version") != RUNTIME_GATE_CANARY_SCHEMA,
            value.get("status") != "PASS",
            value.get("controller") not in CONTROLLERS,
            SHA256_RE.fullmatch(str(value.get("plan_sha256"))) is None,
            engine.UUID_RE.fullmatch(str(value.get("job_uid"))) is None,
            engine.UUID_RE.fullmatch(str(value.get("pod_uid"))) is None,
            value.get("authority_schema_version") != HELD_SCHEMA,
            SHA256_RE.fullmatch(str(value.get("authority_receipt_sha256"))) is None,
            SHA256_RE.fullmatch(str(value.get("package_source_receipt_sha256"))) is None,
            value.get("bootstrap_stage_reached") != "06-runtime-exec",
            any(
                value.get(field) != 0
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
            ),
            value.get("scores_included") is not False,
            value.get("prompts_or_traces_included") is not False,
            value.get("credentials_included") is not False,
            value.get("receipt_sha256") != self_hosted.digest_without(value, "receipt_sha256"),
        )
    ):
        raise RuntimeError("hosted whole-task runtime-gate canary receipt drifted")


def package_source_receipt(
    controller: str, plan: dict[str, Any], data: dict[str, str]
) -> dict[str, Any]:
    """Seal every immutable ConfigMap source byte except the mutable release."""
    if controller not in CONTROLLERS or plan.get("controller") != controller:
        raise ValueError("hosted whole-task package source controller drifted")
    if "release.json" in data or "package-source.json" in data:
        raise ValueError("hosted whole-task package source includes a mutable binding")
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
            set(receipt)
            != {
                "schema_version",
                "controller",
                "plan_sha256",
                "configmap_name",
                "files",
                "file_count",
                "release_excluded",
                "scores_included",
                "prompts_or_traces_included",
                "receipt_sha256",
            },
            receipt.get("schema_version") != PACKAGE_SOURCE_SCHEMA,
            receipt.get("controller") != controller,
            receipt.get("plan_sha256") != plan["plan_sha256"],
            receipt.get("configmap_name") != CONTROLLERS[controller]["configmap_name"],
            not isinstance(files, dict),
            not files,
            any(not isinstance(name, str) or not name for name in files),
            any(SHA256_RE.fullmatch(str(value)) is None for value in files.values()),
            "release.json" in files,
            "package-source.json" in files,
            receipt.get("file_count") != len(files),
            receipt.get("release_excluded") is not True,
            receipt.get("scores_included") is not False,
            receipt.get("prompts_or_traces_included") is not False,
            receipt.get("receipt_sha256") != self_hosted.digest_without(receipt, "receipt_sha256"),
        )
    ):
        raise RuntimeError("hosted whole-task package source receipt drifted")


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
                "supersedes",
                "preclaim_failure",
                "canary_failure",
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
            held.get("controllers") != release_projection(plans, package_sources),
            held.get("supersedes") != SUPERSEDED_HELD,
            held.get("preclaim_failure") != PRECLAIM_FAILURE,
            held.get("canary_failure") != CANARY_FAILURE,
            held.get("ledger_snapshot_path") != LEDGER_PATH,
            held.get("ledger_snapshot_receipt_sha256") != LEDGER_SELF_SHA256,
            held.get("ledger_snapshot_file_sha256") != LEDGER_FILE_SHA256,
            held.get("retry_forbidden_selection_ranks") != [13, 14],
            held.get("predecessor_tombstones") != PREDECESSOR_TOMBSTONES,
            held.get("required_fresh_release_gates")
            != [
                "prior_jobs_terminal_and_pods_absent",
                "endpoint_lease_slots_simultaneously_lockable",
                "no_relevant_active_hosted_q_jobs_or_pods",
                "canonical_claim_collisions_zero",
                "authoritative_session_collisions_zero",
                "accepted_evidence_collisions_zero",
                "output_root_collisions_zero",
                "durable_preparing_before_four_claim_publication",
                "crash_recovery_before_model_boundary",
                "atomic_four_claim_publication_and_validation",
                "immutable_package_source_digest_binding",
                "score_free_private_regular_input_stage06_runtime_gate_canary_pass",
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


def validate_release(
    release: dict[str, Any],
    plans: dict[str, dict[str, Any]],
    package_sources: dict[str, dict[str, Any]],
) -> None:
    del release, plans, package_sources
    raise RuntimeError(
        "hosted whole-task scored release is closed for consumed object and execution identities"
    )


def load_runtime_release(
    plans: dict[str, dict[str, Any]], package_source: dict[str, Any]
) -> dict[str, Any]:
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
    controller = str(package_source.get("controller"))
    if controller not in plans:
        raise RuntimeError("hosted whole-task runtime package controller drifted")
    validate_package_source_receipt(package_source, controller, plans[controller])
    projected = {row["controller"]: row for row in release.get("controllers") or []}
    if projected.get(controller, {}).get("package_source_receipt_sha256") != package_source.get(
        "receipt_sha256"
    ):
        raise RuntimeError("hosted whole-task runtime package source binding drifted")
    # The renderer validates both immutable package sources.  Each create-once
    # controller independently revalidates its own embedded source before use.
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
            raise RuntimeError("hosted whole-task PREPARING transaction drifted")

    def _load_preparings(self) -> list[dict[str, Any]]:
        if not self.preparing_root.exists():
            return []
        if self.preparing_root.is_symlink() or not self.preparing_root.is_dir():
            raise RuntimeError("hosted whole-task PREPARING root is unsafe")
        values: list[dict[str, Any]] = []
        for path in self.preparing_root.iterdir():
            if path.is_symlink() or not path.is_file() or path.suffix != ".json":
                raise RuntimeError("hosted whole-task PREPARING entry is unsafe")
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
                raise RuntimeError("hosted whole-task existing PREPARING bytes drifted") from None
        self._validate_preparing(value)
        return value

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
            raise RuntimeError("hosted whole-task reservation bytes drifted")
        if any((claim["job_uid"], claim["pod_uid"]) != owner for claim in claims.values()):
            raise RuntimeError("hosted whole-task reservation claim owner drifted")
        return claims

    def _recover(self, claim_root: Path) -> bool:
        """Commit a complete transaction or roll back only proven pre-model bytes."""
        preparings = self._load_preparings()
        owners = {(row["job_uid"], row["pod_uid"]) for row in preparings}
        boundary_root = self.out / "model-boundaries"
        if boundary_root.exists() and (boundary_root.is_symlink() or not boundary_root.is_dir()):
            raise RuntimeError("hosted whole-task model boundary root is unsafe")
        model_boundary_exists = boundary_root.exists() and any(boundary_root.iterdir())
        reservation_path = self.out / "RESERVATION.json"
        if reservation_path.exists() or reservation_path.is_symlink():
            if reservation_path.is_symlink() or not reservation_path.is_file():
                raise RuntimeError("hosted whole-task reservation path is unsafe")
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
                        "hosted whole-task committed reservation drifted after model boundary"
                    ) from None
                reservation_path.unlink()
        if model_boundary_exists:
            raise RuntimeError("hosted whole-task claims drifted after model boundary")
        for item in reversed(self.plan["attempts"]):
            path = claim_root / engine.claim_filename(item["execution_id"])
            if not path.exists() and not path.is_symlink():
                continue
            if path.is_symlink() or not path.is_file():
                raise RuntimeError("hosted whole-task partial claim path is unsafe")
            claim = engine._validate_preserved_claim(self.plan, item, claim_root)
            if (claim["job_uid"], claim["pod_uid"]) not in owners:
                raise RuntimeError("hosted whole-task claim lacks PREPARING ownership")
            if claim.get("model_call_started_when_claim_written") is not False:
                raise RuntimeError("hosted whole-task recovery crossed model boundary")
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
                        raise RuntimeError("hosted whole-task claim transaction collided")
                    created[item["run_id"]] = claim
                    self.fault_hook(f"after_claim_{count}")
                for count, item in enumerate(reversed(self.plan["attempts"]), start=1):
                    claim = created[item["run_id"]]
                    validated = engine._validate_preserved_claim(self.plan, item, claim_root)
                    if validated != claim:
                        raise RuntimeError("hosted whole-task claim validation drifted")
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
            raise RuntimeError("hosted whole-task model boundary lacks reserved claim")
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
                raise RuntimeError("hosted whole-task model boundary marker drifted")
            raise RuntimeError("hosted whole-task model boundary already crossed; retry prohibited")
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
            raise RuntimeError("hosted whole-task claim provider plan drifted")
        if not self.claims:
            self._reserve(claim_root, job_uid, pod_uid)
        try:
            return self.claims[item["run_id"]]
        except KeyError as exc:
            raise RuntimeError("hosted whole-task item is outside the reservation") from exc
