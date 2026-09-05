"""Executable, score-blind pre-bulk reconciliation for dedicated bulk v4.

The v4 observer reuses the reviewed generation-7 and exact-inventory readers
from v3, then binds their output to the dedicated-aware 798-cell partition.
It creates no model request and writes only create-once sanitized receipts.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_dedicated_bulk_v4 as bulk
from evals.fleet import exact_pass4_prebulk_reconciliation_v3 as prior
from evals.fleet import self_hosted

OBSERVATION_SCHEMA = "fleet-exact-pass4-dedicated-prebulk-observation-v4"
TERMINAL_SCHEMA = "fleet-exact-pass4-dedicated-prebulk-terminal-v4"
RELEASE_SCHEMA = "fleet-exact-pass4-dedicated-prebulk-release-v4"
SOURCE_JOB = "chris-cyber-exact100-prebulk-reconcile-source-v4"
ACCEPT_JOB = "chris-cyber-exact100-prebulk-reconcile-accept-v4"
NAMESPACE = "fleet-train-jobs"
OUTPUT_ROOT = Path(bulk.PREBULK_ROOT)
RELEASE_PATH = Path(
    "docs/evidence/qwen38-study/"
    "2026-09-05-opencode-exact-pass4-prebulk-reconciliation-release-v4.json"
)


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["receipt_sha256"] = self_hosted.digest_without(result, "receipt_sha256")
    return result


def _uuid(value: Any, label: str) -> str:
    return prior._uuid(value, label)  # noqa: SLF001


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith("sha256:")
        or any(char not in "0123456789abcdef" for char in value[7:])
    ):
        raise prior.ReconciliationError(f"{label}_invalid")
    return value


def _write_once(path: Path, value: dict[str, Any]) -> None:
    prior._write_once(path, value)  # noqa: SLF001


@contextmanager
def _v4_prior_output_root() -> Iterator[None]:
    original = prior.OUTPUT_ROOT
    try:
        prior.OUTPUT_ROOT = OUTPUT_ROOT
        yield
    finally:
        prior.OUTPUT_ROOT = original


def plan_bindings(root: Path) -> dict[str, Any]:
    value = bulk.load(root / bulk.SPEC_PATH)
    bulk.validate_spec(value, root)
    controllers = bulk.plans(value, root)
    canaries = bulk.scored_canary_plans(value, root)
    all_cells = [
        cell
        for plan in [*controllers.values(), *canaries.values()]
        for cell in plan["cells"]
    ]
    if len(all_cells) != 798 or len({cell["cell_id"] for cell in all_cells}) != 798:
        raise prior.ReconciliationError("dedicated_v4_partition_not_exact")
    return {
        "held_receipt_sha256": bulk.held_receipt(root)["receipt_sha256"],
        "controller_plan_sha256": {
            name: plan["plan_sha256"] for name, plan in controllers.items()
        },
        "scored_canary_plan_sha256": {
            replica: plan["plan_sha256"] for replica, plan in canaries.items()
        },
        "planned_execution_count": 798,
        "planned_cell_ids_sha256": self_hosted.sha256(
            self_hosted.canonical_json(sorted(cell["cell_id"] for cell in all_cells))
        ),
        "checked_job_names": sorted(
            plan["job_name"] for plan in [*controllers.values(), *canaries.values()]
        ),
        "checked_output_roots": sorted(
            plan["sfs_root"] for plan in [*controllers.values(), *canaries.values()]
        ),
    }


def build_release(root: Path, package_commit: str) -> dict[str, Any]:
    if bulk.v3.COMMIT_RE.fullmatch(package_commit) is None:
        raise ValueError("prebulk v4 package commit is invalid")
    return _seal(
        {
            "schema_version": RELEASE_SCHEMA,
            "status": "RELEASED",
            "package_commit": package_commit,
            "authority": plan_bindings(root),
            "authorized_jobs": [SOURCE_JOB, ACCEPT_JOB],
            "authorized_namespace": NAMESPACE,
            "cpu_only": True,
            "authorized_job_resource_policy": {
                "gpus": 0,
                "priority_class": "fleet-serve-low",
                "preemption_policy": "Never",
            },
            "scored_model_requests": 0,
            "launch_authorized": True,
            "prompts_traces_flags_or_scores_included": False,
        }
    )


def validate_release(value: dict[str, Any], root: Path) -> None:
    package_commit = value.get("package_commit")
    if value != build_release(root, str(package_commit)):
        raise ValueError("prebulk v4 release drifted")


def _all_plans(root: Path) -> dict[str, dict[str, Any]]:
    value = bulk.load(root / bulk.SPEC_PATH)
    controllers = bulk.plans(value, root)
    canaries = {
        f"glm-dedicated-{replica.lower()}-canary": plan
        for replica, plan in bulk.scored_canary_plans(value, root).items()
    }
    return controllers | canaries


def _assert_v4_absent(
    root: Path,
    kubernetes: Callable[[str, str | None], dict[str, Any]],
) -> None:
    plans = _all_plans(root)
    prior._assert_absent_jobs(plans, kubernetes)  # noqa: SLF001
    for plan in plans.values():
        path = Path(plan["sfs_root"])
        if path.exists() or path.is_symlink():
            raise prior.ReconciliationError("dedicated_v4_sfs_output_collision")


def collect(
    root: Path,
    *,
    job_uid: str,
    pod_uid: str,
    sessions: Callable[[str, str], list[dict[str, Any]]] = prior._default_sessions,  # noqa: SLF001
    kubernetes: Callable[[str, str | None], dict[str, Any]] = prior._default_kubernetes,  # noqa: SLF001
    api_key: str | None = None,
    observed_at: str | None = None,
    evidence_root: Path | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    with _v4_prior_output_root():
        base, canaries, inventory_package = prior.collect(
            root,
            job_uid=job_uid,
            pod_uid=pod_uid,
            sessions=sessions,
            kubernetes=kubernetes,
            api_key=api_key,
            observed_at=observed_at,
            evidence_root=evidence_root,
        )
    _assert_v4_absent(root, kubernetes)
    receipt = _seal(
        {
            "schema_version": OBSERVATION_SCHEMA,
            "status": "CLEAR_PENDING_SOURCE_JOB_ACCEPTANCE",
            "observed_at_utc": base["observed_at_utc"],
            "authority": plan_bindings(root),
            "prior_score_blind_observation": base,
            "v4_collisions": {
                "kubernetes_job_or_pod": 0,
                "sfs_output": 0,
                "accepted_active_claimed_or_model_started_cell": 0,
            },
            "runtime": {
                "namespace": NAMESPACE,
                "kind": "Job",
                "name": SOURCE_JOB,
                "job_uid": _uuid(job_uid, "source_job_uid"),
                "pod_uid": _uuid(pod_uid, "source_pod_uid"),
            },
            "request_counts": {
                "transcript_queries": 0,
                "mutation_calls": 0,
                "model_or_scoring_calls": 0,
            },
            "privacy": {
                "prompts_traces_flags_or_scores_included": False,
                "credentials_included": False,
            },
        }
    )
    validate_observation(receipt, root)
    return receipt, canaries, inventory_package


def validate_observation(receipt: dict[str, Any], root: Path) -> None:
    base = receipt.get("prior_score_blind_observation")
    if not isinstance(base, dict):
        raise prior.ReconciliationError("v4_prior_observation_absent")
    with _v4_prior_output_root():
        prior.validate_observation(base, root)
    runtime = receipt.get("runtime") or {}
    expected = {
        "schema_version": OBSERVATION_SCHEMA,
        "status": "CLEAR_PENDING_SOURCE_JOB_ACCEPTANCE",
        "observed_at_utc": base["observed_at_utc"],
        "authority": plan_bindings(root),
        "prior_score_blind_observation": base,
        "v4_collisions": {
            "kubernetes_job_or_pod": 0,
            "sfs_output": 0,
            "accepted_active_claimed_or_model_started_cell": 0,
        },
        "runtime": {
            "namespace": NAMESPACE,
            "kind": "Job",
            "name": SOURCE_JOB,
            "job_uid": _uuid(runtime.get("job_uid"), "source_job_uid"),
            "pod_uid": _uuid(runtime.get("pod_uid"), "source_pod_uid"),
        },
        "request_counts": {
            "transcript_queries": 0,
            "mutation_calls": 0,
            "model_or_scoring_calls": 0,
        },
        "privacy": {
            "prompts_traces_flags_or_scores_included": False,
            "credentials_included": False,
        },
    }
    if (
        {key: item for key, item in receipt.items() if key != "receipt_sha256"} != expected
        or receipt.get("receipt_sha256")
        != self_hosted.digest_without(receipt, "receipt_sha256")
    ):
        raise prior.ReconciliationError("v4_source_observation_invalid")


def validate_terminal(receipt: dict[str, Any], root: Path) -> None:
    source = receipt.get("source_observation") or {}
    runtime = receipt.get("runtime") or {}
    collector = receipt.get("collector_runtime") or {}
    package_commit = receipt.get("observer_package_commit")
    if not isinstance(package_commit, str) or bulk.v3.COMMIT_RE.fullmatch(package_commit) is None:
        raise prior.ReconciliationError("v4_terminal_package_commit_invalid")
    source_path = Path(str(source.get("path")))
    if (
        source_path != OUTPUT_ROOT / "OBSERVATION.json"
        or source_path.is_symlink()
        or not source_path.is_file()
        or prior._file_sha(source_path)  # noqa: SLF001
        != _sha256(source.get("file_sha256"), "source_file_sha256")
    ):
        raise prior.ReconciliationError("v4_terminal_source_file_invalid")
    source_value = prior._load(source_path)  # noqa: SLF001
    validate_observation(source_value, root)
    if (
        source_value.get("receipt_sha256")
        != _sha256(source.get("receipt_sha256"), "source_receipt_sha256")
        or runtime != source_value.get("runtime")
    ):
        raise prior.ReconciliationError("v4_terminal_source_binding_invalid")
    gates = receipt.get("generation7_gate_receipts")
    if not isinstance(gates, dict) or set(gates) != {"qwen3.8-27b", "glm-5.3"}:
        raise prior.ReconciliationError("v4_terminal_generation7_gate_set_invalid")
    for model, gate in gates.items():
        expected_path = OUTPUT_ROOT / f"{model}-generation7-gate.json"
        path = Path(str(gate.get("path")))
        if (
            path != expected_path
            or path.is_symlink()
            or not path.is_file()
            or prior._file_sha(path)  # noqa: SLF001
            != _sha256(gate.get("file_sha256"), f"{model}_gate_file_sha256")
        ):
            raise prior.ReconciliationError("v4_terminal_generation7_gate_file_invalid")
        gate_value = prior._load(path)  # noqa: SLF001
        prior.bulk.validate_canary_gate(gate_value, model, root)
        source_canary = source_value["prior_score_blind_observation"]["generation7"][model]
        if gate_value.get("receipt_sha256") != _sha256(
            gate.get("receipt_sha256"), f"{model}_gate_receipt_sha256"
        ) or {
            key: gate_value.get(key)
            for key in ("cell_id", "execution_id", "session_id", "verifier_execution_id")
        } != {
            key: source_canary[key]
            for key in ("cell_id", "execution_id", "session_id", "verifier_execution_id")
        }:
            raise prior.ReconciliationError("v4_terminal_generation7_gate_binding_invalid")
    inventory = receipt.get("exact100_inventory_package") or {}
    inventory_path = Path(str(inventory.get("path")))
    expected_inventory_path = OUTPUT_ROOT / prior.INVENTORY_PACKAGE_FILENAME
    if (
        inventory_path != expected_inventory_path
        or inventory_path.is_symlink()
        or not inventory_path.is_file()
        or prior._file_sha(inventory_path)  # noqa: SLF001
        != _sha256(inventory.get("file_sha256"), "inventory_package_file_sha256")
        or inventory.get("package_sha256")
        != source_value["prior_score_blind_observation"]["exact100_inventory_package"][
            "package_sha256"
        ]
    ):
        raise prior.ReconciliationError("v4_terminal_inventory_package_invalid")
    if (
        receipt.get("schema_version") != TERMINAL_SCHEMA
        or receipt.get("status") != "CLEAR"
        or receipt.get("authority") != plan_bindings(root)
        or receipt.get("planned_execution_count") != 798
        or receipt.get("checked_immediately_before_release") is not True
        or receipt.get("observer_job_succeeded") is not True
        or receipt.get("observer_pod_restarts") != 0
        or receipt.get("methods") != ["GET"]
        or receipt.get("mutation_calls") != 0
        or runtime.get("name") != SOURCE_JOB
        or runtime.get("namespace") != NAMESPACE
        or runtime.get("kind") != "Job"
        or collector.get("name") != ACCEPT_JOB
        or collector.get("namespace") != NAMESPACE
        or collector.get("kind") != "Job"
        or receipt.get("prompts_traces_flags_or_scores_included") is not False
        or receipt.get("receipt_sha256")
        != self_hosted.digest_without(receipt, "receipt_sha256")
    ):
        raise prior.ReconciliationError("v4_terminal_invalid")
    for label, value in (
        ("source_job_uid", runtime.get("job_uid")),
        ("source_pod_uid", runtime.get("pod_uid")),
        ("collector_job_uid", collector.get("job_uid")),
        ("collector_pod_uid", collector.get("pod_uid")),
    ):
        _uuid(value, label)


def accept(
    root: Path,
    observation: dict[str, Any],
    *,
    collector_job_uid: str,
    collector_pod_uid: str,
    package_commit: str,
    sessions: Callable[[str, str], list[dict[str, Any]]] = prior._default_sessions,  # noqa: SLF001
    kubernetes: Callable[[str, str | None], dict[str, Any]] = prior._default_kubernetes,  # noqa: SLF001
    api_key: str | None = None,
    observed_at: str | None = None,
    evidence_root: Path | None = None,
) -> dict[str, Any]:
    validate_observation(observation, root)
    source_runtime = observation["runtime"]
    source_state = prior._succeeded_job(  # noqa: SLF001
        SOURCE_JOB,
        source_runtime["job_uid"],
        source_runtime["pod_uid"],
        kubernetes,
    )
    fresh, canaries, inventory_package = collect(
        root,
        job_uid=collector_job_uid,
        pod_uid=collector_pod_uid,
        sessions=sessions,
        kubernetes=kubernetes,
        api_key=api_key,
        observed_at=observed_at,
        evidence_root=evidence_root,
    )
    if (
        fresh["authority"] != observation["authority"]
        or fresh["prior_score_blind_observation"]["generation7"]
        != observation["prior_score_blind_observation"]["generation7"]
        or fresh["prior_score_blind_observation"]["exact100_inventory"]
        != observation["prior_score_blind_observation"]["exact100_inventory"]
    ):
        raise prior.ReconciliationError("v4_source_authority_drifted")
    inventory_package_path = OUTPUT_ROOT / prior.INVENTORY_PACKAGE_FILENAME
    prior._write_bytes_once(inventory_package_path, inventory_package["raw"])  # noqa: SLF001
    if (
        prior._file_sha(inventory_package_path)  # noqa: SLF001
        != fresh["prior_score_blind_observation"]["exact100_inventory_package"]["file_sha256"]
    ):
        raise prior.ReconciliationError("v4_inventory_package_copy_drifted")
    with _v4_prior_output_root():
        gates = {
            model: prior._canary_gate(  # noqa: SLF001
                chain,
                collector_job_uid,
                collector_pod_uid,
                fresh["observed_at_utc"],
            )
            for model, chain in canaries.items()
        }
    for model, gate in gates.items():
        _write_once(OUTPUT_ROOT / f"{model}-generation7-gate.json", gate)
    source_path = OUTPUT_ROOT / "OBSERVATION.json"
    terminal = _seal(
        {
            "schema_version": TERMINAL_SCHEMA,
            "status": "CLEAR",
            "observer_package_commit": package_commit,
            "authority": fresh["authority"],
            "planned_execution_count": 798,
            "checked_immediately_before_release": True,
            "observer_job_succeeded": True,
            "observer_pod_restarts": source_state["pod_restarts"],
            "methods": ["GET"],
            "mutation_calls": 0,
            "runtime": source_runtime,
            "collector_runtime": {
                "namespace": NAMESPACE,
                "kind": "Job",
                "name": ACCEPT_JOB,
                "job_uid": _uuid(collector_job_uid, "collector_job_uid"),
                "pod_uid": _uuid(collector_pod_uid, "collector_pod_uid"),
            },
            "source_observation": {
                "path": str(source_path),
                "file_sha256": prior._file_sha(source_path),  # noqa: SLF001
                "receipt_sha256": observation["receipt_sha256"],
            },
            "exact100_inventory_package": {
                "path": str(inventory_package_path),
                "file_sha256": prior._file_sha(inventory_package_path),  # noqa: SLF001
                "package_sha256": fresh["prior_score_blind_observation"][
                    "exact100_inventory_package"
                ]["package_sha256"],
            },
            "generation7_gate_receipts": {
                model: {
                    "path": str(OUTPUT_ROOT / f"{model}-generation7-gate.json"),
                    "file_sha256": prior._file_sha(  # noqa: SLF001
                        OUTPUT_ROOT / f"{model}-generation7-gate.json"
                    ),
                    "receipt_sha256": gate["receipt_sha256"],
                }
                for model, gate in gates.items()
            },
            "prompts_traces_flags_or_scores_included": False,
        }
    )
    validate_terminal(terminal, root)
    return terminal


def main() -> int:
    global OUTPUT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("render-release", "validate-release", "source", "accept", "validate"),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--package-commit")
    parser.add_argument("--evidence-root", type=Path)
    args = parser.parse_args()
    OUTPUT_ROOT = args.output_root
    root = args.root.resolve(strict=True)
    if args.command == "render-release":
        if not args.package_commit:
            parser.error("render-release requires --package-commit")
        print(json.dumps(build_release(root, args.package_commit), indent=2, sort_keys=True))
        return 0
    release = prior._load(root / RELEASE_PATH)  # noqa: SLF001
    validate_release(release, root)
    if args.command == "validate-release":
        return 0
    if args.command == "validate":
        validate_observation(prior._load(OUTPUT_ROOT / "OBSERVATION.json"), root)  # noqa: SLF001
        validate_terminal(prior._load(OUTPUT_ROOT / "TERMINAL.json"), root)  # noqa: SLF001
        return 0
    if args.command == "source":
        receipt, _, _ = collect(
            root,
            job_uid=os.environ["JOB_UID"],
            pod_uid=os.environ["POD_UID"],
            evidence_root=args.evidence_root,
        )
        _write_once(OUTPUT_ROOT / "OBSERVATION.json", receipt)
        return 0
    if not args.package_commit:
        parser.error("accept requires --package-commit")
    observation = prior._load(OUTPUT_ROOT / "OBSERVATION.json")  # noqa: SLF001
    terminal = accept(
        root,
        observation,
        collector_job_uid=os.environ["JOB_UID"],
        collector_pod_uid=os.environ["POD_UID"],
        package_commit=args.package_commit,
        evidence_root=args.evidence_root,
    )
    _write_once(OUTPUT_ROOT / "TERMINAL.json", terminal)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
