"""Finite-path score-blind release observer for hosted-Qwen rank 17.

Unlike the terminal v1 observer, this successor never recursively parses the
historical jobs tree.  It checks the exact finite paths the bound rank-17
engine can author, plus the authoritative task/model/version session inventory.
"""

from __future__ import annotations

import argparse
import contextlib
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from evals.fleet import qwen_hosted_rank17_g22_release_observer_v1 as base
except ModuleNotFoundError:  # projected ConfigMap runtime
    import qwen_hosted_rank17_g22_release_observer_v1 as base  # type: ignore[no-redef]

SCHEMA = "fleet-qwen38-hosted-rank17-g22-release-observation-v3"
PACKAGE_SCHEMA = "fleet-qwen38-hosted-rank17-g22-release-package-v3"
FAILURE_SCHEMA = "fleet-qwen38-hosted-rank17-g22-release-failure-v3"
SCORED_ROOT = Path("/mnt/sfs/jobs/chris-q38-hosted-r017-whole-task-g22-v1")
SAFE_FAILURE_CODES = base.SAFE_FAILURE_CODES | {
    "finite_path_contract_invalid",
    "observer_failed_safely",
    "release_package_file_drifted",
    "release_package_source_escape",
    "release_package_source_invalid",
}


def finite_paths(binding: dict[str, Any]) -> dict[str, list[str]]:
    base.validate_binding(binding)
    run_ids = [row["run_id"] for row in binding["cells"]]
    paths = {
        "canonical_claim": list(binding["planned_claim_paths"]),
        "legacy_per_run_accepted": list(binding["planned_accepted_paths"]),
        "engine_accepted_registry": [
            str(SCORED_ROOT / "accepted" / f"{run_id}.json") for run_id in run_ids
        ],
    }
    if (
        any(len(rows) != 4 or len(rows) != len(set(rows)) for rows in paths.values())
        or set(paths["canonical_claim"]) != set(binding["planned_claim_paths"])
        or set(paths["legacy_per_run_accepted"]) != set(binding["planned_accepted_paths"])
    ):
        raise base.GateError("finite_path_contract_invalid")
    return paths


def _finite_paths_absent(binding: dict[str, Any]) -> dict[str, int]:
    observed: dict[str, int] = {}
    for receipt_class, rows in finite_paths(binding).items():
        observed[receipt_class] = base._planned_paths_absent(  # noqa: SLF001
            rows, failure_code="planned_statistical_cell_collision"
        )
    return observed


def _session_collisions(
    binding: dict[str, Any], key: str, expected: set[str]
) -> tuple[int, int, int]:
    rows_examined = 0
    gets = 0
    collisions = 0
    offset = 0
    pages_seen = 0
    session_ids_seen: set[str] = set()
    while True:
        pages_seen += 1
        if pages_seen > base.MAX_SESSION_PAGES:
            raise base.GateError("fleet_session_pagination_stalled")
        page = base._fleet_get(  # noqa: SLF001
            "/v1/sessions",
            key,
            {"task_key": binding["task_key"], "limit": 500, "offset": offset},
        )
        gets += 1
        if "sessions" not in page or "has_more" not in page:
            raise base.GateError("fleet_session_inventory_invalid")
        rows = page["sessions"]
        has_more = page["has_more"]
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise base.GateError("fleet_session_inventory_invalid")
        if not isinstance(has_more, bool):
            raise base.GateError("fleet_session_inventory_invalid")
        session_ids = [row.get("session_id") for row in rows]
        if any(not isinstance(value, str) or not value for value in session_ids):
            raise base.GateError("fleet_session_inventory_invalid")
        if len(set(session_ids)) != len(session_ids) or session_ids_seen.intersection(session_ids):
            raise base.GateError("fleet_session_pagination_stalled")
        session_ids_seen.update(session_ids)
        rows_examined += len(rows)
        if rows_examined > base.MAX_SESSION_ROWS:
            raise base.GateError("fleet_session_pagination_stalled")
        for row in rows:
            eval_version = row.get("eval_task_version_id")
            task_version = row.get("task_version_id")
            provided_versions = [
                value for value in (eval_version, task_version) if value is not None
            ]
            if any(not isinstance(value, str) or not value for value in provided_versions):
                raise base.GateError("fleet_session_identity_ambiguous")
            if len(provided_versions) == 2 and provided_versions[0] != provided_versions[1]:
                raise base.GateError("fleet_session_identity_ambiguous")
            projected_version = provided_versions[0] if provided_versions else None
            row_identities = base._identity_values(row)  # noqa: SLF001
            planned_identity_collision = bool(row_identities.intersection(expected))
            model = row.get("model")
            if not isinstance(model, str) or not model:
                raise base.GateError("fleet_session_identity_ambiguous")
            if model != binding["session_model"]:
                collisions += int(planned_identity_collision)
                continue
            if projected_version is None:
                raise base.GateError("fleet_session_identity_ambiguous")
            if projected_version == binding["task_version_id"]:
                collisions += 1
            else:
                collisions += int(planned_identity_collision)
        if has_more is False:
            return rows_examined, gets, collisions
        if not rows:
            raise base.GateError("fleet_session_pagination_stalled")
        offset += len(rows)


def collect(
    binding: dict[str, Any], *, job_uid: str, pod_uid: str, api_key: str
) -> dict[str, Any]:
    base.validate_binding(binding)
    account = base._fleet_get("/v1/account", api_key)  # noqa: SLF001
    if account.get("team_name") != "fleet" or account.get("team_id") != base.FLEET_TEAM_ID:
        raise base.GateError("fleet_team_identity_invalid")
    kubernetes_gets = base._fresh_objects_absent(binding)  # noqa: SLF001
    base._sfs_roots_clear(binding)  # noqa: SLF001
    absent = _finite_paths_absent(binding)
    expected = set(binding["identity_values"])
    session_rows, session_gets, session_collisions = _session_collisions(
        binding, api_key, expected
    )
    slots = base._lease_slots_clear(binding)  # noqa: SLF001
    if session_collisions:
        raise base.GateError("planned_statistical_cell_collision")
    receipt = base.seal(
        {
            "schema_version": SCHEMA,
            "status": "CLEAR_PENDING_TERMINAL_JOB_VALIDATION",
            "observed_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "binding_sha256": binding["binding_sha256"],
            "plan_commit": base.PLAN_COMMIT,
            "plan_sha256": binding["plan_sha256"],
            "held_receipt_sha256": binding["held_receipt_sha256"],
            "authoritative_tally": base.EXPECTED_TALLY,
            "statistical_cell_count": 4,
            "all_planned_cells_observed_unstarted": True,
            "collisions": {
                "canonical_claims": 0,
                "legacy_per_run_accepted": 0,
                "engine_accepted_registry": 0,
                "authoritative_sessions": 0,
                "fresh_kubernetes_objects": 0,
                "sfs_output_roots": 0,
            },
            "observed_aggregates": {
                "canonical_claim_paths_absent": absent["canonical_claim"],
                "legacy_per_run_accepted_paths_absent": absent["legacy_per_run_accepted"],
                "engine_accepted_registry_paths_absent": absent["engine_accepted_registry"],
                "authoritative_session_rows_examined": session_rows,
                "fresh_object_sets_absent": 1,
                "checked_sfs_roots_absent": 2,
                "endpoint_lease_slots_simultaneously_free": slots,
            },
            "request_counts": {
                "fleet_account_gets": 1,
                "fleet_session_inventory_gets": session_gets,
                "kubernetes_gets": kubernetes_gets,
                "transcript_prompt_task_verifier_or_score_gets": 0,
            },
            "runtime": {
                "namespace": base.NAMESPACE,
                "job_uid": base._uuid(job_uid, "observer_job_uid"),  # noqa: SLF001
                "pod_uid": base._uuid(pod_uid, "observer_pod_uid"),  # noqa: SLF001
            },
            "methods": ["GET"],
            "model_calls": 0,
            "task_calls": 0,
            "session_mutations": 0,
            "verifier_calls": 0,
            "scoring_calls": 0,
            "api_mutations": 0,
            "scores_included": False,
            "prompts_traces_flags_included": False,
            "credentials_included": False,
        }
    )
    validate_observation(receipt, binding=binding)
    return receipt


def validate_observation(receipt: Any, *, binding: dict[str, Any]) -> None:
    if not isinstance(receipt, dict):
        raise base.GateError("release_observation_invalid")
    aggregates = receipt.get("observed_aggregates")
    requests = receipt.get("request_counts")
    runtime = receipt.get("runtime")
    if any(
        (
            set(receipt)
            != {
                "schema_version",
                "status",
                "observed_at_utc",
                "binding_sha256",
                "plan_commit",
                "plan_sha256",
                "held_receipt_sha256",
                "authoritative_tally",
                "statistical_cell_count",
                "all_planned_cells_observed_unstarted",
                "collisions",
                "observed_aggregates",
                "request_counts",
                "runtime",
                "methods",
                "model_calls",
                "task_calls",
                "session_mutations",
                "verifier_calls",
                "scoring_calls",
                "api_mutations",
                "scores_included",
                "prompts_traces_flags_included",
                "credentials_included",
                "receipt_sha256",
            },
            receipt.get("schema_version") != SCHEMA,
            receipt.get("status") != "CLEAR_PENDING_TERMINAL_JOB_VALIDATION",
            receipt.get("binding_sha256") != binding["binding_sha256"],
            receipt.get("plan_commit") != base.PLAN_COMMIT,
            receipt.get("plan_sha256") != binding["plan_sha256"],
            receipt.get("held_receipt_sha256") != binding["held_receipt_sha256"],
            receipt.get("authoritative_tally") != base.EXPECTED_TALLY,
            receipt.get("statistical_cell_count") != 4,
            receipt.get("all_planned_cells_observed_unstarted") is not True,
            receipt.get("collisions")
            != {
                "canonical_claims": 0,
                "legacy_per_run_accepted": 0,
                "engine_accepted_registry": 0,
                "authoritative_sessions": 0,
                "fresh_kubernetes_objects": 0,
                "sfs_output_roots": 0,
            },
            not isinstance(aggregates, dict),
            aggregates
            != {
                "canonical_claim_paths_absent": 4,
                "legacy_per_run_accepted_paths_absent": 4,
                "engine_accepted_registry_paths_absent": 4,
                "authoritative_session_rows_examined": aggregates.get(
                    "authoritative_session_rows_examined", -1
                )
                if isinstance(aggregates, dict)
                else -1,
                "fresh_object_sets_absent": 1,
                "checked_sfs_roots_absent": 2,
                "endpoint_lease_slots_simultaneously_free": 2,
            },
            not isinstance(aggregates, dict)
            or not isinstance(aggregates.get("authoritative_session_rows_examined"), int)
            or aggregates.get("authoritative_session_rows_examined", -1) < 0,
            not isinstance(requests, dict),
            set(requests or {})
            != {
                "fleet_account_gets",
                "fleet_session_inventory_gets",
                "kubernetes_gets",
                "transcript_prompt_task_verifier_or_score_gets",
            },
            (requests or {}).get("fleet_account_gets") != 1,
            not isinstance((requests or {}).get("fleet_session_inventory_gets"), int),
            (requests or {}).get("fleet_session_inventory_gets", 0) < 1,
            (requests or {}).get("kubernetes_gets") != 3,
            (requests or {}).get("transcript_prompt_task_verifier_or_score_gets") != 0,
            not isinstance(runtime, dict),
            set(runtime or {}) != {"namespace", "job_uid", "pod_uid"},
            (runtime or {}).get("namespace") != base.NAMESPACE,
            receipt.get("methods") != ["GET"],
            any(
                receipt.get(field) != 0
                for field in (
                    "model_calls",
                    "task_calls",
                    "session_mutations",
                    "verifier_calls",
                    "scoring_calls",
                    "api_mutations",
                )
            ),
            receipt.get("scores_included") is not False,
            receipt.get("prompts_traces_flags_included") is not False,
            receipt.get("credentials_included") is not False,
            receipt.get("receipt_sha256") != base.digest(receipt),
            base.UTC_RE.fullmatch(str(receipt.get("observed_at_utc"))) is None,
        )
    ):
        raise base.GateError("release_observation_invalid")
    base._uuid((runtime or {}).get("job_uid"), "observer_job_uid")  # noqa: SLF001
    base._uuid((runtime or {}).get("pod_uid"), "observer_pod_uid")  # noqa: SLF001


def validate_package_source(path: Path, package_root: Path) -> None:
    try:
        root = package_root.resolve(strict=True)
        source = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise base.GateError("release_package_source_invalid") from exc
    if not source.is_relative_to(root):
        raise base.GateError("release_package_source_escape")
    value = base.load(source)
    files = value.get("files")
    expected = {
        "binding.json",
        "qwen_hosted_rank17_g22_release_observer_v1.py",
        "qwen_hosted_rank17_g22_release_observer_v3.py",
    }
    if (
        not isinstance(files, dict)
        or set(value) != {"schema_version", "files", "file_count", "receipt_sha256"}
        or value.get("schema_version") != PACKAGE_SCHEMA
        or set(files) != expected
        or value.get("file_count") != 3
        or value.get("receipt_sha256") != base.digest(value)
    ):
        raise base.GateError("release_package_source_invalid")
    for name, expected_digest in files.items():
        try:
            target = (package_root / name).resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise base.GateError("release_package_file_drifted") from exc
        if (
            not target.is_relative_to(root)
            or not target.is_file()
            or target.stat().st_size > base.MAX_JSON_BYTES
            or base.sha256(target.read_bytes()) != expected_digest
        ):
            raise base.GateError("release_package_file_drifted")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--package-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    stage = "package-source"
    try:
        validate_package_source(args.package_source, args.package_source.parent)
        stage = "binding"
        binding = base.load_projected(args.binding, args.package_source.parent)
        key = os.environ.get("FLEET_API_KEY")
        if not key:
            raise base.GateError("fleet_api_key_absent")
        stage = "collect"
        receipt = collect(
            binding,
            job_uid=os.environ["JOB_UID"],
            pod_uid=os.environ["POD_UID"],
            api_key=key,
        )
        stage = "write"
        base.write_once(args.output, receipt)
    except Exception as exc:
        code = str(exc)
        failure = base.seal(
            {
                "schema_version": FAILURE_SCHEMA,
                "status": "FAILED",
                "last_stage": stage,
                "failure_code": code if code in SAFE_FAILURE_CODES else "redacted",
                "failure_category": (
                    "safe_gate_failure" if code in SAFE_FAILURE_CODES else "unexpected_failure"
                ),
                "job_uid": os.environ.get("JOB_UID"),
                "pod_uid": os.environ.get("POD_UID"),
                "model_calls": 0,
                "task_calls": 0,
                "session_mutations": 0,
                "verifier_calls": 0,
                "scoring_calls": 0,
                "api_mutations": 0,
                "scores_included": False,
                "prompts_traces_flags_included": False,
                "credentials_included": False,
            }
        )
        with contextlib.suppress(Exception):
            base.write_once(args.output.with_name("FAILED.json"), failure)
        raise base.GateError("observer_failed_safely") from None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
