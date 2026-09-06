"""Conservative score-blind release observer for hosted-Qwen rank 18."""

from __future__ import annotations

import argparse
import contextlib
import os
import ssl
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from evals.fleet import qwen_hosted_rank17_g22_release_observer_v1 as base
    from evals.fleet import score_blind_session_inventory_v1 as stream
except ModuleNotFoundError:  # projected ConfigMap runtime
    import qwen_hosted_rank17_g22_release_observer_v1 as base  # type: ignore[no-redef]
    import score_blind_session_inventory_v1 as stream  # type: ignore[no-redef]

SCHEMA = "fleet-qwen38-hosted-rank18-g23-release-observation-v1"
BINDING_SCHEMA = "fleet-qwen38-hosted-rank18-g23-release-binding-v1"
PACKAGE_SCHEMA = "fleet-qwen38-hosted-rank18-g23-release-package-v1"
FAILURE_SCHEMA = "fleet-qwen38-hosted-rank18-g23-release-failure-v1"
PLAN_COMMIT = "3e79a43ba195dc759dc14a890ba5007cd88343b1"
EXPECTED_TALLY = base.EXPECTED_TALLY
EXPECTED_PLAN_MODULE_SHA256 = (
    "sha256:c435983e5c7f5cdfe663da53bf613a1039c6b9e528bef88ae4b20ae47d3b692f"
)
EXPECTED_HELD_FILE_SHA256 = (
    "sha256:bb5205cf05ec42cd0777e429cef69f21b6efb1f5f3d67cd7b42d1e68c476ad3b"
)
EXPECTED_HELD_RECEIPT_SHA256 = (
    "sha256:7bdd013e9156355c4de991f5f62fbffa98eaeecaa101582a09de6fe1674653bf"
)
EXPECTED_PLAN_SHA256 = (
    "sha256:299cfbc83217bf6e5aea03a17bd85ee57207ccdfc798b632f7cfb30c327f813f"
)
EXPECTED_TASK_KEY = (
    "cysec1-2-fentry-gen_blackbox-ebd29ca3d4ae284704202ebc__blackbox_ctf_v1"
)
EXPECTED_TASK_VERSION_ID = "3aafc5a8-2144-4bc3-919f-fcbbebd9f0d7"
EXPECTED_SESSION_MODEL = base.EXPECTED_SESSION_MODEL
EXPECTED_FRESH_OBJECT = {
    "job_name": "chris-q38-hosted-r018-whole-task-g23-v1",
    "configmap_name": "chris-q38-hosted-r018-whole-task-g23-package-v1",
}
EXPECTED_SFS_ROOTS = [
    "/mnt/sfs/jobs/chris-q38-hosted-r018-whole-task-g23-v1",
    "/mnt/sfs/jobs/chris-q38-hosted-r018-whole-task-g23-v1-diagnostic",
]
EXPECTED_CELLS = [
    {
        "attempt": 1,
        "cell_id": "sha256:4eda1896bf25d2fd6deb9d1cc021fff3a60d66c9662a15a3bd324c6203c6dda4",
        "execution_id": "sha256:f32be93b9c559a697b53653f596991e1cdaad5e69e1549e6e2741484fefe783a",
        "run_id": "chris-q38-ac-g23v1-b-r018-a1-f32be93b",
    },
    {
        "attempt": 2,
        "cell_id": "sha256:7028c4817efa58fbe3406cd348655d31f658de82be8bfef07c6c397b5a1ac561",
        "execution_id": "sha256:3e9b135ff70ddcba6dfdf5dbcdae7f756ebde8c1cc7e35ef2624148440990cae",
        "run_id": "chris-q38-ac-g23v1-b-r018-a2-3e9b135f",
    },
    {
        "attempt": 3,
        "cell_id": "sha256:d88e5f205de4467deda0fa876617588b937c2b3bc90435233cbe7ad87e08ac50",
        "execution_id": "sha256:b7bca0dd03abd61712087825b1e451126f25df5f8f309698a040d48cbc157856",
        "run_id": "chris-q38-ac-g23v1-b-r018-a3-b7bca0dd",
    },
    {
        "attempt": 4,
        "cell_id": "sha256:5ae35b4613ede22ea8151792b9518ef8b16692b7115ad96e8aee328b12e0ae7d",
        "execution_id": "sha256:93a3a6d66c134226ecf0460ebf27d06526eb13a710a7ee460ae00962693aee1b",
        "run_id": "chris-q38-ac-g23v1-b-r018-a4-93a3a6d6",
    },
]
SCORED_ROOT = Path(EXPECTED_SFS_ROOTS[0])
SAFE_FAILURE_CODES = base.SAFE_FAILURE_CODES | {
    "finite_path_contract_invalid",
    "observer_failed_safely",
    "session_inventory_invalid",
    "session_inventory_too_large",
}


def validate_binding(binding: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "plan_commit",
        "plan_module_sha256",
        "held_file_sha256",
        "held_receipt_sha256",
        "plan_sha256",
        "authoritative_tally",
        "statistical_cell_count",
        "cells",
        "identity_values",
        "task_key",
        "task_version_id",
        "session_model",
        "fresh_object",
        "checked_sfs_roots",
        "claim_root",
        "planned_claim_paths",
        "jobs_root",
        "planned_accepted_paths",
        "lease_root",
        "endpoint_key",
        "binding_sha256",
    }
    cells = binding.get("cells")
    identities = binding.get("identity_values")
    if any(
        (
            set(binding) != required,
            binding.get("schema_version") != BINDING_SCHEMA,
            binding.get("plan_commit") != PLAN_COMMIT,
            binding.get("plan_module_sha256") != EXPECTED_PLAN_MODULE_SHA256,
            binding.get("held_file_sha256") != EXPECTED_HELD_FILE_SHA256,
            binding.get("held_receipt_sha256") != EXPECTED_HELD_RECEIPT_SHA256,
            binding.get("plan_sha256") != EXPECTED_PLAN_SHA256,
            binding.get("authoritative_tally") != EXPECTED_TALLY,
            binding.get("statistical_cell_count") != 4,
            not isinstance(cells, list) or cells != EXPECTED_CELLS,
            not isinstance(identities, list) or len(identities) != 12,
            len(set(identities or [])) != 12,
            identities != sorted(base._identity_values(EXPECTED_CELLS)),  # noqa: SLF001
            binding.get("task_key") != EXPECTED_TASK_KEY,
            binding.get("task_version_id") != EXPECTED_TASK_VERSION_ID,
            binding.get("session_model") != EXPECTED_SESSION_MODEL,
            binding.get("fresh_object") != EXPECTED_FRESH_OBJECT,
            binding.get("checked_sfs_roots") != EXPECTED_SFS_ROOTS,
            binding.get("claim_root")
            != "/mnt/sfs/cell-execution-claims/opencode11827-autocontinue-v1",
            binding.get("jobs_root") != "/mnt/sfs/jobs",
            binding.get("lease_root")
            != "/mnt/sfs/endpoint-leases/opencode11827-autocontinue-primary-v1",
            binding.get("endpoint_key") != "qwen-hosted-autocontinue-v1",
            binding.get("planned_claim_paths")
            != [
                f"{binding.get('claim_root')}/{row['execution_id'].removeprefix('sha256:')}.json"
                for row in (cells or [])
            ],
            binding.get("planned_accepted_paths")
            != [f"/mnt/sfs/jobs/{row['run_id']}/ACCEPTED.json" for row in (cells or [])],
            binding.get("binding_sha256") != base.binding_digest(binding),
        )
    ):
        raise base.GateError("release_binding_invalid")


def finite_paths(binding: dict[str, Any]) -> dict[str, list[str]]:
    validate_binding(binding)
    rows = {
        "canonical_claim": list(binding["planned_claim_paths"]),
        "legacy_per_run_accepted": list(binding["planned_accepted_paths"]),
        "engine_accepted_registry": [
            str(SCORED_ROOT / "accepted" / f"{row['run_id']}.json")
            for row in binding["cells"]
        ],
    }
    if any(len(value) != 4 or len(set(value)) != 4 for value in rows.values()):
        raise base.GateError("finite_path_contract_invalid")
    return rows


def _finite_paths_absent(binding: dict[str, Any]) -> dict[str, int]:
    return {
        name: base._planned_paths_absent(  # noqa: SLF001
            paths, failure_code="planned_statistical_cell_collision"
        )
        for name, paths in finite_paths(binding).items()
    }


def _response_chunks(response: Any) -> Any:
    while True:
        chunk = response.read(16 * 1024)
        if not chunk:
            return
        yield chunk


def _session_page(key: str, task_key: str, offset: int) -> stream.SessionPage:
    query = urllib.parse.urlencode({"task_key": task_key, "limit": 500, "offset": offset})
    request = urllib.request.Request(
        f"{base.ORCHESTRATOR}/v1/sessions?{query}",
        method="GET",
        headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
    )
    try:
        opener = urllib.request.build_opener(base._RejectRedirects())  # noqa: SLF001
        with opener.open(request, timeout=60) as response:
            return stream.parse_session_page(_response_chunks(response))
    except stream.InventoryError as exc:
        raise base.GateError(str(exc)) from None
    except (OSError, TimeoutError, urllib.error.URLError, ssl.SSLError):
        raise base.GateError("read_request_failed") from None


def _session_collisions(binding: dict[str, Any], key: str) -> tuple[int, int, int]:
    rows_examined = 0
    collisions = 0
    offset = 0
    session_ids: set[str] = set()
    for gets in range(1, base.MAX_SESSION_PAGES + 1):
        page = _session_page(key, binding["task_key"], offset)
        if page.limit != 500 or page.offset != offset:
            raise base.GateError("fleet_session_inventory_invalid")
        if not page.sessions and page.has_more:
            raise base.GateError("fleet_session_pagination_stalled")
        for row in page.sessions:
            session_id = row.get("session_id")
            eval_task_id = row.get("eval_task_id")
            task_key = row.get("task_key")
            model = row.get("model")
            status = row.get("status")
            if (
                not isinstance(session_id, str)
                or not session_id
                or session_id in session_ids
                or not isinstance(eval_task_id, str)
                or not eval_task_id
                or task_key != binding["task_key"]
                or not isinstance(model, str)
                or not model
                or (status is not None and not isinstance(status, str))
            ):
                raise base.GateError("fleet_session_identity_ambiguous")
            session_ids.add(session_id)
            rows_examined += 1
            if rows_examined > base.MAX_SESSION_ROWS:
                raise base.GateError("fleet_session_pagination_stalled")
            collisions += int(model == binding["session_model"])
        if not page.has_more:
            return rows_examined, gets, collisions
        offset += len(page.sessions)
    raise base.GateError("fleet_session_pagination_stalled")


def collect(
    binding: dict[str, Any], *, job_uid: str, pod_uid: str, api_key: str
) -> dict[str, Any]:
    validate_binding(binding)
    account = base._fleet_get("/v1/account", api_key)  # noqa: SLF001
    if account.get("team_name") != "fleet" or account.get("team_id") != base.FLEET_TEAM_ID:
        raise base.GateError("fleet_team_identity_invalid")
    kubernetes_gets = base._fresh_objects_absent(binding)  # noqa: SLF001
    base._sfs_roots_clear(binding)  # noqa: SLF001
    absent = _finite_paths_absent(binding)
    session_rows, session_gets, session_collisions = _session_collisions(binding, api_key)
    slots = base._lease_slots_clear(binding)  # noqa: SLF001
    if session_collisions:
        raise base.GateError("planned_statistical_cell_collision")
    receipt = base.seal(
        {
            "schema_version": SCHEMA,
            "status": "CLEAR_PENDING_TERMINAL_JOB_VALIDATION",
            "observed_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "binding_sha256": binding["binding_sha256"],
            "plan_commit": PLAN_COMMIT,
            "plan_sha256": binding["plan_sha256"],
            "held_receipt_sha256": binding["held_receipt_sha256"],
            "authoritative_tally": EXPECTED_TALLY,
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
                "engine_accepted_registry_paths_absent": absent[
                    "engine_accepted_registry"
                ],
                "authoritative_session_rows_examined": session_rows,
                "fresh_object_sets_absent": 1,
                "checked_sfs_roots_absent": 2,
                "endpoint_lease_slots_simultaneously_free": slots,
            },
            "request_counts": {
                "fleet_account_gets": 1,
                "fleet_session_inventory_gets": session_gets,
                "kubernetes_gets": kubernetes_gets,
                "protected_value_gets": 0,
            },
            "session_collision_policy": "any_exact_model_row_collides_regardless_version",
            "session_inventory_parser": "bounded_streaming_allowlist_v1",
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
    expected = collect_shape(binding)
    for field in ("observed_at_utc", "runtime", "observed_aggregates", "request_counts"):
        expected[field] = receipt.get(field)
    expected["receipt_sha256"] = receipt.get("receipt_sha256")
    if receipt != expected or receipt.get("receipt_sha256") != base.digest(receipt):
        raise base.GateError("release_observation_invalid")
    if base.UTC_RE.fullmatch(str(receipt.get("observed_at_utc"))) is None:
        raise base.GateError("release_observation_invalid")
    runtime = receipt.get("runtime")
    aggregates = receipt.get("observed_aggregates")
    requests = receipt.get("request_counts")
    if (
        not isinstance(runtime, dict)
        or set(runtime) != {"namespace", "job_uid", "pod_uid"}
        or runtime.get("namespace") != base.NAMESPACE
        or not isinstance(aggregates, dict)
        or set(aggregates)
        != {
            "canonical_claim_paths_absent",
            "legacy_per_run_accepted_paths_absent",
            "engine_accepted_registry_paths_absent",
            "authoritative_session_rows_examined",
            "fresh_object_sets_absent",
            "checked_sfs_roots_absent",
            "endpoint_lease_slots_simultaneously_free",
        }
        or any(not isinstance(value, int) or value < 0 for value in aggregates.values())
        or aggregates.get("canonical_claim_paths_absent") != 4
        or aggregates.get("legacy_per_run_accepted_paths_absent") != 4
        or aggregates.get("engine_accepted_registry_paths_absent") != 4
        or aggregates.get("fresh_object_sets_absent") != 1
        or aggregates.get("checked_sfs_roots_absent") != 2
        or aggregates.get("endpoint_lease_slots_simultaneously_free") != 2
        or not isinstance(requests, dict)
        or set(requests)
        != {
            "fleet_account_gets",
            "fleet_session_inventory_gets",
            "kubernetes_gets",
            "protected_value_gets",
        }
        or requests.get("fleet_account_gets") != 1
        or not isinstance(requests.get("fleet_session_inventory_gets"), int)
        or requests.get("fleet_session_inventory_gets", 0) < 1
        or requests.get("kubernetes_gets") != 3
        or requests.get("protected_value_gets") != 0
    ):
        raise base.GateError("release_observation_invalid")
    base._uuid(runtime.get("job_uid"), "observer_job_uid")  # noqa: SLF001
    base._uuid(runtime.get("pod_uid"), "observer_pod_uid")  # noqa: SLF001


def collect_shape(binding: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "status": "CLEAR_PENDING_TERMINAL_JOB_VALIDATION",
        "observed_at_utc": None,
        "binding_sha256": binding["binding_sha256"],
        "plan_commit": PLAN_COMMIT,
        "plan_sha256": binding["plan_sha256"],
        "held_receipt_sha256": binding["held_receipt_sha256"],
        "authoritative_tally": EXPECTED_TALLY,
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
        "observed_aggregates": None,
        "request_counts": None,
        "session_collision_policy": "any_exact_model_row_collides_regardless_version",
        "session_inventory_parser": "bounded_streaming_allowlist_v1",
        "runtime": None,
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
        "qwen_hosted_rank18_g23_release_observer_v1.py",
        "score_blind_session_inventory_v1.py",
    }
    if (
        not isinstance(files, dict)
        or set(value) != {"schema_version", "files", "file_count", "receipt_sha256"}
        or value.get("schema_version") != PACKAGE_SCHEMA
        or set(files) != expected
        or value.get("file_count") != len(expected)
        or value.get("receipt_sha256") != base.digest(value)
    ):
        raise base.GateError("release_package_source_invalid")
    for name, expected_digest in files.items():
        try:
            target = (root / name).resolve(strict=True)
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
    # Capture and canonicalize downward-API identities before any package,
    # binding, or inventory operation can fail. Raw environment bytes must
    # never cross the sanitized failure boundary.
    stage = "runtime-identity"
    job_uid: str | None = None
    pod_uid: str | None = None
    try:
        job_uid = base._uuid(os.environ.get("JOB_UID"), "observer_job_uid")  # noqa: SLF001
        pod_uid = base._uuid(os.environ.get("POD_UID"), "observer_pod_uid")  # noqa: SLF001
        stage = "package-source"
        validate_package_source(args.package_source, args.package_source.parent)
        stage = "binding"
        binding = base.load_projected(args.binding, args.package_source.parent)
        validate_binding(binding)
        key = os.environ.get("FLEET_API_KEY")
        if not key:
            raise base.GateError("fleet_api_key_absent")
        stage = "collect"
        receipt = collect(
            binding,
            job_uid=job_uid,
            pod_uid=pod_uid,
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
                "job_uid": job_uid,
                "pod_uid": pod_uid,
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
