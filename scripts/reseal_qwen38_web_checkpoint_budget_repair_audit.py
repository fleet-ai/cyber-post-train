#!/usr/bin/env python3
"""Reseal the one current, score-free WebExploitBench budget audit.

The receipt is deliberately narrow: it can refresh only the derived bindings in
``qwen38-web-important-checkpoint-budget-repair-prompt-preflight-20260921``.
It never contacts a provider, reads benchmark content, or makes a model or
scoring request.  Any schema, provenance, or semantic-invariant change needs a
new reviewed audit rather than a silent reseal.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
AUDIT_RELATIVE_PATH = (
    "docs/evidence/qwen38-web-important-checkpoint-budget-repair-prompt-preflight-20260921.json"
)
AUDIT_PATH = ROOT / AUDIT_RELATIVE_PATH

SCHEMA_VERSION = "qwen38_web_important_checkpoint_budget_audit_v6"
STATUS = "runtime_budget_repaired_prompt_preflight_parity_campaign_still_gated"
CLASSIFICATION = "runtime_budget_repaired_prompt_preflight_parity_campaign_still_gated"
REUSABLE_RULE = (
    "Keep model-free qualification in the same order as production. Prove the exact "
    "prompt runtime dependencies and all prompt and memory checks before changing "
    "host-wide containment state; then bind and remove only the qualification-owned "
    "proxy and network and restore the complete prior inventory. An added offline "
    "`--no-deps` wheel must have a sealed dependency closure; do not require a "
    "docs-only CLI package in the evaluator runtime. Preserve earlier immutable audits "
    "and keep collection closed until a fresh unique qualification accepts and releases."
)
SCOPE = (
    "Static, score-blind repository audit only. It performed no provider request, "
    "model request, scoring, collection, or external mutation and makes no "
    "model-capability claim. The exact V3 child error remains underdetermined because "
    "private stderr was not inspected."
)
OPERATION = {
    "provider_requests": 0,
    "model_requests": 0,
    "scoring_requests": 0,
    "external_mutations": 0,
}

SOURCE_PATHS = {
    "historical_netproxy_audit": (
        "docs/evidence/qwen38-web-important-checkpoint-budget-repair-netproxy-contract-20260921.json"
    ),
    "runtime_failure_evidence": (
        "docs/evidence/webexploitbench/2026-09-21-prompt-preflight-order-and-runtime-dependencies.md"
    ),
    "campaign_plan": (
        "configs/evaluation/qwen38-important-checkpoints-opencode-wbe-campaign-v1.json"
    ),
    "collection_runner": "evals/webexploitbench/tensorlake/bootstrap/run_collection_partition.sh",
    "collection_launcher": "evals/webexploitbench/tensorlake/collection_launcher.py",
    "collection_supervisor": "evals/webexploitbench/tensorlake/collection_supervisor.py",
    "qualification_worker": (
        "evals/webexploitbench/tensorlake/collection_snapshot_qualification_worker.py"
    ),
    "qualification_execution_packet": (
        "evals/webexploitbench/tensorlake/qualification_execution_packet.py"
    ),
}
RUNNER_FIELDS = {
    "runner_pre_collection_budget_seconds",
    "collection_dispatch_timeout_seconds",
    "collection_export_timeout_seconds",
    "collection_acceptance_timeout_seconds",
    "runner_shutdown_reserve_seconds",
}
CHECK_FIELD_ORDER = (
    "outer_collection_dispatch_covers_declared_task",
    "runner_envelope_covers_dispatch_and_evidence_preservation",
    "supervisor_horizon_covers_runner_envelope_with_reviewed_headroom",
    "sandbox_covers_supervisor_horizon",
    "prompt_runtime_dependency_probe_bound_to_current_source",
    "prompt_render_before_netproxy_bound_to_current_source",
    "non_strict_prompt_receipt_validation_bound_to_current_source",
    "qualification_netproxy_network_restoration_bound_to_current_source",
    "isolated_release_reconciliation_import_bound_to_current_source",
    "historical_netproxy_receipt_preserved",
    "campaign_is_marked_launchable",
    "fresh_model_free_runtime_qualification_still_required",
)
CHECK_FIELDS = set(CHECK_FIELD_ORDER)
TOP_LEVEL_FIELDS = {
    "schema_version",
    "status",
    "observed_at",
    "source",
    "observed_budget_seconds",
    "checks",
    "classification",
    "reusable_rule",
    "scope",
    "operation",
    "receipt_sha256",
}
SOURCE_FIELDS = {
    "repair_source_commit",
    "repair_base_commit",
    "campaign_plan_sha256",
    *{field for name in SOURCE_PATHS for field in (f"{name}_path", f"{name}_file_sha256")},
}
OBSERVED_BUDGET_FIELDS = {
    "declared_task_max",
    "actual_outer_collection_dispatch_limit",
    "runner_pre_collection_budget",
    "runner_preservation_budget",
    "runner_shutdown_reserve",
    "runner_envelope",
    "supervisor_poll_interval",
    "supervisor_poll_max_cycles",
    "supervisor_poll_horizon",
    "snapshot_restore_and_readiness_allowance",
    "campaign_collection_and_preservation_allowance",
    "sandbox_lifetime",
}
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_UTC_TIMESTAMP = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")


class AuditError(ValueError):
    """The fixed audit cannot be safely resealed from this checkout."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _digest_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _self_digest(value: Mapping[str, Any]) -> str:
    unsigned = dict(value)
    unsigned.pop("receipt_sha256", None)
    return _digest_bytes(_canonical(unsigned))


def _render(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2) + "\n").encode()


def _require_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AuditError(f"{label}_not_object")
    return value


def _require_exact_fields(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise AuditError(f"{label}_fields_mismatch")


def _require_regular_file(root: Path, relative: str, label: str) -> bytes:
    path = root / relative
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise AuditError(f"{label}_missing") from error
    if not resolved.is_relative_to(root.resolve()):
        raise AuditError(f"{label}_outside_repository")
    details = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(details.st_mode):
        raise AuditError(f"{label}_not_regular")
    return path.read_bytes()


def _read_json(root: Path, relative: str, label: str) -> dict[str, Any]:
    try:
        value = json.loads(_require_regular_file(root, relative, label))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AuditError(f"{label}_invalid_json") from error
    return _require_mapping(value, label)


def _require_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise AuditError(f"{label}_invalid_digest")
    return value


def _require_positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise AuditError(f"{label}_invalid_positive_int")
    return value


def _require_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise AuditError(f"{label}_not_bool")
    return value


def _validate_audit_shape(audit: Mapping[str, Any]) -> None:
    _require_exact_fields(audit, TOP_LEVEL_FIELDS, "audit")
    for field, expected in {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "classification": CLASSIFICATION,
        "reusable_rule": REUSABLE_RULE,
        "scope": SCOPE,
        "operation": OPERATION,
    }.items():
        if audit.get(field) != expected:
            raise AuditError(f"audit_{field}_invariant_drift")
    if (
        not isinstance(audit["observed_at"], str)
        or _UTC_TIMESTAMP.fullmatch(audit["observed_at"]) is None
    ):
        raise AuditError("audit_observed_at_invalid")
    _require_digest(audit["receipt_sha256"], "audit_receipt_sha256")
    if audit["receipt_sha256"] != _self_digest(audit):
        raise AuditError("audit_receipt_sha256_invalid")

    source = _require_mapping(audit["source"], "audit_source")
    _require_exact_fields(source, SOURCE_FIELDS, "audit_source")
    for field in ("repair_source_commit", "repair_base_commit"):
        if not isinstance(source[field], str) or _COMMIT.fullmatch(source[field]) is None:
            raise AuditError(f"audit_source_{field}_invalid")
    _require_digest(source["campaign_plan_sha256"], "audit_source_campaign_plan_sha256")
    for name, relative in SOURCE_PATHS.items():
        if source[f"{name}_path"] != relative:
            raise AuditError(f"audit_source_{name}_path_invariant_drift")
        _require_digest(source[f"{name}_file_sha256"], f"audit_source_{name}_file_sha256")

    observed = _require_mapping(audit["observed_budget_seconds"], "audit_observed_budget")
    _require_exact_fields(observed, OBSERVED_BUDGET_FIELDS, "audit_observed_budget")
    for name, value in observed.items():
        _require_positive_int(value, f"audit_observed_budget_{name}")

    checks = _require_mapping(audit["checks"], "audit_checks")
    _require_exact_fields(checks, CHECK_FIELDS, "audit_checks")
    for name, value in checks.items():
        _require_bool(value, f"audit_checks_{name}")


def _runner_values(source: str) -> dict[str, int]:
    values = {
        name: int(match.group(1))
        for name in RUNNER_FIELDS
        if (
            match := re.search(
                rf"^readonly {re.escape(name)}=([0-9]+)$",
                source,
                flags=re.MULTILINE,
            )
        )
    }
    if set(values) != RUNNER_FIELDS or any(value <= 0 for value in values.values()):
        raise AuditError("collection_runner_budget_constants_invalid")
    return values


def _supervisor_values(source: str) -> tuple[int, int]:
    values: list[int] = []
    for name in ("SUPERVISOR_POLL_INTERVAL_SECONDS", "SUPERVISOR_POLL_MAX_CYCLES"):
        match = re.search(rf"^{name} = ([0-9]+)$", source, flags=re.MULTILINE)
        if match is None:
            raise AuditError("collection_supervisor_poll_constants_invalid")
        values.append(int(match.group(1)))
    if any(value <= 0 for value in values):
        raise AuditError("collection_supervisor_poll_constants_invalid")
    return values[0], values[1]


def _plan_budget(plan: Mapping[str, Any]) -> tuple[int, int, int, int, bool]:
    protocol = _require_mapping(plan.get("protocol"), "campaign_plan_protocol")
    budget = _require_mapping(protocol.get("budget"), "campaign_plan_budget")
    execution = _require_mapping(plan.get("execution"), "campaign_plan_execution")
    task_minutes = _require_positive_int(
        budget.get("task_max_duration_minutes"), "campaign_plan_task_max_duration_minutes"
    )
    restore = _require_positive_int(
        budget.get("snapshot_restore_and_readiness_allowance_seconds"),
        "campaign_plan_snapshot_restore_and_readiness_allowance_seconds",
    )
    allowance = _require_positive_int(
        budget.get("collection_and_preservation_allowance_seconds"),
        "campaign_plan_collection_and_preservation_allowance_seconds",
    )
    sandbox = _require_positive_int(
        budget.get("sandbox_timeout_seconds"), "campaign_plan_sandbox_timeout_seconds"
    )
    launchable = _require_bool(execution.get("launchable_now"), "campaign_plan_launchable_now")
    return task_minutes * 60, restore, allowance, sandbox, launchable


def _historical_receipt_is_sealed(root: Path) -> bool:
    value = _read_json(root, SOURCE_PATHS["historical_netproxy_audit"], "historical_netproxy_audit")
    receipt = value.get("receipt_sha256")
    return isinstance(receipt, str) and receipt == _self_digest(value)


def _semantic_checks(
    *,
    worker_source: str,
    packet_source: str,
    runner_values: Mapping[str, int],
    supervisor_interval: int,
    supervisor_cycles: int,
    task_budget: int,
    sandbox_lifetime: int,
    launchable: bool,
    root: Path,
) -> dict[str, bool]:
    preservation = (
        runner_values["collection_export_timeout_seconds"]
        + runner_values["collection_acceptance_timeout_seconds"]
    )
    runner_envelope = (
        runner_values["runner_pre_collection_budget_seconds"]
        + runner_values["collection_dispatch_timeout_seconds"]
        + preservation
        + runner_values["runner_shutdown_reserve_seconds"]
    )
    supervisor_horizon = supervisor_interval * supervisor_cycles
    dependency_call = "runtime_preflight.update(\n        _prompt_runtime_dependency_preflight"
    project_loop = 'for binding in bundle["projects"]:'
    netproxy_call = "runtime_preflight.update(\n        _netproxy_after_prompt_preflight"
    prompt_dependency_before_projects = (
        dependency_call in worker_source
        and project_loop in worker_source
        and worker_source.index(dependency_call) < worker_source.index(project_loop)
    )
    prompt_render_before_netproxy = (
        project_loop in worker_source
        and netproxy_call in worker_source
        and worker_source.index(project_loop) < worker_source.index(netproxy_call)
        and 'raise QualificationError("prompt_preflight_root_remained")' in worker_source
    )
    isolated_release_reconciliation = all(
        marker in packet_source
        for marker in (
            "def _release_reconciliation_dependencies(",
            "sys.path.insert(0, str(repository))",
            "sys.path[:] = original_path",
            "def _reconcile_release_readonly(",
        )
    )
    checks = {
        "runner_envelope_covers_dispatch_and_evidence_preservation": runner_envelope
        == (
            runner_values["runner_pre_collection_budget_seconds"]
            + runner_values["collection_dispatch_timeout_seconds"]
            + preservation
            + runner_values["runner_shutdown_reserve_seconds"]
        ),
        "outer_collection_dispatch_covers_declared_task": (
            runner_values["collection_dispatch_timeout_seconds"] >= task_budget
        ),
        "supervisor_horizon_covers_runner_envelope_with_reviewed_headroom": (
            runner_envelope + 1800 <= supervisor_horizon
        ),
        "sandbox_covers_supervisor_horizon": supervisor_horizon <= sandbox_lifetime,
        "prompt_runtime_dependency_probe_bound_to_current_source": (
            prompt_dependency_before_projects
        ),
        "prompt_render_before_netproxy_bound_to_current_source": prompt_render_before_netproxy,
        "non_strict_prompt_receipt_validation_bound_to_current_source": (
            '"--no-strict-exit"' in worker_source
            and 'raise QualificationError("prompt_render_preflight_receipt_invalid")'
            in worker_source
        ),
        "qualification_netproxy_network_restoration_bound_to_current_source": all(
            marker in worker_source
            for marker in (
                "def _remove_expected_netproxy_and_restore_network_inventory(",
                'raise QualificationError("netproxy_network_inventory_not_restored")',
                '_require_netproxy_network_absent("before")',
            )
        ),
        "isolated_release_reconciliation_import_bound_to_current_source": (
            isolated_release_reconciliation
        ),
        "historical_netproxy_receipt_preserved": _historical_receipt_is_sealed(root),
        "campaign_is_marked_launchable": launchable,
        "fresh_model_free_runtime_qualification_still_required": True,
    }
    if (
        not all(
            checks[name]
            for name in checks
            if name
            not in {
                "campaign_is_marked_launchable",
            }
        )
        or checks["campaign_is_marked_launchable"]
    ):
        raise AuditError("current_web_budget_invariant_drift")
    return checks


def build_audit(*, root: Path = ROOT, audit_path: Path = AUDIT_PATH) -> dict[str, Any]:
    """Return the only permitted reseal of the current fixed audit."""

    root = root.resolve()
    try:
        audit = json.loads(audit_path.read_bytes())
    except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AuditError("audit_invalid_json") from error
    audit = _require_mapping(audit, "audit")
    _validate_audit_shape(audit)

    source_bytes = {
        name: _require_regular_file(root, relative, name) for name, relative in SOURCE_PATHS.items()
    }
    plan = _require_mapping(json.loads(source_bytes["campaign_plan"]), "campaign_plan")
    plan_sha256 = plan.get("sha256")
    if not isinstance(plan_sha256, str) or plan_sha256 != _digest_bytes(
        _canonical({key: value for key, value in plan.items() if key != "sha256"})
    ):
        raise AuditError("campaign_plan_self_digest_invalid")
    task_budget, restore, allowance, sandbox_lifetime, launchable = _plan_budget(plan)
    runner_values = _runner_values(source_bytes["collection_runner"].decode("utf-8"))
    supervisor_interval, supervisor_cycles = _supervisor_values(
        source_bytes["collection_supervisor"].decode("utf-8")
    )
    preservation = (
        runner_values["collection_export_timeout_seconds"]
        + runner_values["collection_acceptance_timeout_seconds"]
    )
    runner_envelope = (
        runner_values["runner_pre_collection_budget_seconds"]
        + runner_values["collection_dispatch_timeout_seconds"]
        + preservation
        + runner_values["runner_shutdown_reserve_seconds"]
    )
    observed = {
        "declared_task_max": task_budget,
        "actual_outer_collection_dispatch_limit": runner_values[
            "collection_dispatch_timeout_seconds"
        ],
        "runner_pre_collection_budget": runner_values["runner_pre_collection_budget_seconds"],
        "runner_preservation_budget": preservation,
        "runner_shutdown_reserve": runner_values["runner_shutdown_reserve_seconds"],
        "runner_envelope": runner_envelope,
        "supervisor_poll_interval": supervisor_interval,
        "supervisor_poll_max_cycles": supervisor_cycles,
        "supervisor_poll_horizon": supervisor_interval * supervisor_cycles,
        "snapshot_restore_and_readiness_allowance": restore,
        "campaign_collection_and_preservation_allowance": allowance,
        "sandbox_lifetime": sandbox_lifetime,
    }
    checks = _semantic_checks(
        worker_source=source_bytes["qualification_worker"].decode("utf-8"),
        packet_source=source_bytes["qualification_execution_packet"].decode("utf-8"),
        runner_values=runner_values,
        supervisor_interval=supervisor_interval,
        supervisor_cycles=supervisor_cycles,
        task_budget=task_budget,
        sandbox_lifetime=sandbox_lifetime,
        launchable=launchable,
        root=root,
    )
    if audit["checks"] != checks:
        raise AuditError("audit_checks_invariant_drift")

    expected = copy.deepcopy(audit)
    source = expected["source"]
    for name, raw in source_bytes.items():
        source[f"{name}_file_sha256"] = _digest_bytes(raw)
    source["campaign_plan_sha256"] = plan_sha256
    expected["observed_budget_seconds"] = observed
    expected["checks"] = {name: checks[name] for name in CHECK_FIELD_ORDER}
    expected["receipt_sha256"] = _self_digest(expected)
    return expected


def _write_atomically(path: Path, content: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        parent_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--write", action="store_true")
    args = parser.parse_args()
    expected = build_audit()
    rendered = _render(expected)
    if args.check:
        if AUDIT_PATH.read_bytes() != rendered:
            raise SystemExit(
                "current WebExploitBench budget audit is stale; run --write after review"
            )
        result = {"audit": AUDIT_RELATIVE_PATH, "external_mutations": 0, "status": "checked"}
    else:
        _write_atomically(AUDIT_PATH, rendered)
        result = {"audit": AUDIT_RELATIVE_PATH, "external_mutations": 0, "status": "resealed"}
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
