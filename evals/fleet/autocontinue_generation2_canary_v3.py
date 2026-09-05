"""Fail-closed evidence contracts for generation-2 autocontinue canaries.

This append-only successor retains the v2 specification and rendered-plan
format while hardening release, generation-claim, and terminal boundaries.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.fleet import autocontinue_canary_controller as legacy
from evals.fleet import autocontinue_generation2_canary as held_v1
from evals.fleet import autocontinue_generation2_canary_v2 as held_v2
from evals.fleet import hosted_sweep_controller as hosted

HELD_SCHEMA = "fleet-opencode-autocontinue-generation2-canary-held-release-v3"
RELEASE_SCHEMA = "fleet-opencode-autocontinue-generation2-canary-scoring-release-v3"
TERMINAL_SCHEMA = "fleet-opencode-autocontinue-generation2-terminal-v3"
V2_HELD_PATH = (
    "docs/evidence/qwen38-study/"
    "2026-09-04-opencode-autocontinue-generation2-canaries-held-v2.json"
)
MODULE_PATH = "evals/fleet/autocontinue_generation2_canary_v3.py"

SHA_RE = re.compile(r"sha256:[0-9a-f]{64}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")


def load(path: Path) -> dict[str, Any]:
    return held_v2.load(path)


def digest(value: dict[str, Any], field: str) -> str:
    return held_v2.digest(value, field)


def file_sha256(path: Path) -> str:
    return held_v2.file_sha256(path)


def _canonical_utc(value: Any) -> bool:
    if not isinstance(value, str) or not value.endswith("Z"):
        return False
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() == UTC.utcoffset(parsed)


def validate_spec(spec: dict[str, Any], root: Path) -> dict[str, Any]:
    return held_v2.validate_spec(spec, root)


def _expected_implementation(root: Path, model: str) -> dict[str, Any]:
    expected = held_v1.EXPECTED[model]
    old = load(root / held_v2.V1_SPEC_PATHS[model])
    return {
        "module_path": MODULE_PATH,
        "module_sha256": file_sha256(root / MODULE_PATH),
        "job_name": expected["job_name"],
        "configmap_name": expected["configmap_name"],
        "sfs_root": old["identities"]["sfs_root"],
    }


def validate_held(release: dict[str, Any], specs: list[dict[str, Any]], root: Path) -> None:
    plans = [validate_spec(spec, root) for spec in specs]
    prior = load(root / V2_HELD_PATH)
    held_v2.validate_held(prior, specs, root)
    expected = {
        "schema_version": HELD_SCHEMA,
        "append_only": True,
        "status": "HELD",
        "observed_at_utc": release.get("observed_at_utc"),
        "supersedes_held_receipt": {
            "path": V2_HELD_PATH,
            "receipt_sha256": prior["receipt_sha256"],
        },
        "generation2_spec_sha256s": [
            spec["generation2_spec_sha256"] for spec in specs
        ],
        "rendered_plan_sha256s": [plan["plan_sha256"] for plan in plans],
        "jobs": [held_v1.EXPECTED[spec["model"]]["job_name"] for spec in specs],
        "execution_generation": 2,
        "statistical_cells": 2,
        "implementation": {
            "module_path": MODULE_PATH,
            "module_sha256": file_sha256(root / MODULE_PATH),
        },
        "launch_authorized": False,
        "bulk_release_authorized": False,
        "dedicated_serving_authorized": False,
        "remaining_gates": [
            "independent_v3_contract_audit",
            "v3_executable_manifest_and_launcher",
            "fresh_hosted_route_and_duplicate_inventory",
            "append_only_model_specific_scoring_releases",
            "explicit_root_launch_authorization",
        ],
        "privacy": {
            "prompts_traces_flags_or_scores_included": False,
            "credentials_included": False,
        },
    }
    if (
        not _canonical_utc(release.get("observed_at_utc"))
        or release.get("receipt_sha256") != digest(release, "receipt_sha256")
        or {key: value for key, value in release.items() if key != "receipt_sha256"}
        != expected
    ):
        raise ValueError("generation-2 v3 held release drifted")


def validate_release(
    release: dict[str, Any],
    spec: dict[str, Any],
    root: Path,
    package_commit: str,
    *,
    authorized_at_utc: str,
    authorization_statement: str,
) -> None:
    plan = validate_spec(spec, root)
    if not COMMIT_RE.fullmatch(package_commit) or not _canonical_utc(authorized_at_utc):
        raise ValueError("generation-2 v3 release authority is invalid")
    if not authorization_statement:
        raise ValueError("generation-2 v3 release authority is invalid")
    expected = {
        "schema_version": RELEASE_SCHEMA,
        "append_only": True,
        "status": "RELEASED",
        "released_at_utc": authorized_at_utc,
        "generation2_spec_sha256": spec["generation2_spec_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "cell_id": spec["statistical_cell"]["cell_id"],
        "execution_id": spec["execution"]["execution_id"],
        "incident_receipt_sha256": held_v1.INCIDENT_SHA,
        "tombstone_bundle_receipt_sha256": held_v1._tombstones(root)[
            "receipt_sha256"
        ],
        "package_commit": package_commit,
        "implementation": _expected_implementation(root, spec["model"]),
        "authorization": {
            "launch_authorized": True,
            "create_once": True,
            "execution_generation": 2,
            "bulk_release_authorized": False,
            "dedicated_serving_authorized": False,
            "author": "/root",
            "authorized_at_utc": authorized_at_utc,
            "statement": authorization_statement,
        },
        "route_and_inventory": {
            "fresh_authenticated_hosted_route_required": True,
            "fresh_exact_treatment_session_inventory_required": True,
            "old_claims_must_exist_and_match": True,
            "generation_2_claim_must_be_absent": True,
        },
        "privacy": {
            "prompts_traces_flags_or_scores_included": False,
            "credentials_included": False,
        },
    }
    if (
        release.get("receipt_sha256") != digest(release, "receipt_sha256")
        or {key: value for key, value in release.items() if key != "receipt_sha256"}
        != expected
    ):
        raise ValueError("generation-2 v3 scoring release is not authoritative")


def validate_claim(
    claim: dict[str, Any],
    spec: dict[str, Any],
    plan: dict[str, Any],
    *,
    job_uid: str,
    pod_uid: str,
    root: Path,
) -> None:
    expected_plan = validate_spec(spec, root)
    if plan != expected_plan or not legacy._is_uuid(job_uid) or not legacy._is_uuid(pod_uid):
        raise ValueError("generation-2 v3 claim identity drifted")
    expected = {
        "schema_version": "fleet-statistical-cell-execution-claim-v2",
        "generation2_spec_sha256": spec["generation2_spec_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "cell_id": spec["statistical_cell"]["cell_id"],
        "execution_id": spec["execution"]["execution_id"],
        "execution_generation": 2,
        "run_id": plan["attempts"][0]["run_id"],
        "job_uid": job_uid,
        "pod_uid": pod_uid,
        "claimed_at_utc": claim.get("claimed_at_utc"),
        "generation_1_claims_preserved": True,
        "immutable": True,
        "automatic_retry": False,
        "scores_included": False,
        "prompts_or_traces_included": False,
    }
    if (
        not _canonical_utc(claim.get("claimed_at_utc"))
        or claim.get("receipt_sha256") != digest(claim, "receipt_sha256")
        or {key: value for key, value in claim.items() if key != "receipt_sha256"}
        != expected
    ):
        raise ValueError("generation-2 v3 claim is not authoritative")


def claim_execution_generation(
    spec: dict[str, Any],
    plan: dict[str, Any],
    claim_root: Path | None = None,
    repo: Path | None = None,
) -> dict[str, Any]:
    root = repo or Path.cwd()
    expected = validate_spec(spec, root)
    if plan != expected:
        raise ValueError("generation-2 v3 spec-plan chain drifted")
    claim = held_v2.claim_execution_generation(spec, plan, claim_root, repo=root)
    validate_claim(
        claim,
        spec,
        plan,
        job_uid=os.environ.get("JOB_UID", ""),
        pod_uid=os.environ.get("POD_UID", ""),
        root=root,
    )
    return claim


def _expected_cell_bindings(plan: dict[str, Any]) -> tuple[str, str]:
    task = plan["tasks"][0]
    item = plan["attempts"][0]
    config = hosted._attempt_config(plan, task, item)
    task_claim = {
        "schema_version": "fleet-hosted-opencode-task-claim-v1",
        "plan_sha256": plan["plan_sha256"],
        "rank": int(task["rank"]),
        "source_rank": int(task["source_rank"]),
        "task_key": task["task"]["key"],
        "task_version_id": task["task"]["version_id"],
        "run_ids": [item["run_id"]],
    }
    task_claim["claim_sha256"] = legacy.digest_without(task_claim, "claim_sha256")
    claim = {
        "schema_version": "fleet-hosted-opencode-attempt-claim-v1",
        "plan_sha256": plan["plan_sha256"],
        "task_claim_sha256": task_claim["claim_sha256"],
        "run_id": item["run_id"],
        "rank": int(task["rank"]),
        "source_rank": int(item["source_rank"]),
        "attempt": 1,
        "network": item["network"],
        "task_key": task["task"]["key"],
        "task_version_id": task["task"]["version_id"],
        "config_sha256": config["config_sha256"],
    }
    return (
        legacy.digest_without(claim, "claim_sha256"),
        config["config_sha256"],
    )


def _validated_result(result: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    common = {"accepted", "quarantined", "claim_sha256", "attempt_config_sha256"}
    accepted_only = {
        "acceptance_receipt_sha256",
        "session_id",
        "verifier_execution_id",
        "session_ingest_completed",
        "cleanup_completed",
    }
    accepted = result.get("accepted") is True
    quarantined = result.get("quarantined") is True
    expected_claim, expected_config = _expected_cell_bindings(plan)
    if (
        accepted == quarantined
        or result.get("claim_sha256") != expected_claim
        or result.get("attempt_config_sha256") != expected_config
    ):
        raise ValueError("generation-2 v3 result is invalid")
    if accepted:
        if set(result) != common | accepted_only:
            raise ValueError("generation-2 v3 result is invalid")
        if (
            not SHA_RE.fullmatch(str(result.get("acceptance_receipt_sha256")))
            or not legacy._is_uuid(result.get("session_id"))
            or not legacy._is_uuid(result.get("verifier_execution_id"))
            or result.get("session_ingest_completed") is not True
            or result.get("cleanup_completed") is not True
        ):
            raise ValueError("generation-2 v3 result is invalid")
    elif set(result) != common:
        raise ValueError("generation-2 v3 result is invalid")
    return copy.deepcopy(result)


def terminal_receipt(
    spec: dict[str, Any],
    plan: dict[str, Any],
    generation_claim: dict[str, Any],
    result: dict[str, Any],
    *,
    root: Path,
    terminal_at_utc: str | None = None,
) -> dict[str, Any]:
    job_uid = os.environ.get("JOB_UID", "")
    pod_uid = os.environ.get("POD_UID", "")
    validate_claim(
        generation_claim,
        spec,
        plan,
        job_uid=job_uid,
        pod_uid=pod_uid,
        root=root,
    )
    validated_result = _validated_result(result, plan)
    terminal = {
        "schema_version": TERMINAL_SCHEMA,
        "generation2_spec_sha256": spec["generation2_spec_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "cell_id": spec["statistical_cell"]["cell_id"],
        "execution_id": spec["execution"]["execution_id"],
        "execution_generation": 2,
        "generation_claim_receipt_sha256": generation_claim["receipt_sha256"],
        "job_uid": job_uid,
        "pod_uid": pod_uid,
        "terminal_at_utc": terminal_at_utc or datetime.now(UTC).isoformat(),
        "result": validated_result,
        "retry_allowed": False,
        "bulk_release_authorized": False,
        "scores_included": False,
        "prompts_or_traces_included": False,
    }
    terminal["receipt_sha256"] = digest(terminal, "receipt_sha256")
    validate_terminal(terminal, spec, plan, generation_claim, root=root)
    return terminal


def validate_terminal(
    terminal: dict[str, Any],
    spec: dict[str, Any],
    plan: dict[str, Any],
    generation_claim: dict[str, Any],
    *,
    root: Path,
) -> None:
    job_uid = generation_claim.get("job_uid", "")
    pod_uid = generation_claim.get("pod_uid", "")
    validate_claim(
        generation_claim,
        spec,
        plan,
        job_uid=job_uid,
        pod_uid=pod_uid,
        root=root,
    )
    result = _validated_result(terminal.get("result", {}), plan)
    expected = {
        "schema_version": TERMINAL_SCHEMA,
        "generation2_spec_sha256": spec["generation2_spec_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "cell_id": spec["statistical_cell"]["cell_id"],
        "execution_id": spec["execution"]["execution_id"],
        "execution_generation": 2,
        "generation_claim_receipt_sha256": generation_claim["receipt_sha256"],
        "job_uid": job_uid,
        "pod_uid": pod_uid,
        "terminal_at_utc": terminal.get("terminal_at_utc"),
        "result": result,
        "retry_allowed": False,
        "bulk_release_authorized": False,
        "scores_included": False,
        "prompts_or_traces_included": False,
    }
    if (
        not _canonical_utc(terminal.get("terminal_at_utc"))
        or terminal.get("receipt_sha256") != digest(terminal, "receipt_sha256")
        or {key: value for key, value in terminal.items() if key != "receipt_sha256"}
        != expected
    ):
        raise ValueError("generation-2 v3 terminal is not authoritative")


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "validate-spec",
            "validate-held",
            "validate-release",
            "validate-terminal",
            "preview",
        ),
    )
    parser.add_argument("--spec", action="append", type=Path, default=[])
    parser.add_argument("--release", type=Path)
    parser.add_argument("--claim", type=Path)
    parser.add_argument("--terminal", type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--package-commit")
    parser.add_argument("--authorized-at-utc")
    parser.add_argument("--authorization-statement")
    args = parser.parse_args()
    specs = [load(path) for path in args.spec]
    if args.command == "validate-spec":
        if len(specs) != 1:
            parser.error("validate-spec requires exactly one --spec")
        validate_spec(specs[0], args.repo)
        return 0
    if args.command == "validate-release":
        if not all(
            (
                args.release,
                len(specs) == 1,
                args.package_commit,
                args.authorized_at_utc,
                args.authorization_statement,
            )
        ):
            parser.error("validate-release requires exact release authority arguments")
        validate_release(
            load(args.release),
            specs[0],
            args.repo,
            args.package_commit,
            authorized_at_utc=args.authorized_at_utc,
            authorization_statement=args.authorization_statement,
        )
        return 0
    if args.command == "validate-terminal":
        if len(specs) != 1 or not args.claim or not args.terminal:
            parser.error("validate-terminal requires one spec, claim, and terminal")
        plan = validate_spec(specs[0], args.repo)
        validate_terminal(
            load(args.terminal), specs[0], plan, load(args.claim), root=args.repo
        )
        return 0
    if not args.release or len(specs) != 2:
        parser.error(f"{args.command} requires two specs and a held release")
    validate_held(load(args.release), specs, args.repo)
    if args.command == "preview":
        print(
            json.dumps(
                {
                    "ok": True,
                    "status": "HELD",
                    "execution_generation": 2,
                    "statistical_cells": 2,
                    "launch_authorized": False,
                    "cluster_objects_created": False,
                    "release_claim_terminal_contract_v3": True,
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
