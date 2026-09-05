"""Self-digesting runtime-plan wrapper for held generation-2 canaries.

The immutable v1 held package remains historical evidence.  This successor
separates the digest of the generation-2 specification from the digest of the
fully rendered runtime plan, then carries both through every mutable boundary.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
import stat
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.fleet import autocontinue_canary_controller as legacy
from evals.fleet import autocontinue_canary_hosted_runtime as hosted_runtime
from evals.fleet import autocontinue_generation2_canary as held_v1
from evals.fleet import hosted_sweep_controller as hosted
from evals.fleet import self_hosted

SPEC_SCHEMA = "fleet-opencode-autocontinue-generation2-canary-spec-v2"
HELD_SCHEMA = "fleet-opencode-autocontinue-generation2-canary-held-release-v2"
RELEASE_SCHEMA = "fleet-opencode-autocontinue-generation2-canary-scoring-release-v2"
MODULE_PATH = "evals/fleet/autocontinue_generation2_canary_v2.py"
MANIFEST_PATH = "evals/fleet/cluster/opencode-autocontinue-generation2-canaries-v2.yaml"
RUN_PATH = "evals/fleet/scripts/run_opencode_autocontinue_generation2_canary_v2.sh"
SUBMIT_PATH = "evals/fleet/scripts/submit_opencode_autocontinue_generation2_canaries_v2.sh"
V1_HELD_PATH = (
    "docs/evidence/qwen38-study/"
    "2026-09-04-opencode-autocontinue-generation2-canaries-held-v1.json"
)

V1_SPEC_PATHS = {
    "qwen3.8-27b": "evals/fleet/configs/qwen38-opencode-autocontinue-canary-generation2-v1.json",
    "glm-5.3": "evals/fleet/configs/glm53-opencode-autocontinue-canary-generation2-v1.json",
}


def load(path: Path) -> dict[str, Any]:
    return held_v1.load(path)


def digest(value: dict[str, Any], field: str) -> str:
    return held_v1.digest(value, field)


def file_sha256(path: Path) -> str:
    return held_v1.file_sha256(path)


def _v1_spec(spec: dict[str, Any], root: Path) -> dict[str, Any]:
    model = spec.get("model")
    if model not in V1_SPEC_PATHS:
        raise ValueError("generation-2 v2 model is unsupported")
    path = V1_SPEC_PATHS[model]
    reference = spec.get("predecessor_generation2_spec")
    if (
        not isinstance(reference, dict)
        or set(reference) != {"path", "generation2_spec_sha256"}
        or reference.get("path") != path
    ):
        raise ValueError("generation-2 v1 specification reference drifted")
    old = load(root / path)
    held_v1.validate_plan(old, root)
    if reference["generation2_spec_sha256"] != old["plan_sha256"]:
        raise ValueError("generation-2 v1 specification digest drifted")
    return old


def validate_spec(spec: dict[str, Any], root: Path) -> dict[str, Any]:
    if (
        set(spec)
        != {
            "schema_version",
            "model",
            "statistical_cell",
            "execution",
            "predecessor_generation2_spec",
            "rendered_plan_sha256",
            "launch_authorized",
            "generation2_spec_sha256",
        }
        or spec.get("schema_version") != SPEC_SCHEMA
        or spec.get("launch_authorized") is not False
        or spec.get("generation2_spec_sha256")
        != digest(spec, "generation2_spec_sha256")
        or not held_v1.exact.SHA256_RE.fullmatch(str(spec.get("rendered_plan_sha256")))
    ):
        raise ValueError("generation-2 v2 specification drifted")

    old = _v1_spec(spec, root)
    if (
        spec.get("model") != old["model"]
        or spec.get("statistical_cell") != old["statistical_cell"]
        or spec.get("execution") != old["execution"]
    ):
        raise ValueError("generation-2 v2 statistical identity drifted")

    plan = copy.deepcopy(held_v1.validate_plan(old, root))
    plan["plan_sha256"] = digest(plan, "plan_sha256")
    if plan["plan_sha256"] != spec["rendered_plan_sha256"]:
        raise ValueError("generation-2 rendered plan digest drifted")
    if plan["plan_sha256"] != digest(plan, "plan_sha256"):
        raise ValueError("generation-2 rendered plan is not self-digesting")
    return plan


def validate_held(release: dict[str, Any], specs: list[dict[str, Any]], root: Path) -> None:
    plans = [validate_spec(spec, root) for spec in specs]
    prior = load(root / V1_HELD_PATH)
    old_specs = [_v1_spec(spec, root) for spec in specs]
    held_v1.validate_held(prior, old_specs, root)
    if (
        set(release)
        != {
            "schema_version",
            "append_only",
            "status",
            "observed_at_utc",
            "supersedes_held_receipt",
            "generation2_spec_sha256s",
            "rendered_plan_sha256s",
            "jobs",
            "execution_generation",
            "statistical_cells",
            "implementation",
            "launch_authorized",
            "bulk_release_authorized",
            "dedicated_serving_authorized",
            "remaining_gates",
            "privacy",
            "receipt_sha256",
        }
        or release.get("schema_version") != HELD_SCHEMA
        or release.get("append_only") is not True
        or release.get("status") != "HELD"
        or not isinstance(release.get("observed_at_utc"), str)
        or not release.get("observed_at_utc")
        or release.get("launch_authorized") is not False
        or release.get("bulk_release_authorized") is not False
        or release.get("dedicated_serving_authorized") is not False
        or release.get("supersedes_held_receipt")
        != {"path": V1_HELD_PATH, "receipt_sha256": prior["receipt_sha256"]}
        or release.get("generation2_spec_sha256s")
        != [spec["generation2_spec_sha256"] for spec in specs]
        or release.get("rendered_plan_sha256s")
        != [plan["plan_sha256"] for plan in plans]
        or release.get("jobs")
        != [held_v1.EXPECTED[spec["model"]]["job_name"] for spec in specs]
        or release.get("execution_generation") != 2
        or release.get("statistical_cells") != 2
        or release.get("implementation")
        != {
            "module_path": MODULE_PATH,
            "module_sha256": file_sha256(root / MODULE_PATH),
            "manifest_path": MANIFEST_PATH,
            "manifest_sha256": file_sha256(root / MANIFEST_PATH),
            "run_path": RUN_PATH,
            "run_sha256": file_sha256(root / RUN_PATH),
            "submit_path": SUBMIT_PATH,
            "submit_sha256": file_sha256(root / SUBMIT_PATH),
        }
        or release.get("remaining_gates")
        != [
            "independent_package_audit",
            "fresh_hosted_route_and_duplicate_inventory",
            "append_only_model_specific_scoring_releases",
            "explicit_root_launch_authorization",
        ]
        or release.get("privacy")
        != {"prompts_traces_flags_or_scores_included": False, "credentials_included": False}
        or release.get("receipt_sha256") != digest(release, "receipt_sha256")
    ):
        raise ValueError("generation-2 v2 held release drifted")


def validate_release(
    release: dict[str, Any], spec: dict[str, Any], root: Path, package_commit: str
) -> None:
    plan = validate_spec(spec, root)
    expected = held_v1.EXPECTED[spec["model"]]
    if (
        set(release)
        != {
            "schema_version",
            "append_only",
            "status",
            "released_at_utc",
            "generation2_spec_sha256",
            "plan_sha256",
            "cell_id",
            "execution_id",
            "incident_receipt_sha256",
            "tombstone_bundle_receipt_sha256",
            "package_commit",
            "implementation",
            "authorization",
            "route_and_inventory",
            "privacy",
            "receipt_sha256",
        }
        or release.get("schema_version") != RELEASE_SCHEMA
        or release.get("append_only") is not True
        or release.get("status") != "RELEASED"
        or not isinstance(release.get("released_at_utc"), str)
        or not release.get("released_at_utc")
        or release.get("generation2_spec_sha256") != spec["generation2_spec_sha256"]
        or release.get("plan_sha256") != plan["plan_sha256"]
        or release.get("cell_id") != spec["statistical_cell"]["cell_id"]
        or release.get("execution_id") != spec["execution"]["execution_id"]
        or release.get("incident_receipt_sha256") != held_v1.INCIDENT_SHA
        or release.get("tombstone_bundle_receipt_sha256")
        != held_v1._tombstones(root)["receipt_sha256"]
        or release.get("package_commit") != package_commit
        or release.get("implementation", {}).get("job_name") != expected["job_name"]
        or release.get("implementation", {}).get("configmap_name")
        != expected["configmap_name"]
    ):
        raise ValueError("generation-2 v2 scoring release is not authoritative")
    if (
        release["implementation"].get("sfs_root")
        != held_v1.load(root / V1_SPEC_PATHS[spec["model"]])["identities"]["sfs_root"]
        or release.get("authorization")
        != {
            "launch_authorized": True,
            "create_once": True,
            "execution_generation": 2,
            "bulk_release_authorized": False,
            "dedicated_serving_authorized": False,
            "author": "/root",
            "statement": release.get("authorization", {}).get("statement"),
        }
        or not release.get("authorization", {}).get("statement")
        or release.get("route_and_inventory")
        != {
            "fresh_authenticated_hosted_route_required": True,
            "fresh_exact_treatment_session_inventory_required": True,
            "old_claims_must_exist_and_match": True,
            "generation_2_claim_must_be_absent": True,
        }
        or release.get("privacy")
        != {"prompts_traces_flags_or_scores_included": False, "credentials_included": False}
        or release.get("receipt_sha256") != digest(release, "receipt_sha256")
    ):
        raise ValueError("generation-2 v2 scoring release is not authoritative")


def claim_execution_generation(
    spec: dict[str, Any],
    plan: dict[str, Any],
    claim_root: Path | None = None,
    repo: Path | None = None,
) -> dict[str, Any]:
    if not legacy._is_uuid(os.environ.get("JOB_UID")) or not legacy._is_uuid(
        os.environ.get("POD_UID")
    ):
        raise RuntimeError("generation claim requires downward API UIDs")
    repository = repo or Path.cwd()
    old = held_v1.load(repository / V1_SPEC_PATHS[spec["model"]])
    root = claim_root or Path(old["identities"]["generation_claim_root"])
    root_fd = legacy._open_directory_nofollow(root)
    try:
        lock_fd = legacy._open_claim_lock(root_fd)
        with os.fdopen(lock_fd, "a+b") as lock:
            if not stat.S_ISREG(os.fstat(lock.fileno()).st_mode):
                raise RuntimeError("generation claim lock is not regular")
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            name = spec["execution"]["execution_id"].removeprefix("sha256:") + ".json"
            try:
                existing = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
            except FileNotFoundError:
                existing = None
            if existing is not None:
                raise RuntimeError("execution generation is already claimed")
            receipt = {
                "schema_version": "fleet-statistical-cell-execution-claim-v2",
                "generation2_spec_sha256": spec["generation2_spec_sha256"],
                "plan_sha256": plan["plan_sha256"],
                "cell_id": spec["statistical_cell"]["cell_id"],
                "execution_id": spec["execution"]["execution_id"],
                "execution_generation": 2,
                "run_id": plan["attempts"][0]["run_id"],
                "job_uid": os.environ["JOB_UID"],
                "pod_uid": os.environ["POD_UID"],
                "claimed_at_utc": datetime.now(UTC).isoformat(),
                "generation_1_claims_preserved": True,
                "immutable": True,
                "automatic_retry": False,
                "scores_included": False,
                "prompts_or_traces_included": False,
            }
            receipt["receipt_sha256"] = digest(receipt, "receipt_sha256")
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            claim_fd = os.open(name, flags, 0o600, dir_fd=root_fd)
            try:
                if not stat.S_ISREG(os.fstat(claim_fd).st_mode):
                    raise RuntimeError("generation claim path is unsafe")
                payload = self_hosted.canonical_json(receipt) + b"\n"
                with os.fdopen(claim_fd, "wb") as stored:
                    claim_fd = -1
                    stored.write(payload)
                    stored.flush()
                    os.fsync(stored.fileno())
            finally:
                if claim_fd >= 0:
                    os.close(claim_fd)
            os.fsync(root_fd)
            return receipt
    finally:
        os.close(root_fd)


def run(
    spec: dict[str, Any],
    release: dict[str, Any],
    launch_route: dict[str, Any],
    out: Path,
    proxy: Path,
    repo: Path,
    package_commit: str,
) -> dict[str, Any]:
    plan = validate_spec(spec, repo)
    validate_release(release, spec, repo, package_commit)
    jobs = tuple(row["job_name"] for row in held_v1.EXPECTED.values())
    hosted_runtime.validate_live_route(
        launch_route,
        caller=hosted_runtime.LAUNCHER_CALLER,
        maximum_age_seconds=None,
        candidate_scored_job_names=jobs,
    )
    key = os.environ.get("FLEET_API_KEY")
    if not key:
        raise RuntimeError("FLEET_API_KEY is required")
    lease = plan["execution"]["endpoint_lease"]
    with legacy.endpoint_lease.acquire_endpoint_lease(
        lease_root=Path(lease["lease_root"]),
        endpoint_key=lease["endpoint_key"],
        maximum_streams=lease["maximum_streams"],
    ):
        runtime_route = hosted_runtime.observe_live_route(
            key,
            caller=hosted_runtime.RUNTIME_CALLER,
            job_uid=os.environ.get("JOB_UID"),
            pod_uid=os.environ.get("POD_UID"),
            candidate_scored_job_names=jobs,
        )
        hosted_runtime.validate_live_route(
            runtime_route,
            caller=hosted_runtime.RUNTIME_CALLER,
            maximum_age_seconds=30,
            job_uid=os.environ.get("JOB_UID"),
            pod_uid=os.environ.get("POD_UID"),
            candidate_scored_job_names=jobs,
        )
        old = _v1_spec(spec, repo)
        held_v1.validate_preserved_claims(old, repo)
        hosted._validate_plan_identity_absence(plan, out)
        with hosted._client(key) as client:
            account = self_hosted._request(client, "GET", "/v1/account")
        if (
            account.get("team_name") != "fleet"
            or account.get("team_id") != self_hosted.FLEET_TEAM_ID
        ):
            raise RuntimeError("FLEET_API_KEY is not scoped to Fleet")
        hosted._validate_inventory_for_task(plan, out, plan["tasks"][0], key)
        generation_claim = claim_execution_generation(spec, plan, repo=repo)
        out.mkdir(mode=0o700)
        for name in ("attempts", "claims", "task-claims", "task-results", "quarantine", "ramps"):
            (out / name).mkdir(mode=0o700)
        self_hosted.write_json_once(out / "PLAN.json", plan)
        self_hosted.write_json_once(out / "SCORING-RELEASE.json", release)
        result = legacy._run_cell(plan, out, proxy, key)
        terminal = terminal_receipt(spec, plan, generation_claim, result)
        self_hosted.write_json_once(out / "CANARY-TERMINAL.json", terminal)
        return terminal


def terminal_receipt(
    spec: dict[str, Any],
    plan: dict[str, Any],
    generation_claim: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    """Bind specification and rendered-plan identities in terminal evidence."""
    terminal = {
            "schema_version": "fleet-opencode-autocontinue-generation2-terminal-v2",
            "generation2_spec_sha256": spec["generation2_spec_sha256"],
            "plan_sha256": plan["plan_sha256"],
            "cell_id": spec["statistical_cell"]["cell_id"],
            "execution_id": spec["execution"]["execution_id"],
            "execution_generation": 2,
            "generation_claim_receipt_sha256": generation_claim["receipt_sha256"],
            "job_uid": os.environ["JOB_UID"],
            "pod_uid": os.environ["POD_UID"],
            "accepted": result["accepted"],
            "quarantined": result["quarantined"],
            "retry_allowed": False,
            "bulk_release_authorized": False,
            "scores_included": False,
            "prompts_or_traces_included": False,
    }
    terminal["receipt_sha256"] = digest(terminal, "receipt_sha256")
    return terminal


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("validate-spec", "validate-held", "validate-release", "preview", "run")
    )
    parser.add_argument("--spec", action="append", type=Path, default=[])
    parser.add_argument("--release", type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--package-commit")
    parser.add_argument("--launch-route", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--proxy", type=Path)
    args = parser.parse_args()
    specs = [load(path) for path in args.spec]
    if args.command == "validate-spec":
        if len(specs) != 1:
            parser.error("validate-spec requires exactly one --spec")
        validate_spec(specs[0], args.repo)
        return 0
    if args.command in {"validate-release", "run"}:
        if not args.release or len(specs) != 1 or not args.package_commit:
            parser.error(f"{args.command} requires one --spec, --release, and --package-commit")
        release = load(args.release)
        validate_release(release, specs[0], args.repo, args.package_commit)
        if args.command == "run":
            if not args.launch_route or not args.out_dir or not args.proxy:
                parser.error("run requires --launch-route, --out-dir, and --proxy")
            run(
                specs[0],
                release,
                load(args.launch_route),
                args.out_dir,
                args.proxy,
                args.repo,
                args.package_commit,
            )
        return 0
    if not args.release or len(specs) != 2:
        parser.error(f"{args.command} requires two --spec values and --release")
    release = load(args.release)
    validate_held(release, specs, args.repo)
    if args.command == "preview":
        print(
            json.dumps(
                {
                    "ok": True,
                    "status": "HELD",
                    "models": sorted(V1_SPEC_PATHS),
                    "statistical_cells": 2,
                    "execution_generation": 2,
                    "launch_authorized": False,
                    "bulk_release_authorized": False,
                    "cluster_objects_created": False,
                    "runtime_plans_self_digesting": True,
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
