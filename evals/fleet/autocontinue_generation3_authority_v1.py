"""Authority, release, claim, and terminal chain for generation-3 canaries."""

from __future__ import annotations

import argparse
import fcntl
import os
import stat
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.fleet import autocontinue_canary_controller as legacy
from evals.fleet import autocontinue_canary_hosted_runtime as hosted_runtime
from evals.fleet import autocontinue_generation2_canary_v3 as generation2
from evals.fleet import autocontinue_generation3_canary as generation3
from evals.fleet import hosted_sweep_controller as hosted
from evals.fleet import self_hosted

AUTH_SCHEMA = "fleet-opencode-autocontinue-generation3-root-authorization-v1"
RELEASE_SCHEMA = "fleet-opencode-autocontinue-generation3-scoring-release-v1"
CLAIM_SCHEMA = "fleet-statistical-cell-execution-claim-v4"
TERMINAL_SCHEMA = "fleet-opencode-autocontinue-generation3-terminal-v2"

AUTH_PATH = (
    "docs/evidence/qwen38-study/"
    "2026-09-04-opencode-autocontinue-generation3-root-authorization-v1.json"
)
SEMANTIC_HELD_PATH = (
    "docs/evidence/qwen38-study/"
    "2026-09-04-opencode-autocontinue-generation3-executable-package-held-v3.json"
)
MODULE_PATH = "evals/fleet/autocontinue_generation3_authority_v1.py"
PACKAGE_MODULE_PATH = "evals/fleet/autocontinue_generation3_authority_package_v1.py"
MANIFEST_PATH = (
    "evals/fleet/cluster/opencode-autocontinue-generation3-authority-held-v4.yaml"
)
RUN_PATH = "evals/fleet/scripts/run_opencode_autocontinue_generation3_authority_v4.sh"
SUBMIT_PATH = (
    "evals/fleet/scripts/submit_opencode_autocontinue_generation3_authority_v4.sh"
)
RELEASE_PATHS = {
    "qwen3.8-27b": (
        "docs/evidence/qwen38-study/"
        "2026-09-04-qwen38-autocontinue-generation3-scoring-release-v1.json"
    ),
    "glm-5.3": (
        "docs/evidence/qwen38-study/"
        "2026-09-04-glm53-autocontinue-generation3-scoring-release-v1.json"
    ),
}
AUTHORIZED_AT = "2026-09-05T05:39:51Z"
STATEMENTS = {
    "qwen3.8-27b": (
        "Authorize exactly one create-once corrected-treatment hosted Qwen3.8 27B "
        "generation-3 scored canary cell under the frozen OpenCode 1.18.27 "
        "auto-compaction-and-continuation protocol, superseding only the terminal "
        "generation-2 pre-Pod admission failure; use fleet-serve-low with "
        "preemptionPolicy Never; no dedicated GPU serving and no bulk release are "
        "authorized."
    ),
    "glm-5.3": (
        "Authorize exactly one create-once corrected-treatment hosted GLM5.3 "
        "generation-3 scored canary cell under the frozen OpenCode 1.18.27 "
        "auto-compaction-and-continuation protocol, superseding only the terminal "
        "generation-2 pre-Pod admission failure; use fleet-serve-low with "
        "preemptionPolicy Never; no dedicated GPU serving and no bulk release are "
        "authorized."
    ),
}


def load(path: Path) -> dict[str, Any]:
    return generation3.load(path)


def digest(value: dict[str, Any], field: str) -> str:
    return generation3.digest(value, field)


def file_sha256(path: Path) -> str:
    return generation3.file_sha256(path)


def specs(root: Path) -> list[dict[str, Any]]:
    return [
        load(root / generation3.G3_SPEC_PATHS[model])
        for model in ("qwen3.8-27b", "glm-5.3")
    ]


def validate_root_authorization(
    receipt: dict[str, Any], root: Path
) -> list[dict[str, Any]]:
    selected = specs(root)
    plans = [generation3.validate_spec(spec, root) for spec in selected]
    expected = {
        "schema_version": AUTH_SCHEMA,
        "append_only": True,
        "status": "AUTHORIZED",
        "author": "/root",
        "scope": {
            "execution_generation": 3,
            "create_once": True,
            "hosted_only": True,
            "scored_launch_authorized": True,
            "dedicated_serving_authorized": False,
            "bulk_release_authorized": False,
            "release_receipts_present": False,
            "cluster_mutation_authorized_by_this_package": False,
        },
        "models": [
            {
                "model": spec["model"],
                "cell_id": spec["statistical_cell"]["cell_id"],
                "generation3_spec_sha256": spec["generation3_spec_sha256"],
                "rendered_plan_sha256": plan["plan_sha256"],
                "execution_id": spec["execution"]["execution_id"],
                "authorized_at_utc": AUTHORIZED_AT,
                "statement": STATEMENTS[spec["model"]],
            }
            for spec, plan in zip(selected, plans, strict=True)
        ],
        "privacy": {
            "prompts_traces_flags_or_scores_included": False,
            "credentials_included": False,
        },
    }
    if (
        receipt.get("receipt_sha256") != digest(receipt, "receipt_sha256")
        or {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        != expected
    ):
        raise ValueError("generation-3 root authorization drifted")
    return plans


def authority_binding(authority: dict[str, Any], root: Path) -> dict[str, Any]:
    validate_root_authorization(authority, root)
    return {
        "path": AUTH_PATH,
        "file_sha256": file_sha256(root / AUTH_PATH),
        "receipt_sha256": authority["receipt_sha256"],
    }


def semantic_held_binding(root: Path) -> dict[str, Any]:
    from evals.fleet import autocontinue_generation3_executable_package_v2 as semantic

    held = load(root / SEMANTIC_HELD_PATH)
    semantic.validate_held(held, root)
    return {
        "path": SEMANTIC_HELD_PATH,
        "file_sha256": file_sha256(root / SEMANTIC_HELD_PATH),
        "receipt_sha256": held["receipt_sha256"],
    }


def implementation(
    root: Path, spec: dict[str, Any], built: dict[str, Any] | None = None
) -> dict[str, Any]:
    from evals.fleet import autocontinue_generation3_authority_package_v1 as package

    expected = generation3.EXPECTED[spec["model"]]
    package_value = built or package.build_package(root)
    return {
        "authority_module_path": MODULE_PATH,
        "authority_module_sha256": file_sha256(root / MODULE_PATH),
        "package_module_path": PACKAGE_MODULE_PATH,
        "package_module_sha256": file_sha256(root / PACKAGE_MODULE_PATH),
        "manifest_path": MANIFEST_PATH,
        "manifest_sha256": file_sha256(root / MANIFEST_PATH),
        "run_path": RUN_PATH,
        "run_sha256": file_sha256(root / RUN_PATH),
        "submit_path": SUBMIT_PATH,
        "submit_sha256": file_sha256(root / SUBMIT_PATH),
        "job_name": expected["job_name"],
        "configmap_name": expected["configmap_name"],
        "intent_configmap_name": generation3.INTENT_NAME,
        "sfs_root": f"/mnt/sfs/jobs/{expected['job_name']}",
        "split_package_aggregate_sha256": package_value["aggregate_sha256"],
        "model_aggregate_sha256": package_value["model_manifests"][spec["model"]][
            "aggregate_sha256"
        ],
    }


def validate_release(
    release: dict[str, Any],
    spec: dict[str, Any],
    authority: dict[str, Any],
    root: Path,
    package_commit: str,
) -> None:
    from evals.fleet import autocontinue_generation3_authority_package_v1 as package

    plans = validate_root_authorization(authority, root)
    plan = generation3.validate_spec(spec, root)
    if plan not in plans or not generation2.COMMIT_RE.fullmatch(package_commit):
        raise ValueError("generation-3 release package is invalid")
    entry = next(row for row in authority["models"] if row["model"] == spec["model"])
    built = package.build_package(root)
    expected = {
        "schema_version": RELEASE_SCHEMA,
        "append_only": True,
        "status": "RELEASED",
        "released_at_utc": entry["authorized_at_utc"],
        "model": spec["model"],
        "generation3_spec_sha256": spec["generation3_spec_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "cell_id": spec["statistical_cell"]["cell_id"],
        "execution_id": spec["execution"]["execution_id"],
        "package_commit": package_commit,
        "root_authorization": authority_binding(authority, root),
        "semantic_held_authority": semantic_held_binding(root),
        "split_configmap_package": {
            "schema_version": package.SCHEMA,
            "aggregate_sha256": built["aggregate_sha256"],
            "model_aggregate_sha256": built["model_manifests"][spec["model"]][
                "aggregate_sha256"
            ],
            "objects": sorted(built["configmaps"]),
            "object_json_bytes": built["object_json_bytes"],
        },
        "implementation": implementation(root, spec, built),
        "authorization": {
            "launch_authorized": True,
            "create_once": True,
            "execution_generation": 3,
            "hosted_only": True,
            "priority_class": "fleet-serve-low",
            "preemption_policy": "Never",
            "bulk_release_authorized": False,
            "dedicated_serving_authorized": False,
            "author": "/root",
            "authorized_at_utc": entry["authorized_at_utc"],
            "statement": entry["statement"],
        },
        "route_and_inventory": {
            "fresh_fleet_team_and_hosted_models_required": True,
            "fresh_exact_treatment_session_inventory_required": True,
            "generation2_tombstones_must_match": True,
            "generation3_claim_must_be_absent": True,
            "generation3_job_configmap_and_sfs_root_must_be_absent": True,
            "dedicated_serving_must_remain_user_stopped": True,
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
        raise ValueError("generation-3 scoring release is not authoritative")


def release_binding(
    release: dict[str, Any], release_file_sha256: str, package_commit: str
) -> dict[str, Any]:
    if not generation2.SHA_RE.fullmatch(release_file_sha256):
        raise ValueError("generation-3 release file digest is invalid")
    return {
        "schema_version": RELEASE_SCHEMA,
        "receipt_sha256": release["receipt_sha256"],
        "file_sha256": release_file_sha256,
        "package_commit": package_commit,
    }


def _claim_receipt(
    spec: dict[str, Any],
    plan: dict[str, Any],
    release: dict[str, Any],
    release_file_sha256: str,
    authority: dict[str, Any],
    root: Path,
    package_commit: str,
    *,
    claimed_at_utc: str,
    job_uid: str | None = None,
    pod_uid: str | None = None,
) -> dict[str, Any]:
    receipt = {
        "schema_version": CLAIM_SCHEMA,
        "generation3_spec_sha256": spec["generation3_spec_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "cell_id": spec["statistical_cell"]["cell_id"],
        "execution_id": spec["execution"]["execution_id"],
        "execution_generation": 3,
        "run_id": plan["attempts"][0]["run_id"],
        "job_uid": job_uid or os.environ.get("JOB_UID"),
        "pod_uid": pod_uid or os.environ.get("POD_UID"),
        "claimed_at_utc": claimed_at_utc,
        "scoring_release": release_binding(
            release, release_file_sha256, package_commit
        ),
        "root_authorization": authority_binding(authority, root),
        "semantic_held_authority": semantic_held_binding(root),
        "generation2_tombstone_receipt_sha256": generation3.TOMBSTONE_SHA,
        "prior_claims_preserved": True,
        "immutable": True,
        "automatic_retry": False,
        "scores_included": False,
        "prompts_or_traces_included": False,
    }
    receipt["receipt_sha256"] = digest(receipt, "receipt_sha256")
    return receipt


def validate_claim(
    claim: dict[str, Any],
    spec: dict[str, Any],
    plan: dict[str, Any],
    release: dict[str, Any],
    release_file_sha256: str,
    authority: dict[str, Any],
    root: Path,
    package_commit: str,
) -> None:
    if plan != generation3.validate_spec(spec, root):
        raise ValueError("generation-3 authority claim spec-plan chain drifted")
    validate_release(release, spec, authority, root, package_commit)
    expected = _claim_receipt(
        spec,
        plan,
        release,
        release_file_sha256,
        authority,
        root,
        package_commit,
        claimed_at_utc=claim.get("claimed_at_utc", ""),
        job_uid=claim.get("job_uid"),
        pod_uid=claim.get("pod_uid"),
    )
    if (
        not legacy._is_uuid(claim.get("job_uid"))
        or not legacy._is_uuid(claim.get("pod_uid"))
        or not generation3._canonical_utc(claim.get("claimed_at_utc"))
        or claim != expected
    ):
        raise ValueError("generation-3 authority claim is not authoritative")


def claim_execution(
    spec: dict[str, Any],
    plan: dict[str, Any],
    release: dict[str, Any],
    release_file_sha256: str,
    authority: dict[str, Any],
    root: Path,
    package_commit: str,
    *,
    claim_root: Path | None = None,
) -> dict[str, Any]:
    validate_release(release, spec, authority, root, package_commit)
    if plan != generation3.validate_spec(spec, root):
        raise ValueError("generation-3 authority claim spec-plan chain drifted")
    if not legacy._is_uuid(os.environ.get("JOB_UID")) or not legacy._is_uuid(
        os.environ.get("POD_UID")
    ):
        raise RuntimeError("generation-3 authority claim requires downward API UIDs")
    generation3.validate_preclaim_tombstone(spec, root)
    claim_dir = claim_root or Path(generation3.CLAIM_ROOT)
    root_fd = legacy._open_directory_nofollow(claim_dir)
    try:
        lock_fd = legacy._open_claim_lock(root_fd)
        with os.fdopen(lock_fd, "a+b") as lock:
            if not stat.S_ISREG(os.fstat(lock.fileno()).st_mode):
                raise RuntimeError("generation-3 authority claim lock is not regular")
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            name = spec["execution"]["execution_id"].removeprefix("sha256:") + ".json"
            try:
                os.stat(name, dir_fd=root_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise RuntimeError("generation-3 execution is already claimed")
            receipt = _claim_receipt(
                spec,
                plan,
                release,
                release_file_sha256,
                authority,
                root,
                package_commit,
                claimed_at_utc=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            )
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(name, flags, 0o600, dir_fd=root_fd)
            try:
                if not stat.S_ISREG(os.fstat(fd).st_mode):
                    raise RuntimeError("generation-3 authority claim path is unsafe")
                with os.fdopen(fd, "wb") as stored:
                    fd = -1
                    stored.write(self_hosted.canonical_json(receipt) + b"\n")
                    stored.flush()
                    os.fsync(stored.fileno())
            finally:
                if fd >= 0:
                    os.close(fd)
            os.fsync(root_fd)
            validate_claim(
                receipt,
                spec,
                plan,
                release,
                release_file_sha256,
                authority,
                root,
                package_commit,
            )
            return receipt
    finally:
        os.close(root_fd)


def _terminal_receipt(
    spec: dict[str, Any],
    plan: dict[str, Any],
    claim: dict[str, Any],
    result: dict[str, Any],
    release: dict[str, Any],
    release_file_sha256: str,
    authority: dict[str, Any],
    root: Path,
    package_commit: str,
    *,
    terminal_at_utc: str | None = None,
) -> dict[str, Any]:
    validate_claim(
        claim,
        spec,
        plan,
        release,
        release_file_sha256,
        authority,
        root,
        package_commit,
    )
    receipt = {
        "schema_version": TERMINAL_SCHEMA,
        "generation3_spec_sha256": spec["generation3_spec_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "cell_id": spec["statistical_cell"]["cell_id"],
        "execution_id": spec["execution"]["execution_id"],
        "execution_generation": 3,
        "generation_claim_receipt_sha256": claim["receipt_sha256"],
        "job_uid": claim["job_uid"],
        "pod_uid": claim["pod_uid"],
        "terminal_at_utc": terminal_at_utc
        or datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "scoring_release": release_binding(
            release, release_file_sha256, package_commit
        ),
        "root_authorization": authority_binding(authority, root),
        "semantic_held_authority": semantic_held_binding(root),
        "result": generation2._validated_result(result, plan),
        "retry_allowed": False,
        "bulk_release_authorized": False,
        "scores_included": False,
        "prompts_or_traces_included": False,
    }
    receipt["receipt_sha256"] = digest(receipt, "receipt_sha256")
    return receipt


def terminal_receipt(
    spec: dict[str, Any],
    plan: dict[str, Any],
    claim: dict[str, Any],
    result: dict[str, Any],
    release: dict[str, Any],
    release_file_sha256: str,
    authority: dict[str, Any],
    root: Path,
    package_commit: str,
    *,
    terminal_at_utc: str | None = None,
) -> dict[str, Any]:
    receipt = _terminal_receipt(
        spec,
        plan,
        claim,
        result,
        release,
        release_file_sha256,
        authority,
        root,
        package_commit,
        terminal_at_utc=terminal_at_utc,
    )
    validate_terminal(
        receipt,
        spec,
        plan,
        claim,
        release,
        release_file_sha256,
        authority,
        root,
        package_commit,
    )
    return receipt


def validate_terminal(
    receipt: dict[str, Any],
    spec: dict[str, Any],
    plan: dict[str, Any],
    claim: dict[str, Any],
    release: dict[str, Any],
    release_file_sha256: str,
    authority: dict[str, Any],
    root: Path,
    package_commit: str,
) -> None:
    expected = _terminal_receipt(
        spec,
        plan,
        claim,
        receipt.get("result", {}),
        release,
        release_file_sha256,
        authority,
        root,
        package_commit,
        terminal_at_utc=receipt.get("terminal_at_utc"),
    )
    if receipt != expected:
        raise ValueError("generation-3 authority terminal is not authoritative")


def run(
    spec: dict[str, Any],
    release: dict[str, Any],
    release_file_sha256: str,
    authority: dict[str, Any],
    launch_route: dict[str, Any],
    out: Path,
    proxy: Path,
    root: Path,
    package_commit: str,
) -> dict[str, Any]:
    plan = generation3.validate_spec(spec, root)
    validate_release(release, spec, authority, root, package_commit)
    jobs = tuple(generation3.EXPECTED[model]["job_name"] for model in generation3.EXPECTED)
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
        live = hosted_runtime.observe_live_route(
            key,
            caller=hosted_runtime.RUNTIME_CALLER,
            job_uid=os.environ.get("JOB_UID"),
            pod_uid=os.environ.get("POD_UID"),
            candidate_scored_job_names=jobs,
        )
        hosted_runtime.validate_live_route(
            live,
            caller=hosted_runtime.RUNTIME_CALLER,
            maximum_age_seconds=30,
            job_uid=os.environ.get("JOB_UID"),
            pod_uid=os.environ.get("POD_UID"),
            candidate_scored_job_names=jobs,
        )
        generation3.validate_preclaim_tombstone(spec, root)
        hosted._validate_plan_identity_absence(plan, out)
        with hosted._client(key) as client:
            account = self_hosted._request(client, "GET", "/v1/account")
        if (
            account.get("team_name") != "fleet"
            or account.get("team_id") != self_hosted.FLEET_TEAM_ID
        ):
            raise RuntimeError("FLEET_API_KEY is not scoped to Fleet")
        hosted._validate_inventory_for_task(plan, out, plan["tasks"][0], key)
        claim = claim_execution(
            spec,
            plan,
            release,
            release_file_sha256,
            authority,
            root,
            package_commit,
        )
        out.mkdir(mode=0o700)
        for name in ("attempts", "claims", "task-claims", "task-results", "quarantine", "ramps"):
            (out / name).mkdir(mode=0o700)
        self_hosted.write_json_once(out / "PLAN.json", plan)
        self_hosted.write_json_once(out / "SCORING-RELEASE.json", release)
        self_hosted.write_json_once(out / "ROOT-AUTHORIZATION.json", authority)
        result = legacy._run_cell(plan, out, proxy, key)
        terminal = terminal_receipt(
            spec,
            plan,
            claim,
            result,
            release,
            release_file_sha256,
            authority,
            root,
            package_commit,
        )
        self_hosted.write_json_once(out / "CANARY-TERMINAL.json", terminal)
        return terminal


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("validate-authority", "validate-release", "run")
    )
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--release", type=Path)
    parser.add_argument("--launch-route", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--proxy", type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--package-commit")
    args = parser.parse_args()
    authority = load(args.authority)
    if args.command == "validate-authority":
        validate_root_authorization(authority, args.repo)
        return 0
    if not args.spec or not args.release or not args.package_commit:
        parser.error(f"{args.command} requires spec, release, and package commit")
    spec = load(args.spec)
    release = load(args.release)
    validate_release(release, spec, authority, args.repo, args.package_commit)
    if args.command == "run":
        if not args.launch_route or not args.out_dir or not args.proxy:
            parser.error("run requires launch route, output directory, and proxy")
        run(
            spec,
            release,
            file_sha256(args.release),
            authority,
            load(args.launch_route),
            args.out_dir,
            args.proxy,
            args.repo,
            args.package_commit,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
