"""Create-once release, claim, execution, and terminal authority for generation 6."""

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
from evals.fleet import autocontinue_generation7_canary as generation7
from evals.fleet import hosted_sweep_controller as hosted
from evals.fleet import self_hosted

AUTH_SCHEMA = "fleet-opencode-autocontinue-generation7-root-authorization-v1"
RELEASE_SCHEMA = "fleet-opencode-autocontinue-generation7-scoring-release-v1"
CLAIM_SCHEMA = "fleet-statistical-cell-execution-claim-v9"
TERMINAL_SCHEMA = "fleet-opencode-autocontinue-generation7-terminal-v1"
AUTH_PATH = (
    "docs/evidence/qwen38-study/"
    "2026-09-05-opencode-autocontinue-generation7-root-authorization-v1.json"
)
MODULE_PATH = "evals/fleet/autocontinue_generation7_authority_v1.py"
PACKAGE_MODULE_PATH = "evals/fleet/autocontinue_generation7_authority_package_v1.py"
MANIFEST_PATH = (
    "evals/fleet/cluster/opencode-autocontinue-generation7-authority-held-v1.yaml"
)
RUN_PATH = "evals/fleet/scripts/run_opencode_autocontinue_generation7_authority_v1.sh"
SUBMIT_PATH = (
    "evals/fleet/scripts/submit_opencode_autocontinue_generation7_authority_v1.sh"
)
RELEASE_PATHS = {
    "qwen3.8-27b": (
        "docs/evidence/qwen38-study/"
        "2026-09-05-qwen38-autocontinue-generation7-scoring-release-v1.json"
    ),
    "glm-5.3": (
        "docs/evidence/qwen38-study/"
        "2026-09-05-glm53-autocontinue-generation7-scoring-release-v1.json"
    ),
}
AUTHORIZED_AT = "2026-09-05T11:30:00Z"


def load(path: Path) -> dict[str, Any]:
    return generation7.load(path)


def digest(value: dict[str, Any], field: str) -> str:
    return generation7.digest(value, field)


def file_sha256(path: Path) -> str:
    return generation7.file_sha256(path)


def specs(root: Path) -> list[dict[str, Any]]:
    return [load(root / generation7.G7_SPEC_PATHS[model]) for model in generation7.EXPECTED]


def validate_root_authorization(receipt: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    selected = specs(root)
    plans = [generation7.validate_spec(spec, root) for spec in selected]
    expected = {
        "schema_version": AUTH_SCHEMA,
        "append_only": True,
        "status": "AUTHORIZED",
        "author": "/root",
        "authorized_at_utc": AUTHORIZED_AT,
        "scope": {
            "execution_generation": 7,
            "create_once": True,
            "hosted_only": True,
            "scored_launch_authorized": True,
            "dedicated_serving_authorized": False,
            "bulk_release_authorized": False,
            "cluster_mutation_authorized_by_package": False,
        },
        "models": [
            {
                "model": spec["model"],
                "cell_id": spec["statistical_cell"]["cell_id"],
                "generation7_spec_sha256": spec["generation7_spec_sha256"],
                "rendered_plan_sha256": plan["plan_sha256"],
                "execution_id": spec["execution"]["execution_id"],
                "job_name": generation7.EXPECTED[spec["model"]]["job_name"],
            }
            for spec, plan in zip(selected, plans, strict=True)
        ],
        "failure_tombstone": {
            "path": generation7.FAILURE_PATH,
            "file_sha256": file_sha256(root / generation7.FAILURE_PATH),
            "receipt_sha256": load(root / generation7.FAILURE_PATH)["receipt_sha256"],
        },
        "runtime_correction": {
            "docker_cli_source_image": generation7.DOCKER_IMAGE,
            "copy_via_shared_emptydir": True,
            "exact_copied_binary_bytes": 105594160,
            "docker_cli_emptydir_size_limit": "256Mi",
            "docker_cli_emptydir_size_limit_bytes": 268435456,
            "docker_cli_emptydir_headroom_bytes": 162841296,
            "network_package_install": False,
            "evaluator_image_unchanged": True,
        },
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
        raise ValueError("generation-7 root authorization drifted")
    return plans


def authority_binding(authority: dict[str, Any], root: Path) -> dict[str, Any]:
    validate_root_authorization(authority, root)
    return {
        "path": AUTH_PATH,
        "file_sha256": file_sha256(root / AUTH_PATH),
        "receipt_sha256": authority["receipt_sha256"],
    }


def implementation(root: Path, spec: dict[str, Any], built: dict[str, Any]) -> dict[str, Any]:
    expected = generation7.EXPECTED[spec["model"]]
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
        "intent_configmap_name": generation7.INTENT_NAME,
        "sfs_root": f"/mnt/sfs/jobs/{expected['job_name']}",
        "split_package_aggregate_sha256": built["aggregate_sha256"],
        "model_aggregate_sha256": built["model_manifests"][spec["model"]][
            "aggregate_sha256"
        ],
    }


def _release_receipt(
    spec: dict[str, Any],
    authority: dict[str, Any],
    root: Path,
    package_commit: str,
    *,
    built: dict[str, Any],
) -> dict[str, Any]:
    plans = validate_root_authorization(authority, root)
    plan = generation7.validate_spec(spec, root)
    if plan not in plans or not generation2.COMMIT_RE.fullmatch(package_commit):
        raise ValueError("generation-7 release package is invalid")
    receipt = {
        "schema_version": RELEASE_SCHEMA,
        "append_only": True,
        "status": "RELEASED",
        "released_at_utc": AUTHORIZED_AT,
        "model": spec["model"],
        "generation7_spec_sha256": spec["generation7_spec_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "cell_id": spec["statistical_cell"]["cell_id"],
        "execution_id": spec["execution"]["execution_id"],
        "package_commit": package_commit,
        "root_authorization": authority_binding(authority, root),
        "generation6_failure_receipt_sha256": load(root / generation7.FAILURE_PATH)[
            "receipt_sha256"
        ],
        "split_configmap_package": {
            "schema_version": built["schema_version"],
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
            "execution_generation": 7,
            "hosted_only": True,
            "priority_class": "fleet-serve-low",
            "preemption_policy": "Never",
            "bulk_release_authorized": False,
            "dedicated_serving_authorized": False,
            "author": "/root",
        },
        "route_and_inventory": {
            "fresh_fleet_team_and_hosted_models_required": True,
            "fresh_exact_treatment_session_inventory_required": True,
            "generation1_through_generation7_claims_must_be_absent": True,
            "generation7_job_configmap_and_sfs_root_must_be_absent": True,
            "generation6_jobs_must_remain_exact_terminal_failures": True,
            "generation6_sessions_and_side_effects_must_remain_absent": True,
        },
        "privacy": {
            "prompts_traces_flags_or_scores_included": False,
            "credentials_included": False,
        },
    }
    receipt["receipt_sha256"] = digest(receipt, "receipt_sha256")
    return receipt


def validate_release(
    release: dict[str, Any],
    spec: dict[str, Any],
    authority: dict[str, Any],
    root: Path,
    package_commit: str,
) -> None:
    from evals.fleet import autocontinue_generation7_authority_package_v1 as package

    if release != _release_receipt(
        spec, authority, root, package_commit, built=package.build_package(root)
    ):
        raise ValueError("generation-7 scoring release is not authoritative")


def release_binding(
    release: dict[str, Any], file_digest: str, package_commit: str
) -> dict[str, Any]:
    if not generation2.SHA_RE.fullmatch(file_digest):
        raise ValueError("generation-7 release file digest is invalid")
    return {
        "schema_version": RELEASE_SCHEMA,
        "receipt_sha256": release["receipt_sha256"],
        "file_sha256": file_digest,
        "package_commit": package_commit,
    }


def _claim_receipt(
    spec: dict[str, Any], plan: dict[str, Any], release: dict[str, Any], file_digest: str,
    authority: dict[str, Any], root: Path, package_commit: str, *, claimed_at_utc: str,
    job_uid: str | None = None, pod_uid: str | None = None,
) -> dict[str, Any]:
    receipt = {
        "schema_version": CLAIM_SCHEMA,
        "generation7_spec_sha256": spec["generation7_spec_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "cell_id": spec["statistical_cell"]["cell_id"],
        "execution_id": spec["execution"]["execution_id"],
        "execution_generation": 7,
        "run_id": plan["attempts"][0]["run_id"],
        "job_uid": job_uid or os.environ.get("JOB_UID"),
        "pod_uid": pod_uid or os.environ.get("POD_UID"),
        "claimed_at_utc": claimed_at_utc,
        "scoring_release": release_binding(release, file_digest, package_commit),
        "root_authorization": authority_binding(authority, root),
        "generation6_failure_receipt_sha256": load(root / generation7.FAILURE_PATH)[
            "receipt_sha256"
        ],
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
    file_digest: str,
    authority: dict[str, Any],
    root: Path,
    package_commit: str,
) -> None:
    expected = _claim_receipt(
        spec,
        plan,
        release,
        file_digest,
        authority,
        root,
        package_commit,
        claimed_at_utc=claim.get("claimed_at_utc", ""),
        job_uid=claim.get("job_uid"),
        pod_uid=claim.get("pod_uid"),
    )
    if (
        plan != generation7.validate_spec(spec, root)
        or not legacy._is_uuid(claim.get("job_uid"))
        or not legacy._is_uuid(claim.get("pod_uid"))
        or not generation7.generation6.generation5.generation3._canonical_utc(
            claim.get("claimed_at_utc")
        )
        or claim != expected
    ):
        raise ValueError("generation-7 claim is not authoritative")


def validate_generation6_output_absence(
    jobs_root: Path = Path("/mnt/sfs/jobs"),
) -> None:
    for predecessor in generation7.generation6.EXPECTED.values():
        path = jobs_root / predecessor["job_name"]
        if path.exists() or path.is_symlink():
            raise RuntimeError("generation-6 output root appeared before generation-7 claim")


def claim_execution(
    spec: dict[str, Any], plan: dict[str, Any], release: dict[str, Any], file_digest: str,
    authority: dict[str, Any], root: Path, package_commit: str,
    *, claim_root: Path | None = None,
) -> dict[str, Any]:
    validate_release(release, spec, authority, root, package_commit)
    if plan != generation7.validate_spec(spec, root):
        raise ValueError("generation-7 claim spec-plan chain drifted")
    if not legacy._is_uuid(os.environ.get("JOB_UID")) or not legacy._is_uuid(
        os.environ.get("POD_UID")
    ):
        raise RuntimeError("generation-7 claim requires downward API UIDs")
    claim_dir = claim_root or Path(generation7.generation6.generation5.CLAIM_ROOT)
    root_fd = legacy._open_directory_nofollow(claim_dir)
    try:
        lock_fd = legacy._open_claim_lock(root_fd)
        with os.fdopen(lock_fd, "a+b") as lock:
            if not stat.S_ISREG(os.fstat(lock.fileno()).st_mode):
                raise RuntimeError("generation-7 claim lock is not regular")
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            for execution_generation in range(1, 8):
                execution = generation7.exact.execution_for(
                    spec["statistical_cell"]["cell_id"], execution_generation
                )
                predecessor_name = (
                    execution["execution_id"].removeprefix("sha256:") + ".json"
                )
                try:
                    os.stat(predecessor_name, dir_fd=root_fd, follow_symlinks=False)
                except FileNotFoundError:
                    continue
                raise RuntimeError(
                    "generation-7 statistical cell has an existing generation claim"
                )
            name = spec["execution"]["execution_id"].removeprefix("sha256:") + ".json"
            receipt = _claim_receipt(
                spec, plan, release, file_digest, authority, root, package_commit,
                claimed_at_utc=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            )
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(name, flags, 0o600, dir_fd=root_fd)
            try:
                if not stat.S_ISREG(os.fstat(fd).st_mode):
                    raise RuntimeError("generation-7 claim path is unsafe")
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
                file_digest,
                authority,
                root,
                package_commit,
            )
            return receipt
    finally:
        os.close(root_fd)


def terminal_receipt(
    spec: dict[str, Any], plan: dict[str, Any], claim: dict[str, Any], result: dict[str, Any],
    release: dict[str, Any], file_digest: str, authority: dict[str, Any], root: Path,
    package_commit: str,
) -> dict[str, Any]:
    validate_claim(
        claim,
        spec,
        plan,
        release,
        file_digest,
        authority,
        root,
        package_commit,
    )
    receipt = {
        "schema_version": TERMINAL_SCHEMA,
        "generation7_spec_sha256": spec["generation7_spec_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "cell_id": spec["statistical_cell"]["cell_id"],
        "execution_id": spec["execution"]["execution_id"],
        "execution_generation": 7,
        "generation_claim_receipt_sha256": claim["receipt_sha256"],
        "job_uid": claim["job_uid"],
        "pod_uid": claim["pod_uid"],
        "terminal_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "scoring_release": release_binding(release, file_digest, package_commit),
        "root_authorization": authority_binding(authority, root),
        "result": generation2._validated_result(result, plan),
        "retry_allowed": False,
        "bulk_release_authorized": False,
        "scores_included": False,
        "prompts_or_traces_included": False,
    }
    receipt["receipt_sha256"] = digest(receipt, "receipt_sha256")
    return receipt


def run(
    spec: dict[str, Any], release: dict[str, Any], file_digest: str,
    authority: dict[str, Any], launch_route: dict[str, Any], out: Path, proxy: Path,
    root: Path, package_commit: str,
) -> dict[str, Any]:
    plan = generation7.validate_spec(spec, root)
    validate_release(release, spec, authority, root, package_commit)
    jobs = tuple(row["job_name"] for row in generation7.EXPECTED.values())
    hosted_runtime.validate_live_route(
        launch_route, caller=hosted_runtime.LAUNCHER_CALLER, maximum_age_seconds=None,
        candidate_scored_job_names=jobs,
    )
    key = os.environ.get("FLEET_API_KEY")
    if not key:
        raise RuntimeError("FLEET_API_KEY is required")
    lease = plan["execution"]["endpoint_lease"]
    with legacy.endpoint_lease.acquire_endpoint_lease(
        lease_root=Path(lease["lease_root"]), endpoint_key=lease["endpoint_key"],
        maximum_streams=lease["maximum_streams"],
    ):
        live = hosted_runtime.observe_live_route(
            key, caller=hosted_runtime.RUNTIME_CALLER,
            job_uid=os.environ.get("JOB_UID"), pod_uid=os.environ.get("POD_UID"),
            candidate_scored_job_names=jobs,
        )
        hosted_runtime.validate_live_route(
            live, caller=hosted_runtime.RUNTIME_CALLER, maximum_age_seconds=30,
            job_uid=os.environ.get("JOB_UID"), pod_uid=os.environ.get("POD_UID"),
            candidate_scored_job_names=jobs,
        )
        generation7.validate_failure(load(root / generation7.FAILURE_PATH), root)
        hosted._validate_plan_identity_absence(plan, out)
        with hosted._client(key) as client:
            account = self_hosted._request(client, "GET", "/v1/account")
        if (
            account.get("team_name") != "fleet"
            or account.get("team_id") != self_hosted.FLEET_TEAM_ID
        ):
            raise RuntimeError("FLEET_API_KEY is not scoped to Fleet")
        predecessor_spec = load(
            root / generation7.generation6.G6_SPEC_PATHS[spec["model"]]
        )
        predecessor_plan = generation7.generation6.validate_spec(predecessor_spec, root)
        hosted._validate_inventory_for_task(
            predecessor_plan, out, predecessor_plan["tasks"][0], key
        )
        hosted._validate_inventory_for_task(plan, out, plan["tasks"][0], key)
        validate_generation6_output_absence()
        claim = claim_execution(
            spec, plan, release, file_digest, authority, root, package_commit
        )
        out.mkdir(mode=0o700)
        for name in ("attempts", "claims", "task-claims", "task-results", "quarantine", "ramps"):
            (out / name).mkdir(mode=0o700)
        self_hosted.write_json_once(out / "PLAN.json", plan)
        self_hosted.write_json_once(out / "SCORING-RELEASE.json", release)
        self_hosted.write_json_once(out / "ROOT-AUTHORIZATION.json", authority)
        result = legacy._run_cell(plan, out, proxy, key)
        terminal = terminal_receipt(
            spec, plan, claim, result, release, file_digest, authority, root, package_commit
        )
        self_hosted.write_json_once(out / "CANARY-TERMINAL.json", terminal)
        return terminal


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("validate-authority", "render-release", "validate-release", "run"),
    )
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--release", type=Path)
    parser.add_argument("--launch-route", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--proxy", type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--package-commit")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    authority = load(args.authority)
    if args.command == "validate-authority":
        validate_root_authorization(authority, args.repo)
        return 0
    if not args.spec or not args.package_commit:
        parser.error(f"{args.command} requires spec and package commit")
    spec = load(args.spec)
    if args.command == "render-release":
        if not args.output:
            parser.error("render-release requires output")
        from evals.fleet import autocontinue_generation7_authority_package_v1 as package
        from evals.fleet import immutable_submission_snapshot as snapshot

        snapshot.assert_stable(args.repo, args.package_commit)
        if args.output.resolve() != (args.repo / RELEASE_PATHS[spec["model"]]).resolve():
            parser.error("render-release output must be the model's exact release path")
        self_hosted.write_json_once(
            args.output,
            _release_receipt(
                spec, authority, args.repo, args.package_commit,
                built=package.build_package(args.repo),
            ),
        )
        return 0
    if not args.release:
        parser.error(f"{args.command} requires release")
    release = load(args.release)
    validate_release(release, spec, authority, args.repo, args.package_commit)
    if args.command == "run":
        if not args.launch_route or not args.out_dir or not args.proxy:
            parser.error("run requires launch route, output directory, and proxy")
        run(
            spec, release, file_sha256(args.release), authority, load(args.launch_route),
            args.out_dir, args.proxy, args.repo, args.package_commit,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
