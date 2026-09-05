"""Release-continuous authority and terminal evidence for generation-2 canaries."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from evals.fleet import autocontinue_canary_controller as legacy
from evals.fleet import autocontinue_canary_hosted_runtime as hosted_runtime
from evals.fleet import autocontinue_generation2_authority_v4 as authority_v4
from evals.fleet import autocontinue_generation2_canary as held_v1
from evals.fleet import autocontinue_generation2_canary_v2 as held_v2
from evals.fleet import autocontinue_generation2_canary_v3 as contract
from evals.fleet import hosted_sweep_controller as hosted
from evals.fleet import self_hosted

RELEASE_SCHEMA = "fleet-opencode-autocontinue-generation2-canary-scoring-release-v5"
TERMINAL_SCHEMA = "fleet-opencode-autocontinue-generation2-terminal-v5"
HELD_SCHEMA = "fleet-opencode-autocontinue-generation2-authority-bound-executable-held-v2"
PREDECESSOR_HELD_PATH = authority_v4.HELD_PATH
AUTH_PATH = authority_v4.AUTH_PATH
HELD_PATH = (
    "docs/evidence/qwen38-study/"
    "2026-09-04-opencode-autocontinue-generation2-authority-bound-held-v2.json"
)
MODULE_PATH = "evals/fleet/autocontinue_generation2_authority_v5.py"
MANIFEST_PATH = "evals/fleet/cluster/opencode-autocontinue-generation2-canaries-v5.yaml"
RUN_PATH = "evals/fleet/scripts/run_opencode_autocontinue_generation2_canary_v5.sh"
SUBMIT_PATH = "evals/fleet/scripts/submit_opencode_autocontinue_generation2_canaries_v5.sh"
SPEC_PATHS = authority_v4.SPEC_PATHS


def load(path: Path) -> dict[str, Any]:
    return contract.load(path)


def digest(value: dict[str, Any], field: str) -> str:
    return contract.digest(value, field)


def file_sha256(path: Path) -> str:
    return contract.file_sha256(path)


def _specs(root: Path) -> list[dict[str, Any]]:
    return [load(root / path) for path in SPEC_PATHS]


def _held_binding(held: dict[str, Any], root: Path) -> dict[str, Any]:
    return {
        "path": HELD_PATH,
        "file_sha256": file_sha256(root / HELD_PATH),
        "receipt_sha256": held["receipt_sha256"],
    }


def _implementation(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    expected = held_v1.EXPECTED[spec["model"]]
    old = load(root / held_v2.V1_SPEC_PATHS[spec["model"]])
    return {
        "module_path": MODULE_PATH,
        "module_sha256": file_sha256(root / MODULE_PATH),
        "manifest_path": MANIFEST_PATH,
        "manifest_sha256": file_sha256(root / MANIFEST_PATH),
        "run_path": RUN_PATH,
        "run_sha256": file_sha256(root / RUN_PATH),
        "submit_path": SUBMIT_PATH,
        "submit_sha256": file_sha256(root / SUBMIT_PATH),
        "job_name": expected["job_name"],
        "configmap_name": expected["configmap_name"],
        "sfs_root": old["identities"]["sfs_root"],
    }


def validate_held(receipt: dict[str, Any], root: Path) -> None:
    specs = _specs(root)
    authority = load(root / AUTH_PATH)
    authority_v4.validate_root_authorization(authority, specs, root)
    prior = load(root / PREDECESSOR_HELD_PATH)
    authority_v4.validate_held(prior, root)
    plans = [contract.validate_spec(spec, root) for spec in specs]
    expected = {
        "schema_version": HELD_SCHEMA,
        "append_only": True,
        "status": "HELD",
        "observed_at_utc": receipt.get("observed_at_utc"),
        "supersedes_authority_bound_held_receipt": {
            "path": PREDECESSOR_HELD_PATH,
            "receipt_sha256": prior["receipt_sha256"],
        },
        "root_authorization": authority_v4._authority_binding(authority, root),
        "generation2_spec_sha256s": [
            spec["generation2_spec_sha256"] for spec in specs
        ],
        "rendered_plan_sha256s": [plan["plan_sha256"] for plan in plans],
        "jobs": [held_v1.EXPECTED[spec["model"]]["job_name"] for spec in specs],
        "implementation": {
            "module_path": MODULE_PATH,
            "module_sha256": file_sha256(root / MODULE_PATH),
            "manifest_path": MANIFEST_PATH,
            "manifest_sha256": file_sha256(root / MANIFEST_PATH),
            "run_path": RUN_PATH,
            "run_sha256": file_sha256(root / RUN_PATH),
            "submit_path": SUBMIT_PATH,
            "submit_sha256": file_sha256(root / SUBMIT_PATH),
        },
        "launch_authorized": False,
        "bulk_release_authorized": False,
        "dedicated_serving_authorized": False,
        "remaining_gates": [
            "independent_release_continuity_audit",
            "fresh_hosted_route_and_duplicate_inventory",
            "append_only_model_specific_scoring_releases",
            "explicit_submit_authorization",
        ],
        "privacy": {
            "prompts_traces_flags_or_scores_included": False,
            "credentials_included": False,
        },
    }
    if (
        not contract._canonical_utc(receipt.get("observed_at_utc"))
        or receipt.get("receipt_sha256") != digest(receipt, "receipt_sha256")
        or {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        != expected
    ):
        raise ValueError("generation-2 v5 executable held receipt drifted")


def validate_release(
    release: dict[str, Any],
    spec: dict[str, Any],
    authority: dict[str, Any],
    held: dict[str, Any],
    root: Path,
    package_commit: str,
) -> None:
    specs = _specs(root)
    authority_v4.validate_root_authorization(authority, specs, root)
    validate_held(held, root)
    plan = contract.validate_spec(spec, root)
    auth_entry = next(row for row in authority["models"] if row["model"] == spec["model"])
    if not contract.COMMIT_RE.fullmatch(package_commit):
        raise ValueError("generation-2 v5 package commit is invalid")
    expected = {
        "schema_version": RELEASE_SCHEMA,
        "append_only": True,
        "status": "RELEASED",
        "released_at_utc": auth_entry["authorized_at_utc"],
        "generation2_spec_sha256": spec["generation2_spec_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "cell_id": spec["statistical_cell"]["cell_id"],
        "execution_id": spec["execution"]["execution_id"],
        "incident_receipt_sha256": held_v1.INCIDENT_SHA,
        "tombstone_bundle_receipt_sha256": held_v1._tombstones(root)[
            "receipt_sha256"
        ],
        "package_commit": package_commit,
        "root_authorization": authority_v4._authority_binding(authority, root),
        "executable_held_authority": _held_binding(held, root),
        "implementation": _implementation(root, spec),
        "authorization": {
            "launch_authorized": True,
            "create_once": True,
            "execution_generation": 2,
            "bulk_release_authorized": False,
            "dedicated_serving_authorized": False,
            "author": "/root",
            "authorized_at_utc": auth_entry["authorized_at_utc"],
            "statement": auth_entry["statement"],
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
        raise ValueError("generation-2 v5 scoring release is not authoritative")


def terminal_receipt(
    terminal_v3: dict[str, Any],
    spec: dict[str, Any],
    plan: dict[str, Any],
    generation_claim: dict[str, Any],
    release: dict[str, Any],
    release_file_sha256: str,
    authority: dict[str, Any],
    held: dict[str, Any],
    package_commit: str,
    *,
    root: Path,
) -> dict[str, Any]:
    validate_release(release, spec, authority, held, root, package_commit)
    contract.validate_terminal(terminal_v3, spec, plan, generation_claim, root=root)
    if not contract.SHA_RE.fullmatch(release_file_sha256):
        raise ValueError("generation-2 v5 release file digest is invalid")
    receipt = {
        "schema_version": TERMINAL_SCHEMA,
        "generation2_spec_sha256": spec["generation2_spec_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "cell_id": spec["statistical_cell"]["cell_id"],
        "execution_id": spec["execution"]["execution_id"],
        "generation_claim_receipt_sha256": generation_claim["receipt_sha256"],
        "root_authorization": authority_v4._authority_binding(authority, root),
        "executable_held_authority": _held_binding(held, root),
        "scoring_release": {
            "schema_version": RELEASE_SCHEMA,
            "receipt_sha256": release["receipt_sha256"],
            "file_sha256": release_file_sha256,
            "package_commit": package_commit,
        },
        "terminal_v3_receipt_sha256": terminal_v3["receipt_sha256"],
        "terminal_v3": terminal_v3,
        "scores_included": False,
        "prompts_or_traces_included": False,
    }
    receipt["receipt_sha256"] = digest(receipt, "receipt_sha256")
    return receipt


def validate_terminal(
    receipt: dict[str, Any],
    spec: dict[str, Any],
    plan: dict[str, Any],
    generation_claim: dict[str, Any],
    release: dict[str, Any],
    release_file_sha256: str,
    authority: dict[str, Any],
    held: dict[str, Any],
    package_commit: str,
    *,
    root: Path,
) -> None:
    expected = terminal_receipt(
        receipt.get("terminal_v3", {}),
        spec,
        plan,
        generation_claim,
        release,
        release_file_sha256,
        authority,
        held,
        package_commit,
        root=root,
    )
    if receipt != expected:
        raise ValueError("generation-2 v5 terminal is not authoritative")


def run(
    spec: dict[str, Any],
    release: dict[str, Any],
    release_file_sha256: str,
    held: dict[str, Any],
    authority: dict[str, Any],
    launch_route: dict[str, Any],
    out: Path,
    proxy: Path,
    repo: Path,
    package_commit: str,
) -> dict[str, Any]:
    plan = contract.validate_spec(spec, repo)
    validate_held(held, repo)
    validate_release(release, spec, authority, held, repo, package_commit)
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
        old = held_v2._v1_spec(spec, repo)
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
        generation_claim = contract.claim_execution_generation(spec, plan, repo=repo)
        out.mkdir(mode=0o700)
        for name in (
            "attempts",
            "claims",
            "task-claims",
            "task-results",
            "quarantine",
            "ramps",
        ):
            (out / name).mkdir(mode=0o700)
        self_hosted.write_json_once(out / "PLAN.json", plan)
        self_hosted.write_json_once(out / "SCORING-RELEASE.json", release)
        self_hosted.write_json_once(out / "ROOT-AUTHORIZATION.json", authority)
        self_hosted.write_json_once(out / "EXECUTABLE-PACKAGE.json", held)
        result = legacy._run_cell(plan, out, proxy, key)
        inner = contract.terminal_receipt(
            spec, plan, generation_claim, result, root=repo
        )
        terminal = terminal_receipt(
            inner,
            spec,
            plan,
            generation_claim,
            release,
            release_file_sha256,
            authority,
            held,
            package_commit,
            root=repo,
        )
        self_hosted.write_json_once(out / "CANARY-TERMINAL.json", terminal)
        return terminal


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("validate-held", "validate-release", "preview", "run")
    )
    parser.add_argument("--spec", action="append", type=Path, default=[])
    parser.add_argument("--authority", type=Path)
    parser.add_argument("--held", type=Path)
    parser.add_argument("--release", type=Path)
    parser.add_argument("--launch-route", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--proxy", type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--package-commit")
    args = parser.parse_args()
    if not args.authority or not args.held:
        parser.error(f"{args.command} requires authority and held receipts")
    authority = load(args.authority)
    held = load(args.held)
    if args.command in {"validate-held", "preview"}:
        validate_held(held, args.repo)
        if args.command == "preview":
            print(
                json.dumps(
                    {
                        "ok": True,
                        "status": "HELD",
                        "launch_authorized": False,
                        "cluster_objects_created": False,
                        "terminal_release_continuity": True,
                    },
                    sort_keys=True,
                )
            )
        return 0
    if len(args.spec) != 1 or not args.release or not args.package_commit:
        parser.error(f"{args.command} requires one spec, release, and package commit")
    spec = load(args.spec[0])
    release = load(args.release)
    validate_release(release, spec, authority, held, args.repo, args.package_commit)
    if args.command == "run":
        if not args.launch_route or not args.out_dir or not args.proxy:
            parser.error("run requires launch route, output directory, and proxy")
        run(
            spec,
            release,
            file_sha256(args.release),
            held,
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
