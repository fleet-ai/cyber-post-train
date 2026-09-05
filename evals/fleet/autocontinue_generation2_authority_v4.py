"""Independent root-authority gate for generation-2 canary releases."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from evals.fleet import autocontinue_canary_controller as legacy
from evals.fleet import autocontinue_canary_hosted_runtime as hosted_runtime
from evals.fleet import autocontinue_generation2_canary as held_v1
from evals.fleet import autocontinue_generation2_canary_v2 as held_v2
from evals.fleet import autocontinue_generation2_canary_v3 as contract
from evals.fleet import autocontinue_generation2_runtime_v3 as runtime_v3
from evals.fleet import hosted_sweep_controller as hosted
from evals.fleet import self_hosted

AUTH_SCHEMA = "fleet-opencode-autocontinue-generation2-root-authorization-v1"
RELEASE_SCHEMA = "fleet-opencode-autocontinue-generation2-canary-scoring-release-v4"
EXECUTABLE_HELD_SCHEMA = (
    "fleet-opencode-autocontinue-generation2-authority-bound-executable-held-v1"
)
AUTH_PATH = (
    "docs/evidence/qwen38-study/"
    "2026-09-04-opencode-autocontinue-generation2-root-authorization-v1.json"
)
HELD_PATH = (
    "docs/evidence/qwen38-study/"
    "2026-09-04-opencode-autocontinue-generation2-authority-bound-held-v1.json"
)
PREDECESSOR_HELD_PATH = runtime_v3.EXECUTABLE_HELD_PATH
MODULE_PATH = "evals/fleet/autocontinue_generation2_authority_v4.py"
MANIFEST_PATH = "evals/fleet/cluster/opencode-autocontinue-generation2-canaries-v4.yaml"
RUN_PATH = "evals/fleet/scripts/run_opencode_autocontinue_generation2_canary_v4.sh"
SUBMIT_PATH = "evals/fleet/scripts/submit_opencode_autocontinue_generation2_canaries_v4.sh"
SPEC_PATHS = runtime_v3.SPEC_PATHS
AUTH_TIME = "2026-09-05T04:34:27Z"
AUTH_STATEMENTS = {
    "qwen3.8-27b": (
        "Authorize exactly one create-once generation-2 hosted Qwen3.8 27B canary "
        "cell after all fresh route, duplicate, package, and runtime gates pass; no "
        "dedicated serving or bulk release is authorized."
    ),
    "glm-5.3": (
        "Authorize exactly one create-once generation-2 hosted GLM5.3 canary cell "
        "after all fresh route, duplicate, package, and runtime gates pass; no dedicated "
        "serving or bulk release is authorized."
    ),
}


def load(path: Path) -> dict[str, Any]:
    return contract.load(path)


def digest(value: dict[str, Any], field: str) -> str:
    return contract.digest(value, field)


def file_sha256(path: Path) -> str:
    return contract.file_sha256(path)


def _specs(root: Path) -> list[dict[str, Any]]:
    return [load(root / path) for path in SPEC_PATHS]


def validate_root_authorization(
    receipt: dict[str, Any], specs: list[dict[str, Any]], root: Path
) -> None:
    plans = [contract.validate_spec(spec, root) for spec in specs]
    expected_models = [
        {
            "model": spec["model"],
            "cell_id": spec["statistical_cell"]["cell_id"],
            "generation2_spec_sha256": spec["generation2_spec_sha256"],
            "rendered_plan_sha256": plan["plan_sha256"],
            "execution_id": spec["execution"]["execution_id"],
            "authorized_at_utc": AUTH_TIME,
            "statement": AUTH_STATEMENTS[spec["model"]],
        }
        for spec, plan in zip(specs, plans, strict=True)
    ]
    expected = {
        "schema_version": AUTH_SCHEMA,
        "append_only": True,
        "status": "AUTHORIZED",
        "author": "/root",
        "scope": {
            "execution_generation": 2,
            "create_once": True,
            "hosted_only": True,
            "scored_launch_authorized": True,
            "dedicated_serving_authorized": False,
            "bulk_release_authorized": False,
            "release_receipts_present": False,
            "cluster_mutation_authorized_by_this_package": False,
        },
        "models": expected_models,
        "privacy": {
            "prompts_traces_flags_or_scores_included": False,
            "credentials_included": False,
        },
    }
    if (
        any(not contract._canonical_utc(row["authorized_at_utc"]) for row in expected_models)
        or receipt.get("receipt_sha256") != digest(receipt, "receipt_sha256")
        or {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        != expected
    ):
        raise ValueError("generation-2 root authorization is not authoritative")


def _authority_binding(authority: dict[str, Any], root: Path) -> dict[str, Any]:
    return {
        "path": AUTH_PATH,
        "file_sha256": file_sha256(root / AUTH_PATH),
        "receipt_sha256": authority["receipt_sha256"],
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


def validate_release(
    release: dict[str, Any],
    spec: dict[str, Any],
    authority: dict[str, Any],
    root: Path,
    package_commit: str,
) -> None:
    specs = _specs(root)
    validate_root_authorization(authority, specs, root)
    plan = contract.validate_spec(spec, root)
    auth_entry = next(row for row in authority["models"] if row["model"] == spec["model"])
    if not contract.COMMIT_RE.fullmatch(package_commit):
        raise ValueError("generation-2 v4 package commit is invalid")
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
        "root_authorization": _authority_binding(authority, root),
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
        raise ValueError("generation-2 v4 scoring release is not authoritative")


def validate_held(receipt: dict[str, Any], root: Path) -> None:
    specs = _specs(root)
    authority = load(root / AUTH_PATH)
    validate_root_authorization(authority, specs, root)
    prior = load(root / PREDECESSOR_HELD_PATH)
    runtime_v3.validate_executable_held(prior, specs, root)
    plans = [contract.validate_spec(spec, root) for spec in specs]
    expected = {
        "schema_version": EXECUTABLE_HELD_SCHEMA,
        "append_only": True,
        "status": "HELD",
        "observed_at_utc": receipt.get("observed_at_utc"),
        "supersedes_executable_held_receipt": {
            "path": PREDECESSOR_HELD_PATH,
            "receipt_sha256": prior["receipt_sha256"],
        },
        "root_authorization": _authority_binding(authority, root),
        "generation2_spec_sha256s": [row["generation2_spec_sha256"] for row in specs],
        "rendered_plan_sha256s": [row["plan_sha256"] for row in plans],
        "jobs": [held_v1.EXPECTED[row["model"]]["job_name"] for row in specs],
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
            "independent_authority_bound_package_audit",
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
        raise ValueError("generation-2 v4 executable held receipt drifted")


def terminal_receipt(
    terminal_v3: dict[str, Any],
    spec: dict[str, Any],
    plan: dict[str, Any],
    generation_claim: dict[str, Any],
    authority: dict[str, Any],
    *,
    root: Path,
) -> dict[str, Any]:
    contract.validate_terminal(terminal_v3, spec, plan, generation_claim, root=root)
    receipt = {
        "schema_version": "fleet-opencode-autocontinue-generation2-terminal-v4",
        "generation2_spec_sha256": spec["generation2_spec_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "cell_id": spec["statistical_cell"]["cell_id"],
        "execution_id": spec["execution"]["execution_id"],
        "root_authorization": _authority_binding(authority, root),
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
    authority: dict[str, Any],
    *,
    root: Path,
) -> None:
    validate_root_authorization(authority, _specs(root), root)
    inner = receipt.get("terminal_v3")
    if not isinstance(inner, dict):
        raise ValueError("generation-2 v4 terminal is not authoritative")
    contract.validate_terminal(inner, spec, plan, generation_claim, root=root)
    expected = terminal_receipt(inner, spec, plan, generation_claim, authority, root=root)
    if receipt != expected:
        raise ValueError("generation-2 v4 terminal is not authoritative")


def run(
    spec: dict[str, Any],
    release: dict[str, Any],
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
    validate_release(release, spec, authority, repo, package_commit)
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
        for name in ("attempts", "claims", "task-claims", "task-results", "quarantine", "ramps"):
            (out / name).mkdir(mode=0o700)
        self_hosted.write_json_once(out / "PLAN.json", plan)
        self_hosted.write_json_once(out / "SCORING-RELEASE.json", release)
        self_hosted.write_json_once(out / "ROOT-AUTHORIZATION.json", authority)
        self_hosted.write_json_once(out / "EXECUTABLE-PACKAGE.json", held)
        result = legacy._run_cell(plan, out, proxy, key)
        inner = contract.terminal_receipt(spec, plan, generation_claim, result, root=repo)
        terminal = terminal_receipt(
            inner, spec, plan, generation_claim, authority, root=repo
        )
        self_hosted.write_json_once(out / "CANARY-TERMINAL.json", terminal)
        return terminal


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "validate-authority",
            "validate-held",
            "validate-release",
            "preview",
            "run",
        ),
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
    specs = [load(path) for path in args.spec]
    if not args.authority:
        parser.error(f"{args.command} requires --authority")
    authority = load(args.authority)
    if args.command == "validate-authority":
        validate_root_authorization(authority, specs, args.repo)
        return 0
    if not args.held:
        parser.error(f"{args.command} requires --held")
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
                        "authority_bound": True,
                    },
                    sort_keys=True,
                )
            )
        return 0
    if len(specs) != 1 or not args.release or not args.package_commit:
        parser.error(f"{args.command} requires one spec, release, and package commit")
    release = load(args.release)
    validate_release(release, specs[0], authority, args.repo, args.package_commit)
    if args.command == "run":
        if not args.launch_route or not args.out_dir or not args.proxy:
            parser.error("run requires launch route, output directory, and proxy")
        run(
            specs[0],
            release,
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
