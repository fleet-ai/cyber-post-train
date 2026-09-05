"""Executable generation-2 canary runtime built on the audited v3 contracts."""

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
from evals.fleet import hosted_sweep_controller as hosted
from evals.fleet import self_hosted

EXECUTABLE_HELD_SCHEMA = (
    "fleet-opencode-autocontinue-generation2-canary-executable-held-release-v1"
)
EXECUTABLE_HELD_PATH = (
    "docs/evidence/qwen38-study/"
    "2026-09-04-opencode-autocontinue-generation2-canaries-executable-held-v1.json"
)
CONTRACT_HELD_PATH = (
    "docs/evidence/qwen38-study/"
    "2026-09-04-opencode-autocontinue-generation2-canaries-held-v3.json"
)
MODULE_PATH = "evals/fleet/autocontinue_generation2_runtime_v3.py"
MANIFEST_PATH = "evals/fleet/cluster/opencode-autocontinue-generation2-canaries-v3.yaml"
RUN_PATH = "evals/fleet/scripts/run_opencode_autocontinue_generation2_canary_v3.sh"
SUBMIT_PATH = "evals/fleet/scripts/submit_opencode_autocontinue_generation2_canaries_v3.sh"
SPEC_PATHS = (
    "evals/fleet/configs/qwen38-opencode-autocontinue-canary-generation2-v2.json",
    "evals/fleet/configs/glm53-opencode-autocontinue-canary-generation2-v2.json",
)


def load(path: Path) -> dict[str, Any]:
    return contract.load(path)


def digest(value: dict[str, Any], field: str) -> str:
    return contract.digest(value, field)


def file_sha256(path: Path) -> str:
    return contract.file_sha256(path)


def validate_executable_held(
    receipt: dict[str, Any], specs: list[dict[str, Any]], root: Path
) -> None:
    prior = load(root / CONTRACT_HELD_PATH)
    contract.validate_held(prior, specs, root)
    plans = [contract.validate_spec(spec, root) for spec in specs]
    expected = {
        "schema_version": EXECUTABLE_HELD_SCHEMA,
        "append_only": True,
        "status": "HELD",
        "observed_at_utc": receipt.get("observed_at_utc"),
        "supersedes_contract_held_receipt": {
            "path": CONTRACT_HELD_PATH,
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
            "contract_module_path": contract.MODULE_PATH,
            "contract_module_sha256": file_sha256(root / contract.MODULE_PATH),
            "runtime_module_path": MODULE_PATH,
            "runtime_module_sha256": file_sha256(root / MODULE_PATH),
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
            "independent_executable_package_audit",
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
        not contract._canonical_utc(receipt.get("observed_at_utc"))
        or receipt.get("receipt_sha256") != digest(receipt, "receipt_sha256")
        or {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        != expected
    ):
        raise ValueError("generation-2 v3 executable held receipt drifted")


def run(
    spec: dict[str, Any],
    release: dict[str, Any],
    executable_held: dict[str, Any],
    launch_route: dict[str, Any],
    out: Path,
    proxy: Path,
    repo: Path,
    package_commit: str,
    *,
    authorized_at_utc: str,
    authorization_statement: str,
) -> dict[str, Any]:
    plan = contract.validate_spec(spec, repo)
    validate_executable_held(
        executable_held,
        [load(repo / path) for path in SPEC_PATHS],
        repo,
    )
    contract.validate_release(
        release,
        spec,
        repo,
        package_commit,
        authorized_at_utc=authorized_at_utc,
        authorization_statement=authorization_statement,
    )
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
        self_hosted.write_json_once(out / "EXECUTABLE-PACKAGE.json", executable_held)
        result = legacy._run_cell(plan, out, proxy, key)
        terminal = contract.terminal_receipt(
            spec, plan, generation_claim, result, root=repo
        )
        self_hosted.write_json_once(out / "CANARY-TERMINAL.json", terminal)
        return terminal


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate-held", "preview", "run"))
    parser.add_argument("--spec", action="append", type=Path, default=[])
    parser.add_argument("--held", type=Path)
    parser.add_argument("--release", type=Path)
    parser.add_argument("--launch-route", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--proxy", type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--package-commit")
    parser.add_argument("--authorized-at-utc")
    parser.add_argument("--authorization-statement")
    args = parser.parse_args()
    specs = [load(path) for path in args.spec]
    if not args.held:
        parser.error(f"{args.command} requires --held")
    held = load(args.held)
    if args.command in {"validate-held", "preview"}:
        if len(specs) != 2:
            parser.error(f"{args.command} requires two --spec values")
        validate_executable_held(held, specs, args.repo)
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
                        "v3_executable_package": True,
                    },
                    sort_keys=True,
                )
            )
        return 0
    if not all(
        (
            len(specs) == 1,
            args.release,
            args.launch_route,
            args.out_dir,
            args.proxy,
            args.package_commit,
            args.authorized_at_utc,
            args.authorization_statement,
        )
    ):
        parser.error("run requires one spec and exact release/runtime authority arguments")
    run(
        specs[0],
        load(args.release),
        held,
        load(args.launch_route),
        args.out_dir,
        args.proxy,
        args.repo,
        args.package_commit,
        authorized_at_utc=args.authorized_at_utc,
        authorization_statement=args.authorization_statement,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
