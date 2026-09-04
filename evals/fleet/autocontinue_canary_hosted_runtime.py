"""Hosted-only release and live-route gates for corrected-treatment canaries."""

from __future__ import annotations

import argparse
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.fleet import autocontinue_canary_controller as canary
from evals.fleet import autocontinue_canary_hosted_release as phase_c
from evals.fleet import autocontinue_hosted_health as health
from evals.fleet import self_hosted

ROUTE_SCHEMA = "fleet-opencode-autocontinue-canary-live-hosted-route-v1"
EXPECTED_MODELS = ("qwen3.8-27b", "glm-5.3")
SCORED_JOBS = ("chris-q38-ac-canary1-v1", "chris-glm53-ac-canary1-v1")
SUCCESSOR_SCORED_JOBS = ("chris-q38-ac-canary1-v2", "chris-glm53-ac-canary1-v2")
LAUNCHER_CALLER = "scored_launcher_precreate"
RUNTIME_CALLER = "scored_runtime_preclaim"
PRIVACY = {
    "credentials_included": False,
    "response_bodies_included": False,
    "prompts_or_traces_included": False,
    "scores_included": False,
}


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError("live hosted route timestamp is invalid")
    return parsed


def observe_live_route(
    api_key: str,
    *,
    caller: str,
    job_uid: str | None = None,
    pod_uid: str | None = None,
    opener: Any = None,
    now: datetime | None = None,
    candidate_scored_job_names: tuple[str, ...] = SCORED_JOBS,
) -> dict[str, Any]:
    if caller not in {LAUNCHER_CALLER, RUNTIME_CALLER}:
        raise ValueError("live hosted route caller is invalid")
    request = health.fetch_json
    kwargs = {} if opener is None else {"opener": opener}
    account = request("fleet_account", health.ACCOUNT_URL, api_key, **kwargs)
    if account.get("team_name") != "fleet" or account.get("team_id") != health.FLEET_TEAM_ID:
        raise health.GateError("fleet_team_identity_mismatch")
    roster = request("hosted_models", health.MODELS_URL, api_key, **kwargs)
    rows = roster.get("data")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise health.GateError("hosted_models_invalid_shape")
    present = []
    for model_id in EXPECTED_MODELS:
        if len([row for row in rows if row.get("id") == model_id]) != 1:
            raise health.GateError(f"{model_id}_availability_mismatch")
        present.append(model_id)
    observed_at = (now or datetime.now(UTC)).isoformat().replace("+00:00", "Z")
    receipt = {
        "schema_version": ROUTE_SCHEMA,
        "append_only": True,
        "status": "PASSED",
        "observed_at_utc": observed_at,
        "campaign_sha256": canary.CAMPAIGN_SHA,
        "caller": caller,
        "fleet_account": {
            "team_name": "fleet",
            "team_id": self_hosted.FLEET_TEAM_ID,
            "authenticated_get_succeeded": True,
        },
        "hosted_route": {
            "origin": "https://inference.flt.build",
            "authenticated_models_get_succeeded": True,
            "served_ids_present_exactly_once": present,
            "dedicated_serving_state": "USER_STOPPED_UNAVAILABLE",
        },
        "runtime": {
            "job_uid": job_uid,
            "pod_uid": pod_uid,
            "candidate_scored_job_names": list(candidate_scored_job_names),
        },
        "request_counts": {
            "fleet_account_get": 1,
            "hosted_models_get": 1,
            "chat_completions": 0,
            "fleet_task_or_scoring": 0,
        },
        "privacy": dict(PRIVACY),
    }
    receipt["receipt_sha256"] = canary.digest_without(receipt, "receipt_sha256")
    return receipt


def validate_live_route(
    receipt: dict[str, Any],
    *,
    caller: str,
    maximum_age_seconds: int | None,
    now: datetime | None = None,
    job_uid: str | None = None,
    pod_uid: str | None = None,
    candidate_scored_job_names: tuple[str, ...] = SCORED_JOBS,
) -> None:
    observed = _utc(str(receipt.get("observed_at_utc")))
    current = now or datetime.now(UTC)
    age = (current - observed).total_seconds()
    expected = {
        "schema_version": ROUTE_SCHEMA,
        "append_only": True,
        "status": "PASSED",
        "observed_at_utc": receipt.get("observed_at_utc"),
        "campaign_sha256": canary.CAMPAIGN_SHA,
        "caller": caller,
        "fleet_account": {
            "team_name": "fleet",
            "team_id": self_hosted.FLEET_TEAM_ID,
            "authenticated_get_succeeded": True,
        },
        "hosted_route": {
            "origin": "https://inference.flt.build",
            "authenticated_models_get_succeeded": True,
            "served_ids_present_exactly_once": list(EXPECTED_MODELS),
            "dedicated_serving_state": "USER_STOPPED_UNAVAILABLE",
        },
        "runtime": {
            "job_uid": job_uid,
            "pod_uid": pod_uid,
            "candidate_scored_job_names": list(candidate_scored_job_names),
        },
        "request_counts": {
            "fleet_account_get": 1,
            "hosted_models_get": 1,
            "chat_completions": 0,
            "fleet_task_or_scoring": 0,
        },
        "privacy": dict(PRIVACY),
    }
    if (
        receipt.get("receipt_sha256") != canary.digest_without(receipt, "receipt_sha256")
        or {key: value for key, value in receipt.items() if key != "receipt_sha256"} != expected
        or age < -5
        or (maximum_age_seconds is not None and age > maximum_age_seconds)
    ):
        raise ValueError("live hosted route receipt is not authoritative or fresh")


def validate_hosted_release(
    release: dict[str, Any],
    plan: dict[str, Any],
    root: Path,
    expected_package_commit: str,
) -> None:
    canary.validate_plan(plan)
    phase_c.validate_preflight_bundle(plan, root)
    bundle = phase_c.BUNDLES[plan["shard_key"]]
    successor = bundle.get("successor") is True
    expected_evidence = {
        "fresh_inventory_receipt_sha256": canary.TASK_INVENTORY_SHA,
        "fresh_inventory_execution_sha256": canary.TASK_INVENTORY_EXECUTION_SHA,
        "fresh_duplicate_inventory_receipt_path": bundle["duplicate_path"],
        "fresh_duplicate_inventory_receipt_sha256": bundle["duplicate_sha"],
        "preflight_authorization_receipt_path": bundle["preauth_path"],
        "preflight_authorization_receipt_sha256": bundle["preauth_sha"],
        "preflight_receipt_path": bundle["preflight_path"],
        "preflight_receipt_sha256": bundle["preflight_sha"],
        "preflight_post_exit_receipt_path": bundle["post_exit_path"],
        "preflight_post_exit_receipt_sha256": bundle["post_exit_sha"],
        "dedicated_parity_receipt_sha256": canary.DEDICATED_PARITY_SHA,
        "dedicated_parity_role": "historical_compatibility_only_not_live_route_authority",
        "hosted_health_receipt_sha256": canary.HOSTED_HEALTH_SHA,
        "shared_pvc_flock_receipt_sha256": canary.FLOCK_GATE_SHA,
        "controller_compatibility_receipt_sha256": (
            bundle["compatibility_sha"]
            if successor
            else canary._compatibility(root)["receipt_sha256"]
        ),
    }
    if not successor:
        expected_evidence.update(
            {
                "hosted_only_route_receipt_path": phase_c.ROUTE_PATH,
                "hosted_only_route_receipt_sha256": phase_c.ROUTE_SHA,
            }
        )
    expected_route = {
        "serving_route": "HOSTED_ONLY",
        "dedicated_serving_state": "USER_STOPPED_UNAVAILABLE",
        "dedicated_recreation_authorized": False,
        "fresh_authenticated_models_get_required_before_scored_job_create": True,
        "fresh_authenticated_models_get_required_before_global_cell_claim": True,
        "required_served_ids": list(EXPECTED_MODELS),
        "launcher_precreate_maximum_route_age_seconds": 120,
        "runtime_does_not_rely_on_launcher_receipt_freshness": True,
        "runtime_maximum_route_age_seconds": 30,
    }
    expected_implementation = {
        "package_commit": expected_package_commit,
        "plan_sha256": plan["plan_sha256"],
        "controller_sha256": canary._sha(root / "evals/fleet/autocontinue_canary_controller.py"),
        "frozen_controller_sha256": canary._sha(root / "evals/fleet/hosted_sweep_controller.py"),
        "phase_c_validator_sha256": canary._sha(
            root / "evals/fleet/autocontinue_canary_hosted_release.py"
        ),
        "hosted_runtime_sha256": canary._sha(
            root / "evals/fleet/autocontinue_canary_hosted_runtime.py"
        ),
        "hosted_health_sha256": canary._sha(root / "evals/fleet/autocontinue_hosted_health.py"),
        "self_hosted_sha256": canary._sha(root / "evals/fleet/self_hosted.py"),
        "runner_sha256": canary._sha(root / "evals/fleet/opencode_train_sweep_runner.py"),
        "endpoint_lease_sha256": canary._sha(root / "evals/fleet/endpoint_lease.py"),
        "fixed_proxy_sha256": canary._sha(root / "evals/fleet/fixed_proxy.py"),
        "dockerfile_sha256": canary._sha(root / "evals/fleet/Dockerfile.opencode"),
        "campaign_file_sha256": canary._sha(
            root / "evals/fleet/configs/q38-glm53-opencode-autocontinue-primary-campaign-v1.json"
        ),
        "compatibility_file_sha256": canary._sha(
            root
            / "docs/evidence/qwen38-study/"
            / (
                "2026-09-04-opencode-autocontinue-canary-controller-compatibility-v3.json"
                if successor
                else "2026-09-04-opencode-autocontinue-canary-controller-compatibility-v2.json"
            )
        ),
        "preflight_manifest_sha256": canary._sha(
            root
            / "evals/fleet/cluster/"
            / (
                "opencode-autocontinue-canary-preflights-v3.yaml"
                if successor
                else "opencode-autocontinue-canary-preflights-v2.yaml"
            )
        ),
        "scored_manifest_sha256": canary._sha(
            root
            / "evals/fleet/cluster/"
            / (
                "opencode-autocontinue-canary-successor-scored-v4.yaml"
                if successor
                else "opencode-autocontinue-canary-scored-v2.yaml"
            )
        ),
        "run_script_sha256": canary._sha(
            root / "evals/fleet/scripts/run_opencode_autocontinue_canary.sh"
        ),
        "launcher_sha256": canary._sha(
            root
            / "evals/fleet/scripts/"
            / (
                "submit_opencode_autocontinue_canaries_v2.sh"
                if successor
                else "submit_opencode_autocontinue_canaries_v1.sh"
            )
        ),
    }
    if successor:
        expected_implementation["manifest_authorization_sha256"] = canary._sha(
            root / "evals/fleet/autocontinue_successor_manifest_authorization.py"
        )
    expected_cell = {
        "source_rank": plan["tasks"][0]["source_rank"],
        "attempt": 1,
        "task_version_id": plan["tasks"][0]["task"]["version_id"],
    }
    if (
        not canary._is_git_commit(expected_package_commit)
        or set(release)
        != {
            "schema_version",
            "append_only",
            "status",
            "released_at_utc",
            "campaign_sha256",
            "plan_sha256",
            "cell",
            "evidence",
            "implementation",
            "authorization",
            "terminal_contract",
            "hosted_only_route",
            "privacy",
            "receipt_sha256",
        }
        or release.get("schema_version") != canary.RELEASE_SCHEMA
        or release.get("append_only") is not True
        or release.get("status") != "RELEASED"
        or not canary._is_utc_timestamp(release.get("released_at_utc"))
        or release.get("campaign_sha256") != canary.CAMPAIGN_SHA
        or release.get("plan_sha256") != plan["plan_sha256"]
        or release.get("cell") != expected_cell
        or release.get("evidence") != expected_evidence
        or release.get("hosted_only_route") != expected_route
        or release.get("implementation") != expected_implementation
        or release.get("authorization")
        != {
            "launch_authorized": True,
            "create_once": True,
            "required_priority_class": canary.PRIORITY,
            "author": "/root",
            "authorized_at_utc": release.get("authorization", {}).get("authorized_at_utc"),
            "statement": release.get("authorization", {}).get("statement"),
        }
        or not canary._is_utc_timestamp(release.get("authorization", {}).get("authorized_at_utc"))
        or not isinstance(release.get("authorization", {}).get("statement"), str)
        or not release.get("authorization", {}).get("statement")
        or release.get("terminal_contract")
        != {
            "downward_job_uid_required": True,
            "downward_pod_uid_required": True,
            "post_exit_k8s_observer_required": True,
            "terminal_schema_version": canary.TERMINAL_SCHEMA,
            "post_exit_schema_version": canary.POST_EXIT_SCHEMA,
        }
        or release.get("privacy") != phase_c.PRIVACY
        or release.get("receipt_sha256") != canary.digest_without(release, "receipt_sha256")
    ):
        raise ValueError("hosted-only canary scoring release is not authoritative")


def run_hosted(
    plan: dict[str, Any],
    release: dict[str, Any],
    launch_route: dict[str, Any],
    root: Path,
    proxy: Path,
    repo: Path,
    package_commit: str,
) -> dict[str, Any]:
    validate_hosted_release(release, plan, repo, package_commit)
    candidate_jobs = SUCCESSOR_SCORED_JOBS if plan["shard_key"].endswith("_v2") else SCORED_JOBS
    validate_live_route(
        launch_route,
        caller=LAUNCHER_CALLER,
        maximum_age_seconds=None,
        candidate_scored_job_names=candidate_jobs,
    )
    key = os.environ.get("FLEET_API_KEY")
    if not key:
        raise RuntimeError("FLEET_API_KEY is required")
    lease = plan["execution"]["endpoint_lease"]
    with canary.endpoint_lease.acquire_endpoint_lease(
        lease_root=Path(lease["lease_root"]),
        endpoint_key=lease["endpoint_key"],
        maximum_streams=lease["maximum_streams"],
    ):
        # This authenticated endpoint check runs while holding the exact endpoint
        # lease and before any permanent global, task, root, or attempt claim.
        runtime_route = observe_live_route(
            key,
            caller=RUNTIME_CALLER,
            job_uid=os.environ.get("JOB_UID"),
            pod_uid=os.environ.get("POD_UID"),
            candidate_scored_job_names=candidate_jobs,
        )
        validate_live_route(
            runtime_route,
            caller=RUNTIME_CALLER,
            maximum_age_seconds=30,
            job_uid=os.environ.get("JOB_UID"),
            pod_uid=os.environ.get("POD_UID"),
            candidate_scored_job_names=candidate_jobs,
        )
        canary.hosted._validate_plan_identity_absence(plan, root)
        with canary.hosted._client(key) as client:
            account = canary.self_hosted._request(client, "GET", "/v1/account")
        if (
            account.get("team_name") != "fleet"
            or account.get("team_id") != self_hosted.FLEET_TEAM_ID
        ):
            raise RuntimeError("FLEET_API_KEY is not scoped to Fleet")
        canary.hosted._validate_inventory_for_task(plan, root, plan["tasks"][0], key)
        global_claim = canary.claim_global_cell(plan)
        root.mkdir(mode=0o700)
        for name in (
            "attempts",
            "claims",
            "task-claims",
            "task-results",
            "quarantine",
            "ramps",
        ):
            (root / name).mkdir(mode=0o700)
        self_hosted.write_json_once(root / "PLAN.json", plan)
        self_hosted.write_json_once(root / "SCORING-RELEASE.json", release)
        result = canary._run_cell(plan, root, proxy, key)
        terminal = {
            "schema_version": canary.TERMINAL_SCHEMA,
            "plan_sha256": plan["plan_sha256"],
            "campaign_sha256": canary.CAMPAIGN_SHA,
            "controller_compatibility_receipt_sha256": release["evidence"][
                "controller_compatibility_receipt_sha256"
            ],
            "release_receipt_sha256": release["receipt_sha256"],
            "job_uid": os.environ["JOB_UID"],
            "pod_uid": os.environ["POD_UID"],
            "source_rank": int(plan["tasks"][0]["source_rank"]),
            "attempt": 1,
            "task_version_id": plan["tasks"][0]["task"]["version_id"],
            "run_id": plan["attempts"][0]["run_id"],
            "global_cell_claim_receipt_sha256": global_claim["receipt_sha256"],
            "model_revision": plan["model"]["revision"],
            "served_id": plan["model"]["served_id"],
            "session_model": plan["model"]["session_model"],
            "model_sha256": self_hosted.sha256(self_hosted.canonical_json(plan["model"])),
            "harness_sha256": self_hosted.sha256(self_hosted.canonical_json(plan["harness"])),
            "treatment_block_sha256": self_hosted.sha256(
                self_hosted.canonical_json(plan["treatment_block"])
            ),
            "context_management": canary.CONTEXT,
            "settings_file_sha256": plan["harness"]["settings_file_sha256"],
            "required_task_tools": ["bash", "submit_report"],
            "required_task_tool_catalog_sha256": plan["execution"][
                "required_task_tool_catalog_sha256"
            ],
            "endpoint_lease": plan["execution"]["endpoint_lease"],
            "attempt_config_sha256": result["attempt_config_sha256"],
            "claim_sha256": result["claim_sha256"],
            "terminal_at_utc": datetime.now(UTC).isoformat(),
            "accepted": result["accepted"],
            "credited": result["accepted"],
            "quarantined": result["quarantined"],
            "exact_cell_count": 1,
            "legacy_credited_sessions": 0,
            "retry_allowed": False,
            "bulk_release_granted": False,
            "post_exit_k8s_observer_required": True,
            "scores_included": False,
            "prompts_or_traces_included": False,
        }
        if result.get("acceptance_receipt_sha256"):
            terminal["acceptance_receipt_sha256"] = result["acceptance_receipt_sha256"]
            terminal["session_id"] = result["session_id"]
            terminal["verifier_execution_id"] = result["verifier_execution_id"]
            terminal["session_ingest_completed"] = result["session_ingest_completed"]
            terminal["cleanup_completed"] = result["cleanup_completed"]
        terminal["receipt_sha256"] = canary.digest_without(terminal, "receipt_sha256")
        canary.validate_terminal(terminal, plan, release)
        self_hosted.write_json_once(root / "CANARY-TERMINAL.json", terminal)
        return terminal


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    observe = sub.add_parser("observe-route")
    observe.add_argument("--out", type=Path, required=True)
    observe.add_argument("--expected-job", action="append")
    validate = sub.add_parser("validate-route")
    validate.add_argument("--receipt", type=Path, required=True)
    validate.add_argument("--maximum-age-seconds", type=int, required=True)
    validate.add_argument("--expected-job", action="append")
    final = sub.add_parser("validate-release")
    final.add_argument("--plan", type=Path, required=True)
    final.add_argument("--release", type=Path, required=True)
    final.add_argument("--repo", type=Path, required=True)
    final.add_argument("--package-commit", required=True)
    execute = sub.add_parser("run-hosted")
    execute.add_argument("--plan", type=Path, required=True)
    execute.add_argument("--release", type=Path, required=True)
    execute.add_argument("--launch-route", type=Path, required=True)
    execute.add_argument("--out-dir", type=Path, required=True)
    execute.add_argument("--proxy", type=Path, required=True)
    execute.add_argument("--repo", type=Path, required=True)
    execute.add_argument("--package-commit", required=True)
    args = parser.parse_args()
    if args.command == "observe-route":
        candidate_jobs = tuple(args.expected_job or SCORED_JOBS)
        receipt = observe_live_route(
            os.environ["FLEET_API_KEY"],
            caller=LAUNCHER_CALLER,
            candidate_scored_job_names=candidate_jobs,
        )
        self_hosted.write_json_once(args.out, receipt)
        print("hosted route gate passed")
        return 0
    if args.command == "validate-route":
        validate_live_route(
            canary.load_object(args.receipt),
            caller=LAUNCHER_CALLER,
            maximum_age_seconds=args.maximum_age_seconds,
            candidate_scored_job_names=tuple(args.expected_job or SCORED_JOBS),
        )
        return 0
    plan = canary.load_object(args.plan)
    release = canary.load_object(args.release)
    if args.command == "validate-release":
        validate_hosted_release(release, plan, args.repo, args.package_commit)
        return 0
    terminal = run_hosted(
        plan,
        release,
        canary.load_object(args.launch_route),
        args.out_dir,
        args.proxy,
        args.repo,
        args.package_commit,
    )
    print(terminal["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
