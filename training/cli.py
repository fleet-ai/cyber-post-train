from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path

from .fleet import (
    FleetClient,
    FleetExportError,
    export_sessions,
    is_blackbox_task_key,
    roster_session_refs,
)
from .io import atomic_write_json, atomic_write_jsonl, file_sha256
from .jobs_api import (
    DEFAULT_BASE_URL,
    JobsAPIError,
    TrainingJobsClient,
    concise_status,
    load_run_config,
)
from .normalize import build_datasets
from .plan import create_run_plan
from .rl_config import (
    FleetTrainingTaskCatalog,
    build_full_rl_config,
    build_treatment_set,
    freeze_task_split,
    read_json,
    read_trajectories,
    write_full_rl_config,
    write_task_snapshot,
)
from .runner import run_plan
from .secrets import secret_values


def _job_ids(args: argparse.Namespace, client: FleetClient) -> list[str]:
    ids = list(args.job_id or [])
    if args.job_ids_file:
        ids.extend(
            line.strip()
            for line in args.job_ids_file.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        )
    if args.discover_completed_cyber:
        if not args.created_after:
            raise ValueError("--discover-completed-cyber requires --created-after")
        after = dt.datetime.fromisoformat(args.created_after.replace("Z", "+00:00"))
        if not after.tzinfo:
            after = after.replace(tzinfo=dt.UTC)
        for job in client.completed_jobs(created_after=after):
            job_id = str(job.get("id") or "")
            roster = client.job_sessions(job_id)
            refs = roster_session_refs(job_id, job.get("name"), roster)
            if any(is_blackbox_task_key(ref.task_key) for ref in refs):
                ids.append(job_id)
    return list(dict.fromkeys(ids))


def _export(args: argparse.Namespace) -> int:
    api_key = os.environ.get("FLEET_API_KEY")
    if not api_key:
        raise ValueError("FLEET_API_KEY must be injected through the environment")
    client = FleetClient(api_key, base_url=args.base_url)
    ids = _job_ids(args, client)
    if not ids:
        raise ValueError("no jobs selected")
    refs = []
    for job_id in ids:
        roster = client.job_sessions(job_id)
        refs.extend(
            ref
            # The session roster is the durable source for historical jobs.
            # Some legacy jobs are intentionally absent from GET /v1/jobs/{id}
            # while remaining readable through the roster endpoint.
            for ref in roster_session_refs(job_id, None, roster)
            if is_blackbox_task_key(ref.task_key)
            and (
                args.include_incomplete
                or str(ref.session.get("status") or "").lower()
                in {"completed", "succeeded", "success"}
            )
        )
    if args.dry_run:
        print(json.dumps({"jobs": ids, "eligible_sessions": len(refs)}, indent=2))
        return 0
    count = atomic_write_jsonl(
        args.output,
        export_sessions(client, refs, workers=args.workers),
        private=True,
    )
    atomic_write_json(
        args.output.with_suffix(args.output.suffix + ".manifest.json"),
        {
            "schema": "fleet_export_manifest_v1",
            "jobs": ids,
            "sessions": count,
            "sha256": file_sha256(args.output),
        },
        private=True,
    )
    print(f"exported {count} session transcripts to {args.output}")
    return 0


def _normalize(args: argparse.Namespace) -> int:
    manifest = build_datasets(args.input, args.output_dir, secrets=secret_values())
    print(json.dumps(manifest["counts"], indent=2))
    return 0


def _plan(args: argparse.Namespace) -> int:
    plan = create_run_plan(
        args.config,
        args.manifest,
        args.compatibility_receipt,
        args.evaluation_protocol,
        args.output,
    )
    print(json.dumps({"plan": str(args.output), "digest": plan["plan_digest"]}, indent=2))
    return 0


def _run(args: argparse.Namespace) -> int:
    outcomes = run_plan(args.plan, args.work_dir, execute=args.execute)
    print(json.dumps(outcomes, indent=2))
    return 0


def _rl_snapshot(args: argparse.Namespace) -> int:
    bearer = os.environ.get("FLEET_TRAINING_API_TOKEN")
    if not bearer:
        raise ValueError("FLEET_TRAINING_API_TOKEN must be injected through the environment")
    manifest = read_json(args.dataset_manifest)
    catalog = FleetTrainingTaskCatalog(bearer, base_url=args.base_url)
    snapshot = freeze_task_split(
        read_trajectories(args.trajectories),
        catalog.options,
        source_job_id=args.source_job_id,
        dataset_manifest_digest=str(manifest.get("manifest_digest") or ""),
        workers=args.workers,
        fetch_environment_versions=catalog.environment_versions,
    )
    write_task_snapshot(args.output, snapshot)
    print(json.dumps({"output": str(args.output), **snapshot["counts"]}, indent=2))
    return 0


def _rl_config(args: argparse.Namespace) -> int:
    snapshot = read_json(args.task_split)
    treatment = None
    if args.exclusions:
        exclusions = read_json(args.exclusions)
        if exclusions.get("schema") != "fleet_rl_task_exclusions_v1" or not isinstance(
            exclusions.get("exclusions"), list
        ):
            raise ValueError("invalid RL exclusions file")
        if not args.treatment_receipt:
            raise ValueError("--exclusions requires --treatment-receipt")
        treatment = build_treatment_set(snapshot, exclusions["exclusions"])
        atomic_write_json(args.treatment_receipt, treatment)
    elif args.treatment_receipt:
        raise ValueError("--treatment-receipt requires --exclusions")
    config = build_full_rl_config(read_json(args.template), snapshot, treatment)
    write_full_rl_config(args.output, config)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "train_tasks": len(config["tasks"]["task_versions"]),
                "dev_tasks": len(config["eval"]["task_versions"]),
                "treatment_receipt": str(args.treatment_receipt)
                if args.treatment_receipt
                else None,
            },
            indent=2,
        )
    )
    return 0


def _jobs_run(args: argparse.Namespace) -> int:
    bearer = os.environ.get("FLEET_TRAINING_API_TOKEN")
    if not bearer:
        raise ValueError("FLEET_TRAINING_API_TOKEN must be injected through the environment")
    config = load_run_config(args.config)
    with TrainingJobsClient(bearer, base_url=args.base_url) as client:
        if args.execute:
            receipt = client.submit(config)
            print(json.dumps({"submitted": True, **receipt}, indent=2, sort_keys=True))
            return 0
        preview = client.preview(config)
        duplicates = client.runs_with_title(str(config["title"]))
    print(
        json.dumps(
            {
                "submitted": False,
                "kind": config["kind"],
                "title": config["title"],
                "duplicate_names": [row.get("name") for row in duplicates],
                "errors": preview.get("errors") or [],
                "warnings": preview.get("warnings") or [],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _jobs_status(args: argparse.Namespace) -> int:
    bearer = os.environ.get("FLEET_TRAINING_API_TOKEN")
    if not bearer:
        raise ValueError("FLEET_TRAINING_API_TOKEN must be injected through the environment")
    with TrainingJobsClient(bearer, base_url=args.base_url) as client:
        run = client.status(args.name)
    print(json.dumps(concise_status(run), indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Fleet cyber post-training utilities")
    commands = root.add_subparsers(dest="command", required=True)

    export = commands.add_parser("export", help="download blackbox Fleet session transcripts")
    export.add_argument("--job-id", action="append")
    export.add_argument("--job-ids-file", type=Path)
    export.add_argument("--discover-completed-cyber", action="store_true")
    export.add_argument("--created-after", help="ISO-8601 lower bound for discovery")
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--base-url", default="https://orchestrator.fleetai.com")
    export.add_argument("--workers", type=int, default=16)
    export.add_argument("--include-incomplete", action="store_true")
    export.add_argument("--dry-run", action="store_true")
    export.set_defaults(run=_export)

    normalize = commands.add_parser(
        "normalize", help="create SFT, preference and online-RL datasets"
    )
    normalize.add_argument("--input", type=Path, required=True)
    normalize.add_argument("--output-dir", type=Path, required=True)
    normalize.set_defaults(run=_normalize)

    plan = commands.add_parser("plan", help="validate gates and seal a run plan")
    plan.add_argument("--config", type=Path, required=True)
    plan.add_argument("--manifest", type=Path, required=True)
    plan.add_argument("--compatibility-receipt", type=Path, required=True)
    plan.add_argument("--evaluation-protocol", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.set_defaults(run=_plan)

    run = commands.add_parser("run", help="dry-run or execute a sealed run plan")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--work-dir", type=Path, required=True)
    run.add_argument(
        "--execute",
        action="store_true",
        help="execute cluster commands; default only prints the planned argv",
    )
    run.set_defaults(run=_run)

    rl_snapshot = commands.add_parser(
        "rl-snapshot", help="freeze exact Fleet task/environment IDs without task payloads"
    )
    rl_snapshot.add_argument("--trajectories", type=Path, required=True)
    rl_snapshot.add_argument("--dataset-manifest", type=Path, required=True)
    rl_snapshot.add_argument("--source-job-id", required=True)
    rl_snapshot.add_argument("--output", type=Path, required=True)
    rl_snapshot.add_argument("--base-url", default=DEFAULT_BASE_URL)
    rl_snapshot.add_argument("--workers", type=int, default=8)
    rl_snapshot.set_defaults(run=_rl_snapshot)

    rl_config = commands.add_parser(
        "rl-config", help="bind a frozen task split into a full typed RL request"
    )
    rl_config.add_argument("--task-split", type=Path, required=True)
    rl_config.add_argument("--template", type=Path, required=True)
    rl_config.add_argument("--output", type=Path, required=True)
    rl_config.add_argument("--exclusions", type=Path)
    rl_config.add_argument("--treatment-receipt", type=Path)
    rl_config.set_defaults(run=_rl_config)

    jobs_run = commands.add_parser(
        "jobs-run", help="preview or submit a typed SFT/RL request through the Jobs API"
    )
    jobs_run.add_argument("--config", type=Path, required=True)
    jobs_run.add_argument("--base-url", default=DEFAULT_BASE_URL)
    jobs_run.add_argument(
        "--execute",
        action="store_true",
        help="submit after preview and duplicate-title checks; default is preview only",
    )
    jobs_run.set_defaults(run=_jobs_run)

    jobs_status = commands.add_parser("jobs-status", help="read concise Jobs API run status")
    jobs_status.add_argument("name")
    jobs_status.add_argument("--base-url", default=DEFAULT_BASE_URL)
    jobs_status.set_defaults(run=_jobs_status)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return int(args.run(args))
    except (FleetExportError, JobsAPIError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=os.sys.stderr)
        return 2
