"""CLI for dry-running and launching the GLM-5.2 Fleet baseline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from .client import (
    FleetEvalError,
    authenticated_client,
    fetch_source_task_keys,
    launch_batch,
    verify_fleet_team,
    verify_gateway_model,
)
from .events import append_event
from .models import (
    DEFAULT_EXPERIMENT,
    DEFAULT_GATEWAY_MODEL,
    DEFAULT_RUNTIME_LABEL,
    DEFAULT_RUNTIME_MODEL,
    DEFAULT_SMOKE_TASK_KEY,
    DEFAULT_SOURCE_JOB_ID,
    build_plan,
)

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)


def _print(value: object) -> None:
    typer.echo(json.dumps(value, indent=2, sort_keys=True))


def _source_keys(client: object, source_job_id: str, expected_count: int) -> tuple[str, ...]:
    return fetch_source_task_keys(client, source_job_id, expected_count=expected_count)  # type: ignore[arg-type]


@app.command()
def preflight() -> None:
    """Verify Fleet-team identity and both names of the served model."""

    try:
        with authenticated_client() as client:
            account = verify_fleet_team(client)
            gateway = verify_gateway_model(client)
        _print({"ok": True, "account": account, "inference": gateway})
    except FleetEvalError as exc:
        typer.echo(f"preflight failed: {exc}", err=True)
        raise typer.Exit(2) from None


@app.command("plan")
def plan_command(
    source_job_id: Annotated[str, typer.Option()] = DEFAULT_SOURCE_JOB_ID,
    expected_count: Annotated[int, typer.Option(min=1)] = 160,
    limit: Annotated[int | None, typer.Option(min=1)] = None,
    task_key: Annotated[str | None, typer.Option()] = None,
    pass_k: Annotated[int, typer.Option(min=1)] = 1,
    batch_session_cap: Annotated[int, typer.Option(min=1, max=6)] = 6,
    max_steps: Annotated[int, typer.Option(min=1)] = 300,
    max_duration_minutes: Annotated[int, typer.Option(min=1)] = 120,
    runtime_model: Annotated[str, typer.Option()] = DEFAULT_RUNTIME_MODEL,
    harness: Annotated[str | None, typer.Option()] = None,
    experiment: Annotated[str, typer.Option()] = DEFAULT_EXPERIMENT,
    gateway_model: Annotated[str, typer.Option()] = DEFAULT_GATEWAY_MODEL,
    runtime_label: Annotated[str, typer.Option()] = DEFAULT_RUNTIME_LABEL,
    run_name: Annotated[str, typer.Option()] = "glm52-fleet-blackbox-baseline-v1",
) -> None:
    """Print a secret-free plan. This command never launches a job."""

    try:
        with authenticated_client() as client:
            verify_fleet_team(client)
            keys = _source_keys(client, source_job_id, expected_count)
        if task_key:
            if task_key not in keys:
                raise FleetEvalError("requested task key is not in the source job")
            keys = (task_key,)
        elif limit:
            keys = keys[:limit]
        plan = build_plan(
            keys,
            source_job_id=source_job_id,
            runtime_model=runtime_model,
            harness=harness,
            experiment=experiment,
            gateway_model=gateway_model,
            runtime_label=runtime_label,
            pass_k=pass_k,
            batch_session_cap=batch_session_cap,
            max_steps=max_steps,
            max_duration_minutes=max_duration_minutes,
            run_name=run_name,
        )
        _print(plan.sanitized_dict())
    except (FleetEvalError, ValueError) as exc:
        typer.echo(f"plan failed: {exc}", err=True)
        raise typer.Exit(2) from None


@app.command()
def launch(
    submit: Annotated[bool, typer.Option("--submit", help="Actually create paid jobs.")] = False,
    source_job_id: Annotated[str, typer.Option()] = DEFAULT_SOURCE_JOB_ID,
    expected_count: Annotated[int, typer.Option(min=1)] = 160,
    smoke: Annotated[
        bool,
        typer.Option("--smoke/--full", help="Select the default smoke task or the corpus slice."),
    ] = True,
    limit: Annotated[int, typer.Option(min=1)] = 1,
    task_key: Annotated[str | None, typer.Option()] = None,
    pass_k: Annotated[int, typer.Option(min=1)] = 1,
    batch_session_cap: Annotated[int, typer.Option(min=1, max=6)] = 6,
    max_steps: Annotated[int, typer.Option(min=1)] = 300,
    max_duration_minutes: Annotated[int, typer.Option(min=1)] = 120,
    runtime_model: Annotated[str, typer.Option()] = DEFAULT_RUNTIME_MODEL,
    harness: Annotated[str | None, typer.Option()] = None,
    experiment: Annotated[str, typer.Option()] = DEFAULT_EXPERIMENT,
    gateway_model: Annotated[str, typer.Option()] = DEFAULT_GATEWAY_MODEL,
    runtime_label: Annotated[str, typer.Option()] = DEFAULT_RUNTIME_LABEL,
    run_name: Annotated[str, typer.Option()] = "glm52-fleet-blackbox-smoke-v1",
    event_log: Annotated[Path, typer.Option()] = Path("runs/fleet/events.jsonl"),
) -> None:
    """Launch a bounded eval; dry-run unless --submit is explicitly supplied."""

    try:
        with authenticated_client() as client:
            verify_fleet_team(client)
            keys = _source_keys(client, source_job_id, expected_count)
            selected_task = task_key or (DEFAULT_SMOKE_TASK_KEY if smoke else None)
            if selected_task:
                if selected_task not in keys:
                    raise FleetEvalError("requested task key is not in the source job")
                keys = (selected_task,)
            else:
                keys = keys[:limit]
            plan = build_plan(
                keys,
                source_job_id=source_job_id,
                runtime_model=runtime_model,
                harness=harness,
                experiment=experiment,
                gateway_model=gateway_model,
                runtime_label=runtime_label,
                pass_k=pass_k,
                batch_session_cap=batch_session_cap,
                max_steps=max_steps,
                max_duration_minutes=max_duration_minutes,
                run_name=run_name,
            )
            if not submit:
                _print({"dry_run": True, **plan.sanitized_dict()})
                return

            receipts = []
            for batch in plan.batches:
                receipt = launch_batch(client, batch)
                event = {
                    "event": "fleet_job_launched",
                    "source_job_id": source_job_id,
                    "source_task_digest": plan.source_task_digest,
                    "batch": batch.sanitized_dict(),
                    **receipt,
                }
                append_event(event_log, event)
                receipts.append(
                    {
                        **receipt,
                        "dashboard": f"https://fleetai.com/dashboard/jobs/{receipt['job_id']}",
                    }
                )
        _print({"dry_run": False, "receipts": receipts})
    except (FleetEvalError, ValueError) as exc:
        typer.echo(f"launch failed: {exc}", err=True)
        raise typer.Exit(2) from None


if __name__ == "__main__":
    app()
