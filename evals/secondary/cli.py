"""Dry-run-first CLI for XBEN and CVE-Bench fallback evaluations."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Annotated

import typer

from .config import SecondaryEvalConfig
from .pins import BENCHMARK_PINS
from .plan import build_plan
from .workspace import (
    WorkspaceError,
    materialize_checkout,
    prepare_xben_compose,
    verify_checkout,
)

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _print(value: object) -> None:
    typer.echo(json.dumps(value, indent=2, sort_keys=True))


@app.command()
def plan(
    config_path: Annotated[Path, typer.Option("--config")],
    checkout: Annotated[Path, typer.Option()],
) -> None:
    """Print a reproducible plan; never clones, starts containers or calls a model."""

    try:
        config = SecondaryEvalConfig.load(config_path)
        _print(build_plan(config, checkout=checkout, project_root=PROJECT_ROOT).to_dict())
    except (OSError, ValueError) as exc:
        typer.echo(f"plan failed: {exc}", err=True)
        raise typer.Exit(2) from None


@app.command()
def bootstrap(
    config_path: Annotated[Path, typer.Option("--config")],
    checkout: Annotated[Path, typer.Option()],
    execute: Annotated[bool, typer.Option("--execute")] = False,
) -> None:
    """Materialize the exact official commit; dry-run unless --execute is present."""

    try:
        config = SecondaryEvalConfig.load(config_path)
        pin = BENCHMARK_PINS[config.benchmark]
        if not execute:
            _print(
                {
                    "dry_run": True,
                    "checkout": str(checkout.resolve()),
                    "upstream": pin.to_dict(),
                }
            )
            return
        materialize_checkout(checkout.resolve(), pin)
        _print({"dry_run": False, "verified": True, "commit": pin.commit})
    except (OSError, ValueError, WorkspaceError) as exc:
        typer.echo(f"bootstrap failed: {exc}", err=True)
        raise typer.Exit(2) from None


@app.command("run")
def run_eval(
    config_path: Annotated[Path, typer.Option("--config")],
    checkout: Annotated[Path, typer.Option()],
    execute: Annotated[bool, typer.Option("--execute")] = False,
) -> None:
    """Run one paid, isolated smoke sample; dry-run unless --execute is present."""

    try:
        config = SecondaryEvalConfig.load(config_path)
        plan = build_plan(config, checkout=checkout, project_root=PROJECT_ROOT)
        if not execute:
            _print({"dry_run": True, **plan.to_dict()})
            return
        verify_checkout(checkout.resolve(), plan.upstream)
        if not os.environ.get("FLEET_API_KEY"):
            raise WorkspaceError("FLEET_API_KEY is not set in this shell")
        result_dir = (
            checkout.resolve().parent / ".cyber-post-train-secondary-results" / config.run_id
        )
        if result_dir.exists() and any(result_dir.iterdir()):
            raise WorkspaceError(
                "result directory is non-empty; use a new run_id or Inspect eval-retry"
            )
        result_dir.mkdir(parents=True, exist_ok=True)
        result_dir.chmod(0o700)
        child_env = os.environ.copy()
        # CVE-Bench v2.1.0 pins Inspect 0.3.103, which predates the named
        # openai-api provider convention. Both supported harness versions use
        # the official OpenAI adapter with an overridden base URL.
        child_env["OPENAI_API_KEY"] = child_env["FLEET_API_KEY"]
        child_env["OPENAI_BASE_URL"] = config.openai_base_url
        existing_pythonpath = child_env.get("PYTHONPATH")
        child_env["PYTHONPATH"] = (
            f"{PROJECT_ROOT}{os.pathsep}{existing_pythonpath}"
            if existing_pythonpath
            else str(PROJECT_ROOT)
        )
        if config.benchmark == "xben":
            if not os.environ.get("XBEN_FLAG"):
                raise WorkspaceError("XBEN_FLAG is not set in this shell")
            child_env["FLAG"] = child_env["XBEN_FLAG"]
            compose_path = prepare_xben_compose(
                checkout=checkout.resolve(),
                target=config.target,
                agent_context=PROJECT_ROOT / "evals" / "secondary",
            )
            if f"compose_file={compose_path}" not in plan.command.argv:
                raise WorkspaceError(
                    "generated XBEN compose path disagrees with the immutable plan"
                )
        subprocess.run(
            list(plan.command.argv),
            cwd=plan.command.cwd,
            env=child_env,
            check=True,
            timeout=14_400,
        )
    except (OSError, ValueError, WorkspaceError, subprocess.SubprocessError) as exc:
        typer.echo(f"run failed: {type(exc).__name__}", err=True)
        raise typer.Exit(2) from None
