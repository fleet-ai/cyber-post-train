"""Prepare, preflight, preview and submit through one small command surface."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Annotated

import typer

from .jobs import Jobs, JobsError, digest, validate_preview, validate_request

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)


def _print(value: object) -> None:
    typer.echo(json.dumps(value, indent=2, sort_keys=True))


def _write(path: Path, value: dict) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def _read(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("expected an object")
    return value


def _prepared(directory: Path) -> tuple[dict, dict]:
    plan, request, receipt = (
        _read(directory / name) for name in ("plan.json", "request.json", "PREPARED.json")
    )
    if receipt != {"plan_sha256": digest(plan), "request_sha256": digest(request)}:
        raise ValueError("prepared inputs changed; prepare a new directory, never edit a launch")
    validate_request(request)
    return plan, request


def _client() -> Jobs:
    return Jobs(os.environ.get("FLEET_TRAINING_API_TOKEN", ""))


def _fail(exc: Exception) -> None:
    # SDK errors can embed source records or secrets. Never emit rich tracebacks.
    message = str(exc) if isinstance(exc, JobsError) else type(exc).__name__
    typer.echo(f"Stopped: {message}. No automatic retry.", err=True)
    raise typer.Exit(2) from None


@app.command()
def data(config: Path) -> None:
    """Prepare private dense SFT data from a frozen split. CPU only; no submission."""
    from training.corpus import build
    from training.sft import read_mapping

    try:
        _print(build(read_mapping(config), relative_to=config.resolve().parent))
    except Exception as exc:
        _fail(exc)


@app.command()
def train(config: Path, output: Annotated[Path, typer.Option("--output")]) -> None:
    """Prepare an immutable SkyRL SFT launch from editable YAML. No network/GPU."""
    from training.sft import compile_sft, job_request, read_mapping

    try:
        plan = compile_sft(read_mapping(config), relative_to=config.resolve().parent)
        request = job_request(plan)
        output.mkdir(parents=True, exist_ok=False, mode=0o700)
        _write(output / "plan.json", plan)
        _write(output / "request.json", request)
        _write(
            output / "PREPARED.json",
            {"plan_sha256": digest(plan), "request_sha256": digest(request)},
        )
        _print(
            {
                "prepared": str(output),
                "model": plan["model"]["repo"],
                "steps": plan["recipe"]["max_steps"],
                "gpus": request["workers"] * request["gpus_per_worker"],
                "submitted": False,
            }
        )
    except Exception as exc:
        _fail(exc)


@app.command()
def preflight(directory: Path) -> None:
    """Validate staged SFT data and runtime in the pinned image, without GPUs."""
    from training.sft import preflight as check

    try:
        plan, _ = _prepared(directory)
        if (directory / "PREFLIGHT.json").exists():
            raise ValueError("preflight already recorded")
        receipt = check(plan)
        _write(directory / "PREFLIGHT.json", {**receipt, "sha256": digest(receipt)})
        _print(receipt)
    except Exception as exc:
        _fail(exc)


@app.command()
def preview(directory: Path) -> None:
    """Read the Jobs API's exact resource/queue render; does not create a run."""
    try:
        _, request = _prepared(directory)
        with _client() as client:
            result = client.preview(request)
        _print({"submitted": False, **validate_preview(request, result)})
    except Exception as exc:
        _fail(exc)


@app.command()
def submit(directory: Path) -> None:
    """Submit once after CPU preflight and operator approval/resource checks.

    Keep this directory on the experiment's shared durable storage. A timeout
    leaves SUBMISSION.jsonl: reconcile it instead of copying or deleting it.
    """
    try:
        plan, request = _prepared(directory)
        proof = _read(directory / "PREFLIGHT.json")
        expected = {
            "schema": "cyber_sft_cpu_preflight_v1",
            "status": "passed",
            "gpus": 0,
            "plan_sha256": digest(plan),
            "request_sha256": digest(request),
        }
        if proof.get("sha256") != digest({k: v for k, v in proof.items() if k != "sha256"}) or any(
            proof.get(k) != v for k, v in expected.items()
        ):
            raise ValueError("missing or mismatched CPU preflight")
        with _client() as client:
            result = client.submit_once(request, directory / "SUBMISSION.jsonl")
        _print(result)
    except Exception as exc:
        _fail(exc)


@app.command()
def status(name: str) -> None:
    """Read sanitized Jobs API state. Does not return private trainer logs."""
    try:
        with _client() as client:
            _print(client.status(name))
    except Exception as exc:
        _fail(exc)


@app.command()
def doctor() -> None:
    """Check installed modules only; NOT model, cluster or scientific readiness."""
    checks = {
        name: importlib.util.find_spec(name) is not None
        for name in ("training", "evals", "httpx", "psycopg", "pyarrow", "torch")
    }
    _print(
        {
            "ok": all(checks.values()),
            "checks": checks,
            "cluster_checked": False,
            "model_qualified": False,
        }
    )
    if not all(checks.values()):
        raise typer.Exit(2)


if __name__ == "__main__":
    app()
