"""Stable preview-first command surface for the repository."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
import yaml
from rich.console import Console
from rich.table import Table

from .catalog import CATALOG, catalog_dict, doctor
from .spec import build_source_spec, compile_plan, load_locked_spec, lock_spec, write_new

app = typer.Typer(
    name="cyber-post-train",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    help="Compose and operate reproducible cyber evaluations and post-training experiments.",
)
console = Console()
experiment_app = typer.Typer(no_args_is_help=True, help="Lock, validate, and compile experiments.")
app.add_typer(experiment_app, name="experiment")


@experiment_app.command("init")
def experiment_init(
    experiment_id: str,
    output: Annotated[Path, typer.Option("--output")],
    adapter: Annotated[
        str,
        typer.Option(help="fleet, webexploitbench, exploitgym, fleet-sft, or fleet-rl"),
    ],
    model: Annotated[Path, typer.Option("--model")],
    dataset: Annotated[Path, typer.Option("--dataset")],
    protocol: Annotated[Path, typer.Option("--protocol")],
    serving: Annotated[Path | None, typer.Option("--serving")] = None,
    harness: Annotated[Path | None, typer.Option("--harness")] = None,
    trainer: Annotated[Path | None, typer.Option("--trainer")] = None,
    backend: Annotated[str, typer.Option("--backend")] = "fleet",
    output_root: Annotated[Path | None, typer.Option("--output-root")] = None,
    max_concurrency: Annotated[int, typer.Option("--max-concurrency", min=1)] = 1,
    serving_block: Annotated[
        str | None,
        typer.Option("--serving-block", help="hosted, dedicated, or matched-self-hosted"),
    ] = None,
) -> None:
    """Create a minimal, editable experiment composition without overwriting files."""

    components = {
        "model": str(model),
        "dataset": str(dataset),
        "protocol": str(protocol),
        "serving": str(serving) if serving else None,
        "harness": str(harness) if harness else None,
        "trainer": str(trainer) if trainer else None,
    }
    value = build_source_spec(
        experiment_id=experiment_id,
        adapter=adapter,
        component_paths=components,
        backend=backend,
        output_root=str(output_root or Path("output") / experiment_id),
        max_concurrency=max_concurrency,
        serving_block=serving_block,
    )
    write_new(output, yaml.safe_dump(value, sort_keys=False))
    typer.echo(str(output))


@app.command("catalog")
def catalog_command(
    as_json: bool = typer.Option(False, "--json", help="Print the machine-readable catalog."),
) -> None:
    """List supported evaluation and training adapters."""

    if as_json:
        typer.echo(json.dumps(catalog_dict(), indent=2, sort_keys=True))
        return
    table = Table(title="cyber-post-train components")
    table.add_column("Kind")
    table.add_column("Name")
    table.add_column("Lifecycle")
    table.add_column("Implementation")
    for component in CATALOG:
        table.add_row(
            component.kind,
            component.name,
            " → ".join(component.lifecycle),
            component.implementation,
        )
    console.print(table)


@app.command("doctor")
def doctor_command(
    as_json: bool = typer.Option(False, "--json", help="Print the machine-readable receipt."),
) -> None:
    """Verify the local interface without credentials or network access."""

    receipt = doctor()
    if as_json:
        typer.echo(json.dumps(receipt, indent=2, sort_keys=True))
    else:
        for name, passed in receipt["checks"].items():
            color = "green" if passed else "red"
            result = "PASS" if passed else "FAIL"
            console.print(f"[{color}]{result}[/] {name}")
    if not receipt["ok"]:
        raise typer.Exit(2)


@experiment_app.command("lock")
def experiment_lock(
    source: Path,
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Resolve component digests and create a locked experiment spec."""

    locked = lock_spec(source)
    write_new(output, yaml.safe_dump(locked, sort_keys=False))
    typer.echo(str(output))


@experiment_app.command("validate")
def experiment_validate(spec: Path) -> None:
    """Validate a locked experiment and every referenced component digest."""

    value = load_locked_spec(spec)
    typer.echo(json.dumps({"ok": True, "experiment_id": value.experiment_id}, sort_keys=True))


@experiment_app.command("compile")
def experiment_compile(
    spec: Path,
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Create an immutable backend-neutral launch plan."""

    plan = compile_plan(spec)
    write_new(output, json.dumps(plan, indent=2, sort_keys=True) + "\n")
    typer.echo(json.dumps({"output": str(output), "plan_sha256": plan["plan_sha256"]}))


if __name__ == "__main__":
    app()
