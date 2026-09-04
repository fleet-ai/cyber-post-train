"""Stable preview-first command surface for the repository."""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.table import Table

from .catalog import CATALOG, catalog_dict, doctor

app = typer.Typer(
    name="cyber-post-train",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    help="Compose and operate reproducible cyber evaluations and post-training experiments.",
)
console = Console()


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


if __name__ == "__main__":
    app()
