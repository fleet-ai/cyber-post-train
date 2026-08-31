"""Offline CLI for freezing and rendering the post-SFT evaluation handoff."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from evals.exploitgym.paired import derive_paired_protocol

from .io import atomic_write_json, file_sha256
from .post_sft import (
    build_fleet_test_holdout_receipt,
    build_post_sft_comparison_receipt,
    build_zero_step_hf_export_request,
    derive_post_sft_registration,
    derive_webexploit_config,
    freeze_final_promoted_checkpoint,
    freeze_final_promoted_sfs_checkpoint,
)

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _plan(path: Path) -> dict[str, Any]:
    value = _read(path)
    if value.get("schema") != "cyber_post_sft_eval_plan_v1":
        raise ValueError("unsupported post-SFT evaluation plan schema")
    return value


def _planned_file(root: Path, binding: dict[str, Any], field: str) -> tuple[Path, dict[str, Any]]:
    path = root / str(binding["path"])
    expected = str(binding["sha256"])
    if file_sha256(path) != expected:
        raise ValueError(f"{field} no longer matches its frozen digest")
    return path, _read(path)


@app.command()
def freeze(
    plan_path: Annotated[Path, typer.Option("--plan")],
    run_observation: Annotated[Path, typer.Option()],
    checkpoints: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()],
) -> None:
    """Select the final promoted checkpoint from captured read-only API observations."""

    plan = _plan(plan_path)
    expected = plan["run"]
    rows = json.loads(checkpoints.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("checkpoints must contain a JSON array of objects")
    receipt = freeze_final_promoted_checkpoint(
        _read(run_observation),
        rows,
        expected_run_name=str(expected["name"]),
        expected_run_config_sha256=str(expected["config_sha256"]),
        expected_rayjob_uid=str(expected["rayjob_uid"]),
        expected_trainer_image=str(expected["trainer_image"]),
        expected_entrypoint_sha256=str(expected["entrypoint_sha256"]),
    )
    atomic_write_json(output, receipt, private=True)
    typer.echo(str(output))


@app.command("freeze-sfs")
def freeze_sfs(
    plan_path: Annotated[Path, typer.Option("--plan")],
    run_observation: Annotated[Path, typer.Option()],
    sfs_observation: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()],
) -> None:
    """Select the final promoted SFS checkpoint when the API index was intentionally disabled."""

    plan = _plan(plan_path)
    expected = plan["run"]
    receipt = freeze_final_promoted_sfs_checkpoint(
        _read(run_observation),
        _read(sfs_observation),
        expected_run_name=str(expected["name"]),
        expected_run_config_sha256=str(expected["config_sha256"]),
        expected_rayjob_uid=str(expected["rayjob_uid"]),
        expected_trainer_image=str(expected["trainer_image"]),
        expected_entrypoint_sha256=str(expected["entrypoint_sha256"]),
        expected_pipeline=plan["source_checkpoint_evidence"]["checkpoint_pipeline"],
        expected_structural_manifest_before_sha256=str(
            plan["source_checkpoint_evidence"]["structural_manifest_before_sha256"]
        ),
    )
    atomic_write_json(output, receipt, private=True)
    typer.echo(str(output))


@app.command()
def render_export(
    plan_path: Annotated[Path, typer.Option("--plan")],
    selection_path: Annotated[Path, typer.Option("--selection")],
    request_output: Annotated[Path, typer.Option("--request-output")],
    receipt_output: Annotated[Path, typer.Option("--receipt-output")],
) -> None:
    """Render, but never submit, a zero-optimizer-step HF export request."""

    plan = _plan(plan_path)
    root = plan_path.resolve().parents[2]
    _, sft = _planned_file(root, plan["fleet"]["sft_config"], "SFT config")
    run = plan["run"]
    receipt = build_zero_step_hf_export_request(
        sft,
        _read(selection_path),
        expected_trainer_version_id=str(run["trainer_version_id"]),
        expected_trainer_image=str(run["trainer_image"]),
        expected_export_binding=plan["export"],
    )
    atomic_write_json(request_output, receipt["request"], private=True)
    atomic_write_json(receipt_output, receipt, private=True)
    typer.echo(str(receipt_output))


@app.command()
def render(
    plan_path: Annotated[Path, typer.Option("--plan")],
    selection_path: Annotated[Path, typer.Option("--selection")],
    export_path: Annotated[Path, typer.Option("--export")],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
) -> None:
    """Render serving and paired-evaluation receipts after a verified HF export exists."""

    plan = _plan(plan_path)
    root = plan_path.resolve().parents[2]
    selection = _read(selection_path)
    export = _read(export_path)
    model = plan["base_model"]

    _, base_registration = _planned_file(
        root, plan["serving"]["base_registration"], "base serving registration"
    )
    base_web_path, base_web = _planned_file(
        root, plan["webexploitbench"]["base_config"], "base WebExploitBench config"
    )
    _, split = _planned_file(root, plan["fleet"]["split_manifest"], "Fleet split manifest")
    _, sft = _planned_file(root, plan["fleet"]["sft_config"], "SFT config")

    serving = derive_post_sft_registration(
        base_registration,
        selection,
        export,
        expected_tokenizer_manifest_sha256=str(model["tokenizer_manifest_sha256"]),
        expected_chat_template_sha256=str(model["chat_template_sha256"]),
        expected_config_sha256=str(model["config_sha256"]),
        expected_export_binding=plan["export"],
    )
    served_model_id = serving["registration"]["id"]
    post_web = derive_webexploit_config(
        base_web,
        served_model_id=served_model_id,
        run_id=str(plan["webexploitbench"]["post_run_id"]),
    )
    holdout = build_fleet_test_holdout_receipt(split, sft)

    output_dir.mkdir(parents=True, exist_ok=True)
    serving_out = output_dir / "serving-registration-receipt.json"
    post_web_out = output_dir / "webexploitbench-post-sft-config.json"
    holdout_out = output_dir / "fleet-test-holdout-receipt.json"
    for path, value in (
        (serving_out, serving),
        (post_web_out, post_web),
        (holdout_out, holdout),
    ):
        atomic_write_json(path, value, private=True)

    comparison = build_post_sft_comparison_receipt(
        selection=selection,
        export=export,
        serving=serving,
        base_webexploit_config_sha256=file_sha256(base_web_path),
        post_webexploit_config_sha256=file_sha256(post_web_out),
        fleet_holdout=holdout,
    )
    atomic_write_json(output_dir / "comparison-receipt.json", comparison, private=True)
    typer.echo(str(output_dir))


@app.command("render-external-benchmarks")
def render_external_benchmarks(
    plan_path: Annotated[Path, typer.Option("--plan")],
    selection_path: Annotated[Path, typer.Option("--selection")],
    export_path: Annotated[Path, typer.Option("--export")],
    control_image_path: Annotated[Path, typer.Option("--control-image")],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
) -> None:
    """Freeze WebExploitBench and ExploitGym inputs after export and staging."""

    plan = _plan(plan_path)
    root = plan_path.resolve().parents[2]
    selection = _read(selection_path)
    export = _read(export_path)
    model = plan["base_model"]
    _, base_registration = _planned_file(
        root, plan["serving"]["base_registration"], "base serving registration"
    )
    _, base_web = _planned_file(
        root, plan["webexploitbench"]["base_config"], "base WebExploitBench config"
    )
    serving = derive_post_sft_registration(
        base_registration,
        selection,
        export,
        expected_tokenizer_manifest_sha256=str(model["tokenizer_manifest_sha256"]),
        expected_chat_template_sha256=str(model["chat_template_sha256"]),
        expected_config_sha256=str(model["config_sha256"]),
        expected_export_binding=plan["export"],
    )
    web = derive_webexploit_config(
        base_web,
        served_model_id=str(serving["registration"]["id"]),
        run_id=str(plan["webexploitbench"]["post_run_id"]),
    )
    base_exploitgym = _read(root / "evals/exploitgym/configs/qwen36-27b-v1-pilot.json")
    exploitgym = derive_paired_protocol(
        base_exploitgym,
        selection,
        export,
        serving,
        _read(control_image_path),
        expected_export_binding=plan["export"],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for path, value in (
        (output_dir / "serving-registration-receipt.json", serving),
        (output_dir / "webexploitbench-post-sft-config.json", web),
        (output_dir / "exploitgym-paired-protocol.json", exploitgym),
    ):
        atomic_write_json(path, value, private=True)
    typer.echo(str(output_dir))


if __name__ == "__main__":
    app()
