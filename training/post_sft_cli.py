"""Offline CLI for freezing and rendering the post-SFT evaluation handoff."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Annotated, Any

import typer

from evals.exploitgym.paired import derive_paired_protocol
from evals.webexploitbench.paired import derive_post_sft_qwen_pair

from .io import file_sha256
from .post_sft import (
    assemble_hf_export_receipt,
    build_fleet_test_holdout_receipt,
    build_post_sft_comparison_receipt,
    build_zero_step_hf_export_request,
    derive_post_sft_registration,
    derive_webexploit_config,
    freeze_final_promoted_checkpoint,
    freeze_final_promoted_sfs_checkpoint,
)
from .post_sft_export_observation import collect_zero_step_export_run_observation
from .post_sft_staging import STAGING_COMMAND_SHA256, STAGING_IMAGE

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)


def _atomic_write_json_new(path: Path, value: Any) -> None:
    """Publish one immutable JSON file without replacing any existing path or symlink."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        raise ValueError(f"refusing to replace pre-existing output path: {path}")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ValueError(f"refusing to replace pre-existing output path: {path}") from exc
    finally:
        with suppress(FileNotFoundError):
            os.unlink(temporary)


def _require_outputs_absent(paths: list[Path]) -> None:
    collisions = [str(path) for path in paths if os.path.lexists(path)]
    if collisions:
        raise ValueError("refusing pre-existing output path(s): " + ", ".join(collisions))


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
    _atomic_write_json_new(output, receipt)
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
    _atomic_write_json_new(output, receipt)
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
    _require_outputs_absent([request_output, receipt_output])
    _atomic_write_json_new(request_output, receipt["request"])
    _atomic_write_json_new(receipt_output, receipt)
    typer.echo(str(receipt_output))


@app.command("assemble-export")
def assemble_export(
    plan_path: Annotated[Path, typer.Option("--plan")],
    selection_path: Annotated[Path, typer.Option("--selection")],
    export_request_path: Annotated[Path, typer.Option("--export-request")],
    export_run_observation_path: Annotated[Path, typer.Option("--export-run-observation")],
    export_observation_path: Annotated[Path, typer.Option("--export-observation")],
    cast_receipt_path: Annotated[Path, typer.Option("--cast-receipt")],
    cast_full_manifest_path: Annotated[Path, typer.Option("--cast-full-manifest")],
    stage_input_path: Annotated[Path, typer.Option("--stage-input")],
    staging_receipt_path: Annotated[Path, typer.Option("--staging-receipt")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Assemble and validate the final immutable HF export receipt."""

    if os.path.lexists(output):
        raise ValueError(f"refusing to replace pre-existing output path: {output}")
    plan = _plan(plan_path)
    model = plan["base_model"]
    export = plan["export"]
    if plan["serving"]["engine_image"] != STAGING_IMAGE:
        raise ValueError("staging image differs from the digest-pinned image in the plan")
    receipt = assemble_hf_export_receipt(
        _read(selection_path),
        _read(export_request_path),
        _read(export_run_observation_path),
        _read(export_observation_path),
        _read(cast_receipt_path),
        _read(cast_full_manifest_path),
        _read(stage_input_path),
        _read(staging_receipt_path),
        expected_tokenizer_manifest_sha256=str(model["tokenizer_manifest_sha256"]),
        expected_chat_template_sha256=str(model["chat_template_sha256"]),
        expected_config_sha256=str(model["config_sha256"]),
        expected_export_binding=export,
        expected_runtime_sidecar_sha256=model["runtime_sidecar_sha256"],
        expected_tokenizer_equivalence_evidence_sha256=str(
            model["tokenizer_equivalence_evidence"]["sha256"]
        ),
        expected_cast_execution=plan["cast_execution"],
        expected_staging_image=STAGING_IMAGE,
        expected_staging_command_sha256=STAGING_COMMAND_SHA256,
    )
    _atomic_write_json_new(output, receipt)
    typer.echo(str(output))


@app.command("collect-export-run")
def collect_export_run(
    export_request_path: Annotated[Path, typer.Option("--export-request")],
    terminal_rayjob_path: Annotated[Path, typer.Option("--terminal-rayjob")],
    runtime_raycluster_path: Annotated[Path, typer.Option("--runtime-raycluster")],
    runtime_pod_path: Annotated[Path, typer.Option("--runtime-pod")],
    submitter_job_path: Annotated[Path, typer.Option("--submitter-job")],
    submitter_pod_path: Annotated[Path, typer.Option("--submitter-pod")],
    api_run_path: Annotated[Path, typer.Option("--api-run")],
    driver_log_path: Annotated[Path, typer.Option("--driver-log")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Build terminal export evidence from read-only Kubernetes/API observations."""

    if os.path.lexists(output):
        raise ValueError(f"refusing to replace pre-existing output path: {output}")
    receipt = collect_zero_step_export_run_observation(
        _read(export_request_path),
        _read(terminal_rayjob_path),
        _read(runtime_raycluster_path),
        _read(runtime_pod_path),
        _read(submitter_job_path),
        _read(submitter_pod_path),
        _read(api_run_path),
        driver_log_path.read_bytes(),
    )
    _atomic_write_json_new(output, receipt)
    typer.echo(str(output))


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
    baseline_terminal_path, _ = _planned_file(
        root,
        plan["webexploitbench"]["baseline_terminal_receipt"],
        "baseline WebExploitBench terminal receipt",
    )
    baseline_protocol_path, _ = _planned_file(
        root,
        plan["webexploitbench"]["baseline_protocol"],
        "baseline WebExploitBench protocol",
    )
    harness_lock_path, _ = _planned_file(
        root, plan["webexploitbench"]["harness_lock"], "Qwen Code harness lock"
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
        expected_runtime_sidecar_sha256=model["runtime_sidecar_sha256"],
    )
    served_model_id = serving["registration"]["id"]
    post_web = derive_webexploit_config(
        base_web,
        served_model_id=served_model_id,
        run_id=str(plan["webexploitbench"]["post_run_id"]),
    )
    holdout = build_fleet_test_holdout_receipt(split, sft)

    serving_out = output_dir / "serving-registration-receipt.json"
    post_web_out = output_dir / "webexploitbench-post-sft-config.json"
    holdout_out = output_dir / "fleet-test-holdout-receipt.json"
    protocol_out = output_dir / "webexploitbench-post-sft-protocol.json"
    paired_out = output_dir / "webexploitbench-paired-identity-receipt.json"
    comparison_out = output_dir / "comparison-receipt.json"
    _require_outputs_absent(
        [serving_out, post_web_out, holdout_out, protocol_out, paired_out, comparison_out]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, value in (
        (serving_out, serving),
        (post_web_out, post_web),
        (holdout_out, holdout),
    ):
        _atomic_write_json_new(path, value)

    post_protocol, paired_identity = derive_post_sft_qwen_pair(
        baseline_terminal_path=baseline_terminal_path,
        baseline_protocol_path=baseline_protocol_path,
        baseline_config_path=base_web_path,
        post_config_path=post_web_out,
        harness_lock_path=harness_lock_path,
        base_registration=base_registration,
        post_serving_receipt=serving,
    )
    _atomic_write_json_new(protocol_out, post_protocol.to_dict())
    _atomic_write_json_new(paired_out, paired_identity)

    comparison = build_post_sft_comparison_receipt(
        selection=selection,
        export=export,
        serving=serving,
        base_webexploit_config_sha256=file_sha256(base_web_path),
        post_webexploit_config_sha256=file_sha256(post_web_out),
        webexploit_paired_identity=paired_identity,
        fleet_holdout=holdout,
    )
    _atomic_write_json_new(comparison_out, comparison)
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
    base_web_path, base_web = _planned_file(
        root, plan["webexploitbench"]["base_config"], "base WebExploitBench config"
    )
    baseline_terminal_path, _ = _planned_file(
        root,
        plan["webexploitbench"]["baseline_terminal_receipt"],
        "baseline WebExploitBench terminal receipt",
    )
    baseline_protocol_path, _ = _planned_file(
        root,
        plan["webexploitbench"]["baseline_protocol"],
        "baseline WebExploitBench protocol",
    )
    harness_lock_path, _ = _planned_file(
        root, plan["webexploitbench"]["harness_lock"], "Qwen Code harness lock"
    )
    serving = derive_post_sft_registration(
        base_registration,
        selection,
        export,
        expected_tokenizer_manifest_sha256=str(model["tokenizer_manifest_sha256"]),
        expected_chat_template_sha256=str(model["chat_template_sha256"]),
        expected_config_sha256=str(model["config_sha256"]),
        expected_export_binding=plan["export"],
        expected_runtime_sidecar_sha256=model["runtime_sidecar_sha256"],
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

    post_web_path = output_dir / "webexploitbench-post-sft-config.json"
    serving_out = output_dir / "serving-registration-receipt.json"
    exploitgym_out = output_dir / "exploitgym-paired-protocol.json"
    protocol_out = output_dir / "webexploitbench-post-sft-protocol.json"
    paired_out = output_dir / "webexploitbench-paired-identity-receipt.json"
    _require_outputs_absent(
        [serving_out, post_web_path, exploitgym_out, protocol_out, paired_out]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, value in (
        (serving_out, serving),
        (post_web_path, web),
        (exploitgym_out, exploitgym),
    ):
        _atomic_write_json_new(path, value)
    post_protocol, paired_identity = derive_post_sft_qwen_pair(
        baseline_terminal_path=baseline_terminal_path,
        baseline_protocol_path=baseline_protocol_path,
        baseline_config_path=base_web_path,
        post_config_path=post_web_path,
        harness_lock_path=harness_lock_path,
        base_registration=base_registration,
        post_serving_receipt=serving,
    )
    _atomic_write_json_new(protocol_out, post_protocol.to_dict())
    _atomic_write_json_new(paired_out, paired_identity)
    typer.echo(str(output_dir))


if __name__ == "__main__":
    app()
