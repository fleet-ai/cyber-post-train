"""Prepare, preflight, preview and submit through one small command surface."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path
from typing import Annotated

import typer

from .jobs import Jobs, JobsError, digest, plan_api_target, validate_preview, validate_request

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
PREPARATION_GATE_VERSION = 2


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


def _prepare(output: Path, plan: dict, request: dict) -> None:
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    _write(output / "plan.json", plan)
    _write(output / "request.json", request)
    _write(
        output / "PREPARED.json",
        {
            "schema": "cyber_post_train_prepared_v1",
            "gate_version": PREPARATION_GATE_VERSION,
            "plan_sha256": digest(plan),
            "request_sha256": digest(request),
        },
    )


def _prepared(directory: Path) -> tuple[dict, dict]:
    plan, request, receipt = (
        _read(directory / name) for name in ("plan.json", "request.json", "PREPARED.json")
    )
    current = {
        "schema": "cyber_post_train_prepared_v1",
        "gate_version": PREPARATION_GATE_VERSION,
        "plan_sha256": digest(plan),
        "request_sha256": digest(request),
    }
    legacy = {key: current[key] for key in ("plan_sha256", "request_sha256")}
    if receipt not in (current, legacy):
        raise ValueError("prepared inputs changed; prepare a new directory, never edit a launch")
    validate_request(request)
    return plan, request


def _current_request(plan: dict) -> dict:
    """Re-render a prepared request through the current, source-bound backend."""
    schema = plan.get("schema")
    if schema == "cyber_miles_conversion_v1":
        from training.miles_conversion import job_request
    elif schema in {"cyber_miles_training_v1", "cyber_miles_training_v2"}:
        from training.miles_training import job_request
    elif schema == "cyber_skyrl_training_v1":
        from training.skyrl_training import job_request
    elif schema == "cyber_skyrl_production_training_v1":
        from training.skyrl_production_training import job_request
    elif schema == "cyber_skyrl_topology_probe_v1":
        from training.skyrl_topology_probe import request as job_request
    else:
        from training.sft import job_request

    return job_request(plan)


def _submission_gate(directory: Path, plan: dict, request: dict) -> None:
    receipt = _read(directory / "PREPARED.json")
    if receipt != {
        "schema": "cyber_post_train_prepared_v1",
        "gate_version": PREPARATION_GATE_VERSION,
        "plan_sha256": digest(plan),
        "request_sha256": digest(request),
    }:
        raise ValueError("prepared request predates the current submission gate; prepare again")
    if _current_request(plan) != request:
        raise ValueError(
            "prepared request is stale under the current source and gates; prepare again"
        )


def _external_action_gate(plan: dict, action: str) -> None:
    """Keep qualification-blocked profiles away from every external job endpoint."""
    qualification = plan.get("qualification")
    if not isinstance(qualification, dict):
        return
    gate = qualification.get("submission_gate")
    field = {
        "preview": "preview_authorized",
        "submit": "submission_authorized",
    }.get(action)
    if field is None:
        raise ValueError("unknown external action")
    if not isinstance(gate, dict) or type(gate.get(field)) is not bool:
        raise ValueError(f"{action} blocked by an incomplete qualification gate")
    if gate[field] is True:
        return
    blockers = gate.get("blockers")
    if (
        not isinstance(blockers, list)
        or not blockers
        or any(not isinstance(item, str) or not item for item in blockers)
    ):
        raise ValueError(f"{action} blocked by an incomplete qualification gate")
    raise ValueError(f"{action} blocked by qualification gate: {', '.join(blockers)}")


def _client(plan: dict | None = None) -> Jobs:
    """Create the client for the route sealed into ``plan``.

    This function intentionally has no cluster argument.  Moving a launch to a
    different cluster requires a newly prepared plan and therefore new plan,
    request and preflight digests.
    """
    _, base_url = plan_api_target(plan)
    return Jobs(os.environ.get("FLEET_API_KEY", ""), base_url=base_url)


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


@app.command("data-subset")
def data_subset(config: Path) -> None:
    """Subset immutable dense SFT corpora to one frozen train split. CPU only."""
    from training.corpus_subset import build
    from training.sft import read_mapping

    try:
        _print(build(read_mapping(config), relative_to=config.resolve().parent))
    except Exception as exc:
        _fail(exc)


@app.command("data-fleet-teachers")
def data_fleet_teachers(config: Path) -> None:
    """Build train-only SFT data from digest-bound Fleet success evidence. CPU only."""
    from training.fleet_teacher_corpus import build
    from training.sft import read_mapping

    try:
        _print(build(read_mapping(config), relative_to=config.resolve().parent))
    except Exception as exc:
        _fail(exc)


@app.command("model-lock")
def model_lock(repo: str, revision: str, output: Annotated[Path, typer.Option("--output")]) -> None:
    """Pin public HF model metadata. No weights, GPUs, remote code or compatibility claim."""
    import httpx

    from training.models import freeze

    try:
        if output.exists():
            raise ValueError("model-lock output must be new")
        with httpx.Client(timeout=60) as client:
            lock, weights = freeze(repo, revision, client)
        output.mkdir(parents=True, exist_ok=False, mode=0o700)
        _write(output / "model.lock.json", lock)
        _write(output / "model.weights.json", weights)
        receipt = {"lock_sha256": digest(lock), "weights_sha256": digest(weights)}
        _write(output / "COMPLETE.json", receipt)
        _print({**receipt, "shards": lock["weights"]["shards"], "model_qualified": False})
    except Exception as exc:
        _fail(exc)


@app.command("rl-data")
def rl_data(config: Path) -> None:
    """CPU-only Miles/SkyRL data from reviewed Fleet versions. GET only; no training."""
    import httpx

    from training.rl_reward_canary import build
    from training.sft import read_mapping

    try:
        token = os.environ["FLEET_API_KEY"]
        if not token:
            raise ValueError("Fleet API key required")
        with httpx.Client(
            headers={"Authorization": "Bearer " + token},
            timeout=60,
            follow_redirects=False,
            transport=httpx.HTTPTransport(retries=0),
        ) as client:
            _print(build(read_mapping(config), relative_to=config.resolve().parent, client=client))
    except Exception as exc:
        _fail(exc)


@app.command()
def train(config: Path, output: Annotated[Path, typer.Option("--output")]) -> None:
    """Prepare an immutable SkyRL SFT launch from editable YAML. No network/GPU."""
    from training.sft import compile_sft, job_request, read_mapping

    try:
        plan = compile_sft(read_mapping(config), relative_to=config.resolve().parent)
        request = job_request(plan)
        _prepare(output, plan, request)
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
def rl(config: Path, output: Annotated[Path, typer.Option("--output")]) -> None:
    """Prepare native Miles or SkyRL RL. No GPU, environment creation or submission."""
    from training.sft import read_mapping

    try:
        value = read_mapping(config)
        if value["backend"] == "miles":
            from training import miles_training as backend
        elif value["backend"] == "skyrl":
            qualification = Path(str(value.get("qualification", ""))).name
            if qualification == "qwen38-skyrl-production-queue-v1.json":
                from training import skyrl_production_training as backend
            else:
                from training import skyrl_training as backend
        else:
            raise ValueError("unsupported RL backend")
        plan = backend.compile_rl(value, relative_to=config.resolve().parent)
        request = backend.job_request(plan)
        _prepare(output, plan, request)
        if plan.get("schema") == "cyber_skyrl_production_training_v1":
            from training.skyrl_production import offline_preview, release_observer_contract

            _write(
                output / "RELEASE_OBSERVER_CONTRACT.json",
                release_observer_contract(plan, request),
            )
            _write(output / "OFFLINE_PREVIEW.json", offline_preview(plan, request))
        _print(
            {
                "prepared": str(output),
                "steps": plan["arguments"]["steps"],
                "gpus": request["workers"] * request["gpus_per_worker"],
                "submitted": False,
            }
        )
    except Exception as exc:
        _fail(exc)


@app.command("rl-topology-probe")
def rl_topology_probe(config: Path, output: Annotated[Path, typer.Option("--output")]) -> None:
    """Prepare the sealed zero-update one-node development setup probe."""
    from training.skyrl_topology_probe import (
        compile_probe,
        fleetjob_manifest,
        fleetjob_packet,
        preflight_job_manifest,
        preflight_job_packet,
        receipt_verify_job_manifest,
        receipt_verify_job_packet,
        request,
    )
    from training.skyrl_topology_rayjob import manifest as direct_rayjob_manifest
    from training.skyrl_topology_rayjob import packet as direct_rayjob_packet

    try:
        plan = compile_probe(config)
        prepared_request = request(plan)
        _prepare(output, plan, prepared_request)
        manifest = fleetjob_manifest(plan)
        _write(output / "fleetjob.json", manifest)
        _write(output / "FLEETJOB_PREPARED.json", fleetjob_packet(plan))
        _write(output / "preflight-job.json", preflight_job_manifest(plan))
        _write(
            output / "PREFLIGHT_JOB_PREPARED.json",
            preflight_job_packet(plan),
        )
        _write(output / "receipt-verify-job.json", receipt_verify_job_manifest(plan))
        _write(
            output / "RECEIPT_VERIFY_JOB_PREPARED.json",
            receipt_verify_job_packet(plan),
        )
        _write(output / "direct-rayjob.json", direct_rayjob_manifest(plan))
        _write(output / "DIRECT_RAYJOB_PREPARED.json", direct_rayjob_packet(plan))
        _print(
            {
                "prepared": str(output),
                "cluster_target": "dev",
                "submission_transport": "fleetjob",
                "qualification_transport": "direct_rayjob",
                "kubernetes_context": plan["execution"]["kubernetes_context"],
                "gpus": 8,
                "rollout_episodes": 0,
                "optimizer_steps": 0,
                "maximum_seconds": 1500,
                "submission_authorized": False,
                "submitted": False,
            }
        )
    except Exception as exc:
        _fail(exc)


@app.command("rl-topology-probe-rayjob-preview")
def rl_topology_probe_rayjob_preview(directory: Path) -> None:
    """Server-dry-run the alert-safe direct RayJob; create nothing."""
    from training.skyrl_topology_probe import SCHEMA
    from training.skyrl_topology_rayjob import manifest, packet, validate_preview

    try:
        plan, _ = _prepared(directory)
        if plan.get("schema") != SCHEMA:
            raise ValueError("prepared directory is not a topology probe")
        expected = _read(directory / "direct-rayjob.json")
        prepared = _read(directory / "DIRECT_RAYJOB_PREPARED.json")
        if expected != manifest(plan) or prepared != packet(plan):
            raise ValueError("direct RayJob packet changed")
        proof_path = directory / "DIRECT_RAYJOB_PREVIEW.json"
        if proof_path.exists() or proof_path.is_symlink():
            raise ValueError("direct RayJob preview already recorded")
        execution = plan["execution"]
        result = subprocess.run(
            [
                "kubectl",
                "--context",
                execution["kubernetes_context"],
                "--namespace",
                execution["namespace"],
                "create",
                "--dry-run=server",
                "--filename",
                str(directory / "direct-rayjob.json"),
                "--output=json",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode:
            raise JobsError("direct RayJob server dry-run failed; nothing was created")
        proof = validate_preview(plan, expected, json.loads(result.stdout))
        _write(proof_path, proof)
        _print(proof)
    except Exception as exc:
        _fail(exc)


@app.command("rl-topology-probe-rayjob-authorize")
def rl_topology_probe_rayjob_authorize(
    directory: Path,
    cpu_result: Annotated[Path, typer.Option("--cpu-result")],
    observer_armed: Annotated[Path, typer.Option("--observer-armed")],
) -> None:
    """Bind exact released CPU evidence and a live direct-RayJob observer."""
    from training.skyrl_topology_probe import SCHEMA
    from training.skyrl_topology_rayjob import authorize

    try:
        plan, _ = _prepared(directory)
        if plan.get("schema") != SCHEMA:
            raise ValueError("prepared directory is not a topology probe")
        value = authorize(
            plan,
            cpu_result=_read(cpu_result),
            cpu_preview=_read(directory / "PREFLIGHT_JOB_PREVIEW.json"),
            receipt_preview=_read(directory / "RECEIPT_VERIFY_JOB_PREVIEW.json"),
            rayjob_preview=_read(directory / "DIRECT_RAYJOB_PREVIEW.json"),
            observer=_read(observer_armed),
        )
        os.kill(value["observer"]["observer_pid"], 0)
        _write(directory / "DIRECT_RAYJOB_LAUNCH_AUTHORIZED.json", value)
        _print(
            {
                "status": value["status"],
                "name": plan["run_name"],
                "plan_sha256": value["plan_sha256"],
                "manifest_sha256": value["manifest_sha256"],
                "authorization_sha256": value["sha256"],
                "observer_pid": value["observer"]["observer_pid"],
                "submitted": False,
            }
        )
    except Exception as exc:
        _fail(exc)


@app.command("rl-topology-probe-rayjob-create")
def rl_topology_probe_rayjob_create(directory: Path) -> None:
    """Create one authorized direct RayJob; never apply, patch, or retry."""
    from training.skyrl_topology_probe import SCHEMA
    from training.skyrl_topology_rayjob import create_once, manifest, packet, write_once_fsynced

    try:
        plan, _ = _prepared(directory)
        if plan.get("schema") != SCHEMA:
            raise ValueError("prepared directory is not a topology probe")
        expected = _read(directory / "direct-rayjob.json")
        if expected != manifest(plan) or _read(directory / "DIRECT_RAYJOB_PREPARED.json") != packet(
            plan
        ):
            raise ValueError("direct RayJob packet changed")
        token = os.environ.get("FLEET_API_KEY", "")
        if not token:
            raise JobsError("FLEET_API_KEY is required for duplicate history checks")
        result = create_once(
            directory,
            plan,
            expected,
            _read(directory / "DIRECT_RAYJOB_PREVIEW.json"),
            _read(directory / "DIRECT_RAYJOB_LAUNCH_AUTHORIZED.json"),
            token=token,
        )
        write_once_fsynced(directory / "DIRECT_RAYJOB_CREATED.json", result)
        _print(result)
    except Exception as exc:
        _fail(exc)


@app.command("rl-topology-probe-preflight-preview")
def rl_topology_probe_preflight_preview(directory: Path) -> None:
    """Server-dry-run the exact zero-GPU dev preflight; create nothing."""
    from training.skyrl_topology_probe import (
        SCHEMA,
        preflight_job_manifest,
        preflight_job_packet,
        validate_preflight_job_preview,
    )

    try:
        plan, _ = _prepared(directory)
        if plan.get("schema") != SCHEMA:
            raise ValueError("prepared directory is not a topology probe")
        gate = plan.get("qualification", {}).get("submission_gate", {})
        if gate.get("cpu_preflight_authorized") is not True:
            raise ValueError("topology probe CPU preflight is not authorized")
        path = directory / "preflight-job.json"
        manifest = _read(path)
        packet = _read(directory / "PREFLIGHT_JOB_PREPARED.json")
        if manifest != preflight_job_manifest(plan) or packet != preflight_job_packet(plan):
            raise ValueError("topology probe preflight packet changed")
        proof_path = directory / "PREFLIGHT_JOB_PREVIEW.json"
        if proof_path.exists():
            raise ValueError("topology probe preflight preview already recorded")
        execution = plan["execution"]
        result = subprocess.run(
            [
                "kubectl",
                "--context",
                execution["kubernetes_context"],
                "--namespace",
                execution["namespace"],
                "create",
                "--dry-run=server",
                "--filename",
                str(path),
                "--output",
                "json",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode:
            raise JobsError("CPU preflight server dry-run failed; nothing was created")
        proof = validate_preflight_job_preview(plan, manifest, json.loads(result.stdout))
        _write(proof_path, proof)
        _print(proof)
    except Exception as exc:
        _fail(exc)


@app.command("rl-topology-probe-receipt-preview")
def rl_topology_probe_receipt_preview(directory: Path) -> None:
    """Server-dry-run the exact zero-GPU durable-receipt verifier."""
    from training.skyrl_topology_probe import (
        SCHEMA,
        receipt_verify_job_manifest,
        receipt_verify_job_packet,
        validate_receipt_verify_job_preview,
    )

    try:
        plan, _ = _prepared(directory)
        if plan.get("schema") != SCHEMA:
            raise ValueError("prepared directory is not a topology probe")
        gate = plan.get("qualification", {}).get("submission_gate", {})
        if gate.get("fleetjob_preview_authorized") is not True:
            raise ValueError("topology probe receipt preview is not authorized")
        path = directory / "receipt-verify-job.json"
        manifest = _read(path)
        packet = _read(directory / "RECEIPT_VERIFY_JOB_PREPARED.json")
        if manifest != receipt_verify_job_manifest(plan) or packet != receipt_verify_job_packet(
            plan
        ):
            raise ValueError("topology probe receipt-verifier packet changed")
        proof_path = directory / "RECEIPT_VERIFY_JOB_PREVIEW.json"
        if proof_path.exists():
            raise ValueError("topology probe receipt preview already recorded")
        execution = plan["execution"]
        result = subprocess.run(
            [
                "kubectl",
                "--context",
                execution["kubernetes_context"],
                "--namespace",
                execution["namespace"],
                "create",
                "--dry-run=server",
                "--filename",
                str(path),
                "--output",
                "json",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode:
            raise JobsError("receipt verifier server dry-run failed; nothing was created")
        proof = validate_receipt_verify_job_preview(plan, manifest, json.loads(result.stdout))
        _write(proof_path, proof)
        _print(proof)
    except Exception as exc:
        _fail(exc)


@app.command("rl-topology-probe-preview")
def rl_topology_probe_preview(directory: Path) -> None:
    """Server-dry-run the exact dev FleetJob; no workload is created."""
    from training.skyrl_topology_probe import (
        SCHEMA,
        fleetjob_manifest,
        fleetjob_packet,
        validate_fleetjob_preview,
    )

    try:
        plan, _ = _prepared(directory)
        if plan.get("schema") != SCHEMA:
            raise ValueError("prepared directory is not a topology probe")
        gate = plan.get("qualification", {}).get("submission_gate", {})
        if gate.get("fleetjob_preview_authorized") is not True:
            raise ValueError("FleetJob server dry-run is not authorized")
        manifest = _read(directory / "fleetjob.json")
        packet = _read(directory / "FLEETJOB_PREPARED.json")
        if manifest != fleetjob_manifest(plan) or packet != fleetjob_packet(plan):
            raise ValueError("topology probe FleetJob packet changed")
        if (directory / "FLEETJOB_PREVIEW.json").exists():
            raise ValueError("FleetJob preview already recorded")
        execution = plan["execution"]
        result = subprocess.run(
            [
                "kubectl",
                "--context",
                execution["kubernetes_context"],
                "--namespace",
                execution["namespace"],
                "create",
                "--dry-run=server",
                "--filename",
                str(directory / "fleetjob.json"),
                "--output",
                "json",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode:
            raise JobsError("FleetJob server dry-run failed; no workload was created")
        rendered = json.loads(result.stdout)
        proof = validate_fleetjob_preview(plan, manifest, rendered)
        _write(directory / "FLEETJOB_PREVIEW.json", proof)
        _print(proof)
    except Exception as exc:
        _fail(exc)


@app.command("rl-topology-probe-authorize")
def rl_topology_probe_authorize(
    directory: Path,
    cpu_result: Annotated[Path, typer.Option("--cpu-result")],
    observer_armed: Annotated[Path, typer.Option("--observer-armed")],
) -> None:
    """Bind a passed CPU gate and a live cleanup observer to one GPU create."""
    from training.skyrl_topology_probe import SCHEMA
    from training.skyrl_topology_probe_launch import authorize

    try:
        plan, _ = _prepared(directory)
        if plan.get("schema") != SCHEMA:
            raise ValueError("prepared directory is not a topology probe")
        value = authorize(
            plan,
            cpu_result=_read(cpu_result),
            cpu_preview=_read(directory / "PREFLIGHT_JOB_PREVIEW.json"),
            receipt_verify_preview=_read(directory / "RECEIPT_VERIFY_JOB_PREVIEW.json"),
            fleetjob_preview=_read(directory / "FLEETJOB_PREVIEW.json"),
            fleetjob_observer=_read(observer_armed),
        )
        pid = value["fleetjob_observer"]["observer_pid"]
        os.kill(pid, 0)
        _write(directory / "LAUNCH_AUTHORIZED.json", value)
        _print(
            {
                "status": value["status"],
                "name": plan["run_name"],
                "plan_sha256": value["plan_sha256"],
                "manifest_sha256": value["fleetjob_manifest_sha256"],
                "authorization_sha256": value["sha256"],
                "observer_pid": pid,
                "submitted": False,
            }
        )
    except Exception as exc:
        _fail(exc)


@app.command("rl-topology-probe-create")
def rl_topology_probe_create(directory: Path) -> None:
    """Create the one evidence-authorized development FleetJob."""
    from uuid import UUID

    from training.skyrl_topology_probe import (
        SCHEMA,
        fleetjob_manifest,
        fleetjob_packet,
        validate_fleetjob_preview,
    )
    from training.skyrl_topology_probe_launch import _seal, validate

    try:
        plan, _ = _prepared(directory)
        if plan.get("schema") != SCHEMA:
            raise ValueError("prepared directory is not a topology probe")
        manifest = _read(directory / "fleetjob.json")
        packet = _read(directory / "FLEETJOB_PREPARED.json")
        if manifest != fleetjob_manifest(plan) or packet != fleetjob_packet(plan):
            raise ValueError("topology probe FleetJob packet changed")
        authorization = _read(directory / "LAUNCH_AUTHORIZED.json")
        validate(plan, authorization)
        observer_pid = authorization["fleetjob_observer"]["observer_pid"]
        os.kill(observer_pid, 0)
        if (directory / "FLEETJOB_CREATED.json").exists():
            raise ValueError("topology probe creation is already recorded")

        execution = plan["execution"]
        contexts = (
            execution["kubernetes_context"],
            "nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6",
        )
        for context in contexts:
            lookup = subprocess.run(
                [
                    "kubectl",
                    "--context",
                    context,
                    "--namespace",
                    execution["namespace"],
                    "get",
                    "fleetjob",
                    plan["run_name"],
                    "--ignore-not-found",
                    "--output=name",
                ],
                capture_output=True,
                text=True,
                timeout=45,
            )
            if lookup.returncode:
                raise JobsError("topology probe duplicate check failed")
            if lookup.stdout.strip():
                raise JobsError("topology probe name already exists")

        preview = subprocess.run(
            [
                "kubectl",
                "--context",
                execution["kubernetes_context"],
                "--namespace",
                execution["namespace"],
                "create",
                "--dry-run=server",
                "--filename",
                str(directory / "fleetjob.json"),
                "--output=json",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if preview.returncode:
            raise JobsError("topology probe final server preview failed")
        validate_fleetjob_preview(plan, manifest, json.loads(preview.stdout))

        result = subprocess.run(
            [
                "kubectl",
                "--context",
                execution["kubernetes_context"],
                "--namespace",
                execution["namespace"],
                "create",
                "--filename",
                str(directory / "fleetjob.json"),
                "--output=json",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode:
            raise JobsError("topology probe create failed")
        resource = json.loads(result.stdout)
        metadata = resource.get("metadata", {})
        uid = metadata.get("uid")
        created_at = metadata.get("creationTimestamp")
        UUID(uid)
        if (
            resource.get("apiVersion") != manifest["apiVersion"]
            or resource.get("kind") != manifest["kind"]
            or metadata.get("name") != plan["run_name"]
            or metadata.get("namespace") != execution["namespace"]
            or resource.get("spec") != manifest["spec"]
            or not isinstance(created_at, str)
            or not created_at
        ):
            raise JobsError("created topology probe differs from its authorization")
        receipt = _seal(
            {
                "schema": "cyber_skyrl_topology_probe_created_v1",
                "status": "created",
                "name": plan["run_name"],
                "uid": uid,
                "created_at": created_at,
                "plan_sha256": authorization["plan_sha256"],
                "manifest_sha256": authorization["fleetjob_manifest_sha256"],
                "authorization_sha256": authorization["sha256"],
            }
        )
        _write(directory / "FLEETJOB_CREATED.json", receipt)
        _print(receipt)
    except Exception as exc:
        _fail(exc)


@app.command()
def preflight(directory: Path) -> None:
    """Validate staged training/conversion inputs in the pinned image, without GPUs."""

    try:
        plan, _ = _prepared(directory)
        if plan.get("schema") == "cyber_miles_conversion_v1":
            from training.miles_conversion import preflight as check
        elif plan.get("schema") == "cyber_miles_training_v1":
            from training.miles_training import preflight as check
        elif plan.get("schema") == "cyber_skyrl_training_v1":
            from training.skyrl_training import preflight as check
        elif plan.get("schema") == "cyber_skyrl_production_training_v1":
            from training.skyrl_production_training import preflight as check
        elif plan.get("schema") == "cyber_skyrl_topology_probe_v1":
            from training.skyrl_topology_probe import preflight as check
        else:
            from training.sft import preflight as check
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
        plan, request = _prepared(directory)
        _external_action_gate(plan, "preview")
        with _client(plan) as client:
            result = client.preview(request)
        if plan.get("schema") == "cyber_skyrl_training_v1":
            from training.skyrl_training import validate_preview as validate_skyrl_preview

            validated = validate_skyrl_preview(plan, request, result)
        elif plan.get("schema") == "cyber_skyrl_production_training_v1":
            from training.skyrl_production_training import (
                validate_preview as validate_skyrl_preview,
            )

            validated = validate_skyrl_preview(plan, request, result)
        elif plan.get("schema") == "cyber_skyrl_topology_probe_v1":
            from training.skyrl_topology_probe import validate_preview as validate_probe_preview

            validated = validate_probe_preview(plan, request, result)
        else:
            validated = validate_preview(request, result)
        _print({"submitted": False, **validated})
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
        _submission_gate(directory, plan, request)
        _external_action_gate(plan, "submit")
        proof = _read(directory / "PREFLIGHT.json")
        expected = {
            "schema": "cyber_miles_conversion_cpu_preflight_v1"
            if plan.get("schema") == "cyber_miles_conversion_v1"
            else "cyber_miles_training_cpu_preflight_v1"
            if plan.get("schema") == "cyber_miles_training_v1"
            else "cyber_skyrl_training_cpu_preflight_v1"
            if plan.get("schema") == "cyber_skyrl_training_v1"
            else "cyber_skyrl_production_cpu_preflight_v1"
            if plan.get("schema") == "cyber_skyrl_production_training_v1"
            else "cyber_skyrl_topology_probe_cpu_preflight_v1"
            if plan.get("schema") == "cyber_skyrl_topology_probe_v1"
            else "cyber_sft_cpu_preflight_v1",
            "status": "passed",
            "gpus": 0,
            "plan_sha256": digest(plan),
            "request_sha256": digest(request),
        }
        if proof.get("sha256") != digest({k: v for k, v in proof.items() if k != "sha256"}) or any(
            proof.get(k) != v for k, v in expected.items()
        ):
            raise ValueError("missing or mismatched CPU preflight")
        with _client(plan) as client:
            if plan.get("schema") == "cyber_skyrl_production_training_v1":
                from training.skyrl_launch_guard import submit_once

                result = submit_once(plan, request, client, directory)
            elif (
                plan.get("schema") == "cyber_skyrl_training_v1"
                and plan.get("qualification", {}).get("profile") == "qwen38_skyrl_reward_canary_v4"
            ):
                from training.skyrl_launch_guard import submit_canary_once

                result = submit_canary_once(plan, request, client, directory)
            else:
                result = client.submit_once(request, directory / "SUBMISSION.jsonl")
        _print(result)
    except Exception as exc:
        _fail(exc)


@app.command("miles-convert")
def miles_convert(config: Path, output: Annotated[Path, typer.Option("--output")]) -> None:
    """Prepare native Qwen checkpoint conversion. No submission, download or optimization."""
    from training.miles_conversion import compile_conversion, job_request
    from training.sft import read_mapping

    try:
        plan = compile_conversion(read_mapping(config), relative_to=config.resolve().parent)
        _prepare(output, plan, job_request(plan))
        _print({"prepared": str(output), "optimizer_steps": 0, "gpus": 8, "submitted": False})
    except Exception as exc:
        _fail(exc)


@app.command("miles-seal")
def miles_seal(directory: Path, output: Annotated[Path, typer.Option("--output")]) -> None:
    """CPU-only integrity seal after terminal native conversion and verified GPU release."""
    from training.miles_conversion import seal

    try:
        plan, _ = _prepared(directory)
        result = seal(plan, output)
        _print({"sha256": result["sha256"], "files": len(result["files"]), "gpu_reload": False})
    except Exception as exc:
        _fail(exc)


@app.command()
def status(
    name: str,
    prepared: Annotated[Path | None, typer.Option("--prepared")] = None,
) -> None:
    """Read sanitized state; a development run requires its prepared plan."""
    try:
        plan = None
        if prepared is not None:
            plan, _ = _prepared(prepared)
            if plan.get("run_name") != name:
                raise ValueError("status name differs from the prepared plan")
        with _client(plan) as client:
            _print(client.status(name))
    except Exception as exc:
        _fail(exc)


@app.command("checkpoint-seal")
def checkpoint_seal(
    directory: Path, step: int, output: Annotated[Path, typer.Option("--output")]
) -> None:
    """CPU-only: hash a trusted run's native checkpoint for export/resume. No GPU reload."""
    from training.checkpoints import seal

    try:
        plan, _ = _prepared(directory)
        result = seal(plan, step, output)
        _print({k: result[k] for k in ("optimizer_step", "total_bytes", "receipt_sha256")})
    except Exception as exc:
        _fail(exc)


@app.command("rl-checkpoint-seal")
def rl_checkpoint_seal(directory: Path, output: Annotated[Path, typer.Option("--output")]) -> None:
    """CPU-only: accept and seal one terminal native SkyRL RL checkpoint."""
    from training.skyrl_posttrain import seal_checkpoint

    try:
        plan, _ = _prepared(directory)
        result = seal_checkpoint(plan, output)
        _print(
            {
                k: result[k]
                for k in (
                    "optimizer_step",
                    "total_bytes",
                    "receipt_sha256",
                    "optimizer_update_verified",
                )
            }
        )
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


@app.command("checkpoint-export")
def checkpoint_export(
    manifest: Path,
    sha256: Annotated[str, typer.Option("--sha256")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """CPU-only: export a sealed Qwen native checkpoint to exact-base-compatible BF16."""
    from training.export import export

    try:
        result = export(manifest, sha256, output)
        _print(
            {
                k: result[k]
                for k in (
                    "output_root",
                    "optimizer_step",
                    "tensor_bytes",
                    "receipt_sha256",
                    "gpu_reload_verified",
                )
            }
        )
    except Exception as exc:
        _fail(exc)


@app.command("rl-checkpoint-export")
def rl_checkpoint_export(
    manifest: Path,
    sha256: Annotated[str, typer.Option("--sha256")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """CPU-only: zero-update BF16 export from a sealed native SkyRL RL checkpoint."""
    from training.skyrl_posttrain import export_checkpoint

    try:
        result = export_checkpoint(manifest, sha256, output)
        _print(
            {
                k: result[k]
                for k in (
                    "output_root",
                    "optimizer_step",
                    "tensor_bytes",
                    "receipt_sha256",
                    "gpu_reload_verified",
                )
            }
        )
    except Exception as exc:
        _fail(exc)


@app.command("checkpoint-check")
def checkpoint_check(
    export: Path,
    sha256: Annotated[str, typer.Option("--sha256")],
    output: Annotated[Path, typer.Option("--output")],
    gpu: Annotated[bool, typer.Option("--gpu")] = False,
) -> None:
    """CPU export integrity/meta check; --gpu runs a one-GPU synthetic reload check."""
    from training.export_check import check

    try:
        _print(check(export, sha256, output, gpu=gpu))
    except Exception as exc:
        _fail(exc)


eval_app = typer.Typer(
    help="Fleet evaluations: prepare, check, initialize, then run bounded workers."
)
app.add_typer(eval_app, name="eval")


@eval_app.command("prepare")
def eval_prepare(config: Path, output: Annotated[Path, typer.Option("--output")]) -> None:
    """Freeze task/model/route identities and plan CSV. No network or execution."""
    from evals.fleet.evaluate import prepare
    from training.sft import read_mapping

    try:
        _print(prepare(read_mapping(config), output, relative_to=config.resolve().parent))
    except Exception as exc:
        _fail(exc)


@eval_app.command("preflight")
def eval_preflight(directory: Path) -> None:
    """Check staged Docker images and live Fleet task/endpoint metadata. No scored tasks."""
    from evals.fleet.evaluate import preflight

    try:
        _print(preflight(directory))
    except Exception as exc:
        _fail(exc)


@eval_app.command("init")
def eval_init(directory: Path) -> None:
    """Initialize an EMPTY dedicated PostgreSQL database. Never reset an existing campaign."""
    from evals.fleet.evaluate import checked_preflight
    from evals.fleet.rollout_postgres import initialize

    try:
        checked_preflight(directory)
        _print(initialize(os.environ["ROLLOUT_DATABASE_URL"], directory / "plan.csv"))
    except Exception as exc:
        _fail(exc)


@eval_app.command("run")
def eval_run(directory: Path, route: str, worker_id: str, limit: int = 1) -> None:
    """Execute at most LIMIT pending sessions on ROUTE, on an authorized Docker worker."""
    from evals.fleet.evaluate import run

    try:
        result = run(
            directory,
            dsn=os.environ["ROLLOUT_DATABASE_URL"],
            route=route,
            worker_id=worker_id,
            limit=limit,
        )
        _print(result)
    except Exception as exc:
        _fail(exc)
    if any("controller_failure_code" in row for row in result["results"]):
        raise typer.Exit(1)


@eval_app.command("status")
def eval_status() -> None:
    """Read score-blind PostgreSQL counts. Does not initialize or repair anything."""
    from evals.fleet.rollout_postgres import summary

    try:
        _print(summary(os.environ["ROLLOUT_DATABASE_URL"]))
    except Exception as exc:
        _fail(exc)


if __name__ == "__main__":
    app()
