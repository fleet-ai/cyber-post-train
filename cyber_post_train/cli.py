"""Prepare, preflight, preview and submit through one small command surface."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from .jobs import API_URLS, Jobs, JobsError, digest, validate_preview, validate_request

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)

_ENGINE_DIAGNOSTIC_ABSENT_RUNTIME_CONTEXT_QUALIFICATION = {
    "image": (
        "661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/skyrl-train@sha256:"
        "89758df2b5f35cdb19efe948c7f6ef54f11e2e2ab47a45d600c25f36914e308f"
    ),
    "image_cpu_qualification_receipt_sha256": (
        "28244d695896b8b9766df66caecd117a33fd5d9c2c5df35faba12bcc785d09f1"
    ),
    "default_user_evidence_path": (
        "docs/evidence/qwen38-study/"
        "2026-09-12-skyrl-worker-rpc-relay-image-default-user-qualification-v1.json"
    ),
    "default_user_evidence_file_sha256": (
        "cf7841536034c599827a938666070075a2e300f777895c81aa40b6da63c1e3a5"
    ),
    "default_user_evidence_receipt_sha256": (
        "d1080784334becc76edb698ce363ecc513e5c98f9fce6c9510de59136192129f"
    ),
    "default_user_probe_pod_uid": "367c7123-873d-43db-a2c7-e4407fd71996",
}


class Cluster(StrEnum):
    dev = "dev"
    prod = "prod"


DEV_KUBE_CONTEXT = "nebius-mk8s-fleetai-training-dev-e04p03enwk5c0va9tb"
_MILES_HF_PLAN_SCHEMAS = frozenset(
    {"cyber_miles_hf_export_job_plan_v1", "cyber_miles_hf_export_job_plan_v2"}
)
_MILES_SERVING_DEV_PLAN_SCHEMA = "cyber_miles_serving_dev_plan_v1"
_SFT_SERVING_DEV_PLAN_SCHEMA = "cyber_sft_serving_dev_plan_v1"
_MILES_PARSER_PROBE_SCHEMA = "cyber_miles_native_parser_probe_v1"
_SERVING_DEV_PLAN_SCHEMAS = frozenset(
    {_MILES_SERVING_DEV_PLAN_SCHEMA, _SFT_SERVING_DEV_PLAN_SCHEMA}
)


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
        {"plan_sha256": digest(plan), "request_sha256": digest(request)},
    )


def _prepared(directory: Path) -> tuple[dict, dict]:
    plan, request, receipt = (
        _read(directory / name) for name in ("plan.json", "request.json", "PREPARED.json")
    )
    if receipt != {"plan_sha256": digest(plan), "request_sha256": digest(request)}:
        raise ValueError("prepared inputs changed; prepare a new directory, never edit a launch")
    if plan.get("schema") in _MILES_HF_PLAN_SCHEMAS and plan.get("stage") == "export":
        from training.miles_hf_export_job import job_request, validate_plan

        validate_plan(plan, check_files=False)
        if request != job_request(plan):
            raise ValueError("prepared Kubernetes Job differs from its immutable plan")
    elif plan.get("schema") in _SERVING_DEV_PLAN_SCHEMAS:
        from training.miles_serving_dev import job_request, validate_plan

        validate_plan(plan, check_files=False)
        if request != job_request(plan, check_files=False):
            raise ValueError("prepared serving canary differs from its immutable plan")
    elif plan.get("schema") == _MILES_PARSER_PROBE_SCHEMA:
        from training.miles_parser_probe import job_request, validate_plan

        validate_plan(plan)
        if request != job_request(plan):
            raise ValueError("prepared parser probe differs from its immutable plan")
    else:
        validate_request(request)
    return plan, request


def _client(cluster: Cluster) -> Jobs:
    return Jobs(os.environ.get("FLEET_API_KEY", ""), base_url=API_URLS[cluster])


def _fail(exc: Exception) -> None:
    # SDK errors can embed source records or secrets. Never emit rich tracebacks.
    message = str(exc) if isinstance(exc, JobsError) else type(exc).__name__
    typer.echo(f"Stopped: {message}. No automatic retry.", err=True)
    raise typer.Exit(2) from None


def _validate_engine_diagnostic_preview(plan: dict, request: dict, preview_result: dict) -> dict:
    """Accept omitted Pod identity only for the exact qualified immutable image.

    The deployed dev Jobs API currently has no request field for a Pod security
    context. An omitted field is distinct from an explicit conflict: omission
    may use the exact image whose default 1000:100 user was proven by a bounded
    dev Pod because the bundled GPU entrypoint rechecks its real uid/gid before
    Ray initialization. Any explicit conflicting value remains fatal.
    """
    import yaml

    from training.skyrl_training import (
        DEV_KUBERNETES_NAMESPACE,
        ENGINE_DIAGNOSTIC_CURRENT_CONFIG_NAME,
        ENGINE_DIAGNOSTIC_CURRENT_OUTPUT_ROOT,
        ENGINE_DIAGNOSTIC_GPUS_PER_WORKER,
        ENGINE_DIAGNOSTIC_WORKERS,
        ENGINE_IMAGE_CPU_QUALIFICATION,
        IMAGE,
        validate_engine_diagnostic_preview,
    )

    try:
        return validate_engine_diagnostic_preview(plan, request, preview_result)
    except JobsError as error:
        if str(error) != "engine diagnostic preview runtime user differs from 1000:100":
            raise
        context_error = error

    qualified = _ENGINE_DIAGNOSTIC_ABSENT_RUNTIME_CONTEXT_QUALIFICATION
    execution = plan.get("execution", {})
    if (
        plan.get("run_name") != ENGINE_DIAGNOSTIC_CURRENT_CONFIG_NAME
        or plan.get("output_root") != ENGINE_DIAGNOSTIC_CURRENT_OUTPUT_ROOT
        or execution.get("cluster_target") != "dev"
        or execution.get("image") != qualified["image"]
        or execution.get("image_cpu_qualification") != ENGINE_IMAGE_CPU_QUALIFICATION
        or request.get("image") != qualified["image"]
        or request.get("workers") != ENGINE_DIAGNOSTIC_WORKERS
        or request.get("gpus_per_worker") != ENGINE_DIAGNOSTIC_GPUS_PER_WORKER
        or request.get("env", {}).get("CYBER_EXPECTED_RUNTIME_UID") != "1000"
        or request.get("env", {}).get("CYBER_EXPECTED_RUNTIME_GID") != "100"
    ):
        raise context_error
    evidence_path = Path(__file__).resolve().parents[1] / qualified["default_user_evidence_path"]
    try:
        evidence_bytes = evidence_path.read_bytes()
        evidence = json.loads(evidence_bytes)
        unsigned_evidence = {
            key: value for key, value in evidence.items() if key != "receipt_sha256"
        }
    except (OSError, TypeError, ValueError) as error:
        raise JobsError("exact engine image default-user qualification is unavailable") from error
    runtime = evidence.get("runtime", {})
    checks = evidence.get("checks", {})
    qualification_scope = evidence.get("qualification_scope", {})
    if (
        hashlib.sha256(evidence_bytes).hexdigest() != qualified["default_user_evidence_file_sha256"]
        or evidence.get("receipt_sha256")
        != hashlib.sha256(
            json.dumps(
                unsigned_evidence, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
            + b"\n"
        ).hexdigest()
        or evidence.get("receipt_sha256") != qualified["default_user_evidence_receipt_sha256"]
        or evidence.get("status") != "qualified"
        or evidence.get("requested_image") != qualified["image"]
        or runtime.get("runtime_image_id") != qualified["image"]
        or runtime.get("pod_uid") != qualified["default_user_probe_pod_uid"]
        or (runtime.get("effective_uid"), runtime.get("effective_gid")) != (1000, 100)
        or runtime.get("terminal_phase") != "Succeeded"
        or runtime.get("exit_code") != 0
        or runtime.get("restart_count") != 0
        or runtime.get("pod_security_context") != {}
        or runtime.get("container_security_context") is not None
        or runtime.get("gpu_request") != 0
        or runtime.get("service_account_token_mounted") is not False
        or evidence.get("cleanup", {}).get("pod_absent_after_delete") is not True
        or checks
        != {
            "container_security_context_absent": True,
            "effective_gid_is_100": True,
            "effective_uid_is_1000": True,
            "exact_runtime_image_id": True,
            "pod_security_context_has_no_identity_fields": True,
            "zero_gpu_and_no_service_account": True,
        }
        or qualification_scope.get("explicit_conflicting_identity_remains_fatal") is not True
        or qualification_scope.get("gpu_runtime_identity_recheck_remains_required") is not True
        or qualification_scope.get("mutable_image_tags_are_not_qualified") is not True
        or qualified["image"] != IMAGE
        or ENGINE_IMAGE_CPU_QUALIFICATION.get("receipt_sha256")
        != qualified["image_cpu_qualification_receipt_sha256"]
    ):
        raise JobsError("absent engine diagnostic runtime user requires the exact qualified image")

    expected = {"runAsUser": 1000, "runAsGroup": 100, "runAsNonRoot": True}
    try:
        obj = yaml.safe_load(preview_result["manifest_yaml"])
        cluster = obj["spec"]["rayClusterSpec"]
        groups = [(1, cluster["headGroupSpec"]["template"])] + [
            (group["replicas"], group["template"]) for group in cluster.get("workerGroupSpecs", [])
        ]
        pods = 0
        omitted_fields = 0
        for replicas, template in groups:
            if type(replicas) is not int or replicas < 0:
                raise JobsError("invalid engine diagnostic preview replica count")
            if replicas == 0:
                continue
            pod = template["spec"]
            containers = pod["containers"]
            if not isinstance(containers, list) or len(containers) != 1:
                raise JobsError("engine diagnostic preview must have one container per Pod")
            pod_context = pod.get("securityContext", {})
            container_context = containers[0].get("securityContext", {})
            if not isinstance(pod_context, dict) or not isinstance(container_context, dict):
                raise JobsError("malformed engine diagnostic Jobs API preview")
            for context in (pod_context, container_context):
                for key, value in expected.items():
                    if key in context and not (
                        type(context[key]) is type(value) and context[key] == value
                    ):
                        raise JobsError(
                            "engine diagnostic preview contains an explicit runtime user conflict"
                        )
            omitted_fields += sum(
                key not in pod_context and key not in container_context for key in expected
            )
            pods += replicas
        if (
            obj["metadata"]["namespace"] != DEV_KUBERNETES_NAMESPACE
            or pods != ENGINE_DIAGNOSTIC_WORKERS
            or omitted_fields == 0
        ):
            raise JobsError("engine diagnostic preview topology or runtime binding changed")
    except JobsError:
        raise
    except (AttributeError, KeyError, TypeError, yaml.YAMLError) as error:
        raise JobsError("malformed engine diagnostic Jobs API preview") from error

    return {
        "runtime_user": {"uid": 1000, "gid": 100},
        "runtime_user_evidence": {
            "mode": "qualified_image_default_with_gpu_entrypoint_recheck",
            "qualification_receipt_sha256": evidence["receipt_sha256"],
            "omitted_preview_fields": omitted_fields,
            "explicit_conflicts_rejected": True,
        },
        "pods": pods,
    }


def _reward_canary_default_user_evidence() -> dict:
    """Reopen the exact clean-pull proof used only by the reward-canary fallback."""
    qualified = _ENGINE_DIAGNOSTIC_ABSENT_RUNTIME_CONTEXT_QUALIFICATION
    evidence_path = Path(__file__).resolve().parents[1] / qualified["default_user_evidence_path"]
    try:
        evidence_bytes = evidence_path.read_bytes()
        evidence = json.loads(evidence_bytes)
        unsigned_evidence = {
            key: value for key, value in evidence.items() if key != "receipt_sha256"
        }
    except (OSError, TypeError, ValueError) as error:
        raise JobsError(
            "exact reward canary image default-user qualification is unavailable"
        ) from error
    runtime = evidence.get("runtime", {})
    checks = evidence.get("checks", {})
    qualification_scope = evidence.get("qualification_scope", {})
    if (
        hashlib.sha256(evidence_bytes).hexdigest() != qualified["default_user_evidence_file_sha256"]
        or evidence.get("receipt_sha256")
        != hashlib.sha256(
            json.dumps(
                unsigned_evidence, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
            + b"\n"
        ).hexdigest()
        or evidence.get("receipt_sha256") != qualified["default_user_evidence_receipt_sha256"]
        or evidence.get("status") != "qualified"
        or evidence.get("requested_image") != qualified["image"]
        or runtime.get("runtime_image_id") != qualified["image"]
        or runtime.get("pod_uid") != qualified["default_user_probe_pod_uid"]
        or (runtime.get("effective_uid"), runtime.get("effective_gid")) != (1000, 100)
        or runtime.get("terminal_phase") != "Succeeded"
        or runtime.get("exit_code") != 0
        or runtime.get("restart_count") != 0
        or runtime.get("pod_security_context") != {}
        or runtime.get("container_security_context") is not None
        or runtime.get("gpu_request") != 0
        or runtime.get("service_account_token_mounted") is not False
        or evidence.get("cleanup", {}).get("pod_absent_after_delete") is not True
        or checks
        != {
            "container_security_context_absent": True,
            "effective_gid_is_100": True,
            "effective_uid_is_1000": True,
            "exact_runtime_image_id": True,
            "pod_security_context_has_no_identity_fields": True,
            "zero_gpu_and_no_service_account": True,
        }
        or qualification_scope.get("explicit_conflicting_identity_remains_fatal") is not True
        or qualification_scope.get("gpu_runtime_identity_recheck_remains_required") is not True
        or qualification_scope.get("mutable_image_tags_are_not_qualified") is not True
    ):
        raise JobsError("absent reward canary runtime user requires the exact qualified image")
    return evidence


def _validate_reward_canary_preview(plan: dict, request: dict, preview_result: dict) -> dict:
    """Accept absent identity only for the exact reward canary and qualified image."""
    import yaml

    from training.skyrl_training import (
        DEV_KUBERNETES_NAMESPACE,
        ENGINE_IMAGE_CPU_QUALIFICATION,
        IMAGE,
        REWARD_CANARY_ARGUMENTS,
        REWARD_CANARY_RUNTIME_USER,
        is_reward_canary,
        validate_reward_canary_preview,
    )

    if not is_reward_canary(plan):
        return validate_reward_canary_preview(plan, request, preview_result)
    validated = None
    context_error = None
    try:
        validated = validate_reward_canary_preview(plan, request, preview_result)
    except JobsError as error:
        if str(error) != "reward canary preview runtime user differs from 1000:100":
            raise
        context_error = error
    qualified = _ENGINE_DIAGNOSTIC_ABSENT_RUNTIME_CONTEXT_QUALIFICATION
    execution = plan.get("execution", {})
    if (
        plan.get("run_name") != REWARD_CANARY_ARGUMENTS["name"]
        or plan.get("output_root") != REWARD_CANARY_ARGUMENTS["output_root"]
        or execution.get("cluster_target") != "dev"
        or execution.get("image") != qualified["image"]
        or execution.get("image_cpu_qualification") != ENGINE_IMAGE_CPU_QUALIFICATION
        or execution.get("runtime_user") != REWARD_CANARY_RUNTIME_USER
        or request.get("image") != qualified["image"]
        or request.get("workers") != 1
        or request.get("gpus_per_worker") != 8
        or request.get("env", {}).get("CYBER_EXPECTED_RUNTIME_UID") != "1000"
        or request.get("env", {}).get("CYBER_EXPECTED_RUNTIME_GID") != "100"
        or qualified["image"] != IMAGE
        or ENGINE_IMAGE_CPU_QUALIFICATION.get("receipt_sha256")
        != qualified["image_cpu_qualification_receipt_sha256"]
    ):
        if context_error is not None:
            raise context_error
        raise JobsError("reward canary preview request differs from its exact plan")

    expected = {"runAsUser": 1000, "runAsGroup": 100, "runAsNonRoot": True}
    try:
        obj = yaml.safe_load(preview_result["manifest_yaml"])
        cluster = obj["spec"]["rayClusterSpec"]
        groups = [(1, cluster["headGroupSpec"]["template"])] + [
            (group["replicas"], group["template"]) for group in cluster.get("workerGroupSpecs", [])
        ]
        pods = 0
        omitted_fields = 0
        for replicas, template in groups:
            if type(replicas) is not int or replicas < 0:
                raise JobsError("invalid reward canary preview replica count")
            if replicas == 0:
                continue
            pod = template["spec"]
            containers = pod["containers"]
            if not isinstance(containers, list) or len(containers) != 1:
                raise JobsError("reward canary preview must have one container per Pod")
            pod_context = pod.get("securityContext", {})
            container_context = containers[0].get("securityContext", {})
            if not isinstance(pod_context, dict) or not isinstance(container_context, dict):
                raise JobsError("malformed reward canary Jobs API preview")
            for context in (pod_context, container_context):
                for key, value in expected.items():
                    if key in context and not (
                        type(context[key]) is type(value) and context[key] == value
                    ):
                        raise JobsError(
                            "reward canary preview contains an explicit runtime user conflict "
                            "with required 1000:100"
                        )
            omitted_fields += sum(
                key not in pod_context and key not in container_context for key in expected
            )
            pods += replicas
        if obj["metadata"]["namespace"] != DEV_KUBERNETES_NAMESPACE or pods != 1:
            raise JobsError("reward canary preview topology or runtime binding changed")
    except JobsError:
        raise
    except (AttributeError, KeyError, TypeError, yaml.YAMLError) as error:
        raise JobsError("malformed reward canary Jobs API preview") from error

    if omitted_fields == 0:
        if validated is None:
            raise context_error
        return validated
    evidence = _reward_canary_default_user_evidence()
    return {
        "runtime_user": {"uid": 1000, "gid": 100},
        "runtime_user_evidence": {
            "mode": "qualified_image_default_with_gpu_entrypoint_recheck",
            "qualification_receipt_sha256": evidence["receipt_sha256"],
            "omitted_preview_fields": omitted_fields,
            "explicit_conflicts_rejected": True,
        },
        "pods": pods,
    }


@app.command()
def data(config: Path) -> None:
    """Prepare private dense SFT data from a frozen split. CPU only; no submission."""
    from training.corpus import build
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


@app.command("rl-data-derive")
def rl_data_derive(config: Path) -> None:
    """Create policy-specific Miles metadata from an accepted base data artifact."""
    from training.rl_data_derive import derive
    from training.sft import read_mapping

    try:
        _print(derive(read_mapping(config), relative_to=config.resolve().parent))
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
            from training import skyrl_training as backend
        else:
            raise ValueError("unsupported RL backend")
        plan = backend.compile_rl(value, relative_to=config.resolve().parent)
        request = backend.job_request(plan)
        _prepare(output, plan, request)
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


@app.command("rl-engine-diagnostic")
def rl_engine_diagnostic(config: Path, output: Annotated[Path, typer.Option("--output")]) -> None:
    """Prepare one dev-only SkyRL engine-start diagnostic. No task execution or submission."""
    from training.sft import read_mapping
    from training.skyrl_training import compile_rl, engine_diagnostic_request

    try:
        value = read_mapping(config)
        if value.get("backend") != "skyrl":
            raise ValueError("engine diagnostic requires the SkyRL backend")
        plan = compile_rl(value, relative_to=config.resolve().parent)
        request = engine_diagnostic_request(plan)
        _prepare(output, plan, request)
        _print(
            {
                "prepared": str(output),
                "gpus": request["workers"] * request["gpus_per_worker"],
                "task_rows": 0,
                "optimizer_steps": 0,
                "submitted": False,
            }
        )
    except Exception as exc:
        _fail(exc)


@app.command("rl-checkpoint-seal")
def rl_checkpoint_seal(
    directory: Path,
    step: int,
    candidate: Annotated[Path, typer.Option("--candidate")],
    controller_audit: Annotated[Path, typer.Option("--controller-audit")],
    sealer_release: Annotated[Path, typer.Option("--sealer-release")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Finalize a checkpoint seal after its exact zero-GPU sealer Pod exits cleanly."""
    from training.skyrl_rl_checkpoint import seal
    from training.skyrl_training import SCHEMA

    try:
        plan, _ = _prepared(directory)
        if plan.get("schema") != SCHEMA:
            raise ValueError("RL checkpoint sealing requires a native SkyRL training plan")
        result = seal(plan, step, output, candidate, controller_audit, sealer_release)
        _print(
            {
                "sha256": result["sha256"],
                "checkpoint_step": result["checkpoint"]["step"],
                "files": len(result["checkpoint"]["files"]),
                "source_gpu_release_verified": True,
                "gpu_reload_verified": False,
            }
        )
    except Exception as exc:
        _fail(exc)


@app.command("rl-checkpoint-candidate")
def rl_checkpoint_candidate(
    directory: Path,
    step: int,
    release_evidence: Annotated[Path, typer.Option("--release-evidence")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Create the native checkpoint candidate inside the exact zero-GPU sealer Pod."""
    from training.skyrl_rl_checkpoint import checkpoint_candidate
    from training.skyrl_training import SCHEMA

    try:
        plan, _ = _prepared(directory)
        if plan.get("schema") != SCHEMA:
            raise ValueError("RL checkpoint candidate requires a native SkyRL training plan")
        result = checkpoint_candidate(plan, step, output, release_evidence)
        _print(
            {
                "sha256": result["sha256"],
                "checkpoint_step": result["checkpoint"]["step"],
                "files": len(result["checkpoint"]["files"]),
                "gpu_reload_verified": False,
            }
        )
    except Exception as exc:
        _fail(exc)


@app.command("rl-checkpoint-sealer-contract")
def rl_checkpoint_sealer_contract(directory: Path, step: int) -> None:
    """Print the exact dev zero-GPU Job contract an external audit must observe."""
    from training.skyrl_rl_checkpoint import checkpoint_sealer_contract
    from training.skyrl_training import SCHEMA

    try:
        plan, _ = _prepared(directory)
        if plan.get("schema") != SCHEMA:
            raise ValueError("RL checkpoint sealer contract requires a native SkyRL plan")
        _print(checkpoint_sealer_contract(plan, step))
    except Exception as exc:
        _fail(exc)


@app.command("rl-reward-canary-accept")
def rl_reward_canary_accept(
    directory: Path,
    checkpoint_manifest: Annotated[Path, typer.Option("--checkpoint-manifest")],
    release_evidence: Annotated[Path, typer.Option("--release-evidence")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Seal the digest-bound reward/update/W&B/checkpoint terminal package."""
    from training.skyrl_training import SCHEMA, seal_reward_canary_terminal

    try:
        plan, _ = _prepared(directory)
        if plan.get("schema") != SCHEMA:
            raise ValueError("reward acceptance requires a native SkyRL training plan")
        result = seal_reward_canary_terminal(
            plan,
            checkpoint_manifest,
            release_evidence,
            output,
        )
        _print({"status": result["status"], "sha256": result["sha256"]})
    except Exception as exc:
        _fail(exc)


@app.command("rl-reload")
def rl_reload(config: Path, output: Annotated[Path, typer.Option("--output")]) -> None:
    """Prepare one dev-only zero-update all-rank SkyRL RL checkpoint reload."""
    from training.sft import read_mapping
    from training.skyrl_rl_checkpoint import compile_reload, reload_request

    try:
        plan = compile_reload(read_mapping(config), relative_to=config.resolve().parent)
        request = reload_request(plan)
        _prepare(output, plan, request)
        _print(
            {
                "prepared": str(output),
                "checkpoint_step": plan["source_manifest"]["checkpoint"]["step"],
                "gpus": request["workers"] * request["gpus_per_worker"],
                "optimizer_updates": 0,
                "rollouts": 0,
                "submitted": False,
            }
        )
    except Exception as exc:
        _fail(exc)


@app.command("rl-reload-accept")
def rl_reload_accept(
    directory: Path,
    release_evidence: Annotated[Path, typer.Option("--release-evidence")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Seal reload acceptance after independent terminal Job/Pod GPU-release evidence."""
    from training.skyrl_rl_checkpoint import RELOAD_SCHEMA, accept_reload

    try:
        plan, _ = _prepared(directory)
        if plan.get("schema") != RELOAD_SCHEMA:
            raise ValueError("reload acceptance requires a prepared RL reload plan")
        result = accept_reload(plan, release_evidence, output)
        _print(
            {
                "sha256": result["sha256"],
                "status": result["status"],
                "optimizer_updates": 0,
                "external_job_gpu_release_verified": True,
            }
        )
    except Exception as exc:
        _fail(exc)


def _skyrl_mode(plan: dict, request: dict) -> str:
    from training import skyrl_training as backend

    if request == backend.engine_diagnostic_request(plan):
        return "engine_diagnostic"
    if request == backend.job_request(plan):
        return "training"
    raise ValueError("prepared SkyRL request does not match its immutable plan")


def _require_prepared_cluster(plan: dict, cluster: Cluster) -> None:
    """Fail closed when an immutable training plan names one cluster."""
    target = plan.get("execution", {}).get("cluster_target")
    if plan.get("schema") in _SERVING_DEV_PLAN_SCHEMAS:
        target = plan.get("cluster_target")
    if target is not None and target != cluster.value:
        raise JobsError(f"prepared plan is {target}-cluster-only")
    if (
        cluster == Cluster.prod
        and plan.get("schema") in {"cyber_skyrl_training_v1", "cyber_miles_training_v1"}
        and target != Cluster.prod.value
    ):
        raise JobsError("production RL requires an explicit prod-cluster plan")


@app.command()
def preflight(directory: Path) -> None:
    """Validate staged training/conversion inputs in the pinned image, without GPUs."""

    try:
        plan, request = _prepared(directory)
        if plan.get("schema") == "cyber_miles_conversion_v1":
            from training.miles_conversion import preflight as check
        elif plan.get("schema") == "cyber_miles_training_v1":
            from training.miles_training import preflight as check
        elif plan.get("schema") == "cyber_miles_rl_reload_v1":
            from training.miles_reload import preflight as check
        elif plan.get("schema") in _MILES_HF_PLAN_SCHEMAS:
            from training.miles_hf_export_job import preflight as check
        elif plan.get("schema") in _SERVING_DEV_PLAN_SCHEMAS:
            from training.miles_serving_dev import preflight as check
        elif plan.get("schema") == _MILES_PARSER_PROBE_SCHEMA:
            from training.miles_parser_probe import preflight as check
        elif plan.get("schema") == "cyber_skyrl_training_v1":
            if _skyrl_mode(plan, request) == "engine_diagnostic":
                from training.skyrl_training import engine_diagnostic_preflight as check
            else:
                from training.skyrl_training import preflight as check
        elif plan.get("schema") == "cyber_skyrl_rl_reload_v1":
            from training.skyrl_rl_checkpoint import preflight as check
        else:
            from training.sft import preflight as check
        if (directory / "PREFLIGHT.json").exists():
            raise ValueError("preflight already recorded")
        receipt = check(plan)
        receipt = {**receipt, "sha256": digest(receipt)}
        _write(directory / "PREFLIGHT.json", receipt)
        _print(receipt)
    except Exception as exc:
        _fail(exc)


@app.command()
def preview(
    directory: Path,
    cluster: Annotated[
        Cluster, typer.Option(help="Jobs API target; independent of kube context.")
    ] = Cluster.dev,
) -> None:
    """Read the selected cluster's render (dev by default); does not create a run."""
    try:
        plan, request = _prepared(directory)
        if (
            plan.get("schema") == "cyber_skyrl_training_v1"
            and _skyrl_mode(plan, request) == "engine_diagnostic"
            and cluster != Cluster.dev
        ):
            raise JobsError("SkyRL engine diagnostics are dev-cluster-only")
        if plan.get("schema") == "cyber_skyrl_rl_reload_v1" and cluster != Cluster.dev:
            raise JobsError("SkyRL RL reload validators are dev-cluster-only")
        if plan.get("schema") in _SERVING_DEV_PLAN_SCHEMAS and cluster != Cluster.dev:
            raise JobsError("serving qualification is dev-cluster-only")
        if plan.get("schema") == _MILES_PARSER_PROBE_SCHEMA and cluster != Cluster.dev:
            raise JobsError("Miles parser probes are dev-cluster-only")
        _require_prepared_cluster(plan, cluster)
        with _client(cluster) as client:
            result = client.preview(request)
        validated = validate_preview(request, result)
        if (
            plan.get("schema") == "cyber_skyrl_training_v1"
            and _skyrl_mode(plan, request) == "engine_diagnostic"
        ):
            validated.update(_validate_engine_diagnostic_preview(plan, request, result))
        elif (
            plan.get("schema") == "cyber_skyrl_training_v1"
            and _skyrl_mode(plan, request) == "training"
        ):
            from training.skyrl_promotion import validate_production_preview

            validated.update(_validate_reward_canary_preview(plan, request, result))
            validated.update(validate_production_preview(plan, request, result))
        elif plan.get("schema") == "cyber_miles_training_v1":
            from training.miles_promotion import validate_production_preview

            validated.update(validate_production_preview(plan, request, result))
        if plan.get("schema") == "cyber_skyrl_rl_reload_v1":
            from training.skyrl_rl_checkpoint import validate_reload_preview

            validated.update(validate_reload_preview(request, result))
        _print(
            {
                "submitted": False,
                "api_base_url": API_URLS[cluster],
                **validated,
            }
        )
    except Exception as exc:
        _fail(exc)


@app.command()
def submit(
    directory: Path,
    cluster: Annotated[
        Cluster, typer.Option(help="Jobs API target; prod requires reviewed dev-run evidence.")
    ] = Cluster.dev,
) -> None:
    """Submit once to dev by default, after CPU preflight and operator resource checks.

    Keep this directory on the experiment's shared durable storage. A timeout
    leaves SUBMISSION.jsonl: reconcile it instead of copying or deleting it.
    --cluster prod is explicit routing, not automatic dev-qualification proof.
    """
    try:
        plan, request = _prepared(directory)
        if plan.get("schema") in _MILES_HF_PLAN_SCHEMAS:
            raise JobsError(
                "Miles HF jobs require the pre-POST TTL-zero watcher; "
                "use python -m training.miles_hf_export_cli submit"
            )
        skyrl_mode = (
            _skyrl_mode(plan, request) if plan.get("schema") == "cyber_skyrl_training_v1" else None
        )
        if skyrl_mode == "engine_diagnostic" and cluster != Cluster.dev:
            raise JobsError("SkyRL engine diagnostics are dev-cluster-only")
        if plan.get("schema") == "cyber_skyrl_rl_reload_v1" and cluster != Cluster.dev:
            raise JobsError("SkyRL RL reload validators are dev-cluster-only")
        if plan.get("schema") in _SERVING_DEV_PLAN_SCHEMAS and cluster != Cluster.dev:
            raise JobsError("serving qualification is dev-cluster-only")
        if plan.get("schema") == _MILES_PARSER_PROBE_SCHEMA and cluster != Cluster.dev:
            raise JobsError("Miles parser probes are dev-cluster-only")
        _require_prepared_cluster(plan, cluster)
        proof = _read(directory / "PREFLIGHT.json")
        expected = {
            "schema": "cyber_miles_conversion_cpu_preflight_v1"
            if plan.get("schema") == "cyber_miles_conversion_v1"
            else "cyber_miles_training_cpu_preflight_v1"
            if plan.get("schema") == "cyber_miles_training_v1"
            else "cyber_miles_native_parser_probe_cpu_preflight_v1"
            if plan.get("schema") == _MILES_PARSER_PROBE_SCHEMA
            else "cyber_miles_rl_reload_cpu_preflight_v1"
            if plan.get("schema") == "cyber_miles_rl_reload_v1"
            else (
                "cyber_sft_serving_dev_cpu_preflight_v1"
                if plan.get("schema") == _SFT_SERVING_DEV_PLAN_SCHEMA
                else "cyber_miles_serving_dev_cpu_preflight_v1"
            )
            if plan.get("schema") in _SERVING_DEV_PLAN_SCHEMAS
            else (
                "cyber_skyrl_engine_diagnostic_cpu_preflight_v1"
                if skyrl_mode == "engine_diagnostic"
                else "cyber_skyrl_training_cpu_preflight_v1"
            )
            if plan.get("schema") == "cyber_skyrl_training_v1"
            else "cyber_skyrl_rl_reload_cpu_preflight_v1"
            if plan.get("schema") == "cyber_skyrl_rl_reload_v1"
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
        if plan.get("schema") == "cyber_miles_rl_reload_v1":
            from training.miles_reload import validate_preflight_receipt

            validate_preflight_receipt(plan, request, proof)
        elif plan.get("schema") == _SFT_SERVING_DEV_PLAN_SCHEMA:
            from training.miles_serving_dev import validate_preflight_receipt

            validate_preflight_receipt(plan, request, proof)
        submission_journal = directory / "SUBMISSION.jsonl"
        if submission_journal.exists() or submission_journal.is_symlink():
            raise JobsError("submission journal already exists; reconcile, never repeat POST")
        with _client(cluster) as client:
            if plan.get("schema") == "cyber_miles_training_v1":
                from training.miles_promotion import (
                    require_live_external,
                    require_live_files,
                    requires_production_promotion,
                    validate_production_preview,
                )

                if requires_production_promotion(plan):
                    preview_result = client.preview(request)
                    validate_preview(request, preview_result)
                    validate_production_preview(plan, request, preview_result)
                    require_live_files(plan, directory)
                    require_live_external(plan, client)
            elif plan.get("schema") == "cyber_skyrl_rl_reload_v1":
                from training.skyrl_rl_checkpoint import validate_reload_preview

                preview_result = client.preview(request)
                validate_preview(request, preview_result)
                validate_reload_preview(request, preview_result)
            elif plan.get("schema") == "cyber_miles_rl_reload_v1":
                # This is deliberately a one-off dev gate. Re-read the live
                # rendered shape immediately before its only allowed POST.
                validate_preview(request, client.preview(request))
            elif plan.get("schema") in _SERVING_DEV_PLAN_SCHEMAS:
                # Re-open the exact artifact and rendered one-GPU shape at the
                # POST boundary. Dev qualification never has a prod route.
                from training.miles_serving_dev import (
                    validate_plan,
                    validate_preflight_receipt,
                )

                validate_plan(plan, check_files=True)
                validate_preview(request, client.preview(request))
                # Rehash the payload and recheck target absence after preview,
                # immediately before the only allowed GPU POST.
                validate_preflight_receipt(plan, request, proof)
            elif skyrl_mode == "engine_diagnostic":
                preview_result = client.preview(request)
                validate_preview(request, preview_result)
                _validate_engine_diagnostic_preview(plan, request, preview_result)
            elif skyrl_mode == "training":
                from training.skyrl_promotion import (
                    require_live_external,
                    require_live_files,
                    requires_production_promotion,
                    validate_production_preview,
                )
                from training.skyrl_training import is_reward_canary

                reward_canary = is_reward_canary(plan)
                production = requires_production_promotion(plan)
                if reward_canary or production:
                    preview_result = client.preview(request)
                    validate_preview(request, preview_result)
                    if reward_canary:
                        _validate_reward_canary_preview(plan, request, preview_result)
                    if production:
                        validate_production_preview(plan, request, preview_result)
                        require_live_files(plan, directory)
                        require_live_external(plan, client)
            result = client.submit_once(request, submission_journal)
        _print({"api_base_url": API_URLS[cluster], **result})
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


@app.command("miles-parser-probe")
def miles_parser_probe(
    source_plan: Path,
    name: Annotated[str, typer.Option("--name")],
    output_root: Annotated[str, typer.Option("--output-root")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Prepare one dev-only, one-GPU native Miles parser check; no training."""
    from training.io import file_sha256
    from training.miles_parser_probe import compile_probe, job_request

    try:
        source = _read(source_plan)
        plan = compile_probe(
            source,
            source_file_sha256=file_sha256(source_plan),
            name=name,
            output_root=output_root,
        )
        request = job_request(plan)
        _prepare(output, plan, request)
        _print(
            {
                "prepared": str(output),
                "cluster": "dev",
                "gpus": 1,
                "optimizer_steps": 0,
                "rollouts": 0,
                "submitted": False,
            }
        )
    except Exception as exc:
        _fail(exc)


@app.command("serving-dev")
@app.command("miles-serving-dev")
def miles_serving_dev(config: Path, output: Annotated[Path, typer.Option("--output")]) -> None:
    """Prepare one exact one-GPU Qwen SGLang canary on dev only."""
    from training.miles_serving_dev import compile_plan, job_request

    try:
        plan = compile_plan(_read(config))
        request = job_request(plan)
        _prepare(output, plan, request)
        _print(
            {
                "prepared": str(output),
                "cluster": "dev",
                "gpus": 1,
                "optimizer_updates": 0,
                "benchmark_attempts": 0,
                "submitted": False,
            }
        )
    except Exception as exc:
        _fail(exc)


@app.command("serving-dev-accept")
@app.command("miles-serving-dev-accept")
def miles_serving_dev_accept(
    directory: Path,
    result: Annotated[Path, typer.Option("--result")],
    result_sha256: Annotated[str, typer.Option("--result-sha256")],
    external: Annotated[Path, typer.Option("--external")],
    external_sha256: Annotated[str, typer.Option("--external-sha256")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Accept measured dev SGLang success only after exact GPU release evidence."""
    from training.miles_serving_dev import accept

    try:
        plan, _ = _prepared(directory)
        if plan.get("schema") not in _SERVING_DEV_PLAN_SCHEMAS:
            raise ValueError("serving acceptance requires a prepared Qwen dev canary")
        receipt = accept(
            plan,
            result.resolve(),
            result_sha256,
            external.resolve(),
            external_sha256,
            output.resolve(),
        )
        _print(
            {
                "status": receipt["status"],
                "receipt_sha256": receipt["receipt_sha256"],
                "gpu_release_verified": True,
                "production_registration_executed": False,
            }
        )
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


@app.command("miles-rl-checkpoint-seal")
def miles_rl_checkpoint_seal(
    directory: Path, output: Annotated[Path, typer.Option("--output")]
) -> None:
    """CPU-only seal of one terminal Miles RL checkpoint; not reload acceptance."""
    from training.miles_reload import seal_training_checkpoint
    from training.miles_training import SCHEMA

    try:
        plan, _ = _prepared(directory)
        if plan.get("schema") != SCHEMA:
            raise ValueError("trained checkpoint sealing requires a prepared Miles RL plan")
        result = seal_training_checkpoint(plan, output)
        _print(
            {
                "sha256": result["sha256"],
                "files": len(result["files"]),
                "rollout_index": result["rollout_index"],
                "gpu_reload_verified": False,
            }
        )
    except Exception as exc:
        _fail(exc)


@app.command("miles-rl-reload")
def miles_rl_reload(config: Path, output: Annotated[Path, typer.Option("--output")]) -> None:
    """Prepare one dev-only, all-rank Miles restore with zero optimizer updates."""
    from training.miles_reload import compile_reload, job_request
    from training.sft import read_mapping

    try:
        plan = compile_reload(read_mapping(config), relative_to=config.resolve().parent)
        request = job_request(plan)
        _prepare(output, plan, request)
        _print(
            {
                "prepared": str(output),
                "checkpoint_rollout_index": plan["source_manifest"]["rollout_index"],
                "gpus": request["workers"] * request["gpus_per_worker"],
                "optimizer_updates": 0,
                "rollouts": 0,
                "submitted": False,
            }
        )
    except Exception as exc:
        _fail(exc)


@app.command("miles-hf-job")
def miles_hf_job(config: Path, output: Annotated[Path, typer.Option("--output")]) -> None:
    """Prepare exact active-dev3 HF export or one-GPU value-checked reload."""
    from training.miles_hf_export_job import _sfs, compile_job, job_request
    from training.sft import read_mapping

    try:
        if _sfs(str(output.resolve()), "HF prepared evidence root") != str(output.resolve()):
            raise ValueError("HF prepared evidence root is not durable SFS")
        plan = compile_job(read_mapping(config))
        request = job_request(plan)
        _prepare(output, plan, request)
        _print(
            {
                "prepared": str(output),
                "stage": plan["stage"],
                "gpus_reserved": plan["execution"]["gpus_per_worker"],
                "cuda_visible": plan["execution"]["cuda_visible"],
                "submitted": False,
            }
        )
    except Exception as exc:
        _fail(exc)


@app.command("miles-hf-bind-submission")
def miles_hf_bind_submission(
    directory: Path,
    source_commit: Annotated[str, typer.Option("--source-commit")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Seal the prepared bundle and create-once dev API journal after one POST."""
    from training.miles_hf_export_job import compile_submission_binding

    try:
        plan, request = _prepared(directory)
        if plan.get("schema") not in _MILES_HF_PLAN_SCHEMAS:
            raise ValueError("submission binding requires a prepared Miles HF job")
        result = compile_submission_binding(
            plan_path=directory / "plan.json",
            request_path=directory / "request.json",
            submission_journal_path=directory / "SUBMISSION.jsonl",
            source_commit=source_commit,
            output=output.resolve(),
        )
        _print({"status": "bound", "api": result["api"], "sha256": result["sha256"]})
    except Exception as exc:
        _fail(exc)


@app.command("miles-hf-accept-export")
def miles_hf_accept_export(directory: Path, submission: Path) -> None:
    """Accept an observed dev export only after its exact UIDs are durably released."""
    from training.miles_hf_export_job import accept_export_job

    try:
        plan, _ = _prepared(directory)
        if plan.get("schema") not in _MILES_HF_PLAN_SCHEMAS or plan.get("stage") != "export":
            raise ValueError("export acceptance requires a prepared Miles HF export job")
        root = Path(plan["output_root"])
        result = accept_export_job(
            plan_path=(directory / "plan.json").resolve(),
            submission_path=submission.resolve(),
            controller_path=root / "HF_EXPORT_CONTROLLER_TERMINAL.json",
            release_path=root / "HF_EXPORT_RELEASE.json",
            output=root / "HF_EXPORT_ACCEPTED.json",
        )
        _print({"status": result["status"], "sha256": result["sha256"], "serving": False})
    except Exception as exc:
        _fail(exc)


@app.command("miles-hf-accept-reload")
def miles_hf_accept_reload(directory: Path, submission: Path) -> None:
    """Accept the one-GPU HF load/probe only after exact external GPU release."""
    from training.miles_hf_export import accept_gpu_reload

    try:
        plan, _ = _prepared(directory)
        if plan.get("schema") not in _MILES_HF_PLAN_SCHEMAS or plan.get("stage") != "reload":
            raise ValueError("reload acceptance requires a prepared Miles HF reload job")
        root = Path(plan["output_root"])
        result = accept_gpu_reload(
            plan_path=(directory / "plan.json").resolve(),
            submission_path=submission.resolve(),
            result_path=root / "HF_RELOAD_VALIDATED.json",
            controller_path=root / "HF_RELOAD_CONTROLLER_TERMINAL.json",
            release_path=root / "HF_RELOAD_RELEASE.json",
            output=root / "HF_RELOAD_ACCEPTED.json",
        )
        _print({"status": result["status"], "sha256": result["sha256"], "serving": False})
    except Exception as exc:
        _fail(exc)


@app.command()
def status(
    name: str,
    cluster: Annotated[
        Cluster, typer.Option(help="Jobs API target; defaults to prod for historical runs.")
    ] = Cluster.prod,
) -> None:
    """Read sanitized state; use --cluster dev for new dev runs. No private logs."""
    try:
        with _client(cluster) as client:
            _print({"api_base_url": API_URLS[cluster], **client.status(name)})
    except Exception as exc:
        _fail(exc)


@app.command("checkpoint-seal")
def checkpoint_seal(
    directory: Path,
    step: int,
    output: Annotated[Path, typer.Option("--output")],
    source_plan_file: Annotated[
        Path | None,
        typer.Option(
            "--source-plan-file",
            help=(
                "Exact producer plan file when its byte SHA, rather than canonical JSON SHA, "
                "was recorded."
            ),
        ),
    ] = None,
) -> None:
    """CPU-only: hash a trusted run's native checkpoint for export/resume. No GPU reload."""
    from training.checkpoints import seal

    try:
        plan, _ = _prepared(directory)
        result = (
            seal(plan, step, output)
            if source_plan_file is None
            else seal(plan, step, output, source_plan_file=source_plan_file)
        )
        _print({k: result[k] for k in ("optimizer_step", "total_bytes", "receipt_sha256")})
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
