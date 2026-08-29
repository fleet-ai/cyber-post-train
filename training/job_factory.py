"""Render suspended, queue-safe SFT and sequential online-RL Jobs."""

from __future__ import annotations

import base64
import io
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .cluster_policy import EXPERIMENT_LABEL, MODEL_LABEL, OWNER_LABEL
from .model_adapter import ModelAdapter

DATASET_DIGEST = "sha256:05c7928155f7d3d0c9d387164e559985c2498d0c40b32a2a0884f6c629fb0494"
DATASET_ROOT = (
    "/mnt/sfs/cyber-post-train/data/"
    "05c7928155f7d3d0c9d387164e559985c2498d0c40b32a2a0884f6c629fb0494"
)
REWARD_CONTRACT = "execution_verifier_x_behavior_x_integrity_v1"
EVALUATION_CONTRACT = "sealed_fleet_holdout_plus_external_webexploitbench_v1"
_DNS = re.compile(r"^[a-z0-9](?:[-a-z0-9]*[a-z0-9])?$")
_BOOTSTRAP = """
import base64, os, pathlib, sys
target = pathlib.Path('/tmp/cyber-post-train-runtime.zip')
target.write_bytes(base64.b64decode(os.environ['TRAINING_RUNTIME_B64'], validate=True))
os.environ['PYTHONPATH'] = str(target) + os.pathsep + os.environ.get('PYTHONPATH', '')
os.execv(sys.executable, [sys.executable, '-m', 'training.job_runtime', os.environ['RUNTIME_MODE']])
""".strip()


@dataclass(frozen=True)
class TrainingInputs:
    dataset_root: str = DATASET_ROOT
    dataset_digest: str = DATASET_DIGEST
    reward_contract: str = REWARD_CONTRACT
    evaluation_contract: str = EVALUATION_CONTRACT

    def __post_init__(self) -> None:
        if (
            self.dataset_root,
            self.dataset_digest,
            self.reward_contract,
            self.evaluation_contract,
        ) != (
            DATASET_ROOT,
            DATASET_DIGEST,
            REWARD_CONTRACT,
            EVALUATION_CONTRACT,
        ):
            raise ValueError("dataset, split, and reward science is sealed for this experiment")


def _runtime_archive() -> str:
    root = Path(__file__).resolve().parent
    files = {
        "training/__init__.py": root / "__init__.py",
        "training/io.py": root / "io.py",
        "training/job_runtime.py": root / "job_runtime.py",
    }
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(files.items()):
            archive.writestr(name, path.read_bytes())
    return base64.b64encode(stream.getvalue()).decode()


def _env(name: str, value: str) -> dict[str, str]:
    return {"name": name, "value": value}


def _common_env(
    adapter: ModelAdapter,
    inputs: TrainingInputs,
    *,
    sft_name: str,
    checkpoint_dir: str,
    manifest_path: str,
    receipt_path: str,
) -> list[dict[str, str]]:
    return [
        _env("TRAINING_RUNTIME_B64", _runtime_archive()),
        _env("MODEL_ID", adapter.model_id),
        _env("MODEL_REVISION", adapter.revision),
        _env("MODEL_CHECKPOINT_LOCK", adapter.checkpoint_lock),
        _env("MODEL_WEIGHTS_MANIFEST_SHA256", adapter.weights_manifest_sha256),
        _env("MODEL_ADAPTER_DIGEST", adapter.digest),
        _env("MODEL_DRIVER_CONTRACT", adapter.driver_contract),
        _env("MODEL_COMPATIBILITY_RECEIPT", adapter.compatibility_receipt),
        _env("TRAINING_IMAGE", adapter.image),
        _env("DATASET_DIGEST", inputs.dataset_digest),
        _env("SFT_JOB_NAME", sft_name),
        _env("SFT_CHECKPOINT_DIR", checkpoint_dir),
        _env("SFT_CHECKPOINT_MANIFEST", manifest_path),
        _env("SFT_CHECKPOINT_RECEIPT", receipt_path),
        _env("HF_HUB_OFFLINE", "1"),
        _env("TRANSFORMERS_OFFLINE", "1"),
    ]


def _container(
    adapter: ModelAdapter,
    *,
    name: str,
    mode: str,
    env: list[dict[str, str]],
    argv: list[str] | None = None,
    gpu: bool = True,
) -> dict[str, Any]:
    values = [*env, _env("RUNTIME_MODE", mode)]
    if argv is not None:
        values.append(_env("TRAINING_ARGV_JSON", json.dumps(argv, separators=(",", ":"))))
    resources = adapter.resources
    requests = {"cpu": resources.cpu_request, "memory": resources.memory_request}
    limits = {"cpu": resources.cpu_limit, "memory": resources.memory_limit}
    if gpu:
        requests["nvidia.com/gpu"] = str(resources.gpus)
        limits["nvidia.com/gpu"] = str(resources.gpus)
    return {
        "name": name,
        "image": adapter.image,
        "imagePullPolicy": "IfNotPresent",
        "workingDir": adapter.working_dir,
        "command": [adapter.python, "-c", _BOOTSTRAP],
        "env": values,
        "resources": {"requests": requests, "limits": limits},
        "volumeMounts": [{"name": "sfs", "mountPath": "/mnt/sfs"}],
    }


def _job(
    adapter: ModelAdapter,
    *,
    name: str,
    experiment: str,
    container: dict[str, Any],
    init_containers: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    labels = {
        "kueue.x-k8s.io/queue-name": "training-lq",
        OWNER_LABEL: "chris",
        EXPERIMENT_LABEL: experiment,
        MODEL_LABEL: adapter.slug,
    }
    pod_spec: dict[str, Any] = {
        "restartPolicy": "Never",
        "preemptionPolicy": "Never",
        "imagePullSecrets": [{"name": "ghcr-pull"}],
        "nodeSelector": {"workload": "fleetai-training-ng-gpu"},
        "tolerations": [
            {
                "key": "workload",
                "operator": "Equal",
                "value": "fleetai-training-ng-gpu",
                "effect": "NoSchedule",
            }
        ],
        "containers": [container],
        "volumes": [{"name": "sfs", "persistentVolumeClaim": {"claimName": "sfs-shared"}}],
    }
    if init_containers:
        pod_spec["initContainers"] = init_containers
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": name,
            "namespace": "fleet-train-jobs",
            "labels": labels,
            "annotations": {
                "cyber-post-train.fleet.ai/model-revision": adapter.revision,
                "cyber-post-train.fleet.ai/model-adapter-digest": adapter.digest,
                "cyber-post-train.fleet.ai/dataset-digest": DATASET_DIGEST,
                "cyber-post-train.fleet.ai/evaluation-contract": EVALUATION_CONTRACT,
                "cyber-post-train.fleet.ai/priority-contract": "priority-0-nonpreempting",
            },
        },
        "spec": {
            "suspend": True,
            "backoffLimit": 0,
            "ttlSecondsAfterFinished": 604800,
            "template": {
                "metadata": {
                    "labels": labels,
                    "annotations": {
                        "kueue.x-k8s.io/podset-required-topology": "kubernetes.io/hostname"
                    },
                },
                "spec": pod_spec,
            },
        },
    }


def render_job_pair(
    adapter: ModelAdapter, run_id: str, inputs: TrainingInputs | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    inputs = inputs or TrainingInputs()
    if not _DNS.fullmatch(run_id):
        raise ValueError("run_id must be a DNS label")
    prefix = f"chris-cyber-{adapter.slug}-{run_id}"
    sft_name, rl_name = f"{prefix}-sft", f"{prefix}-rl"
    if max(len(sft_name), len(rl_name)) > 63:
        raise ValueError("generated job name exceeds DNS-63")
    output_root = f"/mnt/sfs/cyber-post-train/runs/{adapter.slug}/{run_id}"
    checkpoint_dir = f"{output_root}/sft/checkpoint"
    manifest_path = f"{output_root}/sft/checkpoint-manifest.json"
    receipt_path = f"{output_root}/sft/receipt.json"
    common = _common_env(
        adapter,
        inputs,
        sft_name=sft_name,
        checkpoint_dir=checkpoint_dir,
        manifest_path=manifest_path,
        receipt_path=receipt_path,
    )
    values = {
        "model_root": adapter.model_root,
        "checkpoint_dir": checkpoint_dir,
        "sft_train": f"{inputs.dataset_root}/sft_train.jsonl",
        "sft_dev": f"{inputs.dataset_root}/sft_dev.jsonl",
        "rl_prompts": f"{inputs.dataset_root}/rl_prompts.jsonl",
        "reward_contract": inputs.reward_contract,
    }
    sft = _job(
        adapter,
        name=sft_name,
        experiment=run_id,
        container=_container(
            adapter,
            name="sft",
            mode="run-sft",
            env=common,
            argv=adapter.format_argv("sft", values),
        ),
    )
    gate = _container(adapter, name="sft-checkpoint-gate", mode="gate-rl", env=common, gpu=False)
    rl = _job(
        adapter,
        name=rl_name,
        experiment=run_id,
        init_containers=[gate],
        container=_container(
            adapter,
            name="online-rl",
            mode="run-rl",
            env=common,
            argv=adapter.format_argv("rl", values),
        ),
    )
    return sft, rl
