"""Render the queued one-step GLM-5.2 Megatron/LoRA compatibility Job."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml

from .compatibility_job import IMAGE_REPOSITORY, MODEL_ROOT, NEMO_RL_COMMIT, RESULT_ROOT
from .io import atomic_write_text

IMAGE_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
JOB_NAME = "chris-cyber-glm52-model-probe-b4734de4"
PROBE_ROOT = f"{RESULT_ROOT}/model-probe"


def probe_config() -> dict[str, Any]:
    return {
        "defaults": "/opt/nemo-rl/examples/configs/sft.yaml",
        "sft": {
            "max_num_epochs": 1,
            "max_num_steps": 1,
            "val_period": 1,
            "val_batches": 0,
            "val_global_batch_size": 1,
            "val_micro_batch_size": 1,
            "val_at_start": False,
            "val_at_end": False,
            "seed": 314159,
        },
        "checkpointing": {
            "enabled": True,
            "checkpoint_dir": f"{PROBE_ROOT}/checkpoints",
            "keep_top_k": 1,
            "save_period": 1,
            "save_optimizer": True,
        },
        "policy": {
            "model_name": MODEL_ROOT,
            "hf_config_overrides": {"qk_rope_head_dim": 64},
            "tokenizer": {"name": MODEL_ROOT, "chat_template": "default"},
            "train_global_batch_size": 1,
            "train_micro_batch_size": 1,
            "max_total_sequence_length": 256,
            "precision": "bfloat16",
            "dtensor_cfg": {"enabled": False},
            "megatron_cfg": {
                "enabled": True,
                "activation_checkpointing": True,
                "tensor_model_parallel_size": 1,
                "expert_tensor_parallel_size": 1,
                "expert_model_parallel_size": 8,
                "pipeline_model_parallel_size": 1,
                "context_parallel_size": 1,
                "sequence_parallel": False,
                "moe_router_dtype": "fp32",
                "moe_token_dispatcher_type": "allgather",
                "apply_rope_fusion": False,
                "defer_fp32_logits": True,
                "model_overrides": {
                    "dsa_indexer_rope_interleaved": True,
                    "dsa_indexer_rotate_activation": False,
                    "dsa_indexer_k_norm_epsilon": 1.0e-6,
                    "dsa_kernel_backend": "cudnn",
                    "dsa_indexer_loss_coeff": 0.0,
                    "dsa_indexer_use_sparse_loss": False,
                },
                "peft": {
                    "enabled": True,
                    "target_modules": [],
                    "exclude_modules": [],
                    "dim": 8,
                    "alpha": 8,
                    "dropout": 0.0,
                    "dropout_position": "post",
                    "lora_A_init_method": "xavier",
                    "lora_B_init_method": "zero",
                    "a2a_experimental": False,
                    "lora_dtype": None,
                },
                "optimizer": {
                    "lr": 1.0e-6,
                    "min_lr": 1.0e-6,
                    "weight_decay": 0.0,
                    "params_dtype": "bfloat16",
                    "use_precision_aware_optimizer": False,
                    "optimizer_cpu_offload": False,
                    "optimizer_offload_fraction": 0.0,
                },
                "scheduler": {
                    "lr_decay_iters": 1,
                    "lr_warmup_iters": 0,
                    "lr_warmup_init": 1.0e-6,
                },
                "distributed_data_parallel_config": {
                    "grad_reduce_in_fp32": True,
                    "overlap_grad_reduce": False,
                    "overlap_param_gather": False,
                },
            },
            "optimizer": None,
            "sequence_packing": {"enabled": False},
            "dynamic_batching": {"enabled": False},
            "make_sequence_length_divisible_by": 1,
        },
        "data": {
            "max_input_seq_length": 256,
            "add_bos": True,
            "add_eos": True,
            "add_generation_prompt": False,
            "shuffle": False,
            "num_workers": 0,
            "train": {
                "dataset_name": "ResponseDataset",
                "data_path": "/tmp/glm52-model-probe.jsonl",
                "input_key": "input",
                "output_key": "output",
            },
            "validation": None,
            "default": {
                "prompt_file": None,
                "system_prompt_file": None,
                "processor": "sft_processor",
            },
        },
        "logger": {
            "log_dir": f"{PROBE_ROOT}/logs",
            "wandb_enabled": False,
            "tensorboard_enabled": False,
            "mlflow_enabled": False,
            "swanlab_enabled": False,
            "monitor_gpus": False,
        },
        "cluster": {"gpus_per_node": 8, "num_nodes": 1, "segment_size": None},
    }


def _b64(value: bytes) -> str:
    return base64.b64encode(value).decode()


def render_model_probe_job(image_digest: str) -> dict[str, Any]:
    if not IMAGE_DIGEST.fullmatch(image_digest):
        raise ValueError("image digest must be an immutable sha256 digest")
    config_bytes = yaml.safe_dump(probe_config(), sort_keys=False).encode()
    sample = json.dumps(
        {"input": "Reply with exactly: compatibility-ready", "output": "compatibility-ready"},
        sort_keys=True,
    ).encode() + b"\n"
    receipt_filter = (
        '{schema:"glm52_model_probe_v1",status:"succeeded",'
        "image_digest:$image,model_revision:$revision,config_sha256:$config,"
        "checks:{hf_to_megatron:true,expert_parallel_8:true,lora_attached:true,"
        "bf16_forward_backward:true,optimizer_step:true,checkpoint_saved:true,"
        "checkpoint_resumed:true}}"
    )
    command = f"""
test -s {RESULT_ROOT}/environment.json
test -s {RESULT_ROOT}/nccl.json
install -d -m 0755 {PROBE_ROOT}
printf '%s' "$MODEL_PROBE_CONFIG_B64" | base64 -d > /tmp/glm52-model-probe.yaml
printf '%s' "$MODEL_PROBE_DATA_B64" | base64 -d > /tmp/glm52-model-probe.jsonl
printf '%s  %s\n' "$MODEL_PROBE_CONFIG_SHA256" /tmp/glm52-model-probe.yaml | sha256sum -c -
uv run --frozen --extra mcore python examples/run_sft.py --config /tmp/glm52-model-probe.yaml
test -d {PROBE_ROOT}/checkpoints
uv run --frozen --extra mcore python examples/run_sft.py \
  --config /tmp/glm52-model-probe.yaml \
  sft.max_num_epochs=2 sft.max_num_steps=2
jq -n \
  --arg image "$TRAINING_IMAGE_DIGEST" \
  --arg revision "$MODEL_REVISION" \
  --arg config "sha256:$MODEL_PROBE_CONFIG_SHA256" \
  '{receipt_filter}' \
  > {PROBE_ROOT}/receipt.json
echo MODEL_PROBE_COMPLETE {PROBE_ROOT}/receipt.json
""".strip()
    labels = {
        "kueue.x-k8s.io/queue-name": "training-lq",
        "cyber-post-train.fleet.ai/owner": "chris",
        "cyber-post-train.fleet.ai/experiment": "model-probe-b4734de4",
    }
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": JOB_NAME,
            "namespace": "fleet-train-jobs",
            "labels": labels,
            "annotations": {
                "cyber-post-train.fleet.ai/training-image-digest": image_digest,
                "cyber-post-train.fleet.ai/model-revision": MODEL_ROOT.rsplit("/", 1)[-1],
            },
        },
        "spec": {
            "suspend": True,
            "backoffLimit": 0,
            "activeDeadlineSeconds": 86400,
            "ttlSecondsAfterFinished": 604800,
            "template": {
                "metadata": {
                    "labels": {key: value for key, value in labels.items() if "/" in key},
                    "annotations": {
                        "kueue.x-k8s.io/podset-required-topology": "kubernetes.io/hostname"
                    },
                },
                "spec": {
                    "restartPolicy": "Never",
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
                    "containers": [
                        {
                            "name": "model-probe",
                            "image": f"{IMAGE_REPOSITORY}@{image_digest}",
                            "imagePullPolicy": "IfNotPresent",
                            "workingDir": "/opt/nemo-rl",
                            "command": ["/bin/bash", "-ceu", "--"],
                            "args": [command],
                            "env": [
                                {"name": "MODEL_PROBE_CONFIG_B64", "value": _b64(config_bytes)},
                                {"name": "MODEL_PROBE_DATA_B64", "value": _b64(sample)},
                                {
                                    "name": "MODEL_PROBE_CONFIG_SHA256",
                                    "value": hashlib.sha256(config_bytes).hexdigest(),
                                },
                                {"name": "TRAINING_IMAGE_DIGEST", "value": image_digest},
                                {"name": "NEMO_RL_COMMIT", "value": NEMO_RL_COMMIT},
                                {"name": "MODEL_REVISION", "value": MODEL_ROOT.rsplit("/", 1)[-1]},
                                {
                                    "name": "NRL_MEGATRON_CHECKPOINT_DIR",
                                    "value": "/mnt/sfs/cyber-post-train/megatron-cache/b4734de4",
                                },
                                {"name": "HF_HUB_OFFLINE", "value": "1"},
                                {"name": "TRANSFORMERS_OFFLINE", "value": "1"},
                                {
                                    "name": "PYTORCH_CUDA_ALLOC_CONF",
                                    "value": "expandable_segments:False",
                                },
                            ],
                            "resources": {
                                "requests": {
                                    "cpu": "96",
                                    "memory": "1200Gi",
                                    "nvidia.com/gpu": "8",
                                },
                                "limits": {
                                    "cpu": "192",
                                    "memory": "2Ti",
                                    "nvidia.com/gpu": "8",
                                },
                            },
                            "volumeMounts": [{"name": "sfs", "mountPath": "/mnt/sfs"}],
                        }
                    ],
                    "volumes": [
                        {"name": "sfs", "persistentVolumeClaim": {"claimName": "sfs-shared"}}
                    ],
                },
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atomic_write_text(
        args.output, yaml.safe_dump(render_model_probe_job(args.image_digest), sort_keys=False)
    )
    print(args.output)


if __name__ == "__main__":
    main()
