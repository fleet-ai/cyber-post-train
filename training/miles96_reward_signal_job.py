"""Render the create-once CPU Job for the bounded Miles96 reward check."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from evals.fleet import evaluate
from training import miles96_reward_signal as signal

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "cyber_qwen38_miles96_reward_signal_job_packet_v1"
CONTEXT = "nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6"
NAMESPACE = "fleet-train-jobs"
JOB_NAME = "chris-q38-m96-signal-a1"
CONFIG_MAP_NAME = JOB_NAME + "-code"
OUTPUT = "/mnt/sfs/jobs/chris-q38-m96-signal-a1"
DATABASE = "q38_m96_signal_a1"
SERVED_ID = signal.SERVED_ID
INFERENCE_MODEL_UID = "eb8168a9-67fa-4662-b895-241b32f3c290"
MODEL_REVISION = signal.MODEL_REVISION
MODEL_PATH = "/scratch/models/chris-autoresearch/chris-ar-q38-base-0918-v1"
EVALUATOR_IMAGE = (
    "ghcr.io/astral-sh/uv:python3.12-bookworm@"
    "sha256:9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7"
)
DIND_IMAGE = (
    "docker.io/library/docker@"
    "sha256:f649ef046008ca7f926a2571c32b0ac22e5c59eb61b959617f9acc2a4c638cf5"
)
AGENT_IMAGE = "sha256:c7d048c98e6b8e52e5b76ab4006a7626b1ccf63a37bfa4b47ecd0fe9028e1f92"
PROXY_IMAGE = (
    "ghcr.io/astral-sh/uv:python3.12-bookworm@"
    "sha256:9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7"
)
SOURCE_FILES = {
    "jobs.py": "cyber_post_train/jobs.py",
    "cluster_entry.py": "evals/fleet/cluster_entry.py",
    "evaluate.py": "evals/fleet/evaluate.py",
    "exact_pass4_crypto.py": "evals/fleet/exact_pass4_crypto.py",
    "exact_pass4_universe.py": "evals/fleet/exact_pass4_universe.py",
    "fixed_proxy.py": "evals/fleet/fixed_proxy.py",
    "model_artifact.py": "evals/fleet/model_artifact.py",
    "opencode_self_hosted.py": "evals/fleet/opencode_self_hosted.py",
    "rollout_campaign.py": "evals/fleet/rollout_campaign.py",
    "rollout_ledger.py": "evals/fleet/rollout_ledger.py",
    "rollout_postgres.py": "evals/fleet/rollout_postgres.py",
    "rollout_worker.py": "evals/fleet/rollout_worker.py",
    "run.sh": "evals/fleet/scripts/run_miles96_reward_signal_v1.sh",
    "signal.py": "training/miles96_reward_signal.py",
}


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def task_selection() -> dict[str, Any]:
    body = {
        "schema": "cyber_qwen38_miles96_signal_task_selection_v1",
        "source_job_id": None,
        "task_set_sha256": signal.TASK_SET_SHA256,
        "verifier_version_id": signal.VERIFIER_VERSION_ID,
        "tasks": [
            {
                "task_key": signal.TASK_KEY,
                "task_version_id": signal.TASK_VERSION_ID,
                "env_key": "cysec1-2-fira-gen",
                "env_version": "v0.0.3",
                "environment_version_id": "e83377eb-4f65-49be-b9c2-37a1fabc43e4",
                "data_key": "cultivacore",
                "data_version": "v0.0.61",
            }
        ],
    }
    return {**body, "sha256": "sha256:" + digest(body)}


def evaluation_config() -> dict[str, Any]:
    return {
        "name": "q38-m96-signal-a1",
        "task_set": "task-selection.json",
        "models": {
            "base": {
                "repository": "Qwen/Qwen3.8-27B",
                "revision": MODEL_REVISION,
                "session_model": f"qwen/{SERVED_ID}",
            }
        },
        "routes": {
            "base": {
                "model": "base",
                "served_id": SERVED_ID,
                "task_versions": [signal.TASK_VERSION_ID],
                "catalog": {
                    "engine": "sglang",
                    "precision": "bf16",
                    "tensor_parallel_size": 1,
                },
                "model_info": {
                    "model_path": MODEL_PATH,
                    "model_type": "qwen3_5",
                    "architectures": ["Qwen3_5ForConditionalGeneration"],
                },
                "server_info": {
                    "model_path": MODEL_PATH,
                    "context_length": 262144,
                    "tp_size": 1,
                    "quantization": None,
                    "kv_cache_dtype": "fp8_e4m3",
                    "reasoning_parser": "qwen3",
                    "tool_call_parser": "qwen3_coder",
                },
                "endpoint_origin": "https://inference.flt.build",
            }
        },
        "harness": {
            "harness": "opencode",
            "harness_version": "1.18.27",
            "release_asset_sha256": (
                "sha256:4af5494f9433f59db8c1e344198f0ee72a50c06ec009fb4a8aeab4c2d4abd702"
            ),
            "provider_adapter": "@ai-sdk/openai-compatible",
            "context_management": "opencode_1.18.27_native_compaction_autocontinue_v2",
            "context_window_size": 262144,
            "compaction_headroom_tokens": 20000,
            "max_output_tokens": signal.MAX_TOKENS_PER_TURN,
            "max_model_requests": signal.MAX_TURNS,
            "timeout_seconds": signal.EPISODE_TIMEOUT_S,
            "tools": ["bash", "submit_report"],
            "tool_catalog_sha256": signal.TOOL_CATALOG_SHA256,
        },
        "images": {"agent": AGENT_IMAGE, "proxy": PROXY_IMAGE},
        "pass_k": signal.EPISODES,
        "concurrency": signal.EPISODES,
        "max_reviewed_infrastructure_retries": 0,
        "training_data_eligible": False,
        "sampling": {"temperature": 0.6, "top_p": 0.95, "seed": 20260921},
    }


def _sources() -> dict[str, str]:
    return {name: (ROOT / path).read_text() for name, path in SOURCE_FILES.items()}


def _config_map(data: dict[str, str]) -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": CONFIG_MAP_NAME, "namespace": NAMESPACE},
        "immutable": True,
        "data": data,
    }


def _job() -> dict[str, Any]:
    labels = {
        "cyber-post-train.fleet.ai/owner": "chris",
        "cyber-post-train.fleet.ai/role": "miles96-reward-signal",
        "kueue.x-k8s.io/queue-name": "training-lq",
        "kueue.x-k8s.io/priority-class": "q1",
    }
    annotations = {
        "fleet.ai/failure-alerts": "off",
        "cyber-post-train.fleet.ai/create-once": "true",
    }
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": JOB_NAME,
            "namespace": NAMESPACE,
            "labels": labels,
            "annotations": annotations,
        },
        "spec": {
            "activeDeadlineSeconds": 7200,
            "backoffLimit": 0,
            "completions": 1,
            "parallelism": 1,
            "suspend": True,
            "template": {
                "metadata": {
                    "labels": {**labels, "cyber-post-train.fleet.ai/postgres-client": "true"},
                    "annotations": annotations,
                },
                "spec": {
                    "priorityClassName": "c1",
                    "restartPolicy": "Never",
                    "nodeSelector": {
                        "kubernetes.io/arch": "amd64",
                        "workload": "fleetai-training-ng-cpu",
                    },
                    "tolerations": [
                        {
                            "key": "workload",
                            "operator": "Equal",
                            "value": "fleetai-training-ng-cpu",
                            "effect": "NoSchedule",
                        }
                    ],
                    "initContainers": [
                        {
                            "name": "dind",
                            "image": DIND_IMAGE,
                            "restartPolicy": "Always",
                            "args": ["--host=unix:///var/run/docker.sock", "--tls=false"],
                            "securityContext": {"privileged": True},
                            "resources": {
                                "requests": {
                                    "cpu": "2",
                                    "memory": "2Gi",
                                    "ephemeral-storage": "10Gi",
                                },
                                "limits": {
                                    "cpu": "4",
                                    "memory": "4Gi",
                                    "ephemeral-storage": "40Gi",
                                },
                            },
                            "volumeMounts": [
                                {"name": "docker-socket", "mountPath": "/var/run"},
                                {"name": "docker-data", "mountPath": "/var/lib/docker"},
                                {"name": "docker-bind", "mountPath": "/docker-bind"},
                                {"name": "workspace", "mountPath": "/workspace"},
                                {"name": "sfs", "mountPath": "/mnt/sfs"},
                            ],
                        }
                    ],
                    "containers": [
                        {
                            "name": "evaluator",
                            "image": EVALUATOR_IMAGE,
                            "command": ["/bin/bash", "-ceu", "--"],
                            "args": [
                                "apt-get update\n"
                                "apt-get install --yes --no-install-recommends docker.io\n"
                                "exec /bin/bash /bootstrap/run.sh\n"
                            ],
                            "env": [
                                {"name": "DOCKER_HOST", "value": "unix:///var/run/docker.sock"},
                                {"name": "DOCKER_TLS_CERTDIR", "value": ""},
                                {"name": "DOCKER_BIND_ROOT", "value": "/docker-bind"},
                                {"name": "EVAL_OUTPUT", "value": OUTPUT},
                                {"name": "EVAL_DATABASE", "value": DATABASE},
                                {
                                    "name": "FLEET_API_KEY",
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "name": "chris-cyber-opencode-evals-v2",
                                            "key": "FLEET_API_KEY",
                                        }
                                    },
                                },
                                {
                                    "name": "ROLLOUT_DATABASE_URL",
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "name": "chris-cyber-rollout-postgres-v1",
                                            "key": "ROLLOUT_DATABASE_URL",
                                        }
                                    },
                                },
                            ],
                            "resources": {
                                "requests": {
                                    "cpu": "4",
                                    "memory": "32Gi",
                                    "ephemeral-storage": "10Gi",
                                },
                                "limits": {
                                    "cpu": "8",
                                    "memory": "48Gi",
                                    "ephemeral-storage": "40Gi",
                                },
                            },
                            "volumeMounts": [
                                {"name": "docker-socket", "mountPath": "/var/run"},
                                {"name": "bootstrap", "mountPath": "/bootstrap", "readOnly": True},
                                {"name": "docker-bind", "mountPath": "/docker-bind"},
                                {"name": "workspace", "mountPath": "/workspace"},
                                {"name": "sfs", "mountPath": "/mnt/sfs"},
                            ],
                        }
                    ],
                    "volumes": [
                        {"name": "bootstrap", "configMap": {"name": CONFIG_MAP_NAME}},
                        {"name": "docker-data", "emptyDir": {"sizeLimit": "80Gi"}},
                        {"name": "docker-socket", "emptyDir": {}},
                        {"name": "docker-bind", "emptyDir": {"sizeLimit": "2Gi"}},
                        {"name": "workspace", "emptyDir": {"sizeLimit": "5Gi"}},
                        {"name": "sfs", "persistentVolumeClaim": {"claimName": "sfs-shared"}},
                    ],
                },
            },
        },
    }


def build_packet() -> dict[str, Any]:
    config = evaluation_config()
    selection = task_selection()
    source = _sources()
    data = {
        **source,
        "config.json": json.dumps(config, sort_keys=True, separators=(",", ":")),
        "task-selection.json": json.dumps(selection, sort_keys=True, separators=(",", ":")),
    }
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        (root / "task-selection.json").write_text(data["task-selection.json"])
        plan = evaluate.compile_eval(config, relative_to=root)
    config_map = _config_map(data)
    job = _job()
    sequence = {
        "exact_name_duplicate_census_required": True,
        "output_absence_required": True,
        "database_absence_required": True,
        "base_route_exact_readiness_required": True,
        "server_dry_run_count": 2,
        "stable_preview_digests_must_match": True,
        "config_map_create_request_count": 1,
        "config_map_create_retry_allowed": False,
        "job_create_request_count": 1,
        "job_create_retry_allowed": False,
        "job_created_suspended": True,
        "post_create_checks": [
            "exact_config_map_uid_bound",
            "exact_job_uid_bound",
            "rendered_root_failure_alerts_off",
            "rendered_priority_class_c1",
            "rendered_gpu_requests_and_limits_zero",
            "preview_suspend_true",
            "live_suspend_is_controller_managed",
            "exact_base_route_ready_and_parity_bound",
        ],
        "controller_managed_unsuspend": True,
        "operator_patch_request_count": 0,
        "post_create_or_patch_requests_allowed": False,
        "exact_uid_terminal_monitor_and_cleanup_required": True,
    }
    body = {
        "schema": SCHEMA,
        "context": CONTEXT,
        "namespace": NAMESPACE,
        "job_name": JOB_NAME,
        "config_map_name": CONFIG_MAP_NAME,
        "output": OUTPUT,
        "database": DATABASE,
        "task_identity": {
            "task_key": signal.TASK_KEY,
            "task_version_id": signal.TASK_VERSION_ID,
            "verifier_version_id": signal.VERIFIER_VERSION_ID,
            "task_set_sha256": signal.TASK_SET_SHA256,
            "tool_catalog_sha256": signal.TOOL_CATALOG_SHA256,
        },
        "route_identity": {
            "served_id": SERVED_ID,
            "inference_model_uid": INFERENCE_MODEL_UID,
            "model_revision": MODEL_REVISION,
            "must_be_ready_before_create": True,
        },
        "episode": {
            "count": signal.EPISODES,
            "concurrency": signal.EPISODES,
            "max_turns": signal.MAX_TURNS,
            "max_tokens_per_turn": signal.MAX_TOKENS_PER_TURN,
            "timeout_seconds": signal.EPISODE_TIMEOUT_S,
        },
        "resources": {
            "gpus": 0,
            "priority_class": "c1",
            "queue_priority": "q1",
            "failure_alerts": "off",
            "active_deadline_seconds": 7200,
            "backoff_limit": 0,
        },
        "plan_sha256": "sha256:" + plan["sha256"],
        "execution_sequence": sequence,
        "bundle": {"apiVersion": "v1", "kind": "List", "items": [config_map, job]},
    }
    return {**body, "sha256": "sha256:" + digest(body)}


def validate_packet(packet: dict[str, Any]) -> dict[str, Any]:
    if packet != build_packet():
        raise ValueError("Miles96 reward-signal packet differs from renderer")
    config_map, job = packet["bundle"]["items"]
    pod = job["spec"]["template"]["spec"]
    if (
        config_map.get("immutable") is not True
        or job["metadata"]["annotations"].get("fleet.ai/failure-alerts") != "off"
        or job["spec"].get("backoffLimit") != 0
        or job["spec"].get("suspend") is not True
        or pod.get("priorityClassName") != "c1"
        or "nvidia.com/gpu" in json.dumps(pod.get("containers", []))
        or "nvidia.com/gpu" in json.dumps(pod.get("initContainers", []))
    ):
        raise ValueError("Miles96 reward-signal safety contract drifted")
    if len(json.dumps(config_map).encode()) >= 950_000:
        raise ValueError("Miles96 reward-signal ConfigMap is too large")
    sequence = packet["execution_sequence"]
    if (
        sequence.get("config_map_create_request_count") != 1
        or sequence.get("config_map_create_retry_allowed") is not False
        or sequence.get("job_create_request_count") != 1
        or sequence.get("job_create_retry_allowed") is not False
        or sequence.get("job_created_suspended") is not True
        or sequence.get("controller_managed_unsuspend") is not True
        or sequence.get("operator_patch_request_count") != 0
        or sequence.get("post_create_or_patch_requests_allowed") is not False
    ):
        raise ValueError("Miles96 reward-signal create/release sequence drifted")
    return {
        "packet_sha256": packet["sha256"],
        "bundle_sha256": "sha256:" + digest(packet["bundle"]),
        "job": JOB_NAME,
        "config_map": CONFIG_MAP_NAME,
        "planned_episodes": signal.EPISODES,
        "gpus": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    packet = build_packet()
    validate_packet(packet)
    encoded = json.dumps(packet, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(encoded)


if __name__ == "__main__":
    main()
