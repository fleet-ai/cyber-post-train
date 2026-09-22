"""Offline lane2 authority, compiler and alert-safe renderer tests."""

from __future__ import annotations

import base64
import copy
import gzip
import json
from pathlib import Path
from unittest import mock

import pytest
import yaml

from cyber_post_train.jobs import JobsError, digest
from training import sft
from training import skyrl_lane2_authority as authority
from training import skyrl_lane2_direct as direct
from training import skyrl_lane2_training as training

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "configs/qualification/qwen38-skyrl-lane2-run-v1.json"
AUTHORITY = ROOT / "configs/qualification/qwen38-skyrl-lane2-authority-v1.json"


def _manifest() -> dict:
    value = json.loads(
        (ROOT / "configs/qualification/qwen38-rl-reward-canary-manifest-prod-v8.json").read_text()
    )
    value.update(
        {
            "name": "chris-q38-rlreward-lane2-v1",
            "selection_sha256": authority.TASK_SET_SELF_SHA256,
            "split_sha256": authority.SPLIT_SELF_SHA256,
        }
    )
    value["sha256"] = "sha256:" + digest(
        {key: item for key, item in value.items() if key != "sha256"}
    )
    return value


def _compile() -> tuple[dict, dict]:
    run, manifest = json.loads(RUN.read_text()), _manifest()
    original = sft.read_mapping

    def read(path: Path) -> dict:
        if Path(path) == Path(run["data"]["manifest"]):
            return copy.deepcopy(manifest)
        return original(path)

    with mock.patch.object(sft, "read_mapping", side_effect=read):
        plan = training.compile_rl(run, relative_to=RUN.parent)
    return plan, training.job_request(plan)


def _bundle(request: dict) -> dict:
    encoded = request["env"].get("CYBER_RUNTIME_BUNDLE")
    if encoded is None:
        encoded = "".join(
            value
            for key, value in sorted(
                (
                    (key, value)
                    for key, value in request["env"].items()
                    if key.startswith("CYBER_RUNTIME_BUNDLE_")
                ),
                key=lambda item: int(item[0].rsplit("_", 1)[1]),
            )
        )
    return json.loads(gzip.decompress(base64.b64decode(encoded, validate=True)))


def _source_preview(plan: dict, request: dict) -> dict:
    placeholder = request["name"] + "-00000000"
    environment = [
        {"name": key, "value": value}
        for key, value in sorted(
            {
                **request["env"],
                "RUN_DIR": request["run_dir"],
                "FLEET_RUN_ID": "00000000-0000-0000-0000-000000000000",
                "FLEET_RUN_NAME": placeholder,
            }.items()
        )
    ]
    resources = request["resources"]
    value = {
        "apiVersion": "ray.io/v1",
        "kind": "RayJob",
        "metadata": {
            "name": placeholder,
            "namespace": direct.NAMESPACE,
            "labels": {
                "app": "fleet-rl-job",
                "fleet.ai/requeue-if-preempted": "false",
                "fleet.ai/run-id": "00000000-0000-0000-0000-000000000000",
                "fleet.ai/run-name": request["name"],
                "kueue.x-k8s.io/queue-name": "training-lq",
                "kueue.x-k8s.io/priority-class": "q1",
            },
            "annotations": {
                "fleet.ai/run-id": "00000000-0000-0000-0000-000000000000",
                "fleet.ai/job-image": request["image"],
                "fleet.ai/run-dir": plan["output_root"],
            },
        },
        "spec": {
            "entrypoint": request["command"],
            "submissionMode": "HTTPMode",
            "suspend": True,
            "shutdownAfterJobFinishes": True,
            "rayClusterSpec": {
                "headGroupSpec": {
                    "template": {
                        "metadata": {},
                        "spec": {
                            "priorityClassName": "c1",
                            "containers": [
                                {
                                    "name": "ray-head",
                                    "image": request["image"],
                                    "env": environment,
                                    "envFrom": [
                                        {"secretRef": {"name": "fleet-api"}},
                                        {"secretRef": {"name": "wandb-api"}},
                                        {"secretRef": {"name": placeholder + "-fleet-key"}},
                                    ],
                                    "resources": {
                                        "requests": {
                                            "cpu": resources["cpu_request"],
                                            "memory": resources["memory_request"],
                                            "nvidia.com/gpu": request["gpus_per_worker"],
                                        },
                                        "limits": {
                                            "cpu": resources["cpu_limit"],
                                            "memory": resources["memory_limit"],
                                            "nvidia.com/gpu": request["gpus_per_worker"],
                                        },
                                    },
                                }
                            ],
                            "initContainers": [
                                {
                                    "name": "sfs-init",
                                    "command": [
                                        "sh",
                                        "-c",
                                        f"mkdir -p {plan['output_root']} && chown 1000:100 "
                                        f"{plan['output_root']}",
                                    ],
                                }
                            ],
                        },
                    }
                }
            },
        },
    }
    return {"name": placeholder, "warnings": [], "manifest_yaml": yaml.safe_dump(value)}


def test_lane2_is_disjoint_mixed_outcome_and_train_only() -> None:
    value = authority.validate_authority(AUTHORITY)
    optimizer = value["lane2_shard"]["optimizer_task"]
    diagnostic = value["lane2_shard"]["internal_diagnostic_task"]
    prior = value["historical_selection_prior"]

    assert optimizer["task_version_id"] == authority.OPTIMIZER_TASK_VERSION_ID
    assert optimizer["task_version_id"] != diagnostic["task_version_id"]
    assert diagnostic["task_version_id"] == authority.DIAGNOSTIC_TASK_VERSION_ID
    assert optimizer["authority_split"] == diagnostic["authority_split"] == "train"
    assert diagnostic["protected_holdout"] is False
    assert (
        prior["optimizer_passes"],
        prior["optimizer_sessions"],
        prior["optimizer_pass_rate"],
    ) == (
        4,
        8,
        0.5,
    )


def test_lane2_current_runtime_request_and_manifests_are_alert_safe() -> None:
    plan, request = _compile()
    bundle = _bundle(request)
    preview = _source_preview(plan, request)
    rayjob = direct.manifest(plan, request, preview)
    cpu_job = direct.preflight_job_manifest(plan)

    assert plan["schema"] == training.SCHEMA
    assert plan["qualification"]["optimizer_task_version_id"] == authority.OPTIMIZER_TASK_VERSION_ID
    assert request["workers"] == 1 and request["gpus_per_worker"] == 8
    assert request["priority_class"] == "c1" and request["failureAlerts"] is False
    assert request["image"] == authority.IMAGE
    assert bundle["module"] == training.MODULE
    assert (
        bundle["files"]["training/skyrl_lane2_training.py"] == Path(training.__file__).read_text()
    )
    assert rayjob["metadata"]["annotations"]["fleet.ai/failure-alerts"] == "off"
    assert rayjob["metadata"]["labels"]["kueue.x-k8s.io/priority-class"] == "q1"
    assert rayjob["spec"]["rayClusterSpec"].get("workerGroupSpecs") in (None, [])
    assert cpu_job["metadata"]["annotations"] == {"fleet.ai/failure-alerts": "off"}
    assert cpu_job["spec"]["template"]["spec"]["priorityClassName"] == "c1"
    assert "nvidia.com/gpu" not in json.dumps(cpu_job, sort_keys=True)
    assert direct.live_create_is_available() is False


def test_lane2_fails_closed_on_optimizer_or_alert_drift() -> None:
    plan, request = _compile()
    changed = copy.deepcopy(plan)
    changed["qualification"]["optimizer_task_version_id"] = authority.DIAGNOSTIC_TASK_VERSION_ID
    with pytest.raises(ValueError, match="lane2 bundled authority changed"):
        training.job_request(changed)

    preview = _source_preview(plan, request)
    source = yaml.safe_load(preview["manifest_yaml"])
    source["metadata"]["annotations"]["fleet.ai/failure-alerts"] = "off"
    preview["manifest_yaml"] = yaml.safe_dump(source)
    with pytest.raises(JobsError, match="identity or admission changed"):
        direct.manifest(plan, request, preview)
