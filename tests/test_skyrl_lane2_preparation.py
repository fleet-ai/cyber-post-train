"""Offline lane2 authority, compiler and alert-safe renderer tests."""

from __future__ import annotations

import base64
import copy
import gzip
import json
from datetime import UTC, datetime
from pathlib import Path
from unittest import mock

import pytest
import yaml

from cyber_post_train.jobs import JobsError, digest
from training import sft
from training import skyrl_lane2_authority as authority
from training import skyrl_lane2_data as data
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
                "fleet.ai/failure-alerts": "off",
            },
        },
        "spec": {
            "backoffLimit": 0,
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
                                    "securityContext": direct.shared._runtime_context(),
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


def _image_identity_receipt(request: dict) -> dict:
    body = {
        "schema": direct.shared.IMAGE_DEFAULT_IDENTITY_SCHEMA,
        "status": "passed",
        "image": request["image"],
        "uid": 1000,
        "gid": 100,
        "gpus": 0,
    }
    return {**body, "receipt_sha256": digest(body)}


def _direct_render(value: dict) -> dict:
    rendered = copy.deepcopy(value)
    rendered["metadata"].update(
        {
            "creationTimestamp": "2026-09-22T00:00:00Z",
            "generation": 1,
            "uid": "00000000-0000-0000-0000-000000000001",
        }
    )
    rendered["spec"]["ttlSecondsAfterFinished"] = 0
    rendered["spec"]["rayClusterSpec"]["headGroupSpec"].update(
        {"numOfHosts": 1, "scaleStrategy": {}}
    )
    return rendered


def _cpu_render(value: dict) -> dict:
    rendered = copy.deepcopy(value)
    uid = "00000000-0000-0000-0000-000000000002"
    name = value["metadata"]["name"]
    generated = {
        "batch.kubernetes.io/controller-uid": uid,
        "batch.kubernetes.io/job-name": name,
        "controller-uid": uid,
        "job-name": name,
    }
    rendered["metadata"].update(
        {
            "creationTimestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "generation": 1,
            "uid": uid,
            "labels": {**value["metadata"]["labels"], **generated},
        }
    )
    rendered["status"] = {}
    rendered["spec"].update(
        {
            "completionMode": "NonIndexed",
            "completions": 1,
            "manualSelector": False,
            "parallelism": 1,
            "podReplacementPolicy": "TerminatingOrFailed",
            "selector": {"matchLabels": {"batch.kubernetes.io/controller-uid": uid}},
            "suspend": True,
        }
    )
    rendered["spec"]["template"]["metadata"]["labels"] = {
        **value["spec"]["template"]["metadata"]["labels"],
        **generated,
    }
    rendered["spec"]["template"]["spec"].update(
        {
            "dnsPolicy": "ClusterFirst",
            "schedulerName": "default-scheduler",
            "terminationGracePeriodSeconds": 30,
        }
    )
    rendered["spec"]["template"]["spec"]["containers"][0]["imagePullPolicy"] = "IfNotPresent"
    return rendered


def test_lane2_is_disjoint_mixed_outcome_and_train_only() -> None:
    value = authority.validate_authority(AUTHORITY)
    optimizer = value["lane2_shard"]["optimizer_task"]
    diagnostic = value["lane2_shard"]["internal_diagnostic_task"]
    prior = value["historical_selection_prior"]

    assert optimizer["task_version_id"] == authority.OPTIMIZER_TASK_VERSION_ID
    assert optimizer["task_version_id"] != diagnostic["task_version_id"]
    assert diagnostic["task_version_id"] == authority.DIAGNOSTIC_TASK_VERSION_ID
    assert "54425601-6fd2-43d8-8cb9-e565b767676a" not in {
        optimizer["task_version_id"],
        diagnostic["task_version_id"],
    }
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
    assert (
        prior["diagnostic_passes"],
        prior["diagnostic_sessions"],
        prior["diagnostic_pass_rate"],
    ) == (4, 8, 0.5)


def test_lane2_current_runtime_request_and_manifests_are_alert_safe() -> None:
    plan, request = _compile()
    bundle = _bundle(request)
    preview = _source_preview(plan, request)
    rayjob = direct.manifest(plan, request, preview)
    packet = direct.packet(plan, request, preview)
    server_proof = direct.validate_server_preview(
        plan,
        request,
        preview,
        rayjob,
        _direct_render(rayjob),
        context=direct.PROD_CONTEXT,
    )
    cpu_job = direct.preflight_job_manifest(plan)
    data_job = direct.data_job_manifest()

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
    assert rayjob == yaml.safe_load(preview["manifest_yaml"])
    assert packet["name"] == preview["name"]
    assert packet["run_name_prefix"] == plan["run_name"]
    assert server_proof["nodes"] == 1 and server_proof["gpus"] == 8
    assert rayjob["metadata"]["labels"]["kueue.x-k8s.io/priority-class"] == "q1"
    assert rayjob["spec"]["rayClusterSpec"].get("workerGroupSpecs") in (None, [])
    assert cpu_job["metadata"]["annotations"] == {"fleet.ai/failure-alerts": "off"}
    assert cpu_job["spec"]["template"]["spec"]["priorityClassName"] == "c1"
    assert cpu_job["spec"]["template"]["spec"]["containers"][0]["envFrom"] == [
        {"secretRef": {"name": "wandb-api"}}
    ]
    assert "nvidia.com/gpu" not in json.dumps(cpu_job, sort_keys=True)
    data_container = data_job["spec"]["template"]["spec"]["containers"][0]
    data_environment = {item["name"]: item["value"] for item in data_container["env"]}
    assert data_job["metadata"]["annotations"] == {"fleet.ai/failure-alerts": "off"}
    assert data_job["metadata"]["labels"] == {
        "kueue.x-k8s.io/queue-name": "training-lq",
        "kueue.x-k8s.io/priority-class": "q1",
    }
    assert data_job["spec"]["suspend"] is True
    assert data_container["envFrom"] == [{"secretRef": {"name": "fleet-api"}}]
    assert "nvidia.com/gpu" not in json.dumps(data_job, sort_keys=True)
    data_bundle = _bundle({"env": data_environment})
    assert data_bundle["module"] == "training.skyrl_lane2_data"
    assert {
        "configs/data/qwen38-skyrl-production-task-set-v1.json",
        "configs/data/qwen38-skyrl-production-split-v1.json",
        "configs/data/qwen-blackbox-eligible-v1.json",
        "evals/fleet/configs/opencode-easiest-train100-selection-v2.json",
    } <= set(data_bundle["files"])
    data_proof = direct.validate_data_server_preview(
        data_job,
        _cpu_render(data_job),
        context=direct.PROD_CONTEXT,
    )
    assert data_proof["gpus"] == 0 and data_proof["queue_priority"] == "q1"
    live_style = _cpu_render(data_job)
    live_style["metadata"]["labels"] = copy.deepcopy(data_job["metadata"]["labels"])
    live_proof = direct.validate_data_server_preview(
        data_job,
        live_style,
        context=direct.PROD_CONTEXT,
    )
    assert live_proof["gpus"] == 0 and live_proof["queue_priority"] == "q1"
    assert direct.live_create_is_available() is False


def test_lane2_fails_closed_on_optimizer_or_alert_drift() -> None:
    plan, request = _compile()
    changed = copy.deepcopy(plan)
    changed["qualification"]["optimizer_task_version_id"] = authority.DIAGNOSTIC_TASK_VERSION_ID
    with pytest.raises(ValueError, match="lane2 bundled authority changed"):
        training.job_request(changed)

    preview = _source_preview(plan, request)
    source = yaml.safe_load(preview["manifest_yaml"])
    source["metadata"]["annotations"].pop("fleet.ai/failure-alerts")
    preview["manifest_yaml"] = yaml.safe_dump(source)
    with pytest.raises(JobsError, match="root alert-off"):
        direct.manifest(plan, request, preview)

    preview = _source_preview(plan, request)
    source = yaml.safe_load(preview["manifest_yaml"])
    source["spec"]["backoffLimit"] = 1
    preview["manifest_yaml"] = yaml.safe_dump(source)
    with pytest.raises(JobsError, match="execution changed"):
        direct.manifest(plan, request, preview)

    preview = _source_preview(plan, request)
    source = yaml.safe_load(preview["manifest_yaml"])
    source["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"]["containers"][0][
        "securityContext"
    ].pop("allowPrivilegeEscalation")
    preview["manifest_yaml"] = yaml.safe_dump(source)
    with pytest.raises(JobsError, match="runtime security context"):
        direct.manifest(plan, request, preview)

    preview = _source_preview(plan, request)
    source = yaml.safe_load(preview["manifest_yaml"])
    source["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"]["containers"][
        0
    ].pop("securityContext")
    preview["manifest_yaml"] = yaml.safe_dump(source)
    with pytest.raises(JobsError, match="exact-image runtime-user receipt"):
        direct.manifest(plan, request, preview)
    assert direct.manifest(
        plan,
        request,
        preview,
        image_identity_receipt=_image_identity_receipt(request),
    ) == source

    data_job = direct.data_job_manifest()
    rendered = _cpu_render(data_job)
    rendered["spec"]["suspend"] = False
    with pytest.raises(JobsError, match="Kueue suspension"):
        direct.validate_data_server_preview(data_job, rendered, context=direct.PROD_CONTEXT)


def test_lane2_data_main_reports_the_existing_builder_contract(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    built = {
        "manifest_sha256": "sha256:" + "a" * 64,
        "files": {
            "train": {"rows": 1, "sha256": "sha256:" + "b" * 64},
            "dev": {"rows": 1, "sha256": "sha256:" + "c" * 64},
        },
        "submitted": False,
    }

    class Client:
        def __enter__(self) -> object:
            return object()

        def __exit__(self, *_: object) -> None:
            return None

    monkeypatch.setenv("FLEET_API_KEY", "test-only")
    monkeypatch.setattr(data.httpx, "Client", lambda **_: Client())
    monkeypatch.setattr(data.rl_data, "build", lambda *_args, **_kwargs: built)
    monkeypatch.setattr("sys.argv", ["skyrl_lane2_data"])

    data.main()

    assert json.loads(capsys.readouterr().out) == {
        "schema": "cyber_skyrl_lane2_data_receipt_v1",
        "status": "built",
        "submitted": False,
        "gpus": 0,
        "files": {
            "train": {"rows": 1, "sha256": "sha256:" + "b" * 64},
            "dev": {"rows": 1, "sha256": "sha256:" + "c" * 64},
        },
        "manifest_sha256": "sha256:" + "a" * 64,
    }


def test_lane2_data_main_writes_only_the_sanitized_termination_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    built = {
        "manifest_sha256": "sha256:" + "a" * 64,
        "files": {
            "train": {"rows": 1, "sha256": "sha256:" + "b" * 64},
            "dev": {"rows": 1, "sha256": "sha256:" + "c" * 64},
        },
        "submitted": False,
        "private_rows": [{"prompt": "must-not-enter-receipt"}],
    }

    class Client:
        def __enter__(self) -> object:
            return object()

        def __exit__(self, *_: object) -> None:
            return None

    receipt = tmp_path / "termination-log"
    monkeypatch.setenv("FLEET_API_KEY", "test-only")
    monkeypatch.setattr(data, "RECEIPT", receipt)
    monkeypatch.setattr(data.httpx, "Client", lambda **_: Client())
    monkeypatch.setattr(data.rl_data, "build", lambda *_args, **_kwargs: built)
    monkeypatch.setattr("sys.argv", ["skyrl_lane2_data", "--receipt", str(receipt)])

    data.main()

    value = json.loads(receipt.read_bytes())
    body = {key: item for key, item in value.items() if key != "receipt_sha256"}
    assert value["receipt_sha256"] == digest(body)
    assert "must-not-enter-receipt" not in receipt.read_text()
    assert json.loads(capsys.readouterr().out) == body
