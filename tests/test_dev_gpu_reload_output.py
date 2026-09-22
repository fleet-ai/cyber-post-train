import hashlib
import json
import subprocess
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import pytest

from cyber_post_train import direct_submit
from cyber_post_train.direct_submit import Kubectl, validate_direct_dev_gpu_reload_precreate
from cyber_post_train.gpu_capacity import build_capacity_census
from cyber_post_train.jobs import JobsError
from cyber_post_train.sfs_write_identity import (
    render_direct_dev_gpu_reload_pod,
    validate_direct_dev_gpu_reload_output,
)


def dev_gpu_reload_pod() -> dict:
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": "chris-q38-s700-dev-gpu-v2",
            "namespace": "fleet-train-jobs",
            "labels": {
                "operation": "step700-dev-gpu-reload",
                "fleet.ai/run-name": "chris-q38-s700-dev-gpu-v2",
            },
            "annotations": {"fleet.ai/failure-alerts": "off"},
        },
        "spec": {
            "securityContext": {
                "runAsNonRoot": True,
                "runAsUser": 1000,
                "runAsGroup": 100,
                "fsGroup": 100,
            },
            "containers": [
                {
                    "name": "reload",
                    "env": [
                        {
                            "name": "RUN_DIR",
                            "value": "/controls/.preflight-control-s700-gpu-dev-v2",
                        }
                    ],
                    "resources": {
                        "requests": {"nvidia.com/gpu": "1"},
                        "limits": {"nvidia.com/gpu": "1"},
                    },
                    "securityContext": {
                        "allowPrivilegeEscalation": False,
                        "capabilities": {"drop": ["ALL"]},
                    },
                    "volumeMounts": [
                        {"name": "sfs-readonly", "mountPath": "/mnt/sfs", "readOnly": True},
                        {
                            "name": "sfs-control",
                            "mountPath": "/controls",
                            "subPath": "jobs/chris-q38-study-corpora-v1/launch-controls",
                        },
                    ],
                }
            ],
            "volumes": [
                {
                    "name": "sfs-readonly",
                    "persistentVolumeClaim": {"claimName": "sfs-shared"},
                },
                {
                    "name": "sfs-control",
                    "persistentVolumeClaim": {"claimName": "sfs-shared"},
                },
            ],
        },
    }


def qualified_capacity(context: str, **kwargs) -> dict:
    assert context == "dev-context"
    return build_capacity_census(
        {"items": []},
        {"items": []},
        {"items": []},
        {"items": []},
        **kwargs,
    )


def test_validator_accepts_the_proven_v2_output_shape() -> None:
    validate_direct_dev_gpu_reload_output(dev_gpu_reload_pod())


def test_renderer_binds_name_and_capacity_owner_label_from_one_authority() -> None:
    template = dev_gpu_reload_pod()
    template["metadata"]["name"] = "stale-copy-edited-name"
    template["metadata"]["labels"].pop("fleet.ai/run-name")

    rendered = render_direct_dev_gpu_reload_pod(
        template,
        run_name="chris-q38-t3k64-s195-dev-gpu-v1",
    )

    assert template["metadata"]["name"] == "stale-copy-edited-name"
    assert "fleet.ai/run-name" not in template["metadata"]["labels"]
    assert rendered["metadata"]["name"] == "chris-q38-t3k64-s195-dev-gpu-v1"
    assert rendered["metadata"]["labels"]["fleet.ai/run-name"] == "chris-q38-t3k64-s195-dev-gpu-v1"


def test_create_rejects_known_v1_top_level_output_before_kubectl(monkeypatch) -> None:
    calls = []
    pod = dev_gpu_reload_pod()
    pod["metadata"]["name"] = "chris-q38-s700-dev-gpu-v1"
    pod["metadata"]["labels"]["fleet.ai/run-name"] = "chris-q38-s700-dev-gpu-v1"
    pod["spec"]["containers"][0]["env"][0]["value"] = (
        "/mnt/sfs/jobs/chris-q38-t3k32-s700-gpu-dev-v1"
    )
    pod["spec"]["containers"][0]["volumeMounts"] = [{"name": "sfs", "mountPath": "/mnt/sfs"}]
    pod["spec"]["volumes"] = [{"name": "sfs", "persistentVolumeClaim": {"claimName": "sfs-shared"}}]

    def run(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("known v1 output shape reached kubectl")

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(JobsError, match=r"RUN_DIR must be one writable /controls leaf"):
        Kubectl("dev-context").create_once(pod)
    assert calls == []


def test_create_rejects_missing_capacity_owner_before_census_or_kubectl(monkeypatch) -> None:
    calls = []
    pod = dev_gpu_reload_pod()
    pod["metadata"]["labels"].pop("fleet.ai/run-name")

    def unexpected(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("invalid ownership reached a live boundary")

    monkeypatch.setattr(direct_submit, "live_capacity_census", unexpected)
    monkeypatch.setattr(subprocess, "run", unexpected)
    with pytest.raises(JobsError, match=r"canonical fleet\.ai/run-name ownership label"):
        Kubectl("dev-context").create_once(pod)
    assert calls == []


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("writable-source", "read-only SFS"),
        ("wrong-control-subpath", "exact writable control subpath"),
        ("wrong-uid", "proven UID 1000"),
        ("initializer", "no initializer"),
        ("nested-run-dir", "one writable /controls leaf"),
        ("visible-run-dir", "direct hidden child"),
        ("extra-mount", "read-only SFS"),
        ("different-claim", "shared SFS claim"),
        ("mismatched-owner", "canonical fleet.ai/run-name"),
        ("wrong-gpu-count", "exactly one GPU"),
    ],
)
def test_validator_rejects_v2_output_shape_drift(fault, message) -> None:
    pod = dev_gpu_reload_pod()
    spec = pod["spec"]
    container = spec["containers"][0]
    if fault == "writable-source":
        container["volumeMounts"][0]["readOnly"] = False
    elif fault == "wrong-control-subpath":
        container["volumeMounts"][1]["subPath"] = "jobs/new-run"
    elif fault == "wrong-uid":
        spec["securityContext"]["runAsUser"] = 0
    elif fault == "initializer":
        spec["initContainers"] = [{"name": "prepare"}]
    elif fault == "nested-run-dir":
        container["env"][0]["value"] = "/controls/nested/.preflight-control-s700-gpu-dev-v3"
    elif fault == "visible-run-dir":
        container["env"][0]["value"] = "/controls/s700-gpu-dev-v3"
    elif fault == "extra-mount":
        container["volumeMounts"].append({"name": "tmp", "mountPath": "/tmp"})
    elif fault == "different-claim":
        spec["volumes"][1]["persistentVolumeClaim"]["claimName"] = "other"
    elif fault == "mismatched-owner":
        pod["metadata"]["labels"]["fleet.ai/run-name"] = "chris-q38-other-dev-gpu-v1"
    else:
        container["resources"]["requests"]["nvidia.com/gpu"] = "2"

    with pytest.raises(ValueError, match=message):
        validate_direct_dev_gpu_reload_output(pod)


@pytest.mark.parametrize(
    ("method", "server_preview"),
    [("dry_run", True), ("create_once", False)],
)
def test_generic_kubectl_boundary_revalidates_server_render(
    monkeypatch, method, server_preview
) -> None:
    calls = []
    local = dev_gpu_reload_pod()
    rendered = deepcopy(local)
    rendered["spec"]["containers"][0]["volumeMounts"][0]["readOnly"] = False

    def run(command, **kwargs):
        calls.append((command, json.loads(kwargs["input"])))
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(rendered), stderr="")

    monkeypatch.setattr(direct_submit, "live_capacity_census", qualified_capacity)
    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(JobsError, match="read-only SFS"):
        getattr(Kubectl("dev-context"), method)(local)
    assert len(calls) == 1
    assert ("--dry-run=server" in calls[0][0]) is server_preview


def test_create_reads_repo_wide_capacity_immediately_before_the_create(monkeypatch) -> None:
    calls = []
    pod = dev_gpu_reload_pod()

    def run(command, **kwargs):
        calls.append(command)
        if command[4] == "get":
            return subprocess.CompletedProcess(command, 0, stdout='{"items":[]}', stderr="")
        assert command[4] == "create"
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(pod), stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    created = Kubectl("dev-context").create_once(pod)

    assert created == pod
    assert [command[5] for command in calls[:4]] == [
        "pods",
        "inferencemodels.inference.fleet.ai",
        "rayjobs.ray.io",
        "workloads.kueue.x-k8s.io",
    ]
    assert all("--all-namespaces" in command for command in calls[:4])
    assert calls[4][4] == "create"
    assert not any(command[4:6] == ["get", "nodes"] for command in calls)


def test_precreate_rejects_a_local_only_capacity_scope() -> None:
    def local_only_capacity(context: str, **kwargs) -> dict:
        value = qualified_capacity(context, **kwargs)
        value["scope"]["kubernetes_namespaces"] = "fleet-train-jobs"
        value["sha256"] = direct_submit.digest(
            {key: item for key, item in value.items() if key != "sha256"}
        )
        return value

    with pytest.raises(JobsError, match="capacity census is stale or incomplete"):
        validate_direct_dev_gpu_reload_precreate(
            dev_gpu_reload_pod(),
            "dev-context",
            reader=local_only_capacity,
            now=datetime.now(UTC),
        )


def test_incident_evidence_is_self_consistent() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "docs/evidence/qwen38-step700-dev-gpu-reload-output-20260922.json"
    )
    value = json.loads(path.read_text())
    expected = value.pop("receipt_sha256")
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(encoded).hexdigest() == expected
    assert value["failed_v1"]["failure_boundary"] == "before_weight_load"
    assert value["accepted_v2"]["run_dir"].startswith("/controls/.preflight-control-")
    assert value["accepted_v2"]["source_mount_read_only"] is True
    assert value["accepted_v2"]["optimizer_steps_executed"] == 0
