import hashlib
import json
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from cyber_post_train.direct_submit import Kubectl
from cyber_post_train.jobs import JobsError
from cyber_post_train.sfs_write_identity import validate_direct_dev_gpu_reload_output


def dev_gpu_reload_pod() -> dict:
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": "chris-q38-s700-dev-gpu-v2",
            "namespace": "fleet-train-jobs",
            "labels": {"operation": "step700-dev-gpu-reload"},
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


def test_validator_accepts_the_proven_v2_output_shape() -> None:
    validate_direct_dev_gpu_reload_output(dev_gpu_reload_pod())


def test_create_rejects_known_v1_top_level_output_before_kubectl(monkeypatch) -> None:
    calls = []
    pod = dev_gpu_reload_pod()
    pod["metadata"]["name"] = "chris-q38-s700-dev-gpu-v1"
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
    else:
        spec["volumes"][1]["persistentVolumeClaim"]["claimName"] = "other"

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

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(JobsError, match="read-only SFS"):
        getattr(Kubectl("dev-context"), method)(local)
    assert len(calls) == 1
    assert ("--dry-run=server" in calls[0][0]) is server_preview


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
