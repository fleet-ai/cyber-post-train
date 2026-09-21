import json
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from cyber_post_train.direct_submit import (
    CPU_CHECKPOINT_OPERATION_ANNOTATION,
    CPU_NODE_SELECTOR,
    Kubectl,
    direct_submit_lr30_qualification_once,
    direct_submit_sft_once,
    render_lr30_qualification_rayjob,
    render_sft_rayjob,
)
from cyber_post_train.jobs import JobsError, digest
from training import qwen38_lr30_step76_gate as lr30
from training import sft

RUN_ID = "12345678-1234-4234-9234-123456789abc"
CREATED_UID = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"


@pytest.fixture(autouse=True)
def current_sft_renderer(monkeypatch):
    monkeypatch.setattr(sft, "job_request", lambda _: request())


def plan():
    return {"schema": "cyber_sft_runtime_dense_v1", "immutable": "synthetic"}


def request():
    return {
        "name": "researcher-sft",
        "title": "Synthetic SFT fixture",
        "image": "registry/image@sha256:" + "a" * 64,
        "command": "python train.py",
        "workers": 2,
        "gpus_per_worker": 8,
        "run_dir": "/mnt/sfs/jobs/researcher-sft-v1",
        "priority_class": "c1",
        "requeueIfPreempted": False,
        "failureAlerts": False,
        "resources": {
            "cpu_request": "8",
            "cpu_limit": "16",
            "memory_request": "64Gi",
            "memory_limit": "128Gi",
        },
        "env": {"WANDB_MODE": "online", "TOKENIZERS_PARALLELISM": "false"},
        "secrets": ["wandb-api"],
        "image_pull_secrets": ["registry-pull"],
    }


def template(value, container_name):
    placeholder = value["name"] + "-00000000"
    generated = {
        "FLEET_EXTERNAL_RAY": "1",
        "FLEET_GPUS_PER_WORKER": str(value["gpus_per_worker"]),
        "FLEET_RUN_ID": "00000000-0000-0000-0000-000000000000",
        "FLEET_RUN_NAME": placeholder,
        "FLEET_TRACE_ROOT": "/mnt/fleet/trajectory-spool",
        "RAY_memory_usage_threshold": "0.98",
        "RUN_DIR": value["run_dir"],
        "SKYPILOT_NUM_GPUS_PER_NODE": str(value["gpus_per_worker"]),
        "WORKERS": str(value["workers"]),
    }
    return {
        "metadata": {
            "labels": {
                "fleet.ai/run-id": "00000000-0000-0000-0000-000000000000",
                "fleet.ai/run-name": value["name"],
            },
            "annotations": {
                "kueue.x-k8s.io/podset-preferred-topology": "topology.nebius.com/tier-1"
            },
        },
        "spec": {
            "priorityClassName": "c1",
            "imagePullSecrets": [{"name": name} for name in value.get("image_pull_secrets", [])],
            "nodeSelector": {"workload": "fleetai-training-ng-gpu"},
            "containers": [
                {
                    "name": container_name,
                    "image": value["image"],
                    "env": [
                        {"name": name, "value": item}
                        for name, item in {**generated, **value["env"]}.items()
                    ],
                    "envFrom": [
                        *[
                            {"secretRef": {"name": name}}
                            for name in value.get("secrets", [])
                        ],
                        {"secretRef": {"name": placeholder + "-fleet-key"}},
                    ],
                    "resources": {
                        "requests": {
                            "cpu": value["resources"]["cpu_request"],
                            "memory": value["resources"]["memory_request"],
                            "nvidia.com/gpu": value["gpus_per_worker"],
                        },
                        "limits": {
                            "cpu": value["resources"]["cpu_limit"],
                            "memory": value["resources"]["memory_limit"],
                            "nvidia.com/gpu": value["gpus_per_worker"],
                        },
                    },
                    "securityContext": {"privileged": False},
                }
            ],
        },
    }


def manifest(value=None):
    value = value or request()
    placeholder = value["name"] + "-00000000"
    return {
        "apiVersion": "ray.io/v1",
        "kind": "RayJob",
        "metadata": {
            "name": placeholder,
            "namespace": "fleet-train-jobs",
            "labels": {
                "app": "fleet-rl-job",
                "fleet.ai/requeue-if-preempted": "false",
                "kueue.x-k8s.io/queue-name": "training-lq",
                "kueue.x-k8s.io/priority-class": "q1",
                "fleet.ai/run-id": "00000000-0000-0000-0000-000000000000",
                "fleet.ai/run-name": value["name"],
            },
            "annotations": {
                "fleet.ai/run-id": "00000000-0000-0000-0000-000000000000",
                "fleet.ai/submitted-by": "researcher@fleet.so",
                "fleet.ai/submitted-by-profile": "9909b292-d23e-4e2e-8587-e4cd0d5c47bf",
                "fleet.ai/job-image": value["image"],
                "fleet.ai/run-dir": value["run_dir"],
            },
        },
        "spec": {
            "entrypoint": value["command"],
            "submissionMode": "HTTPMode",
            "suspend": True,
            "shutdownAfterJobFinishes": True,
            "rayClusterSpec": {
                "enableInTreeAutoscaling": False,
                "headGroupSpec": {"template": template(value, "ray-head")},
                "workerGroupSpecs": (
                    [
                        {
                            "groupName": "gpu",
                            "replicas": value["workers"] - 1,
                            "minReplicas": value["workers"] - 1,
                            "maxReplicas": value["workers"] - 1,
                            "template": template(value, "ray-worker"),
                        }
                    ]
                    if value["workers"] > 1
                    else []
                ),
            },
        },
    }


def cpu_checkpoint_pod():
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": "researcher-checkpoint-seal-v1",
            "namespace": "fleet-train-jobs",
            "annotations": {
                "fleet.ai/failure-alerts": "off",
                CPU_CHECKPOINT_OPERATION_ANNOTATION: "seal",
            },
        },
        "spec": {
            "priorityClassName": "c1",
            "restartPolicy": "Never",
            "nodeSelector": deepcopy(CPU_NODE_SELECTOR),
            "containers": [
                {
                    "name": "seal",
                    "image": "registry/image@sha256:" + "a" * 64,
                    "resources": {
                        "requests": {"cpu": "4", "memory": "16Gi"},
                        "limits": {"cpu": "4", "memory": "16Gi"},
                    },
                }
            ],
        },
    }


def preview(obj=None):
    return {"manifest_yaml": yaml.safe_dump(obj or manifest()), "warnings": []}


def lr30_plan(*, launchable=False):
    path = (
        Path(__file__).resolve().parents[1]
        / "configs/qualification/qwen38-lr30-step76-gpu-reload-v1.json"
    )
    value = json.loads(path.read_text())
    if launchable:
        value["launchable"] = True
        value["status"] = "approved_for_exact_create"
        value["blockers"] = []
        value["sha256"] = "sha256:" + digest(
            {key: item for key, item in value.items() if key != "sha256"}
        )
    return value


def test_render_changes_only_reviewed_identity_secret_and_root_annotation():
    source = manifest()
    rendered, proof = render_sft_rayjob(plan(), request(), preview(source), run_id=RUN_ID)
    assert proof["name"] == "researcher-sft-12345678"
    assert proof["run_id"] == RUN_ID
    assert proof["removed_api_fleet_secrets"] == 2
    assert rendered["metadata"]["annotations"]["fleet.ai/failure-alerts"] == "off"
    assert rendered["metadata"]["labels"]["fleet.ai/run-id"] == RUN_ID
    assert rendered["metadata"]["annotations"]["fleet.ai/run-id"] == RUN_ID
    assert source["metadata"]["annotations"].get("fleet.ai/failure-alerts") is None

    for group in [
        rendered["spec"]["rayClusterSpec"]["headGroupSpec"],
        *rendered["spec"]["rayClusterSpec"]["workerGroupSpecs"],
    ]:
        pod = group["template"]
        container = pod["spec"]["containers"][0]
        env = {item["name"]: item["value"] for item in container["env"]}
        assert pod["metadata"]["labels"]["fleet.ai/run-id"] == RUN_ID
        assert env["FLEET_RUN_ID"] == RUN_ID
        assert env["FLEET_RUN_NAME"] == proof["name"]
        assert container["envFrom"] == [{"secretRef": {"name": "wandb-api"}}]
    assert proof["preview_manifest_sha256"] == digest(source)
    assert proof["manifest_sha256"] == digest(rendered)


def test_lr30_render_is_exact_one_gpu_root_annotated_and_secret_free():
    request_value = lr30.job_request()
    rendered, proof = render_lr30_qualification_rayjob(
        lr30_plan(), request_value, preview(manifest(request_value)), run_id=RUN_ID
    )
    assert proof["name"] == "chris-q38-lr30-s76-gpu-v1-12345678"
    assert rendered["metadata"]["annotations"]["fleet.ai/failure-alerts"] == "off"
    groups = [
        rendered["spec"]["rayClusterSpec"]["headGroupSpec"],
        *rendered["spec"]["rayClusterSpec"]["workerGroupSpecs"],
    ]
    assert all(group["template"]["spec"]["containers"][0]["envFrom"] == [] for group in groups)
    assert all(
        group["template"]["spec"]["containers"][0]["resources"]["requests"][
            "nvidia.com/gpu"
        ]
        == 1
        for group in groups
    )


@pytest.mark.parametrize(
    "fault",
    [
        "already-annotated",
        "wrong-priority",
        "not-suspended",
        "wrong-image",
        "wrong-resource",
        "extra-env",
        "missing-fleet-secret",
        "extra-secret",
        "unknown-zero-id",
        "unknown-placeholder-name",
        "duplicate-container",
        "warning",
    ],
)
def test_render_fails_closed_on_preview_drift(fault):
    obj = manifest()
    response = preview(obj)
    head = obj["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]
    container = head["spec"]["containers"][0]
    if fault == "already-annotated":
        obj["metadata"]["annotations"]["fleet.ai/failure-alerts"] = "off"
    elif fault == "wrong-priority":
        head["spec"]["priorityClassName"] = "c0"
    elif fault == "not-suspended":
        obj["spec"]["suspend"] = False
    elif fault == "wrong-image":
        container["image"] = "registry/other@sha256:" + "b" * 64
    elif fault == "wrong-resource":
        container["resources"]["requests"]["memory"] = "1Gi"
    elif fault == "extra-env":
        container["env"].append({"name": "UNREVIEWED", "value": "1"})
    elif fault == "missing-fleet-secret":
        container["envFrom"].pop()
    elif fault == "extra-secret":
        container["envFrom"].append({"secretRef": {"name": "other"}})
    elif fault == "unknown-zero-id":
        obj["metadata"]["annotations"]["other"] = "00000000-0000-0000-0000-000000000000"
    elif fault == "unknown-placeholder-name":
        obj["metadata"]["annotations"]["other"] = "researcher-sft-00000000"
    elif fault == "duplicate-container":
        head["spec"]["containers"].append(deepcopy(container))
    else:
        response["warnings"] = ["server changed"]
    response["manifest_yaml"] = yaml.safe_dump(obj)
    with pytest.raises(JobsError):
        render_sft_rayjob(plan(), request(), response, run_id=RUN_ID)


@pytest.mark.parametrize(
    "plan_value,request_change",
    [
        ({"schema": "cyber_miles_training_v2"}, {}),
        (plan(), {"secrets": ["wandb-api", "fleet-key"]}),
        (plan(), {"secrets": []}),
        (plan(), {"priority_class": "c2"}),
    ],
)
def test_direct_fallback_is_sft_only_and_proves_no_fleet_secret(plan_value, request_change):
    value = {**request(), **request_change}
    with pytest.raises(JobsError):
        render_sft_rayjob(plan_value, value, preview(manifest(value)), run_id=RUN_ID)


class FakeJobs:
    def __init__(self, *, rows=None, preview_value=None):
        self.rows = rows or []
        self.preview_value = preview_value or preview()
        self.calls = []

    def all_runs(self):
        self.calls.append("history")
        return deepcopy(self.rows)

    def raw_preview(self, value):
        self.calls.append(("preview", value))
        return deepcopy(self.preview_value)


class FakeKubectl:
    context = "production-context"

    def __init__(self, *, inventories=None, fail_create=False):
        self.inventories = inventories or {
            "rayjobs.ray.io": {"kind": "List", "items": []},
            "jobs.batch": {"kind": "List", "items": []},
        }
        self.fail_create = fail_create
        self.calls = []

    def list(self, resource):
        self.calls.append(("list", resource))
        return deepcopy(self.inventories[resource])

    def dry_run(self, obj):
        self.calls.append(("dry-run", digest(obj)))
        return deepcopy(obj)

    def create_once(self, obj):
        self.calls.append(("create", digest(obj)))
        if self.fail_create:
            raise JobsError("synthetic create uncertainty")
        result = deepcopy(obj)
        result["metadata"]["uid"] = CREATED_UID
        return result


def test_direct_submit_checks_twice_journals_then_creates_exactly_once(tmp_path):
    jobs, kube = FakeJobs(), FakeKubectl()
    journal = tmp_path / "DIRECT_SUBMISSION.jsonl"
    result = direct_submit_sft_once(
        plan=plan(),
        request=request(),
        jobs=jobs,
        kubectl=kube,
        journal=journal,
        run_id=RUN_ID,
    )
    assert result["uid"] == CREATED_UID and result["name"] == "researcher-sft-12345678"
    assert jobs.calls == ["history", ("preview", request()), "history"]
    assert [call[0] for call in kube.calls].count("list") == 4
    assert [call[0] for call in kube.calls].count("dry-run") == 1
    assert [call[0] for call in kube.calls].count("create") == 1
    records = [json.loads(line) for line in journal.read_text().splitlines()]
    assert [record["state"] for record in records] == [
        "KUBECTL_CREATE_INTENT_DO_NOT_RETRY",
        "KUBECTL_CREATE_RESPONSE",
    ]
    assert journal.stat().st_mode & 0o777 == 0o600


def test_lr30_direct_submit_requires_explicit_launchable_plan_and_creates_once(tmp_path):
    request_value = lr30.job_request()
    jobs = FakeJobs(preview_value=preview(manifest(request_value)))
    kube = FakeKubectl()
    journal = tmp_path / "DIRECT_SUBMISSION.jsonl"
    with pytest.raises(JobsError, match="not explicitly approved"):
        direct_submit_lr30_qualification_once(
            plan=lr30_plan(),
            request=request_value,
            jobs=jobs,
            kubectl=kube,
            journal=journal,
            run_id=RUN_ID,
        )
    assert not journal.exists() and jobs.calls == [] and kube.calls == []

    result = direct_submit_lr30_qualification_once(
        plan=lr30_plan(launchable=True),
        request=request_value,
        jobs=jobs,
        kubectl=kube,
        journal=journal,
        run_id=RUN_ID,
    )
    assert result["uid"] == CREATED_UID
    assert [call[0] for call in kube.calls].count("create") == 1
    assert json.loads(journal.read_text().splitlines()[0])["state"] == (
        "KUBECTL_CREATE_INTENT_DO_NOT_RETRY"
    )


def test_ambiguous_create_leaves_intent_and_never_retries(tmp_path):
    jobs, kube = FakeJobs(), FakeKubectl(fail_create=True)
    journal = tmp_path / "DIRECT_SUBMISSION.jsonl"
    with pytest.raises(JobsError):
        direct_submit_sft_once(
            plan=plan(),
            request=request(),
            jobs=jobs,
            kubectl=kube,
            journal=journal,
            run_id=RUN_ID,
        )
    assert json.loads(journal.read_text())["state"] == "KUBECTL_CREATE_INTENT_DO_NOT_RETRY"
    with pytest.raises(JobsError, match="journal already exists"):
        direct_submit_sft_once(
            plan=plan(),
            request=request(),
            jobs=jobs,
            kubectl=kube,
            journal=journal,
            run_id=RUN_ID,
        )
    assert [call[0] for call in kube.calls].count("create") == 1


def test_saved_request_must_match_current_source_before_network(tmp_path):
    jobs, kube = FakeJobs(), FakeKubectl()
    changed = {**request(), "title": "stale saved request"}
    with pytest.raises(JobsError, match="source-bound"):
        direct_submit_sft_once(
            plan=plan(),
            request=changed,
            jobs=jobs,
            kubectl=kube,
            journal=tmp_path / "DIRECT_SUBMISSION.jsonl",
            run_id=RUN_ID,
        )
    assert jobs.calls == [] and kube.calls == []


@pytest.mark.parametrize("authority", ["api-name", "api-output", "kube-name", "kube-output"])
def test_duplicates_stop_before_intent_or_create(tmp_path, authority):
    rows = []
    inventories = {
        "rayjobs.ray.io": {"kind": "List", "items": []},
        "jobs.batch": {"kind": "List", "items": []},
    }
    if authority == "api-name":
        rows = [{"name": "researcher-sft-deadbeef", "run_dir": "/mnt/sfs/jobs/other"}]
    elif authority == "api-output":
        rows = [{"name": "other", "run_dir": request()["run_dir"]}]
    else:
        metadata = {"name": "other", "labels": {}, "annotations": {}}
        if authority == "kube-name":
            metadata["name"] = "researcher-sft-deadbeef"
        else:
            metadata["annotations"]["fleet.ai/run-dir"] = request()["run_dir"]
        inventories["rayjobs.ray.io"]["items"].append({"metadata": metadata})
    jobs, kube = FakeJobs(rows=rows), FakeKubectl(inventories=inventories)
    journal = tmp_path / "DIRECT_SUBMISSION.jsonl"
    with pytest.raises(JobsError, match="already owns"):
        direct_submit_sft_once(
            plan=plan(),
            request=request(),
            jobs=jobs,
            kubectl=kube,
            journal=journal,
            run_id=RUN_ID,
        )
    assert not journal.exists()
    assert not any(call[0] == "create" for call in kube.calls)


def test_kubectl_boundary_uses_only_get_server_dry_run_and_one_create(monkeypatch):
    calls = []
    rendered = manifest()

    def run(command, **kwargs):
        calls.append((command, kwargs.get("input")))
        output = {"kind": "List", "items": []} if "get" in command else deepcopy(rendered)
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(output), stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    kube = Kubectl("prod-context")
    kube.list("rayjobs.ray.io")
    kube.dry_run(rendered)
    kube.create_once(rendered)
    tokens = [token for command, _ in calls for token in command]
    assert "apply" not in tokens and "patch" not in tokens and "delete" not in tokens
    assert sum(command.count("create") for command, _ in calls) == 2
    assert sum("--dry-run=server" in command for command, _ in calls) == 1
    assert (
        sum("create" in command and "--dry-run=server" not in command for command, _ in calls) == 1
    )


def test_cpu_checkpoint_boundary_previews_and_creates_only_unpinned_zero_gpu_pod(monkeypatch):
    calls = []
    expected = cpu_checkpoint_pod()

    def run(command, **kwargs):
        calls.append((command, json.loads(kwargs["input"])))
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(expected), stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    kube = Kubectl("prod-context")
    kube.dry_run_cpu_checkpoint_pod(expected)
    kube.create_cpu_checkpoint_pod_once(expected)
    assert len(calls) == 2
    assert "--dry-run=server" in calls[0][0]
    assert "--dry-run=server" not in calls[1][0]
    assert all(call[1]["spec"]["nodeSelector"] == CPU_NODE_SELECTOR for call in calls)


@pytest.mark.parametrize(
    "fault",
    [
        "hostname-selector",
        "node-name",
        "affinity",
        "gpu-request",
        "wrong-pool",
        "wrong-priority",
        "missing-alert-opt-out",
        "missing-operation",
    ],
)
def test_cpu_checkpoint_create_boundary_rejects_unsafe_placement_before_kubectl(monkeypatch, fault):
    calls = []
    pod = cpu_checkpoint_pod()
    spec = pod["spec"]
    if fault == "hostname-selector":
        spec["nodeSelector"]["kubernetes.io/hostname"] = "busy-host"
    elif fault == "node-name":
        spec["nodeName"] = "busy-host"
    elif fault == "affinity":
        spec["affinity"] = {"nodeAffinity": {}}
    elif fault == "gpu-request":
        spec["containers"][0]["resources"]["requests"]["nvidia.com/gpu"] = 1
    elif fault == "wrong-pool":
        spec["nodeSelector"]["workload"] = "fleetai-training-ng-gpu"
    elif fault == "wrong-priority":
        spec["priorityClassName"] = "c0"
    elif fault == "missing-alert-opt-out":
        pod["metadata"]["annotations"].pop("fleet.ai/failure-alerts")
    else:
        pod["metadata"]["annotations"].pop(CPU_CHECKPOINT_OPERATION_ANNOTATION)

    def run(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("unsafe CPU checkpoint Pod reached kubectl")

    monkeypatch.setattr(subprocess, "run", run)
    kube = Kubectl("prod-context")
    with pytest.raises(JobsError, match="CPU checkpoint Pod"):
        kube.create_cpu_checkpoint_pod_once(pod)
    assert calls == []


def test_invalid_uuid_or_context_fails_locally():
    with pytest.raises(JobsError, match="UUIDv4"):
        render_sft_rayjob(plan(), request(), preview(), run_id="not-a-uuid")
    for value in ("", "--current", "spaces are unsafe"):
        with pytest.raises(JobsError):
            Kubectl(value)
