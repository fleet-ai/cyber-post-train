import importlib.util
import io
import json
import subprocess
import tarfile
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from cyber_post_train import lora_cpu_preflight_driver
from cyber_post_train.direct_submit import (
    CPU_CHECKPOINT_OPERATION_ANNOTATION,
    CPU_NODE_SELECTOR,
    CPU_PREFLIGHT_BUNDLE_SHA256_ANNOTATION,
    CPU_SFS_MEMORY_FLOOR_MIB_ANNOTATION,
    CPU_SFS_OUTPUT_ROOT_ANNOTATION,
    CPU_SFS_OWNED_ROOT_ANNOTATION,
    CPU_SOURCE_ARCHIVE_SHA256_ANNOTATION,
    CPU_SOURCE_COMMIT_ANNOTATION,
    LORA_TRAINER_IMAGE,
    SFT_PRODUCTION_CONTEXT,
    TRAINING_GPU_CLUSTER_SELECTOR,
    Kubectl,
    collect_sfs_output_check,
    create_sfs_output_check_once,
    direct_submit_lr30_qualification_once,
    direct_submit_sft_once,
    render_lr30_qualification_rayjob,
    render_sft_rayjob,
)
from cyber_post_train.jobs import JobsError, digest
from cyber_post_train.lora_cpu_preflight import build_lora_cpu_preflight_package
from cyber_post_train.lora_cpu_preflight_driver import (
    ENV_BUNDLE_SHA256,
    ENV_SOURCE_ARCHIVE,
    ENV_SOURCE_ARCHIVE_SHA256,
    ENV_SOURCE_COMMIT,
)
from cyber_post_train.sfs_output import build_output_absence_receipt
from cyber_post_train.sfs_output_job import build_sfs_output_job
from cyber_post_train.source_bundle import canonical_source_commit_bytes
from training import qwen38_lr30_step76_gate as lr30
from training import sft

RUN_ID = "12345678-1234-4234-9234-123456789abc"
CREATED_UID = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
SOURCE_COMMIT = "a108edff2062558359cc72ebdfaf5d40cadeb333"


@pytest.fixture(autouse=True)
def current_sft_renderer(monkeypatch):
    monkeypatch.setattr(sft, "job_request", lambda _: request())


@pytest.fixture
def sfs_jobs_root(tmp_path):
    root = tmp_path / "jobs"
    root.mkdir()
    return root


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
            "securityContext": {"supplementalGroups": [2000]},
            "tolerations": [
                {
                    "effect": "NoSchedule",
                    "key": "workload",
                    "operator": "Equal",
                    "value": "fleetai-training-ng-gpu",
                }
            ],
            "volumes": [
                {
                    "name": "sfs",
                    "persistentVolumeClaim": {"claimName": "sfs-shared"},
                },
                {
                    "name": "shm",
                    "emptyDir": {"medium": "Memory", "sizeLimit": "64Gi"},
                },
                {
                    "name": "trajectory-spool",
                    "hostPath": {
                        "path": "/scratch/trajectories",
                        "type": "DirectoryOrCreate",
                    },
                },
            ],
            "initContainers": [
                {
                    "name": "sfs-init",
                    "image": "busybox:1.36",
                    "command": [
                        "sh",
                        "-c",
                        f"mkdir -p {value['run_dir']} && chown 1000:100 {value['run_dir']}",
                    ],
                    "securityContext": {"runAsUser": 0},
                    "volumeMounts": [{"name": "sfs", "mountPath": "/mnt/sfs"}],
                },
                {
                    "name": "prepare-trajectory-spool",
                    "image": "public.ecr.aws/docker/library/busybox:1.37.0",
                    "command": ["sh", "-ec"],
                    "args": ["chgrp 2000 /trajectory-spool; chmod 2770 /trajectory-spool"],
                    "securityContext": {
                        "allowPrivilegeEscalation": False,
                        "capabilities": {"add": ["CHOWN", "FOWNER"], "drop": ["ALL"]},
                        "readOnlyRootFilesystem": True,
                        "runAsNonRoot": False,
                        "runAsUser": 0,
                    },
                    "volumeMounts": [
                        {"name": "trajectory-spool", "mountPath": "/trajectory-spool"}
                    ],
                },
            ],
            "containers": [
                {
                    "name": container_name,
                    "image": value["image"],
                    "env": [
                        {"name": name, "value": item}
                        for name, item in {**generated, **value["env"]}.items()
                    ],
                    "envFrom": [
                        *[{"secretRef": {"name": name}} for name in value.get("secrets", [])],
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
                    "volumeMounts": [
                        {"name": "sfs", "mountPath": "/mnt/sfs"},
                        {"name": "shm", "mountPath": "/dev/shm"},
                        {
                            "name": "trajectory-spool",
                            "mountPath": "/mnt/fleet/trajectory-spool",
                        },
                    ],
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
                "headGroupSpec": {
                    "rayStartParams": {"dashboard-host": "0.0.0.0"},
                    "template": template(value, "ray-head"),
                },
                "workerGroupSpecs": (
                    [
                        {
                            "groupName": "gpu",
                            "replicas": value["workers"] - 1,
                            "minReplicas": value["workers"] - 1,
                            "maxReplicas": value["workers"] - 1,
                            "rayStartParams": {},
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


def cpu_sfs_control_pod(*, source_bound=True):
    pod = cpu_checkpoint_pod()
    owned = "/mnt/sfs/jobs/chris-q38-study-corpora-v1/launch-controls"
    output = owned + "/.preflight-control-step60-a21cbe7c"
    pod["metadata"]["annotations"].update(
        {
            CPU_SFS_OWNED_ROOT_ANNOTATION: owned,
            CPU_SFS_OUTPUT_ROOT_ANNOTATION: output,
            CPU_SFS_MEMORY_FLOOR_MIB_ANNOTATION: "26207",
        }
    )
    pod["spec"].update(
        {
            "activeDeadlineSeconds": 3600,
            "automountServiceAccountToken": False,
            "imagePullSecrets": [{"name": "ghcr-pull"}],
            "securityContext": {
                "runAsNonRoot": True,
                "runAsUser": 1000,
                "runAsGroup": 100,
                "fsGroup": 100,
            },
            "volumes": [
                {
                    "name": "bundle",
                    "configMap": {
                        "name": "researcher-checkpoint-seal-v1-source",
                        "defaultMode": 292,
                    },
                },
                {
                    "name": "sfs-readonly",
                    "persistentVolumeClaim": {"claimName": "sfs-shared"},
                },
                {
                    "name": "sfs-control",
                    "persistentVolumeClaim": {"claimName": "sfs-shared"},
                },
            ],
        }
    )
    container = pod["spec"]["containers"][0]
    container.update(
        {
            "image": LORA_TRAINER_IMAGE,
            "command": ["python", "/bundle/preflight_driver.py"],
            "env": [
                {"name": "CYBER_SFS_OWNED_ROOT", "value": owned},
                {"name": "CYBER_SFS_OUTPUT_ROOT", "value": output},
                {"name": "CYBER_SFS_CONTROL_MOUNT", "value": "/controls"},
            ],
            "resources": {
                "requests": {"cpu": "4", "memory": "32Gi"},
                "limits": {"cpu": "8", "memory": "48Gi"},
            },
            "securityContext": {
                "allowPrivilegeEscalation": False,
                "capabilities": {"drop": ["ALL"]},
            },
            "volumeMounts": [
                {"name": "bundle", "mountPath": "/bundle", "readOnly": True},
                {"name": "sfs-readonly", "mountPath": "/mnt/sfs", "readOnly": True},
                {
                    "name": "sfs-control",
                    "mountPath": "/controls",
                    "subPath": "jobs/chris-q38-study-corpora-v1/launch-controls",
                },
            ],
        }
    )
    if source_bound:
        bundle_sha256 = "b" * 64
        archive_sha256 = "c" * 64
        pod["metadata"]["annotations"].update(
            {
                CPU_PREFLIGHT_BUNDLE_SHA256_ANNOTATION: bundle_sha256,
                CPU_SOURCE_ARCHIVE_SHA256_ANNOTATION: archive_sha256,
                CPU_SOURCE_COMMIT_ANNOTATION: SOURCE_COMMIT,
            }
        )
        container["env"].extend(
            [
                {"name": ENV_BUNDLE_SHA256, "value": bundle_sha256},
                {"name": ENV_SOURCE_ARCHIVE, "value": "/mnt/sfs/source.tgz"},
                {"name": ENV_SOURCE_ARCHIVE_SHA256, "value": archive_sha256},
                {"name": ENV_SOURCE_COMMIT, "value": SOURCE_COMMIT},
            ]
        )
    return pod


def source_archive(path: Path, commit: str = SOURCE_COMMIT) -> None:
    value = canonical_source_commit_bytes(commit)
    with tarfile.open(path, "w:gz") as archive:
        member = tarfile.TarInfo("SOURCE_COMMIT")
        member.size = len(value)
        archive.addfile(member, io.BytesIO(value))


def cpu_node_inventory():
    return {
        "apiVersion": "v1",
        "kind": "List",
        "items": [
            {
                "metadata": {
                    "name": "shared-cpu-1",
                    "labels": deepcopy(CPU_NODE_SELECTOR),
                },
                "spec": {},
                "status": {
                    "allocatable": {"cpu": "15900m", "memory": "65216572Ki"},
                    "conditions": [{"type": "Ready", "status": "True"}],
                },
            }
        ],
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
    rendered, proof = render_sft_rayjob(
        plan(),
        request(),
        preview(source),
        kubernetes_context=SFT_PRODUCTION_CONTEXT,
        run_id=RUN_ID,
    )
    assert proof["name"] == "researcher-sft-12345678"
    assert proof["run_id"] == RUN_ID
    assert proof["removed_api_fleet_secrets"] == 2
    assert rendered["metadata"]["annotations"]["fleet.ai/failure-alerts"] == "off"
    assert rendered["metadata"]["labels"]["fleet.ai/run-id"] == RUN_ID
    assert rendered["metadata"]["annotations"]["fleet.ai/run-id"] == RUN_ID
    assert source["metadata"]["annotations"].get("fleet.ai/failure-alerts") is None
    assert proof["kubernetes_context"] == SFT_PRODUCTION_CONTEXT
    assert proof["gpu_cluster_selector"] == TRAINING_GPU_CLUSTER_SELECTOR
    assert proof["bound_gpu_cluster_templates"] == 2

    source_groups = [
        source["spec"]["rayClusterSpec"]["headGroupSpec"],
        *source["spec"]["rayClusterSpec"]["workerGroupSpecs"],
    ]
    rendered_groups = [
        rendered["spec"]["rayClusterSpec"]["headGroupSpec"],
        *rendered["spec"]["rayClusterSpec"]["workerGroupSpecs"],
    ]
    assert all(
        group["template"]["spec"]["nodeSelector"] == {"workload": "fleetai-training-ng-gpu"}
        for group in source_groups
    )
    for group in rendered_groups:
        pod = group["template"]
        container = pod["spec"]["containers"][0]
        env = {item["name"]: item["value"] for item in container["env"]}
        assert pod["spec"]["nodeSelector"] == {
            **TRAINING_GPU_CLUSTER_SELECTOR,
            "workload": "fleetai-training-ng-gpu",
        }
        assert pod["metadata"]["labels"]["fleet.ai/run-id"] == RUN_ID
        assert pod["spec"]["priority"] == 10_000
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
        group["template"]["spec"]["containers"][0]["resources"]["requests"]["nvidia.com/gpu"] == 1
        for group in groups
    )
    assert all(
        group["template"]["spec"]["nodeSelector"] == {"workload": "fleetai-training-ng-gpu"}
        for group in groups
    )


@pytest.mark.parametrize(
    "fault",
    [
        "already-annotated",
        "wrong-priority",
        "wrong-numeric-priority",
        "not-suspended",
        "wrong-image",
        "wrong-resource",
        "extra-env",
        "missing-fleet-secret",
        "extra-secret",
        "unknown-zero-id",
        "unknown-placeholder-name",
        "duplicate-container",
        "privileged-main-container",
        "privileged-secret-init-container",
        "secret-volume",
        "node-name",
        "affinity",
        "runtime-class",
        "scheduling-gate",
        "pod-security-context",
        "autoscaling",
        "extra-pull-secret",
        "missing-storage-bundle",
        "missing-ray-start-params",
        "missing-generic-gpu-selector",
        "wrong-generic-gpu-selector",
        "unexpected-prebound-gpu-cluster-selector",
        "wrong-gpu-cluster-selector",
        "token-submitter",
        "newline-submitter",
        "oversize-submitter",
        "invalid-profile",
        "noncanonical-profile",
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
    elif fault == "wrong-numeric-priority":
        head["spec"]["priority"] = 0
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
    elif fault == "privileged-main-container":
        container["securityContext"] = {"privileged": True}
    elif fault == "privileged-secret-init-container":
        head["spec"]["initContainers"] = [
            {
                "name": "unreviewed",
                "image": request()["image"],
                "envFrom": [{"secretRef": {"name": "unreviewed-secret"}}],
                "resources": {
                    "requests": {"cpu": "1", "memory": "1Gi"},
                    "limits": {"cpu": "1", "memory": "1Gi"},
                },
                "securityContext": {"privileged": True},
            }
        ]
    elif fault == "secret-volume":
        head["spec"]["volumes"] = [
            {"name": "secret", "secret": {"secretName": "unreviewed-secret"}}
        ]
        container["volumeMounts"] = [{"name": "secret", "mountPath": "/secret"}]
    elif fault == "node-name":
        head["spec"]["nodeName"] = "chosen-node"
    elif fault == "affinity":
        head["spec"]["affinity"] = {"nodeAffinity": {}}
    elif fault == "runtime-class":
        head["spec"]["runtimeClassName"] = "unreviewed"
    elif fault == "scheduling-gate":
        head["spec"]["schedulingGates"] = [{"name": "unreviewed"}]
    elif fault == "pod-security-context":
        head["spec"]["securityContext"] = {"runAsUser": 0, "runAsNonRoot": False}
    elif fault == "autoscaling":
        obj["spec"]["rayClusterSpec"]["enableInTreeAutoscaling"] = True
    elif fault == "extra-pull-secret":
        head["spec"]["imagePullSecrets"].append({"name": "unreviewed-secret"})
    elif fault == "missing-storage-bundle":
        for field in ("initContainers", "securityContext", "tolerations", "volumes"):
            head["spec"].pop(field)
        container.pop("volumeMounts")
    elif fault == "missing-ray-start-params":
        obj["spec"]["rayClusterSpec"]["headGroupSpec"].pop("rayStartParams")
    elif fault == "missing-generic-gpu-selector":
        head["spec"]["nodeSelector"].pop("workload")
    elif fault == "wrong-generic-gpu-selector":
        head["spec"]["nodeSelector"]["workload"] = "fleetai-training-ng-cpu"
    elif fault == "unexpected-prebound-gpu-cluster-selector":
        head["spec"]["nodeSelector"].update(TRAINING_GPU_CLUSTER_SELECTOR)
    elif fault == "wrong-gpu-cluster-selector":
        head["spec"]["nodeSelector"]["topology.nebius.com/gpu-cluster-id"] = (
            "computegpucluster-wrong"
        )
    elif fault == "token-submitter":
        obj["metadata"]["annotations"]["fleet.ai/submitted-by"] = "Bearer unreviewed-token"
    elif fault == "newline-submitter":
        obj["metadata"]["annotations"]["fleet.ai/submitted-by"] = "a@b.co\nsecret"
    elif fault == "oversize-submitter":
        obj["metadata"]["annotations"]["fleet.ai/submitted-by"] = "a" * 250 + "@b.co"
    elif fault == "invalid-profile":
        obj["metadata"]["annotations"]["fleet.ai/submitted-by-profile"] = "-" * 36
    elif fault == "noncanonical-profile":
        obj["metadata"]["annotations"]["fleet.ai/submitted-by-profile"] = (
            "9909B292-D23E-4E2E-8587-E4CD0D5C47BF"
        )
    else:
        response["warnings"] = ["server changed"]
    response["manifest_yaml"] = yaml.safe_dump(obj)
    with pytest.raises(JobsError):
        render_sft_rayjob(
            plan(),
            request(),
            response,
            kubernetes_context=SFT_PRODUCTION_CONTEXT,
            run_id=RUN_ID,
        )


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
        render_sft_rayjob(
            plan_value,
            value,
            preview(manifest(value)),
            kubernetes_context=SFT_PRODUCTION_CONTEXT,
            run_id=RUN_ID,
        )


@pytest.mark.parametrize(
    "context",
    [
        "nebius-mk8s-fleetai-training-dev-e04p03enwk5c0va9tb",
        "production-context",
    ],
)
def test_sft_gpu_cluster_binding_rejects_unknown_or_development_context_before_network(
    tmp_path, sfs_jobs_root, context
):
    jobs, kube = FakeJobs(), FakeKubectl()
    kube.context = context
    journal = tmp_path / "DIRECT_SUBMISSION.jsonl"

    with pytest.raises(JobsError, match="exact production context"):
        direct_submit_sft_once(
            plan=plan(),
            request=request(),
            jobs=jobs,
            kubectl=kube,
            journal=journal,
            run_id=RUN_ID,
            jobs_root=sfs_jobs_root,
        )

    assert jobs.calls == [] and kube.calls == []
    assert not journal.exists()


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
    context = SFT_PRODUCTION_CONTEXT

    def __init__(self, *, inventories=None, fail_create=False):
        self.inventories = inventories or {
            "rayjobs.ray.io": {"kind": "List", "items": []},
            "jobs.batch": {"kind": "List", "items": []},
        }
        self.fail_create = fail_create
        self.calls = []
        self.created = None

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
        self.created = deepcopy(result)
        return result

    def get_rayjob(self, name):
        self.calls.append(("get", name))
        assert self.created is not None and self.created["metadata"]["name"] == name
        return deepcopy(self.created)


class FakeOutputCheckKubectl(FakeKubectl):
    def _cpu_node_inventory(self):
        self.calls.append(("list", "nodes"))
        return cpu_node_inventory()


def test_direct_submit_checks_twice_journals_then_creates_exactly_once(tmp_path, sfs_jobs_root):
    jobs, kube = FakeJobs(), FakeKubectl()
    journal = tmp_path / "DIRECT_SUBMISSION.jsonl"
    result = direct_submit_sft_once(
        plan=plan(),
        request=request(),
        jobs=jobs,
        kubectl=kube,
        journal=journal,
        run_id=RUN_ID,
        jobs_root=sfs_jobs_root,
    )
    assert result["uid"] == CREATED_UID and result["name"] == "researcher-sft-12345678"
    assert jobs.calls == ["history", ("preview", request()), "history"]
    assert [call[0] for call in kube.calls].count("list") == 4
    assert [call[0] for call in kube.calls].count("dry-run") == 1
    assert [call[0] for call in kube.calls].count("create") == 1
    assert [call[0] for call in kube.calls].count("get") == 1
    records = [json.loads(line) for line in journal.read_text().splitlines()]
    assert [record["state"] for record in records] == [
        "KUBECTL_CREATE_INTENT_DO_NOT_RETRY",
        "KUBECTL_CREATE_RESPONSE",
    ]
    assert len(records[0]["output_absence_receipt_sha256"]) == 64
    assert records[0]["kubernetes_context"] == SFT_PRODUCTION_CONTEXT
    assert records[0]["gpu_cluster_selector"] == TRAINING_GPU_CLUSTER_SELECTOR
    assert records[0]["bound_gpu_cluster_templates"] == 2
    assert journal.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(
    ("target", "fault"),
    [
        ("head", "missing-generic"),
        ("head", "wrong-generic"),
        ("worker", "prebound-exact-cluster"),
        ("worker", "prebound-wrong-cluster"),
    ],
)
def test_invalid_source_gpu_selector_stops_before_intent_or_create(
    tmp_path, sfs_jobs_root, target, fault
):
    source = manifest()
    ray_cluster = source["spec"]["rayClusterSpec"]
    template_value = (
        ray_cluster["headGroupSpec"]["template"]
        if target == "head"
        else ray_cluster["workerGroupSpecs"][0]["template"]
    )
    selector = template_value["spec"]["nodeSelector"]
    if fault == "missing-generic":
        selector.pop("workload")
    elif fault == "wrong-generic":
        selector["workload"] = "fleetai-training-ng-cpu"
    elif fault == "prebound-exact-cluster":
        selector.update(TRAINING_GPU_CLUSTER_SELECTOR)
    else:
        selector["topology.nebius.com/gpu-cluster-id"] = "computegpucluster-wrong"
    jobs = FakeJobs(preview_value=preview(source))
    kube = FakeKubectl()
    journal = tmp_path / "DIRECT_SUBMISSION.jsonl"

    with pytest.raises(JobsError, match="node selector drift"):
        direct_submit_sft_once(
            plan=plan(),
            request=request(),
            jobs=jobs,
            kubectl=kube,
            journal=journal,
            run_id=RUN_ID,
            jobs_root=sfs_jobs_root,
        )

    assert not journal.exists()
    assert not any(call[0] in {"dry-run", "create"} for call in kube.calls)


def test_output_check_create_is_suspended_queued_journaled_and_create_once(tmp_path):
    kube = FakeOutputCheckKubectl()
    journal = tmp_path / "SFS_OUTPUT_CHECK_A01.jsonl"
    result = create_sfs_output_check_once(
        plan=plan(),
        request=request(),
        attempt=1,
        kubectl=kube,
        journal=journal,
    )
    assert result == {
        "submitted": True,
        "gpus": 0,
        "name": "researcher-sft-sfs-a01",
        "uid": CREATED_UID,
        "attempt": 1,
    }
    assert [call[0] for call in kube.calls].count("list") == 3
    assert [call[0] for call in kube.calls].count("dry-run") == 1
    assert [call[0] for call in kube.calls].count("create") == 1
    records = [json.loads(line) for line in journal.read_text().splitlines()]
    assert [row["state"] for row in records] == [
        "KUBECTL_CREATE_INTENT_DO_NOT_RETRY",
        "KUBECTL_CREATE_RESPONSE",
    ]
    with pytest.raises(JobsError, match="journal already exists"):
        create_sfs_output_check_once(
            plan=plan(),
            request=request(),
            attempt=1,
            kubectl=kube,
            journal=journal,
        )
    assert [call[0] for call in kube.calls].count("create") == 1


def test_output_check_exact_duplicate_stops_before_dry_run_or_intent(tmp_path):
    package = build_sfs_output_job(plan(), request(), 1)
    kube = FakeOutputCheckKubectl(
        inventories={
            "rayjobs.ray.io": {"kind": "List", "items": []},
            "jobs.batch": {"kind": "JobList", "items": [deepcopy(package.job)]},
        }
    )
    journal = tmp_path / "SFS_OUTPUT_CHECK_A01.jsonl"
    with pytest.raises(JobsError, match="already exists"):
        create_sfs_output_check_once(
            plan=plan(),
            request=request(),
            attempt=1,
            kubectl=kube,
            journal=journal,
        )
    assert not journal.exists()
    assert not any(call[0] in {"dry-run", "create"} for call in kube.calls)


def test_output_check_collection_requires_admitted_exact_job_pod_and_receipt(tmp_path):
    package = build_sfs_output_job(plan(), request(), 1)
    job = deepcopy(package.job)
    job["metadata"]["uid"] = CREATED_UID
    job["spec"]["suspend"] = False
    workload_name = "job-researcher-sft-sfs-a01-abcde"
    batch_labels = {
        "batch.kubernetes.io/controller-uid": CREATED_UID,
        "batch.kubernetes.io/job-name": package.job["metadata"]["name"],
        "controller-uid": CREATED_UID,
        "job-name": package.job["metadata"]["name"],
    }
    kueue_labels = {
        "kueue.x-k8s.io/cluster-queue-name": "training-cq",
        "kueue.x-k8s.io/local-queue-name": "training-lq",
        "kueue.x-k8s.io/podset": "main",
    }
    job["spec"]["selector"] = {"matchLabels": {"batch.kubernetes.io/controller-uid": CREATED_UID}}
    live_template = job["spec"]["template"]
    live_template["metadata"]["annotations"]["kueue.x-k8s.io/workload"] = workload_name
    live_template["metadata"]["labels"].update({**batch_labels, **kueue_labels})
    job["status"] = {"conditions": [{"type": "Complete", "status": "True"}]}
    pod = {
        "metadata": {
            "name": "researcher-sft-sfs-a01-abcde",
            "uid": "11111111-2222-4333-8444-555555555555",
            "annotations": deepcopy(live_template["metadata"]["annotations"]),
            "labels": {
                **deepcopy(live_template["metadata"]["labels"]),
                "topology.kubernetes.io/region": "eu-north1",
            },
            "ownerReferences": [
                {
                    "apiVersion": "batch/v1",
                    "kind": "Job",
                    "name": package.job["metadata"]["name"],
                    "uid": CREATED_UID,
                    "controller": True,
                    "blockOwnerDeletion": True,
                }
            ],
        },
        "spec": deepcopy(live_template["spec"]),
        "status": {
            "phase": "Succeeded",
            "containerStatuses": [
                {
                    "restartCount": 0,
                    "imageID": "docker-pullable://registry/image@sha256:" + "a" * 64,
                    "state": {"terminated": {"exitCode": 0}},
                }
            ],
        },
    }
    pod["spec"]["nodeName"] = "shared-cpu-1"
    pod["spec"]["serviceAccount"] = "default"
    pod["spec"]["serviceAccountName"] = "default"
    # Kubernetes only copies the default ServiceAccount pull secret when the
    # submitted PodSpec has no explicit pull secrets. This fixture's request
    # deliberately carries registry-pull, so the scheduled Pod must preserve
    # that exact reviewed value rather than gain ecr-pull.
    workload = {
        "apiVersion": "kueue.x-k8s.io/v1beta2",
        "kind": "Workload",
        "metadata": {
            "name": workload_name,
            "namespace": "fleet-train-jobs",
            "uid": "99999999-2222-4333-8444-555555555555",
            "labels": {"kueue.x-k8s.io/job-uid": CREATED_UID},
            "ownerReferences": [
                {
                    "apiVersion": "batch/v1",
                    "kind": "Job",
                    "name": package.job["metadata"]["name"],
                    "uid": CREATED_UID,
                    "controller": True,
                    "blockOwnerDeletion": True,
                }
            ],
        },
        "spec": {
            "queueName": "training-lq",
            "priority": 10000,
            "priorityClassRef": {
                "group": "kueue.x-k8s.io",
                "kind": "WorkloadPriorityClass",
                "name": "q1",
            },
            "active": True,
            "podSets": [
                {
                    "name": "main",
                    "count": 1,
                    "topologyRequest": {
                        "podIndexLabel": "batch.kubernetes.io/job-completion-index"
                    },
                    "template": {
                        "metadata": {
                            "annotations": deepcopy(
                                package.job["spec"]["template"]["metadata"]["annotations"]
                            ),
                            "labels": {
                                **deepcopy(package.job["spec"]["template"]["metadata"]["labels"]),
                                "batch.kubernetes.io/job-name": package.job["metadata"]["name"],
                            },
                        },
                        "spec": {
                            **deepcopy(package.job["spec"]["template"]["spec"]),
                            "priority": None,
                        },
                    },
                }
            ],
        },
        "status": {
            "admission": {
                "clusterQueue": "training-cq",
                "podSetAssignments": [
                    {
                        "name": "main",
                        "count": 1,
                        "flavors": {"cpu": "cpu", "memory": "cpu"},
                        "resourceUsage": {"cpu": "1", "memory": "1Gi"},
                    }
                ],
            },
            "conditions": [
                {"type": "Admitted", "status": "True"},
                {"type": "Finished", "status": "True", "reason": "Succeeded"},
            ],
        },
    }
    jobs_root = tmp_path / "jobs"
    jobs_root.mkdir()
    receipt = build_output_absence_receipt(plan(), request(), jobs_root=jobs_root, now=time.time())

    class Collector(FakeOutputCheckKubectl):
        def get_output_check_job(self, name):
            assert name == package.job["metadata"]["name"]
            return deepcopy(job)

        def list_output_check_pods(self, name):
            assert name == package.job["metadata"]["name"]
            return {"kind": "List", "items": [deepcopy(pod)]}

        def list_output_check_workloads(self, job_uid):
            assert job_uid == CREATED_UID
            return {"kind": "List", "items": [deepcopy(workload)]}

        def get_output_check_service_account(self):
            return {
                "apiVersion": "v1",
                "kind": "ServiceAccount",
                "metadata": {"name": "default", "namespace": "fleet-train-jobs"},
                "imagePullSecrets": [{"name": "ecr-pull"}],
            }

        def output_check_logs(self, name):
            assert name == pod["metadata"]["name"]
            return "CYBER_SFT_OUTPUT_ABSENCE=" + json.dumps(
                receipt, sort_keys=True, separators=(",", ":")
            )

    assert (
        collect_sfs_output_check(plan=plan(), request=request(), attempt=1, kubectl=Collector())
        == receipt
    )


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


def test_ambiguous_create_leaves_intent_and_never_retries(tmp_path, sfs_jobs_root):
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
            jobs_root=sfs_jobs_root,
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
            jobs_root=sfs_jobs_root,
        )
    assert [call[0] for call in kube.calls].count("create") == 1


def test_saved_request_must_match_current_source_before_network(tmp_path, sfs_jobs_root):
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
            jobs_root=sfs_jobs_root,
        )
    assert jobs.calls == [] and kube.calls == []


@pytest.mark.parametrize(
    "authority", ["api-name", "api-title", "api-output", "kube-name", "kube-output"]
)
def test_duplicates_stop_before_intent_or_create(tmp_path, authority, sfs_jobs_root):
    rows = []
    inventories = {
        "rayjobs.ray.io": {"kind": "List", "items": []},
        "jobs.batch": {"kind": "List", "items": []},
    }
    if authority == "api-name":
        rows = [{"name": "researcher-sft-deadbeef", "run_dir": "/mnt/sfs/jobs/other"}]
    elif authority == "api-title":
        rows = [
            {
                "name": "other",
                "title": request()["title"],
                "run_dir": "/mnt/sfs/jobs/other",
            }
        ]
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
            jobs_root=sfs_jobs_root,
        )
    assert not journal.exists()
    assert not any(call[0] == "create" for call in kube.calls)


def test_completed_cpu_preflight_name_does_not_collide_with_training_identity(
    tmp_path, sfs_jobs_root
):
    inventories = {
        "rayjobs.ray.io": {"kind": "RayJobList", "items": []},
        "jobs.batch": {
            "kind": "JobList",
            "items": [
                {
                    "metadata": {
                        "name": "researcher-sft-pre-v1",
                        "labels": {},
                        "annotations": {},
                    },
                    "status": {"conditions": [{"type": "Complete", "status": "True"}]},
                }
            ],
        },
    }
    jobs, kube = FakeJobs(), FakeKubectl(inventories=inventories)
    result = direct_submit_sft_once(
        plan=plan(),
        request=request(),
        jobs=jobs,
        kubectl=kube,
        journal=tmp_path / "DIRECT_SUBMISSION.jsonl",
        run_id=RUN_ID,
        jobs_root=sfs_jobs_root,
    )
    assert result["name"] == "researcher-sft-12345678"
    assert [call[0] for call in kube.calls].count("create") == 1


def test_remote_submitter_requires_and_revalidates_source_bound_sfs_receipt(tmp_path):
    mounted = tmp_path / "mounted-jobs"
    mounted.mkdir()
    receipt = build_output_absence_receipt(plan(), request(), jobs_root=mounted, now=time.time())
    unavailable = tmp_path / "no-sfs"
    with pytest.raises(JobsError, match="provide a fresh source-bound"):
        direct_submit_sft_once(
            plan=plan(),
            request=request(),
            jobs=FakeJobs(),
            kubectl=FakeKubectl(),
            journal=tmp_path / "missing-receipt.jsonl",
            run_id=RUN_ID,
            jobs_root=unavailable,
        )
    result = direct_submit_sft_once(
        plan=plan(),
        request=request(),
        jobs=FakeJobs(),
        kubectl=FakeKubectl(),
        journal=tmp_path / "with-receipt.jsonl",
        run_id=RUN_ID,
        jobs_root=unavailable,
        output_absence_receipt=receipt,
    )
    assert result["submitted"] is True


def test_output_appearing_after_server_dry_run_stops_before_intent_or_create(
    tmp_path, sfs_jobs_root
):
    class OutputAppearsKubectl(FakeKubectl):
        def dry_run(self, obj):
            result = super().dry_run(obj)
            (sfs_jobs_root / "researcher-sft-v1").mkdir()
            return result

    jobs, kube = FakeJobs(), OutputAppearsKubectl()
    journal = tmp_path / "DIRECT_SUBMISSION.jsonl"
    with pytest.raises(JobsError, match="output already exists"):
        direct_submit_sft_once(
            plan=plan(),
            request=request(),
            jobs=jobs,
            kubectl=kube,
            journal=journal,
            run_id=RUN_ID,
            jobs_root=sfs_jobs_root,
        )
    assert not journal.exists()
    assert not any(call[0] == "create" for call in kube.calls)


@pytest.mark.parametrize("failure", [PermissionError("denied"), OSError("stale mount")])
def test_uninspectable_output_stops_before_network_intent_or_create(
    tmp_path, sfs_jobs_root, monkeypatch, failure
):
    jobs, kube = FakeJobs(), FakeKubectl()
    target = sfs_jobs_root / "researcher-sft-v1"
    original_lstat = Path.lstat

    def fail_target_lstat(path):
        if path == target:
            raise failure
        return original_lstat(path)

    monkeypatch.setattr(Path, "lstat", fail_target_lstat)
    journal = tmp_path / "DIRECT_SUBMISSION.jsonl"
    with pytest.raises(JobsError, match="cannot be inspected"):
        direct_submit_sft_once(
            plan=plan(),
            request=request(),
            jobs=jobs,
            kubectl=kube,
            journal=journal,
            run_id=RUN_ID,
            jobs_root=sfs_jobs_root,
        )
    assert not journal.exists()
    assert jobs.calls == []
    assert kube.calls == []


@pytest.mark.parametrize(
    "fault",
    [
        "priority",
        "secret-env",
        "image-pull-secret",
        "secret-sidecar",
        "pod-security-context",
        "topology-spread",
        "scheduling-gate",
        "autoscaling",
        "worker-max-replicas",
        "backoff-limit",
        "ttl-seconds",
        "worker-num-hosts",
        "head-num-hosts",
        "numeric-priority-missing",
        "numeric-priority-zero",
        "worker-numeric-priority-missing",
        "gpu-cluster-selector",
        "worker-gpu-cluster-selector",
    ],
)
def test_server_dry_run_runtime_drift_stops_before_intent_or_create(tmp_path, sfs_jobs_root, fault):
    class RuntimeDriftKubectl(FakeKubectl):
        def dry_run(self, obj):
            result = super().dry_run(obj)
            pod_spec = result["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"]
            container = pod_spec["containers"][0]
            if fault == "priority":
                pod_spec["priorityClassName"] = "c0"
            elif fault == "secret-env":
                container["envFrom"].append({"secretRef": {"name": "unreviewed"}})
            elif fault == "image-pull-secret":
                pod_spec["imagePullSecrets"].append({"name": "unreviewed"})
            elif fault == "secret-sidecar":
                sidecar = deepcopy(container)
                sidecar["name"] = "unreviewed-sidecar"
                sidecar["envFrom"] = [{"secretRef": {"name": "unreviewed"}}]
                sidecar["resources"] = {
                    "requests": {"cpu": "1", "memory": "1Gi"},
                    "limits": {"cpu": "1", "memory": "1Gi"},
                }
                pod_spec["containers"].append(sidecar)
            elif fault == "pod-security-context":
                pod_spec["securityContext"] = {
                    "runAsNonRoot": False,
                    "runAsUser": 0,
                }
            elif fault == "topology-spread":
                pod_spec["topologySpreadConstraints"] = [
                    {
                        "maxSkew": 1,
                        "topologyKey": "kubernetes.io/hostname",
                        "whenUnsatisfiable": "DoNotSchedule",
                        "labelSelector": {},
                    }
                ]
            elif fault == "scheduling-gate":
                pod_spec["schedulingGates"] = [{"name": "unreviewed"}]
            elif fault == "autoscaling":
                result["spec"]["rayClusterSpec"]["enableInTreeAutoscaling"] = True
            elif fault == "backoff-limit":
                result["spec"]["backoffLimit"] = 1
            elif fault == "ttl-seconds":
                result["spec"]["ttlSecondsAfterFinished"] = 1
            elif fault == "worker-num-hosts":
                result["spec"]["rayClusterSpec"]["workerGroupSpecs"][0]["numOfHosts"] = 2
            elif fault == "head-num-hosts":
                result["spec"]["rayClusterSpec"]["headGroupSpec"]["numOfHosts"] = 1
            elif fault == "numeric-priority-missing":
                pod_spec.pop("priority")
            elif fault == "numeric-priority-zero":
                pod_spec["priority"] = 0
            elif fault == "worker-numeric-priority-missing":
                result["spec"]["rayClusterSpec"]["workerGroupSpecs"][0]["template"]["spec"].pop(
                    "priority"
                )
            elif fault == "gpu-cluster-selector":
                pod_spec["nodeSelector"]["topology.nebius.com/gpu-cluster-id"] = (
                    "computegpucluster-wrong"
                )
            elif fault == "worker-gpu-cluster-selector":
                result["spec"]["rayClusterSpec"]["workerGroupSpecs"][0]["template"]["spec"][
                    "nodeSelector"
                ]["topology.nebius.com/gpu-cluster-id"] = "computegpucluster-wrong"
            else:
                result["spec"]["rayClusterSpec"]["workerGroupSpecs"][0]["maxReplicas"] = 8
            return result

    jobs, kube = FakeJobs(), RuntimeDriftKubectl()
    journal = tmp_path / "DIRECT_SUBMISSION.jsonl"
    with pytest.raises(JobsError):
        direct_submit_sft_once(
            plan=plan(),
            request=request(),
            jobs=jobs,
            kubectl=kube,
            journal=journal,
            run_id=RUN_ID,
            jobs_root=sfs_jobs_root,
        )
    assert not journal.exists()
    assert not any(call[0] == "create" for call in kube.calls)


def test_server_harmless_api_defaults_are_normalized_exactly(tmp_path, sfs_jobs_root):
    class DefaultingKubectl(FakeKubectl):
        def dry_run(self, obj):
            result = super().dry_run(obj)
            result["spec"].update(
                {
                    "backoffLimit": 0,
                    "ttlSecondsAfterFinished": 0,
                }
            )
            for worker_group in result["spec"]["rayClusterSpec"]["workerGroupSpecs"]:
                worker_group["numOfHosts"] = 1
            pod_spec = result["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"]
            pod_spec.update(
                {
                    "dnsPolicy": "ClusterFirst",
                    "enableServiceLinks": True,
                    "preemptionPolicy": "PreemptLowerPriority",
                    "priority": 10000,
                    "schedulerName": "default-scheduler",
                    "serviceAccount": "default",
                    "serviceAccountName": "default",
                    "terminationGracePeriodSeconds": 30,
                }
            )
            pod_spec["containers"][0].update(
                {
                    "terminationMessagePath": "/dev/termination-log",
                    "terminationMessagePolicy": "File",
                }
            )
            result["metadata"]["creationTimestamp"] = None
            return result

        def create_once(self, obj):
            result = super().create_once(obj)
            result["metadata"]["finalizers"] = ["ray.io/rayjob-finalizer"]
            pod_spec = result["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"]
            resources = pod_spec["containers"][0]["resources"]
            resources["requests"]["nvidia.com/gpu"] = "8"
            resources["limits"]["nvidia.com/gpu"] = "8"
            for init_container in pod_spec["initContainers"]:
                init_container["resources"] = {}
            self.created = deepcopy(result)
            return result

    result = direct_submit_sft_once(
        plan=plan(),
        request=request(),
        jobs=FakeJobs(),
        kubectl=DefaultingKubectl(),
        journal=tmp_path / "DIRECT_SUBMISSION.jsonl",
        run_id=RUN_ID,
        jobs_root=sfs_jobs_root,
    )
    assert result["submitted"] is True


@pytest.mark.parametrize(
    ("fault", "value"),
    [
        ("finalizers", ["ray.io/rayjob-finalizer", "unreviewed"]),
        ("gpu-request", "8.0"),
        ("gpu-limit", "7"),
        ("init-resources", {"requests": {"cpu": "1m"}}),
    ],
)
def test_nearby_create_defaults_remain_strict(tmp_path, sfs_jobs_root, fault, value):
    class DefaultDriftKubectl(FakeKubectl):
        def create_once(self, obj):
            result = super().create_once(obj)
            pod_spec = result["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"]
            if fault == "finalizers":
                result["metadata"]["finalizers"] = value
            elif fault == "gpu-request":
                result["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"]["containers"][
                    0
                ]["resources"]["requests"]["nvidia.com/gpu"] = value
            elif fault == "gpu-limit":
                pod_spec["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] = value
            else:
                pod_spec["initContainers"][0]["resources"] = value
            self.created = deepcopy(result)
            return result

    journal = tmp_path / "DIRECT_SUBMISSION.jsonl"
    with pytest.raises(JobsError):
        direct_submit_sft_once(
            plan=plan(),
            request=request(),
            jobs=FakeJobs(),
            kubectl=DefaultDriftKubectl(),
            journal=journal,
            run_id=RUN_ID,
            jobs_root=sfs_jobs_root,
        )
    records = [json.loads(line) for line in journal.read_text().splitlines()]
    assert [row["state"] for row in records] == ["KUBECTL_CREATE_INTENT_DO_NOT_RETRY"]


def test_server_create_runtime_drift_is_not_accepted_as_success(tmp_path, sfs_jobs_root):
    class CreateDriftKubectl(FakeKubectl):
        def create_once(self, obj):
            result = super().create_once(obj)
            pod_spec = result["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"]
            pod_spec["imagePullSecrets"].append({"name": "unreviewed"})
            return result

    jobs, kube = FakeJobs(), CreateDriftKubectl()
    journal = tmp_path / "DIRECT_SUBMISSION.jsonl"
    with pytest.raises(JobsError, match="runtime surface"):
        direct_submit_sft_once(
            plan=plan(),
            request=request(),
            jobs=jobs,
            kubectl=kube,
            journal=journal,
            run_id=RUN_ID,
            jobs_root=sfs_jobs_root,
        )
    records = [json.loads(line) for line in journal.read_text().splitlines()]
    assert [row["state"] for row in records] == ["KUBECTL_CREATE_INTENT_DO_NOT_RETRY"]
    assert [call[0] for call in kube.calls].count("create") == 1


def test_persisted_readback_gpu_cluster_drift_is_not_accepted_as_success(tmp_path, sfs_jobs_root):
    class ReadbackDriftKubectl(FakeKubectl):
        def get_rayjob(self, name):
            result = super().get_rayjob(name)
            selector = result["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"][
                "nodeSelector"
            ]
            selector["topology.nebius.com/gpu-cluster-id"] = "computegpucluster-wrong"
            return result

    jobs, kube = FakeJobs(), ReadbackDriftKubectl()
    journal = tmp_path / "DIRECT_SUBMISSION.jsonl"
    with pytest.raises(JobsError, match="runtime surface"):
        direct_submit_sft_once(
            plan=plan(),
            request=request(),
            jobs=jobs,
            kubectl=kube,
            journal=journal,
            run_id=RUN_ID,
            jobs_root=sfs_jobs_root,
        )

    records = [json.loads(line) for line in journal.read_text().splitlines()]
    assert [row["state"] for row in records] == ["KUBECTL_CREATE_INTENT_DO_NOT_RETRY"]
    assert [call[0] for call in kube.calls].count("create") == 1
    assert [call[0] for call in kube.calls].count("get") == 1


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
    kube.get_rayjob("researcher-sft-12345678")
    tokens = [token for command, _ in calls for token in command]
    assert "apply" not in tokens and "patch" not in tokens and "delete" not in tokens
    assert sum(command.count("create") for command, _ in calls) == 2
    assert sum("--dry-run=server" in command for command, _ in calls) == 1
    assert (
        sum("create" in command and "--dry-run=server" not in command for command, _ in calls) == 1
    )
    assert sum("get" in command for command, _ in calls) == 2


def test_output_check_workload_read_is_scoped_to_exact_job_uid(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps({"kind": "List", "items": []}),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", run)
    kube = Kubectl("prod-context")
    kube.list_output_check_workloads(CREATED_UID)
    assert len(calls) == 1
    assert "--selector=kueue.x-k8s.io/job-uid=" + CREATED_UID in calls[0]
    with pytest.raises(JobsError, match="invalid output-check Job UID"):
        kube.list_output_check_workloads("not-a-uid")
    assert len(calls) == 1


def test_output_check_service_account_read_is_exact_and_read_only(monkeypatch):
    calls = []
    response = {
        "apiVersion": "v1",
        "kind": "ServiceAccount",
        "metadata": {"name": "default", "namespace": "fleet-train-jobs"},
        "imagePullSecrets": [{"name": "ecr-pull"}],
    }

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(response),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", run)
    assert Kubectl("prod-context").get_output_check_service_account() == response
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[-6:] == [
        "get",
        "serviceaccount",
        "default",
        "--namespace",
        "fleet-train-jobs",
        "--output=json",
    ]
    assert kwargs.get("input") is None
    assert not any(token in command for token in ("create", "apply", "patch", "delete"))


def test_cpu_checkpoint_boundary_previews_and_creates_only_unpinned_zero_gpu_pod(monkeypatch):
    calls = []
    expected = cpu_checkpoint_pod()

    def run(command, **kwargs):
        payload = kwargs.get("input")
        calls.append((command, json.loads(payload) if payload is not None else None))
        output = cpu_node_inventory() if "get" in command else expected
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(output), stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    kube = Kubectl("prod-context")
    kube.dry_run_cpu_checkpoint_pod(expected)
    kube.create_cpu_checkpoint_pod_once(expected)
    assert len(calls) == 4
    assert "get" in calls[0][0]
    assert "--dry-run=server" in calls[1][0]
    assert "get" in calls[2][0]
    assert "--dry-run=server" not in calls[3][0]
    manifests = [payload for _, payload in calls if payload is not None]
    assert all(payload["spec"]["nodeSelector"] == CPU_NODE_SELECTOR for payload in manifests)


def test_cpu_checkpoint_boundary_accepts_truthful_export_operation(monkeypatch):
    calls = []
    expected = cpu_checkpoint_pod()
    expected["metadata"]["annotations"][CPU_CHECKPOINT_OPERATION_ANNOTATION] = "export"

    def run(command, **kwargs):
        calls.append((command, kwargs.get("input")))
        output = cpu_node_inventory() if "get" in command else expected
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(output), stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    Kubectl("prod-context").dry_run_cpu_checkpoint_pod(expected)
    assert len(calls) == 2
    assert "get" in calls[0][0]
    assert "--dry-run=server" in calls[1][0]


@pytest.mark.parametrize(
    ("resource", "quantity"),
    [("cpu", "16"), ("memory", "128Gi")],
)
def test_cpu_checkpoint_create_rejects_request_that_cannot_fit_observed_node(
    monkeypatch, resource, quantity
):
    calls = []
    pod = cpu_checkpoint_pod()
    pod["spec"]["containers"][0]["resources"]["requests"][resource] = quantity

    def run(command, **kwargs):
        calls.append((command, kwargs.get("input")))
        if "get" not in command:
            raise AssertionError("an unschedulable CPU checkpoint Pod reached create")
        return subprocess.CompletedProcess(
            command, 0, stdout=json.dumps(cpu_node_inventory()), stderr=""
        )

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(JobsError, match="cannot fit any observed eligible node"):
        Kubectl("prod-context").create_cpu_checkpoint_pod_once(pod)
    assert len(calls) == 1
    assert "get" in calls[0][0]


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


def test_cpu_sfs_control_rejects_the_old_naked_pod_submission_path(monkeypatch):
    calls = []

    def run(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("unpackaged CPU preflight reached kubectl")

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(JobsError, match="source and SFS annotations must be complete"):
        Kubectl("prod-context").create_cpu_checkpoint_pod_once(
            cpu_sfs_control_pod(source_bound=False)
        )
    with pytest.raises(JobsError, match="exact packaged submission path"):
        Kubectl("prod-context").create_cpu_checkpoint_pod_once(cpu_sfs_control_pod())
    assert calls == []


def test_lora_cpu_preflight_package_binds_exact_driver_bytes_before_dry_run_and_create(
    tmp_path, monkeypatch
):
    archive = tmp_path / "source.tgz"
    source_archive(archive)
    package = build_lora_cpu_preflight_package(
        cpu_sfs_control_pod(source_bound=False),
        source_archive=archive,
        runtime_source_archive="/mnt/sfs/reviewed/source.tgz",
        source_commit=SOURCE_COMMIT,
    )
    assert package.config_map["immutable"] is True
    assert package.config_map["data"]["preflight_driver.py"] == (
        Path(lora_cpu_preflight_driver.__file__).read_text()
    )
    calls = []

    def run(command, **kwargs):
        payload = kwargs.get("input")
        obj = json.loads(payload) if payload is not None else None
        calls.append((command, obj))
        output = cpu_node_inventory() if "get" in command else obj
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(output), stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    kube = Kubectl("prod-context")
    preview = kube.dry_run_lora_cpu_preflight(package)
    created = kube.create_lora_cpu_preflight_once(package)

    assert preview["proof"] == created["proof"]
    assert [obj and obj["kind"] for _, obj in calls] == [
        None,
        "ConfigMap",
        "Pod",
        None,
        "ConfigMap",
        "Pod",
    ]
    assert all(call[1]["data"] == package.config_map["data"] for call in (calls[1], calls[4]))
    assert all(
        payload["metadata"]["annotations"]["fleet.ai/failure-alerts"] == "off"
        for payload in (calls[2][1], calls[5][1])
    )


def test_lora_cpu_preflight_driver_invokes_p3_and_p4_guards_from_packaged_bytes(
    tmp_path, monkeypatch
):
    archive = tmp_path / "source.tgz"
    source_archive(archive)
    package = build_lora_cpu_preflight_package(
        cpu_sfs_control_pod(source_bound=False),
        source_archive=archive,
        runtime_source_archive="/mnt/sfs/reviewed/source.tgz",
        source_commit=SOURCE_COMMIT,
    )
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    for name, content in package.config_map["data"].items():
        (bundle_root / name).write_text(content)
    monkeypatch.syspath_prepend(str(bundle_root))
    spec = importlib.util.spec_from_file_location(
        "packaged_preflight_driver", bundle_root / "preflight_driver.py"
    )
    assert spec is not None and spec.loader is not None
    packaged_driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(packaged_driver)
    environment = {
        entry["name"]: entry["value"] for entry in package.pod["spec"]["containers"][0]["env"]
    }
    calls = []

    def verify_archive(path, commit):
        calls.append(("source", str(path), commit))
        return {
            "source_commit": commit,
            "source_commit_file_sha256": "d" * 64,
            "source_archive_sha256": environment[ENV_SOURCE_ARCHIVE_SHA256],
        }

    def verify_output(owned, output, *, writable_mount):
        calls.append(("output", str(owned), str(output), str(writable_mount)))
        return {"owned_root": str(owned), "output_root": str(output), "uid": 1000, "gid": 100}

    monkeypatch.setattr(packaged_driver, "verify_source_archive_commit", verify_archive)
    monkeypatch.setattr(packaged_driver, "verify_owned_output_runtime", verify_output)
    receipt = packaged_driver.run_preflight(environment, bundle_root=bundle_root)

    assert [call[0] for call in calls] == ["source", "output"]
    assert calls[0][1:] == ("/mnt/sfs/reviewed/source.tgz", SOURCE_COMMIT)
    assert calls[1][1:] == (
        "/mnt/sfs/jobs/chris-q38-study-corpora-v1/launch-controls",
        "/mnt/sfs/jobs/chris-q38-study-corpora-v1/launch-controls/.preflight-control-step60-a21cbe7c",
        "/controls",
    )
    assert receipt["preflight_bundle_sha256"] == environment[ENV_BUNDLE_SHA256]


@pytest.mark.parametrize("drift", ["driver", "archive"])
def test_lora_cpu_preflight_package_drift_fails_before_kubectl(tmp_path, monkeypatch, drift):
    archive = tmp_path / "source.tgz"
    source_archive(archive)
    package = build_lora_cpu_preflight_package(
        cpu_sfs_control_pod(source_bound=False),
        source_archive=archive,
        runtime_source_archive="/mnt/sfs/reviewed/source.tgz",
        source_commit=SOURCE_COMMIT,
    )
    if drift == "driver":
        changed = deepcopy(package.config_map)
        changed["data"]["preflight_driver.py"] += "\n# drift\n"
        package = replace(package, config_map=changed)
    else:
        archive.write_bytes(b"not the reviewed archive")
    calls = []

    def run(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("drifted preflight package reached kubectl")

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(JobsError):
        Kubectl("prod-context").dry_run_lora_cpu_preflight(package)
    assert calls == []


@pytest.mark.parametrize(
    "fault",
    [
        "top-level-output",
        "different-parent",
        "root-user",
        "wrong-fsgroup",
        "initializer",
        "low-memory",
        "wrong-memory-floor",
        "secret-env",
        "secret-volume",
        "wrong-pvc",
        "writable-global-sfs",
        "wrong-control-subpath",
        "extra-image-pull-secret",
        "capability",
        "wrong-image",
        "extra-volume",
    ],
)
def test_cpu_sfs_control_drift_fails_before_kubectl(monkeypatch, fault):
    calls = []
    pod = cpu_sfs_control_pod()
    spec = pod["spec"]
    container = spec["containers"][0]
    annotations = pod["metadata"]["annotations"]
    if fault == "top-level-output":
        value = "/mnt/sfs/jobs/.preflight-control-step60-a21cbe7c"
        annotations[CPU_SFS_OUTPUT_ROOT_ANNOTATION] = value
        container["env"][1]["value"] = value
    elif fault == "different-parent":
        value = "/mnt/sfs/jobs/chris-other-run/.preflight-control-step60-a21cbe7c"
        annotations[CPU_SFS_OUTPUT_ROOT_ANNOTATION] = value
        container["env"][1]["value"] = value
    elif fault == "root-user":
        spec["securityContext"]["runAsUser"] = 0
    elif fault == "wrong-fsgroup":
        spec["securityContext"]["fsGroup"] = 200
    elif fault == "initializer":
        spec["initContainers"] = [deepcopy(container)]
    elif fault == "low-memory":
        container["resources"]["requests"]["memory"] = "16Gi"
    elif fault == "wrong-memory-floor":
        annotations[CPU_SFS_MEMORY_FLOOR_MIB_ANNOTATION] = "not-a-number"
    elif fault == "secret-env":
        container["envFrom"] = [{"secretRef": {"name": "credential"}}]
    elif fault == "secret-volume":
        spec["volumes"].append({"name": "credential", "secret": {"secretName": "key"}})
    elif fault == "wrong-pvc":
        spec["volumes"][1]["persistentVolumeClaim"]["claimName"] = "other"
    elif fault == "writable-global-sfs":
        container["volumeMounts"][1]["readOnly"] = False
    elif fault == "wrong-control-subpath":
        container["volumeMounts"][2]["subPath"] = "jobs/chris-q38-study-corpora-v1"
    elif fault == "extra-image-pull-secret":
        spec["imagePullSecrets"].append({"name": "other"})
    elif fault == "wrong-image":
        container["image"] = LORA_TRAINER_IMAGE.replace("7da4", "8da4", 1)
    elif fault == "extra-volume":
        spec["volumes"].append({"name": "host", "hostPath": {"path": "/tmp"}})
    else:
        container["securityContext"]["capabilities"] = {"drop": ["ALL"], "add": ["CHOWN"]}

    def run(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("unsafe CPU SFS control reached kubectl")

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(JobsError):
        Kubectl("prod-context").create_cpu_checkpoint_pod_once(pod)
    assert calls == []


def test_invalid_uuid_or_context_fails_locally():
    with pytest.raises(JobsError, match="UUIDv4"):
        render_sft_rayjob(
            plan(),
            request(),
            preview(),
            kubernetes_context=SFT_PRODUCTION_CONTEXT,
            run_id="not-a-uuid",
        )
    for value in ("", "--current", "spaces are unsafe"):
        with pytest.raises(JobsError):
            Kubectl(value)
