import base64
import json
from copy import deepcopy
from pathlib import Path

import pytest

from cyber_post_train.jobs import digest
from cyber_post_train.sfs_output_driver import (
    ENV_DRIVER_SHA256,
    ENV_DRIVER_SOURCE,
    observe,
)
from cyber_post_train.sfs_output_job import (
    CPU_NODE_SELECTOR,
    DRIVER_ANNOTATION,
    build_sfs_output_job,
    collect_sfs_output_receipt,
    validate_sfs_output_job_node_fit,
    validate_sfs_output_job_response,
)

ROOT = Path(__file__).parents[1]
LIVE_DRIFT_EVIDENCE = (
    ROOT / "docs/evidence/qwen38-sft-sfs-observer-kueue-v1beta2-drift-20260921.json"
)


def plan():
    return {"schema": "cyber_sft_runtime_dense_v1", "immutable": "synthetic"}


def request():
    return {
        "name": "researcher-sft",
        "title": "Synthetic SFT fixture",
        "image": "registry/image@sha256:" + "a" * 64,
        "command": "python train.py",
        "workers": 1,
        "gpus_per_worker": 8,
        "run_dir": "/mnt/sfs/jobs/researcher-sft",
        "priority_class": "c1",
        "requeueIfPreempted": False,
        "failureAlerts": False,
        "resources": {
            "cpu_request": "8",
            "cpu_limit": "16",
            "memory_request": "64Gi",
            "memory_limit": "128Gi",
        },
        "env": {},
        "secrets": ["wandb-api"],
    }


def node_inventory(*, memory="65216572Ki"):
    return {
        "kind": "List",
        "items": [
            {
                "metadata": {"name": "cpu-1", "labels": deepcopy(CPU_NODE_SELECTOR)},
                "spec": {},
                "status": {
                    "allocatable": {"cpu": "15900m", "memory": memory},
                    "conditions": [{"type": "Ready", "status": "True"}],
                },
            }
        ],
    }


def service_account():
    return {
        "apiVersion": "v1",
        "kind": "ServiceAccount",
        "metadata": {"name": "default", "namespace": "fleet-train-jobs"},
        "imagePullSecrets": [{"name": "ecr-pull"}],
    }


def completed_objects(package, receipt):
    job = deepcopy(package.job)
    job["metadata"]["uid"] = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
    workload_name = "job-" + package.job["metadata"]["name"] + "-abcde"
    batch_labels = {
        "batch.kubernetes.io/controller-uid": job["metadata"]["uid"],
        "batch.kubernetes.io/job-name": job["metadata"]["name"],
        "controller-uid": job["metadata"]["uid"],
        "job-name": job["metadata"]["name"],
    }
    kueue_labels = {
        "kueue.x-k8s.io/cluster-queue-name": "training-cq",
        "kueue.x-k8s.io/local-queue-name": "training-lq",
        "kueue.x-k8s.io/podset": "main",
    }
    job["spec"]["selector"] = {
        "matchLabels": {
            "batch.kubernetes.io/controller-uid": job["metadata"]["uid"],
        }
    }
    live_template = job["spec"]["template"]
    live_template["metadata"]["annotations"]["kueue.x-k8s.io/workload"] = workload_name
    live_template["metadata"]["labels"].update({**batch_labels, **kueue_labels})
    job["status"] = {"conditions": [{"type": "Complete", "status": "True"}]}
    pod = {
        "metadata": {
            "name": package.job["metadata"]["name"] + "-abcde",
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
                    "uid": job["metadata"]["uid"],
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
    pod["spec"]["imagePullSecrets"] = [{"name": "ecr-pull"}]
    workload = {
        "apiVersion": "kueue.x-k8s.io/v1beta2",
        "kind": "Workload",
        "metadata": {
            "name": workload_name,
            "namespace": "fleet-train-jobs",
            "uid": "99999999-2222-4333-8444-555555555555",
            "labels": {"kueue.x-k8s.io/job-uid": job["metadata"]["uid"]},
            "ownerReferences": [
                {
                    "apiVersion": "batch/v1",
                    "kind": "Job",
                    "name": package.job["metadata"]["name"],
                    "uid": job["metadata"]["uid"],
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
                                "batch.kubernetes.io/job-name": job["metadata"]["name"],
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
    logs = "CYBER_SFT_OUTPUT_ABSENCE=" + json.dumps(receipt, sort_keys=True, separators=(",", ":"))
    return (
        job,
        {"kind": "List", "items": [workload]},
        {"kind": "List", "items": [pod]},
        logs,
    )


def test_job_is_exact_root_alert_off_c1_q1_zero_gpu_read_only_package():
    request_value = {**request(), "image_pull_secrets": ["ghcr-pull"]}
    package = build_sfs_output_job(plan(), request_value, 1)
    job = package.job
    assert job["metadata"]["name"] == "researcher-sft-sfs-a01"
    assert job["metadata"]["annotations"]["fleet.ai/failure-alerts"] == "off"
    assert job["metadata"]["labels"]["kueue.x-k8s.io/priority-class"] == "q1"
    spec = job["spec"]["template"]["spec"]
    assert spec["priorityClassName"] == "c1"
    assert spec["priority"] == 10_000
    assert package.job["spec"]["suspend"] is True
    assert spec["imagePullSecrets"] == [{"name": "ghcr-pull"}]
    assert spec["automountServiceAccountToken"] is False
    assert spec["securityContext"]["runAsUser"] == 1000
    assert spec["containers"][0]["volumeMounts"][0]["readOnly"] is True
    assert spec["volumes"][0]["persistentVolumeClaim"]["readOnly"] is True
    assert "nvidia.com/gpu" not in str(spec["containers"][0]["resources"])
    environment = {entry["name"]: entry["value"] for entry in spec["containers"][0]["env"]}
    source = spec["containers"][0]["command"][-1].encode()
    assert base64.b64decode(environment[ENV_DRIVER_SOURCE], validate=True) == source
    assert environment[ENV_DRIVER_SHA256] == job["metadata"]["annotations"][DRIVER_ANNOTATION]


@pytest.mark.parametrize(
    "fault",
    [
        "alert",
        "root-priority",
        "pod-priority",
        "gpu",
        "writable",
        "command",
        "pull-secret",
        "env-from-secret",
        "privileged",
        "init-secret",
        "init-gpu",
        "node-name",
        "affinity",
        "overhead",
        "runtime-class",
        "missing-effective-priority",
        "wrong-effective-priority",
    ],
)
def test_server_drift_is_rejected(fault):
    package = build_sfs_output_job(plan(), request(), 1)
    actual = deepcopy(package.job)
    if fault == "alert":
        actual["metadata"]["annotations"]["fleet.ai/failure-alerts"] = "on"
    elif fault == "root-priority":
        actual["metadata"]["labels"]["kueue.x-k8s.io/priority-class"] = "q0"
    elif fault == "pod-priority":
        actual["spec"]["template"]["spec"]["priorityClassName"] = "c0"
    elif fault == "gpu":
        actual["spec"]["template"]["spec"]["containers"][0]["resources"]["limits"][
            "nvidia.com/gpu"
        ] = 1
    elif fault == "writable":
        actual["spec"]["template"]["spec"]["containers"][0]["volumeMounts"][0]["readOnly"] = False
    elif fault == "command":
        actual["spec"]["template"]["spec"]["containers"][0]["command"][-1] += "\n# drift"
    elif fault == "pull-secret":
        actual["spec"]["template"]["spec"]["imagePullSecrets"].append({"name": "unreviewed"})
    elif fault == "env-from-secret":
        actual["spec"]["template"]["spec"]["containers"][0]["envFrom"] = [
            {"secretRef": {"name": "unreviewed"}}
        ]
    elif fault == "privileged":
        actual["spec"]["template"]["spec"]["containers"][0]["securityContext"]["privileged"] = True
    elif fault in {"init-secret", "init-gpu"}:
        resources = {"requests": {"cpu": "1"}, "limits": {"cpu": "1"}}
        init = {
            "name": "injected",
            "image": request()["image"],
            "resources": resources,
        }
        if fault == "init-secret":
            init["envFrom"] = [{"secretRef": {"name": "unreviewed"}}]
        else:
            resources["requests"]["nvidia.com/gpu"] = "1"
            resources["limits"]["nvidia.com/gpu"] = "1"
            init["restartPolicy"] = "Always"
        actual["spec"]["template"]["spec"]["initContainers"] = [init]
    elif fault == "node-name":
        actual["spec"]["template"]["spec"]["nodeName"] = "forced-node"
    elif fault == "affinity":
        actual["spec"]["template"]["spec"]["affinity"] = {
            "nodeAffinity": {"requiredDuringSchedulingIgnoredDuringExecution": {}}
        }
    elif fault == "overhead":
        actual["spec"]["template"]["spec"]["overhead"] = {"nvidia.com/gpu": "1"}
    elif fault == "runtime-class":
        actual["spec"]["template"]["spec"]["runtimeClassName"] = "unreviewed"
    elif fault == "missing-effective-priority":
        del actual["spec"]["template"]["spec"]["priority"]
    else:
        actual["spec"]["template"]["spec"]["priority"] = 0
    with pytest.raises(ValueError):
        validate_sfs_output_job_response(actual, package, require_uid=False)


def test_enumerated_kubernetes_job_defaults_are_accepted_without_behavior_drift():
    package = build_sfs_output_job(plan(), request(), 1)
    actual = deepcopy(package.job)
    controller_uid = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
    actual["metadata"].update({"creationTimestamp": None, "generation": 1, "uid": controller_uid})
    actual["spec"].update(
        {
            "manualSelector": False,
            "podReplacementPolicy": "TerminatingOrFailed",
            "selector": {"matchLabels": {"batch.kubernetes.io/controller-uid": controller_uid}},
        }
    )
    actual["spec"]["template"]["metadata"]["labels"]["batch.kubernetes.io/controller-uid"] = (
        controller_uid
    )
    actual["spec"]["template"]["metadata"]["labels"].update(
        {
            "batch.kubernetes.io/job-name": package.job["metadata"]["name"],
            "controller-uid": controller_uid,
            "job-name": package.job["metadata"]["name"],
        }
    )
    actual["spec"]["template"]["metadata"]["creationTimestamp"] = None
    pod_spec = actual["spec"]["template"]["spec"]
    pod_spec.update(
        {
            "dnsPolicy": "ClusterFirst",
            "enableServiceLinks": True,
            "preemptionPolicy": "PreemptLowerPriority",
            "priority": 10_000,
            "schedulerName": "default-scheduler",
            "serviceAccount": "default",
            "serviceAccountName": "default",
        }
    )
    pod_spec["containers"][0].update(
        {
            "terminationMessagePath": "/dev/termination-log",
            "terminationMessagePolicy": "File",
        }
    )
    validate_sfs_output_job_response(actual, package, require_uid=False)


@pytest.mark.parametrize("pull_secrets", [[], ["ghcr-pull"]])
def test_live_api_server_serialization_defaults_are_canonicalized_exactly(pull_secrets):
    request_value = {**request(), "image_pull_secrets": pull_secrets}
    package = build_sfs_output_job(plan(), request_value, 1)
    actual = deepcopy(package.job)
    actual["status"] = {}
    pod_spec = actual["spec"]["template"]["spec"]
    for key in ("hostIPC", "hostNetwork", "hostPID"):
        del pod_spec[key]
    if not pull_secrets:
        del pod_spec["imagePullSecrets"]
    environment = pod_spec["containers"][0]["env"]
    cuda_visible = next(entry for entry in environment if entry["name"] == "CUDA_VISIBLE_DEVICES")
    del cuda_visible["value"]
    validate_sfs_output_job_response(actual, package, require_uid=False)


def test_node_fit_requires_one_ready_matching_cpu_node():
    package = build_sfs_output_job(plan(), request(), 1)
    assert validate_sfs_output_job_node_fit(package, node_inventory())["fitting_nodes"] == 1
    inventory = node_inventory()
    inventory["items"][0]["status"]["conditions"][0]["status"] = "False"
    with pytest.raises(ValueError, match="cannot fit"):
        validate_sfs_output_job_node_fit(package, inventory)


def test_driver_observation_is_source_bound_non_root_and_symlink_safe(tmp_path, monkeypatch):
    package = build_sfs_output_job(plan(), request(), 1)
    container = package.job["spec"]["template"]["spec"]["containers"][0]
    environment = {entry["name"]: entry["value"] for entry in container["env"]}
    observed = tmp_path / "output"
    monkeypatch.setattr("cyber_post_train.sfs_output_driver.exact_run_dir", lambda *_: observed)
    monkeypatch.setattr("cyber_post_train.sfs_output_driver.os.getuid", lambda: 1000)
    monkeypatch.setattr("cyber_post_train.sfs_output_driver.os.getgid", lambda: 100)
    receipt = observe(environment)
    assert receipt["plan_sha256"] == digest(plan())
    assert receipt["request_sha256"] == digest(request())
    observed.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        observe(environment)


def test_terminal_collection_binds_job_pod_image_logs_and_fresh_receipt():
    package = build_sfs_output_job(plan(), request(), 1)
    unsigned = {
        "schema": "cyber_sft_output_absence_v1",
        "status": "passed",
        "checked_at_epoch": 1000,
        "plan_sha256": digest(plan()),
        "request_sha256": digest(request()),
        "run_name": request()["name"],
        "run_dir": request()["run_dir"],
        "sfs_jobs_root": "/mnt/sfs/jobs",
        "output_absent": True,
    }
    receipt = {**unsigned, "sha256": digest(unsigned)}
    job, workloads, pods, logs = completed_objects(package, receipt)
    job["spec"]["suspend"] = False
    assert (
        collect_sfs_output_receipt(
            package,
            job,
            workloads,
            pods,
            service_account(),
            logs,
            now=1001,
        )
        == receipt
    )
    pods["items"][0]["status"]["containerStatuses"][0]["restartCount"] = 1
    with pytest.raises(ValueError, match="restarted"):
        collect_sfs_output_receipt(
            package,
            job,
            workloads,
            pods,
            service_account(),
            logs,
            now=1001,
        )


def test_live_kueue_v1beta2_drift_evidence_is_sanitized_and_self_digesting():
    evidence = json.loads(LIVE_DRIFT_EVIDENCE.read_text())
    assert evidence["sha256"] == digest(
        {key: value for key, value in evidence.items() if key != "sha256"}
    )
    assert evidence["collector_result"] == {
        "accepted": False,
        "reason": "unreviewed_kueue_v1beta2_workload_fields",
        "log_read_attempted": False,
        "output_absence_receipt_created": False,
        "training_rayjob_created": False,
    }
    assert evidence["live_shape"] == {
        "workload_spec_priority_class_ref": {
            "group": "kueue.x-k8s.io",
            "kind": "WorkloadPriorityClass",
            "name": "q1",
        },
        "pod_set_topology_request": {"podIndexLabel": "batch.kubernetes.io/job-completion-index"},
        "job_template_image_pull_secrets": [],
        "workload_pod_set_image_pull_secrets": [],
        "scheduled_pod_service_account": "default",
        "default_service_account_image_pull_secrets": ["ecr-pull"],
        "scheduled_pod_image_pull_secrets": ["ecr-pull"],
        "crd_storage_version": "v1beta2",
        "crd_priority_class_ref_required_fields": ["group", "kind", "name"],
    }
    assert evidence["privacy"] == {
        "credentials_present": False,
        "prompts_present": False,
        "private_logs_present": False,
        "scores_present": False,
    }


@pytest.mark.parametrize(
    "pull_secrets",
    [
        [{"name": "other"}],
        [{"name": "ecr-pull"}, {"name": "other"}],
        [{"name": "ecr-pull", "unreviewed": True}],
    ],
)
def test_terminal_pod_rejects_every_nonexact_service_account_pull_secret(pull_secrets):
    package = build_sfs_output_job(plan(), request(), 1)
    unsigned = {
        "schema": "cyber_sft_output_absence_v1",
        "status": "passed",
        "checked_at_epoch": 1000,
        "plan_sha256": digest(plan()),
        "request_sha256": digest(request()),
        "run_name": request()["name"],
        "run_dir": request()["run_dir"],
        "sfs_jobs_root": "/mnt/sfs/jobs",
        "output_absent": True,
    }
    receipt = {**unsigned, "sha256": digest(unsigned)}
    job, workloads, pods, logs = completed_objects(package, receipt)
    job["spec"]["suspend"] = False
    pods["items"][0]["spec"]["imagePullSecrets"] = pull_secrets
    with pytest.raises(ValueError, match="imagePullSecrets drifted"):
        collect_sfs_output_receipt(
            package,
            job,
            workloads,
            pods,
            service_account(),
            logs,
            now=1001,
        )


@pytest.mark.parametrize("fault", ["secret-env", "secret-volume"])
def test_terminal_pod_rejects_service_account_unrelated_secret_surfaces(fault):
    package = build_sfs_output_job(plan(), request(), 1)
    unsigned = {
        "schema": "cyber_sft_output_absence_v1",
        "status": "passed",
        "checked_at_epoch": 1000,
        "plan_sha256": digest(plan()),
        "request_sha256": digest(request()),
        "run_name": request()["name"],
        "run_dir": request()["run_dir"],
        "sfs_jobs_root": "/mnt/sfs/jobs",
        "output_absent": True,
    }
    receipt = {**unsigned, "sha256": digest(unsigned)}
    job, workloads, pods, logs = completed_objects(package, receipt)
    job["spec"]["suspend"] = False
    pod_spec = pods["items"][0]["spec"]
    if fault == "secret-env":
        pod_spec["containers"][0]["envFrom"] = [{"secretRef": {"name": "unreviewed-secret"}}]
    else:
        pod_spec["volumes"].append(
            {"name": "unreviewed-secret", "secret": {"secretName": "unreviewed-secret"}}
        )
    with pytest.raises(ValueError):
        collect_sfs_output_receipt(
            package,
            job,
            workloads,
            pods,
            service_account(),
            logs,
            now=1001,
        )


@pytest.mark.parametrize(
    "fault",
    [
        "api-version",
        "kind",
        "name",
        "namespace",
        "missing-pull-secret",
        "extra-pull-secret",
        "legacy-secret",
        "pod-service-account",
    ],
)
def test_terminal_pod_requires_exact_fresh_default_service_account_binding(fault):
    package = build_sfs_output_job(plan(), request(), 1)
    unsigned = {
        "schema": "cyber_sft_output_absence_v1",
        "status": "passed",
        "checked_at_epoch": 1000,
        "plan_sha256": digest(plan()),
        "request_sha256": digest(request()),
        "run_name": request()["name"],
        "run_dir": request()["run_dir"],
        "sfs_jobs_root": "/mnt/sfs/jobs",
        "output_absent": True,
    }
    receipt = {**unsigned, "sha256": digest(unsigned)}
    job, workloads, pods, logs = completed_objects(package, receipt)
    job["spec"]["suspend"] = False
    account = service_account()
    if fault == "api-version":
        account["apiVersion"] = "v2"
    elif fault == "kind":
        account["kind"] = "Secret"
    elif fault == "name":
        account["metadata"]["name"] = "other"
    elif fault == "namespace":
        account["metadata"]["namespace"] = "other"
    elif fault == "missing-pull-secret":
        account["imagePullSecrets"] = []
    elif fault == "extra-pull-secret":
        account["imagePullSecrets"].append({"name": "other"})
    elif fault == "legacy-secret":
        account["secrets"] = [{"name": "token"}]
    else:
        pods["items"][0]["spec"]["serviceAccountName"] = "other"
    with pytest.raises(ValueError, match="ServiceAccount"):
        collect_sfs_output_receipt(
            package,
            job,
            workloads,
            pods,
            account,
            logs,
            now=1001,
        )


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "multiple",
        "wrong-job-uid",
        "wrong-workload-label",
        "wrong-api-version",
        "wrong-queue",
        "wrong-cluster-queue",
        "wrong-class",
        "wrong-priority",
        "missing-priority-ref",
        "wrong-priority-ref-group",
        "wrong-priority-ref-kind",
        "wrong-priority-ref-name",
        "missing-topology-request",
        "wrong-topology-request",
        "extra-topology-request-field",
        "wrong-podset-count",
        "gpu-resource",
        "missing-admission",
        "not-admitted",
        "not-finished",
        "wrong-finished-reason",
        "evicted",
    ],
)
def test_terminal_collection_requires_exact_live_kueue_admission(fault):
    package = build_sfs_output_job(plan(), request(), 1)
    unsigned = {
        "schema": "cyber_sft_output_absence_v1",
        "status": "passed",
        "checked_at_epoch": 1000,
        "plan_sha256": digest(plan()),
        "request_sha256": digest(request()),
        "run_name": request()["name"],
        "run_dir": request()["run_dir"],
        "sfs_jobs_root": "/mnt/sfs/jobs",
        "output_absent": True,
    }
    receipt = {**unsigned, "sha256": digest(unsigned)}
    job, workloads, pods, logs = completed_objects(package, receipt)
    job["spec"]["suspend"] = False
    workload = workloads["items"][0]
    if fault == "missing":
        workloads["items"] = []
    elif fault == "multiple":
        workloads["items"].append(deepcopy(workload))
    elif fault == "wrong-job-uid":
        workload["metadata"]["ownerReferences"][0]["uid"] = "88888888-2222-4333-8444-555555555555"
    elif fault == "wrong-workload-label":
        workload["metadata"]["labels"]["kueue.x-k8s.io/job-uid"] = (
            "88888888-2222-4333-8444-555555555555"
        )
    elif fault == "wrong-api-version":
        workload["apiVersion"] = "kueue.x-k8s.io/v1beta1"
    elif fault == "wrong-queue":
        workload["spec"]["queueName"] = "other"
    elif fault == "wrong-cluster-queue":
        workload["status"]["admission"]["clusterQueue"] = "other"
    elif fault == "wrong-class":
        workload["spec"]["priorityClassName"] = "q0"
    elif fault == "wrong-priority":
        workload["spec"]["priority"] = 0
    elif fault == "missing-priority-ref":
        workload["spec"].pop("priorityClassRef")
    elif fault == "wrong-priority-ref-group":
        workload["spec"]["priorityClassRef"]["group"] = "wrong.example"
    elif fault == "wrong-priority-ref-kind":
        workload["spec"]["priorityClassRef"]["kind"] = "PriorityClass"
    elif fault == "wrong-priority-ref-name":
        workload["spec"]["priorityClassRef"]["name"] = "q0"
    elif fault == "missing-topology-request":
        workload["spec"]["podSets"][0].pop("topologyRequest")
    elif fault == "wrong-topology-request":
        workload["spec"]["podSets"][0]["topologyRequest"] = {
            "podIndexLabel": "unreviewed.example/index"
        }
    elif fault == "extra-topology-request-field":
        workload["spec"]["podSets"][0]["topologyRequest"]["unconstrained"] = True
    elif fault == "wrong-podset-count":
        workload["spec"]["podSets"][0]["count"] = 2
    elif fault == "gpu-resource":
        workload["status"]["admission"]["podSetAssignments"][0]["resourceUsage"][
            "nvidia.com/gpu"
        ] = "1"
    elif fault == "missing-admission":
        workload["status"]["admission"] = None
    elif fault == "not-admitted":
        workload["status"]["conditions"][0]["status"] = "False"
    elif fault == "not-finished":
        workload["status"]["conditions"][1]["status"] = "False"
    elif fault == "wrong-finished-reason":
        workload["status"]["conditions"][1]["reason"] = "Failed"
    else:
        workload["status"]["conditions"].append({"type": "Evicted", "status": "True"})
    with pytest.raises(ValueError):
        collect_sfs_output_receipt(
            package,
            job,
            workloads,
            pods,
            service_account(),
            logs,
            now=1001,
        )


def test_collection_rejects_stale_receipt_even_after_successful_job():
    package = build_sfs_output_job(plan(), request(), 1)
    unsigned = {
        "schema": "cyber_sft_output_absence_v1",
        "status": "passed",
        "checked_at_epoch": 1000,
        "plan_sha256": digest(plan()),
        "request_sha256": digest(request()),
        "run_name": request()["name"],
        "run_dir": request()["run_dir"],
        "sfs_jobs_root": "/mnt/sfs/jobs",
        "output_absent": True,
    }
    receipt = {**unsigned, "sha256": digest(unsigned)}
    job, workloads, pods, logs = completed_objects(package, receipt)
    job["spec"]["suspend"] = False
    with pytest.raises(ValueError, match="stale"):
        collect_sfs_output_receipt(
            package,
            job,
            workloads,
            pods,
            service_account(),
            logs,
            now=2000,
        )


def test_package_reopens_tracked_driver_before_use(tmp_path, monkeypatch):
    package = build_sfs_output_job(plan(), request(), 1)
    changed = deepcopy(package)
    changed.job["spec"]["template"]["spec"]["containers"][0]["command"][-1] += "\n"
    with pytest.raises(ValueError, match="source-bound renderer"):
        from cyber_post_train.sfs_output_job import validate_sfs_output_job_package

        validate_sfs_output_job_package(changed)
