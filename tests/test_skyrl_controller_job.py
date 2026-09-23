import base64
import copy
import json
import os
from pathlib import Path

import pytest

from cyber_post_train.jobs import JobsError
from cyber_post_train.skyrl_controller_driver import (
    ENV_DRIVER_SHA256,
    ENV_JOB_UID,
    ENV_PACKET,
    ENV_POD_NAME,
    ENV_POD_UID,
    BootstrapError,
    observe,
    validate_receipt,
)
from cyber_post_train.skyrl_controller_job import (
    CONTROLS_MOUNT,
    CREATE_ONCE_ROOT,
    DRIVER_ANNOTATION,
    IMAGE,
    NAME,
    NAMESPACE,
    ControllerJobPackage,
    assert_target_post_disabled,
    build_controller_job,
    collect_controller_terminal_from_api,
    collect_controller_terminal_receipt,
    controller_packet,
    create_controller_job_once,
    validate_controller_job_package,
    validate_controller_job_response,
)
from cyber_post_train.skyrl_controller_job import (
    ENV_DRIVER_SOURCE as JOB_ENV_DRIVER_SOURCE,
)
from training.skyrl_prod9_hardening import CREATE_ONCE_ROOT as PROD9_CREATE_ONCE_ROOT
from training.skyrl_training import IMAGE as TRAINING_IMAGE

JOB_UID = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
POD_UID = "11111111-2222-4333-8444-555555555555"


def packet():
    return controller_packet(
        source_commit="a" * 40,
        identity_sha256="sha256:" + "1" * 64,
        plan_sha256="sha256:" + "2" * 64,
        request_sha256="sha256:" + "3" * 64,
        target_manifest_sha256="sha256:" + "4" * 64,
    )


def server_object(package, *, uid=JOB_UID):
    value = copy.deepcopy(package.job)
    metadata = value["metadata"]
    metadata.update(
        {
            "uid": uid,
            "creationTimestamp": "2026-09-23T00:00:00Z",
            "generation": 1,
            "resourceVersion": "123",
        }
    )
    generated = {
        "batch.kubernetes.io/controller-uid": uid,
        "batch.kubernetes.io/job-name": NAME,
        "controller-uid": uid,
        "job-name": NAME,
    }
    metadata["labels"].update(generated)
    spec = value["spec"]
    spec["manualSelector"] = False
    spec["podReplacementPolicy"] = "TerminatingOrFailed"
    spec["selector"] = {
        "matchLabels": {"batch.kubernetes.io/controller-uid": uid},
    }
    template = spec["template"]
    template["metadata"]["creationTimestamp"] = None
    template["metadata"]["labels"].update(generated)
    pod = template["spec"]
    pod.update(
        {
            "dnsPolicy": "ClusterFirst",
            "enableServiceLinks": True,
            "preemptionPolicy": "PreemptLowerPriority",
            "schedulerName": "default-scheduler",
            "serviceAccount": "default",
            "serviceAccountName": "default",
        }
    )
    value["status"] = {}
    return value


def literal_environment(package):
    entries = package.job["spec"]["template"]["spec"]["containers"][0]["env"]
    return {entry["name"]: entry["value"] for entry in entries if "value" in entry}


def runtime_environment(package):
    return literal_environment(package)


def runtime_identity():
    return {
        ENV_JOB_UID: JOB_UID,
        ENV_POD_NAME: NAME + "-abcde",
        ENV_POD_UID: POD_UID,
    }


def test_packet_is_closed_and_names_exact_target_receipts():
    value = packet()
    assert value["status"] == "prepared_not_authorized"
    assert value["identity_sha256"] == "sha256:" + "1" * 64
    assert value["target"] == {
        "name": "chris-q38-rlreward-prod10",
        "nodes": 1,
        "gpus": 8,
        "priority": "c1",
        "queue_priority": "q1",
        "failure_alerts": "off",
    }
    assert value["required_target_receipts"] == {
        "creator_exact_run_id_and_rayjob_uid": True,
        "observer_exact_uid_release": True,
        "terminal_reward_update_checkpoint": True,
    }
    assert set(value["platform_gates"].values()) == {False}
    assert value["launch_authorized"] is False
    assert value["submitted"] is False
    with pytest.raises(ValueError, match="target POST is disabled"):
        assert_target_post_disabled(value)


def test_job_is_exact_alert_off_c1_q1_uid_gid_sfs_and_zero_gpu():
    package = build_controller_job(packet())
    proof = validate_controller_job_package(package)
    job = package.job
    assert IMAGE == TRAINING_IMAGE
    assert str(PROD9_CREATE_ONCE_ROOT) == CREATE_ONCE_ROOT
    assert job["metadata"]["annotations"]["fleet.ai/failure-alerts"] == "off"
    assert job["spec"]["template"]["metadata"]["annotations"]["fleet.ai/failure-alerts"] == "off"
    assert job["metadata"]["labels"]["kueue.x-k8s.io/queue-name"] == "training-lq"
    assert job["metadata"]["labels"]["kueue.x-k8s.io/priority-class"] == "q1"
    assert job["spec"]["backoffLimit"] == 0
    assert job["spec"]["suspend"] is True
    pod = job["spec"]["template"]["spec"]
    assert pod["priorityClassName"] == "c1"
    assert pod["priority"] == 10_000
    assert pod["automountServiceAccountToken"] is False
    assert "serviceAccountName" not in pod
    assert pod["securityContext"]["runAsUser"] == 1000
    assert pod["securityContext"]["runAsGroup"] == 100
    assert pod["securityContext"]["fsGroup"] == 100
    container = pod["containers"][0]
    assert container["securityContext"]["runAsUser"] == 1000
    assert container["securityContext"]["runAsGroup"] == 100
    assert container["image"] == IMAGE
    assert "nvidia.com/gpu" not in json.dumps(container["resources"])
    assert not any("secret" in volume for volume in pod["volumes"])
    assert all(set(entry) == {"name", "value"} for entry in container["env"])
    identity_volume = next(volume for volume in pod["volumes"] if volume["name"] == "identity")
    assert set(identity_volume) == {"name", "downwardAPI"}
    mounts = {row["mountPath"]: row for row in container["volumeMounts"]}
    assert mounts["/mnt/sfs"]["readOnly"] is True
    assert mounts[CONTROLS_MOUNT]["readOnly"] is False
    assert mounts[CONTROLS_MOUNT]["subPath"] == ("jobs/chris-q38-study-corpora-v1/launch-controls")
    volumes = {row["name"]: row for row in pod["volumes"]}
    assert volumes["sfs"]["persistentVolumeClaim"]["readOnly"] is True
    assert set(volumes["controls"]) == {"name", "persistentVolumeClaim"}
    source = container["command"][-1].encode()
    environment = literal_environment(package)
    assert base64.b64decode(environment[JOB_ENV_DRIVER_SOURCE], validate=True) == source
    assert environment[ENV_DRIVER_SHA256] == job["metadata"]["annotations"][DRIVER_ANNOTATION]
    assert proof["gpus"] == 0
    assert proof["submitted"] is False


@pytest.mark.parametrize(
    "fault",
    [
        "root-alert",
        "pod-alert",
        "root-priority",
        "pod-priority",
        "gpu",
        "sidecar",
        "init",
        "service-account",
        "secret-volume",
        "broad-write",
        "runtime-user",
        "driver",
    ],
)
def test_package_drift_is_rejected(fault):
    package = build_controller_job(packet())
    job = copy.deepcopy(package.job)
    pod = job["spec"]["template"]["spec"]
    container = pod["containers"][0]
    if fault == "root-alert":
        job["metadata"]["annotations"]["fleet.ai/failure-alerts"] = "on"
    elif fault == "pod-alert":
        job["spec"]["template"]["metadata"]["annotations"]["fleet.ai/failure-alerts"] = "on"
    elif fault == "root-priority":
        job["metadata"]["labels"]["kueue.x-k8s.io/priority-class"] = "q0"
    elif fault == "pod-priority":
        pod["priorityClassName"] = "c0"
    elif fault == "gpu":
        container["resources"]["limits"]["nvidia.com/gpu"] = "1"
    elif fault == "sidecar":
        pod["containers"].append(copy.deepcopy(container))
    elif fault == "init":
        pod["initContainers"] = [copy.deepcopy(container)]
    elif fault == "service-account":
        pod["serviceAccountName"] = "default"
    elif fault == "secret-volume":
        pod["volumes"].append({"name": "token", "secret": {"secretName": "fleet-api"}})
    elif fault == "broad-write":
        container["volumeMounts"][0]["readOnly"] = False
    elif fault == "runtime-user":
        pod["securityContext"]["runAsUser"] = 0
    elif fault == "driver":
        container["command"][-1] += "\n"
    drifted = ControllerJobPackage(packet=package.packet, job=job)
    with pytest.raises(ValueError, match="differs"):
        validate_controller_job_package(drifted)


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("alert", "differs|alert"),
        ("extra", "unreviewed"),
        ("selector", "selector"),
        ("service-account", "default changed"),
        ("status", "live status"),
        ("gpu", "differs|zero-GPU"),
    ],
)
def test_server_preview_drift_is_rejected(fault, message):
    package = build_controller_job(packet())
    actual = server_object(package)
    if fault == "alert":
        actual["metadata"]["annotations"]["fleet.ai/failure-alerts"] = "on"
    elif fault == "extra":
        actual["surprise"] = True
    elif fault == "selector":
        actual["spec"]["selector"] = {"matchLabels": {"other": "value"}}
    elif fault == "service-account":
        actual["spec"]["template"]["spec"]["serviceAccountName"] = "privileged"
    elif fault == "status":
        actual["status"] = {"active": 1}
    elif fault == "gpu":
        actual["spec"]["template"]["spec"]["containers"][0]["resources"]["limits"][
            "nvidia.com/gpu"
        ] = "1"
    with pytest.raises(ValueError, match=message):
        validate_controller_job_response(actual, package, require_uid=False)


class FakeApi:
    def __init__(self, package, *, create_fault=False):
        self.package = package
        self.create_fault = create_fault
        self.calls = []

    def get_job(self, namespace, name):
        self.calls.append(("get", namespace, name))
        return None

    def server_dry_run_job(self, manifest):
        self.calls.append(("preview", manifest))
        return server_object(self.package)

    def create_job_once(self, manifest):
        self.calls.append(("create", manifest))
        value = server_object(self.package)
        if self.create_fault:
            value["metadata"].pop("uid")
        return value


class MemoryJournal:
    def __init__(self, package, *, path=None):
        self.path = (
            Path(package.packet["controller"]["create_journal"]) if path is None else Path(path)
        )
        self.rows = []

    def exists(self):
        return bool(self.rows)

    def write_once(self, value):
        if self.rows:
            raise JobsError("intent exists")
        self.rows.append(copy.deepcopy(value))

    def append(self, value):
        if not self.rows:
            raise JobsError("intent missing")
        self.rows.append(copy.deepcopy(value))

    def read_rows(self):
        return copy.deepcopy(self.rows)


def test_typed_caller_previews_then_creates_once_and_binds_uid(tmp_path):
    package = build_controller_job(packet())
    api = FakeApi(package)
    journal = MemoryJournal(package)
    receipt = create_controller_job_once(package, api, journal)
    assert [row[0] for row in api.calls] == ["get", "preview", "get", "create"]
    assert receipt["status"] == "created_once"
    assert receipt["job_uid"] == JOB_UID
    assert receipt["failure_alerts"] == "off"
    assert receipt["gpus"] == 0
    assert receipt["target_posts"] == 0
    rows = journal.read_rows()
    assert [row["state"] for row in rows] == [
        "CREATE_INTENT_DO_NOT_RETRY",
        "CREATE_RESPONSE",
    ]
    with pytest.raises(JobsError, match="intent exists"):
        create_controller_job_once(package, FakeApi(package), journal)


def test_ambiguous_create_keeps_do_not_retry_intent(tmp_path):
    package = build_controller_job(packet())
    journal = MemoryJournal(package)
    with pytest.raises(JobsError, match="ambiguous"):
        create_controller_job_once(package, FakeApi(package, create_fault=True), journal)
    rows = journal.read_rows()
    assert len(rows) == 1
    assert rows[0]["state"] == "CREATE_INTENT_DO_NOT_RETRY"


def test_caller_rejects_alternate_non_sfs_journal_before_api_call(tmp_path):
    package = build_controller_job(packet())
    api = FakeApi(package)
    with pytest.raises(JobsError, match="canonical SFS path"):
        create_controller_job_once(
            package,
            api,
            MemoryJournal(package, path=tmp_path / "alternate.jsonl"),
        )
    assert api.calls == []


def test_uidless_exact_preview_is_normalized_without_keyerror():
    package = build_controller_job(packet())
    proof = validate_controller_job_response(package.job, package, require_uid=False)
    assert proof["job_uid"] is None


def test_driver_proves_identity_mounts_and_writes_no_target(tmp_path):
    package = build_controller_job(packet())
    sfs = tmp_path / "sfs"
    controls = sfs / "jobs/chris-q38-study-corpora-v1/launch-controls"
    create_root = controls / "prod9-create-once-v1"
    create_root.mkdir(parents=True, mode=0o700)
    mountinfo = (
        f"1 0 0:1 / {sfs} ro - nfs server:/sfs ro\n"
        f"2 1 0:1 /controls {controls} rw - nfs server:/sfs rw\n"
    )
    receipt = observe(
        runtime_environment(package),
        controls_root=controls,
        sfs_root=sfs,
        mountinfo=mountinfo,
        runtime_user=(1000, 100),
        filesystem_owner=(os.getuid(), os.getgid()),
        runtime_identity=runtime_identity(),
    )
    assert receipt["status"] == "passed_non_submitting_bootstrap"
    assert receipt["runtime_user"] == {"uid": 1000, "gid": 100}
    assert receipt["sfs_root_read_only"] is True
    assert receipt["controls_mount_writable"] is True
    assert receipt["gpus"] == 0
    assert receipt["target_posts"] == 0
    assert receipt["launch_authorized"] is False
    assert receipt["private_rows_read"] == 0
    assert receipt["traces_read"] == 0
    durable = (
        controls
        / "prod10-controller-bootstrap-v1"
        / Path(package.packet["controller"]["bootstrap_receipt"]).name
    )
    assert json.loads(durable.read_bytes()) == receipt
    validate_receipt(
        receipt,
        packet=package.packet,
        job_uid=JOB_UID,
        pod_name=NAME + "-abcde",
        pod_uid=POD_UID,
    )
    with pytest.raises(BootstrapError, match="bootstrap_receipt_exists"):
        observe(
            runtime_environment(package),
            controls_root=controls,
            sfs_root=sfs,
            mountinfo=mountinfo,
            runtime_user=(1000, 100),
            filesystem_owner=(os.getuid(), os.getgid()),
            runtime_identity=runtime_identity(),
        )


@pytest.mark.parametrize(
    ("fault", "code"),
    [
        ("user", "runtime_user_invalid"),
        ("gpu", "gpu_visibility_invalid"),
        ("job-uid", "job_uid_invalid"),
        ("sfs-rw", "sfs_read_only_mount_invalid"),
        ("controls-ro", "controls_writable_mount_invalid"),
        ("driver", "driver_binding_invalid"),
    ],
)
def test_driver_fails_closed_without_exposing_private_data(tmp_path, fault, code):
    package = build_controller_job(packet())
    values = runtime_environment(package)
    identity = runtime_identity()
    user = (1000, 100)
    sfs = tmp_path / "sfs"
    controls = sfs / "jobs/chris-q38-study-corpora-v1/launch-controls"
    (controls / "prod9-create-once-v1").mkdir(parents=True, mode=0o700)
    sfs_mode = "ro"
    controls_mode = "rw"
    if fault == "user":
        user = (0, 0)
    elif fault == "gpu":
        values["NVIDIA_VISIBLE_DEVICES"] = "all"
    elif fault == "job-uid":
        identity[ENV_JOB_UID] = "not-a-uid"
    elif fault == "sfs-rw":
        sfs_mode = "rw"
    elif fault == "controls-ro":
        controls_mode = "ro"
    elif fault == "driver":
        values[ENV_DRIVER_SHA256] = "0" * 64
    mountinfo = (
        f"1 0 0:1 / {sfs} {sfs_mode} - nfs server:/sfs {sfs_mode}\n"
        f"2 1 0:1 /controls {controls} {controls_mode} - nfs server:/sfs {controls_mode}\n"
    )
    with pytest.raises(BootstrapError, match=code):
        observe(
            values,
            controls_root=controls,
            sfs_root=sfs,
            mountinfo=mountinfo,
            runtime_user=user,
            filesystem_owner=(os.getuid(), os.getgid()),
            runtime_identity=identity,
            write=False,
        )


def test_terminal_receipt_binds_creator_job_pod_and_runtime(tmp_path):
    package = build_controller_job(packet())
    journal = MemoryJournal(package)
    creator = create_controller_job_once(
        package,
        FakeApi(package),
        journal,
    )
    sfs = tmp_path / "sfs"
    controls = sfs / "jobs/chris-q38-study-corpora-v1/launch-controls"
    (controls / "prod9-create-once-v1").mkdir(parents=True, mode=0o700)
    mountinfo = (
        f"1 0 0:1 / {sfs} ro - nfs server:/sfs ro\n"
        f"2 1 0:1 /controls {controls} rw - nfs server:/sfs rw\n"
    )
    runtime = observe(
        runtime_environment(package),
        controls_root=controls,
        sfs_root=sfs,
        mountinfo=mountinfo,
        runtime_user=(1000, 100),
        filesystem_owner=(os.getuid(), os.getgid()),
        runtime_identity=runtime_identity(),
    )
    job = server_object(package)
    generated = {
        "batch.kubernetes.io/controller-uid": JOB_UID,
        "batch.kubernetes.io/job-name": NAME,
        "controller-uid": JOB_UID,
        "job-name": NAME,
    }
    for key in generated:
        job["metadata"]["labels"].pop(key)
    workload_name = "job-" + NAME + "-abcde"
    kueue_labels = {
        "kueue.x-k8s.io/cluster-queue-name": "training-cq",
        "kueue.x-k8s.io/local-queue-name": "training-lq",
        "kueue.x-k8s.io/podset": "main",
    }
    job["spec"]["suspend"] = False
    job["spec"]["template"]["metadata"]["annotations"]["kueue.x-k8s.io/workload"] = workload_name
    job["spec"]["template"]["metadata"]["labels"].update(kueue_labels)
    job["status"] = {"conditions": [{"type": "Complete", "status": "True"}]}
    owner = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "name": NAME,
        "uid": JOB_UID,
        "controller": True,
        "blockOwnerDeletion": True,
    }
    podset_spec = copy.deepcopy(package.job["spec"]["template"]["spec"])
    podset_spec["priority"] = None
    workloads = {
        "apiVersion": "kueue.x-k8s.io/v1beta2",
        "kind": "WorkloadList",
        "items": [
            {
                "apiVersion": "kueue.x-k8s.io/v1beta2",
                "kind": "Workload",
                "metadata": {
                    "name": workload_name,
                    "namespace": NAMESPACE,
                    "uid": "99999999-2222-4333-8444-555555555555",
                    "labels": {"kueue.x-k8s.io/job-uid": JOB_UID},
                    "ownerReferences": [owner],
                },
                "spec": {
                    "queueName": "training-lq",
                    "priority": 10_000,
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
                                    "annotations": copy.deepcopy(
                                        package.job["spec"]["template"]["metadata"]["annotations"]
                                    ),
                                    "labels": {
                                        **copy.deepcopy(
                                            package.job["spec"]["template"]["metadata"]["labels"]
                                        ),
                                        "batch.kubernetes.io/job-name": NAME,
                                    },
                                },
                                "spec": podset_spec,
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
                                "resourceUsage": {"cpu": "2", "memory": "2Gi"},
                            }
                        ],
                    },
                    "conditions": [
                        {"type": "Admitted", "status": "True"},
                        {"type": "Finished", "status": "True", "reason": "Succeeded"},
                    ],
                },
            }
        ],
    }
    pod_spec = copy.deepcopy(package.job["spec"]["template"]["spec"])
    pod_spec.update(
        {
            "dnsPolicy": "ClusterFirst",
            "enableServiceLinks": True,
            "preemptionPolicy": "PreemptLowerPriority",
            "schedulerName": "default-scheduler",
            "serviceAccount": "default",
            "serviceAccountName": "default",
            "imagePullSecrets": [{"name": "ecr-pull"}],
            "nodeName": "shared-cpu-1",
        }
    )
    pod_spec["tolerations"].extend(
        [
            {
                "key": "node.kubernetes.io/not-ready",
                "operator": "Exists",
                "effect": "NoExecute",
                "tolerationSeconds": 300,
            },
            {
                "key": "node.kubernetes.io/unreachable",
                "operator": "Exists",
                "effect": "NoExecute",
                "tolerationSeconds": 300,
            },
        ]
    )
    pod = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": NAME + "-abcde",
            "namespace": NAMESPACE,
            "uid": POD_UID,
            "annotations": copy.deepcopy(job["spec"]["template"]["metadata"]["annotations"]),
            "labels": {
                **copy.deepcopy(job["spec"]["template"]["metadata"]["labels"]),
                "topology.kubernetes.io/region": "eu-north1",
            },
            "ownerReferences": [owner],
        },
        "spec": pod_spec,
        "status": {
            "phase": "Succeeded",
            "containerStatuses": [
                {
                    "name": "controller",
                    "image": IMAGE,
                    "restartCount": 0,
                    "imageID": "docker-pullable://" + IMAGE,
                    "state": {
                        "terminated": {
                            "exitCode": 0,
                            "reason": "Completed",
                            "message": json.dumps(
                                runtime,
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                        }
                    },
                }
            ],
        },
    }
    service_account = {
        "apiVersion": "v1",
        "kind": "ServiceAccount",
        "metadata": {"name": "default", "namespace": NAMESPACE},
        "imagePullSecrets": [{"name": "ecr-pull"}],
    }
    pod_list = {"apiVersion": "v1", "kind": "PodList", "items": [pod]}
    logs = (
        "CYBER_SKYRL_PROD10_CONTROLLER_BOOTSTRAP="
        + json.dumps(runtime, sort_keys=True, separators=(",", ":"))
        + "\n"
    )
    terminal = collect_controller_terminal_receipt(
        package,
        creator,
        journal,
        job,
        workloads,
        pod_list,
        service_account,
        logs,
    )
    assert terminal["status"] == "accepted_non_submitting_bootstrap"
    assert terminal["job_uid"] == JOB_UID
    assert terminal["pod_uid"] == POD_UID
    assert terminal["workload_name"] == workload_name
    assert terminal["workload_uid"] == "99999999-2222-4333-8444-555555555555"
    assert terminal["local_queue"] == "training-lq"
    assert terminal["cluster_queue"] == "training-cq"
    assert terminal["effective_priority"] == 10_000
    assert terminal["resolved_image_id"] == "docker-pullable://" + IMAGE
    assert terminal["runtime_receipt_path"] == package.packet["controller"]["bootstrap_receipt"]
    assert terminal["failure_alerts"] == "off"
    assert terminal["gpus"] == 0
    assert terminal["target_posts"] == 0
    assert terminal["launch_authorized"] is False

    class TerminalApi:
        def terminal_evidence(self, *, job_uid):
            assert job_uid == JOB_UID
            return tuple(
                copy.deepcopy(value) for value in (job, workloads, pod_list, service_account, logs)
            )

    assert (
        collect_controller_terminal_from_api(package, creator, journal, TerminalApi()) == terminal
    )

    bad_creator = copy.deepcopy(creator)
    bad_creator["server_render_sha256"] = "not-a-digest"
    with pytest.raises(ValueError, match="creator receipt (?:digest )?changed"):
        collect_controller_terminal_receipt(
            package,
            bad_creator,
            journal,
            job,
            workloads,
            pod_list,
            service_account,
            logs,
        )
    bad_pods = copy.deepcopy(pod_list)
    bad_pods["kind"] = "List"
    with pytest.raises(ValueError, match="Pod inventory GVK"):
        collect_controller_terminal_receipt(
            package,
            creator,
            journal,
            job,
            workloads,
            bad_pods,
            service_account,
            logs,
        )
    with pytest.raises(ValueError, match="log and termination"):
        collect_controller_terminal_receipt(
            package,
            creator,
            journal,
            job,
            workloads,
            pod_list,
            service_account,
            logs.replace("BOOTSTRAP=", "BOOTSTRAP_CHANGED="),
        )


def test_driver_packet_environment_contains_no_credentials_or_raw_task_data():
    package = build_controller_job(packet())
    environment = package.job["spec"]["template"]["spec"]["containers"][0]["env"]
    names = {entry["name"] for entry in environment}
    assert not any(
        marker in name
        for name in names
        for marker in ("TOKEN", "PASSWORD", "SECRET", "API_KEY", "CREDENTIAL")
    )
    packet_value = json.loads(literal_environment(package)[ENV_PACKET])
    assert "rows" not in json.dumps(packet_value).lower()
    assert "trace" not in json.dumps(packet_value).lower()
