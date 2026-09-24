import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from cyber_post_train import direct_submit
from cyber_post_train import sft_cpu_preflight_driver as driver
from cyber_post_train.cli import _prepare
from cyber_post_train.direct_submit import collect_sft_cpu_preflight, create_sft_cpu_preflight_once
from cyber_post_train.jobs import JobsError, digest
from cyber_post_train.sft_cpu_preflight_driver import LOG_PREFIX
from cyber_post_train.sft_cpu_preflight_job import (
    BUNDLE_ANNOTATION,
    DRIVER_ANNOTATION,
    SOURCE_COMMIT_ANNOTATION,
    build_sft_cpu_preflight_job,
    collect_sft_cpu_preflight_failure,
    collect_sft_cpu_preflight_receipt,
    validate_sft_cpu_preflight_job_node_fit,
    validate_sft_cpu_preflight_job_package,
    validate_sft_cpu_preflight_job_response,
)
from tests.test_sfs_output_job import completed_objects, node_inventory, service_account
from training.sft import compile_sft, job_request, read_mapping

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/runs/qwen38-teacher3k-32k-full-b8-lr1e6-v3.json"
SOURCE_COMMIT = "a" * 40


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    plan = compile_sft(read_mapping(CONFIG), relative_to=CONFIG.parent)
    request = job_request(plan)
    target = tmp_path_factory.mktemp("sft-cpu-preflight") / "prepared"
    _prepare(target, plan, request)
    return target


@pytest.fixture(scope="module")
def package(prepared):
    return build_sft_cpu_preflight_job(
        prepared,
        source_commit=SOURCE_COMMIT,
        attempt=1,
    )


def test_package_is_exact_alert_off_c1_q1_zero_gpu_read_only(package):
    proof = validate_sft_cpu_preflight_job_package(package)
    job = package.job
    assert job["metadata"]["name"] == "chris-q38-t3k32-lr1-v3-pre-a01"
    assert job["metadata"]["annotations"]["fleet.ai/failure-alerts"] == "off"
    assert job["spec"]["template"]["metadata"]["annotations"]["fleet.ai/failure-alerts"] == "off"
    assert job["metadata"]["labels"]["kueue.x-k8s.io/queue-name"] == "training-lq"
    assert job["metadata"]["labels"]["kueue.x-k8s.io/priority-class"] == "q1"
    assert job["spec"]["suspend"] is True
    assert job["spec"]["ttlSecondsAfterFinished"] == 1800
    pod = job["spec"]["template"]["spec"]
    container = pod["containers"][0]
    assert container["terminationMessagePolicy"] == "FallbackToLogsOnError"
    assert pod["priorityClassName"] == "c1" and pod["priority"] == 10_000
    assert pod["automountServiceAccountToken"] is False
    assert pod["imagePullSecrets"] == []
    assert pod["containers"][0]["volumeMounts"][0] == {
        "name": "sfs",
        "mountPath": "/mnt/sfs",
        "readOnly": True,
    }
    assert pod["volumes"][0]["persistentVolumeClaim"]["readOnly"] is True
    resources = container["resources"]
    assert resources["requests"]["memory"] == "32Gi"
    assert resources["limits"]["memory"] == "48Gi"
    assert "nvidia.com/gpu" not in json.dumps(resources)
    assert "envFrom" not in pod["containers"][0]
    assert "wandb-api" not in json.dumps(job)
    assert proof["source_commit"] == SOURCE_COMMIT
    assert proof["bundle_sha256"] == job["metadata"]["annotations"][BUNDLE_ANNOTATION]
    assert proof["driver_sha256"] == job["metadata"]["annotations"][DRIVER_ANNOTATION]
    assert job["metadata"]["annotations"][SOURCE_COMMIT_ANNOTATION] == SOURCE_COMMIT


def test_embedded_bundle_reopens_exact_source_and_prepared_bytes(package, monkeypatch):
    environment = {
        entry["name"]: entry["value"]
        for entry in package.job["spec"]["template"]["spec"]["containers"][0]["env"]
    }
    for name, value in environment.items():
        if name == driver.ENV_BUNDLE_SHA256 or (
            name.startswith(driver.CHUNK_PREFIX)
            and name.removeprefix(driver.CHUNK_PREFIX).isdigit()
        ):
            monkeypatch.setenv(name, value)
    blob = driver._bundle_from_environment()
    manifest, files = driver._inspect_bundle(blob)
    assert len(files) <= driver.MAX_FILES
    assert set(files) == set(manifest["files"])
    assert manifest["source_commit"] == SOURCE_COMMIT
    assert manifest["plan_sha256"] == digest(package.plan)
    assert manifest["request_sha256"] == digest(package.request)
    assert (
        files["prepared/plan.json"] == package.prepared_directory.joinpath("plan.json").read_text()
    )


def test_driver_output_absence_fails_closed_on_symlink_and_inspection_error(tmp_path, monkeypatch):
    root = tmp_path / "jobs"
    root.mkdir()
    target = root / "run"
    driver._require_output_absent(target)
    target.symlink_to(root, target_is_directory=True)
    with pytest.raises(FileExistsError):
        driver._require_output_absent(target)
    target.unlink()
    original = Path.lstat

    def denied(path):
        if path == target:
            raise PermissionError("denied")
        return original(path)

    monkeypatch.setattr(Path, "lstat", denied)
    with pytest.raises(ValueError, match="could not be inspected"):
        driver._require_output_absent(target)


def test_driver_reads_only_bounded_cgroup_v2_peak_memory(tmp_path):
    cgroup = tmp_path / "cgroup"
    cgroup.write_text("0::/kubepods/pod/container\n")
    assert driver._cgroup_v2_memory_peak_path(cgroup, tmp_path) == (
        tmp_path / "kubepods/pod/container/memory.peak"
    )
    cgroup.write_text("11:memory:/kubepods/pod/container\n")
    with pytest.raises(ValueError, match="unavailable"):
        driver._cgroup_v2_memory_peak_path(cgroup, tmp_path)

    peak = tmp_path / "memory.peak"
    peak.write_text("123456789\n")
    assert driver._cgroup_v2_peak_memory_bytes(peak) == 123456789

    for malformed in (
        "",
        "max\n",
        "-1\n",
        "0\n",
        f"{driver.MAX_PEAK_MEMORY_BYTES + 1}\n",
        "1" * 33,
    ):
        peak.write_text(malformed)
        with pytest.raises(ValueError, match="peak-memory evidence"):
            driver._cgroup_v2_peak_memory_bytes(peak)
    with pytest.raises(ValueError, match="unavailable"):
        driver._cgroup_v2_peak_memory_bytes(tmp_path / "missing")


def test_driver_sanitizes_failure_stage_without_exposing_message():
    private = "private-value-must-not-appear"
    with (
        pytest.raises(driver._PreflightFailure) as caught,
        driver._failure_stage("native.prepare_rows_train"),
    ):
        raise ValueError(private)
    failure = caught.value
    assert failure.stage == "native.prepare_rows_train"
    assert failure.error_class == "ValueError"
    assert len(failure.error_fingerprint) == 64
    assert private not in str(failure)


@pytest.mark.parametrize(
    "fault",
    [
        "root-alert",
        "template-alert",
        "ttl",
        "priority",
        "gpu",
        "memory-request",
        "memory-limit",
        "termination-policy",
        "secret-env",
        "secret-volume",
        "extra-pull-secret",
        "privileged",
        "sidecar",
        "init-container",
        "node-name",
        "affinity",
        "runtime-class",
        "overhead",
        "writable-sfs",
    ],
)
def test_server_response_rejects_security_resource_and_placement_drift(package, fault):
    actual = deepcopy(package.job)
    pod = actual["spec"]["template"]["spec"]
    container = pod["containers"][0]
    if fault == "root-alert":
        actual["metadata"]["annotations"]["fleet.ai/failure-alerts"] = "on"
    elif fault == "template-alert":
        actual["spec"]["template"]["metadata"]["annotations"]["fleet.ai/failure-alerts"] = "on"
    elif fault == "ttl":
        actual["spec"]["ttlSecondsAfterFinished"] = 0
    elif fault == "priority":
        pod["priority"] = 0
    elif fault == "gpu":
        container["resources"]["limits"]["nvidia.com/gpu"] = "1"
    elif fault == "memory-request":
        container["resources"]["requests"]["memory"] = "31Gi"
    elif fault == "memory-limit":
        container["resources"]["limits"]["memory"] = "47Gi"
    elif fault == "termination-policy":
        container["terminationMessagePolicy"] = "File"
    elif fault == "secret-env":
        container["envFrom"] = [{"secretRef": {"name": "unreviewed"}}]
    elif fault == "secret-volume":
        pod["volumes"].append({"name": "secret", "secret": {"secretName": "unreviewed"}})
    elif fault == "extra-pull-secret":
        pod["imagePullSecrets"] = [{"name": "unreviewed"}]
    elif fault == "privileged":
        container["securityContext"]["privileged"] = True
    elif fault == "sidecar":
        pod["containers"].append(deepcopy(container))
    elif fault == "init-container":
        pod["initContainers"] = [deepcopy(container)]
    elif fault == "node-name":
        pod["nodeName"] = "chosen-node"
    elif fault == "affinity":
        pod["affinity"] = {"nodeAffinity": {}}
    elif fault == "runtime-class":
        pod["runtimeClassName"] = "unreviewed"
    elif fault == "overhead":
        pod["overhead"] = {"cpu": "1"}
    else:
        container["volumeMounts"][0]["readOnly"] = False
    with pytest.raises(ValueError):
        validate_sft_cpu_preflight_job_response(actual, package, require_uid=False)


def test_package_reopens_prepared_and_source_bytes(package):
    changed = deepcopy(package.job)
    changed["metadata"]["annotations"][BUNDLE_ANNOTATION] = "0" * 64
    with pytest.raises(ValueError, match="differs from current source/prepared bytes"):
        validate_sft_cpu_preflight_job_package(replace(package, job=changed))
    request_path = package.prepared_directory / "request.json"
    original = request_path.read_text()
    try:
        request_path.write_text("{}")
        with pytest.raises(ValueError):
            validate_sft_cpu_preflight_job_package(package)
    finally:
        request_path.write_text(original)


def test_node_fit_requires_four_cpu_and_32_gib(package):
    assert validate_sft_cpu_preflight_job_node_fit(package, node_inventory())["fitting_nodes"] == 1
    with pytest.raises(ValueError, match="cannot fit"):
        validate_sft_cpu_preflight_job_node_fit(package, node_inventory(memory="31Gi"))


def test_collect_binds_terminal_job_workload_pod_image_and_native_receipt(package):
    body = {
        "schema": "cyber_sft_cpu_preflight_v1",
        "status": "passed",
        "gpus": 0,
        "plan_sha256": digest(package.plan),
        "request_sha256": digest(package.request),
        "checked": ["native_sources"],
        "counts": {"train": {"rows": 1, "tasks": 1, "supervised_tokens": 1}},
    }
    receipt = {**body, "sha256": digest(body)}
    job, workloads, pods, _ = completed_objects(package, {})
    job["spec"]["suspend"] = False
    assignment = workloads["items"][0]["status"]["admission"]["podSetAssignments"][0]
    requests = package.job["spec"]["template"]["spec"]["containers"][0]["resources"]["requests"]
    assignment["resourceUsage"] = deepcopy(requests)
    assignment["flavors"] = {name: "cpu-head" for name in requests}
    image_digest = package.request["image"].rsplit("@", 1)[-1]
    pods["items"][0]["status"]["containerStatuses"][0]["imageID"] = (
        "docker-pullable://registry/image@" + image_digest
    )
    proof = validate_sft_cpu_preflight_job_package(package)
    envelope = {
        "schema": "cyber_sft_cpu_preflight_observation_v1",
        "status": "passed",
        "gpus": 0,
        "job_name": job["metadata"]["name"],
        "observed_at_unix": 1.0,
        "bundle_sha256": proof["bundle_sha256"],
        "driver_sha256": proof["driver_sha256"],
        "plan_sha256": proof["plan_sha256"],
        "request_sha256": proof["request_sha256"],
        "preflight": receipt,
    }
    logs = LOG_PREFIX + json.dumps(envelope, sort_keys=True, separators=(",", ":"))

    def collect(value):
        return collect_sft_cpu_preflight_receipt(
            package,
            job,
            workloads,
            pods,
            service_account(),
            value,
        )

    assert collect(logs) == receipt
    envelope["peak_memory_bytes"] = 1
    logs = LOG_PREFIX + json.dumps(envelope, sort_keys=True, separators=(",", ":"))
    with pytest.raises(ValueError, match="peak-memory evidence is unexpected"):
        collect(logs)


def test_collect_rejects_admission_resource_drift(package):
    job, workloads, pods, _ = completed_objects(package, {})
    job["spec"]["suspend"] = False
    assignment = workloads["items"][0]["status"]["admission"]["podSetAssignments"][0]
    assignment["resourceUsage"] = {"cpu": "1", "memory": "1Gi"}
    image_digest = package.request["image"].rsplit("@", 1)[-1]
    pods["items"][0]["status"]["containerStatuses"][0]["imageID"] = (
        "docker-pullable://registry/image@" + image_digest
    )
    with pytest.raises(ValueError, match="admitted resources drifted"):
        collect_sft_cpu_preflight_receipt(
            package,
            job,
            workloads,
            pods,
            service_account(),
            "",
        )


@pytest.mark.parametrize("surface", ["workload", "pod"])
def test_collect_rejects_terminal_message_policy_drift(package, surface):
    body = {
        "schema": "cyber_sft_cpu_preflight_v1",
        "status": "passed",
        "gpus": 0,
        "plan_sha256": digest(package.plan),
        "request_sha256": digest(package.request),
        "checked": [],
        "counts": {},
    }
    receipt = {**body, "sha256": digest(body)}
    job, workloads, pods, _ = completed_objects(package, {})
    job["spec"]["suspend"] = False
    assignment = workloads["items"][0]["status"]["admission"]["podSetAssignments"][0]
    requests = package.job["spec"]["template"]["spec"]["containers"][0]["resources"]["requests"]
    assignment["resourceUsage"] = deepcopy(requests)
    assignment["flavors"] = {name: "cpu-head" for name in requests}
    if surface == "workload":
        container = workloads["items"][0]["spec"]["podSets"][0]["template"]["spec"]["containers"][0]
    else:
        container = pods["items"][0]["spec"]["containers"][0]
    container["terminationMessagePolicy"] = "File"
    image_digest = package.request["image"].rsplit("@", 1)[-1]
    pods["items"][0]["status"]["containerStatuses"][0]["imageID"] = (
        "docker-pullable://registry/image@" + image_digest
    )
    proof = validate_sft_cpu_preflight_job_package(package)
    envelope = {
        "schema": driver.ENVELOPE_SCHEMA,
        "status": "passed",
        "gpus": 0,
        "job_name": job["metadata"]["name"],
        "observed_at_unix": 1.0,
        "bundle_sha256": proof["bundle_sha256"],
        "driver_sha256": proof["driver_sha256"],
        "plan_sha256": proof["plan_sha256"],
        "request_sha256": proof["request_sha256"],
        "preflight": receipt,
    }
    logs = LOG_PREFIX + json.dumps(envelope, sort_keys=True, separators=(",", ":"))
    with pytest.raises(ValueError, match="container default drifted"):
        collect_sft_cpu_preflight_receipt(
            package,
            job,
            workloads,
            pods,
            service_account(),
            logs,
        )


def test_collect_preserves_source_and_uid_bound_sanitized_failure(package):
    job, workloads, pods, _ = completed_objects(package, {})
    job["spec"]["suspend"] = False
    job["status"] = {"conditions": [{"type": "Failed", "status": "True"}]}
    workloads["items"][0]["status"]["conditions"][1]["reason"] = "Failed"
    assignment = workloads["items"][0]["status"]["admission"]["podSetAssignments"][0]
    requests = package.job["spec"]["template"]["spec"]["containers"][0]["resources"]["requests"]
    assignment["resourceUsage"] = deepcopy(requests)
    assignment["flavors"] = {name: "cpu-head" for name in requests}
    pod = pods["items"][0]
    pod["status"]["phase"] = "Failed"
    status = pod["status"]["containerStatuses"][0]
    status["state"]["terminated"]["exitCode"] = 1
    image_digest = package.request["image"].rsplit("@", 1)[-1]
    status["imageID"] = "docker-pullable://registry/image@" + image_digest
    proof = validate_sft_cpu_preflight_job_package(package)
    envelope = {
        "schema": driver.ENVELOPE_SCHEMA,
        "status": "failed",
        "gpus": 0,
        "job_name": job["metadata"]["name"],
        "observed_at_unix": 1.0,
        "bundle_sha256": proof["bundle_sha256"],
        "driver_sha256": proof["driver_sha256"],
        "plan_sha256": proof["plan_sha256"],
        "request_sha256": proof["request_sha256"],
        "failure_stage": "native.prepare_rows_train",
        "error_class": "ValueError",
        "error_fingerprint": "b" * 64,
    }
    logs = LOG_PREFIX + json.dumps(envelope, sort_keys=True, separators=(",", ":"))
    receipt = collect_sft_cpu_preflight_failure(
        package,
        job,
        workloads,
        pods,
        service_account(),
        logs,
    )
    assert receipt["schema"] == "cyber_sft_cpu_preflight_failure_v1"
    assert receipt["classification"] == "preflight_failed"
    assert receipt["job"]["uid"] == job["metadata"]["uid"]
    assert receipt["pod"]["uid"] == pod["metadata"]["uid"]
    assert receipt["failure_stage"] == "native.prepare_rows_train"
    assert receipt["sha256"] == digest(
        {key: value for key, value in receipt.items() if key != "sha256"}
    )


@pytest.mark.parametrize("log_result", ["error", "empty"])
def test_direct_collect_validates_failure_then_uses_sanitized_termination_message(
    package, monkeypatch, log_result
):
    job, workloads, pods, _ = completed_objects(package, {})
    job["spec"]["suspend"] = False
    job["status"] = {"conditions": [{"type": "Failed", "status": "True"}]}
    workloads["items"][0]["status"]["conditions"][1]["reason"] = "Failed"
    assignment = workloads["items"][0]["status"]["admission"]["podSetAssignments"][0]
    requests = package.job["spec"]["template"]["spec"]["containers"][0]["resources"]["requests"]
    assignment["resourceUsage"] = deepcopy(requests)
    assignment["flavors"] = {name: "cpu-head" for name in requests}
    pod = pods["items"][0]
    pod["status"]["phase"] = "Failed"
    status = pod["status"]["containerStatuses"][0]
    status["state"]["terminated"]["exitCode"] = 1
    status["imageID"] = (
        "docker-pullable://registry/image@" + package.request["image"].rsplit("@", 1)[-1]
    )
    proof = validate_sft_cpu_preflight_job_package(package)
    envelope = {
        "schema": driver.ENVELOPE_SCHEMA,
        "status": "failed",
        "gpus": 0,
        "job_name": job["metadata"]["name"],
        "observed_at_unix": 1.0,
        "bundle_sha256": proof["bundle_sha256"],
        "driver_sha256": proof["driver_sha256"],
        "plan_sha256": proof["plan_sha256"],
        "request_sha256": proof["request_sha256"],
        "failure_stage": "native.prepare_rows_train",
        "error_class": "ValueError",
        "error_fingerprint": "c" * 64,
    }
    status["state"]["terminated"]["message"] = LOG_PREFIX + json.dumps(
        envelope, sort_keys=True, separators=(",", ":")
    )
    reads = []

    class FakeKubectl:
        def get_output_check_job(self, name):
            return job

        def list_output_check_workloads(self, uid):
            return workloads

        def list_output_check_pods(self, name):
            return pods

        def get_output_check_service_account(self):
            return service_account()

        def sft_cpu_preflight_logs(self, name):
            reads.append(name)
            if log_result == "error":
                raise JobsError("synthetic unavailable logs")
            return ""

    monkeypatch.setattr(
        direct_submit, "build_sft_cpu_preflight_job", lambda *args, **kwargs: package
    )
    receipt = collect_sft_cpu_preflight(
        directory=package.prepared_directory,
        source_commit=package.source_commit,
        attempt=package.attempt,
        kubectl=FakeKubectl(),
    )
    assert reads == [pod["metadata"]["name"]]
    assert receipt["schema"] == "cyber_sft_cpu_preflight_failure_v1"
    assert receipt["failure_stage"] == "native.prepare_rows_train"


def test_direct_collect_rejects_identity_before_reading_failure_logs(package, monkeypatch):
    job = deepcopy(package.job)
    job["metadata"]["uid"] = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
    job["status"] = {"conditions": [{"type": "Failed", "status": "True"}]}
    reads = []

    class FakeKubectl:
        def get_output_check_job(self, name):
            return job

        def list_output_check_workloads(self, uid):
            return {"kind": "List", "items": []}

        def list_output_check_pods(self, name):
            return {"kind": "List", "items": []}

        def get_output_check_service_account(self):
            return service_account()

        def sft_cpu_preflight_logs(self, name):
            reads.append(name)
            return ""

    monkeypatch.setattr(
        direct_submit, "build_sft_cpu_preflight_job", lambda *args, **kwargs: package
    )
    with pytest.raises(JobsError):
        collect_sft_cpu_preflight(
            directory=package.prepared_directory,
            source_commit=package.source_commit,
            attempt=package.attempt,
            kubectl=FakeKubectl(),
        )
    assert reads == []


def test_create_once_server_previews_journals_then_creates(prepared, tmp_path):
    class FakeKubectl:
        context = "prod"

        def __init__(self):
            self.created = []

        def _cpu_node_inventory(self):
            return node_inventory()

        def list(self, resource):
            assert resource == "jobs.batch"
            return {"kind": "List", "items": []}

        def dry_run(self, manifest):
            return deepcopy(manifest)

        def create_once(self, manifest):
            self.created.append(deepcopy(manifest))
            value = deepcopy(manifest)
            value["metadata"]["uid"] = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
            return value

    kubectl = FakeKubectl()
    journal = tmp_path / "create.jsonl"
    result = create_sft_cpu_preflight_once(
        directory=prepared,
        source_commit=SOURCE_COMMIT,
        attempt=1,
        kubectl=kubectl,
        journal=journal,
    )
    assert result == {
        "submitted": True,
        "gpus": 0,
        "name": "chris-q38-t3k32-lr1-v3-pre-a01",
        "uid": "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
        "attempt": 1,
    }
    lines = [json.loads(line) for line in journal.read_text().splitlines()]
    assert [line["state"] for line in lines] == [
        "KUBECTL_CREATE_INTENT_DO_NOT_RETRY",
        "KUBECTL_CREATE_RESPONSE",
    ]
    assert lines[0]["operation"] == "sft_cpu_preflight"
    assert kubectl.created[0]["metadata"]["annotations"]["fleet.ai/failure-alerts"] == "off"
    with pytest.raises(JobsError, match="journal already exists"):
        create_sft_cpu_preflight_once(
            directory=prepared,
            source_commit=SOURCE_COMMIT,
            attempt=1,
            kubectl=kubectl,
            journal=journal,
        )
