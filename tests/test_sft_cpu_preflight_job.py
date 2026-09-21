import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from cyber_post_train import sft_cpu_preflight_driver as driver
from cyber_post_train.cli import _prepare
from cyber_post_train.direct_submit import create_sft_cpu_preflight_once
from cyber_post_train.jobs import JobsError, digest
from cyber_post_train.sft_cpu_preflight_driver import LOG_PREFIX
from cyber_post_train.sft_cpu_preflight_job import (
    BUNDLE_ANNOTATION,
    DRIVER_ANNOTATION,
    SOURCE_COMMIT_ANNOTATION,
    build_sft_cpu_preflight_job,
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
    pod = job["spec"]["template"]["spec"]
    assert pod["priorityClassName"] == "c1" and pod["priority"] == 10_000
    assert pod["automountServiceAccountToken"] is False
    assert pod["imagePullSecrets"] == []
    assert pod["containers"][0]["volumeMounts"][0] == {
        "name": "sfs",
        "mountPath": "/mnt/sfs",
        "readOnly": True,
    }
    assert pod["volumes"][0]["persistentVolumeClaim"]["readOnly"] is True
    assert "nvidia.com/gpu" not in json.dumps(pod["containers"][0]["resources"])
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


@pytest.mark.parametrize(
    "fault",
    [
        "root-alert",
        "template-alert",
        "priority",
        "gpu",
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
    elif fault == "priority":
        pod["priority"] = 0
    elif fault == "gpu":
        container["resources"]["limits"]["nvidia.com/gpu"] = "1"
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
    assert (
        collect_sft_cpu_preflight_receipt(
            package,
            job,
            workloads,
            pods,
            service_account(),
            logs,
        )
        == receipt
    )


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


def test_create_once_server_previews_journals_then_creates(prepared, tmp_path):
    class FakeKubectl:
        context = "prod"

        def __init__(self):
            self.created = []

        def _cpu_node_inventory(self):
            return node_inventory()

        def list_exact_name(self, resource, name):
            assert resource == "jobs.batch"
            assert name.endswith("-pre-a01")
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
