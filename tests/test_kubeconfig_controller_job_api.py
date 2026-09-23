import base64
import copy
import json
import subprocess
from pathlib import Path

import httpx
import pytest
import yaml

from cyber_post_train.jobs import JobsError
from cyber_post_train.kubeconfig_job_api import KubeconfigControllerJobApi
from cyber_post_train.skyrl_controller_job import (
    NAME,
    PROD_CONTEXT,
    build_controller_job,
    controller_packet,
    create_controller_job_once,
)

JOB_UID = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"


class MemoryJournal:
    def __init__(self, package):
        self.path = Path(package.packet["controller"]["create_journal"])
        self.rows = []

    def exists(self):
        return bool(self.rows)

    def write_once(self, value):
        self.rows.append(copy.deepcopy(value))

    def append(self, value):
        self.rows.append(copy.deepcopy(value))

    def read_rows(self):
        return copy.deepcopy(self.rows)


def package():
    packet = controller_packet(
        source_commit="a" * 40,
        identity_sha256="sha256:" + "1" * 64,
        plan_sha256="sha256:" + "2" * 64,
        request_sha256="sha256:" + "3" * 64,
        target_manifest_sha256="sha256:" + "4" * 64,
    )
    return build_controller_job(packet)


def kubeconfig(tmp_path):
    command = tmp_path / "nebius"
    command.write_text("#!/bin/sh\nexit 1\n")
    command.chmod(0o700)
    config = {
        "apiVersion": "v1",
        "kind": "Config",
        "current-context": PROD_CONTEXT,
        "contexts": [
            {
                "name": PROD_CONTEXT,
                "context": {"cluster": "prod-cluster", "user": "prod-user"},
            }
        ],
        "clusters": [
            {
                "name": "prod-cluster",
                "cluster": {
                    "server": "https://cluster.example.test",
                    "certificate-authority-data": base64.b64encode(b"test-ca").decode(),
                },
            }
        ],
        "users": [
            {
                "name": "prod-user",
                "user": {
                    "exec": {
                        "apiVersion": "client.authentication.k8s.io/v1beta1",
                        "args": ["iam", "get-access-token"],
                        "command": str(command),
                        "env": None,
                        "interactiveMode": "IfAvailable",
                        "provideClusterInfo": False,
                    }
                },
            }
        ],
    }
    path = tmp_path / "config"
    path.write_text(yaml.safe_dump(config))
    return path, command


def credential_runner(calls):
    def run(arguments, **kwargs):
        calls.append((arguments, kwargs))
        return subprocess.CompletedProcess(
            arguments,
            0,
            stdout=json.dumps(
                {
                    "apiVersion": "client.authentication.k8s.io/v1beta1",
                    "kind": "ExecCredential",
                    "status": {"token": "short-lived-test-token"},
                }
            ),
            stderr="",
        )

    return run


def test_typed_native_api_previews_then_creates_exact_job_once(tmp_path):
    reviewed = package()
    requests = []
    get_count = 0

    def handler(request):
        nonlocal get_count
        requests.append(request)
        assert request.headers["authorization"] == "Bearer short-lived-test-token"
        if request.method == "GET":
            get_count += 1
            return httpx.Response(404, json={"kind": "Status", "code": 404})
        submitted = json.loads(request.content)
        assert submitted == reviewed.job
        if request.url.params.get("dryRun") == "All":
            return httpx.Response(200, json=reviewed.job)
        created = copy.deepcopy(reviewed.job)
        created["metadata"]["uid"] = JOB_UID
        return httpx.Response(201, json=created)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    auth_calls = []
    config, command = kubeconfig(tmp_path)
    api = KubeconfigControllerJobApi(
        reviewed,
        kubeconfig=config,
        runner=credential_runner(auth_calls),
        client=client,
    )
    receipt = create_controller_job_once(reviewed, api, MemoryJournal(reviewed))
    assert receipt["job_uid"] == JOB_UID
    assert receipt["failure_alerts"] == "off"
    assert receipt["gpus"] == 0
    assert get_count == 2
    posts = [request for request in requests if request.method == "POST"]
    assert len(posts) == 2
    assert posts[0].url.params["dryRun"] == "All"
    assert "dryRun" not in posts[1].url.params
    assert all(request.url.params["fieldValidation"] == "Strict" for request in posts)
    assert all(call[0][0] == str(command) for call in auth_calls)
    assert not any("kubectl" in argument for call in auth_calls for argument in call[0])
    assert all(call[1]["stdin"] is subprocess.DEVNULL for call in auth_calls)
    assert {
        json.loads(call[1]["env"]["KUBERNETES_EXEC_INFO"])["spec"]["interactive"]
        for call in auth_calls
    } == {False}


def test_native_api_terminal_reads_are_exactly_scoped(tmp_path):
    reviewed = package()
    requests = []

    def handler(request):
        requests.append(request)
        path = request.url.path
        if path.endswith("/jobs/" + NAME):
            return httpx.Response(200, json={"metadata": {"uid": JOB_UID}})
        if path.endswith("/workloads"):
            return httpx.Response(
                200,
                json={"apiVersion": "kueue.x-k8s.io/v1beta2", "kind": "WorkloadList", "items": []},
            )
        if path.endswith("/pods"):
            return httpx.Response(
                200,
                json={
                    "apiVersion": "v1",
                    "kind": "PodList",
                    "items": [{"metadata": {"name": NAME + "-abcde"}}],
                },
            )
        if path.endswith("/serviceaccounts/default"):
            return httpx.Response(200, json={"apiVersion": "v1", "kind": "ServiceAccount"})
        if path.endswith("/pods/" + NAME + "-abcde/log"):
            return httpx.Response(200, text="sanitized-receipt\n")
        raise AssertionError(path)

    config, _ = kubeconfig(tmp_path)
    api = KubeconfigControllerJobApi(
        reviewed,
        kubeconfig=config,
        runner=credential_runner([]),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    job, workloads, pods, account, logs = api.terminal_evidence(job_uid=JOB_UID)
    assert job["metadata"]["uid"] == JOB_UID
    assert workloads["kind"] == "WorkloadList"
    assert pods["kind"] == "PodList"
    assert account["kind"] == "ServiceAccount"
    assert logs == "sanitized-receipt\n"
    selectors = [request.url.params.get("labelSelector") for request in requests]
    assert "kueue.x-k8s.io/job-uid=" + JOB_UID in selectors
    assert "batch.kubernetes.io/job-name=" + NAME in selectors
    assert all(request.url.host == "cluster.example.test" for request in requests)


def test_native_api_rejects_context_and_root_annotation_drift(tmp_path):
    reviewed = package()
    config, _ = kubeconfig(tmp_path)
    with pytest.raises(JobsError, match="exact production context"):
        KubeconfigControllerJobApi(
            reviewed,
            kubeconfig=config,
            context="not-production",
            runner=credential_runner([]),
            client=httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(500))),
        )
    api = KubeconfigControllerJobApi(
        reviewed,
        kubeconfig=config,
        runner=credential_runner([]),
        client=httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(500))),
    )
    drifted = copy.deepcopy(reviewed.job)
    drifted["metadata"]["annotations"]["fleet.ai/failure-alerts"] = "on"
    with pytest.raises(JobsError, match="root binding"):
        api.server_dry_run_job(drifted)
