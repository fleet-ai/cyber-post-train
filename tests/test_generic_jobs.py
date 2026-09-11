import json
from copy import deepcopy

import httpx
import pytest
import yaml

from cyber_post_train.jobs import (
    Jobs,
    JobsError,
    quantity,
    safe_status,
    validate_preview,
    validate_request,
)


def config():
    return {
        "name": "researcher-sft",
        "title": "Synthetic fixture",
        "image": "registry/image@sha256:" + "a" * 64,
        "command": "python train.py",
        "workers": 1,
        "gpus_per_worker": 8,
        "run_dir": "/mnt/sfs/jobs/researcher-sft-v1",
        "priority_class": "c1",
        "requeueIfPreempted": False,
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


def test_runtime_bundle_executes_exact_bytes_once(tmp_path):
    import os
    import shlex
    import subprocess
    import sys

    from cyber_post_train.jobs import bundled_request

    files = {
        "synthetic/__init__.py": "",
        "synthetic/run.py": "import pathlib,sys;pathlib.Path('result').write_text(sys.argv[1])",
    }
    request = bundled_request(config(), files, "synthetic.run", ["literal $not-a-shell-command"])
    assert request == bundled_request(
        config(), files, "synthetic.run", ["literal $not-a-shell-command"]
    )
    env = {**os.environ, **request["env"], "RUN_DIR": str(tmp_path)}
    command = [sys.executable, *shlex.split(request["command"])[1:]]
    assert subprocess.run(command, env=env, capture_output=True).returncode == 0
    assert (tmp_path / ".runtime/result").read_text() == "literal $not-a-shell-command"
    assert subprocess.run(command, env=env, capture_output=True).returncode != 0
    broken = {**env, "CYBER_RUNTIME_BUNDLE": "YQ=="}
    assert subprocess.run(command, env=broken, capture_output=True).returncode != 0


@pytest.mark.parametrize("fault", ["missing", "escape", "absolute", "text"])
def test_runtime_bundle_rejects_bad_content(fault):
    from cyber_post_train.jobs import bundled_request

    files = {"run.py": "pass"}
    if fault == "missing":
        files = {}
    elif fault == "escape":
        files["../escape"] = "pass"
    elif fault == "absolute":
        files["/absolute"] = "pass"
    else:
        files["run.py"] = None
    with pytest.raises(JobsError):
        bundled_request(config(), files, "run", [])


def test_large_runtime_bundle_is_chunked_and_digest_checked(tmp_path):
    import os
    import random
    import shlex
    import subprocess
    import sys

    from cyber_post_train.jobs import bundled_request

    noise = random.Random(42).randbytes(160000).hex()
    request = bundled_request(config(), {"run.py": "pass", "noise.txt": noise}, "run", [])
    assert "CYBER_RUNTIME_BUNDLE" not in request["env"]
    assert max(len(v) for v in request["env"].values()) <= 48000
    env = {**os.environ, **request["env"], "RUN_DIR": str(tmp_path)}
    command = [sys.executable, *shlex.split(request["command"])[1:]]
    assert subprocess.run(command, env=env, capture_output=True).returncode == 0
    assert (tmp_path / ".runtime/noise.txt").read_text() == noise
    other = tmp_path / "new"
    other.mkdir()
    env.update(RUN_DIR=str(other), CYBER_RUNTIME_BUNDLE_0="YQ==")
    assert subprocess.run(command, env=env, capture_output=True).returncode != 0
    assert not (other / ".runtime").exists()
    with pytest.raises(JobsError, match="too large"):
        bundled_request(config(), {"run.py": "pass", "noise": noise * 8 + noise[::-1]}, "run", [])
    value = config()
    value["env"]["CYBER_RUNTIME_BUNDLE_0"] = "unreviewed"
    with pytest.raises(JobsError, match="reserved"):
        bundled_request(value, {"run.py": "pass"}, "run", [])


def test_environment_size_limit_counts_utf8_bytes_and_name():
    value = config()
    value["env"]["DATA"] = "a" * (131072 - len("DATA") - 2)
    validate_request(value)
    value["env"]["DATA"] += "a"
    with pytest.raises(JobsError, match="process-start limit"):
        validate_request(value)
    value["env"]["DATA"] = "é" * 70000
    with pytest.raises(JobsError, match="process-start limit"):
        validate_request(value)


def test_gzip_header_is_reproducible_across_platforms(monkeypatch):
    import gzip

    from cyber_post_train.jobs import canonical_gzip

    original = gzip.compress
    results = []
    for os_byte in (3, 19, 255):

        def compress(payload, mtime, selected_os=os_byte):
            blob = original(payload, mtime=mtime)
            return blob[:9] + bytes([selected_os]) + blob[10:]

        monkeypatch.setattr(gzip, "compress", compress)
        result = canonical_gzip(b"synthetic immutable payload")
        assert result[9] == 255 and gzip.decompress(result) == b"synthetic immutable payload"
        results.append(result)
    assert results[0] == results[1] == results[2]


def manifest(request=None):
    c = request or config()
    pod = {
        "spec": {
            "priorityClassName": c["priority_class"],
            "imagePullSecrets": [{"name": s} for s in c["image_pull_secrets"]],
            "containers": [
                {
                    "image": c["image"],
                    "resources": {
                        "requests": {
                            "cpu": c["resources"]["cpu_request"],
                            "memory": c["resources"]["memory_request"],
                            "nvidia.com/gpu": c["gpus_per_worker"],
                        },
                        "limits": {
                            "cpu": c["resources"]["cpu_limit"],
                            "memory": c["resources"]["memory_limit"],
                            "nvidia.com/gpu": c["gpus_per_worker"],
                        },
                    },
                    "env": [
                        {"name": k, "value": v}
                        for k, v in {**c["env"], "RUN_DIR": c["run_dir"]}.items()
                    ],
                    "envFrom": [{"secretRef": {"name": s}} for s in c["secrets"]],
                    "securityContext": {"privileged": c.get("privileged", False)},
                }
            ],
        }
    }
    return {
        "kind": "RayJob",
        "metadata": {
            "namespace": "fleet-train-jobs",
            "labels": {
                "kueue.x-k8s.io/queue-name": "training-lq",
                "kueue.x-k8s.io/priority-class": "q" + c["priority_class"][1:],
                "fleet.ai/requeue-if-preempted": "false",
            },
            "annotations": {"fleet.ai/run-dir": c["run_dir"]},
        },
        "spec": {
            "suspend": True,
            "shutdownAfterJobFinishes": True,
            "entrypoint": c["command"],
            "rayClusterSpec": {
                "headGroupSpec": {"template": pod},
                "workerGroupSpecs": [{"replicas": c["workers"] - 1, "template": deepcopy(pod)}],
            },
        },
    }


def preview(obj=None):
    return {"manifest_yaml": yaml.safe_dump(obj or manifest()), "warnings": []}


@pytest.mark.parametrize("nodes", [1, 2, 4, 5, 8])
@pytest.mark.parametrize("priority", ["c1", "c2"])
def test_resource_preview(nodes, priority):
    request = {**config(), "workers": nodes, "priority_class": priority}
    result = validate_preview(request, preview(manifest(request)))
    assert result["nodes"] == nodes and result["gpus"] == nodes * 8
    assert len(result["manifest_sha256"]) == 64


@pytest.mark.parametrize(
    "field,value",
    [
        ("name", "a" * 32),
        ("name", "ft-run"),
        ("image", "image:latest"),
        ("command", ""),
        ("workers", True),
        ("workers", 0),
        ("workers", 9),
        ("gpus_per_worker", 0),
        ("gpus_per_worker", 9),
        ("priority_class", "c0"),
        ("queue_priority_class", "q1"),
        ("requeueIfPreempted", True),
        ("requeueIfPreempted", "false"),
        ("run_dir", "/mnt/sfs/jobs"),
        ("run_dir", "/mnt/sfs/jobs/a/../b"),
        ("run_dir", "/mnt/sfs/jobs/a/"),
        ("env", {"WANDB_API_KEY": "synthetic-secret"}),
        ("env", {"FLEET_CREDENTIALS_B64": "synthetic-secret"}),
        ("env", {"X": 1}),
        ("secrets", "not-a-list"),
        ("image_pull_secrets", ["INVALID"]),
        ("models", [{"hf": "unbound-model"}]),
        ("entrypoint_wrapper", "unchecked.sh"),
        ("resources", {}),
        ("privileged", "yes"),
    ],
)
def test_invalid_request_rejected_locally(field, value):
    with pytest.raises(JobsError):
        validate_request({**config(), field: value})


def test_privileged_partial_node_is_forbidden():
    with pytest.raises(JobsError, match="every GPU"):
        validate_request({**config(), "privileged": True, "gpus_per_worker": 1})


@pytest.mark.parametrize(
    "path,value",
    [
        (("kind",), "Job"),
        (("metadata", "namespace"), "peer"),
        (("metadata", "labels", "kueue.x-k8s.io/queue-name"), "bypass"),
        (("metadata", "labels", "kueue.x-k8s.io/priority-class"), "q0"),
        (("metadata", "labels", "fleet.ai/requeue-if-preempted"), "true"),
        (("metadata", "annotations", "fleet.ai/run-dir"), "/mnt/sfs/peer"),
        (("spec", "suspend"), False),
        (("spec", "shutdownAfterJobFinishes"), False),
        (("spec", "entrypoint"), "other-command"),
    ],
)
def test_manifest_control_plane_drift(path, value):
    obj = manifest()
    target = obj
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(JobsError):
        validate_preview(config(), preview(obj))


@pytest.mark.parametrize(
    "change",
    [
        "node",
        "priority",
        "gpu",
        "cpu",
        "memory",
        "image",
        "env",
        "secret",
        "pull_secret",
        "privilege",
        "count",
        "extra_container",
    ],
)
def test_pod_resource_and_runtime_drift(change):
    obj = manifest()
    pod = obj["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"]
    c = pod["containers"][0]
    if change == "node":
        pod["nodeName"] = "bypass-scheduler"
    elif change == "priority":
        pod["priorityClassName"] = "c0"
    elif change in {"gpu", "cpu", "memory"}:
        c["resources"]["requests"]["nvidia.com/gpu" if change == "gpu" else change] = "1"
    elif change == "image":
        c["image"] = "image:latest"
    elif change == "env":
        c["env"] = []
    elif change == "secret":
        c["envFrom"] = []
    elif change == "pull_secret":
        pod["imagePullSecrets"] = []
    elif change == "privilege":
        c["securityContext"]["privileged"] = True
    elif change == "count":
        obj["spec"]["rayClusterSpec"]["workerGroupSpecs"][0]["replicas"] = 1
    else:
        pod["containers"].append(deepcopy(c))
    with pytest.raises(JobsError):
        validate_preview(config(), preview(obj))


@pytest.mark.parametrize(
    "payload",
    [{}, {"manifest_yaml": "bad"}, {"manifest_yaml": "["}, {**preview(), "warnings": ["review"]}],
)
def test_missing_or_warned_preview_fails_closed(payload):
    with pytest.raises(JobsError):
        validate_preview(config(), payload)


def test_equal_kubernetes_quantities_are_not_false_drift():
    obj = manifest()
    c = obj["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"]["containers"][0]
    c["resources"]["requests"]["cpu"] = "8000m"
    c["resources"]["limits"]["memory"] = "131072Mi"
    validate_preview(config(), preview(obj))
    assert quantity("0.5") == quantity("500m")


def client(handler):
    return Jobs(
        "synthetic-token", base_url="https://jobs.invalid", transport=httpx.MockTransport(handler)
    )


def test_exhaustive_pagination():
    offsets = []

    def handler(req):
        offset = int(req.url.params["offset"])
        offsets.append(offset)
        return httpx.Response(
            200, json={"items": [{"name": f"run-{offset}"}], "has_more": offset == 0}
        )

    with client(handler) as api:
        assert len(api.all_runs()) == 2
    assert offsets == [0, 1]


@pytest.mark.parametrize(
    "payload",
    [[], {"items": []}, {"items": [], "has_more": True}, {"items": [{}], "has_more": False}],
)
def test_incomplete_history_blocks_submission(payload, tmp_path):
    with client(lambda req: httpx.Response(200, json=payload)) as api, pytest.raises(JobsError):
        api.submit_once(config(), tmp_path / "intent.jsonl")
    assert not (tmp_path / "intent.jsonl").exists()


@pytest.mark.parametrize("outcome", ["ok", "timeout", "http-error", "ambiguous", "wrong-root"])
def test_one_post_with_durable_intent_even_after_uncertain_failure(tmp_path, outcome):
    seen = []
    journal = tmp_path / "intent.jsonl"

    def handler(req):
        seen.append((req.method, req.url.path))
        assert b"synthetic-token" not in req.content
        if req.method == "GET":
            return httpx.Response(200, json={"items": [], "has_more": False})
        if req.url.path.endswith("/preview"):
            return httpx.Response(200, json=preview())
        assert journal.exists()
        assert json.loads(journal.read_text())["state"] == "POST_INTENT_DO_NOT_RETRY"
        if outcome == "timeout":
            raise httpx.ReadTimeout("private-trace-and-secret", request=req)
        if outcome == "http-error":
            return httpx.Response(500, text="private-trace-and-secret")
        response = {
            "name": "researcher-sft-1234abcd",
            "status": "queued",
            "failure_message": "private-trace-and-secret",
        }
        if outcome == "ambiguous":
            response["name"] = "unexpected"
        if outcome == "wrong-root":
            response["run_dir"] = "/mnt/sfs/other"
        return httpx.Response(202, json=response)

    with client(handler) as api:
        if outcome == "ok":
            assert api.submit_once(config(), journal)["status"] == "queued"
        else:
            with pytest.raises(JobsError) as error:
                api.submit_once(config(), journal)
            assert "private-trace-and-secret" not in str(error.value)
        with pytest.raises(JobsError, match="journal already exists"):
            api.submit_once(config(), journal)
    assert seen.count(("POST", "/v1/runs")) == 1
    assert "private-trace-and-secret" not in journal.read_text()
    assert journal.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(
    "record", [{"name": "old", "run_dir": config()["run_dir"]}, {"name": "researcher-sft-1234abcd"}]
)
def test_history_duplicate_is_never_resubmitted(record, tmp_path):
    def handler(req):
        assert req.method == "GET"
        return httpx.Response(200, json={"items": [record], "has_more": False})

    with client(handler) as api, pytest.raises(JobsError, match="already owns"):
        api.submit_once(config(), tmp_path / "intent.jsonl")


def test_status_is_allowlisted_and_name_cannot_inject_route():
    assert "private" not in str(safe_status({"name": "safe", "failure_message": "private"}))
    with client(
        lambda req: httpx.Response(200, json={"name": "safe", "status": "SUCCEEDED"})
    ) as api:
        assert api.status("safe")["status"] == "SUCCEEDED"
        with pytest.raises(JobsError):
            api.status("../runs")
