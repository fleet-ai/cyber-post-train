import json
import os
import stat
import uuid
from copy import deepcopy

import httpx
import pytest
import yaml

from cyber_post_train import jobs as jobs_module
from cyber_post_train.jobs import (
    API_URLS,
    PRIVILEGED_WHOLE_NODE_WARNING,
    Jobs,
    JobsError,
    plan_api_target,
    quantity,
    safe_status,
    validate_creator_response,
    validate_preview,
    validate_request,
)


def test_plan_api_target_is_immutable_and_closed() -> None:
    assert plan_api_target(None) == ("prod", API_URLS["prod"])
    assert plan_api_target({}) == ("prod", API_URLS["prod"])
    for target in ("dev", "prod"):
        plan = {
            "execution": {
                "cluster_target": target,
                "jobs_api_base_url": API_URLS[target],
            }
        }
        assert plan_api_target(plan) == (target, API_URLS[target])
    for execution in (
        {"cluster_target": "dev"},
        {"jobs_api_base_url": API_URLS["dev"]},
        {"cluster_target": "dev", "jobs_api_base_url": API_URLS["prod"]},
        {"cluster_target": "prod", "jobs_api_base_url": API_URLS["dev"]},
        {"cluster_target": "staging", "jobs_api_base_url": "https://example.invalid"},
    ):
        with pytest.raises(JobsError, match="incomplete or mismatched"):
            plan_api_target({"execution": execution})


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


def test_runtime_bundle_executes_exact_bytes_once(tmp_path):
    import os
    import shlex
    import subprocess
    import sys

    from cyber_post_train.jobs import bundled_request

    files = {
        "synthetic/__init__.py": "",
        "synthetic/run.py": (
            "import pathlib,sys;TOKEN='qualified-module';"
            "assert sys.modules[__name__].TOKEN==TOKEN;"
            "pathlib.Path('result').write_text(sys.argv[1])"
        ),
    }
    request = bundled_request(config(), files, "synthetic.run", ["literal $not-a-shell-command"])
    assert request == bundled_request(
        config(), files, "synthetic.run", ["literal $not-a-shell-command"]
    )
    run_dir = tmp_path / "missing" / "run"
    env = {**os.environ, **request["env"], "RUN_DIR": str(run_dir)}
    command = [sys.executable, *shlex.split(request["command"])[1:]]
    assert subprocess.run(command, env=env, capture_output=True).returncode == 0
    assert (run_dir / ".runtime/result").read_text() == "literal $not-a-shell-command"
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


def test_runtime_bundle_can_obey_stricter_fleetjob_env_limit() -> None:
    import random

    from cyber_post_train.jobs import bundled_request

    payload = random.Random(43).randbytes(60000).hex()
    request = bundled_request(
        config(),
        {"run.py": "pass", "payload.txt": payload},
        "run",
        [],
        transport_split_threshold=30000,
        transport_chunk_size=30000,
    )
    chunks = {
        key: value
        for key, value in request["env"].items()
        if key.startswith("CYBER_RUNTIME_BUNDLE")
    }
    assert len(chunks) > 1
    assert max(map(len, chunks.values())) <= 30000
    for threshold, size in ((0, 1), (30000, 30001), (120001, 30000)):
        with pytest.raises(JobsError, match="transport limits"):
            bundled_request(
                config(),
                {"run.py": "pass"},
                "run",
                [],
                transport_split_threshold=threshold,
                transport_chunk_size=size,
            )


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
            "restartPolicy": "Never",
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
            "annotations": {
                "fleet.ai/run-dir": c["run_dir"],
                "fleet.ai/failure-alerts": "off",
            },
        },
        "spec": {
            "suspend": True,
            "shutdownAfterJobFinishes": True,
            "backoffLimit": 0,
            "entrypoint": c["command"],
            "rayClusterSpec": {
                "headGroupSpec": {"template": pod},
                "workerGroupSpecs": [{"replicas": c["workers"] - 1, "template": deepcopy(pod)}],
            },
        },
    }


def preview(obj=None):
    return {"manifest_yaml": yaml.safe_dump(obj or manifest()), "warnings": []}


def creator_row(request=None, **overrides):
    c = request or config()
    row = {
        "image": c["image"],
        "job_id": "e75dfbcc-dada-4bb5-9f4d-49b43322f2aa",
        "message": None,
        "name": c["name"] + "-1234abcd",
        "priority_class": c["priority_class"],
        "priority_reason": None,
        "queue_priority_class": "q" + c["priority_class"][1:],
        "requeueIfPreempted": False,
        "run_dir": c["run_dir"],
        "status": "QUEUED",
        "submitted_by": None,
        "submitted_by_profile_id": None,
    }
    row.update(overrides)
    return row


@pytest.mark.parametrize("nodes", [1, 2, 4, 5, 8])
@pytest.mark.parametrize("priority", ["c1", "c2"])
def test_resource_preview(nodes, priority):
    request = {**config(), "workers": nodes, "priority_class": priority}
    result = validate_preview(request, preview(manifest(request)))
    assert result["nodes"] == nodes and result["gpus"] == nodes * 8
    assert len(result["manifest_sha256"]) == 64


def test_privileged_whole_node_preview_accepts_only_the_reviewed_warning() -> None:
    request = {**config(), "privileged": True}
    payload = {
        "manifest_yaml": yaml.safe_dump(manifest(request)),
        "errors": None,
        "warnings": [PRIVILEGED_WHOLE_NODE_WARNING],
    }
    assert validate_preview(request, payload)["gpus"] == 8


@pytest.mark.parametrize(
    ("errors", "warnings"),
    [
        (None, []),
        (None, None),
        (None, ["different"]),
        (None, [PRIVILEGED_WHOLE_NODE_WARNING, "extra"]),
        ([], [PRIVILEGED_WHOLE_NODE_WARNING]),
        ({}, [PRIVILEGED_WHOLE_NODE_WARNING]),
    ],
)
def test_privileged_whole_node_preview_rejects_every_other_policy(errors, warnings) -> None:
    request = {**config(), "privileged": True}
    payload = {
        "manifest_yaml": yaml.safe_dump(manifest(request)),
        "errors": errors,
        "warnings": warnings,
    }
    with pytest.raises(JobsError, match="whole-node warning policy"):
        validate_preview(request, payload)


def test_generic_preview_keeps_legacy_omitted_errors_semantics() -> None:
    assert validate_preview(config(), preview())["gpus"] == 8


@pytest.mark.parametrize("value", [None, True, False, -1, 1, "0"])
def test_preview_requires_exact_integer_zero_backoff(value) -> None:
    obj = manifest()
    if value is None:
        obj["spec"].pop("backoffLimit")
    else:
        obj["spec"]["backoffLimit"] = value
    with pytest.raises(JobsError, match="controller retries"):
        validate_preview(config(), preview(obj))


@pytest.mark.parametrize("value", [None, "OnFailure", "Always", False])
def test_preview_requires_restart_never_on_every_active_group(value) -> None:
    obj = manifest()
    pod = obj["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"]
    if value is None:
        pod.pop("restartPolicy")
    else:
        pod["restartPolicy"] = value
    with pytest.raises(JobsError, match="restart policy"):
        validate_preview(config(), preview(obj))


@pytest.mark.parametrize(
    "field,value",
    [
        ("name", "a" * 32),
        ("name", "ft-run"),
        ("title", ""),
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
        ("failureAlerts", True),
        ("failureAlerts", "false"),
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


def test_request_without_explicit_failure_alert_opt_out_is_rejected() -> None:
    value = config()
    value.pop("failureAlerts")
    with pytest.raises(JobsError, match="failed-job alerts"):
        validate_request(value)


def test_privileged_partial_node_is_forbidden():
    with pytest.raises(JobsError, match="every GPU"):
        validate_request({**config(), "privileged": True, "gpus_per_worker": 1})


@pytest.mark.parametrize("value", [None, True, "-1", "NaN", "Infinity", "1e3", "1GiB"])
def test_invalid_resource_quantities_never_reach_preview(value):
    with pytest.raises(JobsError, match="unsupported resource quantity"):
        quantity(value)


@pytest.mark.parametrize("kind", ["cpu", "memory"])
@pytest.mark.parametrize("reserved,limit", [("0", "1"), ("2", "1"), ("1", "0")])
def test_resource_reservations_are_positive_and_bounded(kind, reserved, limit):
    value = config()
    value["resources"].update({kind + "_request": reserved, kind + "_limit": limit})
    with pytest.raises(JobsError, match="positive and no greater"):
        validate_request(value)


@pytest.mark.parametrize("replicas", [-1, True, "1", None])
def test_malformed_preview_replicas_fail_before_submission(replicas):
    obj = manifest()
    obj["spec"]["rayClusterSpec"]["workerGroupSpecs"][0]["replicas"] = replicas
    with pytest.raises(JobsError, match="invalid preview replica count"):
        validate_preview(config(), preview(obj))


@pytest.mark.parametrize(
    "path,value",
    [
        (("kind",), "Job"),
        (("metadata", "namespace"), "peer"),
        (("metadata", "labels", "kueue.x-k8s.io/queue-name"), "bypass"),
        (("metadata", "labels", "kueue.x-k8s.io/priority-class"), "q0"),
        (("metadata", "labels", "fleet.ai/requeue-if-preempted"), "true"),
        (("metadata", "annotations", "fleet.ai/run-dir"), "/mnt/sfs/peer"),
        (("metadata", "annotations", "fleet.ai/failure-alerts"), "on"),
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


@pytest.mark.parametrize("worker", [False, True])
@pytest.mark.parametrize(
    "fault",
    [
        "duplicate-env",
        "optional-secret",
        "prefixed-secret",
        "duplicate-secret",
        "failure-alerts",
    ],
)
def test_ambiguous_environment_blocks_the_actual_submit_boundary(tmp_path, worker, fault):
    request = {**config(), "workers": 2}
    obj = manifest(request)
    cluster = obj["spec"]["rayClusterSpec"]
    group = cluster["workerGroupSpecs"][0] if worker else cluster["headGroupSpec"]
    container = group["template"]["spec"]["containers"][0]
    if fault == "duplicate-env":
        # The old dict projection hid a conflicting earlier entry.
        container["env"].insert(0, {"name": "RUN_DIR", "value": "/mnt/sfs/jobs/other"})
    elif fault == "optional-secret":
        container["envFrom"][0]["secretRef"]["optional"] = True
    elif fault == "prefixed-secret":
        container["envFrom"][0]["prefix"] = "RENAMED_"
    elif fault == "duplicate-secret":
        container["envFrom"].append(deepcopy(container["envFrom"][0]))
    else:
        obj["metadata"]["annotations"].pop("fleet.ai/failure-alerts")
    calls = []

    def handler(req):
        calls.append((req.method, req.url.path))
        if req.method == "GET":
            return httpx.Response(200, json={"items": [], "has_more": False})
        if req.url.path.endswith("/preview"):
            return httpx.Response(200, json=preview(obj))
        return httpx.Response(202, json={"name": "researcher-sft-1234abcd"})

    journal = tmp_path / "intent.jsonl"
    with client(handler) as api, pytest.raises(JobsError, match="environment|Secret|failed-job"):
        api.submit_once(request, journal)
    assert calls == [("GET", "/v1/runs"), ("POST", "/v1/runs/preview")]
    assert not journal.exists()


def test_required_secret_can_explicitly_remain_nonoptional_without_renaming():
    obj = manifest()
    container = obj["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"]["containers"][0]
    container["envFrom"][0].update(prefix="")
    container["envFrom"][0]["secretRef"]["optional"] = False
    assert validate_preview(config(), preview(obj))["gpus"] == 8


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"manifest_yaml": "bad"},
        {"manifest_yaml": "["},
        {**preview(), "warnings": ["review"]},
    ],
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
        "synthetic-token",
        base_url="https://jobs.invalid",
        transport=httpx.MockTransport(handler),
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
    [
        [],
        {"items": []},
        {"items": [], "has_more": True},
        {"items": [{}], "has_more": False},
    ],
)
def test_incomplete_history_blocks_submission(payload, tmp_path):
    with (
        client(lambda req: httpx.Response(200, json=payload)) as api,
        pytest.raises(JobsError),
    ):
        api.submit_once(config(), tmp_path / "intent.jsonl")
    assert not (tmp_path / "intent.jsonl").exists()


@pytest.mark.parametrize(
    "outcome",
    [
        "ok",
        "timeout",
        "http-error",
        "invalid-json",
        "ambiguous",
        "wrong-root",
        "bad-job-id",
        "extra-field",
    ],
)
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
        if outcome == "invalid-json":
            return httpx.Response(202, text="private-trace-and-secret")
        response = creator_row()
        if outcome == "ambiguous":
            response["name"] = "unexpected"
        if outcome == "wrong-root":
            response["run_dir"] = "/mnt/sfs/other"
        if outcome == "bad-job-id":
            response["job_id"] = "not-a-uuid"
        if outcome == "extra-field":
            response["unexpected"] = True
        return httpx.Response(202, json=response)

    with client(handler) as api:
        if outcome == "ok":
            assert api.submit_once(config(), journal)["status"] == "QUEUED"
        else:
            with pytest.raises(JobsError) as error:
                api.submit_once(config(), journal)
            assert "private-trace-and-secret" not in str(error.value)
        with pytest.raises(JobsError, match="journal already exists"):
            api.submit_once(config(), journal)
    assert seen.count(("POST", "/v1/runs")) == 1
    assert "private-trace-and-secret" not in journal.read_text()
    assert journal.stat().st_mode & 0o777 == 0o600


def test_intent_file_and_parent_are_fsynced_before_the_sole_post(tmp_path, monkeypatch) -> None:
    events = []
    journal = tmp_path / "intent.jsonl"
    original_fsync = os.fsync

    def fsync(fd):
        events.append("fsync:dir" if stat.S_ISDIR(os.fstat(fd).st_mode) else "fsync:file")
        return original_fsync(fd)

    def handler(req):
        if req.method == "GET":
            return httpx.Response(200, json={"items": [], "has_more": False})
        if req.url.path.endswith("/preview"):
            return httpx.Response(200, json=preview())
        events.append("POST")
        raise httpx.ReadTimeout("uncertain", request=req)

    monkeypatch.setattr(jobs_module.os, "fsync", fsync)
    with client(handler) as api:
        with pytest.raises(JobsError, match="transport failed"):
            api.submit_once(config(), journal)
        with pytest.raises(JobsError, match="journal already exists"):
            api.submit_once(config(), journal)
    assert events[:3] == ["fsync:file", "fsync:dir", "POST"]
    assert stat.S_IMODE(journal.stat().st_mode) == 0o600


def test_before_intent_runs_after_final_preview_and_duplicate_scan(tmp_path) -> None:
    events = []
    journal = tmp_path / "intent.jsonl"
    evidence = "sha256:" + "e" * 64

    def handler(req):
        if req.method == "GET":
            events.append("history")
            return httpx.Response(200, json={"items": [], "has_more": False})
        if req.url.path.endswith("/preview"):
            events.append("preview")
            return httpx.Response(200, json=preview())
        events.append("POST")
        assert journal.exists()
        return httpx.Response(202, json=creator_row())

    def before_intent(proof):
        events.append("before-intent")
        assert proof["gpus"] == 8
        assert not journal.exists()
        return evidence

    with client(handler) as api:
        assert api.submit_once(config(), journal, before_intent=before_intent)["status"] == "QUEUED"
    assert events == ["history", "preview", "history", "before-intent", "POST"]
    intent, response, _ = jobs_module.read_submission_journal(
        journal,
        config(),
        expected_manifest_sha256=validate_preview(config(), preview())["manifest_sha256"],
        expected_intent_evidence_file_sha256=evidence,
    )
    assert intent["intent_evidence_file_sha256"] == evidence
    assert response is not None


def test_before_intent_failure_leaves_no_intent_and_no_post(tmp_path) -> None:
    journal = tmp_path / "intent.jsonl"
    calls = []

    def handler(req):
        if req.method == "GET":
            return httpx.Response(200, json={"items": [], "has_more": False})
        if req.url.path.endswith("/preview"):
            return httpx.Response(200, json=preview())
        calls.append("POST")
        return httpx.Response(202, json=creator_row())

    def stop(_proof):
        calls.append("callback")
        raise JobsError("fresh Kubernetes absence failed")

    with client(handler) as api, pytest.raises(JobsError, match="Kubernetes absence"):
        api.submit_once(config(), journal, before_intent=stop)
    assert calls == ["callback"]
    assert not journal.exists()
    assert not list(tmp_path.glob(".intent.jsonl.*.tmp"))


def test_before_intent_cannot_mutate_reviewed_request_or_preview(tmp_path) -> None:
    journal = tmp_path / "intent.jsonl"
    request = config()
    calls = []

    def handler(req):
        if req.method == "GET":
            return httpx.Response(200, json={"items": [], "has_more": False})
        if req.url.path.endswith("/preview"):
            return httpx.Response(200, json=preview())
        calls.append("POST")
        return httpx.Response(202, json=creator_row())

    def mutate(proof):
        request["image"] = "attacker/image:latest"
        proof["image"] = request["image"]
        return "sha256:" + "f" * 64

    with client(handler) as api, pytest.raises(JobsError, match="mutated"):
        api.submit_once(request, journal, before_intent=mutate)
    assert calls == []
    assert not journal.exists()


def test_expired_authority_stops_before_callback_intent_and_post(tmp_path, monkeypatch) -> None:
    journal = tmp_path / "intent.jsonl"
    calls = []

    def handler(req):
        if req.method == "GET":
            return httpx.Response(200, json={"items": [], "has_more": False})
        if req.url.path.endswith("/preview"):
            return httpx.Response(200, json=preview())
        calls.append("POST")
        return httpx.Response(202, json=creator_row())

    monkeypatch.setattr(jobs_module.time, "time", lambda: 1000.0)
    with client(handler) as api, pytest.raises(JobsError, match="expired"):
        api.submit_once(
            config(),
            journal,
            before_intent=lambda _proof: calls.append("callback") or ("sha256:" + "f" * 64),
            not_after_epoch=1029,
        )
    assert calls == []
    assert not journal.exists()


def test_authority_expiring_after_intent_fsync_stops_before_post(tmp_path, monkeypatch) -> None:
    journal = tmp_path / "intent.jsonl"
    calls = []

    def handler(req):
        if req.method == "GET":
            return httpx.Response(200, json={"items": [], "has_more": False})
        if req.url.path.endswith("/preview"):
            return httpx.Response(200, json=preview())
        calls.append("POST")
        return httpx.Response(202, json=creator_row())

    monkeypatch.setattr(
        jobs_module.time,
        "time",
        lambda: 1031.0 if journal.exists() else 1000.0,
    )
    with client(handler) as api, pytest.raises(JobsError, match="after durable intent"):
        api.submit_once(config(), journal, not_after_epoch=1030)
    assert calls == []
    assert journal.exists()


def test_restrictive_umask_cannot_poison_the_submission_journal(tmp_path) -> None:
    journal = tmp_path / "intent.jsonl"
    posts = []

    def handler(req):
        if req.method == "GET":
            return httpx.Response(200, json={"items": [], "has_more": False})
        if req.url.path.endswith("/preview"):
            return httpx.Response(200, json=preview())
        posts.append(req.url.path)
        return httpx.Response(202, json=creator_row())

    previous = os.umask(0o777)
    try:
        with client(handler) as api:
            api.submit_once(config(), journal)
    finally:
        os.umask(previous)
    assert posts == ["/v1/runs"]
    assert stat.S_IMODE(journal.stat().st_mode) == 0o600


def test_post_link_temp_cleanup_fsync_failure_cannot_suppress_the_post(
    tmp_path, monkeypatch
) -> None:
    journal = tmp_path / "intent.jsonl"
    posts = []
    directory_fsyncs = 0
    original_fsync = os.fsync

    def fsync(descriptor):
        nonlocal directory_fsyncs
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            directory_fsyncs += 1
            if directory_fsyncs == 2:
                raise OSError("injected cleanup-only directory fsync failure")
        return original_fsync(descriptor)

    def handler(req):
        if req.method == "GET":
            return httpx.Response(200, json={"items": [], "has_more": False})
        if req.url.path.endswith("/preview"):
            return httpx.Response(200, json=preview())
        posts.append(req.url.path)
        return httpx.Response(202, json=creator_row())

    monkeypatch.setattr(jobs_module.os, "fsync", fsync)
    with client(handler) as api:
        result = api.submit_once(config(), journal)
    assert result["status"] == "QUEUED"
    assert posts == ["/v1/runs"]
    assert directory_fsyncs == 2
    assert not list(tmp_path.glob(".intent.jsonl.*.tmp"))


def test_post_link_temp_unlink_failure_cannot_replace_the_creator_result(
    tmp_path, monkeypatch
) -> None:
    journal = tmp_path / "intent.jsonl"
    posts = []
    injected = False
    original_unlink = os.unlink

    def unlink(path, *args, **kwargs):
        nonlocal injected
        if not injected and str(path).startswith(".intent.jsonl."):
            injected = True
            raise OSError("injected cleanup-only unlink failure")
        return original_unlink(path, *args, **kwargs)

    def handler(req):
        if req.method == "GET":
            return httpx.Response(200, json={"items": [], "has_more": False})
        if req.url.path.endswith("/preview"):
            return httpx.Response(200, json=preview())
        posts.append(req.url.path)
        return httpx.Response(202, json=creator_row())

    monkeypatch.setattr(jobs_module.os, "unlink", unlink)
    with client(handler) as api:
        result = api.submit_once(config(), journal)
        with pytest.raises(JobsError, match="journal already exists"):
            api.submit_once(config(), journal)
    assert injected is True
    assert result["status"] == "QUEUED"
    assert posts == ["/v1/runs"]


def test_submission_reader_is_blocked_until_the_writer_finishes_response(tmp_path) -> None:
    journal = tmp_path / "intent.jsonl"
    request = config()
    manifest_sha256 = validate_preview(request, preview())["manifest_sha256"]
    blocked = []

    def handler(req):
        if req.method == "GET":
            return httpx.Response(200, json={"items": [], "has_more": False})
        if req.url.path.endswith("/preview"):
            return httpx.Response(200, json=preview())
        with pytest.raises(JobsError, match="publisher is still active"):
            jobs_module.read_submission_journal(
                journal,
                request,
                expected_manifest_sha256=manifest_sha256,
                expected_intent_evidence_file_sha256=None,
            )
        blocked.append(True)
        return httpx.Response(202, json=creator_row())

    with client(handler) as api:
        result = api.submit_once(request, journal)
    assert blocked == [True]
    _, response, _ = jobs_module.read_submission_journal(
        journal,
        request,
        expected_manifest_sha256=manifest_sha256,
        expected_intent_evidence_file_sha256=None,
    )
    assert response == result


def test_reviewed_preview_digest_accepts_only_allowed_root_metadata_drift(tmp_path) -> None:
    request = config()
    reviewed = validate_preview(request, preview())["manifest_sha256"]
    rendered = manifest()
    rendered["metadata"].update(
        uid=str(uuid.uuid4()),
        resourceVersion="12",
        creationTimestamp="2026-09-24T12:00:00Z",
    )
    rendered["status"] = {"jobStatus": "PENDING"}
    posts = []

    def handler(req):
        if req.method == "GET":
            return httpx.Response(200, json={"items": [], "has_more": False})
        if req.url.path.endswith("/preview"):
            return httpx.Response(200, json=preview(rendered))
        posts.append(req.url.path)
        return httpx.Response(202, json=creator_row())

    with client(handler) as api:
        api.submit_once(
            request,
            tmp_path / "intent.jsonl",
            expected_preview_manifest_sha256=reviewed,
        )
    assert posts == ["/v1/runs"]


@pytest.mark.parametrize("drift", ["annotation", "image", "command"])
def test_reviewed_preview_semantic_drift_blocks_before_intent_and_post(tmp_path, drift) -> None:
    request = config()
    reviewed = validate_preview(request, preview())["manifest_sha256"]
    rendered = manifest()
    container = rendered["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"][
        "containers"
    ][0]
    if drift == "annotation":
        rendered["metadata"]["annotations"]["fleet.ai/run-dir"] = "/mnt/sfs/jobs/other"
    elif drift == "image":
        container["image"] = "registry/image@sha256:" + "b" * 64
    else:
        rendered["spec"]["entrypoint"] = "python other.py"
    posts = []

    def handler(req):
        if req.method == "GET":
            return httpx.Response(200, json={"items": [], "has_more": False})
        if req.url.path.endswith("/preview"):
            return httpx.Response(200, json=preview(rendered))
        posts.append(req.url.path)
        return httpx.Response(202, json=creator_row())

    journal = tmp_path / "intent.jsonl"
    with client(handler) as api, pytest.raises(JobsError):
        api.submit_once(
            request,
            journal,
            expected_preview_manifest_sha256=reviewed,
        )
    assert posts == []
    assert not journal.exists()


def test_reviewed_privileged_preview_warning_drift_blocks_before_intent(tmp_path) -> None:
    request = config()
    request["privileged"] = True
    rendered = manifest(request)
    valid = preview(rendered)
    valid.update(errors=None, warnings=[PRIVILEGED_WHOLE_NODE_WARNING])
    reviewed = validate_preview(request, valid)["manifest_sha256"]
    bad = {**valid, "warnings": []}
    journal = tmp_path / "intent.jsonl"

    def handler(req):
        if req.method == "GET":
            return httpx.Response(200, json={"items": [], "has_more": False})
        if req.url.path.endswith("/preview"):
            return httpx.Response(200, json=bad)
        raise AssertionError("unreviewed warning policy reached the create POST")

    with client(handler) as api, pytest.raises(JobsError, match="warning policy"):
        api.submit_once(
            request,
            journal,
            expected_preview_manifest_sha256=reviewed,
        )
    assert not journal.exists()


def test_duplicate_appearing_after_preview_blocks_before_intent(tmp_path) -> None:
    reads = 0

    def handler(req):
        nonlocal reads
        if req.method == "GET":
            reads += 1
            items = [] if reads == 1 else [{"name": "researcher-sft-1234abcd"}]
            return httpx.Response(200, json={"items": items, "has_more": False})
        if req.url.path.endswith("/preview"):
            return httpx.Response(200, json=preview())
        raise AssertionError("late duplicate reached the create POST")

    with client(handler) as api, pytest.raises(JobsError, match="appeared after preview"):
        api.submit_once(config(), tmp_path / "intent.jsonl")
    assert not (tmp_path / "intent.jsonl").exists()


def test_unrelated_history_does_not_block_a_new_create_once_run(tmp_path):
    seen = []

    def handler(req):
        seen.append((req.method, req.url.path))
        if req.method == "GET":
            return httpx.Response(
                200,
                json={
                    "items": [{"name": "peer-run", "run_dir": "/mnt/sfs/jobs/peer"}],
                    "has_more": False,
                },
            )
        if req.url.path.endswith("/preview"):
            return httpx.Response(200, json=preview())
        return httpx.Response(202, json=creator_row())

    with client(handler) as api:
        result = api.submit_once(config(), tmp_path / "intent.jsonl")
    assert result["status"] == "QUEUED"
    assert seen.count(("POST", "/v1/runs")) == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", "other-1234abcd"),
        ("job_id", str(uuid.uuid4()).upper()),
        ("run_dir", "/mnt/sfs/jobs/other"),
        ("image", "registry/image@sha256:" + "b" * 64),
        ("priority_class", "c2"),
        ("queue_priority_class", "q0"),
        ("requeueIfPreempted", True),
        ("status", ""),
        ("message", {}),
    ],
)
def test_creator_response_rejects_every_identity_or_policy_drift(field, value) -> None:
    with pytest.raises(JobsError, match="creator"):
        validate_creator_response(config(), creator_row(**{field: value}))


def test_creator_response_is_safely_projected_from_exact_flat_contract() -> None:
    row = creator_row(message="private server detail", priority_reason="admitted")
    assert validate_creator_response(config(), row) == {
        "name": row["name"],
        "job_id": row["job_id"],
        "run_dir": row["run_dir"],
        "status": row["status"],
    }
    with pytest.raises(JobsError, match="fields changed"):
        validate_creator_response(config(), {**row, "unexpected": True})


@pytest.mark.parametrize(
    "record",
    [
        {"name": "old", "run_dir": config()["run_dir"]},
        {"name": "researcher-sft-1234abcd"},
        {"name": "other", "title": config()["title"]},
    ],
)
def test_history_duplicate_is_never_resubmitted(record, tmp_path):
    def handler(req):
        assert req.method == "GET"
        return httpx.Response(200, json={"items": [record], "has_more": False})

    with client(handler) as api, pytest.raises(JobsError, match="already owns"):
        api.submit_once(config(), tmp_path / "intent.jsonl")


def test_status_is_allowlisted_and_name_cannot_inject_route():
    with pytest.raises(JobsError, match="token is required"):
        Jobs("")
    assert "private" not in str(safe_status({"name": "safe", "failure_message": "private"}))
    with client(
        lambda req: httpx.Response(200, json={"name": "safe", "status": "SUCCEEDED"})
    ) as api:
        assert api.status("safe")["status"] == "SUCCEEDED"
        with pytest.raises(JobsError):
            api.status("../runs")


def test_delete_accepts_exact_empty_204_and_never_parses_json():
    seen = []

    def handler(req):
        seen.append((req.method, req.url.path))
        return httpx.Response(204)

    with client(handler) as api:
        assert api.delete("owned-run-1234abcd") == {
            "name": "owned-run-1234abcd",
            "deleted": True,
            "http_status": 204,
        }
    assert seen == [("DELETE", "/v1/runs/owned-run-1234abcd")]


@pytest.mark.parametrize(
    "response",
    [httpx.Response(200, json={}), httpx.Response(202), httpx.Response(204, content=b"unexpected")],
)
def test_delete_rejects_every_noncanonical_success_response(response):
    with (
        client(lambda req: response) as api,
        pytest.raises(JobsError, match="reconcile|unexpected"),
    ):
        api.delete("owned-run-1234abcd")


def test_delete_rejects_invalid_name_before_network():
    with (
        client(lambda req: pytest.fail("network must not be called")) as api,
        pytest.raises(JobsError, match="invalid run name"),
    ):
        api.delete("../runs")
