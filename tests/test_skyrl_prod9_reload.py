"""Synthetic prod9 one-GPU reload rail tests; no cluster or API mutations."""

from __future__ import annotations

import base64
import copy
import gzip
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from cyber_post_train.gpu_capacity import build_capacity_census
from cyber_post_train.jobs import digest
from training import skyrl_prod9_direct as direct
from training import skyrl_prod9_hardening as hardening
from training import skyrl_prod9_reload as reload


def _spec() -> dict:
    name = "chris-q38-skyrl10-a1-p10-reload-v1"
    plan_sha256 = "1" * 64
    checkpoint_sha256 = "2" * 64
    manifest_file_sha256 = "3" * 64
    return reload._seal(
        {
            "schema": reload.SPEC_SCHEMA,
            "plan_sha256": plan_sha256,
            "name": name,
            "run_dir": "/mnt/sfs/jobs/" + name,
            "image": reload.IMAGE,
            "model": {
                "repo": "Qwen/Qwen3.8-27B",
                "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
                "base_root": "/mnt/sfs/models/qwen3.8-27b-1d4bf0f2",
                "export_root": "/mnt/sfs/jobs/chris-q38-skyrl10-a1/hf-export-step10-v1",
            },
            "checkpoint": {
                "manifest_path": (
                    "/mnt/sfs/jobs/chris-q38-skyrl10-a1/checkpoint-seals-v1/step-10.json"
                ),
                "manifest_file_sha256": manifest_file_sha256,
                "receipt_sha256": checkpoint_sha256,
                "checkpoint_path": (
                    "/mnt/sfs/jobs/chris-q38-skyrl10-a1/checkpoints/global_step_10"
                ),
                "optimizer_step": 10,
            },
            "export": {
                "receipt_path": (
                    "/mnt/sfs/jobs/chris-q38-skyrl10-a1/hf-export-step10-v1/EXPORT.json"
                ),
                "file_sha256": "4" * 64,
                "receipt_sha256": "5" * 64,
                "source_checkpoint_receipt_sha256": checkpoint_sha256,
                "source_manifest_file_sha256": manifest_file_sha256,
                "source_plan_sha256": plan_sha256,
                "optimizer_step": 10,
            },
            "runtime": {
                "module": reload.MODULE,
                "files_sha256": reload._runtime_hashes(),
                "hard_child_seconds": reload.HARD_CHILD_SECONDS,
            },
            "resources": {
                "nodes": 1,
                "gpus": 1,
                "priority_class": "c1",
                "queue_priority_class": "q1",
                "maximum_seconds": reload.MAXIMUM_SECONDS,
            },
            "scientific_work": {
                "optimizer_steps": 0,
                "synthetic_only": True,
                "serving_qualified": False,
            },
        }
    )


def _source_preview(spec: dict, request: dict) -> dict:
    placeholder = request["name"] + "-00000000"
    env = [
        {"name": key, "value": value}
        for key, value in sorted(
            {
                **request["env"],
                "RUN_DIR": request["run_dir"],
                "FLEET_RUN_ID": "00000000-0000-0000-0000-000000000000",
                "FLEET_RUN_NAME": placeholder,
            }.items()
        )
    ]
    resources = request["resources"]
    value = {
        "apiVersion": "ray.io/v1",
        "kind": "RayJob",
        "metadata": {
            "name": placeholder,
            "namespace": reload.NAMESPACE,
            "labels": {
                "app": "fleet-rl-job",
                "fleet.ai/requeue-if-preempted": "false",
                "fleet.ai/run-id": "00000000-0000-0000-0000-000000000000",
                "fleet.ai/run-name": request["name"],
                "kueue.x-k8s.io/queue-name": "training-lq",
                "kueue.x-k8s.io/priority-class": "q1",
            },
            "annotations": {
                "fleet.ai/run-id": "00000000-0000-0000-0000-000000000000",
                "fleet.ai/job-image": reload.IMAGE,
                "fleet.ai/run-dir": spec["run_dir"],
                reload.FAILURE_ALERT_ANNOTATION: reload.FAILURE_ALERT_OFF,
            },
        },
        "spec": {
            "backoffLimit": 0,
            "entrypoint": request["command"],
            "submissionMode": "HTTPMode",
            "suspend": True,
            "shutdownAfterJobFinishes": True,
            "rayClusterSpec": {
                "headGroupSpec": {
                    "template": {
                        "metadata": {},
                        "spec": {
                            "priorityClassName": "c1",
                            "containers": [
                                {
                                    "name": "ray-head",
                                    "image": reload.IMAGE,
                                    "env": env,
                                    "envFrom": [
                                        {"secretRef": {"name": placeholder + "-fleet-key"}}
                                    ],
                                    "securityContext": direct._runtime_context(),
                                    "resources": {
                                        "requests": {
                                            "cpu": resources["cpu_request"],
                                            "memory": resources["memory_request"],
                                            "nvidia.com/gpu": 1,
                                        },
                                        "limits": {
                                            "cpu": resources["cpu_limit"],
                                            "memory": resources["memory_limit"],
                                            "nvidia.com/gpu": 1,
                                        },
                                    },
                                }
                            ],
                            "initContainers": [
                                {
                                    "name": "sfs-init",
                                    "command": [
                                        "sh",
                                        "-c",
                                        (
                                            f"mkdir -p {spec['run_dir']} && chown 1000:100 "
                                            f"{spec['run_dir']}"
                                        ),
                                    ],
                                }
                            ],
                        },
                    }
                }
            },
        },
    }
    return {
        "name": placeholder,
        "warnings": [],
        "manifest_yaml": yaml.safe_dump(value),
    }


def _server_render(value: dict) -> dict:
    result = copy.deepcopy(value)
    result["metadata"].update(
        {
            "creationTimestamp": "2026-09-21T00:00:00Z",
            "generation": 1,
            "uid": "00000000-0000-0000-0000-000000000001",
        }
    )
    result["spec"]["ttlSecondsAfterFinished"] = 0
    result["spec"]["rayClusterSpec"]["headGroupSpec"].update({"numOfHosts": 1, "scaleStrategy": {}})
    return result


def _observer(tmp_path: Path, spec: dict, request: dict, manifest: dict) -> dict:
    value = direct._seal(
        {
            "schema": "cyber_jobs_api_prefix_guard_armed_v1",
            "status": "armed_non_destructive_prefix_guard",
            "context": reload.PROD_CONTEXT,
            "namespace": reload.NAMESPACE,
            "run_name_prefix": request["name"],
            "generated_name_pattern": "^" + re.escape(request["name"]) + r"-[a-f0-9]{8}$",
            "run_dir": request["run_dir"],
            "image": request["image"],
            "maximum_seconds": reload.MAXIMUM_SECONDS,
            "expected_gpus": 1,
            "plan_sha256": "sha256:" + spec["plan_sha256"],
            "manifest_sha256": "sha256:" + digest(manifest),
            "armed_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "observer_pid": os.getpid(),
            "prefix_collision_count_before_post": 0,
        }
    )
    direct.jobs_api_guard_path(tmp_path, "reload").write_text(json.dumps(value))
    return value


def _capacity(_context=None, **kwargs) -> dict:
    return build_capacity_census(
        {"items": []},
        {"items": []},
        {"items": []},
        {"items": []},
        observed_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **kwargs,
    )


def _bundle(request: dict) -> dict:
    encoded = request["env"].get("CYBER_RUNTIME_BUNDLE")
    if encoded is None:
        parts = sorted(
            (
                (int(key.rsplit("_", 1)[1]), value)
                for key, value in request["env"].items()
                if key.startswith("CYBER_RUNTIME_BUNDLE_")
            )
        )
        encoded = "".join(value for _, value in parts)
    return json.loads(gzip.decompress(base64.b64decode(encoded, validate=True)))


def _prepared(tmp_path: Path, monkeypatch):
    spec = _spec()
    request = reload.job_request(spec)
    assert request["command"].startswith(
        "mkdir " + spec["run_dir"] + "/.prod9-reload-create-claim-v1 && exec "
    )
    source = _source_preview(spec, request)
    expected = reload.manifest(spec, request, source)
    rendered = _server_render(expected)
    dev = reload.validate_preview(
        spec, request, source, expected, rendered, context=reload.DEV_CONTEXT
    )
    prod = reload.validate_preview(
        spec, request, source, expected, rendered, context=reload.PROD_CONTEXT
    )
    global_root = tmp_path / "global-create-once"
    global_root.mkdir()
    monkeypatch.setattr(hardening, "CREATE_ONCE_ROOT", global_root)
    operation_root = hardening.reload_operation_root(spec)
    operation_root.mkdir()
    observer = _observer(operation_root, spec, request, expected)
    monkeypatch.setattr(direct.os, "kill", lambda *_: None)
    auth = reload.authorize(
        spec,
        request,
        source,
        expected,
        dev_preview=dev,
        prod_preview=prod,
        observer=observer,
    )
    return spec, request, source, expected, auth


def test_reload_bundle_is_hermetic_and_manifest_is_exact_one_gpu(tmp_path) -> None:
    spec = _spec()
    request = reload.job_request(spec)
    bundle = _bundle(request)
    root = tmp_path / "bundle"
    for relative, source in bundle["files"].items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    imported = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                "import importlib,json,sys;"
                f"sys.path.insert(0,{str(root)!r});"
                "m=importlib.import_module('training.skyrl_prod9_reload');"
                "[importlib.import_module(n) for n in ("
                "'training.checkpoints','training.export_check',"
                "'training.post_sft_artifacts','training.sft_runtime')];"
                "m._spec_identity(json.load(open('spec.json')))"
            ),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert imported.returncode == 0, imported.stderr

    expected = reload.manifest(spec, request, _source_preview(spec, request))
    assert expected["metadata"]["name"] == request["name"] + "-00000000"
    assert expected["metadata"]["annotations"][reload.FAILURE_ALERT_ANNOTATION] == "off"
    cluster = expected["spec"]["rayClusterSpec"]
    assert cluster.get("workerGroupSpecs") in (None, [])
    pod = cluster["headGroupSpec"]["template"]["spec"]
    assert pod["priorityClassName"] == "c1"
    assert pod["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] == 1
    assert pod["containers"][0]["envFrom"] == [
        {"secretRef": {"name": request["name"] + "-00000000-fleet-key"}}
    ]
    assert expected["spec"]["backoffLimit"] == 0


def test_full_reload_authorization_is_no_mutation_and_wrong_root_stops_before_io(
    tmp_path, monkeypatch
) -> None:
    spec, request, source, expected, auth = _prepared(tmp_path, monkeypatch)
    other = tmp_path / "other"
    other.mkdir()
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("no external read or mutation is permitted")

    with pytest.raises(ValueError, match="operation root"):
        reload.create_once(
            other,
            spec,
            request,
            source,
            expected,
            auth,
            token="synthetic",
            runner=forbidden,
            jobs_factory=object(),
            capacity_reader=lambda **_: (_ for _ in ()).throw(AssertionError),
        )
    assert calls == []
    assert list(other.iterdir()) == []


def test_reload_create_is_exactly_once_and_publishes_creator_uid(tmp_path, monkeypatch) -> None:
    spec, request, source, expected, auth = _prepared(tmp_path, monkeypatch)
    operation_root = Path(auth["operation_root"])
    posts = []
    actual_name = request["name"] + "-1a2b3c4d"
    job_id = "00000000-0000-0000-0000-000000000019"

    class FakeJobs:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def all_runs(self):
            return []

        def preview(self, submitted):
            assert submitted == request
            return source

        def request(self, method, path, **kwargs):
            assert (method, path, kwargs.get("json")) == ("POST", "/v1/runs", request)
            posts.append((method, path))
            return {
                "name": actual_name,
                "job_id": job_id,
                "run_dir": request["run_dir"],
                "status": "queued",
            }

    def runner(command, **kwargs):
        if "get" in command and actual_name in command:
            created = copy.deepcopy(expected)
            created["metadata"].update(
                {
                    "name": actual_name,
                    "uid": "00000000-0000-0000-0000-000000000020",
                    "creationTimestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
            )
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps(created), stderr="")
        if "get" in command:
            return subprocess.CompletedProcess(command, 0, stdout='{"items":[]}', stderr="")
        if "--dry-run=server" in command:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(_server_render(json.loads(kwargs["input"]))),
                stderr="",
            )
        pytest.fail(f"unexpected mutating command: {command}")

    proof = reload.create_once(
        operation_root,
        spec,
        request,
        source,
        expected,
        auth,
        token="synthetic",
        runner=runner,
        jobs_factory=FakeJobs,
        capacity_reader=_capacity,
    )
    assert proof["status"] == "submitted_once_and_bound_exact_uid"
    assert proof["rayjob_uid"] == "00000000-0000-0000-0000-000000000020"
    assert posts == [("POST", "/v1/runs")]
    binding = json.loads(hardening.creator_binding_path(operation_root, "reload").read_text())
    assert binding["rayjob_uid"] == proof["rayjob_uid"]
    assert binding["jobs_api_run_name"] == actual_name

    before = len(posts)
    with pytest.raises(ValueError, match="never retry"):
        reload.create_once(
            operation_root,
            spec,
            request,
            source,
            expected,
            auth,
            token="synthetic",
            runner=runner,
            jobs_factory=FakeJobs,
            capacity_reader=_capacity,
        )
    assert len(posts) == before

    alternate = tmp_path / "fresh-alternate-operation"
    alternate.mkdir()
    changed = _observer(alternate, spec, request, expected)
    changed["observer_pid"] += 1
    changed = direct._seal(changed)
    with pytest.raises(ValueError, match="binding changed"):
        reload.authorize(
            spec,
            request,
            source,
            expected,
            dev_preview=auth["dev_preview"],
            prod_preview=auth["prod_preview"],
            observer=changed,
        )
    assert len(posts) == before


def test_reload_preview_rejects_root_alert_or_gpu_drift() -> None:
    spec = _spec()
    request = reload.job_request(spec)
    source = _source_preview(spec, request)
    expected = reload.manifest(spec, request, source)
    missing_alert = copy.deepcopy(source)
    missing_alert_manifest = yaml.safe_load(missing_alert["manifest_yaml"])
    missing_alert_manifest["metadata"]["annotations"].pop(reload.FAILURE_ALERT_ANNOTATION)
    missing_alert["manifest_yaml"] = yaml.safe_dump(missing_alert_manifest)
    with pytest.raises(ValueError, match="root alert-off"):
        reload.manifest(spec, request, missing_alert)
    for fault in ("alert", "gpu"):
        changed = _server_render(expected)
        if fault == "alert":
            changed["metadata"]["annotations"][reload.FAILURE_ALERT_ANNOTATION] = "on"
        else:
            changed["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"]["containers"][0][
                "resources"
            ]["limits"]["nvidia.com/gpu"] = 8
        with pytest.raises(ValueError, match="server preview"):
            reload.validate_preview(
                spec,
                request,
                source,
                expected,
                changed,
                context=reload.PROD_CONTEXT,
            )
