"""Offline contract tests for the zero-update development topology probe."""

from __future__ import annotations

import copy
import os
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest
import yaml

from cyber_post_train import cli
from cyber_post_train.jobs import JobsError, digest
from training import skyrl_topology_probe as probe

ROOT = probe.ROOT
CONFIG = ROOT / "configs/qualification/qwen38-skyrl-topology-probe-dev-v1.json"


@pytest.fixture
def plan():
    return probe.compile_probe(CONFIG)


def test_probe_is_distinct_dev_only_bounded_and_zero_update(plan) -> None:
    request = probe.request(plan)
    assert plan["schema"] == probe.SCHEMA
    assert plan["execution"]["cluster_target"] == "dev"
    assert plan["execution"]["jobs_api_base_url"] == "https://api.ft.dev.flt.build"
    assert (request["workers"], request["gpus_per_worker"]) == (1, 8)
    assert request["secrets"] == []
    assert plan["deadlines"] == {
        "setup_seconds": 1200,
        "cleanup_seconds": 300,
        "total_seconds": 1500,
    }
    assert plan["scientific_work"] == {
        "task_rows": 0,
        "rollout_episodes": 0,
        "verifier_calls": 0,
        "optimizer_steps": 0,
        "checkpoints": 0,
    }
    gate = plan["qualification"]["submission_gate"]
    assert gate["preview_authorized"] is True
    assert gate["submission_authorized"] is False
    assert request["env"]["CYBER_EXPECTED_RUNTIME_UID"] == "1000"
    assert request["env"]["CYBER_EXPECTED_RUNTIME_GID"] == "100"


def test_probe_runtime_bundle_imports_without_source_checkout(plan, tmp_path) -> None:
    bundle = tmp_path / "bundle"
    for name, content in probe._runtime().items():
        path = bundle / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    for directory in ("training", "cyber_post_train", "evals", "evals/fleet"):
        (bundle / directory / "__init__.py").write_text("")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from training import skyrl_topology_probe as p; assert p.SCHEMA.endswith('_v1')",
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(bundle)},
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr.decode()


@pytest.mark.parametrize(
    "field,value",
    [
        ("cluster_target", "prod"),
        ("jobs_api_base_url", "https://api.ft.flt.build"),
        ("workers", 2),
        ("gpus_per_worker", 4),
    ],
)
def test_probe_rejects_topology_or_route_substitution(plan, field, value) -> None:
    plan["execution"][field] = value
    with pytest.raises(ValueError, match="plan changed"):
        probe.request(plan)


def test_probe_preflight_parses_engine_without_tasks_or_gpu(plan, monkeypatch) -> None:
    monkeypatch.setattr(probe.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(probe.os, "getegid", lambda: 100)
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg = NS(generator=NS(inference_engine=object()))
    monkeypatch.setattr(probe.skyrl, "diagnostic_native_config", lambda _: cfg)
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "skyrl.backends.skyrl_train.inference_servers.utils",
        NS(build_vllm_cli_args=lambda value: calls.append(value)),
    )
    receipt = probe.preflight(plan)
    assert receipt["schema"] == "cyber_skyrl_topology_probe_cpu_preflight_v1"
    assert receipt["request_sha256"] == digest(probe.request(plan))
    assert calls == [cfg]
    assert receipt["task_rows_read"] == 0
    assert receipt["rollout_episodes"] == receipt["optimizer_steps"] == 0


def _preview(request, *, identity=True):
    context = (
        {"runAsUser": 1000, "runAsGroup": 100, "runAsNonRoot": True}
        if identity
        else {}
    )
    container = {
        "image": request["image"],
        "securityContext": {"privileged": False, **context},
        "resources": {
            "requests": {
                "cpu": request["resources"]["cpu_request"],
                "memory": request["resources"]["memory_request"],
                "nvidia.com/gpu": 8,
            },
            "limits": {
                "cpu": request["resources"]["cpu_limit"],
                "memory": request["resources"]["memory_limit"],
                "nvidia.com/gpu": 8,
            },
        },
        "env": [
            {"name": key, "value": value}
            for key, value in {**request["env"], "RUN_DIR": request["run_dir"]}.items()
        ],
    }
    template = {
        "spec": {
            "priorityClassName": "c1",
            "containers": [container],
            "imagePullSecrets": [],
        }
    }
    manifest = {
        "kind": "RayJob",
        "metadata": {
            "namespace": "fleet-train-jobs",
            "labels": {
                "kueue.x-k8s.io/queue-name": "training-lq",
                "kueue.x-k8s.io/priority-class": "q1",
                "fleet.ai/requeue-if-preempted": "false",
            },
            "annotations": {"fleet.ai/run-dir": request["run_dir"]},
        },
        "spec": {
            "suspend": True,
            "shutdownAfterJobFinishes": True,
            "entrypoint": request["command"],
            "rayClusterSpec": {
                "headGroupSpec": {"template": template},
                "workerGroupSpecs": [],
            },
        },
    }
    return {"manifest_yaml": yaml.safe_dump(manifest), "warnings": []}


def test_probe_preview_requires_effective_uid_gid(plan) -> None:
    request = probe.request(plan)
    result = probe.validate_preview(plan, request, _preview(request))
    assert result["runtime_user"] == {"uid": 1000, "gid": 100}
    with pytest.raises(JobsError, match="does not prove runtime user"):
        probe.validate_preview(plan, request, _preview(request, identity=False))


def test_release_receipt_requires_exact_terminal_absence_and_zero_gpus(plan) -> None:
    receipt = probe._seal(
        {
            "schema": probe.RECEIPT_SCHEMA,
            "status": "setup_and_internal_cleanup_passed",
            "plan_sha256": digest(plan),
        }
    )
    observation = {
        "api_run_id": "run-uuid",
        "rayjob_uid": "rayjob-uuid",
        "workload_uid": "workload-uuid",
        "pod_uid": "pod-uuid",
        "terminal_status": "SUCCEEDED",
        "rayjob_present": False,
        "workload_present": False,
        "pod_present": False,
        "active_gpus": 0,
    }
    release = probe.validate_release(plan, receipt, observation)
    assert release["schema"] == probe.RELEASE_SCHEMA and release["status"] == "released"
    for field, value in (
        ("pod_present", True),
        ("active_gpus", 8),
        ("terminal_status", "RUNNING"),
    ):
        changed = copy.deepcopy(observation)
        changed[field] = value
        with pytest.raises(ValueError, match="release was not proven"):
            probe.validate_release(plan, receipt, changed)


def test_probe_cli_prepares_but_external_gate_stays_closed(tmp_path) -> None:
    output = tmp_path / "prepared"
    result = cli.app.registered_commands  # prove command registration without network
    assert result
    from typer.testing import CliRunner

    response = CliRunner().invoke(
        cli.app, ["rl-topology-probe", str(CONFIG), "--output", str(output)]
    )
    assert response.exit_code == 0, response.output
    plan, _ = cli._prepared(output)
    cli._external_action_gate(plan, "preview")
    with pytest.raises(ValueError, match="submit blocked by qualification gate"):
        cli._external_action_gate(plan, "submit")
