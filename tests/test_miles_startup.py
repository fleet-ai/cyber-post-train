"""The unfinished 32-GPU gate is inert and cannot accept caller summaries."""

import copy
import json
from pathlib import Path

import pytest

from cyber_post_train.jobs import digest
from training import miles_startup as startup

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def plan():
    return {
        "schema": startup.PLAN_SCHEMA,
        "gate_version": startup.GATE_VERSION,
        "status": "prepared",
        "launchable": True,
        "contract": startup.CONTRACT,
        "bindings": {
            "runtime_source_sha256": "sha256:" + "a" * 64,
            "checkpoint_sha256": "sha256:" + "b" * 64,
            "output_root": "/mnt/sfs/jobs/synthetic-miles-startup",
        },
        "execution": startup.EXECUTION,
        "expected_counters": startup.EXPECTED_COUNTERS,
        "receipt_schema": startup.RECEIPT_SCHEMA,
    }


@pytest.fixture
def prepared_request(plan):
    return {
        "name": "synthetic-miles-startup",
        "run_dir": plan["bindings"]["output_root"],
        "image": startup.RUNTIME_IMAGE,
        "command": "python -m exact_external_startup_runner",
        "workers": 4,
        "gpus_per_worker": 8,
        "priority_class": "c1",
        "requeueIfPreempted": False,
        "resources": {
            "cpu_request": "64",
            "cpu_limit": "128",
            "memory_request": "1536Gi",
            "memory_limit": "2048Gi",
        },
    }


def _seal(value):
    value["sha256"] = "sha256:" + digest(value)
    return value


def _forged_observations(plan, request):
    bindings = {
        "plan_sha256": "sha256:" + digest(plan),
        "request_sha256": "sha256:" + digest(request),
        "command_sha256": "sha256:" + digest(request["command"]),
        "runtime_source_sha256": plan["bindings"]["runtime_source_sha256"],
        "checkpoint_sha256": plan["bindings"]["checkpoint_sha256"],
        "runtime_image": startup.RUNTIME_IMAGE,
    }
    controller = _seal(
        {
            "schema": "cyber_miles_startup_controller_observation_v1",
            "observed_at": "2026-09-15T20:00:00Z",
            "bindings": bindings,
            "api": {
                "run_id": "20000000-0000-4000-8000-000000000000",
                "name": "unrelated-success-12345678",
                "status": "SUCCEEDED",
            },
            "kubernetes": {
                kind: {
                    "name": "unrelated-" + kind,
                    "uid": f"{index}0000000-0000-4000-8000-000000000000",
                }
                for index, kind in enumerate(("rayjob", "workload", "raycluster"), 3)
            }
            | {"namespace": "fleet-train-jobs"},
            "pods": [
                {
                    "name": "unrelated-pod",
                    "uid": f"10000000-0000-4000-8000-00000000000{index}",
                    "observed_image_id": "forged-prefix@" + startup.RUNTIME_IMAGE.rsplit("@", 1)[1],
                    "phase": "Succeeded",
                    "exit_code": 0,
                    "restart_count": 0,
                    "gpus": 8,
                }
                for index in range(1, 5)
            ],
        }
    )
    counters = _seal(
        {
            "schema": "cyber_miles_startup_counter_observation_v1",
            "observed_at": "2020-01-01T00:00:00Z",
            "bindings": bindings,
            "api_run_id": controller["api"]["run_id"],
            "output_root": plan["bindings"]["output_root"],
            "measurement": "read_only_artifact_inventory_and_jobs_metrics",
            "counters": startup.EXPECTED_COUNTERS,
            "episode_paths": [],
            "checkpoint_paths": [],
            "inventory_sha256": "sha256:" + "c" * 64,
        }
    )
    release = _seal(
        {
            "schema": "cyber_miles_startup_release_observation_v1",
            "observed_at": "2026-09-15T20:02:00Z",
            "bindings": bindings,
            "api_run_id": controller["api"]["run_id"],
            "namespace": "fleet-train-jobs",
            "jobs_api_readback_status": 404,
            "resource_queries": {},
            "gpus_allocated_before": 32,
            "gpus_allocated_after": 0,
            "inventory_sha256": "sha256:" + "d" * 64,
        }
    )
    return controller, counters, release


def test_committed_template_is_exact_strictly_typed_and_inert():
    path = ROOT / "configs/qualification/qwen38-miles-opencode-distributed-startup-v2.template.json"
    value = json.loads(path.read_text())
    assert startup.validate_template(value) is value
    assert value["launchable"] is False
    assert value["successor_reward_canary_allowed"] is False
    assert value["contract"]["distributed_backend"] == "cpu:gloo,cuda:nccl"
    assert value["contract"]["tensor_parallel_size"] == 8
    assert value["contract"]["pipeline_parallel_size"] == 1
    assert value["contract"]["context_parallel_size"] == 4
    assert value["contract"]["context_tokens"] == 262144
    assert value["contract"]["response_tokens"] == 245760


def test_template_rejects_python_numeric_aliases():
    path = ROOT / "configs/qualification/qwen38-miles-opencode-distributed-startup-v2.template.json"
    value = json.loads(path.read_text())
    value["gate_version"] = 2.0
    value["contract"]["pipeline_parallel_size"] = True
    value["expected_counters"] = {key: False for key in value["expected_counters"]}
    with pytest.raises(ValueError, match="exact inert contract"):
        startup.validate_template(value)


def test_forged_noop_summaries_cannot_create_acceptance(tmp_path, plan, prepared_request):
    request = copy.deepcopy(prepared_request)
    request.update(command="true", privileged=True, secrets=["unrelated-secret"])
    controller, counters, release = _forged_observations(plan, request)
    output = tmp_path / "ACCEPTED.json"
    with pytest.raises(ValueError, match="acceptance is disabled"):
        startup.seal_receipt(
            plan=plan,
            request=request,
            controller=controller,
            counters=counters,
            release=release,
            output=output,
        )
    assert not output.exists()


def test_no_concrete_plan_or_receipt_can_validate(plan, prepared_request):
    with pytest.raises(ValueError, match="acceptance is disabled"):
        startup.validate_plan(plan)
    with pytest.raises(ValueError, match="acceptance is disabled"):
        startup.validate_receipt({}, plan=plan, request=prepared_request)
