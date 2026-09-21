import copy
import json
import uuid
from pathlib import Path

import pytest
import yaml

from cyber_post_train.jobs import JobsError, validate_preview, validate_request
from training import miles96_mechanics_canary as mechanics


def _sha(char: str) -> str:
    return "sha256:" + char * 64


def _plan() -> dict:
    return mechanics.build_plan(
        name="q38-m96-canary-a1",
        reload_name="q38-m96-reload-a1",
        model_root="/mnt/sfs/jobs/q38-prepared-model-v1",
        model_binding_sha256=_sha("a"),
        task_binding={
            "task_key": "synthetic-blackbox-v1",
            "task_version_id": str(uuid.UUID("11111111-1111-4111-8111-111111111111")),
            "task_set_sha256": _sha("b"),
            "tool_catalog_sha256": _sha("c"),
            "authority_receipt_sha256": _sha("d"),
        },
    )


def _server_preview(request: dict) -> dict:
    pod = {
        "spec": {
            "priorityClassName": request["priority_class"],
            "imagePullSecrets": [{"name": item} for item in request["image_pull_secrets"]],
            "containers": [
                {
                    "image": request["image"],
                    "resources": {
                        "requests": {
                            "cpu": request["resources"]["cpu_request"],
                            "memory": request["resources"]["memory_request"],
                            "nvidia.com/gpu": request["gpus_per_worker"],
                        },
                        "limits": {
                            "cpu": request["resources"]["cpu_limit"],
                            "memory": request["resources"]["memory_limit"],
                            "nvidia.com/gpu": request["gpus_per_worker"],
                        },
                    },
                    "env": [
                        {"name": key, "value": value}
                        for key, value in {**request["env"], "RUN_DIR": request["run_dir"]}.items()
                    ],
                    "envFrom": [{"secretRef": {"name": item}} for item in request["secrets"]],
                    "securityContext": {"privileged": request["privileged"]},
                }
            ],
        }
    }
    obj = {
        "kind": "RayJob",
        "metadata": {
            "namespace": "fleet-train-jobs",
            "labels": {
                "kueue.x-k8s.io/queue-name": "training-lq",
                "kueue.x-k8s.io/priority-class": "q" + request["priority_class"][1:],
                "fleet.ai/requeue-if-preempted": "false",
            },
            "annotations": {
                "fleet.ai/run-dir": request["run_dir"],
                "fleet.ai/failure-alerts": "off",
            },
        },
        "spec": {
            "suspend": True,
            "shutdownAfterJobFinishes": True,
            "entrypoint": request["command"],
            "rayClusterSpec": {
                "headGroupSpec": {"template": pod},
                "workerGroupSpecs": [{"replicas": request["workers"] - 1, "template": pod}],
            },
        },
    }
    return {"manifest_yaml": yaml.safe_dump(obj), "warnings": []}


def _receipt(plan: dict) -> dict:
    body = {
        "schema": mechanics.TRAIN_RECEIPT_SCHEMA,
        "plan_sha256": "sha256:" + mechanics.digest(plan),
        "optimizer_steps": 1,
        "reward_filter": mechanics.REWARD_FILTER,
        "checkpoint_path_sha256": _sha("e"),
        "hf_export_path_sha256": _sha("f"),
        "hf_export_complete": True,
        "save_hook_after_checkpoint": True,
    }
    return {**body, "receipt_sha256": "sha256:" + mechanics.digest(body)}


def test_one_node_current_recipe_request_and_fresh_v1_row() -> None:
    plan = _plan()
    request = mechanics.job_request(plan)
    validate_request(request)
    assert request["image"] == mechanics.IMAGE
    assert request["workers"] == 1 and request["gpus_per_worker"] == 8
    assert request["privileged"] is True
    assert request["priority_class"] == "c1" and request["failureAlerts"] is False
    assert request["requeueIfPreempted"] is False
    assert request["run_dir"] == "/mnt/sfs/jobs/q38-m96-canary-a1"
    assert request["env"]["MILES_SCRIPT_EXTERNAL_RAY"] == "1"
    assert request["env"]["MODEL_DIR"] == plan["prepared_model"]["root"]
    assert plan["task_binding"]["task_key"] not in request["command"]
    assert "Run the versioned Fleet task." not in request["command"]

    row = json.loads(mechanics.task_rows(plan))
    assert row["messages"] == [{"role": "user", "content": "Run the versioned Fleet task."}]
    assert row["metadata"]["task_version_id"] == plan["task_binding"]["task_version_id"]
    assert set(row["metadata"]["fleet"]) == set(plan["episode"])

    argv = mechanics.native_arguments(plan)
    extra = argv[argv.index("--extra-args") + 1]
    worker_env = json.loads(argv[argv.index("--extra-env-vars") + 1])
    assert argv[:3] == ["-m", "fti.trainers.miles.run_fleet", "--model-name"]
    assert "qwen3.8-27b-256k" not in argv
    assert worker_env == {
        "PYTHONPATH": "/mnt/sfs/jobs/q38-m96-canary-a1/.runtime",
        "CYBER_RUNTIME_DIR": "/mnt/sfs/jobs/q38-m96-canary-a1/.runtime",
        "CYBER_PLAN_SHA256": mechanics.digest(plan),
    }
    for expected in (
        "--num-rollout 1",
        "--save-interval 1",
        "--save-hf /mnt/sfs/jobs/q38-m96-canary-a1/hf/step-{rollout_id}",
        mechanics.REWARD_FILTER,
        "training.miles96_mechanics_canary.post_save_hook",
        "--wandb-run-id q38-m96-canary-a1",
    ):
        assert expected in extra


def test_server_preview_proves_the_root_alert_annotation() -> None:
    request = mechanics.job_request(_plan())
    proof = validate_preview(request, _server_preview(request))
    assert proof["nodes"] == 1 and proof["gpus"] == 8

    broken = yaml.safe_load(_server_preview(request)["manifest_yaml"])
    broken["metadata"]["annotations"].pop("fleet.ai/failure-alerts")
    with pytest.raises(JobsError, match="failed-job alerts"):
        validate_preview(request, {"manifest_yaml": yaml.safe_dump(broken), "warnings": []})


@pytest.mark.parametrize(
    "fault", ["same_name", "old_image", "context", "filter", "priority", "reuse"]
)
def test_contract_drift_fails_before_any_request(fault: str) -> None:
    plan = _plan()
    if fault == "same_name":
        plan["identity"]["reload_name"] = plan["identity"]["name"]
    elif fault == "old_image":
        plan["trainer"]["image"] = "registry.invalid/trainer@sha256:" + "0" * 64
    elif fault == "context":
        plan["episode"]["max_tokens_per_turn"] = 4096
    elif fault == "filter":
        plan["optimization"]["reward_filter"] = (
            "miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted"
        )
    elif fault == "priority":
        plan["cluster"]["priority"] = "c0"
    else:
        plan["prepared_model"]["root"] = plan["identity"]["run_dir"] + "/old-model"
    with pytest.raises(ValueError):
        mechanics.validate_plan(plan)


def test_post_save_receipt_is_bound_and_has_no_reward_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()
    written: list[tuple[Path, bytes]] = []
    receipt = _receipt(plan)
    monkeypatch.setattr(mechanics, "_load_runtime_plan", lambda: plan)
    monkeypatch.setattr(mechanics, "_train_receipt", lambda *_args, **_kwargs: receipt)
    monkeypatch.setattr(mechanics, "_write_once", lambda path, data: written.append((path, data)))
    monkeypatch.setenv("CYBER_PLAN_SHA256", mechanics.digest(plan))
    mechanics.post_save_hook(None, 1, "/unused/checkpoint", "/unused/hf")
    assert written[0][0] == Path(plan["identity"]["run_dir"]) / mechanics.UPDATE_FILE
    assert json.loads(written[0][1])["receipt_sha256"] == receipt["receipt_sha256"]
    assert b"reward" in written[0][1] and b"0." not in written[0][1]


def test_reload_request_is_one_gpu_zero_optimizer_and_receipt_bound() -> None:
    plan = _plan()
    receipt = _receipt(plan)
    request = mechanics.reload_request(plan, receipt)
    validate_request(request)
    assert request["workers"] == request["gpus_per_worker"] == 1
    assert request["privileged"] is False
    assert request["priority_class"] == "c1" and request["failureAlerts"] is False
    assert "wandb-api" not in request["secrets"]
    assert request["run_dir"] == plan["identity"]["reload_run_dir"]
    assert mechanics.RELOAD_FILE not in request["command"]

    changed = copy.deepcopy(receipt)
    changed["optimizer_steps"] = 2
    with pytest.raises(ValueError, match="invalid update/export receipt"):
        mechanics.reload_request(plan, changed)
