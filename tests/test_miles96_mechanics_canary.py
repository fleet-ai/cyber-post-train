from __future__ import annotations

import base64
import copy
import gzip
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import yaml

from cyber_post_train.jobs import JobsError, validate_preview, validate_request
from cyber_post_train.jobs import digest as jobs_digest
from cyber_post_train.sfs_output import build_output_absence_receipt
from training import dev_cleanup_observer as cleanup
from training import miles96_mechanics_canary as mechanics
from training import miles96_mechanics_launch as launch


def _sha(char: str) -> str:
    return "sha256:" + char * 64


def _plan() -> dict:
    authority = {
        "task_key": "synthetic-blackbox-v1",
        "task_version_id": str(uuid.UUID("11111111-1111-4111-8111-111111111111")),
        "verifier_version_id": str(uuid.UUID("22222222-2222-4222-8222-222222222222")),
        "task_set_sha256": _sha("b"),
        "tool_catalog_sha256": _sha("c"),
    }
    return mechanics.build_plan(
        name="q38-m96-canary-a1",
        reload_name="q38-m96-reload-a1",
        model_root="/mnt/sfs/jobs/q38-prepared-model-v1",
        model_binding_sha256=_sha("a"),
        task_binding={
            **authority,
            "authority_receipt_sha256": "sha256:" + mechanics.digest(authority),
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
            "namespace": mechanics.NAMESPACE,
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
            "backoffLimit": 0,
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
        "native_rollout_id": 0,
        "optimizer_updates": 1,
        "reward_filter": mechanics.REWARD_FILTER,
        "reward_gate_sha256": _sha("d"),
        "checkpoint_manifest_sha256": _sha("e"),
        "complete_model_manifest_sha256": _sha("f"),
        "complete_model_path_sha256": _sha("0"),
        "changed_trained_tensor_count": 1,
        "all_trained_tensors_finite": True,
        "wandb_run_id": plan["wandb"]["run_id"],
        "wandb_resume": "never",
        "hf_export_complete": True,
        "save_hook_after_checkpoint": True,
    }
    return {**body, "receipt_sha256": "sha256:" + mechanics.digest(body)}


def _bundle(request: dict) -> dict:
    values = request["env"]
    if "CYBER_RUNTIME_BUNDLE" in values:
        encoded = values["CYBER_RUNTIME_BUNDLE"]
    else:
        chunks = sorted(
            (
                (int(key.rsplit("_", 1)[1]), value)
                for key, value in values.items()
                if key.startswith("CYBER_RUNTIME_BUNDLE_")
            )
        )
        encoded = "".join(value for _, value in chunks)
    return json.loads(gzip.decompress(base64.b64decode(encoded, validate=True)))


def test_one_node_current_recipe_request_and_fresh_v1_row() -> None:
    plan = _plan()
    request = mechanics.job_request(plan)
    validate_request(request)
    assert plan["trainer"] == {
        "model": mechanics.MODEL,
        "recipe": mechanics.RECIPE,
        "image": mechanics.IMAGE,
        "theseus_commit": "224d6b81cb698f4785fb16123314583b981493f3",
        "miles_commit": mechanics.MILES_COMMIT,
        "fti_version": "0.10.9",
        "run_fleet_sha256": mechanics.RUN_FLEET_SHA256,
        "fti_common_sha256": mechanics.FTI_COMMON_SHA256,
        "fti_client_recording_sha256": mechanics.FTI_CLIENT_RECORDING_SHA256,
        "miles_inference_rollout_sha256": mechanics.MILES_INFERENCE_ROLLOUT_SHA256,
        "miles_http_utils_sha256": mechanics.MILES_HTTP_UTILS_SHA256,
        "miles_megatron_actor_sha256": mechanics.MILES_MEGATRON_ACTOR_SHA256,
        "miles_hf_export_sha256": mechanics.MILES_HF_EXPORT_SHA256,
    }
    assert request["workers"] == 1 and request["gpus_per_worker"] == 8
    assert request["priority_class"] == "c1" and request["failureAlerts"] is False
    assert request["requeueIfPreempted"] is False
    assert request["env"]["WANDB_RESUME"] == "never"
    assert plan["task_binding"]["task_key"] not in request["command"]

    row = json.loads(mechanics.task_rows(plan))
    assert row["messages"] == [{"role": "user", "content": "Run the versioned Fleet task."}]
    assert row["metadata"]["task_version_id"] == plan["task_binding"]["task_version_id"]
    assert set(row["metadata"]["fleet"]) == set(plan["episode"])

    argv = mechanics.native_arguments(plan)
    extra = argv[argv.index("--extra-args") + 1]
    assert argv[:3] == ["-m", "fti.trainers.miles.run_fleet", "--model-name"]
    assert "qwen3.8-27b-256k" not in argv
    for expected in (
        "--num-rollout 1",
        "--over-sampling-batch-size 1",
        "--save-interval 1",
        "--save-hf /mnt/sfs/jobs/q38-m96-canary-a1/hf/step-{rollout_id}",
        mechanics.REWARD_FILTER,
        "training.miles96_mechanics_canary.record_selected_reward_group",
        "training.miles96_mechanics_canary.post_save_hook",
        "--wandb-run-id q38-m96-canary-a1",
    ):
        assert expected in extra


def test_runtime_bundle_executes_in_an_isolated_interpreter(tmp_path: Path) -> None:
    bundle = _bundle(mechanics.job_request(_plan()))
    runtime = tmp_path / "runtime"
    for name, content in bundle["files"].items():
        path = runtime / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    module_source = (runtime / "training/miles96_mechanics_canary.py").read_text()
    assert "cyber_post_train" not in module_source.split("def job_request", 1)[0]
    code = (
        "import runpy,sys;"
        f"sys.path.insert(0,{str(runtime)!r});"
        "sys.argv=['training.miles96_mechanics_canary','--help'];"
        "runpy.run_module('training.miles96_mechanics_canary',run_name='__main__')"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--train" in result.stdout and "--reload" in result.stdout


def test_server_preview_proves_root_alert_annotation_and_no_retry() -> None:
    request = mechanics.job_request(_plan())
    proof = validate_preview(request, _server_preview(request))
    assert proof["nodes"] == 1 and proof["gpus"] == 8

    broken = yaml.safe_load(_server_preview(request)["manifest_yaml"])
    broken["metadata"]["annotations"].pop("fleet.ai/failure-alerts")
    with pytest.raises(JobsError, match="failed-job alerts"):
        validate_preview(request, {"manifest_yaml": yaml.safe_dump(broken), "warnings": []})


@pytest.mark.parametrize(
    "fault", ["same_name", "old_image", "context", "filter", "priority", "reuse", "authority"]
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
        plan["optimization"]["reward_filter"] = "check_no_aborted"
    elif fault == "priority":
        plan["cluster"]["priority"] = "c0"
    elif fault == "reuse":
        plan["prepared_model"]["root"] = plan["identity"]["run_dir"] + "/old-model"
    else:
        plan["task_binding"]["tool_catalog_sha256"] = _sha("9")
    with pytest.raises(ValueError):
        mechanics.validate_plan(plan)


def test_selected_group_keeps_exact_execution_ids_private(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan()
    evidence = tmp_path / "private"
    evidence.mkdir(mode=0o700)
    refs = []
    for index in range(8):
        body = {
            "schema": mechanics.PRIVATE_EPISODE_SCHEMA,
            "plan_sha256": "sha256:" + mechanics.digest(plan),
            "task_key": plan["task_binding"]["task_key"],
            "task_version_id": plan["task_binding"]["task_version_id"],
            "verifier_version_id": plan["task_binding"]["verifier_version_id"],
            "instance_id": str(uuid.UUID(int=index + 100)),
            "verifier_execution_id": str(uuid.UUID(int=index + 200)),
            "reward": float(index % 2),
            "cleanup_confirmed": True,
        }
        value = {**body, "sha256": "sha256:" + mechanics.digest(body)}
        refs.append(value["sha256"])
        (evidence / (value["sha256"].removeprefix("sha256:") + ".json")).write_text(
            json.dumps(value)
        )
    public: list[dict] = []

    def write_once(path: Path, payload: bytes) -> None:
        if path.parent == evidence:
            path.write_bytes(payload)
        else:
            public.append(json.loads(payload))

    monkeypatch.setattr(mechanics, "_load_runtime_plan", lambda: plan)
    monkeypatch.setattr(mechanics, "_private_directory", lambda _root: evidence)
    monkeypatch.setattr(mechanics, "_write_once", write_once)
    samples = [NS(metadata={"fleet_v1": {"authority_evidence_sha256": ref}}) for ref in refs]
    mechanics.record_selected_reward_group(
        NS(rollout_batch_size=1, n_samples_per_prompt=8), [samples]
    )
    selected = json.loads((evidence / mechanics.PRIVATE_SELECTED_FILE).read_text())
    assert len(selected["verifier_execution_ids"]) == len(selected["instance_ids"]) == 8
    assert set(selected["rewards"]) == {0.0, 1.0}
    assert public[0]["native_rollout_id"] == 0
    assert public[0]["optimizer_updates"] == 1
    assert public[0]["selected_episode_count"] == 8
    assert "rewards" not in public[0] and "verifier_execution_ids" not in public[0]


def _write_model(root: Path, tensors: dict, *, complete: bool = False) -> None:
    import safetensors.torch

    root.mkdir(parents=True)
    safetensors.torch.save_file(tensors, root / "model.safetensors")
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {key: "model.safetensors" for key in tensors},
            }
        )
    )
    if complete:
        (root / ".complete").write_text("native-export-complete\n")


def test_complete_model_restores_only_frozen_qwen_tensors_and_detects_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")

    plan = _plan()
    run_dir = tmp_path / "run"
    model_root = tmp_path / "prepared"
    base = model_root / "Qwen3.8-27B"
    raw = run_dir / mechanics.RAW_HF_STEP
    _write_model(
        base,
        {
            "model.language_model.layers.0.weight": torch.tensor([1.0, 2.0]),
            "model.visual.patch.weight": torch.tensor([3.0]),
            "mtp.weight": torch.tensor([4.0]),
        },
    )
    (base / "config.json").write_text('{"model_type":"qwen3_5"}\n')
    (base / "tokenizer_config.json").write_text("{}\n")
    _write_model(
        raw,
        {"model.language_model.layers.0.weight": torch.tensor([1.0, 2.5])},
        complete=True,
    )
    plan["identity"]["run_dir"] = str(run_dir)
    plan["prepared_model"]["root"] = str(model_root)
    monkeypatch.setattr(mechanics, "validate_plan", lambda value: value)
    manifest = mechanics.compose_complete_model(plan, raw)
    final = run_dir / mechanics.COMPLETE_HF_STEP
    assert manifest["native_rollout_id"] == 0
    assert manifest["optimizer_updates"] == 1
    assert manifest["changed_trained_tensor_count"] == 1
    assert manifest["all_trained_tensors_finite"] is True
    assert manifest["trained_tensor_count"] == 1
    assert manifest["frozen_tensor_count"] == 2
    assert manifest["total_tensor_bytes"] == 16
    index = json.loads((final / "model.safetensors.index.json").read_text())
    assert index["metadata"] == {"total_size": 16}
    assert set(mechanics._hf_index(final)) == {
        "model.language_model.layers.0.weight",
        "model.visual.patch.weight",
        "mtp.weight",
    }
    mechanics.validate_complete_model(plan, final)
    (final / "config.json").write_text('{"tampered":true}\n')
    with pytest.raises(ValueError, match="file manifest changed"):
        mechanics.validate_complete_model(plan, final)


def test_receipt_uses_zero_index_for_one_real_optimizer_update() -> None:
    plan = _plan()
    receipt = _receipt(plan)
    mechanics.validate_train_receipt(plan, receipt)
    changed = copy.deepcopy(receipt)
    changed["native_rollout_id"] = 1
    with pytest.raises(ValueError, match="invalid update/export receipt"):
        mechanics.validate_train_receipt(plan, changed)


def test_train_receipt_requires_private_selected_reward_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan()
    plan["identity"]["run_dir"] = str(tmp_path)
    monkeypatch.setattr(mechanics, "validate_plan", lambda value: value)
    public_body = {
        "schema": mechanics.REWARD_GATE_SCHEMA,
        "plan_sha256": "sha256:" + mechanics.digest(plan),
        "native_rollout_id": 0,
        "optimizer_updates": 1,
        "selected_episode_count": 8,
        "finite_rewards": True,
        "reward_variation": True,
        "unique_verifier_executions": 8,
        "all_instances_released": True,
        "private_evidence_sha256": _sha("a"),
    }
    (tmp_path / mechanics.REWARD_GATE_FILE).write_text(
        json.dumps({**public_body, "sha256": "sha256:" + mechanics.digest(public_body)})
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    raw = tmp_path / mechanics.RAW_HF_STEP
    raw.mkdir(parents=True)
    (raw / ".complete").write_text("complete\n")
    with pytest.raises(FileNotFoundError):
        mechanics._train_receipt(
            plan,
            rollout_id=0,
            checkpoint_dir=str(checkpoint),
            hf_dir=str(raw),
        )


def test_reload_request_is_one_gpu_zero_optimizer_and_receipt_bound() -> None:
    plan = _plan()
    request = mechanics.reload_request(plan, _receipt(plan))
    validate_request(request)
    assert request["workers"] == request["gpus_per_worker"] == 1
    assert request["privileged"] is False
    assert request["priority_class"] == "c1" and request["failureAlerts"] is False
    assert "wandb-api" not in request["secrets"]
    assert request["run_dir"] == plan["identity"]["reload_run_dir"]
    assert (
        mechanics.COMPLETE_HF_STEP
        in _bundle(request)["files"]["training/miles96_mechanics_canary.py"]
    )
    server = mechanics._reload_server_arguments(Path("/model"))
    assert server[server.index("--model-path") + 1] == "/model"
    assert server[server.index("--context-length") + 1] == "98304"
    assert server[server.index("--tp-size") + 1] == "1"
    assert server[server.index("--attention-backend") + 1] == "triton"


class _FakeClient:
    def __init__(self, request: dict) -> None:
        self.request_value = request
        self.posts = 0

    def preview(self, request: dict) -> dict:
        assert request == self.request_value
        return _server_preview(request)

    def all_runs(self) -> list[dict]:
        return []

    def request(self, method: str, path: str, *, json: dict) -> dict:
        assert (method, path, json) == ("POST", "/v1/runs", self.request_value)
        self.posts += 1
        return {
            "name": json["name"] + "-1234abcd",
            "job_id": "33333333-3333-4333-8333-333333333333",
            "run_dir": json["run_dir"],
            "status": "QUEUED",
        }


def test_live_preview_absence_and_exactly_one_post_are_durably_journaled(
    tmp_path: Path,
) -> None:
    plan = _plan()
    request = mechanics.job_request(plan)
    client = _FakeClient(request)
    jobs_root = tmp_path / "sfs-jobs"
    jobs_root.mkdir()

    def start_observer(directory: Path) -> NS:
        preview = json.loads((directory / "SERVER_PREVIEW.json").read_text())
        now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        body = {
            "schema": cleanup.JOBS_API_PREFIX_GUARD_SCHEMA,
            "status": "armed_non_destructive_prefix_guard",
            "context": plan["execution"]["kubernetes_context"],
            "namespace": plan["execution"]["namespace"],
            "run_name_prefix": request["name"],
            "generated_name_pattern": "^" + request["name"] + "-[a-f0-9]{8}$",
            "run_dir": request["run_dir"],
            "image": request["image"],
            "plan_sha256": "sha256:" + mechanics.digest(plan),
            "manifest_sha256": preview["manifest_sha256"],
            "maximum_seconds": 7200,
            "expected_gpus": 8,
            "armed_at": now,
            "observer_pid": os.getpid(),
            "prefix_collision_count_before_post": 0,
        }
        (directory / "OBSERVER_ARMED.json").write_text(
            json.dumps({**body, "sha256": "sha256:" + jobs_digest(body)})
        )
        return NS(terminate=lambda: None)

    def runner(argv: list[str], **_kwargs) -> NS:
        assert argv[:2] == ["kubectl", "--context"]
        return NS(returncode=0, stdout='{"items":[]}', stderr="")

    result = launch.submit_once(
        plan,
        request,
        client,
        tmp_path / "launch",
        runner=runner,
        jobs_root=jobs_root,
        wandb_exists=lambda *_args: False,
        now=time.time,
        start_observer=start_observer,
    )
    rows = [
        json.loads(line) for line in (tmp_path / "launch/SUBMISSION.jsonl").read_text().splitlines()
    ]
    assert result["name"] == request["name"] + "-1234abcd"
    assert client.posts == 1
    assert [row["state"] for row in rows] == [
        "POST_INTENT_DO_NOT_RETRY",
        "POST_RESPONSE",
    ]
    assert rows[0]["preview"]["root_failure_alerts"] == "off"
    assert rows[0]["preview"]["backoff_limit"] == 0
    assert rows[0]["fresh_absence"]["sfs_output_absent"] is True
    assert len(rows[0]["fresh_absence"]["sfs_output_absence_receipt_sha256"]) == 64
    assert len(rows[0]["final_sfs_output_absence_receipt_sha256"]) == 64


def test_off_node_submit_requires_a_fresh_exact_sfs_receipt(tmp_path: Path) -> None:
    plan = _plan()
    request = mechanics.job_request(plan)
    mounted = tmp_path / "mounted"
    mounted.mkdir()
    receipt = build_output_absence_receipt(plan, request, jobs_root=mounted, now=1000)
    unavailable = tmp_path / "unavailable"

    assert (
        launch._prove_sfs_output_absent(
            plan,
            request,
            jobs_root=unavailable,
            receipt=receipt,
            observed_at=1001,
        )
        == receipt
    )
    with pytest.raises(JobsError, match="provide a fresh"):
        launch._prove_sfs_output_absent(
            plan,
            request,
            jobs_root=unavailable,
            receipt=None,
            observed_at=1001,
        )
    with pytest.raises(JobsError, match="stale"):
        launch._prove_sfs_output_absent(
            plan,
            request,
            jobs_root=unavailable,
            receipt=receipt,
            observed_at=1301,
        )
