from __future__ import annotations

import base64
import copy
import gzip
import json
import os
import subprocess
import sys
import time
import types
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import yaml

from cyber_post_train.gpu_capacity import build_capacity_census
from cyber_post_train.jobs import JobsError, validate_preview, validate_request
from cyber_post_train.jobs import digest as jobs_digest
from cyber_post_train.sfs_output import build_output_absence_receipt
from cyber_post_train.sfs_output_job import build_sfs_output_job
from training import dev_cleanup_observer as cleanup
from training import miles96_mechanics_canary as mechanics
from training import miles96_mechanics_launch as launch
from training import miles96_signal_qualification as signal_qualification
from training import miles_signal_wave


def _sha(char: str) -> str:
    return "sha256:" + char * 64


def _plan() -> dict:
    wave = miles_signal_wave.load()
    candidate = wave["candidates"][0]
    authority = miles_signal_wave.task_binding(wave, candidate)
    task_binding = {
        **authority,
        "authority_receipt_sha256": candidate["authority_receipt_sha256"],
    }
    phase1 = signal_qualification.build_plan(
        name=candidate["identity"]["name"],
        model_root=signal_qualification.HF_MODEL_ROOT,
        model_binding_sha256=signal_qualification.HF_MODEL_BINDING_SHA256,
        task_binding=task_binding,
        authority_config_sha256=wave["sha256"],
        current_binding_sha256=candidate["live_binding_receipt_sha256"],
        production_split_sha256=wave["authorities"]["production_split"]["self_sha256"],
    )
    signal_body = {
        "schema": mechanics.TASK_SIGNAL_EVIDENCE_SCHEMA,
        "phase1_plan_sha256": "sha256:" + mechanics.digest(phase1),
        "phase1_run_name": candidate["identity"]["name"],
        "model_revision": signal_qualification.HF_MODEL_REVISION,
        "model_binding_sha256": signal_qualification.HF_MODEL_BINDING_SHA256,
        **phase1["selection_authority"],
        "planned_slot_count": 8,
        "terminal_slot_count": 8,
        "excluded_slot_count": 0,
        "outer_replacement_count": 0,
        **authority,
        "max_turns": 32,
        "max_tokens_per_turn": 8192,
        "episode_timeout_s": 2400,
        "completed_episode_count": 8,
        "finite_rewards": True,
        "reward_variation": True,
        "all_instances_released": True,
        "source_receipt_sha256": _sha("7"),
        "native_terminal_sha256": _sha("9"),
        "optimizer_steps": 0,
        "checkpoint_artifacts_absent": True,
        "runtime_source_manifest_sha256": "sha256:" + mechanics.digest(phase1["runtime_sources"]),
        "runtime_bundle_sha256": _sha("8"),
        "request_binding_sha256": _sha("6"),
        "runtime_preflight_sha256": _sha("5"),
    }
    return mechanics.build_plan(
        name="chris-q38-m96-canary-a1",
        reload_name="chris-q38-m96-reload-a1",
        model_root="/mnt/sfs/jobs/q38-prepared-model-v1",
        model_binding_sha256=_sha("a"),
        task_binding=task_binding,
        task_signal_evidence={
            **signal_body,
            "sha256": "sha256:" + mechanics.digest(signal_body),
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
        "wandb_primary_resume": "never",
        "wandb_secondary_resume": "allow_same_live_id",
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
                and key.removeprefix("CYBER_RUNTIME_BUNDLE_").isdigit()
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
        "theseus_commit": "68da13aa9c226d1bfed0ad59b990b99321d44239",
        "image_source_commit": "d23116f018cb9213f0a3ee6c228d213abcd85d80",
        "miles_commit": mechanics.MILES_COMMIT,
        "fti_version": "0.10.27",
        "run_fleet_sha256": mechanics.RUN_FLEET_SHA256,
        "fti_common_sha256": mechanics.FTI_COMMON_SHA256,
        "fti_client_recording_sha256": mechanics.FTI_CLIENT_RECORDING_SHA256,
        "miles_inference_rollout_sha256": mechanics.MILES_INFERENCE_ROLLOUT_SHA256,
        "miles_http_utils_sha256": mechanics.MILES_HTTP_UTILS_SHA256,
        "miles_megatron_actor_sha256": mechanics.MILES_MEGATRON_ACTOR_SHA256,
        "miles_hf_export_sha256": mechanics.MILES_HF_EXPORT_SHA256,
        "miles_wandb_utils_sha256": mechanics.MILES_WANDB_UTILS_SHA256,
    }
    assert request["workers"] == 1 and request["gpus_per_worker"] == 8
    assert request["priority_class"] == "c1" and request["failureAlerts"] is False
    assert request["requeueIfPreempted"] is False
    assert request["env"]["WANDB_RESUME"] == "never"
    assert request["env"]["WANDB_RUN_ID"] == plan["wandb"]["run_id"]
    assert plan["provenance"] == {
        "mechanics_reference": {
            "repository": "fleet-ai/dataminer_v2",
            "commit": "10afa8d064bb3dd1c11c50768590e432dfa69097",
            "recipe": "v004",
            "proven_scope": "one_node_tp4_cp2_96k_shape_only",
        },
        "maintained_recipe": {
            "max_tokens_per_gpu": 49152,
            "optimizer_cpu_offload": False,
            "qualification_status": "first_qualification_not_v004_proven",
        },
    }
    assert plan["task_signal_evidence"]["reward_variation"] is True
    assert plan["task_binding"]["task_key"] not in request["command"]

    row = json.loads(mechanics.task_rows(plan))
    assert row["messages"] == [{"role": "user", "content": "Run the versioned Fleet task."}]
    assert row["metadata"]["task_version_id"] == plan["task_binding"]["task_version_id"]
    assert set(row["metadata"]["fleet"]) == set(plan["episode"])

    argv = mechanics.native_arguments(plan)
    extra = argv[argv.index("--extra-args") + 1]
    ray_env = json.loads(argv[argv.index("--extra-env-vars") + 1])
    assert ray_env["WANDB_RUN_ID"] == plan["wandb"]["run_id"]
    assert ray_env["WANDB_RESUME"] == "never"
    assert "WANDB_API_KEY" not in ray_env
    assert argv[:3] == ["-m", "fti.trainers.miles.run_fleet", "--model-name"]
    assert "qwen3.8-27b-256k" not in argv
    for expected in (
        "--num-rollout 1",
        "--over-sampling-batch-size 1",
        "--save-interval 1",
        "--save-hf /mnt/sfs/jobs/chris-q38-m96-canary-a1/hf/step-{rollout_id}",
        mechanics.REWARD_FILTER,
        "training.miles96_mechanics_canary.record_selected_reward_group",
        "training.miles96_mechanics_canary.post_save_hook",
        "--wandb-run-id chris-q38-m96-canary-a1",
    ):
        assert expected in extra


def test_pinned_miles_wandb_patch_binds_primary_and_live_secondaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the exact pinned helper behavior that discarded the parsed ID."""
    calls: list[dict[str, object]] = []

    class FakeWandb:
        run = NS(id="")
        existing: set[str] = set()
        live: set[str] = set()

        @classmethod
        def init(cls, **kwargs: object) -> None:
            run_id = str(kwargs.get("id") or "upstream-generated-id")
            resume = kwargs.get("resume")
            if resume == "never" and run_id in cls.existing:
                raise RuntimeError("historical run exists")
            if resume == "allow" and run_id not in cls.live:
                raise RuntimeError("run is not the live primary")
            calls.append(dict(kwargs))
            cls.existing.add(run_id)
            cls.live.add(run_id)
            cls.run = NS(id=run_id)

    command_utils = types.ModuleType("miles.utils.external_utils.command_utils")
    command_utils.get_default_wandb_args = lambda *_args, **_kwargs: "unsafe"
    wandb_utils = types.ModuleType("miles.utils.tracking_utils.wandb_utils")
    wandb_utils.wandb = FakeWandb

    # These two small functions reproduce the relevant behavior of the pinned,
    # hash-checked Miles helper: primary omits id/resume and overwrites the
    # parsed ID, while secondary joins that resulting ID with resume=allow.
    def upstream_primary(args: NS) -> None:
        FakeWandb.init(entity="thefleet", project="cyber-post")
        args.wandb_run_id = FakeWandb.run.id

    secondary_resume = ["allow"]

    def upstream_secondary(args: NS, router_addr: str | None = None) -> None:
        del router_addr
        FakeWandb.init(id=args.wandb_run_id, resume=secondary_resume[0], reinit=True)

    wandb_utils.init_wandb_primary = upstream_primary
    wandb_utils.init_wandb_secondary = upstream_secondary

    packages = {
        "miles": types.ModuleType("miles"),
        "miles.utils": types.ModuleType("miles.utils"),
        "miles.utils.external_utils": types.ModuleType("miles.utils.external_utils"),
        "miles.utils.external_utils.command_utils": command_utils,
        "miles.utils.tracking_utils": types.ModuleType("miles.utils.tracking_utils"),
        "miles.utils.tracking_utils.wandb_utils": wandb_utils,
    }
    for name, module in packages.items():
        if name != "miles.utils.external_utils.command_utils" and name != (
            "miles.utils.tracking_utils.wandb_utils"
        ):
            module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("WANDB_API_KEY", "redacted-test-key")
    monkeypatch.setenv("WANDB_RUN_ID", "chris-q38-m96-canary-a1")
    monkeypatch.setenv("WANDB_RESUME", "never")

    mechanics.install_safe_wandb()
    primary = NS(wandb_run_id="chris-q38-m96-canary-a1")
    with pytest.raises(RuntimeError, match="not the live primary"):
        wandb_utils.init_wandb_secondary(NS(wandb_run_id=primary.wandb_run_id))
    # A historical run at the exact planned ID is never resumed by primary.
    FakeWandb.existing.add("chris-q38-m96-canary-a1")
    with pytest.raises(RuntimeError, match="historical run exists"):
        wandb_utils.init_wandb_primary(primary)
    FakeWandb.existing.clear()
    wandb_utils.init_wandb_primary(primary)
    assert primary.wandb_run_id == "chris-q38-m96-canary-a1"
    assert calls == [
        {
            "entity": "thefleet",
            "project": "cyber-post",
            "id": "chris-q38-m96-canary-a1",
            "resume": "never",
        }
    ]

    wandb_utils.init_wandb_secondary(NS(wandb_run_id=primary.wandb_run_id))
    assert calls[1] == {
        "id": "chris-q38-m96-canary-a1",
        "resume": "allow",
        "reinit": True,
    }
    with pytest.raises(RuntimeError, match="live primary"):
        wandb_utils.init_wandb_secondary(NS(wandb_run_id="historical-run"))
    secondary_resume[0] = "must"
    with pytest.raises(RuntimeError, match="attachment contract drift"):
        wandb_utils.init_wandb_secondary(NS(wandb_run_id=primary.wandb_run_id))

    monkeypatch.delenv("WANDB_API_KEY")
    with pytest.raises(RuntimeError, match="credential is absent"):
        wandb_utils.init_wandb_primary(NS(wandb_run_id="chris-q38-m96-canary-a1"))


def test_phase2_runtime_bundle_validates_closed_sources_in_isolated_interpreter(
    tmp_path: Path,
) -> None:
    bundle = _bundle(mechanics.job_request(_plan()))
    runtime = tmp_path / "runtime"
    for name, content in bundle["files"].items():
        path = runtime / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    code = (
        "import json,sys;"
        f"sys.path.insert(0,{str(runtime)!r});"
        "from training import miles96_mechanics_canary as m;"
        f"m.validate_plan(json.load(open({str(runtime / 'plan.json')!r})))"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    signal_path = runtime / "training/miles96_signal_qualification.py"
    signal_path.write_text(signal_path.read_text() + "\n# drift\n")
    changed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert changed.returncode != 0
    assert "custom runtime source manifest drift" in changed.stderr


def test_phase2_rereads_exact_phase1_source_and_zero_update_filesystem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan()
    signal = plan["task_signal_evidence"]
    phase1 = signal_qualification.build_plan(
        name=signal["phase1_run_name"],
        model_root=signal_qualification.HF_MODEL_ROOT,
        model_binding_sha256=signal["model_binding_sha256"],
        task_binding=plan["task_binding"],
        authority_config_sha256=signal["authority_config_sha256"],
        current_binding_sha256=signal["current_binding_sha256"],
        production_split_sha256=signal["production_split_sha256"],
    )
    root = tmp_path / signal["phase1_run_name"]
    private_dir = root / mechanics.PRIVATE_EVIDENCE_DIR
    private_dir.mkdir(parents=True)
    preflight_body = {
        "schema": signal_qualification.RUNTIME_PREFLIGHT_SCHEMA,
        "plan_sha256": "sha256:" + mechanics.digest(phase1),
        "image": mechanics.IMAGE,
        "fti_v1_sha256": "sha256:" + signal_qualification.FTI_V1_SHA256,
        "miles_eval_sha256": "sha256:" + signal_qualification.MILES_INFERENCE_EVAL_SHA256,
        "sample_index_start": 0,
        "sample_index_end": 7,
        "sample_count": 8,
        "max_concurrent_envs": 2,
        "shielded_close": True,
        "release_absence_http_status": 404,
        "tool_catalog_sha256": plan["task_binding"]["tool_catalog_sha256"],
        "live_tool_schema_gate_at_session_open": True,
        "outer_episode_replacements": 0,
    }
    preflight = {
        **preflight_body,
        "sha256": "sha256:" + mechanics.digest(preflight_body),
    }
    signal["runtime_preflight_sha256"] = preflight["sha256"]
    private_body = {
        "schema": signal_qualification.PRIVATE_GROUP_SCHEMA,
        "plan_sha256": signal["phase1_plan_sha256"],
        "runtime_source_manifest_sha256": signal["runtime_source_manifest_sha256"],
        "runtime_bundle_sha256": signal["runtime_bundle_sha256"],
        "request_binding_sha256": signal["request_binding_sha256"],
        "runtime_preflight_sha256": signal["runtime_preflight_sha256"],
        "episode_receipts": [_sha("1")] * 8,
        "claim_receipts": [_sha("2")] * 8,
        "attempt_receipts": [_sha("3")] * 24,
        "slot_receipts": [_sha("4")] * 8,
        "verifier_execution_ids": [str(uuid.UUID(int=index + 1)) for index in range(8)],
        "instance_ids": [str(uuid.UUID(int=index + 101)) for index in range(8)],
        "rewards": [float(index % 2) for index in range(8)],
        "admitted_episode_bindings": [{"sample_index": index} for index in range(8)],
    }
    private = {**private_body, "sha256": "sha256:" + mechanics.digest(private_body)}
    native_body = {
        "schema": signal_qualification.NATIVE_TERMINAL_SCHEMA,
        "plan_sha256": signal["phase1_plan_sha256"],
        "status": "succeeded",
        "returncode": 0,
        "optimizer_steps": 0,
        "checkpoint_artifacts_absent": True,
        "runtime_source_manifest_sha256": signal["runtime_source_manifest_sha256"],
        "runtime_bundle_sha256": signal["runtime_bundle_sha256"],
        "request_binding_sha256": signal["request_binding_sha256"],
        "runtime_preflight_sha256": signal["runtime_preflight_sha256"],
    }
    native = {**native_body, "sha256": "sha256:" + mechanics.digest(native_body)}
    signal.update(
        source_receipt_sha256=private["sha256"],
        native_terminal_sha256=native["sha256"],
    )
    signal_body = {key: value for key, value in signal.items() if key != "sha256"}
    signal["sha256"] = "sha256:" + mechanics.digest(signal_body)
    (root / signal_qualification.EVIDENCE_FILE).write_text(json.dumps(signal))
    (private_dir / signal_qualification.PRIVATE_GROUP_FILE).write_text(json.dumps(private))
    (root / signal_qualification.NATIVE_TERMINAL_FILE).write_text(json.dumps(native))
    (root / signal_qualification.RUNTIME_PREFLIGHT_FILE).write_text(json.dumps(preflight))
    monkeypatch.setattr(mechanics, "SFS_JOBS_ROOT", tmp_path)

    mechanics._task_signal_source_exists(plan)
    checkpoint = root / "model-output/checkpoints/unexpected.bin"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"unexpected-update")
    with pytest.raises(ValueError, match="source evidence is invalid"):
        mechanics._task_signal_source_exists(plan)


def test_server_preview_proves_root_alert_annotation_and_no_retry() -> None:
    request = mechanics.job_request(_plan())
    proof = validate_preview(request, _server_preview(request))
    assert proof["nodes"] == 1 and proof["gpus"] == 8

    broken = yaml.safe_load(_server_preview(request)["manifest_yaml"])
    broken["metadata"]["annotations"].pop("fleet.ai/failure-alerts")
    with pytest.raises(JobsError, match="failed-job alerts"):
        validate_preview(request, {"manifest_yaml": yaml.safe_dump(broken), "warnings": []})


@pytest.mark.parametrize(
    "fault",
    [
        "same_name",
        "old_image",
        "context",
        "filter",
        "priority",
        "reuse",
        "authority",
        "signal",
        "provenance",
    ],
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
    elif fault == "authority":
        plan["task_binding"]["tool_catalog_sha256"] = _sha("9")
    elif fault == "signal":
        plan["task_signal_evidence"]["reward_variation"] = False
    else:
        plan["provenance"]["maintained_recipe"]["optimizer_cpu_offload"] = True
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
            "done_reason": "submitted",
            "tool_calls": 3,
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


def test_prepared_model_binding_hashes_exact_hf_and_megatron_bytes(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    root = tmp_path / "prepared"
    hf = root / "Qwen3.8-27B"
    megatron = root / "qwen3.8-27B_torch_dist"
    _write_model(hf, {"model.language_model.weight": torch.tensor([1.0])})
    (hf / "config.json").write_text('{"model_type":"qwen3_5"}\n')
    megatron.mkdir(parents=True)
    (megatron / "latest_checkpointed_iteration.txt").write_text("release\n")
    (megatron / "release-state.pt").write_bytes(b"exact-megatron-bytes")

    inventory = mechanics.prepared_model_inventory(root)
    plan = _plan()
    plan["prepared_model"] = {"root": str(root), "binding_sha256": inventory["sha256"]}
    mechanics._prepared_model_exists(plan)
    (megatron / "release-state.pt").write_bytes(b"changed-megatron-bytes")
    with pytest.raises(ValueError, match="immutable binding"):
        mechanics._prepared_model_exists(plan)


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
    megatron = model_root / "qwen3.8-27B_torch_dist"
    megatron.mkdir(parents=True)
    (megatron / "latest_checkpointed_iteration.txt").write_text("release\n")
    (megatron / "release-state.pt").write_bytes(b"exact-megatron-bytes")
    _write_model(
        raw,
        {"model.language_model.layers.0.weight": torch.tensor([1.0, 2.5])},
        complete=True,
    )
    plan["identity"]["run_dir"] = str(run_dir)
    plan["prepared_model"]["root"] = str(model_root)
    plan["prepared_model"]["binding_sha256"] = mechanics.prepared_model_inventory(model_root)[
        "sha256"
    ]
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


def test_post_save_receipt_reads_the_actual_live_wandb_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()
    fake = types.ModuleType("wandb")
    fake.run = NS(id=plan["wandb"]["run_id"])
    monkeypatch.setitem(sys.modules, "wandb", fake)
    assert mechanics._live_wandb_run_id(plan) == plan["wandb"]["run_id"]
    fake.run = NS(id="historical-run")
    with pytest.raises(ValueError, match="planned live W&B run"):
        mechanics._live_wandb_run_id(plan)


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
    assert request["env"]["CYBER_RUNTIME_DIR"] == request["run_dir"] + "/.runtime"
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
            "generated_name_pattern": "^" + launch.re.escape(request["name"]) + "-[a-f0-9]{8}$",
            "run_dir": request["run_dir"],
            "image": request["image"],
            "plan_sha256": "sha256:" + mechanics.digest(plan),
            "manifest_sha256": preview["manifest_sha256"],
            "maximum_seconds": launch._maximum_seconds(plan, request),
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

    capacity_calls = []

    def capacity_reader(context: str, **kwargs) -> dict:
        capacity_calls.append((context, kwargs))
        return build_capacity_census(
            {"items": []},
            {"items": []},
            observed_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            **kwargs,
        )

    result = launch.submit_once(
        plan,
        request,
        client,
        tmp_path / "launch",
        runner=runner,
        jobs_root=jobs_root,
        wandb_exists=lambda *_args: False,
        capacity_reader=capacity_reader,
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
    assert rows[0]["capacity_gate"]["planned"] == {"nodes": 1, "gpus": 8}
    assert rows[0]["capacity_gate"]["capacity_census"]["qualified"] is True
    assert capacity_calls == [
        (
            plan["execution"]["kubernetes_context"],
            {
                "owner_prefixes": launch.PROJECT_OWNER_PREFIXES,
                "max_nodes": launch.PROJECT_MAX_NODES,
                "max_gpus": launch.PROJECT_MAX_GPUS,
                "planned_nodes": 1,
                "planned_gpus": 8,
            },
        )
    ]


def test_capacity_gate_counts_train_and_reload_and_fails_closed() -> None:
    plan = _plan()
    observed_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    planned = []

    def reader(_context: str, **kwargs) -> dict:
        planned.append((kwargs["planned_nodes"], kwargs["planned_gpus"]))
        return build_capacity_census(
            {"items": []}, {"items": []}, observed_at=observed_at, **kwargs
        )

    train = launch.capacity_gate(plan, mechanics.job_request(plan), reader=reader)
    reload = launch.capacity_gate(
        plan, mechanics.reload_request(plan, _receipt(plan)), reader=reader
    )
    assert train["planned"] == {"nodes": 1, "gpus": 8}
    assert reload["planned"] == {"nodes": 1, "gpus": 1}
    assert planned == [(1, 8), (1, 1)]

    def uncertain(_context: str, **kwargs) -> dict:
        return build_capacity_census(
            {
                "items": [
                    {
                        "metadata": {
                            "namespace": "fleet-train-jobs",
                            "name": "chris-q38-unclassified",
                            "uid": "44444444-4444-4444-8444-444444444444",
                            "resourceVersion": "1",
                            "labels": {},
                        },
                        "spec": {
                            "nodeName": "gpu-node",
                            "containers": [
                                {
                                    "resources": {
                                        "requests": {"nvidia.com/gpu": 1},
                                        "limits": {"nvidia.com/gpu": 1},
                                    }
                                }
                            ],
                        },
                        "status": {"phase": "Running", "containerStatuses": []},
                    }
                ]
            },
            {"items": []},
            observed_at=observed_at,
            **kwargs,
        )

    with pytest.raises(JobsError, match="stale or incomplete"):
        launch.capacity_gate(plan, mechanics.job_request(plan), reader=uncertain)

    def unavailable(_context: str, **_kwargs) -> dict:
        raise TimeoutError

    with pytest.raises(JobsError, match="census failed"):
        launch.capacity_gate(plan, mechanics.job_request(plan), reader=unavailable)


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


def test_kubernetes_duplicate_gate_allows_only_exact_terminal_sfs_observer() -> None:
    plan = _plan()
    request = mechanics.job_request(plan)
    package = build_sfs_output_job(plan, request, 1)
    job = copy.deepcopy(package.job)
    job["status"] = {"conditions": [{"type": "Complete", "status": "True"}]}
    pod = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": job["metadata"]["name"] + "-abcde",
            "annotations": copy.deepcopy(job["spec"]["template"]["metadata"]["annotations"]),
            "labels": copy.deepcopy(job["spec"]["template"]["metadata"]["labels"]),
        },
        "status": {
            "phase": "Succeeded",
            "containerStatuses": [{"restartCount": 0, "state": {"terminated": {"exitCode": 0}}}],
        },
    }

    def runner(_argv: list[str], **_kwargs) -> NS:
        return NS(
            returncode=0,
            stdout=json.dumps({"items": [job, pod]}),
            stderr="",
        )

    assert launch._kubernetes_absent(plan, request, runner=runner) == 2

    pod["status"]["containerStatuses"][0]["restartCount"] = 1
    with pytest.raises(JobsError, match="already contains"):
        launch._kubernetes_absent(plan, request, runner=runner)


def test_kubernetes_duplicate_gate_never_exempts_training_resources() -> None:
    plan = _plan()
    request = mechanics.job_request(plan)
    item = build_sfs_output_job(plan, request, 1).job
    item["metadata"]["name"] = request["name"]
    item["status"] = {"conditions": [{"type": "Complete", "status": "True"}]}

    def runner(_argv: list[str], **_kwargs) -> NS:
        return NS(returncode=0, stdout=json.dumps({"items": [item]}), stderr="")

    with pytest.raises(JobsError, match="already contains"):
        launch._kubernetes_absent(plan, request, runner=runner)
