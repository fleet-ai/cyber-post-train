"""Synthetic native-SkyRL checkpoint evidence; no cluster, Fleet, or private logs."""

from __future__ import annotations

import copy
import json
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from test_export import distributed
from test_rl_data import setup  # noqa: F401
from test_skyrl_data import skyrl as data_setup  # noqa: F401
from test_skyrl_training import prepared  # noqa: F401
from torch.distributed.tensor import Shard

from cyber_post_train.jobs import digest
from evals.fleet import opencode_self_hosted as fleet
from training import dev_cleanup_observer as cleanup_observer
from training import export as native_export
from training import skyrl, skyrl_prod9_direct, skyrl_prod9_reload, skyrl_prod9_training
from training import skyrl_posttrain as post
from training import skyrl_prod9_hardening as prod9
from training.sft_runtime import digest as file_digest
from training.sft_runtime import write_receipt


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def sealed(value: dict, field: str = "sha256", prefix: bool = True) -> dict:
    result = copy.deepcopy(value)
    value_digest = digest(result)
    result[field] = ("sha256:" if prefix else "") + value_digest
    return result


def model_base(root: Path) -> dict:
    root.mkdir()
    tensors = {
        key: torch.tensor([float(index)], dtype=torch.bfloat16)
        for index, key in enumerate(native_export.FROZEN_MTP_KEYS)
    }
    tensors["weight"] = torch.zeros(8, dtype=torch.bfloat16)
    save_file(tensors, root / "model-00001-of-00001.safetensors")
    write(
        root / "model.safetensors.index.json",
        {"weight_map": {key: "model-00001-of-00001.safetensors" for key in tensors}},
    )
    for name in native_export.SIDECARS:
        write(root / name, {"synthetic": name})
    return tensors


def checkpoint(root: Path, step: int, *, changed: bool) -> None:
    policy = root / f"checkpoints/global_step_{step}/policy"
    (policy / "huggingface").mkdir(parents=True)
    write(policy / "huggingface/config.json", {})
    write(policy / "fsdp_config.json", {"fsdp_strategy": "fsdp", "world_size": 8})
    torch.save({"global_step": step}, policy.parent / "trainer_state.pt")
    torch.save({"_num_yielded": 1}, policy.parent / "data.pt")
    for rank in range(8):
        local = torch.tensor([float(rank + 1 if changed else 0)], dtype=torch.float32)
        state = {"weight": distributed(local, [8], Shard(0), tuple(range(8)))}
        torch.save(state, policy / f"model_world_size_8_rank_{rank}.pt")
        torch.save({"state": torch.ones(1)}, policy / f"optim_world_size_8_rank_{rank}.pt")
        torch.save({"rng": torch.ones(1)}, policy / f"extra_state_world_size_8_rank_{rank}.pt")


def episode(
    plan: dict, batch_dir: Path, batch: dict, index: int, source: dict, reward: float
) -> None:
    directory = batch_dir / f"episode-{index}"
    directory.mkdir()
    binding = copy.deepcopy(source)
    binding.update(
        run_id=f"{plan['run_name']}-{batch_dir.name}-{index}",
        native_batch={
            "phase": batch["phase"],
            "global_step": batch["global_step"],
            "trajectory_ids": batch["trajectory_ids"],
        },
        sampling={},
    )
    binding["config_sha256"] = fleet.digest_without(binding, "config_sha256")
    instance = f"synthetic-instance-{batch_dir.name}-{index}"
    execution = str(uuid.uuid5(uuid.NAMESPACE_DNS, instance))
    values = {
        "binding.json": binding,
        "instance.json": {"instance_id": instance, "evidence_run_id": instance + "-evidence"},
        "conversation.json": {"messages": [{"role": "assistant", "content": "private"}]},
        "reward.json": {
            "task_key": binding["task"]["key"],
            "task_version_id": binding["task"]["version_id"],
            "instance_id": instance,
            "reward": reward,
            "verifier_execution_id": execution,
            "direct_authority_attestation": {
                "schema_version": fleet.DIRECT_AUTHORITY_ATTESTATION_SCHEMA,
                "context": {
                    "task_key": binding["task"]["key"],
                    "task_version_id": binding["task"]["version_id"],
                    "instance_id": instance,
                    "evidence_run_id": instance + "-evidence",
                    "verifier_version_id": binding["verifier"]["version_id"],
                    "scoring_payload_mode": fleet.RUNTIME_EVIDENCE_ONLY_V3,
                },
                "activity": {
                    "result_schema_version": "cyber_verification_result_v3",
                    "reward": reward,
                    "task_version_id": binding["task"]["version_id"],
                    "verifier_execution_id": execution,
                },
                "shadow": {
                    "mode": "authoritative",
                    "status": "authoritative",
                    "match": True,
                    "production_execution_id": execution,
                    "direct_verifier": {
                        "status": "authoritative",
                        "match": True,
                        "execution_id": execution,
                        "verifier_contract_version": binding["authority"][
                            "required_cyber_contract"
                        ]["verifier_contract"],
                        "context_schema_version": "cyber_verification_context_v1",
                    },
                },
                "data_minimization": {
                    "components_included": False,
                    "diagnostics_included": False,
                    "evidence_payloads_included": False,
                    "prompts_included": False,
                    "traces_included": False,
                    "flags_included": False,
                },
            },
        },
        "cleanup.json": {
            "create_attempted": True,
            "instance_created": True,
            "instance_id": instance,
            "instance_closed": True,
            "possible_instance_leak": False,
        },
        "recording.json": {"samples": [{"tokens": [1], "response_length": 1}]},
    }
    for name, value in values.items():
        write(directory / name, value)
    accepted = {
        "task_version_id": binding["task"]["version_id"],
        "instance_id": instance,
        "verifier_execution_id": execution,
        "done_reason": "report_submitted",
        "config_sha256": binding["config_sha256"],
        "sample_count": 1,
        "files": {name: fleet.sha256((directory / name).read_bytes()) for name in values},
    }
    write(
        directory / "ACCEPTED.json",
        {**accepted, "sha256": fleet.digest_without(accepted, "sha256")},
    )


def batch(plan: dict, phase: str, step: int, rows: list[dict], reward_offset: int) -> None:
    args = plan["arguments"]
    repetitions = args["samples_per_prompt"] if phase == "train" else 1
    count = args["groups"] if phase == "train" else args["dev_rows"]
    trajectories = [[str(uid), rep] for uid in range(count) for rep in range(repetitions)]
    value = {
        "schema": "cyber_skyrl_batch_v1",
        "phase": phase,
        "global_step": step,
        "trajectory_ids": trajectories,
        "data_sha256": plan["data"]["sha256"],
        "optimizer_step_verified": False,
    }
    identifier = fleet.sha256(
        fleet.canonical_json({k: value[k] for k in ("phase", "global_step", "trajectory_ids")})
    ).removeprefix("sha256:")[:24]
    directory = Path(plan["output_root"]) / "episodes/batches" / identifier
    directory.mkdir(parents=True)
    for index, (uid, _) in enumerate(trajectories):
        source = json.loads(rows[int(uid)]["cyber_config_json"])
        episode(plan, directory, value, index, source, float((index + reward_offset) % 2))
    write(directory / "COLLECTED.json", sealed(value))


@pytest.fixture
def completed_rl(prepared, monkeypatch):  # noqa: F811
    plan, tmp = copy.deepcopy(prepared.plan), prepared.state.tmp
    # Unit fixtures use a temporary root rather than shared SFS. The production
    # path policy is tested by SkyRL's own configuration suite.
    monkeypatch.setattr(skyrl.SkyRLConfig, "validate", lambda self: None)
    data = tmp / "out"
    plan["arguments"].update(
        data_manifest=str(data / "manifest.json"),
        train_data=str(data / "train.jsonl"),
        dev_data=str(data / "dev.jsonl"),
        train_rows=1,
        dev_rows=1,
    )
    plan["data"] = json.loads((data / "manifest.json").read_text())
    output = tmp / "result"
    plan["output_root"] = plan["arguments"]["output_root"] = str(output)
    base = tmp / "base"
    model_base(base)
    plan["model"]["root"] = plan["arguments"]["model_root"] = str(base)
    plan["model"]["files"] = [
        {"path": path.name, "sha256": file_digest(path)} for path in sorted(base.iterdir())
    ]
    for split in ("train", "dev"):
        path = data / f"{split}.jsonl"
        row = json.loads(path.read_text())
        config = json.loads(row["cyber_config_json"])
        config["model"]["root"] = str(base)
        config["config_sha256"] = fleet.digest_without(config, "config_sha256")
        row["cyber_config_json"] = fleet.canonical_json(config).decode()
        payload = fleet.canonical_json(row) + b"\n"
        path.write_bytes(payload)
        plan["data"]["files"][split]["sha256"] = fleet.sha256(payload)
    plan["data"]["sha256"] = "sha256:" + digest(
        {key: value for key, value in plan["data"].items() if key != "sha256"}
    )
    write(data / "manifest.json", plan["data"])
    plan["native_overrides"] = skyrl.overrides(skyrl.SkyRLConfig(**plan["arguments"]))
    checkpoint(output, 1, changed=True)
    checkpoint(output, 2, changed=True)
    (output / "checkpoints/latest_ckpt_global_step.txt").write_text("2")
    source_rows = {
        split: [json.loads(line) for line in (data / f"{split}.jsonl").read_text().splitlines()]
        for split in ("train", "dev")
    }
    for phase, step in (("eval", 0), ("train", 1), ("eval", 1), ("train", 2), ("eval", 2)):
        batch(plan, phase, step, source_rows["dev" if phase == "eval" else "train"], step)
    with (output / "metrics.jsonl").open("w") as stream:
        for step in (1, 2):
            stream.write(
                json.dumps(
                    {
                        "optimizer_step": step,
                        "time": float(step),
                        "policy/loss": 0.5,
                        "policy/approx_kl": 0.01,
                        "policy/entropy": 0.2,
                        "policy/grad_norm": 1.0,
                    }
                )
                + "\n"
            )
    terminal = {
        "status": "native_loop_returned",
        "plan_sha256": digest(plan),
        "checkpoint_global_step": 2,
        "completed_batches": 5,
        "completed_at": 1.0,
        "optimizer_update_independently_verified": False,
        "checkpoint_reload_verified": False,
    }
    write(output / "NATIVE_TRAINING_COMPLETE.json", sealed(terminal, prefix=False))
    return SimpleNamespace(plan=plan, root=output, data=data, base=base, manifest=tmp / "seal.json")


@pytest.fixture
def completed_prod9_rl(completed_rl, monkeypatch):
    state = completed_rl
    state.plan.update(
        schema=skyrl_prod9_training.SCHEMA,
        runtime_sha256=digest(skyrl_prod9_training._runtime()),
        prod9_runtime=skyrl_prod9_training._binding(),
    )
    terminal_path = state.root / "NATIVE_TRAINING_COMPLETE.json"
    terminal = json.loads(terminal_path.read_text())
    terminal["plan_sha256"] = digest(state.plan)
    write(
        terminal_path,
        sealed({key: value for key, value in terminal.items() if key != "sha256"}, prefix=False),
    )
    create_once_root = state.root.parent / "prod9-create-once"
    create_once_root.mkdir()
    monkeypatch.setattr(prod9, "CREATE_ONCE_ROOT", create_once_root)
    monkeypatch.setattr(
        skyrl_prod9_training,
        "job_request",
        lambda plan: {"synthetic_request_plan_sha256": digest(plan)},
    )
    monkeypatch.setattr(
        skyrl_prod9_reload,
        "_reload_run_dir",
        lambda plan, step: f"/mnt/sfs/jobs/{plan['run_name']}-p{step}-reload-v1",
    )
    state.manifest = state.root.parent / "prod9-seal.json"
    return state


def test_rl_checkpoint_seal_and_zero_update_bf16_export(completed_rl, monkeypatch):
    state = completed_rl
    before = {path: path.read_bytes() for path in state.root.rglob("*") if path.is_file()}
    manifest = post.seal_checkpoint(state.plan, state.manifest)
    post.verify_manifest(manifest)
    assert manifest["rollout_evidence"]["train_rollouts"] == 16
    assert manifest["rollout_evidence"]["reward_min"] == 0
    assert manifest["rollout_evidence"]["reward_max"] == 1
    assert manifest["update_evidence"]["changed_parameter_tensors"] == 1
    assert manifest["optimizer_update_verified"] is True
    assert before == {path: path.read_bytes() for path in before}
    output = state.root / "hf-export-v1"
    result = post.export_checkpoint(state.manifest, file_digest(state.manifest), output)
    assert result["schema"] == post.EXPORT_SCHEMA
    assert result["optimizer_steps_executed"] == 0
    assert result["dtype"] == "BF16" and result["trained_tensors"] == 1
    assert before == {path: path.read_bytes() for path in before}
    with pytest.raises(FileExistsError):
        post.export_checkpoint(state.manifest, file_digest(state.manifest), output)


def test_prod9_plan_seals_and_verifies_without_historical_plan_rewrite(
    completed_prod9_rl,
) -> None:
    state = completed_prod9_rl
    manifest = post.seal_checkpoint(state.plan, state.manifest)
    post.verify_manifest(manifest)
    assert manifest["source_plan"] == state.plan
    assert manifest["source_plan_sha256"] == digest(state.plan)

    broken = copy.deepcopy(state.plan)
    broken.pop("prod9_runtime")
    with pytest.raises(ValueError, match="runtime binding"):
        post._validate_plan(broken)


def test_prod9_reload_spec_binds_exact_checkpoint_export_and_model(
    completed_prod9_rl, monkeypatch
) -> None:
    state = completed_prod9_rl
    monkeypatch.setattr(
        skyrl_prod9_reload,
        "_reload_run_dir",
        lambda plan, step: f"/mnt/sfs/jobs/{plan['run_name']}-p{step}-reload-v1",
    )
    paths = prod9.terminal_paths(state.plan)
    manifest = post.seal_checkpoint(state.plan, paths["checkpoint_manifest"])
    exported = post.export_checkpoint(
        paths["checkpoint_manifest"],
        file_digest(paths["checkpoint_manifest"]),
        paths["export"].parent,
    )
    spec = skyrl_prod9_reload.build_spec(
        state.plan,
        manifest,
        exported,
        checkpoint_manifest_file_sha256=file_digest(paths["checkpoint_manifest"]),
        export_file_sha256=file_digest(paths["export"]),
    )

    observed_export, layout = skyrl_prod9_reload.validate_source(spec)
    assert spec["name"] == state.plan["run_name"] + "-p2-reload-v1"
    assert spec["resources"] == {
        "nodes": 1,
        "gpus": 1,
        "priority_class": "c1",
        "queue_priority_class": "q1",
        "maximum_seconds": 1800,
    }
    assert spec["model"]["repo"] == "Qwen/Qwen3.8-27B"
    assert spec["model"]["revision"] == state.plan["model"]["revision"]
    assert observed_export == exported
    assert layout

    wrong = copy.deepcopy(exported)
    wrong["source_checkpoint_receipt_sha256"] = "0" * 64
    wrong["receipt_sha256"] = digest(
        {key: item for key, item in wrong.items() if key != "receipt_sha256"}
    )
    with pytest.raises(ValueError, match="checkpoint/export identity"):
        skyrl_prod9_reload.build_spec(
            state.plan,
            manifest,
            wrong,
            checkpoint_manifest_file_sha256=file_digest(paths["checkpoint_manifest"]),
            export_file_sha256=file_digest(paths["export"]),
        )


def _terminal_inputs(state):
    from cyber_post_train.gpu_capacity import build_capacity_census

    paths = prod9.terminal_paths(state.plan)
    manifest = post.seal_checkpoint(state.plan, paths["checkpoint_manifest"])
    exported = post.export_checkpoint(
        paths["checkpoint_manifest"],
        file_digest(paths["checkpoint_manifest"]),
        paths["export"].parent,
    )
    training_receipt = json.loads((state.root / "NATIVE_TRAINING_COMPLETE.json").read_text())
    write_receipt(
        paths["gpu_check"],
        {
            "schema": "cyber_hf_export_check_v1",
            "status": "passed",
            "export_sha256": file_digest(paths["export"]),
            "export_receipt_sha256": exported["receipt_sha256"],
            "checkpoint_manifest_file_sha256": file_digest(paths["checkpoint_manifest"]),
            "checkpoint_receipt_sha256": manifest["receipt_sha256"],
            "source_plan_sha256": digest(state.plan),
            "model_repo": state.plan["model"]["repo"],
            "model_revision": state.plan["model"]["revision"],
            "checker_sha256": file_digest(Path(skyrl_prod9_reload.__file__)),
            "optimizer_steps_executed": 0,
            "gpus": 1,
            "gpu_reload_verified": True,
            "source_unchanged": True,
            "finite_logits": True,
            "generated_tokens": 2,
            "serving_qualified": False,
        },
    )
    gpu_receipt = json.loads(paths["gpu_check"].read_text())
    reload_spec = skyrl_prod9_reload.build_spec(
        state.plan,
        manifest,
        exported,
        checkpoint_manifest_file_sha256=file_digest(paths["checkpoint_manifest"]),
        export_file_sha256=file_digest(paths["export"]),
    )
    reload_operation = prod9.reload_operation_root(reload_spec)
    reload_operation.mkdir()
    paths["reload_creator_binding"] = prod9.creator_binding_path(reload_operation, "reload")
    paths["reload_create_journal"] = reload_operation / "PROD9_RELOAD_RAYJOB_CREATE.jsonl"

    def creator(path, *, prefix, run_dir, gpus, maximum, suffix, uid):
        value = sealed(
            {
                "schema": cleanup_observer.JOBS_API_EXACT_BINDING_SCHEMA,
                "status": "bound_exact_uid_cleanup_not_started",
                "prefix_guard_sha256": "sha256:" + "a" * 64,
                "context": cleanup_observer.PROD_CONTEXT,
                "namespace": cleanup_observer.NAMESPACE,
                "jobs_api_run_name": prefix + "-" + suffix,
                "jobs_api_run_id": "00000000-0000-0000-0000-000000000019",
                "run_dir": run_dir,
                "rayjob_name": prefix + "-" + suffix,
                "rayjob_uid": uid,
                "rayjob_created_at": "2026-09-21T00:00:00Z",
                "bound_at": "2026-09-21T00:00:01Z",
                "failure_alerts": "off",
                "maximum_seconds": maximum,
                "expected_gpus": gpus,
                "cleanup_started": False,
            }
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        write(path, value)
        return value

    def journal(
        path,
        *,
        schema,
        plan_sha,
        request_sha,
        creator_value,
        gpus,
        manifest_sha,
        spec_sha=None,
    ):
        census = build_capacity_census(
            {"items": []},
            {"items": []},
            planned_nodes=1,
            planned_gpus=gpus,
            observed_at="2026-09-21T00:00:00Z",
        )
        capacity = sealed(
            {
                "schema": "synthetic_prod9_capacity_gate_v1",
                "status": "passed",
                "context": prod9.PROD_CONTEXT,
                "plan_sha256": plan_sha,
                "request_sha256": request_sha,
                "manifest_sha256": manifest_sha,
                "planned": {"nodes": 1, "gpus": gpus},
                "capacity_census": census,
            }
        )
        preview = sealed(
            {
                "schema": "synthetic_prod9_live_preview_v1",
                "status": "passed",
                "context": prod9.PROD_CONTEXT,
                "request_sha256": request_sha,
                "manifest_sha256": manifest_sha,
                "failure_alerts": "off",
                "priority": "c1",
                "queue_priority": "q1",
                "nodes": 1,
                "gpus": gpus,
            }
        )
        auth_sha = "sha256:" + "d" * 64
        preview_sha = "sha256:" + "e" * 64
        created_body = {
            "schema": schema,
            "status": "submitted_once_and_bound_exact_uid",
            "plan_sha256": plan_sha,
            "request_sha256": request_sha,
            "manifest_sha256": manifest_sha,
            "authorization_sha256": auth_sha,
            "live_jobs_preview_sha256": preview_sha,
            "live_preview_proof_sha256": preview["sha256"],
            "capacity_gate_sha256": capacity["sha256"],
            "jobs_api_run_name": creator_value["jobs_api_run_name"],
            "jobs_api_run_id": creator_value["jobs_api_run_id"],
            "rayjob_name": creator_value["rayjob_name"],
            "rayjob_uid": creator_value["rayjob_uid"],
            "creator_binding_sha256": creator_value["sha256"],
            "created_at": creator_value["rayjob_created_at"],
            "failure_alerts": "off",
            "priority": "c1",
            "queue_priority": "q1",
            "nodes": 1,
            "gpus": gpus,
        }
        if spec_sha is not None:
            created_body["spec_sha256"] = spec_sha
        created = sealed(created_body)
        rows = [
            {
                "state": "POST_INTENT_DO_NOT_RETRY",
                "plan_sha256": plan_sha,
                "request_sha256": request_sha,
                "manifest_sha256": manifest_sha,
                "authorization_sha256": auth_sha,
                "live_jobs_preview_sha256": preview_sha,
                "live_preview_proof": preview,
                "capacity_gate": capacity,
            },
            {
                "state": "POST_RESPONSE",
                "name": creator_value["jobs_api_run_name"],
                "job_id": creator_value["jobs_api_run_id"],
                "run_dir": creator_value["run_dir"],
                "status": "queued",
                "created_at": None,
            },
            created,
        ]
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        return created

    def observer(path, *, creator_value, gpus, receipt, child_suffix):
        value = sealed(
            {
                "schema": cleanup_observer.JOBS_API_EXACT_OBSERVER_SCHEMA,
                "status": "released_after_terminal",
                "reason": "exact_root_and_observed_children_absent",
                "release_confirmed": True,
                "context": cleanup_observer.PROD_CONTEXT,
                "namespace": cleanup_observer.NAMESPACE,
                "binding_sha256": creator_value["sha256"],
                "jobs_api_run_name": creator_value["jobs_api_run_name"],
                "jobs_api_run_id": creator_value["jobs_api_run_id"],
                "rayjob_name": creator_value["rayjob_name"],
                "rayjob_uid": creator_value["rayjob_uid"],
                "created_at": creator_value["rayjob_created_at"],
                "deadline_at": "2026-09-21T06:00:00Z",
                "maximum_seconds": creator_value["maximum_seconds"],
                "terminal_status": "Succeeded",
                "owned_inventory_observed": True,
                "raycluster_identity_observed": True,
                "workloads": [{"name": child_suffix + "-workload", "uid": str(uuid.uuid4())}],
                "rayclusters": [{"name": child_suffix + "-cluster", "uid": str(uuid.uuid4())}],
                "pods": [{"name": child_suffix + "-pod-23", "uid": str(uuid.uuid4())}],
                "peak_gpus": gpus,
                "active_gpus": 0,
                "restarts": 0,
                "exit_codes": [0],
                "receipt": receipt,
                "cleanup_status": "not_requested_root_already_absent",
                "cleanup_requested": False,
                "private_logs_read": False,
            }
        )
        write(path, value)
        return value

    training_creator = creator(
        paths["training_creator_binding"],
        prefix=state.plan["run_name"],
        run_dir=state.plan["output_root"],
        gpus=8,
        maximum=skyrl_prod9_direct.MAXIMUM_SECONDS,
        suffix="1a2b3c4d",
        uid="00000000-0000-0000-0000-000000000020",
    )
    journal(
        paths["training_create_journal"],
        schema=skyrl_prod9_direct.CREATED_SCHEMA,
        plan_sha="sha256:" + digest(state.plan),
        request_sha="sha256:" + digest(skyrl_prod9_training.job_request(state.plan)),
        creator_value=training_creator,
        gpus=8,
        manifest_sha="sha256:" + "b" * 64,
    )
    observer(
        paths["training_observer"],
        creator_value=training_creator,
        gpus=8,
        receipt=training_receipt,
        child_suffix=state.plan["run_name"],
    )
    reload_request = skyrl_prod9_reload.job_request(reload_spec)
    reload_creator = creator(
        paths["reload_creator_binding"],
        prefix=reload_request["name"],
        run_dir=reload_spec["run_dir"],
        gpus=1,
        maximum=skyrl_prod9_reload.MAXIMUM_SECONDS,
        suffix="2b3c4d5e",
        uid="00000000-0000-0000-0000-000000000021",
    )
    journal(
        paths["reload_create_journal"],
        schema=skyrl_prod9_reload.CREATED_SCHEMA,
        plan_sha="sha256:" + digest(state.plan),
        request_sha="sha256:" + digest(reload_request),
        creator_value=reload_creator,
        gpus=1,
        manifest_sha="sha256:" + "a" * 64,
        spec_sha=reload_spec["sha256"],
    )
    observer(
        paths["reload_observer"],
        creator_value=reload_creator,
        gpus=1,
        receipt=gpu_receipt,
        child_suffix=reload_spec["name"],
    )
    return paths, manifest, exported


def test_rl_terminal_acceptance_requires_exact_seal_export_reload_and_release(
    completed_prod9_rl,
):
    state = completed_prod9_rl
    paths, manifest, exported = _terminal_inputs(state)

    accepted = prod9.accept_terminal(
        state.plan,
        checkpoint_manifest=paths["checkpoint_manifest"],
        export=paths["export"],
        training_observer=paths["training_observer"],
        training_creator_binding=paths["training_creator_binding"],
        training_create_journal=paths["training_create_journal"],
        gpu_check=paths["gpu_check"],
        reload_observer=paths["reload_observer"],
        reload_creator_binding=paths["reload_creator_binding"],
        reload_create_journal=paths["reload_create_journal"],
        output=paths["accepted"],
    )

    assert accepted["schema"] == prod9.ACCEPTANCE_SCHEMA
    assert accepted["checkpoint_manifest_receipt_sha256"] == manifest["receipt_sha256"]
    assert accepted["export_receipt_sha256"] == exported["receipt_sha256"]
    assert accepted["training_rayjob_name"] == state.plan["run_name"] + "-1a2b3c4d"
    assert accepted["training_pod_names"] == [state.plan["run_name"] + "-pod-23"]
    assert accepted["complete_bf16_reload_verified"] is True
    assert accepted["gpu_resources_released"] is True
    with pytest.raises(FileExistsError):
        prod9.accept_terminal(
            state.plan,
            checkpoint_manifest=paths["checkpoint_manifest"],
            export=paths["export"],
            training_observer=paths["training_observer"],
            training_creator_binding=paths["training_creator_binding"],
            training_create_journal=paths["training_create_journal"],
            gpu_check=paths["gpu_check"],
            reload_observer=paths["reload_observer"],
            reload_creator_binding=paths["reload_creator_binding"],
            reload_create_journal=paths["reload_create_journal"],
            output=paths["accepted"],
        )


@pytest.mark.parametrize(
    "fault",
    [
        "gpu",
        "checker",
        "release",
        "terminal",
        "observer_manifest",
        "training_release",
        "training_uid",
        "training_receipt",
        "training_created",
        "training_request",
        "reload_capacity",
        "wrong_path",
    ],
)
def test_rl_terminal_acceptance_rejects_incomplete_reload_or_release(completed_prod9_rl, fault):
    state = completed_prod9_rl
    paths, _, _ = _terminal_inputs(state)
    if fault == "gpu":
        value = json.loads(paths["gpu_check"].read_text())
        value["gpu_reload_verified"] = False
        write_receipt(
            paths["gpu_check"],
            {key: item for key, item in value.items() if key != "receipt_sha256"},
            replace=True,
        )
    elif fault == "checker":
        value = json.loads(paths["gpu_check"].read_text())
        value["checker_sha256"] = "0" * 64
        write_receipt(
            paths["gpu_check"],
            {key: item for key, item in value.items() if key != "receipt_sha256"},
            replace=True,
        )
    elif fault == "release":
        value = json.loads(paths["reload_observer"].read_text())
        value["active_gpus"] = 1
        write(
            paths["reload_observer"],
            sealed({key: item for key, item in value.items() if key != "sha256"}),
        )
    elif fault == "terminal":
        value = json.loads(paths["reload_observer"].read_text())
        value["terminal_status"] = "Failed"
        write(
            paths["reload_observer"],
            sealed({key: item for key, item in value.items() if key != "sha256"}),
        )
    elif fault == "observer_manifest":
        value = json.loads(paths["reload_observer"].read_text())
        value["binding_sha256"] = "sha256:" + "0" * 64
        write(
            paths["reload_observer"],
            sealed({key: item for key, item in value.items() if key != "sha256"}),
        )
    elif fault == "training_release":
        value = json.loads(paths["training_observer"].read_text())
        value["active_gpus"] = 8
        write(
            paths["training_observer"],
            sealed({key: item for key, item in value.items() if key != "sha256"}),
        )
    elif fault == "training_uid":
        value = json.loads(paths["training_observer"].read_text())
        value["uid"] = value["rayjob_uid"] = "00000000-0000-0000-0000-000000000099"
        write(
            paths["training_observer"],
            sealed({key: item for key, item in value.items() if key != "sha256"}),
        )
    elif fault == "training_receipt":
        value = json.loads(paths["training_observer"].read_text())
        value["receipt"]["checkpoint_global_step"] += 1
        write(
            paths["training_observer"],
            sealed({key: item for key, item in value.items() if key != "sha256"}),
        )
    elif fault == "training_created":
        rows = [
            json.loads(line) for line in paths["training_create_journal"].read_text().splitlines()
        ]
        rows[2]["capacity_gate_sha256"] = "sha256:" + "0" * 64
        rows[2] = sealed({key: item for key, item in rows[2].items() if key != "sha256"})
        paths["training_create_journal"].write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        )
    elif fault == "training_request":
        rows = [
            json.loads(line) for line in paths["training_create_journal"].read_text().splitlines()
        ]
        wrong = "sha256:" + "9" * 64
        rows[0]["request_sha256"] = wrong
        for key in ("capacity_gate", "live_preview_proof"):
            rows[0][key]["request_sha256"] = wrong
            rows[0][key] = sealed(
                {item: value for item, value in rows[0][key].items() if item != "sha256"}
            )
        rows[2]["request_sha256"] = wrong
        rows[2]["capacity_gate_sha256"] = rows[0]["capacity_gate"]["sha256"]
        rows[2]["live_preview_proof_sha256"] = rows[0]["live_preview_proof"]["sha256"]
        rows[2] = sealed({key: item for key, item in rows[2].items() if key != "sha256"})
        paths["training_create_journal"].write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        )
    elif fault == "reload_capacity":
        rows = [
            json.loads(line) for line in paths["reload_create_journal"].read_text().splitlines()
        ]
        census = rows[0]["capacity_gate"]["capacity_census"]
        census["planned"] = {"nodes": 1, "gpus": 8}
        census["sha256"] = digest({key: item for key, item in census.items() if key != "sha256"})
        capacity = rows[0]["capacity_gate"]
        rows[0]["capacity_gate"] = sealed(
            {key: item for key, item in capacity.items() if key != "sha256"}
        )
        rows[2]["capacity_gate_sha256"] = rows[0]["capacity_gate"]["sha256"]
        rows[2] = sealed({key: item for key, item in rows[2].items() if key != "sha256"})
        paths["reload_create_journal"].write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        )
    else:
        paths["export"] = paths["export"].with_name("other.json")
    with pytest.raises(ValueError):
        prod9.accept_terminal(
            state.plan,
            checkpoint_manifest=paths["checkpoint_manifest"],
            export=paths["export"],
            training_observer=paths["training_observer"],
            training_creator_binding=paths["training_creator_binding"],
            training_create_journal=paths["training_create_journal"],
            gpu_check=paths["gpu_check"],
            reload_observer=paths["reload_observer"],
            reload_creator_binding=paths["reload_creator_binding"],
            reload_create_journal=paths["reload_create_journal"],
            output=paths["accepted"],
        )
    assert not paths["accepted"].exists()


def test_rl_seal_rejects_source_drift_during_full_rehash(completed_rl):
    state = completed_rl

    def mutate(*_):
        (state.base / "config.json").write_text('{"changed":true}')

    with pytest.raises(ValueError, match="changed during"):
        post.seal_checkpoint(state.plan, state.manifest, progress=mutate)
    assert not state.manifest.exists()


@pytest.mark.parametrize(
    "fault",
    [
        "missing_file",
        "corrupt_file",
        "wrong_step",
        "wrong_plan",
        "partial_payload",
        "source_drift",
        "reward_constant",
        "metric_missing",
    ],
)
def test_rl_seal_rejects_broken_evidence(completed_rl, fault):
    state = completed_rl
    checkpoint_file = (
        state.root / "checkpoints/global_step_2/policy/extra_state_world_size_8_rank_7.pt"
    )
    if fault == "missing_file":
        checkpoint_file.unlink()
    elif fault == "corrupt_file":
        checkpoint_file.write_bytes(b"corrupt")
    elif fault == "wrong_step":
        torch.save({"global_step": 1}, state.root / "checkpoints/global_step_2/trainer_state.pt")
    elif fault == "wrong_plan":
        state.plan["arguments"]["lr"] = 3e-6
    elif fault == "partial_payload":
        accepted = next(state.root.glob("episodes/batches/*/episode-0/ACCEPTED.json"))
        value = json.loads(accepted.read_text())
        value["files"].pop("recording.json")
        write(accepted, sealed({k: v for k, v in value.items() if k != "sha256"}))
    elif fault == "source_drift":
        (state.data / "train.jsonl").write_text("changed")
    elif fault == "reward_constant":
        for path in state.root.glob("episodes/batches/*/episode-*/reward.json"):
            value = json.loads(path.read_text())
            value["reward"] = 0.0
            write(path, value)
            accepted_path = path.parent / "ACCEPTED.json"
            accepted = json.loads(accepted_path.read_text())
            accepted["files"]["reward.json"] = fleet.sha256(path.read_bytes())
            write(accepted_path, sealed({k: v for k, v in accepted.items() if k != "sha256"}))
    else:
        rows = [
            json.loads(line) for line in (state.root / "metrics.jsonl").read_text().splitlines()
        ]
        for row in rows:
            row.pop("policy/entropy")
        (state.root / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises((ValueError, KeyError, json.JSONDecodeError)):
        post.seal_checkpoint(state.plan, state.manifest)
    assert not state.manifest.exists()


@pytest.mark.parametrize("fault", ["wrong_plan", "wrong_step", "partial", "source"])
def test_rl_manifest_reverification_rejects_drift(completed_rl, fault):
    state = completed_rl
    value = copy.deepcopy(post.seal_checkpoint(state.plan, state.manifest))
    if fault == "wrong_plan":
        value["source_plan_sha256"] = "0" * 64
    elif fault == "wrong_step":
        value["optimizer_step"] = 1
    elif fault == "partial":
        value["total_bytes"] -= value["files"].pop("policy/extra_state_world_size_8_rank_7.pt")[
            "bytes"
        ]
    else:
        (state.base / "config.json").write_text("{}")
    if fault != "source":
        value["receipt_sha256"] = digest(
            {key: item for key, item in value.items() if key != "receipt_sha256"}
        )
    with pytest.raises(ValueError):
        post.verify_manifest(value, check_files=fault == "source")
