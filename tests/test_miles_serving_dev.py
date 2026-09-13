from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from cyber_post_train import cli
from cyber_post_train.jobs import digest
from training import miles_serving_dev as serving
from training import serving_registration
from training.io import canonical_json, digest_json, file_sha256
from training.post_sft_artifacts import QWEN36_EXACT_MTP_OMISSION_KEYS
from training.sft_runtime import write_receipt

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "evals/fleet/serving/qwen38-27b-dedicated-v1.registration.json"
SHA = "a" * 64
CHECKER_SHA = serving_registration.EXPORT_CHECK_SHA256


def _write(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n")
    return file_sha256(path)


def _artifacts(export_root: Path) -> tuple[dict, dict, dict[str, str]]:
    files = {
        "config.json": {"bytes": 2, "sha256": SHA},
        "model-00001-of-00001.safetensors": {"bytes": 2, "sha256": "b" * 64},
        "model.safetensors.index.json": {"bytes": 2, "sha256": "c" * 64},
        "tokenizer.json": {"bytes": 2, "sha256": "d" * 64},
        "tokenizer_config.json": {"bytes": 2, "sha256": "e" * 64},
        "chat_template.jinja": {"bytes": 2, "sha256": "f" * 64},
    }
    export = {
        "schema": "cyber_miles_native_hf_export_v2",
        "output_root": str(export_root),
        "sha256": "1" * 64,
        "tensor_inventory_sha256": "2" * 64,
        "dtype": "BF16",
        "optimizer_updates_executed": 0,
        "model": {
            "repo": serving.MODEL_REPO,
            "revision": serving.MODEL_REVISION,
            "base_root": "/mnt/sfs/models/qwen38-exact-base",
            "native_parallelism": {"tensor": 4, "context": 2, "world_size": 8},
        },
        "source": {
            "source_plan_sha256": "3" * 64,
            "checkpoint": {"receipt_sha256": "4" * 64},
        },
        "files": files,
    }
    reload_receipt = {
        "sha256": "5" * 64,
        "serving_qualified": False,
        "export_path": str(export_root / "EXPORT.json"),
        "export_file_sha256": "6" * 64,
        "export_receipt_sha256": export["sha256"],
        "export_tensor_inventory_sha256": export["tensor_inventory_sha256"],
    }
    validation = {
        "export_receipt_sha256": "1" * 64,
        "prediction_sha256": "7" * 64,
        "reload_result_sha256": "8" * 64,
        "controller_terminal_sha256": "9" * 64,
        "external_release_sha256": "0" * 64,
    }
    return export, reload_receipt, validation


def _sft_config(
    tmp_path: Path,
    *,
    export_change: tuple[str, object] | None = None,
    gpu_change: tuple[str, object] | None = None,
) -> dict:
    export_root = tmp_path / "sft-export"
    export_root.mkdir()
    tensors = {
        f"model.synthetic.{index}.weight": torch.ones(1, dtype=torch.bfloat16)
        for index in range(serving_registration.GPU_MODEL_TENSORS)
    }
    tensors.update(
        {name: torch.ones(1, dtype=torch.bfloat16) for name in QWEN36_EXACT_MTP_OMISSION_KEYS}
    )
    shard = export_root / "model.safetensors"
    save_file(tensors, shard)
    (export_root / "model.safetensors.index.json").write_text(
        canonical_json({"weight_map": {name: shard.name for name in tensors}}) + "\n"
    )
    for name, contents in {
        "config.json": "{}\n",
        "tokenizer.json": "{}\n",
        "tokenizer_config.json": "{}\n",
        "chat_template.jinja": "{{ messages }}\n",
    }.items():
        (export_root / name).write_text(contents)
    files = {
        path.name: {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path).removeprefix("sha256:"),
        }
        for path in export_root.iterdir()
    }
    export = {
        "schema": "cyber_native_checkpoint_hf_export_v1",
        "model_repo": serving.MODEL_REPO,
        "model_revision": serving.MODEL_REVISION,
        "output_root": str(export_root),
        "dtype": "BF16",
        "optimizer_steps_executed": 0,
        "optimizer_step": 186,
        "source_checkpoint_receipt_sha256": "2" * 64,
        "source_manifest_file_sha256": "3" * 64,
        "source_plan_sha256": "4" * 64,
        "code_sha256": {"training/export.py": "5" * 64},
        "all_output_tensors_reopened_equal": True,
        "source_inventory_sizes_mtimes_unchanged": True,
        "restored_base_tensors": list(QWEN36_EXACT_MTP_OMISSION_KEYS),
        "trained_tensors": serving_registration.GPU_MODEL_TENSORS,
        "files": files,
    }
    if export_change:
        export[export_change[0]] = export_change[1]
    export_path = export_root / "EXPORT.json"
    write_receipt(export_path, export)
    export_receipt = json.loads(export_path.read_text())
    export_file_sha = file_sha256(export_path)
    gpu_check = {
        "schema": "cyber_hf_export_check_v1",
        "status": "passed",
        "checker_sha256": CHECKER_SHA,
        "export_sha256": export_file_sha,
        "export_receipt_sha256": export_receipt["receipt_sha256"],
        "gpus": 1,
        "gpu_reload_verified": True,
        "source_unchanged": True,
        "finite_logits": True,
        "optimizer_steps_executed": 0,
        "synthetic_only": True,
        "serving_qualified": False,
        "attention_implementation": "eager",
        "generated_tokens": 2,
        "patched_linear_layers": serving_registration.GPU_PATCHED_LINEAR_LAYERS,
        "loader_contract": serving_registration.GPU_LOADER_CONTRACT,
    }
    if gpu_change:
        gpu_check[gpu_change[0]] = gpu_change[1]
    gpu_path = tmp_path / "gpu-check" / "GPU_CHECK.json"
    write_receipt(gpu_path, gpu_check)
    return {
        "schema": serving.SFT_CONFIG_SCHEMA,
        "name": "q38-teacher-v5-serve-dev1",
        "output_root": "/mnt/sfs/jobs/q38-teacher-v5-serving-dev-unit",
        "base_registration": {"path": str(BASE), "file_sha256": file_sha256(BASE)},
        "export": {"path": str(export_path), "file_sha256": export_file_sha},
        "gpu_check": {"path": str(gpu_path), "file_sha256": file_sha256(gpu_path)},
        "dev_sfs": {
            "namespace": serving.NAMESPACE,
            "pvc_name": "dev-sfs-unit",
            "pvc_uid": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            "mount_root": "/mnt/sfs",
        },
        "cluster": {"target": "dev", "priority": "c1"},
    }


def _sft_preflight(plan: dict) -> dict:
    return serving._sign(serving.preflight(plan))


@pytest.fixture()
def plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    export_root = tmp_path / "export"
    export_root.mkdir()
    export, reload_receipt, validation = _artifacts(export_root)
    monkeypatch.setattr(
        serving,
        "_artifacts",
        lambda config: (copy.deepcopy(export), copy.deepcopy(reload_receipt), validation.copy()),
    )
    base_sha = file_sha256(BASE)
    config = {
        "schema": serving.CONFIG_SCHEMA,
        "name": "q38-miles-serve-dev1",
        "output_root": "/mnt/sfs/jobs/q38-miles-serving-dev-unit",
        "base_registration": {"path": str(BASE), "file_sha256": base_sha},
        "export": {
            "path": str(export_root / "EXPORT.json"),
            "file_sha256": "6" * 64,
        },
        "reload_acceptance": {
            "path": str(tmp_path / "reload" / "HF_RELOAD_ACCEPTED.json"),
            "file_sha256": "a" * 64,
        },
        "cluster": {"target": "dev", "priority": "c1"},
    }
    return serving.compile_plan(config)


@pytest.fixture()
def sft_plan(tmp_path: Path) -> dict:
    return serving.compile_plan(_sft_config(tmp_path))


def test_compiles_one_exact_dev_only_serving_request(plan: dict) -> None:
    request = serving.job_request(plan)
    registration = plan["serving"]["registration"]
    assert plan["cluster_target"] == "dev" and plan["priority"] == "c1"
    assert plan["probe"] == {
        "startup_timeout_seconds": 1800,
        "request_timeout_seconds": 300,
        "shutdown_timeout_seconds": 60,
        "task_content_included": False,
        "benchmark_content_included": False,
        "scores_observed": False,
        "webexploitbench_eligible_for_selection": False,
    }
    assert serving_registration.execution_contract(
        registration
    ) == serving_registration.execution_contract(json.loads(BASE.read_text()))
    assert request["workers"] == request["gpus_per_worker"] == 1
    assert request["priority_class"] == "c1"
    assert request["requeueIfPreempted"] is False
    assert request["image"] == (
        "lmsysorg/sglang@sha256:febfb971c7352570fc445c466ebd6ffc9d896024958e544a60f2137fd85856b1"
    )
    assert "webexploitbench" not in request["command"].lower()


def test_legacy_miles_request_shape_and_title_are_unchanged(plan: dict) -> None:
    request = serving.job_request(plan)
    assert set(serving._runtime_sources(plan)) == {
        "training/miles_serving_dev.py",
        "training/io.py",
        "training/__init__.py",
        "cyber_post_train/jobs.py",
        "cyber_post_train/__init__.py",
    }
    assert set(request) == {
        "name",
        "title",
        "run_dir",
        "image",
        "workers",
        "gpus_per_worker",
        "resources",
        "priority_class",
        "requeueIfPreempted",
        "env",
        "command",
    }
    assert request["title"] == plan["run_name"] + " bounded Miles SGLang dev qualification"
    assert request["env"] == {
        "CYBER_RUNTIME_BUNDLE": request["env"]["CYBER_RUNTIME_BUNDLE"],
        "HF_HUB_OFFLINE": "1",
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
    }
    assert {key: request[key] for key in request if key not in {"command", "env"}} == {
        "name": plan["run_name"],
        "title": plan["run_name"] + " bounded Miles SGLang dev qualification",
        "run_dir": plan["output_root"],
        "image": plan["serving"]["image"],
        "workers": 1,
        "gpus_per_worker": 1,
        "resources": plan["serving"]["resources"],
        "priority_class": "c1",
        "requeueIfPreempted": False,
    }
    assert request["command"].startswith("python -c ")


def test_compiles_teacher_sft_with_the_same_one_gpu_execution_contract(sft_plan: dict) -> None:
    request = serving.job_request(sft_plan)
    assert sft_plan["schema"] == serving.SFT_PLAN_SCHEMA
    assert sft_plan["artifact"]["optimizer_step"] == 186
    assert sft_plan["artifact"]["gpu_check_receipt_sha256"].startswith("sha256:")
    assert serving_registration.execution_contract(
        sft_plan["serving"]["registration"]
    ) == serving_registration.execution_contract(json.loads(BASE.read_text()))
    assert request["workers"] == request["gpus_per_worker"] == 1
    assert request["resources"] == {
        "cpu_request": "32",
        "cpu_limit": "96",
        "memory_request": "256Gi",
        "memory_limit": "768Gi",
    }
    assert request["priority_class"] == "c1"
    assert request["requeueIfPreempted"] is False
    assert "webexploitbench" not in json.dumps(request).lower()


def test_generic_cli_reopens_teacher_sft_preparation(sft_plan: dict, tmp_path: Path) -> None:
    request = serving.job_request(sft_plan)
    prepared = tmp_path / "sft-prepared"
    cli._prepare(prepared, sft_plan, request)
    assert cli._prepared(prepared) == (sft_plan, request)
    proof = _sft_preflight(sft_plan)
    serving.validate_preflight_receipt(sft_plan, request, proof)


@pytest.mark.parametrize(
    ("receipt", "field", "value"),
    [
        ("export", "model_revision", "f" * 40),
        ("export", "output_root", "/mnt/sfs/jobs/wrong-export-root"),
        ("gpu", "serving_qualified", True),
        ("gpu", "generated_tokens", 1),
        ("gpu", "patched_linear_layers", 47),
        ("gpu", "checker_sha256", "f" * 64),
    ],
)
def test_teacher_sft_rejects_mutated_export_or_gpu_acceptance(
    tmp_path: Path, receipt: str, field: str, value: object
) -> None:
    kwargs = {f"{receipt}_change": (field, value)}
    with pytest.raises(ValueError):
        serving.compile_plan(_sft_config(tmp_path, **kwargs))


def test_teacher_sft_rejects_a_self_consistent_but_unapproved_checker(tmp_path: Path) -> None:
    bogus = "f" * 64
    config = _sft_config(tmp_path, gpu_change=("checker_sha256", bogus))
    with pytest.raises(ValueError):
        serving.compile_plan(config)


def test_teacher_sft_submit_boundary_rehashes_payload(
    sft_plan: dict,
) -> None:
    request = serving.job_request(sft_plan)
    proof = _sft_preflight(sft_plan)
    serving.validate_preflight_receipt(sft_plan, request, proof)
    payload = Path(sft_plan["serving"]["model_root"]) / "config.json"
    payload.write_text('{"changed":true}\n')
    with pytest.raises(ValueError, match="digest|payload|export"):
        serving.validate_preflight_receipt(sft_plan, request, proof)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (
            "dev_sfs",
            {
                "namespace": "fleet-train-jobs",
                "pvc_name": "wrong",
                "pvc_uid": "dddddddd-dddd-4ddd-8ddd-dddddddddddd",
                "mount_root": "/mnt/sfs",
            },
        ),
        ("payload_rehashed", False),
        ("output_root_absent", False),
        ("output_root_is_symlink", True),
        ("payload_root", "/mnt/sfs/jobs/not-the-export"),
        ("payload_root_is_symlink", True),
    ],
)
def test_teacher_sft_rejects_mutated_zero_gpu_preflight(
    sft_plan: dict, field: str, value: object
) -> None:
    proof = _sft_preflight(sft_plan)
    proof.pop("sha256")
    proof[field] = value
    proof = serving._sign(proof)
    with pytest.raises(ValueError):
        serving.validate_preflight_receipt(sft_plan, serving.job_request(sft_plan), proof)


@pytest.mark.parametrize("kind", ["directory", "symlink"])
def test_teacher_sft_submit_boundary_rejects_claimed_target(tmp_path: Path, kind: str) -> None:
    target = tmp_path / "claimed"
    if kind == "directory":
        target.mkdir()
    else:
        destination = tmp_path / "destination"
        destination.mkdir()
        target.symlink_to(destination)
    with pytest.raises(ValueError, match="already claimed"):
        serving._require_output_absent({"output_root": str(target)})


def test_plan_and_runtime_payload_changes_fail_closed(plan: dict, tmp_path: Path) -> None:
    changed = copy.deepcopy(plan)
    changed["serving"]["args"][1] = "/mnt/sfs/jobs/another-export"
    changed["sha256"] = digest({key: value for key, value in changed.items() if key != "sha256"})
    with pytest.raises(ValueError, match="serving"):
        serving.validate_plan(changed, check_files=False)

    root = Path(plan["serving"]["model_root"])
    for name in plan["artifact"]["files"]:
        (root / name).write_bytes(b"{}")
    (root / "EXPORT.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="runtime Miles export"):
        serving._verify_runtime_payload(plan)


def test_runtime_refuses_a_reused_output_directory(
    plan: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "run"
    (root / ".runtime").mkdir(parents=True)
    (root / "stale.json").write_text("{}\n")
    changed = copy.deepcopy(plan)
    changed["output_root"] = str(root)
    changed["sha256"] = SHA
    plan_path = tmp_path / "plan.json"
    _write(plan_path, changed)
    monkeypatch.setenv("RUN_DIR", str(root))
    monkeypatch.setattr(serving, "validate_plan", lambda value, check_files: value)
    with pytest.raises(ValueError, match="not a fresh Jobs API-owned directory"):
        serving.run(plan_path, SHA)


def test_synthetic_probes_use_the_proven_non_reasoning_tool_smoke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = "q38-miles-serve-dev1"
    chat = {"model": model, "choices": [{"message": {}}], "usage": {"completion_tokens": 1}}
    responses = iter(
        [
            {"data": [{"id": model}]},
            chat,
            chat,
            {
                "model": model,
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "identity",
                                        "arguments": '{"value":"parity"}',
                                    }
                                }
                            ]
                        }
                    }
                ],
                "usage": {"completion_tokens": 1},
            },
        ]
    )
    requests: list[tuple[str, dict | None]] = []

    def fake_request(path: str, body: dict | None = None) -> dict:
        requests.append((path, body))
        return next(responses)

    monkeypatch.setattr(serving, "_request", fake_request)
    probes = serving._probes(model)
    bodies = [body for _, body in requests if body is not None]
    assert probes["tool_name"] == "identity"
    assert all(body["chat_template_kwargs"] == {"enable_thinking": False} for body in bodies)
    assert bodies[-1]["max_tokens"] == 128
    assert "webexploitbench" not in json.dumps(requests).lower()


def test_generic_cli_reopens_exact_prepared_canary(plan: dict, tmp_path: Path) -> None:
    prepared = tmp_path / "prepared"
    request = serving.job_request(plan)
    cli._prepare(prepared, plan, request)
    assert cli._prepared(prepared) == (plan, request)
    changed = json.loads((prepared / "request.json").read_text())
    changed["priority_class"] = "c2"
    (prepared / "request.json").write_text(canonical_json(changed) + "\n")
    with pytest.raises(ValueError, match="prepared inputs changed"):
        cli._prepared(prepared)


def _result(plan: dict) -> dict:
    value = {
        "schema": (
            serving.SFT_RESULT_SCHEMA
            if plan["schema"] == serving.SFT_PLAN_SCHEMA
            else serving.RESULT_SCHEMA
        ),
        "status": "passed",
        "plan_sha256": "sha256:" + plan["sha256"],
        "request_sha256": digest_json(serving.job_request(plan)),
        "api_run_name": plan["run_name"] + "-1234abcd",
        "api_run_id": "11111111-1111-4111-8111-111111111111",
        "execution_contract_sha256": plan["serving"]["execution_contract_sha256"],
        "export_receipt_sha256": plan["artifact"]["export_receipt_sha256"],
        "requested_runtime_image": plan["serving"]["image"],
        "checks": {key: True for key in serving.DEV_CHECKS},
        "probes": {
            "model_list_exact": True,
            "forward_completion_tokens": 1,
            "continuation_completion_tokens": 1,
            "tool_name": "identity",
            "tool_argument_keys": ["value"],
            "response_content_recorded": False,
            "task_content_included": False,
            "benchmark_content_included": False,
            "scores_observed": False,
        },
        "server_pid": 100,
        "server_exit_code": -15,
        "server_process_group_stopped": True,
        "source_export_unchanged": True,
        "optimizer_updates": 0,
        "rollouts": 0,
        "verifier_calls": 0,
        "benchmark_attempts": 0,
        "completed_at": "2026-09-13T09:00:00+00:00",
    }
    if plan["schema"] == serving.SFT_PLAN_SCHEMA:
        value.update(
            model_revision=plan["artifact"]["model_revision"],
            export_file_sha256=plan["artifact"]["export_file_sha256"],
            gpu_check_receipt_sha256=plan["artifact"]["gpu_check_receipt_sha256"],
            gpu_check_file_sha256=plan["artifact"]["gpu_check_file_sha256"],
            gpu_checker_sha256=plan["artifact"]["gpu_checker_sha256"],
            source_checkpoint_receipt_sha256=plan["artifact"]["source_checkpoint_receipt_sha256"],
            source_manifest_file_sha256=plan["artifact"]["source_manifest_file_sha256"],
            source_plan_sha256=plan["artifact"]["source_plan_sha256"],
            export_code_sha256=plan["artifact"]["export_code_sha256"],
            optimizer_step=plan["artifact"]["optimizer_step"],
            export_manifest_sha256=plan["artifact"]["export_manifest_sha256"],
        )
    else:
        value.update(
            reload_acceptance_receipt_sha256=plan["artifact"]["reload_acceptance_receipt_sha256"],
            source_update_identity_sha256=plan["artifact"]["source_update_identity_sha256"],
            staged_manifest_sha256=plan["artifact"]["staged_manifest_sha256"],
        )
    return serving._sign(value)


def _external(plan: dict, result: dict) -> dict:
    observed = "2026-09-13T09:01:00+00:00"
    return {
        "schema": (
            serving.SFT_EXTERNAL_SCHEMA
            if plan["schema"] == serving.SFT_PLAN_SCHEMA
            else serving.EXTERNAL_SCHEMA
        ),
        "status": "succeeded_released",
        "cluster": "dev",
        "api_base_url": "https://api.ft.dev.flt.build",
        "kube_context": serving.DEV_CONTEXT,
        "namespace": serving.NAMESPACE,
        "namespace_uid": "22222222-2222-4222-8222-222222222222",
        "run_name": result["api_run_name"],
        "api_run_id": result["api_run_id"],
        "request_sha256": result["request_sha256"],
        "effective_priority": 10000,
        "automatic_requeue": False,
        "controller": {
            "kind": "RayJob",
            "name": result["api_run_name"],
            "uid": "33333333-3333-4333-8333-333333333333",
            "status": "SUCCEEDED",
            "workload_uid": "44444444-4444-4444-8444-444444444444",
            "raycluster_uid": "55555555-5555-4555-8555-555555555555",
        },
        "pod": {
            "name": result["api_run_name"] + "-worker",
            "uid": "66666666-6666-4666-8666-666666666666",
            "owner_raycluster_uid": "55555555-5555-4555-8555-555555555555",
            "phase": "Succeeded",
            "exit_code": 0,
            "termination_reason": "Completed",
            "terminated_at": "2026-09-13T09:00:30+00:00",
            "container_restarts": 0,
            "gpus": 1,
            "runtime_image_id": "containerd://docker.io/" + plan["serving"]["image"],
        },
        "release": {
            "api_status": "SUCCEEDED",
            "raycluster_present": False,
            "workload_present": False,
            "gpu_pods_present": False,
            "active_gpus": 0,
            "observed_at": observed,
        },
        "observed_at": observed,
    }


def test_accepts_only_measured_success_and_production_validator_reopens(
    plan: dict, tmp_path: Path
) -> None:
    result = _result(plan)
    external = _external(plan, result)
    result_path = tmp_path / "result.json"
    external_path = tmp_path / "external.json"
    result_sha = _write(result_path, result)
    external_sha = _write(external_path, external)
    output = tmp_path / "DEV_SERVING_ACCEPTED.json"
    accepted = serving.accept(plan, result_path, result_sha, external_path, external_sha, output)
    assert accepted["gpu_release_verified"] is True
    assert accepted["serving_ready"] is False
    assert accepted["production_registration_executed"] is False
    production_plan = {
        "schema": serving_registration.MILES_SCHEMA,
        "execution_contract_sha256": plan["serving"]["execution_contract_sha256"],
        "export_receipt_sha256": plan["artifact"]["export_receipt_sha256"],
        "source_update_identity_sha256": plan["artifact"]["source_update_identity_sha256"],
        "reload_acceptance_receipt_sha256": plan["artifact"]["reload_acceptance_receipt_sha256"],
        "staged_manifest_sha256": plan["artifact"]["staged_manifest_sha256"],
        "registration": plan["serving"]["registration"],
    }
    assert (
        serving_registration._qualification(production_plan, output, file_sha256(output))
        == accepted
    )


def test_accepts_teacher_sft_only_after_measured_release_and_reopens_for_registration(
    sft_plan: dict, tmp_path: Path
) -> None:
    result = _result(sft_plan)
    external = _external(sft_plan, result)
    result_path = tmp_path / "sft-result.json"
    external_path = tmp_path / "sft-external.json"
    output = tmp_path / "SFT_DEV_SERVING_ACCEPTED.json"
    accepted = serving.accept(
        sft_plan,
        result_path,
        _write(result_path, result),
        external_path,
        _write(external_path, external),
        output,
    )
    assert accepted["optimizer_step"] == 186
    assert accepted["gpu_release_verified"] is True
    assert accepted["optimizer_updates"] == accepted["benchmark_attempts"] == 0
    artifact = sft_plan["artifact"]
    production_plan = {
        "schema": serving_registration.SFT_SCHEMA,
        "execution_contract_sha256": sft_plan["serving"]["execution_contract_sha256"],
        "model_revision": artifact["model_revision"],
        "export_file_sha256": artifact["export_file_sha256"],
        "export_receipt_sha256": artifact["export_receipt_sha256"],
        "source_checkpoint_receipt_sha256": artifact["source_checkpoint_receipt_sha256"],
        "source_manifest_file_sha256": artifact["source_manifest_file_sha256"],
        "source_plan_sha256": artifact["source_plan_sha256"],
        "export_code_sha256": artifact["export_code_sha256"],
        "gpu_check_file_sha256": artifact["gpu_check_file_sha256"],
        "gpu_check_receipt_sha256": artifact["gpu_check_receipt_sha256"],
        "gpu_checker_sha256": artifact["gpu_checker_sha256"],
        "optimizer_step": artifact["optimizer_step"],
        "staged_manifest_sha256": artifact["export_manifest_sha256"],
        "registration": sft_plan["serving"]["registration"],
    }
    assert (
        serving_registration._qualification(production_plan, output, file_sha256(output))
        == accepted
    )


@pytest.mark.parametrize(
    ("defect", "field"),
    [
        ("wrong", "gpu_checker_sha256"),
        ("missing", "source_manifest_file_sha256"),
        ("missing", "gpu_release_verified"),
        ("missing", "evidence_manifest"),
    ],
)
def test_teacher_sft_registration_rejects_mutated_qualification(
    sft_plan: dict, tmp_path: Path, defect: str, field: str
) -> None:
    result = _result(sft_plan)
    external = _external(sft_plan, result)
    result_path = tmp_path / "sft-result.json"
    external_path = tmp_path / "sft-external.json"
    accepted_path = tmp_path / "SFT_DEV_SERVING_ACCEPTED.json"
    accepted = serving.accept(
        sft_plan,
        result_path,
        _write(result_path, result),
        external_path,
        _write(external_path, external),
        accepted_path,
    )
    artifact = sft_plan["artifact"]
    production_plan = {
        "schema": serving_registration.SFT_SCHEMA,
        "execution_contract_sha256": sft_plan["serving"]["execution_contract_sha256"],
        "model_revision": artifact["model_revision"],
        "export_file_sha256": artifact["export_file_sha256"],
        "export_receipt_sha256": artifact["export_receipt_sha256"],
        "source_checkpoint_receipt_sha256": artifact["source_checkpoint_receipt_sha256"],
        "source_manifest_file_sha256": artifact["source_manifest_file_sha256"],
        "source_plan_sha256": artifact["source_plan_sha256"],
        "export_code_sha256": artifact["export_code_sha256"],
        "gpu_check_file_sha256": artifact["gpu_check_file_sha256"],
        "gpu_check_receipt_sha256": artifact["gpu_check_receipt_sha256"],
        "gpu_checker_sha256": artifact["gpu_checker_sha256"],
        "optimizer_step": artifact["optimizer_step"],
        "staged_manifest_sha256": artifact["export_manifest_sha256"],
        "registration": sft_plan["serving"]["registration"],
    }
    accepted.pop("receipt_sha256")
    if defect == "missing":
        accepted.pop(field)
    else:
        accepted[field] = "sha256:" + "f" * 64
    mutated = serving_registration._signed(accepted)
    with pytest.raises(ValueError):
        serving_registration._qualification(
            production_plan, accepted_path, _write(accepted_path, mutated)
        )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("pod", "container_restarts"), 1),
        (("release", "active_gpus"), 1),
        (("automatic_requeue",), True),
    ],
)
def test_accept_rejects_unreleased_or_drifted_runs(
    plan: dict, tmp_path: Path, path: tuple[str, ...], value: object
) -> None:
    result = _result(plan)
    external = _external(plan, result)
    target = external
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    result_path = tmp_path / "result.json"
    external_path = tmp_path / "external.json"
    result_sha = _write(result_path, result)
    external_sha = _write(external_path, external)
    with pytest.raises(ValueError, match="incomplete"):
        serving.accept(
            plan,
            result_path,
            result_sha,
            external_path,
            external_sha,
            tmp_path / "accepted.json",
        )


def test_accept_rejects_uncontrolled_server_exit(plan: dict, tmp_path: Path) -> None:
    result = _result(plan)
    result.pop("sha256")
    result["server_exit_code"] = 1
    result = serving._sign(result)
    external = _external(plan, result)
    result_path = tmp_path / "result.json"
    external_path = tmp_path / "external.json"
    with pytest.raises(ValueError, match="incomplete"):
        serving.accept(
            plan,
            result_path,
            _write(result_path, result),
            external_path,
            _write(external_path, external),
            tmp_path / "accepted.json",
        )


def test_templates_are_unlaunchable_and_bind_the_reviewed_base(tmp_path: Path) -> None:
    dev = json.loads(
        (ROOT / "configs/qualification/qwen38-miles-serving-dev-v1.template.json").read_text()
    )
    external = json.loads(
        (
            ROOT / "configs/qualification/qwen38-miles-serving-dev-external-v1.template.json"
        ).read_text()
    )
    staging = json.loads(
        (
            ROOT / "configs/qualification/qwen38-miles-inference-staging-evidence-v1.template.json"
        ).read_text()
    )
    production = json.loads(
        (ROOT / "configs/qualification/qwen38-miles-serving-prod-v1.template.json").read_text()
    )
    teacher = json.loads(
        (
            ROOT / "configs/qualification/"
            "qwen38-teacher-sft-v5-step186-serving-dev-v1.template.json"
        ).read_text()
    )
    teacher_prod = json.loads(
        (
            ROOT / "configs/qualification/"
            "qwen38-teacher-sft-v5-step186-serving-prod-v2.template.json"
        ).read_text()
    )
    teacher_prod_v1 = json.loads(
        (
            ROOT / "configs/qualification/"
            "qwen38-teacher-sft-v5-step186-serving-prod-v1.template.json"
        ).read_text()
    )
    assert dev["base_registration"]["file_sha256"] == file_sha256(BASE)
    assert production["base_registration"]["sha256"] == file_sha256(BASE)
    assert dev["name"] is dev["output_root"] is None
    assert "sha256" not in external and external["pod"]["runtime_image_id"] is None
    assert staging["receipt_sha256"] is staging["observer_pod_uid"] is None
    assert production["model_id"] is production["staging"]["sha256"] is None
    assert teacher["schema"] == serving.SFT_CONFIG_SCHEMA
    assert teacher["base_registration"]["file_sha256"] == file_sha256(BASE)
    assert teacher["name"] is teacher["output_root"] is None
    assert teacher["export"]["file_sha256"] == (
        "sha256:96b3181e05c501343c97cfe803aada6884e406c7a112e0dc6c77461f090b8bab"
    )
    assert teacher["gpu_check"]["file_sha256"] == (
        "sha256:7fbfcfe16875722012cc2654d979545e22aa1fdca093987495dfff65173dfd8e"
    )
    assert teacher["dev_sfs"]["pvc_name"] is teacher["dev_sfs"]["pvc_uid"] is None
    assert teacher_prod_v1["schema"] == serving_registration.SCHEMA
    assert teacher_prod["schema"] == serving_registration.SFT_SCHEMA
    with pytest.raises((TypeError, ValueError)):
        serving.compile_plan(teacher)
    output = tmp_path / "must-not-exist"
    with pytest.raises((TypeError, ValueError)):
        serving_registration.prepare(teacher_prod, output)
    assert not output.exists()
