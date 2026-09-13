from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from cyber_post_train import cli
from cyber_post_train.jobs import digest
from training import miles_serving_dev as serving
from training import serving_registration
from training.io import canonical_json, digest_json, file_sha256

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "evals/fleet/serving/qwen38-27b-dedicated-v1.registration.json"
SHA = "a" * 64


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
    return serving._sign(
        {
            "schema": serving.RESULT_SCHEMA,
            "status": "passed",
            "plan_sha256": "sha256:" + plan["sha256"],
            "request_sha256": digest_json(serving.job_request(plan)),
            "api_run_name": plan["run_name"] + "-1234abcd",
            "api_run_id": "11111111-1111-4111-8111-111111111111",
            "execution_contract_sha256": plan["serving"]["execution_contract_sha256"],
            "export_receipt_sha256": plan["artifact"]["export_receipt_sha256"],
            "reload_acceptance_receipt_sha256": plan["artifact"][
                "reload_acceptance_receipt_sha256"
            ],
            "source_update_identity_sha256": plan["artifact"]["source_update_identity_sha256"],
            "staged_manifest_sha256": plan["artifact"]["staged_manifest_sha256"],
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
    )


def _external(plan: dict, result: dict) -> dict:
    observed = "2026-09-13T09:01:00+00:00"
    return {
        "schema": serving.EXTERNAL_SCHEMA,
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
            "phase": "Succeeded",
            "exit_code": 0,
            "container_restarts": 0,
            "gpus": 1,
            "runtime_image_id": plan["serving"]["image"],
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


def test_templates_are_unlaunchable_and_bind_the_reviewed_base() -> None:
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
    assert dev["base_registration"]["file_sha256"] == file_sha256(BASE)
    assert production["base_registration"]["sha256"] == file_sha256(BASE)
    assert dev["name"] is dev["output_root"] is None
    assert "sha256" not in external and external["pod"]["gpus"] == 1
    assert staging["receipt_sha256"] is staging["observer_pod_uid"] is None
    assert production["model_id"] is production["staging"]["sha256"] is None
