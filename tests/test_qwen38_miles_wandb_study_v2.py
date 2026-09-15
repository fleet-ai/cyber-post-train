"""Offline-only contract tests for the inert Miles W&B v2 successor."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from training import miles, miles_study_training, miles_wandb
from training.io import digest_json, file_sha256

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "configs/qualification/qwen38-miles-wandb-study-dev-v9.template.json"
EVIDENCE = (
    ROOT / "docs/evidence/qwen38-study/2026-09-14-miles-wandb-study-v2-offline-preparation.json"
)


def synthetic_plan() -> dict:
    return {
        "schema": "cyber_miles_training_v2",
        "run_name": "synthetic-miles-v2",
        "model": {"repo": "Qwen/Qwen3.8-27B", "revision": "a" * 40},
        "data": {"sha256": "sha256:" + "b" * 64},
        "checkpoint": {"sha256": "sha256:" + "c" * 64},
        "arguments": {
            "nodes": 1,
            "gpus_per_node": 8,
            "steps": 2,
            "groups": 1,
            "samples_per_prompt": 8,
            "lr": 2e-6,
            "temperature": 0.7,
            "kl_loss_coef": 0.001,
            "seed": 42,
        },
    }


def synthetic_wandb() -> dict:
    return {
        "entity": "thefleet",
        "project": "cyber-post-train",
        "group": "q38-fleet-rl-study-a-v1",
        "run_id": "synthetic-miles-v2",
        "name": "synthetic-miles-v2",
        "job_type": "rl-train",
        "tags": ["backend:miles", "model:qwen38-27b", "phase:dev"],
    }


def contract() -> dict:
    return miles_wandb.build_contract(synthetic_wandb(), synthetic_plan(), "sha256:" + "d" * 64)


def test_inert_template_preserves_dev8_and_lists_all_fresh_gates() -> None:
    value = json.loads(TEMPLATE.read_bytes())
    assert value["sha256"] == digest_json(
        {key: item for key, item in value.items() if key != "sha256"}
    )
    assert value["status"] == "blocked_not_launchable"
    assert value["launchable"] is False
    for binding in value["preserved_dev8"].values():
        if isinstance(binding, dict):
            assert file_sha256(ROOT / binding["path"]) == binding["file_sha256"]
    assert all(item is None for item in value["blocked_bindings"].values())
    assert value["future_identity"]["wandb"]["group"] != value["future_identity"]["wandb"]["run_id"]
    assert "no_production_launch_from_this_template" in value["prohibitions"]


def test_offline_evidence_is_self_digesting_and_binds_exact_bytes() -> None:
    value = json.loads(EVIDENCE.read_bytes())
    assert value["sha256"] == digest_json(
        {key: item for key, item in value.items() if key != "sha256"}
    )
    assert value["status"] == "prepared_inert_not_launchable"
    assert value["method"] == {
        "cluster_calls": 0,
        "jobs_api_calls": 0,
        "live_wandb_calls": 0,
        "private_data_reads": 0,
        "secrets_read": 0,
        "submissions": 0,
    }
    for binding in value["bound_files"].values():
        assert file_sha256(ROOT / binding["path"]) == binding["file_sha256"]
    assert value["gates"]["production_authorized"] is False
    assert value["gates"]["fresh_dev_qualification_required"] is True


def test_contract_has_exact_identity_resume_config_allowlist_and_axes() -> None:
    value = contract()
    miles_wandb.validate_contract(value)
    assert value["resume"] == "never"
    assert value["group"] == "q38-fleet-rl-study-a-v1"
    assert value["run_id"] == value["name"] == "synthetic-miles-v2"
    assert value["job_type"] == "rl-train"
    assert value["history_keys"] == list(miles_wandb.HISTORY_KEYS)
    assert value["axes"] == {
        "optimizer": "train/step",
        "token_exposure": "tokens/total_response_tokens",
    }
    assert value["public_config"]["experiment_plan_sha256"] == "sha256:" + "d" * 64
    assert set(value["public_config"]).isdisjoint(
        {"reward", "score", "success", "response", "prompt", "task_text"}
    )
    env = miles_wandb.request_env(value)
    assert env["WANDB_RESUME"] == "never"
    assert env["WANDB_RUN_GROUP"] == value["group"]
    assert env["WANDB__DISABLE_STATS"] == "true"
    assert env["CYBER_EXPERIMENT_PLAN_SHA256"] == "sha256:" + "d" * 64


def test_additive_compiler_binds_full_plan_digest_into_portable_request(
    monkeypatch, tmp_path
) -> None:
    base = {
        **synthetic_plan(),
        "output_root": "/mnt/sfs/jobs/synthetic-miles-v2",
        "runtime_sha256": "legacy",
        "native_driver_sha256": "a" * 64,
        "execution": {
            "image": miles.IMAGE,
            "priority": "c1",
            "cluster_target": "dev",
            "resources": {
                "cpu_request": "64",
                "cpu_limit": "128",
                "memory_request": "1536Gi",
                "memory_limit": "2048Gi",
            },
        },
    }
    legacy_request = {
        "name": "synthetic-miles-v2",
        "run_dir": base["output_root"],
        "image": miles.IMAGE,
        "workers": 1,
        "gpus_per_worker": 8,
        "resources": base["execution"]["resources"],
        "priority_class": "c1",
        "requeueIfPreempted": False,
        "secrets": ["fleet-api", "wandb-api"],
        "env": {
            "WANDB_RUN_ID": "legacy",
            "WANDB_MODE": "online",
            "PYTHONUNBUFFERED": "1",
            "CYBER_RUNTIME_BUNDLE_0": "discarded",
        },
    }
    monkeypatch.setattr(
        miles_study_training.v1,
        "compile_rl",
        lambda _config, relative_to: json.loads(json.dumps(base)),
    )
    monkeypatch.setattr(miles_study_training.v1, "job_request", lambda _plan: legacy_request)
    plan = miles_study_training.compile_rl({"wandb": synthetic_wandb()}, relative_to=tmp_path)
    request = miles_study_training.job_request(plan)
    assert plan["schema"] == miles_study_training.SCHEMA
    assert plan["wandb"] == synthetic_wandb()
    assert request["env"]["WANDB_RUN_ID"] == synthetic_wandb()["run_id"]
    assert request["env"]["WANDB_RUN_GROUP"] == synthetic_wandb()["group"]
    assert request["env"]["WANDB_RESUME"] == "never"
    assert request["env"]["CYBER_EXPERIMENT_PLAN_SHA256"] == "sha256:" + (
        miles_study_training.digest(plan)
    )
    assert "WANDB_API_KEY" not in request["env"]


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda value: value.update(group=value["run_id"]), "stable"),
        (lambda value: value.update(tags=["phase:dev", "backend:miles"]), "sorted"),
        (lambda value: value.update(job_type="has spaces"), "job_type"),
    ],
)
def test_contract_rejects_ambiguous_or_unstable_identity(change, match) -> None:
    value = synthetic_wandb()
    change(value)
    with pytest.raises(ValueError, match=match):
        miles_wandb.build_contract(value, synthetic_plan(), "sha256:" + "d" * 64)


def test_adapter_writes_exact_metadata_and_filters_native_payload(monkeypatch, tmp_path) -> None:
    value = contract()
    calls, definitions, logs = [], [], []

    class FakeRun:
        id = value["run_id"]
        name = value["name"]
        group = value["group"]
        job_type = value["job_type"]
        tags = value["tags"]
        entity = value["entity"]
        project = value["project"]

    fake = SimpleNamespace(
        Settings=lambda **kwargs: kwargs,
        init=lambda **kwargs: calls.append(kwargs) or FakeRun(),
        define_metric=lambda *args, **kwargs: definitions.append((args, kwargs)),
        log=lambda payload: logs.append(payload),
        finish=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.setattr(miles_wandb, "_contract", value)
    for key, item in miles_wandb.request_env(value).items():
        monkeypatch.setenv(key, item)
    backend = SimpleNamespace()
    args = SimpleNamespace(wandb_dir=str(tmp_path / "wandb"), global_batch_size=8)
    miles_wandb._init(backend, args, primary=True)
    assert calls[0]["id"] == value["run_id"]
    assert calls[0]["group"] == value["group"]
    assert calls[0]["name"] == value["name"]
    assert calls[0]["job_type"] == value["job_type"]
    assert calls[0]["tags"] == value["tags"]
    assert calls[0]["resume"] == "never"
    assert calls[0]["settings"]["x_disable_stats"] is True
    assert calls[0]["config"]["experiment_plan_sha256"] == "sha256:" + "d" * 64
    assert (("train/step",), {}) in definitions
    assert (("tokens/total_response_tokens",), {}) in definitions

    miles_wandb._log(
        backend,
        {
            "train/step": 0,
            "train/loss": 1.5,
            "train/grad_norm": 2.0,
            "train/lr-pg_0": 2e-6,
            "reward": 1,
            "eval/passrate": 1,
            "response": "private",
            "rollout/step": 0,
            "rollout/episode_response_length/mean": 5,
        },
    )
    assert logs == [
        {
            "train/step": 0,
            "train/loss": 1.5,
            "train/grad_norm": 2.0,
            "train/lr-pg_0": 2e-6,
            "tokens/rollout_step": 0,
            "tokens/response_tokens": 40,
            "tokens/total_response_tokens": 40,
        }
    ]
    receipt = json.loads((tmp_path / "WANDB_CONTRACT.json").read_bytes())
    assert receipt["sha256"] == digest_json(
        {key: item for key, item in receipt.items() if key != "sha256"}
    )
    assert receipt["contract_sha256"] == value["sha256"]
    assert receipt["experiment_plan_sha256"] == "sha256:" + "d" * 64


def test_adapter_rejects_nonfinite_public_scalars(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(log=lambda _payload: None))
    with pytest.raises(ValueError, match="finite scalars"):
        miles_wandb._log(SimpleNamespace(), {"train/loss": float("nan")})


def test_v2_is_additive_and_dev8_runtime_sources_are_unchanged() -> None:
    audit = json.loads(
        (
            ROOT / "docs/evidence/qwen38-study/2026-09-14-wandb-training-contract-audit-v1.json"
        ).read_bytes()
    )
    for key in ("miles_arguments", "miles_compiler", "miles_acceptance"):
        binding = audit["bound_sources"][key]
        assert file_sha256(ROOT / binding["path"]) == binding["file_sha256"]
    source = (ROOT / "training/miles_study_training.py").read_text()
    assert '"WANDB_RESUME": "never"' in (ROOT / "training/miles_wandb.py").read_text()
    assert '"worker_process_setup_hook": "training.miles_wandb.install_worker"' in source
    assert "training/miles_wandb.py" not in (ROOT / "training/miles_training.py").read_text()
