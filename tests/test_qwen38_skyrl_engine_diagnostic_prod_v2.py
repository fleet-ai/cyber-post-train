"""Exact prod2 Blackwell-sampler engine gate; no task or training work."""

import copy
import json
from pathlib import Path

from cyber_post_train.jobs import digest
from training import skyrl_training

ROOT = Path(__file__).resolve().parents[1]
DATA_CONFIG = ROOT / (
    "configs/qualification/qwen38-rl-filtered-skyrl-engine-diagnostic-data-prod-v2.json"
)
RUN_CONFIG = ROOT / (
    "configs/qualification/qwen38-rl-filtered-skyrl-engine-diagnostic-prod-v2.json"
)


def test_prod2_source_config_is_exact_c1_zero_work() -> None:
    data = json.loads(DATA_CONFIG.read_bytes())
    run = json.loads(RUN_CONFIG.read_bytes())

    assert skyrl_training.is_prod_engine_diagnostic_config(run)
    assert digest(run) == skyrl_training.ENGINE_DIAGNOSTIC_PROD2_CONFIG_SHA256
    assert run["name"] == data["name"] == skyrl_training.ENGINE_DIAGNOSTIC_PROD2_CONFIG_NAME
    assert run["output_root"] == skyrl_training.ENGINE_DIAGNOSTIC_PROD2_OUTPUT_ROOT
    assert run["data"]["root"] == data["output"] == skyrl_training.ENGINE_DIAGNOSTIC_PROD2_DATA_ROOT
    assert run["cluster"]["target"] == "prod"
    assert run["cluster"]["priority"] == "c1"
    assert {key: run["recipe"][key] for key in ("nodes", "steps", "groups", "samples_per_prompt")} == {
        "nodes": 1,
        "steps": 1,
        "groups": 2,
        "samples_per_prompt": 4,
    }


def test_prod2_source_config_rejects_any_drift() -> None:
    run = json.loads(RUN_CONFIG.read_bytes())
    changed = copy.deepcopy(run)
    changed["cluster"]["priority"] = "c0"
    assert not skyrl_training.is_prod_engine_diagnostic_config(changed)


def test_sampler_fallback_is_bound_to_jobs_and_ray_environments(monkeypatch) -> None:
    plan = {
        "arguments": {
            "name": "synthetic",
            "output_root": "/mnt/sfs/jobs/synthetic",
            "model_root": "/mnt/sfs/models/synthetic",
            "train_data": "/mnt/sfs/data/synthetic/train.jsonl",
            "dev_data": "/mnt/sfs/data/synthetic/dev.jsonl",
            "data_manifest": "/mnt/sfs/data/synthetic/manifest.json",
            "train_rows": 2,
            "dev_rows": 1,
            "wandb_entity": "thefleet",
            "wandb_project": "cyber-post-train",
            "wandb_run_id": "synthetic",
            "nodes": 1,
            "steps": 1,
            "groups": 2,
            "samples_per_prompt": 4,
            "lr": 1e-6,
            "eval_interval": 1,
            "checkpoint_interval": 1,
            "keep_checkpoints": 2,
            "seed": 42,
            "context_tokens": 98304,
            "response_tokens": 81920,
            "tokens_per_turn": 4096,
            "max_turns": 80,
            "engine_start_timeout_seconds": 1800,
            "engine_cleanup_timeout_seconds": 300,
            "model": "Qwen/Qwen3.8-27B",
        },
        "native_overrides": {
            "generator.inference_engine.num_engines": 2,
            "generator.inference_engine.tensor_parallel_size": 4,
        },
        "execution": {
            "resources": {},
            "priority": "c1",
        },
    }
    monkeypatch.setattr(skyrl_training, "job_request", lambda _: None)
    monkeypatch.setattr(skyrl_training, "_runtime", lambda: {})
    monkeypatch.setattr(skyrl_training, "bundled_request", lambda request, *_: request)

    request = skyrl_training.engine_diagnostic_request(plan)
    assert request["env"]["VLLM_USE_FLASHINFER_SAMPLER"] == "0"

    native = {"skyrl.train.utils.utils": type("Native", (), {
        "prepare_runtime_environment": staticmethod(lambda _: {})
    })}
    monkeypatch.setattr(skyrl_training, "_prepare_infra_log", lambda _: Path("/tmp/infra"))
    env, _, _ = skyrl_training._ray_environment(plan, object(), native, diagnostic=True)
    assert env["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
