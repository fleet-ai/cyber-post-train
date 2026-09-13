"""Fail-closed boundaries for the exact Miles production promotion."""

import copy
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from cyber_post_train.jobs import JobsError, digest
from training import miles, miles_promotion
from training.models import bound_model

ROOT = Path(__file__).resolve().parents[1]


def _model() -> dict:
    return bound_model(
        json.loads((ROOT / "configs/models/qwen38-27b-1d4bf0f2.lock.json").read_text()),
        json.loads((ROOT / "configs/models/qwen38-27b-1d4bf0f2.weights.json").read_text()),
        "/mnt/sfs/models/qwen3.8-27b-1d4bf0f2",
    )


def _data() -> dict:
    value = {
        "schema": "cyber_miles_data_v1",
        "name": miles_promotion.PROD_NAME,
        "tokenizer": copy.deepcopy(miles_promotion.EXPECTED_DATA["tokenizer"]),
        "template_sha256": "sha256:" + miles.TEMPLATE_SHA256,
        "selection_sha256": miles_promotion.EXPECTED_DATA["selection_sha256"],
        "split_sha256": miles_promotion.EXPECTED_DATA["split_sha256"],
        "tool_catalog_sha256": miles_promotion.EXPECTED_DATA["tool_catalog_sha256"],
        "limits": miles_promotion.EXPECTED_DATA["limits"],
        "files": {
            split: {"path": split + ".jsonl", "rows": rows, "sha256": "sha256:" + "1" * 64}
            for split, rows in miles_promotion.EXPECTED_DATA["rows"].items()
        },
        "gpus": 0,
        "environment_creates": 0,
    }
    value["sha256"] = "sha256:" + digest(value)
    return value


def _promotion() -> dict:
    reference = {
        "path": "/mnt/sfs/jobs/synthetic/evidence.json",
        "file_sha256": "2" * 64,
        "receipt_sha256": "3" * 64,
    }
    value = {
        "schema": miles_promotion.SCHEMA,
        "status": "qualified_for_fresh_live_checks",
        "candidate_run_sha256": miles_promotion.EXPECTED_CANDIDATE_SHA256,
        "active_dev3": miles_promotion.DEV3,
        "reward_terminal": reference,
        "native_reload": reference,
        "production_data_manifest": reference,
        "benchmark_isolation": miles_promotion.BENCHMARK_ISOLATION,
        "live_requirements": miles_promotion.LIVE_REQUIREMENTS,
    }
    value["sha256"] = digest(value)
    return value


def _plan() -> dict:
    model, data, promotion = _model(), _data(), _promotion()
    checkpoint = {
        "schema": "cyber_miles_checkpoint_v1",
        "model": model,
        "image": miles.IMAGE,
        "root": miles_promotion.BASE_CHECKPOINT["root"],
        "optimizer_steps": 0,
        "files": [],
        "sha256": miles_promotion.BASE_CHECKPOINT["receipt_sha256"],
    }
    args = miles.MilesConfig(
        name=miles_promotion.PROD_NAME,
        output_root=miles_promotion.PROD_OUTPUT,
        model_root=model["root"],
        torch_dist_root=checkpoint["root"],
        train_data=miles_promotion.PROD_DATA_ROOT + "/train.jsonl",
        dev_data=miles_promotion.PROD_DATA_ROOT + "/dev.jsonl",
        data_manifest=miles_promotion.PROD_DATA_MANIFEST,
        wandb_entity=miles_promotion.PROD_WANDB["entity"],
        wandb_project=miles_promotion.PROD_WANDB["project"],
        wandb_run_id=miles_promotion.PROD_WANDB["run_id"],
        nodes=1,
        gpus_per_node=8,
        steps=59,
        groups=1,
        samples_per_prompt=8,
        lr=1e-6,
        eval_interval=59,
        checkpoint_interval=10,
        seed=42,
    )
    proof = {
        "path": "/mnt/sfs/jobs/synthetic/PROMOTION.json",
        "file_sha256": "4" * 64,
        "receipt_sha256": promotion["sha256"],
        "receipt": promotion,
    }
    return {
        "run_name": args.name,
        "output_root": args.output_root,
        "model": model,
        "data": data,
        "checkpoint": checkpoint,
        "arguments": asdict(args),
        "execution": {
            "image": miles.IMAGE,
            "priority": "c1",
            "resources": miles_promotion.EXPECTED_RESOURCES,
            "cluster_target": "prod",
            "production_promotion": proof,
        },
    }


def test_every_prod_miles_config_requires_the_exact_candidate_and_receipt() -> None:
    generic = {"backend": "miles", "name": "other", "cluster": {"target": "prod"}}
    assert miles_promotion.requires_production_promotion(generic) is True
    with pytest.raises(ValueError, match="exact reviewed run"):
        miles_promotion.bind_production_promotion(generic, ROOT)


def test_embedded_promotion_rechecks_the_complete_compiled_candidate() -> None:
    plan = _plan()
    assert miles_promotion.validate_embedded_promotion(plan, check_files=False) is True
    for section, key, changed in (
        ("arguments", "steps", 1),
        ("arguments", "samples_per_prompt", 4),
        ("arguments", "lr", 2e-6),
        ("arguments", "data_manifest", "/mnt/sfs/jobs/other/manifest.json"),
        ("execution", "priority", "c2"),
        ("execution", "resources", {**miles_promotion.EXPECTED_RESOURCES, "cpu_request": "63"}),
        ("checkpoint", "root", "/mnt/sfs/jobs/other/torch-dist"),
    ):
        invalid = copy.deepcopy(plan)
        invalid[section][key] = changed
        with pytest.raises(ValueError, match="exact candidate"):
            miles_promotion.validate_embedded_promotion(invalid, check_files=False)


def test_promotion_cannot_hide_external_benchmark_feedback() -> None:
    value = _promotion()
    miles_promotion.validate_promotion(value, check_files=False)
    changed = copy.deepcopy(value)
    changed["benchmark_isolation"]["external_benchmark_hpo_or_checkpoint_inputs"] = 1
    changed["sha256"] = digest({key: item for key, item in changed.items() if key != "sha256"})
    with pytest.raises(ValueError, match="invariant"):
        miles_promotion.validate_promotion(changed, check_files=False)


def test_promotion_cross_binds_native_reload_to_the_dev_checkpoint(monkeypatch) -> None:
    value = _promotion()
    for index, name in enumerate(("reward_terminal", "native_reload", "production_data_manifest")):
        value[name] = {
            "path": (
                miles_promotion.PROD_DATA_MANIFEST
                if name == "production_data_manifest"
                else f"/mnt/sfs/jobs/synthetic/{name}.json"
            ),
            "file_sha256": str(index + 4) * 64,
            "receipt_sha256": str(index + 7) * 64,
        }
    value["sha256"] = digest({key: item for key, item in value.items() if key != "sha256"})
    terminal = {
        "sha256": "sha256:" + "a" * 64,
        "checkpoint_manifest": {"receipt_sha256": "sha256:" + "b" * 64},
    }
    native = {
        "source_terminal_acceptance_sha256": "sha256:" + "a" * 64,
        "source_manifest_sha256": "sha256:" + "b" * 64,
    }
    data = {"schema": "synthetic"}
    observed = {
        value["reward_terminal"]["path"]: terminal,
        value["native_reload"]["path"]: native,
        value["production_data_manifest"]["path"]: data,
    }
    monkeypatch.setattr(
        miles_promotion,
        "_reopen",
        lambda reference, *, check_files: observed[reference["path"]],
    )
    monkeypatch.setattr(miles_promotion, "_exact_dev3", lambda _: None)
    monkeypatch.setattr(miles_promotion, "_exact_data", lambda _: None)
    monkeypatch.setattr("training.miles_acceptance.validate_terminal", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        "training.miles_reload_acceptance.validate_accepted", lambda *_args, **_kwargs: {}
    )

    result = miles_promotion.validate_promotion(value, check_files=True)
    assert result["dev_checkpoint"] == terminal["checkpoint_manifest"]

    native["source_manifest_sha256"] = "sha256:" + "c" * 64
    with pytest.raises(ValueError, match="cross-bound"):
        miles_promotion.validate_promotion(value, check_files=True)


def test_live_gate_reopens_c1_value_and_counts_node_budget(monkeypatch) -> None:
    class Client:
        @staticmethod
        def all_runs():
            return []

    class Wandb:
        @staticmethod
        def run(_):
            raise RuntimeError("Could not find run")

    priority = 10000

    def kubectl(*args):
        if args[:2] == ("get", "namespace"):
            return {"metadata": {"uid": miles_promotion.PROD_NAMESPACE_UID}}
        if args[:2] == ("get", "priorityclass"):
            return {
                "metadata": {"name": "c1"},
                "value": priority,
                "preemptionPolicy": "PreemptLowerPriority",
            }
        return {
            "items": [
                {
                    "status": {"phase": "Pending"},
                    "spec": {"containers": [{"resources": {"limits": {"nvidia.com/gpu": "8"}}}]},
                }
                for _ in range(7)
            ]
        }

    monkeypatch.setattr(miles_promotion, "validate_embedded_promotion", lambda *_a, **_k: True)
    monkeypatch.setattr(miles_promotion, "_kubectl_json", kubectl)
    observed = miles_promotion.require_live_external({}, Client(), wandb_api=Wandb())
    assert observed == {
        "jobs_api_absent": True,
        "wandb_absent": True,
        "active_experiment_nodes": 7,
        "candidate_nodes": 1,
        "node_limit": 8,
        "effective_priority": 10000,
    }
    priority = 9999
    with pytest.raises(JobsError, match="effective priority"):
        miles_promotion.require_live_external({}, Client(), wandb_api=Wandb())
