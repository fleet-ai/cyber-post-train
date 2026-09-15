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


def _active_canary_binding() -> dict:
    value = {
        "schema": miles_promotion.ACTIVE_CANARY_SCHEMA,
        "status": "accepted_dev_canary",
        "reward_terminal_receipt_sha256": "sha256:" + "a" * 64,
        "source_run_name": miles_promotion.EXPECTED_DEV_CANARY["source_run_name"],
        "source_commit": miles_promotion.EXPECTED_DEV_CANARY["source_commit"],
        "source_plan_sha256": miles_promotion.EXPECTED_DEV_CANARY["source_plan_sha256"],
        "source_request_sha256": miles_promotion.EXPECTED_DEV_CANARY["source_request_sha256"],
        "runtime_bundle_sha256": "sha256:" + "e" * 64,
        "api_base_url": miles_promotion.EXPECTED_DEV_CANARY["api_base_url"],
        "api_run_id": "11111111-1111-4111-8111-111111111111",
        "api_run_name": "chris-q38-miles-rlreward-dev5-11111111",
        "rayjob_uid": "22222222-2222-4222-8222-222222222222",
        "workload_uid": "33333333-3333-4333-8333-333333333333",
    }
    value["sha256"] = digest(value)
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
        "active_canary_binding": reference,
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
        lr=2e-6,
        temperature=0.7,
        kl_loss_coef=0.001,
        max_tokens_per_gpu=8192,
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


def test_reward_canary_config_is_c1_without_an_override_reason() -> None:
    path = ROOT / "configs/qualification/qwen38-miles-rl-reward-canary-prod-v1.json"
    config = json.loads(path.read_text())
    miles_promotion._exact_reward_canary_config(config)
    assert config["cluster"]["priority"] == "c1"
    assert "priority_reason" not in config["cluster"]

    for changed_cluster in (
        {**config["cluster"], "priority": "c0"},
        {**config["cluster"], "priority_reason": "override"},
    ):
        changed = copy.deepcopy(config)
        changed["cluster"] = changed_cluster
        with pytest.raises(ValueError, match="exact one-update arm"):
            miles_promotion._exact_reward_canary_config(changed)


def test_reward_canary_v2_preserves_dev8_science_with_fresh_prod_identities() -> None:
    data_path = ROOT / "configs/qualification/qwen38-miles-rl-reward-canary-data-prod-v2.json"
    run_path = ROOT / "configs/qualification/qwen38-miles-rl-reward-canary-prod-v2.json"
    data, config = json.loads(data_path.read_text()), json.loads(run_path.read_text())

    miles_promotion._exact_reward_canary_config(config)
    assert data["name"] == config["name"] == config["wandb"]["run_id"]
    assert config["name"] == miles_promotion.PROD_REWARD_CANARY_V2_NAME
    assert config["output_root"] == miles_promotion.PROD_REWARD_CANARY_V2_OUTPUT
    assert data["output"] == miles_promotion.PROD_REWARD_CANARY_V2_DATA_ROOT
    assert data["limits"] == miles_promotion.EXPECTED_REWARD_CANARY_V2_DATA["limits"]
    assert data["limits"]["max_tokens_per_turn"] == 8192
    assert data["limits"]["tool_result_chars"] == 4000
    assert config["recipe"] == {
        "nodes": 1,
        "gpus_per_node": 8,
        "steps": 1,
        "groups": 1,
        "samples_per_prompt": 8,
        "lr": 2e-6,
        "temperature": 0.7,
        "kl_loss_coef": 0.001,
        "max_tokens_per_gpu": 8192,
        "eval_interval": 1,
        "checkpoint_interval": 1,
        "seed": 42,
    }
    assert config["cluster"] == {
        "target": "prod",
        "priority": "c1",
        "resources": miles_promotion.PROD_REWARD_CANARY_RESOURCES,
    }
    assert config["production_promotion"] == {"mode": miles_promotion.PROD_REWARD_CANARY_V2_MODE}

    changed = copy.deepcopy(config)
    changed["recipe"]["steps"] = 2
    with pytest.raises(ValueError, match="exact one-update arm"):
        miles_promotion._exact_reward_canary_config(changed)


def test_reward_canary_v3_changes_only_the_per_turn_generation_horizon() -> None:
    data_v2 = json.loads(
        (ROOT / "configs/qualification/qwen38-miles-rl-reward-canary-data-prod-v2.json").read_text()
    )
    run_v2 = json.loads(
        (ROOT / "configs/qualification/qwen38-miles-rl-reward-canary-prod-v2.json").read_text()
    )
    data_v3 = json.loads(
        (ROOT / "configs/qualification/qwen38-miles-rl-reward-canary-data-prod-v3.json").read_text()
    )
    run_v3 = json.loads(
        (ROOT / "configs/qualification/qwen38-miles-rl-reward-canary-prod-v3.json").read_text()
    )

    miles_promotion._exact_reward_canary_config(run_v3)
    assert data_v3["limits"]["max_tokens_per_turn"] == 32768
    assert data_v3["limits"] == miles_promotion.EXPECTED_REWARD_CANARY_V3_DATA["limits"]
    assert run_v3["production_promotion"] == {
        "mode": miles_promotion.PROD_REWARD_CANARY_V3_MODE
    }

    def science(data, run):
        data = copy.deepcopy(data)
        run = copy.deepcopy(run)
        data["name"] = run["name"] = run["wandb"]["run_id"] = "IDENTITY"
        data["output"] = "DATA"
        run["output_root"] = "OUTPUT"
        run["data"] = {"manifest": "MANIFEST", "root": "DATA"}
        run["production_promotion"] = {"mode": "MODE"}
        data["limits"].pop("max_tokens_per_turn")
        return data, run

    assert science(data_v3, run_v3) == science(data_v2, run_v2)


def test_long_context_reward_canary_binds_exact_runtime_checkpoint_and_four_nodes() -> None:
    config = json.loads(
        (
            ROOT
            / "configs/qualification/qwen38-miles-opencode-long-context-prod-v1.json"
        ).read_text()
    )

    miles_promotion._exact_reward_canary_config(config)
    assert config["name"] == config["wandb"]["run_id"]
    assert config["checkpoint"] == miles_promotion.PROD_REWARD_CANARY_LONG_V1_CHECKPOINT
    assert config["runtime"] == miles_promotion.PROD_REWARD_CANARY_LONG_V1_RUNTIME
    assert config["recipe"] == {
        "nodes": 4,
        "gpus_per_node": 8,
        "steps": 1,
        "groups": 1,
        "samples_per_prompt": 8,
        "lr": 2e-6,
        "temperature": 0.7,
        "kl_loss_coef": 0.001,
        "max_tokens_per_gpu": 65536,
        "eval_interval": 1,
        "checkpoint_interval": 1,
        "seed": 42,
        "native_profile": "qwen3.8-27b-256k",
        "harness": "opencode",
        "session_node_cap": 4096,
    }
    assert config["cluster"] == {
        "target": "prod",
        "priority": "c1",
        "resources": miles_promotion.PROD_REWARD_CANARY_LONG_V1_RESOURCES,
    }

    for section, key, changed in (
        ("recipe", "nodes", 1),
        ("recipe", "session_node_cap", 1024),
        ("recipe", "harness", "direct"),
        ("runtime", "image", miles.IMAGE),
        ("checkpoint", "sha256", "sha256:" + "0" * 64),
        ("cluster", "priority", "c0"),
    ):
        invalid = copy.deepcopy(config)
        invalid[section][key] = changed
        with pytest.raises(ValueError, match="exact one-update arm"):
            miles_promotion._exact_reward_canary_config(invalid)


def test_long_context_reward_canary_v2_uses_qualified_runtime_and_new_identity() -> None:
    config = json.loads(
        (
            ROOT
            / "configs/qualification/qwen38-miles-opencode-long-context-prod-v2.json"
        ).read_text()
    )

    miles_promotion._exact_reward_canary_config(config)
    assert config["name"] == miles_promotion.PROD_REWARD_CANARY_LONG_V2_NAME
    assert config["output_root"] == miles_promotion.PROD_REWARD_CANARY_LONG_V2_OUTPUT
    assert config["wandb"] == miles_promotion.PROD_REWARD_CANARY_LONG_V2_WANDB
    assert config["runtime"] == miles_promotion.PROD_REWARD_CANARY_LONG_V2_RUNTIME
    assert config["runtime"]["image"].endswith(
        "@sha256:dc1a41ac386c9f7377e7a6f7b92a402830e308855aa2632413af25917f38dd93"
    )
    assert config["production_promotion"] == {
        "mode": miles_promotion.PROD_REWARD_CANARY_LONG_V2_MODE
    }


def test_long_context_reward_canary_preview_is_exact_four_by_eight_c1(monkeypatch) -> None:
    from cyber_post_train import jobs

    plan = {
        "execution": {
            "production_promotion": {
                "mode": miles_promotion.PROD_REWARD_CANARY_LONG_V1_MODE
            }
        }
    }
    request = {
        "workers": 4,
        "gpus_per_worker": 8,
        "priority_class": "c1",
        "requeueIfPreempted": False,
        "resources": miles_promotion.PROD_REWARD_CANARY_LONG_V1_RESOURCES,
    }
    monkeypatch.setattr(miles_promotion, "validate_embedded_promotion", lambda *_a, **_k: True)
    monkeypatch.setattr(jobs, "validate_preview", lambda *_a, **_k: {"nodes": 4})
    assert miles_promotion.validate_production_preview(plan, request, {}) == {
        "production_reward_canary": "validated",
        "rendered_nodes": 4,
        "effective_priority_expected": 10000,
    }

    invalid = {**request, "workers": 1}
    with pytest.raises(JobsError, match="exact 4x8 c1/no-requeue"):
        miles_promotion.validate_production_preview(plan, invalid, {})


def test_reward_canary_preview_requires_c1_without_an_override_reason(monkeypatch) -> None:
    from cyber_post_train import jobs

    plan = {
        "execution": {"production_promotion": {"mode": miles_promotion.PROD_REWARD_CANARY_MODE}}
    }
    request = {
        "workers": 1,
        "gpus_per_worker": 8,
        "priority_class": "c1",
        "requeueIfPreempted": False,
        "resources": miles_promotion.PROD_REWARD_CANARY_RESOURCES,
    }
    monkeypatch.setattr(miles_promotion, "validate_embedded_promotion", lambda *_a, **_k: True)
    monkeypatch.setattr(jobs, "validate_preview", lambda *_a, **_k: {"nodes": 1})
    assert miles_promotion.validate_production_preview(plan, request, {}) == {
        "production_reward_canary": "validated",
        "rendered_nodes": 1,
        "effective_priority_expected": 10000,
    }

    for changed_request in (
        {**request, "priority_class": "c0"},
        {**request, "priority_reason": "override"},
    ):
        with pytest.raises(JobsError, match="exact 1x8 c1/no-requeue"):
            miles_promotion.validate_production_preview(plan, changed_request, {})


def test_reward_canary_live_gate_reads_only_c1_priority(monkeypatch) -> None:
    class Client:
        @staticmethod
        def all_runs():
            return []

    class Wandb:
        @staticmethod
        def run(_):
            raise RuntimeError("Could not find run")

    seen = []

    def kubectl(*args):
        seen.append(args)
        if args[:2] == ("get", "namespace"):
            return {"metadata": {"uid": miles_promotion.PROD_NAMESPACE_UID}}
        if args[:2] == ("get", "priorityclass"):
            return {
                "metadata": {"name": "c1"},
                "value": 10000,
                "preemptionPolicy": "PreemptLowerPriority",
            }
        return {"items": []}

    plan = {
        "execution": {"production_promotion": {"mode": miles_promotion.PROD_REWARD_CANARY_MODE}}
    }
    monkeypatch.setattr(miles_promotion, "validate_embedded_promotion", lambda *_a, **_k: True)
    monkeypatch.setattr(miles_promotion, "_kubectl_json", kubectl)
    observed = miles_promotion.require_live_external(plan, Client(), wandb_api=Wandb())
    assert observed["effective_priority"] == 10000
    assert ("get", "priorityclass", "c1") in seen
    assert all("c0" not in args for args in seen)


def test_long_context_reward_canary_live_gate_reserves_four_nodes(monkeypatch) -> None:
    class Client:
        @staticmethod
        def all_runs():
            return []

    class Wandb:
        @staticmethod
        def run(_):
            raise RuntimeError("Could not find run")

    def kubectl(*args):
        if args[:2] == ("get", "namespace"):
            return {"metadata": {"uid": miles_promotion.PROD_NAMESPACE_UID}}
        if args[:2] == ("get", "priorityclass"):
            return {
                "metadata": {"name": "c1"},
                "value": 10000,
                "preemptionPolicy": "PreemptLowerPriority",
            }
        return {"items": []}

    plan = {
        "execution": {
            "production_promotion": {
                "mode": miles_promotion.PROD_REWARD_CANARY_LONG_V1_MODE
            }
        }
    }
    monkeypatch.setattr(miles_promotion, "validate_embedded_promotion", lambda *_a, **_k: True)
    monkeypatch.setattr(miles_promotion, "_kubectl_json", kubectl)
    observed = miles_promotion.require_live_external(plan, Client(), wandb_api=Wandb())
    assert observed["candidate_nodes"] == 4
    assert observed["node_limit"] == 8


def test_embedded_promotion_rechecks_the_complete_compiled_candidate() -> None:
    plan = _plan()
    assert miles_promotion.validate_embedded_promotion(plan, check_files=False) is True
    explicit = copy.deepcopy(plan)
    explicit["arguments"]["policy_identity_root"] = explicit["arguments"]["model_root"]
    assert miles_promotion.validate_embedded_promotion(explicit, check_files=False) is True
    for section, key, changed in (
        ("arguments", "steps", 1),
        ("arguments", "samples_per_prompt", 4),
        ("arguments", "lr", 1e-6),
        ("arguments", "temperature", 1.0),
        ("arguments", "kl_loss_coef", 0.0),
        ("arguments", "max_tokens_per_gpu", None),
        ("arguments", "data_manifest", "/mnt/sfs/jobs/other/manifest.json"),
        ("arguments", "policy_identity_root", "/mnt/sfs/jobs/other/hf-export"),
        ("execution", "priority", "c2"),
        ("execution", "resources", {**miles_promotion.EXPECTED_RESOURCES, "cpu_request": "63"}),
        ("checkpoint", "root", "/mnt/sfs/jobs/other/torch-dist"),
    ):
        invalid = copy.deepcopy(plan)
        invalid[section][key] = changed
        with pytest.raises(ValueError, match="exact candidate"):
            miles_promotion.validate_embedded_promotion(invalid, check_files=False)


def test_existing_production_promoter_rejects_an_sft_seeded_policy() -> None:
    plan = _plan()
    sft_model = {
        **plan["model"],
        "root": "/mnt/sfs/jobs/synthetic-sft/hf-export",
        "initial_policy": {"kind": "sft_hf_export", "sft_optimizer_step": 44},
    }
    plan["model"] = sft_model
    plan["checkpoint"]["model"] = sft_model
    plan["arguments"]["model_root"] = sft_model["root"]
    with pytest.raises(ValueError, match="exact candidate"):
        miles_promotion.validate_embedded_promotion(plan, check_files=False)


def test_reward_canary_live_files_reopen_its_plan_bound_data(tmp_path, monkeypatch) -> None:
    data = _data()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n")
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    output = tmp_path / "new-output"
    plan = {
        "data": data,
        "execution": {"production_promotion": {"mode": miles_promotion.PROD_REWARD_CANARY_MODE}},
    }
    monkeypatch.setattr(miles_promotion, "PROD_DATA_MANIFEST", str(manifest))
    monkeypatch.setattr(miles_promotion, "PROD_REWARD_CANARY_OUTPUT", str(output))
    monkeypatch.setattr(miles_promotion, "validate_embedded_promotion", lambda *_a, **_k: True)

    miles_promotion.require_live_files(plan, prepared)
    changed = copy.deepcopy(plan)
    changed["data"]["files"]["train"]["rows"] += 1
    with pytest.raises(ValueError, match="data manifest changed"):
        miles_promotion.require_live_files(changed, prepared)


def test_promotion_cannot_hide_external_benchmark_feedback() -> None:
    value = _promotion()
    miles_promotion.validate_promotion(value, check_files=False)
    changed = copy.deepcopy(value)
    changed["benchmark_isolation"]["external_benchmark_hpo_or_checkpoint_inputs"] = 1
    changed["sha256"] = digest({key: item for key, item in changed.items() if key != "sha256"})
    with pytest.raises(ValueError, match="invariant"):
        miles_promotion.validate_promotion(changed, check_files=False)


@pytest.mark.parametrize(
    "reference_name", ("active_canary_binding", "reward_terminal", "native_reload")
)
def test_promotion_rejects_unmaterialized_binding_or_terminal_digest(reference_name) -> None:
    value = _promotion()
    value[reference_name]["file_sha256"] = None
    value["sha256"] = digest({key: item for key, item in value.items() if key != "sha256"})
    with pytest.raises(ValueError, match="reference changed"):
        miles_promotion.validate_promotion(value, check_files=False)


def test_active_canary_binding_is_exactly_dev5_and_bound_to_terminal(monkeypatch) -> None:
    binding = _active_canary_binding()
    terminal = {
        "sha256": "sha256:" + "a" * 64,
        "source_run_name": binding["source_run_name"],
        "source_plan_sha256": binding["source_plan_sha256"],
        "source_request_sha256": binding["source_request_sha256"],
        "submission_binding": {
            "path": "/binding/submission",
            "file_sha256": "1" * 64,
            "receipt_sha256": "2" * 64,
        },
        "controller_observation": {
            "path": "/binding/controller",
            "file_sha256": "3" * 64,
            "receipt_sha256": "4" * 64,
        },
    }
    submission = {
        "source_commit": binding["source_commit"],
        "runtime_bundle_sha256": binding["runtime_bundle_sha256"],
        "api": {
            "base_url": binding["api_base_url"],
            "run_id": binding["api_run_id"],
            "run_name": binding["api_run_name"],
        },
    }
    controller = {
        "kubernetes": {
            "rayjob": {"uid": binding["rayjob_uid"]},
            "workload": {"uid": binding["workload_uid"]},
        }
    }
    observed = {"/binding/submission": submission, "/binding/controller": controller}
    monkeypatch.setattr(
        miles_promotion,
        "_reopen",
        lambda reference, *, check_files: observed[reference["path"]],
    )

    miles_promotion._exact_active_canary(terminal, binding)
    changed = copy.deepcopy(binding)
    changed["workload_uid"] = "44444444-4444-4444-8444-444444444444"
    changed["sha256"] = digest({key: item for key, item in changed.items() if key != "sha256"})
    with pytest.raises(ValueError, match="differs"):
        miles_promotion._exact_active_canary(terminal, changed)

    for key, replacement in (
        ("source_run_name", "chris-q38-miles-rlreward-dev6"),
        ("source_commit", "f" * 40),
        ("source_plan_sha256", "sha256:" + "f" * 64),
        ("source_request_sha256", "sha256:" + "f" * 64),
    ):
        wrong_dev = copy.deepcopy(binding)
        wrong_dev[key] = replacement
        if key == "source_run_name":
            wrong_dev["api_run_name"] = replacement + "-11111111"
        wrong_dev["sha256"] = digest(
            {name: item for name, item in wrong_dev.items() if name != "sha256"}
        )
        with pytest.raises(ValueError, match="exact dev5"):
            miles_promotion._exact_active_canary(terminal, wrong_dev)


def test_active_canary_binding_is_created_once_from_accepted_terminal(
    tmp_path, monkeypatch
) -> None:
    expected = _active_canary_binding()
    terminal = {"sha256": expected["reward_terminal_receipt_sha256"]}
    monkeypatch.setattr(
        miles_promotion,
        "_reference",
        lambda _path: (
            {
                "path": "/accepted/MILES_TERMINAL_ACCEPTED.json",
                "file_sha256": "1" * 64,
                "receipt_sha256": "a" * 64,
            },
            terminal,
        ),
    )
    monkeypatch.setattr(
        miles_promotion,
        "_active_canary_identity",
        lambda _terminal: {
            key: item for key, item in expected.items() if key not in {"schema", "status", "sha256"}
        },
    )
    monkeypatch.setattr("training.miles_acceptance.validate_terminal", lambda *_args, **_kwargs: {})
    output = tmp_path / "ACTIVE_CANARY_BINDING.json"

    assert (
        miles_promotion.accept_active_canary_binding(
            reward_terminal=tmp_path / "MILES_TERMINAL_ACCEPTED.json", output=output
        )
        == expected
    )
    assert json.loads(output.read_text()) == expected
    with pytest.raises(FileExistsError, match="already exists"):
        miles_promotion.accept_active_canary_binding(
            reward_terminal=tmp_path / "MILES_TERMINAL_ACCEPTED.json", output=output
        )


def test_promotion_cross_binds_native_reload_to_the_dev_checkpoint(monkeypatch) -> None:
    value = _promotion()
    for index, name in enumerate(
        ("active_canary_binding", "reward_terminal", "native_reload", "production_data_manifest")
    ):
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
    binding = _active_canary_binding()
    observed = {
        value["active_canary_binding"]["path"]: binding,
        value["reward_terminal"]["path"]: terminal,
        value["native_reload"]["path"]: native,
        value["production_data_manifest"]["path"]: data,
    }
    monkeypatch.setattr(
        miles_promotion,
        "_reopen",
        lambda reference, *, check_files: observed[reference["path"]],
    )
    monkeypatch.setattr(miles_promotion, "_exact_active_canary", lambda *_: None)
    monkeypatch.setattr(miles_promotion, "_exact_data", lambda _: None)
    monkeypatch.setattr("training.miles_acceptance.validate_terminal", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        "training.miles_reload_acceptance.validate_accepted", lambda *_args, **_kwargs: {}
    )

    result = miles_promotion.validate_promotion(value, check_files=True)
    assert result["dev_checkpoint"] == terminal["checkpoint_manifest"]
    assert result["dev_source_plan_sha256"] == miles_promotion.EXPECTED_DEV_CANARY[
        "source_plan_sha256"
    ].removeprefix("sha256:")

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
        owned = [
            {
                "metadata": {
                    "name": f"ray-head-{index}",
                    "labels": {"fleet.ai/run-name": f"chris-q38-cell-{index}"},
                },
                "status": {"phase": "Pending"},
                "spec": {"containers": [{"resources": {"requests": {"nvidia.com/gpu": "8"}}}]},
            }
            for index in range(7)
        ]
        peers = [
            {
                "metadata": {
                    "name": f"peer-ray-head-{index}",
                    "labels": {"fleet.ai/run-name": f"peer-cell-{index}"},
                },
                "status": {"phase": "Running"},
                "spec": {
                    "nodeName": f"peer-node-{index}",
                    "containers": [{"resources": {"limits": {"nvidia.com/gpu": "8"}}}],
                },
            }
            for index in range(23)
        ]
        return {"items": [*owned, *peers]}

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


def test_live_gate_counts_owned_nodes_once_and_rejects_ninth(monkeypatch) -> None:
    class Client:
        @staticmethod
        def all_runs():
            return []

    class Wandb:
        @staticmethod
        def run(_):
            raise RuntimeError("Could not find run")

    pods = [
        {
            "metadata": {
                "name": f"chris-helper-{index}",
                "labels": {},
            },
            "status": {"phase": "Running"},
            "spec": {
                "nodeName": "shared-owned-node" if index < 2 else f"owned-node-{index}",
                "containers": [
                    {
                        "resources": {
                            "requests": {"nvidia.com/gpu": "4"},
                            "limits": {"nvidia.com/gpu": "4"},
                        }
                    }
                ],
            },
        }
        for index in range(8)
    ]

    def kubectl(*args):
        if args[:2] == ("get", "namespace"):
            return {"metadata": {"uid": miles_promotion.PROD_NAMESPACE_UID}}
        if args[:2] == ("get", "priorityclass"):
            return {
                "metadata": {"name": "c1"},
                "value": 10000,
                "preemptionPolicy": "PreemptLowerPriority",
            }
        return {"items": pods}

    monkeypatch.setattr(miles_promotion, "validate_embedded_promotion", lambda *_a, **_k: True)
    monkeypatch.setattr(miles_promotion, "_kubectl_json", kubectl)
    observed = miles_promotion.require_live_external({}, Client(), wandb_api=Wandb())
    assert observed["active_experiment_nodes"] == 7

    pods.append(
        {
            "metadata": {
                "name": "ray-head-peer-looking-name",
                "labels": {"fleet.ai/run-name": "chris-q38-eighth-node"},
            },
            "status": {"phase": "Pending"},
            "spec": {"containers": [{"resources": {"limits": {"nvidia.com/gpu": "8"}}}]},
        }
    )
    with pytest.raises(JobsError, match="exceed eight"):
        miles_promotion.require_live_external({}, Client(), wandb_api=Wandb())
