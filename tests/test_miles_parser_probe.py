import dataclasses
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from cyber_post_train import cli
from training import miles
from training import miles_parser_probe as probe


def source_plan() -> dict:
    args = miles.MilesConfig(
        name="chris-q38-miles-rlreward-dev8",
        output_root="/mnt/sfs/jobs/chris-q38-miles-rlreward-dev8",
        model_root="/mnt/sfs/models/qwen3.8-27b-1d4bf0f2",
        torch_dist_root="/mnt/sfs/jobs/chris-miles-base/torch-dist",
        train_data="/mnt/sfs/jobs/chris-miles-data/train.jsonl",
        dev_data="/mnt/sfs/jobs/chris-miles-data/dev.jsonl",
        data_manifest="/mnt/sfs/jobs/chris-miles-data/manifest.json",
        wandb_entity="thefleet",
        wandb_project="cyber-post-train",
        wandb_run_id="chris-q38-miles-rlreward-dev8",
        samples_per_prompt=8,
        lr=2e-6,
        temperature=0.7,
        kl_loss_coef=0.001,
        max_tokens_per_gpu=8192,
        tokens_per_turn=8192,
    )
    return {
        "schema": "cyber_miles_training_v1",
        "run_name": args.name,
        "output_root": args.output_root,
        "checkpoint": {"root": args.torch_dist_root},
        "arguments": dataclasses.asdict(args),
        "execution": {"cluster_target": "dev"},
    }


def source_request(_source: dict) -> dict:
    return {
        "workers": 1,
        "gpus_per_worker": 8,
        "priority_class": "c1",
        "requeueIfPreempted": False,
    }


@pytest.fixture
def plan(monkeypatch) -> dict:
    monkeypatch.setattr(probe, "training_request", source_request)
    return probe.compile_probe(
        source_plan(),
        source_file_sha256="sha256:" + "a" * 64,
        name="chris-q38-miles-p8k-dev-v1",
        output_root="/mnt/sfs/jobs/chris-q38-miles-p8k-dev-v1",
    )


def test_probe_request_is_one_gpu_dev_c1_without_secrets(plan, monkeypatch) -> None:
    monkeypatch.setattr(probe, "training_request", source_request)
    request = probe.job_request(plan)
    assert request["workers"] == request["gpus_per_worker"] == 1
    assert request["priority_class"] == "c1"
    assert request["requeueIfPreempted"] is False
    assert request["secrets"] == []
    assert request["run_dir"] != plan["source_plan"]["output_root"]
    assert request["env"]["WANDB_MODE"] == "disabled"
    assert "FLEET_API_KEY" not in request["env"]


def test_probe_calls_only_native_parser_and_records_8192(plan, monkeypatch) -> None:
    monkeypatch.setattr(probe, "training_request", source_request)
    calls = []

    def native_args(_source):
        calls.append("native_parser")
        return SimpleNamespace(
            fleet_max_tokens_per_turn=8192,
            rollout_max_context_len=98304,
            rollout_max_response_len=81920,
            actor_num_nodes=1,
            actor_num_gpus_per_node=8,
            num_rollout=1,
            global_batch_size=8,
            load=plan["source_plan"]["checkpoint"]["root"],
            ref_load=plan["source_plan"]["checkpoint"]["root"],
        )

    monkeypatch.setattr(probe, "native_args", native_args)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1),
            distributed=SimpleNamespace(is_initialized=lambda: False),
        ),
    )
    monkeypatch.setattr(probe.Path, "exists", lambda path: False)
    recorded = []
    monkeypatch.setattr(
        probe,
        "_write",
        lambda path, value: recorded.append((path, value)) or {**value, "sha256": "receipt"},
    )
    result = probe.run(plan, run_root=Path(plan["output_root"]))
    assert calls == ["native_parser"]
    assert result["status"] == "validated"
    assert result["native_parser_observation"]["fleet_max_tokens_per_turn"] == 8192
    assert all(result[key] == 0 for key in plan["operations"])
    assert recorded[0][0].name == "PARSER_VALIDATED.json"


def test_probe_cleanly_rejects_without_one_visible_gpu(plan, monkeypatch) -> None:
    monkeypatch.setattr(probe, "training_request", source_request)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0),
            distributed=SimpleNamespace(is_initialized=lambda: False),
        ),
    )
    monkeypatch.setattr(probe.Path, "exists", lambda path: False)
    with pytest.raises(probe.ProbeRejected, match="exactly_one_cuda_device_not_visible"):
        probe.run(plan, run_root=Path(plan["output_root"]))


def test_probe_receipts_are_bound_to_platform_run_dir(plan, monkeypatch) -> None:
    monkeypatch.setenv("RUN_DIR", plan["output_root"])
    monkeypatch.setattr(probe.Path, "is_symlink", lambda _path: False)
    monkeypatch.setattr(probe.Path, "is_dir", lambda _path: True)
    assert probe._run_root(plan) == Path(plan["output_root"])
    monkeypatch.setenv("RUN_DIR", plan["source_plan"]["output_root"])
    with pytest.raises(RuntimeError, match="does not match"):
        probe._run_root(plan)


def test_probe_rejects_any_training_plan_or_resource_drift(plan, monkeypatch) -> None:
    monkeypatch.setattr(probe, "training_request", source_request)
    plan["operations"]["optimizer_steps"] = 1
    with pytest.raises(ValueError, match="plan drift"):
        probe.validate_plan(plan)


def test_public_cli_prepares_probe_once_and_blocks_prod(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(probe, "training_request", source_request)
    source = tmp_path / "source-plan.json"
    source.write_text(json.dumps(source_plan()))
    prepared = tmp_path / "prepared"
    args = [
        "miles-parser-probe",
        str(source),
        "--name",
        "chris-q38-miles-p8k-dev-v1",
        "--output-root",
        "/mnt/sfs/jobs/chris-q38-miles-p8k-dev-v1",
        "--output",
        str(prepared),
    ]
    runner = CliRunner()
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0
    assert json.loads(result.stdout)["submitted"] is False
    assert cli._prepared(prepared)[0]["schema"] == probe.SCHEMA
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )
    checked = runner.invoke(cli.app, ["preflight", str(prepared)])
    assert checked.exit_code == 0
    assert json.loads(checked.stdout)["native_parser_checked"] is False
    assert runner.invoke(cli.app, args).exit_code == 2
    blocked = runner.invoke(cli.app, ["preview", str(prepared), "--cluster", "prod"])
    assert blocked.exit_code == 2
    assert "dev-cluster-only" in blocked.stderr
