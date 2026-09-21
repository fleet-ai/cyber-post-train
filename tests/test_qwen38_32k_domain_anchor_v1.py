import copy
import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cyber_post_train import cli, qwen38_lora_sft_preflight
from training import sft_32k_domain_anchor_v1 as compiler
from training import sft_dispatch
from training import sft_runtime_32k_domain_anchor_v1 as runtime

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "configs" / "runs"
ANCHOR = RUNS / "qwen38-27b-sft-32k-dense-b16-lr5e6-domain-v1.json"
MATCHED_DENSE = RUNS / "qwen38-27b-sft-32k-dense-matched-v1.json"
MATCHED_RUNTIME = ROOT / "training" / "sft_runtime_32k_matched_v1.py"
RUNNER = CliRunner()


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _compile(path: Path) -> tuple[dict, dict, dict]:
    config = _read(path)
    selected = sft_dispatch.compiler_for_config(config)
    plan = selected.compile_sft(config, relative_to=RUNS)
    return config, plan, selected.job_request(plan)


def test_domain_anchor_is_one_exact_fresh_dense_identity():
    config, plan, request = _compile(ANCHOR)
    _, matched, _ = _compile(MATCHED_DENSE)

    assert hashlib.sha256(MATCHED_RUNTIME.read_bytes()).hexdigest() == (
        "5c2fd322305e280902c7f6a805dd12fddd6062b2721109d77b5afb5f5c72d96c"
    )
    assert config["name"] == config["wandb"]["run_id"] == config["wandb"]["name"]
    assert config["name"] == runtime.RUN_NAME
    assert plan["runtime_variant"]["name"] == runtime.RUNTIME_VARIANT
    assert sft_dispatch.compiler_for_plan(plan) is compiler
    assert not qwen38_lora_sft_preflight.is_qwen38_lora_plan(plan)

    assert plan["model"] == matched["model"]
    assert plan["datasets"] == matched["datasets"]
    assert plan["split_manifest_sha256"] == matched["split_manifest_sha256"]
    assert plan["corpus_manifest_sha256"] == matched["corpus_manifest_sha256"]
    for field in (
        "epochs",
        "microbatch_per_gpu",
        "nodes",
        "gpus_per_node",
        "max_length",
        "eval_interval",
        "seed",
    ):
        assert plan["recipe"][field] == matched["recipe"][field]
    assert plan["recipe"] == runtime.base_plan_binding()["recipe"]
    assert plan["recipe"]["batch_size"] == 16
    assert plan["recipe"]["lr"] == 5e-6
    assert plan["recipe"]["max_steps"] == 919
    assert plan["optimizer"] == matched["optimizer"] == runtime.OPTIMIZER

    options = runtime.sft_overrides(plan)
    assert options["strategy"] == "fsdp"
    assert options["optimizer_config.adam_betas"] == [0.9, 0.999]
    assert options["optimizer_config.weight_decay"] == 0.01
    assert options["optimizer_config.max_grad_norm"] == 1.0
    assert options["optimizer_config.scheduler"] == "constant_with_warmup"
    assert options["optimizer_config.num_warmup_steps"] == 0
    assert options["optimizer_config.offload_after_step"] is False
    assert request["workers"] == 1
    assert request["gpus_per_worker"] == 8
    assert request["priority_class"] == "c1"
    assert request["failureAlerts"] is False
    assert request["image"] == runtime.DENSE_IMAGE


def test_domain_anchor_prepares_without_external_work(tmp_path):
    output = tmp_path / "prepared"
    result = RUNNER.invoke(cli.app, ["train", str(ANCHOR), "--output", str(output)])

    assert result.exit_code == 0, result.output
    plan, request = cli._prepared(output)
    assert plan["runtime_variant"]["name"] == runtime.RUNTIME_VARIANT
    assert request == compiler.job_request(plan)
    assert not (output / "PREFLIGHT.json").exists()


@pytest.mark.parametrize(
    ("keys", "replacement"),
    [
        (("recipe", "batch_size"), 8),
        (("recipe", "lr"), 3e-6),
        (("recipe", "epochs"), 2),
        (("recipe", "seed"), 20260920),
        (("optimizer", "weight_decay"), 0.0),
        (("optimizer", "scheduler"), "cosine"),
        (("name",), "chris-q38-other-v1"),
    ],
)
def test_domain_anchor_fails_closed_on_identity_or_optimizer_drift(keys, replacement):
    config = copy.deepcopy(_read(ANCHOR))
    target = config
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = replacement

    with pytest.raises(ValueError):
        compiler.compile_sft(config, relative_to=RUNS)
