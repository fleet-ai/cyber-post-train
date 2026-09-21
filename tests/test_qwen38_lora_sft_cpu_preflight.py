"""Regression coverage for the typed Qwen3.8 LoRA CPU-preflight route."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cyber_post_train import cli, qwen38_lora_sft_preflight
from cyber_post_train.jobs import digest

ROOT = Path(__file__).resolve().parents[1]
A2_CONFIG = ROOT / "configs" / "runs" / "qwen38-27b-lora-sft-r64-a32-anchor-a2-v1.json"
RUNNER = CliRunner()


def _prepared_a2(tmp_path: Path) -> tuple[Path, dict, dict]:
    directory = tmp_path / "prepared"
    result = RUNNER.invoke(cli.app, ["train", str(A2_CONFIG), "--output", str(directory)])
    assert result.exit_code == 0, result.output
    plan, request = cli._prepared(directory)
    return directory, plan, request


def _native_receipt(plan: dict, request: dict) -> dict:
    body = {
        "schema": "cyber_sft_cpu_preflight_v1",
        "status": "passed",
        "gpus": 0,
        "plan_sha256": digest(plan),
        "request_sha256": digest(request),
        "checked": ["native_loader"],
    }
    return {**body, "sha256": digest(body)}


def test_typed_receipt_binds_real_a2_lora_identity_and_rejects_drift(tmp_path):
    _, plan, request = _prepared_a2(tmp_path)
    receipt = qwen38_lora_sft_preflight.build_receipt(plan, request, _native_receipt(plan, request))

    qwen38_lora_sft_preflight.validate_plan_request(plan, request)
    qwen38_lora_sft_preflight.validate_receipt(receipt, plan, request)
    assert receipt["schema"] == qwen38_lora_sft_preflight.SCHEMA
    assert receipt["gpus"] == 0
    assert receipt["lora"] == plan["lora"]
    assert receipt["runtime_variant"] == plan["runtime_variant"]

    wrong_topology = copy.deepcopy(plan)
    wrong_topology["recipe"]["gpus_per_node"] = 4
    with pytest.raises(ValueError, match="one-node eight-GPU"):
        qwen38_lora_sft_preflight.validate_plan_request(wrong_topology, request)

    changed_adapter = copy.deepcopy(receipt)
    changed_adapter["lora"]["rank"] = 32
    changed_adapter["sha256"] = digest(
        {key: value for key, value in changed_adapter.items() if key != "sha256"}
    )
    with pytest.raises(ValueError, match="identity drifted"):
        qwen38_lora_sft_preflight.validate_receipt(changed_adapter, plan, request)

    changed_native = copy.deepcopy(receipt)
    changed_native["native_sft_preflight"]["checked"] = ["different"]
    changed_native["sha256"] = digest(
        {key: value for key, value in changed_native.items() if key != "sha256"}
    )
    with pytest.raises(ValueError, match="digest mismatch"):
        qwen38_lora_sft_preflight.validate_receipt(changed_native, plan, request)

    unexpected = {**receipt, "unexpected": True}
    unexpected["sha256"] = digest(
        {key: value for key, value in unexpected.items() if key != "sha256"}
    )
    with pytest.raises(ValueError, match="unknown or missing"):
        qwen38_lora_sft_preflight.validate_receipt(unexpected, plan, request)


def test_typed_cli_route_reuses_generic_zero_gpu_job_and_enforces_its_receipt(
    tmp_path, monkeypatch
):
    from cyber_post_train import direct_submit

    directory, plan, request = _prepared_a2(tmp_path)
    source_commit = "a" * 40
    native_receipt = _native_receipt(plan, request)
    calls = []

    class SyntheticKubectl:
        def __init__(self, context):
            self.context = context

    def create(**kwargs):
        calls.append(("create", kwargs))
        return {"submitted": True, "name": "synthetic-pre-a02", "gpus": 0}

    def collect(**kwargs):
        calls.append(("collect", kwargs))
        return native_receipt

    monkeypatch.setattr(cli, "_clean_source_commit", lambda: source_commit)
    monkeypatch.setattr(direct_submit, "Kubectl", SyntheticKubectl)
    monkeypatch.setattr(direct_submit, "create_sft_cpu_preflight_once", create)
    monkeypatch.setattr(direct_submit, "collect_sft_cpu_preflight", collect)

    # The generic route fails before it can create a Job for a Qwen LoRA plan.
    blocked = RUNNER.invoke(
        cli.app,
        ["sft-cpu-preflight-job-create", str(directory), "--context", "prod"],
    )
    assert blocked.exit_code == 2
    assert not calls

    created = RUNNER.invoke(
        cli.app,
        [
            "qwen38-lora-sft-cpu-preflight-job-create",
            str(directory),
            "--context",
            "prod",
            "--attempt",
            "2",
        ],
    )
    assert created.exit_code == 0, created.output
    assert calls[0][0] == "create"
    assert calls[0][1]["directory"] == directory
    assert calls[0][1]["source_commit"] == source_commit
    assert calls[0][1]["attempt"] == 2
    assert calls[0][1]["journal"] == directory / "SFT_CPU_PREFLIGHT_A02.jsonl"
    assert calls[0][1]["kubectl"].context == "prod"

    collected = RUNNER.invoke(
        cli.app,
        [
            "qwen38-lora-sft-cpu-preflight-job-collect",
            str(directory),
            "--context",
            "prod",
            "--attempt",
            "2",
        ],
    )
    assert collected.exit_code == 0, collected.output
    assert calls[1][0] == "collect"
    assert calls[1][1]["directory"] == directory
    assert calls[1][1]["source_commit"] == source_commit
    assert calls[1][1]["attempt"] == 2
    assert calls[1][1]["kubectl"].context == "prod"

    receipt = cli._read(directory / "PREFLIGHT.json")
    assert receipt == qwen38_lora_sft_preflight.build_receipt(plan, request, native_receipt)
    cli._require_preflight(directory, plan, request)

    # A raw generic receipt has the right loader checks but is not accepted as
    # proof of the named Qwen-LoRA preflight route.
    (directory / "PREFLIGHT.json").write_text(json.dumps(native_receipt))
    with pytest.raises(ValueError, match="identity drifted"):
        cli._require_preflight(directory, plan, request)
