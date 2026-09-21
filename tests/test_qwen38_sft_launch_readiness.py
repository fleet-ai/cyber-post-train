"""Source-only launch-readiness checks for the next two Qwen SFT arms.

These checks intentionally stop before every external authority.  In particular,
they prove request intent and the zero-GPU preflight Job shape locally, while a
live Jobs API preview or the direct-submit server dry-run remains the only proof
of the final RayJob's root alert annotation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cyber_post_train import cli
from cyber_post_train.direct_submit import _assert_sft_contract
from cyber_post_train.jobs import FAILURE_ALERT_ANNOTATION, FAILURE_ALERT_OFF
from cyber_post_train.sft_cpu_preflight_job import (
    build_sft_cpu_preflight_job,
    validate_sft_cpu_preflight_job_package,
)
from training.sft_dispatch import compiler_for_plan

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "configs" / "runs"
QUALIFICATION = ROOT / "configs" / "qualification"
RUNNER = CliRunner()
SOURCE_COMMIT = "a" * 40


@pytest.mark.parametrize(
    ("label", "config_name", "run_name", "output_root"),
    [
        (
            "dense_96k",
            "qwen38-teacher3k-96k-full-b8-lr3e6-v3.json",
            "chris-q38-t3k96-b8-v3",
            "/mnt/sfs/jobs/chris-q38-t3k96-b8-v3",
        ),
        (
            "lora_a2",
            "qwen38-27b-lora-sft-r64-a32-anchor-a2-v1.json",
            "chris-q38-lora-sft-a2-v1",
            "/mnt/sfs/jobs/chris-q38-lora-sft-a2-v1",
        ),
    ],
)
def test_next_sft_arms_prepare_and_bind_the_same_fail_closed_launch_rail(
    tmp_path: Path,
    label: str,
    config_name: str,
    run_name: str,
    output_root: str,
) -> None:
    """Both arms reach a fresh CPU preflight without a GPU/API/Kubernetes call."""
    prepared = tmp_path / label
    result = RUNNER.invoke(
        cli.app,
        ["train", str(RUNS / config_name), "--output", str(prepared)],
    )
    assert result.exit_code == 0, result.output

    plan, request = cli._prepared(prepared)
    assert plan["run_name"] == request["name"] == request["title"] == run_name
    assert plan["output_root"] == request["run_dir"] == output_root
    assert request["priority_class"] == "c1"
    assert request["workers"] == 1
    assert request["gpus_per_worker"] == 8
    assert request["failureAlerts"] is False
    assert compiler_for_plan(plan).job_request(plan) == request

    # This is the exact local contract used by direct-submit-sft before it
    # requests a live preview and then performs a server dry-run.  It must not
    # be bypassed merely because an arm has a versioned LoRA wrapper.
    _assert_sft_contract(plan, request)

    package = build_sft_cpu_preflight_job(
        prepared,
        source_commit=SOURCE_COMMIT,
        attempt=1,
    )
    proof = validate_sft_cpu_preflight_job_package(package)
    job = package.job
    pod_template = job["spec"]["template"]
    assert proof["gpus"] == 0
    assert job["metadata"]["annotations"][FAILURE_ALERT_ANNOTATION] == FAILURE_ALERT_OFF
    assert pod_template["metadata"]["annotations"][FAILURE_ALERT_ANNOTATION] == FAILURE_ALERT_OFF
    assert pod_template["spec"]["priorityClassName"] == "c1"


def test_lora_a2_packet_keeps_the_annotation_repair_fallback_explicit() -> None:
    packet = json.loads(
        (QUALIFICATION / "qwen38-lora-anchor-parallel-candidate-a2-v1.json").read_text()
    )
    expected = packet.pop("packet_sha256")
    from cyber_post_train.jobs import digest

    assert expected == digest(packet)
    fallback = packet["launch_rail"][
        "sft_only_fallback_when_normal_preview_omits_only_the_root_annotation"
    ]
    assert "direct-submit-sft" in fallback["command_with_local_sfs_mount"]
    assert "direct-submit-sft" in fallback["command_without_local_sfs_mount"]
    assert any("server dry-run" in item for item in fallback["allowed_only_if"])
