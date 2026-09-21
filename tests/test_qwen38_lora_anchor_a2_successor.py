import base64
import copy
import gzip
import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cyber_post_train import cli
from cyber_post_train.jobs import FAILURE_ALERT_ANNOTATION, FAILURE_ALERT_OFF
from cyber_post_train.sft_cpu_preflight_job import (
    build_sft_cpu_preflight_job,
    validate_sft_cpu_preflight_job_package,
)
from training import sft, sft_dispatch, sft_runtime
from training import sft_lora_anchor_a2_v1 as a2_compiler
from training import sft_runtime_lora_anchor_a2_v1 as a2_runtime

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "configs" / "runs"
A1_CONFIG = RUNS / "qwen38-27b-lora-sft-r64-a32-anchor-v1.json"
A2_CONFIG = RUNS / "qwen38-27b-lora-sft-r64-a32-anchor-a2-v1.json"
A2_PACKET = ROOT / "configs" / "qualification" / "qwen38-lora-anchor-parallel-candidate-a2-v1.json"
A1_COLLISION = (
    ROOT
    / "docs"
    / "evidence"
    / "qwen38-lora-anchor-a1-cpu-preflight-output-collision-20260921.json"
)
RUNNER = CliRunner()
HISTORICAL_SFT_RUNTIME_SHA256 = "8cb671f377d089e1303248e237f386c256b212b15e41dadc07ac49cf2695ee17"


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _without_identity(value: dict) -> dict:
    result = copy.deepcopy(value)
    result.pop("name", None)
    result.pop("run_name", None)
    result.pop("output_root", None)
    wandb = result.get("wandb")
    if isinstance(wandb, dict):
        wandb.pop("run_id", None)
        wandb.pop("name", None)
    return result


def _without_execution_wrapper(value: dict) -> dict:
    result = _without_identity(value)
    result.pop("runtime_variant", None)
    result.pop("runtime_sha256", None)
    return result


def _bundle(request: dict) -> dict:
    encoded = request["env"].get("CYBER_SFT_BUNDLE")
    if encoded is None:
        encoded = "".join(
            value
            for _, value in sorted(
                (int(name.rsplit("_", 1)[1]), value)
                for name, value in request["env"].items()
                if name.startswith("CYBER_SFT_BUNDLE_")
            )
        )
    return json.loads(gzip.decompress(base64.b64decode(encoded, validate=True)))


def test_a2_is_an_identity_only_successor_of_the_broad_lora_anchor():
    a1 = _read(A1_CONFIG)
    a2 = _read(A2_CONFIG)

    assert _without_execution_wrapper(a2) == _without_execution_wrapper(a1)
    assert a2["runtime_variant"] == a2_runtime.RUNTIME_VARIANT
    assert a2["name"] == "chris-q38-lora-sft-a2-v1"
    assert a2["output_root"] == "/mnt/sfs/jobs/chris-q38-lora-sft-a2-v1"
    assert a2["name"] == a2["wandb"]["run_id"] == a2["wandb"]["name"]

    plan = a2_compiler.compile_sft(a2, relative_to=RUNS)
    request = a2_compiler.job_request(plan)
    base_plan = sft.compile_sft(a1, relative_to=RUNS)
    assert _without_execution_wrapper(plan) == _without_execution_wrapper(base_plan)
    assert hashlib.sha256((ROOT / "training/sft_runtime.py").read_bytes()).hexdigest() == (
        HISTORICAL_SFT_RUNTIME_SHA256
    )
    assert plan["runtime_variant"] == {
        "schema": a2_runtime.RUNTIME_BINDING_SCHEMA,
        "name": a2_runtime.RUNTIME_VARIANT,
        "base_runtime_sha256": HISTORICAL_SFT_RUNTIME_SHA256,
    }
    assert (
        plan["runtime_sha256"]
        == hashlib.sha256(
            (ROOT / "training/sft_runtime_lora_anchor_a2_v1.py").read_bytes()
        ).hexdigest()
    )
    assert plan["runtime_sha256"] != plan["runtime_variant"]["base_runtime_sha256"]
    assert plan["qualification_gate"] == sft_runtime.QWEN38_LORA_PRODUCTION_QUALIFICATION
    # The wrapper keeps one temporary in-memory identity binding while the
    # historical runtime calls back into its validator. That re-entry must not
    # weaken the sealed identity or fail spuriously.
    with a2_runtime.base_plan_context():
        a2_runtime.validate_plan(plan, check_files=False)
    assert a2_runtime.delegated_plan(plan) == {
        **base_plan,
        "run_name": a2["name"],
        "output_root": a2["output_root"],
        "wandb": plan["wandb"],
    }
    assert request["name"] == request["title"] == a2["name"]
    assert request["run_dir"] == a2["output_root"]
    assert request["env"]["WANDB_RUN_ID"] == request["env"]["WANDB_NAME"] == a2["name"]
    assert request["priority_class"] == "c1"
    assert request["failureAlerts"] is False
    assert sft_dispatch.compiler_for_config(a2) is a2_compiler
    assert sft_dispatch.compiler_for_plan(plan) is a2_compiler

    bundle = _bundle(request)
    assert hashlib.sha256(bundle["runtime"].encode()).hexdigest() == plan["runtime_sha256"]
    base_source = bundle["extra_files"]["training/sft_runtime.py"].encode()
    assert hashlib.sha256(base_source).hexdigest() == plan["runtime_variant"]["base_runtime_sha256"]


@pytest.mark.parametrize(
    "path",
    [
        ("name",),
        ("output_root",),
        ("wandb", "run_id"),
        ("wandb", "name"),
    ],
)
def test_a2_rejects_a_partial_identity_reversion(path):
    a1 = _read(A1_CONFIG)
    a2 = _read(A2_CONFIG)
    target = a2
    source = a1
    for key in path[:-1]:
        target = target[key]
        source = source[key]
    target[path[-1]] = source[path[-1]]

    with pytest.raises(ValueError, match="Qwen3.8 LoRA requires"):
        a2_compiler.compile_sft(a2, relative_to=RUNS)


def test_a2_prepare_uses_the_versioned_compiler_without_an_sfs_mount(tmp_path):
    prepared = tmp_path / "prepared"

    result = RUNNER.invoke(cli.app, ["train", str(A2_CONFIG), "--output", str(prepared)])

    assert result.exit_code == 0, result.output
    plan, request = cli._prepared(prepared)
    assert plan["runtime_variant"]["name"] == a2_runtime.RUNTIME_VARIANT
    assert request == a2_compiler.job_request(plan)
    assert request["failureAlerts"] is False


def test_a2_cpu_preflight_package_keeps_the_root_alert_and_zero_gpu_contract(tmp_path):
    prepared = tmp_path / "prepared"
    result = RUNNER.invoke(cli.app, ["train", str(A2_CONFIG), "--output", str(prepared)])
    assert result.exit_code == 0, result.output

    package = build_sft_cpu_preflight_job(
        prepared,
        source_commit="a" * 40,
        attempt=1,
    )
    proof = validate_sft_cpu_preflight_job_package(package)
    job = package.job
    pod = job["spec"]["template"]["spec"]

    assert proof["gpus"] == 0
    assert job["metadata"]["annotations"][FAILURE_ALERT_ANNOTATION] == FAILURE_ALERT_OFF
    assert pod["containers"][0]["resources"] == {
        "requests": {"cpu": "4", "memory": "32Gi", "ephemeral-storage": "2Gi"},
        "limits": {"cpu": "8", "memory": "48Gi", "ephemeral-storage": "4Gi"},
    }
    assert pod["volumes"][0]["persistentVolumeClaim"]["readOnly"] is True


def test_a1_output_collision_is_sanitized_and_a2_packet_preserves_its_lesson():
    evidence = _read(A1_COLLISION)
    expected_evidence_digest = evidence.pop("sha256")
    assert expected_evidence_digest == _digest(evidence)
    assert evidence["status"] == "terminal_failure_released_no_gpu_allocation"
    assert evidence["attempt"]["job"]["uid"] == "b0fcf2e3-3901-4f5d-8dc8-ff2016b57230"
    assert evidence["attempt"]["pod"]["uid"] == "6461aa33-8472-4256-9aa9-ff4c8c120b7d"
    assert evidence["precreate_contract_observed"]["root_failure_alert_annotation"] == {
        "fleet.ai/failure-alerts": "off"
    }
    assert evidence["precreate_contract_observed"]["gpus"] == 0
    assert evidence["sanitized_terminal_receipt"] == {
        "schema": "cyber_sft_cpu_preflight_observation_v1",
        "status": "failed",
        "error_class": "FileExistsError",
    }
    assert evidence["cleanup"]["post_delete_read_only_recheck"] == {
        "job_present": False,
        "pods_present": False,
    }
    assert not {"logs", "prompts", "traces"} & set(evidence)

    packet = _read(A2_PACKET)
    expected_packet_digest = packet.pop("packet_sha256")
    assert expected_packet_digest == _digest(packet)
    assert packet["candidate"]["config_path"] == str(A2_CONFIG.relative_to(ROOT))
    assert (
        packet["candidate"]["config_file_sha256"]
        == "sha256:" + hashlib.sha256(A2_CONFIG.read_bytes()).hexdigest()
    )
    assert packet["identity_successor_of"]["terminal_preflight_evidence_path"] == str(
        A1_COLLISION.relative_to(ROOT)
    )
    assert packet["identity_successor_of"]["only_allowed_scientific_config_leaf_differences"] == [
        "name",
        "output_root",
        "wandb.run_id",
        "wandb.name",
    ]
    assert (
        packet["identity_successor_of"]["required_execution_binding_difference"]
        == "runtime_variant"
    )
    assert packet["candidate"]["runtime"]["variant"] == a2_runtime.RUNTIME_VARIANT
    contract = packet["data_and_context_contract"]
    assert "visible assistant actions" in contract["supervision"]
    assert "same frozen 32K teacher corpus" in contract["paired_comparison"]
    assert "static 32K packed windows" in contract["offline_context"]
    assert "not part of A2" in contract["future_student_visible_reasoning"]
    assert packet["launch_rail"]["remote_preparation_boundary"]["status"] == (
        "supported_sft_only_zero_gpu_preflight_job"
    )
    assert any("A1's output" in item for item in packet["pre_create_checklist"])
