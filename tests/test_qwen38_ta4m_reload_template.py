from __future__ import annotations

import json
from pathlib import Path

from training.io import digest_json

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "configs/qualification/qwen38-ta4m-dev1-reload-v1.template.json"


def _read() -> dict:
    value = json.loads(TEMPLATE.read_text())
    assert value["sha256"] == digest_json(
        {key: item for key, item in value.items() if key != "sha256"}
    )
    return value


def test_ta4m_reload_template_is_inert_until_terminal_and_seal_are_bound() -> None:
    value = _read()
    assert value["launchable"] is False
    assert value["status"] == "blocked_waiting_source_terminal_and_checkpoint_seal"

    unresolved = value["unresolved_bindings"]
    assert len(unresolved) == len(set(unresolved)) == 12
    for path in unresolved:
        current = value
        for part in path.split("."):
            current = current[part]
        assert current in (None, False)


def test_ta4m_reload_preserves_exact_source_treatment_and_world_size() -> None:
    value = _read()
    source, reload = value["source"], value["reload"]

    assert source["run_name"] == "chris-q38-ta4m-dev1"
    assert source["plan_sha256"] == (
        "sha256:883ba74af9e36c19be8a6683e0a1a16fdbfc074e4555ee17a780c472d41662b3"
    )
    assert source["model"] == {
        "repo": "Qwen/Qwen3.8-27B",
        "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "weight_manifest_sha256": (
            "sha256:06c94e47c0e31fd331ed410665c830ab1b657f90f15a1b11e7bc45e2de00f352"
        ),
    }
    assert source["selection"] == {
        "mode": "task_outcomes_only",
        "fleet_dev_protocol_sha256": (
            "sha256:d9423651420dcf1c39f7b870daafe8a74dec18db6eac8684b3845d881dc61019"
        ),
        "reference_ce_enabled": False,
    }
    assert source["pause_after_step"] == 1
    assert "pause_after_step" not in source["recipe"]
    assert source["recipe"]["gpus_per_node"] == value["checkpoint_seal"]["world_size"] == 4
    assert reload["resources"]["workers"] == 1
    assert reload["resources"]["gpus_per_worker"] == reload["resources"]["total_gpus"] == 4


def test_ta4m_reload_is_zero_update_native_validation_not_continuation() -> None:
    value = _read()
    reload = value["reload"]

    assert reload["mode"] == reload["recovery_config"]["mode"] == "validate"
    assert reload["optimizer_steps_authorized"] == 0
    assert "pause_after_step" in reload["source_plan_transform"]["remove"]
    assert reload["resources"]["priority_class"] == "c1"
    assert reload["resources"]["expected_priority_value"] == 10_000
    assert reload["resources"]["requeueIfPreempted"] is False
    assert "optimizer_steps_executed_equals_zero" in reload["acceptance"]
    assert (
        "validation_scope_equals_checkpoint_and_sampler_reload_only_no_ce" in reload["acceptance"]
    )
    assert all("resume" not in gate for gate in reload["acceptance"])


def test_ta4m_checkpoint_seal_is_cpu_only_and_create_once() -> None:
    value = _read()
    seal = value["checkpoint_seal"]

    assert seal["operation"] == "cpu_only_create_once"
    assert seal["gpu_reload_verified"] is False
    assert seal["source_checkpoint_path"].endswith("/checkpoints/global_step_1")
    assert seal["source_checkpoint_receipt_path"].endswith("/checkpoint_receipts/step-000001.json")
    source = value["source"]["prepared_dir"]
    assert seal["command"][2:5] == ["cyber-post-train", "checkpoint-seal", source]
    assert source == "/mnt/sfs/jobs/chris-q38-study-corpora-v1/metrics-canary-ta4m-v1"
    assert seal["command"][5] == "1"
    assert "create_once_manifest_destination_absent" in seal["requires"]
