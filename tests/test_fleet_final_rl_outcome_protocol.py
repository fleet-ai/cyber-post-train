"""Offline tests for the non-launchable Fleet final RL binding contract."""

from __future__ import annotations

import copy
import json
import stat
from pathlib import Path

import pytest

from evals.fleet import dev_outcome_protocol as legacy
from evals.fleet import final_rl_outcome_protocol as protocol
from training.io import canonical_json, digest_json

ROOT = Path(__file__).parents[1]
PARENT_PATH = (
    ROOT / "configs/evaluation/qwen38-blackbox-fleet-final-rl-outcome-protocol-v1.json"
)
LEGACY_PARENT_PATH = (
    ROOT / "configs/evaluation/qwen38-blackbox-fleet-final-outcome-protocol-v1.json"
)
DIGEST = "sha256:" + "1" * 64


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _arm(prefix: str, *, candidate: bool) -> dict:
    value = {
        field: "sha256:" + ("2" if candidate else "1") * 64
        for field in protocol.ARM_FIELDS
        if field.endswith("_sha256")
    }
    value.update(
        checkpoint_identity=(
            "chris-q38-miles-rl-prod1-iter-0000059"
            if candidate
            else "Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
        ),
        served_model_id=f"qwen38-{prefix}-served-v1",
        serving_image="registry.example/sglang@sha256:" + "3" * 64,
        agent_image="registry.example/opencode@sha256:" + "4" * 64,
        proxy_image="registry.example/proxy@sha256:" + "5" * 64,
    )
    for field in protocol.MATCHED_ARM_FIELDS:
        if field.endswith("_sha256"):
            value[field] = DIGEST
    return value


def _bindings(tmp_path: Path) -> dict:
    return {
        "schema": protocol.BINDINGS_SCHEMA,
        "candidate_arm": "rl",
        "training_boundary": {
            "source_class": "fleet_blackbox_tasks_only",
            "training_arm_id": "chris-q38-miles-rl-prod1",
            "training_plan_sha256": "sha256:" + "6" * 64,
            "training_terminal_acceptance_receipt_sha256": "sha256:" + "7" * 64,
            "checkpoint_freeze_receipt_sha256": "sha256:" + "8" * 64,
            "training_input_manifest_sha256": "sha256:" + "9" * 64,
            "external_benchmark_inputs_used": False,
            "external_benchmark_outputs_used": False,
            "external_benchmark_outcomes_used": False,
            "external_benchmark_derived_inputs_used": False,
        },
        "arms": {
            "base": _arm("base", candidate=False),
            "candidate": _arm("rl", candidate=True),
        },
        "result_isolation": {
            "campaign_id": "qwen38-fleet-final-rl-prod1-v1",
            "prepared_output_root": str(tmp_path / "prepared-output"),
            "result_root": str(tmp_path / "prepared-output" / "private-results"),
            "roots_absent": True,
        },
    }


def test_parent_reuses_exact_final_set_and_preserves_legacy_post_sft() -> None:
    task_set = protocol.load_exact_task_set()
    parent = _load(PARENT_PATH)
    protocol.validate_parent(parent, task_set)

    assert parent["candidate_arm"] == "rl"
    assert parent["launchable"] is False
    assert parent["paid_or_scored_work_authorized"] is False
    assert parent["task_set"]["sha256"] == protocol.EXACT_TASK_SET_SHA256
    assert parent["task_set"]["file_sha256"] == protocol.EXACT_TASK_SET_FILE_SHA256
    assert parent["task_set"]["final_test_lock_sha256"] == protocol.EXACT_FINAL_LOCK_SHA256
    assert parent["harness"] == legacy.HARNESS
    assert parent["sampling"] == legacy.SAMPLING
    assert parent["pass_k"] == legacy.PASS_K
    assert parent["external_benchmark_isolation"] == {
        "webexploitbench_state": "sealed_external_evaluation_only",
        "training_input_eligible": False,
        "reward_input_eligible": False,
        "checkpoint_or_hyperparameter_selection_eligible": False,
        "retry_or_stopping_input_eligible": False,
        "failure_analysis_before_checkpoint_freeze": "forbidden",
    }

    legacy_parent = _load(LEGACY_PARENT_PATH)
    legacy.validate_final_protocol(legacy_parent, task_set)
    assert set(legacy_parent["checkpoint_binding_template"]["arms"]) == {
        "base",
        "post_sft",
    }


def test_parent_templates_are_complete_nulls_and_nonlaunchable() -> None:
    parent = _load(PARENT_PATH)
    template = parent["binding_template"]
    assert template["state"] == "unbound"
    assert template["candidate_arm"] == "rl"
    assert set(template["arms"]) == {"base", "candidate"}
    for arm in template["arms"].values():
        assert set(arm) == set(protocol.ARM_FIELDS)
        assert all(value is None for value in arm.values())
    assert set(template["training_boundary"]) == set(protocol.TRAINING_BOUNDARY_FIELDS)
    assert all(value is None for value in template["training_boundary"].values())
    assert set(template["result_isolation"]) == set(protocol.RESULT_ISOLATION_FIELDS)
    assert all(value is None for value in template["result_isolation"].values())


def test_materializes_create_once_receipt_complete_nonlaunchable_child(tmp_path: Path) -> None:
    bindings_path = tmp_path / "bindings.json"
    bindings_path.write_text(canonical_json(_bindings(tmp_path)) + "\n", encoding="utf-8")
    output = tmp_path / "child.json"
    child = protocol.materialize_child(PARENT_PATH, bindings_path, output)

    assert output.exists()
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert child["state"] == "bound_nonlaunchable"
    assert child["candidate_arm"] == "rl"
    assert child["launchable"] is False
    assert child["paid_or_scored_work_authorized"] is False
    assert child["launch_gate"]["all_receipt_digests_present"] is True
    assert child["launch_gate"]["paid_or_scored_work_authorized"] is False
    assert child["parent"]["protocol_sha256"] == _load(PARENT_PATH)["sha256"]
    protocol.validate_child(
        child,
        _load(PARENT_PATH),
        PARENT_PATH,
        protocol.load_exact_task_set(),
    )
    with pytest.raises(FileExistsError):
        protocol.materialize_child(PARENT_PATH, bindings_path, output)


@pytest.mark.parametrize(
    "field",
    [
        "tokenizer_manifest_sha256",
        "chat_template_sha256",
        "serving_execution_contract_sha256",
        "live_pair_parity_receipt_sha256",
        "serving_image",
        "agent_image",
        "proxy_image",
    ],
)
def test_rejects_any_base_candidate_runtime_or_harness_drift(
    tmp_path: Path, field: str
) -> None:
    value = _bindings(tmp_path)
    value["arms"]["candidate"][field] = (
        "registry.example/drift@sha256:" + "a" * 64
        if field.endswith("image")
        else "sha256:" + "a" * 64
    )
    with pytest.raises(ValueError, match="matched arm field differs"):
        protocol.validate_bindings(value, _load(PARENT_PATH), protocol.load_exact_task_set())


@pytest.mark.parametrize(
    "field",
    [
        "external_benchmark_inputs_used",
        "external_benchmark_outputs_used",
        "external_benchmark_outcomes_used",
        "external_benchmark_derived_inputs_used",
    ],
)
def test_rejects_every_external_benchmark_training_channel(tmp_path: Path, field: str) -> None:
    value = _bindings(tmp_path)
    value["training_boundary"][field] = True
    with pytest.raises(ValueError, match="benchmark-derived training inputs"):
        protocol.validate_bindings(value, _load(PARENT_PATH), protocol.load_exact_task_set())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_receipt", "fields differ"),
        ("mutable_checkpoint", "immutable identity"),
        ("same_weights", "distinct weight manifest"),
        ("same_checkpoint", "distinct checkpoint identity"),
        ("same_route", "distinct served model"),
    ],
)
def test_rejects_incomplete_or_nonintervention_bindings(
    tmp_path: Path, mutation: str, message: str
) -> None:
    value = _bindings(tmp_path)
    if mutation == "missing_receipt":
        value["arms"]["candidate"].pop("export_receipt_sha256")
    elif mutation == "mutable_checkpoint":
        value["arms"]["candidate"]["checkpoint_identity"] = "latest"
    elif mutation == "same_weights":
        value["arms"]["candidate"]["weights_manifest_sha256"] = value["arms"]["base"][
            "weights_manifest_sha256"
        ]
    elif mutation == "same_checkpoint":
        value["arms"]["candidate"]["checkpoint_identity"] = value["arms"]["base"][
            "checkpoint_identity"
        ]
    else:
        value["arms"]["candidate"]["served_model_id"] = value["arms"]["base"][
            "served_model_id"
        ]
    with pytest.raises(ValueError, match=message):
        protocol.validate_bindings(value, _load(PARENT_PATH), protocol.load_exact_task_set())


@pytest.mark.parametrize("existing", ["prepared", "result"])
def test_rejects_preexisting_result_or_prepared_root(tmp_path: Path, existing: str) -> None:
    value = _bindings(tmp_path)
    path = Path(
        value["result_isolation"][
            "prepared_output_root" if existing == "prepared" else "result_root"
        ]
    )
    path.mkdir(parents=True)
    with pytest.raises(ValueError, match="already exists"):
        protocol.validate_bindings(value, _load(PARENT_PATH), protocol.load_exact_task_set())


def test_child_output_must_remain_disjoint_from_future_result_roots(tmp_path: Path) -> None:
    value = _bindings(tmp_path)
    bindings_path = tmp_path / "bindings.json"
    bindings_path.write_text(canonical_json(value) + "\n", encoding="utf-8")
    output = Path(value["result_isolation"]["prepared_output_root"]) / "child.json"
    with pytest.raises(ValueError, match="disjoint from result roots"):
        protocol.materialize_child(PARENT_PATH, bindings_path, output)
    assert not output.exists()


def test_rejects_parent_or_child_tampering_even_after_reseal(tmp_path: Path) -> None:
    parent = _load(PARENT_PATH)
    changed_parent = copy.deepcopy(parent)
    changed_parent["pass_k"] = 1
    changed_parent["sha256"] = digest_json(
        {key: item for key, item in changed_parent.items() if key != "sha256"}
    )
    with pytest.raises(ValueError, match="exact frozen contract"):
        protocol.validate_parent(changed_parent, protocol.load_exact_task_set())

    child = protocol.build_child(
        parent,
        PARENT_PATH,
        _bindings(tmp_path),
        protocol.load_exact_task_set(),
    )
    changed_child = copy.deepcopy(child)
    changed_child["launch_gate"]["paid_or_scored_work_authorized"] = True
    changed_child["child_sha256"] = digest_json(
        {key: item for key, item in changed_child.items() if key != "child_sha256"}
    )
    with pytest.raises(ValueError, match="differs from its exact parent"):
        protocol.validate_child(
            changed_child,
            parent,
            PARENT_PATH,
            protocol.load_exact_task_set(),
        )
