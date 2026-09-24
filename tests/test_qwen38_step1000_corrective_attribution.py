from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "configs" / "qualification" / "qwen38-step1000-corrective-attribution-v1.json"
HELDOUT_PATH = (
    ROOT / "configs" / "evaluation" / "qwen38-teacher3k-fleet-heldout20-pass4-protocol-v2.json"
)
DOC_PATH = ROOT / "docs" / "QWEN38_STEP1000_CORRECTIVE_ATTRIBUTION_2026-09-24.md"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _logical_digest(value: dict) -> str:
    body = {key: item for key, item in value.items() if key != "sha256"}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def test_plan_is_self_digesting_blocked_and_binds_current_sources() -> None:
    plan = _load(PLAN_PATH)
    assert plan["sha256"] == _logical_digest(plan)
    assert plan["status"] == "blocked_no_launch"
    assert plan["launch_authorized"] is False
    assert plan["prepared_from_main"] == "ea5f67dd6a955218c4d4ee2946e365189cfd3a8d"

    for name in ("causal_synthesis", "heldout20", "base_model"):
        source = plan["authority"][name]
        assert _file_digest(ROOT / source["path"]) == source["file_sha256"]
    assert (
        _load(ROOT / plan["authority"]["causal_synthesis"]["path"])["sha256"]
        == (plan["authority"]["causal_synthesis"]["sha256"])
    )
    assert _load(HELDOUT_PATH)["sha256"] == plan["authority"]["heldout20"]["sha256"]
    step1000 = plan["authority"]["step1000"]
    assert _file_digest(ROOT / step1000["acceptance_path"]) == (step1000["acceptance_file_sha256"])
    assert _load(ROOT / step1000["acceptance_path"])["sha256"] == (step1000["acceptance_sha256"])
    tool_contract = plan["authority"]["current_tool_contract"]
    assert (
        _file_digest(ROOT / tool_contract["source_catalog_path"])
        == (tool_contract["source_catalog_file_sha256"])
    )
    assert (
        _canonical_digest(_load(ROOT / tool_contract["source_catalog_path"]))
        == (tool_contract["source_catalog_sha256"])
    )

    assert all(value == 0 for key, value in plan["operation"].items() if key != "results_unsealed")
    assert plan["operation"]["results_unsealed"] is False
    assert plan["shared_exclusions"]["external_evaluations_in_scope"] is False
    assert plan["shared_exclusions"]["external_benchmark_content_present"] is False
    assert plan["shared_exclusions"]["private_prompt_trace_flag_answer_or_score_present"] is False
    assert plan["safety"]["every_future_root_job_or_rayjob_annotation"] == {
        "fleet.ai/failure-alerts": "off"
    }
    assert plan["safety"]["priority"] == "c1"
    assert plan["safety"]["server_preview_required_before_create"] is True


def test_tool_name_discriminator_is_the_exact_four_cell_dev13_factorial() -> None:
    plan = _load(PLAN_PATH)
    experiment = plan["experiments"]["A_tool_name_discriminator"]
    heldout = _load(HELDOUT_PATH)
    clean_dev = [
        row
        for row in heldout["classification"]["tasks"]
        if row["source_role"] == "dev" and row["training_lineage_status"] == "clean"
    ]
    assert len(clean_dev) == len({row["group_id"] for row in clean_dev}) == 13
    assert experiment["selection"] == {
        "stratum": "clean_dev13",
        "task_family_count": 13,
        "final_test_families_unsealed": 0,
        "pass_k": 4,
        "attempt_seeds": [975414717, 3965467, 807861398, 1399333715],
        "planned_sessions": 208,
    }
    assert experiment["selection"]["planned_sessions"] == 13 * 4 * 4
    assert [stage["planned_sessions"] for stage in experiment["stage_order"]] == [104, 104]
    assert experiment["stage_order"][0]["cells"] == [
        "base__current_prefixed",
        "step1000__current_prefixed",
    ]
    assert experiment["stage_order"][1]["start_only_if"].startswith("A0 confirms")
    assert experiment["factorial"]["cells"] == [
        "base__current_prefixed",
        "base__historical_bare_aliases",
        "step1000__current_prefixed",
        "step1000__historical_bare_aliases",
    ]
    assert experiment["catalogs"]["current_prefixed"]["model_facing_names"] == [
        "fleet_bash",
        "fleet_submit_report",
    ]
    assert experiment["catalogs"]["historical_bare_aliases"]["model_facing_names"] == [
        "bash",
        "submit_report",
    ]
    assert (
        experiment["fixed_treatment"]["source_tool_catalog_sha256"]
        == (plan["authority"]["current_tool_contract"]["source_catalog_sha256"])
    )
    assert experiment["fixed_treatment"]["agent_image"].startswith("sha256:")
    assert "@sha256:" in experiment["fixed_treatment"]["proxy_image"]
    assert experiment["fixed_treatment"]["serving_profile"] == {
        "precision": "bf16",
        "quantization": None,
        "tensor_parallel_size": 1,
        "data_parallel_size": 8,
        "context_length": 262144,
        "kv_cache_dtype": "fp8_e4m3",
        "runtime_image": (
            "lmsysorg/sglang@sha256:"
            "d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9"
        ),
        "reasoning_parser": "qwen3",
        "tool_call_parser": "qwen3_coder",
    }
    assert experiment["operator_reuse"]["accepted_merged_commit"] is None
    assert "104 cells" in experiment["operator_reuse"]["known_shape_gap"]
    assert experiment["preregistration_gate"].startswith("Before A0 starts")
    assert {blocker["id"] for blocker in experiment["blockers"]} == {
        "current-heldout-operator-not-merged",
        "bare-alias-adapter-absent",
        "factorial-join-receipt-absent",
        "fresh-route-parity-receipts-absent",
    }


def test_clipping_ablation_changes_only_context_and_has_exact_64_step_shape() -> None:
    plan = _load(PLAN_PATH)
    experiment = plan["experiments"]["B_raw_clipping_ablation"]
    source = experiment["paired_source"]
    training = experiment["training"]

    assert source["source_sessions"] == source["target_occurrences"] == 512
    assert source["rows_per_arm"] == 512
    assert source["targets_per_session"] == 1
    assert source["session_reuse_within_arm"] == 0
    assert source["per_teacher_model_session_cap"] == 80
    assert source["per_reviewed_task_family_session_cap"] == 8
    assert source["deterministic_selection"]["cap_application"].endswith(
        "fail closed if fewer qualify."
    )
    assert source["selection_manifest_sha256"] is None
    assert training["global_batch"] == 8
    assert training["optimizer_steps"] == 64
    assert training["global_batch"] * training["optimizer_steps"] == source["rows_per_arm"]
    assert training["epochs_over_diagnostic_rows"] == 1
    assert training["method"] == "full_weight_fsdp_bf16"
    assert training["shuffle_seed"] == 20260921
    assert training["learning_rate"] == 3e-6
    assert training["optimizer"] == {
        "family": "AdamW",
        "adam_betas": [0.9, 0.999],
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "scheduler": "constant_with_warmup",
        "num_warmup_steps": 0,
        "offload_after_step": False,
    }
    assert experiment["context_budget"]["pre_target_tokens"] == 8192
    assert experiment["arms"]["FIXED"]["anchorless_rows_allowed"] == 0
    assert experiment["arms"]["CLIPPED"]["required_nonzero_clip_offset_rows"] == 512
    assert experiment["evaluation"]["selection"] == "clean_dev13"
    assert experiment["evaluation"]["pass_k"] == 4
    assert experiment["evaluation"]["planned_new_sessions"] == 13 * 2 * 4
    assert {blocker["id"] for blocker in experiment["blockers"]} == {
        "message-aligned-builder-not-merged",
        "paired-512-selection-absent",
        "paired-window-builder-absent",
        "64-step-runtime-successor-absent",
        "training-artifact-chain-absent",
    }


def test_documentation_describes_both_experiments_and_no_launch() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")
    for phrase in (
        "no corpus, training, serving, or evaluation work launched",
        "Test the historical tool names without training",
        "Isolate raw clipping in 64 optimizer steps",
        "does not create another evaluation or training operator",
        'fleet.ai/failure-alerts: "off"',
    ):
        assert phrase in text
