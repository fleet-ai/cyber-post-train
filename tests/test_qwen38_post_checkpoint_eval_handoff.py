"""Offline contract for the Qwen3.8 post-checkpoint evaluation handoff."""

from __future__ import annotations

import json
import re
from pathlib import Path

from evals.fleet import dev_outcome_protocol as fleet
from evals.webexploitbench import study_parent_protocol as web
from evals.webexploitbench.tensorlake import checkpoint_eval_plan
from training.io import digest_json, file_sha256

ROOT = Path(__file__).resolve().parents[1]
HANDOFF_PATH = ROOT / "configs/evaluation/qwen38-post-checkpoint-eval-handoff-v1.template.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def reopen(reference: dict) -> dict:
    path = ROOT / reference["path"]
    assert reference["file_sha256"] == file_sha256(path)
    value = read(path)
    assert reference["embedded_sha256"] == value["sha256"]
    return value


def test_handoff_is_self_digesting_inert_and_unbound() -> None:
    handoff = read(HANDOFF_PATH)
    assert handoff["sha256"] == digest_json(
        {key: value for key, value in handoff.items() if key != "sha256"}
    )
    assert handoff["state"] == "blocked_waiting_exact_candidate_and_live_serving_evidence"
    assert handoff["launchable"] is False
    assert handoff["paid_or_scored_work_authorized"] is False
    assert all(value is None for value in handoff["binding_template"].values())
    assert handoff["execution_record"] == {
        "kind": "offline_template_only_not_an_evaluation_plan",
        "network_calls_performed": 0,
        "cluster_calls_performed": 0,
        "jobs_submitted": 0,
        "tensorlake_sandboxes_created": 0,
        "benchmark_payloads_read": 0,
        "benchmark_outcomes_read": 0,
    }


def test_fleet_split_references_reopen_and_validate() -> None:
    handoff = read(HANDOFF_PATH)
    selection = handoff["fleet_development_selection"]
    assert selection["role"] == "only_hyperparameter_and_checkpoint_selection_surface"
    for split_name, references in selection["splits"].items():
        protocol = reopen(references["parent_protocol"])
        base = reopen(references["base_control"])
        task_set = reopen(references["task_set"])
        fleet.validate_protocol(protocol, task_set)
        fleet.validate_base_control(base, protocol, task_set)
        assert protocol["split_variant"] == split_name
        assert task_set["split_variant"] == split_name
        assert protocol["task_set"]["sha256"] == task_set["sha256"]
        assert protocol["task_set"]["task_count"] == 20
        assert task_set["training_data_eligible"] is False
        assert protocol["metrics"]["primary"]["name"] == (
            "fleet_dev_paired_mean_success_delta_4fixed"
        )
        assert protocol["selection_rule"]["signal"] == (
            "fleet_dev_paired_mean_success_delta_4fixed only"
        )
        assert "WebExploitBench outcomes" in protocol["selection_rule"]["forbidden_tiebreakers"]


def test_training_and_evaluation_harness_contract_is_explicit() -> None:
    handoff = read(HANDOFF_PATH)
    applicability = handoff["applicability"]
    frozen = handoff["fleet_development_selection"]["frozen_design"]
    web_parent = reopen(handoff["webexploitbench_reporting"]["parent_protocol"])
    web_harness = web_parent["harness"]

    assert applicability["training_harness_family"] == frozen["harness"] == "opencode"
    assert (
        applicability["training_harness_version"]
        == frozen["harness_version"]
        == (web_harness["adapter"]["opencode_version"])
    )
    assert (
        applicability["training_harness_release_asset_sha256"]
        == (web_harness["adapter"]["opencode_release_asset_sha256"])
    )
    assert frozen["context_window_size"] == web_harness["context_policy"]["limit"]["context"]
    assert frozen["max_output_tokens"] == web_harness["context_policy"]["limit"]["output"]
    assert frozen["tools"] == ["bash", "submit_report"]
    assert web_harness["tools"] == ["web_bash", "web_submit_findings"]
    assert "different task-specific submission tools" in applicability["cross_surface_limit"]
    assert "unknown" in applicability["rule"]


def test_webexploitbench_is_paired_reporting_only_and_fail_closed() -> None:
    handoff = read(HANDOFF_PATH)
    reporting = handoff["webexploitbench_reporting"]
    parent = reopen(reporting["parent_protocol"])
    paired = reopen(reporting["paired_plan_template"])
    web.validate_parent(parent)
    checkpoint_eval_plan.validate_template(paired)

    assert reporting["role"] == "sealed_external_reporting_only"
    assert reporting["execution"] == {
        "provider": "tensorlake",
        "kubernetes_queue_used_for_scored_sandboxes": False,
        "tensorlake_has_kubernetes_priority_field": False,
        "maximum_priority_for_any_separately_authorized_cluster_helper": "c1/q1",
        "candidate_only_run_is_capability_comparison": False,
        "paired_controller_launch_implemented": False,
    }
    assert all(value is False for value in reporting["selection_firewall"].values())
    assert paired["selection_firewall"]["hyperparameter_selection_eligible"] is False
    assert paired["selection_firewall"]["checkpoint_selection_eligible"] is False
    assert paired["selection_firewall"]["stopping_tiebreaking_or_retry_signal_eligible"] is False
    assert paired["selection_firewall"]["wandb_export_allowed"] is False
    assert paired["launch_gate"]["this_template_may_be_passed_to_tensorlake_controller"] is False
    assert reporting["execution"]["paired_controller_launch_implemented"] is False


def test_priority_and_exact_eligibility_gates_cannot_be_misread_as_launch() -> None:
    handoff = read(HANDOFF_PATH)
    execution = handoff["fleet_development_selection"]["execution"]
    assert execution["new_or_changed_cluster_executable_requires_dev_first"] is True
    assert execution["maximum_pod_priority_class"] == "c1"
    assert execution["expected_queue_priority_class"] == "q1"
    assert execution["expected_priority_value"] == 10_000
    assert execution["c0_or_q0_allowed"] is False
    assert execution["gpu_job_required_for_controller"] is False

    gates = "\n".join(handoff["eligibility_gates_in_order"])
    for required in (
        "terminal training acceptance",
        "zero-update reload",
        "BF16 export",
        "serving registrations",
        "live base/candidate parity",
        "OpenCode 1.18.27",
        "Fleet development split",
        "authoritative paired Fleet outcomes",
        "Fleet development selection decision",
        "WebExploitBench paired duplicate inventory",
        "paired controller",
        "released",
    ):
        assert required in gates


def test_template_contains_no_benchmark_or_secret_payload() -> None:
    handoff = read(HANDOFF_PATH)
    text = HANDOFF_PATH.read_text(encoding="utf-8")
    assert re.search(r"\bsk_[A-Za-z0-9_-]{20,}", text) is None
    assert re.search(r"\btl_apiKey_[A-Za-z0-9_-]{20,}", text) is None
    assert "FLAG{" not in text
    assert "task_graph" not in text
    assert "score" not in handoff["binding_template"]
    assert "prompt" not in handoff["binding_template"]
    assert "trace" not in handoff["binding_template"]
    assert "answer" not in handoff["binding_template"]
    assert "credential" not in handoff["binding_template"]
