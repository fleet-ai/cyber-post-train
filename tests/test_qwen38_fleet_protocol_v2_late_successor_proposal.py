from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts import prepare_qwen38_fleet_protocol_v2_late_successor_proposal as proposal


def _seal(value: dict) -> dict:
    value["sha256"] = proposal._digest(value)  # noqa: SLF001
    return value


def _write(path: Path, value: dict) -> tuple[str, str]:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return proposal._file_digest(path), value["sha256"]  # noqa: SLF001


def _predecessor() -> dict:
    seeds = proposal.PREDECESSOR_SEEDS
    value = {
        "schema": "cyber_qwen38_fleet_dev17_pass8_comparison_definition_v8",
        "aggregation": "eight_predeclared_pass1_replicas_per_task_and_arm",
        "comparison_arms": proposal.ARMS,
        "original_seeds": list(range(46, 54)),
        "excluded_protocol_v2_seeds": [55, 56, 57, 58, 59, 60, 64, 65],
        "included_seeds": seeds,
        "replacement_mapping": [
            {"invalid_original_seed": 46, "replacement_seed": 54},
            {"invalid_original_seed": 47, "replacement_seed": 62},
            {"invalid_original_seed": 49, "replacement_seed": 61},
            {"invalid_original_seed": 50, "replacement_seed": 63},
            {"invalid_original_seed": 51, "replacement_seed": 68},
            {"invalid_original_seed": 52, "replacement_seed": 67},
            {"invalid_original_seed": 53, "replacement_seed": 66},
        ],
        "replica_protocols": [
            {
                "seed": seed,
                "origin": "retained" if seed < 61 else "whole_pair_successor",
                "protocol_id": f"protocol-{seed}",
                "comparison_protocol_sha256": f"sha256:protocol-{seed}",
            }
            for seed in seeds
        ],
        "superseded_replacements": [],
        "task_count": 17,
        "sessions_per_arm": 136,
        "total_sessions": 272,
        "models": {"base": {"revision": "base"}, "candidate": {"revision": "candidate"}},
        "task_selection_sha256": "sha256:tasks",
        "split_manifest_sha256": "sha256:split",
        "binding_roster_sha256": "sha256:roster",
        "harness": {"harness": "opencode", "harness_version": "1.18.27"},
        "images": {"agent": "sha256:agent", "proxy": "sha256:proxy"},
        "sampling_without_seed": {"temperature": 0.6, "top_p": 0.95},
        "pass_k_per_replica": 1,
        "retry_limit": 0,
        "training_data_eligible": False,
        "whole_replica_pairs_only": True,
        "cell_level_replacement_forbidden": True,
    }
    return _seal(value)


def _preparation(seed: int, schema: str) -> dict:
    rows = []
    for arm in proposal.ARMS:
        rows.append(
            {
                "seed": seed,
                "arm_id": arm,
                "packet_file_sha256": f"sha256:packet-{seed}-{arm}",
                "evaluation_identity_sha256": f"sha256:identity-{seed}-{arm}",
                "evaluation_identity": {
                    "sampling_seed": seed,
                    "arm_id": arm,
                    "harness": "opencode",
                    "pass_k": 1,
                    "retry_limit": 0,
                    "protocol_id": f"protocol-{seed}",
                    "comparison_protocol_sha256": f"sha256:protocol-{seed}",
                    "task_selection_sha256": "sha256:tasks",
                    "split_manifest_sha256": "sha256:split",
                },
                "outer_launcher": {
                    "root_failure_alerts": "off",
                    "root_create_once": True,
                    "priority_class": "c1",
                    "gpu_requests": 0,
                },
            }
        )
    return _seal({"schema": schema, "source_commit": proposal.CANONICAL_GITHUB_MAIN, "arms": rows})


@pytest.fixture
def inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    values = {
        "predecessor": _predecessor(),
        "seed66_retirement": _seal(
            {
                "schema": "q38_dev17_protocol_v2_seed66_whole_pair_retirement_receipt_v1",
                "decision": {
                    "gate": "NO-GO",
                    "classification": "infrastructure_invalid_whole_pair",
                    "whole_pair_excluded": True,
                    "replacement_seed": 69,
                    "accepted_rows_replayed": False,
                    "accepted_rows_spliced": False,
                    "partial_subset_used_for_replacement": False,
                },
                "privacy": {"score_blind": True},
            }
        ),
        "seed67_retirement": _seal(
            {
                "schema": "cyber_q38_matched_pair_score_blind_retirement_v2",
                "decision": {
                    "gate": "NO_GO",
                    "classification": "infrastructure-invalid-for-matched-comparison",
                    "whole_pair_excluded": True,
                    "replacement_sampling_seed": 70,
                    "replacement_mapping_effective": True,
                    "accepted_seed67_rows_may_not_be_replayed": True,
                    "accepted_seed67_rows_may_not_be_reconciled_for_retirement": True,
                    "seed67_rows_may_not_be_reused_for_seed70": True,
                },
            }
        ),
        "seed68_snapshot": _seal(
            {"schema": "q38_seed68_base_terminal_candidate_live_score_blind_snapshot_v1"}
        ),
        "seed69_preparation": _preparation(
            69, "cyber_qwen38_seed69_matched_pair_replacement_preparation_v1"
        ),
        "seed69_current_main": _seal(
            {
                "schema": "cyber_qwen38_seed69_main989_offline_rerender_readiness_v1",
                "canonical_runtime_source": {"commit": proposal.CANONICAL_GITHUB_MAIN},
                "render_twins_byte_identical": True,
                "prepared_pair": {"preparation_receipt": {}},
                "execution": {
                    "create_performed": False,
                    "kubernetes_calls_performed": False,
                    "provider_model_or_scorer_calls_performed": False,
                },
            }
        ),
        "seed70_preparation": _preparation(
            70, "cyber_qwen38_fleet_seed70_matched_replacement_offline_preparation_v1"
        ),
        "seed71_preparation": _preparation(
            71, "cyber_qwen38_fleet_seed71_matched_replacement_offline_preparation_v1"
        ),
        "seed71_hold": _seal(
            {
                "schema": "cyber_q38_seed71_matched_replacement_prep_hold_v1",
                "status": "PREP_HOLD",
                "decision": {
                    "create_authorized": False,
                    "create_performed": False,
                    "candidate_stop_or_mutation_authorized": False,
                },
                "seed68_incident_binding": {"base_terminal_pre_session_failure": True},
            }
        ),
    }
    paths = {}
    specs = {}
    for label, value in values.items():
        path = tmp_path / f"{label}.json"
        file_sha256, self_sha256 = _write(path, value)
        paths[label] = path
        specs[label] = (value["schema"], file_sha256, self_sha256)
    values["seed69_current_main"]["prepared_pair"]["preparation_receipt"] = {
        "file_sha256": specs["seed69_preparation"][1],
        "self_sha256": specs["seed69_preparation"][2],
    }
    values["seed69_current_main"] = _seal(
        {key: value for key, value in values["seed69_current_main"].items() if key != "sha256"}
    )
    file_sha256, self_sha256 = _write(paths["seed69_current_main"], values["seed69_current_main"])
    specs["seed69_current_main"] = (
        values["seed69_current_main"]["schema"],
        file_sha256,
        self_sha256,
    )
    monkeypatch.setattr(proposal, "INPUT_SPECS", specs)
    return paths


def test_confirmed_replacements_and_conditional_target_are_fail_closed(
    tmp_path: Path, inputs: dict[str, Path]
) -> None:
    receipt = proposal.prepare(inputs, tmp_path / "output")
    definition = json.loads(
        (tmp_path / "output" / "COMPARISON_DEFINITION_V9_PROPOSAL.json").read_text()
    )
    mapping = {
        row["invalid_original_seed"]: row["replacement_seed"]
        for row in definition["replacement_mapping"]
    }
    assert definition["included_seeds"] == proposal.CONFIRMED_SUCCESSOR_SEEDS
    assert [row["seed"] for row in definition["replica_protocols"]] == (
        proposal.CONFIRMED_SUCCESSOR_SEEDS
    )
    assert (mapping[51], mapping[52], mapping[53]) == (68, 70, 69)
    conditional = definition["conditional_provisional_replacements"][0]
    assert conditional["successor_seed"] == 71
    assert conditional["auto_activation_allowed"] is False
    assert conditional["active_in_included_seeds"] is False
    assert conditional["active_in_replacement_mapping"] is False
    assert conditional["active_in_replica_protocols"] is False
    assert conditional["affects_aggregation"] is False
    assert conditional["create_authorized"] is False
    assert definition["conditional_atomic_target"]["target_included_seeds"] == (
        proposal.TARGET_SEEDS
    )
    assert receipt["decision"]["gate"] == "HOLD"
    assert receipt["decision"]["create_authorized"] is False
    assert receipt["execution"]["kubernetes_preview_performed"] is False


def test_render_is_deterministic_and_no_replace(tmp_path: Path, inputs: dict[str, Path]) -> None:
    proposal.prepare(inputs, tmp_path / "one")
    proposal.prepare(inputs, tmp_path / "two")
    for name in (
        "COMPARISON_DEFINITION_V9_PROPOSAL.json",
        "SUCCESSOR_INTENT_V9_PROPOSAL.json",
        "SUCCESSOR_RECEIPT_V9_PROPOSAL.json",
    ):
        assert (tmp_path / "one" / name).read_bytes() == (tmp_path / "two" / name).read_bytes()
    with pytest.raises(FileExistsError):
        proposal.prepare(inputs, tmp_path / "one")


def test_retirement_drift_fails_before_write(
    tmp_path: Path, inputs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    bad = json.loads(inputs["seed67_retirement"].read_text())
    bad["decision"]["accepted_seed67_rows_may_not_be_replayed"] = False
    bad = _seal({key: value for key, value in bad.items() if key != "sha256"})
    file_sha256, self_sha256 = _write(inputs["seed67_retirement"], bad)
    specs = copy.deepcopy(proposal.INPUT_SPECS)
    specs["seed67_retirement"] = (bad["schema"], file_sha256, self_sha256)
    monkeypatch.setattr(proposal, "INPUT_SPECS", specs)
    with pytest.raises(ValueError, match="retirement evidence differs"):
        proposal.prepare(inputs, tmp_path / "output")
    assert not (tmp_path / "output").exists()
