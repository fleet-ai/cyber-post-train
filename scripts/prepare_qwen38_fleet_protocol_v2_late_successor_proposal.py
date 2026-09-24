#!/usr/bin/env python3
"""Prepare the score-blind seed 69/70/71 successor proposal.

The output is intentionally inert.  Seed 66 and 67 have exact whole-pair
retirement evidence, but seed 68 does not.  This command therefore records one
atomic target roster while refusing to activate any part of it until a later
version binds a root-approved seed-68 whole-pair retirement receipt.

This command reads local evidence only.  It never contacts Kubernetes, a
provider, a model, a scorer, or a database, and it never creates a workload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

CANONICAL_GITHUB_MAIN = "989fc0c2d5d12bffe2f3c82e9050546ca3974431"
PREDECESSOR_SEEDS = [48, 54, 61, 62, 63, 66, 67, 68]
CONFIRMED_SUCCESSOR_SEEDS = [48, 54, 61, 62, 63, 68, 69, 70]
TARGET_SEEDS = [48, 54, 61, 62, 63, 69, 70, 71]
ARMS = ["base", "candidate"]

INPUT_SPECS = {
    "predecessor": (
        "cyber_qwen38_fleet_dev17_pass8_comparison_definition_v8",
        "sha256:b1e1e057bead9ad80be8df7133e8e52f600a80f46b3539e0b82d349c4ddcd0a8",
        "sha256:d86791c19d0dc55a423707e59aa9ad865653a2d9dc48e6aca1c22d942e4ed24e",
    ),
    "seed66_retirement": (
        "q38_dev17_protocol_v2_seed66_whole_pair_retirement_receipt_v1",
        "sha256:0ecaeaa49409da3a2a8953e79059cce2cf9ac73a0a1028265e3f9bc3d144ca84",
        "sha256:e19e55fb7fb3e99b823419a49cd7657dc8070a97fe0a9cb9e3e6df95e45808aa",
    ),
    "seed67_retirement": (
        "cyber_q38_matched_pair_score_blind_retirement_v2",
        "sha256:cd5fb4dcb2245825b06757c232ab1f867e8ed8a89db90337d8ce98d7b6fb40d5",
        "sha256:7ea7804c34705e38c0f15d31386ca814cbd40b5b5b78851a56e5cee11960be02",
    ),
    "seed68_snapshot": (
        "q38_seed68_base_terminal_candidate_live_score_blind_snapshot_v1",
        "sha256:9e9062e9f35ad8b81113a16e42ba7e9a3217a61d0042ae092d63767ead03fa0f",
        "sha256:fd88e876ab54ea6f864ab09b68cc417e9511cb1e89de524bedc2544838e81ee6",
    ),
    "seed69_preparation": (
        "cyber_qwen38_seed69_matched_pair_replacement_preparation_v1",
        "sha256:5afb27e16017eb4fb7bf1af3d31f0666bf8b1d085bb5e91640fe7ebfbc72d850",
        "sha256:0848aa28bd7a616a8ca640b2928f836f0f4d3e1ad8b319d8d731508a82dcf88d",
    ),
    "seed69_current_main": (
        "cyber_qwen38_seed69_main989_offline_rerender_readiness_v1",
        "sha256:61c8943a759213908f0ba048dc68afab6fe646d8e70a436463dbfb301c618e96",
        "sha256:ecbfaac29e92e79ab765864da8ea6d48960a8b869f2760a2e2e4e86d9e18a1cb",
    ),
    "seed70_preparation": (
        "cyber_qwen38_fleet_seed70_matched_replacement_offline_preparation_v1",
        "sha256:2f44f0989638e9f16612ac6e2c344833462e4eaabefa9199148c7a74631f8bce",
        "sha256:2a14a7ec1eb872bb0cede7c9c212977a4bd703ee57701b67be5cd2b33077f2f6",
    ),
    "seed71_preparation": (
        "cyber_qwen38_fleet_seed71_matched_replacement_offline_preparation_v1",
        "sha256:37ecbf1902593db46f4dd3196c5ad3b4012c5c9f9ebc3fe3c4e7222730c79862",
        "sha256:9683057a4f16492fa1a29e45ebd793f53560d5b467c7baf85052637a67b2457f",
    ),
    "seed71_hold": (
        "cyber_q38_seed71_matched_replacement_prep_hold_v1",
        "sha256:b762445dde10830f24cb4ab88dd1e4a13be64d68c6873bfae660e2255f7a1d39",
        "sha256:434b20891c0da8f5c4e3c3300748ed282e2f5ca3c8adeba7ccb0871fffeeafd3",
    ),
}


def _digest(value: dict[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "sha256"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _read_exact(path: Path, label: str) -> dict[str, Any]:
    schema, file_sha256, self_sha256 = INPUT_SPECS[label]
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("schema") != schema
        or _file_digest(path) != file_sha256
        or value.get("sha256") != self_sha256
        or _digest(value) != self_sha256
    ):
        raise ValueError(f"{label} bytes differ")
    return value


def _binding(path: Path, label: str) -> dict[str, str]:
    schema, file_sha256, self_sha256 = INPUT_SPECS[label]
    return {
        "label": label,
        "schema": schema,
        "source_path": str(path),
        "file_sha256": file_sha256,
        "sha256": self_sha256,
    }


def _validate_predecessor(value: dict[str, Any]) -> None:
    mapping = {
        row["invalid_original_seed"]: row["replacement_seed"]
        for row in value.get("replacement_mapping", [])
    }
    replicas = [row.get("seed") for row in value.get("replica_protocols", [])]
    if (
        value.get("included_seeds") != PREDECESSOR_SEEDS
        or replicas != PREDECESSOR_SEEDS
        or mapping.get(51) != 68
        or mapping.get(52) != 67
        or mapping.get(53) != 66
        or value.get("task_count") != 17
        or value.get("sessions_per_arm") != 136
        or value.get("total_sessions") != 272
        or value.get("comparison_arms") != ARMS
        or value.get("whole_replica_pairs_only") is not True
        or value.get("cell_level_replacement_forbidden") is not True
        or value.get("training_data_eligible") is not False
    ):
        raise ValueError("predecessor comparison contract differs")


def _validate_retirements(seed66: dict[str, Any], seed67: dict[str, Any]) -> None:
    decision66 = seed66.get("decision", {})
    decision67 = seed67.get("decision", {})
    if (
        decision66.get("gate") != "NO-GO"
        or decision66.get("classification") != "infrastructure_invalid_whole_pair"
        or decision66.get("whole_pair_excluded") is not True
        or decision66.get("replacement_seed") != 69
        or decision66.get("accepted_rows_replayed") is not False
        or decision66.get("accepted_rows_spliced") is not False
        or decision66.get("partial_subset_used_for_replacement") is not False
        or seed66.get("privacy", {}).get("score_blind") is not True
        or decision67.get("gate") != "NO_GO"
        or decision67.get("classification") != "infrastructure-invalid-for-matched-comparison"
        or decision67.get("whole_pair_excluded") is not True
        or decision67.get("replacement_sampling_seed") != 70
        or decision67.get("replacement_mapping_effective") is not True
        or decision67.get("accepted_seed67_rows_may_not_be_replayed") is not True
        or decision67.get("accepted_seed67_rows_may_not_be_reconciled_for_retirement") is not True
        or decision67.get("seed67_rows_may_not_be_reused_for_seed70") is not True
    ):
        raise ValueError("confirmed whole-pair retirement evidence differs")


def _validate_preparation(
    value: dict[str, Any], seed: int, current_main: dict[str, Any] | None = None
) -> dict[str, Any]:
    rows = value.get("arms") or value.get("replacement", {}).get("arms")
    source_commit = value.get("source_commit") or value.get("source", {}).get("commit")
    if not isinstance(rows, list) or len(rows) != 2:
        raise ValueError(f"seed {seed} preparation lacks two arms")
    by_arm = {row.get("arm_id"): row for row in rows}
    if set(by_arm) != set(ARMS):
        raise ValueError(f"seed {seed} preparation arm roster differs")
    identities = [by_arm[arm].get("evaluation_identity", {}) for arm in ARMS]
    protocol_ids = {item.get("protocol_id") for item in identities}
    protocol_sha256s = {item.get("comparison_protocol_sha256") for item in identities}
    task_selections = {item.get("task_selection_sha256") for item in identities}
    splits = {item.get("split_manifest_sha256") for item in identities}
    if (
        any(row.get("seed") != seed for row in rows)
        or any(identity.get("sampling_seed") != seed for identity in identities)
        or any(identity.get("harness") != "opencode" for identity in identities)
        or any(identity.get("pass_k") != 1 for identity in identities)
        or any(identity.get("retry_limit") != 0 for identity in identities)
        or len(protocol_ids) != 1
        or len(protocol_sha256s) != 1
        or len(task_selections) != 1
        or len(splits) != 1
        or None in protocol_ids
        or None in protocol_sha256s
    ):
        raise ValueError(f"seed {seed} scientific identity differs")
    for row in rows:
        outer = row.get("outer_launcher", {})
        if (
            outer.get("root_failure_alerts") != "off"
            or outer.get("root_create_once") is not True
            or outer.get("priority_class") != "c1"
            or outer.get("gpu_requests") != 0
        ):
            raise ValueError(f"seed {seed} outer policy differs")
    current_main_valid = source_commit == CANONICAL_GITHUB_MAIN
    if current_main is not None:
        receipt_binding = current_main.get("prepared_pair", {}).get("preparation_receipt", {})
        execution = current_main.get("execution", {})
        current_main_valid = (
            current_main.get("canonical_runtime_source", {}).get("commit") == CANONICAL_GITHUB_MAIN
            and current_main.get("render_twins_byte_identical") is True
            and receipt_binding.get("file_sha256") == INPUT_SPECS[f"seed{seed}_preparation"][1]
            and receipt_binding.get("self_sha256") == INPUT_SPECS[f"seed{seed}_preparation"][2]
            and execution.get("create_performed") is False
            and execution.get("kubernetes_calls_performed") is False
            and execution.get("provider_model_or_scorer_calls_performed") is False
        )
    if not current_main_valid:
        raise ValueError(f"seed {seed} preparation is not bound to canonical main")
    return {
        "seed": seed,
        "origin": "whole_pair_successor",
        "protocol_id": protocol_ids.pop(),
        "comparison_protocol_sha256": protocol_sha256s.pop(),
        "packet_bindings": [
            {
                "arm_id": arm,
                "packet_file_sha256": by_arm[arm]["packet_file_sha256"],
                "evaluation_identity_sha256": by_arm[arm]["evaluation_identity_sha256"],
            }
            for arm in ARMS
        ],
    }


def _build_definition(
    predecessor: dict[str, Any],
    protocols: dict[int, dict[str, Any]],
    evidence: dict[str, dict[str, str]],
) -> dict[str, Any]:
    confirmed_mapping = json.loads(json.dumps(predecessor["replacement_mapping"]))
    confirmed_replacements = {52: (67, 70), 53: (66, 69)}
    for row in confirmed_mapping:
        original = row["invalid_original_seed"]
        if original in confirmed_replacements:
            current, target = confirmed_replacements[original]
            if row["replacement_seed"] != current:
                raise ValueError("predecessor replacement mapping differs")
            row["replacement_seed"] = target
    retained = [row for row in predecessor["replica_protocols"] if row["seed"] not in {66, 67}]
    confirmed_protocols = sorted(
        retained
        + [
            {key: value for key, value in protocols[seed].items() if key != "packet_bindings"}
            for seed in (69, 70)
        ],
        key=lambda row: row["seed"],
    )
    target_mapping = json.loads(json.dumps(confirmed_mapping))
    row51 = next(row for row in target_mapping if row["invalid_original_seed"] == 51)
    if row51["replacement_seed"] != 68:
        raise ValueError("seed68 predecessor mapping differs")
    row51["replacement_seed"] = 71
    target_protocols = sorted(
        [row for row in confirmed_protocols if row["seed"] != 68]
        + [{key: value for key, value in protocols[71].items() if key != "packet_bindings"}],
        key=lambda row: row["seed"],
    )
    if (
        [row["replacement_seed"] for row in target_mapping if row["invalid_original_seed"] >= 51]
        != [71, 70, 69]
        or [row["seed"] for row in target_protocols] != TARGET_SEEDS
        or [row["seed"] for row in confirmed_protocols] != CONFIRMED_SUCCESSOR_SEEDS
    ):
        raise ValueError("atomic successor target differs")
    definition = json.loads(
        json.dumps({key: value for key, value in predecessor.items() if key != "sha256"})
    )
    definition.update(
        {
            "schema": "cyber_qwen38_fleet_dev17_pass8_comparison_definition_v9_proposal_v1",
            "canonical_github_main": CANONICAL_GITHUB_MAIN,
            "predecessor_comparison_definition_file_sha256": evidence["predecessor"]["file_sha256"],
            "predecessor_comparison_definition_sha256": predecessor["sha256"],
            "included_seeds": CONFIRMED_SUCCESSOR_SEEDS,
            "replacement_mapping": confirmed_mapping,
            "replica_protocols": confirmed_protocols,
            "full_comparison_packet_state": {
                "state": "seed69_seed70_bound_seed71_inert_pending_seed68_retirement",
                "rendered_successor_seeds": [62, 63, 68, 69, 70],
                "retired_rendered_successor_seeds": [64, 65, 66, 67],
                "prepared_not_created_successor_seeds": [69, 70],
                "provisional_invalid_seed_awaiting_retirement": [68],
                "full_roster_launch_ready": False,
                "final_aggregation_allowed": False,
                "score_unseal_allowed": False,
                "requires_fresh_budget_refresh": True,
                "requires_fresh_destination_absence": True,
                "requires_fresh_live_parity": True,
                "requires_versioned_successor_receipt": True,
            },
            "confirmed_late_replacements": [
                {
                    "superseded_replacement_seed": 66,
                    "successor_seed": 69,
                    "status": "confirmed_whole_pair_retirement",
                    "retirement_evidence": evidence["seed66_retirement"],
                },
                {
                    "superseded_replacement_seed": 67,
                    "successor_seed": 70,
                    "status": "confirmed_whole_pair_retirement",
                    "retirement_evidence": evidence["seed67_retirement"],
                },
            ],
            "conditional_provisional_replacements": [
                {
                    "superseded_replacement_seed": 68,
                    "successor_seed": 71,
                    "status": "inactive_pending_root_approved_whole_pair_retirement",
                    "failure_snapshot": evidence["seed68_snapshot"],
                    "preparation_hold": evidence["seed71_hold"],
                    "activation_mode": "new_versioned_definition_only",
                    "auto_activation_allowed": False,
                    "active_in_included_seeds": False,
                    "active_in_replacement_mapping": False,
                    "active_in_replica_protocols": False,
                    "affects_aggregation": False,
                    "create_authorized": False,
                }
            ],
            "conditional_atomic_target": {
                "target_included_seeds": TARGET_SEEDS,
                "target_replacement_mapping": target_mapping,
                "target_replica_protocols": target_protocols,
                "partial_activation_allowed": False,
                "activation_status": "HOLD",
            },
            "prepared_packet_bindings": {
                str(seed): protocols[seed]["packet_bindings"] for seed in protocols
            },
            "activation": {
                "status": "HOLD",
                "confirmed_seed69_seed70_definition_recorded": True,
                "aggregation_allowed": False,
                "create_authorized": False,
                "score_unseal_allowed": False,
                "seed68_to_seed71_active": False,
                "required_to_finalize_seed71": [
                    "root-approved score-blind seed68 whole-pair retirement receipt",
                    "exact file and self digests for that receipt",
                    "new versioned active comparison definition and receipt",
                    "independent review of the finalized atomic successor",
                ],
            },
            "evidence_bindings": list(evidence.values()),
        }
    )
    definition.setdefault("superseded_replacements", []).extend(
        [
            {
                "invalid_original_seed": 53,
                "superseded_replacement_seed": 66,
                "successor_seed": 69,
                "whole_pair_excluded": True,
                "evidence_file_sha256s": [evidence["seed66_retirement"]["file_sha256"]],
                "evidence_receipt_sha256s": [evidence["seed66_retirement"]["sha256"]],
            },
            {
                "invalid_original_seed": 52,
                "superseded_replacement_seed": 67,
                "successor_seed": 70,
                "whole_pair_excluded": True,
                "evidence_file_sha256s": [evidence["seed67_retirement"]["file_sha256"]],
                "evidence_receipt_sha256s": [evidence["seed67_retirement"]["sha256"]],
            },
        ]
    )
    definition["sha256"] = _digest(definition)
    return definition


def prepare(paths: dict[str, Path], output: Path) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise FileExistsError("successor proposal output already exists")
    if not output.parent.is_dir():
        raise ValueError("successor proposal output parent does not exist")
    values = {label: _read_exact(paths[label], label) for label in INPUT_SPECS}
    _validate_predecessor(values["predecessor"])
    _validate_retirements(values["seed66_retirement"], values["seed67_retirement"])
    hold = values["seed71_hold"]
    if (
        hold.get("status") != "PREP_HOLD"
        or hold.get("decision", {}).get("create_authorized") is not False
        or hold.get("decision", {}).get("create_performed") is not False
        or hold.get("decision", {}).get("candidate_stop_or_mutation_authorized") is not False
        or hold.get("seed68_incident_binding", {}).get("base_terminal_pre_session_failure")
        is not True
    ):
        raise ValueError("seed68 failure or seed71 PREP_HOLD evidence differs")
    protocols = {
        69: _validate_preparation(values["seed69_preparation"], 69, values["seed69_current_main"]),
        70: _validate_preparation(values["seed70_preparation"], 70),
        71: _validate_preparation(values["seed71_preparation"], 71),
    }
    evidence = {label: _binding(paths[label], label) for label in INPUT_SPECS}
    definition = _build_definition(values["predecessor"], protocols, evidence)

    intent = {
        "schema": "cyber_qwen38_fleet_dev17_pass8_successor_intent_v9_proposal_v1",
        "canonical_github_main": CANONICAL_GITHUB_MAIN,
        "definition_sha256": definition["sha256"],
        "confirmed_successor_included_seeds": CONFIRMED_SUCCESSOR_SEEDS,
        "conditional_atomic_target_included_seeds": TARGET_SEEDS,
        "status": "HOLD",
        "partial_activation_allowed": False,
        "external_mutations_authorized": False,
        "evidence_bindings": list(evidence.values()),
    }
    intent["sha256"] = _digest(intent)

    temporary = Path(tempfile.mkdtemp(prefix=".q38-late-successor-", dir=output.parent))
    try:
        files = {
            "COMPARISON_DEFINITION_V9_PROPOSAL.json": definition,
            "SUCCESSOR_INTENT_V9_PROPOSAL.json": intent,
        }
        for name, value in files.items():
            path = temporary / name
            path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            path.chmod(0o600)
        receipt = {
            "schema": "cyber_qwen38_fleet_dev17_pass8_successor_receipt_v9_proposal_v1",
            "canonical_github_main": CANONICAL_GITHUB_MAIN,
            "outputs": {
                name: {
                    "file_sha256": _file_digest(temporary / name),
                    "sha256": value["sha256"],
                }
                for name, value in files.items()
            },
            "decision": {
                "gate": "HOLD",
                "reason": "seed68 to seed71 lacks root-approved whole-pair retirement evidence",
                "confirmed_seed69_seed70_definition_recorded": True,
                "conditional_seed71_target_recorded": True,
                "create_authorized": False,
                "aggregation_allowed": False,
                "score_unseal_allowed": False,
            },
            "execution": {
                "kubernetes_preview_performed": False,
                "kubernetes_mutation_performed": False,
                "database_mutation_performed": False,
                "provider_calls_performed": False,
                "model_generation_performed": False,
                "scoring_performed": False,
            },
            "privacy": {
                "score_blind": True,
                "scores_read_or_included": False,
                "prompts_responses_flags_rewards_or_trace_content_read_or_included": False,
                "private_task_cell_session_or_verifier_ids_included": False,
            },
        }
        receipt["sha256"] = _digest(receipt)
        receipt_path = temporary / "SUCCESSOR_RECEIPT_V9_PROPOSAL.json"
        receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        receipt_path.chmod(0o600)
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for label in INPUT_SPECS:
        parser.add_argument("--" + label.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {label: getattr(args, label) for label in INPUT_SPECS}
    print(json.dumps(prepare(paths, args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
