from __future__ import annotations

import copy
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from cyber_post_train import public_eval_import
from evals.fleet import final_pass8_aggregate as final
from scripts import render_qwen38_fleet_pass8_final_aggregate as renderer

ROOT = Path(__file__).resolve().parents[1]


class FakeSnapshot:
    def __init__(self, gate: final.GateSnapshot, scored: list[dict[str, Any]]) -> None:
        self.value = gate
        self.scored = scored
        self.score_reads = 0

    def gate(self) -> final.GateSnapshot:
        return self.value

    def scored_results(self) -> list[dict[str, Any]]:
        self.score_reads += 1
        return copy.deepcopy(self.scored)


def _signed(value: dict[str, Any], field: str = "sha256") -> dict[str, Any]:
    return {**value, field: final._digest(value)}  # noqa: SLF001


def _protocol(seed: int, *, replacement: bool) -> dict[str, Any]:
    protocol_sha256s = {
        46: "sha256:fc4c31caddf382f064bf4301f7ca88bcbf744207190386c491e055df23d30899",
        48: "sha256:6e8a15d372bda78978ea48c1889b7c5ef4a1f902e3dd85d534ede185cd5a646a",
        49: "sha256:29d56d78a5c2a2ea6f1763c8cc0fdda9bf7018088f71061986b92a27fdcd2315",
        50: "sha256:82f3f5fa3e41f854ef60a555a7bff2f373d5233cafa6802db18cb3347e09361c",
        54: "sha256:8355cf7a9682b96e2d83cdc5de5098f25954f3d21f85706fcdfe59c36073a319",
        55: "sha256:46775efd79f44e0c692be974ee7f48d270a43c6da25be90f60e56ffc20ce340d",
        56: "sha256:2792a3808fe69d14cb6cfe7a38c8860ba20a183f259505215a64eea2bdbdd086",
        57: "sha256:22a256cf7fb861e1720d6f80c2ea9b8bada730f1da6806f997fbd4e0dc9806b7",
    }
    suffix = "replacement-p1-v2" if replacement else "p1-v1"
    return {
        "seed": seed,
        "origin": "whole_pair_replacement" if replacement else "retained_original",
        "protocol_id": f"q38-dev17-s{seed}-base-t3k32s1000-{suffix}",
        "comparison_protocol_sha256": protocol_sha256s[seed],
    }


def _migration(tmp_path: Path, excluded: tuple[int, ...] = (47, 51, 52, 53)) -> Path:
    migration_root = tmp_path / "migration"
    migration_root.mkdir()
    mapping = [
        {"invalid_original_seed": seed, "replacement_seed": 54 + index}
        for index, seed in enumerate(excluded)
    ]
    included = sorted(
        [
            *(seed for seed in final.SOURCE_SEEDS if seed not in excluded),
            *(row["replacement_seed"] for row in mapping),
        ]
    )
    replacement_seeds = {row["replacement_seed"] for row in mapping}
    protocols = [_protocol(seed, replacement=seed in replacement_seeds) for seed in included]
    base = json.loads(renderer.BASE_CONFIG.read_text())
    comparison = _signed(
        {
            "schema": final.COMPARISON_DEFINITION_SCHEMA,
            "protocol_study_id": "q38-dev17-base-step1000-p8-v2",
            "predecessor_comparison_definition_sha256": (
                final.PREDECESSOR_COMPARISON_DEFINITION_SHA256
            ),
            "aggregation": "eight_predeclared_pass1_replicas_per_task_and_arm",
            "original_seeds": list(final.SOURCE_SEEDS),
            "excluded_original_seeds": list(excluded),
            "replacement_mapping": mapping,
            "included_seeds": included,
            "replica_protocols": protocols,
            "task_count": final.TASK_COUNT,
            "sessions_per_arm": final.TASK_COUNT * final.PASS_K,
            "total_sessions": final.TASK_COUNT * final.PASS_K * len(final.ARMS),
            "comparison_arms": list(final.ARMS),
            "models": {
                "base": {
                    "model_id": "qwen3.8-27b",
                    "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
                },
                "candidate": {
                    "model_id": "chris-q38-t3k32-s1000-v1",
                    "revision": (
                        "sha256:023c5f8b0559ba050f0d672a6bc27aabecec7d5837595f8ea5bc914446d26db5"
                    ),
                },
            },
            "task_selection_sha256": (
                "sha256:38ea6686afa068c19e69fea2493e01027fdf99f7e01b20376e107b1a0cfa0b68"
            ),
            "split_manifest_sha256": (
                "sha256:05b3a8dc90ca93adc9671942d75ecb54ff8d0951b0dd48a087e29ec1e641840c"
            ),
            "binding_roster_sha256": (
                "sha256:39ae49c2db322725800b15347d5da5d171e7e03b21dec5b570683414d70e83c5"
            ),
            "harness": base["harness"],
            "images": base["images"],
            "sampling_without_seed": {"temperature": 0.6, "top_p": 0.95},
            "pass_k_per_replica": 1,
            "retry_limit": 0,
            "training_data_eligible": False,
            "whole_replica_pairs_only": True,
            "cell_level_replacement_forbidden": True,
            "included_seed_status": "provisional_until_all_valid8_terminal_gates_pass",
            "later_invalid_seed_policy": (
                "create a versioned successor intent, definition, and receipt before score "
                "unseal; never edit this definition in place"
            ),
        }
    )
    assert comparison["sha256"] == final.FROZEN_COMPARISON_DEFINITION_SHA256
    definition_path = migration_root / "COMPARISON_DEFINITION.json"
    definition_path.write_text(json.dumps(comparison, sort_keys=True) + "\n")
    protocol_by_seed = {row["seed"]: row for row in protocols}
    migrations = []
    replacement_protocols = []
    replacement_arms = []
    for row in mapping:
        original = row["invalid_original_seed"]
        seed = row["replacement_seed"]
        excluded_arms = {}
        for arm_index, arm in enumerate(final.ARMS):
            excluded_arms[arm] = {
                "packet_file_sha256": "sha256:" + f"{original + arm_index + 2000:064x}",
                "evaluation_identity_sha256": ("sha256:" + f"{original + arm_index + 3000:064x}"),
                "evaluation_config_sha256": ("sha256:" + f"{original + arm_index + 4000:064x}"),
                "comparison_protocol_file_sha256": (
                    "sha256:" + f"{original + arm_index + 5000:064x}"
                ),
                "comparison_protocol_sha256": ("sha256:" + f"{original + arm_index + 6000:064x}"),
                "protocol_id": f"source-{original}",
                "job_name": f"source-{original}-{arm}",
                "config_map_name": f"source-{original}-{arm}-code",
                "output_root": f"/private/source/{original}/{arm}",
                "database": f"source_{original}_{arm}",
            }
        migrations.append(
            {
                **row,
                "reason_class": "mixed_infrastructure_invalid_replica",
                "evidence_receipt_sha256s": ["sha256:" + f"{original + 7000:064x}"],
                "excluded_source_arms": excluded_arms,
                "whole_pair_excluded": True,
            }
        )
        replacement_protocols.append(
            {
                **row,
                "protocol_id": protocol_by_seed[seed]["protocol_id"],
                "sha256": protocol_by_seed[seed]["comparison_protocol_sha256"],
                "file_sha256": "sha256:" + f"{seed + 8000:064x}",
            }
        )
        for arm_index, arm in enumerate(final.ARMS):
            replacement_arms.append(
                {
                    "arm_id": arm,
                    **row,
                    "packet_file_sha256": "sha256:" + f"{seed + arm_index + 9000:064x}",
                    "evaluation_identity_sha256": ("sha256:" + f"{seed + arm_index + 10000:064x}"),
                    "serving_proof_file_sha256": ("sha256:" + f"{seed + arm_index + 11000:064x}"),
                    "comparison_protocol_file_sha256": (
                        "sha256:" + f"{seed + arm_index + 12000:064x}"
                    ),
                    "comparison_protocol_sha256": protocol_by_seed[seed][
                        "comparison_protocol_sha256"
                    ],
                    "packet_path": f"seed{seed}/{arm}/LAUNCH_PACKET.json",
                }
            )
    receipt = _signed(
        {
            "schema": final.MIGRATION_SCHEMA,
            "source": {
                "protocol_study_id": "q38-dev17-seeds46to53-base-step1000-p8-v1",
                "preparation_receipt_sha256": "sha256:" + "1" * 64,
                "preparation_receipt_file_sha256": "sha256:" + "2" * 64,
                "seeds": list(final.SOURCE_SEEDS),
                "arm_count": len(final.SOURCE_SEEDS) * len(final.ARMS),
            },
            "migration_intent_sha256": "sha256:" + "3" * 64,
            "comparison_definition": comparison,
            "comparison_definition_file_sha256": final._file_digest(  # noqa: SLF001
                definition_path
            ),
            "mapping_rule": (
                "invalid original seeds sorted ascending map to the smallest unused seeds "
                "greater than 53, ascending"
            ),
            "migrations": migrations,
            "excluded_original_seeds": list(excluded),
            "included_seeds": included,
            "replacement_protocols": replacement_protocols,
            "replacement_arms": replacement_arms,
            "retirement_evidence": {
                "sha256": "sha256:" + "4" * 64,
                "file_sha256": "sha256:" + "5" * 64,
                "targets": 2,
                "model_rollouts": 0,
                "outputs_or_databases_deleted": False,
            },
            "scientific_identity": {
                "task_count_per_arm": final.TASK_COUNT,
                "comparison_arms": list(final.ARMS),
                "pass_k": 1,
                "retry_limit": 0,
                "same_task_versions_models_harness_budgets_and_sampling_recipe": True,
                "whole_replica_pairs_only": True,
                "cell_level_replacement_forbidden": True,
                "seed_reuse_forbidden": True,
                "later_invalid_seed_requires_versioned_successor_before_score_unseal": True,
            },
            "capacity": {
                "new_replacement_rollouts": len(mapping) * len(final.ARMS) * final.TASK_COUNT,
                "final_comparison_rollouts": final.TASK_COUNT * final.PASS_K * len(final.ARMS),
                "original_base_rollouts": final.TASK_COUNT * final.PASS_K,
                "original_candidate_started_seeds": [46, 47, 48, 49, 50, 51],
                "original_candidate_started_seed_rollouts": final.TASK_COUNT * 6,
                "original_candidate_retired_before_start_seeds": [52, 53],
                "original_candidate_retired_before_start_rollouts": 0,
                "scoring_or_metadata_cpu_model_rollouts": 0,
                "cumulative_model_rollouts_consumed_or_planned_today": (
                    final.TASK_COUNT * final.PASS_K
                    + final.TASK_COUNT * 6
                    + len(mapping) * len(final.ARMS) * final.TASK_COUNT
                ),
                "daily_rollout_cap": 500,
                "within_daily_cap": True,
            },
            "selection": {
                "selection_sha256": (
                    "sha256:38ea6686afa068c19e69fea2493e01027fdf99f7e01b20376e107b1a0cfa0b68"
                ),
                "split_sha256": (
                    "sha256:05b3a8dc90ca93adc9671942d75ecb54ff8d0951b0dd48a087e29ec1e641840c"
                ),
                "corpus_dev_windows": 17,
                "binding_roster_sha256": (
                    "sha256:39ae49c2db322725800b15347d5da5d171e7e03b21dec5b570683414d70e83c5"
                ),
            },
            "live_parity_file_sha256": "sha256:" + "6" * 64,
            "live_parity_receipt_sha256": "sha256:" + "7" * 64,
            "privacy": {
                "score_values_read": False,
                "prompts_responses_flags_rewards_or_trace_content_read": False,
                "infrastructure_reason_classes_only": True,
            },
            "external_mutations": 0,
            "launch_performed": False,
        }
    )
    receipt_path = migration_root / "MIGRATION_RECEIPT.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
    return receipt_path


def _plan(tmp_path: Path) -> dict[str, Any]:
    migration = _migration(tmp_path)
    plan = final.build_current_study_plan(
        task_set_path=renderer.TASK_SET,
        roster_path=renderer.ROSTER,
        base_config_path=renderer.BASE_CONFIG,
        migration_receipt_path=migration,
    )
    plan = copy.deepcopy(plan)
    for replica in plan["replicas"]:
        root = tmp_path / f"s{replica['seed']}-{replica['arm']}"
        root.mkdir()
        replica["output_root"] = str(root)
        replica["terminal_receipt_path"] = str(root / "TERMINAL_OBSERVATION.json")
    plan["private_output_root"] = str(tmp_path / "final")
    plan.pop("sha256")
    return _signed(plan)


def _evaluation(plan: dict[str, Any], replica: dict[str, Any]) -> dict[str, Any]:
    arm = plan["arms"][replica["arm"]]
    value = {
        "schema": "cyber_fleet_eval_v1",
        "campaign_id": replica["experiment_id"],
        "tasks": sorted(plan["tasks"], key=lambda row: row["task_version_id"]),
        "models": {arm["model_id"]: arm["model"]},
        "routes": {arm["serving_block"]: arm["route"]},
        "treatment": plan["harness"],
        "images": plan["images"],
        "sampling": {**plan["sampling"], "seed": replica["seed"]},
        "pass_k": 1,
        "automatic_retry": False,
        "max_reviewed_infrastructure_retries": 0,
        "training_data_eligible": False,
        "runtime_files": {
            name: f"{index + 1:064x}" for index, name in enumerate(sorted(final.RUNTIME_FILES))
        },
    }
    value["sha256"] = final._plain_digest(value)  # noqa: SLF001
    return value


def _local_record(cell_id: str, session_id: str, score: float, index: int) -> dict[str, Any]:
    digest = f"{index + 1:064x}"[-64:]
    value = {
        "execution_id": "sha256:" + f"{index + 300:064x}"[-64:],
        "cell_id": cell_id,
        "execution_generation": 1,
        "run_id": f"run-{index}",
        "session_id": session_id,
        "verifier_execution_id": f"verifier-{index}",
        "score": score,
        "config_sha256": digest,
        "artifact_directory": f"attempts/{index}",
        "trace_path": "trace.json",
        "trace_sha256": digest,
        "result_path": "result.json",
        "result_sha256": digest,
        "reward_path": "reward-result.json",
        "reward_sha256": digest,
        "session_ingest_path": "session-ingest.json",
        "session_ingest_sha256": digest,
        "cleanup_path": "cleanup.json",
        "cleanup_sha256": digest,
        "session_ingest_status": "completed",
        "agent_exit_code": 0,
        "agent_termination": "completed",
        "elapsed_seconds": 10.0,
    }
    return {**value, "record_sha256": final._plain_digest(value)}  # noqa: SLF001


def _study(tmp_path: Path) -> tuple[dict[str, Any], dict[tuple[int, str], FakeSnapshot]]:
    plan = _plan(tmp_path)
    snapshots: dict[tuple[int, str], FakeSnapshot] = {}
    for replica in plan["replicas"]:
        evaluation = _evaluation(plan, replica)
        root = Path(replica["output_root"])
        (root / "EVAL.json").write_text(json.dumps(evaluation, sort_keys=True) + "\n")
        arm = plan["arms"][replica["arm"]]
        stored_plan = []
        for task in sorted(plan["tasks"], key=lambda row: row["task_version_id"]):
            row = {
                "experiment_id": replica["experiment_id"],
                "task_key": task["task_key"],
                "task_version_id": task["task_version_id"],
                "model_id": arm["model_id"],
                "model_revision": arm["model"]["revision"],
                "serving_block": arm["serving_block"],
                "endpoint_model_id": arm["route"]["served_id"],
                "harness_id": "protocol-" + evaluation["sha256"],
                "attempt": 1,
                "max_retries": 0,
            }
            stored_plan.append({"cell_id": final._cell_id(row), **row})  # noqa: SLF001
        plan_digest = final._plain_digest(stored_plan)  # noqa: SLF001
        terminal = _signed(
            {
                "schema": final.TERMINAL_SCHEMA,
                "evaluation_identity_sha256": replica["evaluation_identity_sha256"],
                "comparison_protocol_sha256": replica["comparison_protocol_sha256"],
                "protocol_id": replica["protocol_id"],
                "arm_id": replica["arm"],
                "job": {
                    "name": replica["job_name"],
                    "uid": "11111111-1111-4111-8111-111111111111",
                    "terminal_condition": "Complete",
                    "succeeded": 1,
                    "failed": 0,
                },
                "config_map": {
                    "name": replica["config_map_name"],
                    "uid": "22222222-2222-4222-8222-222222222222",
                },
                "workloads": [],
                "pods": [],
                "database": {
                    "name": replica["database"],
                    "summary": {"plan_sha256": plan_digest},
                },
                "output_root": {"path": replica["output_root"], "exists": True},
                "decision": {
                    "capability_result_status": "not_interpreted",
                    "score_blind_reconciliation_required": False,
                    "unresolved_cells": 0,
                    "rollout_retry_performed": False,
                    "score_read_or_generated": False,
                },
                "privacy": {
                    "prompts_responses_flags_rewards_or_trace_content_included": False,
                    "score_values_included": False,
                    "credentials_included": False,
                },
            }
        )
        Path(replica["terminal_receipt_path"]).write_text(
            json.dumps(terminal, sort_keys=True) + "\n"
        )
        cells = []
        local = []
        events = []
        planned_by_task = {row["task_version_id"]: row for row in stored_plan}
        for index, task in enumerate(plan["tasks"]):
            planned = planned_by_task[task["task_version_id"]]
            cell_id = planned["cell_id"]
            session_id = f"private-session-{replica['seed']}-{replica['arm']}-{index}"
            receipt = f"{index + 500:064x}"[-64:]
            cells.append(
                {
                    "cell_id": cell_id,
                    **planned,
                    "state": "accepted",
                    "session_id": session_id,
                    "lease_expires_at": None,
                    "retry_count": 0,
                    "max_retries": 0,
                    "result_class": "valid",
                    "receipt_digest": receipt,
                    "failure_code": None,
                    "reconciliation_digest": None,
                }
            )
            record = _local_record(
                cell_id,
                session_id,
                float(index == 0 and replica["arm"] == "candidate"),
                index,
            )
            local.append(record)
            events.append(
                {
                    "cell_id": cell_id,
                    "event": "accepted",
                    "to_state": "accepted",
                    "detail_json": json.dumps({"receipt_digest": receipt}),
                }
            )
        gate = final.GateSnapshot(
            plan_sha256=plan_digest,
            cells=cells,
            local_metadata=[
                {key: value for key, value in row.items() if key != "score"} for row in local
            ],
            events=events,
            reconciliations=[],
        )
        snapshots[(replica["seed"], replica["arm"])] = FakeSnapshot(gate, local)
    return plan, snapshots


def test_final_gate_opens_scores_only_after_all_replicas_and_emits_safe_public_input(
    tmp_path: Path,
) -> None:
    plan, snapshots = _study(tmp_path)

    result = final.finalize(plan, snapshots, output_root=Path(plan["private_output_root"]))

    assert result["status"] == "final"
    assert result["valid_outcomes_per_task_arm"] == 8
    assert all(snapshot.score_reads == 1 for snapshot in snapshots.values())
    public_path = Path(plan["private_output_root"]) / "SANITIZED_AGGREGATE.json"
    public = json.loads(public_path.read_text())
    validated = public_eval_import._validate(public, final._file_digest(public_path))  # noqa: SLF001
    assert validated["summary"]["paired_valid_tasks"] == 17
    assert all(
        row[arm]["valid_attempts"] == 8 and row[arm]["infrastructure_invalid_attempts"] == 0
        for row in public["task_rows"]
        for arm in final.ARMS
    )
    public_text = public_path.read_text().lower()
    for forbidden in (
        "task_key",
        "task_version_id",
        "session_id",
        "cell_id",
        "prompt",
        "trace",
        "verifier",
        "private-session",
    ):
        assert forbidden not in public_text
    assert oct(public_path.stat().st_mode & 0o777) == "0o600"


@pytest.mark.parametrize(
    ("state", "result_class"),
    (
        ("running", None),
        ("retry_review", "infrastructure_invalid"),
        ("terminal", "infrastructure_invalid"),
    ),
)
def test_one_unready_cell_prevents_every_score_read(
    tmp_path: Path, state: str, result_class: str | None
) -> None:
    plan, snapshots = _study(tmp_path)
    first = snapshots[(46, "base")]
    first.value.cells[0]["state"] = state
    first.value.cells[0]["result_class"] = result_class

    with pytest.raises(final.FinalAggregateError, match="non-authoritative"):
        final.finalize(plan, snapshots, output_root=Path(plan["private_output_root"]))

    assert all(snapshot.score_reads == 0 for snapshot in snapshots.values())
    assert not Path(plan["private_output_root"]).exists()


def test_protocol_v2_uses_only_eight_complete_whole_replica_pairs(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    included = set(plan["included_seeds"])
    excluded = set(plan["excluded_original_seeds"])
    replacements = {row["replacement_seed"] for row in plan["replacement_mapping"]}

    assert len(included) == final.PASS_K
    assert included.isdisjoint(excluded)
    assert replacements <= included
    assert {(row["seed"], row["arm"]) for row in plan["replicas"]} == {
        (seed, arm) for seed in included for arm in final.ARMS
    }
    assert all(
        row["origin"] == "whole_pair_replacement"
        for row in plan["replicas"]
        if row["seed"] in replacements
    )


def test_plan_cannot_mix_an_excluded_original_with_its_replacement(tmp_path: Path) -> None:
    plan, snapshots = _study(tmp_path)
    replacement = plan["replacement_mapping"][0]
    row = next(
        item
        for item in plan["replicas"]
        if item["seed"] == replacement["replacement_seed"] and item["arm"] == "candidate"
    )
    row["seed"] = replacement["invalid_original_seed"]
    plan.pop("sha256")
    plan.update(sha256=final._digest(plan))  # noqa: SLF001

    with pytest.raises(final.FinalAggregateError, match="eight included pairs"):
        final.finalize(plan, snapshots, output_root=Path(plan["private_output_root"]))

    assert all(snapshot.score_reads == 0 for snapshot in snapshots.values())


def test_migration_receipt_tamper_is_rejected_before_plan_build(tmp_path: Path) -> None:
    receipt_path = _migration(tmp_path)
    receipt = json.loads(receipt_path.read_text())
    receipt["included_seeds"][0] = 47
    receipt_path.write_text(json.dumps(receipt) + "\n")

    with pytest.raises(final.FinalAggregateError, match="self digest"):
        final.build_current_study_plan(
            task_set_path=renderer.TASK_SET,
            roster_path=renderer.ROSTER,
            base_config_path=renderer.BASE_CONFIG,
            migration_receipt_path=receipt_path,
        )


def test_resigned_alternate_exclusion_roster_is_rejected(tmp_path: Path) -> None:
    receipt_path = _migration(tmp_path)
    receipt = json.loads(receipt_path.read_text())
    comparison = receipt["comparison_definition"]
    comparison.pop("sha256")
    comparison["excluded_original_seeds"] = [47, 52, 53]
    comparison["sha256"] = final._digest(comparison)  # noqa: SLF001
    definition_path = receipt_path.parent / "COMPARISON_DEFINITION.json"
    definition_path.write_text(json.dumps(comparison, sort_keys=True) + "\n")
    receipt.pop("sha256")
    receipt["comparison_definition"] = comparison
    receipt["comparison_definition_file_sha256"] = final._file_digest(  # noqa: SLF001
        definition_path
    )
    receipt["sha256"] = final._digest(receipt)  # noqa: SLF001
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")

    with pytest.raises(final.FinalAggregateError, match="excluded original seeds differ"):
        final.build_current_study_plan(
            task_set_path=renderer.TASK_SET,
            roster_path=renderer.ROSTER,
            base_config_path=renderer.BASE_CONFIG,
            migration_receipt_path=receipt_path,
        )


def _rewrite_terminal(replica: dict[str, Any], mutate: Any) -> None:
    path = Path(replica["terminal_receipt_path"])
    receipt = json.loads(path.read_text())
    receipt.pop("sha256")
    mutate(receipt)
    path.write_text(json.dumps(_signed(receipt), sort_keys=True) + "\n")


def test_wrong_protocol_prevents_every_score_read(tmp_path: Path) -> None:
    plan, snapshots = _study(tmp_path)
    replica = next(row for row in plan["replicas"] if row["seed"] == 46 and row["arm"] == "base")
    _rewrite_terminal(replica, lambda receipt: receipt.update(protocol_id="wrong"))

    with pytest.raises(final.FinalAggregateError, match="terminal receipt identity"):
        final.finalize(plan, snapshots, output_root=Path(plan["private_output_root"]))

    assert all(snapshot.score_reads == 0 for snapshot in snapshots.values())


def test_live_ledger_plan_must_recompute_from_exact_cells(tmp_path: Path) -> None:
    plan, snapshots = _study(tmp_path)
    identity = (46, "base")
    snapshot = snapshots[identity]
    wrong = "f" * 64
    snapshot.value = final.GateSnapshot(
        plan_sha256=wrong,
        cells=snapshot.value.cells,
        local_metadata=snapshot.value.local_metadata,
        events=snapshot.value.events,
        reconciliations=snapshot.value.reconciliations,
    )
    replica = next(row for row in plan["replicas"] if (row["seed"], row["arm"]) == identity)
    _rewrite_terminal(
        replica,
        lambda receipt: receipt["database"]["summary"].update(plan_sha256=wrong),
    )

    with pytest.raises(final.FinalAggregateError, match="immutable ledger plan"):
        final.finalize(plan, snapshots, output_root=Path(plan["private_output_root"]))

    assert all(value.score_reads == 0 for value in snapshots.values())


@pytest.mark.parametrize(
    "drift",
    ("evaluation_identity", "job_uid", "config_map_name", "config_map_uid", "privacy"),
)
def test_terminal_identity_drift_prevents_every_score_read(tmp_path: Path, drift: str) -> None:
    plan, snapshots = _study(tmp_path)
    replica = next(row for row in plan["replicas"] if row["seed"] == 46 and row["arm"] == "base")

    def mutate(receipt: dict[str, Any]) -> None:
        if drift == "evaluation_identity":
            receipt["evaluation_identity_sha256"] = "sha256:" + "0" * 64
        elif drift == "job_uid":
            receipt["job"]["uid"] = "not-a-uid"
        elif drift == "config_map_name":
            receipt["config_map"]["name"] = "wrong-config-map"
        elif drift == "config_map_uid":
            receipt["config_map"]["uid"] = "not-a-uid"
        else:
            receipt["privacy"]["prompts_responses_flags_rewards_or_trace_content_included"] = True

    _rewrite_terminal(replica, mutate)

    with pytest.raises(final.FinalAggregateError, match="terminal receipt identity"):
        final.finalize(plan, snapshots, output_root=Path(plan["private_output_root"]))

    assert all(value.score_reads == 0 for value in snapshots.values())


def test_scored_query_cannot_change_score_blind_metadata(tmp_path: Path) -> None:
    plan, snapshots = _study(tmp_path)
    snapshot = snapshots[(46, "base")]
    snapshot.scored[0]["trace_sha256"] = "a" * 64

    with pytest.raises(final.FinalAggregateError, match="score-blind gate"):
        final.finalize(plan, snapshots, output_root=Path(plan["private_output_root"]))

    assert snapshot.score_reads == 1
    assert not Path(plan["private_output_root"]).exists()


def _add_reconciliation(
    plan: dict[str, Any], snapshots: dict[tuple[int, str], FakeSnapshot]
) -> dict[str, Any]:
    replica = next(row for row in plan["replicas"] if row["seed"] == 46 and row["arm"] == "base")
    evaluation = json.loads((Path(replica["output_root"]) / "EVAL.json").read_text())
    terminal = json.loads(Path(replica["terminal_receipt_path"]).read_text())
    snapshot = snapshots[(46, "base")]
    cell = snapshot.value.cells[0]
    intent = "9" * 64
    cell["reconciliation_digest"] = intent
    snapshot.value.events[0] = {
        "cell_id": cell["cell_id"],
        "event": "stored_scored_session_reconciled",
        "to_state": "accepted",
        "detail_json": json.dumps(
            {
                "reviewed_intent_sha256": intent,
                "cell_receipt_sha256": cell["receipt_digest"],
                "source_job_terminal_receipt_sha256": terminal["sha256"],
                "action": "accept_existing_scored_session",
            }
        ),
    }
    evidence = _signed(
        {
            "schema_version": "fleet-stored-session-reconciliation-v2",
            "reviewed_intent_sha256": intent,
            "evaluation_plan_sha256": evaluation["sha256"],
            "source_job_uid_sha256": (
                "sha256:" + hashlib.sha256(terminal["job"]["uid"].encode()).hexdigest()
            ),
            "source_job_terminal_receipt_sha256": terminal["sha256"],
            "selected_cell_count": 1,
            "prior_retry_review_count": 1,
            "accepted_existing_completed_session_count": 1,
            "source_agent_exit_code": 1,
            "source_agent_termination": "process_error",
            "source_failure_code_sha256": "sha256:" + "f" * 64,
            "action": "accept_existing_scored_session",
            "model_generation_performed": False,
            "scoring_call_performed": False,
            "score_values_included": False,
            "prompt_response_flag_reward_or_trace_content_included": False,
            "cell_task_session_or_trace_identifiers_included": False,
        },
        "receipt_sha256",
    )
    snapshot.value.reconciliations.append(evidence)
    return evidence


def test_reviewed_stored_session_needs_matching_private_reconciliation(tmp_path: Path) -> None:
    plan, snapshots = _study(tmp_path)
    _add_reconciliation(plan, snapshots)

    final.finalize(plan, snapshots, output_root=Path(plan["private_output_root"]))

    assert Path(plan["private_output_root"], "FINAL.json").is_file()


def test_legacy_reconciliation_receipt_is_fully_bound_before_score_open(tmp_path: Path) -> None:
    plan, snapshots = _study(tmp_path)
    evidence = _add_reconciliation(plan, snapshots)
    evidence.pop("receipt_sha256")
    evidence["schema_version"] = "fleet-stored-session-reconciliation-v1"
    evidence["prior_stale_active_count"] = 0
    for field in (
        "source_agent_exit_code",
        "source_agent_termination",
        "source_failure_code_sha256",
        "action",
    ):
        evidence.pop(field)
    evidence["receipt_sha256"] = final._digest(evidence)  # noqa: SLF001
    event = snapshots[(46, "base")].value.events[0]
    event["event"] = "stored_session_reconciled"
    detail = json.loads(event["detail_json"])
    detail.pop("action")
    event["detail_json"] = json.dumps(detail)

    final.finalize(plan, snapshots, output_root=Path(plan["private_output_root"]))

    assert Path(plan["private_output_root"], "FINAL.json").is_file()


def _as_subset_receipt(evidence: dict[str, Any]) -> None:
    evidence.pop("receipt_sha256")
    evidence.update(
        source_total_cell_count=17,
        prior_arm_state_counts={
            "pending": 0,
            "claimed": 0,
            "running": 0,
            "grading": 0,
            "accepted": 16,
            "retry_review": 1,
            "terminal": 0,
        },
        post_arm_state_counts={
            "pending": 0,
            "claimed": 0,
            "running": 0,
            "grading": 0,
            "accepted": 17,
            "retry_review": 0,
            "terminal": 0,
        },
        nonselected_cell_count=16,
        nonselected_cells_preserved=True,
    )
    evidence["receipt_sha256"] = final._digest(evidence)  # noqa: SLF001


def test_subset_reconciliation_state_transition_is_fully_bound(tmp_path: Path) -> None:
    plan, snapshots = _study(tmp_path)
    evidence = _add_reconciliation(plan, snapshots)
    _as_subset_receipt(evidence)

    final.finalize(plan, snapshots, output_root=Path(plan["private_output_root"]))

    assert Path(plan["private_output_root"], "FINAL.json").is_file()


@pytest.mark.parametrize("drift", ("total", "states", "local_counts"))
def test_subset_reconciliation_count_drift_prevents_score_open(tmp_path: Path, drift: str) -> None:
    plan, snapshots = _study(tmp_path)
    evidence = _add_reconciliation(plan, snapshots)
    _as_subset_receipt(evidence)
    evidence.pop("receipt_sha256")
    if drift == "total":
        evidence["source_total_cell_count"] = 18
    elif drift == "states":
        evidence["post_arm_state_counts"]["accepted"] = 16
        evidence["post_arm_state_counts"]["retry_review"] = 1
    else:
        evidence.update(
            source_local_result_count=15,
            missing_local_result_count=1,
            missing_local_results_are_unselected=True,
            selected_cells_have_local_results=True,
            missing_local_result_cells_preserved=True,
            missing_local_result_failure_code_sha256="sha256:" + "a" * 64,
        )
    evidence["receipt_sha256"] = final._digest(evidence)  # noqa: SLF001

    with pytest.raises(final.FinalAggregateError, match="reconciliation receipt is invalid"):
        final.finalize(plan, snapshots, output_root=Path(plan["private_output_root"]))

    assert all(value.score_reads == 0 for value in snapshots.values())


@pytest.mark.parametrize("drift", ("plan", "source", "action", "count"))
def test_reconciliation_identity_drift_prevents_score_open(tmp_path: Path, drift: str) -> None:
    plan, snapshots = _study(tmp_path)
    evidence = _add_reconciliation(plan, snapshots)
    evidence.pop("receipt_sha256")
    if drift == "plan":
        evidence["evaluation_plan_sha256"] = "sha256:" + "0" * 64
    elif drift == "source":
        evidence["source_job_terminal_receipt_sha256"] = "sha256:" + "0" * 64
    elif drift == "action":
        evidence["action"] = "generate_more_model_output"
    else:
        evidence["selected_cell_count"] = 2
        evidence["accepted_existing_completed_session_count"] = 2
    evidence["receipt_sha256"] = final._digest(evidence)  # noqa: SLF001

    with pytest.raises(final.FinalAggregateError, match="reconciliation receipt is invalid"):
        final.finalize(plan, snapshots, output_root=Path(plan["private_output_root"]))

    assert all(value.score_reads == 0 for value in snapshots.values())


def test_renderer_is_cpu_only_c1_alert_suppressed_and_module_bound(tmp_path: Path) -> None:
    root = tmp_path / "render"

    receipt = renderer.render(output=root, migration_receipt=_migration(tmp_path))

    bundle = yaml.safe_load((root / "final-aggregate.yaml").read_text())
    config_map, job = bundle["items"]
    assert receipt["external_mutations"] == 0
    assert receipt["launch_performed"] is False
    assert receipt["two_server_previews_required_before_create"] is True
    assert job["metadata"]["annotations"] == {
        "fleet.ai/failure-alerts": "off",
        "cyber-post-train.fleet.ai/create-once": "true",
    }
    assert job["spec"]["template"]["spec"]["priorityClassName"] == "c1"
    assert "nvidia.com/gpu" not in json.dumps(job)
    files = json.loads(gzip.decompress(base64_decode(config_map["binaryData"]["bundle.json.gz"])))
    assert "aggregate.py" in files and "study.json" in files and "run.py" in files
    assert receipt["source_files"]["aggregate.py"] == (
        "sha256:" + hashlib.sha256(files["aggregate.py"].encode()).hexdigest()
    )


def base64_decode(value: str) -> bytes:
    import base64

    return base64.b64decode(value, validate=True)


def _server_preview(bundle: dict[str, Any], uid: str) -> dict[str, Any]:
    value = copy.deepcopy(bundle)
    for item in value["items"]:
        item["metadata"].update(
            uid=uid,
            creationTimestamp="2026-09-23T00:00:00Z",
            resourceVersion="1",
        )
        if item["kind"] == "Job":
            item["status"] = {"active": 0}
            item["spec"]["selector"] = {"matchLabels": {"controller-uid": uid}}
            item["spec"]["template"]["metadata"]["labels"].update(
                {
                    "controller-uid": uid,
                    "batch.kubernetes.io/controller-uid": uid,
                }
            )
    return value


def test_two_preview_validator_normalizes_only_server_job_identity(tmp_path: Path) -> None:
    root = tmp_path / "render"
    migration = _migration(tmp_path)
    renderer.render(output=root, migration_receipt=migration)
    bundle = yaml.safe_load((root / "final-aggregate.yaml").read_text())
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(_server_preview(bundle, "11111111-1111-4111-8111-111111111111")))
    second.write_text(json.dumps(_server_preview(bundle, "22222222-2222-4222-8222-222222222222")))

    receipt = renderer.validate_previews(
        render_root=root,
        migration_receipt=migration,
        first=first,
        second=second,
        output=tmp_path / "previews.json",
    )

    assert receipt["root_failure_alerts"] == "off"
    assert receipt["priority_class"] == "c1"
    assert receipt["gpu_requests"] == 0
    assert receipt["create_performed"] is False


def test_two_preview_validator_rejects_render_receipt_drift(tmp_path: Path) -> None:
    root = tmp_path / "render"
    migration = _migration(tmp_path)
    renderer.render(output=root, migration_receipt=migration)
    bundle = yaml.safe_load((root / "final-aggregate.yaml").read_text())
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(_server_preview(bundle, "11111111-1111-4111-8111-111111111111")))
    second.write_text(json.dumps(_server_preview(bundle, "22222222-2222-4222-8222-222222222222")))
    receipt = json.loads((root / "RENDER.json").read_text())
    receipt["gpu_requests"] = 1
    (root / "RENDER.json").write_text(json.dumps(receipt))

    with pytest.raises(renderer.RenderError, match="self digest"):
        renderer.validate_previews(
            render_root=root,
            migration_receipt=migration,
            first=first,
            second=second,
            output=tmp_path / "previews.json",
        )


def test_two_preview_validator_rejects_admission_added_gpu(tmp_path: Path) -> None:
    root = tmp_path / "render"
    migration = _migration(tmp_path)
    renderer.render(output=root, migration_receipt=migration)
    bundle = yaml.safe_load((root / "final-aggregate.yaml").read_text())
    first_value = _server_preview(bundle, "11111111-1111-4111-8111-111111111111")
    second_value = _server_preview(bundle, "22222222-2222-4222-8222-222222222222")
    for value in (first_value, second_value):
        job = next(item for item in value["items"] if item["kind"] == "Job")
        job["spec"]["template"]["spec"]["initContainers"] = [
            {
                "name": "admission-added",
                "image": "example.invalid/image@sha256:" + "0" * 64,
                "resources": {"limits": {"nvidia.com/mig-1g.10gb": 1}},
            }
        ]
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(first_value))
    second.write_text(json.dumps(second_value))

    with pytest.raises(renderer.RenderError, match="unexpectedly requests an accelerator"):
        renderer.validate_previews(
            render_root=root,
            migration_receipt=migration,
            first=first,
            second=second,
            output=tmp_path / "previews.json",
        )


def test_two_preview_validator_rejects_one_preview_replayed_twice(tmp_path: Path) -> None:
    root = tmp_path / "render"
    migration = _migration(tmp_path)
    renderer.render(output=root, migration_receipt=migration)
    bundle = yaml.safe_load((root / "final-aggregate.yaml").read_text())
    preview = tmp_path / "preview.json"
    preview.write_text(json.dumps(_server_preview(bundle, "11111111-1111-4111-8111-111111111111")))

    with pytest.raises(renderer.RenderError, match="distinct regular files"):
        renderer.validate_previews(
            render_root=root,
            migration_receipt=migration,
            first=preview,
            second=preview,
            output=tmp_path / "previews.json",
        )


@pytest.mark.parametrize(
    "drift",
    (
        "host_network",
        "privileged",
        "env_from",
        "parallelism",
        "manual_selector",
        "termination_message",
        "priority",
    ),
)
def test_two_preview_validator_rejects_unsafe_admission_fields(tmp_path: Path, drift: str) -> None:
    root = tmp_path / "render"
    migration = _migration(tmp_path)
    renderer.render(output=root, migration_receipt=migration)
    bundle = yaml.safe_load((root / "final-aggregate.yaml").read_text())
    previews = [
        _server_preview(bundle, "11111111-1111-4111-8111-111111111111"),
        _server_preview(bundle, "22222222-2222-4222-8222-222222222222"),
    ]
    for value in previews:
        job = next(item for item in value["items"] if item["kind"] == "Job")
        pod = job["spec"]["template"]["spec"]
        if drift == "host_network":
            pod["hostNetwork"] = True
        elif drift == "privileged":
            pod["containers"][0]["securityContext"] = {"privileged": True}
        elif drift == "env_from":
            pod["containers"][0]["envFrom"] = [{"secretRef": {"name": "unreviewed-secret"}}]
        elif drift == "parallelism":
            job["spec"]["parallelism"] = 2
        elif drift == "manual_selector":
            job["spec"]["manualSelector"] = True
            job["spec"]["selector"] = {"matchLabels": {"attacker.example/selected": "true"}}
        elif drift == "termination_message":
            pod["containers"][0]["terminationMessagePath"] = (
                "/mnt/sfs/private/PRIVATE_SCORED_OUTCOME_INDEX.json"
            )
        else:
            pod["priority"] = 1_000_000
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(previews[0]))
    second.write_text(json.dumps(previews[1]))

    with pytest.raises(renderer.RenderError):
        renderer.validate_previews(
            render_root=root,
            migration_receipt=migration,
            first=first,
            second=second,
            output=tmp_path / "previews.json",
        )


def test_preview_validator_rebuilds_render_policy_after_resigning_tamper(
    tmp_path: Path,
) -> None:
    root = tmp_path / "render"
    migration = _migration(tmp_path)
    renderer.render(output=root, migration_receipt=migration)
    bundle_path = root / "final-aggregate.yaml"
    bundle = yaml.safe_load(bundle_path.read_text())
    job = next(item for item in bundle["items"] if item["kind"] == "Job")
    job["spec"]["template"]["spec"]["priorityClassName"] = "c0"
    bundle_path.write_text(yaml.safe_dump(bundle, sort_keys=False))
    receipt_path = root / "RENDER.json"
    receipt = json.loads(receipt_path.read_text())
    receipt.pop("sha256")
    receipt["rendered_bundle_file_sha256"] = final._file_digest(bundle_path)  # noqa: SLF001
    receipt["sha256"] = renderer._canonical_digest(receipt)  # noqa: SLF001
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(_server_preview(bundle, "11111111-1111-4111-8111-111111111111")))
    second.write_text(json.dumps(_server_preview(bundle, "22222222-2222-4222-8222-222222222222")))

    with pytest.raises(renderer.RenderError, match="Kubernetes objects differ"):
        renderer.validate_previews(
            render_root=root,
            migration_receipt=migration,
            first=first,
            second=second,
            output=tmp_path / "previews.json",
        )


def test_preview_validator_rejects_resigned_executable_bundle_tamper(tmp_path: Path) -> None:
    root = tmp_path / "render"
    migration = _migration(tmp_path)
    renderer.render(output=root, migration_receipt=migration)
    bundle_path = root / "final-aggregate.yaml"
    original = yaml.safe_load(bundle_path.read_text())
    config_map = next(item for item in original["items"] if item["kind"] == "ConfigMap")
    files = json.loads(gzip.decompress(base64_decode(config_map["binaryData"]["bundle.json.gz"])))
    files["aggregate.py"] += "\nraise RuntimeError('unreviewed executable')\n"
    compressed = gzip.compress(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode(),
        compresslevel=9,
        mtime=0,
    )
    compressed = compressed[:9] + b"\xff" + compressed[10:]
    digests = {
        name: "sha256:" + hashlib.sha256(value.encode()).hexdigest()
        for name, value in files.items()
    }
    plan = json.loads(files["study.json"])
    changed_map, changed_job = renderer._objects(plan, compressed, digests)  # noqa: SLF001
    changed_bundle = {
        "apiVersion": "v1",
        "kind": "List",
        "items": [changed_map, changed_job],
    }
    bundle_path.write_text(yaml.safe_dump(changed_bundle, sort_keys=False))
    changed_receipt = renderer._render_receipt(  # noqa: SLF001
        plan=plan,
        compressed=compressed,
        digests=digests,
        rendered_bundle_file_sha256=final._file_digest(bundle_path),  # noqa: SLF001
    )
    (root / "RENDER.json").write_text(json.dumps(changed_receipt, sort_keys=True) + "\n")
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(
        json.dumps(_server_preview(changed_bundle, "11111111-1111-4111-8111-111111111111"))
    )
    second.write_text(
        json.dumps(_server_preview(changed_bundle, "22222222-2222-4222-8222-222222222222"))
    )

    with pytest.raises(renderer.RenderError, match="executable bytes differ"):
        renderer.validate_previews(
            render_root=root,
            migration_receipt=migration,
            first=first,
            second=second,
            output=tmp_path / "previews.json",
        )


@pytest.mark.parametrize("drift", ("output_root", "replacement_identity"))
def test_preview_validator_rejects_resigned_study_plan_tamper(tmp_path: Path, drift: str) -> None:
    root = tmp_path / "render"
    migration = _migration(tmp_path)
    renderer.render(output=root, migration_receipt=migration)
    bundle_path = root / "final-aggregate.yaml"
    original = yaml.safe_load(bundle_path.read_text())
    config_map = next(item for item in original["items"] if item["kind"] == "ConfigMap")
    files = json.loads(gzip.decompress(base64_decode(config_map["binaryData"]["bundle.json.gz"])))
    plan = json.loads(files["study.json"])
    plan.pop("sha256")
    if drift == "output_root":
        plan["private_output_root"] = "/mnt/sfs/jobs/unreviewed-private-output"
    else:
        replica = next(row for row in plan["replicas"] if row["seed"] == 54)
        replica["evaluation_identity_sha256"] = "sha256:" + "0" * 64
    plan["sha256"] = final._digest(plan)  # noqa: SLF001
    files["study.json"] = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    compressed = gzip.compress(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode(),
        compresslevel=9,
        mtime=0,
    )
    compressed = compressed[:9] + b"\xff" + compressed[10:]
    digests = {
        name: "sha256:" + hashlib.sha256(value.encode()).hexdigest()
        for name, value in files.items()
    }
    changed_map, changed_job = renderer._objects(plan, compressed, digests)  # noqa: SLF001
    changed_bundle = {
        "apiVersion": "v1",
        "kind": "List",
        "items": [changed_map, changed_job],
    }
    bundle_path.write_text(yaml.safe_dump(changed_bundle, sort_keys=False))
    changed_receipt = renderer._render_receipt(  # noqa: SLF001
        plan=plan,
        compressed=compressed,
        digests=digests,
        rendered_bundle_file_sha256=final._file_digest(bundle_path),  # noqa: SLF001
    )
    (root / "RENDER.json").write_text(json.dumps(changed_receipt, sort_keys=True) + "\n")
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(
        json.dumps(_server_preview(changed_bundle, "11111111-1111-4111-8111-111111111111"))
    )
    second.write_text(
        json.dumps(_server_preview(changed_bundle, "22222222-2222-4222-8222-222222222222"))
    )

    with pytest.raises(renderer.RenderError, match="authoritative migration evidence"):
        renderer.validate_previews(
            render_root=root,
            migration_receipt=migration,
            first=first,
            second=second,
            output=tmp_path / "previews.json",
        )
