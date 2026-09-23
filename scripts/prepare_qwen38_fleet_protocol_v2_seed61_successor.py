#!/usr/bin/env python3
"""Prepare the score-blind whole-pair seed-61 successor to protocol v2.

The accepted protocol-v2 evidence stays immutable.  This provider-free command
binds that exact receipt, excludes its now irrecoverable seed-56 pair, and
prepares one symmetric base/candidate pair at fresh seed 61.  It never previews
or creates a cluster object and never reads scores or rollout content.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.fleet import heldout_launch
from scripts import prepare_qwen38_fleet_protocol_v2_replacements as protocol_v2
from scripts import prepare_qwen38_fleet_seed46_pass8_packets as source

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "cyber_qwen38_fleet_protocol_v2_seed61_successor_receipt_v1"
COMPARISON_SCHEMA = "cyber_qwen38_fleet_dev17_pass8_comparison_definition_v3"
PREDECESSOR_RECEIPT_SHA256 = (
    "sha256:9fa3f39f9030dede9efaf38d8e0dfa218a36db885f8f6c72695ba5aa5ec13643"
)
PREDECESSOR_RECEIPT_FILE_SHA256 = (
    "sha256:fcee992ba4bb81527c5a63774fa76e9334163f7d583824ddb4b9f3b46c1c84d4"
)
PREDECESSOR_COMPARISON_SHA256 = (
    "sha256:9813713ef2ab023ac6c64df494ca7cbf39b920e8fba2f05f5949daaa006479f1"
)
PREDECESSOR_COMPARISON_FILE_SHA256 = (
    "sha256:1356a652628619020ab7cbdd6b599a5b162dc99b47ee71b957f8428bb49227c6"
)
RETIREMENT_PRE = (
    ROOT / "docs/evidence/qwen38-study/2026-09-23-q38-dev17-seed56-protocol-v2-retirement-pre.json"
)
RETIREMENT_POST = (
    ROOT
    / "docs/evidence/qwen38-study/2026-09-23-q38-dev17-seed56-protocol-v2-retirement-receipt.json"
)
RETIREMENT_PRE_SHA256 = "sha256:376bd023c6757020f8a3c4c44a49a2348be4ca634f3deaee0e25d92e100ed8fc"
RETIREMENT_PRE_FILE_SHA256 = (
    "sha256:7e2b7987c2d2bf3394464a03da8e8fa91f0c871ccb90fb1537a36308bf5edeb5"
)
RETIREMENT_POST_SHA256 = "sha256:d17610246725bac2a70cea5a3a3ed0f28fcbbfe5cd511a1225b1fbcdecd310f9"
RETIREMENT_POST_FILE_SHA256 = (
    "sha256:61de8e4f15574a2c6be48ffed6524132c6f23f2a4b673214eb19afefabaa22cd"
)
SUPERSEDED_SEED = 56
SUCCESSOR_SEED = 61
ORIGINAL_SEED = 49
INCLUDED_SEEDS = (48, 54, 55, 57, 58, 59, 60, 61)
PREDECESSOR_PROJECTED_ROLLOUTS = 350
NEW_ROLLOUTS = protocol_v2.TASKS_PER_ARM * len(protocol_v2.ARMS)
PROJECTED_ROLLOUTS = PREDECESSOR_PROJECTED_ROLLOUTS + NEW_ROLLOUTS


def _evidence() -> dict[str, Any]:
    pre = protocol_v2._verified(RETIREMENT_PRE, "seed-56 retirement pre-snapshot")  # noqa: SLF001
    post = protocol_v2._verified(RETIREMENT_POST, "seed-56 retirement receipt")  # noqa: SLF001
    missing = [
        row
        for row in pre.get("retry_review", [])
        if row.get("failure_code") == "post_claim.connecterror"
        and row.get("has_local_result") is False
        and row.get("has_session") is False
        and row.get("count") == 1
    ]
    if (
        protocol_v2._file_sha(RETIREMENT_PRE) != RETIREMENT_PRE_FILE_SHA256  # noqa: SLF001
        or pre.get("sha256") != RETIREMENT_PRE_SHA256
        or pre.get("schema") != "q38_dev17_protocol_v2_exact_job_retirement_pre_v1"
        or pre.get("privacy")
        != {
            "score_blind": True,
            "scores_read": False,
            "traces_read": False,
            "prompts_read": False,
            "credentials_read": False,
        }
        or pre.get("reason", {}).get("seed") != SUPERSEDED_SEED
        or pre.get("reason", {}).get("class") != "exact_valid8_replica_exclusion"
        or len(missing) != 1
        or protocol_v2._file_sha(RETIREMENT_POST) != RETIREMENT_POST_FILE_SHA256  # noqa: SLF001
        or post.get("sha256") != RETIREMENT_POST_SHA256
        or post.get("schema") != "q38_dev17_protocol_v2_exact_job_retirement_receipt_v1"
        or post.get("pre_snapshot")
        != {
            "path": str(RETIREMENT_PRE.relative_to(ROOT)),
            "self_sha256": RETIREMENT_PRE_SHA256,
            "file_sha256": RETIREMENT_PRE_FILE_SHA256,
        }
        or post.get("post_release")
        != {
            "exact_job_count": 0,
            "exact_owner_pod_count": 0,
            "exact_owner_workload_count": 0,
        }
        or post.get("scope")
        != {
            "seed": SUPERSEDED_SEED,
            "whole_pair_excluded": True,
            "other_jobs_mutated": False,
            "serving_route_mutated": False,
        }
        or post.get("privacy", {}).get("score_blind") is not True
        or post.get("privacy", {}).get("scores_included") is not False
        or post.get("privacy", {}).get("prompts_or_traces_included") is not False
    ):
        raise ValueError("seed-56 score-blind retirement evidence differs")
    return {
        "pre_snapshot": {
            "path": str(RETIREMENT_PRE.relative_to(ROOT)),
            "sha256": pre["sha256"],
            "file_sha256": RETIREMENT_PRE_FILE_SHA256,
        },
        "post_receipt": {
            "path": str(RETIREMENT_POST.relative_to(ROOT)),
            "sha256": post["sha256"],
            "file_sha256": RETIREMENT_POST_FILE_SHA256,
        },
        "reason_class": "post_claim.connecterror_without_local_result_or_session",
    }


def _predecessor(packet_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt_path = packet_root / "MIGRATION_RECEIPT.json"
    definition_path = packet_root / "COMPARISON_DEFINITION.json"
    receipt = protocol_v2._verified(receipt_path, "protocol-v2 migration receipt")  # noqa: SLF001
    definition = protocol_v2._verified(  # noqa: SLF001
        definition_path, "protocol-v2 comparison definition"
    )
    if (
        receipt.get("sha256") != PREDECESSOR_RECEIPT_SHA256
        or protocol_v2._file_sha(receipt_path) != PREDECESSOR_RECEIPT_FILE_SHA256  # noqa: SLF001
        or definition.get("sha256") != PREDECESSOR_COMPARISON_SHA256
        or protocol_v2._file_sha(definition_path)  # noqa: SLF001
        != PREDECESSOR_COMPARISON_FILE_SHA256
        or receipt.get("comparison_definition") != definition
        or receipt.get("comparison_definition_file_sha256") != PREDECESSOR_COMPARISON_FILE_SHA256
        or receipt.get("included_seeds") != [48, 54, 55, 56, 57, 58, 59, 60]
        or receipt.get("excluded_original_seeds") != list(protocol_v2.FROZEN_INVALID_SEEDS)
        or receipt.get("privacy")
        != {
            "score_values_read": False,
            "prompts_responses_flags_rewards_or_trace_content_read": False,
            "infrastructure_reason_classes_only": True,
        }
        or receipt.get("external_mutations") != 0
        or receipt.get("launch_performed") is not False
        or receipt.get("capacity", {}).get("projected_rollouts_after_reservation")
        != PREDECESSOR_PROJECTED_ROLLOUTS
        or receipt.get("capacity", {}).get("new_replacement_rollouts") != 238
        or receipt.get("capacity", {}).get("daily_rollout_cap") != protocol_v2.DAILY_ROLLOUT_CAP
    ):
        raise ValueError("protocol-v2 predecessor differs from the exact accepted bytes")
    seed56_rows = [
        row
        for row in receipt.get("replacement_arms", [])
        if row.get("replacement_seed") == SUPERSEDED_SEED
    ]
    if {row.get("arm_id") for row in seed56_rows} != set(protocol_v2.ARMS):
        raise ValueError("protocol-v2 predecessor lacks the complete seed-56 pair")
    for row in seed56_rows:
        packet = packet_root / f"seed{SUPERSEDED_SEED}" / row["arm_id"] / "LAUNCH_PACKET.json"
        binding = protocol_v2._evaluation_binding(packet)  # noqa: SLF001
        if any(
            (
                binding["packet_file_sha256"] != row.get("packet_file_sha256"),
                binding["evaluation_identity_sha256"] != row.get("evaluation_identity_sha256"),
                binding["evaluation_plan_sha256"] != row.get("evaluation_plan_sha256"),
                binding["runtime_files_sha256"] != row.get("runtime_files_sha256"),
            )
        ):
            raise ValueError("seed-56 packet pair differs from the predecessor receipt")
    return receipt, definition


def _definition(predecessor: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(predecessor)
    value.pop("sha256")
    value["schema"] = COMPARISON_SCHEMA
    value["predecessor_comparison_definition_sha256"] = PREDECESSOR_COMPARISON_SHA256
    value["predecessor_migration_receipt_sha256"] = PREDECESSOR_RECEIPT_SHA256
    value["excluded_protocol_v2_seeds"] = [SUPERSEDED_SEED]
    value["included_seeds"] = list(INCLUDED_SEEDS)
    value["replacement_mapping"] = [
        {
            **row,
            "replacement_seed": SUCCESSOR_SEED,
        }
        if row == {"invalid_original_seed": ORIGINAL_SEED, "replacement_seed": SUPERSEDED_SEED}
        else row
        for row in predecessor["replacement_mapping"]
    ]
    value["superseded_replacement"] = {
        "invalid_original_seed": ORIGINAL_SEED,
        "superseded_replacement_seed": SUPERSEDED_SEED,
        "successor_seed": SUCCESSOR_SEED,
        "whole_pair_excluded": True,
        "evidence_receipt_sha256s": [RETIREMENT_PRE_SHA256, RETIREMENT_POST_SHA256],
    }
    value["replica_protocols"] = [
        row for row in predecessor["replica_protocols"] if row["seed"] != SUPERSEDED_SEED
    ] + [
        {
            "seed": SUCCESSOR_SEED,
            "origin": "whole_pair_successor",
            "protocol_id": protocol["protocol_id"],
            "comparison_protocol_sha256": protocol["sha256"],
        }
    ]
    value["replica_protocols"].sort(key=lambda row: row["seed"])
    if (
        [row["seed"] for row in value["replica_protocols"]] != list(INCLUDED_SEEDS)
        or value["sessions_per_arm"] != protocol_v2.TASKS_PER_ARM * len(INCLUDED_SEEDS)
        or value["total_sessions"]
        != protocol_v2.TASKS_PER_ARM * len(INCLUDED_SEEDS) * len(protocol_v2.ARMS)
    ):
        raise ValueError("successor comparison does not preserve exact pass@8")
    value["sha256"] = protocol_v2._canonical(value)  # noqa: SLF001
    return value


def prepare(
    *,
    predecessor_packets: Path,
    live_parity: Path,
    output: Path,
    now: datetime | None = None,
) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise FileExistsError("seed-61 successor output already exists")
    if not output.parent.is_dir():
        raise ValueError("seed-61 successor output parent does not exist")
    predecessor_receipt, predecessor_definition = _predecessor(predecessor_packets)
    evidence = _evidence()
    base_template, task_set, split, _corpus, roster = source._inputs()  # noqa: SLF001
    base, candidate = protocol_v2._configs(base_template, SUCCESSOR_SEED)  # noqa: SLF001
    protocol = protocol_v2._protocol(base, SUCCESSOR_SEED)  # noqa: SLF001
    parity_arms = {
        "base": {
            "model_revision": source.BASE_REVISION,
            "served_id": "qwen3.8-27b",
            "expected_source_path": f"/models/qwen3.8-27b/{source.BASE_REVISION}",
        },
        "candidate": {
            "model_revision": source.CANDIDATE_REVISION,
            "served_id": source.CANDIDATE_ID,
            "expected_source_path": f"/models/{source.CANDIDATE_ID}",
        },
    }
    parity = source.shared._live_parity(  # noqa: SLF001
        live_parity,
        readiness={"arms": {"base": parity_arms["base"], "fresh75": parity_arms["candidate"]}},
        configs={"base": base, "fresh75": candidate},
        now=now or datetime.now(UTC),
    )
    temporary = Path(tempfile.mkdtemp(prefix=".fleet-seed61-successor-", dir=output.parent))
    try:
        seed_dir = temporary / f"seed{SUCCESSOR_SEED}"
        seed_dir.mkdir(mode=0o700)
        base_path = seed_dir / "qwen38-base-fleet-dev17-opencode-seed61-pass1-v2.json"
        candidate_path = (
            seed_dir / "qwen38-teacher3k32-step1000-fleet-dev17-opencode-seed61-pass1-v2.json"
        )
        protocol_path = (
            seed_dir / "qwen38-fleet-dev17-seed61-base-step1000-replacement-protocol-v2.json"
        )
        base_provenance = seed_dir / "qwen38-base-seed61-replacement-provenance-v2.json"
        candidate_provenance = seed_dir / "qwen38-step1000-seed61-replacement-provenance-v2.json"
        source._write_json(base_path, base)  # noqa: SLF001
        source._write_json(candidate_path, candidate)  # noqa: SLF001
        source._write_json(protocol_path, protocol)  # noqa: SLF001
        source._write_json(  # noqa: SLF001
            base_provenance,
            source._base_provenance(base_path, roster, base["name"]),  # noqa: SLF001
        )
        source._write_json(  # noqa: SLF001
            candidate_provenance,
            source._candidate_packet(candidate["name"]),  # noqa: SLF001
        )
        arms = {
            "base": {
                **parity_arms["base"],
                "job_name": "chris-q38-dev17-s61-base-repl-p1-v2",
                "config_map_name": "chris-q38-dev17-s61-base-repl-code-v2",
                "output_root": "/mnt/sfs/jobs/chris-q38-fleet-dev17-s61-base-repl-p1-v2",
                "database": "q38_dev17_s61_base_repl_p1_v2",
            },
            "candidate": {
                **parity_arms["candidate"],
                "job_name": "chris-q38-dev17-s61-t3k32s1000-repl-p1-v2",
                "config_map_name": "chris-q38-dev17-s61-t3k32s1000-repl-code-v2",
                "output_root": ("/mnt/sfs/jobs/chris-q38-fleet-dev17-s61-t3k32s1000-repl-p1-v2"),
                "database": "q38_dev17_s61_t3k32s1000_repl_p1_v2",
            },
        }
        replacement_arms = []
        for arm_id, config, config_path, checkpoint in (
            ("base", base, base_path, base_provenance),
            ("candidate", candidate, candidate_path, candidate_provenance),
        ):
            row = source.shared._prepare_arm(  # noqa: SLF001
                seed_dir / arm_id,
                arm_id=arm_id,
                arm=arms[arm_id],
                config_path=config_path,
                config=config,
                task_set_path=source.TASK_SET,
                split_path=source.SPLIT,
                protocol_path=protocol_path,
                protocol=protocol,
                checkpoint_path=checkpoint,
                proof_path=live_parity,
                ledger_path=source.LEDGER,
                source_files=source.V3_SOURCE_FILES if arm_id == "candidate" else None,
            )
            packet_path = seed_dir / arm_id / "LAUNCH_PACKET.json"
            package = heldout_launch.build_package(packet_path)
            route = next(iter(package.evaluation_config["routes"].values()))
            if (
                package.packet.identity["sampling_seed"] != SUCCESSOR_SEED
                or package.packet.identity["pass_k"] != 1
                or package.packet.identity["retry_limit"] != 0
                or package.evaluation_config["max_reviewed_infrastructure_retries"] != 0
                or len(route["task_versions"]) != protocol_v2.TASKS_PER_ARM
                or package.job["metadata"]["annotations"].get(
                    heldout_launch.FAILURE_ALERT_ANNOTATION
                )
                != heldout_launch.FAILURE_ALERT_OFF
                or package.job["spec"]["template"]["spec"].get("priorityClassName") != "c1"
                or "nvidia.com/gpu" in json.dumps(package.job)
            ):
                raise ValueError("seed-61 arm changed the frozen recipe or resource policy")
            replacement_arms.append(
                {
                    **row,
                    **protocol_v2._evaluation_binding(packet_path),  # noqa: SLF001
                    "superseded_replacement_seed": SUPERSEDED_SEED,
                    "successor_seed": SUCCESSOR_SEED,
                    "comparison_protocol_file_sha256": package.packet.identity[
                        "comparison_protocol_file_sha256"
                    ],
                    "comparison_protocol_sha256": package.packet.identity[
                        "comparison_protocol_sha256"
                    ],
                }
            )
        definition = _definition(predecessor_definition, protocol)
        source._write_json(temporary / "COMPARISON_DEFINITION.json", definition)  # noqa: SLF001
        receipt = {
            "schema": SCHEMA,
            "predecessor": {
                "migration_receipt_sha256": predecessor_receipt["sha256"],
                "migration_receipt_file_sha256": PREDECESSOR_RECEIPT_FILE_SHA256,
                "comparison_definition_sha256": predecessor_definition["sha256"],
                "comparison_definition_file_sha256": PREDECESSOR_COMPARISON_FILE_SHA256,
                "included_seeds": predecessor_receipt["included_seeds"],
                "excluded_original_seeds": predecessor_receipt["excluded_original_seeds"],
            },
            "seed56_exclusion_evidence": evidence,
            "lineage": {
                "invalid_original_seed": ORIGINAL_SEED,
                "superseded_replacement_seed": SUPERSEDED_SEED,
                "successor_seed": SUCCESSOR_SEED,
                "whole_pair_excluded": True,
                "cell_level_replacement_forbidden": True,
            },
            "comparison_definition": definition,
            "comparison_definition_file_sha256": protocol_v2._file_sha(  # noqa: SLF001
                temporary / "COMPARISON_DEFINITION.json"
            ),
            "included_seeds": list(INCLUDED_SEEDS),
            "replacement_protocol": {
                "seed": SUCCESSOR_SEED,
                "protocol_id": protocol["protocol_id"],
                "sha256": protocol["sha256"],
                "file_sha256": protocol_v2._file_sha(protocol_path),  # noqa: SLF001
            },
            "replacement_arms": replacement_arms,
            "capacity": {
                "daily_rollout_cap": protocol_v2.DAILY_ROLLOUT_CAP,
                "predecessor_actual_started_at_census": predecessor_receipt["capacity"][
                    "actual_started_rollouts_today"
                ],
                "predecessor_reserved_rollouts": predecessor_receipt["capacity"][
                    "new_replacement_rollouts"
                ],
                "predecessor_projected_after_reservation": PREDECESSOR_PROJECTED_ROLLOUTS,
                "predecessor_reservation_already_includes_seed56_pair": True,
                "excluded_seed56_reservation_credit_claimed": 0,
                "new_seed61_rollouts": NEW_ROLLOUTS,
                "projected_after_seed61_reservation": PROJECTED_ROLLOUTS,
                "remaining_after_seed61_reservation": (
                    protocol_v2.DAILY_ROLLOUT_CAP - PROJECTED_ROLLOUTS
                ),
                "within_daily_cap": PROJECTED_ROLLOUTS <= protocol_v2.DAILY_ROLLOUT_CAP,
            },
            "scientific_identity": {
                "task_count_per_arm": protocol_v2.TASKS_PER_ARM,
                "comparison_arms": list(protocol_v2.ARMS),
                "pass_k": 1,
                "retry_limit": 0,
                "same_task_versions_models_harness_budgets_and_sampling_recipe": True,
                "whole_replica_pairs_only": True,
                "score_blind_exclusion": True,
            },
            "selection": {
                "selection_sha256": task_set["selection_sha256"],
                "split_sha256": split["sha256"],
                "binding_roster_sha256": roster["sha256"],
            },
            "live_parity_file_sha256": protocol_v2._file_sha(live_parity),  # noqa: SLF001
            "live_parity_receipt_sha256": parity["receipt_sha256"],
            "privacy": {
                "score_values_read": False,
                "prompts_responses_flags_rewards_or_trace_content_read": False,
                "infrastructure_reason_classes_only": True,
            },
            "external_mutations": 0,
            "server_preview_performed": False,
            "launch_performed": False,
        }
        if not receipt["capacity"]["within_daily_cap"]:
            raise ValueError("seed-61 successor exceeds the daily rollout cap")
        receipt["sha256"] = protocol_v2._canonical(receipt)  # noqa: SLF001
        source._write_json(temporary / "SUCCESSOR_RECEIPT.json", receipt)  # noqa: SLF001
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predecessor-packets", type=Path, required=True)
    parser.add_argument("--live-parity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                predecessor_packets=args.predecessor_packets,
                live_parity=args.live_parity,
                output=args.output,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
