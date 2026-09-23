#!/usr/bin/env python3
"""Prepare whole-replica protocol-v2 replacements for invalid Fleet replicas.

The source study is immutable.  This module never repairs or cherry-picks one
cell from it.  Instead, one score-blind intent names complete invalid seed
pairs, and the deterministic mapping assigns each pair a fresh seed.  Both
arms and all 17 tasks are then rebuilt under the unchanged evaluation recipe.

This command is provider-free and launch-inert: it reads local sealed packets,
validates one local live-parity receipt, and writes local packets and receipts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.fleet import heldout_launch, model_artifact_v3
from scripts import prepare_qwen38_fleet_seed46_pass8_packets as source

ROOT = Path(__file__).resolve().parents[1]
INTENT_SCHEMA = "cyber_qwen38_fleet_protocol_v2_replica_migration_intent_v1"
RECEIPT_SCHEMA = "cyber_qwen38_fleet_protocol_v2_replica_migration_receipt_v1"
COMPARISON_DEFINITION_SCHEMA = "cyber_qwen38_fleet_dev17_pass8_comparison_definition_v2"
RETIREMENT_EVIDENCE_SCHEMA = "cyber_qwen38_fleet_protocol_v2_retirement_evidence_v1"
PROTOCOL_V2_STUDY_ID = "q38-dev17-base-step1000-p8-v2"
PREDECESSOR_COMPARISON_DEFINITION_SHA256 = (
    "sha256:1154b450624a8b9a567464916da95874c44933175c408b86a80ff5bca0eada70"
)
RETIREMENT_PREFLIGHT = (
    ROOT
    / "docs/evidence/qwen38-fleet-dev17-candidate-seed52-53-v2-retirement-preflight-20260923.json"
)
RETIREMENT_POST = (
    ROOT / "docs/evidence/qwen38-fleet-dev17-candidate-seed52-53-v2-retirement-20260923.json"
)
SOURCE_SEEDS = tuple(range(46, 54))
REQUIRED_INITIAL_INVALID_SEEDS = frozenset({47, 52, 53})
FIRST_REPLACEMENT_SEED = 54
DAILY_ROLLOUT_CAP = 500
TASKS_PER_ARM = 17
ARMS = ("base", "candidate")
ALLOWED_REASON_CLASSES = frozenset(
    {
        "terminal_replica_incomplete",
        "replica_not_started",
        "mixed_infrastructure_invalid_replica",
    }
)
INTENT_FIELDS = {
    "schema",
    "source_protocol_study_id",
    "source_preparation_receipt_sha256",
    "source_preparation_receipt_file_sha256",
    "invalid_original_replicas",
    "score_unsealed",
    "daily_rollout_cap",
    "sha256",
}
INVALID_FIELDS = {"seed", "reason_class", "evidence_receipt_sha256s"}


def _canonical(value: Any) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    )


def _file_sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _path_label(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _read(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be one regular nonsymlink file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not readable JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be one JSON object")
    return value


def _verified(path: Path, label: str) -> dict[str, Any]:
    value = _read(path, label)
    if value.get("sha256") != _canonical(
        {key: item for key, item in value.items() if key != "sha256"}
    ):
        raise ValueError(f"{label} self digest differs")
    return value


def _digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ValueError(f"{label} must be one prefixed SHA-256 digest")
    return value


def _load_intent(
    path: Path, source_receipt: dict[str, Any], source_receipt_path: Path
) -> dict[str, Any]:
    value = _verified(path, "migration intent")
    rows = value.get("invalid_original_replicas")
    if (
        set(value) != INTENT_FIELDS
        or value.get("schema") != INTENT_SCHEMA
        or value.get("source_protocol_study_id") != source.PROTOCOL_STUDY_ID
        or value.get("source_preparation_receipt_sha256") != source_receipt.get("sha256")
        or value.get("source_preparation_receipt_file_sha256") != _file_sha(source_receipt_path)
        or value.get("score_unsealed") is not False
        or value.get("daily_rollout_cap") != DAILY_ROLLOUT_CAP
        or not isinstance(rows, list)
        or not rows
    ):
        raise ValueError("migration intent does not bind the immutable source study")
    seeds: list[int] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != INVALID_FIELDS:
            raise ValueError("invalid replica evidence has unsupported fields")
        seed = row.get("seed")
        evidence = row.get("evidence_receipt_sha256s")
        if (
            type(seed) is not int
            or seed not in SOURCE_SEEDS
            or row.get("reason_class") not in ALLOWED_REASON_CLASSES
            or not isinstance(evidence, list)
            or not evidence
            or len(evidence) != len(set(evidence))
        ):
            raise ValueError("invalid replica evidence is malformed")
        for index, receipt in enumerate(evidence):
            _digest(receipt, f"invalid replica evidence digest {index}")
        seeds.append(seed)
    if seeds != sorted(seeds) or len(seeds) != len(set(seeds)):
        raise ValueError("invalid original seeds must be unique and sorted ascending")
    if not REQUIRED_INITIAL_INVALID_SEEDS.issubset(seeds):
        raise ValueError("migration intent omits an already-proven invalid original replica")
    return value


def deterministic_mapping(invalid_original_seeds: list[int]) -> list[dict[str, int]]:
    """Map sorted invalid original seeds to fresh ascending seeds above 53."""
    if (
        not invalid_original_seeds
        or invalid_original_seeds != sorted(invalid_original_seeds)
        or len(invalid_original_seeds) != len(set(invalid_original_seeds))
        or any(type(seed) is not int or seed not in SOURCE_SEEDS for seed in invalid_original_seeds)
    ):
        raise ValueError("invalid original seed roster is not canonical")
    used = set(SOURCE_SEEDS)
    fresh: list[int] = []
    candidate = FIRST_REPLACEMENT_SEED
    while len(fresh) < len(invalid_original_seeds):
        if candidate not in used:
            fresh.append(candidate)
            used.add(candidate)
        candidate += 1
    return [
        {"invalid_original_seed": original, "replacement_seed": replacement}
        for original, replacement in zip(invalid_original_seeds, fresh, strict=True)
    ]


def _source_inventory(
    packet_root: Path, receipt: dict[str, Any]
) -> dict[int, dict[str, dict[str, Any]]]:
    if (
        receipt.get("schema") != "cyber_qwen38_fleet_seed46_pass8_packet_preparation_v1"
        or receipt.get("protocol_study_id") != source.PROTOCOL_STUDY_ID
        or receipt.get("seeds") != list(SOURCE_SEEDS)
        or len(receipt.get("arms", [])) != len(SOURCE_SEEDS) * len(ARMS)
        or receipt.get("launch_performed") is not False
    ):
        raise ValueError("source packet preparation receipt differs")
    result: dict[int, dict[str, dict[str, Any]]] = {}
    expected_task_versions: list[str] | None = None
    for seed in SOURCE_SEEDS:
        per_seed: dict[str, dict[str, Any]] = {}
        for arm in ARMS:
            packet_path = packet_root / f"seed{seed}" / arm / "LAUNCH_PACKET.json"
            package = heldout_launch.build_package(packet_path)
            identity = package.packet.identity
            route = next(iter(package.evaluation_config["routes"].values()))
            tasks = list(route["task_versions"])
            if (
                identity.get("arm_id") != arm
                or identity.get("sampling_seed") != seed
                or identity.get("pass_k") != 1
                or identity.get("retry_limit") != 0
                or package.evaluation_config.get("max_reviewed_infrastructure_retries") != 0
                or len(tasks) != TASKS_PER_ARM
            ):
                raise ValueError("source arm differs from the frozen pass@8 recipe")
            if expected_task_versions is None:
                expected_task_versions = tasks
            elif tasks != expected_task_versions:
                raise ValueError("source replicas do not share the exact task roster")
            per_seed[arm] = {
                "packet_file_sha256": _file_sha(packet_path),
                "evaluation_identity_sha256": package.packet.identity_sha256,
                "evaluation_config_sha256": identity["evaluation_config_sha256"],
                "comparison_protocol_file_sha256": identity["comparison_protocol_file_sha256"],
                "comparison_protocol_sha256": identity["comparison_protocol_sha256"],
                "protocol_id": identity["protocol_id"],
                "job_name": package.packet.job_name,
                "config_map_name": package.packet.config_map_name,
                "output_root": package.packet.output_root,
                "database": package.packet.database,
            }
        if (
            per_seed["base"]["comparison_protocol_sha256"]
            != per_seed["candidate"]["comparison_protocol_sha256"]
        ):
            raise ValueError("source pair protocol identities differ")
        result[seed] = per_seed
    return result


def _comparison_definition(
    *,
    invalid_seeds: list[int],
    mapping: list[dict[str, int]],
    inventory: dict[int, dict[str, dict[str, Any]]],
    replacement_protocols: list[dict[str, Any]],
    base_template: dict[str, Any],
    task_set: dict[str, Any],
    split: dict[str, Any],
    roster: dict[str, Any],
) -> dict[str, Any]:
    retained = sorted(set(SOURCE_SEEDS) - set(invalid_seeds))
    replacement_seeds = [row["replacement_seed"] for row in mapping]
    included_seeds = sorted([*retained, *replacement_seeds])
    if len(included_seeds) != len(SOURCE_SEEDS) or len(set(included_seeds)) != len(SOURCE_SEEDS):
        raise ValueError("protocol-v2 comparison does not contain exactly eight unique replicas")
    replacement_by_seed = {row["replacement_seed"]: row for row in replacement_protocols}
    replica_protocols = []
    for seed in included_seeds:
        if seed in inventory:
            protocol_id = inventory[seed]["base"]["protocol_id"]
            protocol_sha256 = inventory[seed]["base"]["comparison_protocol_sha256"]
            origin = "retained_original"
        else:
            protocol = replacement_by_seed[seed]
            protocol_id = protocol["protocol_id"]
            protocol_sha256 = protocol["sha256"]
            origin = "whole_pair_replacement"
        replica_protocols.append(
            {
                "seed": seed,
                "origin": origin,
                "protocol_id": protocol_id,
                "comparison_protocol_sha256": protocol_sha256,
            }
        )
    value = {
        "schema": COMPARISON_DEFINITION_SCHEMA,
        "protocol_study_id": PROTOCOL_V2_STUDY_ID,
        "predecessor_comparison_definition_sha256": (PREDECESSOR_COMPARISON_DEFINITION_SHA256),
        "aggregation": "eight_predeclared_pass1_replicas_per_task_and_arm",
        "original_seeds": list(SOURCE_SEEDS),
        "excluded_original_seeds": invalid_seeds,
        "replacement_mapping": mapping,
        "included_seeds": included_seeds,
        "replica_protocols": replica_protocols,
        "task_count": TASKS_PER_ARM,
        "sessions_per_arm": TASKS_PER_ARM * len(included_seeds),
        "total_sessions": TASKS_PER_ARM * len(included_seeds) * len(ARMS),
        "comparison_arms": list(ARMS),
        "models": {
            "base": {
                "model_id": "qwen3.8-27b",
                "revision": source.BASE_REVISION,
            },
            "candidate": {
                "model_id": source.CANDIDATE_ID,
                "revision": source.CANDIDATE_REVISION,
            },
        },
        "task_selection_sha256": task_set["selection_sha256"],
        "split_manifest_sha256": split["sha256"],
        "binding_roster_sha256": roster["sha256"],
        "harness": base_template["harness"],
        "images": base_template["images"],
        "sampling_without_seed": {
            key: item for key, item in base_template["sampling"].items() if key != "seed"
        },
        "pass_k_per_replica": 1,
        "retry_limit": 0,
        "training_data_eligible": False,
        "whole_replica_pairs_only": True,
        "cell_level_replacement_forbidden": True,
        "included_seed_status": "provisional_until_all_valid8_terminal_gates_pass",
        "later_invalid_seed_policy": (
            "create a versioned successor intent, definition, and receipt before score unseal; "
            "never edit this definition in place"
        ),
    }
    value["sha256"] = _canonical(value)
    if value["sha256"] == PREDECESSOR_COMPARISON_DEFINITION_SHA256:
        raise ValueError("protocol-v2 comparison definition reused the v1 public digest")
    return value


def _retirement_evidence(
    preflight_path: Path = RETIREMENT_PREFLIGHT,
    post_path: Path = RETIREMENT_POST,
) -> dict[str, Any]:
    preflight = _verified(preflight_path, "candidate retirement preflight")
    post = _verified(post_path, "candidate retirement receipt")
    preflight_jobs = preflight.get("jobs")
    retirements = post.get("retirements")
    if (
        preflight.get("schema") != "cyber_fleet_eval_queued_job_retirement_preflight_v1"
        or preflight.get("classification") != "queued_unstarted_no_execution"
        or preflight.get("mutation_performed") is not False
        or preflight.get("scores_included") is not False
        or preflight.get("prompts_or_traces_included") is not False
        or not isinstance(preflight_jobs, list)
        or len(preflight_jobs) != 2
        or post.get("schema") != "cyber_fleet_eval_queued_job_retirement_v1"
        or post.get("classification") != "retired_unstarted_zero_execution"
        or post.get("scores_included") is not False
        or post.get("prompts_or_traces_included") is not False
        or not isinstance(retirements, list)
        or len(retirements) != 2
        or post.get("namespace") != preflight.get("namespace")
        or post.get("kube_context") != preflight.get("kube_context")
        or post.get("source_preflight")
        != {
            "path": _path_label(preflight_path),
            "file_sha256": _file_sha(preflight_path),
            "self_sha256": preflight["sha256"],
        }
    ):
        raise ValueError("candidate retirement receipts are incomplete or disagree")
    preflight_by_seed = {row.get("seed"): row for row in preflight_jobs}
    post_by_seed = {row.get("seed"): row for row in retirements}
    if set(preflight_by_seed) != {52, 53} or set(post_by_seed) != {52, 53}:
        raise ValueError("candidate retirement receipts target the wrong replicas")
    targets = []
    for seed in (52, 53):
        before = preflight_by_seed[seed]
        after = post_by_seed[seed]
        job_before = before.get("job", {})
        workload_before = before.get("workload", {})
        execution = before.get("execution_evidence", {})
        job_after = after.get("job", {})
        workload_after = after.get("workload", {})
        postconditions = after.get("postconditions", {})
        config_before = before.get("config_map", {})
        config_after = postconditions.get("config_map_preserved", {})
        if (
            job_before.get("name") != job_after.get("name")
            or job_before.get("uid") != job_after.get("uid")
            or job_before.get("resource_version")
            != job_after.get("delete_precondition_resource_version")
            or job_before.get("suspend") is not True
            or job_before.get("start_time") is not None
            or job_before.get("active") != 0
            or job_before.get("succeeded") != 0
            or job_before.get("failed") != 0
            or job_before.get("priority_class") != "c1"
            or job_before.get("gpu_resources") != 0
            or job_before.get("root_failure_alerts") != "off"
            or workload_before.get("admission") is not None
            or workload_before.get("owner_uid") != job_before.get("uid")
            or workload_after.get("uid") != workload_before.get("uid")
            or workload_after.get("owner_ref_job_uid") != job_before.get("uid")
            or job_after.get("post_state") != "absent"
            or workload_after.get("post_state") != "absent_by_owner_ref_garbage_collection"
            or config_before.get("uid") != config_after.get("uid")
            or config_before.get("name") != config_after.get("name")
            or config_before.get("immutable") is not True
            or config_after.get("immutable") is not True
            or any(
                execution.get(field) != 0
                for field in (
                    "claims",
                    "controller_pods",
                    "model_generation_calls",
                    "sessions",
                )
            )
            or execution.get("dedicated_database_exists") is not False
            or execution.get("output_root_exists") is not False
            or any(
                postconditions.get(field) != 0
                for field in (
                    "claims",
                    "controller_pods",
                    "model_generation_calls",
                    "sessions",
                )
            )
            or postconditions.get("dedicated_database_exists") is not False
            or postconditions.get("output_root_exists") is not False
        ):
            raise ValueError("candidate retirement pre/post evidence disagrees")
        targets.append(
            {
                "seed": seed,
                "job_name": job_before["name"],
                "job_uid": job_before["uid"],
                "workload_name": workload_before["name"],
                "workload_uid": workload_before["uid"],
                "config_map_name": config_before["name"],
                "config_map_uid": config_before["uid"],
                "model_rollouts": 0,
                "outputs_preserved": True,
            }
        )
    if post.get("post_verification") != {
        "exact_config_maps_preserved": 2,
        "exact_controller_pods": 0,
        "exact_dedicated_databases_present": 0,
        "exact_jobs_absent": 2,
        "exact_output_roots_present": 0,
        "exact_owner_workloads_absent": 2,
    }:
        raise ValueError("candidate retirement post-verification differs")
    value = {
        "schema": RETIREMENT_EVIDENCE_SCHEMA,
        "preflight": {
            "path": _path_label(preflight_path),
            "file_sha256": _file_sha(preflight_path),
            "receipt_sha256": preflight["sha256"],
        },
        "post_retirement": {
            "path": _path_label(post_path),
            "file_sha256": _file_sha(post_path),
            "receipt_sha256": post["sha256"],
        },
        "targets": targets,
        "retired_jobs": 2,
        "retired_owner_workloads": 2,
        "preserved_immutable_config_maps": 2,
        "model_rollouts": 0,
        "outputs_or_databases_deleted": False,
        "score_values_read": False,
    }
    value["sha256"] = _canonical(value)
    return value


def _configs(base_template: dict[str, Any], seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    base, candidate = source._configs(base_template, seed, candidate_generation=2)  # noqa: SLF001
    base["name"] = f"q38-dev17-s{seed}-base-replacement-p1-v2"
    candidate["name"] = f"q38-dev17-s{seed}-t3k32s1000-replacement-p1-v2"
    packet_name = f"qwen38-step1000-seed{seed}-replacement-provenance-v2.json"
    artifact = source._candidate_packet(candidate["name"])  # noqa: SLF001
    model_artifact_v3.validate_packet(candidate["models"], artifact)
    candidate["model_artifact_binding"] = source._candidate_binding(  # noqa: SLF001
        artifact, packet_name
    )
    return base, candidate


def _protocol(base: dict[str, Any], seed: int) -> dict[str, Any]:
    value = source._protocol(base, seed)  # noqa: SLF001
    value["protocol_id"] = f"q38-dev17-s{seed}-base-t3k32s1000-replacement-p1-v2"
    value["sha256"] = _canonical({key: item for key, item in value.items() if key != "sha256"})
    return value


def prepare(
    *,
    source_packets: Path,
    migration_intent: Path,
    live_parity: Path,
    output: Path,
    now: datetime | None = None,
) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise FileExistsError("replacement packet output already exists")
    if not output.parent.is_dir():
        raise ValueError("replacement packet output parent does not exist")
    source_receipt_path = source_packets / "PREPARATION_RECEIPT.json"
    source_receipt = _verified(source_receipt_path, "source preparation receipt")
    inventory = _source_inventory(source_packets, source_receipt)
    intent = _load_intent(migration_intent, source_receipt, source_receipt_path)
    invalid = intent["invalid_original_replicas"]
    mapping = deterministic_mapping([row["seed"] for row in invalid])
    reason_by_seed = {row["seed"]: row for row in invalid}
    base_template, task_set, split, corpus, roster = source._inputs()  # noqa: SLF001
    first_seed = mapping[0]["replacement_seed"]
    parity_base, parity_candidate = _configs(base_template, first_seed)
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
        configs={"base": parity_base, "fresh75": parity_candidate},
        now=now or datetime.now(UTC),
    )
    temporary = Path(tempfile.mkdtemp(prefix=".fleet-protocol-v2-", dir=output.parent))
    try:
        replacement_arms: list[dict[str, Any]] = []
        replacement_protocols: list[dict[str, Any]] = []
        migration_rows: list[dict[str, Any]] = []
        for mapping_row in mapping:
            original_seed = mapping_row["invalid_original_seed"]
            seed = mapping_row["replacement_seed"]
            seed_dir = temporary / f"seed{seed}"
            seed_dir.mkdir(mode=0o700)
            base, candidate = _configs(base_template, seed)
            protocol = _protocol(base, seed)
            base_path = seed_dir / f"qwen38-base-fleet-dev17-opencode-seed{seed}-pass1-v2.json"
            candidate_path = seed_dir / (
                f"qwen38-teacher3k32-step1000-fleet-dev17-opencode-seed{seed}-pass1-v2.json"
            )
            protocol_path = seed_dir / (
                f"qwen38-fleet-dev17-seed{seed}-base-step1000-replacement-protocol-v2.json"
            )
            base_provenance = seed_dir / f"qwen38-base-seed{seed}-replacement-provenance-v2.json"
            candidate_provenance = seed_dir / (
                f"qwen38-step1000-seed{seed}-replacement-provenance-v2.json"
            )
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
                    "job_name": f"chris-q38-dev17-s{seed}-base-repl-p1-v2",
                    "config_map_name": f"chris-q38-dev17-s{seed}-base-repl-code-v2",
                    "output_root": f"/mnt/sfs/jobs/chris-q38-fleet-dev17-s{seed}-base-repl-p1-v2",
                    "database": f"q38_dev17_s{seed}_base_repl_p1_v2",
                },
                "candidate": {
                    **parity_arms["candidate"],
                    "job_name": f"chris-q38-dev17-s{seed}-t3k32s1000-repl-p1-v2",
                    "config_map_name": f"chris-q38-dev17-s{seed}-t3k32s1000-repl-code-v2",
                    "output_root": (
                        f"/mnt/sfs/jobs/chris-q38-fleet-dev17-s{seed}-t3k32s1000-repl-p1-v2"
                    ),
                    "database": f"q38_dev17_s{seed}_t3k32s1000_repl_p1_v2",
                },
            }
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
                    package.packet.identity["sampling_seed"] != seed
                    or package.packet.identity["retry_limit"] != 0
                    or package.evaluation_config["max_reviewed_infrastructure_retries"] != 0
                    or len(route["task_versions"]) != TASKS_PER_ARM
                ):
                    raise ValueError("replacement arm changed the frozen scientific recipe")
                replacement_arms.append(
                    {
                        **row,
                        "invalid_original_seed": original_seed,
                        "replacement_seed": seed,
                        "packet_file_sha256": _file_sha(packet_path),
                        "comparison_protocol_file_sha256": package.packet.identity[
                            "comparison_protocol_file_sha256"
                        ],
                        "comparison_protocol_sha256": package.packet.identity[
                            "comparison_protocol_sha256"
                        ],
                    }
                )
            replacement_protocols.append(
                {
                    "invalid_original_seed": original_seed,
                    "replacement_seed": seed,
                    "protocol_id": protocol["protocol_id"],
                    "sha256": protocol["sha256"],
                    "file_sha256": _file_sha(protocol_path),
                }
            )
            migration_rows.append(
                {
                    **mapping_row,
                    "reason_class": reason_by_seed[original_seed]["reason_class"],
                    "evidence_receipt_sha256s": reason_by_seed[original_seed][
                        "evidence_receipt_sha256s"
                    ],
                    "excluded_source_arms": inventory[original_seed],
                    "whole_pair_excluded": True,
                }
            )
        comparison_definition = _comparison_definition(
            invalid_seeds=[row["seed"] for row in invalid],
            mapping=mapping,
            inventory=inventory,
            replacement_protocols=replacement_protocols,
            base_template=base_template,
            task_set=task_set,
            split=split,
            roster=roster,
        )
        source._write_json(  # noqa: SLF001
            temporary / "COMPARISON_DEFINITION.json", comparison_definition
        )
        retirement_evidence = _retirement_evidence()
        source._write_json(  # noqa: SLF001
            temporary / "RETIREMENT_EVIDENCE.json", retirement_evidence
        )
        planned = len(mapping) * len(ARMS) * TASKS_PER_ARM
        cumulative_rollouts = 136 + 102 + planned
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "source": {
                "protocol_study_id": source.PROTOCOL_STUDY_ID,
                "preparation_receipt_sha256": source_receipt["sha256"],
                "preparation_receipt_file_sha256": _file_sha(source_receipt_path),
                "seeds": list(SOURCE_SEEDS),
                "arm_count": len(SOURCE_SEEDS) * len(ARMS),
            },
            "migration_intent_sha256": intent["sha256"],
            "comparison_definition": comparison_definition,
            "comparison_definition_file_sha256": _file_sha(
                temporary / "COMPARISON_DEFINITION.json"
            ),
            "mapping_rule": (
                "invalid original seeds sorted ascending map to the smallest unused seeds "
                "greater than 53, ascending"
            ),
            "migrations": migration_rows,
            "excluded_original_seeds": [row["seed"] for row in invalid],
            "included_seeds": comparison_definition["included_seeds"],
            "replacement_protocols": replacement_protocols,
            "replacement_arms": replacement_arms,
            "retirement_evidence": {
                "sha256": retirement_evidence["sha256"],
                "file_sha256": _file_sha(temporary / "RETIREMENT_EVIDENCE.json"),
                "targets": len(retirement_evidence["targets"]),
                "model_rollouts": 0,
                "outputs_or_databases_deleted": False,
            },
            "scientific_identity": {
                "task_count_per_arm": TASKS_PER_ARM,
                "comparison_arms": list(ARMS),
                "pass_k": 1,
                "retry_limit": 0,
                "same_task_versions_models_harness_budgets_and_sampling_recipe": True,
                "whole_replica_pairs_only": True,
                "cell_level_replacement_forbidden": True,
                "seed_reuse_forbidden": True,
                "future_invalid_original_seeds_added_before_score_unseal": True,
            },
            "capacity": {
                "new_replacement_rollouts": planned,
                "final_comparison_rollouts": (TASKS_PER_ARM * len(SOURCE_SEEDS) * len(ARMS)),
                "original_base_rollouts": 136,
                "original_candidate_started_seeds": [46, 47, 48, 49, 50, 51],
                "original_candidate_started_seed_rollouts": 102,
                "original_candidate_retired_before_start_seeds": [52, 53],
                "original_candidate_retired_before_start_rollouts": 0,
                "scoring_or_metadata_cpu_model_rollouts": 0,
                "cumulative_model_rollouts_consumed_or_planned_today": cumulative_rollouts,
                "daily_rollout_cap": intent["daily_rollout_cap"],
                "within_daily_cap": cumulative_rollouts <= intent["daily_rollout_cap"],
            },
            "selection": {
                "selection_sha256": task_set["selection_sha256"],
                "split_sha256": split["sha256"],
                "corpus_dev_windows": corpus["dev_windows"],
                "binding_roster_sha256": roster["sha256"],
            },
            "live_parity_file_sha256": _file_sha(live_parity),
            "live_parity_receipt_sha256": parity["receipt_sha256"],
            "privacy": {
                "score_values_read": False,
                "prompts_responses_flags_rewards_or_trace_content_read": False,
                "infrastructure_reason_classes_only": True,
            },
            "external_mutations": 0,
            "launch_performed": False,
        }
        if not receipt["capacity"]["within_daily_cap"]:
            raise ValueError("replacement campaign exceeds the daily rollout cap")
        receipt["sha256"] = _canonical(receipt)
        source._write_json(temporary / "MIGRATION_RECEIPT.json", receipt)  # noqa: SLF001
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-packets", type=Path, required=True)
    parser.add_argument("--migration-intent", type=Path, required=True)
    parser.add_argument("--live-parity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                source_packets=args.source_packets,
                migration_intent=args.migration_intent,
                live_parity=args.live_parity,
                output=args.output,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
