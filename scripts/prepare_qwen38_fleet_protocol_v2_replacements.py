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
import subprocess
import sys
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
DAILY_BUDGET_EVIDENCE_SCHEMA = "cyber_qwen38_fleet_protocol_v2_daily_budget_evidence_v1"
PROTOCOL_V2_STUDY_ID = "q38-dev17-base-step1000-p8-v2"
PREDECESSOR_LAUNCH_RECEIPT_SHA256 = (
    "sha256:1154b450624a8b9a567464916da95874c44933175c408b86a80ff5bca0eada70"
)
RETIREMENT_PREFLIGHT = (
    ROOT
    / "docs/evidence/qwen38-fleet-dev17-candidate-seed52-53-v2-retirement-preflight-20260923.json"
)
RETIREMENT_POST = (
    ROOT / "docs/evidence/qwen38-fleet-dev17-candidate-seed52-53-v2-retirement-20260923.json"
)
SEED51_INVALID_EVIDENCE = (
    ROOT / "docs/evidence/qwen38-fleet-dev17-seed51-base-invalid-replica-20260923.json"
)
PARTIAL_RECOVERY_HOLD_EVIDENCE = (
    ROOT / "docs/evidence/qwen38-fleet-dev17-base-prefix-recovery-hold-20260923.json"
)
SOURCE_LAUNCH_EVIDENCE = (
    ROOT / "docs/evidence/qwen38-fleet-dev17-seed46to53-pass8-launch-20260923.json"
)
SOURCE_CREATE_BINDINGS_EVIDENCE = (
    ROOT / "docs/evidence/qwen38-fleet-dev17-seed46to53-pass8-create-bindings-20260923.json"
)
CANDIDATE_SUCCESSOR_CREATE_EVIDENCE = (
    ROOT / "docs/evidence/qwen38-fleet-dev17-seed46to53-candidate-successors-20260923.json"
)
EXPECTED_SOURCE_PREPARATION_RECEIPT_SHA256 = (
    "sha256:22b59a3e9e1ea2ed8e8b92920304f27bbd953925fe0ecf709c7308433db8f0d3"
)
EXPECTED_SOURCE_PREPARATION_RECEIPT_FILE_SHA256 = (
    "sha256:2827250c73d024cab06906d393eba9270784f99b70bb3812d9893c53b1f84859"
)
EXPECTED_CANDIDATE_SUCCESSOR_PREPARATION_SHA256 = (
    "sha256:d3624db05e0665cf0483d802a49c54be601293e5267b5475143a282791269189"
)
EXPECTED_CANDIDATE_SUCCESSOR_PREPARATION_FILE_SHA256 = (
    "sha256:5605623bf5f6205a70a6bad0f02985f985858904badfa7be413a1cead1cbe67a"
)
SOURCE_SEEDS = tuple(range(46, 54))
FROZEN_INVALID_SEEDS = (46, 47, 49, 50, 51, 52, 53)
FIRST_REPLACEMENT_SEED = 54
DAILY_ROLLOUT_CAP = 500
TASKS_PER_ARM = 17
ARMS = ("base", "candidate")
SEALED_EVALUATOR_MODULES = (
    "evaluate.py",
    "rollout_worker.py",
    "rollout_postgres.py",
    "rollout_ledger.py",
    "opencode_self_hosted.py",
    "fixed_proxy.py",
    "exact_pass4_crypto.py",
    "exact_pass4_universe.py",
    "rollout_campaign.py",
)
SEALED_RUNTIME_IDENTITY_FILES = SEALED_EVALUATOR_MODULES[:7]
ALLOWED_REASON_CLASSES = frozenset(
    {
        "terminal_replica_incomplete",
        "replica_not_started",
        "mixed_infrastructure_invalid_replica",
        "persisted_transcript_prefix_mismatch",
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
FROZEN_INVALID_REPLICA_EVIDENCE: dict[int, dict[str, Any]] = {
    46: {
        "reason_class": "persisted_transcript_prefix_mismatch",
        "evidence_receipt_sha256s": [
            "sha256:e7b0031010f097f66dd3c87de94ecbc0d1505de97167c47d0a2d7615a0164e52"
        ],
    },
    47: {
        "reason_class": "terminal_replica_incomplete",
        "evidence_receipt_sha256s": [
            "sha256:9bd0f1c415d32cdeae47744cbaa140c5e39781b7790ea318573d38772d8dd5bf",
            "sha256:a8d3ca88eaab62d4642ca3fab0fdf17b21a9eb0c92bb5de8971d8d0273928b0e",
            "sha256:7bf0cbc8715c51d6945bb133357f392646cf07ccc7e71a84dc887a30c902e083",
        ],
    },
    49: {
        "reason_class": "persisted_transcript_prefix_mismatch",
        "evidence_receipt_sha256s": [
            "sha256:e7b0031010f097f66dd3c87de94ecbc0d1505de97167c47d0a2d7615a0164e52"
        ],
    },
    50: {
        "reason_class": "persisted_transcript_prefix_mismatch",
        "evidence_receipt_sha256s": [
            "sha256:e7b0031010f097f66dd3c87de94ecbc0d1505de97167c47d0a2d7615a0164e52"
        ],
    },
    51: {
        "reason_class": "mixed_infrastructure_invalid_replica",
        "evidence_receipt_sha256s": [
            "sha256:572e330d1340e83d2aaf188665c0fd99b401e33eeec5f858717e2db60952bfa3"
        ],
    },
    52: {
        "reason_class": "mixed_infrastructure_invalid_replica",
        "evidence_receipt_sha256s": [
            "sha256:3a247e72e61da31593223d5fa46f902e4ff546e76277be3e27ef7407d91348c8",
            "sha256:e1718d127e4a2da1667eaa7d2c0a7d1d219e2e0bef3dcaabd4d46aa5f5b8f97b",
            "sha256:9a1296727060d303a3fe6d81fe51d2021482113c0dba8f4a9c4b4ba5efb1ccf3",
            "sha256:952bbaaabbb6c3a2fa105bb3780090d2ff8cc749345ba65904fc401ddb152aad",
        ],
    },
    53: {
        "reason_class": "replica_not_started",
        "evidence_receipt_sha256s": [
            "sha256:fa656de6282a504ad2570a65078625a5c35e79afcf5d4d120c0803a75912ff70",
            "sha256:ef6e969b305cecf8fa5bff882713d87bf3f8c21a44ced4f0ce54119c46c049e1",
            "sha256:9a1296727060d303a3fe6d81fe51d2021482113c0dba8f4a9c4b4ba5efb1ccf3",
            "sha256:952bbaaabbb6c3a2fa105bb3780090d2ff8cc749345ba65904fc401ddb152aad",
        ],
    },
}
RETIREMENT_AUTHORIZATION_GATE = (
    "do_not_delete_until_parent_confirms_protocol_v2_receipt_and_pr_accepted"
)
RETIREMENT_DELETE_ONLY = [
    "Job/chris-q38-dev17-s52-t3k32s1000-p1-v2@55233036-1d27-4b4a-9f46-9d8e8e04a36f",
    "Workload/job-chris-q38-dev17-s52-t3k32s1000-p1-v2-09558@87df4274-bb1e-4b20-9239-554d3ab8b230",
    "Job/chris-q38-dev17-s53-t3k32s1000-p1-v2@76ade7d1-9c35-4563-9e14-e690e05ae52e",
    "Workload/job-chris-q38-dev17-s53-t3k32s1000-p1-v2-918c5@4a11b78c-6a00-4f47-a6b5-6d13599404c9",
]
RETIREMENT_METHOD = (
    "re-read each Job and owned Workload; require same UID, "
    "suspended/unstarted/unadmitted/zero Pods; issue foreground Job DELETE with UID and "
    "current resourceVersion Preconditions; verify owner Workload is garbage-collected; "
    "fail closed on any drift"
)
RETIREMENT_PRESERVE_PLAN = [
    "both immutable ConfigMaps",
    "both absent output paths",
    "shared Secrets",
    "all other Jobs, Workloads, Pods, outputs and databases",
]
RETIREMENT_PRESERVED_SCOPE = [
    "ConfigMap/chris-q38-dev17-s52-t3k32s1000-code-v2@2d0b8155-031d-4f3d-943a-0e7308d147c0",
    "ConfigMap/chris-q38-dev17-s53-t3k32s1000-code-v2@b9e2e70c-780a-4203-9686-136b6928eea8",
    "both exact output paths (absent)",
    "both exact dedicated databases (absent)",
    "shared Secrets",
    "all other Jobs, Workloads, Pods, outputs and databases",
]


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
    if seeds != list(FROZEN_INVALID_SEEDS):
        raise ValueError("migration intent differs from the frozen invalid-replica roster")
    expected_rows = [
        {"seed": seed, **FROZEN_INVALID_REPLICA_EVIDENCE[seed]} for seed in FROZEN_INVALID_SEEDS
    ]
    if rows != expected_rows:
        raise ValueError("migration intent differs from the exact frozen evidence contract")
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


def _sealed_evaluation_plan(package: heldout_launch.Package) -> dict[str, Any]:
    """Compile with the exact evaluator/runtime bytes sealed in the packet ConfigMap."""
    data = package.config_map.get("data")
    if not isinstance(data, dict) or any(
        not isinstance(data.get(name), str) for name in SEALED_EVALUATOR_MODULES
    ):
        raise ValueError("evaluation packet lacks its complete sealed evaluator runtime")
    scientific_config = dict(package.evaluation_config)
    scientific_config.pop("model_artifact_binding", None)
    with tempfile.TemporaryDirectory(prefix="fleet-sealed-evaluator-") as directory:
        root = Path(directory)
        package_root = root / "evals" / "fleet"
        package_root.mkdir(parents=True, mode=0o700)
        (root / "evals" / "__init__.py").write_text("", encoding="utf-8")
        (package_root / "__init__.py").write_text("", encoding="utf-8")
        for name in SEALED_EVALUATOR_MODULES:
            (package_root / name).write_text(data[name], encoding="utf-8")
        task_set = data.get("task-set.json")
        task_set_name = scientific_config.get("task_set")
        if (
            not isinstance(task_set, str)
            or not isinstance(task_set_name, str)
            or not task_set_name
            or Path(task_set_name).name != task_set_name
        ):
            raise ValueError("evaluation packet lacks its sealed task set")
        (root / task_set_name).write_text(task_set, encoding="utf-8")
        (root / "config.json").write_text(
            json.dumps(scientific_config, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        command = (
            "import json; from pathlib import Path; from evals.fleet import evaluate; "
            "config=json.loads(Path('config.json').read_text(encoding='utf-8')); "
            "plan=evaluate.compile_eval(config, relative_to=Path('.')); "
            "print(json.dumps(plan, sort_keys=True, separators=(',', ':'), allow_nan=False))"
        )
        try:
            completed = subprocess.run(
                [sys.executable, "-c", command],
                cwd=root,
                env={
                    "PYTHONPATH": str(root),
                    "PYTHONNOUSERSITE": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            plan = json.loads(completed.stdout)
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
            raise ValueError("sealed evaluator could not compile its exact plan") from exc
    if not isinstance(plan, dict):
        raise ValueError("sealed evaluator returned an invalid plan")
    return plan


def _evaluation_binding(packet_path: Path) -> dict[str, Any]:
    package = heldout_launch.build_package(packet_path)
    plan = _sealed_evaluation_plan(package)
    if (
        package.packet.identity_sha256 != _canonical(package.packet.identity)
        or not isinstance(plan.get("sha256"), str)
        or len(plan["sha256"]) != 64
        or not isinstance(plan.get("runtime_files"), dict)
        or set(plan["runtime_files"]) != set(SEALED_RUNTIME_IDENTITY_FILES)
        or any(
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for digest in plan["runtime_files"].values()
        )
    ):
        raise ValueError("evaluation packet does not compile to an exact runtime identity")
    return {
        "evaluation_identity": package.packet.identity,
        "evaluation_identity_sha256": package.packet.identity_sha256,
        "evaluation_plan_sha256": "sha256:" + plan["sha256"],
        "runtime_files_sha256": plan["runtime_files"],
        "packet_file_sha256": _file_sha(packet_path),
    }


def _retained_arms(
    *,
    source_packets: Path,
    candidate_successor_packets: Path,
    invalid_seeds: list[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    receipt_path = candidate_successor_packets / "PREPARATION_RECEIPT.json"
    receipt = _verified(receipt_path, "candidate successor preparation receipt")
    rows = receipt.get("arms")
    if (
        receipt.get("sha256") != EXPECTED_CANDIDATE_SUCCESSOR_PREPARATION_SHA256
        or _file_sha(receipt_path) != EXPECTED_CANDIDATE_SUCCESSOR_PREPARATION_FILE_SHA256
        or receipt.get("schema") != "cyber_qwen38_fleet_seed46_candidate_successor_preparation_v1"
        or receipt.get("protocol_study_id") != source.PROTOCOL_STUDY_ID
        or receipt.get("successor_generation") != 2
        or receipt.get("seeds") != list(SOURCE_SEEDS)
        or receipt.get("external_mutations") != 0
        or receipt.get("launch_performed") is not False
        or receipt.get("credentials_read") is not False
        or receipt.get("scores_read") is not False
        or receipt.get("final8_opened") is not False
        or not isinstance(rows, list)
        or len(rows) != len(SOURCE_SEEDS)
    ):
        raise ValueError("candidate successor preparation receipt differs")
    rows_by_seed = {row.get("seed"): row for row in rows}
    if set(rows_by_seed) != set(SOURCE_SEEDS):
        raise ValueError("candidate successor packet roster differs")

    retained: list[dict[str, Any]] = []
    retained_seeds = sorted(set(SOURCE_SEEDS) - set(invalid_seeds))
    for seed in retained_seeds:
        pair = {
            "base": source_packets / f"seed{seed}" / "base" / "LAUNCH_PACKET.json",
            "candidate": (
                candidate_successor_packets / f"seed{seed}" / "candidate" / "LAUNCH_PACKET.json"
            ),
        }
        bindings = {arm: _evaluation_binding(path) for arm, path in pair.items()}
        candidate_receipt = rows_by_seed[seed]
        if (
            candidate_receipt.get("arm_id") != "candidate"
            or candidate_receipt.get("evaluation_identity_sha256")
            != bindings["candidate"]["evaluation_identity_sha256"]
            or candidate_receipt.get("packet_file_sha256")
            != bindings["candidate"]["packet_file_sha256"]
            or bindings["base"]["evaluation_identity"]["sampling_seed"] != seed
            or bindings["candidate"]["evaluation_identity"]["sampling_seed"] != seed
            or bindings["base"]["evaluation_identity"]["arm_id"] != "base"
            or bindings["candidate"]["evaluation_identity"]["arm_id"] != "candidate"
            or bindings["base"]["evaluation_identity"]["comparison_protocol_sha256"]
            != bindings["candidate"]["evaluation_identity"]["comparison_protocol_sha256"]
        ):
            raise ValueError("retained candidate successor differs from its sealed receipt")
        for arm in ARMS:
            retained.append({"seed": seed, "arm_id": arm, **bindings[arm]})
    return retained, {
        "path": _path_label(receipt_path),
        "file_sha256": _file_sha(receipt_path),
        "receipt_sha256": receipt["sha256"],
        "successor_generation": 2,
    }


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
        "predecessor_launch_receipt_sha256": PREDECESSOR_LAUNCH_RECEIPT_SHA256,
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
    if value["sha256"] == PREDECESSOR_LAUNCH_RECEIPT_SHA256:
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
    expected_preflight_fields = {
        "authorization_gate",
        "classification",
        "jobs",
        "kube_context",
        "mutation_performed",
        "namespace",
        "observed_at",
        "prompts_or_traces_included",
        "retirement_plan",
        "schema",
        "scores_included",
        "sha256",
    }
    expected_post_fields = {
        "classification",
        "kube_context",
        "namespace",
        "observed_at",
        "post_verification",
        "prompts_or_traces_included",
        "retirements",
        "schema",
        "scope",
        "scores_included",
        "sha256",
        "source_preflight",
    }
    if (
        _file_sha(preflight_path)
        != "sha256:0a36bf0f88ffe35200b4e0826f15c2cc6d9c49a3df536ae7fbd6c3748f1d7628"
        or preflight.get("sha256")
        != "sha256:9a1296727060d303a3fe6d81fe51d2021482113c0dba8f4a9c4b4ba5efb1ccf3"
        or _file_sha(post_path)
        != "sha256:661ecd1974f0f1fa79887e509732bae302b997c67d16d62be72dc1385ca1b22e"
        or post.get("sha256")
        != "sha256:952bbaaabbb6c3a2fa105bb3780090d2ff8cc749345ba65904fc401ddb152aad"
        or set(preflight) != expected_preflight_fields
        or set(post) != expected_post_fields
        or preflight.get("schema") != "cyber_fleet_eval_queued_job_retirement_preflight_v1"
        or preflight.get("classification") != "queued_unstarted_no_execution"
        or preflight.get("mutation_performed") is not False
        or preflight.get("scores_included") is not False
        or preflight.get("prompts_or_traces_included") is not False
        or preflight.get("authorization_gate") != RETIREMENT_AUTHORIZATION_GATE
        or preflight.get("retirement_plan")
        != {
            "delete_only": RETIREMENT_DELETE_ONLY,
            "method": RETIREMENT_METHOD,
            "preserve": RETIREMENT_PRESERVE_PLAN,
        }
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
        or post.get("scope")
        != {
            "deleted_only": RETIREMENT_DELETE_ONLY,
            "preserved": RETIREMENT_PRESERVED_SCOPE,
        }
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
        expected_delete_response = (
            {
                "accepted": True,
                "capture": (
                    "API returned before a local result-serialization AttributeError; "
                    "UID-preconditioned deletion is independently proven by exact "
                    "post-state absence"
                ),
            }
            if seed == 52
            else {"accepted": True, "kind": "Job"}
        )
        if (
            after.get("delete_response") != expected_delete_response
            or job_before.get("name") != job_after.get("name")
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


def _seed51_invalid_evidence() -> dict[str, Any]:
    value = _verified(SEED51_INVALID_EVIDENCE, "seed-51 invalid-replica evidence")
    classification = value.get("classification", {})
    privacy = value.get("privacy", {})
    residual = value.get("unrecoverable_residual", {})
    if (
        value.get("schema") != "cyber_qwen38_fleet_seed51_base_terminal_cohort_audit_v1"
        or value.get("sampling_seed") != 51
        or value.get("arm") != "base"
        or value.get("arm_census")
        != {
            "accepted": 7,
            "local_results": 16,
            "retry_review": 10,
            "stale_active": 0,
            "total": 17,
        }
        or classification.get("reason_class") != "mixed_infrastructure_invalid_replica"
        or classification.get("arm_valid_for_protocol_v1_comparison") is not False
        or classification.get("paired_replica_valid_under_protocol_v2_whole_replica_rule")
        is not False
        or classification.get("external_mutations_performed") != 0
        or classification.get("reroll_created") is not False
        or classification.get("subset_reconciliation_created") is not False
        or privacy.get("scores_read_or_included") is not False
        or privacy.get("prompts_responses_flags_rewards_or_trace_content_read_or_included")
        is not False
        or privacy.get("credentials_included") is not False
        or privacy.get("private_cell_task_session_or_trace_identifiers_included") is not False
        or residual.get("count") != 1
        or residual.get("failure_code") != "post_claim.runtimeerror"
        or residual.get("local_result_present") is not False
        or residual.get("database_session_binding_present") is not False
        or residual.get("recoverable_authoritative_outcome_proven") is not False
        or residual.get("retry_count") != 0
        or residual.get("max_retries") != 0
    ):
        raise ValueError("seed-51 invalid-replica evidence differs")
    return value


def _partial_recovery_hold_evidence() -> dict[str, Any]:
    value = _read(PARTIAL_RECOVERY_HOLD_EVIDENCE, "partial-recovery HOLD evidence")
    raw_self_digest = hashlib.sha256(
        json.dumps(
            {key: item for key, item in value.items() if key != "sha256"},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    if (
        set(value)
        != {
            "schema",
            "observed_at",
            "status",
            "affected_seeds",
            "infrastructure_class",
            "recovery",
            "resources",
            "decision",
            "private_evidence_binding",
            "privacy",
            "sha256",
        }
        or _file_sha(PARTIAL_RECOVERY_HOLD_EVIDENCE)
        != "sha256:d20b4fe2310de95cfd5561877f6d93133dcf7e5ed903f425addfb28bda2583ae"
        or value.get("sha256") != "e7b0031010f097f66dd3c87de94ecbc0d1505de97167c47d0a2d7615a0164e52"
        or value.get("sha256") != raw_self_digest
        or value.get("schema") != "cyber_qwen38_fleet_partial_session_terminal_hold_v1"
        or value.get("observed_at") != "2026-09-23T07:47:03Z"
        or value.get("status") != "terminal_hold"
        or value.get("affected_seeds") != [46, 49, 50]
        or value.get("infrastructure_class") != "persisted_transcript_prefix_mismatch"
        or value.get("recovery")
        != {
            "attempted_cells": 3,
            "suffix_chunks_appended": 0,
            "scoring_calls": 0,
            "model_replays": 0,
            "verifier_replays": 0,
            "ledger_writes": 0,
            "exact_recovery_safe": False,
            "hold_not_guess": True,
        }
        or value.get("resources")
        != {
            "cpu_jobs_terminal": 3,
            "cpu_pods_terminal": 3,
            "cpu_workloads_terminal": 3,
            "all_cpu_resources_released": True,
            "gpu_resources_requested": 0,
        }
        or value.get("decision")
        != {
            "retry_these_identities": False,
            "reroll_missing_base_cells": False,
            "exclude_whole_replica_pairs": True,
            "exclude_both_arms_for_each_seed": True,
        }
        or value.get("private_evidence_binding")
        != {
            "source_file_sha256": (
                "35e96255b827d7c8502fa06a44a010b4c677deae2a1a65bb8aa126d6e5e03e01"
            ),
            "source_receipt_sha256": (
                "45893e380d1d9755cbe514ec85628b7fec930d9e2bf5e61920967155112bda09"
            ),
        }
        or value.get("privacy")
        != {
            "credentials_included": False,
            "task_or_session_ids_included": False,
            "scores_included": False,
            "prompts_responses_flags_answers_or_trace_content_included": False,
            "private_provider_ids_included": False,
        }
    ):
        raise ValueError("partial-recovery HOLD evidence differs")
    return value


def _daily_budget_evidence(retirement: dict[str, Any]) -> dict[str, Any]:
    source_launch = _verified(SOURCE_LAUNCH_EVIDENCE, "source launch evidence")
    source_bindings = _verified(SOURCE_CREATE_BINDINGS_EVIDENCE, "source create bindings evidence")
    successor_create = _verified(
        CANDIDATE_SUCCESSOR_CREATE_EVIDENCE, "candidate successor create evidence"
    )
    source_failure = successor_create.get("source_failure", {})
    packet_preparation = successor_create.get("packet_preparation", {})
    successor_rows = successor_create.get("rows")
    privacy = successor_create.get("privacy", {})
    source_evaluators = source_launch.get("evaluators")
    source_binding_rows = source_bindings.get("rows")
    launch_bindings = (
        {
            (row.get("seed"), row.get("arm"), row.get("job_name"), row.get("job_uid"))
            for row in source_evaluators
        }
        if isinstance(source_evaluators, list)
        else set()
    )
    created_bindings = (
        {
            (row.get("seed"), row.get("arm"), row.get("job_name"), row.get("job_uid"))
            for row in source_binding_rows
        }
        if isinstance(source_binding_rows, list)
        else set()
    )
    if (
        _file_sha(SOURCE_LAUNCH_EVIDENCE)
        != "sha256:e7acd1517e214aecf60b8d1c7e5d31fe5016d91591c37af11322dc92da0a339f"
        or source_launch.get("sha256")
        != "sha256:1154b450624a8b9a567464916da95874c44933175c408b86a80ff5bca0eada70"
        or source_launch.get("schema") != "cyber_qwen38_fleet_dev17_seed46to53_pass8_launch_v1"
        or not str(source_launch.get("observed_at", "")).startswith("2026-09-23T")
        or source_launch.get("study", {}).get("seeds") != list(SOURCE_SEEDS)
        or source_launch.get("study", {}).get("sessions_per_arm") != 136
        or source_launch.get("study", {}).get("total_sessions") != 272
        or not isinstance(source_evaluators, list)
        or len(source_evaluators) != 16
        or _file_sha(SOURCE_CREATE_BINDINGS_EVIDENCE)
        != "sha256:62c27e21cc6cdf3d9773617aaa7aa082cbc9fb6caf62d73428c7508960a1bce3"
        or source_bindings.get("sha256")
        != "sha256:e76ae2aac0c370999d43de1caa4a76e704f4eb410fe7fdef623c705877470768"
        or source_bindings.get("schema") != "cyber_qwen38_fleet_dev17_pass8_create_bindings_v1"
        or not str(source_bindings.get("observed_at", "")).startswith("2026-09-23T")
        or source_bindings.get("binding_count") != 16
        or source_bindings.get("external_mutations") != 0
        or source_bindings.get("credentials_included") is not False
        or source_bindings.get("score_values_included") is not False
        or source_bindings.get("prompts_responses_flags_rewards_or_trace_content_included")
        is not False
        or not isinstance(source_binding_rows, list)
        or len(source_binding_rows) != 16
        or created_bindings != launch_bindings
        or _file_sha(CANDIDATE_SUCCESSOR_CREATE_EVIDENCE)
        != "sha256:924ce77582cc3e8370fc8a4ca1a7cbdfa8edcd69c10e14b373918cbd5b0410fc"
        or successor_create.get("sha256")
        != "sha256:da2a8d3e40e7c8c925164300a66976626b4c36329f2ffe20ad5627846f0f61c8"
        or successor_create.get("schema")
        != "cyber_qwen38_fleet_dev17_candidate_successor_create_v1"
        or not str(successor_create.get("observed_at", "")).startswith("2026-09-23T")
        or source_failure.get("seeds") != list(SOURCE_SEEDS)
        or source_failure.get("uniform_class") != "missing_local_model_artifact_binding"
        or source_failure.get("all_source_jobs_terminal_failed") is not True
        or source_failure.get("all_source_databases_zero_rows") is not True
        or source_failure.get("all_source_outputs_absent") is not True
        or source_failure.get("candidate_sessions_created") != 0
        or source_failure.get("valid_outcomes_replayed") is not False
        or packet_preparation.get("seed_count") != 8
        or packet_preparation.get("candidate_sessions") != 136
        or packet_preparation.get("pass_k_per_seed") != 1
        or packet_preparation.get("retries") != 0
        or not isinstance(successor_rows, list)
        or [row.get("seed") for row in successor_rows] != list(SOURCE_SEEDS)
        or privacy
        != {
            "credentials_included": False,
            "prompts_responses_flags_answers_scores_or_trace_content_included": False,
            "scores_read": False,
        }
        or retirement.get("model_rollouts") != 0
        or [row.get("seed") for row in retirement.get("targets", [])] != [52, 53]
        or retirement.get("outputs_or_databases_deleted") is not False
    ):
        raise ValueError("daily budget source evidence differs from the sealed campaign")

    original_base = len(SOURCE_SEEDS) * TASKS_PER_ARM
    failed_original_candidate = 0
    retired_successor_seeds = [row["seed"] for row in retirement["targets"]]
    successor_seeds = sorted(set(SOURCE_SEEDS) - set(retired_successor_seeds))
    successor_reserved = len(successor_seeds) * TASKS_PER_ARM
    replacements = len(FROZEN_INVALID_SEEDS) * len(ARMS) * TASKS_PER_ARM
    total = original_base + failed_original_candidate + successor_reserved + replacements
    value = {
        "schema": DAILY_BUDGET_EVIDENCE_SCHEMA,
        "budget_date_utc": "2026-09-23",
        "scope": PROTOCOL_V2_STUDY_ID,
        "authoritative_inputs": {
            "source_launch": {
                "path": _path_label(SOURCE_LAUNCH_EVIDENCE),
                "file_sha256": _file_sha(SOURCE_LAUNCH_EVIDENCE),
                "receipt_sha256": source_launch["sha256"],
            },
            "source_create_bindings": {
                "path": _path_label(SOURCE_CREATE_BINDINGS_EVIDENCE),
                "file_sha256": _file_sha(SOURCE_CREATE_BINDINGS_EVIDENCE),
                "receipt_sha256": source_bindings["sha256"],
            },
            "candidate_successor_create": {
                "path": _path_label(CANDIDATE_SUCCESSOR_CREATE_EVIDENCE),
                "file_sha256": _file_sha(CANDIDATE_SUCCESSOR_CREATE_EVIDENCE),
                "receipt_sha256": successor_create["sha256"],
            },
            "candidate_seed52_53_retirement": {
                "receipt_sha256": retirement["sha256"],
                "model_rollouts": 0,
            },
        },
        "line_items": {
            "original_base_seed46_to_53": original_base,
            "failed_original_candidate_pre_model": failed_original_candidate,
            "candidate_successor_nonretired_seeds": successor_seeds,
            "candidate_successor_nonretired_rollouts_reserved": successor_reserved,
            "candidate_successor_retired_before_start_seeds": retired_successor_seeds,
            "candidate_successor_retired_before_start_rollouts": 0,
            "protocol_v2_whole_pair_replacements_planned": replacements,
            "scoring_or_metadata_cpu_model_rollouts": 0,
        },
        "cumulative_model_rollouts_consumed_or_reserved": total,
        "accounting_kind": "campaign_local_conservative_upper_bound",
        "daily_rollout_cap": DAILY_ROLLOUT_CAP,
        "daily_rollout_cap_scope": "this_protocol_v2_campaign",
        "unrelated_campaigns_included": False,
        "within_daily_cap": total <= DAILY_ROLLOUT_CAP,
        "score_values_read": False,
        "prompts_responses_flags_rewards_or_trace_content_read": False,
        "external_mutations": 0,
    }
    value["sha256"] = _canonical(value)
    return value


def _configs(base_template: dict[str, Any], seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    base, candidate = source._configs(base_template, seed, candidate_generation=2)  # noqa: SLF001
    base["name"] = f"q38-s{seed}-base-repl-p1-v2"
    candidate["name"] = f"q38-s{seed}-t3k32s1000-repl-p1-v2"
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
    candidate_successor_packets: Path,
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
    if (
        source_receipt.get("sha256") != EXPECTED_SOURCE_PREPARATION_RECEIPT_SHA256
        or _file_sha(source_receipt_path) != EXPECTED_SOURCE_PREPARATION_RECEIPT_FILE_SHA256
    ):
        raise ValueError("source preparation receipt differs from the exact launched study")
    inventory = _source_inventory(source_packets, source_receipt)
    intent = _load_intent(migration_intent, source_receipt, source_receipt_path)
    invalid = intent["invalid_original_replicas"]
    retained_arms, retained_candidate_source = _retained_arms(
        source_packets=source_packets,
        candidate_successor_packets=candidate_successor_packets,
        invalid_seeds=[row["seed"] for row in invalid],
    )
    mapping = deterministic_mapping([row["seed"] for row in invalid])
    reason_by_seed = {row["seed"]: row for row in invalid}
    seed51_evidence = _seed51_invalid_evidence()
    partial_hold_evidence = _partial_recovery_hold_evidence()
    if (
        seed51_evidence["sha256"] not in reason_by_seed[51]["evidence_receipt_sha256s"]
        or reason_by_seed[51]["reason_class"] != seed51_evidence["classification"]["reason_class"]
    ):
        raise ValueError("migration intent does not bind the exact seed-51 invalid evidence")
    partial_hold_sha256 = "sha256:" + partial_hold_evidence["sha256"]
    for seed in partial_hold_evidence["affected_seeds"]:
        if (
            partial_hold_sha256 not in reason_by_seed[seed]["evidence_receipt_sha256s"]
            or reason_by_seed[seed]["reason_class"] != partial_hold_evidence["infrastructure_class"]
        ):
            raise ValueError("migration intent does not bind the exact partial-recovery HOLD")
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
                        **_evaluation_binding(packet_path),
                        "invalid_original_seed": original_seed,
                        "replacement_seed": seed,
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
        daily_budget_evidence = _daily_budget_evidence(retirement_evidence)
        source._write_json(  # noqa: SLF001
            temporary / "DAILY_BUDGET_EVIDENCE.json", daily_budget_evidence
        )
        planned = daily_budget_evidence["line_items"]["protocol_v2_whole_pair_replacements_planned"]
        cumulative_rollouts = daily_budget_evidence[
            "cumulative_model_rollouts_consumed_or_reserved"
        ]
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "source": {
                "protocol_study_id": source.PROTOCOL_STUDY_ID,
                "preparation_receipt_sha256": source_receipt["sha256"],
                "preparation_receipt_file_sha256": _file_sha(source_receipt_path),
                "seeds": list(SOURCE_SEEDS),
                "arm_count": len(SOURCE_SEEDS) * len(ARMS),
                "retained_candidate_successor_preparation": retained_candidate_source,
            },
            "migration_intent_sha256": intent["sha256"],
            "sanitized_invalid_replica_evidence": [
                {"seed": seed, **FROZEN_INVALID_REPLICA_EVIDENCE[seed]}
                for seed in FROZEN_INVALID_SEEDS
            ],
            "checked_in_seed51_invalid_evidence": {
                "path": str(SEED51_INVALID_EVIDENCE.relative_to(ROOT)),
                "file_sha256": _file_sha(SEED51_INVALID_EVIDENCE),
                "receipt_sha256": seed51_evidence["sha256"],
            },
            "checked_in_partial_recovery_hold_evidence": {
                "path": str(PARTIAL_RECOVERY_HOLD_EVIDENCE.relative_to(ROOT)),
                "file_sha256": _file_sha(PARTIAL_RECOVERY_HOLD_EVIDENCE),
                "receipt_sha256": partial_hold_sha256,
                "source_private_receipt_sha256": (
                    "sha256:"
                    + partial_hold_evidence["private_evidence_binding"]["source_receipt_sha256"]
                ),
            },
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
            "retained_arms": retained_arms,
            "retirement_evidence": {
                "sha256": retirement_evidence["sha256"],
                "file_sha256": _file_sha(temporary / "RETIREMENT_EVIDENCE.json"),
                "targets": len(retirement_evidence["targets"]),
                "model_rollouts": 0,
                "outputs_or_databases_deleted": False,
            },
            "daily_budget_evidence": {
                "sha256": daily_budget_evidence["sha256"],
                "file_sha256": _file_sha(temporary / "DAILY_BUDGET_EVIDENCE.json"),
                "budget_date_utc": daily_budget_evidence["budget_date_utc"],
                "scope": daily_budget_evidence["scope"],
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
                "later_invalid_seed_requires_versioned_successor_before_score_unseal": True,
            },
            "capacity": {
                "new_replacement_rollouts": planned,
                "final_comparison_rollouts": (TASKS_PER_ARM * len(SOURCE_SEEDS) * len(ARMS)),
                "original_base_rollouts": daily_budget_evidence["line_items"][
                    "original_base_seed46_to_53"
                ],
                "original_candidate_pre_model_rollouts": daily_budget_evidence["line_items"][
                    "failed_original_candidate_pre_model"
                ],
                "candidate_successor_nonretired_seeds": daily_budget_evidence["line_items"][
                    "candidate_successor_nonretired_seeds"
                ],
                "candidate_successor_nonretired_rollouts_reserved": daily_budget_evidence[
                    "line_items"
                ]["candidate_successor_nonretired_rollouts_reserved"],
                "candidate_successor_retired_before_start_seeds": daily_budget_evidence[
                    "line_items"
                ]["candidate_successor_retired_before_start_seeds"],
                "candidate_successor_retired_before_start_rollouts": daily_budget_evidence[
                    "line_items"
                ]["candidate_successor_retired_before_start_rollouts"],
                "scoring_or_metadata_cpu_model_rollouts": daily_budget_evidence["line_items"][
                    "scoring_or_metadata_cpu_model_rollouts"
                ],
                "cumulative_model_rollouts_consumed_or_planned_today": cumulative_rollouts,
                "daily_rollout_cap": daily_budget_evidence["daily_rollout_cap"],
                "within_daily_cap": daily_budget_evidence["within_daily_cap"],
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
    parser.add_argument("--candidate-successor-packets", type=Path, required=True)
    parser.add_argument("--migration-intent", type=Path, required=True)
    parser.add_argument("--live-parity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                source_packets=args.source_packets,
                candidate_successor_packets=args.candidate_successor_packets,
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
