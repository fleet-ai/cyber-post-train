#!/usr/bin/env python3
"""Render provider-free launchers for one sealed protocol-v2 replacement set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import yaml

from evals.fleet import heldout_launch
from scripts import prepare_qwen38_fleet_protocol_v2_replacements as protocol_v2
from scripts import render_fleet_heldout_launcher_jobs as shared

ROOT = Path(__file__).resolve().parents[1]
RENDER_SCHEMA = "cyber_qwen38_fleet_protocol_v2_launcher_render_v1"


def _canonical(value: Any) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    )


def _file_sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _migration(packet_root: Path) -> dict[str, Any]:
    path = packet_root / "MIGRATION_RECEIPT.json"
    value = protocol_v2._verified(path, "protocol-v2 migration receipt")  # noqa: SLF001
    arms = value.get("replacement_arms")
    retained_arms = value.get("retained_arms")
    migrations = value.get("migrations")
    definition = value.get("comparison_definition")
    seed51 = protocol_v2._seed51_invalid_evidence()  # noqa: SLF001
    partial_hold = protocol_v2._partial_recovery_hold_evidence()  # noqa: SLF001
    expected_invalid_evidence = [
        {"seed": seed, **protocol_v2.FROZEN_INVALID_REPLICA_EVIDENCE[seed]}
        for seed in protocol_v2.FROZEN_INVALID_SEEDS
    ]
    expected_checked_in_seed51 = {
        "path": str(protocol_v2.SEED51_INVALID_EVIDENCE.relative_to(ROOT)),
        "file_sha256": _file_sha(protocol_v2.SEED51_INVALID_EVIDENCE),
        "receipt_sha256": seed51["sha256"],
    }
    expected_checked_in_partial_hold = {
        "path": str(protocol_v2.PARTIAL_RECOVERY_HOLD_EVIDENCE.relative_to(ROOT)),
        "file_sha256": _file_sha(protocol_v2.PARTIAL_RECOVERY_HOLD_EVIDENCE),
        "receipt_sha256": "sha256:" + partial_hold["sha256"],
        "source_private_receipt_sha256": (
            "sha256:" + partial_hold["private_evidence_binding"]["source_receipt_sha256"]
        ),
    }
    expected_privacy = {
        "score_values_read": False,
        "prompts_responses_flags_rewards_or_trace_content_read": False,
        "infrastructure_reason_classes_only": True,
    }
    if (
        value.get("schema") != protocol_v2.RECEIPT_SCHEMA
        or not isinstance(arms, list)
        or not isinstance(retained_arms, list)
        or not isinstance(migrations, list)
        or len(arms) != len(migrations) * len(protocol_v2.ARMS)
        or len(retained_arms)
        != (len(protocol_v2.SOURCE_SEEDS) - len(protocol_v2.FROZEN_INVALID_SEEDS))
        * len(protocol_v2.ARMS)
        or value.get("scientific_identity", {}).get("whole_replica_pairs_only") is not True
        or value.get("scientific_identity", {}).get("retry_limit") != 0
        or value.get("privacy") != expected_privacy
        or value.get("external_mutations") != 0
        or value.get("launch_performed") is not False
        or value.get("sanitized_invalid_replica_evidence") != expected_invalid_evidence
        or value.get("checked_in_seed51_invalid_evidence") != expected_checked_in_seed51
        or value.get("checked_in_partial_recovery_hold_evidence")
        != expected_checked_in_partial_hold
        or not isinstance(definition, dict)
        or definition.get("schema") != protocol_v2.COMPARISON_DEFINITION_SCHEMA
        or definition.get("sha256")
        != protocol_v2._canonical(  # noqa: SLF001
            {key: item for key, item in definition.items() if key != "sha256"}
        )
        or definition.get("included_seeds") != value.get("included_seeds")
        or len(definition.get("included_seeds", [])) != len(protocol_v2.SOURCE_SEEDS)
        or definition.get("sha256") == protocol_v2.PREDECESSOR_LAUNCH_RECEIPT_SHA256
    ):
        raise ValueError("protocol-v2 migration receipt is invalid")
    for label, rows in (("replacement", arms), ("retained", retained_arms)):
        for row in rows:
            identity = row.get("evaluation_identity")
            plan = row.get("evaluation_plan_sha256")
            runtime = row.get("runtime_files_sha256")
            if (
                not isinstance(identity, dict)
                or row.get("evaluation_identity_sha256") != protocol_v2._canonical(identity)  # noqa: SLF001
                or not isinstance(plan, str)
                or len(plan) != 71
                or not plan.startswith("sha256:")
                or not isinstance(runtime, dict)
                or set(runtime) != set(protocol_v2.SEALED_RUNTIME_IDENTITY_FILES)
                or any(
                    not isinstance(digest, str)
                    or len(digest) != 64
                    or any(character not in "0123456789abcdef" for character in digest)
                    for digest in runtime.values()
                )
            ):
                raise ValueError(f"protocol-v2 {label} arm runtime identity is invalid")
    definition_path = packet_root / "COMPARISON_DEFINITION.json"
    retirement_path = packet_root / "RETIREMENT_EVIDENCE.json"
    daily_budget_path = packet_root / "DAILY_BUDGET_EVIDENCE.json"
    retirement = protocol_v2._verified(  # noqa: SLF001
        retirement_path, "protocol-v2 retirement evidence"
    )
    daily_budget = protocol_v2._verified(  # noqa: SLF001
        daily_budget_path, "protocol-v2 daily budget evidence"
    )
    if (
        protocol_v2._verified(  # noqa: SLF001
            definition_path, "protocol-v2 comparison definition"
        )
        != definition
        or _file_sha(definition_path) != value.get("comparison_definition_file_sha256")
        or retirement != protocol_v2._retirement_evidence()  # noqa: SLF001
        or retirement.get("sha256") != value.get("retirement_evidence", {}).get("sha256")
        or _file_sha(retirement_path) != value.get("retirement_evidence", {}).get("file_sha256")
        or daily_budget != protocol_v2._daily_budget_evidence(retirement)  # noqa: SLF001
        or value.get("daily_budget_evidence")
        != {
            "sha256": daily_budget["sha256"],
            "file_sha256": _file_sha(daily_budget_path),
            "budget_date_utc": daily_budget["budget_date_utc"],
            "scope": daily_budget["scope"],
        }
    ):
        raise ValueError("protocol-v2 auxiliary receipts differ")
    return value


def _assert_exact_packet_roster(packet_path: Path) -> None:
    raw = protocol_v2._read(packet_path, "replacement launch packet")  # noqa: SLF001
    files = raw.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("replacement packet file roster is invalid")
    expected = {"LAUNCH_PACKET.json"}
    for label, binding in files.items():
        if not isinstance(label, str) or not isinstance(binding, dict):
            raise ValueError("replacement packet file binding is invalid")
        relative = binding.get("path")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or Path(relative).parent != Path(".")
            or Path(relative).name != relative
        ):
            raise ValueError("replacement packet file binding leaves its sealed directory")
        expected.add(relative)
    actual = set()
    for child in packet_path.parent.iterdir():
        if child.is_symlink() or not child.is_file():
            raise ValueError("replacement packet directory contains a non-regular entry")
        actual.add(child.name)
    if actual != expected:
        raise ValueError("replacement packet directory contains an unexpected or missing file")


def _runtime_hashes() -> dict[str, str]:
    paths = [*shared.SOURCE_ROOTS, *sorted((ROOT / "evals" / "fleet").glob("*.py"))]
    return {str(path.relative_to(ROOT)): _file_sha(path) for path in paths}


def render(*, packets: Path, output: Path) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise FileExistsError("replacement launcher output already exists")
    if not output.parent.is_dir():
        raise ValueError("replacement launcher output parent does not exist")
    migration = _migration(packets)
    expected = {
        (row["replacement_seed"], row["arm_id"]): row for row in migration["replacement_arms"]
    }
    if len(expected) != len(migration["replacement_arms"]):
        raise ValueError("replacement arm receipt repeats an identity")
    packet_paths = sorted(packets.glob("seed*/base/LAUNCH_PACKET.json")) + sorted(
        packets.glob("seed*/candidate/LAUNCH_PACKET.json")
    )
    if len(packet_paths) != len(expected):
        raise ValueError("replacement packet roster is incomplete")
    temporary = Path(tempfile.mkdtemp(prefix=".fleet-protocol-v2-launchers-", dir=output.parent))
    try:
        items: list[dict[str, Any]] = []
        arms: list[dict[str, Any]] = []
        for packet_path in packet_paths:
            relative = packet_path.relative_to(packets)
            seed_text, arm = relative.parts[:2]
            if not seed_text.startswith("seed"):
                raise ValueError("replacement packet seed directory is invalid")
            seed = int(seed_text.removeprefix("seed"))
            binding = expected.get((seed, arm))
            if binding is None:
                raise ValueError("replacement packet is absent from the migration receipt")
            _assert_exact_packet_roster(packet_path)
            package = heldout_launch.build_package(packet_path)
            identity = package.packet.identity
            if (
                _file_sha(packet_path) != binding["packet_file_sha256"]
                or package.packet.identity_sha256 != binding["evaluation_identity_sha256"]
                or package.packet.identity != binding["evaluation_identity"]
                or identity["comparison_protocol_file_sha256"]
                != binding["comparison_protocol_file_sha256"]
                or identity["comparison_protocol_sha256"] != binding["comparison_protocol_sha256"]
                or identity["sampling_seed"] != seed
                or identity["arm_id"] != arm
                or identity["retry_limit"] != 0
            ):
                raise ValueError("replacement packet differs from the migration receipt")
            plan = protocol_v2._sealed_evaluation_plan(package)  # noqa: SLF001
            if (
                "sha256:" + plan["sha256"] != binding["evaluation_plan_sha256"]
                or plan["runtime_files"] != binding["runtime_files_sha256"]
            ):
                raise ValueError("replacement evaluator plan differs from the migration receipt")
            compressed, evaluator_job = shared._bundle(packet_path.parent)  # noqa: SLF001
            config_map, job = shared._objects(  # noqa: SLF001
                replica=f"seed{seed}-{arm}",
                compressed=compressed,
                evaluator_job=evaluator_job,
                launch_journal_id=evaluator_job,
            )
            if (
                job["metadata"]["annotations"].get(heldout_launch.FAILURE_ALERT_ANNOTATION)
                != heldout_launch.FAILURE_ALERT_OFF
                or job["spec"]["template"]["spec"].get("priorityClassName") != "c1"
                or "nvidia.com/gpu" in json.dumps(job)
            ):
                raise ValueError("replacement launcher resource policy differs")
            items.extend([config_map, job])
            arms.append(
                {
                    "replacement_seed": seed,
                    "arm_id": arm,
                    "evaluation_identity_sha256": package.packet.identity_sha256,
                    "evaluator_job": evaluator_job,
                    "launcher_job": job["metadata"]["name"],
                    "bundle_sha256": "sha256:" + hashlib.sha256(compressed).hexdigest(),
                    "root_failure_alerts": "off",
                    "priority_class": "c1",
                    "gpu_requests": 0,
                }
            )
        bundle = {"apiVersion": "v1", "kind": "List", "items": items}
        bundle_path = temporary / "launchers.yaml"
        bundle_path.write_text(yaml.safe_dump(bundle, sort_keys=False), encoding="utf-8")
        receipt = {
            "schema": RENDER_SCHEMA,
            "migration_receipt_sha256": migration["sha256"],
            "migration_receipt_file_sha256": _file_sha(packets / "MIGRATION_RECEIPT.json"),
            "comparison_definition_sha256": migration["comparison_definition"]["sha256"],
            "included_seeds": migration["included_seeds"],
            "bundle_path": "launchers.yaml",
            "bundle_file_sha256": _file_sha(bundle_path),
            "runtime_module_sha256": _runtime_hashes(),
            "launcher_program_sha256": "sha256:"
            + hashlib.sha256(shared.LAUNCHER.encode()).hexdigest(),
            "arms": sorted(arms, key=lambda row: (row["replacement_seed"], row["arm_id"])),
            "create_gates": {
                "identical_server_previews_required": 2,
                "duplicate_census_before_first_preview": True,
                "duplicate_census_after_second_preview": True,
                "output_database_kubernetes_and_ledger_absence_required": True,
                "authoritative_completed_session_required_before_cell_acceptance": True,
                "one_create_call_per_arm": True,
            },
            "external_mutations": 0,
            "launch_performed": False,
        }
        receipt["sha256"] = _canonical(receipt)
        (temporary / "RENDER.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(packets=args.packets, output=args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
