#!/usr/bin/env python3
"""Prepare deferred seed-65/66 matched pairs without claiming launch readiness.

This provider-free command extends the frozen seed-62/63/64 successor packet
with the already-declared seed-65 pair and the seed-66 whole-pair replacement
for infrastructure-invalid seed 60. It renders immutable inner evaluator
packets and create-once outer launcher objects only. It does not inspect the
current UTC budget, preview a server object, or create anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from evals.fleet import heldout_launch
from scripts import prepare_qwen38_fleet_protocol_v2_replacements as protocol_v2
from scripts import prepare_qwen38_fleet_protocol_v2_seed62_seed63_seed64_successor as successor
from scripts import prepare_qwen38_fleet_seed46_pass8_packets as source
from scripts import render_fleet_heldout_launcher_jobs as launcher

SCHEMA = "cyber_qwen38_fleet_protocol_v2_seed65_seed66_pair_preparation_v2"
DEFERRED_SEEDS = (65, 66)
FROZEN_DEFERRED_SEED = 65
FROZEN_RECEIPT_FILE_SHA256 = (
    "sha256:4708988c3ed2b4c1423b37f066ce94f28b954c8ebeba6dbdef097043e7dd706c"
)
FROZEN_RECEIPT_SHA256 = "sha256:c0a1602053e8865359ed57a4ab7eff7cad16854d4bd643b4e19564743300d57c"
FROZEN_DEFINITION_FILE_SHA256 = (
    "sha256:a44771f30f7a2bfb80faf77cb62bcb1663a8db4df038d46c28e50213827d918a"
)
FROZEN_DEFINITION_SHA256 = "sha256:e6dd12b097062eb4f5b775e7a7043dea3b0b0aca9e3407242ed8683722a6580e"
FROZEN_PROTOCOL_FILE_SHA256 = (
    "sha256:f9ef38fffc005e8bafa874b4cf6e93223777e1bda70249f794a70acdd545aaff"
)
FROZEN_PROTOCOL_SHA256 = "sha256:7ff2c0764b9f6abb8c87e90020ef773849c3674be10be55e0bde113dbf3c918e"
SEED60_RETIREMENT_FILE_SHA256 = (
    "sha256:f0c716088d38c4b149a167a9322cab03f0113558c0279ac20c7135fb5e8d8428"
)
SEED60_RETIREMENT_SHA256 = "sha256:4f8d76d751fe6d1f6697c1d8a89d01cb17f3d04a6cdfe21e67dc08f59a23b5a1"
FROZEN_INCLUDED_SEEDS = (48, 54, 60, 61, 62, 63, 64, 65)
INCLUDED_SEEDS = (48, 54, 61, 62, 63, 64, 65, 66)
PRIOR_RENDERED_SEEDS = (62, 63, 64)
ROLLOUTS = protocol_v2.TASKS_PER_ARM * len(protocol_v2.ARMS) * len(DEFERRED_SEEDS)


def _file_sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _frozen_source(
    root: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, dict[str, Any]],
    Path,
]:
    receipt_path = root / "SUCCESSOR_RECEIPT.json"
    definition_path = root / "COMPARISON_DEFINITION.json"
    protocol_path = (
        root / "seed65" / ("qwen38-fleet-dev17-seed65-base-step1000-replacement-protocol-v2.json")
    )
    receipt = protocol_v2._verified(receipt_path, "frozen seed-62/63/64 receipt")  # noqa: SLF001
    definition = protocol_v2._verified(  # noqa: SLF001
        definition_path, "frozen seed-62/63/64 comparison"
    )
    protocol = protocol_v2._verified(protocol_path, "frozen seed-65 protocol")  # noqa: SLF001
    protocol_row = next(
        (
            row
            for row in receipt.get("replacement_protocols", [])
            if row.get("seed") == FROZEN_DEFERRED_SEED
        ),
        None,
    )
    state = receipt.get("full_comparison_packet_state", {})
    if (
        _file_sha(receipt_path) != FROZEN_RECEIPT_FILE_SHA256
        or receipt.get("sha256") != FROZEN_RECEIPT_SHA256
        or _file_sha(definition_path) != FROZEN_DEFINITION_FILE_SHA256
        or definition.get("sha256") != FROZEN_DEFINITION_SHA256
        or _file_sha(protocol_path) != FROZEN_PROTOCOL_FILE_SHA256
        or protocol.get("sha256") != FROZEN_PROTOCOL_SHA256
        or receipt.get("comparison_definition") != definition
        or receipt.get("included_seeds") != list(FROZEN_INCLUDED_SEEDS)
        or definition.get("included_seeds") != list(FROZEN_INCLUDED_SEEDS)
        or receipt.get("external_mutations") != 0
        or receipt.get("server_preview_performed") is not False
        or receipt.get("launch_performed") is not False
        or state.get("rendered_successor_seeds") != list(PRIOR_RENDERED_SEEDS)
        or state.get("deferred_successor_seeds") != [FROZEN_DEFERRED_SEED]
        or state.get("missing_packet_cells")
        != [{"seed": FROZEN_DEFERRED_SEED, "arm_id": arm} for arm in protocol_v2.ARMS]
        or state.get("full_roster_launch_ready") is not False
        or protocol_row
        != {
            "seed": FROZEN_DEFERRED_SEED,
            "protocol_id": protocol["protocol_id"],
            "sha256": FROZEN_PROTOCOL_SHA256,
            "file_sha256": FROZEN_PROTOCOL_FILE_SHA256,
        }
    ):
        raise ValueError("frozen seed-62/63/64 successor bytes differ")

    rows = {
        (row.get("successor_seed"), row.get("arm_id")): row
        for row in receipt.get("replacement_arms", [])
        if isinstance(row, dict)
    }
    if set(rows) != {(seed, arm) for seed in PRIOR_RENDERED_SEEDS for arm in protocol_v2.ARMS}:
        raise ValueError("frozen successor arm roster differs")
    contracts: dict[str, dict[str, Any]] = {}
    proof_path: Path | None = None
    proof_sha256: str | None = None
    for seed in PRIOR_RENDERED_SEEDS:
        for arm in protocol_v2.ARMS:
            packet_path = root / f"seed{seed}" / arm / "LAUNCH_PACKET.json"
            row = rows[(seed, arm)]
            package = successor._validate_successor_package(packet_path)  # noqa: SLF001
            binding = protocol_v2._evaluation_binding(packet_path)  # noqa: SLF001
            treatment = {
                "config": successor._frozen_config_contract(package.evaluation_config),  # noqa: SLF001
                "runtime_files_sha256": binding["runtime_files_sha256"],
            }
            if (
                binding["packet_file_sha256"] != row.get("packet_file_sha256")
                or binding["evaluation_identity_sha256"] != row.get("evaluation_identity_sha256")
                or package.packet.identity.get("sampling_seed") != seed
                or package.packet.identity.get("arm_id") != arm
                or package.packet.identity.get("harness") != "opencode"
                or package.packet.identity.get("pass_k") != 1
                or package.packet.identity.get("retry_limit") != 0
                or (arm in contracts and contracts[arm] != treatment)
            ):
                raise ValueError("frozen successor packet differs")
            contracts.setdefault(arm, treatment)
            candidate_proof = package.packet.files["serving_route_proof"]
            candidate_sha256 = _file_sha(candidate_proof)
            if proof_sha256 is not None and candidate_sha256 != proof_sha256:
                raise ValueError("frozen successor arms do not share one parity proof")
            proof_sha256 = candidate_sha256
            proof_path = candidate_proof
    if set(contracts) != set(protocol_v2.ARMS) or proof_path is None:
        raise ValueError("frozen successor lacks a complete scientific treatment")
    return receipt, definition, protocol, contracts, proof_path


def _validate_frozen_parity(proof_path: Path, configs: dict[str, dict[str, Any]]) -> dict:
    raw = protocol_v2._read(proof_path, "frozen live-parity proof")  # noqa: SLF001
    observed = raw.get("observed_at")
    if not isinstance(observed, str):
        raise ValueError("frozen live-parity observation time is missing")
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
        proof_path,
        readiness={"arms": {"base": parity_arms["base"], "fresh75": parity_arms["candidate"]}},
        configs={"base": configs["base"], "fresh75": configs["candidate"]},
        now=datetime.fromisoformat(observed.replace("Z", "+00:00")),
    )
    successor._require_weight_probe_difference(parity)  # noqa: SLF001
    return parity


def _seed60_retirement(path: Path) -> dict[str, Any]:
    receipt = protocol_v2._verified(path, "seed-60 whole-pair retirement")  # noqa: SLF001
    if (
        _file_sha(path) != SEED60_RETIREMENT_FILE_SHA256
        or receipt.get("sha256") != SEED60_RETIREMENT_SHA256
        or receipt.get("schema") != "q38_dev17_protocol_v2_seed60_whole_pair_retirement_receipt_v1"
        or receipt.get("source_commit") != "e6329cc81a92170fafa2c2d71fb20866ba95e010"
        or receipt.get("decision", {}).get("gate") != "NO-GO"
        or receipt.get("decision", {}).get("whole_pair_excluded") is not True
        or receipt.get("decision", {}).get("allowlist_widened") is not False
        or receipt.get("decision", {}).get("partial_subset_reconciliation_for_seed60_base_allowed")
        is not False
        or receipt.get("lineage", {}).get("mapping") != "seed60->seed66"
        or receipt.get("lineage", {}).get("original_seed") != 60
        or receipt.get("lineage", {}).get("replacement_seed") != 66
        or receipt.get("lineage", {}).get("whole_pair_replacement_required") is not True
        or any(value is not False for value in receipt.get("execution", {}).values())
        or receipt.get("privacy")
        != {
            "credentials_read_or_included": False,
            "private_cell_task_session_or_verifier_ids_read_or_included": False,
            "prompts_responses_flags_rewards_or_trace_content_read_or_included": False,
            "score_blind": True,
            "scores_read_or_included": False,
        }
    ):
        raise ValueError("seed-60 whole-pair retirement evidence differs")
    return receipt


def _comparison_definition(
    frozen: dict[str, Any], protocols: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    definition = json.loads(
        json.dumps({key: value for key, value in frozen.items() if key != "sha256"})
    )
    definition["schema"] = "cyber_qwen38_fleet_dev17_pass8_comparison_definition_v6"
    definition["predecessor_comparison_definition_sha256"] = frozen["sha256"]
    definition["included_seeds"] = list(INCLUDED_SEEDS)
    definition["excluded_protocol_v2_seeds"] = sorted(
        {*definition["excluded_protocol_v2_seeds"], 60}
    )
    definition["full_comparison_packet_state"] = {
        "state": "seed65_seed66_deferred_next_utc_gates",
        "rendered_successor_seeds": list(PRIOR_RENDERED_SEEDS),
        "deferred_successor_seeds": list(DEFERRED_SEEDS),
        "rendered_packet_count": len(PRIOR_RENDERED_SEEDS) * len(protocol_v2.ARMS),
        "missing_packet_cells": [
            {"seed": seed, "arm_id": arm} for seed in DEFERRED_SEEDS for arm in protocol_v2.ARMS
        ],
        "full_roster_launch_ready": False,
        "final_aggregation_allowed": False,
        "score_unseal_allowed": False,
        "requires_fresh_budget_refresh": True,
        "requires_fresh_destination_absence": True,
        "requires_fresh_live_parity": True,
        "requires_versioned_successor_receipt": True,
    }
    mapping = definition["replacement_mapping"]
    row = next(item for item in mapping if item["invalid_original_seed"] == 53)
    if row.get("replacement_seed") != 60:
        raise ValueError("seed-60 predecessor mapping differs")
    row["replacement_seed"] = 66

    replicas = [item for item in definition["replica_protocols"] if item.get("seed") != 60]
    seed65 = next((item for item in replicas if item.get("seed") == 65), None)
    if seed65 != {
        "comparison_protocol_sha256": protocols[65]["sha256"],
        "origin": "whole_pair_successor",
        "protocol_id": protocols[65]["protocol_id"],
        "seed": 65,
    }:
        raise ValueError("frozen seed-65 protocol row differs")
    replicas.append(
        {
            "comparison_protocol_sha256": protocols[66]["sha256"],
            "origin": "whole_pair_successor",
            "protocol_id": protocols[66]["protocol_id"],
            "seed": 66,
        }
    )
    definition["replica_protocols"] = sorted(replicas, key=lambda item: item["seed"])
    definition["superseded_replacements"].append(
        {
            "evidence_receipt_sha256s": [
                "sha256:4f6fb296751a8aefbfe485b1b0249d761ad2c3f5ba72a6dd287e09ba53d1a2f2",
                "sha256:5b2468d0878b830dcbecf111d0e988613e96b7ce4992ed70496bfab95f62e96c",
                "sha256:9e7e61e3cf2306472219e4b8c74cc597a653a8b4282c242f03fbe85fabc2e112",
                "sha256:817471978fd4248751dc64a733c55d6976e888c824db40c4cd3deb28033b2696",
                "sha256:9f313d81b36e8261bea439532d4037b8587dd30411360c4508e0c13cb11ae0bf",
                SEED60_RETIREMENT_SHA256,
            ],
            "invalid_original_seed": 53,
            "successor_seed": 66,
            "superseded_replacement_seed": 60,
            "whole_pair_excluded": True,
        }
    )
    definition["seed60_pair_retirement_evidence"] = {
        "file_sha256": SEED60_RETIREMENT_FILE_SHA256,
        "sha256": SEED60_RETIREMENT_SHA256,
        "whole_pair_excluded": True,
        "replacement_seed": 66,
    }
    definition["sha256"] = protocol_v2._canonical(definition)  # noqa: SLF001
    return definition


def _launcher_objects(packet_path: Path, seed: int, arm: str) -> tuple[dict, dict, dict[str, Any]]:
    compressed, evaluator_job = launcher._bundle(packet_path.parent)  # noqa: SLF001
    config_map, job = launcher._objects(  # noqa: SLF001
        replica=f"seed{seed}-{arm}",
        compressed=compressed,
        evaluator_job=evaluator_job,
        launch_journal_id=evaluator_job,
    )
    pod = job.get("spec", {}).get("template", {}).get("spec", {})
    labels = job.get("spec", {}).get("template", {}).get("metadata", {}).get("labels", {})
    annotations = job.get("metadata", {}).get("annotations", {})
    if (
        config_map.get("immutable") is not True
        or annotations.get(heldout_launch.FAILURE_ALERT_ANNOTATION)
        != heldout_launch.FAILURE_ALERT_OFF
        or annotations.get(heldout_launch.CREATE_ONCE_ANNOTATION) != "true"
        or pod.get("priorityClassName") != "c1"
        or labels.get(heldout_launch.POSTGRES_CLIENT_LABEL)
        != heldout_launch.POSTGRES_CLIENT_LABEL_VALUE
        or "nvidia.com/gpu" in json.dumps(job)
    ):
        raise ValueError("deferred outer launcher policy differs")
    return (
        config_map,
        job,
        {
            "arm_id": arm,
            "bundle_sha256": "sha256:" + hashlib.sha256(compressed).hexdigest(),
            "config_map_name": config_map["metadata"]["name"],
            "evaluator_job": evaluator_job,
            "launcher_job": job["metadata"]["name"],
            "root_create_once": True,
            "root_failure_alerts": "off",
            "priority_class": "c1",
            "gpu_requests": 0,
        },
    )


def prepare(*, frozen_packets: Path, seed60_retirement: Path, output: Path) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise FileExistsError("seed-65/66 preparation output already exists")
    if not output.parent.is_dir():
        raise ValueError("seed-65/66 preparation output parent does not exist")
    frozen_receipt, frozen_definition, frozen_protocol, contracts, proof_path = _frozen_source(
        frozen_packets
    )
    retirement = _seed60_retirement(seed60_retirement)
    base_template, _task_set, _split, _corpus, roster = source._inputs()  # noqa: SLF001
    configs_by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    protocols: dict[int, dict[str, Any]] = {}
    for seed in DEFERRED_SEEDS:
        base, candidate = protocol_v2._configs(base_template, seed)  # noqa: SLF001
        configs_by_seed[seed] = {"base": base, "candidate": candidate}
        protocols[seed] = protocol_v2._protocol(base, seed)  # noqa: SLF001
    if protocols[FROZEN_DEFERRED_SEED] != frozen_protocol:
        raise ValueError("seed-65 protocol changed after it was frozen")
    parity = _validate_frozen_parity(proof_path, configs_by_seed[FROZEN_DEFERRED_SEED])
    comparison_definition = _comparison_definition(frozen_definition, protocols)

    temporary = Path(
        tempfile.mkdtemp(prefix=".fleet-seed65-seed66-preparation-", dir=output.parent)
    )
    try:
        evidence = temporary / "evidence"
        evidence.mkdir(mode=0o700)
        frozen_bindings = {}
        for label, source_path in (
            ("successor_receipt", frozen_packets / "SUCCESSOR_RECEIPT.json"),
            ("comparison_definition", frozen_packets / "COMPARISON_DEFINITION.json"),
        ):
            destination = evidence / source_path.name
            destination.write_bytes(source_path.read_bytes())
            destination.chmod(0o600)
            frozen_bindings[label] = {
                "path": f"evidence/{destination.name}",
                "file_sha256": _file_sha(destination),
                "sha256": (
                    frozen_receipt["sha256"]
                    if label == "successor_receipt"
                    else frozen_definition["sha256"]
                ),
            }
        retirement_copy = evidence / "SEED60_RETIREMENT_RECEIPT.json"
        retirement_copy.write_bytes(seed60_retirement.read_bytes())
        retirement_copy.chmod(0o600)
        retirement_binding = {
            "path": "evidence/SEED60_RETIREMENT_RECEIPT.json",
            "file_sha256": _file_sha(retirement_copy),
            "sha256": retirement["sha256"],
            "whole_pair_excluded": True,
            "replacement_seed": 66,
        }

        definition_path = temporary / "COMPARISON_DEFINITION_V6.json"
        source._write_json(definition_path, comparison_definition)  # noqa: SLF001
        prior_rows = frozen_receipt["replacement_arms"]
        destination_fields = {
            "job": "job_name",
            "config_map": "config_map_name",
            "output": "output_root",
            "database": "database",
            "identity": "ledger_identity",
        }
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
        arms: list[dict[str, Any]] = []
        objects: list[dict] = []
        protocol_rows: list[dict[str, Any]] = []
        for seed in DEFERRED_SEEDS:
            configs = configs_by_seed[seed]
            protocol = protocols[seed]
            seed_dir = temporary / f"seed{seed}"
            seed_dir.mkdir(mode=0o700)
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
            source._write_json(base_path, configs["base"])  # noqa: SLF001
            source._write_json(candidate_path, configs["candidate"])  # noqa: SLF001
            source._write_json(protocol_path, protocol)  # noqa: SLF001
            source._write_json(  # noqa: SLF001
                base_provenance,
                source._base_provenance(  # noqa: SLF001
                    base_path, roster, configs["base"]["name"]
                ),
            )
            source._write_json(  # noqa: SLF001
                candidate_provenance,
                source._candidate_packet(configs["candidate"]["name"]),  # noqa: SLF001
            )
            protocol_rows.append(
                {
                    "seed": seed,
                    "protocol_id": protocol["protocol_id"],
                    "file_sha256": _file_sha(protocol_path),
                    "sha256": protocol["sha256"],
                }
            )

            destinations = successor._destinations(seed)  # noqa: SLF001
            if any(
                destination[source_field] in {row[receipt_field] for row in prior_rows}
                for destination in destinations.values()
                for source_field, receipt_field in destination_fields.items()
            ):
                raise ValueError("deferred destination collides with a frozen predecessor")
            for arm, config_path, checkpoint in (
                ("base", base_path, base_provenance),
                ("candidate", candidate_path, candidate_provenance),
            ):
                destination = destinations[arm]
                arm_runtime = {
                    **parity_arms[arm],
                    "job_name": destination["job"],
                    "config_map_name": destination["config_map"],
                    "output_root": destination["output"],
                    "database": destination["database"],
                }
                source.shared._prepare_arm(  # noqa: SLF001
                    seed_dir / arm,
                    arm_id=arm,
                    arm=arm_runtime,
                    config_path=config_path,
                    config=configs[arm],
                    task_set_path=source.TASK_SET,
                    split_path=source.SPLIT,
                    protocol_path=protocol_path,
                    protocol=protocol,
                    checkpoint_path=checkpoint,
                    proof_path=proof_path,
                    ledger_path=source.LEDGER,
                    source_files=source.V3_SOURCE_FILES if arm == "candidate" else None,
                )
                packet_path = seed_dir / arm / "LAUNCH_PACKET.json"
                package = successor._harden_and_reseal(packet_path)  # noqa: SLF001
                evaluation_binding = protocol_v2._evaluation_binding(packet_path)  # noqa: SLF001
                identity = package.packet.identity
                route = next(iter(package.evaluation_config["routes"].values()))
                if (
                    successor._frozen_config_contract(package.evaluation_config)  # noqa: SLF001
                    != contracts[arm]["config"]
                    or evaluation_binding["runtime_files_sha256"]
                    != contracts[arm]["runtime_files_sha256"]
                    or identity.get("sampling_seed") != seed
                    or identity.get("arm_id") != arm
                    or identity.get("harness") != "opencode"
                    or identity.get("pass_k") != 1
                    or identity.get("retry_limit") != 0
                    or package.evaluation_config.get("max_reviewed_infrastructure_retries") != 0
                    or len(route.get("task_versions", [])) != protocol_v2.TASKS_PER_ARM
                ):
                    raise ValueError("deferred scientific treatment changed")
                config_map, job, outer = _launcher_objects(packet_path, seed, arm)
                objects.extend([config_map, job])
                arms.append(
                    {
                        "seed": seed,
                        "arm_id": arm,
                        "packet_path": f"seed{seed}/{arm}/LAUNCH_PACKET.json",
                        **evaluation_binding,
                        "job_name": package.packet.job_name,
                        "config_map_name": package.packet.config_map_name,
                        "output_root": package.packet.output_root,
                        "database": package.packet.database,
                        "ledger_identity": identity["evaluation_config_name"],
                        "outer_launcher": outer,
                    }
                )
        unique_fields = (
            "packet_path",
            "job_name",
            "config_map_name",
            "output_root",
            "database",
            "ledger_identity",
        )
        if any(len({row[field] for row in arms}) != 4 for field in unique_fields):
            raise ValueError("deferred pair identities are not unique")
        for field in ("launcher_job", "config_map_name"):
            if len({row["outer_launcher"][field] for row in arms}) != 4:
                raise ValueError("deferred outer launcher identities are not unique")

        bundle = {"apiVersion": "v1", "kind": "List", "items": objects}
        bundle_path = temporary / "launchers.yaml"
        bundle_path.write_text(yaml.safe_dump(bundle, sort_keys=False), encoding="utf-8")
        receipt = {
            "schema": SCHEMA,
            "frozen_successor": frozen_bindings,
            "frozen_successor_receipt_sha256": FROZEN_RECEIPT_SHA256,
            "frozen_comparison_definition_sha256": FROZEN_DEFINITION_SHA256,
            "comparison_definition": {
                "path": definition_path.name,
                "file_sha256": _file_sha(definition_path),
                "sha256": comparison_definition["sha256"],
            },
            "seed60_pair_retirement": retirement_binding,
            "included_seeds": list(INCLUDED_SEEDS),
            "replacement_protocols": protocol_rows,
            "arms": arms,
            "launcher_bundle": {
                "path": "launchers.yaml",
                "file_sha256": _file_sha(bundle_path),
                "object_count": len(objects),
                "job_count": 4,
                "config_map_count": 4,
            },
            "scientific_identity": {
                "task_count_per_arm": protocol_v2.TASKS_PER_ARM,
                "comparison_arms": list(protocol_v2.ARMS),
                "pass_k": 1,
                "retry_limit": 0,
                "harness": "opencode",
                "sampling_seeds": list(DEFERRED_SEEDS),
                "rollouts_requiring_fresh_reservation": ROLLOUTS,
                "frozen_parity_file_sha256": _file_sha(proof_path),
                "frozen_parity_receipt_sha256": parity["receipt_sha256"],
            },
            "launch_readiness": {
                "state": "prepared_unsealed_next_utc_gates_required",
                "launch_ready": False,
                "current_utc_capacity_claimed": False,
                "destination_absence_checked": False,
                "fresh_live_parity_checked": False,
                "server_preview_performed": False,
                "required_before_create": [
                    "fresh_next_utc_global_budget_census",
                    "fresh_all_dimension_destination_absence",
                    "fresh_content_free_live_parity",
                    "two_identical_server_previews",
                    "exact_head_ci_and_independent_review",
                ],
                "packet_bytes_may_change_at_launch": False,
            },
            "job_security": {
                "inner_root_failure_alerts": "off",
                "inner_priority_class": "c1",
                "inner_gpu_requests": 0,
                "inner_postgres_client_label": "true",
                "inner_automountServiceAccountToken": False,
                "outer_root_failure_alerts": "off",
                "outer_create_once": True,
                "outer_priority_class": "c1",
                "outer_gpu_requests": 0,
            },
            "privacy": {
                "score_values_read": False,
                "prompts_responses_flags_rewards_or_trace_content_read": False,
                "infrastructure_only": True,
            },
            "provider_requests": 0,
            "external_mutations": 0,
            "server_preview_performed": False,
            "launch_performed": False,
        }
        receipt["sha256"] = protocol_v2._canonical(receipt)  # noqa: SLF001
        source._write_json(  # noqa: SLF001
            temporary / "SEED65_SEED66_PREPARATION_RECEIPT_V2.json", receipt
        )
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-packets", type=Path, required=True)
    parser.add_argument("--seed60-retirement", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                frozen_packets=args.frozen_packets,
                seed60_retirement=args.seed60_retirement,
                output=args.output,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
