#!/usr/bin/env python3
"""Prepare, but never launch, the Qwen3.8 seed-45 five-arm Fleet packets."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts import prepare_qwen38_fleet_seed44_two_arm_packets as shared

ROOT = Path(__file__).resolve().parents[1]
READINESS = ROOT / "configs/evaluation/qwen38-fleet-dev17-seed45-five-arm-readiness-v1.json"
CANDIDATES = ("fresh75", "teacher186", "self44", "lr30s76")
SOURCE_FILES = {
    **shared.SOURCE_FILES,
    "model_artifact_v2.py": ROOT / "evals/fleet/model_artifact_v2.py",
    "run.sh": ROOT / "evals/fleet/scripts/run_qwen38_dev17_single_arm_v2.sh",
}


def _parity_inputs(values: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        arm, separator, path = value.partition("=")
        if not separator or arm not in CANDIDATES or arm in parsed or not path:
            raise ValueError("--live-parity must be one unique <candidate>=<path> binding")
        parsed[arm] = Path(path)
    if set(parsed) != set(CANDIDATES):
        raise ValueError("one live-parity receipt is required for every candidate")
    return parsed


def _validate_pair(
    path: Path,
    *,
    candidate: str,
    readiness: dict[str, Any],
    configs: dict[str, dict[str, Any]],
    now: datetime,
) -> dict[str, Any]:
    return shared._live_parity(  # noqa: SLF001
        path,
        readiness={
            "arms": {
                "base": readiness["arms"]["base"],
                "fresh75": readiness["arms"][candidate],
            }
        },
        configs={"base": configs["base"], "fresh75": configs[candidate]},
        now=now,
    )


def prepare(
    *,
    output: Path,
    live_parity: dict[str, Path],
    now: datetime | None = None,
) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise FileExistsError("packet output already exists")
    if not output.parent.is_dir():
        raise ValueError("packet output parent does not exist")
    readiness = shared._read_json(READINESS, "seed-45 readiness")  # noqa: SLF001
    shared._verify_self_digest(readiness, "seed-45 readiness")  # noqa: SLF001
    if (
        readiness.get("launchable") is not False
        or readiness.get("selection", {}).get("final8_sealed") is not True
        or readiness.get("operation")
        != {
            "jobs_created": 0,
            "config_maps_created": 0,
            "routes_resumed": 0,
            "databases_created": 0,
            "outputs_created": 0,
        }
    ):
        raise ValueError("seed-45 source readiness is not launch-inert")
    protocol_path = shared._verify_file_binding(  # noqa: SLF001
        readiness["protocol"], "comparison protocol"
    )
    protocol = shared._read_json(protocol_path, "comparison protocol")  # noqa: SLF001
    shared._verify_self_digest(protocol, "comparison protocol")  # noqa: SLF001
    if protocol.get("comparison_arms") != ["base", *CANDIDATES] or protocol.get("retry_limit") != 1:
        raise ValueError("seed-45 comparison protocol differs")
    task_set_path = shared._repo_path(  # noqa: SLF001
        readiness["selection"]["task_set_path"], "task set"
    )
    split_path = shared._repo_path(readiness["selection"]["split_path"], "split manifest")  # noqa: SLF001
    if (
        shared._file_sha256(task_set_path)  # noqa: SLF001
        != readiness["selection"]["task_set_file_sha256"]
        or shared._file_sha256(split_path)  # noqa: SLF001
        != readiness["selection"]["split_file_sha256"]
    ):
        raise ValueError("seed-45 task selection or split bytes changed")

    configs: dict[str, dict[str, Any]] = {}
    sources: dict[str, tuple[Path, Path]] = {}
    for arm_id, arm in readiness["arms"].items():
        config_path = shared._repo_path(arm["config_path"], f"{arm_id} config")  # noqa: SLF001
        checkpoint_path = shared._repo_path(  # noqa: SLF001
            arm["checkpoint_provenance_path"], f"{arm_id} checkpoint provenance"
        )
        if (
            shared._file_sha256(config_path) != arm["config_file_sha256"]  # noqa: SLF001
            or shared._file_sha256(checkpoint_path)  # noqa: SLF001
            != arm["checkpoint_provenance_file_sha256"]
        ):
            raise ValueError(f"{arm_id} source bytes changed")
        config = shared._read_json(config_path, f"{arm_id} config")  # noqa: SLF001
        if config.get("sampling", {}).get("seed") != 45:
            raise ValueError(f"{arm_id} is not a seed-45 config")
        configs[arm_id] = config
        sources[arm_id] = (config_path, checkpoint_path)

    observed_at = now or datetime.now(UTC)
    pairs = {
        candidate: _validate_pair(
            live_parity[candidate],
            candidate=candidate,
            readiness=readiness,
            configs=configs,
            now=observed_at,
        )
        for candidate in CANDIDATES
    }
    temporary = Path(tempfile.mkdtemp(prefix=".seed45-packets-", dir=output.parent))
    try:
        joint = {
            "schema": "cyber_qwen38_fleet_seed45_joint_live_parity_v1",
            "protocol_id": protocol["protocol_id"],
            "pairs": {
                candidate: {
                    "path_label": live_parity[candidate].name,
                    "file_sha256": shared._file_sha256(live_parity[candidate]),  # noqa: SLF001
                    "receipt_sha256": pairs[candidate]["receipt_sha256"],
                    "observed_at": pairs[candidate]["observed_at"],
                    "base_served_model": pairs[candidate]["arms"]["base"]["served_model"],
                    "candidate_served_model": pairs[candidate]["arms"]["candidate"]["served_model"],
                }
                for candidate in CANDIDATES
            },
            "benchmark_content_included": False,
            "response_content_recorded": False,
            "scores_observed": False,
            "task_content_included": False,
            "external_mutations_performed": 0,
        }
        joint["sha256"] = shared._canonical_digest(joint)  # noqa: SLF001
        joint_path = temporary / "joint-live-parity.json"
        joint_path.write_text(json.dumps(joint, indent=2, sort_keys=True) + "\n")

        ledger_path = ROOT / "configs/evaluation/qwen38-checkpoint-eval-ledger-v2.json"
        arms = []
        for arm_id in protocol["comparison_arms"]:
            config_path, checkpoint_path = sources[arm_id]
            arms.append(
                shared._prepare_arm(  # noqa: SLF001
                    temporary / arm_id,
                    arm_id=arm_id,
                    arm=readiness["arms"][arm_id],
                    config_path=config_path,
                    config=configs[arm_id],
                    task_set_path=task_set_path,
                    split_path=split_path,
                    protocol_path=protocol_path,
                    protocol=protocol,
                    checkpoint_path=checkpoint_path,
                    proof_path=joint_path,
                    ledger_path=ledger_path,
                    source_files=SOURCE_FILES,
                )
            )
        receipt = {
            "schema": "cyber_qwen38_fleet_seed45_five_arm_packet_preparation_v1",
            "protocol_id": protocol["protocol_id"],
            "comparison_protocol_sha256": protocol["sha256"],
            "joint_live_parity_sha256": joint["sha256"],
            "joint_live_parity_file_sha256": shared._file_sha256(joint_path),  # noqa: SLF001
            "arms": arms,
            "external_mutations": 0,
            "launch_performed": False,
            "credentials_read": False,
            "final8_opened": False,
        }
        receipt["sha256"] = shared._canonical_digest(receipt)  # noqa: SLF001
        (temporary / "PREPARATION_RECEIPT.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        )
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--live-parity",
        action="append",
        default=[],
        help="repeat as <candidate>=<path> for fresh75, teacher186, self44, and lr30s76",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(output=args.output, live_parity=_parity_inputs(args.live_parity)),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
