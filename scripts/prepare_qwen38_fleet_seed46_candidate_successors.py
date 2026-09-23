#!/usr/bin/env python3
"""Prepare eight no-replay candidate successors after a zero-session setup failure."""

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
from scripts import prepare_qwen38_fleet_seed46_pass8_packets as study

GENERATION = 2


def _runtime(seed: int) -> dict[str, Any]:
    return {
        "model_revision": study.CANDIDATE_REVISION,
        "served_id": study.CANDIDATE_ID,
        "expected_source_path": f"/models/{study.CANDIDATE_ID}",
        "job_name": f"chris-q38-dev17-s{seed}-t3k32s1000-p1-v{GENERATION}",
        "config_map_name": f"chris-q38-dev17-s{seed}-t3k32s1000-code-v{GENERATION}",
        "output_root": (f"/mnt/sfs/jobs/chris-q38-fleet-dev17-s{seed}-t3k32s1000-p1-v{GENERATION}"),
        "database": f"q38_dev17_s{seed}_t3k32s1000_p1_v{GENERATION}",
    }


def prepare(*, output: Path, live_parity: Path, now: datetime | None = None) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise FileExistsError("successor packet output already exists")
    if not output.parent.is_dir():
        raise ValueError("successor packet output parent does not exist")
    base_template, task_set, split, corpus, roster = study._inputs()  # noqa: SLF001
    parity_base, parity_candidate = study._configs(  # noqa: SLF001
        base_template, study.SEEDS[0], candidate_generation=GENERATION
    )
    parity = shared._live_parity(  # noqa: SLF001
        live_parity,
        readiness={
            "arms": {
                "base": {
                    "model_revision": study.BASE_REVISION,
                    "served_id": "qwen3.8-27b",
                    "expected_source_path": (f"/models/qwen3.8-27b/{study.BASE_REVISION}"),
                },
                "fresh75": {
                    "model_revision": study.CANDIDATE_REVISION,
                    "served_id": study.CANDIDATE_ID,
                    "expected_source_path": f"/models/{study.CANDIDATE_ID}",
                },
            }
        },
        configs={"base": parity_base, "fresh75": parity_candidate},
        now=now or datetime.now(UTC),
    )
    temporary = Path(tempfile.mkdtemp(prefix=".seed46-candidate-v2-", dir=output.parent))
    try:
        rows = []
        protocols = []
        for seed in study.SEEDS:
            seed_dir = temporary / f"seed{seed}"
            seed_dir.mkdir(mode=0o700)
            base, candidate = study._configs(  # noqa: SLF001
                base_template, seed, candidate_generation=GENERATION
            )
            protocol = study._protocol(base, seed)  # noqa: SLF001
            config_path = seed_dir / (
                f"qwen38-teacher3k32-step1000-fleet-dev17-opencode-seed{seed}"
                f"-pass1-v{GENERATION}.json"
            )
            packet_path = seed_dir / (f"qwen38-step1000-seed{seed}-provenance-v{GENERATION}.json")
            protocol_path = seed_dir / (
                f"qwen38-fleet-dev17-seed{seed}-base-step1000-pass1-protocol-v1.json"
            )
            packet = study._candidate_packet(candidate["name"])  # noqa: SLF001
            study._write_json(config_path, candidate)  # noqa: SLF001
            study._write_json(packet_path, packet)  # noqa: SLF001
            study._write_json(protocol_path, protocol)  # noqa: SLF001
            row = shared._prepare_arm(  # noqa: SLF001
                seed_dir / "candidate",
                arm_id="candidate",
                arm=_runtime(seed),
                config_path=config_path,
                config=candidate,
                task_set_path=study.TASK_SET,
                split_path=study.SPLIT,
                protocol_path=protocol_path,
                protocol=protocol,
                checkpoint_path=packet_path,
                proof_path=live_parity,
                ledger_path=study.LEDGER,
                source_files=study.V3_SOURCE_FILES,
            )
            row["seed"] = seed
            rows.append(row)
            protocols.append(
                {
                    "seed": seed,
                    "protocol_id": protocol["protocol_id"],
                    "sha256": protocol["sha256"],
                    "file_sha256": study._file_sha(protocol_path),  # noqa: SLF001
                }
            )
        receipt = {
            "schema": "cyber_qwen38_fleet_seed46_candidate_successor_preparation_v1",
            "protocol_study_id": study.PROTOCOL_STUDY_ID,
            "successor_generation": GENERATION,
            "seeds": list(study.SEEDS),
            "protocols": protocols,
            "live_parity_file_sha256": study._file_sha(live_parity),  # noqa: SLF001
            "live_parity_receipt_sha256": parity["receipt_sha256"],
            "selection": {
                "task_count": len(task_set["tasks"]),
                "sessions": len(task_set["tasks"]) * len(study.SEEDS),
                "pass_k_per_seed": 1,
                "aggregate_pass_k": 8,
                "selection_sha256": task_set["selection_sha256"],
                "split_sha256": split["sha256"],
                "binding_roster_sha256": roster["sha256"],
            },
            "source_failure_gate": {
                "source_generation": 1,
                "all_source_jobs_terminal_failed": True,
                "all_source_databases_zero_rows": True,
                "all_source_outputs_absent": True,
                "candidate_sessions_created": 0,
                "failure_class": "missing_local_model_artifact_binding",
                "valid_outcomes_replayed": False,
            },
            "leakage_gate": {
                "exact_dev_task_key_overlap": 0,
                "corpus_dev_windows": corpus["dev_windows"],
            },
            "arms": rows,
            "external_mutations": 0,
            "launch_performed": False,
            "credentials_read": False,
            "scores_read": False,
            "final8_opened": False,
        }
        receipt["sha256"] = study._canonical(receipt)  # noqa: SLF001
        study._write_json(temporary / "PREPARATION_RECEIPT.json", receipt)  # noqa: SLF001
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-parity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(output=args.output, live_parity=args.live_parity), indent=2))


if __name__ == "__main__":
    main()
