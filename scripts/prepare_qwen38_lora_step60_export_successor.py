#!/usr/bin/env python3
"""Version the step-60 zero-update export plan without rewriting its receipt-bound v1."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from cyber_post_train.jobs import digest
from training import qwen38_lora_export as export

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = ROOT / "configs/qualification/qwen38-lora-step60-zero-update-export-v1.json"
SUCCESSOR = ROOT / "configs/qualification/qwen38-lora-step60-zero-update-export-v2.json"
EVIDENCE = ROOT / "docs/evidence/qwen38-lora-step60-export-successor-preparation-20260921.json"
HISTORICAL_FILE_SHA256 = "c3fe7f50033ba4569ef6297991a22dcd6e580e2124a380ec9424e0318640f815"
HISTORICAL_PLAN_SHA256 = "31c9548c7119e01e3941049bcdaa79f9ee7a787a0191ce02c53644959d2d4547"
SUCCESSOR_FILE_SHA256 = "36ab4a0156b875f61d206d1be0f450a46a72e3364fa613b4eae7994f10071ba0"
SUCCESSOR_PLAN_SHA256 = "406a8108abb8f9f0c08bd80ca66f30edddb9eb1895ab933fab8781591029303b"
SUCCESSOR_REQUEST_SHA256 = "31f4b1963c0460457f177fa63a62a958cdb6d30c5d259b2941f0fc1e52043dfc"
EVIDENCE_FILE_SHA256 = "599c7fd46f05c352b756e327f04f283bfb190321409fdf5bd65909747befd71e"
EVIDENCE_RECEIPT_SHA256 = "2e9d596c4a01e78d0e7a482f41342b6b4a7fff1311da0115a6f3cd65c8ba4354"


def raw(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> tuple[dict, dict]:
    historical = json.loads(HISTORICAL.read_text())
    if (
        file_sha256(HISTORICAL) != HISTORICAL_FILE_SHA256
        or digest(historical) != HISTORICAL_PLAN_SHA256
    ):
        raise ValueError("receipt-bound step-60 v1 changed; create a new version instead")

    successor = copy.deepcopy(historical)
    successor.update(
        {
            "run_name": "chris-q38-lora-s60-exp-v2",
            "run_dir": "/mnt/sfs/jobs/chris-q38-lora-s60-exp-v2",
            "output_root": "/mnt/sfs/jobs/chris-q38-lora-s60-exp-v2/merged-hf",
            "code_sha256": export._continuation_code_sha256(),
        }
    )
    export.validate_plan(successor)
    request = export.job_request(successor)
    evidence = {
        "schema": "cyber_qwen38_lora_step60_export_successor_preparation_v1",
        "status": "prepared_not_launched",
        "historical_plan": {
            "path": str(HISTORICAL.relative_to(ROOT)),
            "file_sha256": HISTORICAL_FILE_SHA256,
            "plan_sha256": HISTORICAL_PLAN_SHA256,
            "preserved_byte_identical": True,
        },
        "successor_plan": {
            "path": str(SUCCESSOR.relative_to(ROOT)),
            "file_sha256": hashlib.sha256(raw(successor)).hexdigest(),
            "plan_sha256": digest(successor),
            "request_sha256": digest(request),
            "run_name": successor["run_name"],
            "run_dir": successor["run_dir"],
            "output_root": successor["output_root"],
            "failure_alerts": request["failureAlerts"],
            "priority": request["priority_class"],
        },
        "external_activity": {
            "jobs_api_requests": 0,
            "kubernetes_requests": 0,
            "cluster_mutations": 0,
            "gpus_allocated": 0,
        },
        "launch_authorized": False,
        "next_gate": (
            "Reopen the exact source, prove the fresh output absent, obtain a server preview "
            "with the root failure-alert annotation, and record separate launch authorization."
        ),
    }
    evidence["receipt_sha256"] = digest(evidence)
    if (
        hashlib.sha256(raw(successor)).hexdigest() != SUCCESSOR_FILE_SHA256
        or digest(successor) != SUCCESSOR_PLAN_SHA256
        or evidence["successor_plan"]["request_sha256"] != SUCCESSOR_REQUEST_SHA256
        or hashlib.sha256(raw(evidence)).hexdigest() != EVIDENCE_FILE_SHA256
        or evidence["receipt_sha256"] != EVIDENCE_RECEIPT_SHA256
    ):
        raise ValueError("sealed step-60 v2 drifted; mint v3 instead of rewriting it")
    return successor, evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = parser.parse_args()
    successor, evidence = build()
    expected = {SUCCESSOR: raw(successor), EVIDENCE: raw(evidence)}
    if args.write:
        for path, content in expected.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                if path.read_bytes() != content:
                    raise SystemExit(f"refusing to overwrite sealed artifact: {path}")
            else:
                with path.open("xb") as stream:
                    stream.write(content)
    else:
        stale = [
            str(path.relative_to(ROOT))
            for path, content in expected.items()
            if not path.exists() or path.read_bytes() != content
        ]
        if stale:
            raise SystemExit("stale step-60 successor artifacts: " + ", ".join(stale))
    print(json.dumps({"checked": len(expected), "external_mutations": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
