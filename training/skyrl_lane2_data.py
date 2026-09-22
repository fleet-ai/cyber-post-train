"""GET-only, zero-GPU construction of the sealed lane2 private data package."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import httpx

from . import rl_data
from . import skyrl_lane2_authority as authority
from .sft import digest, read_mapping

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/qualification/qwen38-skyrl-lane2-data-v1.json"
AUTHORITY = ROOT / "configs/qualification/qwen38-skyrl-lane2-authority-v1.json"
RECEIPT = Path("/dev/termination-log")
MAX_RECEIPT_BYTES = 16384


def validate_config(config: dict, *, relative_to: Path) -> None:
    evidence = authority.validate_authority(AUTHORITY)
    expected = read_mapping(CONFIG)
    if (
        config != expected
        or relative_to.resolve() != CONFIG.parent.resolve()
        or config.get("name") != "chris-q38-rlreward-lane2-v1"
        or config.get("output")
        != "/mnt/sfs/jobs/chris-q38-study-corpora-v1/rlreward-inputs-lane2-v1/data"
        or evidence["submission_gate"]["submission_authorized"] is not False
    ):
        raise ValueError("lane2 data-construction identity changed")
    task_set = read_mapping(relative_to / config["task_set"])
    split = read_mapping(relative_to / config["split"])
    selected = rl_data.selection(task_set, split)
    if [(row["split"], row["task_version_id"]) for row in selected] != [
        ("dev", authority.DIAGNOSTIC_TASK_VERSION_ID),
        ("train", authority.OPTIMIZER_TASK_VERSION_ID),
    ]:
        raise ValueError("lane2 data-construction selection changed")


def build(config: dict, *, relative_to: Path, client: httpx.Client) -> dict:
    validate_config(config, relative_to=relative_to)
    value = rl_data.build(config, relative_to=relative_to, client=client)
    if (
        value.get("submitted") is not False
        or not isinstance(value.get("manifest_sha256"), str)
        or not isinstance(value.get("files"), dict)
    ):
        raise ValueError("lane2 data construction crossed its GET-only CPU boundary")
    return value


def sanitized_receipt(value: dict) -> dict:
    """Keep private rows out of the Job termination message."""
    files = value.get("files")
    if not isinstance(files, dict):
        raise ValueError("lane2 data result files are missing")
    body = {
        "schema": "cyber_skyrl_lane2_data_receipt_v1",
        "status": "built",
        "submitted": value.get("submitted"),
        "gpus": 0,
        "files": {
            split: {"rows": item["rows"], "sha256": item["sha256"]} for split, item in files.items()
        },
        "manifest_sha256": value.get("manifest_sha256"),
    }
    if (
        body["submitted"] is not False
        or set(body["files"]) != {"train", "dev"}
        or not isinstance(body["manifest_sha256"], str)
        or any(
            type(item.get("rows")) is not int
            or item["rows"] < 1
            or not isinstance(item.get("sha256"), str)
            for item in body["files"].values()
        )
    ):
        raise ValueError("lane2 data result cannot produce a sanitized receipt")
    return {**body, "receipt_sha256": digest(body)}


def _write_receipt(path: Path, value: dict) -> None:
    if path != RECEIPT:
        raise ValueError("lane2 data receipt path changed")
    encoded = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if len(encoded) > MAX_RECEIPT_BYTES:
        raise ValueError("lane2 data receipt is too large")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    token = os.environ.get("FLEET_API_KEY", "")
    if not token:
        raise SystemExit("Fleet API key required")
    config = read_mapping(config_path)
    with httpx.Client(
        headers={"Authorization": "Bearer " + token},
        timeout=60,
        follow_redirects=False,
        transport=httpx.HTTPTransport(retries=0),
    ) as client:
        value = build(config, relative_to=config_path.parent, client=client)
    receipt = sanitized_receipt(value)
    if args.receipt is not None:
        _write_receipt(args.receipt, receipt)
    print(
        json.dumps(
            {key: item for key, item in receipt.items() if key != "receipt_sha256"}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
