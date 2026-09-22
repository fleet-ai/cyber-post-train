"""GET-only, zero-GPU construction of the sealed lane2 private data package."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import httpx

from . import rl_data
from . import skyrl_lane2_authority as authority
from .sft import read_mapping

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/qualification/qwen38-skyrl-lane2-data-v1.json"
AUTHORITY = ROOT / "configs/qualification/qwen38-skyrl-lane2-authority-v1.json"


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
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
    print(
        json.dumps(
            {
                "status": "built",
                "submitted": value["submitted"],
                "gpus": 0,
                "files": {
                    split: {"rows": item["rows"], "sha256": item["sha256"]}
                    for split, item in value["files"].items()
                },
                "manifest_sha256": value["manifest_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
