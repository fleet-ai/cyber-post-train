"""Create exactly one independently reviewed prod10 preview-difference probe."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from cyber_post_train import skyrl_prod10_operator_job as operator_job
from cyber_post_train import skyrl_prod10_operator_launch as operator_launch


def _load(path: Path) -> object:
    return json.loads(path.read_bytes())


def create(args: argparse.Namespace) -> dict:
    root = args.source_root.resolve()
    operation = operator_launch._operation_directory(args.operation_directory)
    summary = _load(operation / "SUMMARY.json")
    if (
        not isinstance(summary, dict)
        or summary.get("status") != "sealed_for_independent_review_no_create"
        or summary.get("create_authorized") is not False
        or summary.get("source_head") != args.source_head
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        != args.source_head
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    ):
        raise ValueError("reviewed preview-difference source or summary changed")
    packet = _load(operation / "PREVIEW_DIFF_PACKET.json")
    package = operator_job.build_operator_package(packet)
    proof = operator_job.validate_operator_package(package)
    previews = _load(operation / "OPERATOR_PREVIEWS.json")
    duplicate = _load(operation / "OPERATOR_DUPLICATE_PROOF.json")
    if (
        package.source_config_map != _load(operation / "OPERATOR_SOURCE_CONFIG_MAP.json")
        or package.packet_config_map != _load(operation / "OPERATOR_PACKET_CONFIG_MAP.json")
        or package.job != _load(operation / "OPERATOR_JOB.json")
        or proof != summary.get("package")
        or packet.get("sha256") != summary.get("packet_sha256")
        or [value.get("sha256") for value in previews] != summary.get("preview_sha256")
        or duplicate.get("sha256") != summary.get("duplicate_sha256")
        or (operation / "OPERATOR_CREATE.jsonl").exists()
    ):
        raise ValueError("reviewed preview-difference package changed")
    return operator_launch.create_once(
        package,
        previews=previews,
        duplicates=duplicate,
        operation_directory=operation,
    )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--source-root", type=Path, required=True)
    value.add_argument("--source-head", required=True)
    value.add_argument("--operation-directory", type=Path, required=True)
    return value


def main() -> None:
    print(json.dumps(create(parser().parse_args()), sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
