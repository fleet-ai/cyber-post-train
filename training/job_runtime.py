"""Fail-closed runtime gates embedded in generated SFT and online-RL Jobs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from .io import atomic_write_json, digest_json, file_sha256

SFT_RECEIPT_SCHEMA = "cyber_post_train_sft_checkpoint_receipt_v1"
COMPATIBILITY_SCHEMA = "cyber_post_train_model_compatibility_v1"


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required")
    return value


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected an object")
    return value


def verify_compatibility() -> None:
    checkpoint_lock = _json(Path(_required("MODEL_CHECKPOINT_LOCK")))
    lock_expected = {
        "schema": "cyber_post_train_checkpoint_lock_v1",
        "repo": _required("MODEL_ID"),
        "revision": _required("MODEL_REVISION"),
        "weights_manifest_sha256": _required("MODEL_WEIGHTS_MANIFEST_SHA256"),
    }
    for key, value in lock_expected.items():
        if checkpoint_lock.get(key) != value:
            raise ValueError(f"model checkpoint lock {key} mismatch")
    path = Path(_required("MODEL_COMPATIBILITY_RECEIPT"))
    receipt = _json(path)
    expected = {
        "schema": COMPATIBILITY_SCHEMA,
        "status": "succeeded",
        "model_revision": _required("MODEL_REVISION"),
        "training_image": _required("TRAINING_IMAGE"),
        "model_adapter_digest": _required("MODEL_ADAPTER_DIGEST"),
        "driver_contract": _required("MODEL_DRIVER_CONTRACT"),
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"model compatibility receipt {key} mismatch")


def checkpoint_manifest(checkpoint_dir: Path) -> dict[str, Any]:
    files = []
    for path in sorted(item for item in checkpoint_dir.rglob("*") if item.is_file()):
        files.append(
            {
                "path": str(path.relative_to(checkpoint_dir)),
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    if not files:
        raise ValueError("SFT completed without producing checkpoint files")
    return {"schema": "cyber_post_train_checkpoint_manifest_v1", "files": files}


def run_sft() -> None:
    verify_compatibility()
    argv = json.loads(_required("TRAINING_ARGV_JSON"))
    if not isinstance(argv, list) or not argv or not all(isinstance(item, str) for item in argv):
        raise ValueError("TRAINING_ARGV_JSON must be a non-empty argv array")
    subprocess.run(argv, check=True)
    checkpoint_dir = Path(_required("SFT_CHECKPOINT_DIR"))
    manifest = checkpoint_manifest(checkpoint_dir)
    manifest_path = Path(_required("SFT_CHECKPOINT_MANIFEST"))
    atomic_write_json(manifest_path, manifest)
    receipt = {
        "schema": SFT_RECEIPT_SCHEMA,
        "status": "succeeded",
        "sft_job": _required("SFT_JOB_NAME"),
        "model_revision": _required("MODEL_REVISION"),
        "dataset_digest": _required("DATASET_DIGEST"),
        "model_adapter_digest": _required("MODEL_ADAPTER_DIGEST"),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_manifest": str(manifest_path),
        "checkpoint_manifest_digest": digest_json(manifest),
    }
    receipt["receipt_digest"] = digest_json(receipt)
    atomic_write_json(Path(_required("SFT_CHECKPOINT_RECEIPT")), receipt)


def gate_rl() -> None:
    verify_compatibility()
    receipt = _json(Path(_required("SFT_CHECKPOINT_RECEIPT")))
    expected = {
        "schema": SFT_RECEIPT_SCHEMA,
        "status": "succeeded",
        "sft_job": _required("SFT_JOB_NAME"),
        "model_revision": _required("MODEL_REVISION"),
        "dataset_digest": _required("DATASET_DIGEST"),
        "model_adapter_digest": _required("MODEL_ADAPTER_DIGEST"),
        "checkpoint_dir": _required("SFT_CHECKPOINT_DIR"),
        "checkpoint_manifest": _required("SFT_CHECKPOINT_MANIFEST"),
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"SFT checkpoint receipt {key} mismatch")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_digest"}
    if receipt.get("receipt_digest") != digest_json(unsigned):
        raise ValueError("SFT checkpoint receipt digest mismatch")
    manifest = _json(Path(expected["checkpoint_manifest"]))
    if receipt.get("checkpoint_manifest_digest") != digest_json(manifest):
        raise ValueError("SFT checkpoint manifest digest mismatch")
    for item in manifest.get("files") or []:
        path = Path(expected["checkpoint_dir"]) / item["path"]
        if not path.is_file() or path.stat().st_size != item["size"]:
            raise ValueError(f"SFT checkpoint file is missing or changed: {path}")
        if file_sha256(path) != item["sha256"]:
            raise ValueError(f"SFT checkpoint file digest mismatch: {path}")


def run_rl() -> None:
    gate_rl()
    argv = json.loads(_required("TRAINING_ARGV_JSON"))
    if not isinstance(argv, list) or not argv or not all(isinstance(item, str) for item in argv):
        raise ValueError("TRAINING_ARGV_JSON must be a non-empty argv array")
    subprocess.run(argv, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run-sft", "gate-rl", "run-rl"))
    args = parser.parse_args()
    if args.mode == "run-sft":
        run_sft()
    elif args.mode == "gate-rl":
        gate_rl()
    else:
        run_rl()


if __name__ == "__main__":
    main()
