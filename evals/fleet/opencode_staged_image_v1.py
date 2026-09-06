"""Validate the exact staged OpenCode archive before a cluster DinD load."""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
from pathlib import Path
from typing import Any

from evals.fleet import self_hosted

SCHEMA = "fleet-opencode11827-exact-image-stage-v2"
RECEIPT_PATH = Path(
    "/mnt/sfs/jobs/chris-cyber-opencode11827-image-stage-v2/STAGED.json"
)
RECEIPT_FILE_SHA256 = "sha256:9dc98137b5810d5de188314926204b94954ca392a13116f46bb23428f84b0065"
RECEIPT_SHA256 = "sha256:8d48222771313d3acefd16978cda30564acda4a97277ba25ef23b72459da58d7"
ARCHIVE_PATH = Path(
    "/mnt/sfs/jobs/chris-cyber-opencode11827-image-stage-v2/"
    "opencode-1.18.27-amd64.tar.gz"
)
ARCHIVE_SHA256 = "sha256:578ff2a933f17a19d22ebf8634533651cec2f0ce611e16f3c5fe8efea7940cd3"
ARCHIVE_BYTES = 146_627_813
OCI_INDEX_DIGEST = "sha256:ca4f0b8f50bd051d709c7c0ae5ec47ca31bbff7d2a2ad754c67b9cdf585567cb"
RUNTIME_IMAGE_ID = "sha256:4a46e71e98fbbc67f54dfd75fab15730af5ae575070d7b2ba1ad09ad4fa28b11"


def _regular_file(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    return stat.S_ISREG(metadata.st_mode) and not path.is_symlink()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def validate(
    receipt_path: Path = RECEIPT_PATH, archive_path: Path = ARCHIVE_PATH
) -> dict[str, Any]:
    if not _regular_file(receipt_path) or not _regular_file(archive_path):
        raise ValueError("staged OpenCode evidence must be regular files")
    receipt_bytes = receipt_path.read_bytes()
    if not receipt_bytes.endswith(b"\n") or _sha256(receipt_path) != RECEIPT_FILE_SHA256:
        raise ValueError("staged OpenCode receipt file drifted")
    try:
        value = json.loads(receipt_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError("staged OpenCode receipt is invalid") from exc
    if not isinstance(value, dict):
        raise ValueError("staged OpenCode receipt must be an object")
    expected = {
        "schema_version": SCHEMA,
        "status": "STAGED_EXACT",
        "archive_path": str(ARCHIVE_PATH),
        "archive_sha256": ARCHIVE_SHA256,
        "source_oci_index_digest": OCI_INDEX_DIGEST,
        "runtime_image_id": RUNTIME_IMAGE_ID,
        "platform": {"architecture": "amd64", "os": "linux"},
        "runtime": {
            "opencode_version": "1.18.27",
            "user": "node",
            "working_dir": "/workspace",
        },
        "credentials_included": False,
        "prompts_traces_flags_or_scores_included": False,
    }
    observed = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if (
        value.get("receipt_sha256") != RECEIPT_SHA256
        or value.get("receipt_sha256") != self_hosted.digest_without(value, "receipt_sha256")
        or observed != expected
        or archive_path != ARCHIVE_PATH
        or archive_path.stat().st_size != ARCHIVE_BYTES
        or _sha256(archive_path) != ARCHIVE_SHA256
    ):
        raise ValueError("staged OpenCode artifact drifted")
    return value


def identity() -> dict[str, Any]:
    return {
        "stage_receipt_path": str(RECEIPT_PATH),
        "stage_receipt_sha256": RECEIPT_SHA256,
        "stage_receipt_file_sha256": RECEIPT_FILE_SHA256,
        "archive_path": str(ARCHIVE_PATH),
        "archive_sha256": ARCHIVE_SHA256,
        "archive_bytes": ARCHIVE_BYTES,
        "source_oci_index_digest": OCI_INDEX_DIGEST,
        "runtime_image_id": RUNTIME_IMAGE_ID,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=RECEIPT_PATH)
    parser.add_argument("--archive", type=Path, default=ARCHIVE_PATH)
    args = parser.parse_args()
    validate(args.receipt, args.archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
