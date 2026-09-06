"""Stage projected ConfigMap JSON into private regular files for hosted Qwen."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any

SCHEMA = "fleet-qwen38-hosted-private-runtime-inputs-v1"
SOURCE_NAMES = ("package-source.json", "release.json")
MAX_JSON_BYTES = 900_000


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key in hosted Qwen private input")
        value[key] = item
    return value


def _sha256(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _digest_without(value: dict[str, Any], field: str) -> str:
    body = {key: item for key, item in value.items() if key != field}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return _sha256(canonical)


def _parse(raw: bytes) -> dict[str, Any]:
    if not raw or len(raw) > MAX_JSON_BYTES:
        raise ValueError("hosted Qwen private input size drifted")
    value = json.loads(raw, object_pairs_hook=_strict_object)
    if not isinstance(value, dict):
        raise ValueError("hosted Qwen private input root drifted")
    return value


def _validate_private_regular(path: Path) -> bytes:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
        raise ValueError("hosted Qwen private input mode drifted")
    if path.parent.is_symlink() or stat.S_IMODE(path.parent.stat().st_mode) != 0o700:
        raise ValueError("hosted Qwen private input directory drifted")
    return path.read_bytes()


def stage_all(bootstrap_root: Path, destination_root: Path) -> dict[str, Any]:
    """Copy both fixed projected inputs exactly once, then validate the copies."""
    if (
        not bootstrap_root.is_absolute()
        or bootstrap_root == Path("/")
        or bootstrap_root.is_symlink()
        or not bootstrap_root.is_dir()
        or not destination_root.is_absolute()
        or destination_root == Path("/")
        or destination_root.parent.is_symlink()
    ):
        raise ValueError("hosted Qwen private staging root drifted")
    destination_root.mkdir(mode=0o700, parents=False, exist_ok=False)
    bindings = {
        "package-source.json": os.environ.get(
            "QWEN_HOSTED_WHOLE_TASK_PACKAGE_SOURCE_SHA256"
        ),
        "release.json": os.environ.get("QWEN_HOSTED_WHOLE_TASK_RELEASE_SHA256"),
    }
    staged: list[dict[str, Any]] = []
    for name in SOURCE_NAMES:
        source = bootstrap_root / name
        if source.parent != bootstrap_root or not source.is_symlink() or not source.is_file():
            raise ValueError("hosted Qwen projected input shape drifted")
        try:
            source.resolve(strict=True).relative_to(bootstrap_root.resolve(strict=True))
        except (OSError, ValueError) as exc:
            raise ValueError("hosted Qwen projected input target drifted") from exc
        raw = source.read_bytes()
        value = _parse(raw)
        expected = bindings[name]
        if not expected or value.get("receipt_sha256") != expected:
            raise ValueError("hosted Qwen projected input digest binding drifted")
        if _digest_without(value, "receipt_sha256") != expected:
            raise ValueError("hosted Qwen projected input self-digest drifted")
        destination = destination_root / name
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        copied = _validate_private_regular(destination)
        copied_value = _parse(copied)
        if copied != raw or copied_value != value:
            raise ValueError("hosted Qwen private input copy drifted")
        staged.append(
            {
                "name": name,
                "source_was_projected_symlink": True,
                "destination_is_private_regular_file": True,
                "file_sha256": _sha256(copied),
                "receipt_sha256": expected,
                "bytes": len(copied),
            }
        )
    body = {
        "schema_version": SCHEMA,
        "status": "STAGED",
        "job_uid": os.environ.get("JOB_UID"),
        "pod_uid": os.environ.get("POD_UID"),
        "destination_root_sha256": _sha256(str(destination_root).encode()),
        "files": staged,
        "output_roots_created": 0,
        "endpoint_leases_acquired": 0,
        "canonical_claims_created": 0,
        "model_calls": 0,
        "task_calls": 0,
        "session_calls": 0,
        "verifier_calls": 0,
        "scoring_calls": 0,
        "api_mutations": 0,
        "scores_included": False,
        "prompts_or_traces_included": False,
        "credentials_included": False,
    }
    receipt = {**body, "receipt_sha256": _digest_without(body, "receipt_sha256")}
    receipt_path = destination_root / "STAGED.json"
    payload = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    descriptor = os.open(receipt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    _validate_private_regular(receipt_path)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage-all", choices=["stage-all"])
    parser.add_argument("--bootstrap-root", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    args = parser.parse_args()
    stage_all(args.bootstrap_root, args.destination_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
