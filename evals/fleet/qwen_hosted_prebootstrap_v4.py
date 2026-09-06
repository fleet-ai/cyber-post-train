"""Score-free pre-bootstrap evidence for the hosted Qwen runtime-gate canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

SCHEMA = "fleet-qwen38-hosted-prebootstrap-phase-v1"
PHASES = (
    "00-prebootstrap-entry",
    "01-network-package-install-bypassed",
    "02-pinned-docker-cli-ready",
)


def _digest_without(value: dict[str, Any], field: str) -> str:
    body = {key: item for key, item in value.items() if key != field}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def _write_once(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def record(phase: str, root: Path) -> dict[str, Any]:
    """Publish one content-free phase receipt before scored or network boundaries."""
    if phase not in PHASES or not root.is_absolute() or root == Path("/"):
        raise ValueError("hosted Qwen pre-bootstrap phase binding drifted")
    body = {
        "schema_version": SCHEMA,
        "status": "REACHED",
        "phase": phase,
        "job_uid": os.environ.get("JOB_UID"),
        "pod_uid": os.environ.get("POD_UID"),
        "authority_receipt_sha256": os.environ.get(
            "QWEN_HOSTED_WHOLE_TASK_RELEASE_SHA256"
        ),
        "package_source_receipt_sha256": os.environ.get(
            "QWEN_HOSTED_WHOLE_TASK_PACKAGE_SOURCE_SHA256"
        ),
        "network_package_install": "BYPASSED",
        "docker_cli_source": "PINNED_IMAGE_SHARED_EMPTYDIR",
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
    _write_once(root / "prebootstrap-phases" / f"{phase}.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("record", choices=["record"])
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    record(args.phase, args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
