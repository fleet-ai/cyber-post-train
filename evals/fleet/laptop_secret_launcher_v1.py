"""Run laptop qualification with one Kubernetes Secret and no credential output."""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

SECRET_NAME = "chris-cyber-opencode-evals-v2"
SECRET_NAMESPACE = "fleet-train-jobs"
SECRET_KEY = "FLEET_API_KEY"
MODULE = "evals.fleet.laptop_opencode_lane_v1"


class SecretLaunchError(RuntimeError):
    """A stable failure which never contains credential material."""


Runner = Callable[..., subprocess.CompletedProcess[bytes]]


def _run(argv: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(list(argv), check=False, **kwargs)


def _decode_secret(encoded: bytes) -> bytes:
    if not encoded.strip():
        raise SecretLaunchError("cluster_secret_key_absent")
    try:
        decoded = base64.b64decode(encoded.strip(), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise SecretLaunchError("cluster_secret_key_invalid_base64") from exc
    if not decoded or b"\x00" in decoded or b"\n" in decoded or b"\r" in decoded:
        raise SecretLaunchError("cluster_secret_key_invalid_value")
    return decoded


def _assert_not_leaked(secret: bytes, encoded: bytes, payload: bytes, where: str) -> None:
    if secret in payload or encoded.strip() in payload:
        raise SecretLaunchError(f"credential_leaked_to_{where}")


def launch(out: Path, *, runner: Runner = _run) -> dict[str, Any]:
    """Inject the decoded key into exactly one child process environment."""
    if out.exists() or out.is_symlink():
        raise SecretLaunchError("qualification_output_already_exists")
    secret_argv = [
        "kubectl",
        "get",
        "secret",
        SECRET_NAME,
        "--namespace",
        SECRET_NAMESPACE,
        "--output",
        "jsonpath={.data.FLEET_API_KEY}",
    ]
    secret_result = runner(secret_argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if secret_result.returncode != 0:
        raise SecretLaunchError("cluster_secret_read_failed")
    encoded = secret_result.stdout
    secret = _decode_secret(encoded)
    try:
        value = secret.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SecretLaunchError("cluster_secret_key_invalid_utf8") from exc
    child_argv = [sys.executable, "-m", MODULE, "--out", str(out)]
    if value in "\x00".join(child_argv):
        raise SecretLaunchError("credential_in_child_argv")
    env = os.environ.copy()
    env[SECRET_KEY] = value
    child = runner(
        child_argv,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    del env[SECRET_KEY]
    _assert_not_leaked(secret, encoded, child.stdout, "child_stdout")
    _assert_not_leaked(secret, encoded, child.stderr, "child_stderr")
    if child.returncode != 0:
        lines = [line.strip() for line in child.stderr.decode("utf-8", "replace").splitlines()]
        terminal = next((line for line in reversed(lines) if line), "no_stderr")
        digest = hashlib.sha256(child.stderr).hexdigest()
        raise SecretLaunchError(
            f"qualification_child_failed:{terminal[:240]}:stderr_sha256={digest}"
        )
    try:
        raw_receipt = out.read_bytes()
        _assert_not_leaked(secret, encoded, raw_receipt, "qualification_receipt")
        receipt = json.loads(raw_receipt)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SecretLaunchError("qualification_receipt_invalid") from exc
    if receipt.get("status") != "QUALIFIED_NON_SCORED":
        raise SecretLaunchError("qualification_receipt_not_passed")
    return {
        "status": "QUALIFIED_NON_SCORED",
        "receipt_path": str(out),
        "receipt_sha256": receipt.get("receipt_sha256"),
        "credential_source": {
            "kind": "kubernetes_secret",
            "namespace": SECRET_NAMESPACE,
            "name": SECRET_NAME,
            "key": SECRET_KEY,
        },
        "credential_in_argv": False,
        "credential_in_stdout": False,
        "credential_in_stderr": False,
        "credential_in_receipt": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = launch(args.out)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
