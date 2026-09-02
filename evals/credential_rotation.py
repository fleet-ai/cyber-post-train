"""Secret-free rotation receipts and in-memory Kubernetes Secret transport.

Receipt construction and validation operate only on Kubernetes metadata.  The
single helper that reads a Secret value keeps the base64 payload in process
memory and never includes it in a return receipt, command argument, or error.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Any
from uuid import UUID

CREDENTIAL_ROTATION_SCHEMA = "fleet-credential-rotation-receipt-v1"
FLEET_SECRET_REF = {
    "namespace": "fleet-train-jobs",
    "name": "fleet-api",
    "key": "FLEET_API_KEY",
}
_METADATA_FIELDS = {
    "namespace",
    "name",
    "uid",
    "resource_version",
    "data_key_present",
}
_KUBERNETES_NAME = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")
_SECRET_KEY = re.compile(r"^[A-Za-z0-9._-]+$")


class CredentialRotationError(ValueError):
    """A sanitized credential-rotation or Secret-transport failure."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest_without(value: Mapping[str, Any], field: str) -> str:
    unsigned = {key: item for key, item in value.items() if key != field}
    return "sha256:" + hashlib.sha256(canonical_json(unsigned)).hexdigest()


def _timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise CredentialRotationError(f"{label} is not a timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CredentialRotationError(f"{label} is not a timestamp") from exc
    if parsed.tzinfo is None:
        raise CredentialRotationError(f"{label} must be timezone-aware")
    return parsed


def _nonzero_uuid(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise CredentialRotationError(f"{label} is not a UUID")
    try:
        parsed = UUID(value)
    except ValueError as exc:
        raise CredentialRotationError(f"{label} is not a UUID") from exc
    if parsed.int == 0:
        raise CredentialRotationError(f"{label} is a zero UUID")
    return value


def _secret_ref(value: Mapping[str, Any]) -> dict[str, str]:
    if set(value) != {"namespace", "name", "key"}:
        raise CredentialRotationError("Secret reference contains unsupported fields")
    result = {field: value.get(field) for field in ("namespace", "name", "key")}
    if not all(isinstance(item, str) and item for item in result.values()):
        raise CredentialRotationError("Secret reference contains an invalid value")
    if not _KUBERNETES_NAME.fullmatch(result["namespace"]):
        raise CredentialRotationError("Secret namespace is invalid")
    if not _KUBERNETES_NAME.fullmatch(result["name"]):
        raise CredentialRotationError("Secret name is invalid")
    if not _SECRET_KEY.fullmatch(result["key"]):
        raise CredentialRotationError("Secret data key is invalid")
    return result  # type: ignore[return-value]


def safe_secret_metadata(
    value: Mapping[str, Any],
    label: str,
    *,
    expected_secret_ref: Mapping[str, Any] = FLEET_SECRET_REF,
) -> dict[str, Any]:
    expected_ref = _secret_ref(expected_secret_ref)
    if set(value) != _METADATA_FIELDS:
        raise CredentialRotationError(f"{label} contains unsupported fields")
    if (
        value.get("namespace") != expected_ref["namespace"]
        or value.get("name") != expected_ref["name"]
    ):
        raise CredentialRotationError(f"{label} binds the wrong Secret")
    _nonzero_uuid(value.get("uid"), f"{label} UID")
    resource_version = value.get("resource_version")
    if not isinstance(resource_version, str) or not resource_version.isdigit():
        raise CredentialRotationError(f"{label} resourceVersion is invalid")
    if not isinstance(value.get("data_key_present"), bool):
        raise CredentialRotationError(f"{label} data-key marker is invalid")
    return dict(value)


def build_credential_rotation_receipt(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    confirmed_by: str,
    rotation_completed_at: str,
    secret_ref: Mapping[str, Any] = FLEET_SECRET_REF,
) -> dict[str, Any]:
    ref = _secret_ref(secret_ref)
    safe_before = safe_secret_metadata(
        before, "pre-rotation Secret metadata", expected_secret_ref=ref
    )
    safe_after = safe_secret_metadata(
        after, "post-rotation Secret metadata", expected_secret_ref=ref
    )
    if (
        safe_before["uid"] == safe_after["uid"]
        and safe_before["resource_version"] == safe_after["resource_version"]
    ):
        raise CredentialRotationError("Secret UID/resourceVersion did not change across rotation")
    if safe_after["data_key_present"] is not True:
        raise CredentialRotationError("post-rotation Secret lacks the expected data key")
    _timestamp(rotation_completed_at, "rotation completion")
    if not isinstance(confirmed_by, str) or not confirmed_by.strip():
        raise CredentialRotationError("credential rotation receipt lacks an accountable confirmer")
    receipt: dict[str, Any] = {
        "schema_version": CREDENTIAL_ROTATION_SCHEMA,
        "secret_ref": ref,
        "before": safe_before,
        "after": safe_after,
        "rotation_completed_at": rotation_completed_at,
        "rotation_confirmed": True,
        "confirmed_by": confirmed_by,
        "secret_value_observed": False,
        "secret_value_persisted": False,
    }
    receipt["receipt_sha256"] = digest_without(
        {**receipt, "receipt_sha256": None}, "receipt_sha256"
    )
    return receipt


def validate_credential_rotation_receipt(
    receipt: Mapping[str, Any],
    *,
    rotation_not_before: str,
    expected_secret_ref: Mapping[str, Any] = FLEET_SECRET_REF,
) -> dict[str, Any]:
    expected_ref = _secret_ref(expected_secret_ref)
    if receipt.get("schema_version") != CREDENTIAL_ROTATION_SCHEMA:
        raise CredentialRotationError("unsupported credential rotation receipt schema")
    if receipt.get("receipt_sha256") != digest_without(receipt, "receipt_sha256"):
        raise CredentialRotationError("credential rotation receipt digest mismatch")
    if receipt.get("secret_ref") != expected_ref:
        raise CredentialRotationError("credential rotation receipt binds the wrong Secret")
    if receipt.get("rotation_confirmed") is not True:
        raise CredentialRotationError("credential rotation is not confirmed")
    if receipt.get("secret_value_observed") is not False:
        raise CredentialRotationError(
            "credential rotation receipt must not observe the Secret value"
        )
    if receipt.get("secret_value_persisted") is not False:
        raise CredentialRotationError(
            "credential rotation receipt must not persist the Secret value"
        )
    rotated_at = _timestamp(receipt.get("rotation_completed_at"), "rotation completion")
    if rotated_at <= _timestamp(rotation_not_before, "rotation lower bound"):
        raise CredentialRotationError("credential rotation does not postdate the incident boundary")
    confirmed_by = receipt.get("confirmed_by")
    if not isinstance(confirmed_by, str) or not confirmed_by.strip():
        raise CredentialRotationError("credential rotation receipt lacks an accountable confirmer")
    before = safe_secret_metadata(
        receipt.get("before") or {},
        "receipt pre-rotation metadata",
        expected_secret_ref=expected_ref,
    )
    after = safe_secret_metadata(
        receipt.get("after") or {},
        "receipt post-rotation metadata",
        expected_secret_ref=expected_ref,
    )
    if before["uid"] == after["uid"] and before["resource_version"] == after["resource_version"]:
        raise CredentialRotationError(
            "credential rotation receipt does not prove a metadata transition"
        )
    if after["data_key_present"] is not True:
        raise CredentialRotationError(
            "credential rotation receipt lacks the expected post-rotation key"
        )
    return after


def validate_credential_rotation(
    receipt: Mapping[str, Any],
    secret_metadata: Mapping[str, Any],
    *,
    rotation_not_before: str,
    expected_secret_ref: Mapping[str, Any] = FLEET_SECRET_REF,
) -> dict[str, Any]:
    after = validate_credential_rotation_receipt(
        receipt,
        rotation_not_before=rotation_not_before,
        expected_secret_ref=expected_secret_ref,
    )
    live = safe_secret_metadata(
        secret_metadata, "live Secret metadata", expected_secret_ref=expected_secret_ref
    )
    if live != after:
        raise CredentialRotationError("live Secret metadata does not match the rotation receipt")
    return live


def _kubectl_prefix(ref: Mapping[str, Any], context: str | None) -> list[str]:
    parsed = _secret_ref(ref)
    prefix = ["kubectl"]
    if context:
        prefix.extend(("--context", context))
    prefix.extend(("-n", parsed["namespace"], "get", "secret", parsed["name"]))
    return prefix


def _run_secret_safe(
    command: Sequence[str],
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> str:
    try:
        completed = runner(command, check=True, text=True, capture_output=True)
    except (OSError, subprocess.SubprocessError):
        raise CredentialRotationError("Kubernetes Secret read failed") from None
    if not isinstance(completed.stdout, str):
        raise CredentialRotationError("Kubernetes Secret read returned invalid output")
    return completed.stdout


def read_secret_metadata(
    *,
    secret_ref: Mapping[str, Any] = FLEET_SECRET_REF,
    context: str | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Read only safe metadata; the kubectl output cannot contain the Secret value."""

    ref = _secret_ref(secret_ref)
    template = (
        '{{.metadata.namespace}}{{"\\n"}}{{.metadata.name}}{{"\\n"}}'
        '{{.metadata.uid}}{{"\\n"}}{{.metadata.resourceVersion}}{{"\\n"}}'
        f'{{{{if index .data "{ref["key"]}"}}}}true{{{{else}}}}false{{{{end}}}}{{{{"\\n"}}}}'
    )
    output = _run_secret_safe(
        [*_kubectl_prefix(ref, context), "-o", f"go-template={template}"], runner=runner
    )
    lines = output.splitlines()
    if len(lines) != 5 or lines[4] not in {"true", "false"}:
        raise CredentialRotationError("Kubernetes Secret metadata projection is invalid")
    return safe_secret_metadata(
        {
            "namespace": lines[0],
            "name": lines[1],
            "uid": lines[2],
            "resource_version": lines[3],
            "data_key_present": lines[4] == "true",
        },
        "live Secret metadata",
        expected_secret_ref=ref,
    )


def read_secret_snapshot_in_memory(
    *,
    secret_ref: Mapping[str, Any] = FLEET_SECRET_REF,
    context: str | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> tuple[dict[str, Any], str]:
    """Atomically bind safe metadata to a value retained only in process memory."""

    ref = _secret_ref(secret_ref)
    template = (
        '{{.metadata.namespace}}{{"\\n"}}{{.metadata.name}}{{"\\n"}}'
        '{{.metadata.uid}}{{"\\n"}}{{.metadata.resourceVersion}}{{"\\n"}}'
        f'{{{{if index .data "{ref["key"]}"}}}}true{{{{else}}}}false{{{{end}}}}{{{{"\\n"}}}}'
        f'{{{{index .data "{ref["key"]}"}}}}'
    )
    output = _run_secret_safe(
        [*_kubectl_prefix(ref, context), "-o", f"go-template={template}"], runner=runner
    )
    lines = output.splitlines()
    if len(lines) != 6 or lines[4] != "true":
        raise CredentialRotationError("Kubernetes Secret snapshot is invalid")
    metadata = safe_secret_metadata(
        {
            "namespace": lines[0],
            "name": lines[1],
            "uid": lines[2],
            "resource_version": lines[3],
            "data_key_present": True,
        },
        "live Secret metadata",
        expected_secret_ref=ref,
    )
    try:
        secret_bytes = base64.b64decode(lines[5], validate=True)
        secret = secret_bytes.decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        raise CredentialRotationError("Kubernetes Secret value is invalid") from None
    if not secret:
        raise CredentialRotationError("Kubernetes Secret value is empty")
    return metadata, secret
