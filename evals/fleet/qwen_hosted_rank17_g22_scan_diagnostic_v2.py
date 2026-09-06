"""Payload-blind envelope diagnostic for the failed rank-17 release scan.

The diagnostic enumerates only the two receipt namespaces used by the failed
observer.  It extracts the two top-level envelope strings required by that
observer without decoding any other value, and emits only aggregate counts,
receipt classes, file sizes, and salted path hashes.  It never emits paths,
receipt payloads, identities, prompts, traces, flags, scores, or credentials.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from collections import Counter
from pathlib import Path
from typing import Any

SCHEMA = "fleet-qwen38-hosted-rank17-g22-scan-diagnostic-v2"
PACKAGE_SCHEMA = "fleet-qwen38-hosted-rank17-g22-scan-diagnostic-package-v2"
INCIDENT_JOB_UID = "4c5a5991-c00f-4c09-b3b2-c45c6a825a52"
INCIDENT_POD_UID = "ecaab3e5-6bf0-4226-8eed-46af047d5b48"
CLAIM_ROOT = Path("/mnt/sfs/cell-execution-claims/opencode11827-autocontinue-v1")
JOBS_ROOT = Path("/mnt/sfs/jobs")
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_ANOMALIES = 64
SHA_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class DiagnosticError(RuntimeError):
    """The payload-blind diagnostic failed closed."""


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def digest(value: dict[str, Any]) -> str:
    return sha256(canonical({key: item for key, item in value.items() if key != "receipt_sha256"}))


def seal(value: dict[str, Any]) -> dict[str, Any]:
    return {**value, "receipt_sha256": digest(value)}


def validate_package_source(path: Path, package_root: Path) -> None:
    try:
        root = package_root.resolve(strict=True)
        source = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise DiagnosticError("package_source_unsafe") from exc
    if not source.is_relative_to(root) or not source.is_file():
        raise DiagnosticError("package_source_unsafe")
    try:
        value = json.loads(source.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DiagnosticError("package_source_invalid") from exc
    files = value.get("files") if isinstance(value, dict) else None
    if (
        not isinstance(value, dict)
        or set(value) != {"schema_version", "files", "file_count", "receipt_sha256"}
        or value.get("schema_version") != PACKAGE_SCHEMA
        or not isinstance(files, dict)
        or set(files) != {"qwen_hosted_rank17_g22_scan_diagnostic_v2.py"}
        or value.get("file_count") != 1
        or value.get("receipt_sha256") != digest(value)
    ):
        raise DiagnosticError("package_source_invalid")
    for name, expected in files.items():
        try:
            target = (package_root / name).resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise DiagnosticError("package_file_drifted") from exc
        if (
            not target.is_relative_to(root)
            or not target.is_file()
            or target.stat().st_size > MAX_FILE_BYTES
            or sha256(target.read_bytes()) != expected
        ):
            raise DiagnosticError("package_file_drifted")


def _skip_ws(raw: str, index: int) -> int:
    while index < len(raw) and raw[index] in " \t\r\n":
        index += 1
    return index


def _string_end(raw: str, index: int) -> int:
    if index >= len(raw) or raw[index] != '"':
        raise DiagnosticError("envelope_json_invalid")
    index += 1
    while index < len(raw):
        char = raw[index]
        if char == '"':
            return index + 1
        if char == "\\":
            index += 1
            if index >= len(raw):
                break
            if raw[index] == "u":
                if index + 4 >= len(raw) or not all(
                    value in "0123456789abcdefABCDEF" for value in raw[index + 1 : index + 5]
                ):
                    raise DiagnosticError("envelope_json_invalid")
                index += 4
        elif ord(char) < 0x20:
            raise DiagnosticError("envelope_json_invalid")
        index += 1
    raise DiagnosticError("envelope_json_invalid")


def _decode_string(raw: str, index: int) -> tuple[str, int]:
    end = _string_end(raw, index)
    try:
        value = json.loads(raw[index:end])
    except json.JSONDecodeError as exc:
        raise DiagnosticError("envelope_json_invalid") from exc
    if not isinstance(value, str):
        raise DiagnosticError("envelope_json_invalid")
    return value, end


def _skip_value(raw: str, index: int) -> int:
    index = _skip_ws(raw, index)
    if index >= len(raw):
        raise DiagnosticError("envelope_json_invalid")
    if raw[index] == '"':
        return _string_end(raw, index)
    if raw[index] not in "[{":
        end = index
        while end < len(raw) and raw[end] not in ",}":
            end += 1
        if not raw[index:end].strip():
            raise DiagnosticError("envelope_json_invalid")
        return end
    stack = [raw[index]]
    index += 1
    while index < len(raw) and stack:
        char = raw[index]
        if char == '"':
            index = _string_end(raw, index)
            continue
        if char in "[{":
            stack.append(char)
        elif char in "]}":
            opener = stack.pop()
            if (opener, char) not in {("[", "]"), ("{", "}")}:
                raise DiagnosticError("envelope_json_invalid")
        index += 1
    if stack:
        raise DiagnosticError("envelope_json_invalid")
    return index


def _envelope(path: Path) -> tuple[str | None, str | None]:
    if path.is_symlink() or not path.is_file():
        raise DiagnosticError("envelope_file_unsafe")
    size = path.stat().st_size
    if size <= 0 or size > MAX_FILE_BYTES:
        raise DiagnosticError("envelope_file_unsafe")
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise DiagnosticError("envelope_file_unsafe") from exc
    index = _skip_ws(raw, 0)
    if index >= len(raw) or raw[index] != "{":
        raise DiagnosticError("envelope_json_invalid")
    index += 1
    selected: dict[str, str | None] = {}
    while True:
        index = _skip_ws(raw, index)
        if index < len(raw) and raw[index] == "}":
            index = _skip_ws(raw, index + 1)
            if index != len(raw):
                raise DiagnosticError("envelope_json_invalid")
            break
        key, index = _decode_string(raw, index)
        index = _skip_ws(raw, index)
        if index >= len(raw) or raw[index] != ":":
            raise DiagnosticError("envelope_json_invalid")
        index = _skip_ws(raw, index + 1)
        if key in {"schema_version", "receipt_sha256"}:
            if key in selected:
                raise DiagnosticError("envelope_duplicate_selected_key")
            if index < len(raw) and raw[index] == '"':
                selected[key], index = _decode_string(raw, index)
            else:
                selected[key] = None
                index = _skip_value(raw, index)
        else:
            index = _skip_value(raw, index)
        index = _skip_ws(raw, index)
        if index < len(raw) and raw[index] == ",":
            index += 1
            continue
        if index < len(raw) and raw[index] == "}":
            continue
        raise DiagnosticError("envelope_json_invalid")
    return selected.get("schema_version"), selected.get("receipt_sha256")


def _candidate_paths() -> list[tuple[str, Path, Path]]:
    candidates: list[tuple[str, Path, Path]] = []
    if CLAIM_ROOT.is_symlink() or not CLAIM_ROOT.is_dir():
        raise DiagnosticError("claim_root_unsafe")
    if JOBS_ROOT.is_symlink() or not JOBS_ROOT.is_dir():
        raise DiagnosticError("jobs_root_unsafe")
    for path in CLAIM_ROOT.rglob("*.json"):
        candidates.append(("canonical_claim", CLAIM_ROOT, path))
    for path in JOBS_ROOT.rglob("*.json"):
        if path.name == "ACCEPTED.json":
            candidates.append(("accepted_leaf", JOBS_ROOT, path))
        elif path.parent.name == "accepted":
            candidates.append(("accepted_registry", JOBS_ROOT, path))
    return sorted(candidates, key=lambda row: (row[0], str(row[2].relative_to(row[1]))))


def collect(*, path_hash_salt: str) -> dict[str, Any]:
    if not SHA_RE.fullmatch(path_hash_salt):
        raise DiagnosticError("path_hash_salt_invalid")
    class_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    anomalies: list[dict[str, Any]] = []
    for receipt_class, root, path in _candidate_paths():
        class_counts[receipt_class] += 1
        try:
            schema, receipt_sha = _envelope(path)
            if not schema:
                status = "schema_missing_or_invalid"
            elif SHA_RE.fullmatch(str(receipt_sha)) is None:
                status = "self_digest_missing_or_invalid_shape"
            else:
                status = "envelope_present_digest_unverified"
        except DiagnosticError as exc:
            status = str(exc)
            schema = None
        status_counts[status] += 1
        if status != "envelope_present_digest_unverified" and len(anomalies) < MAX_ANOMALIES:
            relative = str(path.relative_to(root))
            anomalies.append(
                {
                    "receipt_class": receipt_class,
                    "path_sha256": sha256((path_hash_salt + "\0" + relative).encode()),
                    "size_bytes": path.lstat().st_size,
                    "envelope_status": status,
                    "schema_value_sha256": None if schema is None else sha256(schema.encode()),
                }
            )
    return seal(
        {
            "schema_version": SCHEMA,
            "status": "DIAGNOSTIC_COMPLETE",
            "incident": {
                "job_uid": INCIDENT_JOB_UID,
                "pod_uid": INCIDENT_POD_UID,
                "failure_code": "scan_receipt_invalid",
            },
            "candidate_counts_by_class": dict(sorted(class_counts.items())),
            "envelope_status_counts": dict(sorted(status_counts.items())),
            "bounded_anomalies": anomalies,
            "anomaly_limit": MAX_ANOMALIES,
            "payload_values_decoded": False,
            "path_values_emitted": False,
            "model_calls": 0,
            "task_calls": 0,
            "session_calls": 0,
            "verifier_calls": 0,
            "scoring_calls": 0,
            "api_mutations": 0,
            "scores_included": False,
            "prompts_traces_flags_included": False,
            "credentials_included": False,
        }
    )


def write_once(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o400)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise DiagnosticError("output_path_unsafe")
        raw = canonical(value) + b"\n"
        if os.write(fd, raw) != len(raw):
            raise DiagnosticError("output_short_write")
        os.fsync(fd)
    finally:
        os.close(fd)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--path-hash-salt", required=True)
    parser.add_argument("--package-source", type=Path, required=True)
    args = parser.parse_args(argv)
    validate_package_source(args.package_source, args.package_source.parent)
    write_once(args.output, collect(path_hash_salt=args.path_hash_salt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
