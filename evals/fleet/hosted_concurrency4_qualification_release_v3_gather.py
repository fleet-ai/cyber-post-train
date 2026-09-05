"""Gather the exact score-blind G7 evidence needed to render qualifier v3."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import ssl
import stat
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

NAMESPACE = "fleet-train-jobs"
ALLIE_NAME = "allie-dev"
ALLIE_UID = "73dabe56-60f8-4879-be9f-365196c502e3"
SCHEMA = "fleet-hosted-concurrency4-qualification-g7-evidence-v1"
TERMINAL_SCHEMA = "fleet-opencode-autocontinue-generation7-terminal-v1"
CLAIM_SCHEMA = "fleet-statistical-cell-execution-claim-v9"
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_BUNDLE_BYTES = 40 * 1024 * 1024
MAX_AGE_SECONDS = 300
G7 = {
    "qwen3.8-27b": {
        "job": "chris-q38-ac-r004-a1-g7-v1",
        "execution": "77deaade2d420d18602ffdfde93d1a2a9957a32a00d1f644d42053ee72c31535",
    },
    "glm-5.3": {
        "job": "chris-glm53-ac-r013-a1-g7-v1",
        "execution": "b6338d535870aee0afa28519c3078dd6749db8d6f961b3a51e84dfc65cc13524",
    },
}


class GatherError(RuntimeError):
    """Score-blind evidence collection failed closed."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def digest_without(value: dict[str, Any], field: str) -> str:
    return sha256(canonical_json({key: item for key, item in value.items() if key != field}))


def strict_json(raw: bytes) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in rows:
            if key in value:
                raise GatherError("duplicate_json_key")
            value[key] = item
        return value

    try:
        value = json.loads(raw, object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GatherError("invalid_json") from exc
    if not isinstance(value, dict):
        raise GatherError("json_root_not_object")
    return value


def _get(client: Any, path: str) -> dict[str, Any]:
    status_code, value = client.request("GET", path)
    if status_code != 200 or not isinstance(value, dict):
        raise GatherError("kubernetes_read_failed")
    return value


def _validate_allie(value: dict[str, Any]) -> None:
    status = value.get("status") or {}
    statuses = status.get("containerStatuses") or []
    if (
        (value.get("metadata") or {}).get("name") != ALLIE_NAME
        or (value.get("metadata") or {}).get("namespace") != NAMESPACE
        or (value.get("metadata") or {}).get("uid") != ALLIE_UID
        or (value.get("spec") or {}).get("serviceAccountName") != "default"
        or status.get("phase") != "Running"
        or not statuses
        or any(row.get("restartCount") != 0 or row.get("ready") is not True for row in statuses)
    ):
        raise GatherError("allie_dev_identity_or_freshness_invalid")


class InClusterReader:
    def __init__(self) -> None:
        host = os.environ.get("KUBERNETES_SERVICE_HOST")
        port = os.environ.get("KUBERNETES_SERVICE_PORT_HTTPS", "443")
        if not host:
            raise GatherError("in_cluster_service_unavailable")
        token = Path("/var/run/secrets/kubernetes.io/serviceaccount/token").read_text().strip()
        self.base = f"https://{host}:{port}"
        self.headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        self.context = ssl.create_default_context(
            cafile="/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
        )

    def request(self, method: str, path: str) -> tuple[int, dict[str, Any]]:
        request = urllib.request.Request(self.base + path, method=method, headers=self.headers)
        try:
            with urllib.request.urlopen(request, timeout=30, context=self.context) as response:
                raw = response.read(MAX_FILE_BYTES + 1)
                status_code = response.status
        except urllib.error.HTTPError as exc:
            raw = exc.read(64 * 1024 + 1)
            status_code = exc.code
        if len(raw) > MAX_FILE_BYTES:
            raise GatherError("kubernetes_response_too_large")
        return status_code, strict_json(raw)


def _read_exact(path: Path, *, schema: str) -> bytes:
    current = path
    while current != current.parent:
        if current.is_symlink():
            raise GatherError("evidence_path_symlink_forbidden")
        current = current.parent
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        details = os.fstat(fd)
        if not stat.S_ISREG(details.st_mode) or details.st_size > MAX_FILE_BYTES:
            raise GatherError("evidence_file_unsafe")
        chunks: list[bytes] = []
        remaining = MAX_FILE_BYTES + 1
        while remaining:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
    finally:
        os.close(fd)
    if len(raw) > MAX_FILE_BYTES:
        raise GatherError("evidence_file_too_large")
    value = strict_json(raw)
    _reject_protected_keys(value)
    if value.get("schema_version") != schema:
        raise GatherError("evidence_schema_invalid")
    if schema == TERMINAL_SCHEMA and (
        value.get("scores_included") is not False
        or value.get("prompts_or_traces_included") is not False
    ):
        raise GatherError("terminal_privacy_contract_invalid")
    if schema == CLAIM_SCHEMA and value.get("scores_included") not in (None, False):
        raise GatherError("claim_privacy_contract_invalid")
    return raw


def _reject_protected_keys(value: Any) -> None:
    forbidden = {
        "prompt",
        "prompts",
        "trace",
        "traces",
        "flag",
        "flags",
        "score",
        "scores",
        "solution",
        "solutions",
        "messages",
        "transcript",
        "transcripts",
    }
    if isinstance(value, dict):
        if forbidden.intersection(value):
            raise GatherError("protected_evidence_key_forbidden")
        for item in value.values():
            _reject_protected_keys(item)
    elif isinstance(value, list):
        for item in value:
            _reject_protected_keys(item)


def gather_in_cluster(
    *, reader: Any | None = None, shared_root: Path = Path("/shared")
) -> dict[str, Any]:
    reader = reader or InClusterReader()
    allie = _get(reader, f"/api/v1/namespaces/{NAMESPACE}/pods/{ALLIE_NAME}")
    _validate_allie(allie)
    files: dict[str, str] = {}
    for model, binding in G7.items():
        job = _get(reader, f"/apis/batch/v1/namespaces/{NAMESPACE}/jobs/{binding['job']}")
        selector = urllib.parse.quote(f"job-name={binding['job']}", safe="")
        pods = _get(reader, f"/api/v1/namespaces/{NAMESPACE}/pods?labelSelector={selector}")
        terminal = _read_exact(
            shared_root / "jobs" / binding["job"] / "CANARY-TERMINAL.json",
            schema=TERMINAL_SCHEMA,
        )
        claim = _read_exact(
            shared_root
            / "cell-execution-claims"
            / "opencode11827-autocontinue-v1"
            / f"{binding['execution']}.json",
            schema=CLAIM_SCHEMA,
        )
        files[f"{model}-job.json"] = base64.b64encode(canonical_json(job) + b"\n").decode()
        files[f"{model}-pods.json"] = base64.b64encode(canonical_json(pods) + b"\n").decode()
        files[f"{model}-terminal.json"] = base64.b64encode(terminal).decode()
        files[f"{model}-claim.json"] = base64.b64encode(claim).decode()
    receipt = {
        "schema_version": SCHEMA,
        "status": "COLLECTED_PENDING_LOCAL_VALIDATION",
        "collected_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "namespace": NAMESPACE,
        "allie_dev": {"name": ALLIE_NAME, "uid": ALLIE_UID, "fresh": True},
        "allowed_sources": {
            model: {
                "job": binding["job"],
                "terminal": f"/shared/jobs/{binding['job']}/CANARY-TERMINAL.json",
                "claim": (
                    "/shared/cell-execution-claims/opencode11827-autocontinue-v1/"
                    f"{binding['execution']}.json"
                ),
            }
            for model, binding in G7.items()
        },
        "files": files,
        "prompts_traces_flags_or_scores_included": False,
        "credentials_included": False,
    }
    receipt["receipt_sha256"] = digest_without(receipt, "receipt_sha256")
    if len(canonical_json(receipt)) > MAX_BUNDLE_BYTES:
        raise GatherError("evidence_bundle_too_large")
    return receipt


def validate_bundle(value: dict[str, Any], *, maximum_age_seconds: int = MAX_AGE_SECONDS) -> None:
    expected_names = {
        f"{model}-{suffix}.json" for model in G7 for suffix in ("job", "pods", "terminal", "claim")
    }
    if (
        set(value)
        != {
            "schema_version",
            "status",
            "collected_at_utc",
            "namespace",
            "allie_dev",
            "allowed_sources",
            "files",
            "prompts_traces_flags_or_scores_included",
            "credentials_included",
            "receipt_sha256",
        }
        or value.get("schema_version") != SCHEMA
        or value.get("status") != "COLLECTED_PENDING_LOCAL_VALIDATION"
        or value.get("namespace") != NAMESPACE
        or value.get("allie_dev") != {"name": ALLIE_NAME, "uid": ALLIE_UID, "fresh": True}
        or value.get("prompts_traces_flags_or_scores_included") is not False
        or value.get("credentials_included") is not False
        or value.get("receipt_sha256") != digest_without(value, "receipt_sha256")
        or set(value.get("files") or {}) != expected_names
        or set(value.get("allowed_sources") or {}) != set(G7)
    ):
        raise GatherError("evidence_bundle_invalid")
    try:
        observed = datetime.fromisoformat(str(value["collected_at_utc"]).replace("Z", "+00:00"))
    except ValueError as exc:
        raise GatherError("evidence_timestamp_invalid") from exc
    age = (datetime.now(UTC) - observed).total_seconds()
    if age < -30 or age > maximum_age_seconds:
        raise GatherError("evidence_bundle_stale")
    for model, binding in G7.items():
        expected = {
            "job": binding["job"],
            "terminal": f"/shared/jobs/{binding['job']}/CANARY-TERMINAL.json",
            "claim": (
                "/shared/cell-execution-claims/opencode11827-autocontinue-v1/"
                f"{binding['execution']}.json"
            ),
        }
        if value["allowed_sources"][model] != expected:
            raise GatherError("evidence_source_allowlist_drifted")
    for name, encoded in value["files"].items():
        if not isinstance(encoded, str):
            raise GatherError("evidence_payload_invalid")
        try:
            raw = base64.b64decode(encoded, validate=True)
        except ValueError as exc:
            raise GatherError("evidence_payload_invalid") from exc
        expected_schema = CLAIM_SCHEMA if name.endswith("-claim.json") else None
        if name.endswith("-terminal.json"):
            expected_schema = TERMINAL_SCHEMA
        if expected_schema and strict_json(raw).get("schema_version") != expected_schema:
            raise GatherError("evidence_payload_schema_invalid")


def write_bytes_once(path: Path, raw: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o400)
    try:
        if os.write(fd, raw) != len(raw):
            raise GatherError("evidence_short_write")
        os.fsync(fd)
    finally:
        os.close(fd)


def materialize_bundle(value: dict[str, Any], output_dir: Path) -> dict[str, str]:
    validate_bundle(value)
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    digests: dict[str, str] = {}
    for name, encoded in value["files"].items():
        raw = base64.b64decode(encoded, validate=True)
        write_bytes_once(output_dir / name, raw)
        digests[name] = sha256(raw)
    return digests


def gather_via_allie() -> dict[str, Any]:
    source = Path(__file__).read_text()
    command = [
        "kubectl",
        "-n",
        NAMESPACE,
        "exec",
        "-i",
        ALLIE_NAME,
        "--",
        "python3",
        "-c",
        source,
        "in-cluster-gather",
    ]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0 or len(result.stdout) > MAX_BUNDLE_BYTES:
        raise GatherError("allie_gather_transport_failed")
    return strict_json(result.stdout)


def gather_and_validate(root: Path, output_dir: Path) -> dict[str, Any]:
    value = gather_via_allie()
    digests = materialize_bundle(value, output_dir)
    from evals.fleet import hosted_concurrency4_qualification_release_v3 as release

    acceptance = release.generation7_acceptance_binding(root, output_dir)
    evidence = {
        "schema_version": SCHEMA,
        "status": "VALIDATED",
        "collected_at_utc": value["collected_at_utc"],
        "allie_dev": value["allie_dev"],
        "source_receipt_sha256": value["receipt_sha256"],
        "file_sha256": digests,
        "generation7_acceptance_binding_sha256": acceptance["binding_sha256"],
        "prompts_traces_flags_or_scores_included": False,
        "credentials_included": False,
    }
    evidence["receipt_sha256"] = digest_without(evidence, "receipt_sha256")
    write_bytes_once(output_dir / "EVIDENCE.json", canonical_json(evidence) + b"\n")
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("gather", "in-cluster-gather"))
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.command == "in-cluster-gather":
        print(json.dumps(gather_in_cluster(), sort_keys=True))
        return 0
    if args.output_dir is None:
        parser.error("gather requires --output-dir")
    evidence = gather_and_validate(args.repo.resolve(strict=True), args.output_dir)
    print(
        json.dumps(
            {
                "status": evidence["status"],
                "allie_dev_uid": ALLIE_UID,
                "generation7_acceptance_binding_sha256": evidence[
                    "generation7_acceptance_binding_sha256"
                ],
                "prompts_traces_flags_or_scores_included": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
