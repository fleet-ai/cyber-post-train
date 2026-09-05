"""Fail-closed identity gates for pooling Fleet evaluation treatments.

The checks in this module are deliberately independent of any live API.  They
consume sanitized evidence collected before a scored run and answer only
whether two treatments are eligible to share one scientific result block.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROUTE_SCHEMA = "fleet_eval_model_route_evidence_v1"
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
ARCHITECTURE_ALIASES = {
    "amd64": "amd64",
    "x86_64": "amd64",
    "arm64": "arm64",
    "aarch64": "arm64",
}


class TreatmentParityError(ValueError):
    """Evidence is incomplete or the candidate belongs in a separate block."""


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TreatmentParityError(f"{label} must be an object")
    return value


def _text(value: Mapping[str, Any], field: str, label: str) -> str:
    item = value.get(field)
    if not isinstance(item, str) or not item.strip():
        raise TreatmentParityError(f"{label}.{field} must be a non-empty string")
    return item.strip()


def _immutable_revision(value: Mapping[str, Any], field: str, label: str) -> str:
    revision = _text(value, field, label)
    if not (COMMIT_RE.fullmatch(revision) or SHA256_RE.fullmatch(revision)):
        raise TreatmentParityError(
            f"{label}.{field} must be an immutable 40-hex commit or sha256 digest"
        )
    return revision


def _sha256_digest(value: Mapping[str, Any], field: str, label: str) -> str:
    digest = _text(value, field, label)
    if not SHA256_RE.fullmatch(digest):
        raise TreatmentParityError(f"{label}.{field} must be a sha256 digest")
    return digest


def _https_endpoint(value: Mapping[str, Any], field: str, label: str) -> str:
    endpoint = _text(value, field, label)
    parsed = urlparse(endpoint)
    if parsed.scheme != "https" or not parsed.netloc:
        raise TreatmentParityError(f"{label}.{field} must be an HTTPS origin")
    return endpoint.rstrip("/")


def _validated_route_evidence(evidence: Mapping[str, Any], label: str) -> dict[str, Any]:
    if evidence.get("schema") != ROUTE_SCHEMA:
        raise TreatmentParityError(f"{label}.schema must be {ROUTE_SCHEMA}")

    model = _object(evidence.get("model"), f"{label}.model")
    route = _object(evidence.get("route"), f"{label}.route")
    harness = _object(evidence.get("harness"), f"{label}.harness")
    tools = _object(evidence.get("tools"), f"{label}.tools")
    tasks = _object(evidence.get("tasks"), f"{label}.tasks")

    normalized = {
        "model.repository": _text(model, "repository", f"{label}.model"),
        "model.revision": _immutable_revision(model, "revision", f"{label}.model"),
        "route.provider": _text(route, "provider", f"{label}.route"),
        "route.endpoint": _https_endpoint(route, "endpoint", f"{label}.route"),
        "route.revision": _immutable_revision(route, "revision", f"{label}.route"),
        "harness.name": _text(harness, "name", f"{label}.harness"),
        "harness.version": _text(harness, "version", f"{label}.harness"),
        "harness.image_digest": _sha256_digest(
            harness, "image_digest", f"{label}.harness"
        ),
        "harness.context_window": harness.get("context_window"),
        "harness.compaction_policy": _text(
            harness, "compaction_policy", f"{label}.harness"
        ),
        "tools.schema_digest": _sha256_digest(tools, "schema_digest", f"{label}.tools"),
        "tasks.binding": _text(tasks, "binding", f"{label}.tasks"),
    }
    context = normalized["harness.context_window"]
    if isinstance(context, bool) or not isinstance(context, int) or context < 1:
        raise TreatmentParityError(
            f"{label}.harness.context_window must be a positive integer"
        )
    if normalized["tasks.binding"] != "exact_eval_task_version_id":
        raise TreatmentParityError(
            f"{label}.tasks.binding must use exact eval_task_version_id values"
        )
    _text(evidence, "observed_at", label)
    _text(evidence, "evidence_source", label)
    return normalized


def assert_poolable_treatments(
    frozen: Mapping[str, Any], candidate: Mapping[str, Any]
) -> None:
    """Require complete immutable evidence and equality for every treatment selector."""

    expected = _validated_route_evidence(frozen, "frozen")
    observed = _validated_route_evidence(candidate, "candidate")
    differences = [
        field for field, expected_value in expected.items() if observed[field] != expected_value
    ]
    if differences:
        raise TreatmentParityError(
            "candidate must remain a separate result block; treatment differs in: "
            + ", ".join(differences)
        )


def normalize_architecture(value: str) -> str:
    """Normalize the architecture names emitted by uname and OCI inspection."""

    normalized = ARCHITECTURE_ALIASES.get(value.strip().lower())
    if normalized is None:
        raise TreatmentParityError(f"unsupported architecture: {value!r}")
    return normalized


def assert_local_image_matches_host(host_architecture: str, image_architecture: str) -> None:
    """Refuse scored local execution through cross-architecture emulation."""

    host = normalize_architecture(host_architecture)
    image = normalize_architecture(image_architecture)
    if host != image:
        raise TreatmentParityError(
            f"local evaluator image architecture {image} does not match host {host}"
        )


def _read(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return _object(value, path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pool = commands.add_parser("pool", help="validate two sanitized treatment receipts")
    pool.add_argument("--frozen", type=Path, required=True)
    pool.add_argument("--candidate", type=Path, required=True)
    image = commands.add_parser("local-image", help="validate local OCI image architecture")
    image.add_argument("--host-architecture", default=platform.machine())
    image.add_argument("--image-architecture", required=True)
    args = parser.parse_args()
    if args.command == "pool":
        assert_poolable_treatments(_read(args.frozen), _read(args.candidate))
    else:
        assert_local_image_matches_host(args.host_architecture, args.image_architecture)


if __name__ == "__main__":
    main()
