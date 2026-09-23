"""Render a no-launch package for exact scored-session Fleet reconciliation.

This adapter is deliberately narrower than a retry worker.  It consumes one
terminal held-out launch packet, that source's exact create journal, and one
private reviewed authorization.  The authorization seals the complete
executable and Kubernetes bundle before this module renders a CPU-only Job that
can only run ``stored_session_reconciliation_v2``.  The runtime reobserves the
exact stored sessions twice before one atomic database transaction.  This
module has no Kubernetes create operation.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from evals.fleet import heldout_launch, rollout_ledger
from evals.fleet import stored_session_reconciliation_v2 as reconciliation
from evals.fleet.evaluate import stable_job_preview

PACKET_SCHEMA = "cyber_fleet_existing_scored_session_reconciliation_packet_v1"
PROOF_SCHEMA = "cyber_fleet_existing_scored_session_reconciliation_proof_v1"
SUBSET_PROOF_SCHEMA = "cyber_fleet_existing_scored_session_subset_reconciliation_proof_v1"
PRIVATE_INTENT_SCHEMA = "cyber_fleet_existing_scored_session_reconciliation_intent_v1"
CLOSURE_SCHEMA = "cyber_fleet_existing_scored_session_reconciliation_closure_v1"
NAMESPACE = "fleet-train-jobs"
FAILURE_CODE = "authoritative_scoring_started.runtimeerror"
EVALUATOR_IMAGE = (
    "ghcr.io/astral-sh/uv:python3.12-bookworm@"
    "sha256:9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7"
)
CODE_FILES = {
    "jobs.py": "cyber_post_train/jobs.py",
    "evaluate.py": "evals/fleet/evaluate.py",
    "exact_pass4_crypto.py": "evals/fleet/exact_pass4_crypto.py",
    "exact_pass4_universe.py": "evals/fleet/exact_pass4_universe.py",
    "fixed_proxy.py": "evals/fleet/fixed_proxy.py",
    "opencode_self_hosted.py": "evals/fleet/opencode_self_hosted.py",
    "retry_review_policy.py": "evals/fleet/retry_review_policy.py",
    "rollout_campaign.py": "evals/fleet/rollout_campaign.py",
    "rollout_ledger.py": "evals/fleet/rollout_ledger.py",
    "rollout_postgres.py": "evals/fleet/rollout_postgres.py",
    "rollout_worker.py": "evals/fleet/rollout_worker.py",
    "stored_session_reconciliation.py": "evals/fleet/stored_session_reconciliation.py",
    "stored_session_reconciliation_v2.py": ("evals/fleet/stored_session_reconciliation_v2.py"),
}
DEPENDENCY_PINS = {
    "httpx": "0.28.1",
    "psycopg[binary]": "3.3.5",
    "pyyaml": "6.0.3",
}
TERMINAL_FIELDS = {
    "schema",
    "observed_at",
    "evaluation_identity_sha256",
    "comparison_protocol_sha256",
    "protocol_id",
    "arm_id",
    "job",
    "config_map",
    "workloads",
    "pods",
    "database",
    "output_root",
    "decision",
    "privacy",
    "sha256",
}
PRIVATE_INTENT_FIELDS = {
    "schema",
    "source_launch_packet_sha256",
    "source_terminal_receipt_sha256",
    "source_create_evidence_sha256",
    "namespace",
    "job_name",
    "config_map_name",
    "secret_name",
    "output_root",
    "runtime_intent",
    "executable_closure",
    "bundle_sha256",
    "sha256",
}
CLOSURE_FIELDS = {
    "schema",
    "image",
    "dependency_pins",
    "files_sha256",
    "runtime_intent_file_sha256",
    "sha256",
}
KUBERNETES_NAME = re.compile(r"[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?")
SHA256 = re.compile(r"(?:sha256:)?([0-9a-f]{64})")


class ReconciliationPacketError(ValueError):
    """The reviewed terminal source or rendered package drifted."""


@dataclass(frozen=True)
class Package:
    config_map: dict[str, Any]
    secret: dict[str, Any]
    job: dict[str, Any]
    proof: dict[str, Any]
    authorization_sha256: str
    authorized_bundle_sha256: str

    @property
    def bundle(self) -> dict[str, Any]:
        return {
            "apiVersion": "v1",
            "kind": "List",
            "items": [self.config_map, self.secret, self.job],
        }


def _canonical_digest(value: Any) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise ReconciliationPacketError("reconciliation value is not canonical") from exc
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReconciliationPacketError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise ReconciliationPacketError(f"{label} is not an object")
    return value


def _intent_from_value(value: Any) -> reconciliation.StoredSessionIntent:
    if not isinstance(value, dict) or not isinstance(value.get("selected_cell_ids"), list):
        raise ReconciliationPacketError("private reconciliation intent schema is unsupported")
    try:
        if (
            value.get("schema_version") == reconciliation.SUBSET_INTENT_SCHEMA
            and set(value) == reconciliation.SUBSET_INTENT_FIELDS
            and isinstance(value.get("unselected_cell_ids"), list)
        ):
            return reconciliation.ExactStoredSessionSubsetIntent(
                evaluation_plan_sha256=value["evaluation_plan_sha256"],
                runtime_files_sha256=value["runtime_files_sha256"],
                serving_block=value["serving_block"],
                source_output_root=value["source_output_root"],
                source_database=value["source_database"],
                source_job_uid=value["source_job_uid"],
                source_job_terminal_receipt_sha256=value["source_job_terminal_receipt_sha256"],
                selected_cell_ids=tuple(value["selected_cell_ids"]),
                unselected_cell_ids=tuple(value["unselected_cell_ids"]),
                expected_arm_state_counts=value["expected_arm_state_counts"],
                expected_agent_exit_code=value["expected_agent_exit_code"],
                expected_agent_termination=value["expected_agent_termination"],
                expected_failure_code=value["expected_failure_code"],
                sha256=value["sha256"],
            )
        if (
            value.get("schema_version") != reconciliation.INTENT_SCHEMA
            or set(value) != reconciliation.INTENT_FIELDS
        ):
            raise ReconciliationPacketError("private reconciliation intent schema is unsupported")
        return reconciliation.ExactStoredSessionIntent(
            evaluation_plan_sha256=value["evaluation_plan_sha256"],
            runtime_files_sha256=value["runtime_files_sha256"],
            serving_block=value["serving_block"],
            source_output_root=value["source_output_root"],
            source_database=value["source_database"],
            source_job_uid=value["source_job_uid"],
            source_job_terminal_receipt_sha256=value["source_job_terminal_receipt_sha256"],
            selected_cell_ids=tuple(value["selected_cell_ids"]),
            expected_agent_exit_code=value["expected_agent_exit_code"],
            expected_agent_termination=value["expected_agent_termination"],
            expected_failure_code=value["expected_failure_code"],
            sha256=value["sha256"],
        )
    except (AttributeError, KeyError, TypeError, ValueError, rollout_ledger.LedgerError) as exc:
        raise ReconciliationPacketError("private reconciliation intent is invalid") from exc


def _json_payload(value: Any) -> bytes:
    try:
        return (
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            + b"\n"
        )
    except (TypeError, ValueError) as exc:
        raise ReconciliationPacketError("private reconciliation value is not canonical") from exc


def _file_digest_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _closure_from_value(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != CLOSURE_FIELDS:
        raise ReconciliationPacketError("executable closure is invalid")
    files = value.get("files_sha256")
    if (
        value.get("schema") != CLOSURE_SCHEMA
        or value.get("image") != EVALUATOR_IMAGE
        or value.get("dependency_pins") != DEPENDENCY_PINS
        or not isinstance(files, dict)
        or set(files) != {*CODE_FILES, "run.sh"}
        or any(
            not isinstance(digest, str) or heldout_launch.SHA256.fullmatch(digest) is None
            for digest in files.values()
        )
        or not isinstance(value.get("runtime_intent_file_sha256"), str)
        or heldout_launch.SHA256.fullmatch(value["runtime_intent_file_sha256"]) is None
        or value.get("sha256")
        != _canonical_digest({key: item for key, item in value.items() if key != "sha256"})
    ):
        raise ReconciliationPacketError("executable closure is invalid")
    return value


def _private_intent_from_value(
    value: Any,
) -> tuple[dict[str, Any], reconciliation.ExactStoredSessionIntent, bytes, bytes]:
    if (
        not isinstance(value, dict)
        or set(value) != PRIVATE_INTENT_FIELDS
        or value.get("schema") != PRIVATE_INTENT_SCHEMA
        or value.get("namespace") != NAMESPACE
        or value.get("sha256")
        != _canonical_digest({key: item for key, item in value.items() if key != "sha256"})
    ):
        raise ReconciliationPacketError("private reconciliation packet intent is invalid")
    for field in (
        "source_launch_packet_sha256",
        "source_terminal_receipt_sha256",
        "source_create_evidence_sha256",
        "bundle_sha256",
    ):
        digest = value.get(field)
        if not isinstance(digest, str) or heldout_launch.SHA256.fullmatch(digest) is None:
            raise ReconciliationPacketError(
                f"{field.replace('_', ' ')} is not an exact SHA-256 digest"
            )
    _name(value.get("job_name"), "reconciliation Job")
    _name(value.get("config_map_name"), "reconciliation ConfigMap")
    _name(value.get("secret_name"), "reconciliation Secret")
    runtime_intent = _intent_from_value(value.get("runtime_intent"))
    closure = _closure_from_value(value.get("executable_closure"))
    runtime_payload = _json_payload(value["runtime_intent"])
    closure_payload = _json_payload(closure)
    if closure["runtime_intent_file_sha256"] != _file_digest_bytes(runtime_payload):
        raise ReconciliationPacketError("runtime intent file binding differs")
    return value, runtime_intent, runtime_payload, closure_payload


def _load_private_intent(
    path: Path,
) -> tuple[dict[str, Any], reconciliation.ExactStoredSessionIntent, bytes, bytes]:
    descriptor = -1
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) & 0o077:
            raise ReconciliationPacketError(
                "private reconciliation intent must be a private regular file"
            )
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            payload = handle.read()
        value = json.loads(payload)
    except ReconciliationPacketError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReconciliationPacketError("private reconciliation intent is unreadable") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    packet_intent, runtime_intent, runtime_payload, closure_payload = _private_intent_from_value(
        value
    )
    return packet_intent, runtime_intent, runtime_payload, closure_payload


def _read_regular_nofollow(path: Path, label: str, *, private: bool = False) -> bytes:
    descriptor = -1
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or (
            private and stat.S_IMODE(metadata.st_mode) & 0o077
        ):
            raise ReconciliationPacketError(f"{label} must be a regular file")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            return handle.read()
    except ReconciliationPacketError:
        raise
    except OSError as exc:
        raise ReconciliationPacketError(f"{label} is unreadable") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def write_private_intent(path: Path, value: dict[str, Any]) -> str:
    """Exclusively write one validated private intent mode 0600 without printing it."""

    packet_intent, _, _, _ = _private_intent_from_value(value)
    payload = _json_payload(packet_intent)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except OSError as exc:
        raise ReconciliationPacketError("private reconciliation intent already exists") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return packet_intent["sha256"]


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or (match := SHA256.fullmatch(value)) is None:
        raise ReconciliationPacketError(f"{label} is not a SHA-256 digest")
    return match.group(1)


def _name(value: Any, label: str) -> str:
    if not isinstance(value, str) or KUBERNETES_NAME.fullmatch(value) is None:
        raise ReconciliationPacketError(f"{label} is not a Kubernetes name")
    return value


def _output_root(value: Any, *, source: str) -> str:
    path = PurePosixPath(value) if isinstance(value, str) else PurePosixPath(".")
    source_path = PurePosixPath(source)
    if (
        not isinstance(value, str)
        or not path.is_absolute()
        or path.parts[:4] != ("/", "mnt", "sfs", "jobs")
        or len(path.parts) < 5
        or ".." in path.parts
        or path.as_posix() != value
        or path == source_path
        or path in source_path.parents
        or source_path in path.parents
    ):
        raise ReconciliationPacketError("reconciliation output root is invalid")
    return value


def _code(repo_root: Path) -> tuple[dict[str, str], dict[str, str]]:
    root = repo_root.resolve()
    values: dict[str, str] = {}
    digests: dict[str, str] = {}
    for name, relative in CODE_FILES.items():
        candidate = root / relative
        path = candidate.resolve()
        if candidate.is_symlink() or root not in path.parents or not path.is_file():
            raise ReconciliationPacketError("reconciliation runtime source is invalid")
        values[name] = path.read_text(encoding="utf-8")
        digests[relative] = hashlib.sha256(values[name].encode()).hexdigest()
    return values, digests


def _run_script() -> str:
    modules = " ".join(name for name in CODE_FILES if name != "jobs.py")
    expected_files = json.dumps(sorted([*CODE_FILES, "run.sh"]), separators=(",", ":"))
    expected_closure_fields = json.dumps(sorted(CLOSURE_FIELDS), separators=(",", ":"))
    expected_pins = json.dumps(DEPENDENCY_PINS, sort_keys=True, separators=(",", ":"))
    return f"""#!/usr/bin/env bash
set -euo pipefail
umask 077
: "${{FLEET_API_KEY:?FLEET_API_KEY is required}}"
: "${{ROLLOUT_DATABASE_URL:?ROLLOUT_DATABASE_URL is required}}"
: "${{EVALUATION_DATABASE:?EVALUATION_DATABASE is required}}"
: "${{EVALUATION_DIRECTORY:?EVALUATION_DIRECTORY is required}}"
: "${{RECONCILIATION_OUTPUT:?RECONCILIATION_OUTPUT is required}}"
python - <<'PY'
import hashlib
import json
from pathlib import Path

closure_path = Path("/intent/closure.json")
intent_path = Path("/intent/intent.json")
closure = json.loads(closure_path.read_bytes())
if (
    not isinstance(closure, dict)
    or set(closure) != set(json.loads({expected_closure_fields!r}))
    or closure.get("schema") != {CLOSURE_SCHEMA!r}
    or closure.get("image") != {EVALUATOR_IMAGE!r}
    or closure.get("dependency_pins") != json.loads({expected_pins!r})
    or sorted(closure.get("files_sha256", {{}})) != json.loads({expected_files!r})
):
    raise SystemExit("reconciliation closure schema differs")
body = {{key: value for key, value in closure.items() if key != "sha256"}}
digest = "sha256:" + hashlib.sha256(
    json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
).hexdigest()
if closure.get("sha256") != digest:
    raise SystemExit("reconciliation closure self digest differs")
if closure.get("runtime_intent_file_sha256") != (
    "sha256:" + hashlib.sha256(intent_path.read_bytes()).hexdigest()
):
    raise SystemExit("reconciliation intent bytes differ")
for name, expected in closure["files_sha256"].items():
    observed = "sha256:" + hashlib.sha256((Path("/bootstrap") / name).read_bytes()).hexdigest()
    if observed != expected:
        raise SystemExit("reconciliation executable closure differs")
PY
root=/workspace/cyber-post-train
mkdir -p "$root/cyber_post_train" "$root/evals/fleet"
touch "$root/cyber_post_train/__init__.py" "$root/evals/__init__.py" \
  "$root/evals/fleet/__init__.py"
install -m 0644 /bootstrap/jobs.py "$root/cyber_post_train/jobs.py"
for name in {modules}; do
  install -m 0644 "/bootstrap/$name" "$root/evals/fleet/$name"
done
cd "$root"
exec uv run --no-project --with httpx=={DEPENDENCY_PINS["httpx"]} \
  --with pyyaml=={DEPENDENCY_PINS["pyyaml"]} \
  --with 'psycopg[binary]=={DEPENDENCY_PINS["psycopg[binary]"]}' \
  python -m evals.fleet.stored_session_reconciliation_v2 \
  --evaluation-directory "$EVALUATION_DIRECTORY" \
  --output-root "$RECONCILIATION_OUTPUT" \
  --postgres-admin-dsn-env ROLLOUT_DATABASE_URL \
  --postgres-database "$EVALUATION_DATABASE" \
  --intent /intent/intent.json
"""


def _terminal_receipt_schema(terminal: dict[str, Any]) -> None:
    try:
        observed_at = datetime.fromisoformat(terminal["observed_at"].replace("Z", "+00:00"))
        job = terminal["job"]
        config_map = terminal["config_map"]
        database = terminal["database"]
        summary = database["summary"]
        output = terminal["output_root"]
        decision = terminal["decision"]
        privacy = terminal["privacy"]
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ReconciliationPacketError("source terminal receipt shape differs") from exc
    resource_rows = (terminal.get("workloads"), terminal.get("pods"))
    if (
        set(terminal) != TERMINAL_FIELDS
        or terminal.get("schema") != heldout_launch.TERMINAL_SCHEMA
        or observed_at.tzinfo is None
        or observed_at.utcoffset() != UTC.utcoffset(observed_at)
        or any(
            not isinstance(terminal.get(field), str)
            or heldout_launch.SHA256.fullmatch(terminal[field]) is None
            for field in (
                "evaluation_identity_sha256",
                "comparison_protocol_sha256",
                "sha256",
            )
        )
        or any(
            not isinstance(terminal.get(field), str) or not terminal[field]
            for field in ("protocol_id", "arm_id")
        )
        or not all(
            isinstance(value, dict)
            for value in (job, config_map, database, summary, output, decision, privacy)
        )
        or set(job) != {"name", "uid", "terminal_condition", "succeeded", "failed"}
        or set(config_map) != {"name", "uid"}
        or set(database) != {"name", "summary"}
        or set(summary)
        != {
            "total",
            "local_results",
            "by_state",
            "by_serving_block",
            "stale_active",
            "plan_sha256",
        }
        or set(output) != {"path", "exists"}
        or set(decision)
        != {
            "capability_result_status",
            "score_blind_reconciliation_required",
            "unresolved_cells",
            "rollout_retry_performed",
            "score_read_or_generated",
        }
        or set(privacy)
        != {
            "prompts_responses_flags_rewards_or_trace_content_included",
            "score_values_included",
            "credentials_included",
        }
        or not all(isinstance(rows, list) for rows in resource_rows)
        or any(
            not isinstance(row, dict)
            or set(row) != {"name", "uid", "phase"}
            or not isinstance(row["name"], str)
            or not isinstance(row["uid"], str)
            or heldout_launch.KUBERNETES_UID.fullmatch(row["uid"]) is None
            or (row["phase"] is not None and not isinstance(row["phase"], str))
            for rows in resource_rows
            for row in rows
        )
        or not isinstance(job["uid"], str)
        or heldout_launch.KUBERNETES_UID.fullmatch(job["uid"]) is None
        or not isinstance(job["name"], str)
        or job["terminal_condition"] not in {"Complete", "Failed"}
        or not isinstance(config_map["uid"], str)
        or heldout_launch.KUBERNETES_UID.fullmatch(config_map["uid"]) is None
        or not isinstance(config_map["name"], str)
        or not isinstance(database["name"], str)
        or any(type(job[field]) is not int or job[field] < 0 for field in ("succeeded", "failed"))
        or any(
            type(summary[field]) is not int or summary[field] < 0
            for field in ("total", "local_results", "stale_active")
        )
        or not isinstance(summary["by_state"], dict)
        or not isinstance(summary["by_serving_block"], list)
        or (
            summary["plan_sha256"] is not None
            and (
                not isinstance(summary["plan_sha256"], str)
                or SHA256.fullmatch(summary["plan_sha256"]) is None
            )
        )
        or not isinstance(output["path"], str)
        or not isinstance(output["exists"], bool)
        or type(decision["unresolved_cells"]) is not int
        or decision["unresolved_cells"] < 0
        or not isinstance(decision["capability_result_status"], str)
        or any(
            not isinstance(decision[field], bool)
            for field in (
                "score_blind_reconciliation_required",
                "rollout_retry_performed",
                "score_read_or_generated",
            )
        )
        or any(not isinstance(value, bool) for value in privacy.values())
    ):
        raise ReconciliationPacketError("source terminal receipt shape differs")


def _source_create_evidence(
    path: Path,
    *,
    source: heldout_launch.Package,
    terminal: dict[str, Any],
) -> str:
    try:
        payload = _read_regular_nofollow(path, "source create evidence")
        records = [json.loads(line) for line in payload.splitlines() if line.strip()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReconciliationPacketError("source create evidence is unreadable") from exc
    if len(records) != 2 or any(not isinstance(record, dict) for record in records):
        raise ReconciliationPacketError("source create evidence is not one successful create")
    intent, response = records
    intent_fields = {
        "schema",
        "state",
        "packet_sha256",
        "evaluation_identity_sha256",
        "comparison_protocol_sha256",
        "protocol_id",
        "arm_id",
        "namespace",
        "job_name",
        "config_map_name",
        "output_root",
        "database",
        "server_preview_sha256",
        "first_duplicate_census",
        "final_duplicate_census",
    }
    response_fields = {
        "state",
        "submitted",
        "gpus",
        "job_name",
        "job_uid",
        "config_map_name",
        "config_map_uid",
        "evaluation_identity_sha256",
        "comparison_protocol_sha256",
    }
    packet = source.packet
    expected_packet_sha256 = _file_digest_bytes(
        _read_regular_nofollow(packet.path, "source launch packet")
    )
    zero_census = {"jobs": 0, "config_maps": 0, "pods": 0}
    job = terminal["job"]
    config_map = terminal["config_map"]
    censuses = (intent.get("first_duplicate_census"), intent.get("final_duplicate_census"))
    if any(
        (
            set(intent) != intent_fields,
            intent.get("schema") != "cyber_fleet_heldout_create_intent_v1",
            intent.get("state") != "KUBECTL_CREATE_INTENT_DO_NOT_RETRY",
            intent.get("packet_sha256") != expected_packet_sha256,
            intent.get("evaluation_identity_sha256") != packet.identity_sha256,
            intent.get("comparison_protocol_sha256")
            != packet.identity["comparison_protocol_sha256"],
            intent.get("protocol_id") != packet.identity["protocol_id"],
            intent.get("arm_id") != packet.identity["arm_id"],
            intent.get("namespace") != packet.namespace,
            intent.get("job_name") != packet.job_name,
            intent.get("config_map_name") != packet.config_map_name,
            intent.get("output_root") != packet.output_root,
            intent.get("database") != packet.database,
            any(
                not isinstance(census, dict)
                or set(census) != set(zero_census)
                or any(type(census[key]) is not int or census[key] != 0 for key in zero_census)
                for census in censuses
            ),
            set(response) != response_fields,
            response.get("state") != "KUBECTL_CREATE_RESPONSE",
            response.get("submitted") is not True,
            type(response.get("gpus")) is not int or response.get("gpus") != 0,
            response.get("job_name") != packet.job_name,
            response.get("job_uid") != job["uid"],
            response.get("config_map_name") != packet.config_map_name,
            response.get("config_map_uid") != config_map["uid"],
            response.get("evaluation_identity_sha256") != packet.identity_sha256,
            response.get("comparison_protocol_sha256")
            != packet.identity["comparison_protocol_sha256"],
        )
    ):
        raise ReconciliationPacketError("source create evidence identity differs")
    if (
        not isinstance(intent.get("server_preview_sha256"), str)
        or heldout_launch.SHA256.fullmatch(intent["server_preview_sha256"]) is None
    ):
        raise ReconciliationPacketError("source server preview is not an exact SHA-256 digest")
    if (
        not isinstance(response.get("job_uid"), str)
        or heldout_launch.KUBERNETES_UID.fullmatch(response["job_uid"]) is None
        or not isinstance(response.get("config_map_uid"), str)
        or heldout_launch.KUBERNETES_UID.fullmatch(response["config_map_uid"]) is None
    ):
        raise ReconciliationPacketError("source create evidence UID differs")
    return _file_digest_bytes(payload)


def _terminal_source(
    source: heldout_launch.Package,
    terminal: dict[str, Any],
    intent: reconciliation.StoredSessionIntent,
) -> tuple[str, int]:
    packet = source.packet
    _terminal_receipt_schema(terminal)
    supplied = _digest(terminal.get("sha256"), "source terminal receipt")
    expected = _canonical_digest({key: value for key, value in terminal.items() if key != "sha256"})
    if supplied != _digest(expected, "source terminal receipt"):
        raise ReconciliationPacketError("source terminal receipt self digest differs")
    job = terminal.get("job")
    database = terminal.get("database")
    decision = terminal.get("decision")
    privacy = terminal.get("privacy")
    output = terminal.get("output_root")
    config_map = terminal.get("config_map")
    if not all(
        isinstance(value, dict) for value in (job, database, decision, privacy, output, config_map)
    ):
        raise ReconciliationPacketError("source terminal receipt shape differs")
    summary = database.get("summary")
    if not isinstance(summary, dict) or not isinstance(summary.get("by_state"), dict):
        raise ReconciliationPacketError("source terminal database summary differs")
    by_state = summary["by_state"]
    total = summary.get("total")
    selected = len(intent.selected_cell_ids)
    subset = isinstance(intent, reconciliation.ExactStoredSessionSubsetIntent)
    expected_states = intent.expected_arm_state_counts if subset else None
    routes = source.evaluation_config.get("routes")
    route = routes.get(intent.serving_block) if isinstance(routes, dict) else None
    task_versions = route.get("task_versions") if isinstance(route, dict) else None
    route_counts = summary.get("by_serving_block")
    state_names = set(rollout_ledger.STATES)
    route_rows_valid = isinstance(route_counts, list) and all(
        isinstance(row, dict)
        and set(row) == {"serving_block", "state", "count"}
        and row["serving_block"] == intent.serving_block
        and row["state"] in state_names
        and type(row["count"]) is int
        and row["count"] > 0
        for row in route_counts
    )
    if (
        type(total) is not int
        or total <= 0
        or not isinstance(routes, dict)
        or set(routes) != {intent.serving_block}
        or not isinstance(task_versions, list)
        or total != len(task_versions) * source.evaluation_config.get("pass_k", 0)
        or selected <= 0
        or set(by_state) != state_names
        or any(type(by_state[state]) is not int or by_state[state] < 0 for state in state_names)
        or sum(by_state.values()) != total
        or (subset and by_state != expected_states)
        or (not subset and by_state.get("retry_review") != selected)
        or (not subset and by_state.get("accepted") != total - selected)
        or any(
            by_state.get(state) != 0
            for state in ("pending", *rollout_ledger.ACTIVE_STATES, "terminal")
        )
        or summary.get("local_results") != total
        or summary.get("stale_active") != 0
        or not route_rows_valid
        or sum(row["count"] for row in route_counts) != total
        or {
            state: sum(row["count"] for row in route_counts if row["state"] == state)
            for state in state_names
        }
        != by_state
    ):
        raise ReconciliationPacketError("source terminal ledger is not exactly review-held")
    plan_sha256 = _digest(summary.get("plan_sha256"), "source evaluation plan")
    if any(
        (
            terminal.get("evaluation_identity_sha256") != packet.identity_sha256,
            terminal.get("comparison_protocol_sha256")
            != packet.identity["comparison_protocol_sha256"],
            terminal.get("protocol_id") != packet.identity["protocol_id"],
            terminal.get("arm_id") != packet.identity["arm_id"],
            job.get("name") != packet.job_name,
            job.get("uid") != intent.source_job_uid,
            job.get("terminal_condition") != "Failed",
            job.get("succeeded") != 0,
            job.get("failed") != 1,
            config_map.get("name") != packet.config_map_name,
            database.get("name") != packet.database,
            output != {"path": packet.output_root, "exists": True},
            plan_sha256 != intent.evaluation_plan_sha256,
            intent.source_output_root != packet.output_root,
            intent.source_database != packet.database,
            intent.source_job_terminal_receipt_sha256 != supplied,
            intent.expected_agent_exit_code != 0,
            intent.expected_agent_termination != "output_limit",
            intent.expected_failure_code != FAILURE_CODE,
            packet.identity.get("retry_limit") != 0,
            source.evaluation_config.get("max_reviewed_infrastructure_retries") != 0,
            decision
            != {
                "capability_result_status": "not_interpreted",
                "score_blind_reconciliation_required": True,
                "unresolved_cells": by_state.get("retry_review"),
                "rollout_retry_performed": False,
                "score_read_or_generated": False,
            },
            privacy
            != {
                "prompts_responses_flags_rewards_or_trace_content_included": False,
                "score_values_included": False,
                "credentials_included": False,
            },
        )
    ):
        raise ReconciliationPacketError("source terminal or private intent identity differs")
    return plan_sha256, total


def _runtime_intent_value(
    source: heldout_launch.Package,
    terminal: dict[str, Any],
    selected_cell_ids: list[str],
    unselected_cell_ids: list[str] | None = None,
) -> dict[str, Any]:
    job = terminal.get("job")
    database = terminal.get("database")
    summary = database.get("summary") if isinstance(database, dict) else None
    routes = source.evaluation_config.get("routes")
    if (
        source.packet.namespace != NAMESPACE
        or not isinstance(job, dict)
        or not isinstance(summary, dict)
        or not isinstance(routes, dict)
        or len(routes) != 1
    ):
        raise ReconciliationPacketError("source cannot bind a private reconciliation intent")
    body = {
        "schema_version": (
            reconciliation.SUBSET_INTENT_SCHEMA
            if unselected_cell_ids is not None
            else reconciliation.INTENT_SCHEMA
        ),
        "evaluation_plan_sha256": _digest(summary.get("plan_sha256"), "source evaluation plan"),
        "runtime_files_sha256": reconciliation.runtime_identity(),
        "serving_block": next(iter(routes)),
        "source_output_root": source.packet.output_root,
        "source_database": source.packet.database,
        "source_job_uid": job.get("uid"),
        "source_job_terminal_receipt_sha256": _digest(
            terminal.get("sha256"), "source terminal receipt"
        ),
        "selected_cell_ids": list(selected_cell_ids),
        "expected_agent_exit_code": 0,
        "expected_agent_termination": "output_limit",
        "expected_failure_code": FAILURE_CODE,
    }
    if unselected_cell_ids is not None:
        by_state = summary.get("by_state")
        if not isinstance(by_state, dict):
            raise ReconciliationPacketError("source terminal arm census differs")
        body.update(
            expected_arm_state_counts=dict(by_state),
            unselected_cell_ids=list(unselected_cell_ids),
        )
    value = {**body, "sha256": _canonical_digest(body).removeprefix("sha256:")}
    _intent_from_value(value)
    return value


def _executable_closure(
    *,
    code_sha256: dict[str, str],
    run_script: str,
    runtime_intent_payload: bytes,
) -> dict[str, Any]:
    files = {name: "sha256:" + code_sha256[relative] for name, relative in CODE_FILES.items()}
    files["run.sh"] = _file_digest_bytes(run_script.encode())
    body = {
        "schema": CLOSURE_SCHEMA,
        "image": EVALUATOR_IMAGE,
        "dependency_pins": DEPENDENCY_PINS,
        "files_sha256": files,
        "runtime_intent_file_sha256": _file_digest_bytes(runtime_intent_payload),
    }
    closure = {**body, "sha256": _canonical_digest(body)}
    _closure_from_value(closure)
    return closure


def _objects(
    *,
    source: heldout_launch.Package,
    job_name: str,
    config_map_name: str,
    secret_name: str,
    output_root: str,
    code: dict[str, str],
    runtime_intent_payload: bytes,
    closure_payload: bytes,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config_map = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": config_map_name, "namespace": NAMESPACE},
        "immutable": True,
        "data": code,
    }
    secret = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {"name": secret_name, "namespace": NAMESPACE},
        "immutable": True,
        "type": "Opaque",
        "data": {
            "closure.json": base64.b64encode(closure_payload).decode("ascii"),
            "intent.json": base64.b64encode(runtime_intent_payload).decode("ascii"),
        },
    }
    labels = {
        "cyber-post-train.fleet.ai/experiment": job_name,
        "cyber-post-train.fleet.ai/owner": "chris",
        "kueue.x-k8s.io/queue-name": "training-lq",
    }
    job = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job_name,
            "namespace": NAMESPACE,
            "labels": labels,
            "annotations": {
                heldout_launch.FAILURE_ALERT_ANNOTATION: heldout_launch.FAILURE_ALERT_OFF,
                heldout_launch.CREATE_ONCE_ANNOTATION: "true",
            },
        },
        "spec": {
            "backoffLimit": 0,
            "activeDeadlineSeconds": 3600,
            "template": {
                "metadata": {
                    "labels": {
                        "cyber-post-train.fleet.ai/experiment": job_name,
                        "cyber-post-train.fleet.ai/owner": "chris",
                        "cyber-post-train.fleet.ai/postgres-client": "true",
                    }
                },
                "spec": {
                    "priorityClassName": "c1",
                    "restartPolicy": "Never",
                    "nodeSelector": {
                        "kubernetes.io/arch": "amd64",
                        "workload": "fleetai-training-ng-cpu",
                    },
                    "tolerations": [
                        {
                            "key": "workload",
                            "operator": "Equal",
                            "value": "fleetai-training-ng-cpu",
                            "effect": "NoSchedule",
                        }
                    ],
                    "containers": [
                        {
                            "name": "reconcile",
                            "image": EVALUATOR_IMAGE,
                            "command": ["/bin/bash", "-ceu", "--"],
                            "args": ["exec /bin/bash /bootstrap/run.sh\n"],
                            "env": [
                                {
                                    "name": "EVALUATION_DIRECTORY",
                                    "value": source.packet.output_root,
                                },
                                {"name": "EVALUATION_DATABASE", "value": source.packet.database},
                                {"name": "RECONCILIATION_OUTPUT", "value": output_root},
                                {
                                    "name": "FLEET_API_KEY",
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "name": "chris-cyber-opencode-evals-v2",
                                            "key": "FLEET_API_KEY",
                                        }
                                    },
                                },
                                {
                                    "name": "ROLLOUT_DATABASE_URL",
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "name": "chris-cyber-rollout-postgres-v1",
                                            "key": "ROLLOUT_DATABASE_URL",
                                        }
                                    },
                                },
                            ],
                            "resources": {
                                "requests": {
                                    "cpu": "1",
                                    "memory": "2Gi",
                                    "ephemeral-storage": "2Gi",
                                },
                                "limits": {
                                    "cpu": "2",
                                    "memory": "4Gi",
                                    "ephemeral-storage": "8Gi",
                                },
                            },
                            "volumeMounts": [
                                {
                                    "name": "bootstrap",
                                    "mountPath": "/bootstrap",
                                    "readOnly": True,
                                },
                                {"name": "intent", "mountPath": "/intent", "readOnly": True},
                                {"name": "workspace", "mountPath": "/workspace"},
                                {"name": "sfs", "mountPath": "/mnt/sfs"},
                            ],
                        }
                    ],
                    "volumes": [
                        {
                            "name": "bootstrap",
                            "configMap": {"name": config_map_name, "defaultMode": 0o444},
                        },
                        {
                            "name": "intent",
                            "secret": {"secretName": secret_name, "defaultMode": 0o400},
                        },
                        {"name": "workspace", "emptyDir": {"sizeLimit": "5Gi"}},
                        {"name": "sfs", "persistentVolumeClaim": {"claimName": "sfs-shared"}},
                    ],
                },
            },
        },
    }
    return config_map, secret, job


def build_private_intent_value(
    *,
    repo_root: Path,
    source_launch_packet: Path,
    source_terminal_receipt: Path,
    source_create_evidence: Path,
    selected_cell_ids: list[str],
    unselected_cell_ids: list[str] | None = None,
    job_name: str,
    config_map_name: str,
    secret_name: str,
    output_root: str,
) -> dict[str, Any]:
    """Build a full, immutable authorization from a reviewed cell roster.

    The caller must store this value mode ``0600``.  This function never queries
    the database, discovers cells, or makes a model/scoring call.
    """

    source = heldout_launch.build_package(source_launch_packet)
    terminal = _load_json(source_terminal_receipt, "source terminal receipt")
    runtime_value = _runtime_intent_value(source, terminal, selected_cell_ids, unselected_cell_ids)
    runtime_intent = _intent_from_value(runtime_value)
    _terminal_source(source, terminal, runtime_intent)
    create_evidence_sha256 = _source_create_evidence(
        source_create_evidence, source=source, terminal=terminal
    )
    names = {
        _name(job_name, "reconciliation Job"),
        _name(config_map_name, "reconciliation ConfigMap"),
        _name(secret_name, "reconciliation Secret"),
    }
    if len(names) != 3 or source.packet.job_name in names or source.packet.config_map_name in names:
        raise ReconciliationPacketError("reconciliation object names are not fresh and distinct")
    output = _output_root(output_root, source=source.packet.output_root)
    code, code_sha256 = _code(repo_root)
    packaged_runtime = {
        name: code_sha256[CODE_FILES[name]] for name in reconciliation.RUNTIME_FILES
    }
    if packaged_runtime != runtime_intent.runtime_files_sha256:
        raise ReconciliationPacketError("packaged reconciliation runtime identity differs")
    run_script = _run_script()
    code["run.sh"] = run_script
    runtime_payload = _json_payload(runtime_value)
    closure = _executable_closure(
        code_sha256=code_sha256,
        run_script=run_script,
        runtime_intent_payload=runtime_payload,
    )
    closure_payload = _json_payload(closure)
    config_map, secret, job = _objects(
        source=source,
        job_name=job_name,
        config_map_name=config_map_name,
        secret_name=secret_name,
        output_root=output,
        code=code,
        runtime_intent_payload=runtime_payload,
        closure_payload=closure_payload,
    )
    bundle_sha256 = _canonical_digest(
        {"apiVersion": "v1", "kind": "List", "items": [config_map, secret, job]}
    )
    body = {
        "schema": PRIVATE_INTENT_SCHEMA,
        "source_launch_packet_sha256": _file_digest_bytes(
            _read_regular_nofollow(source_launch_packet, "source launch packet")
        ),
        "source_terminal_receipt_sha256": terminal["sha256"],
        "source_create_evidence_sha256": create_evidence_sha256,
        "namespace": NAMESPACE,
        "job_name": job_name,
        "config_map_name": config_map_name,
        "secret_name": secret_name,
        "output_root": output,
        "runtime_intent": runtime_value,
        "executable_closure": closure,
        "bundle_sha256": bundle_sha256,
    }
    value = {**body, "sha256": _canonical_digest(body)}
    _private_intent_from_value(value)
    return value


def render(
    *,
    repo_root: Path,
    source_launch_packet: Path,
    source_terminal_receipt: Path,
    source_create_evidence: Path,
    private_intent: Path,
    job_name: str,
    config_map_name: str,
    secret_name: str,
    output_root: str,
) -> Package:
    """Render one immutable package; do not create any Kubernetes object."""

    source = heldout_launch.build_package(source_launch_packet)
    if source.packet.namespace != NAMESPACE:
        raise ReconciliationPacketError("source held-out namespace differs")
    packet_intent, intent, intent_payload, closure_payload = _load_private_intent(private_intent)
    terminal = _load_json(source_terminal_receipt, "source terminal receipt")
    plan_sha256, total = _terminal_source(source, terminal, intent)
    if (
        packet_intent["source_launch_packet_sha256"]
        != _file_digest_bytes(_read_regular_nofollow(source_launch_packet, "source launch packet"))
        or packet_intent["source_terminal_receipt_sha256"] != terminal["sha256"]
        or packet_intent["source_create_evidence_sha256"]
        != _source_create_evidence(source_create_evidence, source=source, terminal=terminal)
    ):
        raise ReconciliationPacketError("reviewed source evidence differs")
    names = {
        _name(job_name, "reconciliation Job"),
        _name(config_map_name, "reconciliation ConfigMap"),
        _name(secret_name, "reconciliation Secret"),
    }
    if len(names) != 3 or source.packet.job_name in names or source.packet.config_map_name in names:
        raise ReconciliationPacketError("reconciliation object names are not fresh and distinct")
    output = _output_root(output_root, source=source.packet.output_root)
    if (
        packet_intent["job_name"] != job_name
        or packet_intent["config_map_name"] != config_map_name
        or packet_intent["secret_name"] != secret_name
        or packet_intent["output_root"] != output
    ):
        raise ReconciliationPacketError("render request differs from reviewed intent")
    code, code_sha256 = _code(repo_root)
    packaged_runtime = {
        name: code_sha256[CODE_FILES[name]] for name in reconciliation.RUNTIME_FILES
    }
    if packaged_runtime != intent.runtime_files_sha256:
        raise ReconciliationPacketError("packaged reconciliation runtime identity differs")
    run_script = _run_script()
    code["run.sh"] = run_script
    observed_closure = _executable_closure(
        code_sha256=code_sha256,
        run_script=run_script,
        runtime_intent_payload=intent_payload,
    )
    if observed_closure != packet_intent["executable_closure"]:
        raise ReconciliationPacketError("reviewed executable closure differs")
    config_map, secret, job = _objects(
        source=source,
        job_name=job_name,
        config_map_name=config_map_name,
        secret_name=secret_name,
        output_root=output,
        code=code,
        runtime_intent_payload=intent_payload,
        closure_payload=closure_payload,
    )
    bundle = {"apiVersion": "v1", "kind": "List", "items": [config_map, secret, job]}
    if _canonical_digest(bundle) != packet_intent["bundle_sha256"]:
        raise ReconciliationPacketError("reviewed Kubernetes bundle differs")
    proof_body = {
        "schema": PROOF_SCHEMA,
        "packet_schema": PACKET_SCHEMA,
        "source_evaluation_identity_sha256": source.packet.identity_sha256,
        "source_terminal_receipt_sha256": "sha256:" + intent.source_job_terminal_receipt_sha256,
        "source_job_name": source.packet.job_name,
        "source_job_uid": intent.source_job_uid,
        "source_database": intent.source_database,
        "source_plan_sha256": "sha256:" + plan_sha256,
        "reviewed_intent_sha256": packet_intent["sha256"],
        "runtime_intent_sha256": "sha256:" + intent.sha256,
        "source_create_evidence_sha256": packet_intent["source_create_evidence_sha256"],
        "source_config_map_uid": terminal["config_map"]["uid"],
        "executable_closure_sha256": observed_closure["sha256"],
        "bundle_sha256": packet_intent["bundle_sha256"],
        "runtime_files_sha256": intent.runtime_files_sha256,
        "selected_cell_count": len(intent.selected_cell_ids),
        "source_total_cell_count": total,
        "operation": "accept_existing_scored_session",
        "model_generation_allowed": False,
        "scoring_call_allowed": False,
        "double_authoritative_observation_before_mutation": True,
        "single_atomic_database_transaction": True,
        "job_name": job_name,
        "config_map_name": config_map_name,
        "secret_name": secret_name,
        "output_root": output,
        "priority_class": "c1",
        "gpu_request": 0,
        "root_failure_alert_annotation": "off",
        "runtime_code_sha256": code_sha256,
        "run_script_sha256": "sha256:" + hashlib.sha256(run_script.encode()).hexdigest(),
        "private_cell_task_session_or_trace_identifiers_included": False,
        "score_values_included": False,
        "prompts_responses_flags_rewards_or_trace_content_included": False,
        "created_or_mutated": False,
    }
    if isinstance(intent, reconciliation.ExactStoredSessionSubsetIntent):
        proof_body.update(
            schema=SUBSET_PROOF_SCHEMA,
            source_retry_review_count=terminal["database"]["summary"]["by_state"]["retry_review"],
            subset_reconciliation=True,
            nonselected_cell_count=len(intent.unselected_cell_ids),
            nonselected_cells_preserved_byte_for_byte_and_state_for_state=True,
        )
    proof = {**proof_body, "sha256": _canonical_digest(proof_body)}
    package = Package(
        config_map=config_map,
        secret=secret,
        job=job,
        proof=proof,
        authorization_sha256=packet_intent["sha256"],
        authorized_bundle_sha256=packet_intent["bundle_sha256"],
    )
    validate(package)
    return package


def validate(package: Package) -> None:
    """Validate the rendered object boundary without reading private content."""

    try:
        pod_spec = package.job["spec"]["template"]["spec"]
        containers = pod_spec["containers"]
        container = containers[0]
        resources = container["resources"]
        annotations = package.job["metadata"]["annotations"]
    except (IndexError, KeyError, TypeError) as exc:
        raise ReconciliationPacketError("rendered reconciliation package is malformed") from exc
    bundle_sha256 = _canonical_digest(package.bundle)
    if any(
        (
            annotations.get(heldout_launch.FAILURE_ALERT_ANNOTATION)
            != heldout_launch.FAILURE_ALERT_OFF,
            annotations.get(heldout_launch.CREATE_ONCE_ANNOTATION) != "true",
            package.job["spec"].get("backoffLimit") != 0,
            pod_spec.get("priorityClassName") != "c1",
            pod_spec.get("restartPolicy") != "Never",
            len(containers) != 1,
            pod_spec.get("initContainers") is not None,
            pod_spec.get("serviceAccount") is not None,
            pod_spec.get("serviceAccountName") is not None,
            container.get("command") != ["/bin/bash", "-ceu", "--"],
            container.get("args") != ["exec /bin/bash /bootstrap/run.sh\n"],
            "nvidia.com/gpu" in resources.get("requests", {}),
            "nvidia.com/gpu" in resources.get("limits", {}),
            package.config_map.get("immutable") is not True,
            package.secret.get("immutable") is not True,
            set(package.config_map.get("data", {})) != {*CODE_FILES, "run.sh"},
            set(package.secret.get("data", {})) != {"intent.json", "closure.json"},
            package.proof.get("reviewed_intent_sha256") != package.authorization_sha256,
            not isinstance(package.authorized_bundle_sha256, str)
            or heldout_launch.SHA256.fullmatch(package.authorized_bundle_sha256) is None,
            bundle_sha256 != package.authorized_bundle_sha256,
            package.proof.get("bundle_sha256") != bundle_sha256,
            not isinstance(package.authorization_sha256, str)
            or heldout_launch.SHA256.fullmatch(package.authorization_sha256) is None,
            package.proof.get("model_generation_allowed") is not False,
            package.proof.get("scoring_call_allowed") is not False,
            package.proof.get("created_or_mutated") is not False,
            package.proof.get("double_authoritative_observation_before_mutation") is not True,
            package.proof.get("single_atomic_database_transaction") is not True,
            package.proof.get("priority_class") != "c1",
            type(package.proof.get("gpu_request")) is not int
            or package.proof.get("gpu_request") != 0,
            package.proof.get("root_failure_alert_annotation") != "off",
            package.proof.get("sha256")
            != _canonical_digest(
                {key: value for key, value in package.proof.items() if key != "sha256"}
            ),
        )
    ):
        raise ReconciliationPacketError("rendered reconciliation package is unsafe")


def validate_server_previews(
    package: Package, first: dict[str, Any], second: dict[str, Any]
) -> str:
    """Require two identical server previews; this function never creates."""

    validate(package)

    def strip_metadata_defaults(value: dict[str, Any]) -> None:
        metadata = value.get("metadata")
        if not isinstance(metadata, dict):
            raise ReconciliationPacketError("server preview metadata is invalid")
        for field in (
            "creationTimestamp",
            "generation",
            "managedFields",
            "resourceVersion",
            "uid",
        ):
            metadata.pop(field, None)
        value.pop("status", None)

    def drop_default(
        actual: dict[str, Any], expected: dict[str, Any], key: str, default: Any
    ) -> None:
        if key not in expected and actual.get(key) == default:
            actual.pop(key)

    def exact_job(value: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
        try:
            metadata = value["metadata"]
            uid = metadata["uid"]
            selector = value["spec"]["selector"]
            projected_labels = value["spec"]["template"]["metadata"]["labels"]
            if (
                not isinstance(uid, str)
                or heldout_launch.KUBERNETES_UID.fullmatch(uid) is None
                or selector != {"matchLabels": {"batch.kubernetes.io/controller-uid": uid}}
                or projected_labels.get("batch.kubernetes.io/controller-uid") != uid
                or projected_labels.get("controller-uid") != uid
            ):
                raise ReconciliationPacketError("server-rendered Job UID projections differ")
            actual = stable_job_preview(value)
            wanted = stable_job_preview(expected)
            strip_metadata_defaults(actual["spec"]["template"])
            actual_spec = actual["spec"]
            wanted_spec = wanted["spec"]
            actual_pod = actual_spec["template"]["spec"]
            wanted_pod = wanted_spec["template"]["spec"]
            actual_container = actual_pod["containers"][0]
            wanted_container = wanted_pod["containers"][0]
        except ReconciliationPacketError:
            raise
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise ReconciliationPacketError("server-rendered Job has no exact stable form") from exc

        actual_labels = actual_spec["template"].get("metadata", {}).get("labels", {})
        wanted_labels = wanted_spec["template"].get("metadata", {}).get("labels", {})
        job_name = wanted["metadata"]["name"]
        for field in ("batch.kubernetes.io/job-name", "job-name"):
            if field not in wanted_labels and actual_labels.get(field) == job_name:
                actual_labels.pop(field)
        for key, default in (
            ("parallelism", 1),
            ("completions", 1),
            ("completionMode", "NonIndexed"),
            ("manualSelector", False),
            ("suspend", False),
            ("podReplacementPolicy", "TerminatingOrFailed"),
        ):
            drop_default(actual_spec, wanted_spec, key, default)
        for key, default in (
            ("dnsPolicy", "ClusterFirst"),
            ("enableServiceLinks", True),
            ("preemptionPolicy", "PreemptLowerPriority"),
            ("priority", 10_000),
            ("schedulerName", "default-scheduler"),
            ("securityContext", {}),
            ("serviceAccount", "default"),
            ("serviceAccountName", "default"),
            ("terminationGracePeriodSeconds", 30),
        ):
            drop_default(actual_pod, wanted_pod, key, default)
        for key, default in (
            ("imagePullPolicy", "IfNotPresent"),
            ("terminationMessagePath", "/dev/termination-log"),
            ("terminationMessagePolicy", "File"),
        ):
            drop_default(actual_container, wanted_container, key, default)
        if actual != wanted:
            raise ReconciliationPacketError("server-rendered package differs")
        return actual

    def exact_non_job(value: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
        actual = json.loads(json.dumps(value))
        wanted = json.loads(json.dumps(expected))
        strip_metadata_defaults(actual)
        strip_metadata_defaults(wanted)
        if actual != wanted:
            raise ReconciliationPacketError("server-rendered package differs")
        return actual

    def stable(response: dict[str, Any]) -> dict[str, Any]:
        items = response.get("items") if response.get("kind") == "List" else None
        if not isinstance(items, list):
            raise ReconciliationPacketError("server preview did not return a List")
        expected = {
            (item["apiVersion"], item["kind"], item["metadata"]["name"]): item
            for item in package.bundle["items"]
        }
        if len(items) != len(expected):
            raise ReconciliationPacketError("server preview object identities differ")
        observed: dict[tuple[str, str, str], dict[str, Any]] = {}
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("metadata"), dict):
                raise ReconciliationPacketError("server preview object is invalid")
            key = (item.get("apiVersion"), item.get("kind"), item["metadata"].get("name"))
            if key not in expected or key in observed:
                raise ReconciliationPacketError("server preview object identities differ")
            observed[key] = item
        if set(observed) != set(expected):
            raise ReconciliationPacketError("server preview object identities differ")
        result: dict[str, Any] = {}
        for key, expected_item in expected.items():
            actual = observed[key]
            if key[1] == "Job":
                if (
                    actual["metadata"]
                    .get("annotations", {})
                    .get(heldout_launch.FAILURE_ALERT_ANNOTATION)
                    != heldout_launch.FAILURE_ALERT_OFF
                ):
                    raise ReconciliationPacketError(
                        "server-rendered root Job is missing failure-alerts off"
                    )
                actual = exact_job(actual, expected_item)
            else:
                actual = exact_non_job(actual, expected_item)
            result["/".join(key)] = actual
        return result

    first_stable = stable(first)
    second_stable = stable(second)
    if first_stable != second_stable:
        raise ReconciliationPacketError("server previews changed across identical requests")
    return _canonical_digest(first_stable)
