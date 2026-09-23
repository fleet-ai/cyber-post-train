"""Private final gate for the matched Fleet dev17 pass@8 comparison.

The gate opens scores only after all sixteen pass@1 replica ledgers are held in
repeatable-read snapshots and every score-blind readiness check has passed.
Its public artifact contains only anonymized task-level counts and evidence
digests; task, cell, session, prompt, trace, and verifier identities stay in
mode-0600 private receipts.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
import re
import secrets
import stat
import tempfile
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Protocol
from urllib.parse import parse_qsl, quote, urlsplit, urlunsplit

PLAN_SCHEMA = "cyber_fleet_matched_pass8_protocol_v2_final_plan_v1"
PUBLIC_SCHEMA = "cyber_sanitized_matched_pass8_aggregate_v1"
BENCHMARK = "fleet_development_dev17"
PREDECESSOR_COMPARISON_DEFINITION_SHA256 = (
    "sha256:1154b450624a8b9a567464916da95874c44933175c408b86a80ff5bca0eada70"
)
FROZEN_COMPARISON_DEFINITION_SHA256 = (
    "sha256:9813713ef2ab023ac6c64df494ca7cbf39b920e8fba2f05f5949daaa006479f1"
)
FROZEN_COMPARISON_DEFINITION_FILE_SHA256 = (
    "sha256:1356a652628619020ab7cbdd6b599a5b162dc99b47ee71b957f8428bb49227c6"
)
FROZEN_MIGRATION_RECEIPT_SHA256 = (
    "sha256:9fa3f39f9030dede9efaf38d8e0dfa218a36db885f8f6c72695ba5aa5ec13643"
)
FROZEN_MIGRATION_RECEIPT_FILE_SHA256 = (
    "sha256:fcee992ba4bb81527c5a63774fa76e9334163f7d583824ddb4b9f3b46c1c84d4"
)
MIGRATION_SCHEMA = "cyber_qwen38_fleet_protocol_v2_replica_migration_receipt_v1"
COMPARISON_DEFINITION_SCHEMA = "cyber_qwen38_fleet_dev17_pass8_comparison_definition_v2"
TERMINAL_SCHEMA = "cyber_fleet_heldout_terminal_observation_v1"
ACCEPTED_SCHEMA = "fleet-rollout-ledger-cell-accepted-v1"
CONTROLLER_TERMINAL_SCHEMA = "fleet-rollout-ledger-controller-terminal-v1"
SOURCE_SEEDS = tuple(range(46, 54))
FROZEN_INVALID_SEEDS = (46, 47, 49, 50, 51, 52, 53)
RETAINED_EVALUATION_IDENTITIES = {
    (48, "base"): "sha256:df6bdb006f7346331ba008a8bfd64e9c14ffa6367943b49832b0d5281b4c3ed4",
    (48, "candidate"): "sha256:a1a7a9f820f1c8efb47d53ca9395627bb77b0fd3c30a387620e2e08b65afa9d5",
}
RETAINED_PACKET_FILE_SHA256 = {
    (48, "base"): "sha256:ef004b3b69017b4fd15f938a35761fd25a6a3ffebbbe767cf416c6125a7c5431",
    (48, "candidate"): "sha256:b4f0163672d791b5d85a09bef1c0d8d435ef69e1a7c499a1c4edf4a30f7cb449",
}
RETAINED_EVALUATION_PLAN_SHA256 = {
    (48, "base"): "sha256:cbd0cd28b486fd75aea20f425e68a4990d7e0edb3e232903548790527592dc4f",
    (48, "candidate"): "sha256:a9d98ee9fc87c3c7fb5f58fe19ad3b7ef01482504f5ce34d39da00f5c4e0f8e7",
}
ARMS = ("base", "candidate")
TASK_COUNT = 17
PASS_K = 8
PRIVATE_OUTPUT_ROOT = (
    "/mnt/sfs/jobs/chris-q38-study-corpora-v1/launch-controls/chris-q38-dev17-pass8-final-v2"
)
PRIVATE_STAGING_ROOT = "/result-staging/chris-q38-dev17-pass8-final-v2"
EXACT_PLAN_FIELD_DIGESTS = {
    "tasks": "sha256:3398757c75f22de021ca325262914d7af52b8e8d32617cb0ba5c25ec1ee463f4",
    "harness": "sha256:cc289a5221a6bfdaf57bf24652c3a8e67d1469d21149e6af889628207a5172bb",
    "images": "sha256:4d9238bf16805846d9ef355364546c762903e29d85080be7b49ba90f2802257f",
    "sampling": "sha256:67a5e03019869fbcc5c68c694e9b0daf72ab20a79c9e8122577889913b42f3c5",
    "arms": "sha256:ba923970850f4eb59456f6788d43b04c712a3787f1a46224affbed8bde4a2c30",
}
SHA256 = re.compile(r"(?:sha256:)?[0-9a-f]{64}")
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")

CELL_FIELDS = (
    "cell_id",
    "experiment_id",
    "task_key",
    "task_version_id",
    "model_id",
    "model_revision",
    "serving_block",
    "endpoint_model_id",
    "harness_id",
    "attempt",
    "state",
    "session_id",
    "lease_expires_at",
    "retry_count",
    "max_retries",
    "result_class",
    "receipt_digest",
    "failure_code",
    "reconciliation_digest",
)
LOCAL_RECORD_FIELDS = (
    "execution_id",
    "cell_id",
    "execution_generation",
    "run_id",
    "session_id",
    "verifier_execution_id",
    "score",
    "config_sha256",
    "artifact_directory",
    "trace_path",
    "trace_sha256",
    "result_path",
    "result_sha256",
    "reward_path",
    "reward_sha256",
    "session_ingest_path",
    "session_ingest_sha256",
    "cleanup_path",
    "cleanup_sha256",
    "session_ingest_status",
    "agent_exit_code",
    "agent_termination",
    "elapsed_seconds",
)
SCORE_BLIND_LOCAL_FIELDS = (
    "execution_id",
    "cell_id",
    "run_id",
    "session_id",
    "verifier_execution_id",
    "config_sha256",
    "artifact_directory",
    "session_ingest_status",
    "agent_exit_code",
    "agent_termination",
)
ACCEPTED_RECEIPT_FIELDS = {
    "schema_version",
    "accepted",
    "campaign_id",
    "cell_id",
    "execution_id",
    "ledger_cell_id",
    "run_id",
    "serving_block",
    "session_id",
    "verifier_execution_id",
    "config_sha256",
    "session_ingest_completed",
    "cleanup_completed",
    "score_persisted_privately",
    "scores_included",
    "prompts_or_traces_included",
    "receipt_sha256",
}
ACCEPTED_EVENT_FIELDS = {"cell_id", "event", "to_state", "detail_json"}
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
PLAN_STORED_FIELDS = (
    "cell_id",
    "experiment_id",
    "task_key",
    "task_version_id",
    "model_id",
    "model_revision",
    "serving_block",
    "endpoint_model_id",
    "harness_id",
    "attempt",
    "max_retries",
)
RUNTIME_FILES = {
    "evaluate.py",
    "rollout_worker.py",
    "rollout_postgres.py",
    "rollout_ledger.py",
    "opencode_self_hosted.py",
    "fixed_proxy.py",
    "exact_pass4_crypto.py",
}
EVALUATION_FIELDS = {
    "schema",
    "campaign_id",
    "run_prefix",
    "selection",
    "tasks",
    "models",
    "routes",
    "treatment",
    "images",
    "pass_k",
    "concurrency",
    "training_data_eligible",
    "automatic_retry",
    "max_reviewed_infrastructure_retries",
    "runtime_files",
    "sampling",
    "interpretation",
    "sha256",
}
EVALUATION_IDENTITY_FIELDS = {
    "protocol_id",
    "comparison_arms",
    "arm_id",
    "evaluation_config_name",
    "evaluation_config_sha256",
    "task_selection_sha256",
    "split_manifest_file_sha256",
    "split_manifest_sha256",
    "comparison_protocol_file_sha256",
    "comparison_protocol_sha256",
    "checkpoint_provenance_sha256",
    "serving_route_proof_sha256",
    "model_revision",
    "harness",
    "harness_version",
    "context_management",
    "sampling_seed",
    "pass_k",
    "retry_limit",
    "output_root",
    "database",
}
RETAINED_ARM_FIELDS = {
    "seed",
    "arm_id",
    "evaluation_identity",
    "evaluation_identity_sha256",
    "evaluation_plan_sha256",
    "runtime_files_sha256",
    "packet_file_sha256",
}
MIGRATION_FIELDS = {
    "invalid_original_seed",
    "replacement_seed",
    "reason_class",
    "evidence_receipt_sha256s",
    "excluded_source_arms",
    "whole_pair_excluded",
}
EXCLUDED_SOURCE_ARM_FIELDS = {
    "packet_file_sha256",
    "evaluation_identity_sha256",
    "evaluation_config_sha256",
    "comparison_protocol_file_sha256",
    "comparison_protocol_sha256",
    "protocol_id",
    "job_name",
    "config_map_name",
    "output_root",
    "database",
}
REPLACEMENT_PROTOCOL_FIELDS = {
    "invalid_original_seed",
    "replacement_seed",
    "protocol_id",
    "sha256",
    "file_sha256",
}
REPLACEMENT_ARM_FIELDS = {
    "arm_id",
    "invalid_original_seed",
    "replacement_seed",
    "evaluation_identity",
    "evaluation_identity_sha256",
    "evaluation_plan_sha256",
    "runtime_files_sha256",
    "packet_file_sha256",
    "serving_proof_file_sha256",
    "comparison_protocol_file_sha256",
    "comparison_protocol_sha256",
    "packet_path",
}
RETIREMENT_EVIDENCE_FIELDS = {
    "sha256",
    "file_sha256",
    "targets",
    "model_rollouts",
    "outputs_or_databases_deleted",
}
CAPACITY_FIELDS = {
    "actual_started_rollouts_today",
    "daily_rollout_cap",
    "final_comparison_rollouts",
    "new_replacement_rollouts",
    "projected_rollouts_after_reservation",
    "remaining_after_reservation",
    "within_daily_cap",
}
GLOBAL_DAILY_BUDGET_FIELDS = {"path", "sha256", "file_sha256", "window_utc"}
SELECTION_FIELDS = {
    "selection_sha256",
    "split_sha256",
    "corpus_dev_windows",
    "binding_roster_sha256",
}
PRIVACY_FIELDS = {
    "score_values_read",
    "prompts_responses_flags_rewards_or_trace_content_read",
    "infrastructure_reason_classes_only",
}
SCIENTIFIC_IDENTITY_FIELDS = {
    "task_count_per_arm",
    "comparison_arms",
    "pass_k",
    "retry_limit",
    "same_task_versions_models_harness_budgets_and_sampling_recipe",
    "whole_replica_pairs_only",
    "cell_level_replacement_forbidden",
    "seed_reuse_forbidden",
    "later_invalid_seed_requires_versioned_successor_before_score_unseal",
}
SOURCE_TASK_JOB_ID = "a62dd51f-a52b-4941-8207-4679e4b25b51"
ALLOWED_EXCLUSION_REASONS = {
    "terminal_replica_incomplete",
    "replica_not_started",
    "mixed_infrastructure_invalid_replica",
    "persisted_transcript_prefix_mismatch",
}
FROZEN_INVALID_REPLICA_EVIDENCE: dict[int, dict[str, Any]] = {
    46: {
        "reason_class": "persisted_transcript_prefix_mismatch",
        "evidence_receipt_sha256s": [
            "sha256:e7b0031010f097f66dd3c87de94ecbc0d1505de97167c47d0a2d7615a0164e52"
        ],
    },
    47: {
        "reason_class": "terminal_replica_incomplete",
        "evidence_receipt_sha256s": [
            "sha256:9bd0f1c415d32cdeae47744cbaa140c5e39781b7790ea318573d38772d8dd5bf",
            "sha256:a8d3ca88eaab62d4642ca3fab0fdf17b21a9eb0c92bb5de8971d8d0273928b0e",
            "sha256:7bf0cbc8715c51d6945bb133357f392646cf07ccc7e71a84dc887a30c902e083",
        ],
    },
    49: {
        "reason_class": "persisted_transcript_prefix_mismatch",
        "evidence_receipt_sha256s": [
            "sha256:e7b0031010f097f66dd3c87de94ecbc0d1505de97167c47d0a2d7615a0164e52"
        ],
    },
    50: {
        "reason_class": "persisted_transcript_prefix_mismatch",
        "evidence_receipt_sha256s": [
            "sha256:e7b0031010f097f66dd3c87de94ecbc0d1505de97167c47d0a2d7615a0164e52"
        ],
    },
    51: {
        "reason_class": "mixed_infrastructure_invalid_replica",
        "evidence_receipt_sha256s": [
            "sha256:572e330d1340e83d2aaf188665c0fd99b401e33eeec5f858717e2db60952bfa3"
        ],
    },
    52: {
        "reason_class": "mixed_infrastructure_invalid_replica",
        "evidence_receipt_sha256s": [
            "sha256:3a247e72e61da31593223d5fa46f902e4ff546e76277be3e27ef7407d91348c8",
            "sha256:e1718d127e4a2da1667eaa7d2c0a7d1d219e2e0bef3dcaabd4d46aa5f5b8f97b",
            "sha256:9a1296727060d303a3fe6d81fe51d2021482113c0dba8f4a9c4b4ba5efb1ccf3",
            "sha256:952bbaaabbb6c3a2fa105bb3780090d2ff8cc749345ba65904fc401ddb152aad",
        ],
    },
    53: {
        "reason_class": "replica_not_started",
        "evidence_receipt_sha256s": [
            "sha256:fa656de6282a504ad2570a65078625a5c35e79afcf5d4d120c0803a75912ff70",
            "sha256:ef6e969b305cecf8fa5bff882713d87bf3f8c21a44ced4f0ce54119c46c049e1",
            "sha256:9a1296727060d303a3fe6d81fe51d2021482113c0dba8f4a9c4b4ba5efb1ccf3",
            "sha256:952bbaaabbb6c3a2fa105bb3780090d2ff8cc749345ba65904fc401ddb152aad",
        ],
    },
}
MIGRATION_RECEIPT_FIELDS = {
    "schema",
    "source",
    "migration_intent_sha256",
    "sanitized_invalid_replica_evidence",
    "checked_in_seed51_invalid_evidence",
    "checked_in_partial_recovery_hold_evidence",
    "comparison_definition",
    "comparison_definition_file_sha256",
    "mapping_rule",
    "migrations",
    "excluded_original_seeds",
    "included_seeds",
    "replacement_protocols",
    "replacement_arms",
    "retained_arms",
    "retirement_evidence",
    "global_daily_budget_evidence",
    "scientific_identity",
    "capacity",
    "selection",
    "live_parity_file_sha256",
    "live_parity_receipt_sha256",
    "privacy",
    "external_mutations",
    "launch_performed",
    "sha256",
}


class FinalAggregateError(ValueError):
    """The final comparison is incomplete, mismatched, or unsafe to open."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise FinalAggregateError("value cannot be canonically encoded") from exc


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(_read_regular_once(path, "digest input")).hexdigest()


def _plain_digest(value: object) -> str:
    return _digest(value).removeprefix("sha256:")


def _cell_id(row: Mapping[str, Any]) -> str:
    identity = "\0".join(
        str(row[field]) for field in ("experiment_id", "task_version_id", "model_id", "attempt")
    )
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"fleet-cyber-rollout-cell-v1:{identity}"))


def _normalized_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise FinalAggregateError(f"{label} is not a SHA-256 digest")
    return value.removeprefix("sha256:")


def _read_regular_once(path: Path, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise FinalAggregateError(f"{label} is not an exact regular file") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise FinalAggregateError(f"{label} is not an exact regular file")
        chunks = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
        ):
            raise FinalAggregateError(f"{label} changed while it was read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _read_json_and_digest(path: Path, label: str) -> tuple[dict[str, Any], str]:
    raw = _read_regular_once(path, label)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FinalAggregateError(f"{label} is not readable JSON") from exc
    if not isinstance(value, dict):
        raise FinalAggregateError(f"{label} must be a JSON object")
    return value, "sha256:" + hashlib.sha256(raw).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    return _read_json_and_digest(path, label)[0]


def _read_json_beneath(
    root: Path, relative: PurePosixPath, label: str
) -> tuple[dict[str, Any], str]:
    """Read one regular JSON file without following a relative-path symlink."""

    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise FinalAggregateError(f"{label} path is not an exact relative path")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(root, directory_flags)
    except OSError as exc:
        raise FinalAggregateError(f"{label} root is not an exact directory") from exc
    try:
        for part in relative.parts[:-1]:
            try:
                child = os.open(part, directory_flags, dir_fd=descriptor)
            except OSError as exc:
                raise FinalAggregateError(f"{label} parent is not an exact directory") from exc
            os.close(descriptor)
            descriptor = child
        try:
            file_descriptor = os.open(
                relative.parts[-1],
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
        except OSError as exc:
            raise FinalAggregateError(f"{label} is not an exact regular file") from exc
        try:
            before = os.fstat(file_descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise FinalAggregateError(f"{label} is not an exact regular file")
            chunks = []
            while chunk := os.read(file_descriptor, 1024 * 1024):
                chunks.append(chunk)
            after = os.fstat(file_descriptor)
            if (
                (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
                or before.st_size != after.st_size
                or before.st_mtime_ns != after.st_mtime_ns
            ):
                raise FinalAggregateError(f"{label} changed while it was read")
            raw = b"".join(chunks)
        finally:
            os.close(file_descriptor)
    finally:
        os.close(descriptor)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FinalAggregateError(f"{label} is not readable JSON") from exc
    if not isinstance(value, dict):
        raise FinalAggregateError(f"{label} must be a JSON object")
    return value, "sha256:" + hashlib.sha256(raw).hexdigest()


def _bound_path(value: str) -> Path:
    """Resolve one plan-bound path; tests replace this without editing the plan."""

    return Path(value)


def _require_self_digest(value: dict[str, Any], field: str, label: str) -> str:
    claimed = value.get(field)
    if not isinstance(claimed, str) or SHA256.fullmatch(claimed) is None:
        raise FinalAggregateError(f"{label} self digest is missing")
    unsigned = {key: item for key, item in value.items() if key != field}
    expected = _digest(unsigned)
    if claimed.removeprefix("sha256:") != expected.removeprefix("sha256:"):
        raise FinalAggregateError(f"{label} self digest differs")
    return expected


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    raw = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a directory without replacing a concurrent claimant."""

    libc = ctypes.CDLL(None, use_errno=True)
    source_raw = os.fsencode(source)
    destination_raw = os.fsencode(destination)
    if hasattr(libc, "renameat2"):
        rename = libc.renameat2
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(-100, source_raw, -100, destination_raw, 1)  # RENAME_NOREPLACE
    elif hasattr(libc, "renamex_np"):
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(source_raw, destination_raw, 0x00000004)  # RENAME_EXCL
    else:
        raise FinalAggregateError("atomic create-once publication is unavailable")
    if result == 0:
        return
    code = ctypes.get_errno()
    if code in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FinalAggregateError("final aggregate output was claimed concurrently")
    raise FinalAggregateError(f"atomic create-once publication failed with errno {code}")


def _task_inputs(
    task_set_path: Path, roster_path: Path
) -> tuple[list[dict[str, str]], dict, str, str]:
    task_set, task_set_file_sha256 = _read_json_and_digest(task_set_path, "dev17 task selection")
    roster, roster_file_sha256 = _read_json_and_digest(roster_path, "dev17 exact binding roster")
    _require_self_digest(roster, "sha256", "dev17 exact binding roster")
    if any(
        (
            task_set.get("schema") != "cyber_eval_task_selection_v2",
            task_set.get("selection_role") != "dev",
            task_set.get("selection_sha256")
            != "sha256:38ea6686afa068c19e69fea2493e01027fdf99f7e01b20376e107b1a0cfa0b68",
            roster.get("binding_count") != TASK_COUNT,
            roster.get("bindings_sha256")
            != "sha256:32230df8759bb74eb306d16cd9e32686618b529aa7c0b0ac1fa1d751ecd38728",
        )
    ):
        raise FinalAggregateError("immutable dev17 selection identity differs")
    tasks = task_set.get("tasks")
    bindings = roster.get("bindings")
    if not isinstance(tasks, list) or not isinstance(bindings, list) or len(tasks) != TASK_COUNT:
        raise FinalAggregateError("immutable dev17 selection is incomplete")
    selected: list[dict[str, str]] = []
    for task, binding in zip(tasks, bindings, strict=True):
        if not isinstance(task, dict) or not isinstance(binding, dict):
            raise FinalAggregateError("immutable dev17 row is invalid")
        version = task.get("task_version_id")
        key = task.get("task_key")
        bound = binding.get("task", {})
        if (
            not isinstance(version, str)
            or UUID.fullmatch(version) is None
            or not isinstance(key, str)
            or not key
            or not isinstance(bound, dict)
            or bound.get("version_id") != version
            or bound.get("key") != key
        ):
            raise FinalAggregateError("dev17 task selection and binding roster differ")
        selected.append(
            {
                field: str(task[field])
                for field in (
                    "task_key",
                    "task_version_id",
                    "env_key",
                    "env_version",
                    "environment_version_id",
                    "data_key",
                    "data_version",
                )
            }
        )
    if len({row["task_version_id"] for row in selected}) != TASK_COUNT:
        raise FinalAggregateError("dev17 task versions are not unique")
    return selected, roster, task_set_file_sha256, roster_file_sha256


def _exact_digest_fields(value: Mapping[str, Any], fields: Sequence[str], label: str) -> None:
    for field in fields:
        _normalized_digest(value.get(field), f"{label} {field}")


def _arm_evidence_binding(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FinalAggregateError(f"{label} is invalid")
    identity = value.get("evaluation_identity")
    runtime = value.get("runtime_files_sha256")
    if (
        not isinstance(identity, dict)
        or set(identity) != EVALUATION_IDENTITY_FIELDS
        or not isinstance(runtime, dict)
        or set(runtime) != RUNTIME_FILES
        or any(
            not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            for digest in runtime.values()
        )
    ):
        raise FinalAggregateError(f"{label} evaluation evidence is incomplete")
    _exact_digest_fields(
        value,
        ("packet_file_sha256", "evaluation_identity_sha256", "evaluation_plan_sha256"),
        label,
    )
    if (
        not str(value["evaluation_plan_sha256"]).startswith("sha256:")
        or _digest(identity) != value["evaluation_identity_sha256"]
    ):
        raise FinalAggregateError(f"{label} evaluation evidence digest differs")
    return {
        "packet_file_sha256": value["packet_file_sha256"],
        "evaluation_identity": identity,
        "evaluation_identity_sha256": value["evaluation_identity_sha256"],
        "evaluation_plan_sha256": value["evaluation_plan_sha256"],
        "runtime_files_sha256": runtime,
    }


def _deterministic_mapping(originals: Sequence[int]) -> list[dict[str, int]]:
    if (
        not originals
        or list(originals) != sorted(originals)
        or tuple(originals) != FROZEN_INVALID_SEEDS
        or len(set(originals)) != len(originals)
        or any(type(seed) is not int or seed not in SOURCE_SEEDS for seed in originals)
    ):
        raise FinalAggregateError("protocol-v2 excluded original seeds differ from the freeze")
    return [
        {"invalid_original_seed": original, "replacement_seed": 54 + index}
        for index, original in enumerate(originals)
    ]


def _comparison_contract(
    comparison: dict[str, Any],
) -> tuple[list[dict[str, int]], list[int], dict[int, dict[str, Any]]]:
    """Validate the stable scientific definition embedded by protocol v2."""

    _require_self_digest(comparison, "sha256", "protocol-v2 comparison definition")
    excluded = comparison.get("excluded_original_seeds")
    if not isinstance(excluded, list):
        raise FinalAggregateError("protocol-v2 excluded original seeds are missing")
    mapping = _deterministic_mapping(excluded)
    included = sorted(
        [
            *(seed for seed in SOURCE_SEEDS if seed not in set(excluded)),
            *(row["replacement_seed"] for row in mapping),
        ]
    )
    if any(
        (
            comparison.get("sha256") != FROZEN_COMPARISON_DEFINITION_SHA256,
            comparison.get("schema") != COMPARISON_DEFINITION_SCHEMA,
            comparison.get("protocol_study_id") != "q38-dev17-base-step1000-p8-v2",
            comparison.get("predecessor_launch_receipt_sha256")
            != PREDECESSOR_COMPARISON_DEFINITION_SHA256,
            comparison.get("aggregation") != "eight_predeclared_pass1_replicas_per_task_and_arm",
            comparison.get("original_seeds") != list(SOURCE_SEEDS),
            comparison.get("replacement_mapping") != mapping,
            comparison.get("included_seeds") != included,
            len(included) != PASS_K,
            len(set(included)) != PASS_K,
            comparison.get("task_count") != TASK_COUNT,
            comparison.get("sessions_per_arm") != TASK_COUNT * PASS_K,
            comparison.get("total_sessions") != TASK_COUNT * PASS_K * len(ARMS),
            comparison.get("comparison_arms") != list(ARMS),
            comparison.get("pass_k_per_replica") != 1,
            comparison.get("retry_limit") != 0,
            comparison.get("training_data_eligible") is not False,
            comparison.get("whole_replica_pairs_only") is not True,
            comparison.get("cell_level_replacement_forbidden") is not True,
            comparison.get("included_seed_status")
            != "provisional_until_all_valid8_terminal_gates_pass",
            comparison.get("later_invalid_seed_policy")
            != (
                "create a versioned successor intent, definition, and receipt before score "
                "unseal; never edit this definition in place"
            ),
            comparison.get("task_selection_sha256")
            != "sha256:38ea6686afa068c19e69fea2493e01027fdf99f7e01b20376e107b1a0cfa0b68",
            comparison.get("split_manifest_sha256")
            != "sha256:05b3a8dc90ca93adc9671942d75ecb54ff8d0951b0dd48a087e29ec1e641840c",
            comparison.get("binding_roster_sha256")
            != "sha256:39ae49c2db322725800b15347d5da5d171e7e03b21dec5b570683414d70e83c5",
        )
    ):
        raise FinalAggregateError("protocol-v2 comparison identity differs")
    expected_models = {
        "base": {
            "model_id": "qwen3.8-27b",
            "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        },
        "candidate": {
            "model_id": "chris-q38-t3k32-s1000-v1",
            "revision": ("sha256:023c5f8b0559ba050f0d672a6bc27aabecec7d5837595f8ea5bc914446d26db5"),
        },
    }
    if (
        comparison.get("models") != expected_models
        or _digest(comparison.get("harness")) != EXACT_PLAN_FIELD_DIGESTS["harness"]
        or _digest(comparison.get("images")) != EXACT_PLAN_FIELD_DIGESTS["images"]
        or comparison.get("sampling_without_seed") != {"temperature": 0.6, "top_p": 0.95}
    ):
        raise FinalAggregateError("protocol-v2 model or treatment identity differs")
    protocols = comparison.get("replica_protocols")
    if (
        not isinstance(protocols, list)
        or len(protocols) != PASS_K
        or [row.get("seed") for row in protocols if isinstance(row, dict)] != included
    ):
        raise FinalAggregateError("protocol-v2 included replica protocol roster is incomplete")
    protocol_by_seed: dict[int, dict[str, Any]] = {}
    replacement_seeds = {row["replacement_seed"] for row in mapping}
    for row in protocols:
        if not isinstance(row, dict) or set(row) != {
            "seed",
            "origin",
            "protocol_id",
            "comparison_protocol_sha256",
        }:
            raise FinalAggregateError("protocol-v2 replica protocol row is malformed")
        seed = row.get("seed")
        expected_origin = (
            "whole_pair_replacement" if seed in replacement_seeds else "retained_original"
        )
        if (
            type(seed) is not int
            or seed not in included
            or seed in protocol_by_seed
            or row.get("origin") != expected_origin
            or not isinstance(row.get("protocol_id"), str)
            or not row["protocol_id"]
        ):
            raise FinalAggregateError("protocol-v2 replica protocol identity differs")
        _normalized_digest(row.get("comparison_protocol_sha256"), "replica protocol")
        protocol_by_seed[seed] = row
    if set(protocol_by_seed) != set(included):
        raise FinalAggregateError("protocol-v2 included replica protocol seeds differ")
    return mapping, included, protocol_by_seed


def _migration_input(path: Path) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    receipt, receipt_file_sha256 = _read_json_and_digest(path, "protocol-v2 migration receipt")
    receipt_sha256 = _require_self_digest(receipt, "sha256", "protocol-v2 migration receipt")
    if (
        receipt_sha256 != FROZEN_MIGRATION_RECEIPT_SHA256
        or receipt_file_sha256 != FROZEN_MIGRATION_RECEIPT_FILE_SHA256
    ):
        raise FinalAggregateError("protocol-v2 migration receipt differs from the freeze")
    if set(receipt) != MIGRATION_RECEIPT_FIELDS:
        raise FinalAggregateError("protocol-v2 migration receipt field roster differs")
    comparison = receipt.get("comparison_definition")
    if not isinstance(comparison, dict):
        raise FinalAggregateError("protocol-v2 comparison definition is missing")
    mapping, included, protocol_by_seed = _comparison_contract(comparison)
    comparison_path = path.parent / "COMPARISON_DEFINITION.json"
    comparison_file, comparison_file_sha256 = _read_json_and_digest(
        comparison_path, "protocol-v2 comparison definition file"
    )
    if (
        comparison_file != comparison
        or comparison_file_sha256 != FROZEN_COMPARISON_DEFINITION_FILE_SHA256
        or receipt.get("comparison_definition_file_sha256") != comparison_file_sha256
    ):
        raise FinalAggregateError("protocol-v2 comparison definition file differs")
    excluded = comparison["excluded_original_seeds"]
    source = receipt.get("source")
    if not isinstance(source, dict) or set(source) != {
        "protocol_study_id",
        "preparation_receipt_sha256",
        "preparation_receipt_file_sha256",
        "seeds",
        "arm_count",
        "retained_candidate_successor_preparation",
    }:
        raise FinalAggregateError("protocol-v2 source study evidence is missing")
    _exact_digest_fields(
        source,
        ("preparation_receipt_sha256", "preparation_receipt_file_sha256"),
        "protocol-v2 source study",
    )
    successor = source.get("retained_candidate_successor_preparation")
    if (
        not isinstance(successor, dict)
        or set(successor) != {"path", "file_sha256", "receipt_sha256", "successor_generation"}
        or not isinstance(successor.get("path"), str)
        or not successor["path"]
        or successor.get("successor_generation") != 2
    ):
        raise FinalAggregateError("retained candidate successor evidence is incomplete")
    _exact_digest_fields(
        successor,
        ("file_sha256", "receipt_sha256"),
        "retained candidate successor evidence",
    )
    if successor != {
        "path": successor["path"],
        "file_sha256": ("sha256:5605623bf5f6205a70a6bad0f02985f985858904badfa7be413a1cead1cbe67a"),
        "receipt_sha256": (
            "sha256:d3624db05e0665cf0483d802a49c54be601293e5267b5475143a282791269189"
        ),
        "successor_generation": 2,
    }:
        raise FinalAggregateError("retained candidate successor evidence differs")
    hold = receipt.get("checked_in_partial_recovery_hold_evidence")
    if (
        not isinstance(hold, dict)
        or set(hold) != {"path", "file_sha256", "receipt_sha256", "source_private_receipt_sha256"}
        or not isinstance(hold.get("path"), str)
        or not hold["path"]
    ):
        raise FinalAggregateError("partial-recovery HOLD evidence is incomplete")
    _exact_digest_fields(
        hold,
        ("file_sha256", "receipt_sha256", "source_private_receipt_sha256"),
        "partial-recovery HOLD evidence",
    )
    if hold != {
        "path": "docs/evidence/qwen38-fleet-dev17-base-prefix-recovery-hold-20260923.json",
        "file_sha256": ("sha256:d20b4fe2310de95cfd5561877f6d93133dcf7e5ed903f425addfb28bda2583ae"),
        "receipt_sha256": (
            "sha256:e7b0031010f097f66dd3c87de94ecbc0d1505de97167c47d0a2d7615a0164e52"
        ),
        "source_private_receipt_sha256": (
            "sha256:45893e380d1d9755cbe514ec85628b7fec930d9e2bf5e61920967155112bda09"
        ),
    }:
        raise FinalAggregateError("partial-recovery HOLD evidence differs")
    checked_seed51 = receipt.get("checked_in_seed51_invalid_evidence")
    if checked_seed51 != {
        "path": "docs/evidence/qwen38-fleet-dev17-seed51-base-invalid-replica-20260923.json",
        "file_sha256": ("sha256:525885e6650d6d742144b12de2ac1807a77cd25f23e07cec58d2832551150497"),
        "receipt_sha256": (
            "sha256:572e330d1340e83d2aaf188665c0fd99b401e33eeec5f858717e2db60952bfa3"
        ),
    }:
        raise FinalAggregateError("seed-51 invalid-replica evidence differs")
    expected_sanitized = [
        {"seed": seed, **FROZEN_INVALID_REPLICA_EVIDENCE[seed]} for seed in FROZEN_INVALID_SEEDS
    ]
    if receipt.get("sanitized_invalid_replica_evidence") != expected_sanitized:
        raise FinalAggregateError("sanitized invalid-replica evidence differs")
    if receipt.get("migration_intent_sha256") != (
        "sha256:0e32d785e8b3570b2e972389b93d70334e9889b8a3d2b9230f0d13ce6137c79c"
    ):
        raise FinalAggregateError("migration intent differs")
    if any(
        (
            receipt.get("schema") != MIGRATION_SCHEMA,
            source.get("protocol_study_id") != "q38-dev17-seeds46to53-base-step1000-p8-v1",
            source.get("preparation_receipt_sha256")
            != "sha256:22b59a3e9e1ea2ed8e8b92920304f27bbd953925fe0ecf709c7308433db8f0d3",
            source.get("preparation_receipt_file_sha256")
            != "sha256:2827250c73d024cab06906d393eba9270784f99b70bb3812d9893c53b1f84859",
            source.get("seeds") != list(SOURCE_SEEDS),
            source.get("arm_count") != len(SOURCE_SEEDS) * len(ARMS),
            receipt.get("mapping_rule")
            != (
                "invalid original seeds sorted ascending map to the smallest unused seeds "
                "greater than 53, ascending"
            ),
            receipt.get("excluded_original_seeds") != excluded,
            receipt.get("included_seeds") != included,
            receipt.get("external_mutations") != 0,
            receipt.get("launch_performed") is not False,
        )
    ):
        raise FinalAggregateError("protocol-v2 migration or comparison identity differs")
    replacement_seeds = {row["replacement_seed"] for row in mapping}
    migrations = receipt.get("migrations")
    if not isinstance(migrations, list) or len(migrations) != len(mapping):
        raise FinalAggregateError("protocol-v2 excluded source evidence is incomplete")
    for expected, row in zip(mapping, migrations, strict=True):
        if (
            not isinstance(row, dict)
            or set(row) != MIGRATION_FIELDS
            or row.get("invalid_original_seed") != expected["invalid_original_seed"]
            or row.get("replacement_seed") != expected["replacement_seed"]
            or row.get("reason_class") not in ALLOWED_EXCLUSION_REASONS
            or row.get("reason_class")
            != FROZEN_INVALID_REPLICA_EVIDENCE[expected["invalid_original_seed"]]["reason_class"]
            or row.get("evidence_receipt_sha256s")
            != FROZEN_INVALID_REPLICA_EVIDENCE[expected["invalid_original_seed"]][
                "evidence_receipt_sha256s"
            ]
            or row.get("whole_pair_excluded") is not True
            or not isinstance(row.get("evidence_receipt_sha256s"), list)
            or not row["evidence_receipt_sha256s"]
            or set(row.get("excluded_source_arms", {})) != set(ARMS)
        ):
            raise FinalAggregateError("protocol-v2 excluded source evidence differs")
        for value in row["evidence_receipt_sha256s"]:
            _normalized_digest(value, "excluded source evidence")
        source_arms = row["excluded_source_arms"]
        for arm in ARMS:
            evidence = source_arms[arm]
            if not isinstance(evidence, dict) or set(evidence) != EXCLUDED_SOURCE_ARM_FIELDS:
                raise FinalAggregateError("protocol-v2 excluded source arm evidence is invalid")
            _exact_digest_fields(
                evidence,
                (
                    "packet_file_sha256",
                    "evaluation_identity_sha256",
                    "evaluation_config_sha256",
                    "comparison_protocol_file_sha256",
                    "comparison_protocol_sha256",
                ),
                "excluded source arm",
            )
    replacement_protocols = receipt.get("replacement_protocols")
    replacement_arms = receipt.get("replacement_arms")
    retained_arms = receipt.get("retained_arms")
    retained_seeds = set(included) - replacement_seeds
    if (
        not isinstance(replacement_protocols, list)
        or len(replacement_protocols) != len(mapping)
        or not isinstance(replacement_arms, list)
        or len(replacement_arms) != len(mapping) * len(ARMS)
        or not isinstance(retained_arms, list)
        or len(retained_arms) != len(retained_seeds) * len(ARMS)
    ):
        raise FinalAggregateError("protocol-v2 replacement evidence is incomplete")
    replacement_protocol_by_seed: dict[int, dict[str, Any]] = {}
    for expected, row in zip(mapping, replacement_protocols, strict=True):
        seed = expected["replacement_seed"]
        if (
            not isinstance(row, dict)
            or set(row) != REPLACEMENT_PROTOCOL_FIELDS
            or row.get("invalid_original_seed") != expected["invalid_original_seed"]
            or row.get("replacement_seed") != seed
            or row.get("protocol_id") != protocol_by_seed[seed]["protocol_id"]
            or row.get("sha256") != protocol_by_seed[seed]["comparison_protocol_sha256"]
        ):
            raise FinalAggregateError("protocol-v2 replacement protocol differs")
        _exact_digest_fields(row, ("sha256", "file_sha256"), "replacement protocol")
        replacement_protocol_by_seed[seed] = row
    arm_by_identity: dict[tuple[int, str], dict[str, Any]] = {}
    for row in retained_arms:
        if not isinstance(row, dict) or set(row) != RETAINED_ARM_FIELDS:
            raise FinalAggregateError("protocol-v2 retained arm evidence is invalid")
        seed = row.get("seed")
        arm = row.get("arm_id")
        identity = (seed, arm)
        binding = _arm_evidence_binding(row, label="retained arm")
        if (
            type(seed) is not int
            or seed not in retained_seeds
            or arm not in ARMS
            or identity in arm_by_identity
            or binding["evaluation_identity_sha256"] != RETAINED_EVALUATION_IDENTITIES.get(identity)
            or binding["packet_file_sha256"] != RETAINED_PACKET_FILE_SHA256.get(identity)
            or binding["evaluation_plan_sha256"] != RETAINED_EVALUATION_PLAN_SHA256.get(identity)
        ):
            raise FinalAggregateError("protocol-v2 retained arm identity differs")
        arm_by_identity[identity] = row
    original_by_replacement = {
        row["replacement_seed"]: row["invalid_original_seed"] for row in mapping
    }
    for row in replacement_arms:
        if not isinstance(row, dict) or set(row) != REPLACEMENT_ARM_FIELDS:
            raise FinalAggregateError("protocol-v2 replacement arm evidence is invalid")
        seed = row.get("replacement_seed")
        arm = row.get("arm_id")
        identity = (seed, arm)
        if (
            type(seed) is not int
            or seed not in replacement_seeds
            or arm not in ARMS
            or identity in arm_by_identity
            or row.get("invalid_original_seed") != original_by_replacement[seed]
            or row.get("comparison_protocol_sha256")
            != protocol_by_seed[seed]["comparison_protocol_sha256"]
        ):
            raise FinalAggregateError("protocol-v2 replacement arm identity differs")
        _exact_digest_fields(
            row,
            (
                "packet_file_sha256",
                "evaluation_identity_sha256",
                "serving_proof_file_sha256",
                "comparison_protocol_file_sha256",
                "comparison_protocol_sha256",
            ),
            "replacement arm",
        )
        binding = _arm_evidence_binding(row, label="replacement arm")
        evaluation_identity = binding["evaluation_identity"]
        if (
            row.get("serving_proof_file_sha256")
            != evaluation_identity["serving_route_proof_sha256"]
            or row.get("serving_proof_file_sha256") != receipt.get("live_parity_file_sha256")
            or row.get("comparison_protocol_file_sha256")
            != evaluation_identity["comparison_protocol_file_sha256"]
            or evaluation_identity["comparison_protocol_file_sha256"]
            != replacement_protocol_by_seed[seed]["file_sha256"]
            or row.get("comparison_protocol_sha256")
            != evaluation_identity["comparison_protocol_sha256"]
        ):
            raise FinalAggregateError("protocol-v2 replacement arm packet identity differs")
        arm_by_identity[identity] = row
    if set(arm_by_identity) != {(seed, arm) for seed in included for arm in ARMS}:
        raise FinalAggregateError("protocol-v2 final arm roster differs")
    scientific = receipt.get("scientific_identity")
    capacity = receipt.get("capacity")
    daily_budget = receipt.get("global_daily_budget_evidence")
    privacy = receipt.get("privacy")
    selection = receipt.get("selection")
    retirement = receipt.get("retirement_evidence")
    if isinstance(retirement, dict) and set(retirement) == RETIREMENT_EVIDENCE_FIELDS:
        _exact_digest_fields(
            retirement,
            ("sha256", "file_sha256"),
            "protocol-v2 retirement evidence",
        )
    _exact_digest_fields(
        receipt,
        ("live_parity_file_sha256", "live_parity_receipt_sha256"),
        "protocol-v2 migration",
    )
    if not isinstance(daily_budget, dict) or set(daily_budget) != GLOBAL_DAILY_BUDGET_FIELDS:
        raise FinalAggregateError("protocol-v2 daily budget evidence is incomplete")
    _exact_digest_fields(
        daily_budget,
        ("sha256", "file_sha256"),
        "protocol-v2 daily budget evidence",
    )
    planned = len(mapping) * len(ARMS) * TASK_COUNT
    if any(
        (
            scientific
            != {
                "task_count_per_arm": TASK_COUNT,
                "comparison_arms": list(ARMS),
                "pass_k": 1,
                "retry_limit": 0,
                "same_task_versions_models_harness_budgets_and_sampling_recipe": True,
                "whole_replica_pairs_only": True,
                "cell_level_replacement_forbidden": True,
                "seed_reuse_forbidden": True,
                "later_invalid_seed_requires_versioned_successor_before_score_unseal": True,
            },
            not isinstance(scientific, dict) or set(scientific) != SCIENTIFIC_IDENTITY_FIELDS,
            not isinstance(capacity, dict) or set(capacity) != CAPACITY_FIELDS,
            daily_budget.get("path")
            != "docs/evidence/qwen38-fleet-global-daily-rollout-budget-20260923.json",
            daily_budget.get("sha256")
            != "sha256:85c5a4f38943abab21ae60db1791fbc6ab005560b665e6680dfaa7f5889819f2",
            daily_budget.get("file_sha256")
            != "sha256:064bad27391869368ce2c14853663e7a181020b0ef6f73a4325c5697c1b46e15",
            daily_budget.get("window_utc")
            != {
                "start_inclusive": "2026-09-23T00:00:00Z",
                "end_exclusive": "2026-09-24T00:00:00Z",
            },
            isinstance(capacity, dict) and capacity.get("new_replacement_rollouts") != planned,
            isinstance(capacity, dict)
            and capacity.get("final_comparison_rollouts") != TASK_COUNT * PASS_K * len(ARMS),
            isinstance(capacity, dict) and capacity.get("actual_started_rollouts_today") != 112,
            isinstance(capacity, dict)
            and capacity.get("projected_rollouts_after_reservation") != 350,
            isinstance(capacity, dict) and capacity.get("remaining_after_reservation") != 150,
            isinstance(capacity, dict) and capacity.get("daily_rollout_cap") != 500,
            isinstance(capacity, dict) and capacity.get("within_daily_cap") is not True,
            not isinstance(retirement, dict) or set(retirement) != RETIREMENT_EVIDENCE_FIELDS,
            isinstance(retirement, dict) and retirement.get("targets") != 2,
            isinstance(retirement, dict) and retirement.get("model_rollouts") != 0,
            isinstance(retirement, dict)
            and retirement.get("outputs_or_databases_deleted") is not False,
            privacy
            != {
                "score_values_read": False,
                "prompts_responses_flags_rewards_or_trace_content_read": False,
                "infrastructure_reason_classes_only": True,
            },
            not isinstance(privacy, dict) or set(privacy) != PRIVACY_FIELDS,
            not isinstance(selection, dict) or set(selection) != SELECTION_FIELDS,
            isinstance(selection, dict)
            and selection.get("selection_sha256")
            != "sha256:38ea6686afa068c19e69fea2493e01027fdf99f7e01b20376e107b1a0cfa0b68",
            isinstance(selection, dict)
            and selection.get("split_sha256")
            != "sha256:05b3a8dc90ca93adc9671942d75ecb54ff8d0951b0dd48a087e29ec1e641840c",
            isinstance(selection, dict)
            and selection.get("binding_roster_sha256")
            != "sha256:39ae49c2db322725800b15347d5da5d171e7e03b21dec5b570683414d70e83c5",
            isinstance(selection, dict) and selection.get("corpus_dev_windows") != 0,
        )
    ):
        raise FinalAggregateError("protocol-v2 migration safety contract differs")
    return receipt, comparison, receipt_file_sha256, comparison_file_sha256


def _replica_descriptor(
    *,
    seed: int,
    arm: str,
    origin: str,
    protocol: Mapping[str, Any],
    arm_evidence: Mapping[tuple[int, str], Mapping[str, Any]],
    original_by_replacement: Mapping[int, int],
) -> dict[str, Any]:
    if origin == "retained_original":
        suffix = "base-p1-v1" if arm == "base" else "t3k32s1000-p1-v2"
        experiment = f"q38-dev17-s{seed}-{suffix}"
        job_name = f"chris-{experiment}"
        config_map_name = (
            f"chris-q38-dev17-s{seed}-base-code-v1"
            if arm == "base"
            else f"chris-q38-dev17-s{seed}-t3k32s1000-code-v2"
        )
        output_root = f"/mnt/sfs/jobs/chris-q38-fleet-dev17-s{seed}-{suffix}"
        database = experiment.replace("-", "_")
        replaces_seed = None
    else:
        model = "base" if arm == "base" else "t3k32s1000"
        experiment = f"q38-s{seed}-{model}-repl-p1-v2"
        short = f"{model}-repl-p1-v2"
        job_name = f"chris-q38-dev17-s{seed}-{short}"
        config_map_name = f"chris-q38-dev17-s{seed}-{model}-repl-code-v2"
        output_root = f"/mnt/sfs/jobs/chris-q38-fleet-dev17-s{seed}-{short}"
        database = f"q38_dev17_s{seed}_{model}_repl_p1_v2"
        replaces_seed = original_by_replacement[seed]
    binding = _arm_evidence_binding(arm_evidence[(seed, arm)], label="final arm")
    identity = binding["evaluation_identity"]
    expected_revision = (
        "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
        if arm == "base"
        else "sha256:023c5f8b0559ba050f0d672a6bc27aabecec7d5837595f8ea5bc914446d26db5"
    )
    if any(
        (
            identity["protocol_id"] != protocol["protocol_id"],
            identity["comparison_arms"] != list(ARMS),
            identity["arm_id"] != arm,
            identity["evaluation_config_name"] != experiment,
            identity["task_selection_sha256"]
            != "sha256:79c834e739246da29aca9513965ecfc7032f8df7744eb2245ce0303aba1b97c5",
            identity["split_manifest_file_sha256"]
            != "sha256:28a3dcaf31f14d724def9023d9435681772d5b8b3a8647cacc7f72ea8fc8adcb",
            identity["split_manifest_sha256"]
            != "sha256:05b3a8dc90ca93adc9671942d75ecb54ff8d0951b0dd48a087e29ec1e641840c",
            identity["comparison_protocol_sha256"] != protocol["comparison_protocol_sha256"],
            identity["model_revision"] != expected_revision,
            identity["harness"] != "opencode",
            identity["harness_version"] != "1.18.27",
            identity["context_management"] != "opencode_1.18.27_native_compaction_autocontinue_v2",
            identity["sampling_seed"] != seed,
            identity["pass_k"] != 1,
            identity["retry_limit"] != 0,
            identity["output_root"] != output_root,
            identity["database"] != database,
        )
    ):
        raise FinalAggregateError("final arm evaluation identity differs")
    _exact_digest_fields(
        identity,
        (
            "evaluation_config_sha256",
            "comparison_protocol_file_sha256",
            "checkpoint_provenance_sha256",
            "serving_route_proof_sha256",
        ),
        "final arm evaluation identity",
    )
    if (
        origin == "retained_original"
        and binding["evaluation_identity_sha256"] != RETAINED_EVALUATION_IDENTITIES[(seed, arm)]
    ):
        raise FinalAggregateError("retained arm evaluation identity differs")
    return {
        "seed": seed,
        "arm": arm,
        "origin": origin,
        "replaces_seed": replaces_seed,
        "experiment_id": experiment,
        "database": database,
        "job_name": job_name,
        "config_map_name": config_map_name,
        "output_root": output_root,
        "terminal_receipt_path": f"{output_root}/TERMINAL_OBSERVATION.json",
        "protocol_id": protocol["protocol_id"],
        "comparison_protocol_sha256": protocol["comparison_protocol_sha256"],
        **binding,
    }


def _sanitized_excluded_source_evidence(
    migrations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Copy only reviewed infrastructure fields into the private plan."""

    output = []
    for row in migrations:
        arms = row["excluded_source_arms"]
        output.append(
            {
                "invalid_original_seed": row["invalid_original_seed"],
                "replacement_seed": row["replacement_seed"],
                "reason_class": row["reason_class"],
                "evidence_receipt_sha256s": list(row["evidence_receipt_sha256s"]),
                "excluded_source_arms": {
                    arm: {field: arms[arm][field] for field in EXCLUDED_SOURCE_ARM_FIELDS}
                    for arm in ARMS
                },
                "whole_pair_excluded": True,
            }
        )
    return output


def build_current_study_plan(
    *,
    task_set_path: Path,
    roster_path: Path,
    base_config_path: Path,
    migration_receipt_path: Path,
) -> dict[str, Any]:
    """Build the exact score-blind gate plan for the protocol-v2 pass@8 study."""

    tasks, roster, task_set_file_sha256, roster_file_sha256 = _task_inputs(
        task_set_path, roster_path
    )
    (
        migration,
        comparison,
        migration_receipt_file_sha256,
        comparison_definition_file_sha256,
    ) = _migration_input(migration_receipt_path)
    base = _read_json(base_config_path, "base evaluation config")
    harness = base.get("harness")
    images = base.get("images")
    routes = base.get("routes")
    if (
        not isinstance(harness, dict)
        or harness.get("harness") != "opencode"
        or harness.get("harness_version") != "1.18.27"
        or harness.get("context_management") != "opencode_1.18.27_native_compaction_autocontinue_v2"
        or not isinstance(images, dict)
        or not isinstance(routes, dict)
        or set(routes) != {"base"}
    ):
        raise FinalAggregateError("base evaluation treatment identity differs")
    base_route = routes["base"]
    if base_route.get("task_versions") != [row["task_version_id"] for row in tasks]:
        raise FinalAggregateError("base route task order differs from dev17")
    candidate_route = json.loads(json.dumps(base_route))
    candidate_route.update(model="teacher3k32-step1000", served_id="chris-q38-t3k32-s1000-v1")
    candidate_route["model_info"]["model_path"] = "/scratch/models/chris-q38-t3k32-s1000-v1"
    candidate_route["server_info"]["model_path"] = "/scratch/models/chris-q38-t3k32-s1000-v1"
    arms = {
        "base": {
            "model": base["models"]["qwen3.8-27b-base"],
            "model_id": "qwen3.8-27b-base",
            "serving_block": "base",
            "route": base_route,
        },
        "candidate": {
            "model": {
                "repository": "/mnt/sfs/jobs/chris-q38-t3k32-b8-v3/hf-export-step1000-v1",
                "revision": (
                    "sha256:023c5f8b0559ba050f0d672a6bc27aabecec7d5837595f8ea5bc914446d26db5"
                ),
                "session_model": "qwen/chris-q38-t3k32-s1000-v1",
            },
            "model_id": "teacher3k32-step1000",
            "serving_block": "teacher3k32",
            "route": candidate_route,
        },
    }
    arm_evidence = {
        **{(row["seed"], row["arm_id"]): row for row in migration["retained_arms"]},
        **{(row["replacement_seed"], row["arm_id"]): row for row in migration["replacement_arms"]},
    }
    original_by_replacement = {
        row["replacement_seed"]: row["invalid_original_seed"]
        for row in comparison["replacement_mapping"]
    }
    protocols = {row["seed"]: row for row in comparison["replica_protocols"]}
    replicas = []
    for seed in comparison["included_seeds"]:
        for arm in ARMS:
            replicas.append(
                _replica_descriptor(
                    seed=seed,
                    arm=arm,
                    origin=protocols[seed]["origin"],
                    protocol=protocols[seed],
                    arm_evidence=arm_evidence,
                    original_by_replacement=original_by_replacement,
                )
            )
    plan = {
        "schema": PLAN_SCHEMA,
        "study_id": comparison["protocol_study_id"],
        "benchmark": BENCHMARK,
        "comparison_definition": comparison,
        "comparison_definition_sha256": comparison["sha256"],
        "comparison_definition_file_sha256": comparison_definition_file_sha256,
        "migration_receipt_sha256": migration["sha256"],
        "migration_receipt_file_sha256": migration_receipt_file_sha256,
        "source_seeds": list(SOURCE_SEEDS),
        "excluded_original_seeds": comparison["excluded_original_seeds"],
        "replacement_mapping": comparison["replacement_mapping"],
        "included_seeds": comparison["included_seeds"],
        "excluded_source_evidence": _sanitized_excluded_source_evidence(migration["migrations"]),
        "task_set_file_sha256": task_set_file_sha256,
        "task_selection_sha256": (
            "sha256:38ea6686afa068c19e69fea2493e01027fdf99f7e01b20376e107b1a0cfa0b68"
        ),
        "binding_roster_file_sha256": roster_file_sha256,
        "binding_roster_sha256": roster["sha256"],
        "bindings_sha256": roster["bindings_sha256"],
        "tasks": tasks,
        "seeds": comparison["included_seeds"],
        "pass_k": PASS_K,
        "replica_pass_k": 1,
        "retry_limit": 0,
        "harness": harness,
        "images": images,
        "sampling": {"temperature": 0.6, "top_p": 0.95},
        "arms": arms,
        "replicas": replicas,
        "public_output_schema": PUBLIC_SCHEMA,
        "private_output_root": PRIVATE_OUTPUT_ROOT,
    }
    return {**plan, "sha256": _digest(plan)}


def validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema") != PLAN_SCHEMA:
        raise FinalAggregateError("final aggregate plan schema differs")
    _require_self_digest(plan, "sha256", "final aggregate plan")
    comparison = plan.get("comparison_definition")
    if not isinstance(comparison, dict):
        raise FinalAggregateError("final aggregate comparison definition is missing")
    mapping, included, protocol_by_seed = _comparison_contract(comparison)
    if any(
        (
            plan.get("study_id") != "q38-dev17-base-step1000-p8-v2",
            plan.get("benchmark") != BENCHMARK,
            plan.get("comparison_definition_sha256") != comparison["sha256"],
            plan.get("source_seeds") != list(SOURCE_SEEDS),
            plan.get("included_seeds") != included,
            plan.get("seeds") != included,
            plan.get("excluded_original_seeds") != comparison["excluded_original_seeds"],
            plan.get("replacement_mapping") != mapping,
            plan.get("pass_k") != PASS_K,
            plan.get("replica_pass_k") != 1,
            plan.get("retry_limit") != 0,
            plan.get("public_output_schema") != PUBLIC_SCHEMA,
            plan.get("private_output_root") != PRIVATE_OUTPUT_ROOT,
            plan.get("task_set_file_sha256")
            != "sha256:79c834e739246da29aca9513965ecfc7032f8df7744eb2245ce0303aba1b97c5",
            plan.get("task_selection_sha256")
            != "sha256:38ea6686afa068c19e69fea2493e01027fdf99f7e01b20376e107b1a0cfa0b68",
            plan.get("binding_roster_file_sha256")
            != "sha256:9d6fc1e4780e49f5054ac44dfabe13310a9b387a8361e4c01f09a9870c76aed3",
            plan.get("binding_roster_sha256")
            != "sha256:39ae49c2db322725800b15347d5da5d171e7e03b21dec5b570683414d70e83c5",
            plan.get("bindings_sha256")
            != "sha256:32230df8759bb74eb306d16cd9e32686618b529aa7c0b0ac1fa1d751ecd38728",
        )
    ):
        raise FinalAggregateError("final aggregate study identity differs")
    _exact_digest_fields(
        plan,
        (
            "comparison_definition_file_sha256",
            "migration_receipt_sha256",
            "migration_receipt_file_sha256",
        ),
        "final aggregate plan",
    )
    if any(
        _digest(plan.get(field)) != digest for field, digest in EXACT_PLAN_FIELD_DIGESTS.items()
    ):
        raise FinalAggregateError(
            "final aggregate immutable model, task, or protocol binding differs"
        )
    tasks = plan.get("tasks")
    replicas = plan.get("replicas")
    arms = plan.get("arms")
    if (
        not isinstance(tasks, list)
        or len(tasks) != TASK_COUNT
        or len({row.get("task_version_id") for row in tasks if isinstance(row, dict)}) != TASK_COUNT
        or not isinstance(replicas, list)
        or len(replicas) != PASS_K * len(ARMS)
        or not isinstance(arms, dict)
        or set(arms) != set(ARMS)
    ):
        raise FinalAggregateError("final aggregate roster is incomplete")
    identities = {(row.get("seed"), row.get("arm")) for row in replicas if isinstance(row, dict)}
    if identities != {(seed, arm) for seed in included for arm in ARMS}:
        raise FinalAggregateError("final aggregate does not contain eight included pairs once")
    original_by_replacement = {
        row["replacement_seed"]: row["invalid_original_seed"] for row in mapping
    }
    arm_evidence: dict[tuple[int, str], dict[str, Any]] = {}
    for replica in replicas:
        if not isinstance(replica, dict):
            raise FinalAggregateError("final aggregate replica identity differs")
        seed = replica["seed"]
        arm = replica["arm"]
        replacement = seed in original_by_replacement
        arm_evidence[(seed, arm)] = _arm_evidence_binding(replica, label="final replica")
        if (
            replica.get("origin") != protocol_by_seed[seed]["origin"]
            or replica.get("protocol_id") != protocol_by_seed[seed]["protocol_id"]
            or replica.get("comparison_protocol_sha256")
            != protocol_by_seed[seed]["comparison_protocol_sha256"]
            or (
                not replacement
                and replica.get("evaluation_identity_sha256")
                != RETAINED_EVALUATION_IDENTITIES[(seed, arm)]
            )
        ):
            raise FinalAggregateError("final aggregate replica protocol identity differs")
    for replica in replicas:
        seed = replica["seed"]
        arm = replica["arm"]
        expected = _replica_descriptor(
            seed=seed,
            arm=arm,
            origin=protocol_by_seed[seed]["origin"],
            protocol=protocol_by_seed[seed],
            arm_evidence=arm_evidence,
            original_by_replacement=original_by_replacement,
        )
        if replica != expected:
            raise FinalAggregateError("final aggregate replica identity differs")
    excluded_source_evidence = plan.get("excluded_source_evidence")
    if not isinstance(excluded_source_evidence, list) or len(excluded_source_evidence) != len(
        mapping
    ):
        raise FinalAggregateError("final aggregate excluded source evidence is incomplete")
    for expected, evidence in zip(mapping, excluded_source_evidence, strict=True):
        if (
            not isinstance(evidence, dict)
            or set(evidence) != MIGRATION_FIELDS
            or evidence.get("invalid_original_seed") != expected["invalid_original_seed"]
            or evidence.get("replacement_seed") != expected["replacement_seed"]
            or evidence.get("whole_pair_excluded") is not True
            or evidence.get("reason_class") not in ALLOWED_EXCLUSION_REASONS
        ):
            raise FinalAggregateError("final aggregate excluded source evidence differs")
        source_arms = evidence.get("excluded_source_arms")
        if not isinstance(source_arms, dict) or set(source_arms) != set(ARMS):
            raise FinalAggregateError("final aggregate excluded source evidence differs")
        if any(
            not isinstance(source_arms[arm], dict)
            or set(source_arms[arm]) != EXCLUDED_SOURCE_ARM_FIELDS
            for arm in ARMS
        ):
            raise FinalAggregateError("final aggregate excluded source evidence differs")
    if plan.get("harness", {}).get("harness_version") != "1.18.27":
        raise FinalAggregateError("final aggregate harness identity differs")


@dataclass(frozen=True)
class GateSnapshot:
    plan_sha256: str
    cells: list[dict[str, Any]]
    local_metadata: list[dict[str, Any]]
    events: list[dict[str, Any]]
    reconciliations: list[dict[str, Any]]


class Snapshot(Protocol):
    def gate(self) -> GateSnapshot: ...

    def scored_results(self) -> list[dict[str, Any]]: ...


def _evaluation_plan(replica: Mapping[str, Any]) -> dict[str, Any]:
    path = _bound_path(str(replica["output_root"])) / "EVAL.json"
    value = _read_json(path, f"seed {replica['seed']} {replica['arm']} EVAL")
    if set(value) != EVALUATION_FIELDS:
        raise FinalAggregateError("evaluation plan field roster differs")
    claimed = value.get("sha256")
    if not isinstance(claimed, str) or not re.fullmatch(r"[0-9a-f]{64}", claimed):
        raise FinalAggregateError("evaluation plan digest is invalid")
    if _plain_digest({key: item for key, item in value.items() if key != "sha256"}) != claimed:
        raise FinalAggregateError("evaluation plan self digest differs")
    runtime = value.get("runtime_files")
    if (
        not isinstance(runtime, dict)
        or set(runtime) != RUNTIME_FILES
        or any(
            not isinstance(item, str) or re.fullmatch(r"[0-9a-f]{64}", item) is None
            for item in runtime.values()
        )
    ):
        raise FinalAggregateError("evaluation runtime identity is incomplete")
    if "sha256:" + claimed != replica.get("evaluation_plan_sha256") or runtime != replica.get(
        "runtime_files_sha256"
    ):
        raise FinalAggregateError("evaluation plan or runtime differs from sealed packet evidence")
    return value


def _terminal_receipt(replica: Mapping[str, Any]) -> tuple[dict[str, Any], str, str]:
    path = _bound_path(str(replica["terminal_receipt_path"]))
    receipt, file_sha256 = _read_json_and_digest(
        path, f"seed {replica['seed']} {replica['arm']} terminal receipt"
    )
    self_digest = _require_self_digest(receipt, "sha256", "held-out terminal receipt")
    job = receipt.get("job", {})
    config_map = receipt.get("config_map", {})
    database = receipt.get("database", {})
    summary = database.get("summary", {}) if isinstance(database, dict) else {}
    output_root = receipt.get("output_root", {})
    decision = receipt.get("decision", {})
    privacy = receipt.get("privacy", {})
    workloads = receipt.get("workloads")
    pods = receipt.get("pods")
    try:
        observed_at = datetime.fromisoformat(
            str(receipt.get("observed_at", "")).replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise FinalAggregateError("held-out terminal receipt shape differs") from exc
    if (
        set(receipt) != TERMINAL_FIELDS
        or not all(
            isinstance(value, dict)
            for value in (job, config_map, database, summary, output_root, decision, privacy)
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
        or set(output_root) != {"path", "exists"}
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
        or observed_at.tzinfo is None
        or observed_at.utcoffset() != UTC.utcoffset(observed_at)
        or not isinstance(workloads, list)
        or not isinstance(pods, list)
        or any(
            not isinstance(row, dict)
            or set(row) != {"name", "uid", "phase"}
            or not isinstance(row["name"], str)
            or not isinstance(row["uid"], str)
            or UUID.fullmatch(row["uid"]) is None
            or (row["phase"] is not None and not isinstance(row["phase"], str))
            for rows in (workloads, pods)
            for row in rows
        )
    ):
        raise FinalAggregateError("held-out terminal receipt shape differs")
    terminal_condition = job.get("terminal_condition")
    terminal_counts_match = (
        terminal_condition == "Complete" and job.get("succeeded") == 1 and job.get("failed") == 0
    ) or (terminal_condition == "Failed" and job.get("succeeded") == 0 and job.get("failed") == 1)
    if any(
        (
            receipt.get("schema") != TERMINAL_SCHEMA,
            receipt.get("protocol_id") != replica["protocol_id"],
            receipt.get("comparison_protocol_sha256") != replica["comparison_protocol_sha256"],
            receipt.get("evaluation_identity_sha256") != replica["evaluation_identity_sha256"],
            receipt.get("arm_id") != replica["arm"],
            job.get("name") != replica["job_name"],
            not isinstance(job.get("uid"), str),
            isinstance(job.get("uid"), str) and UUID.fullmatch(job["uid"]) is None,
            config_map.get("name") != replica["config_map_name"],
            not isinstance(config_map.get("uid"), str),
            isinstance(config_map.get("uid"), str) and UUID.fullmatch(config_map["uid"]) is None,
            not terminal_counts_match,
            database.get("name") != replica["database"],
            output_root.get("path") != replica["output_root"],
            output_root.get("exists") is not True,
            decision.get("capability_result_status") != "not_interpreted",
            not isinstance(decision.get("score_blind_reconciliation_required"), bool),
            type(decision.get("unresolved_cells")) is not int,
            type(decision.get("unresolved_cells")) is int and decision["unresolved_cells"] < 0,
            isinstance(decision.get("score_blind_reconciliation_required"), bool)
            and type(decision.get("unresolved_cells")) is int
            and decision["score_blind_reconciliation_required"]
            != (decision["unresolved_cells"] > 0),
            decision.get("rollout_retry_performed") is not False,
            decision.get("score_read_or_generated") is not False,
            privacy.get("prompts_responses_flags_rewards_or_trace_content_included") is not False,
            privacy.get("score_values_included") is not False,
            privacy.get("credentials_included") is not False,
        )
    ):
        raise FinalAggregateError("held-out terminal receipt identity differs")
    return receipt, self_digest, file_sha256


def _expected_task_map(plan: Mapping[str, Any]) -> dict[str, str]:
    return {row["task_version_id"]: row["task_key"] for row in plan["tasks"]}


def _event_detail(row: Mapping[str, Any]) -> dict[str, Any]:
    value = row.get("detail_json")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise FinalAggregateError("ledger accepted event detail is invalid") from exc
    if not isinstance(value, dict):
        raise FinalAggregateError("ledger accepted event detail is invalid")
    return value


RECONCILIATION_COMMON_FIELDS = {
    "schema_version",
    "reviewed_intent_sha256",
    "evaluation_plan_sha256",
    "source_job_uid_sha256",
    "source_job_terminal_receipt_sha256",
    "selected_cell_count",
    "prior_retry_review_count",
    "accepted_existing_completed_session_count",
    "model_generation_performed",
    "scoring_call_performed",
    "score_values_included",
    "prompt_response_flag_reward_or_trace_content_included",
    "cell_task_session_or_trace_identifiers_included",
    "receipt_sha256",
}
RECONCILIATION_V1_FIELDS = RECONCILIATION_COMMON_FIELDS | {"prior_stale_active_count"}
RECONCILIATION_V2_FIELDS = RECONCILIATION_COMMON_FIELDS | {
    "source_agent_exit_code",
    "source_agent_termination",
    "source_failure_code_sha256",
    "action",
}
RECONCILIATION_V2_SUBSET_FIELDS = RECONCILIATION_V2_FIELDS | {
    "source_total_cell_count",
    "prior_arm_state_counts",
    "post_arm_state_counts",
    "nonselected_cell_count",
    "nonselected_cells_preserved",
}
RECONCILIATION_V2_LOCAL_GAP_FIELDS = {
    "source_local_result_count",
    "missing_local_result_count",
    "missing_local_results_are_unselected",
    "selected_cells_have_local_results",
    "missing_local_result_cells_preserved",
    "missing_local_result_failure_code_sha256",
}
RECONCILIATION_STATES = (
    "pending",
    "claimed",
    "running",
    "grading",
    "accepted",
    "retry_review",
    "terminal",
)


def _reconciliation_shape(evidence: Mapping[str, Any], schema: str) -> None:
    fields = set(evidence)
    if schema == "fleet-stored-session-reconciliation-v1":
        expected = RECONCILIATION_V1_FIELDS
    elif schema == "fleet-stored-session-reconciliation-v2":
        expected = RECONCILIATION_V2_FIELDS
        if "source_total_cell_count" in fields:
            expected = set(RECONCILIATION_V2_SUBSET_FIELDS)
            if fields & RECONCILIATION_V2_LOCAL_GAP_FIELDS:
                expected |= RECONCILIATION_V2_LOCAL_GAP_FIELDS
    else:
        raise FinalAggregateError("accepted reconciliation receipt is invalid")
    if fields != expected:
        raise FinalAggregateError("accepted reconciliation receipt is invalid")


def _validate_acceptance_evidence(
    cell: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    reconciliations: Sequence[dict],
    *,
    evaluation_plan_sha256: str,
    terminal_receipt_sha256: str,
    source_job_uid: str,
    reconciled_cell_count: int,
) -> None:
    cell_events = [row for row in events if row.get("cell_id") == cell["cell_id"]]
    if (
        len(cell_events) != 1
        or set(cell_events[0]) != {"cell_id", "event", "to_state", "detail_json"}
        or cell_events[0].get("to_state") != "accepted"
    ):
        raise FinalAggregateError("accepted cell lacks one exact acceptance event")
    detail = _event_detail(cell_events[0])
    receipt = _normalized_digest(cell.get("receipt_digest"), "accepted receipt")
    event_receipt = detail.get("receipt_digest", detail.get("cell_receipt_sha256"))
    if _normalized_digest(event_receipt, "accepted event receipt") != receipt:
        raise FinalAggregateError("accepted event receipt differs from ledger cell")
    reconciliation = cell.get("reconciliation_digest")
    if reconciliation is None:
        if cell_events[0].get("event") != "accepted" or set(detail) != {"receipt_digest"}:
            raise FinalAggregateError("unreconciled acceptance event kind differs")
        return
    reconciliation = _normalized_digest(reconciliation, "reconciliation digest")
    event_intent = detail.get("reviewed_intent_sha256", detail.get("reconciliation_digest"))
    if _normalized_digest(event_intent, "accepted reconciliation event") != reconciliation:
        raise FinalAggregateError("accepted reconciliation event differs")
    matches = []
    for value in reconciliations:
        if not isinstance(value, dict):
            continue
        reviewed = value.get("reviewed_intent_sha256")
        if (
            isinstance(reviewed, str)
            and SHA256.fullmatch(reviewed)
            and reviewed.removeprefix("sha256:") == reconciliation
        ):
            matches.append(value)
    if len(matches) != 1:
        raise FinalAggregateError("accepted reconciliation receipt is missing or ambiguous")
    evidence = matches[0]
    _require_self_digest(evidence, "receipt_sha256", "accepted reconciliation receipt")
    schema = evidence.get("schema_version")
    _reconciliation_shape(evidence, schema)
    expected_event = {
        "fleet-stored-session-reconciliation-v1": "stored_session_reconciled",
        "fleet-stored-session-reconciliation-v2": "stored_scored_session_reconciled",
    }.get(schema)
    expected_detail_fields = {
        "fleet-stored-session-reconciliation-v1": {
            "reviewed_intent_sha256",
            "cell_receipt_sha256",
            "source_job_terminal_receipt_sha256",
            "failure_code",
        },
        "fleet-stored-session-reconciliation-v2": {
            "reviewed_intent_sha256",
            "cell_receipt_sha256",
            "source_job_terminal_receipt_sha256",
            "action",
        },
    }.get(schema)
    source_job_uid_sha256 = "sha256:" + hashlib.sha256(source_job_uid.encode()).hexdigest()
    selected = evidence.get("selected_cell_count")
    prior_retry = evidence.get("prior_retry_review_count")
    if type(selected) is not int or type(prior_retry) is not int or selected < 1 or prior_retry < 0:
        raise FinalAggregateError("accepted reconciliation receipt is invalid")
    if schema == "fleet-stored-session-reconciliation-v1":
        stale = evidence.get("prior_stale_active_count")
        if type(stale) is not int or stale < 0 or prior_retry + stale != selected:
            raise FinalAggregateError("accepted reconciliation receipt is invalid")
    else:
        if prior_retry < selected:
            raise FinalAggregateError("accepted reconciliation receipt is invalid")
        _normalized_digest(evidence.get("source_failure_code_sha256"), "source failure code")
        if type(evidence.get("source_agent_exit_code")) is not int or (
            evidence.get("source_agent_exit_code"),
            evidence.get("source_agent_termination"),
        ) not in {(0, "output_limit"), (1, "process_error")}:
            raise FinalAggregateError("accepted reconciliation receipt is invalid")
        if "source_total_cell_count" in evidence:
            total = evidence.get("source_total_cell_count")
            nonselected = evidence.get("nonselected_cell_count")
            prior_states = evidence.get("prior_arm_state_counts")
            post_states = evidence.get("post_arm_state_counts")
            if (
                type(total) is not int
                or type(nonselected) is not int
                or total != TASK_COUNT
                or total != selected + nonselected
                or nonselected < 1
                or evidence.get("nonselected_cells_preserved") is not True
                or not isinstance(prior_states, dict)
                or not isinstance(post_states, dict)
                or set(prior_states) != set(RECONCILIATION_STATES)
                or set(post_states) != set(RECONCILIATION_STATES)
                or any(type(value) is not int or value < 0 for value in prior_states.values())
                or any(type(value) is not int or value < 0 for value in post_states.values())
                or sum(prior_states.values()) != total
                or sum(post_states.values()) != total
                or prior_states["retry_review"] != prior_retry
            ):
                raise FinalAggregateError("accepted reconciliation receipt is invalid")
            expected_post = dict(prior_states)
            expected_post["retry_review"] -= selected
            expected_post["accepted"] += selected
            if expected_post != post_states:
                raise FinalAggregateError("accepted reconciliation receipt is invalid")
        if "source_local_result_count" in evidence:
            _normalized_digest(
                evidence.get("missing_local_result_failure_code_sha256"),
                "missing local result failure code",
            )
            if any(
                evidence.get(field) is not True
                for field in (
                    "missing_local_results_are_unselected",
                    "selected_cells_have_local_results",
                    "missing_local_result_cells_preserved",
                )
            ) or (
                type(evidence.get("source_local_result_count")) is not int
                or type(evidence.get("missing_local_result_count")) is not int
                or evidence["missing_local_result_count"] < 1
                or evidence["source_local_result_count"] + evidence["missing_local_result_count"]
                != evidence["source_total_cell_count"]
                or evidence["missing_local_result_count"] > evidence["nonselected_cell_count"]
            ):
                raise FinalAggregateError("accepted reconciliation receipt is invalid")
    if any(
        (
            expected_event is None,
            expected_detail_fields is None,
            isinstance(expected_detail_fields, set) and set(detail) != expected_detail_fields,
            cell_events[0].get("event") != expected_event,
            _normalized_digest(
                evidence.get("evaluation_plan_sha256"), "reconciliation evaluation plan"
            )
            != _normalized_digest(evaluation_plan_sha256, "evaluation plan"),
            _normalized_digest(
                evidence.get("source_job_terminal_receipt_sha256"),
                "reconciliation source terminal",
            )
            != _normalized_digest(terminal_receipt_sha256, "source terminal"),
            evidence.get("source_job_uid_sha256") != source_job_uid_sha256,
            evidence.get("selected_cell_count") != reconciled_cell_count,
            evidence.get("accepted_existing_completed_session_count") != reconciled_cell_count,
            evidence.get("model_generation_performed") is not False,
            evidence.get("scoring_call_performed") is not False,
            evidence.get("score_values_included") is not False,
            evidence.get("prompt_response_flag_reward_or_trace_content_included") is not False,
            evidence.get("cell_task_session_or_trace_identifiers_included") is not False,
            _normalized_digest(
                detail.get("source_job_terminal_receipt_sha256"),
                "accepted event source terminal",
            )
            != _normalized_digest(terminal_receipt_sha256, "source terminal"),
            schema == "fleet-stored-session-reconciliation-v2"
            and evidence.get("action") != "accept_existing_scored_session",
            schema == "fleet-stored-session-reconciliation-v2"
            and detail.get("action") != "accept_existing_scored_session",
        )
    ):
        raise FinalAggregateError("accepted reconciliation receipt is invalid")


def _validate_eval_identity(
    plan: Mapping[str, Any], replica: Mapping[str, Any], evaluation: Mapping[str, Any]
) -> None:
    arm = plan["arms"][replica["arm"]]
    expected_tasks = sorted(plan["tasks"], key=lambda row: row["task_version_id"])
    if any(
        (
            evaluation.get("schema") != "cyber_fleet_eval_v1",
            evaluation.get("campaign_id") != replica["experiment_id"],
            evaluation.get("run_prefix") != replica["experiment_id"],
            evaluation.get("selection") != {"source_job_id": SOURCE_TASK_JOB_ID},
            evaluation.get("tasks") != expected_tasks,
            evaluation.get("models") != {arm["model_id"]: arm["model"]},
            evaluation.get("routes") != {arm["serving_block"]: arm["route"]},
            evaluation.get("treatment") != plan["harness"],
            evaluation.get("images") != plan["images"],
            evaluation.get("sampling") != {**plan["sampling"], "seed": replica["seed"]},
            evaluation.get("pass_k") != 1,
            evaluation.get("concurrency") != 4,
            evaluation.get("automatic_retry") is not False,
            evaluation.get("max_reviewed_infrastructure_retries") != 0,
            evaluation.get("training_data_eligible") is not False,
            evaluation.get("interpretation") != "serving-block descriptive evaluation",
        )
    ):
        raise FinalAggregateError("evaluation plan model, task, harness, or protocol differs")


def _validate_worker_accepted_receipt(
    replica: Mapping[str, Any],
    cell: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> str:
    artifact_directory = metadata.get("artifact_directory")
    if not isinstance(artifact_directory, str):
        raise FinalAggregateError("accepted local result artifact directory is invalid")
    relative = PurePosixPath(artifact_directory) / "ACCEPTED.json"
    receipt, _file_sha256 = _read_json_beneath(
        _bound_path(str(replica["output_root"])),
        relative,
        f"seed {replica['seed']} {replica['arm']} accepted cell receipt",
    )
    if set(receipt) != ACCEPTED_RECEIPT_FIELDS:
        raise FinalAggregateError("accepted cell receipt shape differs")
    receipt_sha256 = _require_self_digest(receipt, "receipt_sha256", "accepted cell receipt")
    if any(
        (
            receipt.get("schema_version") != ACCEPTED_SCHEMA,
            receipt.get("accepted") is not True,
            receipt.get("campaign_id") != replica["experiment_id"],
            receipt.get("cell_id") != cell["cell_id"],
            receipt.get("ledger_cell_id") != cell["cell_id"],
            receipt.get("execution_id") != metadata.get("execution_id"),
            receipt.get("run_id") != metadata.get("run_id"),
            receipt.get("serving_block") != cell["serving_block"],
            receipt.get("session_id") != cell["session_id"],
            receipt.get("session_id") != metadata.get("session_id"),
            receipt.get("verifier_execution_id") != metadata.get("verifier_execution_id"),
            _normalized_digest(receipt.get("config_sha256"), "accepted config")
            != _normalized_digest(metadata.get("config_sha256"), "local config"),
            receipt.get("session_ingest_completed") is not True,
            receipt.get("cleanup_completed") is not True,
            receipt.get("score_persisted_privately") is not True,
            receipt.get("scores_included") is not False,
            receipt.get("prompts_or_traces_included") is not False,
            cell.get("reconciliation_digest") is None
            and _normalized_digest(cell.get("receipt_digest"), "ledger accepted receipt")
            != _normalized_digest(receipt_sha256, "accepted cell receipt"),
        )
    ):
        raise FinalAggregateError("accepted cell receipt identity differs")
    return receipt_sha256


def validate_gate(
    plan: Mapping[str, Any],
    replica: Mapping[str, Any],
    snapshot: GateSnapshot,
    evaluation: Mapping[str, Any],
    terminal: Mapping[str, Any],
) -> dict[str, str | None]:
    """Validate one replica without reading its score column."""

    _validate_eval_identity(plan, replica, evaluation)
    arm = plan["arms"][replica["arm"]]
    tasks = _expected_task_map(plan)
    expected_harness = "protocol-" + str(evaluation["sha256"])
    if snapshot.plan_sha256 != terminal.get("database", {}).get("summary", {}).get("plan_sha256"):
        raise FinalAggregateError("terminal and live database plan identities differ")
    cells = snapshot.cells
    if len(cells) != TASK_COUNT or {row.get("task_version_id") for row in cells} != set(tasks):
        raise FinalAggregateError("replica ledger does not contain the exact dev17 roster")
    metadata = snapshot.local_metadata
    counts = Counter(row.get("cell_id") for row in metadata)
    if (
        len(metadata) != TASK_COUNT
        or any(
            not isinstance(row, dict) or set(row) != set(SCORE_BLIND_LOCAL_FIELDS)
            for row in metadata
        )
        or set(counts) != {row["cell_id"] for row in cells}
        or any(count != 1 for count in counts.values())
    ):
        raise FinalAggregateError("accepted cells do not each have one score-blind local result")
    cells_by_id = {row["cell_id"]: row for row in cells}
    metadata_by_cell = {row["cell_id"]: row for row in metadata}
    event_counts = Counter(row.get("cell_id") for row in snapshot.events if isinstance(row, dict))
    if (
        len(snapshot.events) != TASK_COUNT
        or any(
            not isinstance(row, dict)
            or set(row) != ACCEPTED_EVENT_FIELDS
            or row.get("to_state") != "accepted"
            for row in snapshot.events
        )
        or set(event_counts) != set(cells_by_id)
        or any(count != 1 for count in event_counts.values())
    ):
        raise FinalAggregateError("accepted event roster differs from the exact cell roster")
    expected_reconciliation_intents = {
        _normalized_digest(row["reconciliation_digest"], "reconciliation digest")
        for row in cells
        if row.get("reconciliation_digest") is not None
    }
    reconciliation_by_intent: dict[str, dict[str, Any]] = {}
    for evidence in snapshot.reconciliations:
        if not isinstance(evidence, dict):
            raise FinalAggregateError("database reconciliation receipt is invalid")
        _require_self_digest(evidence, "receipt_sha256", "database reconciliation receipt")
        _reconciliation_shape(evidence, evidence.get("schema_version"))
        intent = _normalized_digest(
            evidence.get("reviewed_intent_sha256"), "reviewed reconciliation intent"
        )
        if intent in reconciliation_by_intent:
            raise FinalAggregateError("database reconciliation receipt roster is ambiguous")
        reconciliation_by_intent[intent] = evidence
    if set(reconciliation_by_intent) != expected_reconciliation_intents:
        raise FinalAggregateError("database reconciliation receipt roster differs")
    for row in metadata:
        if (
            row.get("session_ingest_status") != "completed"
            or row.get("session_id") != cells_by_id[row["cell_id"]]["session_id"]
        ):
            raise FinalAggregateError("accepted local result lacks completed session ingestion")
    reconciliation_counts = Counter(
        _normalized_digest(row["reconciliation_digest"], "reconciliation digest")
        for row in cells
        if row.get("reconciliation_digest") is not None
    )
    accepted_receipts: dict[str, str | None] = {}
    for cell in cells:
        version = cell["task_version_id"]
        if any(
            (
                cell.get("experiment_id") != replica["experiment_id"],
                cell.get("task_key") != tasks[version],
                cell.get("model_id") != arm["model_id"],
                cell.get("model_revision") != arm["model"]["revision"],
                cell.get("serving_block") != arm["serving_block"],
                cell.get("endpoint_model_id") != arm["route"]["served_id"],
                cell.get("harness_id") != expected_harness,
                cell.get("attempt") != 1,
                cell.get("state") != "accepted",
                cell.get("result_class") != "valid",
                cell.get("lease_expires_at") is not None,
                cell.get("retry_count") != 0,
                cell.get("max_retries") != 0,
                cell.get("failure_code") is not None,
                not isinstance(cell.get("session_id"), str),
            )
        ):
            raise FinalAggregateError("replica contains a non-authoritative or mismatched cell")
        _normalized_digest(cell.get("receipt_digest"), "accepted receipt")
        if cell.get("cell_id") != _cell_id(cell):
            raise FinalAggregateError("replica cell identity differs from its immutable plan row")
        reconciliation = cell.get("reconciliation_digest")
        _validate_acceptance_evidence(
            cell,
            snapshot.events,
            snapshot.reconciliations,
            evaluation_plan_sha256=evaluation["sha256"],
            terminal_receipt_sha256=terminal["sha256"],
            source_job_uid=terminal["job"]["uid"],
            reconciled_cell_count=(
                reconciliation_counts[_normalized_digest(reconciliation, "reconciliation digest")]
                if reconciliation is not None
                else 0
            ),
        )
        local_row = metadata_by_cell[cell["cell_id"]]
        if reconciliation is None:
            expected_lifecycle = (0, "completed")
        else:
            reconciliation_digest = _normalized_digest(reconciliation, "reconciliation digest")
            evidence = reconciliation_by_intent[reconciliation_digest]
            expected_lifecycle = (
                (1, "process_error")
                if evidence.get("schema_version") == "fleet-stored-session-reconciliation-v1"
                else (
                    evidence["source_agent_exit_code"],
                    evidence["source_agent_termination"],
                )
            )
        if (
            local_row.get("agent_exit_code"),
            local_row.get("agent_termination"),
        ) != expected_lifecycle:
            raise FinalAggregateError("accepted local result lifecycle differs")
        accepted_receipts[cell["cell_id"]] = (
            _validate_worker_accepted_receipt(replica, cell, local_row)
            if reconciliation is None
            else None
        )
    stored_plan = [
        {field: row.get(field) for field in PLAN_STORED_FIELDS}
        for row in sorted(
            cells,
            key=lambda row: (
                row["experiment_id"],
                row["task_version_id"],
                row["model_id"],
                row["attempt"],
            ),
        )
    ]
    if _plain_digest(stored_plan) != snapshot.plan_sha256:
        raise FinalAggregateError("replica immutable ledger plan digest differs")
    reconciled_count = sum(cell.get("reconciliation_digest") is not None for cell in cells)
    if terminal["decision"]["unresolved_cells"] != reconciled_count:
        raise FinalAggregateError("terminal unresolved census differs from reconciliation evidence")
    return accepted_receipts


def _validate_scored_results(
    snapshot: GateSnapshot, scored: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    if len(scored) != TASK_COUNT:
        raise FinalAggregateError("score opening did not return exactly seventeen rows")
    cells = {row["cell_id"]: row for row in snapshot.cells}
    if {row.get("cell_id") for row in scored} != set(cells):
        raise FinalAggregateError("opened scores differ from the accepted cell roster")
    metadata_fields = SCORE_BLIND_LOCAL_FIELDS
    expected_metadata = sorted(
        ({field: row.get(field) for field in metadata_fields} for row in snapshot.local_metadata),
        key=lambda row: row["cell_id"],
    )
    opened_metadata = sorted(
        ({field: row.get(field) for field in metadata_fields} for row in scored),
        key=lambda row: row["cell_id"],
    )
    if opened_metadata != expected_metadata:
        raise FinalAggregateError(
            "opened private result metadata differs from the score-blind gate"
        )
    output = []
    for row in scored:
        normalized = {field: row.get(field) for field in LOCAL_RECORD_FIELDS}
        score = normalized["score"]
        elapsed = normalized["elapsed_seconds"]
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or not 0.0 <= float(score) <= 1.0
            or isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(float(elapsed))
            or float(elapsed) < 0
        ):
            raise FinalAggregateError("opened private result has an invalid numeric value")
        for field in (
            "config_sha256",
            "trace_sha256",
            "result_sha256",
            "reward_sha256",
            "session_ingest_sha256",
            "cleanup_sha256",
        ):
            normalized[field] = _normalized_digest(normalized[field], field)
        if _plain_digest(normalized) != _normalized_digest(row.get("record_sha256"), "record"):
            raise FinalAggregateError("opened private result record digest differs")
        if normalized["session_id"] != cells[normalized["cell_id"]]["session_id"]:
            raise FinalAggregateError("opened private result session differs")
        output.append({**normalized, "record_sha256": row["record_sha256"]})
    return output


def finalize(
    plan: dict[str, Any],
    snapshots: Mapping[tuple[int, str], Snapshot],
    *,
    output_root: Path,
    publication_root: Path | None = None,
) -> dict[str, Any]:
    """Gate all replicas, then and only then open scores and publish receipts."""

    validate_plan(plan)
    expected_output_root = _bound_path(str(plan["private_output_root"]))
    if output_root != expected_output_root:
        raise FinalAggregateError("final output root differs from the sealed plan")
    publication_root = output_root if publication_root is None else publication_root
    if publication_root != output_root and publication_root.as_posix() != PRIVATE_STAGING_ROOT:
        raise FinalAggregateError("final publication root differs from the sealed runtime")
    if publication_root.exists() or publication_root.is_symlink():
        raise FinalAggregateError("final aggregate output already exists")
    if not publication_root.parent.is_dir() or publication_root.parent.is_symlink():
        raise FinalAggregateError("final aggregate parent is not an exact directory")
    replicas = {(row["seed"], row["arm"]): row for row in plan["replicas"]}
    if set(snapshots) != set(replicas):
        raise FinalAggregateError("database snapshots differ from the sealed replica roster")

    gated: dict[
        tuple[int, str], tuple[GateSnapshot, dict, dict, str, str, dict[str, str | None]]
    ] = {}
    common_runtime: dict[str, str] | None = None
    protocol_by_seed: dict[int, dict[str, str]] = defaultdict(dict)
    for identity in sorted(replicas):
        replica = replicas[identity]
        evaluation = _evaluation_plan(replica)
        runtime = evaluation["runtime_files"]
        if common_runtime is None:
            common_runtime = runtime
        elif runtime != common_runtime:
            raise FinalAggregateError("base and candidate evaluator runtime files differ")
        terminal, terminal_sha, terminal_file_sha = _terminal_receipt(replica)
        protocol_by_seed[replica["seed"]][replica["arm"]] = terminal.get(
            "comparison_protocol_sha256"
        )
        snapshot = snapshots[identity].gate()
        accepted_receipts = validate_gate(plan, replica, snapshot, evaluation, terminal)
        gated[identity] = (
            snapshot,
            evaluation,
            terminal,
            terminal_sha,
            terminal_file_sha,
            accepted_receipts,
        )
    if any(
        set(values) != set(ARMS) or values["base"] != values["candidate"]
        for values in protocol_by_seed.values()
    ):
        raise FinalAggregateError("matched base and candidate protocol identity differs")

    # This is the sole score-opening boundary. No scored_results() call occurs
    # until all sixteen terminal, identity, state, and evidence gates above pass.
    outcomes: list[dict[str, Any]] = []
    for identity in sorted(replicas):
        replica = replicas[identity]
        (
            snapshot,
            evaluation,
            _terminal,
            _terminal_sha,
            _terminal_file_sha,
            accepted_receipts,
        ) = gated[identity]
        for row in _validate_scored_results(snapshot, snapshots[identity].scored_results()):
            cell = next(item for item in snapshot.cells if item["cell_id"] == row["cell_id"])
            outcomes.append(
                {
                    "seed": replica["seed"],
                    "arm": replica["arm"],
                    "task_key": cell["task_key"],
                    "task_version_id": cell["task_version_id"],
                    "score": float(row["score"]),
                    "record_sha256": "sha256:" + _normalized_digest(row["record_sha256"], "record"),
                    "accepted_receipt_sha256": "sha256:"
                    + _normalized_digest(cell["receipt_digest"], "accepted receipt"),
                    "worker_accepted_receipt_sha256": accepted_receipts[cell["cell_id"]],
                    "evaluation_plan_sha256": "sha256:" + evaluation["sha256"],
                }
            )
    coverage = Counter((row["task_version_id"], row["arm"], row["seed"]) for row in outcomes)
    if len(outcomes) != TASK_COUNT * len(ARMS) * PASS_K or any(
        coverage[(task["task_version_id"], arm, seed)] != 1
        for task in plan["tasks"]
        for arm in ARMS
        for seed in plan["included_seeds"]
    ):
        raise FinalAggregateError("opened outcomes do not cover each task, arm, and seed once")

    permutation = list(range(TASK_COUNT))
    secrets.SystemRandom().shuffle(permutation)
    anonymization_body = {
        "schema": "cyber_private_task_anonymization_v1",
        "method": "private_random_permutation_v1",
        "private_nonce": secrets.token_hex(32),
        "rows": [
            {
                "public_task_index": public_index,
                "task_key": plan["tasks"][source_index]["task_key"],
                "task_version_id": plan["tasks"][source_index]["task_version_id"],
            }
            for public_index, source_index in enumerate(permutation)
        ],
    }
    anonymization = {
        **anonymization_body,
        "receipt_sha256": _digest(anonymization_body),
    }
    terminal_body = {
        "schema": "cyber_fleet_matched_pass8_terminal_index_v2",
        "study_plan_sha256": plan["sha256"],
        "protocol_v2_migration": {
            "migration_receipt_sha256": plan["migration_receipt_sha256"],
            "migration_receipt_file_sha256": plan["migration_receipt_file_sha256"],
            "comparison_definition_sha256": plan["comparison_definition_sha256"],
            "comparison_definition_file_sha256": plan["comparison_definition_file_sha256"],
            "source_seeds": plan["source_seeds"],
            "excluded_original_seeds": plan["excluded_original_seeds"],
            "replacement_mapping": plan["replacement_mapping"],
            "included_seeds": plan["included_seeds"],
            "excluded_source_evidence": plan["excluded_source_evidence"],
        },
        "rows": [
            {
                "seed": replica["seed"],
                "arm": replica["arm"],
                "origin": replica["origin"],
                "replaces_seed": replica["replaces_seed"],
                "evaluation_identity_sha256": terminal["evaluation_identity_sha256"],
                "comparison_protocol_sha256": terminal["comparison_protocol_sha256"],
                "job_uid": terminal["job"]["uid"],
                "terminal_receipt_file_sha256": terminal_file_sha,
                "terminal_receipt_sha256": terminal_sha,
                "database_plan_sha256": snapshot.plan_sha256,
            }
            for identity, replica in sorted(replicas.items())
            for snapshot, _evaluation, terminal, terminal_sha, terminal_file_sha, _accepted in [
                gated[identity]
            ]
        ],
    }
    terminal_index = {**terminal_body, "receipt_sha256": _digest(terminal_body)}
    outcome_body = {
        "schema": "cyber_fleet_matched_pass8_protocol_v2_private_scored_outcome_index_v1",
        "study_plan_sha256": plan["sha256"],
        "outcome_count": len(outcomes),
        "rows": sorted(
            outcomes,
            key=lambda row: (row["task_version_id"], row["arm"], row["seed"]),
        ),
    }
    outcome_index = {**outcome_body, "receipt_sha256": _digest(outcome_body)}
    outcomes_by_task = defaultdict(list)
    for row in outcomes:
        outcomes_by_task[row["task_version_id"]].append(row)
    public_rows = []
    for source_index in permutation:
        version = plan["tasks"][source_index]["task_version_id"]
        task_outcomes = outcomes_by_task[version]
        row = {"metric_denominator": 1}
        for arm in ARMS:
            scores = [item["score"] for item in task_outcomes if item["arm"] == arm]
            if len(scores) != PASS_K:
                raise FinalAggregateError("anonymized task row is missing an arm attempt")
            row[arm] = {
                "valid_attempts": PASS_K,
                "infrastructure_invalid_attempts": 0,
                "metric_numerator": int(any(score == 1.0 for score in scores)),
            }
        public_rows.append(row)
    public_body = {
        "schema_version": PUBLIC_SCHEMA,
        "status": "final",
        "benchmark": BENCHMARK,
        "comparison_definition_sha256": plan["comparison_definition_sha256"],
        "anonymization_method": "private_random_permutation_v1",
        "anonymization_receipt_sha256": anonymization["receipt_sha256"],
        "task_rows": public_rows,
        "evidence": {
            "terminal_index_sha256": terminal_index["receipt_sha256"],
            "scored_outcome_index_sha256": outcome_index["receipt_sha256"],
            "terminal_schemas": {
                "launch_terminal": TERMINAL_SCHEMA,
                "accepted_cell": ACCEPTED_SCHEMA,
                "controller_terminal": CONTROLLER_TERMINAL_SCHEMA,
            },
        },
    }
    public = {**public_body, "receipt_sha256": _digest(public_body)}
    final_body = {
        "schema": "cyber_fleet_matched_pass8_protocol_v2_final_receipt_v1",
        "study_plan_sha256": plan["sha256"],
        "status": "final",
        "task_count": TASK_COUNT,
        "valid_outcomes_per_task_arm": PASS_K,
        "private_terminal_index_sha256": terminal_index["receipt_sha256"],
        "private_scored_outcome_index_sha256": outcome_index["receipt_sha256"],
        "private_anonymization_receipt_sha256": anonymization["receipt_sha256"],
        "sanitized_aggregate_receipt_sha256": public["receipt_sha256"],
        "comparison_definition_sha256": plan["comparison_definition_sha256"],
        "migration_receipt_sha256": plan["migration_receipt_sha256"],
        "included_seeds": plan["included_seeds"],
        "excluded_original_seeds": plan["excluded_original_seeds"],
        "public_contains_private_identifiers": False,
        "prompts_responses_flags_or_traces_read": False,
    }
    final = {**final_body, "receipt_sha256": _digest(final_body)}

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{publication_root.name}.", dir=publication_root.parent)
    )
    os.chmod(temporary, 0o700)
    try:
        for name, value in (
            ("PRIVATE_TERMINAL_INDEX.json", terminal_index),
            ("PRIVATE_SCORED_OUTCOME_INDEX.json", outcome_index),
            ("PRIVATE_ANONYMIZATION.json", anonymization),
            ("SANITIZED_AGGREGATE.json", public),
            ("FINAL.json", final),
        ):
            _write_exclusive(temporary / name, value)
        _fsync_directory(temporary)
        _rename_noreplace(temporary, publication_root)
        _fsync_directory(publication_root.parent)
    except BaseException:
        # The final path is never exposed partially. A failed temporary tree is
        # retained for private diagnosis and cannot be mistaken for acceptance.
        raise
    return final


class _PostgresSnapshot:
    def __init__(self, connection: Any) -> None:
        self.connection = connection
        self.score_opened = False

    def gate(self) -> GateSnapshot:
        plan = self.connection.execute(
            "SELECT value FROM ledger_metadata WHERE key = 'plan_sha256'"
        ).fetchall()
        if len(plan) != 1:
            raise FinalAggregateError("replica database plan identity is missing")
        cells = self.connection.execute(
            f"SELECT {', '.join(CELL_FIELDS)} FROM rollout_cells ORDER BY cell_id"
        ).fetchall()
        metadata_fields = SCORE_BLIND_LOCAL_FIELDS
        metadata = self.connection.execute(
            f"SELECT {', '.join(metadata_fields)} FROM rollout_local_results ORDER BY cell_id"
        ).fetchall()
        events = self.connection.execute(
            """
            SELECT cell_id, event, to_state, detail_json
            FROM rollout_events WHERE to_state = 'accepted' ORDER BY sequence
            """
        ).fetchall()
        reconciliation_rows = self.connection.execute(
            "SELECT receipt_json FROM ledger_reconciliations ORDER BY receipt_sha256"
        ).fetchall()
        reconciliations = []
        for row in reconciliation_rows:
            value = row["receipt_json"]
            if isinstance(value, str):
                value = json.loads(value)
            if not isinstance(value, dict):
                raise FinalAggregateError("database reconciliation receipt is invalid")
            reconciliations.append(value)
        return GateSnapshot(
            plan_sha256=plan[0]["value"],
            cells=[dict(row) for row in cells],
            local_metadata=[dict(row) for row in metadata],
            events=[dict(row) for row in events],
            reconciliations=reconciliations,
        )

    def scored_results(self) -> list[dict[str, Any]]:
        self.score_opened = True
        rows = self.connection.execute(
            f"SELECT {', '.join(LOCAL_RECORD_FIELDS)}, record_sha256 "
            "FROM rollout_local_results ORDER BY cell_id"
        ).fetchall()
        return [dict(row) for row in rows]


def _database_dsn(root_dsn: str, database: str) -> str:
    try:
        parsed = urlsplit(root_dsn)
    except ValueError as exc:
        raise FinalAggregateError("PostgreSQL root connection is invalid") from exc
    query = {key.casefold() for key, _value in parse_qsl(parsed.query, keep_blank_values=True)}
    authority_overrides = {"database", "dbname", "host", "hostaddr", "port", "service"}
    if (
        parsed.scheme not in {"postgres", "postgresql"}
        or not parsed.hostname
        or parsed.fragment
        or query & authority_overrides
        or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,62}", database)
    ):
        raise FinalAggregateError("PostgreSQL root connection or database name is invalid")
    return urlunsplit(
        (parsed.scheme, parsed.netloc, "/" + quote(database, safe=""), parsed.query, "")
    )


@contextmanager
def open_postgres_snapshots(
    plan: Mapping[str, Any], root_dsn: str
) -> Iterator[dict[tuple[int, str], Snapshot]]:
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise FinalAggregateError("psycopg is required for the private final gate") from exc
    with ExitStack() as stack:
        snapshots: dict[tuple[int, str], Snapshot] = {}
        for replica in plan["replicas"]:
            connection = stack.enter_context(
                psycopg.connect(
                    _database_dsn(root_dsn, replica["database"]),
                    autocommit=False,
                    row_factory=dict_row,
                    connect_timeout=30,
                    application_name="fleet-dev17-pass8-final-v2",
                    options="-c default_transaction_read_only=on -c statement_timeout=30000",
                )
            )
            connection.execute("BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            snapshots[(replica["seed"], replica["arm"])] = _PostgresSnapshot(connection)
        yield snapshots


def run_from_environment(plan_path: Path) -> dict[str, Any]:
    plan, plan_file_sha256 = _read_json_and_digest(plan_path, "bundled final aggregate plan")
    expected = os.environ.get("FINAL_STUDY_PLAN_FILE_SHA256")
    if expected != plan_file_sha256:
        raise FinalAggregateError("bundled final aggregate plan bytes differ")
    validate_plan(plan)
    dsn = os.environ.get("ROLLOUT_DATABASE_URL")
    if not dsn:
        raise FinalAggregateError("PostgreSQL root connection is missing")
    output = Path(os.environ.get("FINAL_OUTPUT_ROOT", ""))
    if output.as_posix() != plan["private_output_root"]:
        raise FinalAggregateError("final output environment differs from the sealed plan")
    staging = Path(os.environ.get("FINAL_STAGING_ROOT", ""))
    if staging.as_posix() != PRIVATE_STAGING_ROOT:
        raise FinalAggregateError("final staging environment differs from the sealed runtime")
    with open_postgres_snapshots(plan, dsn) as snapshots:
        return finalize(plan, snapshots, output_root=output, publication_root=staging)
