"""Score-blind status ledger for the exact Qwen3.8/GLM5.3 pass@4 universe.

Only immutable acceptance receipts, execution claims, and retry-safe
pre-model tombstones are eligible inputs.  The command never opens rollout
transcripts, prompts, flags, verifier payloads, or score artifacts.
"""

from __future__ import annotations

import argparse
import json
import stat
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_bulk_runtime_v3 as bulk_runtime
from evals.fleet import exact_pass4_bulk_v3 as bulk
from evals.fleet import exact_pass4_universe as exact
from evals.fleet import qwen38_dedicated_scored_canary_v1 as qwen_dedicated
from evals.fleet import qwen_bulk_generation16 as qwen_generation16
from evals.fleet import self_hosted

DEFAULT_CAMPAIGN = Path("evals/fleet/configs/q38-glm53-exact-easiest100-pass4-campaign-v1.json")

GENERATION7_TERMINAL_SCHEMA = "fleet-opencode-autocontinue-generation7-terminal-v1"
GENERATION7_CLAIM_SCHEMA = "fleet-statistical-cell-execution-claim-v9"
GENERATION15_TERMINAL_SCHEMA = "fleet-opencode-generation15-simple-terminal-v1"
GENERATION15_CLAIM_SCHEMA = "fleet-statistical-cell-execution-claim-v15"
DEDICATED_QWEN_ACCEPTED_SCHEMA = "fleet-qwen38-dedicated-tp1-cell-accepted-v1"
ACCEPTED_SCHEMAS = {
    "fleet-exact-pass4-bulk-cell-accepted-v3",
    GENERATION7_TERMINAL_SCHEMA,
    GENERATION15_TERMINAL_SCHEMA,
    DEDICATED_QWEN_ACCEPTED_SCHEMA,
}
CLAIM_SCHEMAS = {
    bulk_runtime.CLAIM_SCHEMA,
    GENERATION7_CLAIM_SCHEMA,
    GENERATION15_CLAIM_SCHEMA,
}
GENERATION7_BINDINGS = {
    "qwen3.8-27b": {
        "cell_id": "sha256:631c9d7cc5328849ce137393943927192b1b50dc60458cdb3425fbce893ecf5a",
        "execution_id": "sha256:77deaade2d420d18602ffdfde93d1a2a9957a32a00d1f644d42053ee72c31535",
        "spec_sha256": "sha256:b3b81a36ef7e5140eb10fda3e5ba02589b3cbaca0fa2b63ba70553518e44dd34",
        "plan_sha256": "sha256:fb173b93ffc6a1b2b368f290c60b1c3e1fe4214afbbd419f9a927cb0703ede53",
        "run_id": "chris-q38-ac-g7-r004-a1-77deaade",
    },
    "glm-5.3": {
        "cell_id": "sha256:905051f141077d3aa5086c1f5dc6ad015d5ee6173a1ea7515025805cf9a24b41",
        "execution_id": "sha256:b6338d535870aee0afa28519c3078dd6749db8d6f961b3a51e84dfc65cc13524",
        "spec_sha256": "sha256:bab642484368c507f53c2b99b29f63d80eb765705899b3d8955e2bb03a84f36a",
        "plan_sha256": "sha256:178230b505ea06290a853b9090adc79937076adac915eb4a0978aa9011570f37",
        "run_id": "chris-glm53-ac-g7-r013-a1-b6338d53",
    },
}
GENERATION15_BINDINGS = {
    "qwen3.8-27b": {
        "cell_id": "sha256:631c9d7cc5328849ce137393943927192b1b50dc60458cdb3425fbce893ecf5a",
        "execution_id": "sha256:35620e0d9c69534f169868f681a90484cdc604eaf13530eb29fe1ad6e035e7c6",
        "spec_sha256": "sha256:600a21ac4424bfca5c62caac5415358c812b89684a0d2df510993048a1ac722b",
        "run_id": "chris-q38-ac-g15-r004-a1-35620e0d",
    },
    "glm-5.3": {
        "cell_id": "sha256:905051f141077d3aa5086c1f5dc6ad015d5ee6173a1ea7515025805cf9a24b41",
        "execution_id": "sha256:db08455e87afd8d07ab36bfac0a30c6d4896341da48679ba9958a4183c78de1a",
        "spec_sha256": "sha256:e5e55a4b3912fdd6e163b251d30597f6ea4eef0ac471378c2343fb009502b64f",
        "run_id": "chris-glm53-ac-g15-r013-a1-db08455e",
    },
}
PROHIBITED_CONTENT_KEYS = {
    "flag",
    "flags",
    "prompt",
    "prompts",
    "reward",
    "rewards",
    "score",
    "scores",
    "solution",
    "solutions",
    "tool_content",
    "trace",
    "traces",
    "transcript",
    "transcripts",
}
STATES = (
    "accepted",
    "active",
    "retryable_infra_failed",
    "blocked_nonrepeatable",
    "unstarted",
)
BULK_ACCEPTED_FIELDS = {
    "schema_version",
    "accepted",
    "credited",
    "retry_allowed",
    "controller",
    "cell_id",
    "execution_id",
    "run_id",
    "selection_rank",
    "attempt",
    "task_key",
    "task_version_id",
    "session_id",
    "verifier_execution_id",
    "agent_exit_code",
    "agent_process_exit_success",
    "config_sha256",
    "claim_sha256",
    "cleanup_completed",
    "session_ingest_completed",
    "scores_included",
    "prompts_or_traces_included",
    "receipt_sha256",
}
BULK_ACCEPTED_OPTIONAL_PROJECTION_FIELD = "authoritative_session_optional_fields"
BULK_CLAIM_FIELDS = {
    "schema_version",
    "plan_sha256",
    "controller",
    "cell_id",
    "execution_id",
    "execution_generation",
    "run_id",
    "selection_rank",
    "attempt",
    "job_uid",
    "pod_uid",
    "claimed_at_utc",
    "immutable",
    "automatic_retry",
    "model_call_started_when_claim_written",
    "scores_included",
    "prompts_or_traces_included",
    "receipt_sha256",
}
GENERATION7_TERMINAL_FIELDS = {
    "schema_version",
    "generation7_spec_sha256",
    "plan_sha256",
    "cell_id",
    "execution_id",
    "execution_generation",
    "generation_claim_receipt_sha256",
    "job_uid",
    "pod_uid",
    "terminal_at_utc",
    "scoring_release",
    "root_authorization",
    "result",
    "retry_allowed",
    "bulk_release_authorized",
    "scores_included",
    "prompts_or_traces_included",
    "receipt_sha256",
}
GENERATION7_CLAIM_FIELDS = {
    "schema_version",
    "generation7_spec_sha256",
    "plan_sha256",
    "cell_id",
    "execution_id",
    "execution_generation",
    "run_id",
    "job_uid",
    "pod_uid",
    "claimed_at_utc",
    "scoring_release",
    "root_authorization",
    "generation6_failure_receipt_sha256",
    "prior_claims_preserved",
    "immutable",
    "automatic_retry",
    "scores_included",
    "prompts_or_traces_included",
    "receipt_sha256",
}
GENERATION15_TERMINAL_FIELDS = {
    "schema_version",
    "accepted",
    "spec_sha256",
    "claim_sha256",
    "acceptance_receipt_sha256",
    "run_id",
    "session_id",
    "verifier_execution_id",
    "job_uid",
    "pod_uid",
    "scores_included",
    "prompts_or_traces_included",
    "receipt_sha256",
}
GENERATION15_CLAIM_FIELDS = {
    "schema_version",
    "spec_sha256",
    "cell_id",
    "execution_id",
    "execution_generation",
    "run_id",
    "job_uid",
    "pod_uid",
    "claimed_at_utc",
    "generation14_tombstone_receipt_sha256",
    "automatic_retry",
    "immutable",
    "prompts_or_traces_included",
    "scores_included",
    "receipt_sha256",
}
DEDICATED_QWEN_ACCEPTED_FIELDS = {
    "schema_version",
    "accepted",
    "credited",
    "retry_allowed",
    "serving_block",
    "cell_id",
    "execution_id",
    "run_id",
    "selection_rank",
    "attempt",
    "task_version_id",
    "session_id",
    "verifier_execution_id",
    "agent_exit_code",
    "agent_process_exit_success",
    "claim_sha256",
    "config_sha256",
    "cleanup_completed",
    "session_ingest_completed",
    "scores_included",
    "prompts_or_traces_included",
    "receipt_sha256",
}


class LedgerError(RuntimeError):
    """An input cannot be reconciled without risking duplicate work."""


@dataclass(frozen=True)
class Evidence:
    state: str
    cell_id: str
    execution_id: str
    execution_generation: int
    receipt_sha256: str
    path: Path


@dataclass(frozen=True)
class Authority:
    repo_root: Path
    cells: dict[str, dict[str, Any]]
    bulk_items: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]]
    qwen_generation16_items: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]]
    dedicated_qwen_items: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]]
    generation7: dict[tuple[str, str], dict[str, Any]]
    generation15: dict[tuple[str, str], dict[str, Any]]


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise LedgerError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_sensitive_content(value: Any) -> None:
    if isinstance(value, dict):
        forbidden = PROHIBITED_CONTENT_KEYS.intersection(value)
        if forbidden:
            raise LedgerError(
                "receipt contains prohibited content fields: " + ", ".join(sorted(forbidden))
            )
        for child in value.values():
            _reject_sensitive_content(child)
    elif isinstance(value, list):
        for child in value:
            _reject_sensitive_content(child)


def load_receipt(path: Path) -> dict[str, Any]:
    """Load one explicit score-blind receipt without following a symlink."""
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise LedgerError(f"receipt is absent: {path}") from exc
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise LedgerError(f"receipt is not a regular non-symlink file: {path}")
    if metadata.st_size > 1_000_000:
        raise LedgerError(f"receipt exceeds the 1 MB score-blind limit: {path}")
    try:
        value = json.loads(path.read_text(), object_pairs_hook=_strict_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LedgerError(f"receipt is not strict JSON: {path}") from exc
    if not isinstance(value, dict):
        raise LedgerError(f"receipt root is not an object: {path}")
    _reject_sensitive_content(value)
    if value.get("receipt_sha256") != self_hosted.digest_without(value, "receipt_sha256"):
        raise LedgerError(f"receipt digest mismatch: {path}")
    return value


def _build_authority(repo_root: Path, campaign_path: Path) -> Authority:
    campaign = exact.read_object(campaign_path)
    universe = exact.build_universe(campaign, repo_root)
    cells = {row["cell_id"]: row for row in universe["cells"]}

    bulk_items: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]] = {}
    for plan in bulk.validate_all(repo_root).values():
        if plan.get("treatment") != exact.EXPECTED_TREATMENT:
            raise LedgerError("bulk plan treatment drifted from the exact universe")
        for item in plan["attempts"]:
            key = (item["cell_id"], item["execution_id"])
            if key in bulk_items:
                raise LedgerError("bulk execution authority is duplicated")
            bulk_items[key] = (plan, item)

    qwen_generation16_items: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]] = {}
    for plan in qwen_generation16.validate_all(repo_root).values():
        if plan.get("treatment") != exact.EXPECTED_TREATMENT:
            raise LedgerError("Qwen generation-16 plan treatment drifted from the exact universe")
        for item in plan["attempts"]:
            key = (item["cell_id"], item["execution_id"])
            if key in qwen_generation16_items:
                raise LedgerError("Qwen generation-16 execution authority is duplicated")
            qwen_generation16_items[key] = (plan, item)

    dedicated_plan = qwen_dedicated.build_plan(repo_root)
    dedicated_item = dedicated_plan["item"]
    dedicated_key = (dedicated_item["cell_id"], dedicated_item["execution_id"])
    if dedicated_plan.get("treatment") != exact.EXPECTED_TREATMENT:
        raise LedgerError("dedicated Qwen plan treatment drifted from the exact universe")
    dedicated_qwen_items = {dedicated_key: (dedicated_plan, dedicated_item)}

    generation7_items = _validated_fixed_bindings(cells, GENERATION7_BINDINGS, 7)
    generation15_items = _validated_fixed_bindings(cells, GENERATION15_BINDINGS, 15)
    return Authority(
        repo_root=repo_root,
        cells=cells,
        bulk_items=bulk_items,
        qwen_generation16_items=qwen_generation16_items,
        dedicated_qwen_items=dedicated_qwen_items,
        generation7=generation7_items,
        generation15=generation15_items,
    )


def _validated_fixed_bindings(
    cells: dict[str, dict[str, Any]],
    bindings: dict[str, dict[str, str]],
    generation: int,
) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for model, binding in bindings.items():
        cell = cells.get(binding["cell_id"])
        if (
            cell is None
            or cell["model"] != model
            or exact.execution_for(cell["cell_id"], generation)["execution_id"]
            != binding["execution_id"]
        ):
            raise LedgerError(f"generation-{generation} immutable binding drifted")
        result[(binding["cell_id"], binding["execution_id"])] = {
            **binding,
            "model": model,
            "execution_generation": generation,
        }
    return result


def _require_cell_execution(
    authority: Authority,
    cell_id: Any,
    execution_id: Any,
    generation: Any,
    path: Path,
) -> tuple[dict[str, Any], int]:
    if not isinstance(cell_id, str) or not isinstance(execution_id, str):
        raise LedgerError(f"receipt omits exact cell/execution identity: {path}")
    cell = authority.cells.get(cell_id)
    if cell is None:
        raise LedgerError(f"receipt cell is outside the exact 800-cell universe: {path}")
    if not isinstance(generation, int) or isinstance(generation, bool) or generation < 1:
        raise LedgerError(f"receipt execution generation is invalid: {path}")
    if exact.execution_for(cell_id, generation)["execution_id"] != execution_id:
        raise LedgerError(f"receipt execution identity contradicts its cell/generation: {path}")
    return cell, generation


def _require_uuid(value: Any, label: str, path: Path) -> None:
    try:
        exact._uuid(value, label)
    except ValueError as exc:
        raise LedgerError(f"{label} is invalid: {path}") from exc


def _require_sha256(value: Any, label: str, path: Path) -> None:
    try:
        exact._require_sha256(value, label)
    except ValueError as exc:
        raise LedgerError(f"{label} is invalid: {path}") from exc


def _require_exact_fields(
    value: dict[str, Any], expected: set[str], path: Path, *, optional: set[str] | None = None
) -> None:
    actual = set(value)
    if not expected <= actual or actual - expected != (actual & (optional or set())):
        raise LedgerError(f"receipt fields drifted from its score-blind schema: {path}")


def _accepted_bulk(value: dict[str, Any], path: Path, authority: Authority) -> Evidence:
    _require_exact_fields(
        value,
        BULK_ACCEPTED_FIELDS,
        path,
        optional={BULK_ACCEPTED_OPTIONAL_PROJECTION_FIELD},
    )
    key = (value.get("cell_id"), value.get("execution_id"))
    pair = authority.qwen_generation16_items.get(key) or authority.bulk_items.get(key)
    if pair is None:
        raise LedgerError(f"bulk acceptance is absent from the exact frozen plans: {path}")
    plan, item = pair
    if value.get("controller") != plan["controller"]:
        raise LedgerError(f"bulk acceptance controller contradicts exact authority: {path}")
    generation = item["execution_generation"]
    cell, _ = _require_cell_execution(authority, *key, generation, path)
    try:
        bulk_runtime._validate_existing_receipt(value, item, plan["controller"], "accepted")
    except RuntimeError as exc:
        raise LedgerError(f"bulk acceptance identity binding is not authoritative: {path}") from exc
    expected = {
        "controller": plan["controller"],
        "cell_id": cell["cell_id"],
        "execution_id": item["execution_id"],
        "run_id": item["run_id"],
        "selection_rank": cell["selection_rank"],
        "attempt": cell["attempt"],
        "task_key": cell["task_key"],
        "task_version_id": cell["task_version_id"],
    }
    if any(value.get(field) != expected_value for field, expected_value in expected.items()):
        raise LedgerError(f"bulk acceptance identity or treatment binding drifted: {path}")
    if any(
        (
            value.get("accepted") is not True,
            value.get("credited") is not True,
            value.get("retry_allowed") is not False,
            value.get("cleanup_completed") is not True,
            value.get("session_ingest_completed") is not True,
            value.get("scores_included") is not False,
            value.get("prompts_or_traces_included") is not False,
            type(value.get("agent_exit_code")) is not int,
            value.get("agent_process_exit_success") is not (value.get("agent_exit_code") == 0),
        )
    ):
        raise LedgerError(f"bulk acceptance is not an authoritative valid outcome: {path}")
    projection = value.get(BULK_ACCEPTED_OPTIONAL_PROJECTION_FIELD)
    if projection is not None and (
        not isinstance(projection, dict)
        or set(projection) != {"model", "task_version_id", "run_id", "execution_id", "cell_id"}
        or any(status not in {"matched", "omitted"} for status in projection.values())
    ):
        raise LedgerError(f"bulk optional session projection evidence drifted: {path}")
    _require_uuid(value.get("session_id"), "session id", path)
    _require_uuid(value.get("verifier_execution_id"), "verifier execution id", path)
    return Evidence(
        "accepted",
        cell["cell_id"],
        item["execution_id"],
        generation,
        value["receipt_sha256"],
        path,
    )


def _accepted_generation7(value: dict[str, Any], path: Path, authority: Authority) -> Evidence:
    _require_exact_fields(value, GENERATION7_TERMINAL_FIELDS, path)
    key = (value.get("cell_id"), value.get("execution_id"))
    binding = authority.generation7.get(key)
    if binding is None:
        raise LedgerError(f"generation-7 acceptance is absent from exact authority: {path}")
    cell, generation = _require_cell_execution(
        authority, *key, value.get("execution_generation"), path
    )
    result = value.get("result")
    if not isinstance(result, dict):
        raise LedgerError(f"generation-7 acceptance result is absent: {path}")
    common = {"accepted", "quarantined", "claim_sha256", "attempt_config_sha256"}
    accepted_only = {
        "acceptance_receipt_sha256",
        "session_id",
        "verifier_execution_id",
        "session_ingest_completed",
        "cleanup_completed",
    }
    if any(
        (
            value.get("generation7_spec_sha256") != binding["spec_sha256"],
            value.get("plan_sha256") != binding["plan_sha256"],
            value.get("retry_allowed") is not False,
            value.get("scores_included") is not False,
            value.get("prompts_or_traces_included") is not False,
            set(result) != common | accepted_only,
            result.get("accepted") is not True,
            result.get("quarantined") is not False,
            result.get("session_ingest_completed") is not True,
            result.get("cleanup_completed") is not True,
        )
    ):
        raise LedgerError(f"generation-7 acceptance identity or treatment drifted: {path}")
    for field in ("claim_sha256", "attempt_config_sha256", "acceptance_receipt_sha256"):
        _require_sha256(result.get(field), field, path)
    _require_sha256(
        value.get("generation_claim_receipt_sha256"),
        "generation claim receipt sha256",
        path,
    )
    _require_uuid(result.get("session_id"), "session id", path)
    _require_uuid(result.get("verifier_execution_id"), "verifier execution id", path)
    _require_uuid(value.get("job_uid"), "job uid", path)
    _require_uuid(value.get("pod_uid"), "pod uid", path)
    return Evidence("accepted", cell["cell_id"], key[1], generation, value["receipt_sha256"], path)


def _accepted_generation15(value: dict[str, Any], path: Path, authority: Authority) -> Evidence:
    _require_exact_fields(value, GENERATION15_TERMINAL_FIELDS, path)
    run_id = value.get("run_id")
    if not isinstance(run_id, str):
        raise LedgerError(f"generation-15 acceptance omits its run id: {path}")
    candidates = [
        (key, spec)
        for key, spec in authority.generation15.items()
        if spec["spec_sha256"] == value.get("spec_sha256") and spec["run_id"] == value.get("run_id")
    ]
    if len(candidates) != 1:
        raise LedgerError(f"generation-15 acceptance lacks unique exact spec binding: {path}")
    key, spec = candidates[0]
    cell, generation = _require_cell_execution(authority, *key, spec["execution_generation"], path)
    if any(
        (
            value.get("accepted") is not True,
            value.get("claim_sha256") is None,
            value.get("acceptance_receipt_sha256") is None,
            value.get("scores_included") is not False,
            value.get("prompts_or_traces_included") is not False,
        )
    ):
        raise LedgerError(f"generation-15 acceptance is not authoritative: {path}")
    for field in ("claim_sha256", "acceptance_receipt_sha256"):
        _require_sha256(value.get(field), field, path)
    _require_uuid(value.get("session_id"), "session id", path)
    _require_uuid(value.get("verifier_execution_id"), "verifier execution id", path)
    _require_uuid(value.get("job_uid"), "job uid", path)
    _require_uuid(value.get("pod_uid"), "pod uid", path)
    return Evidence("accepted", cell["cell_id"], key[1], generation, value["receipt_sha256"], path)


def _accepted_dedicated_qwen(value: dict[str, Any], path: Path, authority: Authority) -> Evidence:
    _require_exact_fields(value, DEDICATED_QWEN_ACCEPTED_FIELDS, path)
    key = (value.get("cell_id"), value.get("execution_id"))
    pair = authority.dedicated_qwen_items.get(key)
    if pair is None:
        raise LedgerError(f"dedicated Qwen acceptance lacks exact plan authority: {path}")
    plan, item = pair
    cell, generation = _require_cell_execution(authority, *key, item["execution_generation"], path)
    config = plan["config"]
    expected = {
        "serving_block": config["serving"]["serving_block"],
        "cell_id": item["cell_id"],
        "execution_id": item["execution_id"],
        "run_id": item["run_id"],
        "selection_rank": item["selection_rank"],
        "attempt": item["attempt"],
        "task_version_id": item["task_version_id"],
        "config_sha256": config["config_sha256"],
    }
    if any(value.get(field) != expected_value for field, expected_value in expected.items()):
        raise LedgerError(f"dedicated Qwen acceptance identity or treatment drifted: {path}")
    if any(
        (
            value.get("accepted") is not True,
            value.get("credited") is not True,
            value.get("retry_allowed") is not False,
            value.get("cleanup_completed") is not True,
            value.get("session_ingest_completed") is not True,
            value.get("scores_included") is not False,
            value.get("prompts_or_traces_included") is not False,
            type(value.get("agent_exit_code")) is not int,
            value.get("agent_process_exit_success") is not (value.get("agent_exit_code") == 0),
        )
    ):
        raise LedgerError(f"dedicated Qwen acceptance is not authoritative: {path}")
    _require_sha256(value.get("claim_sha256"), "claim sha256", path)
    _require_uuid(value.get("session_id"), "session id", path)
    _require_uuid(value.get("verifier_execution_id"), "verifier execution id", path)
    return Evidence(
        "accepted",
        cell["cell_id"],
        item["execution_id"],
        generation,
        value["receipt_sha256"],
        path,
    )


def accepted_evidence(path: Path, authority: Authority) -> Evidence:
    value = load_receipt(path)
    schema = value.get("schema_version")
    if schema not in ACCEPTED_SCHEMAS:
        raise LedgerError(f"unsupported acceptance receipt schema at {path}: {schema!r}")
    if schema == "fleet-exact-pass4-bulk-cell-accepted-v3":
        return _accepted_bulk(value, path, authority)
    if schema == DEDICATED_QWEN_ACCEPTED_SCHEMA:
        return _accepted_dedicated_qwen(value, path, authority)
    if schema == GENERATION7_TERMINAL_SCHEMA:
        return _accepted_generation7(value, path, authority)
    return _accepted_generation15(value, path, authority)


def _claim_bulk(value: dict[str, Any], path: Path, authority: Authority) -> Evidence:
    _require_exact_fields(value, BULK_CLAIM_FIELDS, path)
    key = (value.get("cell_id"), value.get("execution_id"))
    pair = authority.qwen_generation16_items.get(key) or authority.bulk_items.get(key)
    if pair is None:
        raise LedgerError(f"bulk claim is absent from exact frozen plans: {path}")
    plan, item = pair
    if value.get("controller") != plan["controller"]:
        raise LedgerError(f"bulk claim controller contradicts exact authority: {path}")
    cell, generation = _require_cell_execution(
        authority, *key, value.get("execution_generation"), path
    )
    _require_canonical_claim_filename(path, value)
    expected = {
        "plan_sha256": plan["plan_sha256"],
        "controller": plan["controller"],
        "cell_id": item["cell_id"],
        "execution_id": item["execution_id"],
        "execution_generation": item["execution_generation"],
        "run_id": item["run_id"],
        "selection_rank": item["selection_rank"],
        "attempt": item["attempt"],
    }
    if any(value.get(field) != expected_value for field, expected_value in expected.items()):
        raise LedgerError(f"bulk claim identity or treatment drifted: {path}")
    if any(
        (
            value.get("immutable") is not True,
            value.get("automatic_retry") is not False,
            value.get("model_call_started_when_claim_written") is not False,
            value.get("scores_included") is not False,
            value.get("prompts_or_traces_included") is not False,
            bulk_runtime.ISO_UTC_RE.fullmatch(str(value.get("claimed_at_utc"))) is None,
        )
    ):
        raise LedgerError(f"bulk claim is not immutable score-blind authority: {path}")
    _require_uuid(value.get("job_uid"), "job uid", path)
    _require_uuid(value.get("pod_uid"), "pod uid", path)
    return Evidence("", cell["cell_id"], key[1], generation, value["receipt_sha256"], path)


def _claim_dedicated_qwen(value: dict[str, Any], path: Path, authority: Authority) -> Evidence:
    _require_exact_fields(value, BULK_CLAIM_FIELDS, path)
    key = (value.get("cell_id"), value.get("execution_id"))
    pair = authority.dedicated_qwen_items.get(key)
    if pair is None:
        raise LedgerError(f"dedicated Qwen claim lacks exact plan authority: {path}")
    plan, item = pair
    cell, generation = _require_cell_execution(
        authority, *key, value.get("execution_generation"), path
    )
    _require_canonical_claim_filename(path, value)
    expected = {
        "plan_sha256": plan["plan_sha256"],
        "controller": plan["controller"],
        "cell_id": item["cell_id"],
        "execution_id": item["execution_id"],
        "execution_generation": item["execution_generation"],
        "run_id": item["run_id"],
        "selection_rank": item["selection_rank"],
        "attempt": item["attempt"],
    }
    if any(value.get(field) != expected_value for field, expected_value in expected.items()):
        raise LedgerError(f"dedicated Qwen claim identity or treatment drifted: {path}")
    if any(
        (
            value.get("immutable") is not True,
            value.get("automatic_retry") is not False,
            value.get("model_call_started_when_claim_written") is not False,
            value.get("scores_included") is not False,
            value.get("prompts_or_traces_included") is not False,
            bulk_runtime.ISO_UTC_RE.fullmatch(str(value.get("claimed_at_utc"))) is None,
        )
    ):
        raise LedgerError(f"dedicated Qwen claim is not immutable score-blind authority: {path}")
    _require_uuid(value.get("job_uid"), "job uid", path)
    _require_uuid(value.get("pod_uid"), "pod uid", path)
    return Evidence("", cell["cell_id"], key[1], generation, value["receipt_sha256"], path)


def _require_canonical_claim_filename(path: Path, value: dict[str, Any]) -> None:
    expected_name = str(value.get("execution_id", "")).removeprefix("sha256:") + ".json"
    if path.name != expected_name:
        raise LedgerError(f"bulk claim filename does not bind its execution id: {path}")


def _claim_generation7(value: dict[str, Any], path: Path, authority: Authority) -> Evidence:
    _require_exact_fields(value, GENERATION7_CLAIM_FIELDS, path)
    key = (value.get("cell_id"), value.get("execution_id"))
    binding = authority.generation7.get(key)
    if binding is None:
        raise LedgerError(f"generation-7 claim is absent from exact authority: {path}")
    cell, generation = _require_cell_execution(
        authority, *key, value.get("execution_generation"), path
    )
    if any(
        (
            value.get("generation7_spec_sha256") != binding["spec_sha256"],
            value.get("plan_sha256") != binding["plan_sha256"],
            value.get("run_id") != binding["run_id"],
            value.get("immutable") is not True,
            value.get("automatic_retry") is not False,
            value.get("scores_included") is not False,
            value.get("prompts_or_traces_included") is not False,
        )
    ):
        raise LedgerError(f"generation-7 claim identity or treatment drifted: {path}")
    _require_uuid(value.get("job_uid"), "job uid", path)
    _require_uuid(value.get("pod_uid"), "pod uid", path)
    return Evidence("", cell["cell_id"], key[1], generation, value["receipt_sha256"], path)


def _claim_generation15(value: dict[str, Any], path: Path, authority: Authority) -> Evidence:
    _require_exact_fields(value, GENERATION15_CLAIM_FIELDS, path)
    key = (value.get("cell_id"), value.get("execution_id"))
    spec = authority.generation15.get(key)
    if spec is None:
        raise LedgerError(f"generation-15 claim is absent from exact authority: {path}")
    cell, generation = _require_cell_execution(
        authority, *key, value.get("execution_generation"), path
    )
    if any(
        (
            value.get("spec_sha256") != spec["spec_sha256"],
            value.get("run_id") != spec["run_id"],
            value.get("immutable") is not True,
            value.get("automatic_retry") is not False,
            value.get("scores_included") is not False,
            value.get("prompts_or_traces_included") is not False,
        )
    ):
        raise LedgerError(f"generation-15 claim identity or treatment drifted: {path}")
    _require_uuid(value.get("job_uid"), "job uid", path)
    _require_uuid(value.get("pod_uid"), "pod uid", path)
    return Evidence("", cell["cell_id"], key[1], generation, value["receipt_sha256"], path)


def claim_evidence(path: Path, authority: Authority, *, active: bool) -> Evidence:
    value = load_receipt(path)
    schema = value.get("schema_version")
    if schema not in CLAIM_SCHEMAS:
        raise LedgerError(f"unsupported claim receipt schema at {path}: {schema!r}")
    if schema == bulk_runtime.CLAIM_SCHEMA:
        if value.get("controller") == qwen_dedicated.CONTROLLER:
            evidence = _claim_dedicated_qwen(value, path, authority)
        else:
            evidence = _claim_bulk(value, path, authority)
    elif schema == GENERATION7_CLAIM_SCHEMA:
        evidence = _claim_generation7(value, path, authority)
    else:
        evidence = _claim_generation15(value, path, authority)
    return Evidence(
        "active" if active else "blocked_nonrepeatable",
        evidence.cell_id,
        evidence.execution_id,
        evidence.execution_generation,
        evidence.receipt_sha256,
        evidence.path,
    )


def tombstone_evidence(path: Path, authority: Authority) -> Evidence:
    value = load_receipt(path)
    cell_id = value.get("cell_id")
    cell = authority.cells.get(cell_id)
    if cell is None:
        raise LedgerError(f"tombstone cell is outside the exact 800-cell universe: {path}")
    try:
        exact.validate_tombstone(value, cell)
    except (TypeError, ValueError) as exc:
        raise LedgerError(f"tombstone is not retry-safe: {path}") from exc
    return Evidence(
        "retryable_infra_failed",
        cell_id,
        value["execution_id"],
        value["execution_generation"],
        value["receipt_sha256"],
        path,
    )


def _mount_maps(values: Sequence[str]) -> list[tuple[Path, Path]]:
    mappings: list[tuple[Path, Path]] = []
    for value in values:
        source_text, separator, observer_text = value.partition("=")
        source, observer = Path(source_text), Path(observer_text)
        if not separator or not source.is_absolute() or not observer.is_absolute():
            raise LedgerError("mount maps must be absolute PRODUCER=OBSERVER paths")
        if source in {item[0] for item in mappings}:
            raise LedgerError(f"producer mount prefix is mapped more than once: {source}")
        mappings.append((source, observer))
    return sorted(mappings, key=lambda item: len(item[0].parts), reverse=True)


def _observer_path(path: Path, mappings: Sequence[tuple[Path, Path]]) -> Path:
    if not path.is_absolute():
        return path
    for source, observer in mappings:
        try:
            suffix = path.relative_to(source)
        except ValueError:
            continue
        return observer / suffix
    return path


def _paths(
    explicit: Sequence[Path],
    roots: Sequence[Path],
    pattern: str,
    mappings: Sequence[tuple[Path, Path]],
) -> list[Path]:
    paths = [_observer_path(path, mappings) for path in explicit]
    for supplied_root in roots:
        root = _observer_path(supplied_root, mappings)
        if root.is_symlink() or not root.is_dir():
            raise LedgerError(f"evidence root is not a regular directory: {root}")
        paths.extend(sorted(root.rglob(pattern)))
    normalized = [path.absolute() for path in paths]
    if len(normalized) != len(set(normalized)):
        raise LedgerError("the same evidence path was supplied more than once")
    return normalized


def reconcile(
    authority: Authority,
    *,
    accepted: Iterable[Evidence],
    active_claims: Iterable[Evidence],
    blocked_claims: Iterable[Evidence],
    tombstones: Iterable[Evidence],
) -> dict[str, Any]:
    by_cell: dict[str, list[Evidence]] = {cell_id: [] for cell_id in authority.cells}
    for row in [*accepted, *active_claims, *blocked_claims, *tombstones]:
        by_cell[row.cell_id].append(row)

    rows: list[dict[str, Any]] = []
    counts = {model: Counter({state: 0 for state in STATES}) for model in exact.EXPECTED_MODELS}
    seen_receipts: set[str] = set()
    for cell_id, evidence in by_cell.items():
        cell = authority.cells[cell_id]
        for item in evidence:
            if item.receipt_sha256 in seen_receipts:
                raise LedgerError("one receipt was presented for more than one ledger entry")
            seen_receipts.add(item.receipt_sha256)

        accepted_rows = [item for item in evidence if item.state == "accepted"]
        active_rows = [item for item in evidence if item.state == "active"]
        blocked_rows = [item for item in evidence if item.state == "blocked_nonrepeatable"]
        tombstone_rows = [item for item in evidence if item.state == "retryable_infra_failed"]
        if len(accepted_rows) > 1:
            raise LedgerError(f"cell has duplicate accepted outcomes: {cell_id}")
        if len(active_rows) > 1 or len(blocked_rows) > 1:
            raise LedgerError(f"cell has duplicate claim classifications: {cell_id}")
        if active_rows and blocked_rows:
            raise LedgerError(f"cell is simultaneously active and blocked: {cell_id}")
        if accepted_rows and (active_rows or blocked_rows):
            raise LedgerError(f"accepted cell also has a live/nonrepeatable claim input: {cell_id}")

        ordered_tombstones = sorted(tombstone_rows, key=lambda item: item.execution_generation)
        if ordered_tombstones:
            if [item.execution_generation for item in ordered_tombstones] != list(
                range(1, len(ordered_tombstones) + 1)
            ):
                raise LedgerError(
                    f"retry-safe tombstones are not contiguous from generation 1: {cell_id}"
                )
            if len({item.execution_id for item in ordered_tombstones}) != len(ordered_tombstones):
                raise LedgerError(f"cell has duplicate tombstone executions: {cell_id}")
        terminal = accepted_rows or active_rows or blocked_rows
        if terminal and ordered_tombstones:
            current = terminal[0]
            if current.execution_generation <= ordered_tombstones[-1].execution_generation:
                raise LedgerError(f"cell terminal/claim execution overlaps a tombstone: {cell_id}")

        if accepted_rows:
            state = "accepted"
        elif active_rows:
            state = "active"
        elif blocked_rows:
            state = "blocked_nonrepeatable"
        elif ordered_tombstones:
            state = "retryable_infra_failed"
        else:
            state = "unstarted"
        counts[cell["model"]][state] += 1
        rows.append(
            {
                "model": cell["model"],
                "selection_rank": cell["selection_rank"],
                "attempt": cell["attempt"],
                "cell_id": cell_id,
                "task_version_id": cell["task_version_id"],
                "state": state,
                "latest_execution_generation": max(
                    (item.execution_generation for item in evidence), default=0
                ),
            }
        )

    summary: dict[str, dict[str, int]] = {}
    for model, counter in counts.items():
        summary[model] = {state: counter[state] for state in STATES}
        if sum(summary[model].values()) != 400:
            raise LedgerError(f"{model} ledger denominator is not exactly 400")
    return {
        "schema_version": "fleet-exact-pass4-score-blind-ledger-v1",
        "campaign_id": exact.EXPECTED_CAMPAIGN_ID,
        "universe_cell_count": 800,
        "models": summary,
        "cells": sorted(
            rows,
            key=lambda row: (row["model"], row["selection_rank"], row["attempt"]),
        ),
        "privacy": {
            "prompts_read": False,
            "traces_read": False,
            "flags_read": False,
            "scores_read": False,
        },
    }


def render_table(ledger: dict[str, Any]) -> str:
    headers = (
        "model",
        "target",
        "accepted",
        "active",
        "retryable infra-failed",
        "blocked/nonrepeatable",
        "unstarted",
    )
    table_rows: list[tuple[str, ...]] = []
    total = Counter({state: 0 for state in STATES})
    for model in exact.EXPECTED_MODELS:
        row = ledger["models"][model]
        total.update(row)
        table_rows.append(
            (
                model,
                "400",
                str(row["accepted"]),
                str(row["active"]),
                str(row["retryable_infra_failed"]),
                str(row["blocked_nonrepeatable"]),
                str(row["unstarted"]),
            )
        )
    table_rows.append(
        (
            "TOTAL",
            "800",
            str(total["accepted"]),
            str(total["active"]),
            str(total["retryable_infra_failed"]),
            str(total["blocked_nonrepeatable"]),
            str(total["unstarted"]),
        )
    )
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in table_rows))
        for index in range(len(headers))
    ]
    line = "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    divider = "  ".join("-" * width for width in widths)
    body = [
        "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in table_rows
    ]
    return "\n".join([line, divider, *body])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--accepted", type=Path, action="append", default=[])
    parser.add_argument("--accepted-root", type=Path, action="append", default=[])
    parser.add_argument("--active-claim", type=Path, action="append", default=[])
    parser.add_argument("--active-claim-root", type=Path, action="append", default=[])
    parser.add_argument("--nonrepeatable-claim", type=Path, action="append", default=[])
    parser.add_argument("--nonrepeatable-claim-root", type=Path, action="append", default=[])
    parser.add_argument("--tombstone", type=Path, action="append", default=[])
    parser.add_argument("--tombstone-root", type=Path, action="append", default=[])
    parser.add_argument(
        "--mount-map",
        action="append",
        default=[],
        metavar="PRODUCER=OBSERVER",
        help="map a producer path such as /mnt/sfs/jobs to its read-only observer mount",
    )
    parser.add_argument("--json", action="store_true", help="emit the full cell ledger as JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        repo_root = args.repo_root.resolve(strict=True)
        campaign = (args.campaign or repo_root / DEFAULT_CAMPAIGN).resolve(strict=True)
        authority = _build_authority(repo_root, campaign)
        mappings = _mount_maps(args.mount_map)
        accepted_paths = _paths(args.accepted, args.accepted_root, "ACCEPTED.json", mappings)
        active_paths = _paths(args.active_claim, args.active_claim_root, "*.json", mappings)
        blocked_paths = _paths(
            args.nonrepeatable_claim,
            args.nonrepeatable_claim_root,
            "*.json",
            mappings,
        )
        tombstone_paths = _paths(args.tombstone, args.tombstone_root, "*.json", mappings)
        ledger = reconcile(
            authority,
            accepted=(accepted_evidence(path, authority) for path in accepted_paths),
            active_claims=(claim_evidence(path, authority, active=True) for path in active_paths),
            blocked_claims=(
                claim_evidence(path, authority, active=False) for path in blocked_paths
            ),
            tombstones=(tombstone_evidence(path, authority) for path in tombstone_paths),
        )
    except (LedgerError, OSError, ValueError) as exc:
        print(f"ledger error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(ledger, sort_keys=True, indent=2))
    else:
        print(render_table(ledger))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
