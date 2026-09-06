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
from evals.fleet import hosted_glm_exact_bulk_v1 as hosted_glm_bulk
from evals.fleet import hosted_glm_s2_c2_canary_v1 as hosted_glm_c2
from evals.fleet import qwen38_dedicated_dp8_canary_v1 as qwen_dedicated_dp8
from evals.fleet import qwen38_dedicated_rank2_v3 as qwen_dedicated_v3
from evals.fleet import qwen38_dedicated_rank3_a2_g22_v1 as qwen_dedicated_rank3_g22
from evals.fleet import qwen38_dedicated_rank3_v3 as qwen_dedicated_rank3
from evals.fleet import qwen38_dedicated_rank97_bundle_v1 as qwen_dedicated_rank97
from evals.fleet import qwen38_dedicated_scored_canary_v1 as qwen_dedicated
from evals.fleet import qwen_bulk_generation16 as qwen_generation16
from evals.fleet import qwen_hosted_generation18 as qwen_generation18
from evals.fleet import qwen_hosted_generation19_v4 as qwen_generation19_v4
from evals.fleet import self_hosted

DEFAULT_CAMPAIGN = Path("evals/fleet/configs/q38-glm53-exact-easiest100-pass4-campaign-v1.json")
EVIDENCE_MANIFEST_SCHEMA = "fleet-exact-pass4-ledger-evidence-manifest-v1"
EVIDENCE_MANIFEST_V2_SCHEMA = "fleet-exact-pass4-ledger-evidence-manifest-v2"
HOSTED_GLM_INVENTORY_PATH = Path(
    "/mnt/sfs/jobs/chris-cyber-exact100-pass4-inventory-v2/TERMINAL.json"
)
HOSTED_GLM_INVENTORY_FILE_SHA256 = (
    "sha256:8bc50021e520a3afbefe8e4062fdb4e1a0a6afa66a5dab53c92a78296f561890"
)
HOSTED_GLM_INVENTORY_RECEIPT_SHA256 = (
    "sha256:bf7fae379833aca5c914e1a2761ab608a805ee6f2424c714c3ef060e318fb9d3"
)
EVIDENCE_MANIFEST_KINDS = {
    "accepted",
    "active_claim",
    "nonrepeatable_claim",
    "operational_incident",
    "tombstone",
}

GENERATION7_TERMINAL_SCHEMA = "fleet-opencode-autocontinue-generation7-terminal-v1"
LEGACY_GLM_GENERATION7_ACCEPTED_SCHEMA = "fleet-hosted-opencode-attempt-accepted-v1"
GENERATION7_CLAIM_SCHEMA = "fleet-statistical-cell-execution-claim-v9"
GENERATION15_TERMINAL_SCHEMA = "fleet-opencode-generation15-simple-terminal-v1"
GENERATION15_CLAIM_SCHEMA = "fleet-statistical-cell-execution-claim-v15"
GENERATION15_ACCEPTED_GATE_SCHEMA = "fleet-qwen38-generation15-accepted-gate-v1"
DEDICATED_QWEN_ACCEPTED_SCHEMA = "fleet-qwen38-dedicated-tp1-cell-accepted-v1"
DEDICATED_QWEN_VALIDATED_SCHEMA = "fleet-qwen38-dedicated-tp1-accepted-validated-v1"
DEDICATED_QWEN_VALIDATED_V2_SCHEMA = "fleet-qwen38-dedicated-tp1-accepted-validated-v2"
GLM_C2_ACCEPTED_VALIDATED_SCHEMA = "fleet-glm53-hosted-c2-accepted-validated-v1"
GLM_C2_RUNTIME_AUTHORITY_SCHEMA = "fleet-glm53-hosted-c2-runtime-authority-v1"
GLM_C2_RUNTIME_AUTHORITY_PATH = Path(
    "docs/evidence/qwen38-study/2026-09-05-glm53-hosted-c2-runtime-authority-v1.json"
)
SUPPLEMENTAL_RUNTIME_AUTHORITY_SCHEMA = (
    "fleet-exact-pass4-supplemental-runtime-authority-v5"
)
SUPPLEMENTAL_RUNTIME_AUTHORITY_PATH = Path(
    "docs/evidence/qwen38-study/"
    "2026-09-05-exact-pass4-supplemental-runtime-authority-v5.json"
)
DEDICATED_QWEN_ATTEMPT1_BINDING = {
    "plan_sha256": "sha256:5358ae8d0c81fd18d815f5289eabf771274101e099c799c49a85d0713687aa67",
    "controller": "qwen-dedicated-tp1-rank2-v1",
    "config_sha256": "sha256:df57b47039edab8dbfa610e728afd3d5e6e2b813dbb11abf5d73ff667e821d31",
    "serving_block": "dedicated-qwen-tp1-v1",
    "cell_id": "sha256:6351b9167d5846a2a30f0725f09f9918cbc749df963c5eba63dc1fc2a835b713",
    "execution_id": "sha256:d84a03547e45240e9702097b47d68ee122312987c5b18fcfa6ab7ee2ab738f0f",
    "execution_generation": 1,
    "run_id": "chris-cyber-q38-opencode11827-ded-tp1-r002-a1-v2",
    "selection_rank": 2,
    "attempt": 1,
    "task_version_id": "09a3fea6-f691-4841-9218-d04459041a1f",
    "claim_sha256": "sha256:be390c624629780fbd3ad723272c60387ebc89186206263423ab2c231aabb555",
    "source_terminal_receipt_sha256": (
        "sha256:d756b797be1c493fcc7ed630e75787a0dad9b14c592f93e0ca6183ab5ae707f2"
    ),
    "source_terminal_stale_claim_sha256": (
        "sha256:68fa8b63d14d26e8bb7f48f4bd103eafb5ac0165545c72e3d1765d7609aeb746"
    ),
    "source_terminal_actual_canonical_sha256": (
        "sha256:161f822c67b3dda33964f1883981dd0a6997cbe3f09aa3e1a8611aa94cb0be84"
    ),
    "accepted_receipt_sha256": (
        "sha256:1b045f8e5174ae607cbb0cdddfd5c29b02e9dada63172a1f899c526f849a743e"
    ),
    "acceptance_terminal_receipt_sha256": (
        "sha256:93f8bb0313ef7f34bab1e5e1e064efd5426bfcb6a4da5b72658e00b68a8fb5d9"
    ),
    "collector_job_uid": "c16b0001-df99-4c02-a210-1c7ab523ea11",
    "collector_pod_uid": "79c1623d-0330-42bf-bb2a-93dbfc8ed061",
    "validator_job_uid": "9368f7fe-dd7f-497f-86ba-f3ba4cf277d5",
    "validator_pod_uid": "9f407539-d4c1-4a51-91da-c57eaea73bd0",
    "session_id": "89a321f1-51dd-4ace-832b-6782385f6bf0",
    "verifier_execution_id": "9ed9b1d5-2f96-4f6c-aaa3-3f2e82c67adb",
    "validated_receipt_sha256": (
        "sha256:108903b167e40d8628776a6bed35ef7fd871b687dd8c81413fe66ca0f2e43212"
    ),
    "artifact_file_sha256": {
        "plan_file_sha256": (
            "sha256:ca86be8f389ae17e9d4d05ffe82323d591184bfd7630a75510140813526dcfb7"
        ),
        "result_file_sha256": (
            "sha256:58e2f0dc9165da00597f700268a2c40f723c4dca88ab168c7e161cda80f5c8b1"
        ),
        "reward_file_sha256": (
            "sha256:c56ef9d56c12bafeb6167271eceea99d93c1f5ba809ee189c7a58c18589ebc59"
        ),
        "session_ingest_file_sha256": (
            "sha256:e86244a9ce9f16978ac7a35aa5299cb2ba0cda9b5324a69bb4aca525d9ca1c69"
        ),
        "cleanup_file_sha256": (
            "sha256:a2b8b54bd9eeb559c978e31b0ae889923e39d411a468d1af205b16486bbd290c"
        ),
        "claim_file_sha256": (
            "sha256:ba7f1b19f6242b7fbe162c19dd6afbadcdf28f5de3e2e408ad641226ded27ee4"
        ),
    },
}
ACCEPTED_SCHEMAS = {
    "fleet-exact-pass4-bulk-cell-accepted-v3",
    GENERATION7_TERMINAL_SCHEMA,
    LEGACY_GLM_GENERATION7_ACCEPTED_SCHEMA,
    GENERATION15_TERMINAL_SCHEMA,
    DEDICATED_QWEN_ACCEPTED_SCHEMA,
    DEDICATED_QWEN_VALIDATED_SCHEMA,
    DEDICATED_QWEN_VALIDATED_V2_SCHEMA,
    GENERATION15_ACCEPTED_GATE_SCHEMA,
    GLM_C2_ACCEPTED_VALIDATED_SCHEMA,
}
LEGACY_GLM_GENERATION7_ACCEPTED_FIELDS = {
    "schema_version",
    "accepted",
    "credited",
    "retry_allowed",
    "run_id",
    "rank",
    "source_rank",
    "attempt",
    "task_key",
    "task_version_id",
    "session_id",
    "verifier_execution_id",
    "agent_exit_code",
    "config_sha256",
    "claim_sha256",
    "cleanup_completed",
    "session_ingest_completed",
    "scores_included",
    "prompts_or_traces_included",
    "receipt_sha256",
}
LEGACY_GLM_GENERATION7_SOURCE = Path(
    "docs/evidence/qwen38-study/2026-09-04-completed-exit1-gap-source-v1.json"
)
LEGACY_GLM_GENERATION7_SOURCE_SHA256 = (
    "sha256:30338f0561867bda8ab8b10cd2601b3a1ce0c60ad27d7fd55f9a7ad87d4b46a2"
)
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
GLM_C2_ACCEPTED_VALIDATED_FIELDS = {
    "schema_version",
    "status",
    "accepted",
    "credited",
    "retry_allowed",
    "controller",
    "cell_id",
    "execution_id",
    "execution_generation",
    "plan_sha256",
    "run_id",
    "selection_rank",
    "attempt",
    "task_key",
    "task_version_id",
    "config_sha256",
    "release_path",
    "release_receipt_sha256",
    "release_file_sha256",
    "release_observer_job_uid",
    "release_observer_pod_uid",
    "claim_sha256",
    "claim_file_sha256",
    "terminal_path",
    "terminal_receipt_sha256",
    "terminal_file_sha256",
    "accepted_path",
    "accepted_receipt_sha256",
    "accepted_file_sha256",
    "source_job_uid",
    "source_pod_uid",
    "session_id",
    "verifier_execution_id",
    "all_source_receipt_byte_digests_matched",
    "fresh_authoritative_session_reconciled",
    "scores_included",
    "prompts_or_traces_included",
    "credentials_included",
    "receipt_sha256",
}
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
GENERATION15_ACCEPTED_GATE_FIELDS = {
    "schema_version",
    "status",
    "model",
    "cell_id",
    "execution_id",
    "task_key",
    "task_version_id",
    "config_sha256",
    "inner_acceptance_receipt_sha256",
    "top_acceptance_receipt_sha256",
    "api_session",
    "job",
    "pod",
    "workload_uid",
    "cleanup_completed",
    "credentials_included",
    "prompts_traces_flags_or_scores_included",
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
DEDICATED_QWEN_PROJECTION_FIELDS = {
    "authoritative_session_task_key_matched",
    "authoritative_projection_omissions",
    "authoritative_projection_rule",
}
DEDICATED_QWEN_VALIDATED_FIELDS = {
    "schema_version",
    "status",
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
    "claim_sha256",
    "config_sha256",
    "authoritative_projection_omissions",
    "authoritative_projection_rule",
    "plan_file_sha256",
    "claim_file_sha256",
    "claim_receipt_sha256",
    "artifact_file_sha256",
    "all_artifact_byte_digests_matched",
    "source_terminal_receipt_sha256",
    "source_terminal_stale_claim_sha256",
    "source_terminal_actual_canonical_sha256",
    "accepted_receipt_sha256",
    "acceptance_terminal_receipt_sha256",
    "collector_job_uid",
    "collector_pod_uid",
    "validator_job_uid",
    "validator_pod_uid",
    "fleet_api_mutations",
    "fresh_authoritative_session_reconciled",
    "scores_included",
    "prompts_or_traces_included",
    "credentials_included",
    "receipt_sha256",
}
DEDICATED_QWEN_VALIDATED_V2_FIELDS = {
    "schema_version",
    "status",
    "accepted",
    "credited",
    "retry_allowed",
    "serving_block",
    "serving_parity_receipt_sha256",
    "cell_id",
    "execution_id",
    "run_id",
    "selection_rank",
    "attempt",
    "task_version_id",
    "session_id",
    "verifier_execution_id",
    "claim_sha256",
    "claim_receipt_sha256",
    "config_sha256",
    "plan_sha256",
    "authoritative_projection_omissions",
    "authoritative_projection_rule",
    "artifact_file_sha256",
    "all_artifact_byte_digests_matched",
    "accepted_receipt_sha256",
    "terminal_receipt_sha256",
    "source_job_uid",
    "source_pod_uid",
    "validator_job_uid",
    "validator_pod_uid",
    "fleet_api_mutations",
    "fresh_authoritative_session_reconciled",
    "scores_included",
    "prompts_or_traces_included",
    "credentials_included",
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
    qwen_generation18_items: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]]
    qwen_generation19_v4_items: dict[
        tuple[str, str], tuple[dict[str, Any], dict[str, Any]]
    ]
    hosted_glm_bulk_items: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]]
    hosted_glm_c2_items: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]]
    supplemental_bulk_items: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]]
    rollforward_precedence: dict[str, tuple[str, str, str, str]]
    dedicated_qwen_items: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]]
    dedicated_qwen_v3_items: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]]
    dedicated_qwen_rank3_items: dict[
        tuple[str, str], tuple[dict[str, Any], dict[str, Any]]
    ]
    dedicated_qwen_rank97_items: dict[
        tuple[str, str], tuple[dict[str, Any], dict[str, Any]]
    ]
    dedicated_qwen_dp8_items: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]]
    generation7: dict[tuple[str, str], dict[str, Any]]
    generation15: dict[tuple[str, str], dict[str, Any]]


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise LedgerError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_sensitive_content(value: Any, parent_key: str | None = None) -> None:
    if isinstance(value, dict):
        if parent_key == "artifact_file_sha256":
            if not value or any(
                not isinstance(item, str) or exact.SHA256_RE.fullmatch(item) is None
                for item in value.values()
            ):
                raise LedgerError("artifact_file_sha256 must contain digests only")
            return
        forbidden = PROHIBITED_CONTENT_KEYS.intersection(value)
        if forbidden:
            raise LedgerError(
                "receipt contains prohibited content fields: " + ", ".join(sorted(forbidden))
            )
        for key, child in value.items():
            _reject_sensitive_content(child, key)
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

    qwen_generation18_items: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]] = {}
    for plan in qwen_generation18.validate_all(repo_root).values():
        if plan.get("treatment") != exact.EXPECTED_TREATMENT:
            raise LedgerError("Qwen generation-18 plan treatment drifted from the exact universe")
        for item in plan["attempts"]:
            key = (item["cell_id"], item["execution_id"])
            if key in qwen_generation18_items:
                raise LedgerError("Qwen generation-18 execution authority is duplicated")
            qwen_generation18_items[key] = (plan, item)

    qwen_generation19_v4_items: dict[
        tuple[str, str], tuple[dict[str, Any], dict[str, Any]]
    ] = {}
    for plan in qwen_generation19_v4.validate_all(repo_root).values():
        if plan.get("treatment") != exact.EXPECTED_TREATMENT:
            raise LedgerError(
                "Qwen generation-19 v4 plan treatment drifted from the exact universe"
            )
        for item in plan["attempts"]:
            key = (item["cell_id"], item["execution_id"])
            if key in qwen_generation19_v4_items:
                raise LedgerError("Qwen generation-19 v4 execution authority is duplicated")
            qwen_generation19_v4_items[key] = (plan, item)

    hosted_glm_bulk_items: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]] = {}
    hosted_glm_plans = hosted_glm_bulk.validate_all(repo_root)
    if HOSTED_GLM_INVENTORY_PATH.is_file():
        if (
            self_hosted.sha256(HOSTED_GLM_INVENTORY_PATH.read_bytes())
            != HOSTED_GLM_INVENTORY_FILE_SHA256
        ):
            raise LedgerError("hosted GLM runtime inventory file digest drifted")
        inventory = load_receipt(HOSTED_GLM_INVENTORY_PATH)
        if inventory.get("receipt_sha256") != HOSTED_GLM_INVENTORY_RECEIPT_SHA256:
            raise LedgerError("hosted GLM runtime inventory receipt digest drifted")
        hosted_glm_plans = {
            controller: hosted_glm_bulk.build_runtime_plan(controller, inventory, repo_root)
            for controller in hosted_glm_bulk.CONTROLLERS
        }
    for plan in hosted_glm_plans.values():
        if plan.get("treatment") != exact.EXPECTED_TREATMENT:
            raise LedgerError("hosted GLM bulk treatment drifted from the exact universe")
        for item in plan["attempts"]:
            key = (item["cell_id"], item["execution_id"])
            if key in hosted_glm_bulk_items:
                raise LedgerError("hosted GLM bulk execution authority is duplicated")
            hosted_glm_bulk_items[key] = (plan, item)

    c2_plan = hosted_glm_c2.build_plan(hosted_glm_c2.CONTROLLER, repo_root)
    if HOSTED_GLM_INVENTORY_PATH.is_file():
        c2_plan = hosted_glm_c2.build_runtime_plan("glm-hosted-s2", inventory, repo_root)
    c2_item = c2_plan["attempts"][0]
    hosted_glm_c2_items = {
        (c2_item["cell_id"], c2_item["execution_id"]): (c2_plan, c2_item)
    }

    supplemental_path = repo_root / SUPPLEMENTAL_RUNTIME_AUTHORITY_PATH
    supplemental = load_receipt(supplemental_path)
    _require_exact_fields(
        supplemental,
        {
            "schema_version",
            "entries",
            "execution_rollforwards",
            "privacy",
            "receipt_sha256",
        },
        supplemental_path,
    )
    if (
        supplemental.get("schema_version") != SUPPLEMENTAL_RUNTIME_AUTHORITY_SCHEMA
        or supplemental.get("privacy")
        != {
            "scores_read": False,
            "prompts_traces_flags_read": False,
            "credentials_included": False,
        }
        or not isinstance(supplemental.get("entries"), list)
        or not isinstance(supplemental.get("execution_rollforwards"), list)
    ):
        raise LedgerError("supplemental runtime authority envelope drifted")
    supplemental_bulk_items: dict[
        tuple[str, str], tuple[dict[str, Any], dict[str, Any]]
    ] = {}
    supplemental_fields = {
        "model",
        "controller",
        "plan_sha256",
        "cell_id",
        "execution_id",
        "execution_generation",
        "run_id",
        "selection_rank",
        "attempt",
        "task_version_id",
    }
    for row in supplemental["entries"]:
        if not isinstance(row, dict) or set(row) != supplemental_fields:
            raise LedgerError("supplemental runtime authority entry fields drifted")
        key = (row["cell_id"], row["execution_id"])
        cell = cells.get(row["cell_id"])
        if (
            cell is None
            or cell["model"] != row["model"]
            or cell["selection_rank"] != row["selection_rank"]
            or cell["attempt"] != row["attempt"]
            or cell["task_version_id"] != row["task_version_id"]
            or exact.execution_for(row["cell_id"], row["execution_generation"])[
                "execution_id"
            ]
            != row["execution_id"]
            or key in supplemental_bulk_items
        ):
            raise LedgerError("supplemental runtime authority statistical binding drifted")
        _require_sha256(row["plan_sha256"], "supplemental plan sha256", supplemental_path)
        item = {
            name: row[name]
            for name in (
                "cell_id",
                "execution_id",
                "execution_generation",
                "run_id",
                "selection_rank",
                "attempt",
                "task_version_id",
            )
        }
        item["task_key"] = cell["task_key"]
        supplemental_bulk_items[key] = (
            {"controller": row["controller"], "plan_sha256": row["plan_sha256"]},
            item,
        )

    rollforward_precedence: dict[str, tuple[str, str, str, str]] = {}
    rollforward_fields = {
        "cell_id",
        "prior_execution_id",
        "prior_execution_generation",
        "prior_state",
        "prior_claim_receipt_sha256",
        "successor_execution_id",
        "successor_execution_generation",
        "successor_states_allowed",
        "successor_claim_receipt_sha256",
        "preserver_clearance_path",
        "preserver_clearance_receipt_sha256",
        "preserver_clearance_file_sha256",
        "scoring_release_path",
        "scoring_release_receipt_sha256",
        "scoring_release_file_sha256",
        "package_canary_path",
        "package_canary_receipt_sha256",
        "package_canary_file_sha256",
        "prior_generations_retry_prohibited",
        "fresh_exact_target_session_matches_at_release",
        "fresh_accepted_generation_matches_at_release",
        "atomic_whole_task_reservation_required",
    }

    def load_bound_repo_receipt(
        row: dict[str, Any], prefix: str
    ) -> tuple[Path, dict[str, Any]]:
        supplied = row[f"{prefix}_path"]
        if not isinstance(supplied, str):
            raise LedgerError(f"supplemental {prefix} path is invalid")
        relative = Path(supplied)
        if relative.is_absolute() or ".." in relative.parts:
            raise LedgerError(f"supplemental {prefix} path escapes the repository")
        path = repo_root / relative
        receipt = load_receipt(path)
        if (
            receipt.get("receipt_sha256") != row[f"{prefix}_receipt_sha256"]
            or self_hosted.sha256(path.read_bytes()) != row[f"{prefix}_file_sha256"]
        ):
            raise LedgerError(f"supplemental {prefix} binding drifted")
        return path, receipt

    expected_rollforward_cells: set[str] = set()
    for row in supplemental["execution_rollforwards"]:
        if not isinstance(row, dict) or set(row) != rollforward_fields:
            raise LedgerError("supplemental execution roll-forward fields drifted")
        cell = cells.get(row["cell_id"])
        prior_key = (row["cell_id"], row["prior_execution_id"])
        successor_key = (row["cell_id"], row["successor_execution_id"])
        if (
            cell is None
            or cell["model"] != "qwen3.8-27b"
            or cell["selection_rank"] != 99
            or row["prior_execution_generation"] != 19
            or row["successor_execution_generation"] != 23
            or exact.execution_for(row["cell_id"], 19)["execution_id"]
            != row["prior_execution_id"]
            or exact.execution_for(row["cell_id"], 23)["execution_id"]
            != row["successor_execution_id"]
            or prior_key not in supplemental_bulk_items
            or successor_key not in supplemental_bulk_items
            or row["prior_state"] != "blocked_nonrepeatable"
            or row["successor_states_allowed"] != ["active", "accepted"]
            or row["prior_generations_retry_prohibited"] != [20, 21, 22]
            or row["fresh_exact_target_session_matches_at_release"] != 0
            or row["fresh_accepted_generation_matches_at_release"] != 0
            or row["atomic_whole_task_reservation_required"] is not True
            or row["cell_id"] in rollforward_precedence
        ):
            raise LedgerError("supplemental execution roll-forward identity drifted")
        for field in (
            "prior_claim_receipt_sha256",
            "successor_claim_receipt_sha256",
            "preserver_clearance_receipt_sha256",
            "preserver_clearance_file_sha256",
            "scoring_release_receipt_sha256",
            "scoring_release_file_sha256",
            "package_canary_receipt_sha256",
            "package_canary_file_sha256",
        ):
            _require_sha256(row[field], field, supplemental_path)

        _preserver_path, preserver = load_bound_repo_receipt(row, "preserver_clearance")
        _release_path, release = load_bound_repo_receipt(row, "scoring_release")
        _canary_path, canary = load_bound_repo_receipt(row, "package_canary")
        successors = {
            (item.get("cell_id"), item.get("execution_id"))
            for item in preserver.get("successors", [])
            if isinstance(item, dict)
        }
        prior_claims = {
            (item.get("execution_id"), item.get("receipt_sha256"))
            for item in preserver.get("prior_execution_chain", {}).get(
                "generation19_claims", []
            )
            if isinstance(item, dict)
        }
        terminal_chain = preserver.get("prior_execution_chain", {})
        fresh = preserver.get("fresh_get_only_reconciliation", {})
        release_cells = {
            (item.get("cell_id"), item.get("execution_id"))
            for item in release.get("cells", [])
            if isinstance(item, dict)
        }
        release_rollforward = release.get("execution_generation_rollforward", {})
        if (
            preserver.get("status")
            != "CLEAR_FOR_FIXED_RENDERED_CLUSTER_PACKAGE_CANARY_ONLY"
            or preserver.get("selection_rank") != 99
            or successor_key not in successors
            or (
                row["prior_execution_id"],
                row["prior_claim_receipt_sha256"],
            )
            not in prior_claims
            or any(
                terminal_chain.get(f"generation{generation}_terminal", {}).get(
                    "retry_allowed"
                )
                is not False
                for generation in (20, 21, 22)
            )
            or fresh.get("exact_target_model_cell_or_execution_session_matches") != 0
            or fresh.get("accepted_outcome_matches") != 0
            or fresh.get("generation23_canonical_claim_paths_present") != 0
            or release.get("status") != "RELEASED_TO_DEDICATED"
            or release.get("selection_rank") != 99
            or release.get("execution_generation") != 23
            or release.get("whole_task_boundary_reserved") is not True
            or release.get("scoring_launch_authorized") is not True
            or successor_key not in release_cells
            or release_rollforward.get("preserver_clearance_receipt_sha256")
            != preserver["receipt_sha256"]
            or release_rollforward.get("generation19_claims_preserved") is not True
            or release_rollforward.get("generation20_21_22_retry_prohibited") is not True
            or release_rollforward.get("fresh_zero_accepted_generations") is not True
            or release_rollforward.get("fresh_exact_target_session_matches") != 0
            or canary.get("status")
            != "PASSED_CALLBACK_IDENTICAL_CLAIMS_DISABLED_PRECLAIM_BOUNDARY"
            or canary.get("execution_generation") != 23
            or canary.get("selection_rank") != 99
            or canary.get("clearance_receipt_sha256") != preserver["receipt_sha256"]
            or canary.get("release_receipt_sha256") != release["receipt_sha256"]
            or canary.get("claims_disabled") is not True
            or canary.get("claims_created") != 0
            or canary.get("reservation_created") is not False
            or canary.get("model_requests") != 0
            or canary.get("task_instance_session_verifier_scoring_calls") != 0
        ):
            raise LedgerError("supplemental execution roll-forward evidence drifted")
        expected_rollforward_cells.add(row["cell_id"])
        rollforward_precedence[row["cell_id"]] = (
            row["prior_execution_id"],
            row["successor_execution_id"],
            row["prior_claim_receipt_sha256"],
            row["successor_claim_receipt_sha256"],
        )
    if len(expected_rollforward_cells) != 4:
        raise LedgerError("supplemental execution roll-forward boundary is incomplete")

    dedicated_qwen_items: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]] = {}
    for attempt in sorted(qwen_dedicated.EXPECTED_IDENTITIES):
        generated_plan = qwen_dedicated.build_plan(repo_root, attempt)
        if generated_plan.get("treatment") != exact.EXPECTED_TREATMENT:
            raise LedgerError("dedicated Qwen plan treatment drifted from the exact universe")
        if attempt == 1:
            binding = DEDICATED_QWEN_ATTEMPT1_BINDING
            item = {
                key: binding[key]
                for key in (
                    "cell_id",
                    "execution_id",
                    "execution_generation",
                    "run_id",
                    "selection_rank",
                    "attempt",
                    "task_version_id",
                )
            }
            if item != generated_plan["item"]:
                raise LedgerError("dedicated Qwen attempt-1 statistical binding drifted")
            plan = {
                "plan_sha256": binding["plan_sha256"],
                "controller": binding["controller"],
                "treatment": exact.EXPECTED_TREATMENT,
                "config": {
                    "config_sha256": binding["config_sha256"],
                    "serving": {"serving_block": binding["serving_block"]},
                },
                "item": item,
            }
        else:
            plan, item = generated_plan, generated_plan["item"]
        key = (item["cell_id"], item["execution_id"])
        if key in dedicated_qwen_items:
            raise LedgerError("dedicated Qwen execution authority is duplicated")
        dedicated_qwen_items[key] = (plan, item)

    dedicated_qwen_v3_items: dict[
        tuple[str, str], tuple[dict[str, Any], dict[str, Any]]
    ] = {}
    for attempt in (2, 3, 4):
        dedicated_qwen_v3_plan = qwen_dedicated_v3.build_plan(repo_root, attempt)
        dedicated_qwen_v3_item = dedicated_qwen_v3_plan["item"]
        key = (dedicated_qwen_v3_item["cell_id"], dedicated_qwen_v3_item["execution_id"])
        if key in dedicated_qwen_v3_items:
            raise LedgerError("dedicated Qwen v3 execution authority is duplicated")
        dedicated_qwen_v3_items[key] = (dedicated_qwen_v3_plan, dedicated_qwen_v3_item)

    dedicated_qwen_rank3_items: dict[
        tuple[str, str], tuple[dict[str, Any], dict[str, Any]]
    ] = {}
    for attempt in sorted(qwen_dedicated_rank3.EXPECTED_IDENTITIES):
        plan = qwen_dedicated_rank3._released_plan(  # noqa: SLF001
            repo_root,
            attempt,
            release_path=(
                repo_root
                / "docs/evidence/qwen38-study/"
                "2026-09-05-qwen38-dedicated-rank3-g21-b-v2-release-v3.json"
            ),
        )
        item = plan["item"]
        key = (item["cell_id"], item["execution_id"])
        if key in dedicated_qwen_rank3_items:
            raise LedgerError("dedicated Qwen rank-3 execution authority is duplicated")
        dedicated_qwen_rank3_items[key] = (plan, item)
    rank3_g22_release = (
        repo_root
        / "docs/evidence/qwen38-study/"
        "2026-09-05-qwen38-dedicated-rank3-a2-g22-release-v1.json"
    )
    rank3_g22_plan = qwen_dedicated_rank3_g22._released_plan(  # noqa: SLF001
        repo_root, release_path=rank3_g22_release
    )
    rank3_g22_item = rank3_g22_plan["item"]
    rank3_g22_key = (rank3_g22_item["cell_id"], rank3_g22_item["execution_id"])
    if rank3_g22_key in dedicated_qwen_rank3_items:
        raise LedgerError("dedicated Qwen rank-3 g22 execution authority is duplicated")
    dedicated_qwen_rank3_items[rank3_g22_key] = (rank3_g22_plan, rank3_g22_item)

    dedicated_qwen_dp8_plan = qwen_dedicated_dp8.released_plan(
        repo_root,
        release_path=(
            repo_root
            / "docs/evidence/qwen38-study/"
            "2026-09-05-qwen38-dedicated-dp8-r004-a2-g22-release-v1.json"
        ),
    )
    dedicated_qwen_dp8_item = dedicated_qwen_dp8_plan["item"]
    dedicated_qwen_dp8_items = {
        (
            dedicated_qwen_dp8_item["cell_id"],
            dedicated_qwen_dp8_item["execution_id"],
        ): (dedicated_qwen_dp8_plan, dedicated_qwen_dp8_item)
    }

    dedicated_qwen_rank97_items: dict[
        tuple[str, str], tuple[dict[str, Any], dict[str, Any]]
    ] = {}
    rank97_release = (
        repo_root
        / "docs/evidence/qwen38-study/"
        "2026-09-05-qwen38-dedicated-rank97-tp1-d-release-v2.json"
    )
    rank97_plans = qwen_dedicated_rank97._released_plans(  # noqa: SLF001
        repo_root, release_path=rank97_release
    )
    for plan in rank97_plans:
        item = plan["item"]
        key = (item["cell_id"], item["execution_id"])
        if key in dedicated_qwen_rank97_items:
            raise LedgerError("dedicated Qwen rank-97 execution authority is duplicated")
        dedicated_qwen_rank97_items[key] = (plan, item)

    generation7_items = _validated_fixed_bindings(cells, GENERATION7_BINDINGS, 7)
    generation15_items = _validated_fixed_bindings(cells, GENERATION15_BINDINGS, 15)
    return Authority(
        repo_root=repo_root,
        cells=cells,
        bulk_items=bulk_items,
        qwen_generation16_items=qwen_generation16_items,
        qwen_generation18_items=qwen_generation18_items,
        qwen_generation19_v4_items=qwen_generation19_v4_items,
        hosted_glm_bulk_items=hosted_glm_bulk_items,
        hosted_glm_c2_items=hosted_glm_c2_items,
        supplemental_bulk_items=supplemental_bulk_items,
        rollforward_precedence=rollforward_precedence,
        dedicated_qwen_items=dedicated_qwen_items,
        dedicated_qwen_v3_items=dedicated_qwen_v3_items,
        dedicated_qwen_rank3_items=dedicated_qwen_rank3_items,
        dedicated_qwen_rank97_items=dedicated_qwen_rank97_items,
        dedicated_qwen_dp8_items=dedicated_qwen_dp8_items,
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


def _bulk_pair(
    authority: Authority,
    key: tuple[Any, Any],
    controller: Any,
    plan_sha256: Any = None,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    supplemental = authority.supplemental_bulk_items.get(key)
    if (
        supplemental is not None
        and supplemental[0].get("controller") == controller
        and (
            plan_sha256 is None
            or supplemental[0].get("plan_sha256") == plan_sha256
        )
    ):
        # The immutable supplemental authority exists specifically to resolve
        # a runtime identity that is otherwise represented by more than one
        # frozen controller plan.  Prefer it before considering those broader
        # plans; accepting both as peers would recreate the ambiguity the
        # authority receipt closed.
        return supplemental
    candidates = [
        pair
        for mapping in (
            authority.qwen_generation18_items,
            authority.qwen_generation19_v4_items,
            authority.qwen_generation16_items,
            authority.hosted_glm_bulk_items,
            authority.hosted_glm_c2_items,
            authority.bulk_items,
        )
        if (pair := mapping.get(key)) is not None
        and pair[0].get("controller") == controller
        and (plan_sha256 is None or pair[0].get("plan_sha256") == plan_sha256)
    ]
    if len(candidates) > 1:
        raise LedgerError("bulk receipt controller is ambiguous across frozen plans")
    return candidates[0] if candidates else None


def _accepted_bulk(value: dict[str, Any], path: Path, authority: Authority) -> Evidence:
    _require_exact_fields(
        value,
        BULK_ACCEPTED_FIELDS,
        path,
        optional={BULK_ACCEPTED_OPTIONAL_PROJECTION_FIELD},
    )
    key = (value.get("cell_id"), value.get("execution_id"))
    pair = _bulk_pair(authority, key, value.get("controller"), value.get("plan_sha256"))
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


def _accepted_legacy_glm_generation7(
    value: dict[str, Any], path: Path, authority: Authority
) -> Evidence:
    """Admit the one reviewed legacy receipt imported as exact generation 7."""
    _require_exact_fields(value, LEGACY_GLM_GENERATION7_ACCEPTED_FIELDS, path)
    binding = GENERATION7_BINDINGS["glm-5.3"]
    key = (binding["cell_id"], binding["execution_id"])
    if key not in authority.generation7:
        raise LedgerError(f"legacy GLM acceptance lacks generation-7 authority: {path}")
    cell, generation = _require_cell_execution(authority, *key, 7, path)
    source = load_receipt(authority.repo_root / LEGACY_GLM_GENERATION7_SOURCE)
    entries = [
        row
        for row in source.get("accepted_cells", [])
        if row.get("model_block") == "glm_hosted_v12"
        and row.get("source_rank") == 13
        and row.get("attempt") == 1
    ]
    if (
        source.get("receipt_sha256") != LEGACY_GLM_GENERATION7_SOURCE_SHA256
        or len(entries) != 1
        or entries[0].get("receipt") != value
    ):
        raise LedgerError(f"legacy GLM acceptance identity or outcome drifted: {path}")
    if cell["selection_rank"] != 13 or cell["attempt"] != 1:
        raise LedgerError(f"legacy GLM acceptance statistical cell drifted: {path}")
    _require_uuid(value.get("session_id"), "session id", path)
    _require_uuid(value.get("verifier_execution_id"), "verifier execution id", path)
    return Evidence("accepted", cell["cell_id"], key[1], generation, value["receipt_sha256"], path)


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


def _accepted_generation15_gate(
    value: dict[str, Any], path: Path, authority: Authority
) -> Evidence:
    """Admit the exact durable score-blind G15 acceptance gate."""
    _require_exact_fields(value, GENERATION15_ACCEPTED_GATE_FIELDS, path)
    binding = GENERATION15_BINDINGS["qwen3.8-27b"]
    key = (value.get("cell_id"), value.get("execution_id"))
    if key not in authority.generation15 or key != (
        binding["cell_id"],
        binding["execution_id"],
    ):
        raise LedgerError(f"generation-15 gate lacks exact immutable authority: {path}")
    cell, generation = _require_cell_execution(authority, *key, 15, path)
    expected = {
        "status": "ACCEPTED",
        "model": "qwen3.8-27b",
        "cell_id": binding["cell_id"],
        "execution_id": binding["execution_id"],
        "task_key": cell["task_key"],
        "task_version_id": cell["task_version_id"],
        "config_sha256": (
            "sha256:b4d07e3b4c5ff610ccf9eb8c334cbd4dd3eb43873204ecfbe08ce3e9b1cde053"
        ),
        "inner_acceptance_receipt_sha256": (
            "sha256:887cc71ed706f60b6364a5e639e0f891e09c70847d86853fef6a270beea9074a"
        ),
        "top_acceptance_receipt_sha256": (
            "sha256:398e7aadd91d77306dcff464747e5f026809bc61ec543b63879a2ab0a9c6915b"
        ),
        "workload_uid": "f3ff2cf1-7c37-4c2f-8f6b-3db39d821718",
        "cleanup_completed": True,
        "credentials_included": False,
        "prompts_traces_flags_or_scores_included": False,
        "receipt_sha256": (
            "sha256:e0aef9a97d146fe5fcc686efafd4399bd7c2ee65a325e64fc089613507ab6745"
        ),
    }
    if any(value.get(field) != expected_value for field, expected_value in expected.items()):
        raise LedgerError(f"generation-15 gate identity or evidence drifted: {path}")
    expected_session = {
        "exact_session_present": True,
        "model_projection": "omitted",
        "session_id": "e98906f6-5878-4f88-b30a-e65e84706b02",
        "transcript_content_read": False,
        "transcript_route_status": 200,
        "verifier_execution_id": "0ddbe2b9-7135-4f87-be5d-383ad5f36e24",
    }
    expected_job = {
        "exclusive_complete": True,
        "name": "chris-q38-ac-r004-a1-g15-v1",
        "uid": "2e35df7c-e4f6-45fa-aa27-21839b211944",
    }
    expected_pod = {
        "name": "chris-q38-ac-r004-a1-g15-v1-wmf4f",
        "restarts": 0,
        "uid": "f66d1e33-1062-4e58-97b4-41d2d25f76e2",
    }
    if (
        value.get("api_session") != expected_session
        or value.get("job") != expected_job
        or value.get("pod") != expected_pod
    ):
        raise LedgerError(f"generation-15 gate authoritative observation drifted: {path}")
    _require_uuid(expected_session["session_id"], "session id", path)
    _require_uuid(expected_session["verifier_execution_id"], "verifier execution id", path)
    return Evidence("accepted", cell["cell_id"], key[1], generation, value["receipt_sha256"], path)


def _accepted_dedicated_qwen(value: dict[str, Any], path: Path, authority: Authority) -> Evidence:
    _require_exact_fields(
        value,
        DEDICATED_QWEN_ACCEPTED_FIELDS,
        path,
        optional=DEDICATED_QWEN_PROJECTION_FIELDS,
    )
    key = (value.get("cell_id"), value.get("execution_id"))
    candidates = [
        pair
        for mapping in (
            authority.dedicated_qwen_items,
            authority.dedicated_qwen_v3_items,
            authority.dedicated_qwen_rank3_items,
            authority.dedicated_qwen_rank97_items,
            authority.dedicated_qwen_dp8_items,
        )
        if (pair := mapping.get(key)) is not None
        and pair[0]["config"]["serving"]["serving_block"] == value.get("serving_block")
    ]
    pair = candidates[0] if len(candidates) == 1 else None
    if pair is None:
        if any(
            key in mapping
            for mapping in (
                authority.dedicated_qwen_items,
                authority.dedicated_qwen_v3_items,
                authority.dedicated_qwen_rank3_items,
                authority.dedicated_qwen_rank97_items,
                authority.dedicated_qwen_dp8_items,
            )
        ):
            raise LedgerError(f"dedicated Qwen acceptance identity or treatment drifted: {path}")
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
    projection_fields = set(value).intersection(DEDICATED_QWEN_PROJECTION_FIELDS)
    if projection_fields and projection_fields != DEDICATED_QWEN_PROJECTION_FIELDS:
        raise LedgerError(f"dedicated Qwen API projection evidence is incomplete: {path}")
    if projection_fields:
        omissions = value.get("authoritative_projection_omissions")
        if (
            value.get("authoritative_session_task_key_matched") is not True
            or value.get("authoritative_projection_rule")
            != "legacy_list_fields_may_be_null_but_never_mismatched_v1"
            or not isinstance(omissions, list)
            or omissions != sorted(set(omissions))
            or not set(omissions) <= {"metadata", "model", "task_version_id"}
        ):
            raise LedgerError(f"dedicated Qwen API projection evidence drifted: {path}")
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


def _accepted_validated_dedicated_qwen(
    value: dict[str, Any], path: Path, authority: Authority
) -> Evidence:
    """Admit the one reviewed post-terminal acceptance chain, fail closed."""
    _require_exact_fields(value, DEDICATED_QWEN_VALIDATED_FIELDS, path)
    binding = DEDICATED_QWEN_ATTEMPT1_BINDING
    key = (value.get("cell_id"), value.get("execution_id"))
    expected_key = (binding["cell_id"], binding["execution_id"])
    if key != expected_key:
        raise LedgerError(f"validated dedicated Qwen receipt is not the reviewed attempt: {path}")
    pair = authority.dedicated_qwen_items.get(key)
    if pair is None:
        raise LedgerError(f"validated dedicated Qwen receipt lacks exact plan authority: {path}")
    plan, item = pair
    cell, generation = _require_cell_execution(authority, *key, item["execution_generation"], path)
    expected = {
        "serving_block": binding["serving_block"],
        "cell_id": binding["cell_id"],
        "execution_id": binding["execution_id"],
        "run_id": binding["run_id"],
        "selection_rank": binding["selection_rank"],
        "attempt": binding["attempt"],
        "task_version_id": binding["task_version_id"],
        "claim_sha256": binding["claim_sha256"],
        "claim_receipt_sha256": binding["claim_sha256"],
        "config_sha256": binding["config_sha256"],
        "source_terminal_receipt_sha256": binding["source_terminal_receipt_sha256"],
        "source_terminal_stale_claim_sha256": binding["source_terminal_stale_claim_sha256"],
        "source_terminal_actual_canonical_sha256": binding[
            "source_terminal_actual_canonical_sha256"
        ],
        "accepted_receipt_sha256": binding["accepted_receipt_sha256"],
        "acceptance_terminal_receipt_sha256": binding["acceptance_terminal_receipt_sha256"],
        "collector_job_uid": binding["collector_job_uid"],
        "collector_pod_uid": binding["collector_pod_uid"],
        "validator_job_uid": binding["validator_job_uid"],
        "validator_pod_uid": binding["validator_pod_uid"],
        "session_id": binding["session_id"],
        "verifier_execution_id": binding["verifier_execution_id"],
        "plan_file_sha256": binding["artifact_file_sha256"]["plan_file_sha256"],
        "claim_file_sha256": binding["artifact_file_sha256"]["claim_file_sha256"],
    }
    if any(value.get(field) != expected_value for field, expected_value in expected.items()):
        raise LedgerError(f"validated dedicated Qwen identity or chain binding drifted: {path}")
    if plan["plan_sha256"] != binding["plan_sha256"] or item["cell_id"] != cell["cell_id"]:
        raise LedgerError(f"validated dedicated Qwen immutable plan authority drifted: {path}")
    artifact_digests = value.get("artifact_file_sha256")
    if artifact_digests != binding["artifact_file_sha256"]:
        raise LedgerError(f"validated dedicated Qwen artifact digest chain drifted: {path}")
    if value.get("receipt_sha256") != binding["validated_receipt_sha256"]:
        raise LedgerError(f"validated dedicated Qwen reviewed receipt digest drifted: {path}")
    omissions = value.get("authoritative_projection_omissions")
    if omissions != ["metadata", "model", "task_version_id"]:
        raise LedgerError(f"validated dedicated Qwen API omission evidence drifted: {path}")
    if any(
        (
            value.get("status") != "ACCEPTED_VALIDATED",
            value.get("accepted") is not True,
            value.get("credited") is not True,
            value.get("retry_allowed") is not False,
            value.get("authoritative_projection_rule")
            != "legacy_list_fields_may_be_null_but_never_mismatched_v1",
            value.get("all_artifact_byte_digests_matched") is not True,
            value.get("fleet_api_mutations") != 0,
            value.get("fresh_authoritative_session_reconciled") is not True,
            value.get("scores_included") is not False,
            value.get("prompts_or_traces_included") is not False,
            value.get("credentials_included") is not False,
        )
    ):
        raise LedgerError(f"validated dedicated Qwen outcome is not authoritative: {path}")
    for field in ("validator_job_uid", "validator_pod_uid"):
        _require_uuid(value.get(field), field, path)
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


def _accepted_validated_dedicated_qwen_v2(
    value: dict[str, Any], path: Path, authority: Authority
) -> Evidence:
    """Validate the reusable v3-server acceptance chain without opening artifacts."""
    _require_exact_fields(value, DEDICATED_QWEN_VALIDATED_V2_FIELDS, path)
    key = (value.get("cell_id"), value.get("execution_id"))
    candidates = [
        pair
        for mapping in (
            authority.dedicated_qwen_v3_items,
            authority.dedicated_qwen_rank3_items,
        )
        if (pair := mapping.get(key)) is not None
    ]
    pair = candidates[0] if len(candidates) == 1 else None
    if pair is None:
        raise LedgerError(f"validated dedicated Qwen v2 receipt lacks exact plan authority: {path}")
    plan, item = pair
    cell, generation = _require_cell_execution(authority, *key, item["execution_generation"], path)
    expected = {
        "serving_block": plan["config"]["serving"]["serving_block"],
        "serving_parity_receipt_sha256": plan["config"]["serving"][
            "parity_receipt_sha256"
        ],
        "cell_id": item["cell_id"],
        "execution_id": item["execution_id"],
        "run_id": item["run_id"],
        "selection_rank": item["selection_rank"],
        "attempt": item["attempt"],
        "task_version_id": item["task_version_id"],
        "config_sha256": plan["config"]["config_sha256"],
        "plan_sha256": plan["plan_sha256"],
    }
    if any(value.get(field) != expected_value for field, expected_value in expected.items()):
        raise LedgerError(f"validated dedicated Qwen v2 identity or plan binding drifted: {path}")
    artifact_digests = value.get("artifact_file_sha256")
    if not isinstance(artifact_digests, dict) or set(artifact_digests) != {
        "plan",
        "claim",
        "result",
        "reward",
        "session_ingest",
        "cleanup",
        "accepted",
        "terminal",
    }:
        raise LedgerError(f"validated dedicated Qwen v2 artifact digest map drifted: {path}")
    for field in (
        "claim_sha256",
        "claim_receipt_sha256",
        "accepted_receipt_sha256",
        "terminal_receipt_sha256",
    ):
        _require_sha256(value.get(field), field, path)
    if value.get("claim_sha256") != value.get("claim_receipt_sha256"):
        raise LedgerError(f"validated dedicated Qwen v2 claim chain drifted: {path}")
    if any(
        (
            value.get("status") != "ACCEPTED_VALIDATED",
            value.get("accepted") is not True,
            value.get("credited") is not True,
            value.get("retry_allowed") is not False,
            value.get("authoritative_projection_omissions")
            != ["metadata", "model", "task_version_id"],
            value.get("authoritative_projection_rule")
            != "legacy_list_fields_may_be_null_but_never_mismatched_v1",
            value.get("all_artifact_byte_digests_matched") is not True,
            value.get("fleet_api_mutations") != 0,
            value.get("fresh_authoritative_session_reconciled") is not True,
            value.get("scores_included") is not False,
            value.get("prompts_or_traces_included") is not False,
            value.get("credentials_included") is not False,
        )
    ):
        raise LedgerError(f"validated dedicated Qwen v2 outcome is not authoritative: {path}")
    for field in (
        "session_id",
        "verifier_execution_id",
        "source_job_uid",
        "source_pod_uid",
        "validator_job_uid",
        "validator_pod_uid",
    ):
        _require_uuid(value.get(field), field, path)
    return Evidence(
        "accepted",
        cell["cell_id"],
        item["execution_id"],
        generation,
        value["receipt_sha256"],
        path,
    )


def _accepted_validated_glm_c2(
    value: dict[str, Any], path: Path, authority: Authority
) -> Evidence:
    """Validate the reviewed GLM c2 receipt chain and exact runtime-plan mapping."""
    _require_exact_fields(value, GLM_C2_ACCEPTED_VALIDATED_FIELDS, path)
    mapping_path = authority.repo_root / GLM_C2_RUNTIME_AUTHORITY_PATH
    mapping = load_receipt(mapping_path)
    _require_exact_fields(
        mapping,
        {
            "schema_version",
            "controller",
            "cell_id",
            "execution_id",
            "execution_generation",
            "plan_sha256",
            "run_id",
            "selection_rank",
            "attempt",
            "task_key",
            "task_version_id",
            "config_sha256",
            "release",
            "claim",
            "terminal",
            "accepted",
            "privacy",
            "receipt_sha256",
        },
        mapping_path,
    )
    if mapping.get("schema_version") != GLM_C2_RUNTIME_AUTHORITY_SCHEMA:
        raise LedgerError(f"GLM c2 runtime authority schema drifted: {mapping_path}")
    nested_fields = {
        "release": {
            "path",
            "receipt_sha256",
            "file_sha256",
            "observer_job_uid",
            "observer_pod_uid",
        },
        "claim": {"path", "receipt_sha256", "file_sha256"},
        "terminal": {
            "path",
            "receipt_sha256",
            "file_sha256",
            "source_job_uid",
            "source_pod_uid",
        },
        "accepted": {
            "path",
            "receipt_sha256",
            "file_sha256",
            "session_id",
            "verifier_execution_id",
        },
    }
    for name, fields in nested_fields.items():
        child = mapping.get(name)
        if not isinstance(child, dict) or set(child) != fields:
            raise LedgerError(f"GLM c2 runtime authority {name} fields drifted: {mapping_path}")
    if mapping.get("privacy") != {
        "scores_read": False,
        "prompts_traces_flags_read": False,
        "credentials_included": False,
    }:
        raise LedgerError(f"GLM c2 runtime authority privacy drifted: {mapping_path}")

    key = (mapping.get("cell_id"), mapping.get("execution_id"))
    pair = authority.hosted_glm_c2_items.get(key)
    if pair is None:
        raise LedgerError(f"GLM c2 mapping is absent from exact statistical authority: {path}")
    plan, item = pair
    cell, generation = _require_cell_execution(
        authority,
        *key,
        mapping.get("execution_generation"),
        path,
    )
    item_expected = {
        "cell_id": item["cell_id"],
        "execution_id": item["execution_id"],
        "execution_generation": item["execution_generation"],
        "run_id": item["run_id"],
        "selection_rank": item["selection_rank"],
        "attempt": item["attempt"],
        "task_key": item["task_key"],
        "task_version_id": item["task_version_id"],
    }
    if any(mapping.get(name) != expected for name, expected in item_expected.items()):
        raise LedgerError(f"GLM c2 runtime authority statistical binding drifted: {mapping_path}")
    if HOSTED_GLM_INVENTORY_PATH.is_file() and plan.get("plan_sha256") != mapping.get(
        "plan_sha256"
    ):
        raise LedgerError(f"GLM c2 runtime plan digest drifted: {mapping_path}")

    release = mapping["release"]
    claim = mapping["claim"]
    terminal = mapping["terminal"]
    accepted = mapping["accepted"]
    expected = {
        **item_expected,
        "controller": mapping["controller"],
        "plan_sha256": mapping["plan_sha256"],
        "config_sha256": mapping["config_sha256"],
        "release_path": release["path"],
        "release_receipt_sha256": release["receipt_sha256"],
        "release_file_sha256": release["file_sha256"],
        "release_observer_job_uid": release["observer_job_uid"],
        "release_observer_pod_uid": release["observer_pod_uid"],
        "claim_sha256": claim["receipt_sha256"],
        "claim_file_sha256": claim["file_sha256"],
        "terminal_path": terminal["path"],
        "terminal_receipt_sha256": terminal["receipt_sha256"],
        "terminal_file_sha256": terminal["file_sha256"],
        "accepted_path": accepted["path"],
        "accepted_receipt_sha256": accepted["receipt_sha256"],
        "accepted_file_sha256": accepted["file_sha256"],
        "source_job_uid": terminal["source_job_uid"],
        "source_pod_uid": terminal["source_pod_uid"],
        "session_id": accepted["session_id"],
        "verifier_execution_id": accepted["verifier_execution_id"],
    }
    if any(value.get(name) != expected_value for name, expected_value in expected.items()):
        raise LedgerError(f"validated GLM c2 identity or source receipt chain drifted: {path}")
    for field in (
        "plan_sha256",
        "config_sha256",
        "release_receipt_sha256",
        "release_file_sha256",
        "claim_sha256",
        "claim_file_sha256",
        "terminal_receipt_sha256",
        "terminal_file_sha256",
        "accepted_receipt_sha256",
        "accepted_file_sha256",
    ):
        _require_sha256(value.get(field), field, path)
    for field in (
        "release_observer_job_uid",
        "release_observer_pod_uid",
        "source_job_uid",
        "source_pod_uid",
        "session_id",
        "verifier_execution_id",
    ):
        _require_uuid(value.get(field), field, path)
    if any(
        (
            value.get("status") != "ACCEPTED_VALIDATED",
            value.get("accepted") is not True,
            value.get("credited") is not True,
            value.get("retry_allowed") is not False,
            value.get("all_source_receipt_byte_digests_matched") is not True,
            value.get("fresh_authoritative_session_reconciled") is not True,
            value.get("scores_included") is not False,
            value.get("prompts_or_traces_included") is not False,
            value.get("credentials_included") is not False,
        )
    ):
        raise LedgerError(f"validated GLM c2 outcome is not authoritative: {path}")
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
    if schema == DEDICATED_QWEN_VALIDATED_SCHEMA:
        return _accepted_validated_dedicated_qwen(value, path, authority)
    if schema == DEDICATED_QWEN_VALIDATED_V2_SCHEMA:
        return _accepted_validated_dedicated_qwen_v2(value, path, authority)
    if schema == GLM_C2_ACCEPTED_VALIDATED_SCHEMA:
        return _accepted_validated_glm_c2(value, path, authority)
    if schema == GENERATION15_ACCEPTED_GATE_SCHEMA:
        return _accepted_generation15_gate(value, path, authority)
    if schema == LEGACY_GLM_GENERATION7_ACCEPTED_SCHEMA:
        return _accepted_legacy_glm_generation7(value, path, authority)
    if schema == GENERATION7_TERMINAL_SCHEMA:
        return _accepted_generation7(value, path, authority)
    return _accepted_generation15(value, path, authority)


def _claim_bulk(value: dict[str, Any], path: Path, authority: Authority) -> Evidence:
    _require_exact_fields(value, BULK_CLAIM_FIELDS, path)
    key = (value.get("cell_id"), value.get("execution_id"))
    pair = _bulk_pair(authority, key, value.get("controller"), value.get("plan_sha256"))
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
    candidates = [
        pair
        for mapping in (
            authority.dedicated_qwen_items,
            authority.dedicated_qwen_v3_items,
            authority.dedicated_qwen_rank3_items,
            authority.dedicated_qwen_rank97_items,
            authority.dedicated_qwen_dp8_items,
        )
        if (pair := mapping.get(key)) is not None
        and pair[0]["controller"] == value.get("controller")
    ]
    pair = candidates[0] if len(candidates) == 1 else None
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
    try:
        expected_name = bulk_runtime.claim_filename(str(value.get("execution_id", "")))
    except ValueError as exc:
        raise LedgerError(f"bulk claim execution id is invalid: {path}") from exc
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
        key = (value.get("cell_id"), value.get("execution_id"))
        dedicated_pair_exists = any(
            (pair := mapping.get(key)) is not None
            and pair[0]["controller"] == value.get("controller")
            for mapping in (
                authority.dedicated_qwen_items,
                authority.dedicated_qwen_v3_items,
                authority.dedicated_qwen_rank3_items,
                authority.dedicated_qwen_rank97_items,
                authority.dedicated_qwen_dp8_items,
            )
        )
        if dedicated_pair_exists:
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


def _manifest_paths(
    path: Path,
    *,
    repo_root: Path,
    campaign: Path,
    mappings: Sequence[tuple[Path, Path]],
) -> dict[str, list[Path]]:
    """Resolve one immutable score-blind evidence-path snapshot.

    The manifest is deliberately a list of exact files rather than directory
    globs. Every referenced receipt digest is checked before the existing
    schema and identity validators see the receipt, so operators do not have
    to reconstruct the authoritative path set from chat or mutable listings.
    """
    manifest_path = path if path.is_absolute() else repo_root / path
    manifest = load_receipt(manifest_path.absolute())
    expected_fields = {
        "schema_version",
        "campaign_id",
        "campaign_path",
        "entries",
        "privacy",
        "receipt_sha256",
    }
    if set(manifest) != expected_fields:
        raise LedgerError(f"evidence manifest fields drifted: {manifest_path}")
    manifest_schema = manifest.get("schema_version")
    if (
        manifest_schema not in {EVIDENCE_MANIFEST_SCHEMA, EVIDENCE_MANIFEST_V2_SCHEMA}
        or manifest.get("campaign_id") != exact.EXPECTED_CAMPAIGN_ID
        or manifest.get("campaign_path") != str(campaign.relative_to(repo_root))
        or manifest.get("privacy")
        != {
            "prompts_read": False,
            "traces_read": False,
            "flags_read": False,
            "scores_read": False,
        }
    ):
        raise LedgerError(f"evidence manifest authority or privacy drifted: {manifest_path}")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise LedgerError(f"evidence manifest entries are absent: {manifest_path}")

    resolved: dict[str, list[Path]] = {kind: [] for kind in EVIDENCE_MANIFEST_KINDS}
    seen_paths: set[Path] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != {
            "kind",
            "path",
            "expected_receipt_sha256",
        }:
            raise LedgerError(f"evidence manifest entry {index} fields drifted: {manifest_path}")
        kind = entry.get("kind")
        supplied = entry.get("path")
        expected_sha = entry.get("expected_receipt_sha256")
        if kind not in EVIDENCE_MANIFEST_KINDS:
            raise LedgerError(f"evidence manifest entry {index} kind is invalid: {manifest_path}")
        if kind == "operational_incident" and manifest_schema != EVIDENCE_MANIFEST_V2_SCHEMA:
            raise LedgerError(
                f"evidence manifest entry {index} requires the v2 manifest schema: "
                f"{manifest_path}"
            )
        if not isinstance(supplied, str) or not supplied:
            raise LedgerError(f"evidence manifest entry {index} path is invalid: {manifest_path}")
        if not isinstance(expected_sha, str) or exact.SHA256_RE.fullmatch(expected_sha) is None:
            raise LedgerError(f"evidence manifest entry {index} digest is invalid: {manifest_path}")
        producer_path = Path(supplied)
        if producer_path.is_absolute():
            observer_path = _observer_path(producer_path, mappings).absolute()
        else:
            if ".." in producer_path.parts:
                raise LedgerError(
                    f"evidence manifest entry {index} escapes the repository: {manifest_path}"
                )
            observer_path = (repo_root / producer_path).absolute()
            try:
                observer_path.relative_to(repo_root)
            except ValueError as exc:
                raise LedgerError(
                    f"evidence manifest entry {index} escapes the repository: {manifest_path}"
                ) from exc
        if observer_path in seen_paths:
            raise LedgerError(f"evidence manifest repeats a path: {observer_path}")
        seen_paths.add(observer_path)
        receipt = load_receipt(observer_path)
        if receipt.get("receipt_sha256") != expected_sha:
            raise LedgerError(f"evidence manifest receipt digest drifted: {observer_path}")
        resolved[kind].append(observer_path)
    return resolved


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
        if accepted_rows and active_rows:
            raise LedgerError(f"accepted cell also has a live claim input: {cell_id}")
        current_rows = accepted_rows or active_rows
        if current_rows and blocked_rows:
            precedence = authority.rollforward_precedence.get(cell_id)
            current = current_rows[0]
            blocked = blocked_rows[0]
            binding_matches = precedence is not None and precedence[:3] == (
                blocked.execution_id,
                current.execution_id,
                blocked.receipt_sha256,
            )
            if current.state == "active":
                binding_matches = binding_matches and precedence[3] == current.receipt_sha256
            else:
                accepted_receipt = load_receipt(current.path)
                binding_matches = (
                    binding_matches
                    and accepted_receipt.get("claim_sha256") == precedence[3]
                )
            if not binding_matches:
                raise LedgerError(
                    f"cell has no exact append-only roll-forward precedence: {cell_id}"
                )

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
        "--evidence-manifest",
        type=Path,
        help=(
            "exact score-blind evidence-path snapshot with expected receipt digests; "
            "cannot be combined with individual evidence path options"
        ),
    )
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
        individual_inputs = (
            args.accepted
            + args.accepted_root
            + args.active_claim
            + args.active_claim_root
            + args.nonrepeatable_claim
            + args.nonrepeatable_claim_root
            + args.tombstone
            + args.tombstone_root
        )
        if args.evidence_manifest is not None and individual_inputs:
            raise LedgerError(
                "--evidence-manifest cannot be combined with individual evidence paths"
            )
        if args.evidence_manifest is not None:
            manifest_paths = _manifest_paths(
                args.evidence_manifest,
                repo_root=repo_root,
                campaign=campaign,
                mappings=mappings,
            )
            accepted_paths = manifest_paths["accepted"]
            active_paths = manifest_paths["active_claim"]
            blocked_paths = manifest_paths["nonrepeatable_claim"]
            tombstone_paths = manifest_paths["tombstone"]
        else:
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
