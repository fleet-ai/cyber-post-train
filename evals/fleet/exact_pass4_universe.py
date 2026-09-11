"""Build the exact Qwen3.8/GLM5.3 easiest-100 pass@4 scientific universe.

This module defines statistical cells independently from execution attempts.  A
pre-model infrastructure failure may therefore be preserved as a tombstoned
execution generation without silently consuming, changing, or duplicating the
statistical cell.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import uuid
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_crypto as crypto

CAMPAIGN_SCHEMA = "fleet-exact-easiest100-pass4-campaign-v1"
EXPECTED_CAMPAIGN_ID = "chris-cyber-q38-glm53-exact-easiest100-pass4-v1"
UNIVERSE_SCHEMA = "fleet-exact-easiest100-pass4-universe-v1"
CELL_SCHEMA = "fleet-exact-easiest100-pass4-cell-v1"
EXECUTION_SCHEMA = "fleet-statistical-cell-execution-v1"
TOMBSTONE_SCHEMA = "fleet-pre-model-infrastructure-tombstone-v1"
SELECTION_SCHEMA = "fleet-opencode-easiest-train100-selection-v2"

EXPECTED_SELECTION_PATH = "evals/fleet/configs/opencode-easiest-train100-selection-v2.json"
EXPECTED_SELECTION_FILE_SHA256 = (
    "sha256:9f5e83123cfe7d8a3f4cc0b791ac1dc9032a3d5faf5eba2a431c7246069255df"
)
EXPECTED_SELECTION_SHA256 = (
    "sha256:38bd544c74f4e45cb67b271849d657f49e356ccf41f97acca9fb6cd7eb7f56b8"
)
EXPECTED_SOURCE_JOB_ID = "a62dd51f-a52b-4941-8207-4679e4b25b51"
EXPECTED_SPLIT_MANIFEST_SHA256 = (
    "sha256:fb09668f8de77e6baee37dc1e3162a7617ca9a0a450da9ca53422c17bfbc942a"
)
EXPECTED_RANKING = [
    "historical_pass_rate_desc",
    "historical_passes_desc",
    "historical_sessions_desc",
    "task_key_asc",
    "task_version_id_asc",
]
EXPECTED_INTERPRETATION = (
    "historical_strong_model_lineage_ease_proxy_not_target_model_outcome; "
    "execution_uses_the_frozen_split_version"
)
EXPECTED_CYBER_CONTRACT = {
    "evidence_schema": "1.0.0",
    "submission_protocol": "2.0.0",
    "verifier_contract": "3.0.0",
}
EXPECTED_MODELS = {
    "qwen3.8-27b": {
        "repository": "Qwen/Qwen3.8-27B",
        "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "served_id": "qwen3.8-27b",
        # Fleet's session catalog stores the unprefixed ID.  The exact endpoint,
        # model revision, harness and compaction treatment remain separately bound.
        "session_model": "qwen/qwen3.8-27b",
    },
    "glm-5.3": {
        "repository": "zai-org/GLM-5.3",
        "revision": "30333038ada1f1dacb294a93270305a890b50c14",
        "served_id": "glm-5.3",
        "session_model": "z-ai/glm-5.3",
    },
}
EXPECTED_TREATMENT = {
    "harness": "opencode",
    "harness_version": "1.18.27",
    "release_asset_sha256": (
        "sha256:4af5494f9433f59db8c1e344198f0ee72a50c06ec009fb4a8aeab4c2d4abd702"
    ),
    "provider_adapter": "@ai-sdk/openai-compatible",
    "context_management": "opencode_1.18.27_native_compaction_autocontinue_v1",
    "context_window_size": 262144,
    "compaction_headroom_tokens": 20000,
    "max_output_tokens": 32768,
    "max_model_requests": 600,
    "timeout_seconds": 28800,
    "tools": ["bash", "submit_report"],
    "tool_catalog_sha256": (
        "sha256:85fad6bdc3a835bf52a11a99b3387740eb06eb3d1720ad9bb33f3feac215b44a"
    ),
}
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(), object_pairs_hook=_strict_object)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a sha256 digest")
    return value


def _file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _uuid(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a UUID")
    try:
        return str(uuid.UUID(value))
    except ValueError as exc:
        raise ValueError(f"{label} must be a UUID") from exc


def _selection_rank_key(row: dict[str, Any]) -> tuple[float, int, int, str, str]:
    ease = row.get("historical_ease")
    if not isinstance(ease, dict):
        raise ValueError("selection task omits historical ease provenance")
    rate = ease.get("pass_rate")
    passes = ease.get("passes")
    sessions = ease.get("sessions")
    exact = ease.get("exact_task_version")
    if not isinstance(rate, (int, float)) or isinstance(rate, bool) or not 0 <= rate <= 1:
        raise ValueError("selection historical pass rate is invalid")
    if not isinstance(passes, int) or isinstance(passes, bool) or passes < 0:
        raise ValueError("selection historical passes are invalid")
    if not isinstance(sessions, int) or isinstance(sessions, bool) or sessions <= 0:
        raise ValueError("selection historical sessions are invalid")
    if passes > sessions or abs(float(rate) - passes / sessions) > 1e-12:
        raise ValueError("selection historical aggregate is inconsistent")
    if not isinstance(exact, bool):
        raise ValueError("selection exact-version provenance is invalid")
    task_key = row.get("task_key")
    version_id = row.get("task_version_id")
    if not isinstance(task_key, str) or not task_key:
        raise ValueError("selection task key is invalid")
    _uuid(version_id, "selection task version id")
    return (-float(rate), -passes, -sessions, task_key, version_id)


def validate_selection(campaign: dict[str, Any], repo_root: Path) -> list[dict[str, Any]]:
    """Validate all available frozen-selection provenance before returning tasks."""
    binding = campaign.get("selection")
    if not isinstance(binding, dict):
        raise ValueError("campaign selection binding is missing")
    expected_binding = {
        "path": EXPECTED_SELECTION_PATH,
        "file_sha256": EXPECTED_SELECTION_FILE_SHA256,
        "selection_sha256": EXPECTED_SELECTION_SHA256,
        "source_job_id": EXPECTED_SOURCE_JOB_ID,
        "split_manifest_digest": EXPECTED_SPLIT_MANIFEST_SHA256,
        "selected_count": 100,
        "source_aggregate": {"task_count": 160, "session_count": 1280},
        "ranking": EXPECTED_RANKING,
        "interpretation": EXPECTED_INTERPRETATION,
        "prior_qwen_task_versions_excluded": 5,
        "eligibility": {
            "required_cyber_contract": EXPECTED_CYBER_CONTRACT,
            "runtime_seed_overlay_required": True,
            "excluded_count": 8,
        },
    }
    if binding != expected_binding:
        raise ValueError("campaign selection provenance binding drifted")

    relative = Path(binding["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("selection path must be repository-relative")
    selection_path = (repo_root / relative).resolve(strict=True)
    root = repo_root.resolve(strict=True)
    if not selection_path.is_relative_to(root):
        raise ValueError("selection path escapes the repository")
    if _file_sha256(selection_path) != binding["file_sha256"]:
        raise ValueError("selection file digest mismatch")
    selection = read_object(selection_path)
    if selection.get("schema_version") != SELECTION_SCHEMA:
        raise ValueError("selection schema drifted")
    if selection.get("selection_sha256") != crypto.digest_without(selection, "selection_sha256"):
        raise ValueError("selection semantic digest mismatch")
    if selection.get("selection_sha256") != binding["selection_sha256"]:
        raise ValueError("selection semantic binding drifted")

    actual_provenance = {
        "source_job_id": selection.get("source_job_id"),
        "split_manifest_digest": selection.get("split_manifest_digest"),
        "selected_count": selection.get("selected_count"),
        "source_aggregate": selection.get("source_aggregate"),
        "ranking": selection.get("ranking"),
        "interpretation": selection.get("interpretation"),
        "prior_qwen_task_versions_excluded": selection.get("prior_qwen_task_versions_excluded"),
        "eligibility": {
            "required_cyber_contract": (selection.get("self_hosted_eligibility") or {}).get(
                "required_cyber_contract"
            ),
            "runtime_seed_overlay_required": (selection.get("self_hosted_eligibility") or {}).get(
                "runtime_seed_overlay_required"
            ),
            "excluded_count": (selection.get("self_hosted_eligibility") or {}).get(
                "excluded_count"
            ),
        },
    }
    if actual_provenance != {
        key: value
        for key, value in binding.items()
        if key not in {"path", "file_sha256", "selection_sha256"}
    }:
        raise ValueError("selection source/ranking/eligibility/aggregate provenance drifted")

    eligibility = selection.get("self_hosted_eligibility") or {}
    excluded = eligibility.get("excluded")
    if not isinstance(excluded, list) or len(excluded) != 8:
        raise ValueError("selection exclusion evidence is incomplete")
    excluded_versions: set[str] = set()
    allowed_reasons = {"not_verifier_contract_v3", "missing_runtime_seed_overlay"}
    for row in excluded:
        if not isinstance(row, dict) or not isinstance(row.get("historical_rank"), int):
            raise ValueError("selection exclusion row is invalid")
        version_id = _uuid(row.get("task_version_id"), "excluded task version id")
        reasons = row.get("reasons")
        if (
            version_id in excluded_versions
            or not isinstance(reasons, list)
            or not reasons
            or len(reasons) != len(set(reasons))
            or not set(reasons).issubset(allowed_reasons)
        ):
            raise ValueError("selection exclusion evidence is invalid")
        excluded_versions.add(version_id)

    privacy = selection.get("privacy")
    if privacy != {
        "prompts_included": False,
        "transcripts_included": False,
        "tool_content_included": False,
        "verifier_content_included": False,
    }:
        raise ValueError("selection privacy declaration drifted")
    if selection.get("shared_qwen_glm_prefix_count") != 50:
        raise ValueError("selection shared-prefix provenance drifted")

    tasks = selection.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 100:
        raise ValueError("selection must contain exactly 100 tasks")
    task_keys: set[str] = set()
    version_ids: set[str] = set()
    historical_ranks: set[int] = set()
    keys: list[tuple[float, int, int, str, str]] = []
    required_strings = (
        "task_version",
        "env_key",
        "env_version",
        "data_key",
        "data_version",
    )
    for rank, row in enumerate(tasks, 1):
        if not isinstance(row, dict) or row.get("rank") != rank or row.get("split") != "train":
            raise ValueError("selection is not an ordered train-only slate")
        historical_rank = row.get("historical_rank")
        if not isinstance(historical_rank, int) or historical_rank <= 0:
            raise ValueError("selection historical rank is invalid")
        task_key = row.get("task_key")
        version_id = _uuid(row.get("task_version_id"), "selection task version id")
        _uuid(row.get("environment_version_id"), "selection environment version id")
        if (
            task_key in task_keys
            or version_id in version_ids
            or historical_rank in historical_ranks
            or any(
                not isinstance(row.get(field), str) or not row[field] for field in required_strings
            )
        ):
            raise ValueError("selection task identities are incomplete or duplicated")
        task_keys.add(task_key)
        version_ids.add(version_id)
        historical_ranks.add(historical_rank)
        keys.append(_selection_rank_key(row))
    if keys != sorted(keys):
        raise ValueError("selection tasks violate the declared ranking")
    return tasks


def validate_campaign(campaign: dict[str, Any], repo_root: Path) -> list[dict[str, Any]]:
    expected_keys = {
        "schema_version",
        "campaign_id",
        "status",
        "launch_authorized",
        "selection",
        "models",
        "treatment",
        "pass_k",
        "attempts",
        "task_count_per_model",
        "cell_count",
        "execution_generation_policy",
        "expected_universe_sha256",
    }
    if set(campaign) != expected_keys:
        raise ValueError("campaign fields drifted")
    if campaign.get("schema_version") != CAMPAIGN_SCHEMA:
        raise ValueError("campaign schema drifted")
    if campaign.get("campaign_id") != EXPECTED_CAMPAIGN_ID:
        raise ValueError("campaign identity drifted")
    if campaign.get("status") != "HELD" or campaign.get("launch_authorized") is not False:
        raise ValueError("new scientific universe must remain held and unlaunched")
    if campaign.get("models") != EXPECTED_MODELS:
        raise ValueError("campaign model bindings drifted")
    if campaign.get("treatment") != EXPECTED_TREATMENT:
        raise ValueError("campaign treatment binding drifted")
    if campaign.get("pass_k") != 4 or campaign.get("attempts") != [1, 2, 3, 4]:
        raise ValueError("campaign must be pass@4")
    if campaign.get("task_count_per_model") != 100 or campaign.get("cell_count") != 800:
        raise ValueError("campaign count contract drifted")
    policy = campaign.get("execution_generation_policy")
    if policy != {
        "schema_version": EXECUTION_SCHEMA,
        "initial_generation": 1,
        "statistical_cell_identity_excludes_execution_generation": True,
        "one_open_generation_per_cell": True,
        "retry_requires_terminal_pre_model_infrastructure_tombstone": True,
        "model_started_generation_is_nonrepeatable": True,
        "session_created_generation_is_nonrepeatable": True,
        "authoritative_outcome_generation_is_nonrepeatable": True,
        "automatic_retry": False,
    }:
        raise ValueError("campaign execution-generation policy drifted")
    expected_universe = campaign.get("expected_universe_sha256")
    _require_sha256(expected_universe, "expected universe digest")
    return validate_selection(campaign, repo_root)


def _cell_identity(model: str, task: dict[str, Any], attempt: int) -> dict[str, Any]:
    return {
        "schema_version": CELL_SCHEMA,
        "model": model,
        "model_revision": EXPECTED_MODELS[model]["revision"],
        "task_key": task["task_key"],
        "task_version_id": task["task_version_id"],
        # This is the immutable selected-slate rank (1..100), not historical_rank.
        "selection_rank": task["rank"],
        "attempt": attempt,
        "treatment": EXPECTED_TREATMENT,
    }


def execution_for(cell_id: str, generation: int) -> dict[str, Any]:
    if not isinstance(generation, int) or isinstance(generation, bool) or generation < 1:
        raise ValueError("execution generation must be a positive integer")
    identity = {
        "schema_version": EXECUTION_SCHEMA,
        "cell_id": cell_id,
        "execution_generation": generation,
    }
    return {**identity, "execution_id": crypto.sha256(crypto.canonical_json(identity))}


def build_universe(campaign: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    tasks = validate_campaign(campaign, repo_root)
    cells: list[dict[str, Any]] = []
    for model in EXPECTED_MODELS:
        for task in tasks:
            for attempt in campaign["attempts"]:
                identity = _cell_identity(model, task, attempt)
                cell_id = crypto.sha256(crypto.canonical_json(identity))
                cells.append(
                    {
                        **identity,
                        "cell_id": cell_id,
                        "initial_execution": execution_for(cell_id, 1),
                    }
                )
    cell_ids = [row["cell_id"] for row in cells]
    statistical_keys = [(row["model"], row["task_version_id"], row["attempt"]) for row in cells]
    if len(cells) != 800 or len(set(cell_ids)) != 800 or len(set(statistical_keys)) != 800:
        raise ValueError("expanded universe contains missing or duplicate statistical cells")
    body = {
        "schema_version": UNIVERSE_SCHEMA,
        "selection_sha256": EXPECTED_SELECTION_SHA256,
        "cell_count": 800,
        "model_counts": {"qwen3.8-27b": 400, "glm-5.3": 400},
        "task_counts": {"qwen3.8-27b": 100, "glm-5.3": 100},
        "attempts": [1, 2, 3, 4],
        "cells": cells,
        "privacy": {
            "prompts_included": False,
            "transcripts_included": False,
            "scores_included": False,
            "verifier_content_included": False,
        },
    }
    universe = {
        **body,
        "universe_sha256": crypto.sha256(crypto.canonical_json(body)),
    }
    if universe["universe_sha256"] != campaign["expected_universe_sha256"]:
        raise ValueError("expanded universe digest does not match the campaign binding")
    return universe


def validate_tombstone(tombstone: dict[str, Any], cell: dict[str, Any]) -> None:
    """Fail closed unless a generation ended before any stochastic/model side effect."""
    required_keys = {
        "schema_version",
        "cell_id",
        "execution_id",
        "execution_generation",
        "classification",
        "terminal",
        "model_called",
        "verifier_called",
        "session_created",
        "authoritative_outcome_created",
        "retry_allowed",
        "evidence_receipt_sha256",
        "receipt_sha256",
    }
    if set(tombstone) != required_keys:
        raise ValueError("tombstone fields drifted")
    generation = tombstone.get("execution_generation")
    expected_execution = execution_for(cell["cell_id"], generation)
    expected = {
        "schema_version": TOMBSTONE_SCHEMA,
        "cell_id": cell["cell_id"],
        "execution_id": expected_execution["execution_id"],
        "execution_generation": generation,
        "classification": "pre_model_infrastructure_failure",
        "terminal": True,
        "model_called": False,
        "verifier_called": False,
        "session_created": False,
        "authoritative_outcome_created": False,
        "retry_allowed": True,
    }
    for key, value in expected.items():
        if tombstone.get(key) != value:
            raise ValueError("tombstone is not a retry-safe pre-model terminal generation")
    _require_sha256(tombstone.get("evidence_receipt_sha256"), "tombstone evidence receipt")
    if tombstone.get("receipt_sha256") != crypto.digest_without(tombstone, "receipt_sha256"):
        raise ValueError("tombstone digest mismatch")


def next_execution(cell: dict[str, Any], tombstones: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the next execution identity after a contiguous safe tombstone chain."""
    if not tombstones:
        return execution_for(cell["cell_id"], 1)
    for expected_generation, tombstone in enumerate(tombstones, 1):
        validate_tombstone(tombstone, cell)
        if tombstone["execution_generation"] != expected_generation:
            raise ValueError("tombstone execution generations are not contiguous")
    return execution_for(cell["cell_id"], len(tombstones) + 1)


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    universe = build_universe(read_object(args.campaign), args.repo_root)
    if args.summary:
        universe = {key: value for key, value in universe.items() if key != "cells"}
    print(json.dumps(universe, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
