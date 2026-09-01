"""Fail-closed preparation for a train-only Qwen prompt-curriculum pilot.

The module never creates a Fleet task, task group, job, or environment.  It
validates an immutable pilot plan, proves the selected rows are training-only
and were clean zero-reward model outcomes in the cited calibration snapshot,
then (only with a Fleet credential) hydrates each exact source task and writes a
private task-group *dry-run* payload plus a prompt-free review receipt.

Prompt bodies and task metadata stay in a mode-0600 file below an already
ignored output root.  Only hashes and non-secret binding identities are safe to
review or commit.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from evals.fleet import self_hosted

PLAN_SCHEMA = "fleet-qwen-code-prompt-curriculum-plan-v1"
DRY_RUN_SCHEMA = "fleet-qwen-code-prompt-curriculum-dry-run-v1"
REVIEW_PLAN_SCHEMA = "fleet-qwen-code-prompt-curriculum-review-plan-v1"
REQUEST_AUDIT_SCHEMA = "fleet-read-only-request-audit-v1"
EXPECTED_SPLIT_SCHEMA = "fleet_rl_task_split_v1"
EXPECTED_MODEL_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
EXPECTED_VARIANT_COUNT = 4
EXPECTED_BASELINE_ATTEMPTS = 4
MAX_PLANNED_SESSIONS_PER_GROUP = 6

# These are intentionally generic, cumulative process cues.  They disclose no
# task-specific route, payload, object, credential, flag, or verifier behavior.
CUE_LIBRARY = {
    "bounded-surface-map": (
        "Work methodically. Start with a small number of read-only requests to map the "
        "challenge application's documented and discoverable public surface. Compare response "
        "status, shape, and authorization behavior before choosing the next probe."
    ),
    "boundary-comparison": (
        "Prioritize boundaries where an identifier, object reference, filter, or user-controlled "
        "value may be trusted more than intended. Change one input at a time, keep probes bounded, "
        "and use observed response differences rather than broad guessing."
    ),
    "evidence-to-report": (
        "When you obtain evidence of unintended access, follow that narrow path only far enough to "
        "recover the required marker, then submit it with a concise explanation. Avoid brute "
        "force, destructive changes, high-volume collection, or probing outside the challenge."
    ),
}

_FORBIDDEN_CUE_PATTERNS = (
    re.compile(r"https?://", re.IGNORECASE),
    re.compile(r"FLAG\{", re.IGNORECASE),
    re.compile(r"\bcurl\b", re.IGNORECASE),
    re.compile(r"/api/", re.IGNORECASE),
    re.compile(r"\b(?:select|insert|update|delete)\s+.+\b(?:from|into|set)\b", re.IGNORECASE),
    re.compile(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bverifier\b", re.IGNORECASE),
)

_TASK_SPEC_FIELDS = (
    "key",
    "environment_id",
    "verifier_id",
    "version",
    "env_variables",
    "metadata",
    "data_id",
    "data_version",
    "multi_app_seed_versions",
    "output_json_schema",
    "factual_answer",
    "task_modality",
    "task_scenario_id",
    "task_lifecycle_status",
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _digest_without(value: dict[str, Any], field: str) -> str:
    return self_hosted.sha256(
        self_hosted.canonical_json({key: item for key, item in value.items() if key != field})
    )


def validate_generic_cues() -> None:
    if set(CUE_LIBRARY) != {
        "bounded-surface-map",
        "boundary-comparison",
        "evidence-to-report",
    }:
        raise ValueError("generic cue library identity drifted")
    for cue_id, cue in CUE_LIBRARY.items():
        if not cue.strip() or any(pattern.search(cue) for pattern in _FORBIDDEN_CUE_PATTERNS):
            raise ValueError(f"cue {cue_id!r} contains task-specific or privileged material")


def validate_plan(
    plan: dict[str, Any], split: dict[str, Any], campaign_state: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    validate_generic_cues()
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unsupported prompt-curriculum plan schema")
    if plan.get("plan_sha256") != _digest_without(plan, "plan_sha256"):
        raise ValueError("prompt-curriculum plan digest mismatch")
    if split.get("schema") != EXPECTED_SPLIT_SCHEMA:
        raise ValueError("unsupported task split schema")
    if split.get("manifest_digest") != _digest_without(split, "manifest_digest"):
        raise ValueError("task split manifest digest mismatch")
    if plan.get("split_manifest_digest") != split.get("manifest_digest"):
        raise ValueError("prompt-curriculum plan does not pin this split")
    if (plan.get("model") or {}).get("revision") != EXPECTED_MODEL_REVISION:
        raise ValueError("prompt-curriculum model revision drifted")

    execution = plan.get("execution") or {}
    if execution != {
        "pass_k": 1,
        "max_concurrent": 1,
        "required_task_tools": ["bash", "submit_report"],
        "training_data_eligible": False,
    }:
        raise ValueError("prompt-curriculum execution controls drifted")
    target = plan.get("target_success_rate") or {}
    if target != {"minimum": 0.2, "maximum": 0.7, "selection": "least_revealing_in_band"}:
        raise ValueError("prompt-curriculum target band drifted")

    variants = plan.get("variants") or []
    if len(variants) != EXPECTED_VARIANT_COUNT:
        raise ValueError("prompt curriculum must contain exactly four variants")
    expected_ladders = [
        [],
        ["bounded-surface-map"],
        ["bounded-surface-map", "boundary-comparison"],
        ["bounded-surface-map", "boundary-comparison", "evidence-to-report"],
    ]
    if [row.get("cue_ids") for row in variants] != expected_ladders:
        raise ValueError("prompt curriculum must use the reviewed cumulative cue ladder")
    if [row.get("id") for row in variants] != ["level-0", "level-1", "level-2", "level-3"]:
        raise ValueError("prompt-curriculum variant ids drifted")

    train = {
        row["task_version_id"]: row for row in split.get("tasks", []) if row.get("split") == "train"
    }
    sealed = {
        row["task_version_id"] for row in split.get("tasks", []) if row.get("split") != "train"
    }
    selected = plan.get("tasks") or []
    if len(selected) != 2:
        raise ValueError("prompt-curriculum v1 pilot must select exactly two tasks")
    version_ids = [row.get("task_version_id") for row in selected]
    if len(version_ids) != len(set(version_ids)):
        raise ValueError("prompt-curriculum task versions must be unique")
    if set(version_ids) & sealed:
        raise ValueError("prompt curriculum selected a sealed dev/test task")
    if len({row.get("family") for row in selected}) != len(selected):
        raise ValueError("prompt-curriculum pilot must use distinct app families")

    rows: list[dict[str, Any]] = []
    for selected_row in selected:
        split_row = train.get(selected_row.get("task_version_id"))
        if split_row is None:
            raise ValueError("prompt curriculum selected an unknown or non-train task version")
        for field in (
            "task_key",
            "task_version_id",
            "task_version",
            "environment_version_id",
            "env_key",
            "env_version",
            "data_key",
            "data_version",
        ):
            if selected_row.get(field) != split_row.get(field):
                raise ValueError(f"prompt-curriculum exact task binding drifted: {field}")
        baseline = selected_row.get("baseline_evidence") or {}
        if baseline != {
            "attempts": EXPECTED_BASELINE_ATTEMPTS,
            "positive_rewards": 0,
            "classification": "valid_model_outcomes",
        }:
            raise ValueError("selected task lacks the reviewed zero-reward baseline evidence")
        rows.append(copy.deepcopy(selected_row))

    planned_sessions = len(variants)
    if planned_sessions > MAX_PLANNED_SESSIONS_PER_GROUP:
        raise ValueError("prompt-curriculum task-group job exceeds the six-session cap")
    if campaign_state is not None:
        _validate_campaign_state(plan, rows, campaign_state)
    return rows


def _validate_campaign_state(
    plan: dict[str, Any], selected: list[dict[str, Any]], state: dict[str, Any]
) -> None:
    baseline_campaign = plan.get("baseline_campaign") or {}
    if self_hosted.sha256(self_hosted.canonical_json(state)) != baseline_campaign.get(
        "state_sha256"
    ):
        raise ValueError("baseline campaign state digest drifted")
    if state.get("campaign_id") != baseline_campaign.get("campaign_id"):
        raise ValueError("baseline campaign identity drifted")
    if state.get("planned_sessions") != 96:
        raise ValueError("baseline campaign planned-session count drifted")
    outcomes = state.get("outcomes") or []
    for task in selected:
        rows = [row for row in outcomes if row.get("task_version_id") == task["task_version_id"]]
        if len(rows) != EXPECTED_BASELINE_ATTEMPTS:
            raise ValueError("selected task does not have four completed baseline attempts")
        if {row.get("attempt") for row in rows} != {1, 2, 3, 4}:
            raise ValueError("selected task baseline attempt identities drifted")

        def exact_zero_score(row: dict[str, Any]) -> bool:
            score = row.get("score")
            return (
                "score" in row
                and not isinstance(score, bool)
                and isinstance(score, (int, float))
                and math.isfinite(score)
                and float(score) == 0.0
            )

        if any(
            row.get("task_key") != task["task_key"]
            or row.get("status") != "model_outcome"
            or row.get("agent_termination") != "completed"
            or row.get("session_ingest_status") != "completed"
            or not exact_zero_score(row)
            or not row.get("session_id")
            or not row.get("verifier_execution_id")
            for row in rows
        ):
            raise ValueError("selected task baseline evidence is incomplete or nonzero")


def render_variant(base_prompt: str, cue_ids: list[str]) -> str:
    if not cue_ids:
        return base_prompt
    cues = "\n\n".join(CUE_LIBRARY[cue_id] for cue_id in cue_ids)
    return f"{base_prompt.rstrip()}\n\nAdditional testing guidance:\n{cues}\n"


def _task_spec(task: dict[str, Any]) -> dict[str, Any]:
    spec = {field: copy.deepcopy(task.get(field)) for field in _TASK_SPEC_FIELDS}
    spec["key"] = task.get("key")
    return {key: value for key, value in spec.items() if value is not None}


def _exact_registry_source(metadata: dict[str, Any]) -> dict[str, Any]:
    source = metadata.get("task_graph_source")
    if not isinstance(source, dict):
        raise RuntimeError("exact source task has no Artifact Registry task-graph locator")
    artifact_key = source.get("artifact_key")
    version_index = source.get("version_index")
    manifest_sha256 = source.get("manifest_sha256")
    if (
        not isinstance(artifact_key, str)
        or not artifact_key.startswith("cyber/task-graphs/")
        or isinstance(version_index, bool)
        or not isinstance(version_index, int)
        or version_index < 0
        or not isinstance(manifest_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", manifest_sha256) is None
    ):
        raise RuntimeError("Artifact Registry task-graph locator is incomplete or unpinned")
    return {
        "artifact_key": artifact_key,
        "version_index": version_index,
        "manifest_sha256": manifest_sha256,
    }


def build_task_group_payload(
    plan: dict[str, Any], selected: dict[str, Any], task: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    verifier = task.get("verifier") or {}
    metadata = task.get("metadata") or {}
    registry_source = _exact_registry_source(metadata)
    live_binding = {
        "task_key": task.get("key"),
        "env_key": task.get("environment_id"),
        "env_version": task.get("version"),
        "data_key": task.get("data_id"),
        "data_version": task.get("data_version"),
        "prompt_sha256": self_hosted.sha256((task.get("prompt") or "").encode()),
        "env_variables_sha256": self_hosted.sha256(
            self_hosted.canonical_json(task.get("env_variables") or {})
        ),
        "output_json_schema_sha256": self_hosted.sha256(
            self_hosted.canonical_json(task.get("output_json_schema"))
        ),
        "runtime_seed_content_sha256": (metadata.get("runtime_seed_manifest") or {}).get(
            "content_sha256"
        ),
        "verifier_id": task.get("verifier_id"),
        "verifier_version_id": verifier.get("verifier_version_id"),
        "verifier_version": verifier.get("version"),
        "verifier_sha256": verifier.get("sha256"),
    }
    expected = selected["source_receipt"]
    if live_binding != expected:
        raise RuntimeError("live source task no longer matches the frozen calibration receipt")

    prompt = task.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise RuntimeError("exact source task has no prompt")
    members = [
        {
            "label": variant["id"],
            "prompt": render_variant(prompt, variant["cue_ids"]),
        }
        for variant in plan["variants"]
    ]
    payload = {
        "project_id": plan["project_id"],
        "name": f"{plan['campaign_id']}-{selected['family']}",
        "task": _task_spec(task),
        "members": members,
    }
    non_prompt_task = copy.deepcopy(payload["task"])
    non_prompt_task.pop("prompt", None)
    receipt = {
        "schema_version": DRY_RUN_SCHEMA,
        "campaign_id": plan["campaign_id"],
        "task_key": selected["task_key"],
        "source_task_version_id": selected["task_version_id"],
        "exact_version_bindings": {
            "task": {
                "key": selected["task_key"],
                "version": selected["task_version"],
                "version_id": selected["task_version_id"],
            },
            "environment": {
                "key": selected["env_key"],
                "version": selected["env_version"],
                "version_id": selected["environment_version_id"],
            },
            "data": {
                "key": selected["data_key"],
                "version": selected["data_version"],
            },
            "verifier": {
                "id": live_binding["verifier_id"],
                "version": live_binding["verifier_version"],
                "version_id": live_binding["verifier_version_id"],
                "sha256": live_binding["verifier_sha256"],
            },
            "runtime_seed_content_sha256": live_binding["runtime_seed_content_sha256"],
        },
        "family": selected["family"],
        "task_group_name": payload["name"],
        "project_id": plan["project_id"],
        "variant_count": len(members),
        "planned_sessions": len(members),
        "variant_prompt_sha256": {
            member["label"]: self_hosted.sha256(member["prompt"].encode()) for member in members
        },
        "variant_ladder": [
            {
                "id": variant["id"],
                "cue_ids": list(variant["cue_ids"]),
                "prompt_sha256": self_hosted.sha256(member["prompt"].encode()),
            }
            for variant, member in zip(plan["variants"], members, strict=True)
        ],
        "source_binding": live_binding,
        "registry_task_graph_source": registry_source,
        "non_prompt_task_spec_sha256": self_hosted.sha256(
            self_hosted.canonical_json(non_prompt_task)
        ),
        "private_task_group_payload_sha256": self_hosted.sha256(
            self_hosted.canonical_json(payload)
        ),
        "task_group_create_performed": False,
        "paid_job_submitted": False,
        "prepared_payload_invariants": {
            "scope": "pre_create_task_group_payload",
            "member_override_fields": ["prompt"],
            "environment_unchanged": True,
            "runtime_seed_unchanged": True,
            "data_unchanged": True,
            "atoms_unchanged": True,
            "verifier_unchanged": True,
            "flags_unchanged": True,
            "evidence": {
                "exact_source_task_version_id": selected["task_version_id"],
                "exact_task_graph_source": registry_source,
                "source_binding": live_binding,
                "non_prompt_task_spec_sha256": self_hosted.sha256(
                    self_hosted.canonical_json(non_prompt_task)
                ),
            },
            "created_member_hydration_pending": True,
        },
        "post_create_gates": [
            "re-hydrate every created member version",
            "require exact source environment, data, runtime seed, verifier version, and schema",
            "require one shared task key and four unique prompt hashes",
            "require a create-once task-group receipt before any job submission",
        ],
    }
    receipt["receipt_sha256"] = self_hosted.sha256(self_hosted.canonical_json(receipt))
    return payload, receipt


def _expected_gets(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"method": "GET", "path": "/v1/account", "params": {}},
        *[
            {
                "method": "GET",
                "path": f"/v1/tasks/{row['task_key']}",
                "params": {"version_id": row["task_version_id"]},
            }
            for row in selected
        ],
    ]


class _PredeclaredGetClient:
    """Allow exactly one ordered pass over a frozen set of safe GETs."""

    def __init__(self, delegate: Any, expected: list[dict[str, Any]]) -> None:
        self._delegate = delegate
        self._expected = copy.deepcopy(expected)
        self._observed: list[dict[str, Any]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> Any:
        parsed = urlsplit(url)
        if parsed.query or parsed.fragment:
            raise RuntimeError("prompt-curriculum preparation attempted an undeclared request")
        actual = {
            "method": method,
            "origin": f"{parsed.scheme}://{parsed.netloc}",
            "path": parsed.path,
            "params": copy.deepcopy(kwargs.get("params") or {}),
        }
        expected = (
            {**self._expected[len(self._observed)], "origin": self_hosted.ORCHESTRATOR}
            if len(self._observed) < len(self._expected)
            else None
        )
        if actual != expected:
            raise RuntimeError("prompt-curriculum preparation attempted an undeclared request")
        if method != "GET":
            raise RuntimeError("prompt-curriculum preparation attempted a mutation")
        self._observed.append({key: actual[key] for key in ("method", "path", "params")})
        return self._delegate.request(method, url, **kwargs)

    def receipt(self) -> dict[str, Any]:
        if self._observed != self._expected:
            raise RuntimeError("prompt-curriculum preparation did not complete every declared GET")
        receipt = {
            "schema_version": REQUEST_AUDIT_SCHEMA,
            "orchestrator_origin": self_hosted.ORCHESTRATOR,
            "expected_request_count": len(self._expected),
            "observed_request_count": len(self._observed),
            "mutation_request_count": 0,
            "requests": copy.deepcopy(self._observed),
        }
        receipt["request_audit_sha256"] = self_hosted.sha256(self_hosted.canonical_json(receipt))
        return receipt


def _hydrate_dry_run_material(
    client: Any,
    plan: dict[str, Any],
    selected: list[dict[str, Any]],
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], dict[str, Any]]:
    audited = _PredeclaredGetClient(client, _expected_gets(selected))
    account = self_hosted._request(audited, "GET", "/v1/account")
    if account.get("team_name") != "fleet" or account.get("team_id") not in {
        None,
        self_hosted.FLEET_TEAM_ID,
    }:
        raise RuntimeError("FLEET_API_KEY is not scoped to the Fleet team")

    material = []
    for row in selected:
        task = self_hosted._request(
            audited,
            "GET",
            f"/v1/tasks/{row['task_key']}",
            params={"version_id": row["task_version_id"]},
        )
        material.append(build_task_group_payload(plan, row, task))
    return material, audited.receipt()


def build_review_plan(
    plan: dict[str, Any],
    receipts: list[dict[str, Any]],
    request_audit: dict[str, Any],
) -> dict[str, Any]:
    if len(receipts) != len(plan["tasks"]):
        raise ValueError("review plan task count does not match the frozen plan")
    if request_audit.get("schema_version") != REQUEST_AUDIT_SCHEMA or request_audit.get(
        "request_audit_sha256"
    ) != _digest_without(request_audit, "request_audit_sha256"):
        raise ValueError("review plan request audit is malformed")
    if (
        request_audit.get("requests") != _expected_gets(plan["tasks"])
        or request_audit.get("orchestrator_origin") != self_hosted.ORCHESTRATOR
        or request_audit.get("expected_request_count") != 3
        or request_audit.get("observed_request_count") != 3
        or request_audit.get("mutation_request_count") != 0
    ):
        raise ValueError("review plan request audit does not prove the exact GET-only contract")
    for selected, receipt in zip(plan["tasks"], receipts, strict=True):
        registry_source = receipt.get("registry_task_graph_source")
        expected_exact_bindings = {
            "task": {
                "key": selected["task_key"],
                "version": selected["task_version"],
                "version_id": selected["task_version_id"],
            },
            "environment": {
                "key": selected["env_key"],
                "version": selected["env_version"],
                "version_id": selected["environment_version_id"],
            },
            "data": {"key": selected["data_key"], "version": selected["data_version"]},
            "verifier": {
                "id": selected["source_receipt"]["verifier_id"],
                "version": selected["source_receipt"]["verifier_version"],
                "version_id": selected["source_receipt"]["verifier_version_id"],
                "sha256": selected["source_receipt"]["verifier_sha256"],
            },
            "runtime_seed_content_sha256": selected["source_receipt"][
                "runtime_seed_content_sha256"
            ],
        }
        invariant = receipt.get("prepared_payload_invariants") or {}
        if (
            receipt.get("schema_version") != DRY_RUN_SCHEMA
            or receipt.get("receipt_sha256") != _digest_without(receipt, "receipt_sha256")
            or receipt.get("campaign_id") != plan["campaign_id"]
            or receipt.get("task_key") != selected["task_key"]
            or receipt.get("source_task_version_id") != selected["task_version_id"]
            or receipt.get("variant_count") != EXPECTED_VARIANT_COUNT
            or receipt.get("planned_sessions") != EXPECTED_VARIANT_COUNT
            or receipt.get("task_group_create_performed") is not False
            or receipt.get("paid_job_submitted") is not False
            or receipt.get("source_binding") != selected["source_receipt"]
            or receipt.get("exact_version_bindings") != expected_exact_bindings
            or _exact_registry_source({"task_graph_source": registry_source}) != registry_source
            or [
                {"id": row.get("id"), "cue_ids": row.get("cue_ids")}
                for row in receipt.get("variant_ladder") or []
            ]
            != plan["variants"]
            or invariant.get("member_override_fields") != ["prompt"]
            or invariant.get("scope") != "pre_create_task_group_payload"
            or invariant.get("created_member_hydration_pending") is not True
            or invariant.get("evidence")
            != {
                "exact_source_task_version_id": selected["task_version_id"],
                "exact_task_graph_source": registry_source,
                "source_binding": selected["source_receipt"],
                "non_prompt_task_spec_sha256": receipt.get("non_prompt_task_spec_sha256"),
            }
            or any(
                invariant.get(field) is not True
                for field in (
                    "environment_unchanged",
                    "runtime_seed_unchanged",
                    "data_unchanged",
                    "atoms_unchanged",
                    "verifier_unchanged",
                    "flags_unchanged",
                )
            )
        ):
            raise ValueError("review plan contains a mismatched or malformed task receipt")
    review = {
        "schema_version": REVIEW_PLAN_SCHEMA,
        "campaign_id": plan["campaign_id"],
        "study_role": plan["study_role"],
        "baseline_campaign": copy.deepcopy(plan["baseline_campaign"]),
        "model": copy.deepcopy(plan["model"]),
        "harness": copy.deepcopy(plan["harness"]),
        "execution": copy.deepcopy(plan["execution"]),
        "target_success_rate": copy.deepcopy(plan["target_success_rate"]),
        "request_audit": copy.deepcopy(request_audit),
        "tasks": copy.deepcopy(receipts),
        "task_group_create_performed": False,
        "paid_job_submitted": False,
        "registry_source_published": False,
        "pipeline_lane_touched": False,
    }
    review["review_plan_sha256"] = self_hosted.sha256(self_hosted.canonical_json(review))
    return review


def _write_create_once(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists():
        raise FileExistsError(f"refusing to replace existing dry-run artifact: {path}")
    self_hosted.write_json_once(path, value)


def prepare_live_dry_runs(
    client: httpx.Client,
    plan: dict[str, Any],
    selected: list[dict[str, Any]],
    out_dir: Path,
) -> dict[str, Any]:
    material, request_audit = _hydrate_dry_run_material(client, plan, selected)
    if out_dir.exists():
        raise FileExistsError(f"refusing to reuse prompt-curriculum output directory: {out_dir}")
    out_dir.mkdir(parents=True, mode=0o700)

    receipts = []
    for row, (payload, receipt) in zip(selected, material, strict=True):
        task_dir = out_dir / row["family"]
        _write_create_once(task_dir / "private-task-group-payload.json", payload)
        _write_create_once(task_dir / "dry-run-receipt.json", receipt)
        receipts.append(receipt)

    summary = {
        "schema_version": DRY_RUN_SCHEMA,
        "campaign_id": plan["campaign_id"],
        "task_count": len(receipts),
        "planned_paid_sessions": sum(row["planned_sessions"] for row in receipts),
        "task_group_create_performed": False,
        "paid_job_submitted": False,
        "request_audit": request_audit,
        "receipt_sha256s": [row["receipt_sha256"] for row in receipts],
    }
    summary["summary_sha256"] = self_hosted.sha256(self_hosted.canonical_json(summary))
    _write_create_once(out_dir / "dry-run-summary.json", summary)
    return summary


def prepare_live_review_plan(
    client: Any,
    plan: dict[str, Any],
    selected: list[dict[str, Any]],
    out_dir: Path,
) -> dict[str, Any]:
    """Hydrate exact source tasks but persist no prompt-bearing payload."""
    material, request_audit = _hydrate_dry_run_material(client, plan, selected)
    review = build_review_plan(plan, [receipt for _, receipt in material], request_audit)
    if out_dir.exists():
        raise FileExistsError(f"refusing to reuse prompt-curriculum output directory: {out_dir}")
    out_dir.mkdir(parents=True, mode=0o700)
    _write_create_once(out_dir / "review-plan.json", review)
    return review


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "prepare", "prepare-review"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--campaign-state", type=Path)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    plan, split = load_json(args.config), load_json(args.split)
    state = load_json(args.campaign_state) if args.campaign_state else None
    selected = validate_plan(plan, split, state)
    if args.command == "validate":
        print(
            json.dumps(
                {
                    "ok": True,
                    "campaign_id": plan["campaign_id"],
                    "task_count": len(selected),
                    "variant_count_per_task": len(plan["variants"]),
                    "planned_sessions_per_group": len(plan["variants"]),
                    "paid_job_submitted": False,
                },
                sort_keys=True,
            )
        )
        return 0

    if state is None:
        raise ValueError("prepare requires --campaign-state")
    if args.out_dir is None:
        raise ValueError("prepare requires --out-dir")
    api_key = os.environ.get("FLEET_API_KEY")
    if not api_key:
        raise RuntimeError("FLEET_API_KEY is required for live read-only preparation")
    with httpx.Client(
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=180,
    ) as client:
        if args.command == "prepare-review":
            summary = prepare_live_review_plan(client, plan, selected, args.out_dir)
        else:
            summary = prepare_live_dry_runs(client, plan, selected, args.out_dir)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
