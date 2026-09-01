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
import os
import re
import sys
from pathlib import Path
from typing import Any

import httpx

from evals.fleet import self_hosted

PLAN_SCHEMA = "fleet-qwen-code-prompt-curriculum-plan-v1"
DRY_RUN_SCHEMA = "fleet-qwen-code-prompt-curriculum-dry-run-v1"
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
        row["task_version_id"]: row
        for row in split.get("tasks", [])
        if row.get("split") == "train"
    }
    sealed = {
        row["task_version_id"]
        for row in split.get("tasks", [])
        if row.get("split") != "train"
    }
    selected = plan.get("tasks") or []
    if not 2 <= len(selected) <= 3:
        raise ValueError("prompt-curriculum pilot must select two or three tasks")
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
        rows = [
            row for row in outcomes if row.get("task_version_id") == task["task_version_id"]
        ]
        if len(rows) != EXPECTED_BASELINE_ATTEMPTS:
            raise ValueError("selected task does not have four completed baseline attempts")
        if {row.get("attempt") for row in rows} != {1, 2, 3, 4}:
            raise ValueError("selected task baseline attempt identities drifted")
        if any(
            row.get("task_key") != task["task_key"]
            or row.get("status") != "model_outcome"
            or row.get("agent_termination") != "completed"
            or row.get("session_ingest_status") != "completed"
            or float(row.get("score") or 0) != 0
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
        "family": selected["family"],
        "task_group_name": payload["name"],
        "project_id": plan["project_id"],
        "variant_count": len(members),
        "planned_sessions": len(members),
        "variant_prompt_sha256": {
            member["label"]: self_hosted.sha256(member["prompt"].encode())
            for member in members
        },
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
        "post_create_gates": [
            "re-hydrate every created member version",
            "require exact source environment, data, runtime seed, verifier version, and schema",
            "require one shared task key and four unique prompt hashes",
            "require a create-once task-group receipt before any job submission",
        ],
    }
    receipt["receipt_sha256"] = self_hosted.sha256(self_hosted.canonical_json(receipt))
    return payload, receipt


def _write_create_once(path: Path, value: dict[str, Any], mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists():
        raise FileExistsError(f"refusing to replace existing dry-run artifact: {path}")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(self_hosted.canonical_json(value) + b"\n")
    except Exception:
        path.unlink(missing_ok=True)
        raise


def prepare_live_dry_runs(
    client: httpx.Client,
    plan: dict[str, Any],
    selected: list[dict[str, Any]],
    out_dir: Path,
) -> dict[str, Any]:
    account = self_hosted._request(client, "GET", "/v1/account")
    if account.get("team_name") != "fleet" or account.get("team_id") not in {
        None,
        self_hosted.FLEET_TEAM_ID,
    }:
        raise RuntimeError("FLEET_API_KEY is not scoped to the Fleet team")
    if out_dir.exists():
        raise FileExistsError(f"refusing to reuse prompt-curriculum output directory: {out_dir}")
    out_dir.mkdir(parents=True, mode=0o700)

    receipts = []
    for row in selected:
        task = self_hosted._request(
            client,
            "GET",
            f"/v1/tasks/{row['task_key']}",
            params={"version_id": row["task_version_id"]},
        )
        payload, receipt = build_task_group_payload(plan, row, task)
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
        "receipt_sha256s": [row["receipt_sha256"] for row in receipts],
    }
    summary["summary_sha256"] = self_hosted.sha256(self_hosted.canonical_json(summary))
    _write_create_once(out_dir / "dry-run-summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "prepare"))
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
        summary = prepare_live_dry_runs(client, plan, selected, args.out_dir)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
