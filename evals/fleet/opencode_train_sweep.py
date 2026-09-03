"""Prepare exact train-only OpenCode smokes for the Qwen3.8/GLM5.3 sweep."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any

import httpx

from evals.fleet import holdout, self_hosted

SOURCE_JOB_ID = "a62dd51f-a52b-4941-8207-4679e4b25b51"
SELECTION_SCHEMA = "fleet-opencode-easiest-train100-selection-v1"
HARNESS = {
    "name": "opencode",
    "version": "1.18.27",
    "release_asset_sha256": (
        "sha256:4af5494f9433f59db8c1e344198f0ee72a50c06ec009fb4a8aeab4c2d4abd702"
    ),
    "provider_adapter": "@ai-sdk/openai-compatible",
    "max_model_requests": 600,
    "context_window_size": 262144,
    "max_output_tokens": 32768,
    "timeout_seconds": 28800,
    "context_management": "opencode_1.18.27_native_compaction_no_autocontinue",
}
MODELS = {
    "qwen38": {
        "repository": "Qwen/Qwen3.8-27B",
        "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "served_id": "qwen3.8-27b",
        "endpoint_origin": "https://inference.flt.build",
        "session_model": "fleet-cluster-opencode-1.18.27/qwen3.8-27b",
        "task_count": 50,
        "campaign_id": "chris-cyber-q38-opencode11827-fleet-train50-p4-v1",
        "resource_prefix": "q38oc-tr50-v1",
    },
    "glm53": {
        "repository": "zai-org/GLM-5.3",
        "revision": "30333038ada1f1dacb294a93270305a890b50c14",
        "served_id": "glm-5.3",
        "endpoint_origin": "https://inference.flt.build",
        "session_model": "fleet-cluster-opencode-1.18.27/glm-5.3",
        "task_count": 100,
        "campaign_id": "chris-cyber-glm53-opencode11827-fleet-train100-p4-v1",
        "resource_prefix": "glm53oc-tr100-v1",
    },
}
AUTHORITY = {
    "provisioning_route_template": (
        "/v1/rollout-rewards/{task_key}/versions/{task_version_id}/instances"
    ),
    "scoring_route_template": "/v1/rollout-rewards/{task_key}/versions/{task_version_id}",
    "required_cyber_contract": {
        "submission_protocol": "2.0.0",
        "evidence_schema": "1.0.0",
        "verifier_contract": "3.0.0",
    },
    "scoring_mode": "partial",
    "multi_app_aggregation_mode": "fractional",
}
TOOL_CATALOG_SHA256 = "sha256:85fad6bdc3a835bf52a11a99b3387740eb06eb3d1720ad9bb33f3feac215b44a"
PRIOR_QWEN_TASK_VERSION_IDS = {
    "2b9ba166-6f43-44aa-894d-75761314c150",
    "25b185c5-6aee-479d-97ba-d94ed9df1f42",
    "4a4aba06-38d4-4fc7-83ee-b625fc934d00",
    "53669bab-1389-48cf-9d08-68e411f9cd78",
    "2c47ef51-29f9-4736-9916-18b748059cc6",
}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def digest_without(value: dict[str, Any], field: str) -> str:
    return self_hosted.sha256(
        self_hosted.canonical_json({key: item for key, item in value.items() if key != field})
    )


def build_selection(client: httpx.Client, split: dict[str, Any]) -> dict[str, Any]:
    if (split.get("source") or {}).get("job_id") != SOURCE_JOB_ID:
        raise ValueError("split source job drifted")
    stats = self_hosted._request(client, "GET", f"/v1/sessions/job/{SOURCE_JOB_ID}")
    rows = stats.get("tasks") or []
    if len(rows) != len(split.get("tasks") or []):
        raise RuntimeError("source-job task statistics are incomplete")
    stats_by_version: dict[str, dict[str, Any]] = {}
    stats_by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        task = row.get("task") or {}
        version_id = task.get("eval_task_version_id")
        if not isinstance(version_id, str) or not version_id:
            raise RuntimeError("source-job statistics omit an exact task version")
        scored = row.get("scored_sessions")
        if not isinstance(scored, int) or scored <= 0:
            raise RuntimeError("source-job task has no scored historical sessions")
        if int(row.get("verifier_completed_sessions") or 0) < scored:
            raise RuntimeError("source-job scored sessions lack completed verifier executions")
        stats_by_version[version_id] = row
        task_key = task.get("key")
        if not isinstance(task_key, str) or not task_key or task_key in stats_by_key:
            raise RuntimeError("source-job statistics contain invalid task lineages")
        stats_by_key[task_key] = row
    ranked: list[dict[str, Any]] = []
    for source in split.get("tasks") or []:
        if source.get("split") != "train":
            continue
        if source.get("task_version_id") in PRIOR_QWEN_TASK_VERSION_IDS:
            continue
        stat = stats_by_version.get(source.get("task_version_id"))
        exact_historical_version = stat is not None
        if stat is None:
            stat = stats_by_key.get(source.get("task_key"))
        if stat is None:
            raise RuntimeError("train task has no source-job lineage statistics")
        ranked.append(
            {
                field: source[field]
                for field in (
                    "task_key",
                    "task_version_id",
                    "task_version",
                    "environment_version_id",
                    "env_key",
                    "env_version",
                    "data_key",
                    "data_version",
                    "split",
                )
            }
            | {
                "historical_ease": {
                    "pass_rate": stat["passed_sessions"] / stat["scored_sessions"],
                    "passes": stat["passed_sessions"],
                    "sessions": stat["scored_sessions"],
                    "exact_task_version": exact_historical_version,
                }
            }
        )
    ranked.sort(
        key=lambda row: (
            -float(row["historical_ease"]["pass_rate"]),
            -int(row["historical_ease"]["passes"]),
            -int(row["historical_ease"]["sessions"]),
            row["task_key"],
            row["task_version_id"],
        )
    )
    selected = ranked[:100]
    if len(selected) != 100:
        raise RuntimeError("fewer than 100 train tasks have complete historical evidence")
    for rank, row in enumerate(selected, 1):
        row["rank"] = rank
    receipt = {
        "schema_version": SELECTION_SCHEMA,
        "source_job_id": SOURCE_JOB_ID,
        "split_manifest_digest": split["manifest_digest"],
        "ranking": [
            "historical_pass_rate_desc",
            "historical_passes_desc",
            "historical_sessions_desc",
            "task_key_asc",
            "task_version_id_asc",
        ],
        "interpretation": (
            "historical_strong_model_lineage_ease_proxy_not_target_model_outcome; "
            "execution_uses_the_frozen_split_version"
        ),
        "selected_count": 100,
        "shared_qwen_glm_prefix_count": 50,
        "prior_qwen_task_versions_excluded": len(PRIOR_QWEN_TASK_VERSION_IDS),
        "tasks": selected,
        "privacy": {
            "prompts_included": False,
            "transcripts_included": False,
            "tool_content_included": False,
            "verifier_content_included": False,
        },
        "source_aggregate": {
            "task_count": len(rows),
            "session_count": stats.get("total_sessions"),
        },
    }
    receipt["selection_sha256"] = digest_without(receipt, "selection_sha256")
    return receipt


def validate_selection(selection: dict[str, Any], split: dict[str, Any]) -> list[dict[str, Any]]:
    if selection.get("schema_version") != SELECTION_SCHEMA:
        raise ValueError("unsupported sweep selection schema")
    if selection.get("selection_sha256") != digest_without(selection, "selection_sha256"):
        raise ValueError("sweep selection digest mismatch")
    if selection.get("split_manifest_digest") != split.get("manifest_digest"):
        raise ValueError("sweep selection split binding drifted")
    tasks = selection.get("tasks") or []
    if len(tasks) != 100 or len({row.get("task_version_id") for row in tasks}) != 100:
        raise ValueError("sweep selection must contain 100 unique task versions")
    split_by_version = {row["task_version_id"]: row for row in split.get("tasks") or []}
    for rank, row in enumerate(tasks, 1):
        source = split_by_version.get(row.get("task_version_id"))
        if row.get("rank") != rank or source is None or source.get("split") != "train":
            raise ValueError("sweep selection is not an ordered train-only slate")
        if row.get("task_key") != source.get("task_key"):
            raise ValueError("sweep task lineage drifted")
    return tasks


def duplicate_preflight(
    client: httpx.Client, rows: list[dict[str, Any]], session_model: str
) -> dict[str, Any]:
    observations = []
    for row in rows:
        offset = 0
        exact = 0
        scanned = 0
        pages = 0
        while True:
            payload = self_hosted._request(
                client,
                "GET",
                "/v1/sessions",
                params={"task_key": row["task_key"], "limit": 500, "offset": offset},
            )
            sessions = payload.get("sessions") or []
            pages += 1
            scanned += len(sessions)
            exact += sum(1 for session in sessions if session.get("model") == session_model)
            if payload.get("has_more") is False:
                break
            if not sessions:
                raise RuntimeError("duplicate inventory pagination made no progress")
            offset += len(sessions)
        if exact:
            raise RuntimeError("equivalent OpenCode treatment sessions already exist")
        observations.append(
            {
                "task_key": row["task_key"],
                "task_version_id": row["task_version_id"],
                "pages": pages,
                "sessions_scanned": scanned,
                "equivalent_treatment_sessions": 0,
            }
        )
    receipt = {
        "route": "/v1/sessions?task_key=<task-lineage-key>",
        "scope": "team_non_archived_sessions",
        "session_model": session_model,
        "queries": observations,
        "transcripts_read": False,
        "session_ids_persisted": False,
    }
    receipt["receipt_sha256"] = digest_without(receipt, "receipt_sha256")
    return receipt


def build_smoke_config(
    client: httpx.Client,
    selection: dict[str, Any],
    split: dict[str, Any],
    model_key: str,
) -> dict[str, Any]:
    model = copy.deepcopy(MODELS[model_key])
    tasks = validate_selection(selection, split)[: int(model.pop("task_count"))]
    row = holdout._task_receipt(client, tasks[0], 1)
    if row.get("cyber_contract") != AUTHORITY["required_cyber_contract"]:
        raise RuntimeError("smoke task is not exact Verifier Contract v3")
    # The paid smoke touches only the first task. The full expansion performs
    # a fresh exhaustive inventory over every selected task immediately before
    # those additional attempts are created.
    duplicate = duplicate_preflight(client, tasks[:1], model["session_model"])
    account = self_hosted._request(client, "GET", "/v1/account")
    if account.get("team_id") != self_hosted.FLEET_TEAM_ID or account.get("team_name") != "fleet":
        raise RuntimeError("FLEET_API_KEY is not scoped to the Fleet team")
    self_hosted.assert_authoritative_routes_deployed(client, {"authority": AUTHORITY})
    model_identity = holdout.live_model_identity(
        os.environ["FLEET_API_KEY"],
        {
            **model,
            "live_identity": {
                "catalog": {
                    "id": model["served_id"],
                    "model_revision": model["revision"],
                    "routed": True,
                    "status": "ready",
                },
                "model_info": {
                    "model_path": f"/scratch/models/{model['served_id']}/{model['revision']}"
                },
                "server_info": {
                    "model_path": f"/scratch/models/{model['served_id']}/{model['revision']}",
                    "served_model_name": model["served_id"],
                    "context_length": HARNESS["context_window_size"],
                },
            },
        },
    )
    key_digest = self_hosted.sha256(row["task_key"].encode()).split(":", 1)[1][:8]
    config = {
        "schema_version": "fleet-selfhosted-opencode-smoke-v1",
        "run_id": f"{model['campaign_id']}-smoke-r1-{key_digest}",
        "campaign_id": model["campaign_id"],
        "source_job_id": SOURCE_JOB_ID,
        "task": {
            "key": row["task_key"],
            "version_id": row["task_version_id"],
            "prompt_sha256": row["prompt_sha256"],
            "env_variables_sha256": row["env_variables_sha256"],
            "output_json_schema_sha256": row["output_json_schema_sha256"],
            "cyber_contract": row["cyber_contract"],
        },
        "environment": {
            "id": row["env_key"],
            "version": row["env_version"],
            "version_id": row["environment_version_id"],
            "data_id": row["data_key"],
            "data_version": row["data_version"],
            "runtime_seed_content_sha256": row["runtime_seed_content_sha256"],
            "ttl_seconds": 32400,
        },
        "verifier": row["verifier"],
        "authority": copy.deepcopy(AUTHORITY),
        "model": model,
        "harness": copy.deepcopy(HARNESS),
        "execution": {
            "pass_k": 1,
            "planned_full_pass_k": 4,
            "max_concurrent": 1,
            "network": f"{model['resource_prefix']}-smoke-{key_digest}",
            "training_data_eligible": True,
            "required_task_tools": ["bash", "submit_report"],
            "required_task_tool_catalog_sha256": TOOL_CATALOG_SHA256,
        },
        "preflight": {
            "fleet_account": {"team_name": account["team_name"], "team_id": account["team_id"]},
            "duplicate_receipt": duplicate,
            "live_model_identity": model_identity,
            "selection_sha256": selection["selection_sha256"],
            "selected_task_count": len(tasks),
        },
    }
    config["config_sha256"] = digest_without(config, "config_sha256")
    return config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("select", "smoke-config"))
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--model", choices=sorted(MODELS))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    key = os.environ.get("FLEET_API_KEY")
    if not key:
        raise RuntimeError("FLEET_API_KEY is required")
    split = load_json(args.split)
    with httpx.Client(
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        timeout=1800,
    ) as client:
        if args.command == "select":
            value = build_selection(client, split)
        else:
            if not args.selection or not args.model:
                parser.error("smoke-config requires --selection and --model")
            value = build_smoke_config(client, load_json(args.selection), split, args.model)
    self_hosted.write_json_once(args.output, value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
