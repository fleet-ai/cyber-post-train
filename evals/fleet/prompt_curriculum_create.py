"""GET-only, fail-closed preflight for the Qwen3.6 prompt curriculum.

This module deliberately contains no mutation request.  It re-hydrates the two
exact source tasks, regenerates the private prompt-mode task-group payloads in
memory, checks their hashes against the reviewed prompt-free plan, and proves
that no job matches either deterministic launch name.  Its output is a
prompt-free creation claim for later human review.

The public task-group API has no list-by-name or create-idempotency contract.
Consequently, a later creator must write the claim durably before its one POST,
must save the response before doing anything else, and must never retry an
ambiguous POST.  This module prepares that contract; it does not execute it.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import NAMESPACE_URL, uuid5

from evals.fleet import client as fleet_client
from evals.fleet import prompt_curriculum, self_hosted
from evals.fleet.models import DEFAULT_SALES_PRODUCT_ID

CREATE_PREFLIGHT_SCHEMA = "fleet-qwen-code-prompt-curriculum-create-preflight-v1"
CREATE_CLAIM_SCHEMA = "fleet-qwen-code-prompt-curriculum-create-claim-v1"
HYDRATION_RECEIPT_SCHEMA = "fleet-qwen-code-prompt-curriculum-hydration-v1"
JOB_INVENTORY_OBSERVATION_SCHEMA = "fleet-qwen-code-job-inventory-observation-v1"
EXPECTED_RUNTIME_MODEL = "qwen/qwen3-6-27b-cyber-baseline"
EXPECTED_HARNESS = "qwen-code"
EXPECTED_RUNTIME_LABEL = "qwen-code-0.22.3"
EXPECTED_CAMPAIGN_ID = "chris-cyber-q36-prompt-curriculum-pilot-v1"
EXPECTED_PROJECT_ID = "63d6fda8-48c4-4726-9ec3-d1028f2c47f5"
EXPECTED_MODEL_IDENTITY = {
    "repository": "Qwen/Qwen3.6-27B",
    "revision": "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
    "served_id": "qwen3-6-27b-cyber-baseline",
    "weights_manifest_sha256": (
        "sha256:14ad10368de9b9e5974ff12a4b70ea7884194b58e670177bbac79daeb81f16b9"
    ),
    "tokenizer_revision": (
        "sha256:27f02770d03b60343350ad948bf1065673968c60274f1e34f4afaed16940a1ea"
    ),
    "chat_template_revision": (
        "sha256:e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259"
    ),
}
EXPECTED_HARNESS_IDENTITY = {
    "name": "qwen_code",
    "version": "0.22.3",
    "source_commit": "09825973e7d3c3fd07e17909c396aa62f48ce51f",
    "max_model_requests": 600,
    "timeout_seconds": 7200,
}
EXPECTED_FAMILIES = ("current", "fakelook")
EXPECTED_LABELS = ("level-0", "level-1", "level-2", "level-3")
MAX_SPREAD_MEMBERS = 4
MAX_SESSIONS_PER_JOB = 6
JOB_PAGE_SIZE = 200


def _digest_without(value: dict[str, Any], field: str) -> str:
    return self_hosted.sha256(
        self_hosted.canonical_json({key: item for key, item in value.items() if key != field})
    )


def _validate_review_plan(
    plan: dict[str, Any], split: dict[str, Any], review: dict[str, Any]
) -> None:
    prompt_curriculum.validate_plan(plan, split)
    if (
        plan.get("campaign_id") != EXPECTED_CAMPAIGN_ID
        or plan.get("project_id") != EXPECTED_PROJECT_ID
        or plan.get("model") != EXPECTED_MODEL_IDENTITY
        or plan.get("harness") != EXPECTED_HARNESS_IDENTITY
    ):
        raise ValueError("Qwen3.6 prompt-curriculum study identity or protocol drifted")
    rebuilt = prompt_curriculum.build_review_plan(
        plan, review.get("tasks") or [], review.get("request_audit") or {}
    )
    if rebuilt != review:
        raise ValueError("review plan does not exactly match the frozen curriculum plan")
    if review.get("review_plan_sha256") != _digest_without(review, "review_plan_sha256"):
        raise ValueError("review plan digest mismatch")
    if tuple(row.get("family") for row in review["tasks"]) != EXPECTED_FAMILIES:
        raise ValueError("create preflight requires the exact two reviewed task families")


def planned_job_name(campaign_id: str, family: str) -> str:
    return f"{campaign_id}-{family}-qcode-p1"


def _job_inventory_request(offset: int) -> dict[str, Any]:
    return {
        "method": "GET",
        "origin": fleet_client.ANALYSIS_BASE_URL,
        "path": "/v2/jobs",
        "params": {
            "team_id": self_hosted.FLEET_TEAM_ID,
            "source_type": "job",
            "project_id": EXPECTED_PROJECT_ID,
            "include_archived": True,
            "sort_order": "desc",
            "limit": JOB_PAGE_SIZE,
            "offset": offset,
        },
    }


def _expected_static_gets(plan: dict[str, Any]) -> list[dict[str, Any]]:
    requests = [
        {
            "method": "GET",
            "origin": self_hosted.ORCHESTRATOR,
            "path": "/v1/account",
            "params": {},
        }
    ]
    requests.extend(
        {
            "method": "GET",
            "origin": self_hosted.ORCHESTRATOR,
            "path": f"/v1/tasks/{row['task_key']}",
            "params": {"version_id": row["task_version_id"]},
        }
        for row in plan["tasks"]
    )
    return requests


class _AuditedReadClient:
    """Allow the three exact reads followed only by derived inventory pages."""

    def __init__(self, delegate: Any, expected: list[dict[str, Any]]) -> None:
        self._delegate = delegate
        self._expected = copy.deepcopy(expected)
        self._observed: list[dict[str, Any]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> Any:
        parsed = urlsplit(url)
        if parsed.query or parsed.fragment:
            raise RuntimeError("create preflight attempted an undeclared request")
        actual = {
            "method": method,
            "origin": f"{parsed.scheme}://{parsed.netloc}",
            "path": parsed.path,
            "params": copy.deepcopy(kwargs.get("params") or {}),
        }
        index = len(self._observed)
        if index < len(self._expected):
            expected = self._expected[index]
        else:
            expected = _job_inventory_request((index - len(self._expected)) * JOB_PAGE_SIZE)
        if method != "GET" or actual != expected:
            raise RuntimeError("create preflight attempted an undeclared or mutating request")
        self._observed.append(actual)
        return self._delegate.request(method, url, **kwargs)

    def receipt(self) -> dict[str, Any]:
        if self._observed[: len(self._expected)] != self._expected or len(self._observed) <= len(
            self._expected
        ):
            raise RuntimeError("create preflight did not complete its exact reads and inventory")
        receipt = {
            "schema_version": prompt_curriculum.REQUEST_AUDIT_SCHEMA,
            "expected_request_count": len(self._observed),
            "observed_request_count": len(self._observed),
            "mutation_request_count": 0,
            "requests": copy.deepcopy(self._observed),
        }
        receipt["request_audit_sha256"] = self_hosted.sha256(self_hosted.canonical_json(receipt))
        return receipt


def _read_job_inventory(client: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pages: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    expected_total: int | None = None
    offset = 0
    while True:
        request = _job_inventory_request(offset)
        value = _read_json(client, request["origin"], request["path"], params=request["params"])
        if not isinstance(value, dict) or not isinstance(value.get("jobs"), list):
            raise RuntimeError("Fleet job inventory returned malformed data")
        if (
            value.get("limit") != JOB_PAGE_SIZE
            or value.get("offset") != offset
            or not isinstance(value.get("total"), int)
            or isinstance(value.get("total"), bool)
            or value["total"] < 0
            or not isinstance(value.get("has_more"), bool)
        ):
            raise RuntimeError("Fleet job inventory pagination is ambiguous")
        if expected_total is None:
            expected_total = value["total"]
        elif value["total"] != expected_total:
            raise RuntimeError("Fleet job inventory changed while it was being paged")
        page = value["jobs"]
        if any(not isinstance(row, dict) or not row.get("id") for row in page):
            raise RuntimeError("Fleet job inventory contains an unidentifiable row")
        if value["has_more"] and len(page) != JOB_PAGE_SIZE:
            raise RuntimeError("Fleet job inventory returned a short non-terminal page")
        page_ids = [str(row["id"]) for row in page]
        if len(page_ids) != len(set(page_ids)) or seen_ids.intersection(page_ids):
            raise RuntimeError("Fleet job inventory contains duplicate rows across pages")
        rows.extend(copy.deepcopy(page))
        seen_ids.update(page_ids)
        pages.append(
            {
                "offset": offset,
                "row_count": len(page),
                "has_more": value["has_more"],
                "total": value["total"],
                "response_sha256": self_hosted.sha256(self_hosted.canonical_json(value)),
            }
        )
        if not value["has_more"]:
            break
        if not page:
            raise RuntimeError("Fleet job inventory claims another page after an empty page")
        offset += JOB_PAGE_SIZE
    if expected_total != len(rows):
        raise RuntimeError("Fleet job inventory did not exhaust the reported project total")
    observation = {
        "schema_version": JOB_INVENTORY_OBSERVATION_SCHEMA,
        "team_id": self_hosted.FLEET_TEAM_ID,
        "project_id": EXPECTED_PROJECT_ID,
        "active_and_archived_included": True,
        "row_count": len(rows),
        "page_count": len(pages),
        "pages": pages,
        "inventory_sha256": self_hosted.sha256(self_hosted.canonical_json(rows)),
    }
    observation["observation_sha256"] = self_hosted.sha256(self_hosted.canonical_json(observation))
    return rows, observation


def _is_possible_semantic_duplicate(
    row: dict[str, Any], *, plan: dict[str, Any], reviewed: dict[str, Any]
) -> bool:
    expected_name = planned_job_name(plan["campaign_id"], reviewed["family"])
    if row.get("name") == expected_name:
        return True
    value = row.get("input")
    if not isinstance(value, dict):
        return False
    if any(
        (
            value.get("models") != [EXPECTED_RUNTIME_MODEL],
            value.get("harness") != EXPECTED_HARNESS,
            value.get("mode") != "tool-use",
            value.get("agent_runtime") is not True,
            value.get("pass_k") != 1,
            value.get("max_steps") != EXPECTED_HARNESS_IDENTITY["max_model_requests"],
            value.get("max_duration_minutes") != EXPECTED_HARNESS_IDENTITY["timeout_seconds"] // 60,
            value.get("project_id") != reviewed["project_id"],
        )
    ):
        return False
    task_group_id = value.get("task_group_id")
    if not isinstance(task_group_id, str) or not task_group_id:
        return False
    members = value.get("task_group_members")
    if not isinstance(members, dict) or len(members) != MAX_SPREAD_MEMBERS:
        return True
    if any(not isinstance(member, dict) for member in members.values()):
        return True
    member_rows = list(members.values())
    required_fields = {
        "task_group_id",
        "task_group_member_id",
        "label",
        "task_key",
        "eval_task_id",
        "eval_task_version_id",
    }
    if any(
        not required_fields <= set(member)
        or any(
            not isinstance(member.get(field), str) or not member[field] for field in required_fields
        )
        for member in member_rows
    ):
        return True
    if (
        {member["task_group_id"] for member in member_rows} != {task_group_id}
        or {member["label"] for member in member_rows} != set(EXPECTED_LABELS)
        or len({member["task_group_member_id"] for member in member_rows}) != MAX_SPREAD_MEMBERS
        or len({member["eval_task_version_id"] for member in member_rows}) != MAX_SPREAD_MEMBERS
        or len({member["eval_task_id"] for member in member_rows}) != 1
        or len({member["task_key"] for member in member_rows}) != 1
    ):
        return True
    return member_rows[0]["task_key"] == reviewed["task_key"]


def _prove_no_duplicate_jobs(
    inventory: list[dict[str, Any]], *, plan: dict[str, Any], reviewed: dict[str, Any]
) -> dict[str, Any]:
    matches = [
        row
        for row in inventory
        if _is_possible_semantic_duplicate(row, plan=plan, reviewed=reviewed)
    ]
    if matches:
        raise RuntimeError(
            f"possible semantic duplicate Fleet job exists for {reviewed['family']!r}"
        )
    return {
        "scope": "exhaustive_fleet_project_job_inventory",
        "job_name": planned_job_name(plan["campaign_id"], reviewed["family"]),
        "inventory_rows_examined": len(inventory),
        "possible_semantic_duplicates": 0,
        "active_and_archived_included": True,
    }


def _read_json(client: Any, origin: str, path: str, *, params: dict[str, Any]) -> Any:
    response = client.request("GET", f"{origin}{path}", params=params)
    if response.status_code >= 400:
        raise RuntimeError(f"Fleet GET {path} failed with HTTP {response.status_code}")
    try:
        return response.json()
    except ValueError:
        raise RuntimeError(f"Fleet GET {path} returned non-JSON data") from None


def _job_contract(plan: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    contract = {
        "name": planned_job_name(plan["campaign_id"], task["family"]),
        "task_group_id": "PENDING_CREATED_GROUP_HYDRATION",
        "models": [EXPECTED_RUNTIME_MODEL],
        "harness": EXPECTED_HARNESS,
        "mode": "tool-use",
        "agent_runtime": True,
        "pass_k": 1,
        "max_steps": plan["harness"]["max_model_requests"],
        "max_duration_minutes": plan["harness"]["timeout_seconds"] // 60,
        "cost_team_id": self_hosted.FLEET_TEAM_ID,
        "project_id": task["project_id"],
        "sales_product_id": DEFAULT_SALES_PRODUCT_ID,
        "capture_db_diff": True,
        "job_type": "evaluation",
        "planned_sessions": task["variant_count"],
        "metadata": {
            "experiment": plan["campaign_id"],
            "gateway_model": plan["model"]["served_id"],
            "runtime": EXPECTED_RUNTIME_LABEL,
            "prompt_curriculum_family": task["family"],
            "review_plan_sha256": None,
        },
    }
    return contract


def build_create_preflight(
    plan: dict[str, Any],
    split: dict[str, Any],
    review: dict[str, Any],
    live_material: list[tuple[dict[str, Any], dict[str, Any]]],
    job_inventory: list[dict[str, Any]],
    inventory_observation: dict[str, Any],
    request_audit: dict[str, Any],
) -> dict[str, Any]:
    """Build a sanitized, reviewable claim; all prompt bodies remain in memory."""

    _validate_review_plan(plan, split, review)
    static_gets = _expected_static_gets(plan)
    observed = request_audit.get("requests") or []
    inventory_gets = observed[len(static_gets) :]
    if (
        request_audit.get("schema_version") != prompt_curriculum.REQUEST_AUDIT_SCHEMA
        or request_audit.get("request_audit_sha256")
        != _digest_without(request_audit, "request_audit_sha256")
        or observed[: len(static_gets)] != static_gets
        or not inventory_gets
        or inventory_gets
        != [_job_inventory_request(index * JOB_PAGE_SIZE) for index in range(len(inventory_gets))]
        or request_audit.get("expected_request_count") != len(observed)
        or request_audit.get("observed_request_count") != len(observed)
        or request_audit.get("mutation_request_count") != 0
    ):
        raise ValueError("create preflight lacks its exact GET-only exhaustive-read audit")
    if len(live_material) != 2:
        raise ValueError("create preflight requires exactly two hydrated tasks")
    inventory_ids = [str(row.get("id") or "") for row in job_inventory]
    if "" in inventory_ids or len(set(inventory_ids)) != len(inventory_ids):
        raise ValueError("create preflight job inventory contains duplicate or missing ids")
    inventory_sha256 = self_hosted.sha256(self_hosted.canonical_json(job_inventory))
    pages = inventory_observation.get("pages") or []
    if (
        inventory_observation.get("schema_version") != JOB_INVENTORY_OBSERVATION_SCHEMA
        or inventory_observation.get("observation_sha256")
        != _digest_without(inventory_observation, "observation_sha256")
        or inventory_observation.get("team_id") != self_hosted.FLEET_TEAM_ID
        or inventory_observation.get("project_id") != EXPECTED_PROJECT_ID
        or inventory_observation.get("active_and_archived_included") is not True
        or inventory_observation.get("row_count") != len(job_inventory)
        or inventory_observation.get("page_count") != len(inventory_gets)
        or inventory_observation.get("page_count") != len(pages)
        or inventory_observation.get("inventory_sha256") != inventory_sha256
        or [page.get("offset") for page in pages]
        != [index * JOB_PAGE_SIZE for index in range(len(pages))]
        or sum(page.get("row_count", -1) for page in pages) != len(job_inventory)
        or any(page.get("total") != len(job_inventory) for page in pages)
        or [page.get("has_more") for page in pages] != [True] * (len(pages) - 1) + [False]
        or any(page.get("row_count") != JOB_PAGE_SIZE for page in pages[:-1])
        or not 0 <= pages[-1].get("row_count", -1) <= JOB_PAGE_SIZE
        or any(
            not isinstance(page.get("response_sha256"), str)
            or not page["response_sha256"].startswith("sha256:")
            for page in pages
        )
    ):
        raise ValueError("job inventory is not bound to the exhaustive Fleet read observation")

    groups = []
    duplicate_plan = {**plan, "review_plan_sha256": review["review_plan_sha256"]}
    for selected, reviewed, material in zip(
        plan["tasks"], review["tasks"], live_material, strict=True
    ):
        payload, live_receipt = material
        if live_receipt != reviewed:
            raise RuntimeError("live source task or private payload drifted from reviewed evidence")
        if self_hosted.sha256(self_hosted.canonical_json(payload)) != reviewed.get(
            "private_task_group_payload_sha256"
        ):
            raise RuntimeError("private task-group payload digest drifted")
        if len(payload.get("members") or []) != MAX_SPREAD_MEMBERS:
            raise ValueError("task group must contain exactly four prompt members")
        labels = tuple(member.get("label") for member in payload["members"])
        if labels != EXPECTED_LABELS or any(
            set(member) != {"label", "prompt"} for member in payload["members"]
        ):
            raise ValueError("task-group member shape drifted from prompt-only spread")
        job = _job_contract(plan, reviewed)
        job["metadata"]["review_plan_sha256"] = review["review_plan_sha256"]
        if job["pass_k"] != 1 or job["planned_sessions"] > MAX_SESSIONS_PER_JOB:
            raise ValueError("planned job violates pass@1 or six-session cap")
        duplicate = _prove_no_duplicate_jobs(job_inventory, plan=duplicate_plan, reviewed=reviewed)
        pre_group_job_contract_sha256 = self_hosted.sha256(self_hosted.canonical_json(job))
        groups.append(
            {
                "family": selected["family"],
                "task_group_name": reviewed["task_group_name"],
                "shared_task_key": selected["task_key"],
                "exact_source_task_version_id": selected["task_version_id"],
                "project_id": reviewed["project_id"],
                "member_labels": list(labels),
                "member_prompt_sha256": copy.deepcopy(reviewed["variant_prompt_sha256"]),
                "private_task_group_payload_sha256": reviewed["private_task_group_payload_sha256"],
                "non_prompt_task_spec_sha256": reviewed["non_prompt_task_spec_sha256"],
                "exact_version_bindings": copy.deepcopy(reviewed["exact_version_bindings"]),
                "registry_task_graph_source": copy.deepcopy(reviewed["registry_task_graph_source"]),
                "job_contract": job,
                "pre_group_job_contract_sha256": pre_group_job_contract_sha256,
                "future_job_idempotency_namespace": str(
                    uuid5(NAMESPACE_URL, pre_group_job_contract_sha256)
                ),
                "duplicate_job_proof": duplicate,
                "create_state": "not_created",
                "hydration_state": "pending_created_group",
            }
        )

    claim = {
        "schema_version": CREATE_CLAIM_SCHEMA,
        "campaign_id": plan["campaign_id"],
        "review_plan_sha256": review["review_plan_sha256"],
        "team_id": self_hosted.FLEET_TEAM_ID,
        "family_count": 2,
        "task_groups_planned": 2,
        "spread_members_per_group": 4,
        "pass_k": 1,
        "planned_sessions_per_job": 4,
        "model_identity": copy.deepcopy(plan["model"]),
        "harness_identity": copy.deepcopy(plan["harness"]),
        "groups": groups,
        "task_group_api_idempotency_available": False,
        "task_group_api_exact_verifier_version_input_available": False,
        "ambiguous_create_policy": "stop_without_retry_and_reconcile_manually",
        "post_create_verifier_drift_policy": "block_without_launch",
        "paid_launch_blocked_on": [
            "task_group_creation",
            "created_member_hydration",
            "authoritative_environment_version_id_receipt",
            "live_model_serving_parity",
        ],
        "pipeline_lane_operations": 0,
        "registry_publications": 0,
        "task_group_creations": 0,
        "paid_job_submissions": 0,
    }
    claim["create_claim_sha256"] = self_hosted.sha256(self_hosted.canonical_json(claim))
    result = {
        "schema_version": CREATE_PREFLIGHT_SCHEMA,
        "campaign_id": plan["campaign_id"],
        "review_plan_sha256": review["review_plan_sha256"],
        "fleet_team": {"name": "fleet", "id": self_hosted.FLEET_TEAM_ID},
        "request_audit": request_audit,
        "duplicate_job_proof_complete": True,
        "job_inventory_row_count": len(job_inventory),
        "job_inventory_sha256": inventory_sha256,
        "job_inventory_observation": copy.deepcopy(inventory_observation),
        "create_claim": claim,
        "post_create_required_gates": [
            "save the one task-group POST response durably before any next action",
            "never retry an ambiguous task-group POST",
            "GET the exact created group and all four exact member task versions",
            "prove one shared task id/key and four unique labelled member versions",
            "prove prompt hashes match and all non-prompt task bindings remain unchanged",
            "obtain an authoritative exact environment-version UUID for every member",
            "only then derive a job idempotency key that binds the created task_group_id",
        ],
        "task_group_create_performed": False,
        "paid_job_submitted": False,
        "pipeline_lane_touched": False,
    }
    result["preflight_sha256"] = self_hosted.sha256(self_hosted.canonical_json(result))
    return result


def prepare_live_create_preflight(
    client: Any,
    plan: dict[str, Any],
    split: dict[str, Any],
    review: dict[str, Any],
) -> dict[str, Any]:
    """Execute the exact source reads plus exhaustive GET-only job inventory."""

    _validate_review_plan(plan, split, review)
    audited = _AuditedReadClient(client, _expected_static_gets(plan))
    account = self_hosted._request(audited, "GET", "/v1/account")
    if account.get("team_name") != "fleet" or account.get("team_id") != self_hosted.FLEET_TEAM_ID:
        raise RuntimeError("FLEET_API_KEY is not scoped to the exact Fleet team")

    material = []
    for selected in plan["tasks"]:
        task = self_hosted._request(
            audited,
            "GET",
            f"/v1/tasks/{selected['task_key']}",
            params={"version_id": selected["task_version_id"]},
        )
        material.append(prompt_curriculum.build_task_group_payload(plan, selected, task))

    inventory, observation = _read_job_inventory(audited)
    return build_create_preflight(
        plan, split, review, material, inventory, observation, audited.receipt()
    )


def _member_binding(task: dict[str, Any]) -> dict[str, Any]:
    verifier = task.get("verifier") or {}
    metadata = task.get("metadata") or {}
    return {
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


def validate_created_group(
    reviewed: dict[str, Any],
    group: dict[str, Any],
    member_tasks: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Validate the mandatory post-create hydration before any paid job."""

    members = group.get("members") or []
    group_id = str(group.get("id") or group.get("task_group_id") or "")
    if (
        not group_id
        or group.get("task_group_id") not in (None, group_id)
        or group.get("team_id") != self_hosted.FLEET_TEAM_ID
        or group.get("project_id") != reviewed.get("project_id")
        or group.get("name") != reviewed.get("task_group_name")
        or group.get("warnings") not in (None, [])
        or len(members) != MAX_SPREAD_MEMBERS
    ):
        raise ValueError("created task group identity or member count drifted")
    members_by_label = {member.get("label"): member for member in members}
    if set(members_by_label) != set(EXPECTED_LABELS) or len(members_by_label) != 4:
        raise ValueError("created task-group labels drifted or are duplicated")
    if {str(member.get("task_group_id")) for member in members} != {group_id}:
        raise ValueError("created member points at a different task group")
    task_ids = {str(member.get("eval_task_id")) for member in members}
    version_ids = [str(member.get("eval_task_version_id")) for member in members]
    if len(task_ids) != 1 or "" in task_ids or len(set(version_ids)) != 4 or "" in version_ids:
        raise ValueError("created members do not share one task with four unique versions")
    shared_task_id = next(iter(task_ids))
    if group.get("eval_task_id") not in (None, shared_task_id):
        raise ValueError("created task-group summary names a different shared task")
    if set(member_tasks) != set(version_ids):
        raise ValueError("created-member hydration is incomplete or contains an extra version")

    expected_source = reviewed["source_binding"]
    expected_prompt_hashes = reviewed["variant_prompt_sha256"]
    hydrated = []
    for label in EXPECTED_LABELS:
        member = members_by_label[label]
        version_id = str(member["eval_task_version_id"])
        task = member_tasks[version_id]
        binding = _member_binding(task)
        expected = {**expected_source, "prompt_sha256": expected_prompt_hashes[label]}
        if binding != expected:
            raise ValueError(f"created member {label!r} changed a frozen task binding")
        if task.get("id") not in (None, member.get("eval_task_id")):
            raise ValueError("hydrated member task id disagrees with task-group response")
        if task.get("version_id") not in (None, version_id):
            raise ValueError("hydrated member version id disagrees with task-group response")
        if self_hosted.sha256(
            self_hosted.canonical_json(prompt_curriculum._task_spec(task))
        ) != reviewed.get("non_prompt_task_spec_sha256"):
            raise ValueError("created member non-prompt task specification drifted")
        source = prompt_curriculum._exact_registry_source(task.get("metadata") or {})
        if source != reviewed.get("registry_task_graph_source"):
            raise ValueError("created member task-graph source drifted")
        hydrated.append(
            {
                "label": label,
                "eval_task_id": str(member["eval_task_id"]),
                "eval_task_version_id": version_id,
                "prompt_sha256": binding["prompt_sha256"],
            }
        )

    receipt = {
        "schema_version": HYDRATION_RECEIPT_SCHEMA,
        "task_group_id": group_id,
        "task_group_name": reviewed["task_group_name"],
        "shared_task_key": reviewed["task_key"],
        "shared_eval_task_id": shared_task_id,
        "members": hydrated,
        "member_count": 4,
        "environment_key_and_version_label_unchanged": True,
        "expected_environment_version_id": reviewed["exact_version_bindings"]["environment"][
            "version_id"
        ],
        "environment_version_id_verified": False,
        "environment_version_id_blocker": (
            "the public task-version read omits environment_version_id; require an "
            "authoritative server-side member-version receipt before launch"
        ),
        "data_unchanged": True,
        "runtime_seed_unchanged": True,
        "atoms_unchanged": True,
        "verifier_unchanged": True,
        "flags_unchanged": True,
        "non_prompt_task_spec_unchanged": True,
        "paid_job_submission_unblocked": False,
    }
    receipt["hydration_sha256"] = self_hosted.sha256(self_hosted.canonical_json(receipt))
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--review-plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    plan = prompt_curriculum.load_json(args.config)
    split = prompt_curriculum.load_json(args.split)
    review = prompt_curriculum.load_json(args.review_plan)
    with fleet_client.authenticated_client() as client:
        result = prepare_live_create_preflight(client, plan, split, review)
    if args.out.exists():
        raise FileExistsError(f"refusing to replace existing create preflight: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    self_hosted.write_json_once(args.out, result)
    print(json.dumps({"preflight_sha256": result["preflight_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
