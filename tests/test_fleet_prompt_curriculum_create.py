from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from evals.fleet import prompt_curriculum, prompt_curriculum_create, self_hosted

PLAN = Path("evals/fleet/configs/qwen36-27b-prompt-curriculum-pilot-v1.json")
SPLIT = Path("configs/data/fleet-a62-task-split-v1.json")


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _live_task(selected: dict[str, Any]) -> dict[str, Any]:
    receipt = selected["source_receipt"]
    prompt = f"Private {selected['family']} source prompt"
    env_variables = {"PRIVATE": selected["family"]}
    output_schema = {"type": "object"}
    receipt.update(
        {
            "prompt_sha256": self_hosted.sha256(prompt.encode()),
            "env_variables_sha256": self_hosted.sha256(self_hosted.canonical_json(env_variables)),
            "output_json_schema_sha256": self_hosted.sha256(
                self_hosted.canonical_json(output_schema)
            ),
        }
    )
    return {
        "key": selected["task_key"],
        "prompt": prompt,
        "environment_id": selected["env_key"],
        "version": selected["env_version"],
        "data_id": selected["data_key"],
        "data_version": selected["data_version"],
        "verifier_id": receipt["verifier_id"],
        "verifier": {
            "verifier_version_id": receipt["verifier_version_id"],
            "version": receipt["verifier_version"],
            "sha256": receipt["verifier_sha256"],
        },
        "env_variables": env_variables,
        "output_json_schema": output_schema,
        "metadata": {
            "runtime_seed_manifest": {
                "content_sha256": receipt["runtime_seed_content_sha256"],
                "files": [],
            },
            "task_graph_source": {
                "artifact_key": f"cyber/task-graphs/{selected['family']}-curriculum-test",
                "version_index": 1,
                "manifest_sha256": "a" * 64,
            },
        },
        "task_modality": "tool_use",
    }


def _audit(requests: list[dict[str, Any]]) -> dict[str, Any]:
    value = {
        "schema_version": prompt_curriculum.REQUEST_AUDIT_SCHEMA,
        "expected_request_count": len(requests),
        "observed_request_count": len(requests),
        "mutation_request_count": 0,
        "requests": copy.deepcopy(requests),
    }
    value["request_audit_sha256"] = self_hosted.sha256(self_hosted.canonical_json(value))
    return value


def _inventory_observation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    value = {
        "schema_version": prompt_curriculum_create.JOB_INVENTORY_OBSERVATION_SCHEMA,
        "team_id": self_hosted.FLEET_TEAM_ID,
        "project_id": prompt_curriculum_create.EXPECTED_PROJECT_ID,
        "active_and_archived_included": True,
        "row_count": len(rows),
        "page_count": 1,
        "pages": [
            {
                "offset": 0,
                "row_count": len(rows),
                "has_more": False,
                "total": len(rows),
                "response_sha256": "sha256:" + "a" * 64,
            }
        ],
        "inventory_sha256": self_hosted.sha256(self_hosted.canonical_json(rows)),
    }
    value["observation_sha256"] = self_hosted.sha256(self_hosted.canonical_json(value))
    return value


def _fixture() -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[tuple[dict[str, Any], dict[str, Any]]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    plan, split = _json(PLAN), _json(SPLIT)
    live_material = []
    for selected in plan["tasks"]:
        live_material.append(
            prompt_curriculum.build_task_group_payload(plan, selected, _live_task(selected))
        )
    plan["plan_sha256"] = self_hosted.sha256(
        self_hosted.canonical_json(
            {key: value for key, value in plan.items() if key != "plan_sha256"}
        )
    )
    old_gets = [
        {"method": "GET", "path": "/v1/account", "params": {}},
        *[
            {
                "method": "GET",
                "path": f"/v1/tasks/{selected['task_key']}",
                "params": {"version_id": selected["task_version_id"]},
            }
            for selected in plan["tasks"]
        ],
    ]
    old_audit = {
        **_audit(old_gets),
        "orchestrator_origin": self_hosted.ORCHESTRATOR,
    }
    old_audit["request_audit_sha256"] = self_hosted.sha256(
        self_hosted.canonical_json(
            {key: value for key, value in old_audit.items() if key != "request_audit_sha256"}
        )
    )
    review = prompt_curriculum.build_review_plan(
        plan, [receipt for _, receipt in live_material], old_audit
    )
    inventory: list[dict[str, Any]] = []
    observation = _inventory_observation(inventory)
    requests = [
        *prompt_curriculum_create._expected_static_gets(plan),
        prompt_curriculum_create._job_inventory_request(0),
    ]
    request_audit = _audit(requests)
    return plan, split, review, live_material, inventory, observation, request_audit


def test_create_preflight_is_exact_bounded_and_prompt_free() -> None:
    plan, split, review, material, inventory, observation, audit = _fixture()
    result = prompt_curriculum_create.build_create_preflight(
        plan, split, review, material, inventory, observation, audit
    )

    claim = result["create_claim"]
    assert claim["family_count"] == claim["task_groups_planned"] == 2
    assert claim["spread_members_per_group"] == 4
    assert claim["pass_k"] == 1
    assert claim["planned_sessions_per_job"] == 4
    assert claim["pipeline_lane_operations"] == 0
    assert claim["registry_publications"] == 0
    assert claim["task_group_creations"] == 0
    assert claim["paid_job_submissions"] == 0
    assert {group["family"] for group in claim["groups"]} == {"current", "fakelook"}
    assert all(group["job_contract"]["planned_sessions"] <= 6 for group in claim["groups"])
    serialized = json.dumps(result)
    assert "Private current source prompt" not in serialized
    assert "Private fakelook source prompt" not in serialized


def _semantic_duplicate(
    plan: dict[str, Any], reviewed: dict[str, Any], *, name: str
) -> dict[str, Any]:
    contract = prompt_curriculum_create._job_contract(plan, reviewed)
    contract["task_group_id"] = "existing-group"
    contract["task_group_members"] = {
        label: {
            "task_group_id": "existing-group",
            "task_group_member_id": f"member-{label}",
            "label": label,
            "task_key": reviewed["task_key"],
            "eval_task_id": "existing-shared-task",
            "eval_task_version_id": f"version-{label}",
        }
        for label in prompt_curriculum_create.EXPECTED_LABELS
    }
    return {"id": f"job-{name}", "name": name, "input": contract}


@pytest.mark.parametrize("name", ["expected", "completely-different-name"])
def test_create_preflight_rejects_name_or_semantic_duplicate(name: str) -> None:
    plan, split, review, material, inventory, _, audit = _fixture()
    reviewed = review["tasks"][0]
    actual_name = (
        prompt_curriculum_create.planned_job_name(plan["campaign_id"], reviewed["family"])
        if name == "expected"
        else name
    )
    inventory.append(_semantic_duplicate(plan, reviewed, name=actual_name))
    observation = _inventory_observation(inventory)
    with pytest.raises(RuntimeError, match="possible semantic duplicate"):
        prompt_curriculum_create.build_create_preflight(
            plan, split, review, material, inventory, observation, audit
        )


def test_create_preflight_blocks_ambiguous_group_job_without_member_provenance() -> None:
    plan, split, review, material, inventory, _, audit = _fixture()
    row = _semantic_duplicate(plan, review["tasks"][0], name="renamed-job")
    row["input"].pop("task_group_members")
    inventory.append(row)
    observation = _inventory_observation(inventory)
    with pytest.raises(RuntimeError, match="possible semantic duplicate"):
        prompt_curriculum_create.build_create_preflight(
            plan, split, review, material, inventory, observation, audit
        )


@pytest.mark.parametrize(
    "members",
    [
        pytest.param({}, id="empty-map"),
        pytest.param(
            {
                "level-0": {
                    "task_group_id": "existing-group",
                    "task_group_member_id": "member-level-0",
                    "label": "level-0",
                    "task_key": "some-other-task",
                    "eval_task_id": "existing-shared-task",
                    "eval_task_version_id": "version-level-0",
                }
            },
            id="partial-map",
        ),
        pytest.param({"level-0": "not-a-member-record"}, id="malformed-map"),
        pytest.param(["not", "a", "map"], id="unparseable-shape"),
    ],
)
def test_create_preflight_blocks_incomplete_task_group_member_expansion(members: Any) -> None:
    plan, split, review, material, inventory, _, audit = _fixture()
    row = _semantic_duplicate(plan, review["tasks"][0], name="renamed-job")
    row["input"]["task_group_members"] = members
    inventory.append(row)
    observation = _inventory_observation(inventory)
    with pytest.raises(RuntimeError, match="possible semantic duplicate"):
        prompt_curriculum_create.build_create_preflight(
            plan, split, review, material, inventory, observation, audit
        )


def test_complete_authoritative_member_map_can_prove_a_different_task() -> None:
    plan, split, review, material, inventory, _, audit = _fixture()
    row = _semantic_duplicate(plan, review["tasks"][0], name="renamed-job")
    for member in row["input"]["task_group_members"].values():
        member["task_key"] = "provably-different-task"
    inventory.append(row)
    observation = _inventory_observation(inventory)
    result = prompt_curriculum_create.build_create_preflight(
        plan, split, review, material, inventory, observation, audit
    )
    assert result["duplicate_job_proof_complete"] is True


def test_create_preflight_rejects_task_or_prompt_payload_drift() -> None:
    plan, split, review, material, inventory, observation, audit = _fixture()
    material[0][0]["members"][1]["prompt"] += " drift"
    with pytest.raises(RuntimeError, match="payload digest drifted"):
        prompt_curriculum_create.build_create_preflight(
            plan, split, review, material, inventory, observation, audit
        )


def test_create_preflight_rejects_any_added_family() -> None:
    plan, split, review, material, inventory, observation, audit = _fixture()
    plan["tasks"].append(copy.deepcopy(plan["tasks"][0]))
    with pytest.raises(ValueError):
        prompt_curriculum_create.build_create_preflight(
            plan, split, review, material, inventory, observation, audit
        )


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("model", "repository", "other/repository"),
        ("model", "revision", "0" * 40),
        ("model", "served_id", "different-model"),
        ("model", "weights_manifest_sha256", "sha256:" + "0" * 64),
        ("model", "tokenizer_revision", "sha256:" + "1" * 64),
        ("model", "chat_template_revision", "sha256:" + "2" * 64),
        ("harness", "name", "different_harness"),
        ("harness", "max_model_requests", 1),
        ("harness", "timeout_seconds", 60),
        ("harness", "version", "0.22.4"),
        ("harness", "source_commit", "3" * 40),
    ],
)
def test_create_preflight_rejects_resealed_model_or_harness_drift(
    section: str, field: str, value: Any
) -> None:
    plan, split, review, material, inventory, observation, audit = _fixture()
    plan[section][field] = value
    plan["plan_sha256"] = self_hosted.sha256(
        self_hosted.canonical_json(
            {key: item for key, item in plan.items() if key != "plan_sha256"}
        )
    )
    review[section] = copy.deepcopy(plan[section])
    review["review_plan_sha256"] = self_hosted.sha256(
        self_hosted.canonical_json(
            {key: item for key, item in review.items() if key != "review_plan_sha256"}
        )
    )
    with pytest.raises(ValueError, match="drifted"):
        prompt_curriculum_create.build_create_preflight(
            plan, split, review, material, inventory, observation, audit
        )


class _Response:
    def __init__(self, value: dict[str, Any]) -> None:
        self.status_code = 200
        self._value = value

    def json(self) -> dict[str, Any]:
        return copy.deepcopy(self._value)


class _ReadOnlyDelegate:
    def __init__(
        self,
        plan: dict[str, Any],
        tasks: list[dict[str, Any]],
        *,
        account: dict[str, Any] | None = None,
        job_pages: dict[int, dict[str, Any]] | None = None,
    ) -> None:
        self.plan = plan
        self.tasks = tasks
        self.account = (
            account
            if account is not None
            else {"team_name": "fleet", "team_id": self_hosted.FLEET_TEAM_ID}
        )
        self.job_pages = job_pages or {
            0: {"jobs": [], "total": 0, "limit": 200, "offset": 0, "has_more": False}
        }
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> _Response:
        self.calls.append((method, url, copy.deepcopy(kwargs)))
        if url.endswith("/v1/account"):
            return _Response(self.account)
        for selected, task in zip(self.plan["tasks"], self.tasks, strict=False):
            if url.endswith(f"/v1/tasks/{selected['task_key']}"):
                return _Response(task)
        if url.endswith("/v2/jobs"):
            return _Response(self.job_pages[kwargs["params"]["offset"]])
        raise AssertionError(url)


def test_live_preflight_executes_exact_source_gets_plus_complete_inventory() -> None:
    plan, split, review, _, _, _, _ = _fixture()
    tasks = [_live_task(selected) for selected in plan["tasks"]]
    delegate = _ReadOnlyDelegate(plan, tasks)
    result = prompt_curriculum_create.prepare_live_create_preflight(delegate, plan, split, review)
    assert len(delegate.calls) == 4
    assert {method for method, _, _ in delegate.calls} == {"GET"}
    assert result["request_audit"]["mutation_request_count"] == 0


def test_job_inventory_pages_until_reported_total_is_exhausted() -> None:
    plan = _json(PLAN)
    first = [{"id": f"job-{index}", "input": {}} for index in range(200)]
    final = [{"id": "job-200", "input": {}}]
    delegate = _ReadOnlyDelegate(
        plan,
        [],
        job_pages={
            0: {"jobs": first, "total": 201, "limit": 200, "offset": 0, "has_more": True},
            200: {
                "jobs": final,
                "total": 201,
                "limit": 200,
                "offset": 200,
                "has_more": False,
            },
        },
    )
    rows, observation = prompt_curriculum_create._read_job_inventory(delegate)
    assert len(rows) == 201
    assert observation["row_count"] == 201
    assert observation["page_count"] == 2
    assert [call[2]["params"]["offset"] for call in delegate.calls] == [0, 200]


def test_job_inventory_rejects_changed_total_across_pages() -> None:
    plan = _json(PLAN)
    first = [{"id": f"job-{index}", "input": {}} for index in range(200)]
    delegate = _ReadOnlyDelegate(
        plan,
        [],
        job_pages={
            0: {"jobs": first, "total": 201, "limit": 200, "offset": 0, "has_more": True},
            200: {"jobs": [], "total": 200, "limit": 200, "offset": 200, "has_more": False},
        },
    )
    with pytest.raises(RuntimeError, match="changed while it was being paged"):
        prompt_curriculum_create._read_job_inventory(delegate)


def test_live_preflight_requires_exact_fleet_team_id() -> None:
    plan, split, review, _, _, _, _ = _fixture()
    tasks = [_live_task(selected) for selected in plan["tasks"]]
    delegate = _ReadOnlyDelegate(plan, tasks, account={"team_name": "fleet"})
    with pytest.raises(RuntimeError, match="exact Fleet team"):
        prompt_curriculum_create.prepare_live_create_preflight(delegate, plan, split, review)
    assert len(delegate.calls) == 1
    assert delegate.calls[0][0] == "GET"


def _created_group_fixture() -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    plan, _, review, material, _, _, _ = _fixture()
    reviewed = review["tasks"][0]
    payload = material[0][0]
    task_id = "11111111-1111-4111-8111-111111111111"
    group_id = "22222222-2222-4222-8222-222222222222"
    members = []
    tasks = {}
    source_task = _live_task(plan["tasks"][0])
    for index, member in enumerate(payload["members"], start=1):
        version_id = f"00000000-0000-4000-8000-{index:012d}"
        members.append(
            {
                "id": f"member-{index}",
                "task_group_id": group_id,
                "eval_task_id": task_id,
                "eval_task_version_id": version_id,
                "label": member["label"],
            }
        )
        task = copy.deepcopy(source_task)
        task["prompt"] = member["prompt"]
        task["id"] = task_id
        task["version_id"] = version_id
        tasks[version_id] = task
    group = {
        "id": group_id,
        "task_group_id": group_id,
        "team_id": self_hosted.FLEET_TEAM_ID,
        "project_id": reviewed["project_id"],
        "name": reviewed["task_group_name"],
        "members": members,
        "warnings": [],
    }
    return reviewed, group, tasks


def test_created_group_hydration_proves_shared_task_and_all_frozen_bindings() -> None:
    reviewed, group, tasks = _created_group_fixture()
    receipt = prompt_curriculum_create.validate_created_group(reviewed, group, tasks)
    assert receipt["member_count"] == 4
    assert len({row["eval_task_id"] for row in receipt["members"]}) == 1
    assert all(
        receipt[field]
        for field in (
            "environment_key_and_version_label_unchanged",
            "data_unchanged",
            "runtime_seed_unchanged",
            "atoms_unchanged",
            "verifier_unchanged",
            "flags_unchanged",
            "non_prompt_task_spec_unchanged",
        )
    )
    assert receipt["environment_version_id_verified"] is False
    assert receipt["paid_job_submission_unblocked"] is False
    assert (
        receipt["expected_environment_version_id"]
        == reviewed["exact_version_bindings"]["environment"]["version_id"]
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda group, tasks: group["members"][1].update(eval_task_id="peer"), "share one task"),
        (
            lambda group, tasks: group["members"][1].update(
                eval_task_version_id=group["members"][0]["eval_task_version_id"]
            ),
            "four unique versions",
        ),
        (
            lambda group, tasks: tasks[group["members"][0]["eval_task_version_id"]].update(
                environment_id="other-env"
            ),
            "changed a frozen task binding",
        ),
        (
            lambda group, tasks: tasks[group["members"][0]["eval_task_version_id"]]["metadata"][
                "task_graph_source"
            ].update(manifest_sha256="b" * 64),
            "specification drifted",
        ),
    ],
)
def test_created_group_hydration_rejects_adversarial_drift(mutation, message) -> None:
    reviewed, group, tasks = _created_group_fixture()
    mutation(group, tasks)
    with pytest.raises(ValueError, match=message):
        prompt_curriculum_create.validate_created_group(reviewed, group, tasks)
