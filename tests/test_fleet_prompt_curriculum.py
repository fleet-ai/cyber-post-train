from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from evals.fleet import prompt_curriculum, self_hosted

PLAN = Path("evals/fleet/configs/qwen36-27b-prompt-curriculum-pilot-v1.json")
SPLIT = Path("configs/data/fleet-a62-task-split-v1.json")


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _seal(plan: dict) -> dict:
    plan["plan_sha256"] = self_hosted.sha256(
        self_hosted.canonical_json(
            {key: value for key, value in plan.items() if key != "plan_sha256"}
        )
    )
    return plan


def _campaign_state(plan: dict) -> dict:
    outcomes = []
    for task in plan["tasks"]:
        for attempt in range(1, 5):
            outcomes.append(
                {
                    "attempt": attempt,
                    "agent_termination": "completed",
                    "score": 0.0,
                    "session_id": f"session-{task['family']}-{attempt}",
                    "session_ingest_status": "completed",
                    "status": "model_outcome",
                    "task_key": task["task_key"],
                    "task_version_id": task["task_version_id"],
                    "verifier_execution_id": f"verifier-{task['family']}-{attempt}",
                }
            )
    state = {
        "campaign_id": plan["baseline_campaign"]["campaign_id"],
        "planned_sessions": 96,
        "completed_attempts": len(outcomes),
        "outcomes": outcomes,
    }
    plan["baseline_campaign"]["state_sha256"] = self_hosted.sha256(
        self_hosted.canonical_json(state)
    )
    _seal(plan)
    return state


def _live_task(selected: dict) -> dict:
    prompt = "Private source task prompt"
    env_variables = {"PRIVATE_BINDING": "not-for-receipt"}
    output_schema = {"type": "object"}
    runtime_seed = selected["source_receipt"]["runtime_seed_content_sha256"]
    verifier = {
        "verifier_version_id": selected["source_receipt"]["verifier_version_id"],
        "version": selected["source_receipt"]["verifier_version"],
        "sha256": selected["source_receipt"]["verifier_sha256"],
    }
    selected["source_receipt"].update(
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
        "verifier_id": selected["source_receipt"]["verifier_id"],
        "verifier": verifier,
        "env_variables": env_variables,
        "output_json_schema": output_schema,
        "metadata": {
            "private_note": "not-for-receipt",
            "runtime_seed_manifest": {"content_sha256": runtime_seed, "files": []},
            "task_graph_source": {
                "artifact_key": f"cyber/task-graphs/{selected['family']}-pilot",
                "version_index": 7,
                "manifest_sha256": "a" * 64,
            },
        },
        "task_modality": "tool_use",
    }


def test_plan_selects_only_completed_zero_reward_train_tasks() -> None:
    plan, split = _json(PLAN), _json(SPLIT)
    state = _campaign_state(plan)
    rows = prompt_curriculum.validate_plan(plan, split, state)
    assert len(rows) == 2
    assert {row["family"] for row in rows} == {"current", "fakelook"}
    train_ids = {row["task_version_id"] for row in split["tasks"] if row["split"] == "train"}
    assert {row["task_version_id"] for row in rows} <= train_ids


def test_plan_rejects_any_sealed_task_version() -> None:
    plan, split = _json(PLAN), _json(SPLIT)
    sealed = next(row for row in split["tasks"] if row["split"] == "test")
    plan["tasks"][0].update(
        {
            key: sealed[key]
            for key in (
                "task_key",
                "task_version_id",
                "task_version",
                "environment_version_id",
                "env_key",
                "env_version",
                "data_key",
                "data_version",
            )
        }
    )
    _seal(plan)
    with pytest.raises(ValueError, match="sealed dev/test"):
        prompt_curriculum.validate_plan(plan, split)


def test_v1_plan_requires_exactly_two_tasks() -> None:
    plan, split = _json(PLAN), _json(SPLIT)
    plan["tasks"] = plan["tasks"][:1]
    _seal(plan)
    with pytest.raises(ValueError, match="exactly two"):
        prompt_curriculum.validate_plan(plan, split)


@pytest.mark.parametrize("score", [pytest.param(None, id="null"), float("nan"), False, True])
def test_campaign_score_must_be_finite_numeric_exact_zero(score) -> None:
    plan, split = _json(PLAN), _json(SPLIT)
    state = _campaign_state(plan)
    state["outcomes"][0]["score"] = score
    plan["baseline_campaign"]["state_sha256"] = self_hosted.sha256(
        self_hosted.canonical_json(state)
    )
    _seal(plan)
    with pytest.raises(ValueError, match="incomplete or nonzero"):
        prompt_curriculum.validate_plan(plan, split, state)


def test_campaign_score_field_is_required() -> None:
    plan, split = _json(PLAN), _json(SPLIT)
    state = _campaign_state(plan)
    state["outcomes"][0].pop("score")
    plan["baseline_campaign"]["state_sha256"] = self_hosted.sha256(
        self_hosted.canonical_json(state)
    )
    _seal(plan)
    with pytest.raises(ValueError, match="incomplete or nonzero"):
        prompt_curriculum.validate_plan(plan, split, state)


def test_generic_ladder_is_cumulative_and_contains_no_forbidden_material() -> None:
    plan = _json(PLAN)
    prompt_curriculum.validate_generic_cues()
    rendered = [
        prompt_curriculum.render_variant("base", row["cue_ids"]) for row in plan["variants"]
    ]
    assert rendered[0] == "base"
    assert len(set(rendered)) == 4
    assert all(rendered[index - 1] in rendered[index] for index in range(1, 4))


def test_task_group_dry_run_changes_only_prompt_and_receipt_is_prompt_free() -> None:
    plan = _json(PLAN)
    selected = copy.deepcopy(plan["tasks"][0])
    task = _live_task(selected)
    payload, receipt = prompt_curriculum.build_task_group_payload(plan, selected, task)

    assert payload["task"]["key"] == selected["task_key"]
    assert len(payload["members"]) == 4
    assert [member["label"] for member in payload["members"]] == [
        "level-0",
        "level-1",
        "level-2",
        "level-3",
    ]
    assert payload["members"][0]["prompt"] == task["prompt"]
    assert receipt["planned_sessions"] == 4
    assert receipt["exact_version_bindings"] == {
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
            "id": receipt["source_binding"]["verifier_id"],
            "version": receipt["source_binding"]["verifier_version"],
            "version_id": receipt["source_binding"]["verifier_version_id"],
            "sha256": receipt["source_binding"]["verifier_sha256"],
        },
        "runtime_seed_content_sha256": receipt["source_binding"]["runtime_seed_content_sha256"],
    }
    assert receipt["task_group_create_performed"] is False
    assert receipt["paid_job_submitted"] is False
    assert receipt["prepared_payload_invariants"] == {
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
            "exact_task_graph_source": receipt["registry_task_graph_source"],
            "source_binding": receipt["source_binding"],
            "non_prompt_task_spec_sha256": receipt["non_prompt_task_spec_sha256"],
        },
        "created_member_hydration_pending": True,
    }
    serialized_receipt = json.dumps(receipt)
    assert "Private source task prompt" not in serialized_receipt
    assert "not-for-receipt" not in serialized_receipt


def test_create_once_writer_refuses_duplicate_artifact(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    prompt_curriculum._write_create_once(path, {"first": True})
    with pytest.raises(FileExistsError, match="refusing to replace"):
        prompt_curriculum._write_create_once(path, {"second": True})


def test_prepare_is_get_only_and_never_calls_post(tmp_path: Path) -> None:
    plan = _json(PLAN)
    selected = copy.deepcopy(plan["tasks"])
    tasks = {}
    for row in selected:
        tasks[row["task_key"]] = _live_task(row)

    methods = []

    class Client:
        def request(self, method: str, url: str, **kwargs):
            methods.append(method)
            if url.endswith("/v1/account"):
                body = {
                    "team_name": "fleet",
                    "team_id": self_hosted.FLEET_TEAM_ID,
                }
            else:
                key = next(key for key in tasks if url.endswith(f"/v1/tasks/{key}"))
                assert kwargs["params"] == {
                    "version_id": next(
                        row["task_version_id"] for row in selected if row["task_key"] == key
                    )
                }
                body = tasks[key]
            return type(
                "Response",
                (),
                {"status_code": 200, "json": lambda self: body},
            )()

    summary = prompt_curriculum.prepare_live_dry_runs(
        Client(), plan, selected, tmp_path / "prepared"
    )
    assert methods == ["GET", "GET", "GET"]
    assert summary["request_audit"]["requests"] == prompt_curriculum._expected_gets(selected)
    assert summary["request_audit"]["mutation_request_count"] == 0
    assert summary["task_group_create_performed"] is False
    assert summary["paid_job_submitted"] is False


def test_prepare_review_persists_only_prompt_free_audited_plan(tmp_path: Path) -> None:
    plan = _json(PLAN)
    selected = plan["tasks"]
    tasks = {row["task_key"]: _live_task(row) for row in selected}

    class Client:
        def request(self, method: str, url: str, **kwargs):
            if url.endswith("/v1/account"):
                body = {"team_name": "fleet", "team_id": self_hosted.FLEET_TEAM_ID}
            else:
                key = next(key for key in tasks if url.endswith(f"/v1/tasks/{key}"))
                body = tasks[key]
            return type(
                "Response",
                (),
                {"status_code": 200, "json": lambda self: body},
            )()

    out_dir = tmp_path / "review"
    review = prompt_curriculum.prepare_live_review_plan(Client(), plan, selected, out_dir)
    assert sorted(path.name for path in out_dir.iterdir()) == ["review-plan.json"]
    assert review["request_audit"]["expected_request_count"] == 3
    assert review["request_audit"]["observed_request_count"] == 3
    assert review["request_audit"]["mutation_request_count"] == 0
    assert all(task["variant_count"] == 4 for task in review["tasks"])
    serialized = (out_dir / "review-plan.json").read_text()
    assert "Private source task prompt" not in serialized
    assert "not-for-receipt" not in serialized


def test_review_plan_rejects_forged_request_audit_and_task_receipt() -> None:
    plan = _json(PLAN)
    selected = plan["tasks"]
    receipts = []
    for row in selected:
        _, receipt = prompt_curriculum.build_task_group_payload(plan, row, _live_task(row))
        receipts.append(receipt)
    expected = prompt_curriculum._expected_gets(selected)
    client = prompt_curriculum._PredeclaredGetClient(object(), expected)
    client._observed = copy.deepcopy(expected)
    audit = client.receipt()

    forged_audit = copy.deepcopy(audit)
    forged_audit["requests"][0]["method"] = "POST"
    forged_audit["request_audit_sha256"] = self_hosted.sha256(
        self_hosted.canonical_json(
            {key: value for key, value in forged_audit.items() if key != "request_audit_sha256"}
        )
    )
    with pytest.raises(ValueError, match="exact GET-only contract"):
        prompt_curriculum.build_review_plan(plan, receipts, forged_audit)

    forged_receipts = copy.deepcopy(receipts)
    forged_receipts[0]["paid_job_submitted"] = True
    forged_receipts[0]["receipt_sha256"] = self_hosted.sha256(
        self_hosted.canonical_json(
            {key: value for key, value in forged_receipts[0].items() if key != "receipt_sha256"}
        )
    )
    with pytest.raises(ValueError, match="malformed task receipt"):
        prompt_curriculum.build_review_plan(plan, forged_receipts, audit)


@pytest.mark.parametrize(
    ("method", "url", "params"),
    [
        ("POST", f"{self_hosted.ORCHESTRATOR}/v1/account", {}),
        ("GET", f"{self_hosted.ORCHESTRATOR}/v1/tasks/unplanned", {}),
        ("GET", "https://example.invalid/v1/account", {}),
        ("GET", f"{self_hosted.ORCHESTRATOR}/v1/account?extra=true", {}),
    ],
)
def test_request_auditor_rejects_mutation_route_and_origin_drift(
    method: str, url: str, params: dict
) -> None:
    class Delegate:
        def request(self, method: str, url: str, **kwargs):
            raise AssertionError("undeclared request reached the network delegate")

    client = prompt_curriculum._PredeclaredGetClient(
        Delegate(), [{"method": "GET", "path": "/v1/account", "params": {}}]
    )
    with pytest.raises(RuntimeError, match="undeclared request|mutation"):
        client.request(method, url, params=params)
