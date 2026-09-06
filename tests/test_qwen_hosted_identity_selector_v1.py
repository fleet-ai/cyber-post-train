from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from evals.fleet import qwen_hosted_identity_selector_package_v1 as package
from evals.fleet import qwen_hosted_identity_selector_v1 as selector
from evals.fleet import qwen_hosted_rank17_g22_release_observer_v1 as base
from evals.fleet import score_blind_session_inventory_v1 as stream

ROOT = Path(__file__).parents[1]
JOB_UID = "11111111-1111-4111-8111-111111111111"
POD_UID = "22222222-2222-4222-8222-222222222222"


def _page(rows: list[dict[str, object]]) -> stream.SessionPage:
    return stream.SessionPage(sessions=tuple(rows), limit=500, offset=0, has_more=False)


def _row(task_key: str, model: object, suffix: str) -> dict[str, object]:
    return {
        "session_id": f"session-{suffix}",
        "eval_task_id": f"eval-task-{suffix}",
        "task_key": task_key,
        "model": model,
        "status": "completed",
    }


def _account(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        base,
        "_fleet_get",
        lambda *_args, **_kwargs: {"team_name": "fleet", "team_id": base.FLEET_TEAM_ID},
    )


def test_binding_exactly_binds_100_task_authority_and_tally() -> None:
    binding = package.build_binding(ROOT)
    selector.validate_binding(binding)
    assert binding["task_count"] == 100
    assert [row["rank"] for row in binding["task_roster"]] == list(range(1, 101))
    assert binding["authoritative_tally"] == {
        "accepted": 51,
        "active": 0,
        "blocked_nonrepeatable": 9,
        "unstarted": 340,
    }


def test_binding_rejects_rehashed_roster_or_extra() -> None:
    for mutate in (
        lambda value: value["task_roster"][0].__setitem__("task_key", "drift"),
        lambda value: value.__setitem__("protected", True),
    ):
        binding = package.build_binding(ROOT)
        mutate(binding)
        binding["binding_sha256"] = base.binding_digest(binding)
        with pytest.raises(base.GateError, match="identity_selector_binding_invalid"):
            selector.validate_binding(binding)


def test_one_scan_classifies_all_tasks_and_emits_only_aggregate_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = package.build_binding(ROOT)
    rank1, rank2, rank3, rank4 = [row["task_key"] for row in binding["task_roster"][:4]]
    marker = "SECRET_SCORE_OR_VERIFIER_VALUE"
    rows = [
        _row(rank1, selector.EXPECTED_MODEL, marker),
        _row(rank2, "different-model", "other"),
        _row(rank3, None, "ambiguous"),
        _row(rank4, marker, "secret-model"),
    ]
    _account(monkeypatch)
    monkeypatch.setattr(selector, "_session_page", lambda *_args, **_kwargs: _page(rows))
    value = selector.select(
        binding, "not-persisted", job_uid=JOB_UID, pod_uid=POD_UID
    )
    selector.validate_observation(value, binding)
    assert value["classification_counts"] == {
        "IDENTITY_CLEAR": 98,
        "EXACT_MODEL_COLLISION": 1,
        "AMBIGUOUS_BLOCK": 1,
    }
    assert value["earliest_clear_rank"] == 2
    raw = json.dumps(value)
    assert marker not in raw
    assert "session-" not in raw
    assert "eval-task-" not in raw
    assert "side_effects" not in value


def test_missing_global_identity_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    binding = package.build_binding(ROOT)
    _account(monkeypatch)
    monkeypatch.setattr(
        selector,
        "_session_page",
        lambda *_args, **_kwargs: _page(
            [
                {
                    "session_id": "session-1",
                    "eval_task_id": "eval-task-1",
                    "task_key": None,
                    "model": "different-model",
                    "status": "completed",
                }
            ]
        ),
    )
    with pytest.raises(base.GateError, match="identity_selector_inventory_ambiguous"):
        selector.select(binding, "unused", job_uid=JOB_UID, pod_uid=POD_UID)


def test_streaming_parser_skips_protected_values_without_emitting_them() -> None:
    marker = "SECRET_SCORE_PROMPT_VERIFIER"
    raw = json.dumps(
        {
            "sessions": [
                {
                    "session_id": "one",
                    "eval_task_id": "two",
                    "task_key": "three",
                    "model": "four",
                    "status": "completed",
                    "verifier_execution": {
                        "score": 0.75,
                        "success": True,
                        "prompt": marker,
                        "nested": [{"flag": marker}],
                    },
                }
            ],
            "limit": 500,
            "offset": 0,
            "has_more": False,
        }
    ).encode()
    page = stream.parse_session_page([raw[:17], raw[17:]])
    assert marker not in repr(page)
    assert set(page.sessions[0]) == {
        "session_id",
        "eval_task_id",
        "task_key",
        "model",
        "status",
    }


def test_observation_rejects_rehashed_extra_or_effect() -> None:
    binding = package.build_binding(ROOT)
    value = base.seal(
        {
            "schema_version": selector.SCHEMA,
            "status": "CLEAR_CANDIDATE_IDENTIFIED",
            "selection_file_sha256": selector.SELECTION_FILE_SHA256,
            "selection_sha256": selector.SELECTION_SHA256,
            "roster_sha256": selector.ROSTER_SHA256,
            "binding_sha256": binding["binding_sha256"],
            "authoritative_tally": selector.EXPECTED_TALLY,
            "task_count": 100,
            "classification_counts": {
                "IDENTITY_CLEAR": 100,
                "EXACT_MODEL_COLLISION": 0,
                "AMBIGUOUS_BLOCK": 0,
            },
            "earliest_clear_rank": 1,
            "session_pages_read": 1,
            "session_rows_examined": 0,
            "runtime": {"job_uid": JOB_UID, "pod_uid": POD_UID},
            "methods": ["GET"],
            "model_calls": 0,
            "task_calls": 0,
            "session_mutations": 0,
            "verifier_calls": 0,
            "scoring_calls": 0,
            "api_mutations": 0,
            "protected_values_materialized": False,
            "scores_included": False,
            "prompts_traces_flags_included": False,
            "credentials_included": False,
        }
    )
    selector.validate_observation(value, binding)
    for mutate in (
        lambda row: row.__setitem__("protected", "SECRET"),
        lambda row: row.__setitem__("scoring_calls", 1),
    ):
        changed = copy.deepcopy(value)
        mutate(changed)
        changed["receipt_sha256"] = base.digest(changed)
        with pytest.raises(base.GateError, match="release_observation_invalid"):
            selector.validate_observation(changed, binding)


def test_manifest_is_immutable_create_once_score_free_and_held() -> None:
    configmap, job = package.render(ROOT)["items"]
    assert configmap["immutable"] is True
    assert job["metadata"]["annotations"] | {
        "cyber-post-train.fleet.ai/create-once": "true",
        "cyber-post-train.fleet.ai/launch-authorized": "false",
        "cyber-post-train.fleet.ai/score-free": "true",
    } == job["metadata"]["annotations"]
    assert job["spec"]["backoffLimit"] == 0
    assert job["spec"]["template"]["spec"]["preemptionPolicy"] == "Never"
    assert job["spec"]["template"]["spec"]["containers"][0]["resources"]["requests"].get(
        "nvidia.com/gpu"
    ) is None
    held = package.held(ROOT)
    persisted = json.loads((ROOT / package.HELD_PATH).read_text())
    assert persisted == held
    assert held["launch_authorized"] is held["scoring_authorized"] is False
    assert set(held["side_effects"].values()) == {0}


def test_projected_package_validates_and_source_drift_fails(tmp_path: Path) -> None:
    configmap = package.render(ROOT)["items"][0]
    for name, value in configmap["data"].items():
        (tmp_path / name).write_text(value)
    selector.validate_package_source(tmp_path / "package-source.json", tmp_path)
    (tmp_path / "score_blind_session_inventory_v1.py").write_text("drift")
    with pytest.raises(base.GateError, match="identity_selector_package_file_drifted"):
        selector.validate_package_source(tmp_path / "package-source.json", tmp_path)
