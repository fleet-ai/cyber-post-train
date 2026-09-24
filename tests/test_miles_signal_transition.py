from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

from training import miles_signal_transition as transition
from training import miles_signal_wave

NOW = dt.datetime(2026, 9, 24, 9, 0, tzinfo=dt.UTC)


def SHA(digit):
    return "sha256:" + hashlib.sha256(str(digit).encode()).hexdigest()


class FakeClient:
    def __init__(self, wave):
        self.wave = wave

    def _get(self, path, params=None):
        if path == "/v1/account":
            return {"team_id": transition.fleet.FLEET_TEAM_ID, "team_name": "fleet"}
        version = params["version_id"]
        row = next(row for row in self.wave["candidates"] if row["task"]["version_id"] == version)
        return {
            "key": row["task"]["key"],
            "eval_task_version_id": version,
            "task_lifecycle_status": "production",
            "metadata": {
                "cyber_contract": {
                    "evidence_schema": "1.0.0",
                    "submission_protocol": "2.0.0",
                    "verifier_contract": "3.0.0",
                }
            },
        }


def _safe_bind(monkeypatch, wave):
    by_version = {row["task"]["version_id"]: row for row in wave["candidates"]}

    def bind(_response, selected):
        row = by_version[selected["task_version_id"]]
        task = {
            "key": row["task"]["key"],
            "version_id": row["task"]["version_id"],
            "prompt_sha256": row["task"]["prompt_sha256"],
            "env_variables_sha256": row["task"]["env_variables_sha256"],
            "output_json_schema_sha256": row["task"]["output_json_schema_sha256"],
        }
        environment = {**row["environment"], "ttl_seconds": 32400}
        verifier = {**row["verifier"], "function_name": "verify"}
        return task, environment, verifier

    monkeypatch.setattr(transition.fleet, "bind_task", bind)


@pytest.fixture
def live_receipt(monkeypatch):
    wave = miles_signal_wave.load()
    _safe_bind(monkeypatch, wave)
    return transition.collect_live_task_receipt(FakeClient(wave), observed_at=NOW)


def _evidence(wave, observed_at=NOW):
    lanes = []
    for index, row in enumerate(wave["candidates"]):
        digit = str(index + 1)
        lanes.append(
            {
                "name": row["identity"]["name"],
                "plan_sha256": SHA(digit),
                "request_sha256": SHA(str(index + 5)),
                "optimizer_steps": 0,
                "checkpoint": False,
                "authority_config_sha256": wave["sha256"],
                "live_binding_receipt_sha256": row["live_binding_receipt_sha256"],
                "preview": {
                    "request_sha256": SHA(str(index + 5)),
                    "receipt_sha256": SHA(chr(ord("a") + index)),
                    "observed_at": transition._stamp(observed_at),
                    "root_annotations": {"fleet.ai/failure-alerts": "off"},
                    "priority_class": "c1",
                    "queue_priority": "q1",
                    "backoff_limit": 0,
                    "nodes": 1,
                    "gpus_per_node": 8,
                    "requeue_if_preempted": False,
                    "optimizer_steps": 0,
                    "checkpoint": False,
                },
                "absence": {
                    "receipt_sha256": SHA(chr(ord("e") + index)),
                    "observed_at": transition._stamp(observed_at),
                    "jobs_api_duplicates": 0,
                    "kubernetes_duplicates": 0,
                    "sfs_output_absent": True,
                },
                "observer": {
                    "receipt_sha256": SHA(chr(ord("i") + index)),
                    "armed": True,
                    "healthy": True,
                    "uid_bound_release": True,
                    "delete_drain_supervised": True,
                    "active_deadline_s": 18_900,
                },
            }
        )
    return {
        "adapter": {
            "commit": "1" * 40,
            "clean": True,
            "source_closure_sha256": SHA("a"),
            "runtime_image": "example.invalid/fti@" + SHA("b"),
            "tests_receipt_sha256": SHA("c"),
            "exact_image_preflight_sha256": SHA("d"),
            "sample_indexes": list(range(8)),
            "max_concurrent_episodes": 2,
            "optimizer_steps": 0,
            "checkpoint": False,
            "live_session_open_tool_schema_gate": True,
            "tool_catalog_sha256": wave["authorities"]["tool_catalog"]["self_sha256"],
            "current_binding_source": "live_binding_receipt_sha256",
            "no_outer_retry_or_replacement": True,
            "all_slots_terminally_accounted": True,
            "unique_verifier_execution_ids": True,
            "exact_cleanup_required": True,
        },
        "operator": {
            "commit": "2" * 40,
            "clean": True,
            "tests_receipt_sha256": SHA("f"),
        },
        "lanes": lanes,
        "cleanup": {
            "receipt_sha256": SHA("f"),
            "prompt_removed_after_terminal": True,
            "private_episode_material_restricted": True,
            "output_create_once": True,
        },
    }


def _load_schema(name):
    return json.loads((Path("configs/qualification") / name).read_text())


def test_live_receipt_is_sanitized_exact_and_schema_valid(live_receipt):
    transition.validate_live_task_receipt(live_receipt, now=NOW, require_fresh=True)
    jsonschema.Draft202012Validator(
        _load_schema("qwen38-miles-signal-live-task-receipt-v1.schema.json"),
        format_checker=jsonschema.FormatChecker(),
    ).validate(live_receipt)
    encoded = json.dumps(live_receipt)
    for forbidden in ('"prompt"', '"code"', '"env_variables"', '"credentials"'):
        assert forbidden not in encoded


def test_stale_or_drifted_live_receipt_fails_closed(live_receipt):
    with pytest.raises(ValueError, match="stale"):
        transition.validate_live_task_receipt(
            live_receipt, now=NOW + dt.timedelta(seconds=901), require_fresh=True
        )
    drifted = copy.deepcopy(live_receipt)
    drifted["candidates"][0]["task"]["version_id"] = "00000000-0000-4000-8000-000000000000"
    drifted = transition._sealed({k: v for k, v in drifted.items() if k != "sha256"})
    with pytest.raises(ValueError, match="candidates"):
        transition.validate_live_task_receipt(drifted)


def test_default_transition_is_unlaunchable_and_lists_all_nine_gates(live_receipt):
    candidate = transition.build_review_candidate(live_receipt, observed_at=NOW)
    transition.validate_review_candidate(candidate)
    assert candidate["launchable"] is False
    assert candidate["state"] == "parent_review_required"
    assert [row["id"] for row in candidate["gates"]] == list(transition.GATES)
    assert [row["passed"] for row in candidate["gates"]] == [
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
    ]


def test_committed_receipts_are_sanitized_and_unlaunchable():
    live = json.loads(
        Path(
            "configs/qualification/qwen38-miles-signal-live-task-receipt-20260924.json"
        ).read_text()
    )
    candidate = json.loads(
        Path(
            "configs/qualification/qwen38-miles-signal-transition-candidate-20260924.json"
        ).read_text()
    )
    transition.validate_live_task_receipt(live)
    transition.validate_review_candidate(candidate)
    assert candidate["authority"]["task_get_receipt_sha256"] == live["sha256"]
    assert candidate["launchable"] is False
    assert candidate["future_bindings"] == {}


def test_complete_evidence_still_requires_separate_parent_review(live_receipt):
    wave = miles_signal_wave.load()
    candidate = transition.build_review_candidate(live_receipt, _evidence(wave), observed_at=NOW)
    assert all(row["passed"] for row in candidate["gates"][:-1])
    assert candidate["gates"][-1]["passed"] is False
    assert candidate["launchable"] is False
    jsonschema.Draft202012Validator(
        _load_schema("qwen38-miles-signal-reviewed-transition-v1.schema.json"),
        format_checker=jsonschema.FormatChecker(),
    ).validate(candidate)
    transition.validate_review_candidate(candidate, live_receipt)


def test_gate_booleans_cannot_be_forged(live_receipt):
    candidate = transition.build_review_candidate(live_receipt, observed_at=NOW)
    candidate["gates"][0]["passed"] = True
    candidate = transition._sealed({k: v for k, v in candidate.items() if k != "sha256"})
    with pytest.raises(ValueError, match="not reproducible"):
        transition.validate_review_candidate(candidate, live_receipt)


def test_parent_review_only_approves_exact_complete_candidate(live_receipt):
    candidate = transition.build_review_candidate(
        live_receipt, _evidence(miles_signal_wave.load()), observed_at=NOW
    )
    review = transition._sealed(
        {
            "schema": transition.REVIEW_SCHEMA,
            "candidate_sha256": candidate["sha256"],
            "approved": True,
            "reviewer": "root",
            "reviewed_at": transition._stamp(NOW),
        }
    )
    approved = transition.approve_review_candidate(candidate, review, live_receipt, reviewed_at=NOW)
    assert approved["launchable"] is True
    assert approved["gates"][-1]["passed"] is True
    wrong = copy.deepcopy(review)
    wrong["candidate_sha256"] = SHA("0")
    wrong = transition._sealed({k: v for k, v in wrong.items() if k != "sha256"})
    with pytest.raises(ValueError, match="exact candidate"):
        transition.approve_review_candidate(candidate, wrong, live_receipt, reviewed_at=NOW)
    with pytest.raises(ValueError, match="incomplete"):
        transition.approve_review_candidate(
            candidate,
            review,
            live_receipt,
            reviewed_at=NOW + dt.timedelta(seconds=301),
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda evidence: evidence["adapter"].__setitem__("max_concurrent_episodes", 8),
        lambda evidence: evidence["adapter"].__setitem__(
            "live_session_open_tool_schema_gate", False
        ),
        lambda evidence: evidence["lanes"][0]["preview"].__setitem__("root_annotations", {}),
        lambda evidence: evidence["lanes"][0]["absence"].__setitem__("jobs_api_duplicates", 1),
        lambda evidence: evidence["lanes"][0]["observer"].__setitem__("active_deadline_s", 7_200),
    ],
)
def test_each_critical_defect_keeps_transition_unlaunchable(live_receipt, mutate):
    evidence = _evidence(miles_signal_wave.load())
    mutate(evidence)
    candidate = transition.build_review_candidate(live_receipt, evidence, observed_at=NOW)
    assert any(row["passed"] is False for row in candidate["gates"][:-1])
    with pytest.raises(ValueError, match="incomplete"):
        transition.approve_review_candidate(
            candidate,
            transition._sealed(
                {
                    "schema": transition.REVIEW_SCHEMA,
                    "candidate_sha256": candidate["sha256"],
                    "approved": True,
                    "reviewer": "root",
                    "reviewed_at": transition._stamp(NOW),
                }
            ),
            live_receipt,
            reviewed_at=NOW,
        )
