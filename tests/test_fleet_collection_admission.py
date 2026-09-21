"""Exercise the metadata-only Fleet collection admission boundary.

All fixtures are synthetic metadata and digests.  They deliberately contain no
prompts, transcripts, messages, tool output, or other rollout payload.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from cyber_post_train.jobs import digest
from training import fleet_collection_admission as admission
from training import task_family_split as splits
from training.io import file_sha256


def _sha(label: str) -> str:
    return "sha256:" + digest({"synthetic_label": label})


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    return {**value, "sha256": "sha256:" + digest(value)}


def _rows() -> list[dict[str, Any]]:
    return [
        {
            "task_key": f"task-{index:02d}",
            "task_version_id": f"version-{index:02d}",
            "lineage": {
                "application": f"app-{index % 3}",
                "environment": f"environment-{index % 3}",
                "task_family": f"family-{index:02d}",
                "difficulty": f"difficulty-{index % 3}",
                "vulnerability_family": [f"vulnerability-{index % 3}"],
            },
        }
        for index in range(15)
    ]


def _write_json(path: Path, value: dict[str, Any]) -> dict[str, str]:
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return {"path": path.name, "sha256": file_sha256(path)}


def _write_attempts(path: Path, rows: list[dict[str, Any]]) -> dict[str, str]:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return {"path": path.name, "sha256": file_sha256(path)}


def _fixture(
    tmp_path: Path,
    *,
    train_count: int,
    dev_count: int,
    final_test_count: int = 0,
    pass_k: int = 1,
) -> dict[str, Any]:
    rows = _rows()
    inventory = _seal({"schema": "synthetic_sanitized_task_catalog_v1", "task_versions": rows})
    split = splits.build(
        rows,
        inventory_sha256=inventory["sha256"],
        seed="admission-fixture-v1",
        ratios={"train": 0.6, "dev": 0.2, "final_test": 0.2},
        max_group_task_version_fraction=0.7,
    )
    role_anchor = splits.freeze_role_anchor(split, rows)
    split = splits.build_anchored(
        rows,
        inventory_sha256=inventory["sha256"],
        role_anchor=role_anchor,
        seed="admission-fixture-v1",
        ratios={"train": 0.6, "dev": 0.2, "final_test": 0.2},
        max_group_task_version_fraction=0.7,
    )
    by_role: dict[str, list[dict[str, str]]] = {"train": [], "dev": [], "final_test": []}
    for row in split["tasks"]:
        by_role[row["split"]].append(row)
    selected = (
        by_role["train"][:train_count]
        + by_role["dev"][:dev_count]
        + by_role["final_test"][:final_test_count]
    )
    assert len(selected) == train_count + dev_count + final_test_count
    heldout_groups = sorted({row["group_id"] for row in split["tasks"] if row["split"] != "train"})
    lock = _seal(
        {
            "schema": admission.PROTECTED_FAMILY_LOCK_SCHEMA,
            "source_split_sha256": split["sha256"],
            "heldout_group_ids": heldout_groups,
        }
    )
    model = {
        "repository": "Qwen/Qwen3.8-27B",
        "revision": "synthetic-model-revision-v1",
        "session_model": "qwen3.8-27b",
    }
    treatment = {"tools": ["bash", "submit_report"], "tool_catalog_sha256": _sha("tools")}
    campaign = {
        "schema": admission.CAMPAIGN_SCHEMA,
        "training_data_eligible": True,
        "models": {"qwen": model},
        "treatment": treatment,
        "tasks": [
            {"task_key": row["task_key"], "task_version_id": row["task_version_id"]}
            for row in selected
        ],
        "pass_k": pass_k,
        "routes": {
            "qwen-route": {
                "model": "qwen",
                "task_versions": [row["task_version_id"] for row in selected],
            }
        },
    }
    campaign["sha256"] = digest(campaign)
    refs = {
        "campaign": _write_json(tmp_path / "campaign.json", campaign),
        "inventory": _write_json(tmp_path / "inventory.json", inventory),
        "family_split": _write_json(tmp_path / "split.json", split),
        "role_anchor": _write_json(tmp_path / "role-anchor.json", role_anchor),
        "protected_family_lock": _write_json(tmp_path / "protected-lock.json", lock),
    }
    return {
        "campaign": campaign,
        "inventory": inventory,
        "split": split,
        "role_anchor": role_anchor,
        "lock": lock,
        "model": model,
        "treatment": treatment,
        "by_role": by_role,
        "selected": selected,
        "refs": refs,
        "tmp_path": tmp_path,
    }


def _attempt(
    fixture: dict[str, Any],
    row: dict[str, str],
    *,
    session_id: str,
    attempt_number: int = 1,
    **overrides: Any,
) -> dict[str, Any]:
    campaign = fixture["campaign"]
    policy = {
        "target_mode": "visible_actions_only",
        "reasoning_visibility": "absent",
        "compaction": "none",
    }
    policy.update(overrides.pop("content_policy", {}))
    outcome = {
        "status": "completed",
        "verifier_process_success": True,
        "score_at_least_one": True,
    }
    outcome.update(overrides.pop("outcome", {}))
    value = {
        "schema": admission.ATTEMPT_SCHEMA,
        "session_id": session_id,
        "campaign_plan_sha256": "sha256:" + campaign["sha256"],
        "cell_sha256": admission._cell_sha(
            campaign_sha256="sha256:" + campaign["sha256"],
            task_key=row["task_key"],
            task_version_id=row["task_version_id"],
            model_alias="qwen",
            model_revision=fixture["model"]["revision"],
            attempt=attempt_number,
        ),
        "task_key": row["task_key"],
        "task_version_id": row["task_version_id"],
        "model_alias": "qwen",
        "attempt": attempt_number,
        "model": fixture["model"],
        "harness": {
            "treatment_sha256": "sha256:" + digest(fixture["treatment"]),
            "tool_catalog_sha256": fixture["treatment"]["tool_catalog_sha256"],
        },
        "template_sha256": _sha("template"),
        "outcome": outcome,
        "ingestion": {
            "status": "complete",
            "normalized_record_sha256": _sha(f"record:{session_id}"),
            "normalized_trajectory_sha256": _sha(f"trajectory:{session_id}"),
            "transcript_sha256": _sha(f"transcript:{session_id}"),
        },
        "content_policy": policy,
    }
    value.update(overrides)
    return _seal(value)


def _request(
    fixture: dict[str, Any],
    attempts: list[dict[str, Any]],
    *,
    output: str = "admission",
    cap: int = 2,
):
    attempts_path = fixture["tmp_path"] / "attempts.jsonl"
    return {
        "schema": admission.REQUEST_SCHEMA,
        **fixture["refs"],
        "attempts": _write_attempts(attempts_path, attempts),
        "source": {
            "kind": "teacher",
            "model_alias": "qwen",
            "template_sha256": _sha("template"),
        },
        "max_sessions_per_task_version": cap,
        "output": output,
    }


def _run(fixture: dict[str, Any], attempts: list[dict[str, Any]]) -> tuple[dict[str, Any], Path]:
    request = _request(fixture, attempts)
    result = admission.build(request, relative_to=fixture["tmp_path"])
    return result, fixture["tmp_path"] / request["output"]


def test_metadata_only_handoff_excludes_heldout_private_opaque_and_duplicates(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, train_count=4, dev_count=1)
    train = fixture["by_role"]["train"]
    dev = fixture["by_role"]["dev"]
    first = _attempt(fixture, train[0], session_id="session-train-a")
    duplicate = _attempt(fixture, train[1], session_id="session-train-b")
    duplicate["ingestion"]["normalized_trajectory_sha256"] = first["ingestion"][
        "normalized_trajectory_sha256"
    ]
    duplicate = _seal({key: value for key, value in duplicate.items() if key != "sha256"})
    attempts = [
        first,
        duplicate,
        _attempt(fixture, dev[0], session_id="session-dev"),
        _attempt(
            fixture,
            train[2],
            session_id="session-opaque",
            content_policy={"compaction": "opaque"},
        ),
        _attempt(
            fixture,
            train[3],
            session_id="session-private-reasoning",
            content_policy={"reasoning_visibility": "private"},
        ),
    ]

    result, output = _run(fixture, attempts)
    receipt = json.loads((output / "ADMISSION.json").read_text())
    selection = json.loads((output / "selection.private.json").read_text())

    assert result["artifact_kind"] == admission.HANDOFF_KIND
    assert result["trainable_corpus_created"] is False
    assert result["parquet_created"] is False
    assert result["source_text_read"] is False
    assert "output" not in result
    assert receipt["catalog_inventory_sha256"] == fixture["inventory"]["sha256"]
    assert receipt["source_model_alias"] == "qwen"
    assert receipt["counts"]["admitted_sessions"] == 1
    assert receipt["counts"]["rejections"] == {
        **{reason: 0 for reason in admission._REJECTION_REASONS},
        "exact_trajectory_duplicate": 1,
        "heldout_or_nontrain": 1,
        "opaque_compaction": 1,
        "private_or_unknown_reasoning": 1,
    }
    assert receipt["trainable_corpus_created"] is False
    assert receipt["parquet_created"] is False
    assert receipt["sha256"] == "sha256:" + digest(
        {key: value for key, value in receipt.items() if key != "sha256"}
    )
    assert "session-train-a" not in json.dumps(receipt)
    assert "task-" not in json.dumps(receipt)
    assert selection["trainable_corpus_created"] is False
    assert selection["parquet_created"] is False
    assert selection["source_model_alias"] == "qwen"
    assert selection["sha256"] == "sha256:" + digest(
        {key: value for key, value in selection.items() if key != "sha256"}
    )
    assert len(selection["selected"]) == 1
    assert (output / "selection.private.json").stat().st_mode & 0o777 == 0o600
    assert (output / "ADMISSION.json").stat().st_mode & 0o777 == 0o600


def test_private_reasoning_marker_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, train_count=1, dev_count=0)
    result, _ = _run(
        fixture,
        [
            _attempt(
                fixture,
                fixture["by_role"]["train"][0],
                session_id="session-private",
                content_policy={"reasoning_visibility": "private"},
            )
        ],
    )
    assert result["admitted_sessions"] == 0
    assert result["rejections"]["private_or_unknown_reasoning"] == 1


def test_campaign_must_explicitly_authorize_training_data_collection(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, train_count=1, dev_count=0)
    changed = copy.deepcopy(fixture["campaign"])
    changed["training_data_eligible"] = False
    changed["sha256"] = digest({key: value for key, value in changed.items() if key != "sha256"})
    fixture["campaign"] = changed
    fixture["refs"]["campaign"] = _write_json(tmp_path / "campaign.json", changed)
    request = _request(
        fixture,
        [_attempt(fixture, fixture["by_role"]["train"][0], session_id="not-authorized")],
    )

    with pytest.raises(ValueError, match="not explicitly eligible"):
        admission.build(request, relative_to=tmp_path)
    assert not (tmp_path / request["output"]).exists()


def test_opaque_compaction_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, train_count=1, dev_count=0)
    result, _ = _run(
        fixture,
        [
            _attempt(
                fixture,
                fixture["by_role"]["train"][0],
                session_id="session-opaque",
                content_policy={"compaction": "opaque"},
            )
        ],
    )
    assert result["admitted_sessions"] == 0
    assert result["rejections"]["opaque_compaction"] == 1


def test_duplicate_sessions_are_excluded_deterministically(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, train_count=2, dev_count=0)
    result, _ = _run(
        fixture,
        [
            _attempt(fixture, fixture["by_role"]["train"][0], session_id="same-session"),
            _attempt(fixture, fixture["by_role"]["train"][1], session_id="same-session"),
        ],
    )
    assert result["admitted_sessions"] == 0
    assert result["rejections"]["duplicate_session"] == 2


def test_per_task_session_cap_is_exact_and_deterministic(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, train_count=1, dev_count=0, pass_k=3)
    task = fixture["by_role"]["train"][0]
    attempts = [
        _attempt(fixture, task, session_id=f"session-{attempt}", attempt_number=attempt)
        for attempt in (1, 2, 3)
    ]
    request = _request(fixture, attempts, cap=2)
    result = admission.build(request, relative_to=tmp_path)
    selection = json.loads((tmp_path / request["output"] / "selection.private.json").read_text())

    assert result["admitted_sessions"] == 2
    assert result["rejections"]["per_task_session_cap"] == 1
    assert [row["session_id"] for row in selection["selected"]] == [
        row["session_id"] for row in sorted(attempts, key=admission._selection_rank)[:2]
    ]


def test_missing_authoritative_grading_fails_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, train_count=1, dev_count=0)
    attempt = _attempt(fixture, fixture["by_role"]["train"][0], session_id="missing-grade")
    attempt["outcome"].pop("score_at_least_one")
    attempt = _seal({key: value for key, value in attempt.items() if key != "sha256"})
    request = _request(fixture, [attempt])

    with pytest.raises(ValueError, match="missing authoritative grading"):
        admission.build(request, relative_to=tmp_path)
    assert not (tmp_path / request["output"]).exists()


def test_catalog_split_digest_drift_fails_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, train_count=1, dev_count=0)
    changed = copy.deepcopy(fixture["inventory"])
    changed["catalog_revision"] = "synthetic-catalog-revision-v2"
    changed = _seal({key: value for key, value in changed.items() if key != "sha256"})
    fixture["inventory"] = changed
    fixture["refs"]["inventory"] = _write_json(tmp_path / "inventory.json", changed)
    request = _request(
        fixture,
        [_attempt(fixture, fixture["by_role"]["train"][0], session_id="catalog-drift")],
    )

    with pytest.raises(ValueError, match="current sanitized task catalog"):
        admission.build(request, relative_to=tmp_path)
    assert not (tmp_path / request["output"]).exists()


def test_protected_family_lock_cannot_drop_a_heldout_family(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, train_count=1, dev_count=0)
    changed = copy.deepcopy(fixture["lock"])
    changed["heldout_group_ids"] = changed["heldout_group_ids"][1:]
    changed = _seal({key: value for key, value in changed.items() if key != "sha256"})
    fixture["lock"] = changed
    fixture["refs"]["protected_family_lock"] = _write_json(
        tmp_path / "protected-lock.json", changed
    )
    request = _request(
        fixture,
        [_attempt(fixture, fixture["by_role"]["train"][0], session_id="locked-family")],
    )

    with pytest.raises(ValueError, match="exactly every immutable nontraining family"):
        admission.build(request, relative_to=tmp_path)
    assert not (tmp_path / request["output"]).exists()


def test_anchored_admission_rejects_a_resealed_different_parent_anchor(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, train_count=1, dev_count=0)
    changed = copy.deepcopy(fixture["role_anchor"])
    changed["source"]["split_sha256"] = _sha("different-parent")
    changed["sha256"] = splits.canonical_digest(
        {key: value for key, value in changed.items() if key != "sha256"}
    )
    fixture["refs"]["role_anchor"] = _write_json(tmp_path / "role-anchor.json", changed)
    request = _request(
        fixture,
        [_attempt(fixture, fixture["by_role"]["train"][0], session_id="wrong-anchor")],
    )

    with pytest.raises(ValueError, match="parent role anchor digest mismatch"):
        admission.build(request, relative_to=tmp_path)
    assert not (tmp_path / request["output"]).exists()


def test_raw_payload_field_is_not_accepted_as_metadata(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, train_count=1, dev_count=0)
    attempt = _attempt(fixture, fixture["by_role"]["train"][0], session_id="bad-payload")
    attempt["messages"] = []
    attempt = _seal({key: value for key, value in attempt.items() if key != "sha256"})
    request = _request(fixture, [attempt])

    with pytest.raises(ValueError, match="unknown fields"):
        admission.build(request, relative_to=tmp_path)
    assert not (tmp_path / request["output"]).exists()
