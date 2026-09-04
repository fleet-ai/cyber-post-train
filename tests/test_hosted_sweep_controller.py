from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

import pytest

from evals.fleet import hosted_sweep_controller as hosted
from evals.fleet import opencode_train_sweep as sweep
from evals.fleet import self_hosted

SOURCE = Path(
    "docs/evidence/qwen38-study/2026-09-04-hosted-opencode-successor-source-v1.json"
)
REMAINDER_SOURCE = Path(
    "docs/evidence/qwen38-study/2026-09-04-hosted-opencode-remainder-source-v1.json"
)
REMAINDER_SOURCE_V3 = Path(
    "docs/evidence/qwen38-study/2026-09-04-hosted-opencode-remainder-source-v3.json"
)
REPLACEMENT_LOCK = Path(
    "docs/evidence/qwen38-study/2026-09-04-opencode-replacement-selection-lock-v1.json"
)
HYDRATION_JOB_V2 = Path(
    "evals/fleet/cluster/glm53-dedicated-a-hydration-job-v2.yaml"
)
DEDICATED_A_PLAN = Path(
    "evals/fleet/configs/glm53-opencode-dedicated-a-even27-pass4-v1.json"
)
DEDICATED_B_PLAN = Path(
    "evals/fleet/configs/glm53-opencode-dedicated-b-even27-pass4-v1.json"
)
QWEN_REPLACEMENT_PLAN = Path(
    "evals/fleet/configs/qwen38-opencode-hosted-replacements3-pass4-v1.json"
)
DEDICATED_SCORING_RELEASE = Path(
    "docs/evidence/qwen38-study/2026-09-04-glm53-dedicated-scoring-release-v1.json"
)
REPLACEMENT_SUPPLEMENT = Path(
    "docs/evidence/qwen38-study/2026-09-04-opencode-replacement-selection-supplement-v1.json"
)
REPLACEMENT_SUPPLEMENT_V2 = Path(
    "docs/evidence/qwen38-study/2026-09-04-opencode-replacement-selection-supplement-v2.json"
)
REPLACEMENT_SUPPLEMENT_V2 = Path(
    "docs/evidence/qwen38-study/2026-09-04-opencode-replacement-selection-supplement-v2.json"
)
GLM_HOSTED_REPLACEMENT_HYDRATION = Path(
    "docs/evidence/qwen38-study/2026-09-04-glm53-hosted-r106-hydration-v1.json"
)
GLM_HOSTED_REPLACEMENT_PLAN = Path(
    "evals/fleet/configs/glm53-opencode-hosted-replacement-r106-pass4-v1.json"
)
GLM_HOSTED_REPLACEMENT_RELEASE = Path(
    "docs/evidence/qwen38-study/2026-09-04-glm53-hosted-r106-scoring-release-v1.json"
)
DEDICATED_B_STOP_TOMBSTONE = Path(
    "docs/evidence/qwen38-study/2026-09-04-glm53-dedicated-b-forced-stop-tombstone-v1.json"
)
DEDICATED_A_STOP_TOMBSTONE_V2 = Path(
    "docs/evidence/qwen38-study/2026-09-04-glm53-dedicated-a-controller-stop-tombstone-v2.json"
)
DEDICATED_B_R107_HYDRATION = Path(
    "docs/evidence/qwen38-study/2026-09-04-glm53-dedicated-b-r107-hydration-v1.json"
)
GLM_HOSTED_REASSIGNED_B_PLAN = Path(
    "evals/fleet/configs/glm53-opencode-hosted-reassigned-b27-pass4-v1.json"
)
QWEN_HTTP500_REPLACEMENT_SUPPLEMENT = Path(
    "docs/evidence/qwen38-study/2026-09-04-qwen38-replacement-selection-supplement-v2.json"
)
GLM_HTTP500_REPLACEMENT_SUPPLEMENT = Path(
    "docs/evidence/qwen38-study/2026-09-04-opencode-replacement-selection-supplement-v3.json"
)
GLM_DEDICATED_A_PREEMPTION_SUPPLEMENT = Path(
    "docs/evidence/qwen38-study/2026-09-04-opencode-replacement-selection-supplement-v4.json"
)
QWEN_SOURCE6_REPLACEMENT_SUPPLEMENT = Path(
    "docs/evidence/qwen38-study/2026-09-04-qwen38-replacement-selection-supplement-v3.json"
)
GLM_SOURCE13_REPLACEMENT_SUPPLEMENT = Path(
    "docs/evidence/qwen38-study/2026-09-04-opencode-replacement-selection-supplement-v5.json"
)
HOSTED_HTTP500_INCIDENT = Path(
    "docs/evidence/qwen38-study/2026-09-04-hosted-scoring-api-common-mode-http500-incident-v1.json"
)
QWEN_HTTP500_HYDRATION = Path(
    "docs/evidence/qwen38-study/2026-09-04-qwen38-r54-r55-hydration-v1.json"
)
GLM_HTTP500_HYDRATION = Path(
    "docs/evidence/qwen38-study/2026-09-04-glm53-r108-r109-hydration-v1.json"
)
QWEN_HTTP500_SUCCESSOR_PLAN = Path(
    "evals/fleet/configs/qwen38-opencode-hosted-http500-successor49-pass4-v8.json"
)
QWEN_HTTP500_SCORING_RELEASE = Path(
    "docs/evidence/qwen38-study/2026-09-04-qwen38-http500-successor-scoring-release-v1.json"
)
GLM_HTTP500_SUCCESSOR_PLAN = Path(
    "evals/fleet/configs/glm53-opencode-hosted-http500-successor73-pass4-v11.json"
)
GLM_HTTP500_HOSTED_PRIMARY_PLAN = Path(
    "evals/fleet/configs/glm53-opencode-hosted-http500-primary46-pass4-v12.json"
)
GLM_HTTP500_SCORING_RELEASE = Path(
    "docs/evidence/qwen38-study/2026-09-04-glm53-http500-hosted-primary-scoring-release-v1.json"
)
GLM_DEDICATED_B_V5_PLAN = Path(
    "evals/fleet/configs/glm53-opencode-dedicated-b-v5-successor27-pass4-v2.json"
)
GLM_DEDICATED_A_V5_PLAN = Path(
    "evals/fleet/configs/glm53-opencode-dedicated-a-v5-successor27-pass4-v2.json"
)
GLM_DEDICATED_A_V5_RELEASE = Path(
    "docs/evidence/qwen38-study/2026-09-04-glm53-dedicated-a-v5-scoring-release-v1.json"
)
QWEN_ATTRITION_REPLACEMENT_PLAN = Path(
    "evals/fleet/configs/qwen38-opencode-hosted-attrition-r56-pass4-v1.json"
)
GLM_ATTRITION_REPLACEMENT_PLAN = Path(
    "evals/fleet/configs/glm53-opencode-hosted-attrition-r111-pass4-v1.json"
)
GLM_DEDICATED_B_V5_RELEASE = Path(
    "docs/evidence/qwen38-study/2026-09-04-glm53-dedicated-b-v5-scoring-release-v1.json"
)
FROZEN_SPLIT = Path("configs/data/fleet-a62-task-split-v1.json")
FROZEN_SELECTION = Path(
    "evals/fleet/configs/opencode-easiest-train100-selection-v2.json"
)
FROZEN_RUNNABLE_INVENTORY = Path(
    "configs/runs/qwen36-27b-rl-base-full-runnable.json"
)


@pytest.mark.parametrize(
    ("model", "source_path", "tasks", "new_sessions", "credits", "excluded_rank"),
    [
        (
            "qwen38",
            Path("evals/fleet/configs/qwen38-opencode-train50-pass4-v3.json"),
            49,
            196,
            0,
            1,
        ),
        (
            "glm53",
            Path("evals/fleet/configs/glm53-opencode-train100-pass4-v4.json"),
            99,
            394,
            2,
            2,
        ),
    ],
)
def test_successor_owns_only_complete_task_boundaries(
    model: str,
    source_path: Path,
    tasks: int,
    new_sessions: int,
    credits: int,
    excluded_rank: int,
) -> None:
    plan = hosted.build_plan(
        hosted.load_object(source_path), hosted.load_object(SOURCE), model
    )
    hosted.validate_plan(plan)
    assert plan["task_count"] == tasks
    assert plan["new_session_count"] == new_sessions
    assert len(plan["credited_sessions"]) == credits
    assert plan["excluded_tasks"][0]["source_rank"] == excluded_rank
    assert excluded_rank not in {row["source_rank"] for row in plan["tasks"]}
    cells = {
        *( (row["rank"], row["attempt"]) for row in plan["credited_sessions"] ),
        *( (row["rank"], row["attempt"]) for row in plan["attempts"] ),
    }
    assert cells == {
        (rank, attempt)
        for rank in range(1, tasks + 1)
        for attempt in range(1, 5)
    }


def test_source_fence_cannot_be_retried_or_credited() -> None:
    source = hosted.load_object(SOURCE)
    source["runs"]["qwen38"]["fenced_noncreditable"][0]["retry_allowed"] = True
    source["receipt_sha256"] = self_hosted.digest_without(source, "receipt_sha256")
    with pytest.raises(ValueError, match="policy drifted"):
        hosted.build_plan(
            hosted.load_object(
                Path("evals/fleet/configs/qwen38-opencode-train50-pass4-v3.json")
            ),
            source,
            "qwen38",
        )


def test_glm_clean_shard_defers_all_rank1_history_and_fenced_rank2() -> None:
    plan = hosted.build_plan(
        hosted.load_object(
            Path("evals/fleet/configs/glm53-opencode-train100-pass4-v4.json")
        ),
        hosted.load_object(SOURCE),
        "glm53_clean",
    )
    hosted.validate_plan(plan)
    assert plan["task_count"] == 98
    assert plan["new_session_count"] == 392
    assert plan["credited_sessions"] == []
    assert {row["source_rank"] for row in plan["tasks"]} == set(range(3, 101))
    assert {row["source_rank"] for row in plan["excluded_tasks"]} == {1, 2}
    assert plan["execution"]["inventory_policy"] == (
        "conservative_no_same_model_session_for_task_key_v1"
    )


def test_glm_hosted_odd_shard_is_disjoint_from_dedicated_even_shard() -> None:
    plan = hosted.build_plan(
        hosted.load_object(
            Path("evals/fleet/configs/glm53-opencode-train100-pass4-v4.json")
        ),
        hosted.load_object(SOURCE),
        "glm53_hosted_odd",
    )
    hosted.validate_plan(plan)
    assert plan["task_count"] == 49
    assert plan["new_session_count"] == 196
    assert plan["credited_sessions"] == []
    assert {row["source_rank"] for row in plan["tasks"]} == set(range(3, 100, 2))
    assert {row["source_rank"] for row in plan["excluded_tasks"]} == {1, 2}
    assert {row["source_rank"] for row in plan["reserved_tasks"]} == set(
        range(4, 101, 2)
    )


def test_plan_rejects_partial_or_duplicate_cells() -> None:
    plan = hosted.load_object(
        Path("evals/fleet/configs/qwen38-opencode-hosted-complete49-pass4-v5.json")
    )
    plan["attempts"][0]["attempt"] = plan["attempts"][1]["attempt"]
    plan["plan_sha256"] = self_hosted.digest_without(plan, "plan_sha256")
    with pytest.raises(ValueError, match="Cartesian"):
        hosted.validate_plan(plan)


def test_score_blind_worker_cap_is_fixed_until_new_headroom_review() -> None:
    assert hosted._worker_cap(0) == 1
    assert hosted._worker_cap(4) == 1
    assert hosted._worker_cap(16) == 1
    assert hosted._worker_cap(10_000) == 1
    assert hosted.SCHEDULE[0]["headroom_gate"] == (
        "fixed_at_launch_no_automatic_widening"
    )
    assert all("score" not in key for stage in hosted.SCHEDULE for key in stage)


def test_exact_treatment_inventory_requires_harness_and_tool_digest(monkeypatch) -> None:
    plan = hosted.load_object(
        Path("evals/fleet/configs/glm53-opencode-hosted-complete99-pass4-v5.json")
    )
    task_key = plan["tasks"][0]["task"]["key"]
    exact_metadata = {
        "self_hosted_harness": "opencode-1.18.27",
        "tool_catalog_sha256": plan["execution"]["required_task_tool_catalog_sha256"],
    }
    rows = [
        {"session_id": "exact", "model": "glm-5.3", "metadata": exact_metadata},
        {"session_id": "other-harness", "model": "glm-5.3", "metadata": {}},
        {"session_id": "other-model", "model": "qwen3.8-27b", "metadata": exact_metadata},
    ]
    monkeypatch.setattr(
        self_hosted,
        "_task_sessions",
        lambda _client, key: rows if key == task_key else [],
    )
    exact = hosted._exact_treatment_sessions(object(), plan, task_key)
    assert [row["session_id"] for row in exact] == ["exact"]


def test_preflight_rejects_current_plan_claim_in_any_sfs_job_root(tmp_path: Path) -> None:
    plan = hosted.load_object(
        Path("evals/fleet/configs/glm53-opencode-hosted-odd49-pass4-v7.json")
    )
    jobs = tmp_path / "jobs"
    other = jobs / "older-controller" / "claims"
    other.mkdir(parents=True)
    run_id = plan["attempts"][0]["run_id"]
    (other / f"{run_id}.json").write_text("{}")
    with pytest.raises(RuntimeError, match="run identity already exists"):
        hosted._validate_plan_identity_absence(plan, jobs / plan["campaign_id"])


@pytest.mark.parametrize(
    ("model", "predecessor", "tasks", "source_ranks", "excluded"),
    [
        (
            "qwen38_remainder",
            "qwen38-opencode-hosted-complete49-pass4-v5.json",
            48,
            set(range(3, 51)),
            {2},
        ),
        (
            "glm53_remainder",
            "glm53-opencode-hosted-odd49-pass4-v7.json",
            47,
            set(range(7, 100, 2)),
            {3, 5},
        ),
    ],
)
def test_remainder_shards_exclude_every_partially_touched_task(
    model: str,
    predecessor: str,
    tasks: int,
    source_ranks: set[int],
    excluded: set[int],
) -> None:
    plan = hosted.build_remainder_plan(
        hosted.load_object(Path("evals/fleet/configs") / predecessor),
        hosted.load_object(REMAINDER_SOURCE),
        model,
    )
    hosted.validate_plan(plan)
    assert plan["task_count"] == tasks
    assert plan["new_session_count"] == tasks * 4
    assert {row["source_rank"] for row in plan["tasks"]} == source_ranks
    assert {row["source_rank"] for row in plan["excluded_tasks"]} == excluded


def test_glm_third_remainder_fences_ingest_incomplete_task_and_preserves_history() -> None:
    source = hosted.load_object(REMAINDER_SOURCE_V3)
    plan = hosted.build_remainder_plan(
        hosted.load_object(
            Path("evals/fleet/configs/glm53-opencode-hosted-odd46-pass4-v9.json")
        ),
        source,
        "glm53_remainder3",
    )
    hosted.validate_plan(plan)
    assert plan["task_count"] == 45
    assert plan["new_session_count"] == 180
    assert [row["source_rank"] for row in plan["tasks"]] == list(range(11, 100, 2))
    assert [row["source_rank"] for row in plan["excluded_tasks"]] == [9]
    assert [row["source_rank"] for row in plan["upstream_excluded_tasks"]] == [3, 5, 7]
    outcomes = source["runs"]["glm53"]["outcomes"]
    assert [row["attempt"] for row in outcomes] == [1, 2]
    assert all(row["retry_allowed"] is False for row in outcomes)


def _bound_accepted_root(plan: dict, tmp_path: Path) -> tuple[Path, dict]:
    root = tmp_path / "job"
    (root / "attempts").mkdir(parents=True)
    (root / "claims").mkdir()
    (root / "PLAN.json").write_text(json.dumps(plan))
    item = plan["attempts"][0]
    claim = {
        "plan_sha256": plan["plan_sha256"],
        "run_id": item["run_id"],
        "rank": item["rank"],
        "attempt": item["attempt"],
        "config_sha256": "sha256:config",
    }
    claim["claim_sha256"] = self_hosted.digest_without(claim, "claim_sha256")
    (root / "claims" / f"{item['run_id']}.json").write_text(json.dumps(claim))
    receipt = {
        "run_id": item["run_id"],
        "session_id": str(uuid.uuid4()),
        "verifier_execution_id": str(uuid.uuid4()),
        "config_sha256": claim["config_sha256"],
        "claim_sha256": claim["claim_sha256"],
    }
    receipt["receipt_sha256"] = self_hosted.digest_without(receipt, "receipt_sha256")
    attempt = root / "attempts" / item["run_id"]
    attempt.mkdir()
    (attempt / "ACCEPTED.json").write_text(json.dumps(receipt))
    return root, receipt


def test_local_bound_accepted_receipt_survives_missing_public_metadata(
    monkeypatch, tmp_path: Path
) -> None:
    plan = hosted.load_object(
        Path("evals/fleet/configs/qwen38-opencode-hosted-complete47-pass4-v7.json")
    )
    root, receipt = _bound_accepted_root(plan, tmp_path)
    task = plan["tasks"][0]
    rows = [{
        "session_id": receipt["session_id"],
        "status": "completed",
        "model": "qwen3.8-27b",
        "verifier_execution": {"id": receipt["verifier_execution_id"]},
    }]

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    monkeypatch.setattr(hosted, "_client", lambda _key: Client())
    monkeypatch.setattr(self_hosted, "_task_sessions", lambda _client, _key: rows)
    assert hosted._validate_inventory_for_task(plan, root, task, "key") == 1


def test_invalid_or_unbound_accepted_receipt_is_rejected(tmp_path: Path) -> None:
    plan = hosted.load_object(
        Path("evals/fleet/configs/qwen38-opencode-hosted-complete47-pass4-v7.json")
    )
    root, receipt = _bound_accepted_root(plan, tmp_path)
    receipt["receipt_sha256"] = "sha256:invalid"
    item = plan["attempts"][0]
    path = root / "attempts" / item["run_id"] / "ACCEPTED.json"
    path.write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError, match="digest mismatch"):
        hosted._accepted(plan, root)


def test_unbound_public_session_row_cannot_become_accepted(
    monkeypatch, tmp_path: Path
) -> None:
    plan = hosted.load_object(
        Path("evals/fleet/configs/qwen38-opencode-hosted-complete47-pass4-v7.json")
    )
    root = tmp_path / "job"
    (root / "attempts").mkdir(parents=True)
    (root / "claims").mkdir()
    row = {
        "session_id": str(uuid.uuid4()),
        "status": "completed",
        "model": "qwen3.8-27b",
        "verifier_execution": {"id": str(uuid.uuid4())},
    }

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    monkeypatch.setattr(hosted, "_client", lambda _key: Client())
    monkeypatch.setattr(self_hosted, "_task_sessions", lambda _client, _key: [row])
    assert hosted._validate_inventory_for_task(plan, root, plan["tasks"][0], "key") == 0


def test_generated_plans_are_reproducible() -> None:
    source = hosted.load_object(SOURCE)
    for model, source_name, generated_name in (
        (
            "qwen38",
            "qwen38-opencode-train50-pass4-v3.json",
            "qwen38-opencode-hosted-complete49-pass4-v5.json",
        ),
        (
            "glm53",
            "glm53-opencode-train100-pass4-v4.json",
            "glm53-opencode-hosted-complete99-pass4-v5.json",
        ),
    ):
        expected = hosted.load_object(Path("evals/fleet/configs") / generated_name)
        actual = hosted.build_plan(
            hosted.load_object(Path("evals/fleet/configs") / source_name), source, model
        )
        assert json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)


def test_replacement_selection_lock_is_exact_disjoint_and_fail_closed() -> None:
    receipt = hosted.load_object(REPLACEMENT_LOCK)
    assert receipt["receipt_sha256"] == self_hosted.digest_without(
        receipt, "receipt_sha256"
    )

    split_path = Path(receipt["source"]["split"]["path"])
    selection_path = Path(receipt["source"]["locked_selection"]["path"])
    split = hosted.load_object(split_path)
    selection = hosted.load_object(selection_path)
    assert "sha256:" + hashlib.sha256(split_path.read_bytes()).hexdigest() == (
        receipt["source"]["split"]["file_sha256"]
    )
    assert "sha256:" + hashlib.sha256(selection_path.read_bytes()).hexdigest() == (
        receipt["source"]["locked_selection"]["file_sha256"]
    )
    assert split["manifest_digest"] == receipt["source"]["split"]["manifest_digest"]
    assert selection["selection_sha256"] == receipt["source"]["locked_selection"][
        "selection_sha256"
    ]
    assert selection["ranking"] == receipt["source"]["ranking"]

    qwen = receipt["qwen38"]
    assert qwen["fenced_original_source_ranks"] == [1, 2, 3]
    assert qwen["intact_original_task_count"] + qwen["replacement_task_count"] == 50
    assert qwen["replacement_task_count"] * qwen["pass_k"] == 12
    selected_qwen = [row for row in selection["tasks"] if 51 <= row["rank"] <= 53]
    assert [row["replacement_rank"] for row in qwen["tasks"]] == [51, 52, 53]
    binding_fields = (
        "historical_rank",
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
    assert [
        {field: row[field] for field in binding_fields} for row in qwen["tasks"]
    ] == [{field: row[field] for field in binding_fields} for row in selected_qwen]

    hydrated = hosted.load_object(
        Path("evals/fleet/configs/glm53-opencode-train100-pass4-v4.json")
    )
    hydrated_by_rank = {row["rank"]: row for row in hydrated["tasks"]}
    for row in qwen["tasks"]:
        source = hydrated_by_rank[row["replacement_rank"]]
        assert source["task"]["version_id"] == row["task_version_id"]
        assert source["task"]["cyber_contract"] == row["cyber_contract"]
        assert source["environment"]["runtime_seed_content_sha256"] == row[
            "runtime_seed_content_sha256"
        ]

    glm = receipt["glm53"]
    assert glm["fenced_original_source_ranks"] == [1, 2, 3, 5, 7]
    assert glm["intact_original_task_count"] + glm["replacement_task_count"] == 100
    assert glm["replacement_task_count"] * glm["pass_k"] == 20
    assert [row["replacement_rank"] for row in glm["tasks"]] == list(range(101, 106))
    assert [row["historical_rank"] for row in glm["tasks"]] == list(range(109, 114))
    assert [row["serving_block"] for row in glm["tasks"]] == [
        "dedicated_b",
        "dedicated_a",
        "dedicated_b",
        "dedicated_a",
        "dedicated_b",
    ]
    selected_ids = {row["task_version_id"] for row in selection["tasks"]}
    split_by_version = {row["task_version_id"]: row for row in split["tasks"]}
    runnable = hosted.load_object(
        Path("configs/runs/qwen36-27b-rl-base-full-runnable.json")
    )
    runnable_ids = {
        row["task_version_id"] for row in runnable["tasks"]["task_versions"]
    }
    for row in glm["tasks"]:
        assert row["task_version_id"] not in selected_ids
        assert row["task_version_id"] in runnable_ids
        source = split_by_version[row["task_version_id"]]
        assert source["split"] == "train"
        for field in binding_fields[1:]:
            assert row[field] == source[field]
    assert glm["hydration_gate"]["status"] == "required_not_satisfied"
    assert glm["hydration_gate"]["required_before_paid_launch"] is True
    assert glm["hydration_gate"]["blocks_released_for_scoring"] == []
    assert glm["hydration_gate"]["checks"] == [
        "exact_frozen_task_and_environment_binding",
        "cyber_contract_v3",
        "nonempty_runtime_seed_manifest",
        "complete_verifier_receipt",
        "fleet_team_authority",
    ]

    blocks = glm["dedicated_block_assignment"]
    assert blocks["rule"] == (
        "assign_replacements_in_rank_order_round_robin_beginning_with_the_smaller_24_task_original_block"
    )
    assert blocks["hosted"]["task_count"] == 46
    assert blocks["hosted"]["cell_count"] == 184
    assert blocks["dedicated_a"]["replacement_ranks"] == [102, 104]
    assert blocks["dedicated_a"]["task_count"] == 27
    assert blocks["dedicated_a"]["cell_count"] == 108
    assert blocks["dedicated_b"]["replacement_ranks"] == [101, 103, 105]
    assert blocks["dedicated_b"]["task_count"] == 27
    assert blocks["dedicated_b"]["cell_count"] == 108
    assert blocks["total_task_count"] == 100
    assert blocks["total_cell_count"] == 400
    assert sum(block["task_count"] for block in (
        blocks["hosted"], blocks["dedicated_a"], blocks["dedicated_b"]
    )) == 100
    assert sum(block["cell_count"] for block in (
        blocks["hosted"], blocks["dedicated_a"], blocks["dedicated_b"]
    )) == 400

    qwen_ids = {row["task_version_id"] for row in qwen["tasks"]}
    glm_ids = {row["task_version_id"] for row in glm["tasks"]}
    assert qwen_ids.isdisjoint(glm_ids)
    assert receipt["overlap_audit"]["candidate_claims"] == 0
    assert receipt["overlap_audit"]["candidate_accepted_receipts"] == 0
    assert receipt["overlap_audit"]["candidate_noncreditable_receipts"] == 0
    assert receipt["denominator_policy"]["qwen_primary_sessions"] == 50 * 4
    assert receipt["denominator_policy"]["glm_primary_sessions"] == 100 * 4


def test_dedicated_a_plan_is_exactly_partitioned_and_endpoint_bound(
    monkeypatch,
) -> None:
    assignment = hosted.load_object(REPLACEMENT_LOCK)
    by_key = {}
    for row in assignment["glm53"]["tasks"]:
        by_key[row["task_key"]] = {
            "key": row["task_key"],
            "environment_id": row["env_key"],
            "version": row["env_version"],
            "data_id": row["data_key"],
            "data_version": row["data_version"],
            "prompt": "sealed",
            "env_variables": {"sealed": True},
            "output_json_schema": {"type": "object"},
            "verifier_id": f"verifier-{row['replacement_rank']}",
            "verifier": {
                "verifier_version_id": f"verifier-version-{row['replacement_rank']}",
                "version": 1,
                "sha256": f"sha256:verifier-{row['replacement_rank']}",
                "function_name": "verify",
            },
            "metadata": {
                "cyber_contract": {
                    "evidence_schema": "1.0.0",
                    "submission_protocol": "2.0.0",
                    "verifier_contract": "3.0.0",
                },
                "runtime_seed_manifest": {
                    "content_sha256": f"sha256:seed-{row['replacement_rank']}",
                    "files": [{"target_path": "sealed"}],
                },
            },
        }

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    def request(_client, _method, path, **_kwargs):
        if path == "/v1/account":
            return {"team_name": "fleet", "team_id": self_hosted.FLEET_TEAM_ID}
        return by_key[path.removeprefix("/v1/tasks/")]

    monkeypatch.setattr(hosted, "_client", lambda _key: Client())
    monkeypatch.setattr(self_hosted, "_request", request)
    hydration = hosted.hydrate_glm53_replacements(assignment, "secret")
    canary = hosted.load_object(
        Path(
            "docs/evidence/qwen38-study/2026-09-03-glm53-dedicated-canary-v4-pass.json"
        )
    )
    source = hosted.load_object(
        Path("evals/fleet/configs/glm53-opencode-train100-pass4-v4.json")
    )
    plan = hosted.build_dedicated_a_plan(source, assignment, hydration, canary)
    hosted.validate_plan(plan)
    assert plan["task_count"] == 27
    assert plan["new_session_count"] == 108
    assert [row["source_rank"] for row in plan["tasks"]] == [
        *range(4, 53, 2),
        102,
        104,
    ]
    assert plan["reserved_source_ranks"] == [*range(54, 101, 2), 101, 103, 105]
    assert plan["treatment_block"]["service_uid"] == canary["network"]["service_uid"]
    assert plan["execution"]["inventory_policy"] == (
        "immutable_plan_claim_and_endpoint_uid_v1"
    )


def test_hydration_job_bootstraps_every_controller_import() -> None:
    manifest = HYDRATION_JOB_V2.read_text()
    assert "/bootstrap/self_hosted.py" in manifest
    assert "/bootstrap/runner.py" in manifest
    assert "opencode_train_sweep_runner.py" in manifest
    assert "/bootstrap/controller.py" in manifest


def test_committed_dedicated_a_plan_is_digest_valid() -> None:
    plan = hosted.load_object(DEDICATED_A_PLAN)
    hosted.validate_plan(plan)
    assert plan["plan_sha256"] == (
        "sha256:4f8d4fcf50af8abccf3b9d272a18755f9be0bc862a66fe25bba691bee8a22208"
    )


def test_dedicated_scoring_release_binds_both_disjoint_plans() -> None:
    release = hosted.load_object(DEDICATED_SCORING_RELEASE)
    assert release["receipt_sha256"] == self_hosted.digest_without(
        release, "receipt_sha256"
    )
    hosted.validate_dedicated_scoring_release(
        hosted.load_object(DEDICATED_A_PLAN), release
    )
    hosted.validate_dedicated_scoring_release(
        hosted.load_object(DEDICATED_B_PLAN), release
    )
    assert release["primary_estimator"]["session_count"] == 400


@pytest.mark.parametrize("plan_path", [DEDICATED_A_PLAN, DEDICATED_B_PLAN])
def test_dedicated_preflight_requires_scoring_release(
    plan_path: Path, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="release receipt is required"):
        hosted.preflight_plan(
            hosted.load_object(plan_path), tmp_path / "unused-root", "not-used"
        )


def test_dedicated_scoring_release_rejects_overlap() -> None:
    plan = hosted.load_object(DEDICATED_B_PLAN)
    release = hosted.load_object(DEDICATED_SCORING_RELEASE)
    release["replicas"]["B"]["plan"]["source_ranks"][0] = 9
    release["receipt_sha256"] = self_hosted.digest_without(release, "receipt_sha256")
    with pytest.raises(ValueError, match="bind this plan|arithmetic"):
        hosted.validate_dedicated_scoring_release(plan, release)


def test_committed_dedicated_b_plan_is_disjoint_and_digest_valid() -> None:
    plan = hosted.load_object(DEDICATED_B_PLAN)
    hosted.validate_plan(plan)
    assert plan["plan_sha256"] == (
        "sha256:008386c1bbc6d82229f2afdb85e074a0d5e717853dc7cab5b720ef13271e9cb5"
    )
    assert [row["source_rank"] for row in plan["tasks"]] == [
        *range(54, 101, 2),
        101,
        103,
        105,
    ]
    a = hosted.load_object(DEDICATED_A_PLAN)
    a_cells = {
        (row["task"]["version_id"], attempt)
        for row in a["tasks"]
        for attempt in range(1, 5)
    }
    b_cells = {
        (row["task"]["version_id"], attempt)
        for row in plan["tasks"]
        for attempt in range(1, 5)
    }
    assert a_cells.isdisjoint(b_cells)


def test_qwen_replacement_plan_is_exact_and_disjoint_from_v7() -> None:
    plan = hosted.load_object(QWEN_REPLACEMENT_PLAN)
    hosted.validate_plan(plan)
    assert plan["plan_sha256"] == (
        "sha256:9f879c054e770c120f2d4b77750475f402ba067873fbb93d0e356b062168841d"
    )
    assert [row["source_rank"] for row in plan["tasks"]] == [51, 52, 53]
    v7 = hosted.load_object(
        Path("evals/fleet/configs/qwen38-opencode-hosted-complete47-pass4-v7.json")
    )
    replacement_cells = {
        (row["task"]["version_id"], attempt)
        for row in plan["tasks"]
        for attempt in range(1, 5)
    }
    v7_cells = {
        (row["task"]["version_id"], attempt)
        for row in v7["tasks"]
        for attempt in range(1, 5)
    }
    assert replacement_cells.isdisjoint(v7_cells)
    assert plan["treatment_block"] == v7["treatment_block"]


def test_glm_r106_supplement_replays_frozen_order_and_restores_denominator() -> None:
    supplement = hosted.load_object(REPLACEMENT_SUPPLEMENT)
    assert supplement["receipt_sha256"] == self_hosted.digest_without(
        supplement, "receipt_sha256"
    )

    parent = hosted.load_object(REPLACEMENT_LOCK)
    assert parent["receipt_sha256"] == supplement["parent_selection_lock"][
        "receipt_sha256"
    ]
    assert hashlib.sha256(REPLACEMENT_LOCK.read_bytes()).hexdigest() == supplement[
        "parent_selection_lock"
    ]["file_sha256"].removeprefix("sha256:")

    selection = hosted.load_object(FROZEN_SELECTION)
    split = hosted.load_object(FROZEN_SPLIT)
    runnable = hosted.load_object(FROZEN_RUNNABLE_INVENTORY)
    selected = {row["task_version_id"] for row in selection["tasks"]}
    train = {
        row["task_version_id"]
        for row in split["tasks"]
        if row["split"] == "train"
    }
    ineligible = {
        row["task_version_id"]
        for row in selection["self_hosted_eligibility"]["excluded"]
    }
    remaining = [
        row
        for row in runnable["tasks"]["task_versions"]
        if row["task_version_id"]
        in train - selected - sweep.PRIOR_QWEN_TASK_VERSION_IDS - ineligible
    ]
    remaining.sort(key=lambda row: (row["task_key"], row["task_version_id"]))
    locked_prefix = [
        row["task_version_id"]
        for row in parent["glm53"]["tasks"]
    ]
    assert [row["task_version_id"] for row in remaining[:5]] == locked_prefix
    replacement = supplement["replacement"]
    assert remaining[5]["task_version_id"] == replacement["task_version_id"]
    assert replacement["replacement_rank"] == 106
    assert replacement["historical_rank"] == 114
    assert replacement["serving_block"] == "hosted"

    split_row = next(
        row
        for row in split["tasks"]
        if row["task_version_id"] == replacement["task_version_id"]
    )
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
    ):
        assert replacement[field] == split_row[field]

    current_plans = [
        Path("evals/fleet/configs/glm53-opencode-hosted-odd45-pass4-v10.json"),
        Path("evals/fleet/configs/glm53-opencode-hosted-odd46-pass4-v9.json"),
        DEDICATED_A_PLAN,
        DEDICATED_B_PLAN,
        Path("evals/fleet/configs/qwen38-opencode-hosted-complete47-pass4-v7.json"),
        QWEN_REPLACEMENT_PLAN,
    ]
    for path in current_plans:
        plan = hosted.load_object(path)
        assert replacement["task_version_id"] not in {
            row["task"]["version_id"] for row in plan["tasks"]
        }

    estimator = supplement["revised_primary_estimator"]
    assert estimator["unique_task_count"] == 45 + 27 + 27 + 1 == 100
    assert estimator["cell_count"] == 180 + 108 + 108 + 4 == 400
    assert estimator["pairwise_disjoint"] is True
    assert replacement["hydration_gate"]["required_before_paid_launch"] is True
    assert replacement["scored_launch_authorized"] is False


def test_glm_r106_hydration_and_hosted_plan_are_exact(monkeypatch) -> None:
    supplement = hosted.load_object(REPLACEMENT_SUPPLEMENT)
    row = supplement["replacement"]
    task_response = {
        "key": row["task_key"],
        "environment_id": row["env_key"],
        "version": row["env_version"],
        "data_id": row["data_key"],
        "data_version": row["data_version"],
        "prompt": "sealed",
        "env_variables": {"sealed": True},
        "output_json_schema": {"type": "object"},
        "verifier_id": "verifier-106",
        "verifier": {
            "verifier_version_id": "verifier-version-106",
            "version": 1,
            "sha256": "sha256:verifier-106",
            "function_name": "verify",
        },
        "metadata": {
            "cyber_contract": {
                "evidence_schema": "1.0.0",
                "submission_protocol": "2.0.0",
                "verifier_contract": "3.0.0",
            },
            "runtime_seed_manifest": {
                "content_sha256": "sha256:seed-106",
                "files": [{"target_path": "sealed"}],
            },
        },
    }

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    def request(_client, _method, path, **_kwargs):
        if path == "/v1/account":
            return {"team_name": "fleet", "team_id": self_hosted.FLEET_TEAM_ID}
        return task_response

    monkeypatch.setattr(hosted, "_client", lambda _key: Client())
    monkeypatch.setattr(self_hosted, "_request", request)
    hydration = hosted.hydrate_glm53_replacements(supplement, "secret")
    assert hydration["schema_version"] == "fleet-glm53-hosted-replacement-hydration-v1"
    assert hydration["selection_supplement_receipt_sha256"] == supplement["receipt_sha256"]
    assert hydration["tasks_hydrated"] == 1
    assert hydration["tasks"][0]["replacement_rank"] == 106

    plan = hosted.build_glm53_hosted_replacement_plan(
        hosted.load_object(
            Path("evals/fleet/configs/glm53-opencode-hosted-odd45-pass4-v10.json")
        ),
        supplement,
        hydration,
    )
    hosted.validate_plan(plan)
    assert plan["task_count"] == 1
    assert plan["new_session_count"] == 4
    assert plan["tasks"][0]["source_rank"] == 106
    assert plan["existing_hosted_source_ranks"] == list(range(11, 100, 2))
    assert plan["execution"]["inventory_policy"] == (
        "conservative_no_same_model_session_for_task_key_v1"
    )


def test_committed_glm_r106_plan_requires_exact_scoring_release() -> None:
    hydration = hosted.load_object(GLM_HOSTED_REPLACEMENT_HYDRATION)
    plan = hosted.load_object(GLM_HOSTED_REPLACEMENT_PLAN)
    release = hosted.load_object(GLM_HOSTED_REPLACEMENT_RELEASE)
    assert hydration["receipt_sha256"] == self_hosted.digest_without(
        hydration, "receipt_sha256"
    )
    hosted.validate_plan(plan)
    hosted.validate_hosted_replacement_scoring_release(plan, release)
    with pytest.raises(ValueError, match="release receipt is required"):
        hosted.validate_hosted_replacement_scoring_release(plan, None)
    tampered = json.loads(json.dumps(release))
    tampered["authorization"]["must_not_repeat"] = False
    with pytest.raises(ValueError, match="does not bind"):
        hosted.validate_hosted_replacement_scoring_release(plan, tampered)


def test_glm_r107_supplement_replays_order_and_tombstones_aborted_source56() -> None:
    supplement = hosted.load_object(REPLACEMENT_SUPPLEMENT_V2)
    assert supplement["receipt_sha256"] == self_hosted.digest_without(
        supplement, "receipt_sha256"
    )
    assert supplement["prior_supplement"]["receipt_sha256"] == hosted.load_object(
        REPLACEMENT_SUPPLEMENT
    )["receipt_sha256"]
    assert hashlib.sha256(REPLACEMENT_SUPPLEMENT.read_bytes()).hexdigest() == supplement[
        "prior_supplement"
    ]["file_sha256"].removeprefix("sha256:")

    selection = hosted.load_object(FROZEN_SELECTION)
    split = hosted.load_object(FROZEN_SPLIT)
    runnable = hosted.load_object(FROZEN_RUNNABLE_INVENTORY)
    selected = {row["task_version_id"] for row in selection["tasks"]}
    train = {
        row["task_version_id"]
        for row in split["tasks"]
        if row["split"] == "train"
    }
    ineligible = {
        row["task_version_id"]
        for row in selection["self_hosted_eligibility"]["excluded"]
    }
    remaining = [
        row
        for row in runnable["tasks"]["task_versions"]
        if row["task_version_id"]
        in train - selected - sweep.PRIOR_QWEN_TASK_VERSION_IDS - ineligible
    ]
    remaining.sort(key=lambda row: (row["task_key"], row["task_version_id"]))
    replayed_prefix = [
        row["task_version_id"]
        for row in supplement["selection_replay"]["locked_prefix_reproduced"]
    ]
    assert [row["task_version_id"] for row in remaining[:6]] == replayed_prefix
    replacement = supplement["replacement"]
    assert remaining[6]["task_version_id"] == replacement["task_version_id"]
    assert replacement["replacement_rank"] == 107
    assert replacement["historical_rank"] == 115
    assert replacement["serving_block"] == "dedicated_b_successor"

    split_row = next(
        row
        for row in split["tasks"]
        if row["task_version_id"] == replacement["task_version_id"]
    )
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
    ):
        assert replacement[field] == split_row[field]

    aborted = supplement["aborted_successor_claim"]
    assert aborted["source_rank"] == 56
    assert aborted["stream_bytes"] == aborted["stderr_bytes"] == 0
    assert all(
        aborted[field] is False
        for field in (
            "proxy_result_present",
            "reward_result_present",
            "trace_manifest_present",
            "session_ingest_present",
            "verifier_execution_present",
            "accepted_or_noncreditable_receipt_present",
        )
    )
    assert aborted["planned_run_id_api_matches"] == 0
    assert aborted["eligible_fresh_under_new_plan_and_treatment"] is True
    assert aborted["old_plan_cell_must_not_resume"] is True
    tombstone_evidence = supplement["forced_stop_tombstone"]
    tombstone_path = Path(tombstone_evidence["path"])
    tombstone = hosted.load_object(tombstone_path)
    assert tombstone["receipt_sha256"] == tombstone_evidence["receipt_sha256"]
    assert hashlib.sha256(tombstone_path.read_bytes()).hexdigest() == tombstone_evidence[
        "file_sha256"
    ].removeprefix("sha256:")
    assert tombstone["fenced_source54"]["whole_task_fenced"] is True
    assert tombstone["source56_aborted_before_execution"]["scored_or_terminal_cell"] is False
    assert tombstone["source56_aborted_before_execution"]["eligible_under_fresh_plan"] is True

    lifecycle = supplement["preemption_and_readmission"]
    assert lifecycle["ray_job_identity_continuity_is_not_runtime_treatment_continuity"]
    assert lifecycle["old_runtime"]["ray_cluster_uid"] != lifecycle[
        "readmitted_runtime"
    ]["ray_cluster_uid"]
    assert lifecycle["old_runtime"]["head_pod_uid"] != lifecycle[
        "readmitted_runtime"
    ]["head_pod_uid"]
    assert lifecycle["old_runtime"]["service_uid"] != lifecycle[
        "readmitted_runtime"
    ]["service_uid"]
    assert lifecycle["fresh_health_and_non_scored_parity_required"] is True

    estimator = supplement["revised_primary_estimator"]
    assert estimator["dedicated_b_task_count"] == 23 + 3 + 1 == 27
    assert estimator["unique_task_count"] == 46 + 27 + 27 == 100
    assert estimator["cell_count"] == 184 + 108 + 108 == 400
    assert replacement["hydration_gate"]["required_before_paid_launch"] is True
    assert replacement["scored_launch_authorized"] is False


def test_dedicated_b_stop_tombstone_preserves_zero_execution_boundary() -> None:
    tombstone = hosted.load_object(DEDICATED_B_STOP_TOMBSTONE)
    assert tombstone["receipt_sha256"] == self_hosted.digest_without(
        tombstone, "receipt_sha256"
    )
    assert tombstone["fenced_source54"]["whole_task_fenced"] is True
    aborted = tombstone["source56_aborted_before_execution"]
    assert aborted["opencode_stream_bytes"] == 0
    assert aborted["old_run_id_session_matches"] == 0
    assert aborted["old_run_id_verifier_matches"] == 0
    assert aborted["scored_or_terminal_cell"] is False
    assert aborted["eligible_under_fresh_plan"] is True
    assert aborted["old_run_ids_reusable"] is False
    successor = tombstone["successor_gate"]
    assert successor["task_count"] == 27
    assert successor["cell_count"] == 108
    assert all(successor[field] is True for field in (
        "required_old_job_absent",
        "required_old_pod_absent",
        "required_tombstone_digest",
        "required_new_serving_uids",
        "required_new_parity_receipt",
        "required_new_r107_lock_and_hydration",
    ))


def test_dedicated_a_stop_tombstone_cannot_hide_nonempty_agent_stream() -> None:
    receipt = hosted.load_object(DEDICATED_A_STOP_TOMBSTONE_V2)
    hosted.validate_dedicated_a_stop_tombstone(receipt)
    tampered = json.loads(json.dumps(receipt))
    tampered["source4_attempt4"]["agent_execution_started"] = False
    tampered["receipt_sha256"] = self_hosted.digest_without(
        tampered, "receipt_sha256"
    )
    with pytest.raises(ValueError, match="classification drifted"):
        hosted.validate_dedicated_a_stop_tombstone(tampered)


def test_glm_r107_hydration_requires_tombstone_bound_supplement(monkeypatch) -> None:
    supplement = hosted.load_object(REPLACEMENT_SUPPLEMENT_V2)
    row = supplement["replacement"]
    task_response = {
        "key": row["task_key"],
        "environment_id": row["env_key"],
        "version": row["env_version"],
        "data_id": row["data_key"],
        "data_version": row["data_version"],
        "prompt": "sealed",
        "env_variables": {"sealed": True},
        "output_json_schema": {"type": "object"},
        "verifier_id": "verifier-107",
        "verifier": {
            "verifier_version_id": "verifier-version-107",
            "version": 1,
            "sha256": "sha256:verifier-107",
            "function_name": "verify",
        },
        "metadata": {
            "cyber_contract": {
                "evidence_schema": "1.0.0",
                "submission_protocol": "2.0.0",
                "verifier_contract": "3.0.0",
            },
            "runtime_seed_manifest": {
                "content_sha256": "sha256:seed-107",
                "files": [{"target_path": "sealed"}],
            },
        },
    }

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    def request(_client, _method, path, **_kwargs):
        if path == "/v1/account":
            return {"team_name": "fleet", "team_id": self_hosted.FLEET_TEAM_ID}
        return task_response

    monkeypatch.setattr(hosted, "_client", lambda _key: Client())
    monkeypatch.setattr(self_hosted, "_request", request)
    hydration = hosted.hydrate_glm53_replacements(supplement, "secret")
    assert hydration["schema_version"] == (
        "fleet-glm53-dedicated-b-replacement-hydration-v1"
    )
    assert hydration["tasks"][0]["replacement_rank"] == 107
    tampered = json.loads(json.dumps(supplement))
    tampered["forced_stop_tombstone"]["receipt_sha256"] = "sha256:wrong"
    tampered["receipt_sha256"] = self_hosted.digest_without(tampered, "receipt_sha256")
    with pytest.raises(ValueError, match="stop tombstone drifted"):
        hosted.hydrate_glm53_replacements(tampered, "secret")


@pytest.mark.parametrize(
    ("supplement_path", "expected_schema", "expected_ranks"),
    [
        (
            QWEN_HTTP500_REPLACEMENT_SUPPLEMENT,
            "fleet-qwen38-hosted-replacement-hydration-v2",
            [54, 55],
        ),
        (
            GLM_HTTP500_REPLACEMENT_SUPPLEMENT,
            "fleet-glm53-hosted-replacement-hydration-v2",
            [108, 109],
        ),
        (
            GLM_DEDICATED_A_PREEMPTION_SUPPLEMENT,
            "fleet-glm53-dedicated-a-replacement-hydration-v1",
            [110],
        ),
        (
            QWEN_SOURCE6_REPLACEMENT_SUPPLEMENT,
            "fleet-qwen38-hosted-replacement-hydration-v3",
            [56],
        ),
        (
            GLM_SOURCE13_REPLACEMENT_SUPPLEMENT,
            "fleet-glm53-hosted-replacement-hydration-v3",
            [111],
        ),
    ],
)
def test_http500_replacement_hydration_is_metadata_only(
    monkeypatch, supplement_path: Path, expected_schema: str, expected_ranks: list[int]
) -> None:
    supplement = hosted.load_object(supplement_path)
    rows = supplement.get("replacements") or [supplement["replacement"]]
    by_key = {row["task_key"]: row for row in rows}

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    def request(_client, _method, path, **_kwargs):
        if path == "/v1/account":
            return {"team_name": "fleet", "team_id": self_hosted.FLEET_TEAM_ID}
        row = by_key[path.removeprefix("/v1/tasks/")]
        return {
            "key": row["task_key"],
            "environment_id": row["env_key"],
            "version": row["env_version"],
            "data_id": row["data_key"],
            "data_version": row["data_version"],
            "prompt": "sealed",
            "env_variables": {"sealed": True},
            "output_json_schema": {"type": "object"},
            "verifier_id": f"verifier-{row['replacement_rank']}",
            "verifier": {
                "verifier_version_id": f"verifier-version-{row['replacement_rank']}",
                "version": 1,
                "sha256": f"sha256:verifier-{row['replacement_rank']}",
                "function_name": "verify",
            },
            "metadata": {
                "cyber_contract": {
                    "evidence_schema": "1.0.0",
                    "submission_protocol": "2.0.0",
                    "verifier_contract": "3.0.0",
                },
                "runtime_seed_manifest": {
                    "content_sha256": f"sha256:seed-{row['replacement_rank']}",
                    "files": [{"target_path": "sealed"}],
                },
            },
        }

    monkeypatch.setattr(hosted, "_client", lambda _key: Client())
    monkeypatch.setattr(self_hosted, "_request", request)
    hydration = hosted.hydrate_glm53_replacements(supplement, "secret")
    assert hydration["schema_version"] == expected_schema
    assert hydration["selection_supplement_receipt_sha256"] == supplement[
        "receipt_sha256"
    ]
    assert [row["replacement_rank"] for row in hydration["tasks"]] == expected_ranks
    assert hydration["scores_read"] is False
    assert hydration["task_content_retained"] is False


def test_glm_hosted_reassigned_b_plan_is_preview_only_and_disjoint() -> None:
    dedicated = hosted.load_object(DEDICATED_B_PLAN)
    hosted_plan = hosted.load_object(
        Path("evals/fleet/configs/glm53-opencode-hosted-odd45-pass4-v10.json")
    )
    tombstone = hosted.load_object(DEDICATED_B_STOP_TOMBSTONE)
    plan = hosted.build_glm53_hosted_reassigned_b_plan(
        dedicated,
        hosted_plan,
        tombstone,
        hosted.load_object(REPLACEMENT_SUPPLEMENT_V2),
        hosted.load_object(DEDICATED_B_R107_HYDRATION),
    )
    assert plan == hosted.load_object(GLM_HOSTED_REASSIGNED_B_PLAN)
    hosted.validate_plan(plan)
    assert plan["task_count"] == 27
    assert plan["total_session_count"] == plan["new_session_count"] == 108
    assert [row["source_rank"] for row in plan["tasks"]] == [
        *range(56, 101, 2),
        101,
        103,
        105,
        107,
    ]
    assert plan["treatment_block"]["serving_block"] == (
        "hosted-reassigned-after-dedicated-preemption"
    )
    assert plan["execution"]["launch_authorized"] is False
    old_source56 = next(
        row for row in dedicated["attempts"] if row["source_rank"] == 56
    )
    new_source56 = next(row for row in plan["attempts"] if row["source_rank"] == 56)
    assert new_source56["run_id"] != old_source56["run_id"]
    assert new_source56["network"] != old_source56["network"]


def test_qwen_http500_successor_is_exactly_50_tasks_with_prior_complete() -> None:
    plan = hosted.build_qwen_http500_successor_plan(
        hosted.load_object(
            Path("evals/fleet/configs/qwen38-opencode-hosted-complete47-pass4-v7.json")
        ),
        hosted.load_object(QWEN_REPLACEMENT_PLAN),
        hosted.load_object(HOSTED_HTTP500_INCIDENT),
        hosted.load_object(QWEN_HTTP500_REPLACEMENT_SUPPLEMENT),
        hosted.load_object(QWEN_HTTP500_HYDRATION),
    )
    assert plan == hosted.load_object(QWEN_HTTP500_SUCCESSOR_PLAN)
    hosted.validate_plan(plan)
    assert plan["task_count"] == 49
    assert plan["new_session_count"] == 196
    assert plan["prior_complete_source_ranks"] == [4]
    assert plan["primary_estimator_task_count"] == 50
    assert {row["source_rank"] for row in plan["tasks"]} == {
        *range(6, 51),
        52,
        53,
        54,
        55,
    }
    assert plan["execution"]["launch_authorized"] is False
    assert plan["execution"]["required_priority_class"] == "fleet-train-high"


def test_qwen_http500_successor_requires_exact_scoring_release() -> None:
    plan = hosted.load_object(QWEN_HTTP500_SUCCESSOR_PLAN)
    release = hosted.load_object(QWEN_HTTP500_SCORING_RELEASE)
    hosted.validate_qwen_http500_scoring_release(plan, release)
    with pytest.raises(ValueError, match="release is required"):
        hosted.validate_qwen_http500_scoring_release(plan, None)
    tampered = json.loads(json.dumps(release))
    tampered["authorization"]["must_not_repeat"] = False
    tampered["receipt_sha256"] = self_hosted.digest_without(
        tampered, "receipt_sha256"
    )
    with pytest.raises(ValueError, match="does not bind"):
        hosted.validate_qwen_http500_scoring_release(plan, tampered)


def test_glm_http500_hosted_successor_is_exactly_73_plus_dedicated_a() -> None:
    plan = hosted.build_glm_http500_successor_plan(
        hosted.load_object(
            Path("evals/fleet/configs/glm53-opencode-hosted-odd45-pass4-v10.json")
        ),
        hosted.load_object(GLM_HOSTED_REASSIGNED_B_PLAN),
        hosted.load_object(HOSTED_HTTP500_INCIDENT),
        hosted.load_object(GLM_HTTP500_REPLACEMENT_SUPPLEMENT),
        hosted.load_object(GLM_HTTP500_HYDRATION),
    )
    assert plan == hosted.load_object(GLM_HTTP500_SUCCESSOR_PLAN)
    hosted.validate_plan(plan)
    assert plan["task_count"] == 73
    assert plan["new_session_count"] == 292
    assert plan["dedicated_a_task_count"] == 27
    assert plan["primary_estimator_task_count"] == 100
    assert {row["source_rank"] for row in plan["tasks"]} == {
        *range(13, 100, 2),
        *range(56, 101, 2),
        101,
        103,
        105,
        107,
        108,
        109,
    }
    assert plan["execution"]["launch_authorized"] is False
    assert plan["execution"]["required_priority_class"] == "fleet-train-high"


def test_glm_http500_primary_splits_out_the_dedicated_b_partition() -> None:
    combined = hosted.load_object(GLM_HTTP500_SUCCESSOR_PLAN)
    plan = hosted.build_glm_http500_hosted_primary_plan(combined)
    assert plan == hosted.load_object(GLM_HTTP500_HOSTED_PRIMARY_PLAN)
    hosted.validate_plan(plan)
    selected = {row["source_rank"] for row in plan["tasks"]}
    assert selected == {*range(13, 100, 2), 108, 109}
    assert plan["task_count"] == 46
    assert plan["new_session_count"] == 184
    assert plan["dedicated_a_task_count"] == 27
    assert plan["dedicated_b_task_count"] == 27
    assert plan["primary_estimator_task_count"] == 100
    assert plan["execution"]["launch_authorized"] is False


def test_glm_http500_primary_requires_exact_scoring_release() -> None:
    plan = hosted.load_object(GLM_HTTP500_HOSTED_PRIMARY_PLAN)
    release = hosted.load_object(GLM_HTTP500_SCORING_RELEASE)
    hosted.validate_glm_http500_scoring_release(plan, release)
    with pytest.raises(ValueError, match="release is required"):
        hosted.validate_glm_http500_scoring_release(plan, None)
    tampered = json.loads(json.dumps(release))
    tampered["authorization"]["must_not_repeat"] = False
    tampered["receipt_sha256"] = self_hosted.digest_without(
        tampered, "receipt_sha256"
    )
    with pytest.raises(ValueError, match="does not bind"):
        hosted.validate_glm_http500_scoring_release(plan, tampered)


def test_glm_dedicated_b_v5_plan_binds_fresh_serving_and_fences_source54() -> None:
    plan = hosted.build_glm_dedicated_b_v5_plan(
        hosted.load_object(DEDICATED_B_PLAN),
        hosted.load_object(DEDICATED_B_STOP_TOMBSTONE),
        hosted.load_object(REPLACEMENT_SUPPLEMENT_V2),
        hosted.load_object(DEDICATED_B_R107_HYDRATION),
        hosted.load_object(
            Path(
                "docs/evidence/qwen38-study/"
                "2026-09-04-glm53-dedicated-replica-b-v5-pass.json"
            )
        ),
    )
    assert plan == hosted.load_object(GLM_DEDICATED_B_V5_PLAN)
    hosted.validate_plan(plan)
    assert [row["source_rank"] for row in plan["tasks"]] == [
        *range(56, 101, 2),
        101,
        103,
        105,
        107,
    ]
    assert 54 in plan["fenced_source_ranks"]
    assert plan["task_count"] == 27
    assert plan["new_session_count"] == 108
    assert plan["treatment_block"]["serving_generation"] == "v5"
    assert plan["execution"]["required_priority_class"] == "fleet-train-high"
    assert plan["execution"]["launch_authorized"] is False


def test_glm_dedicated_b_v5_requires_exact_root_scoring_release() -> None:
    plan = hosted.load_object(GLM_DEDICATED_B_V5_PLAN)
    release = hosted.load_object(GLM_DEDICATED_B_V5_RELEASE)
    hosted.validate_glm_dedicated_b_v5_scoring_release(plan, release)
    tampered = json.loads(json.dumps(release))
    tampered["gates"]["source56_zero_execution_reuse_proven"] = False
    tampered["receipt_sha256"] = self_hosted.digest_without(
        tampered, "receipt_sha256"
    )
    with pytest.raises(ValueError, match="does not bind"):
        hosted.validate_glm_dedicated_b_v5_scoring_release(plan, tampered)


def test_glm_dedicated_a_v5_plan_binds_fresh_serving_and_fences_source4() -> None:
    plan = hosted.build_glm_dedicated_a_v5_plan(
        hosted.load_object(DEDICATED_A_PLAN),
        hosted.load_object(DEDICATED_A_STOP_TOMBSTONE_V2),
        hosted.load_object(GLM_DEDICATED_A_PREEMPTION_SUPPLEMENT),
        hosted.load_object(
            Path(
                "docs/evidence/qwen38-study/"
                "2026-09-04-glm53-dedicated-a-r110-hydration-v1.json"
            )
        ),
        hosted.load_object(
            Path(
                "docs/evidence/qwen38-study/"
                "2026-09-04-glm53-dedicated-replica-a-v5-pass.json"
            )
        ),
    )
    assert plan == hosted.load_object(GLM_DEDICATED_A_V5_PLAN)
    hosted.validate_plan(plan)
    assert [row["source_rank"] for row in plan["tasks"]] == [
        *range(6, 53, 2),
        102,
        104,
        110,
    ]
    assert 4 in plan["fenced_source_ranks"]
    assert plan["task_count"] == 27
    assert plan["new_session_count"] == 108
    assert plan["treatment_block"]["serving_generation"] == "v5"
    assert plan["execution"]["required_priority_class"] == "fleet-train-high"
    assert plan["execution"]["launch_authorized"] is False


def test_glm_dedicated_a_v5_requires_exact_root_scoring_release() -> None:
    plan = hosted.load_object(GLM_DEDICATED_A_V5_PLAN)
    release = hosted.load_object(GLM_DEDICATED_A_V5_RELEASE)
    hosted.validate_glm_dedicated_a_v5_scoring_release(plan, release)
    tampered = json.loads(json.dumps(release))
    tampered["gates"]["source4_fenced"] = False
    tampered["receipt_sha256"] = self_hosted.digest_without(
        tampered, "receipt_sha256"
    )
    with pytest.raises(ValueError, match="does not bind"):
        hosted.validate_glm_dedicated_a_v5_scoring_release(plan, tampered)


@pytest.mark.parametrize(
    ("predecessor", "supplement", "hydration", "plan_path", "source_rank"),
    [
        (
            QWEN_HTTP500_SUCCESSOR_PLAN,
            QWEN_SOURCE6_REPLACEMENT_SUPPLEMENT,
            Path(
                "docs/evidence/qwen38-study/2026-09-04-qwen38-r56-hydration-v1.json"
            ),
            QWEN_ATTRITION_REPLACEMENT_PLAN,
            56,
        ),
        (
            GLM_HTTP500_HOSTED_PRIMARY_PLAN,
            GLM_SOURCE13_REPLACEMENT_SUPPLEMENT,
            Path(
                "docs/evidence/qwen38-study/2026-09-04-glm53-r111-hydration-v1.json"
            ),
            GLM_ATTRITION_REPLACEMENT_PLAN,
            111,
        ),
    ],
)
def test_hosted_attrition_replacement_plan_is_same_treatment_and_preview_only(
    predecessor: Path,
    supplement: Path,
    hydration: Path,
    plan_path: Path,
    source_rank: int,
) -> None:
    original = hosted.load_object(predecessor)
    plan = hosted.build_hosted_attrition_replacement_plan(
        original,
        hosted.load_object(supplement),
        hosted.load_object(hydration),
    )
    assert plan == hosted.load_object(plan_path)
    hosted.validate_plan(plan)
    assert plan["treatment_block"] == original["treatment_block"]
    assert [row["source_rank"] for row in plan["tasks"]] == [source_rank]
    assert plan["task_count"] == 1
    assert plan["new_session_count"] == 4
    assert plan["execution"]["required_priority_class"] == "fleet-train-high"
    assert plan["execution"]["launch_authorized"] is False


def test_qwen_http500_supplement_selects_unused_r54_r55_and_restores_200_cells() -> None:
    supplement = hosted.load_object(QWEN_HTTP500_REPLACEMENT_SUPPLEMENT)
    assert supplement["receipt_sha256"] == self_hosted.digest_without(
        supplement, "receipt_sha256"
    )

    selection = hosted.load_object(FROZEN_SELECTION)
    fields = (
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
    for replacement in supplement["replacements"]:
        selected = selection["tasks"][replacement["replacement_rank"] - 1]
        assert selected["rank"] == replacement["replacement_rank"]
        assert selected["historical_rank"] == replacement["historical_rank"]
        assert all(replacement[field] == selected[field] for field in fields)

    v7 = hosted.load_object(
        Path("evals/fleet/configs/qwen38-opencode-hosted-complete47-pass4-v7.json")
    )
    prior_replacements = hosted.load_object(QWEN_REPLACEMENT_PLAN)
    replacement_ids = {
        row["task_version_id"] for row in supplement["replacements"]
    }
    current_ids = {
        row["task"]["version_id"] for row in [*v7["tasks"], *prior_replacements["tasks"]]
    }
    assert replacement_ids.isdisjoint(current_ids)

    primary_ids = {
        row["task"]["version_id"]
        for row in v7["tasks"]
        if row["source_rank"] != 5
    }
    primary_ids.update(
        row["task"]["version_id"]
        for row in prior_replacements["tasks"]
        if row["source_rank"] != 51
    )
    primary_ids.update(replacement_ids)
    assert len(primary_ids) == 46 + 2 + 2 == 50
    estimator = supplement["revised_primary_estimator"]
    assert estimator["unique_task_count"] == 50
    assert estimator["pass_k"] == 4
    assert estimator["cell_count"] == 200

    failure = supplement["common_infrastructure_failure"]
    assert failure["http_status"] == 500
    assert failure["agent_output_present"] is True
    assert failure["scoring_intent_present"] is True
    assert failure["cleanup_instance_closed"] is True
    assert failure["verifier_execution_present"] is False
    assert failure["reward_result_present"] is False
    assert failure["session_ingest_present"] is False
    assert failure["retry_same_output"] is False
    assert {row["source_rank"] for row in supplement["fenced_tasks"]} == {5, 51}
    assert all(row["whole_task_fenced"] for row in supplement["fenced_tasks"])

    assert supplement["hydration_gate"]["required_before_paid_launch"] is True
    assert supplement["concurrency"]["third_hosted_stream_authorized"] is False
    scheduling = supplement["future_submission_gate"]
    assert scheduling["required_priority_class"] == "fleet-train-high"
    assert scheduling["rejected_priority_class"] == "fleet-infra-quiet"
    assert scheduling["true_non_preemptible_available"] is False
    assert scheduling["higher_priority_may_preempt_lower_priority"] is True


def test_glm_http500_supplement_selects_unused_r108_r109_and_restores_400_cells() -> None:
    supplement = hosted.load_object(GLM_HTTP500_REPLACEMENT_SUPPLEMENT)
    assert supplement["receipt_sha256"] == self_hosted.digest_without(
        supplement, "receipt_sha256"
    )
    prior_path = Path(supplement["prior_supplement"]["path"])
    prior = hosted.load_object(prior_path)
    assert supplement["prior_supplement"]["receipt_sha256"] == prior["receipt_sha256"]
    assert hashlib.sha256(prior_path.read_bytes()).hexdigest() == supplement[
        "prior_supplement"
    ]["file_sha256"].removeprefix("sha256:")

    selection = hosted.load_object(FROZEN_SELECTION)
    split = hosted.load_object(FROZEN_SPLIT)
    runnable = hosted.load_object(FROZEN_RUNNABLE_INVENTORY)
    selected = {row["task_version_id"] for row in selection["tasks"]}
    train = {
        row["task_version_id"]
        for row in split["tasks"]
        if row["split"] == "train"
    }
    ineligible = {
        row["task_version_id"]
        for row in selection["self_hosted_eligibility"]["excluded"]
    }
    remaining = [
        row
        for row in runnable["tasks"]["task_versions"]
        if row["task_version_id"]
        in train - selected - sweep.PRIOR_QWEN_TASK_VERSION_IDS - ineligible
    ]
    remaining.sort(key=lambda row: (row["task_key"], row["task_version_id"]))
    assert [row["task_version_id"] for row in remaining[7:9]] == [
        row["task_version_id"] for row in supplement["replacements"]
    ]
    assert [row["replacement_rank"] for row in supplement["replacements"]] == [108, 109]
    assert [row["historical_rank"] for row in supplement["replacements"]] == [116, 117]

    fields = (
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
    for replacement in supplement["replacements"]:
        split_row = next(
            row
            for row in split["tasks"]
            if row["task_version_id"] == replacement["task_version_id"]
        )
        assert all(replacement[field] == split_row[field] for field in fields)

    hosted_original = hosted.load_object(
        Path("evals/fleet/configs/glm53-opencode-hosted-odd45-pass4-v10.json")
    )
    dedicated_a = hosted.load_object(DEDICATED_A_PLAN)
    reassigned_b = hosted.load_object(GLM_HOSTED_REASSIGNED_B_PLAN)
    blocks = [
        {
            row["task"]["version_id"]
            for row in hosted_original["tasks"]
            if row["source_rank"] != 11
        },
        {row["task"]["version_id"] for row in dedicated_a["tasks"]},
        {row["task"]["version_id"] for row in reassigned_b["tasks"]},
        {row["task_version_id"] for row in supplement["replacements"]},
    ]
    assert [len(block) for block in blocks] == [44, 27, 27, 2]
    for index, block in enumerate(blocks):
        assert all(block.isdisjoint(other) for other in blocks[index + 1 :])
    assert len(set().union(*blocks)) == 100
    fenced_r106 = hosted.load_object(GLM_HOSTED_REPLACEMENT_PLAN)["tasks"][0]["task"][
        "version_id"
    ]
    assert fenced_r106 not in set().union(*blocks)

    estimator = supplement["revised_primary_estimator"]
    assert estimator["hosted_task_count"] == 44 + 27 + 2 == 73
    assert estimator["hosted_cell_count"] == 292
    assert estimator["dedicated_a_task_count"] == 27
    assert estimator["dedicated_a_cell_count"] == 108
    assert estimator["unique_task_count"] == 100
    assert estimator["cell_count"] == 400
    treatment = supplement["treatment_reassignment"]
    assert treatment["preview_plan_launch_authorized"] is False
    assert treatment["must_be_rebuilt_to_include_r108_r109_and_exclude_fenced_r106"] is True

    failure = supplement["common_infrastructure_failure"]
    assert failure["http_status"] == 500
    assert failure["agent_output_present"] is True
    assert failure["scoring_intent_present"] is True
    assert failure["cleanup_containers_removed"] is True
    assert failure["verifier_execution_present"] is False
    assert failure["reward_result_present"] is False
    assert failure["session_ingest_present"] is False
    assert failure["retry_same_output"] is False
    assert {row["source_rank"] for row in supplement["fenced_tasks"]} == {11, 106}
    assert all(row["whole_task_fenced"] for row in supplement["fenced_tasks"])

    assert supplement["hydration_gate"]["required_before_paid_launch"] is True
    assert supplement["concurrency"]["third_hosted_stream_authorized"] is False
    scheduling = supplement["future_submission_gate"]
    assert scheduling["required_priority_class"] == "fleet-train-high"
    assert scheduling["rejected_priority_class"] == "fleet-infra-quiet"
    assert scheduling["true_non_preemptible_available"] is False
    assert scheduling["higher_priority_may_preempt_lower_priority"] is True


def test_glm_dedicated_a_preemption_selects_unused_r110_and_restores_400_cells() -> None:
    supplement = hosted.load_object(GLM_DEDICATED_A_PREEMPTION_SUPPLEMENT)
    assert supplement["receipt_sha256"] == self_hosted.digest_without(
        supplement, "receipt_sha256"
    )

    prior_path = Path(supplement["prior_supplement"]["path"])
    prior = hosted.load_object(prior_path)
    assert supplement["prior_supplement"]["receipt_sha256"] == prior[
        "receipt_sha256"
    ]
    assert hashlib.sha256(prior_path.read_bytes()).hexdigest() == supplement[
        "prior_supplement"
    ]["file_sha256"].removeprefix("sha256:")

    stop = supplement["controller_stop_evidence"]
    tombstone_path = Path(stop["tombstone_path"])
    retirement_path = Path(stop["retirement_path"])
    tombstone = hosted.load_object(tombstone_path)
    retirement = hosted.load_object(retirement_path)
    assert stop["tombstone_receipt_sha256"] == tombstone["receipt_sha256"]
    assert stop["retirement_receipt_sha256"] == retirement["receipt_sha256"]
    assert hashlib.sha256(tombstone_path.read_bytes()).hexdigest() == stop[
        "tombstone_file_sha256"
    ].removeprefix("sha256:")
    assert hashlib.sha256(retirement_path.read_bytes()).hexdigest() == stop[
        "retirement_file_sha256"
    ].removeprefix("sha256:")
    assert stop["corrected_tombstone_binds_nonempty_agent_stream"] is True
    assert tombstone["source4_attempt4"]["opencode_stream_bytes"] == 301387
    assert tombstone["successor_gate"][
        "original_zero_execution_classification_forbidden"
    ] is True
    assert stop["stop_and_post_stop_absence_facts_retained"] is True

    selection = hosted.load_object(FROZEN_SELECTION)
    split = hosted.load_object(FROZEN_SPLIT)
    runnable = hosted.load_object(FROZEN_RUNNABLE_INVENTORY)
    selected = {row["task_version_id"] for row in selection["tasks"]}
    train = {
        row["task_version_id"]
        for row in split["tasks"]
        if row["split"] == "train"
    }
    ineligible = {
        row["task_version_id"]
        for row in selection["self_hosted_eligibility"]["excluded"]
    }
    remaining = [
        row
        for row in runnable["tasks"]["task_versions"]
        if row["task_version_id"]
        in train - selected - sweep.PRIOR_QWEN_TASK_VERSION_IDS - ineligible
    ]
    remaining.sort(key=lambda row: (row["task_key"], row["task_version_id"]))
    replacement = supplement["replacement"]
    assert remaining[9]["task_version_id"] == replacement["task_version_id"]
    assert replacement["replacement_rank"] == 110
    assert replacement["historical_rank"] == 118
    assert replacement["serving_block"] == "dedicated_a_successor"
    split_row = next(
        row
        for row in split["tasks"]
        if row["task_version_id"] == replacement["task_version_id"]
    )
    fields = (
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
    assert all(replacement[field] == split_row[field] for field in fields)

    fenced = supplement["fenced_source4"]
    assert [row["attempt"] for row in fenced["accepted_attempts"]] == [1, 2, 3]
    interrupted = fenced["interrupted_attempt"]
    assert interrupted["agent_activity_present"] is True
    assert interrupted["agent_stream_bytes"] > 0
    assert interrupted["result_present"] is False
    assert interrupted["scoring_intent_present"] is False
    assert interrupted["reward_result_present"] is False
    assert interrupted["session_ingest_present"] is False
    assert interrupted["verifier_execution_present"] is False
    assert interrupted["accepted_or_noncreditable_receipt_present"] is False
    assert interrupted["retry_same_output"] is False
    stopped = supplement["stopped_scored_controller"]
    assert stopped["source6_claim_present"] is False
    assert stopped["source6_execution_present"] is False
    assert stopped["old_plan_must_not_resume"] is True

    hosted_primary = hosted.load_object(GLM_HTTP500_HOSTED_PRIMARY_PLAN)
    dedicated_a = hosted.load_object(DEDICATED_A_PLAN)
    reassigned_b = hosted.load_object(GLM_HOSTED_REASSIGNED_B_PLAN)
    blocks = [
        {row["task"]["version_id"] for row in hosted_primary["tasks"]},
        {
            row["task"]["version_id"]
            for row in dedicated_a["tasks"]
            if row["source_rank"] != 4
        }
        | {replacement["task_version_id"]},
        {row["task"]["version_id"] for row in reassigned_b["tasks"]},
    ]
    assert [len(block) for block in blocks] == [46, 27, 27]
    for index, block in enumerate(blocks):
        assert all(block.isdisjoint(other) for other in blocks[index + 1 :])
    assert len(set().union(*blocks)) == 100

    estimator = supplement["revised_primary_estimator"]
    assert estimator["dedicated_a_original_task_count"] == 24
    assert estimator["dedicated_a_replacement_ranks"] == [102, 104, 110]
    assert estimator["unique_task_count"] == 100
    assert estimator["cell_count"] == 400
    assert estimator["partially_accepted_sessions_outside_estimator"] == 8
    assert estimator["formal_accept_count_if_primary_completes"] == 408

    assert replacement["hydration_gate"]["required_before_paid_launch"] is True
    assert replacement["scored_launch_authorized"] is False
    scheduling = supplement["future_submission_gate"]
    assert scheduling["required_a_priority_class"] == "fleet-train-high"
    assert scheduling["rejected_priority_class"] == "fleet-infra-quiet"
    assert scheduling["fresh_a_non_scored_parity_required"] is True
    assert scheduling["true_non_preemptible_available"] is False
    assert scheduling["higher_priority_may_preempt_lower_priority"] is True


def test_qwen_source6_fence_selects_r56_and_restores_200_cells() -> None:
    supplement = hosted.load_object(QWEN_SOURCE6_REPLACEMENT_SUPPLEMENT)
    assert supplement["receipt_sha256"] == self_hosted.digest_without(
        supplement, "receipt_sha256"
    )
    prior_path = Path(supplement["prior_supplement"]["path"])
    prior = hosted.load_object(prior_path)
    assert supplement["prior_supplement"]["receipt_sha256"] == prior[
        "receipt_sha256"
    ]
    assert hashlib.sha256(prior_path.read_bytes()).hexdigest() == supplement[
        "prior_supplement"
    ]["file_sha256"].removeprefix("sha256:")

    replacement = supplement["replacement"]
    selected = hosted.load_object(FROZEN_SELECTION)["tasks"][55]
    fields = (
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
    assert replacement["replacement_rank"] == selected["rank"] == 56
    assert replacement["historical_rank"] == selected["historical_rank"] == 61
    assert all(replacement[field] == selected[field] for field in fields)

    fence = supplement["source6_fence"]
    assert fence["whole_task_fenced"] is True
    assert fence["accepted_attempt"]["attempt"] == 1
    assert fence["accepted_attempt"]["attrition_only"] is True
    assert fence["noncreditable_attempt"]["attempt"] == 2
    assert fence["noncreditable_attempt"]["agent_exit_code"] == 1
    assert fence["noncreditable_attempt"]["session_ingest_completed"] is True
    assert fence["noncreditable_attempt"]["verifier_execution_present"] is True
    assert fence["noncreditable_attempt"]["retry_allowed"] is False
    assert fence["attempts_3_and_4_launched"] is False
    assert fence["controller_advanced_to_source7"] is True

    successor = hosted.load_object(QWEN_HTTP500_SUCCESSOR_PLAN)
    retained = hosted.load_object(
        Path("evals/fleet/configs/qwen38-opencode-hosted-complete47-pass4-v7.json")
    )
    primary_ids = {
        row["task"]["version_id"]
        for row in successor["tasks"]
        if row["source_rank"] != 6
    }
    primary_ids.add(
        next(
            row["task"]["version_id"]
            for row in retained["tasks"]
            if row["source_rank"] == 4
        )
    )
    assert replacement["task_version_id"] not in primary_ids
    primary_ids.add(replacement["task_version_id"])
    assert len(primary_ids) == 50
    overlap = supplement["overlap_audit"]
    assert overlap["same_qwen_plan_or_claim_references"] == 0
    assert overlap["cross_model_glm_references_present"] is True
    assert overlap["cross_model_references_are_not_same_treatment_cells"] is True
    assert overlap["unused_for_current_qwen_experiment"] is True

    estimator = supplement["revised_primary_estimator"]
    assert estimator["retained_complete_source4_task_count"] == 1
    assert estimator["current_v8_intact_task_count_after_source6_fence"] == 48
    assert estimator["unique_task_count"] == 50
    assert estimator["cell_count"] == 200
    assert estimator["partially_accepted_sessions_outside_estimator"] == 6
    assert estimator["formal_accept_count_if_primary_completes"] == 206
    assert replacement["scored_launch_authorized"] is False
    assert supplement["future_submission_gate"]["required_priority_class"] == (
        "fleet-train-high"
    )


def test_glm_source13_fence_selects_r111_and_restores_400_cells() -> None:
    supplement = hosted.load_object(GLM_SOURCE13_REPLACEMENT_SUPPLEMENT)
    assert supplement["receipt_sha256"] == self_hosted.digest_without(
        supplement, "receipt_sha256"
    )
    prior_path = Path(supplement["prior_supplement"]["path"])
    prior = hosted.load_object(prior_path)
    assert supplement["prior_supplement"]["receipt_sha256"] == prior[
        "receipt_sha256"
    ]
    assert hashlib.sha256(prior_path.read_bytes()).hexdigest() == supplement[
        "prior_supplement"
    ]["file_sha256"].removeprefix("sha256:")

    selection = hosted.load_object(FROZEN_SELECTION)
    split = hosted.load_object(FROZEN_SPLIT)
    runnable = hosted.load_object(FROZEN_RUNNABLE_INVENTORY)
    selected = {row["task_version_id"] for row in selection["tasks"]}
    train = {
        row["task_version_id"]
        for row in split["tasks"]
        if row["split"] == "train"
    }
    ineligible = {
        row["task_version_id"]
        for row in selection["self_hosted_eligibility"]["excluded"]
    }
    remaining = [
        row
        for row in runnable["tasks"]["task_versions"]
        if row["task_version_id"]
        in train - selected - sweep.PRIOR_QWEN_TASK_VERSION_IDS - ineligible
    ]
    remaining.sort(key=lambda row: (row["task_key"], row["task_version_id"]))
    replacement = supplement["replacement"]
    assert remaining[10]["task_version_id"] == replacement["task_version_id"]
    assert replacement["replacement_rank"] == 111
    assert replacement["historical_rank"] == 119
    split_row = next(
        row
        for row in split["tasks"]
        if row["task_version_id"] == replacement["task_version_id"]
    )
    fields = (
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
    assert all(replacement[field] == split_row[field] for field in fields)

    fence = supplement["source13_fence"]
    assert fence["whole_task_fenced"] is True
    assert fence["accepted_attempt"]["attempt"] == 1
    assert fence["accepted_attempt"]["attrition_only"] is True
    assert fence["noncreditable_attempt"]["attempt"] == 2
    assert fence["noncreditable_attempt"]["agent_exit_code"] == 1
    assert fence["noncreditable_attempt"]["session_ingest_completed"] is True
    assert fence["noncreditable_attempt"]["verifier_execution_present"] is True
    assert fence["noncreditable_attempt"]["retry_allowed"] is False
    assert fence["attempts_3_and_4_launched"] is False
    assert fence["controller_advanced_to_source15"] is True

    hosted_primary = hosted.load_object(GLM_HTTP500_HOSTED_PRIMARY_PLAN)
    dedicated_a = hosted.load_object(GLM_DEDICATED_A_V5_PLAN)
    dedicated_b = hosted.load_object(GLM_DEDICATED_B_V5_PLAN)
    blocks = [
        {
            row["task"]["version_id"]
            for row in hosted_primary["tasks"]
            if row["source_rank"] != 13
        }
        | {replacement["task_version_id"]},
        {row["task"]["version_id"] for row in dedicated_a["tasks"]},
        {row["task"]["version_id"] for row in dedicated_b["tasks"]},
    ]
    assert [len(block) for block in blocks] == [46, 27, 27]
    for index, block in enumerate(blocks):
        assert all(block.isdisjoint(other) for other in blocks[index + 1 :])
    assert len(set().union(*blocks)) == 100

    estimator = supplement["revised_primary_estimator"]
    assert estimator["hosted_v12_intact_task_count_after_source13_fence"] == 45
    assert estimator["hosted_task_count"] == 46
    assert estimator["unique_task_count"] == 100
    assert estimator["cell_count"] == 400
    assert estimator["partially_accepted_sessions_outside_estimator"] == 9
    assert estimator["formal_accept_count_if_primary_completes"] == 409
    assert replacement["scored_launch_authorized"] is False
    assert supplement["future_submission_gate"]["required_priority_class"] == (
        "fleet-train-high"
    )


def test_completed_ingested_exit1_is_a_creditable_pass_at_k_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_version_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())
    verifier_execution_id = str(uuid.uuid4())
    instance_id = str(uuid.uuid4())
    result = {
        "run_id": "run-1",
        "task_key": "task-1",
        "task_version_id": task_version_id,
        "instance_id": instance_id,
        "agent_termination": "completed",
        "agent_exit_code": 1,
        "session_ingest_status": "completed",
        "session_id": session_id,
        "verifier_execution_id": verifier_execution_id,
    }
    (tmp_path / "result.json").write_text(json.dumps(result))
    (tmp_path / "reward-result.json").write_text(
        json.dumps(
            {
                "task_key": "task-1",
                "task_version_id": task_version_id,
                "instance_id": instance_id,
                "verifier_execution_id": verifier_execution_id,
                "reward": 0,
            }
        )
    )
    (tmp_path / "session-ingest.json").write_text(
        json.dumps({"status": "completed", "session_id": session_id})
    )
    (tmp_path / "cleanup.json").write_text(
        json.dumps(
            {
                "instance_created": True,
                "instance_closed": True,
                "containers_removed": True,
            }
        )
    )
    config = {
        "run_id": "run-1",
        "task": {"key": "task-1", "version_id": task_version_id},
        "model": {"session_model": "fleet-cluster-opencode-1.18.27/test-model"},
        "config_sha256": "sha256:" + "1" * 64,
    }
    item = {"rank": 1, "source_rank": 7, "attempt": 1}
    monkeypatch.setattr(
        self_hosted,
        "_task_sessions",
        lambda _client, _task_key: [
            {
                "session_id": session_id,
                "status": "completed",
                "model": "test-model",
                "verifier_execution": {"id": verifier_execution_id},
                "metadata": {},
            }
        ],
    )

    receipt = hosted._classify_result(
        tmp_path, config, item, "sha256:" + "2" * 64, "test-key"
    )

    assert receipt["accepted"] is True
    assert receipt["credited"] is True
    assert receipt["agent_exit_code"] == 1
    assert receipt["agent_process_exit_success"] is False
    assert receipt["schema_version"] == "fleet-hosted-opencode-attempt-accepted-v1"
    assert (tmp_path / "ACCEPTED.json").exists()
    assert not (tmp_path / "NONCREDITABLE.json").exists()


def test_exit1_without_authoritative_ingest_remains_infrastructure_incomplete(
    tmp_path: Path,
) -> None:
    task_version_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())
    verifier_execution_id = str(uuid.uuid4())
    instance_id = str(uuid.uuid4())
    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "task_key": "task-1",
                "task_version_id": task_version_id,
                "instance_id": instance_id,
                "agent_termination": "completed",
                "agent_exit_code": 1,
                "session_ingest_status": "failed",
                "session_id": session_id,
                "verifier_execution_id": verifier_execution_id,
            }
        )
    )
    (tmp_path / "reward-result.json").write_text(
        json.dumps(
            {
                "task_key": "task-1",
                "task_version_id": task_version_id,
                "instance_id": instance_id,
                "verifier_execution_id": verifier_execution_id,
                "reward": 0,
            }
        )
    )
    (tmp_path / "session-ingest.json").write_text(
        json.dumps({"status": "failed", "session_id": session_id})
    )
    (tmp_path / "cleanup.json").write_text(
        json.dumps(
            {
                "instance_created": True,
                "instance_closed": True,
                "containers_removed": True,
            }
        )
    )
    config = {
        "run_id": "run-1",
        "task": {"key": "task-1", "version_id": task_version_id},
        "config_sha256": "sha256:" + "1" * 64,
    }

    with pytest.raises(RuntimeError, match="infrastructure-incomplete"):
        hosted._classify_result(
            tmp_path,
            config,
            {"rank": 1, "source_rank": 7, "attempt": 1},
            "sha256:" + "2" * 64,
            "test-key",
        )


@pytest.mark.parametrize("reward_state", ["missing", "verifier_mismatch"])
def test_exit1_reward_evidence_must_exist_and_match_verifier(
    tmp_path: Path,
    reward_state: str,
) -> None:
    task_version_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())
    verifier_execution_id = str(uuid.uuid4())
    instance_id = str(uuid.uuid4())
    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "task_key": "task-1",
                "task_version_id": task_version_id,
                "instance_id": instance_id,
                "agent_termination": "completed",
                "agent_exit_code": 1,
                "session_ingest_status": "completed",
                "session_id": session_id,
                "verifier_execution_id": verifier_execution_id,
            }
        )
    )
    if reward_state != "missing":
        (tmp_path / "reward-result.json").write_text(
            json.dumps(
                {
                    "task_key": "task-1",
                    "task_version_id": task_version_id,
                    "instance_id": instance_id,
                    "verifier_execution_id": str(uuid.uuid4()),
                    "reward": 0,
                }
            )
        )
    (tmp_path / "session-ingest.json").write_text(
        json.dumps({"status": "completed", "session_id": session_id})
    )
    (tmp_path / "cleanup.json").write_text(
        json.dumps(
            {
                "instance_created": True,
                "instance_closed": True,
                "containers_removed": True,
            }
        )
    )
    config = {
        "run_id": "run-1",
        "task": {"key": "task-1", "version_id": task_version_id},
        "config_sha256": "sha256:" + "1" * 64,
    }

    with pytest.raises((FileNotFoundError, RuntimeError, ValueError)):
        hosted._classify_result(
            tmp_path,
            config,
            {"rank": 1, "source_rank": 7, "attempt": 1},
            "sha256:" + "2" * 64,
            "test-key",
        )


def test_completed_exit1_reconciliation_is_exact_plan_bound_and_score_blind() -> None:
    receipt_path = Path(
        "docs/evidence/qwen38-study/"
        "2026-09-04-completed-exit1-reconciliation-v1.json"
    )
    qwen_plan = hosted.load_object(
        Path(
            "evals/fleet/configs/"
            "qwen38-opencode-hosted-http500-successor49-pass4-v8.json"
        )
    )
    glm_plan = hosted.load_object(
        Path(
            "evals/fleet/configs/"
            "glm53-opencode-hosted-http500-primary46-pass4-v12.json"
        )
    )
    receipt = hosted.load_object(receipt_path)

    hosted.validate_completed_exit1_reconciliation(receipt, qwen_plan, glm_plan)
    assert {
        (row["model_block"], row["source_rank"], row["attempt"])
        for row in receipt["reconciled_cells"]
    } == {
        ("qwen_hosted_v8", 6, 2),
        ("qwen_hosted_v8", 7, 1),
        ("glm_hosted_v12", 13, 2),
        ("glm_hosted_v12", 15, 1),
    }
    assert all(
        row["reconciled_outcome"] == "RECONCILED_ACCEPTED"
        for row in receipt["reconciled_cells"]
    )
    assert receipt["primary_denominators_after_gap_completion"]["qwen"]["cells"] == 200
    assert receipt["primary_denominators_after_gap_completion"]["glm"]["cells"] == 400
    assert receipt["privacy"]["scores_read"] is False
    assert receipt["privacy"]["prompts_or_traces_read"] is False

    tampered = json.loads(json.dumps(receipt))
    tampered["reconciled_cells"][0]["evidence"]["reward_result_present"] = False
    tampered["receipt_sha256"] = self_hosted.digest_without(
        tampered, "receipt_sha256"
    )
    with pytest.raises(ValueError, match="not fully scored"):
        hosted.validate_completed_exit1_reconciliation(
            tampered, qwen_plan, glm_plan
        )
