"""Synthetic metadata only: no task payloads or private corpus fixtures."""

import copy

import pytest

from training.io import digest_json
from training.study_data import (
    EPISODE_SCHEMA,
    INVENTORY_SCHEMA,
    audit_inventory,
    filter_records,
    freeze_final_test,
    seal,
    split_variant,
)
from training.study_data import (
    select_sources as select_metadata_sources,
)

EVIDENCE = "sha256:" + "1" * 64


def select_sources(*args, **kwargs):
    policy = {
        "max_episodes_per_family": 2,
        "max_supervised_tokens_per_family": 1000,
        "source_seed": "frozen-synthetic-source-seed",
        "target_policy_sha256": EVIDENCE,
    }
    return select_metadata_sources(*args, **(policy | kwargs))


def reseal(value):
    return seal({k: v for k, v in value.items() if k != "sha256"})


def verified(value):
    return {"status": "verified", "value": value, "evidence_sha256": EVIDENCE}


def inventory():
    tasks = []
    for i in range(12):
        for version in range(2):
            tasks.append(
                {
                    "task_key": f"synthetic-task-{i}",
                    "task_version_id": f"synthetic-version-{i}-{version}",
                    "lineage_id": f"synthetic-lineage-{i}",
                    "taxonomy": {
                        "task_family": verified(f"reviewed-family-{i}"),
                        "application": verified(f"app-{i % 3}"),
                        "environment": verified(f"environment-{i % 2}"),
                        "vulnerability_family": verified(f"vuln-{i % 4}"),
                        "difficulty": verified(["easy", "hard"][i % 2]),
                    },
                    "certification": {"status": "accepted", "receipt_sha256": EVIDENCE},
                    "exposure": {"status": "unexposed", "receipt_sha256": EVIDENCE},
                }
            )
    return seal({"schema": INVENTORY_SCHEMA, "origin": "fleet_cyber", "tasks": tasks})


def study():
    inv = inventory()
    lock = freeze_final_test(inv, groups_count=3, seed="test-before-eval")
    return split_variant(inv, lock, dev_groups=3, seed="variant-a")


def record(task, sid="session-1", model="synthetic-teacher"):
    return {
        "record_id": sid,
        "lineage": {
            "task_key": task["task_key"],
            "eval_task_version_id": task["task_version_id"],
            "application": "synthetic-app",
            "task_family": task["task_key"],
        },
        "source": {"model": model},
        "eligibility": {"sft": True},
        "outcome": {"infra_valid": True, "success": True, "score": 1},
    }


def episode(private_record, **changes):
    value = {
        "schema": EPISODE_SCHEMA,
        "task_key": private_record["lineage"]["task_key"],
        "task_version_id": private_record["lineage"]["eval_task_version_id"],
        "episode_id": private_record["record_id"],
        "model_id": private_record["source"]["model"],
        "source_kind": "teacher",
        "validity": "valid",
        "verified_success": True,
        "acceptance_sha256": EVIDENCE,
        "trace_sha256": EVIDENCE,
        "normalized_record_sha256": digest_json(private_record),
        "coverage": {
            "status": "certified",
            "receipt_sha256": EVIDENCE,
            "scope": "eligible_dense_targets",
            "target_policy_sha256": EVIDENCE,
            "assistant_responses": 4,
            "supervised_tokens": 100,
            "submit_report_responses": 1,
            "submit_report_tokens": 10,
            "non_submit_tool_responses": 2,
            "decision_responses": 1,
            "other_responses": 0,
            "completed_non_submit_tool_rounds": 2,
        },
    }
    value.update(changes)
    return seal(value)


def train_record(split, **kwargs):
    return record(split["training_split"]["tasks"][0], **kwargs)


def test_grouped_versioned_split_has_no_teacher_references_or_eval_targets():
    split = study()
    assignment = {}
    for row in split["tasks"]:
        assert assignment.setdefault(row["group_id"], row["split"]) == row["split"]
    assert split["training_split"]["schema"] == "cyber_task_split_v2"
    assert len(split["training_split"]["tasks"]) == 12
    assert all(
        set(row) == {"task_key", "task_version_id", "split"} and row["split"] == "train"
        for row in split["training_split"]["tasks"]
    )
    for name in ("dev", "final_test"):
        assert len(split["evaluation"][name]["tasks"]) == 6
        assert all(r["split"] == name for r in split["evaluation"][name]["tasks"])
    assert "reference_session_id" not in str(split)
    assert split["generalization_scope"].endswith("applications may be shared")


def test_deterministic_representative_variants_share_exact_final_test():
    inv = inventory()
    lock = freeze_final_test(inv, groups_count=3, seed="frozen")
    a = split_variant(inv, lock, dev_groups=3, seed="a")
    assert a == split_variant(inv, lock, dev_groups=3, seed="a")
    alternatives = [split_variant(inv, lock, dev_groups=3, seed=str(i)) for i in range(5)]
    assert all(s["evaluation"]["final_test"] == a["evaluation"]["final_test"] for s in alternatives)
    assert any(s["evaluation"]["dev"] != a["evaluation"]["dev"] for s in alternatives)
    assert a["representation"]["final_test"]["application"]["split_group_counts"] == {
        "app-0": 1,
        "app-1": 1,
        "app-2": 1,
    }
    reverse = reseal({**inv, "tasks": list(reversed(inv["tasks"]))})
    assert freeze_final_test(reverse, groups_count=3, seed="frozen")["tasks"] == lock["tasks"]


def test_existing_final_test_protects_whole_family_not_one_version():
    inv = inventory()
    identity = (inv["tasks"][0]["task_key"], inv["tasks"][0]["task_version_id"])
    lock = freeze_final_test(inv, groups_count=3, seed="frozen", protected_task_versions=[identity])
    assert len([r for r in lock["tasks"] if r["task_key"] == identity[0]]) == 2


@pytest.mark.parametrize("status", ["exposed", "unknown"])
def test_exposed_or_unknown_versions_exclude_whole_family_from_test(status):
    inv = inventory()
    inv["tasks"][0]["exposure"] = {"status": status}
    inv = reseal(inv)
    assert all(
        r["task_key"] != inv["tasks"][0]["task_key"]
        for r in freeze_final_test(inv, groups_count=3, seed="x")["tasks"]
    )
    with pytest.raises(ValueError, match="protected final test"):
        freeze_final_test(
            inv,
            groups_count=3,
            seed="x",
            protected_task_versions=[
                (inv["tasks"][1]["task_key"], inv["tasks"][1]["task_version_id"])
            ],
        )


def test_rejected_exposed_version_still_disqualifies_accepted_sibling_from_test():
    inv = inventory()
    inv["tasks"][0]["certification"] = {"status": "rejected"}
    inv["tasks"][0]["exposure"] = {"status": "exposed"}
    inv = reseal(inv)
    lock = freeze_final_test(inv, groups_count=3, seed="x")
    assert not any(r["task_key"] == inv["tasks"][0]["task_key"] for r in lock["tasks"])


def test_missing_non_grouping_taxonomy_is_reported_not_fabricated():
    inv = inventory()
    inv["tasks"][0]["taxonomy"].pop("difficulty")
    inv["tasks"][0]["taxonomy"]["vulnerability_family"] = {"status": "unverified"}
    inv = reseal(inv)
    audit = audit_inventory(inv)
    assert not audit["full_taxonomy_available"]
    assert audit["eligible_task_versions"] == 24
    assert audit["missing_or_unverified_taxonomy_task_versions"]["difficulty"] == 1
    lock = freeze_final_test(inv, groups_count=3, seed="x")
    assert "__missing__" in lock["representation"]["difficulty"]["population_group_counts"]


def test_missing_reviewed_family_quarantines_its_known_aliases():
    inv = inventory()
    inv["tasks"][0]["taxonomy"].pop("task_family")
    audit = audit_inventory(reseal(inv))
    assert audit["eligible_task_versions"] == 22
    assert audit["quarantined_task_versions"]["missing_verified_task_family"] == 1
    assert audit["quarantined_task_versions"]["family_has_unresolved_alias"] == 1


def test_version_family_conflict_is_not_silently_relabelled():
    inv = inventory()
    inv["tasks"][0]["taxonomy"]["task_family"] = verified("different")
    with pytest.raises(ValueError, match="changes reviewed family"):
        audit_inventory(reseal(inv))


@pytest.mark.parametrize("change", ["origin", "payload", "digest", "evidence", "exposure_evidence"])
def test_invalid_inventory_and_non_metadata_inputs_fail_closed(change):
    inv = inventory()
    if change == "origin":
        inv["origin"] = "external_benchmark"
    elif change == "payload":
        inv["tasks"][0]["prompt"] = "synthetic forbidden input"
    elif change == "digest":
        inv["sha256"] = EVIDENCE
    elif change == "evidence":
        inv["tasks"][0]["taxonomy"]["application"]["evidence_sha256"] = "unknown"
    else:
        inv["tasks"][0]["exposure"].pop("receipt_sha256")
    with pytest.raises(ValueError):
        audit_inventory(inv if change == "digest" else reseal(inv))


def test_final_test_inventory_and_family_members_cannot_drift_between_variants():
    inv = inventory()
    lock = freeze_final_test(inv, groups_count=3, seed="x")
    changed = reseal({**inv, "note": "different snapshot"})
    with pytest.raises(ValueError, match="inventory changed"):
        split_variant(changed, lock, dev_groups=3, seed="x")
    lock["tasks"].pop()
    with pytest.raises(ValueError, match="drops a family version"):
        split_variant(inv, reseal(lock), dev_groups=3, seed="x")


@pytest.mark.parametrize("size", [0, -1, 12, True])
def test_invalid_split_sizes_rejected(size):
    with pytest.raises(ValueError):
        freeze_final_test(inventory(), groups_count=size, seed="x")


def test_train_only_teacher_self_audit_and_private_record_digest_adapter():
    split = study()
    teacher = train_record(split)
    own = train_record(split, sid="session-2", model="synthetic-student")
    selection = select_sources(
        [episode(teacher), episode(own, source_kind="self")],
        split,
        models=["synthetic-teacher", "synthetic-student"],
    )
    assert selection["status"] == "ready"
    assert selection["totals"]["assistant_responses"] == 8
    assert selection["totals"]["supervised_tokens"] == 200
    assert selection["coverage"]["covered_train_task_versions"] == 1
    assert selection["dominance"]["submission_token_share"] == 0.1
    assert selection["by_source"]["self:synthetic-student"]["covered_task_versions"] == 1
    selected = filter_records([own, teacher], selection, split)
    assert selected == [teacher, own]
    assert selected[0] is teacher  # Preserve original messages/masks: no rewriting or unmasking.


def test_heldout_metadata_beyond_exact_identity_is_never_read():
    class IdentityOnly(dict):
        def items(self):
            raise AssertionError("heldout fields were inspected")

        def __getitem__(self, key):
            assert key in {"task_key", "task_version_id"}
            return super().__getitem__(key)

        def get(self, *args):
            raise AssertionError("heldout fields were inspected")

    split = study()
    heldout = IdentityOnly(split["evaluation"]["dev"]["tasks"][0])
    selected = select_sources(
        [heldout, episode(train_record(split))], split, models=["synthetic-teacher"]
    )
    assert selected["status"] == "ready"
    assert selected["excluded"] == {"outside_training_split": 1}


@pytest.mark.parametrize(
    "kind", ["submission_only", "no_tool_result", "invalid_counts", "missing_receipt"]
)
def test_source_coverage_and_final_submit_only_gate(kind):
    split = study()
    item = episode(train_record(split))
    c = item["coverage"]
    if kind == "submission_only":
        c.update(
            assistant_responses=1,
            submit_report_responses=1,
            submit_report_tokens=100,
            non_submit_tool_responses=0,
            decision_responses=0,
            completed_non_submit_tool_rounds=0,
        )
    elif kind == "no_tool_result":
        c["completed_non_submit_tool_rounds"] = 0
    elif kind == "invalid_counts":
        c["assistant_responses"] = 1
    else:
        c.pop("receipt_sha256")
    result = select_sources([reseal(item)], split, models=["synthetic-teacher"])
    assert result["status"] == "blocked"
    assert not result["selected_episode_ids"]
    assert sum(result["excluded"].values()) == 1


def test_aggregate_submit_token_dominance_blocks_without_dropping_final_targets():
    split = study()
    item = episode(train_record(split))
    item["coverage"]["submit_report_tokens"] = 80
    result = select_sources([reseal(item)], split, models=["synthetic-teacher"])
    assert result["status"] == "blocked"
    assert result["blockers"] == ["final_submission_token_dominance"]
    assert result["totals"]["assistant_responses"] == 4
    assert result["totals"]["supervised_tokens"] == 100
    with pytest.raises(ValueError, match="blocked"):
        filter_records([train_record(split)], result, split)


def test_source_metadata_forbids_scores_and_payloads():
    split = study()
    item = episode(train_record(split), score=1)
    with pytest.raises(ValueError, match="only certified metadata"):
        select_sources([item], split, models=["synthetic-teacher"])


def test_unverified_failed_or_unbound_records_cannot_enter_training():
    split = study()
    bad = [
        episode(train_record(split, sid=f"session-{i}"), **fields)
        for i, fields in enumerate(
            [
                {"validity": "invalid"},
                {"verified_success": False},
                {"acceptance_sha256": "unknown"},
                {"normalized_record_sha256": "unknown"},
                {"model_id": "another-model"},
            ]
        )
    ]
    result = select_sources(bad, split, models=["synthetic-teacher"])
    assert result["status"] == "blocked"
    assert not result["selected_episode_ids"]
    assert sum(result["excluded"].values()) == 5


@pytest.mark.parametrize("failure", ["missing", "duplicate", "tampered", "wrong_task"])
def test_filter_rejects_missing_duplicate_modified_or_cross_split_sources(failure):
    split = study()
    private = train_record(split)
    selected = select_sources([episode(private)], split, models=["synthetic-teacher"])
    records = [copy.deepcopy(private)]
    if failure == "missing":
        records = []
    elif failure == "duplicate":
        records *= 2
    elif failure == "tampered":
        records[0]["source"]["model"] = "changed"
    else:
        heldout = split["evaluation"]["dev"]["tasks"][0]
        records[0]["lineage"] = {
            "task_key": heldout["task_key"],
            "eval_task_version_id": heldout["task_version_id"],
        }
    with pytest.raises(ValueError):
        filter_records(records, selected, split)


def test_resealed_wrapper_cannot_smuggle_heldout_or_reference_targets_into_corpus_split():
    split = study()
    private = train_record(split)
    heldout = split["evaluation"]["dev"]["tasks"][0]
    split["training_split"]["tasks"].append({**heldout, "split": "train"})
    split["training_split"] = reseal(split["training_split"])
    with pytest.raises(ValueError, match="exactly train-only"):
        select_sources([episode(private)], reseal(split), models=["synthetic-teacher"])


def test_coverage_policy_drift_or_raw_pre_exclusion_counts_are_not_eligible():
    split = study()
    item = episode(train_record(split))
    item["coverage"]["scope"] = "raw_trajectory_before_context_exclusions"
    result = select_sources([reseal(item)], split, models=["synthetic-teacher"])
    assert result["status"] == "blocked"
    item["coverage"]["scope"] = "eligible_dense_targets"
    item["coverage"]["target_policy_sha256"] = "sha256:" + "2" * 64
    result = select_sources([reseal(item)], split, models=["synthetic-teacher"])
    assert result["status"] == "blocked"


def test_existing_corpus_selector_consumes_training_only_v2_without_dev_references():
    from training import corpus

    split = study()
    private = train_record(split)
    metadata = select_sources([episode(private)], split, models=["synthetic-teacher"])
    approved = filter_records([private], metadata, split)
    train, dev, skipped = corpus.select_sources(
        approved, split["training_split"], ["synthetic-teacher"], reference_validation=False
    )
    assert train == [private]
    assert not dev and not skipped


def test_predeclared_family_caps_preserve_model_diversity_not_high_volume_successes():
    split = study()
    candidates = [
        episode(train_record(split, sid=f"teacher-a-{i}", model="teacher-a")) for i in range(10)
    ]
    candidates.append(episode(train_record(split, sid="teacher-b-1", model="teacher-b")))
    a = select_sources(candidates, split, models=["teacher-a", "teacher-b"])
    b = select_sources(list(reversed(candidates)), split, models=["teacher-b", "teacher-a"])
    assert a == b
    assert a["status"] == "ready"
    assert a["totals"]["episodes"] == 2
    assert a["by_source"]["teacher:teacher-a"]["episodes"] == 1
    assert a["by_source"]["teacher:teacher-b"]["episodes"] == 1
    assert a["excluded"]["family_episode_cap"] == 9
    assert a["coverage"]["qualified_candidate_episodes_before_caps"] == 11


def test_family_supervised_token_ceiling_selects_whole_episodes_without_truncation():
    split = study()
    candidates = [episode(train_record(split, sid=f"session-{i}")) for i in range(3)]
    result = select_sources(
        candidates, split, models=["synthetic-teacher"], max_supervised_tokens_per_family=150
    )
    assert result["totals"]["episodes"] == 1
    assert result["totals"]["supervised_tokens"] == 100
    assert result["excluded"]["family_supervised_token_cap"] == 2
    assert next(iter(result["per_family"].values()))["supervised_tokens"] == 100


def test_family_episode_cap_spans_all_versions_not_separate_version_buckets():
    split = study()
    rows = split["training_split"]["tasks"][:2]
    assert rows[0]["task_key"] == rows[1]["task_key"]
    candidates = [episode(record(task, sid=f"version-{i}")) for i, task in enumerate(rows)]
    result = select_sources(
        candidates, split, models=["synthetic-teacher"], max_episodes_per_family=1
    )
    assert result["totals"]["episodes"] == 1
    assert result["excluded"]["family_episode_cap"] == 1


@pytest.mark.parametrize("field", ["max_episodes_per_family", "max_supervised_tokens_per_family"])
def test_family_caps_must_be_explicit_positive_integers(field):
    split = study()
    with pytest.raises(ValueError):
        select_sources(
            [episode(train_record(split))], split, models=["synthetic-teacher"], **{field: 0}
        )
