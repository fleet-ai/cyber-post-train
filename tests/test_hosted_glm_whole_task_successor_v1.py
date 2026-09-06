import copy
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from evals.fleet import hosted_glm_whole_task_engine_v1 as engine
from evals.fleet import hosted_glm_whole_task_package_v1 as package
from evals.fleet import hosted_glm_whole_task_runtime_v1 as runtime
from evals.fleet import hosted_glm_whole_task_successor_v1 as successor
from evals.fleet import self_hosted

ROOT = Path(__file__).resolve().parents[1]
JOB_UID = "11111111-1111-4111-8111-111111111111"
POD_UID = "22222222-2222-4222-8222-222222222222"


@pytest.fixture(autouse=True)
def _safe_runtime_task_projection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Static held plans omit task bodies; model only identity-safe session inputs."""
    monkeypatch.setattr(
        engine,
        "_task_for_item",
        lambda _plan, item: {"task": {"key": item["task_key"]}},
    )
    monkeypatch.setattr(
        engine,
        "_attempt_config",
        lambda _plan, _task, item: {
            "run_id": item["run_id"],
            "execution": {
                "cell_id": item["cell_id"],
                "execution_id": item["execution_id"],
            },
        },
    )


def _plans() -> dict[str, dict]:
    return successor.validate_all(ROOT)


def _provider(tmp_path: Path, plan: dict, **kwargs) -> successor.AtomicWholeTaskClaims:
    jobs = tmp_path / "jobs"
    claims = tmp_path / "claims"
    reservations = tmp_path / "reservations"
    output = jobs / Path(plan["sfs_root"]).name
    for path in (jobs, claims, reservations, output):
        path.mkdir(parents=True, exist_ok=True)
    return successor.AtomicWholeTaskClaims(
        plan,
        output,
        key="inert",
        jobs_root=jobs,
        reservation_root=reservations,
        session_check=lambda _config, _key: None,
        **kwargs,
    )


def test_exact_two_whole_task_partition_and_treatment() -> None:
    plans = _plans()
    assert set(plans) == set(successor.CONTROLLERS)
    seen = set()
    for controller, plan in plans.items():
        rank = successor.CONTROLLERS[controller]["rank"]
        assert plan["partition"] == {
            "whole_task_rank": rank,
            "attempts": [1, 2, 3, 4],
            "all_cells_previously_unstarted_required": True,
            "rank29_cells_excluded_and_preserved": True,
            "other_ranks_excluded": True,
        }
        assert plan["model"]["served_id"] == "glm-5.3"
        assert plan["model"]["revision"] == "30333038ada1f1dacb294a93270305a890b50c14"
        assert plan["harness"]["version"] == "1.18.27"
        assert plan["harness"]["context_management"] == (
            "opencode_1.18.27_native_compaction_autocontinue_v1"
        )
        assert plan["execution"]["required_task_tools"] == ["bash", "submit_report"]
        assert plan["execution"]["endpoint_lease"]["maximum_streams"] == 2
        assert plan["execution"]["workers"] == 1
        assert [row["attempt"] for row in plan["attempts"]] == [1, 2, 3, 4]
        seen.update((row["selection_rank"], row["attempt"]) for row in plan["attempts"])
    assert seen == {(rank, attempt) for rank in (30, 31) for attempt in range(1, 5)}
    assert all(rank != 29 for rank, _attempt in seen)


def test_exact_cells_are_frozen_generation_one_identities() -> None:
    expected = {
        30: [
            "sha256:5af2152499b8af043a847707e47cc3c04b81d24c47aa99c33c04a8688cbcafa6",
            "sha256:7e2fffc5a21d93d76e97ab387800a687669bf39e69763d8a5b874dfea4da1f41",
            "sha256:b77c81587baee31eabe1bf65903749e8a89cf7719c38e8f527fd5bc53e087121",
            "sha256:2b6f93f57e1862f339e55ff758ce6258a9665c66df80ad6d67fee21fa9fbe66f",
        ],
        31: [
            "sha256:a432fedf8c7698a95bb5f0af6c93d3ad8b42de7b7f2049e25254ec9e70f3cb63",
            "sha256:5c0c9dd638233c314fe71e6e4f6154cfc98c52c4e270cb207003e1597f4ead13",
            "sha256:debb0df658849b3af576852d65a0237056f231b3b01dcef73cff37cb02073cf0",
            "sha256:8712942282c3755cd5ce49bdb2d24fdcf43856a9e1a22d78595663878aa8776d",
        ],
    }
    for plan in _plans().values():
        rank = plan["attempts"][0]["selection_rank"]
        assert [row["cell_id"] for row in plan["attempts"]] == expected[rank]
        assert all(row["execution_generation"] == 1 for row in plan["attempts"])


def test_held_package_is_immutable_create_once_and_closed() -> None:
    rendered = package.render(ROOT)
    assert rendered["launch_authorized"] is False
    assert rendered["scoring_authorized"] is False
    assert rendered["reason"] == "requires_fresh_digest_valid_release"
    items = rendered["objects"]["items"]
    configmaps = [item for item in items if item["kind"] == "ConfigMap"]
    jobs = [item for item in items if item["kind"] == "Job"]
    assert len(configmaps) == len(jobs) == 2
    assert all(item["immutable"] is True for item in configmaps)
    assert all(
        item["metadata"]["annotations"]["cyber-post-train.fleet.ai/launch-authorized"]
        == "false"
        for item in jobs
    )
    assert all(item["spec"]["activeDeadlineSeconds"] == 129_600 for item in jobs)
    assert len({item["metadata"]["name"] for item in items}) == 4
    assert all("release.json" not in item["data"] for item in configmaps)


def test_tracked_held_receipt_binds_exact_plans_and_package() -> None:
    path = (
        ROOT
        / "docs/evidence/glm53-study/"
        "2026-09-06-glm53-hosted-rank30-rank31-whole-task-held-v1.json"
    )
    value = json.loads(path.read_text())
    plans = _plans()
    rendered = package.render(ROOT)
    assert value["controllers"] == successor.release_projection(plans)
    assert value["source_package_sha256"] == rendered["source_package_sha256"]
    assert value["held_package_sha256"] == rendered["package_sha256"]
    assert value["ledger_authority"] == successor.LEDGER_AUTHORITY
    assert value["selection_authority"] == successor.SELECTION_AUTHORITY
    assert value["launch_authorized"] is False
    assert value["scoring_authorized"] is False
    assert value["receipt_sha256"] == self_hosted.digest_without(
        value, "receipt_sha256"
    )


def test_specialized_engine_scopes_atomic_provider_and_restores_base() -> None:
    source = Path(engine.__file__).read_text()
    assert "claim_provider" in source
    assert "base.claim_cell = claim_provider" in source
    assert "base.claim_cell = prior_claim_cell" in source
    assert "claim_provider=provider" in Path(runtime.__file__).read_text()
    engine.base.validate_bulk_adapter(successor)


def test_atomic_reservation_publishes_and_validates_all_four_before_return(
    tmp_path: Path,
) -> None:
    plan = next(iter(_plans().values()))
    provider = _provider(tmp_path, plan)
    claim = provider(plan, plan["attempts"][0], tmp_path / "claims", JOB_UID, POD_UID)
    assert claim["cell_id"] == plan["attempts"][0]["cell_id"]
    assert len(provider.claims) == 4
    assert provider.out.joinpath("RESERVATION.json").is_file()
    reservation = json.loads(provider.out.joinpath("RESERVATION.json").read_text())
    assert reservation["all_claims_before_model_call"] is True
    assert reservation["all_claims_byte_validated"] is True
    assert len(reservation["claim_sha256s"]) == 4
    assert len(list((tmp_path / "claims").glob("*.json"))) == 4


@pytest.mark.parametrize("partial_count", [1, 2, 3, 4])
def test_restart_recovers_crash_like_partial_claim_publication(
    tmp_path: Path, partial_count: int
) -> None:
    plan = next(iter(_plans().values()))
    provider = _provider(tmp_path, plan)
    claim_root = tmp_path / "claims"
    provider._ensure_preparing(JOB_UID, POD_UID)  # noqa: SLF001
    for item in list(reversed(plan["attempts"]))[:partial_count]:
        assert (
            engine.claim_cell(
                plan,
                item,
                claim_root=claim_root,
                job_uid=JOB_UID,
                pod_uid=POD_UID,
            )
            is not None
        )
    assert len(list(claim_root.glob("*.json"))) == partial_count

    restarted = _provider(tmp_path, plan)
    claim = restarted(plan, plan["attempts"][0], claim_root, JOB_UID, POD_UID)
    assert claim["cell_id"] == plan["attempts"][0]["cell_id"]
    assert len(restarted.claims) == 4
    assert len(list(claim_root.glob("*.json"))) == 4
    assert restarted.out.joinpath("RESERVATION.json").is_file()


@pytest.mark.parametrize(
    "stage",
    [
        "after_preparing",
        "after_claim_1",
        "after_claim_2",
        "after_claim_3",
        "after_claim_4",
        "after_validation_1",
        "after_validation_2",
        "after_validation_3",
        "after_validation_4",
        "before_reservation_write",
        "after_reservation_write",
    ],
)
@pytest.mark.parametrize("error", [RuntimeError("fault"), SystemExit(9)])
def test_every_publication_boundary_rolls_back_before_model(
    tmp_path: Path, stage: str, error: BaseException
) -> None:
    plan = next(iter(_plans().values()))

    def fault(observed: str) -> None:
        if observed == stage:
            raise error

    provider = _provider(tmp_path, plan, fault_hook=fault)
    with pytest.raises(type(error)):
        provider(plan, plan["attempts"][0], tmp_path / "claims", JOB_UID, POD_UID)
    if stage == "after_reservation_write":
        assert provider.out.joinpath("RESERVATION.json").is_file()
        assert len(list((tmp_path / "claims").glob("*.json"))) == 4
        recovered = _provider(tmp_path, plan)
        claim = recovered(
            plan, plan["attempts"][0], tmp_path / "claims", JOB_UID, POD_UID
        )
        assert claim["cell_id"] == plan["attempts"][0]["cell_id"]
        assert len(recovered.claims) == 4
    else:
        assert not provider.out.joinpath("RESERVATION.json").exists()
        assert list((tmp_path / "claims").glob("*.json")) == []


def _release(plans: dict[str, dict], source_sha: str) -> dict:
    body = {
        "schema_version": successor.RELEASE_SCHEMA,
        "status": "CLEAR",
        "checked_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "launch_authorized": True,
        "scoring_authorized": True,
        "controller_cap": 2,
        "controllers": successor.release_projection(plans),
        "source_package_sha256": source_sha,
        "ledger_authority": successor.LEDGER_AUTHORITY,
        "selection_authority": successor.SELECTION_AUTHORITY,
        "fresh_collision_reconciliation": {
            "checked_immediately_before_create": True,
            "observer_job_uid": JOB_UID,
            "observer_pod_uid": POD_UID,
            "observed_cells": 8,
            "canonical_claim_collisions": 0,
            "authoritative_session_collisions": 0,
            "accepted_evidence_collisions": 0,
            "output_root_collisions": 0,
            "new_job_collisions": 0,
            "new_configmap_collisions": 0,
            "endpoint_lease_slots_available": 2,
            "api_mutations": 0,
        },
        "rank29_disposition": {
            "preserved": True,
            "selected": False,
            "retry_or_supersession_authorized": False,
        },
        "privacy": {
            "scores_read": False,
            "prompts_traces_flags_read": False,
            "credentials_included": False,
        },
    }
    return {**body, "receipt_sha256": self_hosted.digest_without(body, "receipt_sha256")}


def test_release_requires_every_collision_and_rank29_fence() -> None:
    plans = _plans()
    source_sha = package.source_package_sha256(ROOT)
    valid = _release(plans, source_sha)
    successor.validate_release(valid, plans, source_sha)
    for field in (
        "canonical_claim_collisions",
        "authoritative_session_collisions",
        "accepted_evidence_collisions",
        "output_root_collisions",
        "new_job_collisions",
        "new_configmap_collisions",
    ):
        changed = copy.deepcopy(valid)
        changed["fresh_collision_reconciliation"][field] = 1
        changed["receipt_sha256"] = self_hosted.digest_without(changed, "receipt_sha256")
        with pytest.raises(RuntimeError, match="release drifted"):
            successor.validate_release(changed, plans, source_sha)
    changed = copy.deepcopy(valid)
    changed["rank29_disposition"]["selected"] = True
    changed["receipt_sha256"] = self_hosted.digest_without(changed, "receipt_sha256")
    with pytest.raises(RuntimeError, match="release drifted"):
        successor.validate_release(changed, plans, source_sha)
    changed = copy.deepcopy(valid)
    changed["ledger_authority"]["file_sha256"] = "sha256:" + "0" * 64
    changed["receipt_sha256"] = self_hosted.digest_without(changed, "receipt_sha256")
    with pytest.raises(RuntimeError, match="release drifted"):
        successor.validate_release(changed, plans, source_sha)
    stale = copy.deepcopy(valid)
    stale["checked_at_utc"] = (datetime.now(UTC) - timedelta(seconds=601)).isoformat().replace(
        "+00:00", "Z"
    )
    stale["receipt_sha256"] = self_hosted.digest_without(stale, "receipt_sha256")
    with pytest.raises(RuntimeError, match="release is stale"):
        successor.validate_release(stale, plans, source_sha)
