from __future__ import annotations

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from evals.fleet import campaign_supervisor as supervisor

CAMPAIGN = Path("evals/fleet/configs/q38-glm53-primary-campaign-v1.json")
WORKER_UID = "11111111-1111-4111-8111-111111111111"


def _released_fixture_campaign() -> dict:
    """Release only an isolated test fixture; the checked-in campaign stays blocked."""
    campaign = supervisor.read_object(CAMPAIGN)
    campaign["release_status"] = "released"
    campaign["launch_authorized"] = True
    campaign["campaign_sha256"] = supervisor.digest_without(campaign, "campaign_sha256")
    return campaign


def _ledger(tmp_path: Path) -> tuple[Path, dict, dict]:
    campaign = _released_fixture_campaign()
    root = tmp_path / "ledger"
    universe = supervisor.initialize_ledger(campaign, root, cluster=False)
    qwen = next(
        cell for cell in universe["cells"] if cell["component_id"] == "qwen-hosted-v8-primary49"
    )
    registration = {
        "schema_version": supervisor.WORKER_SCHEMA,
        "campaign_sha256": campaign["campaign_sha256"],
        "worker_name": "qwen-tail-a-v9",
        "worker_uid": WORKER_UID,
        "namespace": "fleet-train-jobs",
        "plan_sha256": "sha256:" + "a" * 64,
        "run_id_prefix": "chris-cyber-q38-opencode11827-hosted-tail-a22-p4-v9-",
        "owned_partitions": [
            {
                "component_id": "qwen-hosted-v8-primary49",
                "source_ranks": list(range(11, 33)),
            }
        ],
        "priority_class": "fleet-train-high",
        "owned": True,
        "scores_included": False,
        "prompts_or_traces_included": False,
    }
    registration["registration_sha256"] = supervisor.digest_without(
        registration, "registration_sha256"
    )
    supervisor.register_worker(root, registration)
    return root, universe, qwen


def test_frozen_campaign_is_exact_600_cell_disjoint_universe(tmp_path: Path) -> None:
    campaign = _released_fixture_campaign()
    universe = supervisor.initialize_ledger(campaign, tmp_path / "ledger", cluster=False)
    assert universe["cell_count"] == 600
    assert universe["model_counts"] == {"glm-5.3": 400, "qwen3.8-27b": 200}
    assert universe["component_counts"] == {
        "glm-dedicated-a-v5-primary27": 108,
        "glm-dedicated-b-v5-primary27": 108,
        "glm-hosted-v12-primary46": 184,
        "qwen-hosted-retained-source4": 4,
        "qwen-hosted-v8-primary49": 196,
    }
    identities = [
        (cell["model"], cell["task_version_id"], cell["attempt"]) for cell in universe["cells"]
    ]
    assert len(identities) == len(set(identities)) == 600


def test_checked_in_campaign_cannot_initialize_before_replacements_are_bound(
    tmp_path: Path,
) -> None:
    campaign = supervisor.read_object(CAMPAIGN)
    with pytest.raises(ValueError, match="blocked pending"):
        supervisor.initialize_ledger(campaign, tmp_path / "ledger", cluster=False)


def test_campaign_rejects_capacity_or_plan_drift() -> None:
    campaign = _released_fixture_campaign()
    drifted = copy.deepcopy(campaign)
    drifted["expected"]["dedicated_gpus"] = 24
    drifted["campaign_sha256"] = supervisor.digest_without(drifted, "campaign_sha256")
    with pytest.raises(ValueError, match="denominator or dedicated capacity"):
        supervisor.validate_campaign(drifted)

    drifted = copy.deepcopy(campaign)
    drifted["components"][0]["plan_sha256"] = "sha256:" + "0" * 64
    drifted["campaign_sha256"] = supervisor.digest_without(drifted, "campaign_sha256")
    with pytest.raises(ValueError, match="plan digest binding"):
        supervisor.build_universe(drifted, cluster=False)


def test_atomic_claim_allows_exactly_one_controller(tmp_path: Path) -> None:
    root, universe, _ = _ledger(tmp_path)
    cell = next(
        row
        for row in universe["cells"]
        if row["component_id"] == "qwen-hosted-v8-primary49"
        and row["source_rank"] == 11
        and row["attempt"] == 1
    )
    kwargs = {
        "cell_id": cell["cell_id"],
        "worker_name": "qwen-tail-a-v9",
        "worker_uid": WORKER_UID,
        "run_id": "chris-cyber-q38-opencode11827-hosted-tail-a22-p4-v9-sr011-a1",
    }
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(supervisor.claim_cell, root, **kwargs) for _ in range(2)]
    successes = [future.result() for future in futures if future.exception() is None]
    failures = [future.exception() for future in futures if future.exception() is not None]
    assert len(successes) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], FileExistsError)


def test_claim_rejects_foreign_partition_and_run_namespace(tmp_path: Path) -> None:
    root, universe, _ = _ledger(tmp_path)
    foreign = next(
        row for row in universe["cells"] if row["component_id"] == "glm-hosted-v12-primary46"
    )
    with pytest.raises(ValueError, match="partition does not own"):
        supervisor.claim_cell(
            root,
            cell_id=foreign["cell_id"],
            worker_name="qwen-tail-a-v9",
            worker_uid=WORKER_UID,
            run_id="chris-cyber-q38-opencode11827-hosted-tail-a22-p4-v9-x",
        )
    owned = next(
        row
        for row in universe["cells"]
        if row["component_id"] == "qwen-hosted-v8-primary49" and row["source_rank"] == 11
    )
    with pytest.raises(ValueError, match="run ID"):
        supervisor.claim_cell(
            root,
            cell_id=owned["cell_id"],
            worker_name="qwen-tail-a-v9",
            worker_uid=WORKER_UID,
            run_id="foreign-run",
        )


def test_outcomes_are_append_only_and_mutually_exclusive(tmp_path: Path) -> None:
    root, universe, _ = _ledger(tmp_path)
    cell = next(
        row
        for row in universe["cells"]
        if row["component_id"] == "qwen-hosted-v8-primary49"
        and row["source_rank"] == 11
        and row["attempt"] == 1
    )
    supervisor.claim_cell(
        root,
        cell_id=cell["cell_id"],
        worker_name="qwen-tail-a-v9",
        worker_uid=WORKER_UID,
        run_id="chris-cyber-q38-opencode11827-hosted-tail-a22-p4-v9-sr011-a1",
    )
    accepted = {
        "schema_version": "fleet-hosted-opencode-attempt-accepted-v1",
        "accepted": True,
        "credited": True,
        "task_version_id": cell["task_version_id"],
        "attempt": 1,
        "scores_included": False,
        "prompts_or_traces_included": False,
    }
    accepted["receipt_sha256"] = supervisor.digest_without(accepted, "receipt_sha256")
    accepted_path = tmp_path / "accepted.json"
    accepted_path.write_text(json.dumps(accepted))
    supervisor.record_outcome(
        root, cell_id=cell["cell_id"], kind="accepted", evidence_path=accepted_path
    )
    quarantine = {
        "schema_version": "fleet-score-blind-quarantine-v1",
        "retry_allowed": False,
        "task_version_id": cell["task_version_id"],
        "attempt": 1,
        "scores_included": False,
        "prompts_or_traces_included": False,
    }
    quarantine["receipt_sha256"] = supervisor.digest_without(quarantine, "receipt_sha256")
    quarantine_path = tmp_path / "quarantine.json"
    quarantine_path.write_text(json.dumps(quarantine))
    with pytest.raises(RuntimeError, match="contradictory terminal"):
        supervisor.record_outcome(
            root,
            cell_id=cell["cell_id"],
            kind="quarantine",
            evidence_path=quarantine_path,
        )
    snapshot = supervisor.status(root)
    qwen = next(row for row in snapshot["blocks"] if row["model"] == "qwen3.8-27b")
    assert qwen == {
        "model": "qwen3.8-27b",
        "serving_block": "qwen-hosted-no-autocontinue",
        "accepted": 1,
        "quarantined": 0,
        "claimed": 0,
        "unclaimed": 199,
        "total": 200,
    }


def test_outcome_rejects_score_bearing_evidence(tmp_path: Path) -> None:
    root, universe, _ = _ledger(tmp_path)
    cell = next(
        row
        for row in universe["cells"]
        if row["component_id"] == "qwen-hosted-v8-primary49" and row["source_rank"] == 11
    )
    supervisor.claim_cell(
        root,
        cell_id=cell["cell_id"],
        worker_name="qwen-tail-a-v9",
        worker_uid=WORKER_UID,
        run_id="chris-cyber-q38-opencode11827-hosted-tail-a22-p4-v9-sr011-a1",
    )
    evidence = {
        "accepted": True,
        "score": 0.5,
        "scores_included": True,
        "prompts_or_traces_included": False,
    }
    evidence["receipt_sha256"] = supervisor.digest_without(evidence, "receipt_sha256")
    path = tmp_path / "score-bearing.json"
    path.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="score/content blind"):
        supervisor.record_outcome(
            root, cell_id=cell["cell_id"], kind="accepted", evidence_path=path
        )


def test_heartbeat_uses_exact_name_uid_and_high_priority(tmp_path: Path) -> None:
    root, _, _ = _ledger(tmp_path)

    class Kube:
        routes: list[str] = []

        def get(self, route: str):
            self.routes.append(route)
            return {
                "metadata": {"uid": WORKER_UID},
                "spec": {"template": {"spec": {"priorityClassName": "fleet-train-high"}}},
                "status": {"active": 1},
            }

    kube = Kube()
    receipt = supervisor.heartbeat_once(root, kube)  # type: ignore[arg-type]
    assert kube.routes == ["/apis/batch/v1/namespaces/fleet-train-jobs/jobs/qwen-tail-a-v9"]
    assert receipt["workers"][0]["state"] == "active"
    assert receipt["exact_named_gets_only"] is True
    assert receipt["peer_objects_mutated"] is False
    assert len(list((root / "heartbeats").glob("*.json"))) == 1


def test_heartbeat_fails_closed_on_uid_or_priority_drift(tmp_path: Path) -> None:
    root, _, _ = _ledger(tmp_path)

    class Kube:
        def __init__(self, uid: str, priority: str) -> None:
            self.uid = uid
            self.priority = priority

        def get(self, _route: str):
            return {
                "metadata": {"uid": self.uid},
                "spec": {"template": {"spec": {"priorityClassName": self.priority}}},
                "status": {"active": 1},
            }

    with pytest.raises(RuntimeError, match="UID drifted"):
        supervisor.heartbeat_once(
            root,
            Kube("22222222-2222-4222-8222-222222222222", "fleet-train-high"),  # type: ignore[arg-type]
        )
    with pytest.raises(RuntimeError, match="priority drifted"):
        supervisor.heartbeat_once(root, Kube(WORKER_UID, "fleet-infra-quiet"))  # type: ignore[arg-type]
