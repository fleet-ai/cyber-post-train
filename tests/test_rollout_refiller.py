from __future__ import annotations

from evals.fleet import rollout_refiller


def _job(
    name: str,
    *,
    active: int = 0,
    terminal: str | None = None,
    managed: bool = False,
    worker_prefix: str | None = None,
):
    conditions = [] if terminal is None else [{"type": terminal, "status": "True"}]
    labels = {rollout_refiller.REFILL_LABEL: "q38-glm53-pass4-v1"} if managed else {}
    annotations = (
        {rollout_refiller.WORKER_PREFIX_ANNOTATION: worker_prefix} if worker_prefix else {}
    )
    return {
        "metadata": {"name": name, "labels": labels, "annotations": annotations},
        "status": {"active": active, "conditions": conditions},
    }


def _models(*, missing: str | None = None):
    return [
        {
            "metadata": {"name": name},
            "status": {
                "conditions": [{"type": "Ready", "status": "False" if route == missing else "True"}]
            },
        }
        for route, name in rollout_refiller.ENDPOINTS.items()
    ]


def test_route_plan_counts_jobs_and_claims_without_double_counting() -> None:
    jobs = [
        _job("chris-cyber-rollout-ledger-glm-dedicated-live-v1", active=1),
        _job("chris-cyber-rollout-ledger-glm-shared-initializing-v1", managed=True),
        _job("chris-cyber-rollout-ledger-qwen-shared-old-v1", terminal="Complete"),
    ]
    statuses = {
        status.route: status
        for status in rollout_refiller.plan_routes(
            jobs=jobs,
            active_claims={"glm-dedicated": 1, "glm-shared": 0, "qwen-shared": 2},
            models=_models(),
            target_per_route=4,
        )
    }
    assert statuses["glm-dedicated"].occupancy == 1
    assert statuses["glm-dedicated"].deficit == 3
    assert statuses["glm-shared"].occupancy == 1
    assert statuses["qwen-shared"].occupancy == 2
    assert statuses["qwen-dedicated"].deficit == 4


def test_unready_endpoint_never_proposes_refill() -> None:
    statuses = {
        status.route: status
        for status in rollout_refiller.plan_routes(
            jobs=[],
            active_claims={},
            models=_models(missing="qwen-dedicated"),
            target_per_route=4,
        )
    }
    assert statuses["qwen-dedicated"].endpoint_ready is False
    assert statuses["qwen-dedicated"].deficit == 0


def test_historical_nonterminal_job_is_not_counted_without_active_pod_or_refill_label() -> None:
    jobs = [_job("chris-cyber-rollout-ledger-qwen-dedicated-old-v1")]
    counts = rollout_refiller.active_jobs_by_route(jobs)
    assert counts["qwen-dedicated"] == 0


def test_orphan_claim_blocks_all_refills(monkeypatch) -> None:
    jobs = [
        _job(
            "chris-cyber-rollout-ledger-glm-shared-live-v1",
            active=1,
            managed=True,
            worker_prefix="live-worker",
        )
    ]

    def runner(command):
        if command[3] == "get" and command[4] == "jobs":
            return {"items": jobs}
        return {"items": _models()}

    monkeypatch.setattr(
        rollout_refiller,
        "_ledger_claims",
        lambda _namespace, _pod, _kind: {
            "active_claims": {"qwen-dedicated": 1},
            "stale_active": 1,
            "owners": [
                {
                    "serving_block": "qwen-dedicated",
                    "worker_id": "terminal-worker-qwen-dedicated-0",
                    "claim_id": "claim",
                    "cell_id": "cell",
                    "stale": True,
                }
            ],
        },
    )
    observed = rollout_refiller.observe(
        namespace="fleet-train-jobs",
        ledger_pod="postgres-0",
        ledger_kind="postgres",
        target_per_route=8,
        runner=runner,
    )

    assert observed["refill_blocked"] is True
    assert observed["orphan_active"] == 1
    assert observed["stale_active"] == 1
    assert observed["proposed_refills"] == []
