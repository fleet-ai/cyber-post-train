from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.fleet import rollout_refiller, rollout_successor

SOURCE = "chris-cyber-rollout-ledger-qwen-dedicated-load16-r21-v1"
NEW = "chris-cyber-rollout-ledger-qwen-dedicated-load16-r23-v1"
OLD_WORKER = "chris-qwen-dedicated-load16-r21-v1"
NEW_WORKER = "chris-qwen-dedicated-load16-r23-v1"


def _job() -> dict:
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": SOURCE,
            "uid": "source-uid",
            "resourceVersion": "10",
            "labels": {"controller-uid": "old-controller"},
        },
        "spec": {
            "backoffLimit": 0,
            "activeDeadlineSeconds": 36000,
            "podFailurePolicy": {"rules": [{"action": "FailJob"}]},
            "selector": {"matchLabels": {"controller-uid": "old-controller"}},
            "template": {
                "metadata": {
                    "labels": {"job-name": SOURCE, "controller-uid": "old-controller"},
                    "annotations": {"kueue.x-k8s.io/workload": "old-workload"},
                },
                "spec": {
                    "containers": [
                        {
                            "name": "evaluator",
                            "args": ["fleet-rollout-controller-clean-exit-v1 " + OLD_WORKER],
                        }
                    ],
                    "volumes": [{"configMap": {"name": SOURCE}}],
                },
            },
        },
        "status": {"succeeded": 1},
    }


def _config() -> dict:
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": SOURCE, "uid": "config-uid"},
        "data": {"run.sh": f"--worker-id {OLD_WORKER}"},
    }


def test_clone_replaces_full_and_short_identities_and_removes_runtime_fields() -> None:
    job = rollout_successor.clone_job(
        _job(),
        source_name=SOURCE,
        new_name=NEW,
        old_worker_id=OLD_WORKER,
        new_worker_id=NEW_WORKER,
        refiller_id="test-refiller",
    )
    config = rollout_successor.clone_config_map(
        _config(),
        source_name=SOURCE,
        new_name=NEW,
        old_worker_id=OLD_WORKER,
        new_worker_id=NEW_WORKER,
        refiller_id="test-refiller",
    )
    encoded = json.dumps([config, job], sort_keys=True)
    assert SOURCE not in encoded
    assert OLD_WORKER not in encoded
    assert NEW in encoded
    assert NEW_WORKER in encoded
    assert "selector" not in job["spec"]
    assert job["spec"]["backoffLimit"] == rollout_successor.ALERT_SAFE_BACKOFF_LIMIT
    assert "activeDeadlineSeconds" not in job["spec"]
    assert "podFailurePolicy" not in job["spec"]
    assert (
        job["metadata"]["annotations"][rollout_successor.ALERT_SAFE_POLICY_ANNOTATION]
        == rollout_successor.ALERT_SAFE_POLICY_VERSION
    )
    assert "status" not in job
    assert "uid" not in job["metadata"]
    assert job["metadata"]["labels"][rollout_refiller.REFILL_LABEL] == "test-refiller"


def test_clone_rejects_missing_clean_exit_wrapper() -> None:
    source = _job()
    source["spec"]["template"]["spec"]["containers"][0]["args"] = [OLD_WORKER]
    with pytest.raises(rollout_successor.SuccessorError, match="clean-exit"):
        rollout_successor.clone_job(
            source,
            source_name=SOURCE,
            new_name=NEW,
            old_worker_id=OLD_WORKER,
            new_worker_id=NEW_WORKER,
            refiller_id="test-refiller",
        )


def test_postgres_successor_injects_backend_and_secret_reference() -> None:
    config = _config()
    config["data"]["run.sh"] = """
for name in rollout_worker.py rollout_campaign.py rollout_ledger.py opencode_self_hosted.py; do
  true
done
exec uv run --no-project --with httpx==0.28.1 python -m evals.fleet.rollout_worker \\
  --database "$state/ledger.sqlite3"
"""
    job = _job()
    job["spec"]["template"]["spec"]["containers"][0]["env"] = []

    rollout_successor.configure_postgres(
        config,
        job,
        repo_root=Path.cwd(),
        postgres_secret="postgres-secret",
        worker_prefix=NEW_WORKER,
    )

    assert "rollout_postgres.py" in config["data"]
    assert "--postgres-dsn-env ROLLOUT_DATABASE_URL" in config["data"]["run.sh"]
    assert '--database "$state/ledger.sqlite3"' not in config["data"]["run.sh"]
    assert job["metadata"]["annotations"][rollout_refiller.WORKER_PREFIX_ANNOTATION] == NEW_WORKER
    assert (
        job["spec"]["template"]["metadata"]["labels"][rollout_successor.POSTGRES_CLIENT_LABEL]
        == "true"
    )
    database_environment = job["spec"]["template"]["spec"]["containers"][0]["env"][-1]
    assert database_environment["valueFrom"]["secretKeyRef"] == {
        "name": "postgres-secret",
        "key": "ROLLOUT_DATABASE_URL",
    }


def test_render_live_stamps_requested_concurrency_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    source_job = _job()
    source_job["status"] = {"conditions": [{"type": "Complete", "status": "True"}]}

    def fake_kubectl(arguments: list[str], *, expect_absent: bool = False) -> dict | None:
        if expect_absent:
            return None
        return source_job if arguments[3] == "job" else _config()

    monkeypatch.setattr(rollout_successor, "_kubectl_json", fake_kubectl)
    rendered = rollout_successor.render_live(
        namespace="fleet-train-jobs",
        source_name=SOURCE,
        expected_source_uid="source-uid",
        new_name=NEW,
        old_worker_id=OLD_WORKER,
        new_worker_id=NEW_WORKER,
        refiller_id="test-refiller",
        concurrency_stage=32,
    )
    job = rendered["items"][1]
    assert job["metadata"]["annotations"][rollout_successor.CONCURRENCY_STAGE_ANNOTATION] == "32"


@pytest.mark.parametrize("priority_class", [None, "q1"])
def test_priority_opt_in_preserves_pod_policy(monkeypatch, priority_class) -> None:
    source = _job()
    source["metadata"]["labels"][rollout_successor.WORKLOAD_PRIORITY_LABEL] = "q0"
    source["status"] = {"conditions": [{"type": "Complete", "status": "True"}]}
    source["spec"]["template"]["spec"]["priorityClassName"] = "c2"

    def read(arguments, *, expect_absent=False):
        if expect_absent:
            return None
        if arguments[0] == "get":
            return {"metadata": {"uid": "q1-uid"}, "value": 10000}
        return source if arguments[3] == "job" else _config()

    monkeypatch.setattr(rollout_successor, "_kubectl_json", read)
    rendered = rollout_successor.render_live(
        namespace="test",
        source_name=SOURCE,
        expected_source_uid="source-uid",
        new_name=NEW,
        old_worker_id=OLD_WORKER,
        new_worker_id=NEW_WORKER,
        refiller_id="test",
        workload_priority_class=priority_class,
        expected_priority_class_uid="q1-uid" if priority_class else None,
    )
    job = rendered["items"][1]
    assert (
        job["metadata"]["labels"].get(rollout_successor.WORKLOAD_PRIORITY_LABEL) == priority_class
    )
    assert job["spec"]["template"]["spec"]["priorityClassName"] == "c2"
    assert "priority" not in job["spec"]
    assert "admission" not in job["spec"]
    assert job["spec"]["backoffLimit"] == 2_147_483_647


def test_priority_requires_immutable_uid() -> None:
    with pytest.raises(rollout_successor.SuccessorError, match="supplied together"):
        rollout_successor.render_live(
            namespace="test",
            source_name=SOURCE,
            expected_source_uid="source-uid",
            new_name=NEW,
            old_worker_id=OLD_WORKER,
            new_worker_id=NEW_WORKER,
            refiller_id="test",
            workload_priority_class="q1",
        )


def test_priority_class_uid_drift_is_rejected(monkeypatch) -> None:
    source = _job()
    source["status"] = {"conditions": [{"type": "Complete", "status": "True"}]}

    def read(arguments, *, expect_absent=False):
        if expect_absent:
            return None
        if arguments[0] == "get":
            return {"metadata": {"uid": "replacement-uid"}, "value": 10000}
        return source if arguments[3] == "job" else _config()

    monkeypatch.setattr(rollout_successor, "_kubectl_json", read)
    with pytest.raises(rollout_successor.SuccessorError, match="identity changed"):
        rollout_successor.render_live(
            namespace="test",
            source_name=SOURCE,
            expected_source_uid="source-uid",
            new_name=NEW,
            old_worker_id=OLD_WORKER,
            new_worker_id=NEW_WORKER,
            refiller_id="test",
            workload_priority_class="q1",
            expected_priority_class_uid="q1-uid",
        )
