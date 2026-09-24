import json
from pathlib import Path

from training import miles96_signal_cleanup as cleanup


def test_packet_has_silent_zero_gpu_root_job() -> None:
    value = cleanup.packet()
    assert value["sha256"] == cleanup.digest({k: v for k, v in value.items() if k != "sha256"})
    job = value["bundle"]["items"][1]
    assert job["metadata"]["annotations"]["fleet.ai/failure-alerts"] == "off"
    assert job["spec"]["backoffLimit"] == 0
    assert job["spec"]["suspend"] is True
    assert job["spec"]["template"]["spec"]["priorityClassName"] == "c1"
    assert "nvidia.com/gpu" not in json.dumps(job)


def test_reconcile_releases_only_exact_remaining_instance(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "signal"
    attempts = source / "attempts"
    attempts.mkdir(parents=True)
    ids = []
    for number in range(8):
        attempt = attempts / str(number)
        attempt.mkdir()
        instance_id = f"instance-opaque-{number}"
        ids.append(instance_id)
        (attempt / "binding.json").write_text(
            json.dumps(
                {
                    "task": {"version_id": cleanup.TASK_VERSION_ID},
                    "verifier": {"version_id": cleanup.VERIFIER_VERSION_ID},
                    "model": {
                        "revision": cleanup.MODEL_REVISION,
                        "session_model": cleanup.SERVED_MODEL,
                    },
                }
            )
        )
        (attempt / "runtime-binding.json").write_text(
            json.dumps(
                {
                    "instance_id": instance_id,
                    "env_key": "env",
                    "environment_version": "v1",
                    "data_key": "data",
                    "data_version": "v1",
                }
            )
        )
        if number:
            (attempt / "cleanup.json").write_text(
                json.dumps(
                    {
                        "instance_created": True,
                        "instance_closed": True,
                        "containers_removed": True,
                    }
                )
            )
    live = {ids[0]}
    calls = []

    def request(instance_id: str, method: str):
        calls.append((instance_id, method))
        if method == "DELETE":
            live.remove(instance_id)
            return 204, None
        if instance_id in live:
            return 200, {
                "env_key": "env",
                "version": "v1",
                "data_key": "data",
                "data_version": "v1",
            }
        return 404, None

    monkeypatch.setattr(cleanup, "_request", request)
    monkeypatch.setattr(cleanup, "RECEIPT", source / "LEAK_RECONCILED.json")
    receipt = cleanup.reconcile(source)
    assert receipt["all_instances_absent_after"] is True
    assert receipt["exact_delete_attempted"] is True
    assert calls.count((ids[0], "DELETE")) == 1
    assert all(instance_id not in json.dumps(receipt) for instance_id in ids)


def test_reconcile_accepts_eight_cleanup_receipts_without_delete(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "signal"
    attempts = source / "attempts"
    attempts.mkdir(parents=True)
    ids = []
    for number in range(8):
        attempt = attempts / str(number)
        attempt.mkdir()
        instance_id = f"instance-opaque-{number}"
        ids.append(instance_id)
        (attempt / "binding.json").write_text(
            json.dumps(
                {
                    "task": {"version_id": cleanup.TASK_VERSION_ID},
                    "verifier": {"version_id": cleanup.VERIFIER_VERSION_ID},
                    "model": {
                        "revision": cleanup.MODEL_REVISION,
                        "session_model": cleanup.SERVED_MODEL,
                    },
                }
            )
        )
        (attempt / "runtime-binding.json").write_text(
            json.dumps(
                {
                    "instance_id": instance_id,
                    "env_key": "env",
                    "environment_version": "v1",
                    "data_key": "data",
                    "data_version": "v1",
                }
            )
        )
        (attempt / "cleanup.json").write_text(
            json.dumps(
                {
                    "instance_created": True,
                    "instance_closed": True,
                    "containers_removed": True,
                }
            )
        )

    calls = []

    def request(instance_id: str, method: str):
        calls.append((instance_id, method))
        assert method == "GET"
        return 404, None

    monkeypatch.setattr(cleanup, "_request", request)
    monkeypatch.setattr(cleanup, "RECEIPT", source / "LEAK_RECONCILED.json")
    receipt = cleanup.reconcile(source)
    assert receipt["preexisting_cleanup_receipt_count"] == 8
    assert receipt["exact_delete_attempted"] is False
    assert len(calls) == 16
    assert all(instance_id not in json.dumps(receipt) for instance_id in ids)


def test_failure_summary_preserves_only_sanitized_counts(tmp_path: Path) -> None:
    source = tmp_path / "signal"
    source.mkdir()
    (source / "TERMINAL-base-v1.json").write_text(
        json.dumps(
            {
                "schema_version": "fleet-rollout-ledger-controller-terminal-v1",
                "accepted": False,
                "results": [
                    {"accepted": False, "failure_code": "model_trace_created.runtimeerror"},
                    {"accepted": False, "failure_code": "model_trace_created.runtimeerror"},
                ],
                "scores_included": False,
                "prompts_or_traces_included": False,
            }
        )
    )
    (source / "EVAL_TERMINAL.json").write_text(
        json.dumps(
            {
                "schema": "fleet_eval_campaign_terminal_v1",
                "summary": {"by_state": {"retry_review": 8, "accepted": 0}},
            }
        )
    )
    assert cleanup._failure_summary(source) == {
        "worker_terminal_present": True,
        "worker_failure_code_counts": {"model_trace_created.runtimeerror": 2},
        "campaign_terminal_present": True,
        "campaign_state_counts": {"accepted": 0, "retry_review": 8},
    }
