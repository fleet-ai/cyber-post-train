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
        return 200, {
            "instance_id": instance_id,
            "env_key": "env",
            "version": "v1",
            "data_key": "data",
            "data_version": "v1",
            "terminated_at": None if instance_id in live else "2026-09-24T00:00:00Z",
        }

    monkeypatch.setattr(cleanup, "_request", request)
    monkeypatch.setattr(cleanup, "RECEIPT", source / "LEAK_RECONCILED.json")
    receipt = cleanup.reconcile(source)
    assert receipt["all_instances_released_after"] is True
    assert receipt["terminated_instance_count_after"] == 8
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

    (source / "TERMINAL-base-v1.json").write_text(
        json.dumps(
            {
                "worker_id": "base-v1",
                "plan_sha256": "sha256:opaque",
                "results": [{"accepted": False, "failure_code": "opaque.failure"}],
                "accepted": 0,
                "receipt_sha256": "sha256:opaque",
            }
        )
    )

    calls = []

    def request(instance_id: str, method: str):
        calls.append((instance_id, method))
        assert method == "GET"
        if instance_id == ids[-1]:
            return 404, None
        return 200, {
            "instance_id": instance_id,
            "env_key": "env",
            "version": "v1",
            "data_key": "data",
            "data_version": "v1",
            "terminated_at": "2026-09-24T00:00:00Z",
        }

    monkeypatch.setattr(cleanup, "_request", request)
    monkeypatch.setattr(cleanup, "RECEIPT", source / "LEAK_RECONCILED.json")
    receipt = cleanup.reconcile(source)
    assert receipt["preexisting_cleanup_receipt_count"] == 8
    assert receipt["exact_delete_attempted"] is False
    assert receipt["all_instances_released_after"] is True
    assert receipt["absent_instance_count_after"] == 1
    assert receipt["terminated_instance_count_after"] == 7
    assert receipt["reward_terminal_metadata_read"] is False
    assert len(calls) == 16
    assert all(instance_id not in json.dumps(receipt) for instance_id in ids)
