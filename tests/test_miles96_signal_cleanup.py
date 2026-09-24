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
        instance_id = f"00000000-0000-0000-0000-{number:012d}"
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
