from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from evals.fleet import endpoint_lease
from evals.fleet import hosted_c4_addon_qualifier_v1 as addon


class Reader:
    def get(self, path: str) -> dict[str, Any]:
        for binding in addon.EXPECTED_CONTROLLERS.values():
            if f"/jobs/{binding['job_name']}" in path:
                return {
                    "metadata": {"name": binding["job_name"], "uid": binding["job_uid"]},
                    "status": {"active": 1},
                }
            if "pods?labelSelector=" in path and binding["job_name"] in path:
                return {
                    "items": [
                        {
                            "metadata": {
                                "uid": binding["pod_uid"],
                                "ownerReferences": [
                                    {"kind": "Job", "uid": binding["job_uid"]}
                                ],
                            },
                            "status": {
                                "phase": "Running",
                                "containerStatuses": [{"ready": True, "restartCount": 0}],
                            },
                        }
                    ]
                }
        raise AssertionError(path)


def _set_runtime_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLEET_API_KEY", "test-only")
    monkeypatch.setenv("JOB_UID", "11111111-1111-4111-8111-111111111111")
    monkeypatch.setenv("POD_UID", "22222222-2222-4222-8222-222222222222")
    monkeypatch.setenv("EXPECTED_SECRET_UID", addon.EXPECTED_SECRET_UID)


def test_addon_requires_and_uses_only_slots_three_and_four(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_runtime_env(monkeypatch)
    first = endpoint_lease.acquire_endpoint_lease(
        lease_root=tmp_path, endpoint_key=addon.LEASE_KEY, maximum_streams=2
    )
    second = endpoint_lease.acquire_endpoint_lease(
        lease_root=tmp_path, endpoint_key=addon.LEASE_KEY, maximum_streams=2
    )
    calls: list[tuple[str, str]] = []

    def caller(model: str, tool: str, _key: str) -> float:
        calls.append((model, tool))
        return 0.1

    receipt = addon.run(
        tmp_path / "out",
        reader=Reader(),
        caller=caller,
        identity_checker=lambda _key: {
            "context_length_observable": True,
            "observed_context_length": 262144,
        },
        lease_root=tmp_path,
    )
    assert receipt["status"] == "PASSED_NON_SCORED"
    assert receipt["lease"]["addon_slots_acquired"] == [3, 4]
    assert receipt["request_counts"] == {
        "chat_completions": 4,
        "task_instance": 0,
        "session": 0,
        "scoring": 0,
        "verifier": 0,
    }
    assert sorted(calls) == sorted(
        [(addon.MODEL, "bash"), (addon.MODEL, "submit_report")] * 2
    )
    first.close()
    second.close()


def test_addon_fails_before_model_when_existing_slots_are_not_both_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_runtime_env(monkeypatch)
    first = endpoint_lease.acquire_endpoint_lease(
        lease_root=tmp_path, endpoint_key=addon.LEASE_KEY, maximum_streams=2
    )
    calls: list[str] = []
    with pytest.raises(
        addon.AddonQualificationError,
        match="existing_controller_slots_not_exactly_one_and_two",
    ):
        addon.run(
            tmp_path / "out",
            reader=Reader(),
            caller=lambda *_args: calls.append("called") or 0.1,
            identity_checker=lambda _key: {
                "context_length_observable": True,
                "observed_context_length": 262144,
            },
            lease_root=tmp_path,
        )
    assert calls == []
    # The rejected add-on released its partial reservation.
    with endpoint_lease.acquire_endpoint_lease(
        lease_root=tmp_path, endpoint_key=addon.LEASE_KEY, maximum_streams=4
    ) as replacement:
        assert replacement.slot == 2
    first.close()


def test_controller_uid_or_restart_drift_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_runtime_env(monkeypatch)

    class RestartedReader(Reader):
        def get(self, path: str) -> dict[str, Any]:
            value = super().get(path)
            if "pods?labelSelector=" in path:
                value["items"][0]["status"]["containerStatuses"][0]["restartCount"] = 1
            return value

    with pytest.raises(addon.AddonQualificationError, match="controller_pod_identity"):
        addon.run(
            tmp_path / "out",
            reader=RestartedReader(),
            identity_checker=lambda _key: {},
            lease_root=tmp_path,
        )
