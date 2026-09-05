from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from evals.fleet import endpoint_lease
from evals.fleet import hosted_c3_addon_qualifier_v1 as c3


class Reader:
    def get(self, path: str) -> dict[str, Any]:
        for binding in c3.c4.EXPECTED_CONTROLLERS.values():
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


def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLEET_API_KEY", "test-only")
    monkeypatch.setenv("JOB_UID", "11111111-1111-4111-8111-111111111111")
    monkeypatch.setenv("POD_UID", "22222222-2222-4222-8222-222222222222")
    monkeypatch.setenv("EXPECTED_SECRET_UID", "e0febd8e-94a2-46b0-a0bf-dd6b3154187b")


def test_c3_addon_acquires_only_slot_three_and_runs_one_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _env(monkeypatch)
    first = endpoint_lease.acquire_endpoint_lease(
        lease_root=tmp_path, endpoint_key=c3.LEASE_KEY, maximum_streams=2
    )
    second = endpoint_lease.acquire_endpoint_lease(
        lease_root=tmp_path, endpoint_key=c3.LEASE_KEY, maximum_streams=2
    )
    calls: list[str] = []
    receipt = c3.run(
        tmp_path / "out",
        reader=Reader(),
        caller=lambda _model, tool, _key: calls.append(tool) or 0.1,
        identity_checker=lambda _key: {
            "context_length_observable": True,
            "observed_context_length": 262144,
        },
        lease_root=tmp_path,
    )
    assert receipt["status"] == "PASSED_NON_SCORED"
    assert receipt["lease"]["addon_slot_acquired"] == 3
    assert calls == ["bash", "submit_report"]
    assert receipt["request_counts"] == {
        "chat_completions": 2,
        "task_instance": 0,
        "session": 0,
        "scoring": 0,
        "verifier": 0,
    }
    first.close()
    second.close()


def test_c3_addon_rejects_missing_second_holder_before_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _env(monkeypatch)
    first = endpoint_lease.acquire_endpoint_lease(
        lease_root=tmp_path, endpoint_key=c3.LEASE_KEY, maximum_streams=2
    )
    calls: list[str] = []
    with pytest.raises(c3.C3QualificationError, match="slots_not_exactly"):
        c3.run(
            tmp_path / "out",
            reader=Reader(),
            caller=lambda *_args: calls.append("called") or 0.1,
            identity_checker=lambda _key: {},
            lease_root=tmp_path,
        )
    assert calls == []
    first.close()
