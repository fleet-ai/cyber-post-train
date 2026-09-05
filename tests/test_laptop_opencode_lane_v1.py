from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from evals.fleet import laptop_opencode_lane_v1 as lane

ROOT = Path(__file__).resolve().parents[1]


def _runner(argv: list[str] | tuple[str, ...]) -> str:
    if argv[:2] == ["docker", "version"]:
        return json.dumps(
            {
                "Client": {"Os": "darwin", "Arch": "arm64"},
                "Server": {"Os": "linux", "Arch": "arm64"},
            }
        )
    if argv[:3] == ["docker", "image", "inspect"]:
        return json.dumps(
            {
                "Id": lane.IMAGE_ID,
                "Os": "linux",
                "Architecture": "amd64",
                "Config": {"User": "node", "WorkingDir": "/workspace"},
            }
        )
    if argv[-2:] == ["opencode", "--version"]:
        return "1.18.27"
    if argv[-2:] == ["-c", "sha256sum /usr/local/bin/opencode"]:
        return f"{lane.OPENCODE_BINARY_SHA256}  /usr/local/bin/opencode"
    raise AssertionError(argv)


def test_held_plan_reserves_one_complete_sequential_task() -> None:
    plan = lane.held_plan(ROOT)
    assert plan["selection_rank"] == 3
    assert [row["attempt"] for row in plan["cells"]] == [1, 2, 3, 4]
    assert [row["cell_id"] for row in plan["cells"]] == lane.EXPECTED_CELLS
    assert plan["execution"] == {
        "scored_launch_authorized": False,
        "whole_task_partition": True,
        "attempt_order": [1, 2, 3, 4],
        "maximum_concurrent_attempts": 1,
        "automatic_retry": False,
    }
    assert len(plan["required_before_first_scored_create"]) == 5
    assert plan["glm_hosted_laptop_lane"]["authorized"] is False


def test_local_qualification_is_network_inert_without_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FLEET_API_KEY", raising=False)
    calls: list[list[str]] = []

    def runner(argv: list[str] | tuple[str, ...]) -> str:
        calls.append(list(argv))
        return _runner(argv)

    receipt = lane.qualify_local(ROOT, runner=runner)
    assert receipt["status"] == "LOCAL_PASSED_NETWORK_BLOCKED"
    assert receipt["scored_launch_authorized"] is False
    assert receipt["network_qualification"] == {
        "credential_present": False,
        "fleet_team_verified": False,
        "authoritative_routes_verified_get_only": False,
        "qwen_served_id_and_structured_tools_verified": False,
        "requests_made": 0,
        "mutations_made": 0,
    }
    assert len(calls) == 4
    assert receipt["privacy"]["credentials_included"] is False


def test_credential_presence_is_only_a_boolean(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "inert-never-persist-this"
    monkeypatch.setenv("FLEET_API_KEY", secret)
    receipt = lane.qualify_local(ROOT, runner=_runner)
    assert receipt["status"] == "LOCAL_PASSED"
    assert receipt["network_qualification"]["credential_present"] is True
    assert secret not in json.dumps(receipt)
    assert receipt["scored_launch_authorized"] is False


def test_cell_drift_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lane, "EXPECTED_CELLS", ["sha256:" + "0" * 64] * 4)
    with pytest.raises(lane.QualificationError, match="reserved_cell_identity_drift"):
        lane.held_plan(ROOT)


def test_image_drift_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FLEET_API_KEY", raising=False)

    def runner(argv: list[str] | tuple[str, ...]) -> str:
        result = _runner(argv)
        if argv[:3] == ["docker", "image", "inspect"]:
            value = json.loads(result)
            value["Id"] = "sha256:" + "0" * 64
            return json.dumps(value)
        return result

    with pytest.raises(lane.QualificationError, match="local_harness_qualification_failed"):
        lane.qualify_local(ROOT, runner=runner)


def test_plan_digest_covers_launch_gates() -> None:
    original = lane.held_plan(ROOT)
    changed = copy.deepcopy(original)
    changed["required_before_first_scored_create"].pop()
    assert lane.crypto.digest_without(changed, "plan_sha256") != original["plan_sha256"]


class _NetworkClient:
    pass


def _behavior(*_args: str) -> dict:
    body = {
        "served_id": "qwen3.8-27b",
        "exact_served_id_advertised": True,
        "observations": [{"tool": "bash"}, {"tool": "submit_report"}],
        "passed": True,
    }
    return {**body, "receipt_sha256": lane.crypto.digest_without(body, "receipt_sha256")}


def test_network_qualification_is_non_scored_and_stays_held(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        lane.self_hosted,
        "_request",
        lambda *_args, **_kwargs: {
            "team_name": "fleet",
            "team_id": lane.self_hosted.FLEET_TEAM_ID,
        },
    )
    monkeypatch.setattr(
        lane.self_hosted,
        "assert_authoritative_routes_deployed",
        lambda *_args: {
            "mode": "behavioral_method_not_allowed",
            "method": "GET",
            "statuses": {"provisioning": 405, "scoring": 405},
        },
    )
    receipt = lane.qualify_network(
        ROOT, "inert-test-key", client=_NetworkClient(), model_probe=_behavior
    )
    assert receipt["status"] == "NETWORK_PASSED_LAUNCH_STILL_HELD"
    assert receipt["scored_launch_authorized"] is False
    assert receipt["request_policy"] == {
        "task_instance_session_verifier_or_scoring_mutations": 0,
        "benchmark_requests": 0,
        "generic_non_benchmark_model_requests": 2,
    }
    assert "inert-test-key" not in json.dumps(receipt)


def test_network_qualification_rejects_wrong_team(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        lane.self_hosted,
        "_request",
        lambda *_args, **_kwargs: {"team_name": "other", "team_id": "wrong"},
    )
    with pytest.raises(lane.QualificationError, match="fleet_team_identity_mismatch"):
        lane.qualify_network(
            ROOT, "inert-test-key", client=_NetworkClient(), model_probe=_behavior
        )


def test_network_qualification_rejects_incomplete_tool_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        lane.self_hosted,
        "_request",
        lambda *_args, **_kwargs: {
            "team_name": "fleet",
            "team_id": lane.self_hosted.FLEET_TEAM_ID,
        },
    )
    monkeypatch.setattr(
        lane.self_hosted, "assert_authoritative_routes_deployed", lambda *_args: {}
    )
    with pytest.raises(
        lane.QualificationError, match="qwen_hosted_structured_tool_parity_failed"
    ):
        lane.qualify_network(
            ROOT,
            "inert-test-key",
            client=_NetworkClient(),
            model_probe=lambda *_args: {"passed": False},
        )
