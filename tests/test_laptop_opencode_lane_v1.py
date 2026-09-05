from __future__ import annotations

import base64
import copy
import json
import subprocess
from pathlib import Path

import pytest

from evals.fleet import laptop_opencode_lane_v1 as lane
from evals.fleet import laptop_secret_launcher_v1 as secret_launcher

ROOT = Path(__file__).resolve().parents[1]
TERMINAL_RECEIPT = ROOT / (
    "docs/evidence/qwen38-study/"
    "2026-09-05-qwen38-laptop-rank3-tool-parity-terminal-v1.json"
)
GLOBAL_COORDINATION_HOLD = ROOT / (
    "docs/evidence/qwen38-study/"
    "2026-09-05-qwen38-laptop-rank3-held-global-coordination-v1.json"
)


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
        "global_claim_root": lane.GLOBAL_CLAIM_ROOT,
        "endpoint_lease": {
            "lease_root": lane.HOSTED_LEASE_ROOT,
            "endpoint_key": lane.HOSTED_ENDPOINT_KEY,
            "maximum_streams": 2,
        },
    }
    assert len(plan["required_before_first_scored_create"]) == 10
    assert plan["required_before_first_scored_create"][-2:] == [
        "hosted_stream_is_not_third_stream_above_qualified_cap_two",
        "attempt_one_is_accepted_before_attempts_two_through_four",
    ]
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


def test_global_coordination_hold_is_digest_valid_and_non_scored() -> None:
    receipt = json.loads(GLOBAL_COORDINATION_HOLD.read_text())
    assert receipt["status"] == "HELD_NON_SCORED"
    assert receipt["decision"] == "DO_NOT_SCORE"
    assert receipt["held_partition"] == {
        "attempts": [1, 2, 3, 4],
        "model": "qwen3.8-27b",
        "selection_rank": 3,
        "start_only_attempt_one_after_release": True,
        "whole_task": True,
    }
    assert receipt["blockers"]["canonical_sfs_claim_root_mounted"] is False
    assert receipt["blockers"]["safe_atomic_global_claim_available"] is False
    assert receipt["request_summary"]["scored_sessions_created"] == 0
    assert receipt["receipt_sha256"] == lane.crypto.digest_without(
        receipt, "receipt_sha256"
    )


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


def test_secret_launcher_keeps_key_out_of_argv_output_and_receipt(tmp_path: Path) -> None:
    secret = b"inert-test-secret-value"
    encoded = base64.b64encode(secret)
    calls: list[tuple[list[str], dict]] = []
    out = tmp_path / "qualified.json"

    def runner(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append((list(argv), dict(kwargs)))
        if argv[0] == "kubectl":
            return subprocess.CompletedProcess(argv, 0, encoded, b"")
        env = kwargs["env"]
        assert isinstance(env, dict)
        assert env["FLEET_API_KEY"] == secret.decode()
        assert secret.decode() not in "\x00".join(argv)
        out.write_text(
            json.dumps({"status": "QUALIFIED_NON_SCORED", "receipt_sha256": "sha256:x"})
        )
        return subprocess.CompletedProcess(argv, 0, b"sanitized", b"")

    result = secret_launcher.launch(out, runner=runner)
    rendered = json.dumps(result).encode()
    assert secret not in rendered and encoded not in rendered
    assert result["credential_in_argv"] is False
    assert len(calls) == 2


def test_secret_launcher_runs_only_allowlisted_actual_qwen_parity(tmp_path: Path) -> None:
    secret = b"inert-test-secret-value"
    encoded = base64.b64encode(secret)
    out = tmp_path / "parity.json"
    child_argv: list[str] = []

    def runner(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        if argv[0] == "kubectl":
            return subprocess.CompletedProcess(argv, 0, encoded, b"")
        child_argv.extend(argv)
        env = kwargs["env"]
        assert isinstance(env, dict)
        assert env["FLEET_API_KEY"] == secret.decode()
        out.write_text(
            json.dumps({"status": "PASSED_NON_SCORED", "receipt_sha256": "sha256:x"})
        )
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    result = secret_launcher.launch(
        out, action="qwen-hosted-actual-parity", runner=runner
    )
    assert child_argv == [
        secret_launcher.sys.executable,
        "-m",
        secret_launcher.ACTUAL_PARITY_MODULE,
        "--model",
        "qwen3.8-27b",
        "--out",
        str(out),
    ]
    assert result["status"] == "PASSED_NON_SCORED"
    assert result["action"] == "qwen-hosted-actual-parity"


def test_secret_launcher_rejects_unlisted_action_without_reading_secret(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        secret_launcher.SecretLaunchError, match="secret_launcher_action_not_allowed"
    ):
        secret_launcher.launch(tmp_path / "unused.json", action="arbitrary-module")


def test_secret_launcher_fails_before_child_when_key_absent(tmp_path: Path) -> None:
    calls = 0

    def runner(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    with pytest.raises(secret_launcher.SecretLaunchError, match="cluster_secret_key_absent"):
        secret_launcher.launch(tmp_path / "qualified.json", runner=runner)
    assert calls == 1


@pytest.mark.parametrize("leak_target", ["stdout", "stderr", "receipt"])
def test_secret_launcher_rejects_any_child_leak(tmp_path: Path, leak_target: str) -> None:
    secret = b"inert-sensitive-value"
    encoded = base64.b64encode(secret)
    out = tmp_path / "qualified.json"

    def runner(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        if argv[0] == "kubectl":
            return subprocess.CompletedProcess(argv, 0, encoded, b"")
        receipt = {"status": "QUALIFIED_NON_SCORED", "receipt_sha256": "sha256:x"}
        raw = json.dumps(receipt).encode()
        if leak_target == "receipt":
            raw += secret
        out.write_bytes(raw)
        stdout = secret if leak_target == "stdout" else b""
        stderr = secret if leak_target == "stderr" else b""
        return subprocess.CompletedProcess(argv, 0, stdout, stderr)

    with pytest.raises(secret_launcher.SecretLaunchError, match="credential_leaked"):
        secret_launcher.launch(out, runner=runner)


def test_terminal_tool_parity_failure_receipt_is_digest_valid_and_non_scored() -> None:
    receipt = json.loads(TERMINAL_RECEIPT.read_text())
    assert receipt["schema_version"] == (
        "fleet-qwen38-laptop-rank3-tool-parity-terminal-v1"
    )
    assert receipt["status"] == "FAILED"
    assert receipt["classification"] == "NON_SCORED_PARITY_BLOCKER"
    assert receipt["final_probe"]["exact_model_identity"]["server_info_exact"] is True
    assert receipt["final_probe"]["result_class"] == (
        "STRUCTURED_SUBMIT_REPORT_CONTRACT_FAILED"
    )
    assert receipt["request_summary"] == {
        "benchmark_requests": 0,
        "content_free_chat_completion_calls": 5,
        "model_started_scored_cells": 0,
        "task_instance_session_verifier_or_scoring_mutations": 0,
    }
    assert receipt["held_partition"]["scored_launch_authorized"] is False
    assert receipt["privacy"] == {
        "credentials_included": False,
        "model_outputs_included": False,
        "prompts_traces_flags_or_scores_included": False,
        "response_content_included": False,
        "tool_arguments_included": False,
    }
    assert receipt["receipt_sha256"] == lane.crypto.digest_without(
        receipt, "receipt_sha256"
    )
