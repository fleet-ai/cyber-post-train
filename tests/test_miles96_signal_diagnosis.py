import json
from pathlib import Path

from training import miles96_signal_diagnosis as diagnosis


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _output(root: Path) -> tuple[Path, str]:
    source = root / "signal"
    attempts = source / "attempts"
    attempts.mkdir(parents=True)
    release_body = {
        "source_job_uid": diagnosis.SOURCE_JOB_UID,
        "all_instances_released_after": True,
        "live_instance_count_after": 0,
    }
    release = {**release_body, "sha256": diagnosis.digest(release_body)}
    _write(source / "LEAK_RECONCILED.json", release)
    results = [
        {
            "serving_block": "base",
            "claimed": True,
            "accepted": False,
            "ledger_cell_id": f"cell-{index}",
            "failure_code": "model_trace_created.runtimeerror",
        }
        for index in range(8)
    ]
    terminal_body = {
        "schema": "fleet_eval_campaign_terminal_v1",
        "plan_sha256": diagnosis.PLAN_SHA256,
        "routes": {"base": {"results": results}},
        "summary": {"total": 8, "by_state": {"retry_review": 8}},
    }
    _write(
        source / "EVAL_TERMINAL.json",
        {**terminal_body, "sha256": diagnosis.digest(terminal_body, prefix=False)},
    )
    for index in range(8):
        attempt = attempts / str(index)
        attempt.mkdir()
        score = float(index % 2)
        execution_id = f"execution-{index}"
        _write(
            attempt / "binding.json",
            {
                "task": {"version_id": diagnosis.TASK_VERSION_ID},
                "verifier": {"version_id": diagnosis.VERIFIER_VERSION_ID},
                "model": {
                    "revision": diagnosis.MODEL_REVISION,
                    "session_model": diagnosis.SERVED_MODEL,
                },
            },
        )
        _write(
            attempt / "result.json",
            {
                "task_version_id": diagnosis.TASK_VERSION_ID,
                "verifier_execution_id": execution_id,
                "score": score,
                "agent_termination": "completed",
            },
        )
        _write(
            attempt / "reward-result.json",
            {
                "reward": score,
                "verifier_execution_id": execution_id,
                "direct_authority_attestation": {
                    "context": {
                        "task_version_id": diagnosis.TASK_VERSION_ID,
                        "verifier_version_id": diagnosis.VERIFIER_VERSION_ID,
                    }
                },
            },
        )
        _write(
            attempt / "cleanup.json",
            {"instance_created": True, "instance_closed": True, "containers_removed": True},
        )
    return source, release["sha256"]


def test_diagnosis_reports_variation_without_values(tmp_path: Path, monkeypatch) -> None:
    source, release_sha256 = _output(tmp_path)
    monkeypatch.setattr(diagnosis, "RECEIPT", source / "SIGNAL_DIAGNOSIS.json")
    monkeypatch.setattr(diagnosis, "RELEASE_SHA256", release_sha256)
    receipt = diagnosis.diagnose(source)
    assert receipt["finite_reward_count"] == 8
    assert receipt["distinct_reward_count"] == 2
    assert receipt["reward_multiplicities"] == [4, 4]
    assert receipt["reward_variation"] is True
    assert receipt["failure_code_counts"] == {"model_trace_created.runtimeerror": 8}
    assert receipt["all_instances_released"] is True
    assert "rewards" not in receipt
    assert all(value not in json.dumps(receipt) for value in ("execution-0", "cell-0"))


def test_packet_has_silent_zero_gpu_root_job() -> None:
    value = diagnosis.packet()
    job = value["bundle"]["items"][1]
    assert value["sha256"] == diagnosis.digest(
        {key: item for key, item in value.items() if key != "sha256"}
    )
    assert job["metadata"]["annotations"]["fleet.ai/failure-alerts"] == "off"
    assert job["spec"]["backoffLimit"] == 0
    assert job["spec"]["suspend"] is True
    assert job["spec"]["template"]["spec"]["priorityClassName"] == "c1"
    assert "nvidia.com/gpu" not in json.dumps(job)


def test_header_probe_emits_only_identity_booleans(tmp_path: Path) -> None:
    source, _release_sha256 = _output(tmp_path)
    receipt = diagnosis.header_probe(source)
    assert receipt["schema_matches"] is True
    assert receipt["self_digest_matches"] is True
    assert receipt["plan_matches"] is True
    assert set(receipt) == {
        "plan_matches",
        "reward_or_trace_content_read",
        "schema",
        "schema_matches",
        "self_digest_matches",
        "sha256",
        "source_job_uid",
        "terminal_fields_included",
        "terminal_file_sha256",
    }


def test_header_packet_has_silent_zero_gpu_root_job() -> None:
    value = diagnosis.header_packet()
    job = value["bundle"]["items"][1]
    assert value["sha256"] == diagnosis.digest(
        {key: item for key, item in value.items() if key != "sha256"}
    )
    assert job["metadata"]["annotations"]["fleet.ai/failure-alerts"] == "off"
    assert job["spec"]["backoffLimit"] == 0
    assert job["spec"]["suspend"] is True
    assert job["spec"]["template"]["spec"]["priorityClassName"] == "c1"
    assert "nvidia.com/gpu" not in json.dumps(job)
