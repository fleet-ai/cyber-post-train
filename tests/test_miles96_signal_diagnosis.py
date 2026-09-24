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
    terminal = {**terminal_body, "sha256": diagnosis.digest(terminal_body, prefix=False)}
    terminal["receipt_sha256"] = diagnosis.digest(terminal)
    _write(source / "EVAL_TERMINAL.json", terminal)
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


def _process_artifacts(source: Path) -> None:
    traces = [
        ([{"type": "step_finish", "part": {"reason": "stop"}}], 1),
        ([{"type": "step_finish", "part": {"reason": "length"}}], 1),
        ([{"type": "error"}, {"type": "step_finish", "part": {"reason": "stop"}}], 1),
        ([{"type": "step_start"}], 1),
        (
            [
                {"type": "step_finish", "part": {"reason": "stop"}},
                {"type": "step_start"},
            ],
            1,
        ),
        ([{"type": "step_finish", "part": {"reason": "stop"}}], 137),
        ([{"type": "step_finish", "part": {"reason": "stop"}}, "malformed"], 1),
        ([], 1),
    ]
    for index, (events, exit_code) in enumerate(traces):
        attempt = source / "attempts" / str(index)
        result_path = attempt / "result.json"
        result = json.loads(result_path.read_text())
        result.update(agent_exit_code=exit_code, agent_termination="process_error")
        _write(result_path, result)
        _write(
            attempt / "agent-process.json",
            {"harness": "opencode", "exit_code": exit_code, "timed_out": False},
        )
        agent_output = attempt / "agent-output"
        agent_output.mkdir()
        trace = agent_output / "opencode-stream.jsonl"
        with trace.open("w") as stream:
            for event in events:
                if isinstance(event, str):
                    stream.write(event + "\n")
                else:
                    stream.write(json.dumps(event, sort_keys=True) + "\n")
        event_count = sum(isinstance(event, dict) for event in events)
        malformed = len(events) - event_count
        _write(
            attempt / "trace-manifest.json",
            {
                "canonical_trace": "agent-output/opencode-stream.jsonl",
                "event_count": event_count,
                "malformed_line_count": malformed,
                "agent_termination": "process_error",
            },
        )
        _write(attempt / "session-ingest.json", {"status": "completed"})


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


def test_diagnosis_rejects_either_tampered_terminal_digest(tmp_path: Path, monkeypatch) -> None:
    source, release_sha256 = _output(tmp_path)
    monkeypatch.setattr(diagnosis, "RECEIPT", source / "SIGNAL_DIAGNOSIS.json")
    monkeypatch.setattr(diagnosis, "RELEASE_SHA256", release_sha256)
    terminal_path = source / "EVAL_TERMINAL.json"
    terminal = json.loads(terminal_path.read_text())
    terminal["sha256"] = "0" * 64
    _write(terminal_path, terminal)
    try:
        diagnosis.diagnose(source)
    except ValueError as exc:
        assert str(exc) == "evaluation terminal differs"
    else:
        raise AssertionError("plain terminal digest tampering was accepted")

    terminal = json.loads(terminal_path.read_text())
    terminal["sha256"] = diagnosis.digest(
        {key: value for key, value in terminal.items() if key not in {"sha256", "receipt_sha256"}},
        prefix=False,
    )
    terminal["receipt_sha256"] = "sha256:" + "0" * 64
    _write(terminal_path, terminal)
    try:
        diagnosis.diagnose(source)
    except ValueError as exc:
        assert str(exc) == "evaluation terminal differs"
    else:
        raise AssertionError("outer terminal receipt tampering was accepted")


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


def test_predicate_probe_emits_only_aggregate_counts(tmp_path: Path, monkeypatch) -> None:
    source, release_sha256 = _output(tmp_path)
    monkeypatch.setattr(diagnosis, "PREDICATE_RECEIPT", source / "SIGNAL_QUALIFICATION_PROBE.json")
    monkeypatch.setattr(diagnosis, "RELEASE_SHA256", release_sha256)
    for reward_path in (source / "attempts").glob("*/reward-result.json"):
        reward = json.loads(reward_path.read_text())
        reward.pop("direct_authority_attestation")
        _write(reward_path, reward)
    result = json.loads((source / "attempts/0/result.json").read_text())
    result["agent_termination"] = "output_limit"
    _write(source / "attempts/0/result.json", result)
    receipt = diagnosis.predicate_probe(source)
    assert receipt["complete_contract_count"] == 7
    assert receipt["failed_predicate_counts"]["completed_termination"] == 1
    assert sum(receipt["failed_predicate_counts"].values()) == 1
    assert receipt["termination_category_counts"] == {
        "completed": 7,
        "execution_timeout": 0,
        "process_error": 0,
        "malformed_trace": 0,
        "harness_error": 0,
        "missing_terminal_step": 0,
        "incomplete_terminal_step": 0,
        "output_limit": 1,
        "other": 0,
    }
    assert receipt["finite_reward_count"] == 8
    assert receipt["distinct_reward_count"] == 2
    assert receipt["reward_multiplicities"] == [4, 4]
    assert receipt["reward_variation"] is True
    assert receipt["identities_or_values_included"] is False
    assert receipt["prompts_traces_flags_answers_rewards_or_scores_included"] is False
    assert "execution-" not in json.dumps(receipt)
    assert "direct_authority_attestation" not in json.dumps(receipt)


def test_predicate_packet_has_silent_zero_gpu_root_job() -> None:
    value = diagnosis.predicate_packet()
    job = value["bundle"]["items"][1]
    assert value["sha256"] == diagnosis.digest(
        {key: item for key, item in value.items() if key != "sha256"}
    )
    assert job["metadata"]["annotations"]["fleet.ai/failure-alerts"] == "off"
    assert job["spec"]["template"]["metadata"]["annotations"]["fleet.ai/failure-alerts"] == ("off")
    assert job["spec"]["backoffLimit"] == 0
    assert job["spec"]["suspend"] is True
    assert job["spec"]["template"]["spec"]["priorityClassName"] == "c1"
    assert job["spec"]["template"]["spec"]["containers"][0]["command"][-1] == ("--predicate-probe")
    assert "nvidia.com/gpu" not in json.dumps(job)


def test_process_probe_emits_only_aggregate_categories(tmp_path: Path, monkeypatch) -> None:
    source, release_sha256 = _output(tmp_path)
    _process_artifacts(source)
    monkeypatch.setattr(diagnosis, "PROCESS_RECEIPT", source / "SIGNAL_PROCESS_CLASSIFICATION.json")
    monkeypatch.setattr(diagnosis, "RELEASE_SHA256", release_sha256)
    receipt = diagnosis.process_probe(source)
    assert receipt["process_error_count"] == 8
    assert receipt["exit_category_counts"] == {"generic_failure": 7, "sigkill_or_oom": 1}
    assert receipt["structural_trace_category_counts"] == {
        "completed": 2,
        "empty_trace": 1,
        "harness_error": 1,
        "incomplete_terminal_step": 1,
        "malformed_trace": 1,
        "missing_terminal_step": 1,
        "output_limit": 1,
    }
    assert receipt["repair_category_counts"] == {
        "malformed_trace": 1,
        "nonzero_exit_after_gradeable_trace": 2,
        "signal_or_resource_exit": 1,
        "startup_or_cli_failure": 1,
        "structured_harness_error": 1,
        "unfinished_trace": 2,
    }
    assert receipt["session_ingest_category_counts"] == {"completed": 8}
    assert receipt["result_process_exit_binding_count"] == 8
    assert receipt["trace_manifest_binding_count"] == 8
    assert receipt["scoring_result_present_count"] == 8
    assert receipt["exact_cleanup_count"] == 8
    assert receipt["stderr_or_log_text_read"] is False
    assert receipt["identifiers_reward_values_or_private_content_included"] is False
    encoded = json.dumps(receipt)
    assert all(value not in encoded for value in ("execution-0", "cell-0", "stop", "length"))


def test_process_packet_has_silent_zero_gpu_root_job() -> None:
    value = diagnosis.process_packet()
    job = value["bundle"]["items"][1]
    assert value["sha256"] == diagnosis.digest(
        {key: item for key, item in value.items() if key != "sha256"}
    )
    assert job["metadata"]["annotations"]["fleet.ai/failure-alerts"] == "off"
    assert job["spec"]["template"]["metadata"]["annotations"]["fleet.ai/failure-alerts"] == ("off")
    assert job["spec"]["backoffLimit"] == 0
    assert job["spec"]["suspend"] is True
    assert job["spec"]["template"]["spec"]["priorityClassName"] == "c1"
    assert job["spec"]["template"]["spec"]["containers"][0]["command"][-1] == ("--process-probe")
    assert "nvidia.com/gpu" not in json.dumps(job)
