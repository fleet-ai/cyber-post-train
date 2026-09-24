from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from training import miles96_harness_error_probe as probe


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _source(root: Path, errors: list[dict]) -> Path:
    body = {
        "process_error_count": 8,
        "exit_category_counts": {"generic_failure": 8},
        "structural_trace_category_counts": {"harness_error": 8},
        "session_ingest_category_counts": {"completed": 8},
        "scoring_result_present_count": 8,
        "exact_cleanup_count": 8,
        "all_instances_released": True,
    }
    digest = probe.digest(body)
    _write(root / "SIGNAL_PROCESS_CLASSIFICATION.json", {**body, "sha256": digest})
    for index in range(8):
        trace = root / "attempts" / str(index) / "agent-output" / "opencode-stream.jsonl"
        trace.parent.mkdir(parents=True)
        trace.write_text(
            "".join(json.dumps({"type": "error", "error": error}) + "\n" for error in errors)
        )
        _write(
            trace.parents[1] / "trace-manifest.json",
            {
                "canonical_trace": "agent-output/opencode-stream.jsonl",
                "canonical_trace_sha256": "sha256:"
                + hashlib.sha256(trace.read_bytes()).hexdigest(),
            },
        )
    return root


def test_classifies_only_allowlisted_structure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    errors = [
        {
            "name": "APIError",
            "data": {
                "message": "Unsupported parameter in private request",
                "statusCode": 400,
                "isRetryable": False,
                "responseBody": json.dumps({"error": {"code": "unsupported_parameter"}}),
            },
        }
    ]
    root = _source(tmp_path / "source", errors)
    source_body = json.loads((root / "SIGNAL_PROCESS_CLASSIFICATION.json").read_text())
    monkeypatch.setattr(probe, "SOURCE_RECEIPT_SHA256", source_body["sha256"])
    receipt = probe.classify(root, root / "SIGNAL_HARNESS_ERROR_KIND.json")

    assert receipt["error_event_count"] == 8
    assert receipt["error_name_counts"] == {"APIError": 8}
    assert receipt["api_status_counts"] == {"400": 8}
    assert receipt["retryable_counts"] == {"false": 8}
    assert receipt["response_code_counts"] == {"unsupported_parameter": 8}
    assert receipt["message_category_counts"] == {"unsupported_parameter": 8}
    assert receipt["distinct_message_sha256_count"] == 1
    encoded = json.dumps(receipt)
    assert "private request" not in encoded
    assert receipt["sha256"] == "sha256:" + probe.digest(
        {key: value for key, value in receipt.items() if key != "sha256"}
    )


def test_rejects_source_receipt_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _source(tmp_path / "source", [{"name": "UnknownError", "data": {"message": "x"}}])
    monkeypatch.setattr(probe, "SOURCE_RECEIPT_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="source classifier identity changed"):
        probe.classify(root, root / "output.json")


def test_packet_is_create_once_zero_gpu_c1_and_alert_off() -> None:
    value = probe.packet()
    config_map, job = value["bundle"]["items"]
    assert value["sha256"] == "sha256:" + probe.digest(
        {key: item for key, item in value.items() if key != "sha256"}
    )
    assert config_map["immutable"] is True
    assert config_map["data"]["probe.py"] == Path(probe.__file__).read_text()
    assert job["metadata"]["annotations"]["fleet.ai/failure-alerts"] == "off"
    assert job["metadata"]["annotations"]["cyber-post-train.fleet.ai/create-once"] == "true"
    assert job["spec"]["template"]["metadata"]["annotations"]["fleet.ai/failure-alerts"] == "off"
    assert job["spec"]["suspend"] is False
    assert job["spec"]["backoffLimit"] == 0
    assert job["spec"]["activeDeadlineSeconds"] == 300
    assert job["spec"]["template"]["spec"]["priorityClassName"] == "c1"
    assert job["spec"]["template"]["spec"]["automountServiceAccountToken"] is False
    assert "nvidia.com/gpu" not in json.dumps(job)
