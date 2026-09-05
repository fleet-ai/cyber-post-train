from __future__ import annotations

from evals.fleet import hosted_behavioral_preflight as preflight
from evals.fleet import self_hosted


def test_probe_persists_only_content_free_protocol_evidence() -> None:
    def caller(model: str, tool: str, key: str) -> dict[str, object]:
        assert model == "glm-5.3"
        assert key == "secret"
        return {"http_status": 200, "latency_ms": 123, "result_class": "VALID_TOOL_STRUCTURE"}

    receipt = preflight.probe(
        "glm-5.3", "secret", caller=caller, advertised_ids={"glm-5.3"}
    )
    assert receipt["passed"] is True
    assert [row["tool"] for row in receipt["observations"]] == ["bash", "submit_report"]
    assert all(row["response_content_persisted"] is False for row in receipt["observations"])
    assert receipt["request"] == {
        "generic_non_benchmark": True,
        "max_tokens": 64,
        "prompt_content_persisted": False,
        "response_content_persisted": False,
    }
    encoded = self_hosted.canonical_json(receipt).decode()
    assert "secret" not in encoded
    assert "prompt" not in encoded.replace("prompt_content_persisted", "")
    assert receipt["receipt_sha256"] == self_hosted.digest_without(
        receipt, "receipt_sha256"
    )


def test_probe_fails_closed_on_tool_structure() -> None:
    def caller(_model: str, tool: str, _key: str) -> dict[str, object]:
        result = "VALID_TOOL_STRUCTURE" if tool == "bash" else "INVALID_TOOL_STRUCTURE"
        return {"http_status": 200, "latency_ms": 1, "result_class": result}

    receipt = preflight.probe(
        "qwen3.8-27b", "secret", caller=caller, advertised_ids={"qwen3.8-27b"}
    )
    assert receipt["passed"] is False
