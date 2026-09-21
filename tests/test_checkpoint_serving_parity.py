from __future__ import annotations

import pytest

import training.checkpoint_serving_parity as serving_parity
from training.checkpoint_serving_parity import (
    capture_base_route,
    catalog_projection,
    model_projection,
    probe_projection,
    server_projection,
)
from training.checkpoint_serving_route import RouteError


def test_projections_accept_exact_ready_route() -> None:
    assert (
        catalog_projection(
            {
                "data": [
                    {
                        "id": "candidate",
                        "model_revision": "sha256:" + "a" * 64,
                        "status": "ready",
                        "routed": True,
                        "ready_replicas": 1,
                        "engine": "sglang",
                        "precision": "bf16",
                        "tensor_parallel_size": 1,
                        "data_parallel_size": 8,
                        "data_parallel_attention": False,
                        "capabilities": ["chat_completions", "tool_calling"],
                    }
                ]
            },
            "candidate",
            "sha256:" + "a" * 64,
        )["routed"]
        is True
    )
    assert (
        model_projection(
            {
                "model_path": "/scratch/candidate",
                "tokenizer_path": "/scratch/candidate",
                "served_model_name": "candidate",
                "architectures": ["Qwen"],
                "model_type": "qwen",
                "load_format": "auto",
                "reasoning_parser": "qwen3",
                "tool_call_parser": "qwen3_coder",
                "weight_version": "default",
            },
            "/scratch/candidate",
            "candidate",
        )["weight_version"]
        == "default"
    )
    assert (
        server_projection(
            {
                "model_path": "/scratch/candidate",
                "served_model_name": "candidate",
                "context_length": 262144,
                "tp_size": 1,
                "dp_size": 8,
                "quantization": None,
            },
            "/scratch/candidate",
            "candidate",
        )["context_length"]
        == 262144
    )


def test_probe_projection_keeps_no_generated_text() -> None:
    tool = {
        "model": "candidate",
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "identity",
                                "arguments": '{"value":"parity"}',
                            }
                        }
                    ]
                }
            }
        ],
    }
    logits = {
        "model": "candidate",
        "choices": [{"logprobs": {"content": [{"token": "x", "bytes": [120], "logprob": -0.25}]}}],
    }
    projected = probe_projection(tool, logits, logits, "candidate")
    assert projected["tool_call_passed"] is True
    assert projected["logit_token_identity_deterministic"] is True
    assert projected["logit_values_exactly_equal"] is True
    assert projected["max_abs_logprob_delta"] == 0
    assert "choices" not in projected


def test_probe_projection_accepts_replica_score_jitter_with_stable_tokens() -> None:
    tool = {
        "model": "candidate",
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "identity",
                                "arguments": '{"value":"parity"}',
                            }
                        }
                    ]
                }
            }
        ],
    }
    first = {
        "model": "candidate",
        "choices": [{"logprobs": {"content": [{"token": "x", "bytes": [120], "logprob": -0.25}]}}],
    }
    second = {
        "model": "candidate",
        "choices": [{"logprobs": {"content": [{"token": "x", "bytes": [120], "logprob": -0.30}]}}],
    }
    projected = probe_projection(tool, first, second, "candidate")
    assert projected["logit_token_identity_deterministic"] is True
    assert projected["logit_values_exactly_equal"] is False
    assert projected["max_abs_logprob_delta"] == pytest.approx(0.05)


def test_probe_projection_rejects_unstable_token_identity() -> None:
    tool = {
        "model": "candidate",
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "identity",
                                "arguments": '{"value":"parity"}',
                            }
                        }
                    ]
                }
            }
        ],
    }
    first = {
        "model": "candidate",
        "choices": [{"logprobs": {"content": [{"token": "x", "bytes": [120], "logprob": -0.25}]}}],
    }
    second = {
        "model": "candidate",
        "choices": [{"logprobs": {"content": [{"token": "y", "bytes": [121], "logprob": -0.25}]}}],
    }
    with pytest.raises(RouteError, match="fixed_logit_token_identity_not_deterministic"):
        probe_projection(tool, first, second, "candidate")


def test_capture_base_route_is_a_standalone_content_free_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = {
        "served_model": "qwen3.8-27b",
        "model_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "source_path": "/models/qwen3.8-27b/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "probes": {"tool_call_passed": True},
    }
    calls: list[dict[str, object]] = []

    def fake_capture_route(**kwargs: object) -> dict[str, object]:
        calls.append(dict(kwargs))
        return observed

    monkeypatch.setattr(serving_parity, "_capture_route", fake_capture_route)
    receipt = capture_base_route(
        "test-key",
        "test-context",
        "qwen3.8-27b",
        observed["model_revision"],
        observed["source_path"],
    )

    assert receipt["schema"] == "cyber_base_serving_live_route_proof_v1"
    assert receipt["status"] == "passed"
    assert receipt["route"] == observed
    assert receipt["benchmark_content_included"] is False
    assert receipt["response_content_recorded"] is False
    assert receipt["scores_observed"] is False
    assert receipt["task_content_included"] is False
    assert receipt["external_mutations_performed"] == 0
    assert isinstance(receipt["receipt_sha256"], str)
    assert len(calls) == 1
    assert calls[0]["key"] == "test-key"
    assert calls[0]["context"] == "test-context"
    assert calls[0]["label"] == "base"
    assert calls[0]["model"] == "qwen3.8-27b"
    assert calls[0]["expected_revision"] == observed["model_revision"]
    assert calls[0]["expected_source_path"] == observed["source_path"]
    assert calls[0]["expected_replicas"] is None
