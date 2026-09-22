"""Fail-closed regressions for the completion-budget runtime overlay."""

from __future__ import annotations

import json

import pytest

from cyber_post_train.jobs import digest
from evals.fleet import collection_completion_budget as budget
from evals.fleet import collection_fixed_proxy_v2 as proxy


def _receipt(path, maximum: int = 600) -> None:
    proxy._write_budget_receipt(  # noqa: SLF001
        path,
        {
            "schema": proxy.BUDGET_RECEIPT_SCHEMA,
            "reason": "exact_completion_budget_exhausted",
            "maximum_total_requests": maximum + budget.TOTAL_REQUEST_HEADROOM,
            "completed_total_requests": maximum + 1,
            "maximum_completions": maximum,
            "completed_completions": maximum,
            "rejected_completion_number": maximum + 1,
            "observed_request_number": maximum + 2,
            "prompts_responses_scores_or_credentials_included": False,
        },
    )


def test_only_exact_budget_receipt_reclassifies_process_error(tmp_path) -> None:
    out_dir = tmp_path / "attempt"
    receipt = budget.receipt_path(out_dir)
    receipt.parent.mkdir(parents=True)
    _receipt(receipt)
    budget._STATE.out_dir = out_dir  # noqa: SLF001
    budget._STATE.maximum_completions = 600  # noqa: SLF001
    try:
        assert (
            budget._budget_termination(  # noqa: SLF001
                [], malformed_lines=0, exit_code=1, timed_out=False
            )
            == "output_limit"
        )
        value = json.loads(receipt.read_text())
        value["completed_completions"] = 599
        receipt.write_text(json.dumps(value))
        assert (
            budget._budget_termination(  # noqa: SLF001
                [], malformed_lines=0, exit_code=1, timed_out=False
            )
            == "process_error"
        )
    finally:
        del budget._STATE.out_dir  # noqa: SLF001
        del budget._STATE.maximum_completions  # noqa: SLF001


def test_receipt_rejects_a_collapsed_total_request_counter(tmp_path) -> None:
    receipt = tmp_path / "receipt.json"
    _receipt(receipt)
    value = json.loads(receipt.read_text())
    value["maximum_total_requests"] = value["maximum_completions"]
    value["sha256"] = "sha256:" + digest(
        {key: item for key, item in value.items() if key != "sha256"}
    )
    receipt.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="does not prove exact exhaustion"):
        budget.validate_receipt(receipt, maximum_completions=600)


def test_budget_output_limit_gets_a_specific_retry_review_exception(tmp_path) -> None:
    out_dir = tmp_path / "attempt"
    receipt = budget.receipt_path(out_dir)
    receipt.parent.mkdir(parents=True)
    _receipt(receipt, maximum=2)
    with pytest.raises(budget.CompletionBudgetOutputLimit):
        budget._budget_accepted(  # noqa: SLF001
            object(),
            {"harness": {"max_model_requests": 2}},
            {},
            {"agent_termination": "output_limit"},
            out_dir,
        )


def test_native_output_limit_without_budget_receipt_keeps_base_classification(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    expected = {"base": "classification"}
    monkeypatch.setattr(budget, "_BASE_ACCEPTED", lambda *_args, **_kwargs: expected)
    result = budget._budget_accepted(  # noqa: SLF001
        object(),
        {"harness": {"max_model_requests": 600}},
        {},
        {"agent_termination": "output_limit"},
        tmp_path,
    )
    assert result is expected


def test_model_proxy_gets_distinct_total_and_completion_limits(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    captured: dict = {}

    def docker(*arguments, **kwargs):
        captured["arguments"] = arguments
        captured["environment"] = kwargs["env"]
        return object()

    out_dir = tmp_path / "attempt"
    out_dir.mkdir()
    monkeypatch.setattr(budget, "_BASE_DOCKER", docker)
    budget._STATE.out_dir = out_dir  # noqa: SLF001
    budget._STATE.maximum_completions = 600  # noqa: SLF001
    try:
        budget._budget_docker(  # noqa: SLF001
            "run",
            "-d",
            "--name",
            "model-proxy",
            "-e",
            "FIXED_MAX_REQUESTS",
            "-v",
            "/source/proxy.py:/proxy.py:ro",
            "proxy-image",
            "python",
            "/proxy.py",
            env={
                "FIXED_ALLOWED_PATHS": "/v1/chat/completions,/v1/models",
                "FIXED_MAX_REQUESTS": "600",
            },
        )
    finally:
        del budget._STATE.out_dir  # noqa: SLF001
        del budget._STATE.maximum_completions  # noqa: SLF001

    assert captured["environment"]["FIXED_MAX_REQUESTS"] == "630"
    assert captured["environment"]["FIXED_MAX_COMPLETIONS"] == "600"
    assert captured["environment"]["FIXED_BUDGET_EVIDENCE_PATH"] == (
        "/budget/COMPLETION_BUDGET_EXHAUSTED.json"
    )
    arguments = captured["arguments"]
    assert "FIXED_MAX_COMPLETIONS" in arguments
    assert "FIXED_BUDGET_EVIDENCE_PATH" in arguments
    assert f"{(out_dir / budget.BUDGET_DIRECTORY).resolve()}:/budget" in arguments


def test_total_budget_drift_fails_before_proxy_start(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        budget,
        "_BASE_DOCKER",
        lambda *_args, **_kwargs: pytest.fail("drift reached Docker"),
    )
    out_dir = tmp_path / "attempt"
    out_dir.mkdir()
    budget._STATE.out_dir = out_dir  # noqa: SLF001
    budget._STATE.maximum_completions = 600  # noqa: SLF001
    try:
        with pytest.raises(RuntimeError, match="ceiling changed"):
            budget._budget_docker(  # noqa: SLF001
                "run",
                "proxy-image",
                "python",
                "/proxy.py",
                env={
                    "FIXED_ALLOWED_PATHS": "/v1/chat/completions,/v1/models",
                    "FIXED_MAX_REQUESTS": "599",
                },
            )
    finally:
        del budget._STATE.out_dir  # noqa: SLF001
        del budget._STATE.maximum_completions  # noqa: SLF001
