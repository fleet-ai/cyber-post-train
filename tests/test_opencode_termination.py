"""Synthetic terminal events only; no task text or stored rollout inspection."""

import pytest

from evals.fleet.opencode_self_hosted import load_opencode_trace, opencode_termination


def finish(reason):
    return {"type": "step_finish", "part": {"type": "step-finish", "reason": reason}}


@pytest.mark.parametrize(
    "events,expected",
    [
        ([finish("stop")], "completed"),
        ([finish("tool-calls"), finish("stop")], "completed"),
        ([finish("length"), finish("stop")], "completed"),
        (
            [{"type": "tool_use", "part": {"state": {"status": "error"}}}, finish("stop")],
            "completed",
        ),
        ([finish("stop"), finish("length")], "output_limit"),
        ([finish("tool-calls")], "incomplete_terminal_step"),
        ([finish("unknown")], "incomplete_terminal_step"),
        ([finish("stop"), {"type": "step_start"}], "incomplete_terminal_step"),
        ([finish(None)], "incomplete_terminal_step"),
        ([], "missing_terminal_step"),
        ([{"type": "step_finish", "part": None}], "missing_terminal_step"),
        ([{"type": "error", "error": {"message": "private"}}, finish("stop")], "harness_error"),
    ],
)
def test_terminal_classification(events, expected):
    result = opencode_termination(events, malformed_lines=0, exit_code=0, timed_out=False)
    assert result == expected
    assert "private" not in result


@pytest.mark.parametrize(
    "malformed,exit_code,timed_out,expected",
    [
        (1, 0, False, "malformed_trace"),
        (0, 1, False, "process_error"),
        (0, 124, True, "execution_timeout"),
    ],
)
def test_outer_failure_cannot_be_overridden_by_stop(malformed, exit_code, timed_out, expected):
    assert (
        opencode_termination(
            [finish("stop")], malformed_lines=malformed, exit_code=exit_code, timed_out=timed_out
        )
        == expected
    )


def test_non_object_json_is_malformed_not_silently_lost(tmp_path):
    path = tmp_path / "synthetic.jsonl"
    path.write_text('[]\nnull\nfalse\n42\n{"type":"step_start"}\n')
    assert load_opencode_trace(path) == ([{"type": "step_start"}], 4)
