from __future__ import annotations

import json
from pathlib import Path

from training.harness_parity import analyze_export, tool_schema_receipt


def test_analyzer_emits_hashes_and_counts_but_no_private_trace_content(tmp_path: Path) -> None:
    secret = "PRIVATE PROMPT AND TOOL OUTPUT"
    row = {
        "source": {"job_id": "job-1"},
        "session": {
            "model": "model-a",
            "started_at": "2026-01-01T00:00:00Z",
            "ended_at": "2026-01-01T00:00:10Z",
            "step_count": 3,
            "verifier_execution": {"score": 1},
        },
        "transcript_envelope": {
            "harness": {
                "tool_names": ["bash", "submit_report"],
                "job_launch_params": {"max_steps": 600},
            },
            "task": {"key": "task-a", "eval_task_version_id": "version-a", "prompt": secret},
            "transcript": [
                {"role": "system", "content": secret},
                {"role": "user", "content": secret},
                {
                    "role": "assistant",
                    "content": secret,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "function": {
                                "name": "bash",
                                "arguments": json.dumps({"script": secret}),
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-1", "content": secret},
            ],
        },
    }
    source = tmp_path / "export.jsonl"
    source.write_text(json.dumps(row) + "\n")
    evidence = analyze_export(
        source,
        treatment_task_keys={"task-a"},
        export_manifest={"sessions": 1, "jobs": ["job-1"], "sha256": "sha256:fixture"},
    )
    rendered = json.dumps(evidence)
    assert secret not in rendered
    assert evidence["task_tool_surfaces"] == [{"tools": ["bash", "submit_report"], "sessions": 1}]
    assert evidence["as_treated_train"]["passes"] == 1
    assert evidence["task_prompt_hydration"]["mismatches"] == 0
    assert evidence["privacy"]["tool_arguments_emitted"] is False
    golden = json.loads(
        (Path(__file__).parent / "fixtures" / "harness_parity_golden.json").read_text()
    )
    assert evidence == golden


def test_tool_schema_receipt_requires_exact_set_and_is_order_stable() -> None:
    bash = {"type": "function", "function": {"name": "bash", "parameters": {"type": "object"}}}
    report = {
        "type": "function",
        "function": {"name": "submit_report", "parameters": {"type": "object"}},
    }
    first = tool_schema_receipt([bash, report], required=["bash", "submit_report"])
    second = tool_schema_receipt([report, bash], required=["bash", "submit_report"])
    assert first == second

    import pytest

    with pytest.raises(ValueError, match="exactly"):
        tool_schema_receipt([bash], required=["bash", "submit_report"])
