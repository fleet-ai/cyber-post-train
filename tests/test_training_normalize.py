import json
import tempfile
import unittest
from pathlib import Path

from training.normalize import build_datasets, normalize_export_row


def row(
    session_id="s1",
    score=1.0,
    status="completed",
    task_key="cysec1-2-fira-gen_demo__blackbox_ctf_v1",
):
    return {
        "export_schema": "fleet_session_export_v1",
        "source": {
            "job_id": "j1",
            "session_id": session_id,
            "roster_task_binding": {
                "key": task_key,
                "version": "3",
                "eval_task_version_id": "7ec7aeca-0c69-4df6-989e-81278980d0a9",
                "verifier_id": "verifier-uuid",
                "data_version": "11",
            },
        },
        "session": {"status": status, "model": "glm-5.2-fp8", "score": score},
        "transcript_envelope": {
            "task": {"key": task_key, "prompt": "find the flag", "version": "3"},
            "instance": {"env_key": "cysec1-2-fira", "version": "v9"},
            "verifier_execution": {"score": score, "success": score == 1.0},
            "transcript": [
                {"role": "system", "content": "authorized"},
                {"role": "user", "content": "find the flag"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
                {"role": "tool", "tool_call_id": "c1", "content": "ok"},
                {"role": "assistant", "content": "done"},
            ],
            "grading": {"executions": [{"stdout": "SECRET GRADER"}]},
        },
    }


class NormalizeTests(unittest.TestCase):
    def test_success_is_sft_eligible_and_grader_is_excluded(self):
        normalized = normalize_export_row(row())
        self.assertTrue(normalized["eligibility"]["sft"])
        self.assertEqual(normalized["lineage"]["application"], "fira")
        self.assertEqual(normalized["lineage"]["lineage_quality"], "fleet_task_version_id")
        self.assertEqual(
            normalized["lineage"]["eval_task_version_id"],
            "7ec7aeca-0c69-4df6-989e-81278980d0a9",
        )
        self.assertNotIn("SECRET GRADER", json.dumps(normalized))

    def test_failure_is_not_imitated_but_is_rl_prompt(self):
        normalized = normalize_export_row(row(score=0.0))
        self.assertFalse(normalized["eligibility"]["sft"])
        self.assertTrue(normalized["eligibility"]["preference"])
        self.assertTrue(normalized["eligibility"]["online_rl_prompt"])

    def test_verifier_process_success_does_not_turn_score_zero_into_task_success(self):
        value = row(score=0.0)
        value["transcript_envelope"]["verifier_execution"]["success"] = True
        normalized = normalize_export_row(value)
        self.assertFalse(normalized["outcome"]["success"])
        self.assertFalse(normalized["eligibility"]["sft"])

    def test_exact_control_plane_secret_redaction(self):
        value = row()
        value["transcript_envelope"]["transcript"][4]["content"] = "oops control-secret-value"
        normalized = normalize_export_row(value, secrets=("control-secret-value",))
        self.assertNotIn("control-secret-value", json.dumps(normalized))

    def test_builds_pair_from_same_prompt_and_lineage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw.jsonl"
            rows = [row("win", 1.0), row("lose", 0.0)]
            raw.write_text("\n".join(json.dumps(item) for item in rows) + "\n")
            manifest = build_datasets(raw, root / "out")
            sft_count = sum(
                manifest["files"][name]["rows"] for name in ("sft_train", "sft_dev", "sft_test")
            )
            self.assertEqual(sft_count, 1)
            self.assertEqual(manifest["files"]["preferences"]["rows"], 1)
            self.assertEqual(manifest["files"]["rl_prompts"]["rows"], 1)
            self.assertTrue((root / "out" / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
