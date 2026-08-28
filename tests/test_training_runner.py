import json
import tempfile
import unittest
from pathlib import Path

from training.io import digest_json
from training.runner import run_plan, stage_argv


class RunnerTests(unittest.TestCase):
    def test_stage_command_must_exist(self):
        with self.assertRaisesRegex(ValueError, "SFT_TRAIN_COMMAND"):
            stage_argv("sft", {})

    def test_plan_defaults_to_dry_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            unsigned = {
                "schema": "cyber_post_train_run_plan_v1",
                "stages": [{"name": "sft"}, {"name": "pre_rl_eval"}],
            }
            plan = {**unsigned, "plan_digest": digest_json(unsigned)}
            path = root / "plan.json"
            path.write_text(json.dumps(plan))
            outcomes = run_plan(
                path,
                root / "run",
                env={
                    "SFT_TRAIN_COMMAND": "python -m trainer.sft",
                    "CYBER_EVAL_HOOK": "python -m frozen_eval",
                },
            )
            self.assertEqual([item["status"] for item in outcomes], ["dry_run"] * 2)
            self.assertFalse((root / "run" / "events.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
