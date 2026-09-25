"""Offline restore-only request tests; no source checkpoint or GPU required."""

import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training import long_context_reload as reload
from training.long_context_launch import _old_python, historical_request


class ReloadTests(unittest.TestCase):
    def test_sealed_step1_is_required_and_request_is_zero_step(self):
        with self.assertRaises(FileNotFoundError):
            reload.prepare()
        source, _ = historical_request(successor=True)
        names = {"data.pt", "trainer_state.pt", "policy/fsdp_config.json",
                 "policy/huggingface/config.json"} | {
                     f"policy/{kind}_world_size_32_rank_{rank}.pt"
                     for kind in ("model", "optim", "extra_state") for rank in range(32)}
        manifest = {"schema": "cyber_skyrl_checkpoint_manifest_v1", "source_plan": source,
                    "source_plan_sha256": reload.SOURCE_SHA, "checkpoint_path": reload.CHECKPOINT,
                    "optimizer_step": 1, "world_size": 32, "sampler_batches_in_epoch": 1,
                    "gpu_reload_verified": False,
                    "files": {name: {"bytes": 1, "sha256": "0" * 64} for name in names},
                    "total_bytes": len(names)}
        manifest["receipt_sha256"] = reload.sha(reload.canonical(manifest))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sealed.json"
            path.write_text(json.dumps(manifest))
            with patch.object(reload, "MANIFEST", path):
                plan, request = reload.prepare(path)
            staged = Path(tmp) / "staged"
            staged.mkdir()
            reload._stage(staged)
            (staged / "training/long_context_reload.py").write_bytes(Path(reload.__file__).read_bytes())
            _old_python(staged, """
import json,sys
from pathlib import Path
from training import long_context_reload as r
v=json.loads(sys.stdin.read());r.MANIFEST=Path(v['path']);r.check(v['plan']);print('{}')
""", stdin=json.dumps({"path": str(path), "plan": plan}))
        self.assertEqual((request["workers"], request["gpus_per_worker"]), (4, 8))
        self.assertEqual(request["priority_class"], "c1")
        self.assertIs(request["failureAlerts"], False)
        self.assertIs(request["requeueIfPreempted"], False)
        self.assertEqual(plan["recovery"]["mode"], "validate")
        self.assertIs(plan["reload_gate"]["submission_authorized"], False)
        self.assertEqual((plan["model"], plan["datasets"], plan["recipe"]),
                         (source["model"], source["datasets"], source["recipe"]))
        manifest["optimizer_step"] = 2
        with self.assertRaisesRegex(ValueError, "step-1"):
            reload.build_plan(manifest)
        tree = ast.parse(Path(reload.__file__).read_text())
        calls = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute)}
        self.assertFalse({"train", "optim_step", "backward", "run_eval", "save_checkpoint"} & calls)
