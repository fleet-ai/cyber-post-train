"""No-network checks for the held historical four-node 262K request."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from training.long_context_launch import cpu_preflight_job, historical_request, prepare, stage_old_code, successor_spec, verify_bundle


class LongContextLaunchTests(unittest.TestCase):
    def test_compiles_exact_historical_bundle_without_submission(self):
        plan, request = historical_request()
        self.assertFalse(plan["qualification"]["submission_gate"]["submission_authorized"])
        self.assertEqual((request["workers"], request["gpus_per_worker"]), (4, 8))
        self.assertEqual(request["priority_class"], "c1")
        self.assertIs(request["failureAlerts"], False)
        verify_bundle(plan, request)
        tampered = {**plan, "run_name": "wrong"}
        with self.assertRaisesRegex(ValueError, "bundle plan changed"):
            verify_bundle(tampered, request)

    def test_preparation_is_create_once_and_preview_only(self):
        with tempfile.TemporaryDirectory() as parent:
            output = Path(parent) / "candidate"
            summary = prepare(output)
            self.assertEqual(summary["state"], "PREVIEW_ONLY_SUBMISSION_BLOCKED")
            self.assertEqual(summary["gpus"], 32)
            self.assertEqual(json.loads((output / "request.json").read_text())["run_dir"], summary["output_root"])
            self.assertFalse((output / "rendered-rayjob.yaml").exists())
            with self.assertRaises(FileExistsError):
                prepare(output)

    def test_v3_repairs_only_runtime_plan_field_and_identity(self):
        old_plan, old_request = historical_request()
        plan, request = historical_request(successor=True)
        self.assertTrue(plan["qualification"]["submission_gate"]["submission_authorized"])
        self.assertEqual(plan["qualification"]["submission_gate"]["blockers"], [])
        self.assertEqual(plan["qualification"]["submission_gate"]["approval_evidence"]["v1_cpu_receipt_sha256"], "39c827afa6db9733e175c6ebb3b5c75478e661d5266a265fa6402f40132b5cf0")
        self.assertEqual(plan["runtime_variant"], old_plan["runtime_variant"])
        self.assertEqual(plan["model"], old_plan["model"])
        self.assertEqual(plan["datasets"], old_plan["datasets"])
        self.assertEqual(plan["recipe"], old_plan["recipe"])
        self.assertEqual(request["workers"], old_request["workers"])
        self.assertEqual(request["gpus_per_worker"], old_request["gpus_per_worker"])
        self.assertEqual(request["priority_class"], old_request["priority_class"])
        self.assertIs(request["failureAlerts"], False)
        self.assertEqual(request["name"], "chris-q38-t3k262-4n-can-v3")
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            stage_old_code(Path(a))
            stage_old_code(Path(b), successor=True)
            for name in ("training/sft_262k_4node_v1.py", "training/sft_262k_runtime.py", "configs/runs/qwen38-teacher3k-262k-4node-canary-v1.json"):
                before = (Path(a) / name).read_text().replace("chris-q38-t3k262-4n-can-v1", "chris-q38-t3k262-4n-can-v3")
                after = (Path(b) / name).read_text()
                if name.endswith("sft_262k_runtime.py"):
                    old_gate = '''    "submission_authorized": False,
    "blockers": [
        "zero-GPU preflight receipt absent",
        "four-node GPU launch has not received root review",
    ],'''
                    new_gate = '    "submission_authorized": True,\n    "blockers": [],\n    "approval_evidence": ' + repr(successor_spec()["root_review"]) + ','
                    before = before.replace(old_gate, new_gate)
                    before = before.replace('        set(plan) != expected_keys', '        set(plan) - {"plan_sha256"} != expected_keys')
                    old = '    _BASE_VALIDATE_PLAN(plan, check_files=check_files)'
                    before = before.replace(old, old + '\n    if "plan_sha256" in plan and plan["plan_sha256"] != base._unsigned_digest({k: v for k, v in plan.items() if k != "plan_sha256"}):\n        raise ValueError("runtime plan digest changed")')
                self.assertEqual(before, after)

            program = "import json,sys; from training.sft_262k_runtime import validate_plan; from cyber_post_train.jobs import digest; p=json.load(sys.stdin); p['plan_sha256']=digest(p); validate_plan(p,check_files=False)"
            for root, candidate, success in ((a, old_plan, False), (b, plan, True)):
                result = subprocess.run([sys.executable, "-c", program], input=json.dumps(candidate), text=True, capture_output=True, cwd=root, env={**os.environ, "PYTHONPATH": root})
                self.assertEqual(result.returncode == 0, success)

    def test_cpu_preflight_job_is_bounded_and_zero_gpu(self):
        job = cpu_preflight_job()
        pod = job["spec"]["template"]["spec"]
        container = pod["containers"][0]
        self.assertEqual(job["metadata"]["annotations"], {"fleet.ai/failure-alerts": "off"})
        self.assertEqual(job["metadata"]["labels"]["kueue.x-k8s.io/priority-class"], "q1")
        self.assertEqual(pod["priorityClassName"], "c1")
        self.assertEqual((job["spec"]["backoffLimit"], job["spec"]["activeDeadlineSeconds"]), (0, 1800))
        self.assertEqual(pod["restartPolicy"], "Never")
        self.assertNotIn("nvidia.com/gpu", container["resources"]["requests"])
        self.assertTrue(pod["volumes"][0]["persistentVolumeClaim"]["readOnly"])
        self.assertTrue(container["volumeMounts"][0]["readOnly"])
        self.assertLess(len(container["env"][0]["value"]), 120000)
        compile(container["command"][2], "preflight", "exec")
        successor = cpu_preflight_job(successor=True)
        self.assertEqual(successor["metadata"]["name"], "chris-q38-262k4n-cpu-pre-v6")
        self.assertEqual(successor["metadata"]["annotations"], {"fleet.ai/failure-alerts": "off"})
        self.assertNotIn("nvidia.com/gpu", successor["spec"]["template"]["spec"]["containers"][0]["resources"]["requests"])
        compile(successor["spec"]["template"]["spec"]["containers"][0]["command"][2], "successor-preflight", "exec")


if __name__ == "__main__":
    unittest.main()
