import unittest
from pathlib import Path

from training.cluster_policy import validate_training_manifest
from training.job_factory import DATASET_DIGEST, TrainingInputs, render_job_pair
from training.model_adapter import load_model_adapter


class TrainingJobFactoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = load_model_adapter(
            Path("training/configs/models/qwen36-27b.json")
        )

    def test_renders_suspended_priority_zero_nonpreempting_jobs(self):
        for manifest in render_job_pair(self.adapter, "poc-v1"):
            validate_training_manifest(manifest)
            self.assertTrue(manifest["spec"]["suspend"])
            pod = manifest["spec"]["template"]["spec"]
            self.assertNotIn("priorityClassName", pod)
            self.assertEqual(pod["preemptionPolicy"], "Never")
            self.assertEqual(
                manifest["metadata"]["annotations"]["cyber-post-train.fleet.ai/dataset-digest"],
                DATASET_DIGEST,
            )
            for container in [*(pod.get("initContainers") or []), *pod["containers"]]:
                self.assertEqual(container["command"][0], "python")
                self.assertNotIn("/bin/bash", container["command"])

    def test_online_rl_has_checkpoint_receipt_init_gate(self):
        sft, rl = render_job_pair(self.adapter, "poc-v1")
        pod = rl["spec"]["template"]["spec"]
        self.assertEqual(pod["initContainers"][0]["name"], "sft-checkpoint-gate")
        gate_env = {item["name"]: item["value"] for item in pod["initContainers"][0]["env"]}
        self.assertEqual(gate_env["RUNTIME_MODE"], "gate-rl")
        self.assertEqual(gate_env["SFT_JOB_NAME"], sft["metadata"]["name"])
        rl_env = {item["name"]: item["value"] for item in pod["containers"][0]["env"]}
        self.assertEqual(rl_env["RUNTIME_MODE"], "run-rl")
        self.assertEqual(rl_env["SFT_CHECKPOINT_RECEIPT"], gate_env["SFT_CHECKPOINT_RECEIPT"])

    def test_model_adapter_cannot_change_science(self):
        with self.assertRaisesRegex(ValueError, "science is sealed"):
            TrainingInputs(reward_contract="model-specific-reward")


if __name__ == "__main__":
    unittest.main()
