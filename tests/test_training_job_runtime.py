import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from training.job_runtime import gate_rl, run_sft


class TrainingJobRuntimeTests(unittest.TestCase):
    def test_sft_receipt_allows_rl_only_while_checkpoint_is_intact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint"
            manifest = root / "manifest.json"
            receipt = root / "receipt.json"
            compatibility = root / "compatibility.json"
            image = "example.invalid/trainer@sha256:" + "a" * 64
            revision = "b" * 40
            compatibility.write_text(
                json.dumps(
                    {
                        "schema": "cyber_post_train_model_compatibility_v1",
                        "status": "succeeded",
                        "model_revision": revision,
                        "training_image": image,
                        "model_adapter_digest": "sha256:" + "d" * 64,
                        "driver_contract": "cyber_post_train_verl_fsdp2_driver_v1",
                    }
                )
            )
            checkpoint_lock = root / "checkpoint-lock.json"
            checkpoint_lock.write_text(
                json.dumps(
                    {
                        "schema": "cyber_post_train_checkpoint_lock_v1",
                        "repo": "Qwen/Qwen3.6-27B",
                        "revision": revision,
                        "weights_manifest_sha256": "sha256:" + "e" * 64,
                    }
                )
            )
            command = [
                sys.executable,
                "-c",
                f"import pathlib; p=pathlib.Path(r'{checkpoint}'); "
                "p.mkdir(); (p/'weights').write_bytes(b'ok')",
            ]
            env = {
                "MODEL_COMPATIBILITY_RECEIPT": str(compatibility),
                "MODEL_CHECKPOINT_LOCK": str(checkpoint_lock),
                "MODEL_ID": "Qwen/Qwen3.6-27B",
                "MODEL_REVISION": revision,
                "MODEL_WEIGHTS_MANIFEST_SHA256": "sha256:" + "e" * 64,
                "TRAINING_IMAGE": image,
                "TRAINING_ARGV_JSON": json.dumps(command),
                "SFT_CHECKPOINT_DIR": str(checkpoint),
                "SFT_CHECKPOINT_MANIFEST": str(manifest),
                "SFT_CHECKPOINT_RECEIPT": str(receipt),
                "SFT_JOB_NAME": "chris-cyber-qwen-test-sft",
                "DATASET_DIGEST": "sha256:" + "c" * 64,
                "MODEL_ADAPTER_DIGEST": "sha256:" + "d" * 64,
                "MODEL_DRIVER_CONTRACT": "cyber_post_train_verl_fsdp2_driver_v1",
            }
            with mock.patch.dict(os.environ, env, clear=True):
                run_sft()
                gate_rl()
                (checkpoint / "weights").write_bytes(b"tampered")
                with self.assertRaisesRegex(ValueError, "changed|mismatch"):
                    gate_rl()


if __name__ == "__main__":
    unittest.main()
