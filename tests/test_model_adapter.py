import json
import tempfile
import unittest
from pathlib import Path

from training.model_adapter import load_model_adapter

CONFIG = Path("training/configs/models/qwen36-27b.json")


class ModelAdapterTests(unittest.TestCase):
    def test_loads_staged_qwen_adapter(self):
        adapter = load_model_adapter(CONFIG)
        self.assertEqual(adapter.slug, "qwen36-27b")
        self.assertEqual(adapter.resources.gpus, 8)
        self.assertIn("@sha256:", adapter.image)
        self.assertTrue(adapter.digest.startswith("sha256:"))

    def test_rejects_mutable_image(self):
        config = json.loads(CONFIG.read_text())
        config["runtime"]["image"] = "ghcr.io/fleet-ai/miles-fleet/trainer:latest"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.json"
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "pinned"):
                load_model_adapter(path)

    def test_rejects_model_root_not_bound_to_revision(self):
        config = json.loads(CONFIG.read_text())
        config["model"]["root"] = "/mnt/sfs/models/Qwen/Qwen3.6-27B/current"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.json"
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "model.root"):
                load_model_adapter(path)

    def test_rejects_science_in_model_adapter(self):
        config = json.loads(CONFIG.read_text())
        config["reward"] = {"contract": "model-specific"}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.json"
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "forbidden"):
                load_model_adapter(path)


if __name__ == "__main__":
    unittest.main()
