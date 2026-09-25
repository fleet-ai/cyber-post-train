"""Boundary tests for the compact 96k launcher; no cluster or private data."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training import launch


CONFIG = Path(__file__).resolve().parents[1] / "configs/runs/qwen38-96k-debug-v1.json"


def fake_manifest() -> dict:
    train = {
        "path": "train.parquet", "sha256": "sha256:" + "a" * 64,
        "rows": 16, "task_keys": ["train-family"],
        "format": "chat_messages_last_assistant_v2",
        "supervised_tokens": 100, "assistant_responses": 2,
        "source_sessions": 1, "source_total_assistant_responses": 2,
        "excluded_assistant_responses": 0,
    }
    dev = {
        "path": "dev.parquet", "sha256": "sha256:" + "b" * 64,
        "rows": 1, "task_keys": ["dev-family"],
        "format": "chat_messages_last_assistant_v2",
    }
    value = {
        "schema": "qwen38_tool_aware_parquet_v1",
        "trainer_ready": True,
        "source_receipt_sha256": "sha256:" + "c" * 64,
        "split_sha256": "sha256:" + "d" * 64,
        "validation_mode": "teacher_cross_entropy",
        "tokenizer": {
            "repo": "Qwen/Qwen3.8-27B",
            "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
            "sha256": "sha256:" + "e" * 64,
        },
        "files": {"train": train, "dev": dev},
    }
    value["sha256"] = "sha256:" + launch.sha(launch.canonical(value))
    return value


def fake_preview(request: dict) -> dict:
    resources = request["resources"]
    c = {
        "image": request["image"],
        "resources": {
            "requests": {"cpu": resources["cpu_request"], "memory": resources["memory_request"],
                         "nvidia.com/gpu": request["gpus_per_worker"]},
            "limits": {"cpu": resources["cpu_limit"], "memory": resources["memory_limit"],
                       "nvidia.com/gpu": request["gpus_per_worker"]},
        },
        "env": [{"name": "RUN_DIR", "value": request["run_dir"]}]
        + [{"name": k, "value": v} for k, v in request["env"].items()],
        "envFrom": [{"secretRef": {"name": name, "optional": False}}
                    for name in request["secrets"]],
    }
    obj = {
        "kind": "RayJob",
        "metadata": {
            "namespace": "fleet-train-jobs",
            "labels": {
                "kueue.x-k8s.io/queue-name": "training-lq",
                "kueue.x-k8s.io/priority-class": "q1",
                "fleet.ai/requeue-if-preempted": "false",
            },
            "annotations": {
                "fleet.ai/run-dir": request["run_dir"],
                "fleet.ai/failure-alerts": "off",
            },
        },
        "spec": {
            "suspend": True, "shutdownAfterJobFinishes": True,
            "entrypoint": request["command"],
            "rayClusterSpec": {
                "headGroupSpec": {"template": {"spec": {
                    "priorityClassName": "c1", "containers": [c],
                }}},
                "workerGroupSpecs": [],
            },
        },
    }
    return {"manifest_yaml": json.dumps(obj), "errors": [], "warnings": []}


class LaunchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = json.loads(CONFIG.read_text())
        self.config["data"]["manifest"] = "corpus.json"
        (self.root / "corpus.json").write_text(json.dumps(fake_manifest()))
        self.config_path = self.root / "run.json"

    def prepare(self) -> tuple[Path, dict]:
        self.config_path.write_text(json.dumps(self.config))
        dest = self.root / "prepared"
        launch.prepare(self.config_path, dest)
        return dest, json.loads((dest / "request.json").read_text())

    def test_prepares_exact_historical_runtime_and_immutable_request(self) -> None:
        dest, request = self.prepare()
        plan, rebuilt, receipt = launch.prepared(dest)
        self.assertEqual(request, rebuilt)
        self.assertEqual(plan["recipe"]["max_steps"], 2)
        self.assertEqual(plan["recipe"]["max_length"], 98304)
        self.assertEqual(plan["validation_mode"], "teacher_cross_entropy")
        self.assertEqual(request["priority_class"], "c1")
        self.assertIs(request["failureAlerts"], False)
        self.assertEqual(receipt["historical_commit"], launch.COMMIT)
        self.assertEqual(launch._legacy(
            "validate_preview", {"request": request, "preview": fake_preview(request)}
        )["gpus"], 8)

    def test_missing_root_alert_annotation_fails_closed(self) -> None:
        _dest, request = self.prepare()
        preview = fake_preview(request)
        obj = json.loads(preview["manifest_yaml"])
        del obj["metadata"]["annotations"]["fleet.ai/failure-alerts"]
        preview["manifest_yaml"] = json.dumps(obj)
        with self.assertRaisesRegex(ValueError, "validate_preview gate rejected"):
            launch._legacy("validate_preview", {"request": request, "preview": preview})

    def test_rendered_priority_drift_fails_closed(self) -> None:
        _dest, request = self.prepare()
        preview = fake_preview(request)
        obj = json.loads(preview["manifest_yaml"])
        obj["metadata"]["labels"]["kueue.x-k8s.io/priority-class"] = "q0"
        preview["manifest_yaml"] = json.dumps(obj)
        with self.assertRaisesRegex(ValueError, "validate_preview gate rejected"):
            launch._legacy("validate_preview", {"request": request, "preview": preview})

    def test_cpu_gate_must_bind_both_splits_and_exact_request(self) -> None:
        dest, _request = self.prepare()
        _plan, _request, receipt = launch.prepared(dest)
        result = {
            "status": "passed", "gpus": 0,
            "request_sha256": receipt["request_sha256"],
            "plan_sha256": receipt["plan_sha256"],
            "checked": [
                "native_sources", "model_files", "dataset_files", "native_config",
                "native_forward_backward_signature", "native_train_only_loader",
                "tokenization", "target_accounting",
            ],
            "counts": {
                "train": {"rows": 16, "supervised_tokens": 100},
                "dev": {"rows": 1, "supervised_tokens": 10},
            },
        }
        self.assertTrue(launch._preflight_matches(result, receipt))
        result["counts"].pop("dev")
        self.assertFalse(launch._preflight_matches(result, receipt))
        result["counts"]["dev"] = {"rows": 1, "supervised_tokens": 10}
        result["request_sha256"] = "0" * 64
        self.assertFalse(launch._preflight_matches(result, receipt))

    def test_config_and_prepared_tampering_fail_closed(self) -> None:
        self.config["cluster"]["priority"] = "c0"
        self.config_path.write_text(json.dumps(self.config))
        with self.assertRaisesRegex(ValueError, "SkyRL and c1"):
            launch.prepare(self.config_path, self.root / "bad")
        self.config["cluster"]["priority"] = "c1"
        dest, request = self.prepare()
        request["failureAlerts"] = True
        (dest / "request.json").write_text(json.dumps(request))
        with self.assertRaisesRegex(ValueError, "prepared binding changed"):
            launch.prepared(dest)

    def test_old_corpus_and_unbounded_debug_are_rejected(self) -> None:
        self.config["data"]["manifest"] = "../data/qwen38-teacher3k-96k-v1.manifest.json"
        with self.assertRaisesRegex(ValueError, "old clipped/leaky corpus"):
            launch._require_debug_config(self.config)
        self.config["data"]["manifest"] = "corpus.json"
        manifest = fake_manifest()
        manifest["files"]["train"]["rows"] = 264  # 33 optimizer steps
        manifest.pop("sha256")
        manifest["sha256"] = "sha256:" + launch.sha(launch.canonical(manifest))
        (self.root / "corpus.json").write_text(json.dumps(manifest))
        self.config_path.write_text(json.dumps(self.config))
        with self.assertRaisesRegex(ValueError, "immutable safety/science gate"):
            launch.prepare(self.config_path, self.root / "too-long")


if __name__ == "__main__":
    unittest.main()
