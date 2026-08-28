import json
import tempfile
import unittest
from pathlib import Path

from training.fleet import is_blackbox_task_key, roster_session_refs
from training.io import digest_json
from training.plan import REQUIRED_COMPATIBILITY_CHECKS, create_run_plan
from training.rewards import compute_reward
from training.science import build_comparison_manifest, validate_eval_protocol
from training.splits import apply_splits


class IntegrityTests(unittest.TestCase):
    def test_reward_integrity_gate(self):
        self.assertEqual(compute_reward(verifier_score=1, infra_valid=True)["total"], 1)
        self.assertEqual(compute_reward(verifier_score=1, infra_valid=False)["total"], 0)
        hacked = compute_reward(verifier_score=1, infra_valid=True, reward_hack_detected=True)
        safe_partial = compute_reward(verifier_score=1, infra_valid=True, behavior_score=0.25)
        self.assertEqual(hacked["total"], 0)
        self.assertEqual(safe_partial["total"], 0.25)

    def test_roster_shape(self):
        roster = {
            "tasks": [
                {
                    "task": {
                        "key": "x__blackbox_ctf_v1",
                        "eval_task_version_id": "task-version-uuid",
                        "prompt": "must not be copied into the export binding",
                    },
                    "sessions": [{"session_id": "s", "status": "completed"}],
                }
            ]
        }
        refs = roster_session_refs("j", "job", roster)
        self.assertEqual(refs[0].session_id, "s")
        self.assertTrue(is_blackbox_task_key(refs[0].task_key))
        self.assertEqual(refs[0].task_binding["eval_task_version_id"], "task-version-uuid")
        self.assertNotIn("prompt", refs[0].task_binding)

    def test_lineage_never_crosses_splits(self):
        base = {
            "lineage": {
                "application": "fira",
                "task_family": "demo",
                "lineage_key": "registry:demo@1",
            }
        }
        records = [json.loads(json.dumps(base)) for _ in range(4)]
        apply_splits(records)
        self.assertEqual(len({record["split"] for record in records}), 1)

    def test_plan_fails_closed_then_passes_complete_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            revision = "a" * 40
            config = {
                "schema": "cyber_post_train_config_v1",
                "model": {
                    "name": "zai-org/GLM-5.2",
                    "revision": revision,
                    "weights_manifest_sha256": "sha256:" + "d" * 64,
                },
                "data": {"use_all_eligible_fleet_data": True},
            }
            unsigned_manifest = {"schema": "fleet_cyber_dataset_manifest_v1", "files": {}}
            manifest = {**unsigned_manifest, "manifest_digest": digest_json(unsigned_manifest)}
            protocol_unsigned = {
                "schema": "cyber_prepost_eval_protocol_v1",
                "model": {
                    "repo": "zai-org/GLM-5.2",
                    "base_checkpoint_revision": revision,
                    "weights_manifest_sha256": "sha256:" + "d" * 64,
                    "tokenizer_revision": revision,
                    "tokenizer_manifest_sha256": "sha256:" + "e" * 64,
                    "chat_template_sha256": "sha256:" + "b" * 64,
                },
                "serving": {
                    "image": "registry/serve@sha256:" + "1" * 64,
                    "engine_revision": "2" * 40,
                    "precision": "bf16",
                    "quantization_manifest_sha256": "sha256:" + "3" * 64,
                },
                "harness": {
                    "image": "registry/harness@sha256:" + "4" * 64,
                    "source_revision": "5" * 40,
                    "tool_schema_sha256": "sha256:" + "6" * 64,
                    "system_prompt_sha256": "sha256:" + "7" * 64,
                },
                "benchmark": {
                    "id": "frozen-benchmark-v1",
                    "source_revision": "8" * 40,
                    "task_manifest_sha256": "sha256:" + "9" * 64,
                    "environment_manifest_sha256": "sha256:" + "a" * 64,
                    "verifier_manifest_sha256": "sha256:" + "b" * 64,
                    "prompt_manifest_sha256": "sha256:" + "c" * 64,
                },
                "sampling": {"temperature": 0.6, "top_p": 0.95, "max_output_tokens": 32768},
                "budgets": {"max_agent_steps": 300, "max_duration_minutes": 120},
                "random_seeds": [17, 29, 43],
            }
            protocol = {
                **protocol_unsigned,
                "protocol_digest": digest_json(protocol_unsigned),
            }
            receipt = {
                "schema": "glm52_training_compatibility_v1",
                "model_revision": revision,
                "tokenizer_revision": revision,
                "chat_template_sha256": "sha256:" + "b" * 64,
                "training_image_digest": "sha256:" + "c" * 64,
                "framework_revision": "nemo-commit+megatron-commit",
                "weights_manifest_sha256": "sha256:" + "d" * 64,
                "inference_image_digest": "sha256:" + "1" * 64,
                "evaluation_protocol_digest": protocol["protocol_digest"],
                "cluster_gpu_topology": "2x8 H200 IB",
                "checks": {name: True for name in REQUIRED_COMPATIBILITY_CHECKS},
            }
            paths = [
                root / name
                for name in ("config.json", "manifest.json", "receipt.json", "protocol.json")
            ]
            for path, value in zip(paths, (config, manifest, receipt, protocol), strict=True):
                path.write_text(json.dumps(value))
            plan = create_run_plan(*paths, root / "plan.json")
            self.assertEqual(plan["model"]["revision"], revision)
            receipt["checks"]["forward_backward_bf16"] = False
            paths[2].write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "forward_backward_bf16"):
                create_run_plan(*paths, root / "bad.json")

    def test_protocol_digest_detects_harness_drift(self):
        protocol = {
            "schema": "cyber_prepost_eval_protocol_v1",
            "model": {
                "repo": "zai-org/GLM-5.2",
                "base_checkpoint_revision": "a" * 40,
                "weights_manifest_sha256": "sha256:" + "b" * 64,
                "tokenizer_revision": "c" * 40,
                "tokenizer_manifest_sha256": "sha256:" + "d" * 64,
                "chat_template_sha256": "sha256:" + "e" * 64,
            },
            "serving": {
                "image": "registry/serve@sha256:" + "f" * 64,
                "engine_revision": "1" * 40,
                "precision": "bf16",
                "quantization_manifest_sha256": "sha256:" + "2" * 64,
            },
            "harness": {
                "image": "registry/harness@sha256:" + "3" * 64,
                "source_revision": "4" * 40,
                "tool_schema_sha256": "sha256:" + "5" * 64,
                "system_prompt_sha256": "sha256:" + "6" * 64,
            },
            "benchmark": {
                "id": "benchmark-v1",
                "source_revision": "7" * 40,
                "task_manifest_sha256": "sha256:" + "8" * 64,
                "environment_manifest_sha256": "sha256:" + "9" * 64,
                "verifier_manifest_sha256": "sha256:" + "a" * 64,
                "prompt_manifest_sha256": "sha256:" + "b" * 64,
            },
            "sampling": {"temperature": 0.6, "top_p": 0.95, "max_output_tokens": 32768},
            "budgets": {"max_agent_steps": 300, "max_duration_minutes": 120},
            "random_seeds": [17, 29, 43],
        }
        protocol["protocol_digest"] = digest_json(protocol)
        self.assertEqual(validate_eval_protocol(protocol), protocol["protocol_digest"])
        protocol["harness"]["tool_schema_sha256"] = "sha256:" + "0" * 64
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            validate_eval_protocol(protocol)

    def test_comparison_manifest_allows_only_checkpoint_intervention(self):
        digest = "sha256:" + "a" * 64
        manifest = build_comparison_manifest(
            protocol_digest=digest,
            base_checkpoint_manifest_sha256="sha256:" + "b" * 64,
            intervention_checkpoint_manifest_sha256="sha256:" + "c" * 64,
            training_plan_digest="sha256:" + "d" * 64,
        )
        self.assertEqual(manifest["controlled_protocol_digest"], digest)
        self.assertEqual(manifest["allowed_difference"], "checkpoint_or_adapter_bytes_only")


if __name__ == "__main__":
    unittest.main()
