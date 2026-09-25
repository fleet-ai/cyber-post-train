"""Offline regressions for the unqualified 4×8 Qwen3.8 capacity canary."""

from copy import deepcopy
import json
import unittest

from training.long_context import SPEC, preflight, validate_cpu_preflight, validate_example, validate_historical_hooks, validate_length_audit, validate_real_row_witness, validate_rendered_job, validate_spec


class LongContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads(SPEC.read_text())

    def test_exact_source_and_hypothesis(self):
        validate_spec(self.spec)
        validate_historical_hooks(self.spec)
        validate_length_audit(self.spec)
        self.assertEqual(validate_cpu_preflight(self.spec)["status"], "Succeeded")
        row = validate_real_row_witness(self.spec)
        self.assertEqual((row["sequence_tokens"], row["supervised_tokens"]), (262144, 153984))
        self.assertEqual(self.spec["status"], "unqualified_hypothesis")

    def test_scientific_and_submission_drift_fails(self):
        for section, key, bad in (
            ("recipe", "nodes", 8),
            ("recipe", "world_size", 64),
            ("recipe", "max_length", 98304),
            ("recipe", "max_steps", 1),
            ("recipe", "pause_after_optimizer_step", 2),
            ("recipe", "layer_checkpoint_group_size", 2),
            ("recipe", "lm_head_chunk_tokens", 262144),
            ("recipe", "reload_optimizer_steps", 1),
            ("cluster", "priority_class", "c0"),
            ("cluster", "root_annotation", {}),
        ):
            with self.subTest(section=section, key=key):
                wrong = deepcopy(self.spec)
                wrong[section][key] = bad
                with self.assertRaises(ValueError):
                    validate_spec(wrong)
        wrong = deepcopy(self.spec)
        wrong["accepted"] = True
        with self.assertRaisesRegex(ValueError, "unproven acceptance"):
            validate_spec(wrong)
        wrong = deepcopy(self.spec)
        wrong["parent"]["accepted"] = True
        with self.assertRaisesRegex(ValueError, "unexpected historical parent"):
            validate_spec(wrong)

    def test_real_full_length_example_is_required(self):
        example = {"schema": "qwen38_262k_real_row_summary_v1", "source_kind": "real_training_row", "dataset_sha256": self.spec["data"]["sha256"], "row_sha256": "a" * 64, "sequence_tokens": 262144, "nonpadding_tokens": 250000, "supervised_tokens": 1000}
        validate_example(self.spec, example)
        for field, bad in (("sequence_tokens", 250000), ("nonpadding_tokens", 249999), ("supervised_tokens", 0), ("dataset_sha256", "wrong"), ("row_sha256", "")):
            with self.subTest(field=field):
                wrong = {**example, field: bad}
                with self.assertRaises(ValueError):
                    validate_example(self.spec, wrong)
        aggregate = json.loads((SPEC.parents[2] / self.spec["length_audit"]["path"]).read_text())
        with self.assertRaisesRegex(ValueError, "wrong example summary"):
            validate_example(self.spec, aggregate)

    def test_only_root_alert_off_and_effective_4x8_c1_render_pass(self):
        image = self.spec["cluster"]["image"]
        group = {"template": {"spec": {"priorityClassName": "c1", "containers": [{"image": image, "resources": {"requests": {"nvidia.com/gpu": 8}, "limits": {"nvidia.com/gpu": 8}}}]}}}
        job = {"kind": "RayJob", "metadata": {"annotations": {"fleet.ai/failure-alerts": "off"}, "labels": {"kueue.x-k8s.io/priority-class": "q1"}}, "spec": {"shutdownAfterJobFinishes": True, "ttlSecondsAfterFinished": 0, "rayClusterSpec": {"headGroupSpec": group, "workerGroupSpecs": [{**group, "replicas": 3}]}}}
        validate_rendered_job(self.spec, job)
        omitted_ttl = deepcopy(job)
        del omitted_ttl["spec"]["ttlSecondsAfterFinished"]
        validate_rendered_job(self.spec, omitted_ttl)
        example = validate_real_row_witness(self.spec)
        result = preflight(self.spec, example, job)
        self.assertEqual((result["real_row_verified"], result["native_cpu_preflight"], result["gpu_qualified"]), (True, "passed", False))
        for section, key, bad in (("metadata", "annotations", {}), ("metadata", "labels", {}), ("spec", "ttlSecondsAfterFinished", 30)):
            with self.subTest(section=section, key=key):
                wrong = deepcopy(job)
                wrong[section][key] = bad
                with self.assertRaises(ValueError):
                    validate_rendered_job(self.spec, wrong)
        wrong = deepcopy(job)
        wrong["spec"]["rayClusterSpec"]["workerGroupSpecs"][0]["replicas"] = 2
        with self.assertRaises(ValueError):
            validate_rendered_job(self.spec, wrong)


if __name__ == "__main__":
    unittest.main()
