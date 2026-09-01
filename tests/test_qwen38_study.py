import unittest

from evals.qwen38_study import validate_static_controls


class Qwen38ExternalEvaluationControlTest(unittest.TestCase):
    def test_static_controls_match_qwen36_harness_and_benchmarks(self):
        receipt = validate_static_controls()
        self.assertTrue(receipt["valid"])
        self.assertEqual(receipt["model"], "qwen3.8-27b")
        self.assertEqual(
            receipt["temporal_interpretation"],
            "paired_before_after_only_not_temporally_clean_holdout",
        )
        self.assertIn("@sha256:", receipt["control_image"])


if __name__ == "__main__":
    unittest.main()
