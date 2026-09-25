"""Offline checks for the historical 57M-token four-node plan."""

import unittest

from training.long_context_full import build


class FullPlanTests(unittest.TestCase):
    def test_exact_plan_is_not_yet_submission_authorized(self):
        plan, request = build()
        self.assertEqual((request["workers"], request["gpus_per_worker"]), (4, 8))
        self.assertEqual((request["priority_class"], request["failureAlerts"]), ("c1", False))
        self.assertEqual((plan["recipe"]["max_length"], plan["recipe"]["max_steps"]), (262144, 115))
        self.assertEqual(plan["datasets"]["train"]["supervised_tokens"], 57384881)
        self.assertFalse(plan["qualification"]["submission_authorized"])


if __name__ == "__main__":
    unittest.main()
