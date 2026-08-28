import unittest

from training.cluster_policy import validate_scheduler_state, validate_training_manifest


def rayjob():
    return {
        "apiVersion": "ray.io/v1",
        "kind": "RayJob",
        "metadata": {
            "name": "chris-cyber-glm52-compat-b4734de4",
            "namespace": "fleet-train-jobs",
            "labels": {
                "kueue.x-k8s.io/queue-name": "training-lq",
                "cyber-post-train.fleet.ai/owner": "chris",
                "cyber-post-train.fleet.ai/experiment": "compat-b4734de4",
            },
        },
        "spec": {
            "rayClusterSpec": {
                "headGroupSpec": {
                    "template": {"spec": {"preemptionPolicy": "Never", "containers": []}}
                },
                "workerGroupSpecs": [
                    {"template": {"spec": {"preemptionPolicy": "Never", "containers": []}}}
                ],
            }
        },
    }


class ClusterPolicyTests(unittest.TestCase):
    def test_accepts_owned_queued_nonpreempting_workload(self):
        validate_training_manifest(rayjob())

    def test_rejects_queue_bypass(self):
        manifest = rayjob()
        del manifest["metadata"]["labels"]["kueue.x-k8s.io/queue-name"]
        with self.assertRaisesRegex(ValueError, "training-lq"):
            validate_training_manifest(manifest)

    def test_rejects_priority_escalation(self):
        manifest = rayjob()
        manifest["spec"]["rayClusterSpec"]["workerGroupSpecs"][0]["template"]["spec"][
            "priorityClassName"
        ] = "critical"
        with self.assertRaisesRegex(ValueError, "priorityClassName"):
            validate_training_manifest(manifest)

    def test_accepts_nonpreempting_live_queue_contract(self):
        validate_scheduler_state(
            {
                "metadata": {"name": "training-cq"},
                "spec": {
                    "preemption": {
                        "withinClusterQueue": "LowerPriority",
                        "reclaimWithinCohort": "Never",
                        "borrowWithinCohort": {"policy": "Never"},
                    }
                },
            },
            {
                "metadata": {"name": "training-lq"},
                "spec": {"clusterQueue": "training-cq", "stopPolicy": "None"},
            },
        )

    def test_rejects_equal_priority_preemption_contract_drift(self):
        with self.assertRaisesRegex(ValueError, "LowerPriority"):
            validate_scheduler_state(
                {
                    "metadata": {"name": "training-cq"},
                    "spec": {"preemption": {"withinClusterQueue": "Any"}},
                },
                {
                    "metadata": {"name": "training-lq"},
                    "spec": {"clusterQueue": "training-cq", "stopPolicy": "None"},
                },
            )


if __name__ == "__main__":
    unittest.main()
