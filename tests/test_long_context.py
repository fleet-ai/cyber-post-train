"""Retired v1 canary: preserve provenance and rendered-job safety regressions."""

from copy import deepcopy
import json
import unittest

from training.long_context import SPEC, validate_historical_hooks, validate_rendered_job


class LongContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads(SPEC.read_text())

    def test_historical_source_and_rendered_job(self):
        validate_historical_hooks(self.spec)
        image = self.spec["cluster"]["image"]
        group = {"template": {"spec": {"priorityClassName": "c1", "containers": [{"image": image, "resources": {"requests": {"nvidia.com/gpu": 8}, "limits": {"nvidia.com/gpu": 8}}}]}}}
        job = {"kind": "RayJob", "metadata": {"annotations": {"fleet.ai/failure-alerts": "off"}, "labels": {"kueue.x-k8s.io/priority-class": "q1"}}, "spec": {"shutdownAfterJobFinishes": True, "ttlSecondsAfterFinished": 0, "rayClusterSpec": {"headGroupSpec": group, "workerGroupSpecs": [{**group, "replicas": 3}]}}}
        validate_rendered_job(self.spec, job)
        omitted_ttl = deepcopy(job)
        del omitted_ttl["spec"]["ttlSecondsAfterFinished"]
        validate_rendered_job(self.spec, omitted_ttl)
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
        for part, bad in (("priorityClassName", "c0"), ("containers", [])):
            wrong = deepcopy(job)
            wrong["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"][part] = bad
            with self.assertRaises(ValueError):
                validate_rendered_job(self.spec, wrong)


if __name__ == "__main__":
    unittest.main()
