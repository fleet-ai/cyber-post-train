"""Synthetic staging safety checks; never load private source bytes in tests."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training import stage_source as stage


class StageTests(unittest.TestCase):
    def test_verify_seals_once_after_bound_source_check(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "source"
            (root / "raw/sessions").mkdir(parents=True)
            root.chmod(0o700)
            (root / "source-selection.private.jsonl").touch(mode=0o600)
            (root / "raw/HYDRATE_REQUEST.json").touch(mode=0o600)
            (root / "raw/HYDRATED.json").write_text(json.dumps({"sha256": stage.RECEIPT_SHA}))
            expected = {"selected_sessions": 2886, "selection_sha256": stage.SELECTION_SHA,
                        "hydration_receipt_file_sha256": stage.RECEIPT_FILE_SHA,
                        "sessions_sha256": stage.SESSIONS_SHA}
            with patch.object(stage, "DEST", root), patch.object(stage, "verify_hydration_cache", return_value=expected) as checked:
                result = stage.verify(root, seal=True)
                self.assertEqual(json.loads((root / "STAGED.json").read_text()), result)
                self.assertEqual(result["sha256"], stage.digest({k: v for k, v in result.items() if k != "sha256"}))
                checked.assert_called_once_with(root / "raw", root / "source-selection.private.jsonl", stage.RECEIPT_FILE_SHA)
                with self.assertRaises(stage.SourceError):
                    stage.verify(root, seal=True)

    def test_job_is_cpu_only_and_silent(self):
        job = stage.job()
        self.assertEqual(job["metadata"]["annotations"]["fleet.ai/failure-alerts"], "off")
        self.assertEqual(job["metadata"]["labels"]["kueue.x-k8s.io/priority-class"], "q1")
        pod = job["spec"]["template"]["spec"]
        self.assertEqual((pod["priorityClassName"], pod["priority"]), ("c1", 10000))
        self.assertEqual(pod["securityContext"]["supplementalGroups"], [2000])
        self.assertEqual(pod["initContainers"][0]["securityContext"]["runAsUser"], 0)
        self.assertIn("p.mkdir(mode=0o700); os.chown(p,1000,100)", pod["initContainers"][0]["command"][2])
        self.assertEqual(pod["containers"][0]["securityContext"], {"runAsUser": 1000, "runAsGroup": 100})
        self.assertEqual(pod["volumes"][0]["persistentVolumeClaim"]["claimName"], "sfs-shared")
        self.assertTrue(job["spec"]["suspend"])
        self.assertNotIn("nvidia.com/gpu", json.dumps(job))


if __name__ == "__main__":
    unittest.main()
