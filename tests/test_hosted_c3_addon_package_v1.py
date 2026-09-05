from __future__ import annotations

from pathlib import Path

import yaml

from evals.fleet import hosted_c3_addon_package_v1 as package

ROOT = Path(__file__).parents[1]


def test_c3_held_package_is_isolated_from_failed_c4_objects() -> None:
    manifest = yaml.safe_load((ROOT / package.MANIFEST_PATH).read_text())
    assert manifest["metadata"]["name"] == package.JOB_NAME
    assert manifest["metadata"]["annotations"] == {
        "cyber-post-train.fleet.ai/preview-only": "true",
        "cyber-post-train.fleet.ai/launch-authorized": "false",
    }
    assert package.JOB_NAME == "chris-cyber-hosted-q38-c3-addon-v1"
    assert package.CONFIGMAP_NAME == "chris-cyber-hosted-q38-c3-addon-bootstrap-v1"
    assert package.OUT_ROOT == "/mnt/sfs/jobs/chris-cyber-hosted-q38-c3-addon-v1"
    assert "c4-addon-v1" not in package.JOB_NAME + package.CONFIGMAP_NAME + package.OUT_ROOT
    assert manifest["spec"]["template"]["spec"]["preemptionPolicy"] == "Never"


def test_c3_package_embeds_materializer_and_exact_zero_score_contract() -> None:
    assert package.FILES["package.py"] == "evals/fleet/hosted_c3_addon_package_v1.py"
    source = (ROOT / "evals/fleet/hosted_c3_addon_package_v1.py").read_text()
    assert '"addon_slot": 3' in source
    assert '"maximum_total_owned_streams": 3' in source
    assert '"synthetic_chat_completion_requests": 2' in source
    assert '"task_instance_session_scoring_or_verifier_calls": 0' in source
    assert '"launch_authorized": False' in source
