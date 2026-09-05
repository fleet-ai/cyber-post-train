from __future__ import annotations

from pathlib import Path

import yaml

from evals.fleet import hosted_c4_addon_package_v1 as package

ROOT = Path(__file__).parents[1]


def test_held_manifest_is_non_scoring_cpu_only() -> None:
    value = yaml.safe_load((ROOT / package.MANIFEST_PATH).read_text())
    assert value["metadata"]["annotations"] == {
        "cyber-post-train.fleet.ai/preview-only": "true",
        "cyber-post-train.fleet.ai/launch-authorized": "false",
    }
    pod = value["spec"]["template"]["spec"]
    assert pod["preemptionPolicy"] == "Never"
    assert pod["priorityClassName"] == "fleet-serve-low"
    assert "nvidia.com/gpu" not in pod["containers"][0]["resources"]["requests"]
    assert value["spec"]["backoffLimit"] == 0


def test_package_contract_is_held_and_exactly_two_addon_streams() -> None:
    source = (ROOT / "evals/fleet/hosted_c4_addon_package_v1.py").read_text()
    assert '"status": "HELD"' in source
    assert '"addon_slots": [3, 4]' in source
    assert '"maximum_total_owned_streams": 4' in source
    assert '"task_instance_session_scoring_or_verifier_calls": 0' in source
    assert '"launch_authorized": False' in source
    assert '"package.py": "evals/fleet/hosted_c4_addon_package_v1.py"' in source
