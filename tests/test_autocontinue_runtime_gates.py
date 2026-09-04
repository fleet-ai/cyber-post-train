from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import yaml

from evals.fleet import autocontinue_campaign

POD_TEMPLATE = Path("evals/fleet/cluster/opencode-autocontinue-controller-pod-template-v1.yaml")
FLOCK_MANIFEST = Path("evals/fleet/cluster/opencode-autocontinue-endpoint-flock-preflight-v1.yaml")


def _identity_env(container: dict) -> dict[str, str]:
    return {row["name"]: row["valueFrom"]["fieldRef"]["fieldPath"] for row in container["env"]}


def test_controller_template_packages_exact_downward_job_and_pod_uids() -> None:
    template = yaml.safe_load(POD_TEMPLATE.read_text())
    assert template["kind"] == "PodTemplate"
    assert template["metadata"]["annotations"] == {
        "cyber-post-train.fleet.ai/preview-only": "true",
        "cyber-post-train.fleet.ai/launch-authorized": "false",
    }
    spec = template["template"]["spec"]
    assert spec["priorityClassName"] == "fleet-train-high"
    assert spec["restartPolicy"] == "Never"
    assert _identity_env(spec["containers"][0]) == {
        "JOB_UID": "metadata.labels['batch.kubernetes.io/controller-uid']",
        "POD_UID": "metadata.uid",
    }


def test_flock_preflight_is_two_pod_high_priority_and_held() -> None:
    jobs = list(yaml.safe_load_all(FLOCK_MANIFEST.read_text()))
    assert [job["metadata"]["name"] for job in jobs] == [
        "chris-opencode11827-ac-flock-holder-v1",
        "chris-opencode11827-ac-flock-prober-v1",
    ]
    for job in jobs:
        assert job["metadata"]["annotations"] == {
            "cyber-post-train.fleet.ai/launch-authorized": "false"
        }
        assert job["spec"]["backoffLimit"] == 0
        spec = job["spec"]["template"]["spec"]
        assert spec["priorityClassName"] == "fleet-train-high"
        assert _identity_env(spec["containers"][0]) == {
            "JOB_UID": "metadata.labels['batch.kubernetes.io/controller-uid']",
            "POD_UID": "metadata.uid",
        }


def test_flock_preflight_protocol_proves_exclusion_then_reacquisition(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cross-process"
    module = "evals.fleet.endpoint_lease_preflight"
    holder_env = os.environ | {
        "JOB_UID": str(uuid.uuid4()),
        "POD_UID": str(uuid.uuid4()),
    }
    prober_env = os.environ | {
        "JOB_UID": str(uuid.uuid4()),
        "POD_UID": str(uuid.uuid4()),
    }
    holder = subprocess.Popen(
        [sys.executable, "-m", module, "hold", "--root", str(root)],
        env=holder_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    for _ in range(100):
        if (root / "HOLDER-READY.json").exists():
            break
        time.sleep(0.01)
    else:
        holder.kill()
        raise AssertionError("holder did not acquire the test lock")
    prober = subprocess.run(
        [sys.executable, "-m", module, "probe", "--root", str(root)],
        env=prober_env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    holder_stdout, holder_stderr = holder.communicate(timeout=10)
    assert (prober.returncode, prober.stdout, prober.stderr) == (0, "", "")
    assert (holder.returncode, holder_stdout, holder_stderr) == (0, "", "")
    terminal = json.loads((root / "TERMINAL.json").read_text())
    assert terminal["contention_excluded_second_pod"] is True
    assert terminal["slot_reacquired_after_release"] is True
    assert terminal["scores_included"] is False
    assert terminal["prompts_or_traces_included"] is False


def test_campaign_binds_runtime_files_and_keeps_launches_held() -> None:
    campaign = autocontinue_campaign.load_object(
        Path("evals/fleet/configs/q38-glm53-opencode-autocontinue-primary-campaign-v1.json")
    )
    summary = autocontinue_campaign.validate_campaign(campaign)
    runtime = campaign["controller_runtime"]
    assert summary["launch_authorized"] is False
    assert runtime["shared_pvc_cross_pod_flock_preflight_passed"] is False
    assert runtime["shared_pvc_cross_pod_flock_preflight_receipt_sha256"] is None
    assert all(row["launch_authorized"] is False for row in campaign["partitions"])
