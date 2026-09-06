from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "evals/fleet/cluster/opencode-exact-image-stage-v1.yaml"


def _objects() -> tuple[dict, dict]:
    configmap, job = [row for row in yaml.safe_load_all(MANIFEST.read_text()) if row]
    return configmap, job


def test_stage_is_cpu_only_nonpreempting_secret_free_and_create_once() -> None:
    configmap, job = _objects()
    name = "chris-cyber-opencode11827-image-stage-v1"
    assert configmap["metadata"] == {"name": name, "namespace": "fleet-train-jobs"}
    assert configmap["immutable"] is True
    assert job["metadata"]["name"] == name
    pod = job["spec"]["template"]["spec"]
    assert pod["priorityClassName"] == "fleet-serve-low"
    assert pod["preemptionPolicy"] == "Never"
    assert pod["nodeSelector"] == {
        "kubernetes.io/arch": "amd64",
        "workload": "fleetai-training-ng-cpu",
    }
    assert "nvidia.com/gpu" not in MANIFEST.read_text()
    assert "secretKeyRef" not in MANIFEST.read_text()
    assert job["spec"]["backoffLimit"] == 0


def test_stage_verifies_exact_bytes_and_emits_success_or_failure_receipt() -> None:
    configmap, _ = _objects()
    script = configmap["data"]["stage.sh"]
    assert "578ff2a933f17a19d22ebf8634533651cec2f0ce611e16f3c5fe8efea7940cd3" in script
    assert "sha256:ca4f0b8f50bd051d709c7c0ae5ec47ca31bbff7d2a2ad754c67b9cdf585567cb" in script
    assert 'test "$inspect" = "$expected_image_id linux amd64 node /workspace"' in script
    assert "opencode --version" in script
    assert 'chmod 0444 "$archive"' in script
    assert 'test ! -e "$archive"' in script
    assert "STAGED.json" in script
    assert "TERMINAL.json" in script
    assert '"last_completed_stage": sys.argv[2]' in script
    assert "prompts_traces_flags_or_scores_included" in script
