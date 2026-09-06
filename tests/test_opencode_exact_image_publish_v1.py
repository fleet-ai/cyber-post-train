from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "evals/fleet/cluster/opencode-exact-image-publish-v1.yaml"


def _objects() -> tuple[dict, dict]:
    configmap, job = [row for row in yaml.safe_load_all(MANIFEST.read_text()) if row]
    return configmap, job


def test_exact_image_publication_is_cpu_only_nonpreempting_and_secret_safe() -> None:
    configmap, job = _objects()
    name = "chris-cyber-opencode11827-image-publish-v1"
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
    publisher = pod["containers"][0]
    token = next(row for row in publisher["env"] if row["name"] == "GH_TOKEN")
    assert token == {
        "name": "GH_TOKEN",
        "valueFrom": {
            "secretKeyRef": {"name": "img-build-secrets", "key": "GH_TOKEN"}
        },
    }
    assert "sk_" not in MANIFEST.read_text()


def test_publication_requires_exact_archive_and_image_identity_before_push() -> None:
    configmap, _ = _objects()
    script = configmap["data"]["publish.sh"]
    assert "578ff2a933f17a19d22ebf8634533651cec2f0ce611e16f3c5fe8efea7940cd3" in script
    assert "sha256:ca4f0b8f50bd051d709c7c0ae5ec47ca31bbff7d2a2ad754c67b9cdf585567cb" in script
    assert "test \"$inspect\" = \"$expected_image_id linux amd64 node /workspace\"" in script
    assert "opencode --version" in script
    assert "docker logout ghcr.io" in script
    assert "docker manifest inspect \"$repo_digest\"" in script
    assert "PUBLISHED.json" in script
    assert "rm -f \"$archive\" \"$ready\"" in script
