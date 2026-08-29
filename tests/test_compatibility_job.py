import base64
import hashlib
import zipfile
from io import BytesIO

import pytest

from training.cluster_policy import validate_training_manifest
from training.compatibility_job import render_preflight_job


def test_preflight_is_queued_exact_and_self_contained():
    digest = "sha256:" + "a" * 64
    job = render_preflight_job(digest)
    validate_training_manifest(job)
    assert job["spec"]["suspend"] is True
    template = job["spec"]["template"]
    assert template["metadata"]["annotations"] == {
        "kueue.x-k8s.io/podset-required-topology": "kubernetes.io/hostname"
    }
    pod = template["spec"]
    assert "priorityClassName" not in pod
    container = pod["containers"][0]
    assert container["image"].endswith("@" + digest)
    assert container["resources"]["requests"]["nvidia.com/gpu"] == "8"
    command = container["args"][0]
    assert "uv run --frozen --extra mcore --extra vllm python" in command
    assert "uv run --frozen --extra mcore --extra vllm torchrun" in command
    assert "\npython -m training.compatibility_probe" not in command
    env = {item["name"]: item["value"] for item in container["env"]}
    archive = base64.b64decode(env["PROBE_ARCHIVE_B64"])
    assert hashlib.sha256(archive).hexdigest() == env["PROBE_ARCHIVE_SHA256"]
    with zipfile.ZipFile(BytesIO(archive)) as bundle:
        assert sorted(bundle.namelist()) == [
            "training/__init__.py",
            "training/compatibility_probe.py",
            "training/io.py",
        ]


def test_preflight_refuses_mutable_image_reference():
    with pytest.raises(ValueError, match="immutable"):
        render_preflight_job("latest")
