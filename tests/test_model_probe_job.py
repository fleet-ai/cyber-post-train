import base64
import hashlib

import pytest
import yaml

from training.cluster_policy import validate_training_manifest
from training.model_probe_job import render_model_probe_job


def test_model_probe_is_queued_exact_and_exercises_training_path():
    digest = "sha256:" + "b" * 64
    job = render_model_probe_job(digest)
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
    assert container["args"][0].count("examples/run_sft.py") == 2
    assert "sft.max_num_epochs=2 sft.max_num_steps=2" in container["args"][0]
    assert "checkpoint_resumed:true" in container["args"][0]
    assert "test -s /mnt/sfs/cyber-post-train/compatibility/b4734de4/nccl.json" in container[
        "args"
    ][0]
    env = {item["name"]: item["value"] for item in container["env"]}
    config = base64.b64decode(env["MODEL_PROBE_CONFIG_B64"])
    assert hashlib.sha256(config).hexdigest() == env["MODEL_PROBE_CONFIG_SHA256"]
    parsed = yaml.safe_load(config)
    assert parsed["policy"]["megatron_cfg"]["expert_model_parallel_size"] == 8
    assert parsed["policy"]["megatron_cfg"]["peft"]["enabled"] is True
    assert parsed["sft"]["max_num_steps"] == 1


def test_model_probe_refuses_mutable_image_reference():
    with pytest.raises(ValueError, match="immutable"):
        render_model_probe_job("latest")
