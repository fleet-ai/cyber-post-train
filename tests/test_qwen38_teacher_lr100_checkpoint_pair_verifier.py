"""Offline contract for the bounded LR100 step-20/21 pair verifier Pod."""

from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT / "configs/qualification/qwen38-teacher-lr100-step20-step21-seal-verify-dev-v2.pod.yaml"
)
IMAGE = (
    "661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/skyrl-train@sha256:"
    "ba288751cd227c5be146d28f4a03237545d87d2cbd4c48464945b17fde566ff4"
)


def pod() -> dict:
    return yaml.safe_load(ARTIFACT.read_text(encoding="utf-8"))


def embedded_python() -> str:
    command = pod()["spec"]["containers"][0]["command"][2]
    marker = "PYTHONPATH=/tmp/lr100-pair-verifier-package python3 - <<'PY'\n"
    return command.split(marker, 1)[1].rsplit("\nPY\n", 1)[0]


def test_verifier_is_bounded_c1_zero_gpu_and_read_only() -> None:
    value = pod()
    spec = value["spec"]
    container = spec["containers"][0]
    assert value["apiVersion"] == "v1"
    assert value["kind"] == "Pod"
    assert value["metadata"]["name"] == "chris-q38-lr100-seal-verify-v2"
    assert value["metadata"]["namespace"] == "fleet-train-jobs"
    assert value["metadata"]["annotations"] == {
        "fleet.ai/prepared-offline-only": "true",
        "fleet.ai/create-requires-fresh-auth-and-live-gates": "true",
    }
    assert spec["automountServiceAccountToken"] is False
    assert spec["restartPolicy"] == "Never"
    assert spec["priorityClassName"] == "c1"
    assert spec["activeDeadlineSeconds"] == 3600
    assert spec["terminationGracePeriodSeconds"] == 30
    assert container["image"] == IMAGE
    assert "nvidia.com/gpu" not in container["resources"]["requests"]
    assert "nvidia.com/gpu" not in container["resources"]["limits"]
    assert container["volumeMounts"] == [{"name": "sfs", "mountPath": "/mnt/sfs", "readOnly": True}]
    assert spec["volumes"] == [
        {
            "name": "sfs",
            "persistentVolumeClaim": {"claimName": "sfs-shared", "readOnly": True},
        }
    ]


def test_verifier_main_process_is_the_rehash_not_a_sleeping_exec_target() -> None:
    command = pod()["spec"]["containers"][0]["command"]
    assert command[:2] == ["/bin/sh", "-lc"]
    assert "sleep" not in command[2]
    assert "verify(manifest, check_files=True)" in command[2]
    assert "raw_tensor_values_read" in command[2]
    assert '"gpu_count": 0' in command[2]


def test_embedded_verifier_is_locally_syntax_compiled_with_correct_recipe_key() -> None:
    shell = pod()["spec"]["containers"][0]["command"][2]
    subprocess.run(["/bin/sh", "-n"], input=shell, text=True, check=True)
    source = embedded_python()
    compile(source, str(ARTIFACT), "exec")
    assert 'recipe["lr"]' in source
    assert "learning_rate" not in source
    assert "cyber_qwen38_lr100_checkpoint_pair_verifier_result_v1" in source


def test_verifier_binds_both_exact_seals_and_source_checkpoints() -> None:
    source = embedded_python()
    assert "lr100-dev-v2-checkpoint-seals-v1/step-20.json" in source
    assert "lr100-dev-v2-checkpoint-seals-v1/step-21.json" in source
    assert "54216c0e2a506f39b5437101e604e87c3e753adccf84ecdc937ac0c6177f89b8" in source
    assert "b82fd516e2e3c4bffa99d9cae718e6caaba4d7b9d1679885c41676ef3d754fa0" in source
    assert "f42894e871ee3ec2204e435cdf240d2641706f4d151bd4484b56edcf6716ada3" in source
    assert 'f"global_step_{step}"' in source
    assert "324_627_486_731" in source
    assert 'len(manifest["files"]) != 33' in source
    assert 'manifest["world_size"] != 8' in source
    assert "file_sha256(path) != manifest_file_sha256" in source


def test_proof_is_emitted_only_after_both_full_rehashes() -> None:
    source = embedded_python()
    verify_position = source.index("verify(manifest, check_files=True)")
    append_position = source.index("rows.append(")
    print_position = source.index("print(json.dumps(proof")
    assert verify_position < append_position < print_position
    assert source.count("for item in expected:") == 1
    assert '"mismatch_count": 0' in source
    assert '"full_source_rehash": True' in source
    assert 'proof["receipt_sha256"]' in source
