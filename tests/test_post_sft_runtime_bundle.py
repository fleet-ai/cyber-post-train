import ast
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from evals.webexploitbench.post_sft_evidence import ARTIFACT_CONFIG_MAP_FILES

ROOT = Path(__file__).resolve().parents[1]
MARKER = ROOT / "evals/post_sft/runtime/training__init__.py"
WEB_MARKER = ROOT / "evals/post_sft/runtime/webexploitbench__init__.py"
PLAN = ROOT / "configs/evaluation/qwen36-27b-ft-run-574bd7b3-post-sft.json"


def test_isolated_training_package_marker_has_no_transitive_imports():
    tree = ast.parse(MARKER.read_text())
    assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree))


def test_isolated_webexploitbench_package_marker_has_no_transitive_imports():
    tree = ast.parse(WEB_MARKER.read_text())
    assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree))


def test_every_post_sft_bundle_uses_and_freezes_the_minimal_marker():
    expected = "sha256:" + hashlib.sha256(MARKER.read_bytes()).hexdigest()
    plan = json.loads(PLAN.read_text())
    contracts = (
        plan["staging_execution"],
        plan["cast_execution"],
        plan["evidence_execution"]["base_artifact_inspector"],
        plan["evidence_execution"]["registration"],
    )
    for item in contracts:
        assert item["config_map_code_sha256"]["training/__init__.py"] == expected


def test_frozen_plan_uses_one_authoritative_bf16_cast_destination():
    plan = json.loads(PLAN.read_text())
    destination = plan["export"]["bf16_cast_destination"]
    assert destination == plan["cast_execution"]["destination_path"]


def test_registration_v3_uses_the_proven_ecr_runtime_image():
    plan = json.loads(PLAN.read_text())
    pod = plan["evidence_execution"]["registration"]["pod_spec"]
    assert "imagePullSecrets" not in pod
    assert pod["container"]["image"].startswith(
        "661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/skyrl-train:"
    )
    manifest = (ROOT / "evals/post_sft/cluster/qwen36-sft-register-job.yaml").read_text()
    assert "chris-cyber-qwen36-sft-register-574bd7b3-v3" in manifest
    assert "imagePullSecrets:" not in manifest


def test_base_artifact_inspector_v9_binds_ecr_auth_and_minimal_package():
    plan = json.loads(PLAN.read_text())
    pod = plan["evidence_execution"]["base_artifact_inspector"]["pod_spec"]
    assert pod["serviceAccountName"].endswith("observer-v9")
    assert pod["imagePullSecrets"] == [{"name": "ecr-pull"}]
    assert pod["container"]["image"].startswith(
        "661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/skyrl-train:"
    )
    manifests = list(
        yaml.safe_load_all(
            (ROOT / "evals/post_sft/cluster/qwen36-base-artifact-inspect-job.yaml").read_text()
        )
    )
    job = next(value for value in manifests if value["kind"] == "Job")
    assert job["metadata"]["name"] == "chris-cyber-qwen36-base-artifact-inspect-6a9e13bd-v9"
    assert job["spec"]["template"]["spec"]["imagePullSecrets"] == [{"name": "ecr-pull"}]
    mounted_items = {
        row["key"]: row["path"]
        for row in job["spec"]["template"]["spec"]["volumes"][0]["configMap"]["items"]
        if row["key"] != "post-sft-plan.json"
    }
    assert mounted_items == ARTIFACT_CONFIG_MAP_FILES


def test_base_artifact_inspector_bundle_imports_without_repository_modules(tmp_path):
    package = tmp_path / "evals" / "webexploitbench"
    package.mkdir(parents=True)
    shutil.copyfile(ROOT / "evals/__init__.py", tmp_path / "evals" / "__init__.py")
    shutil.copyfile(WEB_MARKER, package / "__init__.py")
    shutil.copyfile(
        ROOT / "evals/webexploitbench/post_sft_evidence.py",
        package / "post_sft_evidence.py",
    )
    training = tmp_path / "training"
    training.mkdir()
    shutil.copyfile(MARKER, training / "__init__.py")
    for name in (
        "io.py",
        "post_sft_artifacts.py",
        "post_sft_base_surface.py",
        "post_sft_cast.py",
    ):
        shutil.copyfile(ROOT / "training" / name, training / name)
    result = subprocess.run(
        [sys.executable, "-m", "evals.webexploitbench.post_sft_evidence", "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_v4_evidence_and_cast_modules_import_from_only_the_mounted_bundle(tmp_path):
    package = tmp_path / "training"
    package.mkdir()
    shutil.copyfile(MARKER, package / "__init__.py")
    for name in (
        "io.py",
        "post_sft_artifacts.py",
        "post_sft_base_surface.py",
        "post_sft_cast.py",
        "post_sft_staging.py",
        "tokenizer_equivalence.py",
    ):
        shutil.copyfile(ROOT / "training" / name, package / name)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import training.post_sft_artifacts; "
                "import training.post_sft_cast; "
                "import training.post_sft_staging; "
                "import training.tokenizer_equivalence"
            ),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
