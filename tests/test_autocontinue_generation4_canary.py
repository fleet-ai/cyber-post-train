from __future__ import annotations

import copy
import os
import stat
import subprocess
from pathlib import Path

import pytest
import yaml

from evals.fleet import autocontinue_generation3_executable_package_v2 as semantic
from evals.fleet import autocontinue_generation4_canary as generation4
from evals.fleet import autocontinue_generation4_executable_package_v1 as package

ROOT = Path.cwd()


@pytest.fixture(scope="module")
def built() -> dict:
    return package.build_package(ROOT)


def test_incident_and_tombstones_bind_zero_side_effects_and_both_causes() -> None:
    incident = generation4.load(ROOT / generation4.INCIDENT_PATH)
    tombstones = generation4.load(ROOT / generation4.TOMBSTONE_PATH)
    generation4.validate_incident(incident, ROOT)
    generation4.validate_tombstones(tombstones, ROOT)
    assert incident["side_effects"] == {
        "model_calls": 0,
        "sessions": 0,
        "verifier_executions": 0,
        "generation3_claims": 0,
        "output_roots": 0,
    }
    assert incident["package_dependency_proof"]["core_b_omitted_required_paths"] == [
        generation4.generation3_authority.SEMANTIC_HELD_PATH,
        semantic.MANIFEST_PATH,
        semantic.RUN_PATH,
        semantic.SUBMIT_PATH,
    ]
    assert not incident["source_ordering_proof"][
        "working_directory_changed_to_repo_before_module_validation"
    ]


@pytest.mark.parametrize("field", ["sessions", "model_calls"])
def test_incident_resealed_side_effect_tamper_rejects(field: str) -> None:
    receipt = generation4.load(ROOT / generation4.INCIDENT_PATH)
    changed = copy.deepcopy(receipt)
    changed["side_effects"][field] = 1
    changed["receipt_sha256"] = generation4.digest(changed, "receipt_sha256")
    with pytest.raises(ValueError, match="incident"):
        generation4.validate_incident(changed, ROOT)


def test_generation4_specs_preserve_cells_and_use_fresh_execution_identities() -> None:
    for model, expected in generation4.EXPECTED.items():
        spec = generation4.load(ROOT / generation4.G4_SPEC_PATHS[model])
        plan = generation4.validate_spec(spec, ROOT)
        old = generation4.load(ROOT / generation4.generation3.G3_SPEC_PATHS[model])
        assert spec["statistical_cell"] == old["statistical_cell"]
        assert spec["execution"]["execution_generation"] == 4
        assert spec["execution"]["execution_id"] != old["execution"]["execution_id"]
        assert plan["scored_job_name"] == expected["job_name"]
        assert plan["attempts"][0]["run_id"] == expected["run_id"]
        assert plan["execution"]["required_priority_class"] == "fleet-serve-low"
        assert plan["execution"]["launch_authorized"] is False


def test_package_includes_missing_dependencies_and_fits_configmaps(built: dict) -> None:
    paths = {
        entry["source_path"]
        for obj in built["object_manifests"].values()
        for entry in obj["entries"]
    }
    assert {
        generation4.generation3_authority.SEMANTIC_HELD_PATH,
        semantic.MANIFEST_PATH,
        semantic.RUN_PATH,
        semantic.SUBMIT_PATH,
    } <= paths
    assert len(built["configmaps"]) == 4
    assert max(built["object_json_bytes"].values()) < package.PACKAGE_OBJECT_LIMIT


def test_runner_changes_to_reconstructed_root_before_module_call(
    built: dict, tmp_path: Path
) -> None:
    model = "qwen3.8-27b"
    bootstrap = tmp_path / "bootstrap"
    destination = tmp_path / "reconstructed"
    outside = tmp_path / "outside"
    bin_dir = tmp_path / "bin"
    for directory in (bootstrap, outside, bin_dir):
        directory.mkdir()
    selected = (
        package.CORE_A_NAME,
        package.CORE_B_NAME,
        package.MODEL_NAMES[model],
    )
    for name in selected:
        for key, value in built["configmaps"][name]["data"].items():
            (bootstrap / key).write_text(value)
    marker = tmp_path / "uv-invocation"
    fake_uv = bin_dir / "uv"
    fake_uv.write_text(
        "#!/bin/sh\nprintf '%s\\n%s\\n' \"$PWD\" \"$*\" > \"$CWD_MARKER\"\nexit 42\n"
    )
    fake_uv.chmod(0o755)
    env = dict(os.environ)
    env.update(
        {
            "CYBER_ROOT": str(destination),
            "GENERATION4_BOOTSTRAP": str(bootstrap),
            "GENERATION4_PACKAGE_AGGREGATE_SHA256": built["model_manifests"][model][
                "aggregate_sha256"
            ],
            "CWD_MARKER": str(marker),
            "PATH": f"{bin_dir}:{env['PATH']}",
        }
    )
    result = subprocess.run(
        ["bash", str(ROOT / generation4.RUN_PATH)],
        cwd=outside,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 42
    cwd, invocation = marker.read_text().splitlines()
    assert cwd == str(destination)
    assert "evals.fleet.autocontinue_generation4_executable_package_v1" in invocation


def test_manifest_and_submitter_are_held_only() -> None:
    manifest = yaml.safe_load((ROOT / generation4.MANIFEST_PATH).read_text())
    assert len(manifest["items"]) == 2
    for item in manifest["items"]:
        assert item["metadata"]["annotations"][
            "cyber-post-train.fleet.ai/launch-authorized"
        ] == "false"
        assert item["spec"]["backoffLimit"] == 0
        pod = item["spec"]["template"]["spec"]
        assert pod["priorityClassName"] == "fleet-serve-low"
        assert pod["preemptionPolicy"] == "Never"
    run = (ROOT / generation4.RUN_PATH).read_text()
    assert run.index('cd "$ROOT"') < run.index("uv run")
    submit = (ROOT / generation4.SUBMIT_PATH).read_text()
    assert "only preview is available" in submit
    assert "kubectl delete" not in submit and "kubectl apply" not in submit
    assert (ROOT / generation4.RUN_PATH).stat().st_mode & stat.S_IXUSR
    assert (ROOT / generation4.SUBMIT_PATH).stat().st_mode & stat.S_IXUSR
