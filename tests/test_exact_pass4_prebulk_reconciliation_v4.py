from __future__ import annotations

import copy
import subprocess
from pathlib import Path

import pytest
import yaml

from evals.fleet import exact_pass4_dedicated_bulk_v4 as bulk
from evals.fleet import exact_pass4_prebulk_reconciliation_package_v4 as package
from evals.fleet import exact_pass4_prebulk_reconciliation_v4 as reconcile

ROOT = Path.cwd()


def test_plan_bindings_cover_exact_798_with_split_canaries() -> None:
    binding = reconcile.plan_bindings(ROOT)
    assert binding["planned_execution_count"] == 798
    assert set(binding["controller_plan_sha256"]) == {
        row["name"] for row in bulk.load(ROOT / bulk.SPEC_PATH)["controllers"]
    }
    assert set(binding["scored_canary_plan_sha256"]) == {"A", "B"}
    assert len(binding["checked_job_names"]) == 10
    assert len(binding["checked_output_roots"]) == 10


def test_release_is_cpu_only_and_binds_exact_authority() -> None:
    release = reconcile.build_release(ROOT, "1" * 40)
    reconcile.validate_release(release, ROOT)
    assert release["authorized_jobs"] == [reconcile.SOURCE_JOB, reconcile.ACCEPT_JOB]
    assert release["cpu_only"] is True
    assert release["authorized_job_resource_policy"] == {
        "gpus": 0,
        "priority_class": "fleet-serve-low",
        "preemption_policy": "Never",
    }
    assert release["scored_model_requests"] == 0
    broken = copy.deepcopy(release)
    broken["authority"]["planned_execution_count"] = 797
    with pytest.raises(ValueError, match="release drifted"):
        reconcile.validate_release(broken, ROOT)


def test_package_is_immutable_bounded_release_free_and_materializes(tmp_path: Path) -> None:
    built = package.build_package(ROOT)
    assert built["launch_authorized"] is False
    assert built["release_included"] is False
    assert set(built["configmaps"]) == {
        package.CORE_A_NAME,
        package.CORE_B_NAME,
        package.OBSERVER_NAME,
    }
    assert all(value["immutable"] is True for value in built["configmaps"].values())
    assert all(size < package.SAFETY_LIMIT for size in built["object_json_bytes"].values())
    projected = tmp_path / "projected"
    projected.mkdir()
    for configmap in built["configmaps"].values():
        for key, value in configmap["data"].items():
            target = projected / key
            assert not target.exists()
            target.write_text(value)
    destination = tmp_path / "materialized"
    materialized = package.materialize(
        projected, destination, built["aggregate_sha256"]
    )
    assert materialized == built["manifest"]
    assert (destination / package.MODULE_PATH).is_file()


def test_manifest_is_cpu_only_never_preempted_and_create_once() -> None:
    manifest = yaml.safe_load((ROOT / package.MANIFEST_PATH).read_text())
    jobs = [item for item in manifest["items"] if item["kind"] == "Job"]
    assert {job["metadata"]["name"] for job in jobs} == {
        reconcile.SOURCE_JOB,
        reconcile.ACCEPT_JOB,
    }
    for job in jobs:
        pod = job["spec"]["template"]["spec"]
        assert pod["priorityClassName"] == "fleet-serve-low"
        assert pod["preemptionPolicy"] == "Never"
        assert pod["restartPolicy"] == "Never"
        assert pod["nodeSelector"]["workload"] == "fleetai-training-ng-cpu"
        assert all(
            "nvidia.com/gpu" not in container.get("resources", {}).get("limits", {})
            for container in pod["containers"]
        )
        assert job["spec"]["backoffLimit"] == 0
        sources = pod["volumes"][0]["projected"]["sources"]
        for name, path in (
            ("chris-q38-ac-r004-a1-g7-run-v1", "qwen-g7-release.json"),
            ("chris-glm53-ac-r013-a1-g7-run-v1", "glm-g7-release.json"),
        ):
            source = next(row["configMap"] for row in sources if row["configMap"]["name"] == name)
            assert source["items"] == [{"key": "release.json", "path": path}]


def test_prebulk_scripts_are_syntax_valid_and_have_no_unconditional_hold() -> None:
    paths = [
        ROOT / package.RUN_PATH,
        ROOT / package.SUBMIT_PATH,
        ROOT / bulk.SUBMIT_PATH,
    ]
    for path in paths:
        subprocess.run(["bash", "-n", str(path)], check=True)
        assert "exit 78" not in path.read_text()
