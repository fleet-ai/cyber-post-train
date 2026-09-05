import subprocess
import sys
from pathlib import Path

from evals.fleet import hosted_glm_exact_bulk_package_v1 as package
from evals.fleet import hosted_glm_exact_bulk_release_package_v1 as release_package
from evals.fleet import hosted_glm_exact_bulk_runtime_v1 as runtime
from evals.fleet import hosted_glm_exact_bulk_v1 as bulk
from evals.fleet import self_hosted

ROOT = Path(__file__).resolve().parents[1]


def test_exact_394_cell_reviewed_partition_is_disjoint() -> None:
    plans = bulk.validate_all(ROOT)
    assert [plan["new_session_count"] for plan in plans.values()] == [94, 100, 100, 100]
    cells = {
        (row["selection_rank"], row["attempt"])
        for plan in plans.values()
        for row in plan["attempts"]
    }
    assert len(cells) == 394
    assert all((12, attempt) not in cells for attempt in range(1, 5))
    assert (1, 1) not in cells
    assert (13, 1) not in cells
    assert all((1, attempt) in cells for attempt in (2, 3, 4))


def test_treatment_and_bounded_execution_are_frozen() -> None:
    for plan in bulk.validate_all(ROOT).values():
        assert plan["model"]["served_id"] == "glm-5.3"
        assert plan["harness"]["version"] == "1.18.27"
        assert plan["harness"]["context_window_size"] == 262144
        assert plan["harness"]["max_output_tokens"] == 32768
        assert plan["harness"]["compaction_headroom_tokens"] == 20000
        assert plan["execution"]["required_task_tools"] == ["bash", "submit_report"]
        assert plan["execution"]["endpoint_lease"]["maximum_streams"] == 4
        assert plan["execution"]["priority_class"] == "fleet-serve-low"
        assert plan["execution"]["preemption_policy"] == "Never"
        assert plan["launch_authorized"] is False


def test_release_gate_is_fail_closed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(runtime, "RELEASE_PATH", tmp_path / "absent.json")
    monkeypatch.setattr(bulk, "CANARY_ROOT", tmp_path / "absent-canary")
    plan = next(iter(bulk.validate_all(ROOT).values()))
    try:
        runtime.validate_release({**plan, "repo_root": str(ROOT)})
    except ValueError:
        pass
    else:
        raise AssertionError("absent acceptance/release evidence must block bulk")


def test_runtime_hydration_preserves_static_cells() -> None:
    inventory = {
        "receipt_sha256": "sha256:" + "0" * 64,
    }
    # The accepted inventory is deliberately not in the repo.  This regression
    # checks the static plan digest and launch hold without fabricating task data.
    plans = bulk.validate_all(ROOT)
    assert self_hosted.sha256(self_hosted.canonical_json(sorted(plans))).startswith("sha256:")
    assert inventory["receipt_sha256"].startswith("sha256:")


def test_held_package_is_immutable_and_never_launch_authorized() -> None:
    rendered = package.render(ROOT)
    assert rendered["launch_authorized"] is False
    configmap, *jobs = rendered["objects"]["items"]
    assert configmap["immutable"] is True
    assert len(jobs) == 4
    assert len({job["metadata"]["name"] for job in jobs}) == 4
    for job in jobs:
        assert (
            job["metadata"]["annotations"]["cyber-post-train.fleet.ai/launch-authorized"] == "false"
        )
        pod = job["spec"]["template"]["spec"]
        assert pod["priorityClassName"] == "fleet-serve-low"
        assert pod["preemptionPolicy"] == "Never"
        assert pod["volumes"][0]["configMap"]["name"] == package.CONFIGMAP_NAME


def test_exact_bundle_imports_in_isolated_tree(tmp_path: Path) -> None:
    configmap = package.render(ROOT)["objects"]["items"][0]
    for name, relative in package.INSTALL_PATHS.items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(configmap["data"][name])
    (tmp_path / "evals/__init__.py").touch()
    (tmp_path / "evals/fleet/__init__.py").touch()
    code = (
        "import sys; from pathlib import Path; "
        f"sys.path.insert(0, {str(tmp_path)!r}); "
        "from evals.fleet import hosted_glm_exact_bulk_runtime_v1; "
        "from evals.fleet import hosted_glm_exact_bulk_v1 as b; "
        f"b.validate_all(Path({str(tmp_path)!r}))"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_release_observer_is_held_create_once_cpu_job() -> None:
    rendered = release_package.render(ROOT)
    assert rendered["launch_authorized"] is False
    configmap, job = rendered["objects"]["items"]
    assert configmap["immutable"] is True
    assert job["spec"]["backoffLimit"] == 0
    assert job["metadata"]["annotations"]["cyber-post-train.fleet.ai/create-once"] == "true"
    assert job["metadata"]["annotations"]["cyber-post-train.fleet.ai/launch-authorized"] == "false"
    pod = job["spec"]["template"]["spec"]
    assert pod["priorityClassName"] == "fleet-serve-low"
    assert pod["preemptionPolicy"] == "Never"


def test_release_bundle_imports_in_isolated_tree(tmp_path: Path) -> None:
    configmap = release_package.render(ROOT)["objects"]["items"][0]
    for name, relative in release_package.INSTALL_PATHS.items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(configmap["data"][name])
    (tmp_path / "evals/__init__.py").touch()
    (tmp_path / "evals/fleet/__init__.py").touch()
    code = (
        "import sys; from pathlib import Path; "
        f"sys.path.insert(0, {str(tmp_path)!r}); "
        "from evals.fleet import hosted_glm_exact_bulk_release_v1; "
        "from evals.fleet import hosted_glm_exact_bulk_v1 as b; "
        f"b.validate_all(Path({str(tmp_path)!r}))"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
