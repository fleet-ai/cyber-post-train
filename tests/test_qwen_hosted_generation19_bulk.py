import inspect
import subprocess
import sys
from pathlib import Path

from evals.fleet import qwen_hosted_generation19_bulk as g19
from evals.fleet import qwen_hosted_generation19_package as package
from evals.fleet import qwen_hosted_generation19_preflight as preflight

ROOT = Path(__file__).resolve().parents[1]


def test_partition_shape_is_whole_task_and_held() -> None:
    assert len(g19.ALLOWED_RANKS) == 96
    assert not set(g19.ALLOWED_RANKS) & {2, 3, 4, 5}
    assert {len(row["ranks"]) for row in g19.CONTROLLERS.values()} == {48}
    owners = {
        rank: controller
        for controller, authority in g19.CONTROLLERS.items()
        for rank in authority["ranks"]
    }
    assert set(owners) == set(g19.ALLOWED_RANKS)


def test_plan_builder_requires_live_inventory_fixture() -> None:
    # The paid renderer consumes the accepted SFS inventory, while this unit
    # test fixes the immutable source campaign and partition at review time.
    assert (
        ROOT / "evals/fleet/configs/q38-glm53-exact-easiest100-pass4-campaign-v1.json"
    ).is_file()


def test_committed_plans_are_held_and_exact() -> None:
    plans = g19.validate_all(ROOT)
    assert {key: len(plan["attempts"]) for key, plan in plans.items()} == {
        "qwen-a": 192,
        "qwen-b": 192,
    }
    assert all(plan["launch_authorized"] is False for plan in plans.values())


def test_held_package_is_whole_task_and_nonpreemptible() -> None:
    rendered = package.render(ROOT)
    preflight_cm = next(
        row
        for row in rendered["items"]
        if row["kind"] == "ConfigMap" and row["metadata"]["name"] == package.PREFLIGHT_CM
    )
    assert "qwen_hosted_generation18.py" in preflight_cm["data"]
    jobs = [row for row in rendered["items"] if row["kind"] == "Job"]
    assert len(jobs) == 3
    for job in jobs:
        pod = job["spec"]["template"]["spec"]
        assert pod["priorityClassName"] == "fleet-serve-low"
        assert pod["preemptionPolicy"] == "Never"
        assert job["metadata"]["labels"]["kueue.x-k8s.io/queue-name"] == "training-lq"
    scored = [job for job in jobs if job["metadata"]["name"] != package.PREFLIGHT_JOB]
    assert all(
        job["metadata"]["annotations"]["cyber-post-train.fleet.ai/launch-authorized"] == "false"
        for job in scored
    )


def test_preflight_package_has_closed_python_imports(tmp_path: Path) -> None:
    rendered = package.render(ROOT)
    preflight_cm = next(
        row
        for row in rendered["items"]
        if row["kind"] == "ConfigMap" and row["metadata"]["name"] == package.PREFLIGHT_CM
    )
    module_root = tmp_path / "evals" / "fleet"
    module_root.mkdir(parents=True)
    (tmp_path / "evals" / "__init__.py").touch()
    (module_root / "__init__.py").touch()
    for name, value in preflight_cm["data"].items():
        if name.endswith(".py"):
            (module_root / name).write_text(value)
    run_script = preflight_cm["data"]["run_qwen_hosted_generation19_preflight.sh"]
    assert all(name in run_script for name in preflight_cm["data"] if name.endswith(".py"))
    subprocess.run(
        [sys.executable, "-c", "import evals.fleet.qwen_hosted_generation19_preflight"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )


def test_preflight_every_boundary_has_stage_and_failure_receipt() -> None:
    source = inspect.getsource(preflight.run)
    assert all(f'_stage(output, "{stage}")' in source for stage in preflight.STAGES)
    main_source = inspect.getsource(preflight.main)
    assert 'args.output.parent / "FAILED.json"' in main_source
    assert '"error_type": type(exc).__name__' in main_source
    assert '"error_sha256": self_hosted.sha256(str(exc).encode())' in main_source


def test_scored_packages_have_closed_python_imports_and_hydration(tmp_path: Path) -> None:
    rendered = package.render(
        ROOT, "sha256:5e2ea6d4d1417f0db242984f85b3466736dcb29b13272b5187fa340ae90c1ac7"
    )
    for cm in [row for row in rendered["items"] if row["kind"] == "ConfigMap"][1:]:
        module_root = tmp_path / cm["metadata"]["name"] / "evals" / "fleet"
        module_root.mkdir(parents=True)
        (module_root.parent / "__init__.py").touch()
        (module_root / "__init__.py").touch()
        for name, value in cm["data"].items():
            if name.endswith(".py"):
                (module_root / name).write_text(value)
        run_script = cm["data"]["run_qwen_hosted_generation19.sh"]
        assert all(name in run_script for name in cm["data"] if name.endswith(".py"))
        subprocess.run(
            [sys.executable, "-c", "import evals.fleet.qwen_hosted_generation19_runtime"],
            cwd=module_root.parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
