import os
import subprocess
from pathlib import Path

from evals.fleet import hosted_glm_rank3_a2a4_c2_package_v1 as package
from evals.fleet import exact_pass4_bulk_runtime_v3 as engine
from evals.fleet import hosted_glm_rank3_a2a4_c2_successor_v1 as successor
from evals.fleet import hosted_glm_rank3_a2a4_c2_successor_v2 as successor_v2
from evals.fleet import hosted_glm_rank3_a2a4_c2_release_package_v3 as release_package_v3
from evals.fleet import self_hosted


def test_successor_is_exact_rank3_tail() -> None:
    engine.validate_bulk_adapter(successor)
    plan = successor.validate_all(Path.cwd())[successor.CONTROLLER]
    assert [(row["selection_rank"], row["attempt"]) for row in plan["attempts"]] == [
        (3, 2),
        (3, 3),
        (3, 4),
    ]
    assert plan["execution"]["endpoint_lease"]["maximum_streams"] == 2
    assert plan["execution"]["required_task_tools"] == ["bash", "submit_report"]
    assert plan["execution"]["preemption_policy"] == "Never"
    assert plan["plan_sha256"] == self_hosted.digest_without(plan, "plan_sha256")


def test_release_package_is_create_once_and_nonpreempting() -> None:
    value = package.render_release(Path.cwd())
    configmap, job = value["objects"]["items"]
    assert configmap["immutable"] is True
    assert configmap["metadata"]["name"] == package.release.CONFIGMAP_NAME
    assert job["metadata"]["name"] == package.release.JOB_NAME
    assert job["spec"]["backoffLimit"] == 0
    assert job["spec"]["template"]["spec"]["preemptionPolicy"] == "Never"
    assert job["spec"]["template"]["spec"]["priorityClassName"] == "fleet-infra-quiet"
    assert "hosted_glm_rank3_a2a4_c2_release_v1" in configmap["data"]["run.sh"]


def test_fresh_v2_adapter_is_engine_complete() -> None:
    engine.validate_bulk_adapter(successor_v2)
    plan = successor_v2.validate_all(Path.cwd())[successor_v2.CONTROLLER]
    assert [(row["selection_rank"], row["attempt"]) for row in plan["attempts"]] == [
        (3, 2),
        (3, 3),
        (3, 4),
    ]
    assert plan["partition"]["failed_v1_effects"] == {
        "claims": 0,
        "model_requests": 0,
        "task_instance_session_verifier_scoring_calls": 0,
    }


def test_v3_release_projected_import_closure(tmp_path: Path) -> None:
    data = release_package_v3.render(Path.cwd())["objects"]["items"][0]["data"]
    target = tmp_path / "evals" / "fleet"
    target.mkdir(parents=True)
    (tmp_path / "evals" / "__init__.py").write_text("")
    (target / "__init__.py").write_text("")
    mapping = {
        "self_hosted.py": "self_hosted.py",
        "runner.py": "opencode_train_sweep_runner.py",
        "endpoint_lease.py": "endpoint_lease.py",
        "predecessor.py": "exact_pass4_bulk_v3.py",
        "engine.py": "exact_pass4_bulk_runtime_v3.py",
        "universe.py": "exact_pass4_universe.py",
        "crypto.py": "exact_pass4_crypto.py",
        "inventory.py": "exact_pass4_task_inventory.py",
        "bulk.py": "hosted_glm_exact_bulk_v1.py",
        "bulk_runtime.py": "hosted_glm_exact_bulk_runtime_v1.py",
        "original_release.py": "hosted_glm_exact_bulk_release_v1.py",
        "prior_successor.py": "hosted_glm_rank3_a2a4_c2_successor_v1.py",
        "successor.py": "hosted_glm_rank3_a2a4_c2_successor_v2.py",
        "prior_runtime.py": "hosted_glm_rank3_a2a4_c2_runtime_v1.py",
        "successor_runtime.py": "hosted_glm_rank3_a2a4_c2_runtime_v2.py",
        "prior_release.py": "hosted_glm_rank3_a2a4_c2_release_v1.py",
        "release.py": "hosted_glm_rank3_a2a4_c2_release_v3.py",
        "fixed_proxy.py": "fixed_proxy.py",
    }
    for source_name, target_name in mapping.items():
        (target / target_name).write_text(data[source_name])
    env = {**os.environ, "PYTHONPATH": str(tmp_path)}
    subprocess.run(
        ["python3", "-c", "import evals.fleet.hosted_glm_rank3_a2a4_c2_release_v3"],
        cwd=tmp_path,
        env=env,
        check=True,
    )
