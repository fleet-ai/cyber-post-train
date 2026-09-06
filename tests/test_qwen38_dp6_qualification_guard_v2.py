from __future__ import annotations

import copy
import os
import subprocess
import sys
from pathlib import Path

import pytest

from evals.fleet import qwen38_dp6_e_scorefree_v1 as held
from evals.fleet import qwen38_dp6_metric_observer_v4 as observer
from evals.fleet import qwen38_dp6_qualification_guard_v2 as guard

ROOT = Path(__file__).resolve().parents[1]


def _binding() -> dict[str, str]:
    return {
        "api_run_id": "ft-run-example",
        "head_pod_name": "server-head",
        "head_pod_uid": "11111111-1111-4111-8111-111111111111",
        "service_uid": "22222222-2222-4222-8222-222222222222",
        "receipt_sha256": "sha256:" + "a" * 64,
    }


def _producer() -> tuple[dict, dict, dict]:
    binding = _binding()
    baseline = observer.baseline_observation(
        7,
        server_run_dir=held.RUN_DIR,
        pod_name=binding["head_pod_name"],
        pod_uid=binding["head_pod_uid"],
        api_run_id=binding["api_run_id"],
        service_uid=binding["service_uid"],
        server_binding_receipt_sha256=binding["receipt_sha256"],
        observed_at_epoch=1,
    )
    state = {observer.STATE_KEY: 7}
    status = observer.observer_status(
        "STABLE_BOUND_COUNTER_BASELINE", "stable_baseline", observed_at_epoch=1
    )
    return baseline, state, status


def _pod(
    name: str,
    uid: str,
    *,
    ready: bool = True,
    restarts: int = 0,
    volume_name: str = "shared",
    mount_path: str = "/shared",
) -> dict:
    return {
        "kind": "Pod",
        "metadata": {"name": name, "uid": uid},
        "spec": {
            "volumes": [
                {
                    "name": volume_name,
                    "persistentVolumeClaim": {"claimName": "sfs-shared"},
                }
            ],
            "containers": [
                {
                    "name": "observer",
                    "volumeMounts": [{"name": volume_name, "mountPath": mount_path}],
                }
            ],
        },
        "status": {
            "phase": "Running",
            "containerStatuses": [{"ready": ready, "restartCount": restarts}],
        },
    }


def test_observer_environment_requires_package_root_not_fleet_directory() -> None:
    dependency = "/tmp/example/evals/fleet"
    assert guard.observer_pythonpath(dependency) == "/tmp/example"
    guard.validate_observer_environment({"PYTHONPATH": "/tmp/example"}, dependency)
    with pytest.raises(ValueError, match="absent from PYTHONPATH"):
        guard.validate_observer_environment({}, dependency)
    with pytest.raises(ValueError):
        guard.observer_pythonpath("relative/evals/fleet")


def test_exact_staged_observer_import_needs_declared_pythonpath(tmp_path: Path) -> None:
    dependency = tmp_path / "deps/evals/fleet"
    dependency.mkdir(parents=True)
    (dependency.parent / "__init__.py").touch()
    (dependency / "__init__.py").touch()
    for version in (1, 2, 3):
        source = ROOT / f"evals/fleet/qwen38_dp6_metric_observer_v{version}.py"
        (dependency / source.name).write_bytes(source.read_bytes())
    script = tmp_path / "qwen38_dp6_metric_observer_v4.py"
    script.write_bytes((ROOT / "evals/fleet/qwen38_dp6_metric_observer_v4.py").read_bytes())
    clean_env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    failed = subprocess.run(
        [sys.executable, str(script), "--help"], capture_output=True, env=clean_env, text=True
    )
    assert failed.returncode != 0
    env = {**clean_env, "PYTHONPATH": str(dependency.parents[1])}
    passed = subprocess.run(
        [sys.executable, str(script), "--help"], capture_output=True, env=env, text=True
    )
    assert passed.returncode == 0


def test_counter_producer_must_be_digest_valid_and_bound_before_authorization() -> None:
    baseline, state, status = _producer()
    value = guard.validate_counter_producer_ready(
        baseline, state, status, _binding(), held.RUN_DIR
    )
    assert value["status"] == "READY_BOUND_STABLE_COUNTER_PRODUCER"
    assert value["state_matches_baseline"] is True
    for target, key, replacement in (
        (baseline, "pod_uid", "drift"),
        (state, observer.STATE_KEY, 8),
        (status, "phase", "failed"),
    ):
        changed = [copy.deepcopy(item) for item in (baseline, state, status)]
        index = (baseline, state, status).index(target)
        changed[index][key] = replacement
        with pytest.raises(ValueError, match="not ready"):
            guard.validate_counter_producer_ready(
                changed[0], changed[1], changed[2], _binding(), held.RUN_DIR
            )


def test_terminal_observer_excludes_target_and_requires_ready_restart_zero() -> None:
    target = _pod("target", "target-uid")
    stable = _pod("stable", "stable-uid")
    assert guard.select_stable_sfs_observer([target, stable], "target-uid") == (
        "stable",
        "stable-uid",
        "/shared",
    )
    for changed in (
        _pod("stable", "stable-uid", ready=False),
        _pod("stable", "stable-uid", restarts=1),
    ):
        with pytest.raises(RuntimeError, match="no stable"):
            guard.select_stable_sfs_observer([target, changed], "target-uid")


def test_observer_path_translates_canonical_sfs_to_bound_mount() -> None:
    assert (
        guard.observer_sfs_path(
            "/shared", "/mnt/sfs/jobs/chris-cyber-evalserve-q38-dp6-g-v1/lifecycle"
        )
        == "/shared/jobs/chris-cyber-evalserve-q38-dp6-g-v1/lifecycle"
    )
    assert (
        guard.observer_sfs_path(
            "/mnt/sfs", "/mnt/sfs/jobs/chris-cyber-evalserve-q38-dp6-g-v1"
        )
        == "/mnt/sfs/jobs/chris-cyber-evalserve-q38-dp6-g-v1"
    )
    for mount, path in (
        ("relative", "/mnt/sfs/jobs/x"),
        ("/", "/mnt/sfs/jobs/x"),
        ("/shared", "/tmp/x"),
        ("/shared", "/mnt/sfs"),
    ):
        with pytest.raises(ValueError):
            guard.observer_sfs_path(mount, path)


@pytest.mark.parametrize("mount", ["/shared/../escape", "/shared/./nested", "//shared"])
def test_observer_path_rejects_non_normalized_mount(mount: str) -> None:
    with pytest.raises(ValueError, match="normalized and bounded"):
        guard.observer_sfs_path(mount, "/mnt/sfs/jobs/example")


def test_observer_selection_rejects_unmounted_or_ambiguous_sfs() -> None:
    missing_mount = _pod("stable", "stable-uid")
    missing_mount["spec"]["containers"][0]["volumeMounts"] = []
    ambiguous = _pod("stable", "stable-uid")
    ambiguous["spec"]["containers"].append(
        {"name": "other", "volumeMounts": [{"name": "shared", "mountPath": "/other"}]}
    )
    for pod in (missing_mount, ambiguous):
        with pytest.raises(RuntimeError, match="no stable"):
            guard.select_stable_sfs_observer([pod], "target-uid")
