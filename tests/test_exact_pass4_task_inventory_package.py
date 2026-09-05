from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from evals.fleet import exact_pass4_task_inventory as inventory
from evals.fleet import exact_pass4_task_inventory_package as package

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_IMAGE = (
    "ghcr.io/astral-sh/uv:python3.12-bookworm@"
    "sha256:9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7"
)


def _committed_repo(root: Path) -> tuple[Path, str]:
    repo = root / "repo"
    repo.mkdir()
    paths = [
        *(source for source, _install in package.PACKAGE_FILES.values()),
        package.MANIFEST_PATH,
        "evals/fleet/scripts/submit_exact_pass4_task_inventory_v1.sh",
    ]
    for relative in paths:
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@fleet.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Fleet Test"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repo, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    return repo, commit


def _project_configmap(configmap: dict, root: Path) -> Path:
    projected = root / "projected"
    generation = projected / "..2026_09_05_00_00_00"
    generation.mkdir(parents=True)
    for key, value in configmap["data"].items():
        (generation / key).write_text(value)
        (projected / key).symlink_to(Path(generation.name) / key)
    return projected


def test_package_reconstructs_empty_root_and_prepare_cli_is_offline(tmp_path: Path) -> None:
    repo, commit = _committed_repo(tmp_path)
    configmap = package.build_configmap(repo, commit)
    projected = _project_configmap(configmap, tmp_path)
    runtime = tmp_path / "runtime"
    summary = package.materialize_projected(
        projected, runtime, expected_commit=commit
    )
    assert summary["installed_file_count"] == 6
    assert summary["package_sha256"].startswith("sha256:")

    expected_path = runtime / "EXPECTED.json"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(runtime)
    subprocess.run(
        [
            sys.executable,
            "-S",
            "-m",
            "evals.fleet.exact_pass4_task_inventory",
            "--root",
            str(runtime),
            "--prepare",
            str(expected_path),
        ],
        cwd=runtime,
        env=environment,
        check=True,
    )
    expected = json.loads(expected_path.read_text())
    assert expected["task_count"] == 100
    assert expected["total_cell_count"] == 800
    assert expected["expected_sha256"] == inventory.digest_without(
        expected, "expected_sha256"
    )


def test_cached_pinned_image_runs_offline_empty_root_bootstrap(tmp_path: Path) -> None:
    if shutil.which("docker") is None:
        pytest.skip("docker is unavailable")
    docker_info = subprocess.run(
        ["docker", "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    if docker_info.returncode:
        pytest.skip("docker daemon is unavailable")
    if subprocess.run(
        ["docker", "image", "inspect", RUNTIME_IMAGE],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode:
        pytest.skip("exact pinned runtime image is not cached; test never pulls mutable inputs")

    repo, commit = _committed_repo(tmp_path)
    configmap = package.build_configmap(repo, commit)
    projected = _project_configmap(configmap, tmp_path)
    workspace = tmp_path / "container-workspace"
    workspace.mkdir()
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network=none",
            "-v",
            f"{projected.resolve()}:/bootstrap:ro",
            "-v",
            f"{workspace.resolve()}:/workspace",
            "-e",
            f"PACKAGE_COMMIT={commit}",
            RUNTIME_IMAGE,
            "/bin/bash",
            "-ceu",
            "python /bootstrap/package.py materialize --projected /bootstrap "
            "--destination /workspace/runtime --package-commit \"$PACKAGE_COMMIT\" >/dev/null; "
            "cd /workspace/runtime; python -S -m evals.fleet.exact_pass4_task_inventory "
            "--root /workspace/runtime --prepare /workspace/runtime/EXPECTED.json",
        ],
        check=True,
    )
    assert json.loads((workspace / "runtime/EXPECTED.json").read_text())["task_count"] == 100


def test_package_and_manifest_bind_create_once_low_resource_get_only_observer(
    tmp_path: Path,
) -> None:
    repo, commit = _committed_repo(tmp_path)
    configmap = package.build_configmap(repo, commit)
    intent = package.build_intent(configmap)
    assert configmap["immutable"] is True
    assert intent["immutable"] is True
    assert set(configmap["data"]) == set(package.PACKAGE_FILES) | {
        "package.json",
        "package_commit",
    }
    packaged = json.loads(configmap["data"]["package.json"])
    package.validate_package_manifest(packaged)
    assert packaged["runtime_contract"]["fleet_methods"] == ["GET"]
    assert packaged["runtime_contract"]["model_or_scoring_calls"] == 0

    manifest = yaml.safe_load((repo / package.MANIFEST_PATH).read_text())
    assert manifest["metadata"]["name"] == intent["data"]["job"]
    assert manifest["spec"]["backoffLimit"] == 0
    pod = manifest["spec"]["template"]["spec"]
    assert pod["priorityClassName"] == "fleet-serve-low"
    assert pod["preemptionPolicy"] == "Never"
    assert pod["restartPolicy"] == "Never"
    assert pod["containers"][0]["image"] == RUNTIME_IMAGE
    assert pod["containers"][0]["command"] == ["/bin/bash", "/bootstrap/run.sh"]
    assert pod["containers"][0]["env"][1]["valueFrom"]["secretKeyRef"] == {
        "name": "chris-cyber-opencode-evals-v2",
        "key": "FLEET_API_KEY",
    }


def test_projected_package_tamper_and_escape_fail_closed(tmp_path: Path) -> None:
    repo, commit = _committed_repo(tmp_path)
    configmap = package.build_configmap(repo, commit)
    projected = _project_configmap(configmap, tmp_path)
    projected.resolve().joinpath("..2026_09_05_00_00_00", "inventory.py").write_text(
        "tampered"
    )
    with pytest.raises(package.PackageError, match="projected_package_file_digest_mismatch"):
        package.materialize_projected(
            projected, tmp_path / "tampered-runtime", expected_commit=commit
        )

    configmap = package.build_configmap(repo, commit)
    second = tmp_path / "second"
    projected = _project_configmap(configmap, second)
    outside = tmp_path / "outside.py"
    outside.write_text("outside")
    (projected / "inventory.py").unlink()
    (projected / "inventory.py").symlink_to(outside)
    with pytest.raises(package.PackageError, match="projected_package_file_unsafe"):
        package.materialize_projected(
            projected, tmp_path / "escaped-runtime", expected_commit=commit
        )


def test_package_reads_commit_objects_not_dirty_worktree(tmp_path: Path) -> None:
    repo, commit = _committed_repo(tmp_path)
    source = repo / "evals/fleet/exact_pass4_task_inventory.py"
    committed = source.read_text()
    source.write_text("dirty and not executable")
    configmap = package.build_configmap(repo, commit)
    assert configmap["data"]["inventory.py"] == committed
    packaged = json.loads(configmap["data"]["package.json"])
    assert packaged["job_manifest"]["sha256"] == package.sha256(
        package.job_manifest_bytes(repo, commit)
    )


def test_submitter_preview_is_server_validated_and_create_free(tmp_path: Path) -> None:
    repo, _commit = _committed_repo(tmp_path)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "kubectl.calls"
    fake = fake_bin / "kubectl"
    fake.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$*\" >>\"$KUBECTL_CALLS\"\n"
        "case \"$*\" in\n"
        "  *\"get secret chris-cyber-opencode-evals-v2\"*) "
        "printf '%s' e0febd8e-94a2-46b0-a0bf-dd6b3154187b;;\n"
        "esac\n"
    )
    fake.chmod(0o755)
    environment = dict(os.environ)
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
    environment["KUBECTL_CALLS"] = str(calls)
    result = subprocess.run(
        [
            "bash",
            str(
                repo
                / "evals/fleet/scripts/submit_exact_pass4_task_inventory_v1.sh"
            ),
            "preview",
        ],
        cwd=repo,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "no objects created" in result.stdout
    recorded = calls.read_text().splitlines()
    assert sum("create --dry-run=server" in row for row in recorded) == 3
    assert not any(" exec " in f" {row} " for row in recorded)
    assert not any(
        " create " in f" {row} " and "--dry-run=server" not in row
        for row in recorded
    )


def test_submitter_rejects_head_change_before_create(tmp_path: Path) -> None:
    repo, _commit = _committed_repo(tmp_path)
    fake_bin = tmp_path / "head-change-bin"
    fake_bin.mkdir()
    calls = tmp_path / "head-change.calls"
    count = tmp_path / "dryrun.count"
    fake = fake_bin / "kubectl"
    fake.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$*\" >>\"$KUBECTL_CALLS\"\n"
        "case \"$*\" in\n"
        "  *\"get secret chris-cyber-opencode-evals-v2\"*) "
        "printf '%s' e0febd8e-94a2-46b0-a0bf-dd6b3154187b;;\n"
        "  *\"create --dry-run=server\"*)\n"
        "    n=0; test ! -f \"$DRYRUN_COUNT\" || n=$(cat \"$DRYRUN_COUNT\")\n"
        "    n=$((n+1)); printf '%s' \"$n\" >\"$DRYRUN_COUNT\"\n"
        "    if test \"$n\" = 3; then git commit --allow-empty -qm head-change; fi;;\n"
        "esac\n"
    )
    fake.chmod(0o755)
    environment = dict(os.environ)
    environment.update(
        {
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "KUBECTL_CALLS": str(calls),
            "DRYRUN_COUNT": str(count),
            "SFS_OBSERVER_UID": "73dabe56-60f8-4879-be9f-365196c502e3",
        }
    )
    result = subprocess.run(
        [
            "bash",
            str(repo / "evals/fleet/scripts/submit_exact_pass4_task_inventory_v1.sh"),
            "submit",
        ],
        cwd=repo,
        env=environment,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    recorded = calls.read_text().splitlines()
    assert sum("create --dry-run=server" in row for row in recorded) == 3
    assert not any(
        " create " in f" {row} " and "--dry-run=server" not in row
        for row in recorded
    )


def test_submitter_rejects_dirty_tree_before_cluster_access(tmp_path: Path) -> None:
    repo, _commit = _committed_repo(tmp_path)
    (repo / "evals/fleet/exact_pass4_task_inventory.py").write_text("dirty")
    fake_bin = tmp_path / "dirty-bin"
    fake_bin.mkdir()
    calls = tmp_path / "dirty.calls"
    fake = fake_bin / "kubectl"
    fake.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$*\" >>\"$KUBECTL_CALLS\"\n"
        "exit 99\n"
    )
    fake.chmod(0o755)
    environment = dict(os.environ)
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
    environment["KUBECTL_CALLS"] = str(calls)
    result = subprocess.run(
        [
            "bash",
            str(repo / "evals/fleet/scripts/submit_exact_pass4_task_inventory_v1.sh"),
            "submit",
        ],
        cwd=repo,
        env=environment,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert not calls.exists()


def test_runner_uses_fixed_create_once_output_and_never_prints_secret() -> None:
    runner = (
        ROOT / "evals/fleet/scripts/run_exact_pass4_task_inventory_v2.sh"
    ).read_text()
    submitter = (
        ROOT / "evals/fleet/scripts/submit_exact_pass4_task_inventory_v1.sh"
    ).read_text()
    assert "OUT_ROOT=/mnt/sfs/jobs/chris-cyber-exact100-pass4-inventory-v2" in runner
    assert "FLEET_API_KEY" in runner
    assert "uv run" not in runner
    assert "httpx" not in runner
    assert "self_hosted" not in (
        ROOT / "evals/fleet/exact_pass4_task_inventory.py"
    ).read_text()
    assert "echo $FLEET_API_KEY" not in runner
    assert "get secret" in submitter
    assert ".data.FLEET_API_KEY" not in submitter
    assert submitter.index('create -f "$intent"') < submitter.index(
        'create -f "$package"'
    )
