from __future__ import annotations

import os
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest
import yaml

from evals.fleet import exact_pass4_bulk_package_v3 as package
from evals.fleet import exact_pass4_bulk_release_renderer_v3 as renderer
from evals.fleet import exact_pass4_bulk_runtime_v3 as runtime
from evals.fleet import exact_pass4_bulk_v3 as bulk
from evals.fleet import exact_pass4_universe as exact
from evals.fleet import self_hosted

ROOT = Path.cwd()


def _seal(body: dict[str, object]) -> dict[str, object]:
    return {
        **body,
        "receipt_sha256": self_hosted.sha256(self_hosted.canonical_json(body)),
    }


def test_global_claim_filename_is_canonical_and_strict() -> None:
    execution_id = "sha256:" + "a" * 64
    assert runtime.claim_filename(execution_id) == "a" * 64 + ".json"
    for invalid in ("", "a" * 64, "sha256:../escape", "sha256:" + "g" * 64):
        with pytest.raises(ValueError, match="sha256 digest"):
            runtime.claim_filename(invalid)


def test_four_partitions_cover_exact_universe_minus_two_canaries() -> None:
    plans = bulk.validate_all(ROOT)
    assert {key: plan["new_session_count"] for key, plan in plans.items()} == {
        "qwen-a": 199,
        "qwen-b": 200,
        "glm-a": 199,
        "glm-b": 200,
    }
    all_rows = [row for plan in plans.values() for row in plan["attempts"]]
    keys = {
        (plan["model"]["served_id"], row["selection_rank"], row["attempt"])
        for plan in plans.values()
        for row in plan["attempts"]
    }
    assert len(all_rows) == len(keys) == 798
    assert ("qwen3.8-27b", 4, 1) not in keys
    assert ("glm-5.3", 13, 1) not in keys
    assert sum(key[0] == "qwen3.8-27b" for key in keys) == 399
    assert sum(key[0] == "glm-5.3" for key in keys) == 399


def test_canary_remainders_and_complete_task_boundaries_are_exact() -> None:
    plans = bulk.validate_all(ROOT)
    for controller, canary_rank in (("qwen-a", 4), ("glm-a", 13)):
        attempts = [
            row["attempt"]
            for row in plans[controller]["attempts"]
            if row["selection_rank"] == canary_rank
        ]
        assert attempts == [2, 3, 4]
    for controller, plan in plans.items():
        by_rank: dict[int, list[int]] = {}
        for row in plan["attempts"]:
            by_rank.setdefault(row["selection_rank"], []).append(row["attempt"])
        partial_rank = 4 if controller == "qwen-a" else 13 if controller == "glm-a" else None
        for rank, attempts in by_rank.items():
            assert attempts == ([2, 3, 4] if rank == partial_rank else [1, 2, 3, 4])


def test_rank_interleave_and_endpoint_caps_are_fixed() -> None:
    plans = bulk.validate_all(ROOT)
    assert set(bulk.CONTROLLERS["qwen-a"]["full_ranks"]) == set(range(1, 98, 2))
    assert set(bulk.CONTROLLERS["glm-a"]["full_ranks"]) == set(range(2, 100, 2))
    for model_prefix in ("qwen", "glm"):
        selected = [plans[f"{model_prefix}-a"], plans[f"{model_prefix}-b"]]
        assert all(plan["execution"]["workers"] == 1 for plan in selected)
        assert all(plan["execution"]["endpoint_lease"]["maximum_streams"] == 2 for plan in selected)
        assert len({plan["serving_block"] for plan in selected}) == 1


def test_execution_ids_run_ids_and_cell_ids_are_globally_unique() -> None:
    plans = bulk.validate_all(ROOT)
    rows = [row for plan in plans.values() for row in plan["attempts"]]
    for field in ("cell_id", "execution_id", "run_id", "network"):
        values = [row[field] for row in rows]
        assert len(values) == len(set(values)) == 798
    assert all(row["execution_generation"] == 1 for row in rows)


def test_attempt_config_projects_all_three_session_duplicate_identities() -> None:
    plan = bulk.validate_all(ROOT)["qwen-a"]
    item = plan["attempts"][0]
    plan = {
        **plan,
        "source_job_id": plan["job_name"],
        "authority": {},
        "harness": {},
        "execution": {
            **plan["execution"],
            "required_task_tool_catalog_sha256": "sha256:" + "a" * 64,
        },
    }
    task = {
        "task": {"key": item["task_key"], "version_id": item["task_version_id"]},
        "environment": {"version_id": item["environment_version_id"]},
        "verifier": {},
    }
    config = runtime._attempt_config(plan, task, item)  # noqa: SLF001
    assert config["run_id"] == item["run_id"]
    assert self_hosted.session_execution_metadata(config) == {
        "cell_id": item["cell_id"],
        "execution_id": item["execution_id"],
    }


def test_treatment_and_failure_policy_are_frozen() -> None:
    plans = bulk.validate_all(ROOT)
    for plan in plans.values():
        assert plan["treatment"] == exact.EXPECTED_TREATMENT
        assert plan["treatment"]["tools"] == ["bash", "submit_report"]
        assert plan["treatment"]["context_window_size"] == 262144
        assert plan["treatment"]["context_management"].endswith("autocontinue_v1")
        execution = plan["execution"]
        assert execution["global_execution_claim_before_model_call"] is True
        assert execution["automatic_retry"] is False
        assert execution["future_nonzero_exit_policy"] == bulk.NONZERO_EXIT_POLICY
        assert execution["tail_survives_single_cell_failure"] is True
        assert execution["priority_class"] == "fleet-serve-low"
        assert execution["preemption_policy"] == "Never"


def test_held_receipt_is_valid_and_release_is_impossible_without_all_gates() -> None:
    held = bulk.load(ROOT / bulk.HELD_PATH)
    bulk.validate_held(held, ROOT)
    assert any("generation5_pre_model_failures" in gate for gate in held["required_gates"])
    with pytest.raises(ValueError):
        bulk.validate_release(
            {
                "schema_version": bulk.RELEASE_SCHEMA,
                "status": "RELEASED",
                "launch_authorized": True,
                "gate_receipts": {},
            },
            ROOT,
        )


def test_release_builder_uses_only_fixed_gate_paths_and_seals_before_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: list[dict[str, object]] = []
    monkeypatch.setattr(
        bulk,
        "validate_release",
        lambda release, *_args, **_kwargs: observed.append(release),
    )
    release = bulk.build_release(
        ROOT,
        package_commit="a" * 40,
        released_at="2026-09-05T12:00:00Z",
        evidence_root=tmp_path,
        collector_job={},
        collector_pods={},
    )
    assert observed == [release]
    assert release["gate_receipts"] == bulk.FIXED_GATE_PATHS
    assert release["receipt_sha256"] == self_hosted.digest_without(release, "receipt_sha256")


def test_inventory_producer_receipt_is_bound_and_summary_only_canary_is_rejected() -> None:
    campaign = exact.read_object(ROOT / bulk.CAMPAIGN_PATH)
    universe = exact.build_universe(campaign, ROOT)
    selection = exact.validate_selection(campaign, ROOT)
    tasks = [
        {
            "selection_rank": row["rank"],
            "task": {"key": row["task_key"], "version_id": row["task_version_id"]},
            "environment": {"version_id": row["environment_version_id"]},
        }
        for row in selection
    ]
    inventory = _seal(
        {
            "schema_version": bulk.INVENTORY_GATE_SCHEMA,
            "status": "PASSED",
            "campaign_id": exact.EXPECTED_CAMPAIGN_ID,
            "campaign_sha256": self_hosted.sha256(self_hosted.canonical_json(campaign)),
            "universe_sha256": universe["universe_sha256"],
            "selection_sha256": exact.EXPECTED_SELECTION_SHA256,
            "models": exact.EXPECTED_MODELS,
            "task_count": 100,
            "cell_counts": {"qwen3.8-27b": 400, "glm-5.3": 400},
            "total_cell_count": 800,
            "attempts": [1, 2, 3, 4],
            "expected_inventory_sha256": __import__(
                "evals.fleet.exact_pass4_task_inventory", fromlist=["prepare_expected"]
            ).prepare_expected(ROOT)["expected_sha256"],
            "task_bindings_sha256": self_hosted.sha256(self_hosted.canonical_json(tasks)),
            "binding_mismatch_count": 0,
            "tasks": tasks,
            "fleet_account": {
                "team_name": "fleet",
                "team_id": "a1025f0b-ad67-49fc-a023-51800ab43e84",
            },
            "request_counts": {
                "account_get": 1,
                "exact_task_version_get": 100,
                "redirects_followed": 0,
                "post_put_patch_delete": 0,
                "model_or_scoring_calls": 0,
                "session_calls": 0,
            },
            "runtime": {
                "job_uid": "11111111-1111-4111-8111-111111111111",
                "pod_uid": "22222222-2222-4222-8222-222222222222",
            },
            "privacy": {
                "task_payloads_persisted": False,
                "task_content_included": False,
                "prompts_traces_flags_or_scores_included": False,
                "credentials_included": False,
            },
        }
    )
    with pytest.raises(ValueError, match="inventory gate"):
        bulk.validate_inventory_gate(inventory, ROOT)

    fake_summary = _seal(
        {
            "schema_version": bulk.CANARY_GATE_SCHEMA,
            "status": "ACCEPTED",
            "model": "qwen3.8-27b",
            "job_succeeded": True,
            "pod_restarts": 0,
        }
    )
    with pytest.raises(ValueError, match="evidence paths"):
        bulk.validate_canary_gate(fake_summary, "qwen3.8-27b", ROOT)


def test_manifest_is_held_only_and_has_four_nonpreempting_one_worker_jobs() -> None:
    manifest = yaml.safe_load((ROOT / bulk.MANIFEST_PATH).read_text())
    assert len(manifest["items"]) == 4
    assert {item["metadata"]["name"] for item in manifest["items"]} == {
        value["job_name"] for value in bulk.CONTROLLERS.values()
    }
    for item in manifest["items"]:
        assert item["metadata"]["labels"]["kueue.x-k8s.io/queue-name"] == "training-lq"
        assert item["metadata"]["labels"]["cyber-post-train.fleet.ai/owner"] == "chris"
        assert (
            item["metadata"]["annotations"]["cyber-post-train.fleet.ai/launch-authorized"]
            == "false"
        )
        pod = item["spec"]["template"]["spec"]
        assert pod["priorityClassName"] == "fleet-serve-low"
        assert pod["preemptionPolicy"] == "Never"
        assert len(pod["containers"]) == 1
        assert "exit 78" in pod["containers"][0]["args"][0]


def test_released_job_is_queue_managed_nonpreempting_and_restartable() -> None:
    plan = {"sfs_root": "/mnt/sfs/jobs/test-only"}
    job = renderer._job(  # noqa: SLF001 - the renderer contract is the unit under test
        "qwen-a",
        {
            "release_receipt_sha256": "sha256:" + "a" * 64,
            "package_aggregate_sha256": "sha256:" + "b" * 64,
            "package_commit": "c" * 40,
            "runtime_gates": {
                "BULK_INVENTORY_GATE_PATH": "/mnt/sfs/jobs/inventory/TERMINAL.json",
                "BULK_INVENTORY_GATE_SHA256": "sha256:" + "d" * 64,
            },
        },
        plan,
    )
    labels = job["metadata"]["labels"]
    assert labels == {
        "cyber-post-train.fleet.ai/owner": "chris",
        "cyber-post-train.fleet.ai/experiment": "exact100-bulk-v3",
        "kueue.x-k8s.io/queue-name": "training-lq",
    }
    assert job["spec"]["backoffLimit"] == 3
    assert job["spec"]["activeDeadlineSeconds"] == 7_776_000
    pod = job["spec"]["template"]["spec"]
    assert pod["preemptionPolicy"] == "Never"
    assert pod["restartPolicy"] == "Never"
    cli, dind = pod["initContainers"]
    assert cli["name"] == "docker-cli"
    assert cli["image"] == renderer.DIND_IMAGE
    assert renderer.DOCKER_CLI_SHA256 in cli["args"][0]
    assert renderer.DOCKER_BUILDX_SHA256 in cli["args"][0]
    assert dind["name"] == "dind"
    assert dind["image"] == renderer.DIND_IMAGE
    env = {row["name"]: row.get("value") for row in pod["containers"][0]["env"]}
    assert env["PATH"].startswith("/docker-cli/bin:")
    assert env["DOCKER_CONFIG"] == "/workspace/docker-config"
    assert env["BULK_PACKAGE_COMMIT"] == "c" * 40
    assert env["BULK_INVENTORY_GATE_PATH"] == "/mnt/sfs/jobs/inventory/TERMINAL.json"
    cli_mount = next(
        row for row in pod["containers"][0]["volumeMounts"] if row["name"] == "docker-cli"
    )
    assert cli_mount == {
        "name": "docker-cli",
        "mountPath": "/docker-cli",
        "readOnly": True,
    }
    assert next(row for row in pod["volumes"] if row["name"] == "docker-cli") == {
        "name": "docker-cli",
        "emptyDir": {"sizeLimit": "256Mi"},
    }
    assert renderer.DOCKER_CLI_VOLUME_BYTES >= 2 * renderer.DOCKER_CLI_TOTAL_BYTES
    assert str(renderer.DOCKER_CLI_TOTAL_BYTES) in cli["args"][0]
    assert all("apt" not in str(container) for container in pod["initContainers"])


def test_bulk_runner_proves_exact_cli_and_buildx_before_docker_use() -> None:
    run = (ROOT / bulk.RUN_PATH).read_text()
    cli = 'test "$(command -v docker)" = /docker-cli/bin/docker'
    info = "docker info >/dev/null"
    assert run.index(cli) < run.index(renderer.DOCKER_CLI_SHA256) < run.index(info)
    assert run.index(renderer.DOCKER_BUILDX_SHA256) < run.index(info)
    assert str(renderer.DOCKER_CLI_TOTAL_BYTES) in run
    assert "Docker version 27.5.1, build 9f9e405" in run
    assert "v0.20.1" in run
    assert "apt-get" not in run and "curl" not in run


def test_package_closure_validates_from_an_empty_root(tmp_path: Path) -> None:
    built = package.build_package(ROOT)
    for controller, controller_name in package.CONTROLLER_NAMES.items():
        controller_sources = {
            entry["source_path"] for entry in built["object_manifests"][controller_name]["entries"]
        }
        assert controller_sources == {bulk.RUNTIME_PATH, bulk.RUN_PATH}
        bootstrap = tmp_path / f"bootstrap-{controller}"
        isolated = tmp_path / f"isolated-{controller}"
        bootstrap.mkdir()
        isolated.mkdir()
        seen: set[str] = set()
        for name in (*package.CORE_NAMES, controller_name):
            for key, value in built["configmaps"][name]["data"].items():
                assert key not in seen
                seen.add(key)
                (bootstrap / key).write_text(value)
        manifest = package.verify_mounted(bootstrap / "package-manifest.json", bootstrap)
        assert manifest["controller"] == controller
        package.materialize_mounted(bootstrap / "package-manifest.json", bootstrap, isolated)
        (isolated / "evals").mkdir(exist_ok=True)
        (isolated / "evals" / "__init__.py").touch()
        (isolated / "evals" / "fleet" / "__init__.py").touch()
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "evals.fleet.exact_pass4_bulk_v3",
                "validate-held",
                "--repo",
                str(isolated),
                "--receipt",
                str(isolated / bulk.HELD_PATH),
            ],
            cwd=isolated,
            env={**os.environ, "PYTHONPATH": str(isolated)},
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        runtime_help = subprocess.run(
            [sys.executable, "-m", "evals.fleet.exact_pass4_bulk_runtime_v3", "--help"],
            cwd=isolated,
            env={**os.environ, "PYTHONPATH": str(isolated)},
            check=False,
            capture_output=True,
            text=True,
        )
        assert runtime_help.returncode == 0, runtime_help.stderr


def test_package_payload_tamper_is_rejected(tmp_path: Path) -> None:
    built = package.build_package(ROOT)
    controller = "qwen-a"
    controller_name = package.CONTROLLER_NAMES[controller]
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    for name in (*package.CORE_NAMES, controller_name):
        for key, value in built["configmaps"][name]["data"].items():
            (bootstrap / key).write_text(value)
    source_key = package.data_key(bulk.CAMPAIGN_PATH)
    (bootstrap / source_key).write_text("{}\n")
    with pytest.raises(ValueError, match="payload drifted"):
        package.verify_mounted(bootstrap / "package-manifest.json", bootstrap)


def test_submitter_renders_only_from_immutable_package_commit_snapshot() -> None:
    submitter = (ROOT / bulk.SUBMIT_PATH).read_text()
    assert "immutable_submission_snapshot assert-stable" in submitter
    assert "immutable_submission_snapshot materialize" in submitter
    assert 'cd "$SNAPSHOT"' in submitter
    assert '--package-commit-authority "$SOURCE_ROOT"' in submitter
    assert '--evidence-root "$EVIDENCE_ROOT"' in submitter
    assert "observer_uid=${SFS_OBSERVER_UID:?" in submitter
    assert 'get job "$ACCEPT_JOB"' in submitter
    assert 'cmp "$MANIFEST" "$MANIFEST_RECHECK"' in submitter
    assert "prepare-release" in submitter
    assert "build-release" in submitter
    assert "/shared/jobs/$relative" in submitter
    assert "chris-cyber-exact100-pass4-inventory-v2/TERMINAL.json" in submitter
    assert "exact100-inventory-package.json" in submitter
    assert "chris-cyber-exact100-pass4-inventory-v1/TERMINAL.json" not in submitter
    assert bulk.SNAPSHOT_PATH in package.CORE_PATHS


def test_reconciliation_accept_collector_requires_uid_bound_clean_success() -> None:
    receipt = {
        "collector_runtime": {
            "job_uid": "11111111-1111-4111-8111-111111111111",
            "pod_uid": "22222222-2222-4222-8222-222222222222",
        }
    }
    job = {
        "metadata": {
            "name": bulk.RECONCILIATION_ACCEPT_JOB,
            "namespace": "fleet-train-jobs",
            "uid": receipt["collector_runtime"]["job_uid"],
        },
        "status": {"conditions": [{"type": "Complete", "status": "True"}]},
    }
    pods = {
        "items": [
            {
                "metadata": {
                    "uid": receipt["collector_runtime"]["pod_uid"],
                    "ownerReferences": [
                        {
                            "apiVersion": "batch/v1",
                            "kind": "Job",
                            "name": bulk.RECONCILIATION_ACCEPT_JOB,
                            "uid": receipt["collector_runtime"]["job_uid"],
                            "controller": True,
                        }
                    ],
                },
                "status": {
                    "phase": "Succeeded",
                    "initContainerStatuses": [],
                    "containerStatuses": [
                        {
                            "restartCount": 0,
                            "state": {"terminated": {"exitCode": 0}},
                        }
                    ],
                },
            }
        ]
    }
    bulk.validate_reconciliation_collector_completion(receipt, job, pods)
    changed = {**job, "status": {"conditions": [{"type": "Failed", "status": "True"}]}}
    with pytest.raises(ValueError, match="not exclusively Complete"):
        bulk.validate_reconciliation_collector_completion(receipt, changed, pods)
    pods["items"][0]["status"]["containerStatuses"][0]["restartCount"] = 1
    with pytest.raises(ValueError, match="not cleanly Succeeded"):
        bulk.validate_reconciliation_collector_completion(receipt, job, pods)
    pods["items"][0]["status"]["containerStatuses"][0]["restartCount"] = 0
    owner_references = pods["items"][0]["metadata"].pop("ownerReferences")
    with pytest.raises(ValueError, match="not cleanly Succeeded"):
        bulk.validate_reconciliation_collector_completion(receipt, job, pods)
    pods["items"][0]["metadata"]["ownerReferences"] = owner_references
    owner_references[0]["uid"] = "33333333-3333-4333-8333-333333333333"
    with pytest.raises(ValueError, match="not cleanly Succeeded"):
        bulk.validate_reconciliation_collector_completion(receipt, job, pods)


def test_runtime_release_gate_rechecks_exact_798_cell_partition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plans = bulk.validate_all(ROOT)
    remaining = sorted(
        (
            {
                "model": plan["model"]["served_id"],
                "controller": controller,
                "selection_rank": row["selection_rank"],
                "attempt": row["attempt"],
                "task_version_id": row["task_version_id"],
                "cell_id": row["cell_id"],
                "execution_id": row["execution_id"],
                "run_id": row["run_id"],
            }
            for controller, plan in plans.items()
            for row in plan["attempts"]
        ),
        key=lambda row: (row["model"], row["selection_rank"], row["attempt"]),
    )
    execution_ids = sorted(row["execution_id"] for row in remaining)
    inventory = _seal({"kind": "inventory"})
    qwen = _seal(
        {
            "schema_version": bulk.CANARY_GATE_SCHEMA,
            "status": "ACCEPTED",
            "model": "qwen3.8-27b",
        }
    )
    glm = _seal(
        {
            "schema_version": bulk.CANARY_GATE_SCHEMA,
            "status": "ACCEPTED",
            "model": "glm-5.3",
        }
    )
    reconciliation = _seal(
        {
            "schema_version": bulk.RECONCILIATION_GATE_SCHEMA,
            "status": "CLEAR",
            "observer_package_commit": "a" * 40,
            "planned_execution_count": 798,
            "planned_execution_ids_sha256": self_hosted.sha256(
                self_hosted.canonical_json(execution_ids)
            ),
            "remaining_cells": remaining,
            "remaining_cells_sha256": self_hosted.sha256(self_hosted.canonical_json(remaining)),
            "checked_immediately_before_release": True,
            "observer_job_succeeded": True,
            "observer_pod_restarts": 0,
            "fleet_api_collisions": 0,
            "kubernetes_job_or_pod_collisions": 0,
            "sfs_output_collisions": 0,
            "global_claim_collisions": 0,
            "active_or_accepted_cell_collisions": 0,
            "mutation_calls": 0,
            "exact100_inventory": {"receipt_sha256": inventory["receipt_sha256"]},
            "generation7_gate_receipts": {
                "qwen3.8-27b": {"receipt_sha256": qwen["receipt_sha256"]},
                "glm-5.3": {"receipt_sha256": glm["receipt_sha256"]},
            },
        }
    )
    paths = {
        "/mnt/sfs/jobs/inventory.json": inventory,
        "/mnt/sfs/jobs/reconciliation.json": reconciliation,
        "/mnt/sfs/jobs/qwen.json": qwen,
        "/mnt/sfs/jobs/glm.json": glm,
    }
    for key, path in (
        ("BULK_INVENTORY_GATE", "/mnt/sfs/jobs/inventory.json"),
        ("BULK_RECONCILIATION_GATE", "/mnt/sfs/jobs/reconciliation.json"),
        ("BULK_QWEN_CANARY_GATE", "/mnt/sfs/jobs/qwen.json"),
        ("BULK_GLM_CANARY_GATE", "/mnt/sfs/jobs/glm.json"),
    ):
        monkeypatch.setenv(f"{key}_PATH", path)
        monkeypatch.setenv(f"{key}_SHA256", paths[path]["receipt_sha256"])
    monkeypatch.setenv("BULK_PACKAGE_COMMIT", "a" * 40)
    monkeypatch.setattr(bulk, "load", lambda path: paths[str(path)])
    monkeypatch.setattr(bulk, "validate_inventory_gate", lambda *_args: None)
    monkeypatch.setattr(bulk, "validate_all", lambda *_args: plans)
    runtime._runtime_release_gate_check({"repo_root": "."})  # noqa: SLF001

    reconciliation["remaining_cells"] = remaining[:-1]
    reconciliation["receipt_sha256"] = self_hosted.digest_without(reconciliation, "receipt_sha256")
    monkeypatch.setenv("BULK_RECONCILIATION_GATE_SHA256", reconciliation["receipt_sha256"])
    with pytest.raises(RuntimeError, match="release gate chain drifted"):
        runtime._runtime_release_gate_check({"repo_root": "."})  # noqa: SLF001


def _runtime_plan(tmp_path: Path) -> dict[str, object]:
    attempts = [
        {
            "cell_id": "sha256:" + str(index) * 64,
            "execution_id": "sha256:" + str(index + 2) * 64,
            "execution_generation": 1,
            "selection_rank": 1,
            "attempt": index,
            "task_key": "task",
            "task_version_id": "version",
            "run_id": f"chris-q38-ac-bulk-a-r001-a{index}-aaaaaaaa",
            "network": f"q38-ac-bulk-a-r001-a{index}-aaaaaaaa",
        }
        for index in (1, 2)
    ]
    plan: dict[str, object] = {
        "controller": "qwen-a",
        "campaign_id": bulk.CONTROLLERS["qwen-a"]["job_name"],
        "source_job_id": bulk.CONTROLLERS["qwen-a"]["job_name"],
        "repo_root": ".",
        "inventory_receipt": {},
        "model": {"served_id": "qwen3.8-27b"},
        "harness": {},
        "authority": {},
        "tasks": [
            {
                "rank": 1,
                "source_rank": 1,
                "task": {"key": "task", "version_id": "version"},
                "environment": {},
                "verifier": {},
            }
        ],
        "attempts": attempts,
        "execution": {
            "claim_root": str(tmp_path / "claims"),
            "required_task_tool_catalog_sha256": "sha256:" + "a" * 64,
            "endpoint_lease": {
                "lease_root": str(tmp_path / "leases"),
                "endpoint_key": "qwen-hosted-autocontinue-v1",
                "maximum_streams": 2,
            },
        },
    }
    plan["plan_sha256"] = self_hosted.digest_without(plan, "plan_sha256")
    return plan


def test_claim_is_o_excl_and_never_reacquired(tmp_path: Path) -> None:
    plan = _runtime_plan(tmp_path)
    item = plan["attempts"][0]
    first = runtime.claim_cell(
        plan,
        item,
        claim_root=tmp_path / "claims",
        job_uid="11111111-1111-4111-8111-111111111111",
        pod_uid="22222222-2222-4222-8222-222222222222",
    )
    second = runtime.claim_cell(
        plan,
        item,
        claim_root=tmp_path / "claims",
        job_uid="11111111-1111-4111-8111-111111111111",
        pod_uid="22222222-2222-4222-8222-222222222222",
    )
    assert first is not None
    assert second is None
    assert len(list((tmp_path / "claims").glob("*.json"))) == 1


def test_infrastructure_quarantine_continues_tail_and_restart_never_repeats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _runtime_plan(tmp_path)
    monkeypatch.setenv("FLEET_API_KEY", "test-only")
    monkeypatch.setenv("JOB_UID", "11111111-1111-4111-8111-111111111111")
    monkeypatch.setenv("POD_UID", "22222222-2222-4222-8222-222222222222")
    monkeypatch.setattr(bulk, "build_runtime_plan", lambda *_args: plan)
    calls: list[str] = []

    def injected_runner(config: dict, out: Path, _proxy: Path) -> dict:
        calls.append(config["run_id"])
        out.mkdir(parents=True)
        if len(calls) == 1:
            self_hosted.write_json_once(
                out / "session-ingest.json",
                {
                    "status": "failed",
                    "error_type": "FleetRequestError",
                    "error_code": "fleet_http_error",
                    "http_status": 503,
                },
            )
            raise RuntimeError("injected transport failure")
        return {}

    def accepted(_out: Path, config: dict, item: dict, claim: dict, _key: str) -> dict:
        return runtime._seal(  # noqa: SLF001 - construct the exact durable contract
            {
                "schema_version": "fleet-exact-pass4-bulk-cell-accepted-v3",
                "accepted": True,
                "credited": True,
                "retry_allowed": False,
                "controller": "qwen-a",
                "cell_id": item["cell_id"],
                "execution_id": item["execution_id"],
                "run_id": config["run_id"],
                "selection_rank": item["selection_rank"],
                "attempt": item["attempt"],
                "task_key": config["task"]["key"],
                "task_version_id": config["task"]["version_id"],
                "claim_sha256": claim["receipt_sha256"],
                "session_id": "33333333-3333-4333-8333-333333333333",
                "verifier_execution_id": "44444444-4444-4444-8444-444444444444",
                "agent_exit_code": 0,
                "agent_process_exit_success": True,
                "config_sha256": config["config_sha256"],
                "cleanup_completed": True,
                "session_ingest_completed": True,
                "scores_included": False,
                "prompts_or_traces_included": False,
            }
        )

    terminal = runtime.run_controller(
        plan,
        out=tmp_path / "first",
        proxy=tmp_path / "proxy.py",
        model_runner=injected_runner,
        classifier=accepted,
        check_run_absent=lambda *_args: None,
        route_check=lambda *_args: None,
        runtime_gate_check=lambda *_args: None,
    )
    assert terminal["quarantined_cells"] == 1
    assert terminal["accepted_cells"] == 1
    assert terminal["accounted_cells"] == 2
    assert terminal["scientific_complete"] is False
    assert terminal["successor_reconciliation_required"] is True
    assert calls == [row["run_id"] for row in plan["attempts"]]

    # Simulate a Pod loss after all per-cell receipts were durable but before
    # the Job controller observed completion.  A replacement Pod uses the same
    # SFS root, not a conveniently empty test directory.
    (tmp_path / "first" / "TERMINAL.json").unlink()
    repeated: list[str] = []
    restarted = runtime.run_controller(
        plan,
        out=tmp_path / "first",
        proxy=tmp_path / "proxy.py",
        model_runner=lambda config, *_args: repeated.append(config["run_id"]),
        classifier=accepted,
        check_run_absent=lambda *_args: None,
        route_check=lambda *_args: None,
        runtime_gate_check=lambda *_args: None,
    )
    assert repeated == []
    assert restarted["accepted_cells"] == 1
    assert restarted["quarantined_cells"] == 1
    assert restarted["preserved_nonrepeatable_cells"] == 0
    assert restarted["scientific_complete"] is False


def test_same_root_restart_preserves_interrupted_claim_and_runs_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _runtime_plan(tmp_path)
    monkeypatch.setenv("FLEET_API_KEY", "test-only")
    monkeypatch.setenv("JOB_UID", "11111111-1111-4111-8111-111111111111")
    monkeypatch.setenv("POD_UID", "22222222-2222-4222-8222-222222222222")
    monkeypatch.setattr(bulk, "build_runtime_plan", lambda *_args: plan)

    with pytest.raises(SystemExit, match="controller crash"):
        runtime.run_controller(
            plan,
            out=tmp_path / "resume",
            proxy=tmp_path / "proxy.py",
            # A process/Pod loss does not execute the caught-Exception abort
            # path.  The durable claim is therefore the restart authority.
            model_runner=lambda *_args: (_ for _ in ()).throw(SystemExit("controller crash")),
            classifier=lambda *_args: {},
            check_run_absent=lambda *_args: None,
            route_check=lambda *_args: None,
            runtime_gate_check=lambda *_args: None,
        )

    called: list[str] = []

    def accepted(_out: Path, config: dict, item: dict, claim: dict, _key: str) -> dict:
        return runtime._seal(  # noqa: SLF001
            {
                "schema_version": "fleet-exact-pass4-bulk-cell-accepted-v3",
                "accepted": True,
                "credited": True,
                "retry_allowed": False,
                "controller": "qwen-a",
                "cell_id": item["cell_id"],
                "execution_id": item["execution_id"],
                "run_id": config["run_id"],
                "selection_rank": item["selection_rank"],
                "attempt": item["attempt"],
                "task_key": config["task"]["key"],
                "task_version_id": config["task"]["version_id"],
                "claim_sha256": claim["receipt_sha256"],
                "session_id": "33333333-3333-4333-8333-333333333333",
                "verifier_execution_id": "44444444-4444-4444-8444-444444444444",
                "agent_exit_code": 0,
                "agent_process_exit_success": True,
                "config_sha256": config["config_sha256"],
                "cleanup_completed": True,
                "session_ingest_completed": True,
                "scores_included": False,
                "prompts_or_traces_included": False,
            }
        )

    terminal = runtime.run_controller(
        plan,
        out=tmp_path / "resume",
        proxy=tmp_path / "proxy.py",
        model_runner=lambda config, *_args: called.append(config["run_id"]),
        classifier=accepted,
        check_run_absent=lambda *_args: None,
        route_check=lambda *_args: None,
        runtime_gate_check=lambda *_args: None,
    )
    assert called == [plan["attempts"][1]["run_id"]]
    assert terminal["accepted_cells"] == 1
    assert terminal["preserved_nonrepeatable_cells"] == 1
    assert terminal["accounted_cells"] == 2
    assert terminal["scientific_complete"] is False
    assert terminal["successor_reconciliation_required"] is True


def test_complete_local_protocol_mismatch_aborts_instead_of_quarantining(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _runtime_plan(tmp_path)
    monkeypatch.setenv("FLEET_API_KEY", "test-only")
    monkeypatch.setenv("JOB_UID", "11111111-1111-4111-8111-111111111111")
    monkeypatch.setenv("POD_UID", "22222222-2222-4222-8222-222222222222")
    monkeypatch.setattr(bulk, "build_runtime_plan", lambda *_args: plan)
    called: list[str] = []

    def runner(config: dict, out: Path, _proxy: Path) -> dict:
        called.append(config["run_id"])
        out.mkdir(parents=True)
        for name in ("result.json", "reward-result.json", "session-ingest.json", "cleanup.json"):
            self_hosted.write_json_once(out / name, {"complete": True})
        return {}

    with pytest.raises(RuntimeError, match="binding mismatch"):
        runtime.run_controller(
            plan,
            out=tmp_path / "protocol-drift",
            proxy=tmp_path / "proxy.py",
            model_runner=runner,
            classifier=lambda *_args: (_ for _ in ()).throw(
                RuntimeError("bulk attempt local binding mismatch")
            ),
            check_run_absent=lambda *_args: None,
            route_check=lambda *_args: None,
            runtime_gate_check=lambda *_args: None,
        )
    assert called == [plan["attempts"][0]["run_id"]]
    assert list((tmp_path / "protocol-drift" / "quarantine").glob("*.json")) == []


def test_self_hosted_binding_failure_aborts_tail_instead_of_mass_quarantine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _runtime_plan(tmp_path)
    monkeypatch.setenv("FLEET_API_KEY", "test-only")
    monkeypatch.setenv("JOB_UID", "11111111-1111-4111-8111-111111111111")
    monkeypatch.setenv("POD_UID", "22222222-2222-4222-8222-222222222222")
    monkeypatch.setattr(bulk, "build_runtime_plan", lambda *_args: plan)
    called: list[str] = []

    def runner(config: dict, out: Path, _proxy: Path) -> dict:
        called.append(config["run_id"])
        out.mkdir(parents=True)
        self_hosted.write_json_once(
            out / "failure.json",
            {
                "error_type": "RuntimeError",
                "run_id": config["run_id"],
            },
        )
        self_hosted.write_json_once(
            out / "cleanup.json",
            {
                "instance_created": False,
                "instance_closed": False,
                "containers_removed": True,
            },
        )
        raise RuntimeError("exact task prompt/verifier/runtime-seed binding drifted")

    with pytest.raises(RuntimeError, match="binding drifted"):
        runtime.run_controller(
            plan,
            out=tmp_path / "systemic-drift",
            proxy=tmp_path / "proxy.py",
            model_runner=runner,
            classifier=lambda *_args: {},
            check_run_absent=lambda *_args: None,
            route_check=lambda *_args: None,
            runtime_gate_check=lambda *_args: None,
        )
    assert called == [plan["attempts"][0]["run_id"]]
    assert list((tmp_path / "systemic-drift" / "quarantine").glob("*.json")) == []
    assert (tmp_path / "systemic-drift" / "ABORT.json").is_file()

    restarted: list[str] = []
    with pytest.raises(RuntimeError, match="durable systemic abort"):
        runtime.run_controller(
            plan,
            out=tmp_path / "systemic-drift",
            proxy=tmp_path / "proxy.py",
            model_runner=lambda config, *_args: restarted.append(config["run_id"]),
            classifier=lambda *_args: {},
            check_run_absent=lambda *_args: None,
            route_check=lambda *_args: None,
            runtime_gate_check=lambda *_args: None,
        )
    assert restarted == []


def test_ingest_binding_drift_is_not_mislabeled_as_infrastructure(tmp_path: Path) -> None:
    out = tmp_path / "attempt"
    out.mkdir()
    self_hosted.write_json_once(
        out / "session-ingest.json",
        {"status": "failed", "error_type": "ResponseBindingDrift"},
    )
    self_hosted.write_json_once(
        out / "result.json",
        {"session_ingest_status": "failed"},
    )
    assert (
        runtime._infrastructure_evidence(  # noqa: SLF001
            out, RuntimeError("bulk attempt is infrastructure-incomplete")
        )
        is False
    )


def test_preclaim_stage_observer_covers_every_runtime_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _runtime_plan(tmp_path)
    monkeypatch.setenv("FLEET_API_KEY", "test-only")
    monkeypatch.setenv("JOB_UID", "11111111-1111-4111-8111-111111111111")
    monkeypatch.setenv("POD_UID", "22222222-2222-4222-8222-222222222222")
    monkeypatch.setattr(bulk, "build_runtime_plan", lambda *_args: plan)
    observed: list[str] = []

    with pytest.raises(SystemExit, match="stop after claim"):
        runtime.run_controller(
            plan,
            out=tmp_path / "stages",
            proxy=tmp_path / "proxy.py",
            model_runner=lambda *_args: (_ for _ in ()).throw(SystemExit("stop after claim")),
            classifier=lambda *_args: {},
            check_run_absent=lambda *_args: None,
            route_check=lambda *_args: None,
            runtime_gate_check=lambda *_args: None,
            stage_observer=lambda stage, _item: observed.append(stage),
        )

    assert observed == [
        "01-plan-rebuilt",
        "02-runtime-gate-valid",
        "03-output-root-initialized",
        "04-endpoint-lease-acquired",
        "05-run-identity-absent",
        "06-route-valid",
        "07-claim-written",
        "08-model-runner-entered",
    ]


def test_bulk_adapter_interface_fails_closed_before_execution() -> None:
    class Incomplete:
        CONTROLLERS = {}

    with pytest.raises(TypeError, match="SHA256_RE"):
        runtime.validate_bulk_adapter(Incomplete())


def test_transient_ingest_failure_quarantines_only_the_exact_cell(tmp_path: Path) -> None:
    out = tmp_path / "attempt"
    out.mkdir()
    self_hosted.write_json_once(
        out / "session-ingest.json",
        {
            "status": "failed",
            "error_type": "FleetRequestError",
            "error_code": "fleet_http_error",
            "http_status": 503,
        },
    )
    assert (
        runtime._infrastructure_evidence(  # noqa: SLF001
            out, RuntimeError("bulk attempt is infrastructure-incomplete")
        )
        is True
    )


def test_transient_authoritative_read_failure_quarantines_after_complete_local_result(
    tmp_path: Path,
) -> None:
    out = tmp_path / "attempt"
    out.mkdir()
    for name in ("result.json", "reward-result.json", "session-ingest.json", "cleanup.json"):
        self_hosted.write_json_once(out / name, {"present": True})
    assert (
        runtime._infrastructure_evidence(  # noqa: SLF001
            out, self_hosted.FleetRequestError("GET", "/v1/sessions", 503)
        )
        is True
    )


def test_completed_nonzero_agent_exit_is_credited_only_with_full_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_id = "33333333-3333-4333-8333-333333333333"
    verifier_id = "44444444-4444-4444-8444-444444444444"
    config = {
        "campaign_id": bulk.CONTROLLERS["qwen-a"]["job_name"],
        "run_id": "chris-q38-ac-bulk-a-r001-a1-aaaaaaaa",
        "task": {"key": "task", "version_id": "version"},
        "execution": {
            "cell_id": "sha256:" + "b" * 64,
            "execution_id": "sha256:" + "c" * 64,
        },
        "config_sha256": "sha256:" + "a" * 64,
    }
    item = {
        "cell_id": "sha256:" + "b" * 64,
        "execution_id": "sha256:" + "c" * 64,
        "selection_rank": 1,
        "attempt": 1,
    }
    claim = {"receipt_sha256": "sha256:" + "d" * 64}
    rows = {
        "result.json": {
            "run_id": config["run_id"],
            "task_key": "task",
            "task_version_id": "version",
            "agent_termination": "completed",
            "agent_exit_code": 1,
            "session_ingest_status": "completed",
            "session_id": session_id,
            "verifier_execution_id": verifier_id,
            "instance_id": "instance",
        },
        "reward-result.json": {
            "task_key": "task",
            "task_version_id": "version",
            "verifier_execution_id": verifier_id,
            "instance_id": "instance",
            "reward": 0,
        },
        "session-ingest.json": {"status": "completed", "session_id": session_id},
        "cleanup.json": {
            "instance_created": True,
            "instance_closed": True,
            "containers_removed": True,
        },
    }
    for name, value in rows.items():
        self_hosted.write_json_once(tmp_path / name, value)
    monkeypatch.setattr(runtime, "_client", lambda _key: nullcontext(object()))
    session_metadata = {
        "run_id": config["run_id"],
        "cell_id": item["cell_id"],
        "execution_id": item["execution_id"],
    }
    monkeypatch.setattr(
        self_hosted,
        "_task_sessions",
        lambda *_args: [
            {
                "session_id": session_id,
                "status": "completed",
                "model": "expected-model",
                "eval_task_version_id": "version",
                "metadata": session_metadata,
                "verifier_execution": {"id": verifier_id},
            }
        ],
    )
    monkeypatch.setattr(
        self_hosted, "persisted_session_model_identity", lambda _config: "expected-model"
    )
    accepted = runtime._classify_result(  # noqa: SLF001
        tmp_path, config, item, claim, "test-only"
    )
    assert accepted["accepted"] is True
    assert accepted["credited"] is True
    assert accepted["agent_exit_code"] == 1
    assert accepted["agent_process_exit_success"] is False
    assert accepted["authoritative_session_optional_fields"] == {
        "model": "matched",
        "task_version_id": "matched",
        "run_id": "matched",
        "execution_id": "matched",
        "cell_id": "matched",
    }
    session_metadata["execution_id"] = "sha256:" + "e" * 64
    with pytest.raises(RuntimeError, match="session binding drifted"):
        runtime._classify_result(tmp_path, config, item, claim, "test-only")  # noqa: SLF001


def test_classification_accepts_omitted_session_projection_but_rejects_contradiction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = {
        "run_id": "run",
        "campaign_id": bulk.CONTROLLERS["qwen-a"]["job_name"],
        "task": {"key": "task", "version_id": "version"},
        "config_sha256": "sha256:" + "d" * 64,
    }
    item = {
        "cell_id": "sha256:" + "a" * 64,
        "execution_id": "sha256:" + "b" * 64,
        "run_id": "run",
        "selection_rank": 1,
        "attempt": 1,
    }
    claim = {"receipt_sha256": "sha256:" + "c" * 64}
    session_id = "11111111-1111-4111-8111-111111111111"
    verifier_id = "22222222-2222-4222-8222-222222222222"
    rows = {
        "result.json": {
            "run_id": "run",
            "task_key": "task",
            "task_version_id": "version",
            "agent_termination": "completed",
            "agent_exit_code": 0,
            "session_ingest_status": "completed",
            "session_id": session_id,
            "verifier_execution_id": verifier_id,
            "instance_id": "instance",
        },
        "reward-result.json": {
            "task_key": "task",
            "task_version_id": "version",
            "verifier_execution_id": verifier_id,
            "instance_id": "instance",
            "reward": 0,
        },
        "session-ingest.json": {"status": "completed", "session_id": session_id},
        "cleanup.json": {
            "instance_created": True,
            "instance_closed": True,
            "containers_removed": True,
        },
    }
    for name, value in rows.items():
        self_hosted.write_json_once(tmp_path / name, value)
    session = {
        "session_id": session_id,
        "status": "completed",
        "model": None,
        "verifier_execution": {"id": verifier_id},
    }
    monkeypatch.setattr(runtime, "_client", lambda _key: nullcontext(object()))
    monkeypatch.setattr(self_hosted, "_task_sessions", lambda *_args: [session])
    monkeypatch.setattr(
        self_hosted, "persisted_session_model_identity", lambda _config: "qwen3.8-27b"
    )
    accepted = runtime._classify_result(  # noqa: SLF001
        tmp_path, config, item, claim, "test-only"
    )
    assert set(accepted["authoritative_session_optional_fields"].values()) == {"omitted"}
    session["model"] = "contradictory-model"
    with pytest.raises(RuntimeError, match="session binding drifted"):
        runtime._classify_result(tmp_path, config, item, claim, "test-only")  # noqa: SLF001
