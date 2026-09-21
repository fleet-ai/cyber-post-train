from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from cyber_post_train.jobs import digest
from evals.fleet import opencode_self_hosted as self_hosted
from evals.fleet import (
    reviewed_recovery_v2,
    reviewed_recovery_worker_v2,
    rollout_ledger,
    rollout_worker,
    seed44_base_repair_job,
    stored_session_reconciliation_v2,
)


def _sha(character: str) -> str:
    return "sha256:" + character * 64


def _stored_intent_value(**changes) -> dict:
    body = {
        "schema_version": stored_session_reconciliation_v2.INTENT_SCHEMA,
        "evaluation_plan_sha256": "a" * 64,
        "runtime_files_sha256": stored_session_reconciliation_v2.runtime_identity(),
        "serving_block": "base",
        "source_output_root": "/mnt/sfs/jobs/source-base",
        "source_database": "source_base",
        "source_job_uid": str(uuid.uuid4()),
        "source_job_terminal_receipt_sha256": "b" * 64,
        "selected_cell_ids": [str(uuid.uuid4()) for _ in range(5)],
        "expected_agent_exit_code": 0,
        "expected_agent_termination": "output_limit",
        "expected_failure_code": "authoritative_scoring_started.runtimeerror",
    }
    body.update(changes)
    return {**body, "sha256": stored_session_reconciliation_v2._body_digest(body)}  # noqa: SLF001


def _private_file(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    os.chmod(path, 0o600)
    return path


def test_exact_stored_session_intent_binds_output_limit_and_runtime(tmp_path: Path):
    intent = stored_session_reconciliation_v2.load_intent(
        _private_file(tmp_path / "stored.json", _stored_intent_value())
    )
    assert len(intent.selected_cell_ids) == 5
    assert intent.expected_agent_exit_code == 0
    assert intent.expected_agent_termination == "output_limit"
    assert intent.expected_failure_code == "authoritative_scoring_started.runtimeerror"


def test_exact_stored_session_intent_rejects_outcome_drift_and_public_permissions(
    tmp_path: Path,
):
    path = _private_file(
        tmp_path / "bad-outcome.json",
        _stored_intent_value(expected_agent_termination="completed"),
    )
    with pytest.raises(rollout_ledger.LedgerError, match="outcome is unsupported"):
        stored_session_reconciliation_v2.load_intent(path)
    path = _private_file(tmp_path / "public.json", _stored_intent_value())
    path.chmod(0o644)
    with pytest.raises(rollout_ledger.LedgerError, match="group or other"):
        stored_session_reconciliation_v2.load_intent(path)


def _recovery_intent_value(selected_cells: list[dict], **changes) -> dict:
    body = {
        "schema_version": reviewed_recovery_v2.INTENT_SCHEMA,
        "evaluation_plan_sha256": "a" * 64,
        "runtime_files_sha256": reviewed_recovery_v2.runtime_identity(),
        "serving_block": "base",
        "source_output_root": "/mnt/sfs/jobs/source-base",
        "source_database": "source_base",
        "source_job_uid": str(uuid.uuid4()),
        "source_job_terminal_receipt_sha256": "b" * 64,
        "prior_stored_session_intent_sha256": "c" * 64,
        "selected_cells": selected_cells,
    }
    body.update(changes)
    return {**body, "sha256": reviewed_recovery_v2._body_digest(body)}  # noqa: SLF001


def _selected_cell(cell_id: str | None = None) -> dict:
    return {
        "cell_id": cell_id or str(uuid.uuid4()),
        "claim_file_sha256": _sha("1"),
        "binding_file_sha256": _sha("2"),
        "prompt_file_sha256": _sha("3"),
        "failure_file_sha256": _sha("4"),
        "cleanup_file_sha256": _sha("5"),
    }


def test_provisioning_timeout_intent_is_private_self_digesting_and_exact(tmp_path: Path):
    cells = [_selected_cell(), _selected_cell()]
    intent = reviewed_recovery_v2.load_intent(
        _private_file(tmp_path / "recovery.json", _recovery_intent_value(cells))
    )
    assert intent.selected_cell_ids == tuple(row["cell_id"] for row in cells)
    assert intent.selected_index[cells[0]["cell_id"]].failure_file_sha256 == _sha("4")
    changed = _recovery_intent_value(cells)
    changed["selected_cells"][0]["failure_file_sha256"] = _sha("f")
    with pytest.raises(rollout_ledger.LedgerError, match="self digest differs"):
        reviewed_recovery_v2.load_intent(_private_file(tmp_path / "drift.json", changed))


def _write(path: Path, value: bytes) -> str:
    path.write_bytes(value)
    return reviewed_recovery_v2._prefixed_file_sha256(path)  # noqa: SLF001


def _json_bytes(value: dict) -> bytes:
    return self_hosted.canonical_json(value) + b"\n"


def _provisioning_fixture(tmp_path: Path):
    source = tmp_path / "source"
    execution = _sha("6")
    execution_name = execution.removeprefix("sha256:")
    attempt = source / "attempts" / execution_name
    claims = source / "claims"
    attempt.mkdir(parents=True)
    claims.mkdir(parents=True)
    cell_id = str(uuid.uuid4())
    row = {
        "cell_id": cell_id,
        "task_key": "task-key",
        "task_version_id": str(uuid.uuid4()),
        "model_id": "base-model",
        "model_revision": "revision",
        "serving_block": "base",
        "endpoint_model_id": "served-base",
        "harness_id": "opencode-1.18.27",
        "attempt": 1,
    }
    config = {
        "schema_version": "fleet-selfhosted-opencode-ledger-cell-v1",
        "run_id": "synthetic-run-g1",
        "campaign_id": "synthetic-campaign",
        "source_job_id": "source-job",
        "task": {"key": row["task_key"], "version_id": row["task_version_id"]},
        "environment": {"id": "env", "version": "v1"},
        "verifier": {"version_id": "verifier"},
        "authority": {
            "provisioning_route_template": (
                "/v1/rollout-rewards/{task_key}/versions/{task_version_id}/instances"
            )
        },
        "model": {"served_id": row["endpoint_model_id"]},
        "harness": {"name": "opencode", "version": "1.18.27"},
        "execution": {
            "cell_id": _sha("7"),
            "execution_id": execution,
            "execution_generation": 1,
        },
    }
    claim = rollout_worker._claim_receipt(config, row)  # noqa: SLF001
    binding = {
        "schema_version": config["schema_version"],
        "run_id": config["run_id"],
        "source_job_id": config["source_job_id"],
        "task": config["task"],
        "environment": config["environment"],
        "verifier": config["verifier"],
        "authority": config["authority"],
        "authority_gate": {"mode": "synthetic"},
        "model": config["model"],
        "harness": config["harness"],
    }
    failure = {
        "error_type": "FleetRequestError",
        "elapsed_seconds": 1.25,
        "run_id": config["run_id"],
        "method": "POST",
        "route": self_hosted.authoritative_route(config, "provisioning"),
        "http_status": 504,
        "response_sha256": _sha("8"),
        "reason": "unclassified",
    }
    cleanup = {
        "instance_created": False,
        "instance_closed": False,
        "containers_removed": True,
    }
    paths = {
        "claim_file_sha256": claims / f"{execution_name}.json",
        "binding_file_sha256": attempt / "binding.json",
        "prompt_file_sha256": attempt / "prompt.txt",
        "failure_file_sha256": attempt / "failure.json",
        "cleanup_file_sha256": attempt / "cleanup.json",
    }
    values = {
        "claim_file_sha256": _json_bytes(claim),
        "binding_file_sha256": _json_bytes(binding),
        "prompt_file_sha256": b"private prompt\n",
        "failure_file_sha256": _json_bytes(failure),
        "cleanup_file_sha256": _json_bytes(cleanup),
    }
    selected = {"cell_id": cell_id}
    for field, path in paths.items():
        selected[field] = _write(path, values[field])
    return source, row, config, reviewed_recovery_v2.SelectedCell(**selected)


def test_provisioning_observation_requires_exact_504_and_absence(tmp_path: Path, monkeypatch):
    source, row, config, selected = _provisioning_fixture(tmp_path)
    monkeypatch.setattr(reviewed_recovery_v2.self_hosted, "_task_sessions", lambda *_: [])
    observation = reviewed_recovery_v2._observe_cell(  # noqa: SLF001
        row=row,
        selected_cell=selected,
        config=config,
        source_root=source.resolve(),
        client=object(),
    )
    assert observation["provisioning_method"] == "POST"
    assert observation["provisioning_http_status"] == 504
    assert observation["model_execution_artifacts_absent"] is True
    assert observation["scoring_artifacts_absent"] is True
    assert observation["receipt_sha256"] == self_hosted.digest_without(
        observation, "receipt_sha256"
    )

    attempt = source / "attempts" / config["execution"]["execution_id"].removeprefix("sha256:")
    (attempt / "trace-manifest.json").write_text("{}")
    with pytest.raises(rollout_ledger.LedgerError, match="file set differs"):
        reviewed_recovery_v2._observe_cell(  # noqa: SLF001
            row=row,
            selected_cell=selected,
            config=config,
            source_root=source.resolve(),
            client=object(),
        )


def test_provisioning_observation_rejects_an_exact_authoritative_session(
    tmp_path: Path, monkeypatch
):
    source, row, config, selected = _provisioning_fixture(tmp_path)
    monkeypatch.setattr(
        reviewed_recovery_v2.self_hosted,
        "_task_sessions",
        lambda *_: [{"metadata": {"execution_id": config["execution"]["execution_id"]}}],
    )
    with pytest.raises(rollout_ledger.LedgerError, match="authoritative session exists"):
        reviewed_recovery_v2._observe_cell(  # noqa: SLF001
            row=row,
            selected_cell=selected,
            config=config,
            source_root=source.resolve(),
            client=object(),
        )


def test_rollout_repair_uses_fresh_generation_two_execution_identity(tmp_path: Path, monkeypatch):
    original = {
        ("model", str(uuid.uuid4()), 1): {
            "model_id": "model",
            "task_version_id": str(uuid.uuid4()),
            "attempt": 1,
            "initial_execution": {
                "cell_id": _sha("a"),
                "execution_id": _sha("b"),
                "execution_generation": 1,
            },
        }
    }
    monkeypatch.setattr(
        reviewed_recovery_worker_v2.reviewed_recovery_v2,
        "_scientific_index",
        lambda *_: original,
    )
    repaired = reviewed_recovery_worker_v2._repair_universe_index(  # noqa: SLF001
        {}, tmp_path / "plan.csv"
    )
    row = next(iter(repaired.values()))
    cell_id = original[next(iter(original))]["initial_execution"]["cell_id"]
    assert row["initial_execution"] == {
        "cell_id": cell_id,
        "execution_id": "sha256:" + digest({"cell": cell_id, "generation": 2}),
        "execution_generation": 2,
    }
    assert row["initial_execution"] != original[next(iter(original))]["initial_execution"]


ROOT = Path(__file__).resolve().parents[1]
REPAIR_PLAN = ROOT / "configs/evaluation/qwen38-base-fleet-dev17-seed44-narrow-repair-v1.json"


def _packet_intents(tmp_path: Path) -> tuple[Path, Path]:
    plan = json.loads(REPAIR_PLAN.read_text())
    source = plan["source"]
    stored_body = {
        "schema_version": stored_session_reconciliation_v2.INTENT_SCHEMA,
        "evaluation_plan_sha256": source["evaluation_plan_sha256"].removeprefix("sha256:"),
        "runtime_files_sha256": stored_session_reconciliation_v2.runtime_identity(),
        "serving_block": "base",
        "source_output_root": source["evaluation_directory"],
        "source_database": source["database"],
        "source_job_uid": source["job_uid"],
        "source_job_terminal_receipt_sha256": source["terminal_evidence_sha256"].removeprefix(
            "sha256:"
        ),
        "selected_cell_ids": [str(uuid.uuid4()) for _ in range(5)],
        "expected_agent_exit_code": 0,
        "expected_agent_termination": "output_limit",
        "expected_failure_code": "authoritative_scoring_started.runtimeerror",
    }
    stored_value = {
        **stored_body,
        "sha256": stored_session_reconciliation_v2._body_digest(stored_body),  # noqa: SLF001
    }
    selected_cells = [_selected_cell(), _selected_cell()]
    recovery_body = {
        "schema_version": reviewed_recovery_v2.INTENT_SCHEMA,
        "evaluation_plan_sha256": source["evaluation_plan_sha256"].removeprefix("sha256:"),
        "runtime_files_sha256": reviewed_recovery_v2.runtime_identity(),
        "serving_block": "base",
        "source_output_root": source["evaluation_directory"],
        "source_database": source["database"],
        "source_job_uid": source["job_uid"],
        "source_job_terminal_receipt_sha256": source["terminal_evidence_sha256"].removeprefix(
            "sha256:"
        ),
        "prior_stored_session_intent_sha256": stored_value["sha256"],
        "selected_cells": selected_cells,
    }
    recovery_value = {
        **recovery_body,
        "sha256": reviewed_recovery_v2._body_digest(recovery_body),  # noqa: SLF001
    }
    return (
        _private_file(tmp_path / "stored-intent.json", stored_value),
        _private_file(tmp_path / "recovery-intent.json", recovery_value),
    )


def test_seed44_base_repair_package_is_two_stage_alert_off_and_cpu_only(tmp_path: Path):
    stored_intent, recovery_intent = _packet_intents(tmp_path)
    packages = seed44_base_repair_job.render(
        repo_root=ROOT,
        plan_path=REPAIR_PLAN,
        stored_intent_path=stored_intent,
        recovery_intent_path=recovery_intent,
    )
    assert list(packages) == [
        "accept_existing_scored_session",
        "single_rollout_repair",
    ]
    for stage, package in packages.items():
        job = package.job
        assert job["metadata"]["annotations"] == {
            "fleet.ai/failure-alerts": "off",
            "cyber-post-train.fleet.ai/create-once": "true",
        }
        assert job["spec"]["backoffLimit"] == 0
        assert job["spec"]["template"]["spec"]["priorityClassName"] == "c1"
        assert "nvidia.com/gpu" not in json.dumps(job)
        assert package.config_map["immutable"] is True
        assert package.secret["immutable"] is True
        assert package.proof["selected_cell_count"] == (
            5 if stage == "accept_existing_scored_session" else 2
        )
        assert package.proof["private_intent_content_included"] is False
        public = json.dumps(package.proof)
        assert all(
            cell_id not in public
            for cell_id in (
                stored_session_reconciliation_v2.load_intent(stored_intent).selected_cell_ids
                + reviewed_recovery_v2.load_intent(recovery_intent).selected_cell_ids
            )
        )
    assert (
        packages["accept_existing_scored_session"]
        .job["spec"]["template"]["spec"]
        .get("initContainers")
        is None
    )
    assert (
        packages["single_rollout_repair"].job["spec"]["template"]["spec"]["initContainers"][0][
            "name"
        ]
        == "dind"
    )
    assert packages["single_rollout_repair"].proof["execution_generation"] == 2
    plan = json.loads(REPAIR_PLAN.read_text())
    recovery = packages["single_rollout_repair"]
    staging = plan["stages"]["single_rollout_repair"]["image_staging"]
    assert recovery.proof["image_staging"] == staging
    assert {"cluster_entry.py", "model_artifact.py"} <= set(recovery.config_map["data"])
    environment = {
        row["name"]: row.get("value")
        for row in recovery.job["spec"]["template"]["spec"]["containers"][0]["env"]
    }
    assert environment["HARNESS_TAR"] == staging["harness_tar"]
    assert environment["HARNESS_TAR_SHA256"] == staging["harness_tar_sha256"]
    assert environment["HARNESS_RECEIPT"] == staging["harness_receipt"]
    assert environment["HARNESS_RECEIPT_SHA256"] == staging["harness_receipt_sha256"]
    run_script = recovery.config_map["data"]["run.sh"]
    assert '--harness-tar "$HARNESS_TAR"' in run_script
    assert '--harness-tar-sha256 "$HARNESS_TAR_SHA256"' in run_script
    assert '--harness-receipt "$HARNESS_RECEIPT"' in run_script
    assert '--harness-receipt-sha256 "$HARNESS_RECEIPT_SHA256"' in run_script


def test_recovery_stages_exact_harness_before_fresh_dind_image_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    harness = tmp_path / "harness.tar"
    harness.write_bytes(b"exact reviewed harness archive")
    harness_sha256 = "sha256:" + hashlib.sha256(harness.read_bytes()).hexdigest()
    plan = {
        "images": {"agent": _sha("a"), "proxy": "proxy.example/image@" + _sha("b")},
        "treatment": {"release_asset_sha256": _sha("c")},
    }
    receipt = tmp_path / "BUILD.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "fleet_harness_build_v1",
                "tar_sha256": harness_sha256,
                "image_id": plan["images"]["agent"],
                "platform": "linux/amd64",
                "release_asset_sha256": plan["treatment"]["release_asset_sha256"],
            }
        ),
        encoding="utf-8",
    )
    receipt_sha256 = "sha256:" + hashlib.sha256(receipt.read_bytes()).hexdigest()
    images: set[str] = set()
    events: list[str] = []

    def docker_run(command, **_kwargs):
        if command[:2] == ["docker", "load"]:
            assert images == set()
            images.add(plan["images"]["agent"])
            events.append("load_agent")
        elif command[:3] == ["docker", "image", "inspect"]:
            assert command[3] in images
            events.append("inspect_agent")
        elif command[:2] == ["docker", "pull"]:
            assert images == {plan["images"]["agent"]}
            images.add(plan["images"]["proxy"])
            events.append("pull_proxy")
        else:  # pragma: no cover - protects the exact staging command surface
            raise AssertionError(command)
        return subprocess.CompletedProcess(command, 0)

    def check_images(value):
        assert value is plan
        assert images == {plan["images"]["agent"], plan["images"]["proxy"]}
        events.append("check_images")

    monkeypatch.setattr(reviewed_recovery_worker_v2.cluster_entry.subprocess, "run", docker_run)
    monkeypatch.setattr(reviewed_recovery_worker_v2.evaluate, "check_images", check_images)
    reviewed_recovery_worker_v2._stage_and_check_images(  # noqa: SLF001
        plan=plan,
        harness_tar=harness,
        harness_tar_sha256=harness_sha256,
        harness_receipt=receipt,
        harness_receipt_sha256=receipt_sha256,
    )
    assert events == ["load_agent", "inspect_agent", "pull_proxy", "check_images"]
    with pytest.raises(ValueError, match="archive digest differs"):
        reviewed_recovery_worker_v2._stage_and_check_images(  # noqa: SLF001
            plan=plan,
            harness_tar=harness,
            harness_tar_sha256=_sha("e"),
            harness_receipt=receipt,
            harness_receipt_sha256=receipt_sha256,
        )
    assert events == ["load_agent", "inspect_agent", "pull_proxy", "check_images"]


def test_seed44_recovery_bootstrap_imports_from_only_the_rendered_closure(tmp_path: Path):
    stored_intent, recovery_intent = _packet_intents(tmp_path)
    recovery = seed44_base_repair_job.render(
        repo_root=ROOT,
        plan_path=REPAIR_PLAN,
        stored_intent_path=stored_intent,
        recovery_intent_path=recovery_intent,
    )["single_rollout_repair"]
    data = recovery.config_map["data"]
    isolated = tmp_path / "isolated"
    (isolated / "cyber_post_train").mkdir(parents=True)
    (isolated / "evals/fleet").mkdir(parents=True)
    for path in (
        isolated / "cyber_post_train/__init__.py",
        isolated / "evals/__init__.py",
        isolated / "evals/fleet/__init__.py",
    ):
        path.write_text("", encoding="utf-8")
    (isolated / "cyber_post_train/jobs.py").write_text(data["jobs.py"], encoding="utf-8")
    for name, source in data.items():
        if name not in {"jobs.py", "run.sh"}:
            (isolated / "evals/fleet" / name).write_text(source, encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            f"import sys;sys.path.insert(0,{str(isolated)!r});"
            "import evals.fleet.reviewed_recovery_worker_v2",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_seed44_private_packages_are_create_once_and_not_printed(tmp_path: Path):
    stored_intent, recovery_intent = _packet_intents(tmp_path)
    output = tmp_path / "private-packages"
    receipt = seed44_base_repair_job.write_private_packages(
        repo_root=ROOT,
        plan_path=REPAIR_PLAN,
        stored_intent_path=stored_intent,
        recovery_intent_path=recovery_intent,
        output_root=output,
    )
    assert receipt["jobs_created"] == 0
    assert receipt["private_cell_task_session_or_trace_identifiers_included"] is False
    for stage in (
        "accept_existing_scored_session",
        "single_rollout_repair",
    ):
        bundle = output / stage / "bundle.json"
        assert bundle.stat().st_mode & 0o077 == 0
        value = json.loads(bundle.read_text())
        job = next(item for item in value["items"] if item["kind"] == "Job")
        assert job["metadata"]["annotations"]["fleet.ai/failure-alerts"] == "off"
    with pytest.raises(FileExistsError, match="already exists"):
        seed44_base_repair_job.write_private_packages(
            repo_root=ROOT,
            plan_path=REPAIR_PLAN,
            stored_intent_path=stored_intent,
            recovery_intent_path=recovery_intent,
            output_root=output,
        )


def test_seed44_base_repair_plan_is_launch_inert_and_final8_sealed():
    plan = json.loads(REPAIR_PLAN.read_text())
    assert plan["sha256"] == seed44_base_repair_job._canonical_digest(  # noqa: SLF001
        {key: value for key, value in plan.items() if key != "sha256"}
    )
    assert plan["launchable"] is False
    assert plan["repair_contract"]["accepted_cell_replay_count"] == 0
    assert plan["repair_contract"]["stored_session_rescore_count"] == 0
    assert plan["repair_contract"]["stored_session_regeneration_count"] == 0
    assert plan["repair_contract"]["single_rollout_repair_count"] == 2
    assert plan["stages"]["single_rollout_repair"]["execution_generation"] == 2
    assert plan["repair_contract"]["final_eight_task_set_accessed"] is False
    assert set(plan["operation"].values()) == {0}
    assert plan["source"]["evaluation_plan_sha256"] == (
        "sha256:844d8ebdaed6e8588c02187c8520eaad942a142e0a83961c19b6df3a46cf0883"
    )
    assert plan["source"]["ledger_plan_sha256"] == (
        "sha256:04dc90eb59e133cdc40e375b09aa0a3d7cc9c325f7dddc438fd10450661dbae6"
    )
    assert plan["source"]["evaluation_plan_sha256"] != plan["source"]["ledger_plan_sha256"]
