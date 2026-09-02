from __future__ import annotations

import copy
import json
import shutil
import threading
import time
from collections import Counter
from pathlib import Path

import pytest
import yaml

from evals.fleet import holdout, qwen38_fleet50

PLAN_PATH = Path("evals/fleet/configs/qwen38-27b-qwen-code-ranked50-base-v1.json")
SELECTION_PATH = Path("evals/fleet/configs/qwen38-27b-ranked50-selection-v1.json")
EXCLUSIONS_PATH = Path("evals/fleet/configs/qwen38-prior-attempt-exclusions-v1.json")
SPLIT_PATH = Path("configs/data/fleet-a62-task-split-v1.json")
JOB_PATH = Path("evals/fleet/cluster/qwen38-code-selfhosted-ranked50-job.yaml")
SUBMIT_PATH = Path("evals/fleet/scripts/submit_selfhosted_qwen38_ranked50.sh")
RUN_PATH = Path("evals/fleet/scripts/run_selfhosted_qwen38_ranked50.sh")
CAPTURE_PATH = Path("evals/fleet/scripts/capture_fleet_secret_metadata.sh")
COLLECT_PATH = Path("evals/fleet/scripts/collect_selfhosted_qwen38_ranked50.sh")
ACCEPT_JOB_PATH = Path("evals/fleet/cluster/qwen38-ranked50-acceptance-job.yaml")


def _successful_result(score: float = 0.0) -> dict:
    return {
        "score": score,
        "verifier_execution_id": "33333333-3333-4333-8333-333333333333",
        "qwen_exit_code": 0,
        "session_ingest_status": "completed",
        "session_id": "77777777-7777-4777-8777-777777777777",
    }


def _source_acceptance_and_workload() -> tuple[dict, dict, dict, dict, dict]:
    artifact_manifest = {"files": []}
    artifact_manifest["manifest_sha256"] = qwen38_fleet50.digest_without(
        artifact_manifest, "manifest_sha256"
    )
    runtime_images = {"schema_version": "fleet-eval-runtime-images-v1", "images": {}}
    runtime_images["receipt_sha256"] = qwen38_fleet50.digest_without(
        runtime_images, "receipt_sha256"
    )
    acceptance = {
        "schema_version": qwen38_fleet50.RUN_ACCEPTANCE_SCHEMA,
        "campaign_id": qwen38_fleet50.EXPECTED_SOURCE_JOB,
        "accepted": True,
        "pod_identity": {
            "job_name": qwen38_fleet50.EXPECTED_SOURCE_JOB,
            "pod_name": "ranked50-pod",
            "pod_uid": "44444444-4444-4444-8444-444444444444",
        },
        "plan_sha256": "sha256:" + "1" * 64,
        "frozen_receipt_sha256": "sha256:" + "2" * 64,
        "selection_sha256": "sha256:" + "3" * 64,
        "prior_attempt_exclusions_sha256": "sha256:" + "4" * 64,
        "duplicate_preflight_sha256": "sha256:" + "5" * 64,
        "summary_sha256": "sha256:" + "6" * 64,
        "fleet_account": {
            "team_name": "fleet",
            "team_id": holdout.self_hosted.FLEET_TEAM_ID,
        },
        "artifact_manifest": artifact_manifest,
        "runtime_images": runtime_images,
        "outcomes": [{"index": index} for index in range(1, 51)],
        "data_minimization": {
            "prompts_included": False,
            "transcripts_included": False,
            "tool_content_included": False,
            "verifier_content_included": False,
            "flags_included": False,
            "credentials_included": False,
        },
    }
    acceptance["acceptance_sha256"] = qwen38_fleet50.digest_without(
        acceptance, "acceptance_sha256"
    )
    job = list(yaml.safe_load_all(JOB_PATH.read_text()))[-1]
    job["metadata"]["uid"] = "55555555-5555-4555-8555-555555555555"
    job["spec"]["suspend"] = False
    job["status"] = {
        "conditions": [{"type": "Complete", "status": "True"}],
        "succeeded": 1,
        "completionTime": "2026-09-02T01:00:00Z",
    }
    pod = {
        "metadata": {
            "name": "ranked50-pod",
            "uid": "44444444-4444-4444-8444-444444444444",
            "ownerReferences": [
                {
                    "apiVersion": "batch/v1",
                    "kind": "Job",
                    "name": qwen38_fleet50.EXPECTED_SOURCE_JOB,
                    "uid": job["metadata"]["uid"],
                    "controller": True,
                }
            ],
        },
        "status": {
            "phase": "Succeeded",
            "containerStatuses": [
                {
                    "name": "evaluator",
                    "restartCount": 0,
                    "state": {"terminated": {"exitCode": 0}},
                }
            ],
        },
    }
    config_map = {
        "metadata": {
            "name": qwen38_fleet50.EXPECTED_SOURCE_JOB,
            "uid": "88888888-8888-4888-8888-888888888888",
        },
        "immutable": True,
        "data": {"config.json": "{}"},
    }
    binding = {
        "schema_version": "fleet-qwen38-ranked50-acceptance-source-binding-v1",
        "job_uid": job["metadata"]["uid"],
        "pod_uid": pod["metadata"]["uid"],
        "configmap_uid": config_map["metadata"]["uid"],
    }
    return acceptance, job, {"items": [pod]}, config_map, binding


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _receipt(plan: dict) -> dict:
    tasks = []
    for index, row in enumerate(plan["selection"]["tasks"], 1):
        tasks.append(
            {
                "index": index,
                **{field: row[field] for field in qwen38_fleet50.TASK_BINDING_FIELDS},
                "prompt_sha256": "sha256:prompt",
                "env_variables_sha256": "sha256:env",
                "output_json_schema_sha256": "sha256:schema",
                "runtime_seed_content_sha256": "sha256:seed",
                "runtime_seed_file_count": 1,
                "cyber_contract": plan["authority"]["required_cyber_contract"],
                "verifier": {
                    "id": "11111111-1111-4111-8111-111111111111",
                    "version_id": "22222222-2222-4222-8222-222222222222",
                    "version": 1,
                    "sha256": "sha256:verifier",
                    "function_name": "verify",
                },
            }
        )
    duplicate_proof = {
        "schema_version": holdout.DUPLICATE_PROOF_SCHEMA,
        "route": "/v1/sessions?task_key=<task-lineage-key>",
        "inventory_scope": "live_non_archived_sessions",
        "served_model": plan["model"]["served_id"],
        "queried_task_count": 50,
        "queries": [
            {
                "task_key": row["task_key"],
                "task_version_id": row["task_version_id"],
                "pages": 1,
                "returned_sessions": 0,
                "equivalent_qwen38_sessions": 0,
            }
            for row in tasks
        ],
        "transcripts_read": False,
        "session_identifiers_persisted": False,
    }
    duplicate_proof["proof_sha256"] = holdout._digest_without(
        duplicate_proof, "proof_sha256"
    )
    receipt = {
        "schema_version": holdout.RECEIPT_SCHEMA,
        "plan_schema_version": holdout.RANKED50_SCHEMA,
        "campaign_id": plan["campaign_id"],
        "task_count": 50,
        "planned_sessions": 50,
        "model": plan["model"],
        "authority": plan["authority"],
        "fleet_account": {
            "team_name": "fleet",
            "team_id": holdout.self_hosted.FLEET_TEAM_ID,
        },
        "tasks": tasks,
        "duplicate_preflight": duplicate_proof,
    }
    receipt["receipt_sha256"] = holdout._digest_without(receipt, "receipt_sha256")
    return receipt


def _mock_wave_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    def observation(plan: dict, wave: int) -> dict:
        value = {
            "wave": wave,
            "observed_at": f"2026-09-02T00:{wave:02d}:00Z",
            "identity": plan["model"]["live_identity"],
        }
        value["observation_sha256"] = holdout._digest_without(
            value, "observation_sha256"
        )
        return value

    monkeypatch.setattr(holdout, "_wave_model_identity_observation", observation)
    monkeypatch.setattr(holdout, "build_terminal_acceptance", lambda *args: None)
    _mock_sanitizer(monkeypatch)


def _mock_sanitizer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLEET_EVAL_SCRATCH_ROOT", "/tmp")

    def persist(scratch: Path, task_out: Path, config: dict, result: dict) -> None:
        del config, result
        shutil.copyfile(scratch / "cleanup.json", task_out / "cleanup.json")

    monkeypatch.setattr(holdout, "_persist_sanitized_task_artifacts", persist)


def test_ranked50_is_exact_easiest_first_train_dev_selection() -> None:
    plan = _json(PLAN_PATH)
    selection = _json(SELECTION_PATH)
    exclusions = _json(EXCLUSIONS_PATH)
    split = _json(SPLIT_PATH)
    assert plan["selection"] == selection
    assert plan["prior_attempt_exclusions"] == exclusions
    rows = holdout.validate_plan(plan, split)
    assert len(rows) == 50
    assert Counter(row["split"] for row in rows) == {"train": 48, "dev": 2}
    assert all(row["split"] != "test" for row in rows)
    assert rows[0]["historical_ease"]["pass_rate"] == 1.0
    assert rows[-1]["historical_ease"]["pass_rate"] == 0.75
    assert rows == sorted(rows, key=qwen38_fleet50._rank_key)
    assert Counter(row["historical_ease"]["pass_rate"] for row in rows) == {
        1.0: 41,
        0.875: 8,
        0.75: 1,
    }
    assert selection["ranking_interpretation"] == (
        "historical_frontier_capability_sweep_not_qwen_specific_difficulty_estimate"
    )


def test_historical_identity_uses_roster_version_and_rejects_key_or_schema_drift() -> None:
    record = {
        "export_schema": "fleet_session_export_v1",
        "source": {
            "task_key_from_roster": "task-key",
            "roster_task_binding": {
                "key": "task-key",
                "eval_task_version_id": "11111111-1111-4111-8111-111111111111",
            },
        },
        "transcript_envelope": {
            "task": {
                "key": "task-key",
                "eval_task_version_id": "22222222-2222-4222-8222-222222222222",
            }
        },
    }
    assert qwen38_fleet50.authoritative_task_identity(record) == (
        "task-key",
        "11111111-1111-4111-8111-111111111111",
    )
    drifted = copy.deepcopy(record)
    drifted["transcript_envelope"]["task"]["key"] = "other-key"
    with pytest.raises(ValueError, match="task keys disagree"):
        qwen38_fleet50.authoritative_task_identity(drifted)
    drifted = copy.deepcopy(record)
    drifted["export_schema"] = "unsupported"
    with pytest.raises(ValueError, match="schema is unsupported"):
        qwen38_fleet50.authoritative_task_identity(drifted)
def test_historical_identity_uses_roster_version_not_stale_transcript_version() -> None:
    roster_version = "55555555-5555-4555-8555-555555555555"
    record = {
        "export_schema": "fleet_session_export_v1",
        "source": {
            "roster_task_binding": {
                "key": "task-key",
                "eval_task_version_id": roster_version,
            },
            "task_key_from_roster": "task-key",
        },
        "transcript_envelope": {
            "task": {
                "key": "task-key",
                "eval_task_version_id": "66666666-6666-4666-8666-666666666666",
            }
        },
    }
    assert qwen38_fleet50.authoritative_task_identity(record) == (
        "task-key",
        roster_version,
    )
    drifted = copy.deepcopy(record)
    drifted["source"]["task_key_from_roster"] = "different-key"
    with pytest.raises(ValueError, match="projections disagree"):
        qwen38_fleet50.authoritative_task_identity(drifted)
    drifted = copy.deepcopy(record)
    drifted["export_schema"] = "unknown"
    with pytest.raises(ValueError, match="schema is unsupported"):
        qwen38_fleet50.authoritative_task_identity(drifted)


def test_ranked50_excludes_every_known_qwen38_attempt() -> None:
    plan = _json(PLAN_PATH)
    exclusions = _json(EXCLUSIONS_PATH)
    excluded = qwen38_fleet50.validate_exclusions(exclusions)
    selected = {row["task_version_id"] for row in plan["selection"]["tasks"]}
    assert len(excluded) == 5
    assert selected.isdisjoint(excluded)
    assert Counter(row["disposition"] for row in exclusions["attempts"]) == {
        "valid_model_outcome": 2,
        "unresolved_excluded": 3,
    }


def test_live_duplicate_preflight_is_exhaustive_and_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _json(PLAN_PATH)
    rows = plan["selection"]["tasks"]
    calls: list[str] = []

    def no_duplicates(client: object, method: str, path: str, **kwargs: object) -> dict:
        del client
        assert method == "GET"
        assert path == "/v1/sessions"
        params = kwargs["params"]
        assert isinstance(params, dict)
        assert params["limit"] == 500
        assert params["offset"] == 0
        calls.append(str(params["task_key"]))
        return {
            "sessions": [{"model": "openai/gpt-5.6"}],
            "has_more": False,
        }

    monkeypatch.setattr(holdout.self_hosted, "_request", no_duplicates)
    proof = holdout.live_duplicate_preflight(object(), rows, plan["model"])
    assert len(calls) == 50
    assert calls == [row["task_key"] for row in rows]
    assert proof["queried_task_count"] == 50
    assert proof["session_identifiers_persisted"] is False
    assert all("session_id" not in row for row in proof["queries"])
    holdout.validate_duplicate_preflight(proof, rows, plan["model"])

    def duplicate(client: object, method: str, path: str, **kwargs: object) -> dict:
        del client, method, path
        return {"sessions": [{"model": "Qwen/Qwen3.8-27B"}], "has_more": False}

    monkeypatch.setattr(holdout.self_hosted, "_request", duplicate)
    with pytest.raises(RuntimeError, match="already exists"):
        holdout.live_duplicate_preflight(object(), rows, plan["model"])


def test_ranked50_binds_exact_harness_tools_and_first_task_gate() -> None:
    plan = _json(PLAN_PATH)
    assert plan["model"]["revision"] == "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    assert plan["harness"]["version"] == "0.22.3"
    assert plan["harness"]["source_commit"] == ("09825973e7d3c3fd07e17909c396aa62f48ce51f")
    assert plan["execution"]["required_task_tools"] == ["bash", "submit_report"]
    assert plan["execution"]["max_concurrent"] == 3
    assert plan["execution"]["pass_k"] == 1
    assert plan["execution"]["training_data_eligible"] is False
    assert plan["first_task_release_gate"] == {
        "task_index": 1,
        "task_version_id": plan["selection"]["tasks"][0]["task_version_id"],
        "required_outcome_status": "model_outcome",
        "required_verifier_execution_id": "nonzero_uuid",
        "required_cleanup_verified": True,
        "required_qwen_exit_code": 0,
        "required_session_ingest_status": "completed",
        "required_session_id": "nonzero_uuid",
        "accept_zero": True,
    }
    assert plan["credential_gate"]["rotation_not_before"] == "2026-09-01T23:39:05Z"
    assert plan["credential_gate"]["secret_value_must_not_be_decoded_by_submitter"] is True
    payload = holdout.self_hosted.build_scoring_payload(
        plan,
        instance_id="instance-slug",
        final_answer="FLAG{private}",
        messages=[{"role": "tool", "content": "private"}],
    )
    assert payload == {
        "instance_id": "instance-slug",
        "scoring_mode": "partial",
        "multi_app_aggregation_mode": "fractional",
    }


@pytest.mark.parametrize("score", [0.0, 1.0])
def test_valid_authoritative_first_outcome_releases_remaining(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, score: float
) -> None:
    plan = _json(PLAN_PATH)
    receipt = _receipt(plan)
    _mock_wave_identity(monkeypatch)

    def successful_run(
        config: dict, out_dir: Path, proxy_script: Path, **kwargs: object
    ) -> dict:
        del config, proxy_script, kwargs
        out_dir.mkdir(parents=True)
        (out_dir / "cleanup.json").write_text(
            json.dumps(
                {"instance_created": True, "instance_closed": True, "containers_removed": True}
            )
        )
        return _successful_result(score)

    monkeypatch.setattr(holdout.self_hosted, "run", successful_run)
    summary = holdout.run_campaign(plan, receipt, tmp_path / "campaign", Path("proxy.py"))
    assert summary["model_outcomes"] == 50
    assert summary["first_task_release_gate_satisfied"] is True


def test_first_outcome_without_verifier_uuid_does_not_release_remaining(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _json(PLAN_PATH)
    receipt = _receipt(plan)
    _mock_wave_identity(monkeypatch)

    def invalid_run(config: dict, out_dir: Path, proxy_script: Path, **kwargs: object) -> dict:
        del config, proxy_script, kwargs
        out_dir.mkdir(parents=True)
        (out_dir / "cleanup.json").write_text(
            json.dumps(
                {"instance_created": True, "instance_closed": True, "containers_removed": True}
            )
        )
        return {"score": 0.0, "verifier_execution_id": None}

    monkeypatch.setattr(holdout.self_hosted, "run", invalid_run)
    summary = holdout.run_campaign(plan, receipt, tmp_path / "campaign", Path("proxy.py"))
    assert summary["model_outcomes"] == 0
    assert summary["infrastructure_errors"] == 1
    assert summary["first_task_release_gate_satisfied"] is False


def test_post_gate_outcome_without_verifier_uuid_is_infrastructure_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _json(PLAN_PATH)
    receipt = _receipt(plan)
    _mock_wave_identity(monkeypatch)
    lock = threading.Lock()
    calls = 0

    def run_with_missing_post_gate_verifier(
        config: dict, out_dir: Path, proxy_script: Path, **kwargs: object
    ) -> dict:
        nonlocal calls
        del config, proxy_script, kwargs
        with lock:
            calls += 1
            call = calls
        out_dir.mkdir(parents=True)
        (out_dir / "cleanup.json").write_text(
            json.dumps(
                {"instance_created": True, "instance_closed": True, "containers_removed": True}
            )
        )
        return {
            **_successful_result(),
            "verifier_execution_id": (
                "33333333-3333-4333-8333-333333333333" if call == 1 else None
            ),
        }

    monkeypatch.setattr(holdout.self_hosted, "run", run_with_missing_post_gate_verifier)
    summary = holdout.run_campaign(plan, receipt, tmp_path / "campaign", Path("proxy.py"))
    assert summary["authoritative_verifier_backed_outcomes"] == 1
    assert summary["model_outcomes"] == 1
    assert summary["infrastructure_errors"] == 3
    assert calls == 4


def test_ranked50_runs_first_alone_then_bounded_waves_of_three(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _json(PLAN_PATH)
    receipt = _receipt(plan)
    _mock_wave_identity(monkeypatch)
    lock = threading.Lock()
    active = 0
    max_active = 0
    started: list[int] = []

    def successful_run(
        config: dict, out_dir: Path, proxy_script: Path, **kwargs: object
    ) -> dict:
        nonlocal active, max_active
        del proxy_script, kwargs
        index = int(config["run_id"].rsplit("-t", 1)[1].split("-", 1)[0])
        with lock:
            started.append(index)
            active += 1
            max_active = max(max_active, active)
            if index > 1:
                assert 1 in started
        time.sleep(0.003)
        out_dir.mkdir(parents=True)
        (out_dir / "cleanup.json").write_text(
            json.dumps(
                {"instance_created": True, "instance_closed": True, "containers_removed": True}
            )
        )
        with lock:
            active -= 1
        return _successful_result()

    monkeypatch.setattr(holdout.self_hosted, "run", successful_run)
    summary = holdout.run_campaign(plan, receipt, tmp_path / "campaign", Path("proxy.py"))
    assert summary["model_outcomes"] == 50
    assert started[0] == 1
    assert max_active == 3


def test_wave_identity_drift_stops_before_next_wave(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _json(PLAN_PATH)
    receipt = _receipt(plan)
    monkeypatch.setattr(holdout, "build_terminal_acceptance", lambda *args: None)

    def observe(plan: dict, wave: int) -> dict:
        del plan
        if wave == 2:
            raise RuntimeError("model identity drift")
        return {"wave": wave, "observation_sha256": "sha256:wave-1"}

    calls = 0

    def successful_run(
        config: dict, out_dir: Path, proxy_script: Path, **kwargs: object
    ) -> dict:
        nonlocal calls
        del config, proxy_script, kwargs
        calls += 1
        out_dir.mkdir(parents=True)
        (out_dir / "cleanup.json").write_text(
            json.dumps(
                {"instance_created": True, "instance_closed": True, "containers_removed": True}
            )
        )
        return _successful_result()

    monkeypatch.setattr(holdout, "_wave_model_identity_observation", observe)
    _mock_sanitizer(monkeypatch)
    monkeypatch.setattr(holdout.self_hosted, "run", successful_run)
    summary = holdout.run_campaign(plan, receipt, tmp_path / "campaign", Path("proxy.py"))
    assert calls == 4
    assert summary["model_outcomes"] == 4
    assert summary["infrastructure_errors"] == 1
    assert summary["wave_identity_error"] == "RuntimeError"


def test_terminal_acceptance_is_create_once_digest_bound_and_all_or_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _json(PLAN_PATH)
    receipt = _receipt(plan)
    out_dir = tmp_path / "campaign"
    out_dir.mkdir()
    outcomes = []
    for index, row in enumerate(plan["selection"]["tasks"], 1):
        task_dir = out_dir / f"task-{index:02d}-artifact"
        task_dir.mkdir()
        frozen = receipt["tasks"][index - 1]
        expected_config = holdout.task_config(plan, frozen)
        run_id = expected_config["run_id"]
        instance_id = f"instance-{index}"
        evidence_run_id = "66666666-6666-4666-8666-666666666666"
        verifier_id = "33333333-3333-4333-8333-333333333333"
        session_id = "77777777-7777-4777-8777-777777777777"
        resource = {
            "run_id": run_id,
            "instance_id": instance_id,
            "evidence_run_id": evidence_run_id,
        }
        resource["resource_plan_sha256"] = holdout._digest_without(
            resource, "resource_plan_sha256"
        )
        scoring = {
            "run_id": run_id,
            "task_key": expected_config["task"]["key"],
            "task_version_id": expected_config["task"]["version_id"],
            "instance_id": instance_id,
            "evidence_run_id": evidence_run_id,
        }
        scoring["scoring_intent_sha256"] = holdout._digest_without(
            scoring, "scoring_intent_sha256"
        )
        provisioning = {
            "schema_version": "fleet-qwen-provisioning-intent-v1",
            "run_id": expected_config["run_id"],
            "task_key": expected_config["task"]["key"],
            "task_version_id": expected_config["task"]["version_id"],
            "request_id": holdout.self_hosted.provisioning_request_id(expected_config),
            "request_payload_sha256": holdout.self_hosted.sha256(
                holdout.self_hosted.canonical_json({})
            ),
        }
        provisioning["intent_sha256"] = holdout._digest_without(
            provisioning, "intent_sha256"
        )
        artifacts = {
            "binding.json": {
                "run_id": run_id,
                "source_job_id": expected_config["source_job_id"],
                "task": expected_config["task"],
                "environment": expected_config["environment"],
                "verifier": expected_config["verifier"],
                "authority": expected_config["authority"],
                "model": expected_config["model"],
                "harness": expected_config["harness"],
            },
            "cleanup.json": {"instance_closed": True, "containers_removed": True},
            "provisioning-intent.json": provisioning,
            "result.json": {
                "run_id": run_id,
                "session_id": session_id,
                "score": 0.0,
                "verifier_execution_id": verifier_id,
                "qwen_exit_code": 0,
            },
            "reward-result.json": {
                "instance_id": instance_id,
                "task_key": row["task_key"],
                "task_version_id": row["task_version_id"],
                "reward": 0.0,
                "verifier_execution_id": verifier_id,
            },
            "resource-plan.json": resource,
            "runtime-binding.json": {
                "instance_id": instance_id,
                "evidence_run_id": evidence_run_id,
                    "env_key": row["env_key"],
                    "environment_version": row["env_version"],
                    "environment_version_id": row["environment_version_id"],
                    "data_key": row["data_key"],
                "data_version": row["data_version"],
                "tool_names": ["bash", "submit_report"],
                "tool_catalog_sha256": plan["execution"][
                    "required_task_tool_catalog_sha256"
                ],
            },
            "scoring-intent.json": scoring,
            "session-ingest.json": {
                "status": "completed",
                "session_id": session_id,
                "chunk_count": 1,
                "chunks_completed": 1,
            },
        }
        for name, value in artifacts.items():
            (task_dir / name).write_text(json.dumps(value) + "\n")
        outcomes.append(
                {
                "index": index,
                "task_key": row["task_key"],
                "task_version_id": row["task_version_id"],
                "status": "model_outcome",
                "score": 0.0,
                    "verifier_execution_id": verifier_id,
                    "cleanup_verified": True,
                    "qwen_exit_code": 0,
                    "session_ingest_status": "completed",
                    "session_id": session_id,
                }
        )
    summary = {
        "model_outcomes": 50,
        "authoritative_verifier_backed_outcomes": 50,
        "infrastructure_errors": 0,
        "wave_identity_error": None,
        "wave_identity_observations": [],
        "outcomes": outcomes,
    }
    for index in range(18):
        observation = {"wave": index, "identity": {"served_id": "qwen3.8-27b"}}
        observation["observation_sha256"] = holdout._digest_without(
            observation, "observation_sha256"
        )
        summary["wave_identity_observations"].append(observation)
    (out_dir / "summary.json").write_bytes(holdout.self_hosted.canonical_json(summary) + b"\n")
    (out_dir / "campaign-state.json").write_text("{}\n")
    runtime_images = {
        "schema_version": "fleet-eval-runtime-images-v1",
        "images": {
            "qwen_code": {
                "requested_ref": "chris/qwen-code:0.22.3-q38-fleet-v1",
                "runtime_id": "sha256:" + "1" * 64,
                "architecture": "amd64",
            },
            "fixed_proxy": {
                "requested_ref": "proxy@sha256:" + "2" * 64,
                "runtime_id": "sha256:" + "3" * 64,
                "architecture": "amd64",
            },
        },
    }
    runtime_images["receipt_sha256"] = holdout._digest_without(
        runtime_images, "receipt_sha256"
    )
    runtime_path = tmp_path / "runtime-images.json"
    runtime_path.write_text(json.dumps(runtime_images))
    monkeypatch.setenv("FLEET_EVAL_RUNTIME_IMAGES_RECEIPT", str(runtime_path))
    monkeypatch.setenv("FLEET_EVAL_JOB_NAME", "chris-cyber-qwen38-qcode-fleet-ranked50-base-v1")
    monkeypatch.setenv("FLEET_EVAL_POD_NAME", "ranked50-pod")
    monkeypatch.setenv("FLEET_EVAL_POD_UID", "44444444-4444-4444-8444-444444444444")
    monkeypatch.setenv("QWEN_CODE_IMAGE", "chris/qwen-code:0.22.3-q38-fleet-v1")
    monkeypatch.setenv("FIXED_PROXY_IMAGE", "proxy@sha256:" + "2" * 64)
    accepted = holdout.build_terminal_acceptance(plan, receipt, summary, out_dir)
    assert accepted is not None
    assert accepted["accepted"] is True
    assert accepted["acceptance_sha256"] == holdout._digest_without(
        accepted, "acceptance_sha256"
    )
    holdout.self_hosted.write_json_once(out_dir / "ACCEPTED.json", accepted)
    qwen38_fleet50.validate_durable_run_tree(out_dir, accepted)
    with pytest.raises(FileExistsError):
        holdout.self_hosted.write_json_once(out_dir / "ACCEPTED.json", accepted)
    (out_dir / "prompt.txt").write_text("private")
    with pytest.raises(ValueError, match="unaccepted artifacts"):
        qwen38_fleet50.validate_durable_run_tree(out_dir, accepted)
    (out_dir / "prompt.txt").unlink()
    first_artifact = out_dir / accepted["artifact_manifest"]["files"][0]["path"]
    first_artifact.write_text("tampered")
    with pytest.raises(ValueError, match="unaccepted artifacts"):
        qwen38_fleet50.validate_durable_run_tree(out_dir, accepted)
    invalid = copy.deepcopy(summary)
    invalid["outcomes"][1]["verifier_execution_id"] = None
    assert holdout.build_terminal_acceptance(plan, receipt, invalid, out_dir) is None


def test_credential_rotation_gate_binds_only_sanitized_live_metadata() -> None:
    before = {
        "namespace": "fleet-train-jobs",
        "name": "fleet-api",
        "uid": "44444444-4444-4444-8444-444444444444",
        "resource_version": "123455",
        "data_key_present": True,
    }
    metadata = {**before, "resource_version": "123456"}
    receipt = qwen38_fleet50.build_credential_rotation_receipt(
        before,
        metadata,
        confirmed_by="fleet-credential-operator",
        rotation_completed_at="2026-09-02T00:00:00Z",
    )
    assert (
        qwen38_fleet50.validate_credential_rotation(
            receipt, metadata, rotation_not_before="2026-09-01T23:39:05Z"
        )
        == metadata
    )
    stale = copy.deepcopy(receipt)
    stale["rotation_completed_at"] = "2026-09-01T23:00:00Z"
    stale["receipt_sha256"] = qwen38_fleet50.digest_without(stale, "receipt_sha256")
    with pytest.raises(ValueError, match="does not postdate"):
        qwen38_fleet50.validate_credential_rotation(
            stale, metadata, rotation_not_before="2026-09-01T23:39:05Z"
        )
    with pytest.raises(ValueError, match="did not change"):
        qwen38_fleet50.build_credential_rotation_receipt(
            before,
            before,
            confirmed_by="fleet-credential-operator",
            rotation_completed_at="2026-09-02T00:00:00Z",
        )


def test_ranked50_cluster_launch_is_suspended_and_create_only() -> None:
    resources = list(yaml.safe_load_all(JOB_PATH.read_text()))
    job = resources[-1]
    assert job["metadata"]["name"] == "chris-cyber-qwen38-qcode-fleet-ranked50-base-v1"
    assert job["spec"]["suspend"] is True
    assert job["spec"]["backoffLimit"] == 0
    assert job["spec"]["activeDeadlineSeconds"] == 604800
    submit = SUBMIT_PATH.read_text()
    run = RUN_PATH.read_text()
    assert "kubectl apply" not in submit
    assert "base64 --decode" not in submit
    assert "secretKeyRef" not in JOB_PATH.read_text()
    assert "read_secret_snapshot_in_memory" in JOB_PATH.read_text()
    assert "credential-rotation-receipt.json" in JOB_PATH.read_text()
    role = resources[1]
    assert role["rules"] == [
        {
            "apiGroups": [""],
            "resources": ["secrets"],
            "resourceNames": ["fleet-api"],
            "verbs": ["get"],
        }
    ]
    assert "FLEET_CREDENTIAL_ROTATION_RECEIPT" in submit
    assert "already exists; refusing to replace" in submit
    assert 'test ! -e "$OUT_DIR"' in run
    assert 'summary["planned_sessions"] == 50' in run
    assert 'summary["first_task_release_gate_satisfied"] is True' in run
    assert 'summary["authoritative_verifier_backed_outcomes"] == 50' in run
    assert "--from-file=v2-incident.json=" in submit
    assert "--from-file=v3-terminal.json=" in submit
    assert "/bootstrap/v2-incident.json" in JOB_PATH.read_text()
    assert "/bootstrap/v3-terminal.json" in JOB_PATH.read_text()
    capture = CAPTURE_PATH.read_text()
    assert "base64 --decode" not in capture
    assert ".data.FLEET_API_KEY" not in capture
    assert "read_secret_metadata" in capture
    assert "-o json" not in capture
    assert 'test ! -e "$OUT"' in capture


def test_bootstrap_evidence_tree_can_validate_the_embedded_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    module_path = repository / "evals/fleet/qwen38_fleet50.py"
    module_path.parent.mkdir(parents=True)
    module_path.touch()
    for source in (
        Path(qwen38_fleet50.V2_EVIDENCE_PATH),
        Path(qwen38_fleet50.V3_EVIDENCE_PATH),
        Path(qwen38_fleet50.V2_PLAN_PATH),
        Path(qwen38_fleet50.V3_PLAN_PATH),
    ):
        destination = repository / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    monkeypatch.setattr(qwen38_fleet50, "__file__", str(module_path))
    assert len(holdout.validate_plan(_json(PLAN_PATH), _json(SPLIT_PATH))) == 50


def test_ranked50_rejects_selection_or_exclusion_drift() -> None:
    split = _json(SPLIT_PATH)
    exclusions = _json(EXCLUSIONS_PATH)
    selection = _json(SELECTION_PATH)
    drifted = copy.deepcopy(selection)
    drifted["tasks"][0]["historical_ease"]["pass_rate"] = 0.0
    with pytest.raises(ValueError, match="selection digest mismatch"):
        qwen38_fleet50.validate_selection(drifted, split, exclusions)
    drifted_exclusions = copy.deepcopy(exclusions)
    drifted_exclusions["harness_version"] = "0.22.4"
    drifted_exclusions["receipt_sha256"] = qwen38_fleet50.digest_without(
        drifted_exclusions, "receipt_sha256"
    )
    with pytest.raises(ValueError, match="harness version drifted"):
        qwen38_fleet50.validate_exclusions(drifted_exclusions)
    drifted_exclusions = copy.deepcopy(exclusions)
    drifted_exclusions["attempts"][1]["score"] = 0
    drifted_exclusions["receipt_sha256"] = qwen38_fleet50.digest_without(
        drifted_exclusions, "receipt_sha256"
    )
    with pytest.raises(ValueError, match="must not claim score"):
        qwen38_fleet50.validate_exclusions(drifted_exclusions)
    drifted_exclusions = copy.deepcopy(exclusions)
    drifted_exclusions["attempts"][4]["terminal_evidence"]["cleanup_instance_closed"] = False
    drifted_exclusions["receipt_sha256"] = qwen38_fleet50.digest_without(
        drifted_exclusions, "receipt_sha256"
    )
    with pytest.raises(ValueError, match="terminal outcome evidence drifted"):
        qwen38_fleet50.validate_exclusions(drifted_exclusions)
    drifted_exclusions = copy.deepcopy(exclusions)
    drifted_exclusions["evidence"]["v3_terminal_summary_raw_file_sha256"] = "sha256:tampered"
    drifted_exclusions["receipt_sha256"] = qwen38_fleet50.digest_without(
        drifted_exclusions, "receipt_sha256"
    )
    with pytest.raises(ValueError, match="evidence binding drifted"):
        qwen38_fleet50.validate_exclusions(drifted_exclusions)


def test_final_acceptance_binds_terminal_job_pod_and_is_create_once(tmp_path: Path) -> None:
    source, job, pods, config_map, binding = _source_acceptance_and_workload()
    final = qwen38_fleet50.build_final_acceptance(
        source, job, pods, config_map, binding
    )
    assert final["accepted"] is True
    assert final["source_job"]["uid"] == job["metadata"]["uid"]
    assert final["source_pod"]["evaluator_exit_code"] == 0
    assert final["acceptance_sha256"] == qwen38_fleet50.digest_without(
        final, "acceptance_sha256"
    )
    path = tmp_path / "FINAL_ACCEPTED.json"
    qwen38_fleet50.write_json_once(path, final)
    with pytest.raises(FileExistsError):
        qwen38_fleet50.write_json_once(path, final)


def test_final_acceptance_rejects_workload_or_source_receipt_drift() -> None:
    source, job, pods, config_map, binding = _source_acceptance_and_workload()
    mutations = []
    changed = copy.deepcopy(job)
    changed["spec"]["suspend"] = True
    mutations.append((source, changed, pods))
    changed = copy.deepcopy(job)
    changed["status"]["active"] = 1
    mutations.append((source, changed, pods))
    changed = copy.deepcopy(job)
    changed["spec"]["template"]["spec"]["containers"][0]["image"] = "drifted"
    mutations.append((source, changed, pods))
    changed_pods = copy.deepcopy(pods)
    changed_pods["items"][0]["metadata"]["ownerReferences"][0]["uid"] = (
        "99999999-9999-4999-8999-999999999999"
    )
    mutations.append((source, job, changed_pods))
    changed_pods = copy.deepcopy(pods)
    changed_pods["items"][0]["status"]["phase"] = "Failed"
    mutations.append((source, job, changed_pods))
    changed_pods = copy.deepcopy(pods)
    status = changed_pods["items"][0]["status"]["containerStatuses"][0]
    status["restartCount"] = 1
    mutations.append((source, job, changed_pods))
    changed_pods = copy.deepcopy(pods)
    status = changed_pods["items"][0]["status"]["containerStatuses"][0]
    status["state"]["terminated"]["exitCode"] = 1
    mutations.append((source, job, changed_pods))
    changed_source = copy.deepcopy(source)
    changed_source["campaign_id"] = "wrong"
    changed_source["acceptance_sha256"] = qwen38_fleet50.digest_without(
        changed_source, "acceptance_sha256"
    )
    mutations.append((changed_source, job, pods))
    for candidate_source, candidate_job, candidate_pods in mutations:
        with pytest.raises(ValueError):
            qwen38_fleet50.build_final_acceptance(
                candidate_source, candidate_job, candidate_pods, config_map, binding
            )
    changed_binding = copy.deepcopy(binding)
    changed_binding["configmap_uid"] = "99999999-9999-4999-8999-999999999999"
    with pytest.raises(ValueError):
        qwen38_fleet50.build_final_acceptance(
            source, job, pods, config_map, changed_binding
        )


def test_acceptance_collector_is_create_only_context_bound_and_sfs_mounted() -> None:
    script = COLLECT_PATH.read_text()
    assert 'KUBECTL=(kubectl --context "$EXPECTED_CONTEXT")' in script
    assert "kubectl apply" not in script
    assert "require_acceptance_absent" in script
    assert "source Job is not exclusively Complete" in script
    assert "source ConfigMap is not immutable" in script
    manifest = list(yaml.safe_load_all(ACCEPT_JOB_PATH.read_text()))
    collector = manifest[-1]
    assert collector["kind"] == "Job"
    assert collector["spec"]["backoffLimit"] == 0
    pod_spec = collector["spec"]["template"]["spec"]
    assert pod_spec["serviceAccountName"].endswith("acceptor-v1")
    volumes = {row["name"]: row for row in pod_spec["volumes"]}
    assert volumes["sfs"]["persistentVolumeClaim"]["claimName"] == "sfs-shared"
    command = pod_spec["containers"][0]["args"][0]
    assert "FINAL_ACCEPTED.json" in command
    assert "qwen38_fleet50 finalize" in command


def test_campaign_persists_only_sanitized_receipts_and_removes_private_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _json(PLAN_PATH)
    receipt = _receipt(plan)
    scratch_root = tmp_path / "scratch"
    monkeypatch.setenv("FLEET_EVAL_SCRATCH_ROOT", str(scratch_root))
    monkeypatch.setattr(
        holdout,
        "_wave_model_identity_observation",
        lambda _plan, wave: {"wave": wave, "observation_sha256": "sha256:observed"},
    )
    monkeypatch.setattr(holdout, "_first_task_gate_satisfied", lambda *_args: False)
    monkeypatch.setattr(holdout, "build_terminal_acceptance", lambda *_args: None)
    sentinel = "FLAG{private-do-not-persist} private prompt and tool trace"

    def sensitive_run(
        config: dict, out_dir: Path, proxy_script: Path, **kwargs: object
    ) -> dict:
        del proxy_script, kwargs
        out_dir.mkdir(parents=True)
        instance_id = "instance-slug"
        evidence_run_id = "66666666-6666-4666-8666-666666666666"
        verifier_id = "33333333-3333-4333-8333-333333333333"
        session_id = "77777777-7777-4777-8777-777777777777"
        resource = {
            "schema_version": "fleet-selfhosted-resource-plan-v1",
            "run_id": config["run_id"],
            "instance_id": instance_id,
            "evidence_run_id": evidence_run_id,
            "containers": [sentinel],
        }
        resource["resource_plan_sha256"] = holdout._digest_without(
            resource, "resource_plan_sha256"
        )
        scoring = {
            "schema_version": "fleet-selfhosted-scoring-intent-v1",
            "run_id": config["run_id"],
            "task_key": config["task"]["key"],
            "task_version_id": config["task"]["version_id"],
            "instance_id": instance_id,
            "evidence_run_id": evidence_run_id,
            "request_sha256": "sha256:" + "a" * 64,
        }
        scoring["scoring_intent_sha256"] = holdout._digest_without(
            scoring, "scoring_intent_sha256"
        )
        artifacts = {
            "binding.json": {"private": sentinel},
            "cleanup.json": {
                "instance_created": True,
                "instance_closed": True,
                "containers_removed": True,
            },
            "result.json": {"private": sentinel},
            "reward-result.json": {
                "instance_id": instance_id,
                "task_key": config["task"]["key"],
                "task_version_id": config["task"]["version_id"],
                "reward": 0.0,
                "verifier_execution_id": verifier_id,
                "cyber_evidence": sentinel,
            },
            "resource-plan.json": resource,
            "runtime-binding.json": {
                "instance_id": instance_id,
                "evidence_run_id": evidence_run_id,
                "env_key": config["environment"]["id"],
                "environment_version": config["environment"]["version"],
                "environment_version_id": config["environment"]["version_id"],
                "data_key": config["environment"]["data_id"],
                "data_version": config["environment"]["data_version"],
                "tool_names": ["bash", "submit_report"],
                "tool_catalog_sha256": config["execution"][
                    "required_task_tool_catalog_sha256"
                ],
            },
            "scoring-intent.json": scoring,
            "session-ingest.json": {
                "status": "completed",
                "session_id": session_id,
                "message_count": 10,
                "chunks_completed": 1,
                "chunk_count": 1,
            },
        }
        for name, value in artifacts.items():
            (out_dir / name).write_text(json.dumps(value))
        (out_dir / "prompt.txt").write_text(sentinel)
        (out_dir / "final-answer.txt").write_text(sentinel)
        private_dir = out_dir / "agent-output"
        private_dir.mkdir()
        (private_dir / "qwen-stream.jsonl").write_text(sentinel)
        return {
            **_successful_result(),
            "run_id": config["run_id"],
            "agent_termination": "completed",
            "elapsed_seconds": 1.0,
        }

    monkeypatch.setattr(holdout.self_hosted, "run", sensitive_run)
    durable = tmp_path / "durable"
    holdout.run_campaign(plan, receipt, durable, Path("proxy.py"))
    task_dirs = list(durable.glob("task-01-*"))
    assert len(task_dirs) == 1
    assert {path.name for path in task_dirs[0].iterdir()} == holdout.SANITIZED_TASK_FILES
    assert sentinel.encode() not in b"".join(
        path.read_bytes() for path in durable.rglob("*") if path.is_file()
    )
    assert not list(scratch_root.iterdir())


@pytest.mark.parametrize("failure_stage", ["scoring", "session_ingest"])
def test_campaign_failure_never_persists_private_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_stage: str
) -> None:
    plan = _json(PLAN_PATH)
    receipt = _receipt(plan)
    scratch_root = tmp_path / "scratch"
    monkeypatch.setenv("FLEET_EVAL_SCRATCH_ROOT", str(scratch_root))
    monkeypatch.setattr(
        holdout,
        "_wave_model_identity_observation",
        lambda _plan, wave: {"wave": wave, "observation_sha256": "sha256:observed"},
    )
    monkeypatch.setattr(holdout, "build_terminal_acceptance", lambda *_args: None)
    sentinel = f"FLAG{{private-{failure_stage}}} private trace"

    def failing_run(
        config: dict, out_dir: Path, proxy_script: Path, **kwargs: object
    ) -> dict:
        del proxy_script
        out_dir.mkdir(parents=True)
        (out_dir / "prompt.txt").write_text(sentinel)
        private_dir = out_dir / "qwen-home"
        private_dir.mkdir()
        (private_dir / "canonical-trace.jsonl").write_text(sentinel)
        (out_dir / "cleanup.json").write_text(
            json.dumps(
                {
                    "instance_created": True,
                    "instance_closed": True,
                    "containers_removed": True,
                }
            )
        )
        intent = {
            "schema_version": "fleet-selfhosted-scoring-intent-v1",
            "run_id": config["run_id"],
            "task_key": config["task"]["key"],
            "task_version_id": config["task"]["version_id"],
            "instance_id": "instance-slug",
            "evidence_run_id": "66666666-6666-4666-8666-666666666666",
            "request_sha256": "sha256:" + "a" * 64,
        }
        intent["scoring_intent_sha256"] = holdout._digest_without(
            intent, "scoring_intent_sha256"
        )
        sink = kwargs["safe_scoring_intent_sink"]
        assert callable(sink)
        sink(intent)
        raise RuntimeError(failure_stage)

    monkeypatch.setattr(holdout.self_hosted, "run", failing_run)
    durable = tmp_path / "durable"
    summary = holdout.run_campaign(plan, receipt, durable, Path("proxy.py"))
    assert summary["infrastructure_errors"] == 1
    task_dir = next(durable.glob("task-01-*"))
    assert (task_dir / "scoring-intent.json").is_file()
    assert sentinel.encode() not in b"".join(
        path.read_bytes() for path in durable.rglob("*") if path.is_file()
    )
    assert not list(scratch_root.iterdir())
