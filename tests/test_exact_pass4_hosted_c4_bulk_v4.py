from __future__ import annotations

import copy
from pathlib import Path

import pytest

from evals.fleet import exact_pass4_bulk_v3 as predecessor
from evals.fleet import exact_pass4_hosted_c4_bulk_runtime_v4 as runtime
from evals.fleet import exact_pass4_hosted_c4_bulk_v4 as bulk
from evals.fleet import hosted_concurrency4_qualification_v1 as qualifier

ROOT = Path(__file__).parents[1]
JOB_UID = "11111111-1111-4111-8111-111111111111"
POD_UID = "22222222-2222-4222-8222-222222222222"


def test_held_receipt_and_preview_contract() -> None:
    receipt = bulk.load(ROOT / bulk.HELD_PATH)
    bulk.validate_held(receipt, ROOT)
    assert receipt == bulk.expected_held(ROOT)
    assert receipt["launch_authorized"] is False
    assert receipt["objects_created"] is False
    assert receipt["planned_sessions"] == {
        "qwen3.8-27b": 399,
        "glm-5.3": 399,
        "total": 798,
    }


def test_eight_controllers_preserve_every_predecessor_identity() -> None:
    plans = bulk.validate_all(ROOT)
    assert len(plans) == 8
    prior = bulk._predecessor_rows(ROOT)
    current = {
        (plan["model"]["served_id"], row["selection_rank"], row["attempt"]): row
        for plan in plans.values()
        for row in plan["attempts"]
    }
    assert set(current) == set(prior)
    for key in prior:
        assert {
            field: current[key][field]
            for field in ("cell_id", "execution_id", "run_id", "network", "task_version_id")
        } == {
            field: prior[key][field]
            for field in ("cell_id", "execution_id", "run_id", "network", "task_version_id")
        }
    assert len({row["execution_id"] for row in current.values()}) == 798
    assert len({row["run_id"] for row in current.values()}) == 798


def test_four_streams_per_model_keep_complete_task_boundaries() -> None:
    plans = bulk.validate_all(ROOT)
    for model, partial_rank in (("qwen3.8-27b", 4), ("glm-5.3", 13)):
        selected = {
            controller: plan
            for controller, plan in plans.items()
            if plan["model"]["served_id"] == model
        }
        assert len(selected) == 4
        assert sorted(plan["new_session_count"] for plan in selected.values()) == [
            99,
            100,
            100,
            100,
        ]
        owners: dict[int, str] = {}
        for controller, plan in selected.items():
            assert plan["execution"]["endpoint_lease"] == {
                "lease_root": predecessor.LEASE_ROOT,
                "endpoint_key": plan["serving_block"],
                "maximum_streams": 4,
            }
            assert plan["execution"]["priority_class"] == "fleet-serve-low"
            assert plan["execution"]["preemption_policy"] == "Never"
            assert plan["execution"]["cpu_only"] is True
            assert plan["serving_treatment"]["dedicated_and_hosted_not_pooled"] is True
            for row in plan["attempts"]:
                owners.setdefault(row["selection_rank"], controller)
                assert owners[row["selection_rank"]] == controller
        assert set(owners) == set(range(1, 101))
        partial = [
            row
            for plan in selected.values()
            for row in plan["attempts"]
            if row["selection_rank"] == partial_rank
        ]
        assert [row["attempt"] for row in partial] == [2, 3, 4]


def test_executable_plans_preserve_exact_harness_model_tools_and_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prior_plans = predecessor.validate_all(ROOT)
    monkeypatch.setattr(bulk, "_predecessor_plans", lambda root: prior_plans)
    monkeypatch.setattr(bulk, "validate_inventory_gate", lambda receipt, root: None)
    inventory = {
        "receipt_sha256": "sha256:" + "8" * 64,
        "tasks": [
            {
                "selection_rank": rank,
                "environment": {
                    "version_id_authority": "synthetic",
                    "runtime_seed_file_count": 0,
                    "ttl_seconds": 1,
                },
                "task": {"key": f"synthetic-{rank}", "version": 1},
                "verifier": {"type": "synthetic"},
            }
            for rank in range(1, 101)
        ],
    }
    for controller, repository, revision in (
        (
            "q38-s1",
            "Qwen/Qwen3.8-27B",
            "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        ),
        (
            "glm53-s1",
            "zai-org/GLM-5.3",
            "30333038ada1f1dacb294a93270305a890b50c14",
        ),
    ):
        plan = bulk.build_runtime_plan(controller, inventory, ROOT)
        assert plan["model"]["repository"] == repository
        assert plan["model"]["revision"] == revision
        assert plan["harness"]["name"] == "opencode"
        assert plan["harness"]["version"] == "1.18.27"
        assert plan["harness"]["context_window_size"] == 262144
        assert plan["harness"]["compaction_headroom_tokens"] == 20000
        assert plan["execution"]["required_task_tools"] == ["bash", "submit_report"]
        assert plan["execution"]["training_data_eligible"] is False
        assert plan["treatment"] == predecessor.exact.EXPECTED_TREATMENT


def _bridge() -> dict:
    plans = bulk.validate_all(ROOT)
    body = {
        "schema_version": bulk.BRIDGE_SCHEMA,
        "status": "CLEAR",
        "planned_execution_count": 798,
        "exact_identity_sha256": bulk.sha256(bulk.canonical(bulk._identity_rows(ROOT))),
        "checked_job_names": sorted(plan["job_name"] for plan in plans.values()),
        "checked_output_roots": sorted(plan["sfs_root"] for plan in plans.values()),
        "checked_configmap_names": sorted(plan["configmap_name"] for plan in plans.values()),
        "collisions": {
            "fleet_api": 0,
            "kubernetes_job_pod_or_configmap": 0,
            "sfs_output": 0,
            "global_claim": 0,
            "accepted_active_or_model_started_cell": 0,
        },
        "methods": ["GET"],
        "mutation_calls": 0,
        "checked_immediately_before_release": True,
        "observed_at_utc": "2026-09-05T12:00:00Z",
        "observer_package_commit": bulk.BASE_COMMIT,
        "observer_job_succeeded": True,
        "observer_pod_restarts": 0,
        "observer_job_uid": JOB_UID,
        "observer_pod_uid": POD_UID,
        "prompts_traces_flags_or_scores_included": False,
    }
    return {**body, "receipt_sha256": bulk.digest(body, "receipt_sha256")}


def test_fresh_duplicate_bridge_binds_all_successor_names_and_cells() -> None:
    receipt = _bridge()
    bulk.validate_bridge(receipt, ROOT)
    changed = copy.deepcopy(receipt)
    changed["collisions"]["global_claim"] = 1
    changed["receipt_sha256"] = bulk.digest(changed, "receipt_sha256")
    with pytest.raises(ValueError, match="not clear"):
        bulk.validate_bridge(changed, ROOT)


def _wave(concurrency: int, streams_per_minute: float) -> dict:
    count = concurrency * 2
    stats = {"min": 1.0, "median": 1.0, "p95": 1.0, "max": 1.0}
    return {
        "concurrency": concurrency,
        "streams_started": concurrency,
        "streams_succeeded": concurrency,
        "requests_expected": count,
        "requests_succeeded": count,
        "protocol_valid_requests": count,
        "errors": 0,
        "wave_elapsed_seconds": 2.0,
        "streams_per_minute": streams_per_minute,
        "request_latency_seconds": stats,
        "stream_latency_seconds": stats,
    }


def _qualifier_model(model: str) -> dict:
    waves = [_wave(2, 60.0), _wave(4, 120.0)]
    source = qualifier.EXPECTED_MODELS[model]
    body = {
        "schema_version": qualifier.SCHEMA,
        "status": "PASSED",
        "model": {
            "served_id": model,
            "repository": source["repository"],
            "revision": source["revision"],
            "endpoint_origin": qualifier.ORIGIN,
            "response_model_exact": True,
        },
        "context_contract": {
            "expected_context_length": 262144,
            "context_length_observable": False,
            "observed_context_length": None,
            "unobservable_roster_context_requires_existing_exact_harness_gate": True,
        },
        "tool_contract": {
            "names": ["bash", "submit_report"],
            "openai_tool_schema_sha256": qualifier.sha256(
                qualifier.canonical_json(qualifier.TOOLS)
            ),
            "forced_selection_order_per_stream": ["bash", "submit_report"],
            "argument_shapes_valid": True,
            "tool_execution_performed": False,
        },
        "waves": waves,
        "decision": qualifier.evaluate_waves(waves),
        "lease": {"namespace": qualifier.LEASE_NAMESPACE, "exclusive": True},
        "request_counts": {
            "chat_completions": 12,
            "task_instance": 0,
            "session": 0,
            "scoring": 0,
            "verifier": 0,
        },
        "privacy": {
            "request_bodies_included": False,
            "response_bodies_included": False,
            "tool_arguments_included": False,
            "prompts_traces_flags_or_scores_included": False,
            "credentials_included": False,
        },
    }
    return {**body, "receipt_sha256": bulk.digest(body, "receipt_sha256")}


def _job_and_pods() -> tuple[dict, dict]:
    job = {
        "metadata": {"name": "chris-cyber-hosted-c4-qualification-v1", "uid": JOB_UID},
        "status": {
            "conditions": [{"type": "Complete", "status": "True"}],
            "active": 0,
            "succeeded": 1,
            "failed": 0,
        },
    }
    pods = {
        "items": [
            {
                "metadata": {
                    "uid": POD_UID,
                    "ownerReferences": [
                        {
                            "apiVersion": "batch/v1",
                            "kind": "Job",
                            "name": "chris-cyber-hosted-c4-qualification-v1",
                            "uid": JOB_UID,
                            "controller": True,
                        }
                    ],
                },
                "status": {
                    "phase": "Succeeded",
                    "containerStatuses": [
                        {
                            "name": "qualifier",
                            "restartCount": 0,
                            "state": {"terminated": {"exitCode": 0}},
                        }
                    ],
                },
            }
        ]
    }
    return job, pods


def _qualifier_terminal(models: dict[str, dict]) -> dict:
    body = {
        "schema_version": qualifier.SCHEMA,
        "status": "PASSED",
        "classification": "operational_gate_no_capability_claim",
        "observed_at_utc": "2026-09-05T12:00:00Z",
        "runtime": {
            "job_uid": JOB_UID,
            "pod_uid": POD_UID,
            "secret_name": qualifier.EXPECTED_SECRET_NAME,
            "secret_uid": qualifier.EXPECTED_SECRET_UID,
            "package_commit": bulk.BASE_COMMIT,
            "package_sha256": "sha256:" + "1" * 64,
            "priority_class": "fleet-serve-low",
            "preemption_policy": "Never",
            "cpu_only": True,
        },
        "models": [
            {
                "served_id": model,
                "status": "PASSED",
                "receipt_sha256": models[model]["receipt_sha256"],
            }
            for model in ("qwen3.8-27b", "glm-5.3")
        ],
        "request_counts": {
            "fleet_account_get": 1,
            "hosted_models_get": 1,
            "chat_completions": 24,
            "task_instance": 0,
            "session": 0,
            "scoring": 0,
            "verifier": 0,
        },
        "endpoint_lease": {
            "namespace": qualifier.LEASE_NAMESPACE,
            "separate_from_scored_endpoint_leases": True,
            "released": True,
        },
        "scored_bulk_launch_authorized": False,
        "privacy": {
            "request_bodies_included": False,
            "response_bodies_included": False,
            "tool_arguments_included": False,
            "prompts_traces_flags_or_scores_included": False,
            "credentials_included": False,
        },
    }
    return {**body, "receipt_sha256": bulk.digest(body, "receipt_sha256")}


def test_qualifier_gate_requires_terminal_and_both_model_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bulk.qualifier_release, "validate_release", lambda value, root: None)
    release = {
        "receipt_sha256": "sha256:" + "2" * 64,
        "implementation": {"commit": bulk.BASE_COMMIT},
        "probe_package": {"sha256": "sha256:" + "1" * 64},
    }
    models = {model: _qualifier_model(model) for model in ("qwen3.8-27b", "glm-5.3")}
    terminal = _qualifier_terminal(models)
    job, pods = _job_and_pods()
    binding = bulk.validate_qualifier_pass(release, terminal, models, job, pods, ROOT)
    assert binding["terminal_receipt_sha256"] == terminal["receipt_sha256"]
    rejected = copy.deepcopy(terminal)
    rejected["status"] = "REJECTED"
    rejected["receipt_sha256"] = bulk.digest(rejected, "receipt_sha256")
    with pytest.raises(ValueError, match="not a PASS"):
        bulk.validate_qualifier_pass(release, rejected, models, job, pods, ROOT)


def test_runtime_wrapper_selects_v4_authority(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed = {}

    def fake(plan: dict, **kwargs: object) -> dict:
        observed["bulk"] = runtime.runtime.bulk
        observed["plan"] = plan
        observed["runtime_gate_check"] = kwargs["runtime_gate_check"]
        return {"ok": True}

    monkeypatch.setattr(runtime.runtime, "run_controller", fake)
    plan = {"campaign_id": "synthetic"}
    assert runtime.run(plan, out=tmp_path / "out", proxy=tmp_path / "proxy") == {"ok": True}
    assert observed == {
        "bulk": bulk,
        "plan": plan,
        "runtime_gate_check": runtime._runtime_release_gate_check,
    }


def test_release_builder_is_fail_closed_behind_all_gates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bulk, "validate_package_commit", lambda root, commit: None)
    monkeypatch.setattr(bulk.predecessor, "validate_inventory_gate", lambda value, root: None)
    monkeypatch.setattr(
        bulk.predecessor,
        "validate_reconciliation_gate",
        lambda value, root, evidence_root=None: None,
    )
    monkeypatch.setattr(
        bulk,
        "validate_qualifier_pass",
        lambda *args, **kwargs: {
            "release_receipt_sha256": "sha256:" + "2" * 64,
            "terminal_receipt_sha256": "sha256:" + "3" * 64,
            "job_uid": JOB_UID,
            "pod_uid": POD_UID,
            "models": [
                {"model": model, "receipt_sha256": "sha256:" + digit * 64}
                for model, digit in (("qwen3.8-27b", "6"), ("glm-5.3", "7"))
            ],
        },
    )
    bridge = _bridge()
    terminal = {"receipt_sha256": "sha256:" + "4" * 64}
    inventory = {"receipt_sha256": "sha256:" + "5" * 64}
    release = bulk.build_release(
        ROOT,
        bulk.BASE_COMMIT,
        "2026-09-05T12:01:00Z",
        qualifier_release_receipt={},
        qualifier_terminal={},
        qualifier_models={},
        qualifier_job={},
        qualifier_pods={},
        bridge=bridge,
        prebulk_terminal=terminal,
        inventory_terminal=inventory,
    )
    assert release["launch_authorized"] is True
    assert release["hosted_only"] is True
    assert release["dedicated_serving_authorized"] is False
    assert len(release["controllers"]) == 8
    assert sum(row["session_count"] for row in release["controllers"]) == 798
    assert release["receipt_sha256"] == bulk.digest(release, "receipt_sha256")

    plan = next(iter(bulk.validate_all(ROOT).values()))
    runtime_plan = {**plan, "inventory_receipt": {}, "repo_root": str(ROOT)}
    monkeypatch.setattr(bulk, "build_runtime_plan", lambda *args, **kwargs: runtime_plan)
    bulk.validate_runtime_release(
        release,
        runtime_plan,
        ROOT,
        package_commit=bulk.BASE_COMMIT,
    )
    drifted = copy.deepcopy(release)
    drifted["execution"]["endpoint_lease_maximum_streams_per_model"] = 5
    drifted["receipt_sha256"] = bulk.digest(drifted, "receipt_sha256")
    with pytest.raises(ValueError, match="runtime release drifted"):
        bulk.validate_runtime_release(
            drifted,
            runtime_plan,
            ROOT,
            package_commit=bulk.BASE_COMMIT,
        )
