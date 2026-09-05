from __future__ import annotations

import copy
from datetime import UTC, datetime
from pathlib import Path

import pytest

from evals.fleet import exact_pass4_final_bulk_package_v5 as package
from evals.fleet import exact_pass4_final_bulk_renderer_v5 as renderer
from evals.fleet import exact_pass4_final_bulk_runtime_v5 as runtime
from evals.fleet import exact_pass4_final_bulk_v5 as bulk
from evals.fleet import exact_pass4_final_dedicated_evidence_v5 as dedicated_evidence

ROOT = Path(__file__).parents[1]


def _inventory() -> dict:
    return {
        "receipt_sha256": "sha256:" + "8" * 64,
        "tasks": [
            {
                "selection_rank": rank,
                "environment": {
                    "version_id_authority": "synthetic",
                    "runtime_seed_file_count": 0,
                    "ttl_seconds": 1,
                },
                "task": {
                    "key": f"synthetic-{rank}",
                    "version": 1,
                    "version_id": f"00000000-0000-4000-8000-{rank:012d}",
                },
                "verifier": {"type": "synthetic"},
            }
            for rank in range(1, 101)
        ],
    }


def test_partition_is_exact_stratified_and_whole_task() -> None:
    plans = bulk.validate_all(ROOT)
    assert len(plans) == 14
    assert {
        group: sum(plans[key]["new_session_count"] for key in keys)
        for group, keys in bulk.GROUPS.items()
    } == {
        "hosted-qwen": 399,
        "hosted-glm": 199,
        "dedicated-a-canary": 1,
        "dedicated-a-bulk": 99,
        "dedicated-b-canary": 1,
        "dedicated-b-bulk": 99,
    }
    cells = [cell for plan in plans.values() for cell in plan["attempts"]]
    assert len(cells) == len({cell["cell_id"] for cell in cells}) == 798
    owners: dict[tuple[str, int], set[str]] = {}
    for plan in plans.values():
        for cell in plan["attempts"]:
            owners.setdefault((plan["model"], cell["selection_rank"]), set()).add(
                plan["serving_block"]
            )
    assert all(len(values) == 1 for values in owners.values())
    assert {
        rank
        for plan in plans.values()
        if plan["model"] == "glm-5.3" and plan["serving_kind"] == "hosted"
        for rank in plan["task_ranks"]
    } == set(range(1, 51))
    assert {
        rank
        for plan in plans.values()
        if plan["serving_kind"] == "dedicated"
        for rank in plan["task_ranks"]
    } == set(range(51, 101))


def test_every_identity_is_inherited_and_resource_caps_are_exact() -> None:
    plans = bulk.validate_all(ROOT)
    source = bulk.hosted._predecessor_rows(ROOT)  # noqa: SLF001
    for plan in plans.values():
        expected_max = 4 if plan["serving_kind"] == "hosted" else 2
        assert plan["endpoint_lease"]["maximum_streams"] == expected_max
        assert plan["resource_policy"] == {
            "cpu_only": True,
            "priority_class": "fleet-serve-low",
            "preemption_policy": "Never",
        }
        for cell in plan["attempts"]:
            key = (plan["model"], cell["selection_rank"], cell["attempt"])
            assert {
                field: cell[field]
                for field in ("cell_id", "execution_id", "run_id", "network", "task_version_id")
            } == {
                field: source[key][field]
                for field in ("cell_id", "execution_id", "run_id", "network", "task_version_id")
            }


def test_fresh_duplicate_gate_binds_exact_selected_group() -> None:
    group = "dedicated-a-canary"
    plans = bulk.validate_all(ROOT)
    selected = [plans[key] for key in bulk.GROUPS[group]]
    body = {
        "schema_version": bulk.FRESH_SCHEMA,
        "status": "CLEAR",
        "group": group,
        "planned_execution_count": 1,
        "exact_identity_sha256": bulk.sha256(
            bulk.canonical(bulk._group_identity(group, ROOT))  # noqa: SLF001
        ),
        "checked_job_names": sorted(row["job_name"] for row in selected),
        "checked_configmap_names": sorted(row["configmap_name"] for row in selected),
        "checked_output_roots": sorted(row["sfs_root"] for row in selected),
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
        "observer_job_succeeded": True,
        "observer_pod_restarts": 0,
        "observer_package_commit": bulk.BASE_COMMIT,
        "observer_job_uid": "00000000-0000-4000-8000-000000000020",
        "observer_pod_uid": "00000000-0000-4000-8000-000000000021",
        "observed_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "prompts_traces_flags_or_scores_included": False,
        "credentials_included": False,
    }
    receipt = {**body, "receipt_sha256": bulk.digest(body)}
    bulk.validate_fresh_duplicate(receipt, group, ROOT, bulk.BASE_COMMIT)
    changed = copy.deepcopy(receipt)
    changed["checked_job_names"] = []
    changed["receipt_sha256"] = bulk.digest(changed)
    with pytest.raises(ValueError, match="not clear"):
        bulk.validate_fresh_duplicate(changed, group, ROOT, bulk.BASE_COMMIT)


def test_runtime_plans_keep_exact_harness_and_separate_endpoint_origins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bulk.hosted, "validate_inventory_gate", lambda value, root: None)
    inventory = _inventory()
    hosted_plan = bulk.build_runtime_plan("glm53-hosted-s1", inventory, ROOT)
    dedicated_plan = bulk.build_runtime_plan("glm53-dedicated-a-s1", inventory, ROOT)
    for plan in (hosted_plan, dedicated_plan):
        assert plan["harness"]["name"] == "opencode"
        assert plan["harness"]["version"] == "1.18.27"
        assert plan["harness"]["context_window_size"] == 262144
        assert plan["harness"]["compaction_headroom_tokens"] == 20000
        assert plan["execution"]["required_task_tools"] == ["bash", "submit_report"]
        settings = bulk.self_hosted.opencode_settings(plan)
        canonical = bulk.self_hosted.canonical_json(settings)
        assert plan["harness"]["settings_canonical_sha256"] == bulk.self_hosted.sha256(canonical)
    assert hosted_plan["model"]["endpoint_origin"] == "https://inference.flt.build"
    assert "glm53-dedicated-a-v7" in dedicated_plan["model"]["endpoint_origin"]
    assert hosted_plan["serving_block"] != dedicated_plan["serving_block"]


def _wave(concurrency: int, throughput: float) -> dict:
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
        "streams_per_minute": throughput,
        "request_latency_seconds": stats,
        "stream_latency_seconds": stats,
    }


def _qualifier_model(model: str) -> dict:
    waves = [_wave(2, 60.0), _wave(4, 120.0)]
    source = bulk.qualifier.EXPECTED_MODELS[model]
    body = {
        "schema_version": bulk.qualifier.SCHEMA,
        "status": "PASSED",
        "model": {
            "served_id": model,
            "repository": source["repository"],
            "revision": source["revision"],
            "endpoint_origin": bulk.qualifier.ORIGIN,
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
            "openai_tool_schema_sha256": bulk.qualifier.sha256(
                bulk.qualifier.canonical_json(bulk.qualifier.TOOLS)
            ),
            "forced_selection_order_per_stream": ["bash", "submit_report"],
            "argument_shapes_valid": True,
            "tool_execution_performed": False,
        },
        "waves": waves,
        "decision": bulk.qualifier.evaluate_waves(waves),
        "lease": {"namespace": bulk.qualifier.LEASE_NAMESPACE, "exclusive": True},
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
    return {**body, "receipt_sha256": bulk.digest(body)}


def _qualifier_terminal(model_receipt: dict) -> dict:
    body = {
        "schema_version": bulk.qualifier.SCHEMA,
        "status": "REJECTED",
        "classification": "operational_gate_no_capability_claim",
        "models": [
            {
                "served_id": model_receipt["model"]["served_id"],
                "status": "PASSED",
                "receipt_sha256": model_receipt["receipt_sha256"],
            },
            {"served_id": "other", "status": "REJECTED", "receipt_sha256": "sha256:" + "9" * 64},
        ],
        "endpoint_lease": {
            "namespace": bulk.qualifier.LEASE_NAMESPACE,
            "separate_from_scored_endpoint_leases": True,
            "released": True,
        },
        "scored_bulk_launch_authorized": False,
    }
    return {**body, "receipt_sha256": bulk.digest(body)}


def test_model_specific_hosted_release_does_not_require_other_model_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bulk, "validate_package_commit", lambda root, commit: None)
    monkeypatch.setattr(bulk, "_validate_prebulk", lambda value, root: None)
    monkeypatch.setattr(bulk, "validate_fresh_duplicate", lambda value, group, root, commit: None)
    monkeypatch.setattr(
        bulk,
        "validate_selected_qualifier",
        lambda *args: {
            "launch_release_receipt_sha256": "sha256:" + "7" * 64,
            "terminal_receipt_sha256": "sha256:" + "6" * 64,
            "model_receipt_sha256": "sha256:" + "5" * 64,
            "job_uid": "00000000-0000-4000-8000-000000000003",
            "pod_uid": "00000000-0000-4000-8000-000000000004",
        },
    )
    model = _qualifier_model("qwen3.8-27b")
    terminal = _qualifier_terminal(model)
    release = bulk.build_release(
        "hosted-qwen",
        ROOT,
        bulk.BASE_COMMIT,
        prebulk_terminal={"receipt_sha256": "sha256:" + "1" * 64},
        fresh_duplicate={"receipt_sha256": "sha256:" + "4" * 64},
        qualifier_launch_release={},
        qualifier_model=model,
        qualifier_terminal=terminal,
        qualifier_job={},
        qualifier_pods={},
    )
    assert release["group"] == "hosted-qwen"
    assert release["credential_authority"] == {
        "secret_name": bulk.SECRET_NAME,
        "secret_uid": bulk.SECRET_UID,
        "fleet_team_id": bulk.FLEET_TEAM_ID,
    }
    assert len(release["controllers"]) == 4
    assert sum(row["session_count"] for row in release["controllers"]) == 399
    changed = copy.deepcopy(model)
    changed["decision"]["accepted"] = False
    changed["receipt_sha256"] = bulk.digest(changed)
    with pytest.raises(ValueError, match="did not pass"):
        bulk.build_release(
            "hosted-qwen",
            ROOT,
            bulk.BASE_COMMIT,
            prebulk_terminal={"receipt_sha256": "sha256:" + "1" * 64},
            fresh_duplicate={"receipt_sha256": "sha256:" + "4" * 64},
            qualifier_launch_release={},
            qualifier_model=changed,
            qualifier_terminal=terminal,
            qualifier_job={},
            qualifier_pods={},
        )


def test_held_package_is_bounded_and_content_addressed() -> None:
    bulk.validate_held(bulk.load(ROOT / bulk.HELD_PATH), ROOT)
    built = package.build_package(ROOT)
    assert built["launch_authorized"] is False
    assert built["release_included"] is False
    assert len(built["controller_manifests"]) == 14
    assert all(
        size < package.prior.PACKAGE_OBJECT_LIMIT for size in built["object_json_bytes"].values()
    )


def test_renderer_emits_independent_create_only_hosted_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bulk.hosted, "validate_inventory_gate", lambda value, root: None)
    release = {
        "group": "hosted-qwen",
        "package_commit": bulk.BASE_COMMIT,
        "receipt_sha256": "sha256:" + "2" * 64,
    }
    monkeypatch.setattr(bulk, "build_release", lambda *args, **kwargs: release)
    rendered = renderer.render(release, _inventory(), {}, {}, ROOT)
    jobs = [item for item in rendered["items"] if item["kind"] == "Job"]
    services = [item for item in rendered["items"] if item["kind"] == "Service"]
    assert len(jobs) == 4
    assert services == []
    for job in jobs:
        spec = job["spec"]["template"]["spec"]
        assert spec["priorityClassName"] == "fleet-serve-low"
        assert spec["preemptionPolicy"] == "Never"
        assert job["metadata"]["annotations"]["cyber-post-train.fleet.ai/create-once"] == "true"
        assert job["metadata"]["annotations"]["cyber-post-train.fleet.ai/secret-uid"] == (
            bulk.SECRET_UID
        )
        env = {row["name"]: row for row in spec["containers"][0]["env"]}
        assert env["FLEET_API_KEY"]["valueFrom"]["secretKeyRef"]["name"] == bulk.SECRET_NAME
        assert (
            job["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] == "exact100-final-v5"
        )


def test_renderer_emits_one_dedicated_canary_and_exact_alias_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bulk.hosted, "validate_inventory_gate", lambda value, root: None)
    release = {
        "group": "dedicated-a-canary",
        "package_commit": bulk.BASE_COMMIT,
        "receipt_sha256": "sha256:" + "3" * 64,
    }
    parity = {
        "ray_cluster_uid": "00000000-0000-4000-8000-000000000010",
        "service_uid": "00000000-0000-4000-8000-000000000011",
        "service_selector": {
            "ray.io/cluster": "exact-ray-cluster",
            "ray.io/node-type": "head",
        },
    }
    monkeypatch.setattr(bulk, "build_release", lambda *args, **kwargs: release)
    rendered = renderer.render(release, _inventory(), {}, {}, ROOT, dedicated_parity=parity)
    jobs = [item for item in rendered["items"] if item["kind"] == "Job"]
    services = [item for item in rendered["items"] if item["kind"] == "Service"]
    assert len(jobs) == len(services) == 1
    assert services[0]["metadata"]["name"] == "chris-cyber-glm53-dedicated-a-v7"
    assert services[0]["spec"]["selector"] == parity["service_selector"]
    assert services[0]["spec"]["ports"] == [
        {"name": "http", "port": 8000, "targetPort": 8000, "protocol": "TCP"}
    ]
    assert jobs[0]["spec"]["template"]["spec"]["priorityClassName"] == "fleet-serve-low"
    assert jobs[0]["spec"]["template"]["spec"]["preemptionPolicy"] == "Never"


def test_runtime_wrapper_overrides_v3_gate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen = {}

    def fake(plan: dict, **kwargs: object) -> dict:
        seen.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(runtime.engine, "run_controller", fake)
    monkeypatch.setattr(
        runtime, "_runtime_release_gate", lambda plan: seen.setdefault("gated", True)
    )
    plan = {"serving_kind": "hosted"}
    assert runtime.run(plan, out=tmp_path / "out", proxy=tmp_path / "proxy") == {"ok": True}
    assert seen["gated"] is True
    assert seen["runtime_gate_check"](plan) is None
    assert seen["route_check"] is runtime._route_check
    assert runtime.engine.bulk is bulk


def test_runtime_receipt_producer_is_exact_and_revalidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = {}
    monkeypatch.setattr(dedicated_evidence.dedicated, "spec", lambda root: {"ok": True})
    monkeypatch.setattr(
        dedicated_evidence.dedicated,
        "_parity_runtime_identity",
        lambda parity, value, replica: {"replica_identity": replica},
    )
    monkeypatch.setattr(
        dedicated_evidence.dedicated,
        "validate_runtime_gate",
        lambda receipt, parity, canary, value, root, replica: seen.update(
            {"receipt": receipt, "replica": replica}
        ),
    )
    parity = {"receipt_sha256": "sha256:" + "a" * 64}
    canary = {
        "cell_id": "cell",
        "execution_id": "sha256:" + "b" * 64,
        "claim_path": "/claims/b.json",
        "claim_receipt_sha256": "sha256:" + "c" * 64,
        "run_id": "chris-cyber-run",
        "session_id": "00000000-0000-4000-8000-000000000001",
        "verifier_execution_id": "00000000-0000-4000-8000-000000000002",
        "receipt_sha256": "sha256:" + "d" * 64,
    }
    receipt = dedicated_evidence.runtime_receipt(parity, canary, ROOT, "A")
    assert receipt == seen["receipt"]
    assert seen["replica"] == "A"
    assert receipt["released_streams"] == ["dedicated-a-1", "dedicated-a-2"]
    assert receipt["receipt_sha256"] == bulk.digest(receipt)
