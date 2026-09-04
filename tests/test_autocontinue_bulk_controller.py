from __future__ import annotations

import copy
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from evals.fleet import autocontinue_bulk_controller as bulk

ROOT = Path(__file__).parents[1]
PACKAGE_PATH = ROOT / "evals/fleet/configs/q38-glm53-opencode-autocontinue-bulk-held-v1.json"
HELD_AUTHORIZATION_PATH = (
    ROOT / "docs/evidence/qwen38-study/"
    "2026-09-04-opencode-autocontinue-bulk-held-authorization-v1.json"
)


def reseal(value: dict, field: str) -> None:
    value[field] = bulk.digest_without(value, field)


def observer(model: str) -> dict:
    expected = bulk.CANARIES[model]
    claim = "sha256:" + "1" * 64
    accepted = "sha256:" + "2" * 64
    final_release = "sha256:" + "3" * 64
    terminal = "sha256:" + "4" * 64
    session_id = str(uuid.uuid4())
    verifier_id = str(uuid.uuid4())
    value = {
        "schema_version": bulk.OBSERVER_SCHEMA,
        "status": "PASSED",
        "campaign_sha256": bulk.CAMPAIGN_SHA,
        "plan_sha256": expected["plan_sha256"],
        "final_release_receipt_sha256": final_release,
        "canary_terminal_receipt_sha256": terminal,
        "bulk_release_eligible": True,
        "job": {
            "name": expected["job_name"],
            "uid": str(uuid.uuid4()),
            "succeeded": 1,
            "failed": 0,
        },
        "pod": {
            "name": expected["job_name"] + "-abc12",
            "uid": str(uuid.uuid4()),
            "phase": "Succeeded",
            "exit_code": 0,
            "restart_count": 0,
        },
        "cell": {
            "source_rank": expected["source_rank"],
            "attempt": 1,
            "task_version_id": expected["task_version_id"],
            "run_id": expected["job_name"] + "-cell",
            "claim_sha256": claim,
            "session_id": session_id,
            "verifier_execution_id": verifier_id,
            "acceptance_receipt_sha256": accepted,
            "accepted": True,
            "credited": True,
            "retry_allowed": False,
        },
        "evidence": {
            "exact_claim_count": 1,
            "exact_accepted_count": 1,
            "exact_session_count": 1,
            "exact_verifier_execution_count": 1,
            "claim_sha256": claim,
            "acceptance_receipt_sha256": accepted,
            "session_id": session_id,
            "verifier_execution_id": verifier_id,
            "final_release_receipt_sha256": final_release,
            "canary_terminal_receipt_sha256": terminal,
            "quarantine_count": 0,
            "session_ingest_completed": True,
            "cleanup_completed": True,
        },
        "privacy": {
            "credentials_included": False,
            "prompts_or_traces_included": False,
            "scores_included": False,
        },
    }
    reseal(value, "receipt_sha256")
    return value


def authorization(package: dict, qwen: dict, glm: dict) -> dict:
    value = {
        "schema_version": bulk.AUTHORIZATION_SCHEMA,
        "append_only": True,
        "status": "RELEASED",
        "campaign_sha256": bulk.CAMPAIGN_SHA,
        "package_sha256": package["package_sha256"],
        "evidence": {
            "inventory_expected_sha256": bulk.INVENTORY_EXPECTED_SHA,
            "inventory_terminal_receipt_sha256": bulk.INVENTORY_TERMINAL_SHA,
            "dedicated_parity_receipt_sha256": bulk.PARITY_SHA,
            "hosted_health_receipt_sha256": bulk.HOSTED_HEALTH_SHA,
            "shared_pvc_flock_receipt_sha256": bulk.FLOCK_SHA,
            "bulk_controller_compatibility_receipt_sha256": package["evidence"][
                "bulk_controller_compatibility"
            ]["sha256"],
            "qwen_canary_observer_receipt_sha256": qwen["receipt_sha256"],
            "qwen_canary_session_id": qwen["cell"]["session_id"],
            "qwen_canary_verifier_execution_id": qwen["cell"]["verifier_execution_id"],
            "glm_canary_observer_receipt_sha256": glm["receipt_sha256"],
            "glm_canary_session_id": glm["cell"]["session_id"],
            "glm_canary_verifier_execution_id": glm["cell"]["verifier_execution_id"],
        },
        "authorization": {
            "launch_authorized": True,
            "create_once": True,
            "must_not_repeat": True,
            "required_priority_class": bulk.PRIORITY,
            "author": "/root",
            "statement": bulk.authorization_statement(package, qwen, glm),
        },
        "privacy": package["privacy"],
    }
    reseal(value, "receipt_sha256")
    return value


def preflight_receipt(plan: dict) -> dict:
    value = {
        "schema_version": "fleet-opencode-autocontinue-bulk-preflight-v2",
        "plan_sha256": plan["plan_sha256"],
        "bulk_authorization_receipt_sha256": plan["source"]["bulk_authorization_receipt_sha256"],
        "fleet_team_id": bulk.self_hosted.FLEET_TEAM_ID,
        "tasks_reconciled": plan["task_count"],
        "exact_treatment_sessions_reconciled": len(plan["credited_sessions"]),
        "active_source_attempts": 0,
        "sfs_job_roots_reconciled": 123,
        "current_plan_run_and_claim_identities_absent": True,
        "global_corrected_treatment_cells_reconciled": plan["new_session_count"],
        "global_corrected_treatment_cells_absent": True,
        "global_claim_root": bulk.GLOBAL_CLAIM_ROOT,
        "output_root_absent": True,
        "preflight_job_name": plan["preflight_job_name"],
        "preflight_configmap_name": plan["preflight_configmap_name"],
        "scored_configmap_name": plan["scored_configmap_name"],
        "configmaps_distinct": True,
        "job_uid": str(uuid.uuid4()),
        "pod_uid": str(uuid.uuid4()),
        "configmap_uid": str(uuid.uuid4()),
        "scores_read": False,
        "prompts_or_traces_read": False,
    }
    reseal(value, "receipt_sha256")
    return value


def preflight_observer(plan: dict, preflight: dict) -> dict:
    value = {
        "schema_version": "fleet-opencode-autocontinue-bulk-preflight-post-exit-v1",
        "status": "PASSED",
        "plan_sha256": plan["plan_sha256"],
        "preflight_receipt_sha256": preflight["receipt_sha256"],
        "job": {
            "name": plan["preflight_job_name"],
            "uid": preflight["job_uid"],
            "succeeded": 1,
            "failed": 0,
        },
        "pod": {
            "name": plan["preflight_job_name"] + "-abc12",
            "uid": preflight["pod_uid"],
            "phase": "Succeeded",
            "exit_code": 0,
            "restart_count": 0,
        },
        "configmap": {
            "name": plan["preflight_configmap_name"],
            "uid": preflight["configmap_uid"],
            "immutable": True,
        },
        "scored_configmap_name": plan["scored_configmap_name"],
        "stage_configmaps_distinct": True,
        "privacy": {
            "credentials_included": False,
            "prompts_or_traces_included": False,
            "scores_included": False,
        },
    }
    reseal(value, "receipt_sha256")
    return value


@pytest.fixture
def bundle(tmp_path: Path) -> tuple[dict, dict, dict, dict, Path]:
    package = bulk.load_object(PACKAGE_PATH)
    qwen = observer("qwen3.8-27b")
    glm = observer("glm-5.3")
    release = authorization(package, qwen, glm)
    out_dir = tmp_path / "materialized"
    bulk.materialize(package, release, qwen, glm, ROOT, out_dir)
    assert not (out_dir / "releases").exists()
    releases_dir = out_dir / "releases"
    preflights_dir = out_dir / "preflights"
    observers_dir = out_dir / "preflight-observers"
    releases_dir.mkdir()
    preflights_dir.mkdir()
    observers_dir.mkdir()
    for path in (out_dir / "plans").glob("*.json"):
        plan = bulk.load_object(path)
        preflight = preflight_receipt(plan)
        post_exit = preflight_observer(plan, preflight)
        shard_release = bulk.authorize_shard(
            plan,
            package,
            release,
            qwen,
            glm,
            preflight,
            post_exit,
            ROOT,
        )
        bulk.self_hosted.write_json_once(preflights_dir / path.name, preflight)
        bulk.self_hosted.write_json_once(observers_dir / path.name, post_exit)
        bulk.self_hosted.write_json_once(releases_dir / path.name, shard_release)
    return package, release, qwen, glm, out_dir


def test_held_package_is_exact_and_non_executable() -> None:
    package = bulk.load_object(PACKAGE_PATH)
    held = bulk.load_object(HELD_AUTHORIZATION_PATH)
    bulk.validate_package(package, ROOT)
    bulk.validate_held_authorization(held, package)
    assert package["launch_authorized"] is False
    assert held["launch_authorized"] is False
    assert sum(row["planned_cell_count"] for row in package["partitions"].values()) == 598


def test_authoritative_canaries_materialize_all_eight_exact_shards(bundle) -> None:
    package, release, qwen, glm, out_dir = bundle
    bulk.validate_authorization(release, package, qwen, glm)
    plans = {
        path.stem: bulk.load_object(path) for path in sorted((out_dir / "plans").glob("*.json"))
    }
    assert set(plans) == set(bulk.PARTITIONS)
    assert (
        sum(
            plan["new_session_count"]
            for plan in plans.values()
            if plan["model"]["served_id"] == "qwen3.8-27b"
        )
        == 199
    )
    assert (
        sum(
            plan["new_session_count"]
            for plan in plans.values()
            if plan["model"]["served_id"] == "glm-5.3"
        )
        == 399
    )
    all_cells: dict[str, set[tuple[int, int]]] = {
        "qwen3.8-27b": set(),
        "glm-5.3": set(),
    }
    for partition_id, plan in plans.items():
        release_row = bulk.load_object(out_dir / "releases" / f"{partition_id}.json")
        preflight = bulk.load_object(out_dir / "preflights" / f"{partition_id}.json")
        post_exit = bulk.load_object(out_dir / "preflight-observers" / f"{partition_id}.json")
        bulk.validate_release(release_row, plan, preflight, post_exit, ROOT)
        assert plan["harness"]["context_management"] == bulk.CONTEXT_POLICY
        assert plan["execution"]["required_task_tools"] == ["bash", "submit_report"]
        assert plan["execution"]["endpoint_lease"]["maximum_streams"] == 2
        assert plan["execution"]["required_priority_class"] == "fleet-train-high"
        assert plan["preflight_configmap_name"] != plan["scored_configmap_name"]
        assert plan["legacy_credited_sessions"] == 0
        model = plan["model"]["served_id"]
        cells = {(int(row["source_rank"]), int(row["attempt"])) for row in plan["attempts"]}
        assert not all_cells[model] & cells
        all_cells[model] |= cells
        by_task = {int(row["source_rank"]): 0 for row in plan["tasks"]}
        for source_rank, _ in cells:
            by_task[source_rank] += 1
        for credit in plan["credited_sessions"]:
            by_task[int(credit["source_rank"])] += 1
        assert set(by_task.values()) == {4}


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.__setitem__("launch_authorized", True),
        lambda value: value.__setitem__("legacy_credit", 1),
        lambda value: value["partitions"]["qwen-hosted-a99"].__setitem__(
            "endpoint_maximum_streams", 3
        ),
        lambda value: value["partitions"]["glm-dedicated-b56"].__setitem__(
            "planned_cell_count", 55
        ),
        lambda value: value["evidence"]["inventory_terminal"].__setitem__(
            "path", value["evidence"]["hosted_health"]["path"]
        ),
    ],
)
def test_package_rejects_resealed_tampering(mutate) -> None:
    package = copy.deepcopy(bulk.load_object(PACKAGE_PATH))
    mutate(package)
    reseal(package, "package_sha256")
    with pytest.raises(ValueError):
        bulk.validate_package(package, ROOT)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.__setitem__("bulk_release_eligible", False),
        lambda value: value["job"].__setitem__("succeeded", 0),
        lambda value: value["pod"].__setitem__("exit_code", 1),
        lambda value: value["pod"].__setitem__("restart_count", 1),
        lambda value: value["cell"].__setitem__("attempt", 2),
        lambda value: value["cell"].__setitem__("accepted", False),
        lambda value: value["evidence"].__setitem__("quarantine_count", 1),
        lambda value: value["evidence"].__setitem__("session_id", str(uuid.uuid4())),
    ],
)
def test_canary_observer_rejects_resealed_tampering(mutate) -> None:
    value = observer("qwen3.8-27b")
    mutate(value)
    reseal(value, "receipt_sha256")
    with pytest.raises(ValueError, match="canary post-exit observer"):
        bulk.validate_canary_observer(value, "qwen3.8-27b")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["authorization"].__setitem__("launch_authorized", False),
        lambda value: value["authorization"].__setitem__("author", "someone-else"),
        lambda value: value["evidence"].__setitem__(
            "inventory_terminal_receipt_sha256", "sha256:" + "0" * 64
        ),
        lambda value: value["evidence"].__setitem__(
            "bulk_controller_compatibility_receipt_sha256", "sha256:" + "0" * 64
        ),
        lambda value: value["evidence"].__setitem__("qwen_canary_session_id", str(uuid.uuid4())),
    ],
)
def test_authorization_rejects_resealed_tampering(mutate) -> None:
    package = bulk.load_object(PACKAGE_PATH)
    qwen = observer("qwen3.8-27b")
    glm = observer("glm-5.3")
    value = authorization(package, qwen, glm)
    mutate(value)
    reseal(value, "receipt_sha256")
    with pytest.raises(ValueError, match="bulk launch authorization"):
        bulk.validate_authorization(value, package, qwen, glm)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["harness"].__setitem__(
            "context_management", "opencode_1.18.27_native_compaction_no_autocontinue"
        ),
        lambda value: value["execution"].__setitem__(
            "required_task_tools", ["bash", "submit_report", "text_editor"]
        ),
        lambda value: value["execution"]["endpoint_lease"].__setitem__("maximum_streams", 3),
        lambda value: value["execution"].__setitem__(
            "required_priority_class", "fleet-infra-quiet"
        ),
        lambda value: value.__setitem__("scored_configmap_name", value["preflight_configmap_name"]),
        lambda value: value.__setitem__("legacy_credited_sessions", 1),
        lambda value: value["attempts"].__setitem__(1, value["attempts"][0]),
        lambda value: value["model"].__setitem__("revision", "0" * 40),
    ],
)
def test_materialized_plan_rejects_resealed_tampering(bundle, mutate) -> None:
    plan = bulk.load_object(bundle[-1] / "plans" / "qwen-hosted-a99.json")
    mutate(plan)
    reseal(plan, "plan_sha256")
    with pytest.raises(ValueError, match="bulk shard plan"):
        bulk.validate_plan(plan)


def test_held_authorization_cannot_materialize(tmp_path: Path) -> None:
    package = bulk.load_object(PACKAGE_PATH)
    held = bulk.load_object(HELD_AUTHORIZATION_PATH)
    with pytest.raises(ValueError, match="bulk launch authorization"):
        bulk.materialize(
            package,
            held,
            observer("qwen3.8-27b"),
            observer("glm-5.3"),
            ROOT,
            tmp_path / "must-not-exist",
        )
    assert not (tmp_path / "must-not-exist").exists()


def test_preflight_observer_rejects_resealed_uid_or_stage_drift(bundle) -> None:
    out_dir = bundle[-1]
    plan = bulk.load_object(out_dir / "plans" / "glm-dedicated-a52.json")
    preflight = bulk.load_object(out_dir / "preflights" / "glm-dedicated-a52.json")
    post_exit = bulk.load_object(out_dir / "preflight-observers" / "glm-dedicated-a52.json")
    post_exit["pod"]["restart_count"] = 1
    reseal(post_exit, "receipt_sha256")
    with pytest.raises(ValueError, match="preflight post-exit observer"):
        bulk._validate_preflight_observer(post_exit, preflight, plan)


def test_global_cell_claim_allows_only_one_concurrent_contender(bundle, tmp_path: Path) -> None:
    plan = bulk.load_object(bundle[-1] / "plans" / "qwen-hosted-b100.json")
    task = plan["tasks"][0]
    items = [row for row in plan["attempts"] if row["rank"] == task["rank"]]
    claim_root = tmp_path / "global-claims"

    def contender() -> str:
        try:
            bulk._claim_global_task_cells(plan, task, items, claim_root)
        except RuntimeError:
            return "rejected"
        return "claimed"

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(contender), pool.submit(contender)]
        outcomes = sorted(future.result() for future in futures)
    assert outcomes == ["claimed", "rejected"]
    assert len(list((claim_root / "cells").glob("*.json"))) == 4


def test_prepare_script_has_no_submit_or_cluster_mutation() -> None:
    script = (ROOT / "evals/fleet/scripts/prepare_opencode_autocontinue_bulk_v1.sh").read_text()
    assert "kubectl" not in script
    assert "--submit" not in script
    assert "materialize" in script
