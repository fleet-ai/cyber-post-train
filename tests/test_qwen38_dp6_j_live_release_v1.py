from __future__ import annotations

import copy
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from evals.fleet import qwen38_dp6_j_live_release_v1 as release
from evals.fleet import qwen38_dp6_j_postcreate_v1 as postcreate
from evals.fleet import qwen38_dp6_j_scorefree_v1 as held
from evals.fleet import self_hosted

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "a" * 40
HELD_PATH = (
    ROOT
    / "docs/evidence/qwen38-study/2026-09-06-qwen38-dp6-j-live-release-held-v1.json"
)


class Response:
    status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"manifest_yaml": "preview"}


class Client:
    def post(self, _path: str, *, json: dict) -> Response:  # noqa: ARG002
        return Response()


def _shape() -> dict:
    return {
        "current_gpu_nodes": 1,
        "current_gpus": 8,
        "projected_gpu_nodes": 2,
        "projected_gpus": 14,
        "maximum_gpu_nodes": 2,
        "maximum_gpus": 16,
        "active_project_rayjobs": 1,
        "active_project_gpu_pods": 1,
        "coexisting_glm_v29": copy.deepcopy(held.COEXISTING_GLM_V29),
        "capacity": {
            "eligible_six_gpu_node_count": 1,
            "eligible_node_uids": ["11111111-1111-4111-8111-111111111111"],
            "b300_nominal_gpu_quota": 128,
            "b300_used_gpu_quota": 64,
            "b300_gpu_quota_headroom": 64,
            "local_queue_uid": "22222222-2222-4222-8222-222222222222",
            "cluster_queue_uid": "33333333-3333-4333-8333-333333333333",
            "priority_class": {
                "name": "fleet-infra-quiet",
                "value": -1000,
                "preemption_policy": "Never",
            },
            "peer_preemption_required": False,
        },
    }


def _current() -> dict:
    value = {
        "schema_version": release.CURRENT_INVENTORY_SCHEMA,
        "status": "CLEAR_CURRENT_EXACT_GLM_V29_COEXISTENCE",
        "observed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "source_commit": SOURCE_COMMIT,
        "active_project_serving_runs": 1,
        "allowed_existing_serving_binding": copy.deepcopy(held.COEXISTING_GLM_V29),
        "project_resource_shape": _shape(),
        "target_identity_matches": {"jobs_api": 0, "kubernetes": 0, "sfs": 0},
        "sfs_observation": {
            "observer_pod_name": "observer",
            "observer_pod_uid": "44444444-4444-4444-8444-444444444444",
            "observer_sfs_mount_path": "/shared",
            "run_dir_exists": False,
        },
        "api_mutations": 0,
        "scored_calls": 0,
        "prompts_traces_flags_or_scores_included": False,
    }
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    return value


def _server_release(current: dict) -> dict:
    config, plan, preview, historical, held_release = held.load_all(ROOT)
    value = {
        "schema_version": release.SERVER_RELEASE_SCHEMA,
        "status": "RELEASED_FOR_ONE_SCORE_FREE_DP6_J_SERVER",
        "launch_authorized": True,
        "scoring_authorized": False,
        "source_commit": SOURCE_COMMIT,
        "title": held.TITLE,
        "run_dir": held.RUN_DIR,
        "serving_block": held.SERVING_BLOCK,
        "config_sha256": config["config_sha256"],
        "plan_receipt_sha256": plan["receipt_sha256"],
        "preview_receipt_sha256": preview["receipt_sha256"],
        "historical_review_inventory_receipt_sha256": historical["receipt_sha256"],
        "held_release_receipt_sha256": held_release["receipt_sha256"],
        "current_inventory_receipt_sha256": current["receipt_sha256"],
        "current_inventory": current,
        "exact_glm_v29_coexistence_reconciled": True,
        "server_create_limit": 1,
        "qualifier_create_limit": 0,
        "statistical_cells_selected": 0,
        "scored_calls": 0,
        "prompts_traces_flags_or_scores_included": False,
    }
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    return value


def test_current_inventory_is_fresh_exact_coexistence_and_release_binds_it() -> None:
    current = _current()
    release.validate_current_inventory(current, SOURCE_COMMIT)
    value = _server_release(current)
    release.validate_server_release(value, ROOT, SOURCE_COMMIT)
    assert value["current_inventory_receipt_sha256"] == current["receipt_sha256"]
    assert postcreate.server_live is release


def test_actual_current_inventory_and_live_gate_round_trip_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sfs = {
        "observer_pod_name": "observer",
        "observer_pod_uid": "44444444-4444-4444-8444-444444444444",
        "observer_sfs_mount_path": "/shared",
        "run_dir_exists": False,
    }
    monkeypatch.setattr(
        release.base, "_active_project_runs", lambda _client: [release.base.EXPECTED_ACTIVE_API]
    )
    monkeypatch.setattr(release.base, "_kubernetes_gate", lambda _active: _shape())
    monkeypatch.setattr(
        release.base,
        "_observer_pod",
        lambda: ("observer", "44444444-4444-4444-8444-444444444444", "/shared"),
    )
    monkeypatch.setattr(release.base, "_sfs_absence", lambda _observer: copy.deepcopy(sfs))
    rendered = held._load(ROOT / held.PREVIEW_PATH)["rendered"]  # noqa: SLF001
    monkeypatch.setattr(held, "preview_identity", lambda *_args: copy.deepcopy(rendered))
    client = Client()
    current = release.current_inventory(client, SOURCE_COMMIT)
    release.validate_current_inventory(current, SOURCE_COMMIT)
    server_release = _server_release(current)
    _, gate = release.live_gate(client, server_release, ROOT, SOURCE_COMMIT)
    receipt = release.base.submission_receipt(
        "ft-run-dp6g", gate, server_release, SOURCE_COMMIT, ROOT
    )
    release.validate_submission(receipt, server_release, ROOT, SOURCE_COMMIT)
    assert gate["sfs_observation"] == sfs


def test_live_release_successor_is_held_and_source_bound() -> None:
    value = json.loads(HELD_PATH.read_text())
    assert value["receipt_sha256"] == self_hosted.digest_without(value, "receipt_sha256")
    assert value["live_release_source_sha256"] == self_hosted.sha256(
        (ROOT / value["live_release_source_path"]).read_bytes()
    )
    assert value["postcreate_source_sha256"] == self_hosted.sha256(
        (ROOT / value["postcreate_source_path"]).read_bytes()
    )
    assert value["fresh_current_exact_glm_v29_coexistence_required"] is True
    assert value["any_other_active_or_drifted_server_rejected"] is True
    assert value["launch_authorized"] is value["scoring_authorized"] is False


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("active_project_serving_runs", 0),
        ("target_identity_matches", {"jobs_api": 1, "kubernetes": 0, "sfs": 0}),
        ("scored_calls", 1),
    ],
)
def test_current_inventory_rejects_any_drift(field: str, replacement: object) -> None:
    value = _current()
    value[field] = replacement
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    with pytest.raises(ValueError):
        release.validate_current_inventory(value, SOURCE_COMMIT)


def test_current_inventory_rejects_stale_observation() -> None:
    value = _current()
    value["observed_at_utc"] = (
        datetime.now(UTC) - timedelta(seconds=release.MAX_INVENTORY_AGE_SECONDS + 1)
    ).isoformat().replace("+00:00", "Z")
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    with pytest.raises(ValueError):
        release.validate_current_inventory(value, SOURCE_COMMIT)


def test_server_release_rejects_nested_inventory_mismatch() -> None:
    current = _current()
    value = _server_release(current)
    changed = copy.deepcopy(current)
    changed["observed_at_utc"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    changed["receipt_sha256"] = self_hosted.digest_without(changed, "receipt_sha256")
    value["current_inventory"] = changed
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    with pytest.raises(ValueError):
        release.validate_server_release(value, ROOT, SOURCE_COMMIT)
