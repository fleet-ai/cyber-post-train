from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from evals.fleet import qwen38_dp6_e_postcreate_v1 as postcreate
from evals.fleet import qwen38_dp6_e_qualifier_runtime_v1 as runtime
from evals.fleet import qwen38_dp6_e_scorefree_v1 as held
from evals.fleet import self_hosted

ROOT = Path(__file__).resolve().parents[1]
HELD_PATH = ROOT / "docs/evidence/qwen38-study/2026-09-06-qwen38-dp6-e-postcreate-held-v1.json"


def _receipt(value: dict) -> dict:
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    return value


def _submission() -> dict:
    return _receipt(
        {
            "api_run_id": "ft-run-example",
            "receipt_sha256": "",
        }
    )


def _objects() -> list[dict]:
    rayjob_uid = "11111111-1111-4111-8111-111111111111"
    cluster_uid = "22222222-2222-4222-8222-222222222222"
    return [
        {
            "kind": "RayJob",
            "metadata": {"name": "ft-run-example", "uid": rayjob_uid},
            "status": {"jobStatus": "RUNNING", "rayClusterName": "ft-run-example-abcd1"},
        },
        {
            "kind": "Workload",
            "metadata": {
                "name": "workload",
                "uid": "33333333-3333-4333-8333-333333333333",
                "ownerReferences": [{"uid": rayjob_uid}],
            },
            "status": {
                "conditions": [
                    {"type": "Admitted", "status": "True"},
                    {"type": "QuotaReserved", "status": "True"},
                ]
            },
        },
        {
            "kind": "RayCluster",
            "metadata": {
                "name": "ft-run-example-abcd1",
                "uid": cluster_uid,
                "ownerReferences": [{"uid": rayjob_uid}],
            },
        },
        {
            "kind": "Service",
            "metadata": {
                "name": "ft-run-example-abcd1-head-svc",
                "uid": "44444444-4444-4444-8444-444444444444",
                "ownerReferences": [{"uid": cluster_uid}],
            },
        },
        {
            "kind": "Pod",
            "metadata": {
                "name": "ft-run-example-abcd1-head-abcde",
                "uid": "55555555-5555-4555-8555-555555555555",
                "ownerReferences": [{"uid": cluster_uid}],
            },
            "status": {
                "phase": "Running",
                "containerStatuses": [{"ready": True, "restartCount": 0}],
            },
        },
    ]


def test_postcreate_authority_is_held_and_source_bound() -> None:
    value = json.loads(HELD_PATH.read_text())
    source = ROOT / value["source_path"]
    assert value["receipt_sha256"] == self_hosted.digest_without(value, "receipt_sha256")
    assert value["source_sha256"] == self_hosted.sha256(source.read_bytes())
    assert value["launch_authorized"] is False
    assert value["scoring_authorized"] is False
    assert value["api_mutations"] == value["scored_calls"] == 0


def test_build_binding_follows_exact_uid_owner_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    submission = _submission()
    monkeypatch.setattr(runtime, "validate_binding", lambda *_args: {})
    value = postcreate.build_binding(
        {"name": "ft-run-example", "run_dir": held.RUN_DIR, "status": "RUNNING"},
        submission,
        _objects(),
        ROOT,
    )
    assert value["rayjob_uid"] == "11111111-1111-4111-8111-111111111111"
    assert value["ray_cluster_name"] == "ft-run-example-abcd1"
    assert value["ray_cluster_uid"] == "22222222-2222-4222-8222-222222222222"
    assert value["service_name"] == "ft-run-example-abcd1-head-svc"
    assert value["head_pod_name"] == "ft-run-example-abcd1-head-abcde"
    assert value["receipt_sha256"] == self_hosted.digest_without(value, "receipt_sha256")


@pytest.mark.parametrize("kind", ["Workload", "RayCluster", "Service", "Pod"])
def test_build_binding_rejects_missing_or_ambiguous_uid_chain(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime, "validate_binding", lambda *_args: {})
    items = _objects()
    missing = [row for row in items if row["kind"] != kind]
    with pytest.raises(RuntimeError):
        postcreate.build_binding(
            {"name": "ft-run-example", "run_dir": held.RUN_DIR, "status": "RUNNING"},
            _submission(),
            missing,
            ROOT,
        )
    duplicate = copy.deepcopy(items)
    duplicate.append(copy.deepcopy(next(row for row in items if row["kind"] == kind)))
    with pytest.raises(RuntimeError):
        postcreate.build_binding(
            {"name": "ft-run-example", "run_dir": held.RUN_DIR, "status": "RUNNING"},
            _submission(),
            duplicate,
            ROOT,
        )


def test_postcreate_release_is_score_free_gap_bound_and_exact() -> None:
    server_release = {"receipt_sha256": "sha256:" + "a" * 64}
    source_commit = "b" * 40
    value = {
        "schema_version": postcreate.POSTCREATE_RELEASE_SCHEMA,
        "status": "RELEASED_FOR_ONE_GAP_FREE_SCORE_FREE_DP6_E_SEQUENCE",
        "launch_authorized": True,
        "scoring_authorized": False,
        "source_commit": source_commit,
        "server_release_receipt_sha256": server_release["receipt_sha256"],
        "title": held.TITLE,
        "run_dir": held.RUN_DIR,
        "serving_block": held.SERVING_BLOCK,
        "qualifier_job": postcreate.qualifier.JOB_NAME,
        "qualifier_output_root": postcreate.qualifier.OUTPUT_ROOT,
        "server_create_limit": 1,
        "qualifier_create_limit": 1,
        "idle_release_seconds": 600,
        "gap_free_monitor_required": True,
        "statistical_cells_selected": 0,
        "scored_calls": 0,
        "prompts_traces_flags_or_scores_included": False,
    }
    _receipt(value)
    postcreate.validate_release(value, server_release, source_commit)
    for field, replacement in (
        ("scoring_authorized", True),
        ("idle_release_seconds", 601),
        ("gap_free_monitor_required", False),
    ):
        changed = {**value, field: replacement}
        _receipt(changed)
        with pytest.raises(ValueError):
            postcreate.validate_release(changed, server_release, source_commit)


def test_monitor_releases_immediately_at_terminal_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        postcreate,
        "_kubectl_json",
        lambda *_args: {
            "metadata": {"uid": "66666666-6666-4666-8666-666666666666"},
            "status": {"succeeded": 1},
        },
    )
    monkeypatch.setattr(
        postcreate,
        "_release_server",
        lambda *_args: {
            "get_before": 200,
            "delete": 204,
            "get_after": 404,
            "rayjob_workload_raycluster_pod_service_absent": True,
        },
    )
    monkeypatch.setattr(
        postcreate,
        "_stop_qualifier",
        lambda *_args: {"job_known": True, "delete": "terminal_no_action"},
    )
    monkeypatch.setattr(
        postcreate,
        "_sfs_json",
        lambda _path: {"status": "PASSED", "receipt_sha256": "sha256:" + "a" * 64},
    )
    monkeypatch.setattr(runtime, "validate_binding", lambda *_args: {})
    monkeypatch.setattr(runtime, "qualification_plan", lambda *_args: {})
    monkeypatch.setattr(runtime, "validate_result", lambda *_args: 6)
    binding = {
        "rayjob_uid": "1",
        "workload_uid": "2",
        "ray_cluster_uid": "3",
        "head_pod_uid": "4",
        "service_uid": "5",
        "head_pod_name": "pod",
        "service_origin": "http://service:8000",
    }
    receipt = postcreate.monitor_and_release(
        object(),
        "ft-run-example",
        {},
        binding,
        "66666666-6666-4666-8666-666666666666",
        ROOT,
    )
    assert receipt["qualifier_phase"] == "Succeeded"
    assert receipt["jobs_api_release"]["delete"] == 204
    assert receipt["jobs_api_release"]["rayjob_workload_raycluster_pod_service_absent"] is True
    assert receipt["scored_calls"] == 0


def test_qualifier_stop_is_uid_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    class Completed:
        returncode = 0
        stdout = '{"metadata":{"uid":"different"}}'

    monkeypatch.setattr(postcreate.subprocess, "run", lambda *_args, **_kwargs: Completed())
    with pytest.raises(RuntimeError, match="UID drift"):
        postcreate._stop_qualifier("66666666-6666-4666-8666-666666666666")  # noqa: SLF001


def test_server_release_requires_api404_and_bound_uid_absence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Response:
        def __init__(self, status_code: int) -> None:
            self.status_code = status_code

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError(self.status_code)

    class Client:
        def __init__(self) -> None:
            self.gets = 0

        def get(self, _path: str) -> Response:
            self.gets += 1
            return Response(200 if self.gets == 1 else 404)

        def delete(self, _path: str) -> Response:
            return Response(204)

    monkeypatch.setattr(postcreate, "_inventory", lambda: [])
    binding = {
        "rayjob_uid": "1",
        "workload_uid": "2",
        "ray_cluster_uid": "3",
        "head_pod_uid": "4",
        "service_uid": "5",
    }
    value = postcreate._release_server(Client(), "ft-run-example", binding)  # noqa: SLF001
    assert value == {
        "get_before": 200,
        "delete": 204,
        "get_after": 404,
        "rayjob_workload_raycluster_pod_service_absent": True,
    }
