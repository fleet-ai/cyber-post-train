from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parents[1]
EVIDENCE = ROOT / "docs/evidence/qwen38-self-teacher-idle-route-release-20260921.json"


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def test_idle_checkpoint_routes_are_bound_to_terminal_release_evidence() -> None:
    value = json.loads(EVIDENCE.read_bytes())
    unsigned = dict(value)
    unsigned.pop("receipt_sha256")
    assert value["receipt_sha256"] == ("sha256:" + hashlib.sha256(_canonical(unsigned)).hexdigest())

    observed = datetime.fromisoformat(value["observed_at"].replace("Z", "+00:00"))
    expected = {
        "self44": (
            "05057134-546f-484e-9672-3f9ca50cec77",
            "c1271f58-ca3e-41e7-8a01-1a3603cb9619",
        ),
        "teacher186": (
            "10b1b0ce-7b02-429e-8361-084a3039f248",
            "4be8c23d-2762-44cc-ab1c-11fa19f17763",
        ),
    }
    for arm, (model_uid, former_pod_uid) in expected.items():
        route = value["routes"][arm]
        assert route["model_uid"] == model_uid
        assert route["former_serving_pod_uid"] == former_pod_uid
        assert (
            datetime.fromisoformat(route["last_useful_request_at"].replace("Z", "+00:00"))
            < observed
        )
        assert route["terminal_phase"] == "paused"
        assert route["terminal_active_pods"] == 0
        assert route["terminal_ready_replicas"] == 0
        assert route["former_serving_pod_absent"] is True
        assert route["deployment_replicas"] == 0

    assert value["consumer_census"]["active_jobs"] == 0
    assert value["consumer_census"]["active_workloads"] == 0
    assert value["lifecycle_contract"]["other_routes_touched"] is False
    assert value["lifecycle_contract"]["lr30_route_touched"] is False
    assert value["operation"] == {
        "external_mutations": 2,
        "routes_paused": 2,
        "jobs_created_or_deleted": 0,
        "model_or_benchmark_requests": 0,
        "scores_prompts_traces_flags_or_responses_read": False,
    }
    assert value["capability_result"] is False
