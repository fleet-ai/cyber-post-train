import copy
import json
from pathlib import Path

import pytest

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import glm53_dedicated_v23_request_counter_watchdog_v1 as runtime
from evals.fleet import glm53_dedicated_v24_watchdog_live_release_v1 as live_engine
from evals.fleet import glm53_dedicated_v28_watchdog_live_release_v1 as v28_adapter
from evals.fleet import glm53_dedicated_v29_controller_package_v1 as controller_package
from evals.fleet import glm53_dedicated_v29_create_v1 as server

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/glm53_v28_watchdog_live_prevalidation.json"


def test_exact_deployed_v28_metric_family_order_replays() -> None:
    receipt = json.loads(FIXTURE.read_text())
    assert receipt["receipt_sha256"] == crypto.digest_without(
        receipt, "receipt_sha256"
    )
    live = receipt["live_state"]
    binding = live["server_binding"]
    assert live["receipt_sha256"] == crypto.digest_without(live, "receipt_sha256")
    assert live["activity_metric_families"] == sorted(runtime.ACTIVITY_METRICS)
    with v28_adapter.bound_engine():
        live_engine._validate_live_state_at(  # noqa: SLF001
            live,
            binding,
            observed_now_epoch=live["observed_at_epoch"] + 0.1,
        )


def test_noncanonical_metric_family_permutation_remains_rejected() -> None:
    live = copy.deepcopy(json.loads(FIXTURE.read_text())["live_state"])
    binding = live["server_binding"]
    live["activity_metric_families"] = list(runtime.ACTIVITY_METRICS)
    live["receipt_sha256"] = crypto.digest_without(live, "receipt_sha256")
    with v28_adapter.bound_engine(), pytest.raises(
        live_engine.LiveReleaseError, match="v24_watchdog_live_state_invalid"
    ):
        live_engine._validate_live_state_at(  # noqa: SLF001
            live,
            binding,
            observed_now_epoch=live["observed_at_epoch"] + 0.1,
        )


def test_v29_server_and_controller_identities_are_fresh_and_held() -> None:
    payload = server.payload()
    server.validate_payload(payload)
    assert payload["title"].endswith("tp8-a-v29")
    assert payload["run_dir"].endswith("tp8-a-v29")
    assert "v28" not in crypto.canonical_json(payload).decode()
    job = controller_package.build_job("0" * 40)
    pod = job["spec"]["template"]["spec"]
    command = pod["containers"][0]["command"][-1]
    assert controller_package.JOB_NAME.endswith("controller-v1")
    assert pod["priorityClassName"] == "fleet-serve-low"
    assert pod["preemptionPolicy"] == "Never"
    assert "export PYTHONPATH=/workspace" in command
    assert command.index("cd /workspace") < command.index(
        "python -m evals.fleet.glm53_dedicated_v29_controller_v1"
    )
    assert "evals/fleet/glm53_dedicated_v24_watchdog_live_release_v1.py" in (
        controller_package.FILES
    )
    assert "evals/fleet/glm53_dedicated_v22_concurrency_package_v1.py" in (
        controller_package.FILES
    )
    assert not any("v28" in path for path in controller_package.FILES)
    held = controller_package.build_held()
    assert held["server_launch_authorized"] is False
    assert held["qualification_launch_authorized"] is False
    assert held["scored_launch_authorized"] is False
    assert held["api_mutation_calls"] == 0
    assert held["receipt_sha256"] == crypto.digest_without(held, "receipt_sha256")
