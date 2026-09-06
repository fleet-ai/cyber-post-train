import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import glm53_dedicated_v23_request_counter_watchdog_v1 as runtime
from evals.fleet import glm53_dedicated_v24_server_v1 as v24
from evals.fleet import glm53_dedicated_v24_watchdog_live_release_v1 as live_engine
from evals.fleet import glm53_dedicated_v28_watchdog_live_release_v1 as v28_adapter
from evals.fleet import glm53_dedicated_v29_controller_package_v1 as controller_package
from evals.fleet import glm53_dedicated_v29_create_v1 as server
from evals.fleet import glm53_dedicated_v29_watchdog_live_release_v1 as adapter

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/glm53_v28_watchdog_live_prevalidation.json"
COMMIT = "1fab83191b4c1455d4df72d80212377e70f9fb36"
EVIDENCE = ROOT / "docs/evidence/glm53-study"


def authorization() -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": server.SCHEMA,
        "status": "PASSED_LIVE_CREATE_GATES",
        "observed_at_epoch": time.time(),
        "server_title": server.TITLE,
        "server_run_dir": server.RUN_DIR,
        "request_sha256": server.request_sha256(),
        "preview_http_status": 200,
        "jobs_api_title_matches": 0,
        "jobs_api_run_dir_matches": 0,
        "kubernetes_identity_or_remnant_matches": 0,
        "sfs_run_dir_absent": True,
        "control_result_absent": True,
        "active_dedicated_nodes": 0,
        "active_dedicated_gpus": 0,
        "planned_nodes_after_create": 1,
        "planned_gpus_after_create": 8,
        "priority_class": v24.PRIORITY_CLASS,
        "preemption_policy": v24.PREEMPTION_POLICY,
        "server_launch_authorized": True,
        "watchdog_handoff_required_immediately": True,
        "qualification_launch_authorized": False,
        "scored_launch_authorized": False,
        "api_mutation_calls": 0,
        "protected_content_included": False,
    }
    value["receipt_sha256"] = crypto.digest_without(value, "receipt_sha256")
    return value


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


def test_v29_exact_projected_package_imports_in_isolation(tmp_path: Path) -> None:
    rendered = controller_package.render(ROOT, COMMIT, authorization())
    source, auth, job = rendered["objects"]["items"]
    package = json.loads(source["data"]["package.json"])
    assert package["package_commit"] == COMMIT
    assert set(package["files"]) == set(controller_package.FILES)
    assert len(json.dumps(source)) < 1_000_000
    assert auth["immutable"] is True
    assert job["metadata"]["name"].endswith("controller-v1")
    for relative in controller_package.FILES:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source["data"][relative.replace("/", "__SLASH__")])
    result = subprocess.run(
        [sys.executable, "-c", "import evals.fleet.glm53_dedicated_v29_controller_v1"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("2026-09-06-glm53-dedicated-v29-create-wrapper-held-v1.json", server.build_held()),
        (
            "2026-09-06-glm53-dedicated-v29-watchdog-adapter-held-v1.json",
            adapter.build_held(COMMIT),
        ),
        (
            "2026-09-06-glm53-dedicated-v29-controller-held-v1.json",
            controller_package.build_held(),
        ),
    ],
)
def test_v29_tracked_held_receipts_are_exact(
    name: str, expected: dict[str, object]
) -> None:
    observed = json.loads((EVIDENCE / name).read_text())
    assert observed == expected
    assert observed["receipt_sha256"] == crypto.digest_without(
        observed, "receipt_sha256"
    )


def test_v28_order_failure_tombstone_is_zero_effect() -> None:
    observed = json.loads(
        (
            EVIDENCE
            / "2026-09-06-glm53-dedicated-v28-watchdog-order-failure-v1.json"
        ).read_text()
    )
    assert observed["sole_failed_predicate"] == "activity_metric_families_order"
    assert observed["api_get_after_release_http_status"] == 404
    assert observed["rayjob_workload_raycluster_pod_service_remnants"] == 0
    assert observed["gpu_allocation_count_after_release"] == 0
    assert observed["task_session_verifier_scoring_calls"] == 0
    assert observed["retry_same_generation"] is False
    assert observed["receipt_sha256"] == crypto.digest_without(
        observed, "receipt_sha256"
    )
