import json
import shlex
import subprocess
from pathlib import Path

import pytest

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import glm53_dedicated_v23_request_counter_watchdog_v1 as watchdog
from evals.fleet import glm53_dedicated_v24_server_v1 as server

ROOT = Path(__file__).resolve().parents[1]


def test_exact_payload_has_no_internal_idle_or_process_kill_authority() -> None:
    value = server.payload()
    server.validate_payload(value)
    command = value["command"]
    assert "sglang.launch_server" in command
    assert "--context-length 262144" in command
    assert "--enable-metrics" in command
    for forbidden in (
        "GLM53_IDLE_SECONDS",
        "IDLE_SECONDS",
        "metrics_digest",
        "MODEL-TRAFFIC",
        "kill -TERM",
    ):
        assert forbidden not in command
    assert value["env"].get("GLM53_IDLE_SECONDS") is None
    assert value["workers"] == 1
    assert value["gpus_per_worker"] == 8
    assert value["priority_class"] == "fleet-infra-quiet"


def test_command_executes_only_the_exact_frozen_sglang_argv() -> None:
    value = server.payload()
    prefix = "bash -lc 'exec \"$@\"' -- "
    assert value["command"].startswith(prefix)
    assert tuple(shlex.split(value["command"][len(prefix) :])) == server.SERVER_ARGV
    assert server.SERVER_ARGV.count("--model-path") == 1
    assert server.SERVER_ARGV.count("--served-model-name") == 1


def test_held_contract_makes_external_watcher_the_only_idle_authority() -> None:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()
    value = server.build_held(commit)
    authority = value["idle_release_authority"]
    assert authority["server_internal_idle_killer_present"] is False
    assert authority["external_watchdog_required"] is True
    assert authority["implementation_sha256"] == watchdog.source_sha256()
    assert authority["activity_metrics"] == list(watchdog.ACTIVITY_METRICS)
    assert authority["active_request_or_queue_refreshes"] is True
    assert authority["idle_release_seconds"] == 600
    assert value["server_launch_authorized"] is False
    assert value["qualification_launch_authorized"] is False
    assert value["scored_launch_authorized"] is False
    assert value["receipt_sha256"] == crypto.digest_without(value, "receipt_sha256")


def test_payload_validation_rejects_any_reintroduced_idle_killer() -> None:
    value = json.loads(json.dumps(server.payload()))
    value["command"] += " ; kill -TERM 1"
    with pytest.raises(server.ServerPlanError, match="idle_authority"):
        server.validate_payload(value)


def test_tracked_preview_is_digest_valid_and_remains_held() -> None:
    path = (
        ROOT / "docs/evidence/glm53-study/"
        "2026-09-06-glm53-dedicated-v24-server-preview-held-v1.json"
    )
    value = json.loads(path.read_text())
    assert value["status"] == "PASSED_PREVIEW_ONLY_HELD"
    assert value["preview"]["http_status"] == 200
    assert value["create_request_sha256"] == crypto.sha256(crypto.canonical_json(server.payload()))
    assert value["duplicate_gate"]["jobs_api_title_matches"] == 0
    assert value["duplicate_gate"]["jobs_api_run_dir_matches"] == 0
    assert value["duplicate_gate"]["kubernetes_identity_or_remnant_matches"] == 0
    assert value["duplicate_gate"]["sfs_run_dir_absent"] is True
    assert value["capacity_gate"]["eligible_eight_gpu_nodes"] >= 1
    assert value["idle_release_authority"]["server_internal_idle_killer_present"] is False
    assert value["server_launch_authorized"] is False
    assert value["qualification_launch_authorized"] is False
    assert value["scored_launch_authorized"] is False
    assert value["receipt_sha256"] == crypto.digest_without(value, "receipt_sha256")
