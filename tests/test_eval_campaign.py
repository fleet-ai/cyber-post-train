from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import evals.campaign as campaign
from evals.campaign import CampaignError, build_plan, digest, prepare, record, run, status, step

SHA = "sha256:" + "1" * 64

DRIVER = r"""
import argparse, hashlib, json
from pathlib import Path

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
def digest(value):
    return "sha256:" + hashlib.sha256(canonical(value)).hexdigest()

p = argparse.ArgumentParser()
p.add_argument("action")
p.add_argument("phase")
p.add_argument("packet")
p.add_argument("receipt")
p.add_argument("--previous")
p.add_argument("--readiness")
p.add_argument("--bad-alert", action="store_true")
p.add_argument("--provider", default="fleet")
p.add_argument("--defer-once", action="store_true")
p.add_argument("--bad-provider-gate", action="store_true")
p.add_argument("--fail-launch", action="store_true")
a = p.parse_args()
packet = json.loads(Path(a.packet).read_text())
value = {
    "schema": "cyber_eval_campaign_driver_receipt_v1",
    "experiment_key": packet["experiment_key"],
    "phase": a.phase,
    "action": a.action,
    "provider": a.provider,
}
previous = json.loads(Path(a.previous).read_text()) if a.previous else None
if a.phase == "score":
    value["collection_terminal_receipt_sha256"] = (
        previous["receipt_sha256"] if a.action == "preview"
        else previous["collection_terminal_receipt_sha256"]
    )
if a.action == "preview":
    value |= {"status": "accepted"}
    if a.phase == "rollout":
        value |= {
            "request": {"failureAlerts": False},
            "rendered_objects": [{
                "apiVersion": "batch/v1", "kind": "Job",
                "metadata": {"annotations": {
                    "fleet.ai/failure-alerts": "on" if a.bad_alert else "off"
                }},
            }],
        }
elif a.action == "ready":
    marker = Path(a.packet).with_name("deferred-once")
    if a.defer_once and not marker.exists():
        marker.write_text("deferred")
        value |= {"status": "deferred_not_ready", "defer_reason_code": "capacity_unavailable"}
    else:
        value |= {"status": "ready"}
    value["preview_receipt_sha256"] = previous["receipt_sha256"]
elif a.action == "launch":
    if a.fail_launch:
        marker = Path(a.packet).with_name("launch-attempts")
        marker.write_text((marker.read_text() if marker.exists() else "") + "x")
        raise SystemExit(1)
    readiness = json.loads(Path(a.readiness).read_text())
    value |= {"status": "created", "remote_id": "remote-" + packet["experiment_key"][-12:],
              "preview_receipt_sha256": previous["receipt_sha256"],
              "readiness_receipt_path": a.readiness,
              "readiness_receipt_sha256": readiness["receipt_sha256"]}
else:
    target = packet["identity"]["target"]["id"]
    terminal = "infrastructure_invalid" if a.phase == "rollout" and target == "bad" else "accepted"
    value |= {"status": terminal, "remote_id": previous["remote_id"],
              "launch_receipt_sha256": previous["receipt_sha256"],
              "terminal_evidence_sha256": "sha256:" + "9" * 64}
if a.provider == "tensorlake" and a.action in {"preview", "ready", "launch"}:
    value["remote_name"] = "tl-eval-" + packet["experiment_key"].removeprefix("sha256:")[:12]
    value["provider_preflight"] = {
        "shared_capacity_receipt_sha256": "sha256:" + "1" * 64,
        "provider_inventory_receipt_sha256": "sha256:" + "2" * 64,
        "create_claim_absent": not a.bad_provider_gate,
        "start_claim_absent": True,
        "remote_name_absent": True,
        "output_root_absent": True,
        "checked_at_epoch": {"preview": 1, "ready": 2, "launch": 3}[a.action],
    }
value["receipt_sha256"] = digest(value)
Path(a.receipt).write_bytes(canonical(value) + b"\n")
"""


@pytest.fixture(autouse=True)
def _canonical_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(campaign, "CANONICAL_REGISTRY", tmp_path / "registry")


def _driver(
    script: Path,
    phase: str,
    *,
    provider: str | None = None,
    bad_alert: bool = False,
    defer_once: bool = False,
    bad_provider_gate: bool = False,
    fail_launch: bool = False,
) -> dict:
    provider = provider or ("fleet" if phase == "rollout" else "local")
    common = [sys.executable, str(script)]
    preview = [
        *common,
        "preview",
        phase,
        "{packet}",
        "{receipt}",
        "--provider",
        provider,
    ]
    if phase == "score":
        preview += ["--previous", "{terminal_receipt}"]
    if bad_alert:
        preview.append("--bad-alert")
    if bad_provider_gate:
        preview.append("--bad-provider-gate")
    ready = [
        *common,
        "ready",
        phase,
        "{packet}",
        "{receipt}",
        "--previous",
        "{preview_receipt}",
        "--provider",
        provider,
    ]
    if defer_once:
        ready.append("--defer-once")
    if bad_provider_gate:
        ready.append("--bad-provider-gate")
    launch = [
        *common,
        "launch",
        phase,
        "{packet}",
        "{receipt}",
        "--previous",
        "{preview_receipt}",
        "--readiness",
        "{readiness_receipt}",
        "--provider",
        provider,
    ]
    if bad_provider_gate:
        launch.append("--bad-provider-gate")
    if fail_launch:
        launch.append("--fail-launch")
    return {
        "provider": provider,
        "submission": "jobs_api" if provider == "fleet" else "none",
        "source_sha256": SHA,
        "timeout_seconds": 10,
        "commands": {
            "preview": preview,
            "ready": ready,
            "launch": launch,
            "observe": [
                *common,
                "observe",
                phase,
                "{packet}",
                "{receipt}",
                "--previous",
                "{launch_receipt}",
                "--provider",
                provider,
            ],
        },
    }


def _config(
    script: Path,
    campaign_id: str = "matched-eval-v1",
    *,
    provider: str = "fleet",
    bad_alert: bool = False,
    defer_once: bool = False,
    bad_provider_gate: bool = False,
    fail_launch: bool = False,
) -> dict:
    models = []
    for model_id, character in (("base", "2"), ("checkpoint", "3")):
        models.append(
            {
                "id": model_id,
                "checkpoint_id": f"qwen38-{model_id}@exact",
                "weights_sha256": "sha256:" + character * 64,
                "matched_treatment_receipt_sha256": "sha256:" + "4" * 64,
                "serving_route_receipt_sha256": "sha256:" + character * 64,
                "live_parity_receipt_sha256": "sha256:" + character * 64,
            }
        )
    return {
        "schema": "cyber_eval_campaign_v1",
        "campaign_id": campaign_id,
        "scheduler": {"max_launches_per_step": 100, "serial_canaries": False},
        "pass_k": 4,
        "budgets_sha256": "sha256:" + "7" * 64,
        "matrix_sha256": "sha256:" + "6" * 64,
        "models": models,
        "benchmarks": [
            {
                "id": "web",
                "task_set_sha256": "sha256:" + "8" * 64,
                "harness": {
                    "name": "opencode",
                    "identity_receipt_sha256": "sha256:" + "b" * 64,
                },
                "scoring_protocol_sha256": "sha256:" + "e" * 64,
                "sampling": {
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "attempt_seeds": [43, 44, 45, 46],
                },
                "targets": [
                    {
                        "id": target,
                        "identity_receipt_sha256": SHA,
                        "canary": False,
                    }
                    for target in ("bad", "good")
                ],
                "rollout_driver": _driver(
                    script,
                    "rollout",
                    provider=provider,
                    bad_alert=bad_alert,
                    defer_once=defer_once,
                    bad_provider_gate=bad_provider_gate,
                    fail_launch=fail_launch,
                ),
                "score_driver": _driver(script, "score"),
            }
        ],
    }


def _prepare(tmp_path: Path, config: dict, name: str = "state") -> Path:
    source = tmp_path / f"{name}.json"
    source.write_text(json.dumps(config))
    destination = tmp_path / name
    prepare(source, destination)
    return destination


def test_plan_is_deterministic_pass4_and_campaign_name_is_not_a_duplicate_escape(tmp_path: Path):
    script = tmp_path / "driver.py"
    script.write_text(DRIVER)
    first = build_plan(_config(script, "first-campaign"))
    second = build_plan(_config(script, "second-campaign"))
    assert len(first["targets"]) == 16
    assert [row["experiment_key"] for row in first["targets"]] == [
        row["experiment_key"] for row in second["targets"]
    ]
    assert {row["identity"]["attempt"] for row in first["targets"]} == {1, 2, 3, 4}
    assert {row["identity"]["seed"] for row in first["targets"]} == {43, 44, 45, 46}


def test_logical_keys_ignore_driver_and_serving_receipt_repairs(tmp_path: Path):
    script = tmp_path / "driver.py"
    script.write_text(DRIVER)
    first_config = _config(script)
    second_config = _config(script)
    second_config["benchmarks"][0]["rollout_driver"]["source_sha256"] = "sha256:" + "0" * 64
    second_config["models"][0]["serving_route_receipt_sha256"] = "sha256:" + "f" * 64
    second_config["matrix_sha256"] = "sha256:" + "e" * 64
    first = build_plan(first_config)
    second = build_plan(second_config)
    assert [row["experiment_key"] for row in first["targets"]] == [
        row["experiment_key"] for row in second["targets"]
    ]
    assert first["plan_sha256"] != second["plan_sha256"]


def test_seed_policy_is_explicit_all_null_or_four_reviewed_integers(tmp_path: Path):
    script = tmp_path / "driver.py"
    script.write_text(DRIVER)
    config = _config(script)
    config["benchmarks"][0]["sampling"]["attempt_seeds"] = [None, None, None, None]
    assert {row["identity"]["seed"] for row in build_plan(config)["targets"]} == {None}
    config["benchmarks"][0]["sampling"]["attempt_seeds"] = [None, 2, 3, 4]
    with pytest.raises(CampaignError, match="invalid_sampling"):
        build_plan(config)


def test_campaign_wide_sampling_override_is_rejected(tmp_path: Path):
    script = tmp_path / "driver.py"
    script.write_text(DRIVER)
    config = _config(script)
    config["sampling"] = {"temperature": 0.1, "top_p": 1.0, "attempt_seeds": [1, 2, 3, 4]}
    with pytest.raises(CampaignError, match="invalid_campaign"):
        build_plan(config)


def test_model_runtime_drift_fails_closed(tmp_path: Path):
    script = tmp_path / "driver.py"
    script.write_text(DRIVER)
    config = _config(script)
    config["models"][1]["matched_treatment_receipt_sha256"] = "sha256:" + "0" * 64
    with pytest.raises(CampaignError, match="model_parity_mismatch"):
        build_plan(config)


def test_step_resumes_each_cell_and_keeps_collection_separate_from_scoring(tmp_path: Path):
    script = tmp_path / "driver.py"
    script.write_text(DRIVER)
    state = _prepare(tmp_path, _config(script))
    assert step(state)["advanced"] == 16  # rollout previews
    assert step(state)["advanced"] == 0  # no create without --execute
    assert step(state, execute=True)["advanced"] == 16
    assert step(state, execute=True)["advanced"] == 16  # rollout terminals
    current = status(state)["counts"]
    assert current == {"score_pending": 8, "rollout_infrastructure_invalid": 8}
    assert step(state)["advanced"] == 8  # score previews only for accepted rollouts
    assert step(state, execute=True)["advanced"] == 8
    assert step(state, execute=True)["advanced"] == 8
    assert status(state)["counts"] == {"complete": 8, "rollout_infrastructure_invalid": 8}


def test_fleet_preview_requires_root_alert_opt_out(tmp_path: Path):
    script = tmp_path / "driver.py"
    script.write_text(DRIVER)
    state = _prepare(tmp_path, _config(script, bad_alert=True))
    result = step(state)
    assert result["advanced"] == 0
    assert len(result["errors"]) == 16
    assert result["counts"] == {"rollout_pending": 16}


def test_capacity_deferral_precedes_claim_and_launch_intent(tmp_path: Path):
    script = tmp_path / "driver.py"
    script.write_text(DRIVER)
    state = _prepare(tmp_path, _config(script, defer_once=True))
    step(state)
    deferred = step(state, execute=True)
    assert deferred["advanced"] == 0
    assert deferred["counts"] == {"rollout_ready": 16}
    assert not list((state / "targets").glob("*/rollout/launch-intent.json"))
    assert not list((campaign.CANONICAL_REGISTRY / "claims").glob("*.json"))
    assert step(state, execute=True)["advanced"] == 16


def test_tensorlake_requires_provider_inventory_and_precreate_gates(tmp_path: Path):
    script = tmp_path / "driver.py"
    script.write_text(DRIVER)
    bad = _prepare(
        tmp_path,
        _config(script, provider="tensorlake", bad_provider_gate=True),
        "bad-provider",
    )
    result = step(bad)
    assert result["advanced"] == 0
    assert result["counts"] == {"rollout_pending": 16}

    good = _prepare(tmp_path, _config(script, provider="tensorlake"), "good-provider")
    assert step(good)["advanced"] == 16
    assert step(good, execute=True)["advanced"] == 16


def test_scheduler_bounds_launches_and_round_robins_benchmarks(tmp_path: Path):
    script = tmp_path / "driver.py"
    script.write_text(DRIVER)
    config = _config(script)
    second = json.loads(json.dumps(config["benchmarks"][0]))
    second["id"] = "web-two"
    second["task_set_sha256"] = "sha256:" + "0" * 64
    config["benchmarks"].append(second)
    config["scheduler"]["max_launches_per_step"] = 4
    state = _prepare(tmp_path, config)
    step(state)
    assert step(state, execute=True)["advanced"] == 4
    launched = [
        json.loads(path.parents[1].joinpath("packet.json").read_text())
        for path in (state / "targets").glob("*/rollout/launch.json")
    ]
    assert {row["identity"]["benchmark"]["id"] for row in launched} == {"web", "web-two"}


def test_registry_blocks_the_same_immutable_cells_from_a_second_campaign(tmp_path: Path):
    script = tmp_path / "driver.py"
    script.write_text(DRIVER)
    first = _prepare(tmp_path, _config(script, "campaign-one"), "first")
    second = _prepare(tmp_path, _config(script, "campaign-two"), "second")
    step(first)
    assert step(first, execute=True)["advanced"] == 16
    step(second)
    blocked = step(second, execute=True)
    assert blocked["advanced"] == 0
    assert len(blocked["errors"]) == 16
    assert blocked["counts"] == {"rollout_ready": 16}


def test_running_observation_does_not_block_later_terminal_evidence(tmp_path: Path):
    script = tmp_path / "driver.py"
    script.write_text(DRIVER)
    state = _prepare(tmp_path, _config(script))
    step(state)
    step(state, execute=True)
    packet = json.loads(next((state / "targets").glob("*/packet.json")).read_text())
    key = packet["experiment_key"]
    launch = json.loads((state / "targets" / key / "rollout/launch.json").read_text())

    def observation(outcome: str, name: str) -> None:
        value = {
            "schema": "cyber_eval_campaign_driver_receipt_v1",
            "experiment_key": key,
            "phase": "rollout",
            "action": "observe",
            "provider": "fleet",
            "status": outcome,
            "remote_id": launch["remote_id"],
            "launch_receipt_sha256": launch["receipt_sha256"],
        }
        if outcome != "running":
            value["terminal_evidence_sha256"] = "sha256:" + "9" * 64
        value["receipt_sha256"] = digest(value)
        path = tmp_path / name
        path.write_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n")
        record(state, key, "rollout", "observe", path)

    observation("running", "running.json")
    assert (
        next(row for row in status(state)["targets"] if row["experiment_key"] == key)["state"]
        == "rollout_created"
    )
    observation("accepted", "terminal.json")
    assert (
        next(row for row in status(state)["targets"] if row["experiment_key"] == key)["state"]
        == "score_pending"
    )


def test_run_reaches_terminal_while_preserving_infrastructure_failures(tmp_path: Path):
    script = tmp_path / "driver.py"
    script.write_text(DRIVER)
    state = _prepare(tmp_path, _config(script))

    result = run(state, poll_seconds=1, sleep=lambda _seconds: None)

    assert result["run_status"] == "terminal"
    assert result["held_targets"] == []
    assert result["counts"] == {"complete": 8, "rollout_infrastructure_invalid": 8}


def test_run_waits_through_capacity_deferral(tmp_path: Path):
    script = tmp_path / "driver.py"
    script.write_text(DRIVER)
    state = _prepare(tmp_path, _config(script, defer_once=True))
    sleeps: list[float] = []

    result = run(state, poll_seconds=3, sleep=sleeps.append)

    assert result["run_status"] == "terminal"
    assert sleeps == [3.0]


def test_run_paces_driver_errors_while_siblings_advance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    script = tmp_path / "driver.py"
    script.write_text(DRIVER)
    state = _prepare(tmp_path, _config(script))
    calls = 0

    def fake_step(_state: Path, *, execute: bool = False, _skip: set[str] | None = None) -> dict:
        nonlocal calls
        calls += 1
        return {
            "advanced": 1 if calls == 1 else 0,
            "errors": [{"experiment_key": "one", "error": "CampaignError"}] if calls == 1 else [],
            **status(state),
        }

    def stop_after_first_sleep(_seconds: float) -> None:
        raise RuntimeError("paced")

    monkeypatch.setattr(campaign, "step", fake_step)
    with pytest.raises(RuntimeError, match="paced"):
        run(state, poll_seconds=3, sleep=stop_after_first_sleep)

    assert calls == 1


def test_step_counts_failed_launch_intents_against_round_limit(tmp_path: Path):
    script = tmp_path / "driver.py"
    script.write_text(DRIVER)
    config = _config(script, fail_launch=True)
    config["scheduler"]["max_launches_per_step"] = 4
    state = _prepare(tmp_path, config)
    step(state)

    result = step(state, execute=True)

    assert len(result["errors"]) == 4
    assert result["counts"] == {"rollout_launch_uncertain": 4, "rollout_ready": 12}
    assert len(list((state / "targets").glob("*/launch-attempts"))) == 4


def test_run_holds_after_three_identical_driver_error_rounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    script = tmp_path / "driver.py"
    script.write_text(DRIVER)
    state = _prepare(tmp_path, _config(script))
    keys = [row["experiment_key"] for row in status(state)["targets"]]
    calls = 0
    sleeps: list[float] = []

    def fake_step(_state: Path, *, execute: bool = False, _skip: set[str] | None = None) -> dict:
        nonlocal calls
        calls += 1
        return {
            "advanced": 0,
            "errors": [{"experiment_key": key, "error": "CampaignError"} for key in keys],
            **status(state),
        }

    monkeypatch.setattr(campaign, "step", fake_step)
    result = run(state, poll_seconds=2, sleep=sleeps.append)

    assert calls == 3
    assert sleeps == [2.0, 2.0]
    assert result["run_status"] == "held"
    assert len(result["held_targets"]) == 16
    assert {row["reason"] for row in result["held_targets"]} == {"driver_error"}


def test_step_never_reexecutes_driver_held_targets(tmp_path: Path):
    script = tmp_path / "driver.py"
    script.write_text(DRIVER)
    state = _prepare(tmp_path, _config(script))
    step(state)
    keys = [row["experiment_key"] for row in status(state)["targets"]]

    result = step(state, execute=True, _skip=set(keys[:8]))

    assert result["counts"] == {"rollout_created": 8, "rollout_ready": 8}


def test_run_driver_held_serial_canary_holds_broad_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    script = tmp_path / "driver.py"
    script.write_text(DRIVER)
    config = _config(script)
    config["scheduler"]["serial_canaries"] = True
    config["benchmarks"][0]["targets"][0]["canary"] = True
    state = _prepare(tmp_path, config)
    canaries = {
        row["experiment_key"] for row in campaign.load_plan(state)["targets"] if row["canary"]
    }
    calls = 0

    def fake_step(_state: Path, *, execute: bool = False, _skip: set[str] | None = None) -> dict:
        nonlocal calls
        calls += 1
        return {
            "advanced": 0,
            "errors": [
                {"experiment_key": key, "error": "CampaignError"}
                for key in canaries - (_skip or set())
            ],
            **status(state),
        }

    monkeypatch.setattr(campaign, "step", fake_step)
    result = run(state, poll_seconds=1, sleep=lambda _seconds: None)

    assert calls == 3
    assert result["run_status"] == "held"
    assert len(result["held_targets"]) == 16
    assert {row["reason"] for row in result["held_targets"]} == {
        "driver_error",
        "serial_canary_not_accepted",
    }


def test_run_holds_ambiguous_launch_without_replay(tmp_path: Path):
    script = tmp_path / "driver.py"
    script.write_text(DRIVER)
    state = _prepare(tmp_path, _config(script, fail_launch=True))

    result = run(state, poll_seconds=1, sleep=lambda _seconds: None)

    assert result["run_status"] == "held"
    assert len(result["held_targets"]) == 16
    assert {row["reason"] for row in result["held_targets"]} == {"launch_uncertain"}
    markers = list((state / "targets").glob("*/launch-attempts"))
    assert len(markers) == 16
    assert all(path.read_text() == "x" for path in markers)

    rerun = run(state, poll_seconds=1, sleep=lambda _seconds: None)
    assert rerun["run_status"] == "held"
    assert all(path.read_text() == "x" for path in markers)


def test_run_stops_after_first_uncertain_serial_canary(tmp_path: Path):
    script = tmp_path / "driver.py"
    script.write_text(DRIVER)
    config = _config(script, fail_launch=True)
    config["scheduler"]["serial_canaries"] = True
    config["benchmarks"][0]["targets"][0]["canary"] = True
    state = _prepare(tmp_path, config)

    result = run(state, poll_seconds=1, sleep=lambda _seconds: None)

    assert result["run_status"] == "held"
    assert len(list((state / "targets").glob("*/launch-attempts"))) == 1
    assert len(result["held_targets"]) == 16
    assert {row["reason"] for row in result["held_targets"]} == {
        "launch_uncertain",
        "serial_canary_not_accepted",
    }


def test_serial_canary_observation_never_launches_the_next_canary(tmp_path: Path):
    script = tmp_path / "driver.py"
    script.write_text(DRIVER)
    config = _config(script)
    config["scheduler"]["serial_canaries"] = True
    config["benchmarks"][0]["targets"][0]["canary"] = True
    state = _prepare(tmp_path, config)
    step(state)

    first = step(state, execute=True)
    observed = step(state, execute=True)
    following = step(state, execute=True)

    assert first["counts"] == {"rollout_created": 1, "rollout_ready": 15}
    assert observed["counts"] == {"rollout_infrastructure_invalid": 1, "rollout_ready": 15}
    assert following["counts"] == {
        "rollout_created": 1,
        "rollout_infrastructure_invalid": 1,
        "rollout_ready": 14,
    }


def test_run_reports_serial_canary_barrier_as_explicit_hold(tmp_path: Path):
    script = tmp_path / "driver.py"
    script.write_text(DRIVER)
    config = _config(script)
    config["scheduler"]["serial_canaries"] = True
    config["benchmarks"][0]["targets"][0]["canary"] = True
    state = _prepare(tmp_path, config)

    result = run(state, poll_seconds=1, sleep=lambda _seconds: None)

    assert result["run_status"] == "held"
    assert result["counts"] == {"rollout_infrastructure_invalid": 8, "rollout_ready": 8}
    assert len(result["held_targets"]) == 8
    assert {row["reason"] for row in result["held_targets"]} == {"serial_canary_not_accepted"}


def test_run_does_not_hold_failed_canary_when_scheduler_is_not_serial(tmp_path: Path):
    script = tmp_path / "driver.py"
    script.write_text(DRIVER)
    config = _config(script)
    config["benchmarks"][0]["targets"][0]["canary"] = True
    state = _prepare(tmp_path, config)

    result = run(state, poll_seconds=1, sleep=lambda _seconds: None)

    assert result["run_status"] == "terminal"
    assert result["held_targets"] == []
    assert result["counts"] == {"complete": 8, "rollout_infrastructure_invalid": 8}


def test_run_cli_requires_explicit_execute(tmp_path: Path):
    script = tmp_path / "driver.py"
    script.write_text(DRIVER)
    state = _prepare(tmp_path, _config(script))

    with pytest.raises(SystemExit, match="2"):
        campaign.main(["run", str(state)])
