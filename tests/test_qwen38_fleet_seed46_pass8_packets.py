"""Offline regressions for the matched seed-46-through-53 Fleet study."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import yaml

from evals.fleet import heldout_launch
from scripts import prepare_qwen38_fleet_seed46_pass8_packets as packets
from scripts import render_fleet_heldout_launcher_jobs as launchers


def _prepare(tmp_path: Path, monkeypatch) -> tuple[Path, dict]:
    proof = tmp_path / "live-parity.json"
    proof.write_text("{}\n", encoding="utf-8")
    digest = "sha256:" + "a" * 64
    monkeypatch.setattr(
        packets.shared,
        "_live_parity",
        lambda *_args, **_kwargs: {"receipt_sha256": digest},
    )
    output = tmp_path / "packets"
    receipt = packets.prepare(
        output=output,
        live_parity=proof,
        now=datetime(2026, 9, 22, 23, tzinfo=UTC),
    )
    return output, receipt


def test_preparer_seals_eight_matched_pass1_pairs(tmp_path, monkeypatch) -> None:
    output, receipt = _prepare(tmp_path, monkeypatch)

    assert receipt["seeds"] == list(range(46, 54))
    assert receipt["selection"]["task_count"] == 17
    assert receipt["selection"]["rollouts_per_aggregate_arm"] == 136
    assert receipt["selection"]["total_rollouts"] == 272
    assert receipt["leakage_gate"]["exact_dev_task_key_overlap"] == 0
    assert receipt["launch_performed"] is False
    assert len(receipt["arms"]) == 16

    identities = set()
    for seed in range(46, 54):
        per_seed = []
        for arm in ("base", "candidate"):
            package = heldout_launch.build_package(
                output / f"seed{seed}" / arm / "LAUNCH_PACKET.json"
            )
            per_seed.append(package)
            identities.add(package.packet.identity_sha256)
            assert package.evaluation_config["sampling"]["seed"] == seed
            assert package.evaluation_config["pass_k"] == 1
            assert package.evaluation_config["concurrency"] == 4
            assert package.evaluation_config["max_reviewed_infrastructure_retries"] == 0
            assert package.evaluation_config["training_data_eligible"] is False
            assert package.packet.identity["retry_limit"] == 0
            assert package.job["spec"]["activeDeadlineSeconds"] == 172800
            assert package.job["metadata"]["annotations"] == {
                "fleet.ai/failure-alerts": "off",
                "cyber-post-train.fleet.ai/create-once": "true",
            }
            assert package.job["spec"]["template"]["spec"]["priorityClassName"] == "c1"
            assert "nvidia.com/gpu" not in json.dumps(package.job)
        assert (
            per_seed[0].packet.identity["comparison_protocol_sha256"]
            == per_seed[1].packet.identity["comparison_protocol_sha256"]
        )
        assert (
            next(iter(per_seed[0].evaluation_config["routes"].values()))["task_versions"]
            == next(iter(per_seed[1].evaluation_config["routes"].values()))["task_versions"]
        )
    assert len(identities) == 16


def test_renderer_makes_sixteen_cpu_only_alert_suppressed_launchers(
    tmp_path, monkeypatch
) -> None:
    packet_root, _receipt = _prepare(tmp_path, monkeypatch)
    output = tmp_path / "launchers"

    receipt = launchers.render(packets=packet_root, output=output)

    assert receipt["external_mutations"] == 0
    assert receipt["launch_performed"] is False
    assert len(receipt["arms"]) == 16
    assert {row["replica"] for row in receipt["arms"]} == {
        f"seed{seed}-{arm}"
        for seed in range(46, 54)
        for arm in ("base", "candidate")
    }
    assert all(row["failure_alerts"] == "off" for row in receipt["arms"])
    assert all(row["priority_class"] == "c1" for row in receipt["arms"])
    assert all(row["gpu_requests"] == 0 for row in receipt["arms"])

    bundle = yaml.safe_load((output / "launchers.yaml").read_text(encoding="utf-8"))
    jobs = [item for item in bundle["items"] if item["kind"] == "Job"]
    config_maps = [item for item in bundle["items"] if item["kind"] == "ConfigMap"]
    assert len(jobs) == len(config_maps) == 16
    for job in jobs:
        assert job["metadata"]["annotations"]["fleet.ai/failure-alerts"] == "off"
        assert job["spec"]["template"]["spec"]["priorityClassName"] == "c1"
        assert "nvidia.com/gpu" not in json.dumps(job)
