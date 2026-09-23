"""Regression coverage for the zero-session Teacher3K candidate repair."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from evals.fleet import heldout_launch, model_artifact_v3
from scripts import prepare_qwen38_fleet_seed44_two_arm_packets as shared
from scripts import prepare_qwen38_fleet_seed46_candidate_successors as successors
from scripts import prepare_qwen38_fleet_seed46_pass8_packets as study
from scripts import render_fleet_heldout_launcher_jobs as launchers

ROOT = Path(__file__).resolve().parents[1]


def _prepare(tmp_path: Path, monkeypatch) -> Path:
    parity = tmp_path / "live-parity.json"
    parity.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        shared,
        "_live_parity",
        lambda *_args, **_kwargs: {"receipt_sha256": "sha256:" + "a" * 64},
    )
    output = tmp_path / "candidate-successors"
    successors.prepare(
        output=output,
        live_parity=parity,
        now=datetime(2026, 9, 23, 5, 30, tzinfo=UTC),
    )
    return output


def test_step1000_acceptance_is_self_digested_and_names_the_real_pod() -> None:
    path = study.ARTIFACT_ACCEPTANCE
    value = json.loads(path.read_text(encoding="utf-8"))
    claimed = value.pop("sha256")
    actual = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    assert claimed == actual
    assert value["schema"] == model_artifact_v3.ACCEPTANCE_SCHEMA
    assert value["gpu_check"]["workload_kind"] == "Pod"
    assert value["gpu_check"]["workload_uid"] == value["gpu_check"]["pod_uid"]
    assert "rayjob_uid" not in value["gpu_check"]
    assert not any(value["privacy"].values())


def test_successors_stage_one_exact_v3_packet_per_seed(tmp_path, monkeypatch) -> None:
    output = _prepare(tmp_path, monkeypatch)
    receipt = json.loads((output / "PREPARATION_RECEIPT.json").read_text())

    assert receipt["seeds"] == list(range(46, 54))
    assert receipt["selection"]["sessions"] == 136
    assert receipt["source_failure_gate"] == {
        "source_generation": 1,
        "all_source_jobs_terminal_failed": True,
        "all_source_databases_zero_rows": True,
        "all_source_outputs_absent": True,
        "candidate_sessions_created": 0,
        "failure_class": "missing_local_model_artifact_binding",
        "valid_outcomes_replayed": False,
    }
    assert len(receipt["arms"]) == 8

    for seed in range(46, 54):
        seed_dir = output / f"seed{seed}"
        packet_path = seed_dir / f"qwen38-step1000-seed{seed}-provenance-v2.json"
        config_path = seed_dir / (
            f"qwen38-teacher3k32-step1000-fleet-dev17-opencode-seed{seed}-pass1-v2.json"
        )
        packet = json.loads(packet_path.read_text())
        config = json.loads(config_path.read_text())
        binding = config["model_artifact_binding"]
        assert packet["campaign_name"] == config["name"]
        assert packet["schema"] == model_artifact_v3.PACKET_SCHEMA
        assert model_artifact_v3.validate_packet(config["models"], packet) == packet
        assert binding["packet_file_sha256"] == (
            "sha256:" + hashlib.sha256(packet_path.read_bytes()).hexdigest()
        )
        assert binding["validator_file_sha256"] == study._file_sha(  # noqa: SLF001
            Path(model_artifact_v3.__file__)
        )

        package = heldout_launch.build_package(seed_dir / "candidate/LAUNCH_PACKET.json")
        assert package.packet.identity["sampling_seed"] == seed
        assert package.packet.identity["pass_k"] == 1
        assert package.packet.identity["retry_limit"] == 0
        assert package.evaluation_config["training_data_eligible"] is False
        assert package.job["metadata"]["name"].endswith("-p1-v2")
        assert package.job["metadata"]["annotations"]["fleet.ai/failure-alerts"] == "off"
        assert package.job["spec"]["template"]["spec"]["priorityClassName"] == "c1"
        assert "nvidia.com/gpu" not in json.dumps(package.job)
        data = package.config_map["data"]
        assert {
            "model-artifact.json",
            "model-artifact-acceptance.json",
            "model_artifact_v2.py",
            "model_artifact_v3.py",
        }.issubset(data)
        assert "model_artifact_v3.py" in data["run.sh"]


def test_renderer_rejects_a_local_model_without_artifact_binding(tmp_path, monkeypatch) -> None:
    output = _prepare(tmp_path, monkeypatch)
    seed_dir = output / "seed46"
    config_path = seed_dir / (
        "qwen38-teacher3k32-step1000-fleet-dev17-opencode-seed46-pass1-v2.json"
    )
    config = json.loads(config_path.read_text())
    config.pop("model_artifact_binding")
    protocol_path = seed_dir / ("qwen38-fleet-dev17-seed46-base-step1000-pass1-protocol-v1.json")
    protocol = json.loads(protocol_path.read_text())
    with pytest.raises(ValueError, match="local SFS model requires a staged artifact binding"):
        shared._prepare_arm(  # noqa: SLF001
            tmp_path / "rejected",
            arm_id="candidate",
            arm=successors._runtime(46),  # noqa: SLF001
            config_path=config_path,
            config=config,
            task_set_path=study.TASK_SET,
            split_path=study.SPLIT,
            protocol_path=protocol_path,
            protocol=protocol,
            checkpoint_path=seed_dir / "qwen38-step1000-seed46-provenance-v2.json",
            proof_path=tmp_path / "live-parity.json",
            ledger_path=study.LEDGER,
            source_files=study.V3_SOURCE_FILES,
        )


def test_candidate_successor_launchers_are_fresh_alert_suppressed_and_cpu_only(
    tmp_path, monkeypatch
) -> None:
    packets = _prepare(tmp_path, monkeypatch)
    output = tmp_path / "candidate-launchers"
    receipt = launchers.render(packets=packets, output=output, candidate_only=True)

    assert receipt["candidate_only"] is True
    assert len(receipt["arms"]) == 8
    assert {row["replica"] for row in receipt["arms"]} == {
        f"seed{seed}-candidate" for seed in range(46, 54)
    }
    bundle = yaml.safe_load((output / "launchers.yaml").read_text(encoding="utf-8"))
    jobs = [item for item in bundle["items"] if item["kind"] == "Job"]
    assert len(jobs) == 8
    for job in jobs:
        assert job["metadata"]["annotations"]["fleet.ai/failure-alerts"] == "off"
        assert job["spec"]["template"]["spec"]["priorityClassName"] == "c1"
        assert "nvidia.com/gpu" not in json.dumps(job)
        environment = job["spec"]["template"]["spec"]["containers"][0]["env"]
        journal = next(item["value"] for item in environment if item["name"] == "CREATE_JOURNAL")
        assert "-p1-v2-CREATE_INTENT.jsonl" in journal
