"""The seed-45 five-arm Fleet source stays matched and launch-inert."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from evals.fleet import model_artifact, model_artifact_v2
from training.io import file_sha256

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "configs/evaluation"

CONFIGS = {
    "base": EVAL / "qwen38-base-fleet-dev17-opencode-seed45-pass1-v1.json",
    "fresh75": EVAL / "qwen38-fresh75-fleet-dev17-opencode-seed45-pass1-v1.json",
    "teacher186": EVAL / "qwen38-teacher-v5-fleet-dev17-opencode-seed45-pass1-v1.json",
    "self44": EVAL / "qwen38-self-sft-step44-fleet-dev17-opencode-seed45-pass1-v1.json",
    "lr30s76": EVAL / "qwen38-lr30-step76-fleet-dev17-opencode-seed45-pass1-v1.json",
}
PROTOCOL = EVAL / "qwen38-fleet-dev17-seed45-five-arm-matched-protocol-v1.json"
READINESS = EVAL / "qwen38-fleet-dev17-seed45-five-arm-readiness-v1.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_seed45_renderer_is_deterministic() -> None:
    subprocess.run(
        ["uv", "run", "python", "scripts/prepare_qwen38_fleet_seed45_configs.py", "--check"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_all_five_arms_share_the_exact_protected_protocol() -> None:
    configs = {name: read(path) for name, path in CONFIGS.items()}
    expected_tasks = next(iter(configs["base"]["routes"].values()))["task_versions"]
    assert len(expected_tasks) == len(set(expected_tasks)) == 17
    assert len({value["name"] for value in configs.values()}) == 5
    for config in configs.values():
        route = next(iter(config["routes"].values()))
        assert config["task_set"] == "qwen38-fresh75-fleet-dev17-task-set-v1.json"
        assert route["task_versions"] == expected_tasks
        assert config["harness"] == configs["base"]["harness"]
        assert config["images"] == configs["base"]["images"]
        assert config["pass_k"] == 1
        assert config["concurrency"] == 8
        assert config["training_data_eligible"] is False
        assert config["sampling"] == {"temperature": 0.6, "top_p": 0.95, "seed": 45}
        assert config["max_reviewed_infrastructure_retries"] == 1
    assert "model_artifact_binding" not in configs["base"]


def test_local_arms_bind_exact_packets_and_acceptance_evidence() -> None:
    for name, config_path in CONFIGS.items():
        if name == "base":
            continue
        config = read(config_path)
        runtime = config["model_artifact_binding"]
        packet_path = ROOT / runtime["packet_path"]
        evidence_path = ROOT / runtime["acceptance_evidence_path"]
        packet = read(packet_path)
        evidence = read(evidence_path)
        assert runtime["packet_file_sha256"] == file_sha256(packet_path)
        assert runtime["packet_sha256"] == "sha256:" + packet["sha256"]
        assert runtime["acceptance_evidence_file_sha256"] == file_sha256(evidence_path)
        assert runtime["acceptance_evidence_sha256"] == evidence["sha256"]
        assert (
            evidence["sha256"]
            == "sha256:"
            + hashlib.sha256(
                json.dumps(
                    {key: item for key, item in evidence.items() if key != "sha256"},
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
        )
        alias = next(iter(config["models"]))
        binding = packet["models"][alias]
        if name == "fresh75":
            assert runtime["validator_file_sha256"] == file_sha256(Path(model_artifact.__file__))
            assert model_artifact.validate_packet(config["models"], packet) == packet
            continue
        assert runtime["validator_file_sha256"] == file_sha256(Path(model_artifact_v2.__file__))
        assert model_artifact_v2.validate_packet(config["models"], packet) == packet
        model_artifact_v2._validate_acceptance_evidence(  # noqa: SLF001
            alias,
            evidence,
            binding,
            binding["checkpoint_manifest"]["required_fields"],
            binding["export_receipt"]["required_fields"],
            binding["gpu_reload_receipt"]["required_fields"],
        )


def test_teacher_revision_basis_is_explicit_not_conflated_with_export_inventory() -> None:
    config = read(CONFIGS["teacher186"])
    packet = read(ROOT / config["model_artifact_binding"]["packet_path"])
    payload = packet["models"]["teacher-v5-step186"]["payload"]
    assert payload["revision_basis"] == "export_receipt_sha256"
    assert payload["served_revision"] == config["models"]["teacher-v5-step186"]["revision"]
    assert payload["served_revision"] != payload["export_files_sha256"]


def test_readiness_is_launch_inert_and_binds_every_create_once_identity() -> None:
    protocol, readiness = read(PROTOCOL), read(READINESS)
    for value in (protocol, readiness):
        assert (
            value["sha256"]
            == "sha256:"
            + hashlib.sha256(
                json.dumps(
                    {key: item for key, item in value.items() if key != "sha256"},
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
        )
    assert protocol["comparison_arms"] == [
        "base",
        "fresh75",
        "teacher186",
        "self44",
        "lr30s76",
    ]
    assert readiness["launchable"] is False
    assert readiness["selection"]["final8_sealed"] is True
    assert readiness["create_once_gate"]["root_failure_alert_annotation"] == "off"
    assert readiness["live_evidence_binding"] == {
        "kubernetes_context": "nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6",
        "namespace": "fleet-train-jobs",
        "assert_before_every_live_read": True,
        "object_absence_valid_only_after_exact_binding_assertion": True,
    }
    assert readiness["eligibility"] == {
        "seed44_terminal_reconciled": False,
        "fresh_live_parity_complete": False,
        "duplicate_absence_complete": False,
        "root_alert_off_server_preview_complete": False,
        "eligible_now": False,
    }
    assert readiness["operation"] == {
        "jobs_created": 0,
        "config_maps_created": 0,
        "routes_resumed": 0,
        "databases_created": 0,
        "outputs_created": 0,
    }
    assert readiness["protocol"] == {
        "path": str(PROTOCOL.relative_to(ROOT)),
        "file_sha256": file_sha256(PROTOCOL),
        "sha256": protocol["sha256"],
    }
    identities = set()
    for arm, row in readiness["arms"].items():
        assert row["config_file_sha256"] == file_sha256(ROOT / row["config_path"])
        assert row["checkpoint_provenance_file_sha256"] == file_sha256(
            ROOT / row["checkpoint_provenance_path"]
        )
        identity = (row["job_name"], row["config_map_name"], row["output_root"], row["database"])
        assert identity not in identities, arm
        identities.add(identity)
