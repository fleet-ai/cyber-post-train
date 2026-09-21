import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/prepare_qwen38_skyrl_prod8_resume_packets.py"
PACKET = ROOT / "configs/qualification/qwen38-rl-reward-prod8-resume-qualification-v1.json"


def load_script():
    spec = importlib.util.spec_from_file_location("prod8_resume_packets", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_packet_is_reproducible_and_non_authorizing():
    module = load_script()
    packet = json.loads(PACKET.read_bytes())
    assert packet == module.build()
    body = {key: value for key, value in packet.items() if key != "sha256"}
    assert packet["sha256"] == module.digest(body)
    assert packet["status"] == "awaiting_accepted_prod8_step1"
    assert packet["launch_authorized"] is False
    assert packet["implementation_gate"]["ready"] is False
    assert packet["final_acceptance"] == {
        "broad_training_stays_blocked": True,
        "requires_all_owned_gpus_released": True,
        "requires_all_three_stages": True,
        "resume_qualified": False,
    }


def test_live_source_evidence_is_deliberately_unbound():
    packet = json.loads(PACKET.read_bytes())
    source = packet["source"]
    assert source["plan_sha256"] == (
        "09cabfc727e8b8448bd00a5ea3cee03914844e67cfc80b1f033bdbe2671212de"
    )
    assert source["runtime_sha256"] == (
        "acb1d1a1ff5aec9d8c153dfc57e52a13bb0b081fc666a4a0d0db139e8cb6831b"
    )
    assert all(
        source[key] is None
        for key in (
            "terminal_receipt_sha256",
            "checkpoint_manifest_sha256",
            "checkpoint_inventory_sha256",
            "reward_variation_receipt_sha256",
            "finite_update_receipt_sha256",
        )
    )
    assert "new immutable successor" in packet["binding_rule"]


def test_stage_sequence_and_resources_are_exact():
    packet = json.loads(PACKET.read_bytes())
    stages = packet["stages"]
    assert [
        (item["kind"], item["source_optimizer_step"], item["target_optimizer_step"])
        for item in stages
    ] == [
        ("zero_update_step1_reload", 1, 1),
        ("one_update_step1_to_step2", 1, 2),
        ("zero_update_step2_reload", 2, 2),
    ]
    assert [item["optimizer_updates"] for item in stages] == [0, 1, 0]
    assert len({item["name"] for item in stages}) == 3
    assert len({item["output_root"] for item in stages}) == 3
    assert len({item["wandb_run_id"] for item in stages}) == 3
    for item in stages:
        assert item["resume_mode"] == "from_path"
        assert item["launch_authorized"] is False
        assert item["resources"] == {
            "failure_alerts": "off",
            "gpus_per_node": 8,
            "image": (
                "661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/"
                "skyrl-train@sha256:89758df2b5f35cdb19efe948c7f6ef54f11e2e2ab47a45d600c25f36914e308f"
            ),
            "nodes": 1,
            "priority": "c1",
            "queue_priority": "q1",
            "requeue_if_preempted": False,
            "total_gpus": 8,
        }
    assert stages[0]["required_result"]["optimizer_calls"] == 0
    assert stages[1]["required_result"]["new_train_batches"] == [2]
    assert stages[1]["required_result"]["forbidden_train_batches"] == [1]
    assert stages[2]["required_result"]["optimizer_calls"] == 0


def test_generator_fails_if_prod8_source_identity_drifts(monkeypatch, tmp_path):
    module = load_script()
    source = json.loads(module.SOURCE.read_bytes())
    source["digests"]["runtime_sha256"] = "0" * 64
    changed = tmp_path / "source.json"
    changed.write_text(json.dumps(source))
    monkeypatch.setattr(module, "SOURCE", changed)
    try:
        module.build()
    except ValueError as error:
        assert str(error) == "prod8 launch packet identity changed"
    else:
        raise AssertionError("source drift was accepted")
