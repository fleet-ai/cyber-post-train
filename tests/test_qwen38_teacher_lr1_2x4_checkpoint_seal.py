"""The LR1 2x4 checkpoint seal accepts CPU integrity only."""

from __future__ import annotations

import json
from pathlib import Path

from training.io import digest_json, file_sha256

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/qwen38-study/2026-09-12-lr1-2x4-dev-checkpoint-seal-v1.json"
TERMINAL = ROOT / "docs/evidence/qwen38-study/2026-09-12-lr1-2x4-dev-terminal-v1.json"
RELOAD = ROOT / "configs/qualification/qwen38-teacher-lr1-cosine-2x4-reload-dev-v1.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_seal_evidence_is_self_digesting_and_binds_terminal_history() -> None:
    evidence, terminal = read(EVIDENCE), read(TERMINAL)
    assert evidence["sha256"] == digest_json(
        {key: value for key, value in evidence.items() if key != "sha256"}
    )
    bound = evidence["source"]["terminal_evidence"]
    assert bound["path"] == str(TERMINAL.relative_to(ROOT))
    assert bound["file_sha256"] == file_sha256(TERMINAL)
    assert bound["embedded_sha256"] == terminal["sha256"]
    assert evidence["source"]["api_run_id"] == terminal["run"]["api_run_id"]


def test_seal_accepts_cpu_integrity_but_not_gpu_reload_or_training() -> None:
    evidence, reload = read(EVIDENCE), read(RELOAD)
    seal, acceptance = evidence["seal"], evidence["acceptance"]
    assert seal["independent_full_file_rehash_passed"] is True
    assert seal["source_inventory_unchanged_after_sealing"] is True
    assert seal["gpu_reload_verified"] is False
    assert seal["optimizer_step"] == evidence["source"]["optimizer_step"] == 6
    assert seal["world_size"] == evidence["source"]["world_size"] == 8
    assert seal["files"] == evidence["source"]["files"] == 33
    assert seal["bytes"] == evidence["source"]["bytes"] == 324_627_486_731
    assert reload["recovery"]["manifest"] == seal["manifest_path"]
    assert reload["recovery"]["sha256"] == seal["manifest_file_sha256"].removeprefix(
        "sha256:"
    )
    assert reload["recovery"]["mode"] == "validate"
    assert acceptance["cpu_integrity_seal_accepted"] is True
    assert acceptance["checkpoint_reusable_for_bounded_zero_update_reload_test"] is True
    assert acceptance["checkpoint_gpu_reload_accepted"] is False
    assert acceptance["optimizer_continuation_authorized"] is False
    assert acceptance["production_training_authorized"] is False


def test_cpu_helper_is_released_and_privacy_boundary_is_explicit() -> None:
    evidence = read(EVIDENCE)
    execution, privacy = evidence["execution"], evidence["privacy"]
    assert execution["helper_kind"] == "standalone_cpu_pod"
    assert execution["gpus_requested"] == execution["restarts"] == 0
    assert execution["jobs_api_posts"] == 0
    assert execution["helper_deleted"] is True
    assert execution["helper_absence_verified"] is True
    assert not any(privacy.values())
