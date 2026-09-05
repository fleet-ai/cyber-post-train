from __future__ import annotations

import copy
import os
import stat
from pathlib import Path

import pytest
import yaml

from evals.fleet import autocontinue_generation2_canary_v3 as generation2
from evals.fleet import autocontinue_generation3_canary as generation3

ROOT = Path.cwd()
INCIDENT = ROOT / generation3.INCIDENT_PATH
TOMBSTONES = ROOT / generation3.TOMBSTONE_PATH
HELD = ROOT / generation3.HELD_PATH


def _load(path: Path) -> dict:
    return generation3.load(path)


def _resign(value: dict) -> dict:
    changed = copy.deepcopy(value)
    changed["receipt_sha256"] = generation3.digest(changed, "receipt_sha256")
    return changed


def test_generation2_admission_incident_is_exact_and_tamper_evident() -> None:
    receipt = _load(INCIDENT)
    generation3.validate_incident(receipt, ROOT)
    changes = (
        ("side_effects", "sessions", 1),
        ("priority_classes", "highest_installed_never", {}),
        ("retry_disposition", "fresh_execution_generation_required", 2),
    )
    for outer, inner, value in changes:
        changed = copy.deepcopy(receipt)
        changed[outer][inner] = value
        with pytest.raises(ValueError, match="incident"):
            generation3.validate_incident(_resign(changed), ROOT)


def test_generation2_tombstones_bind_exact_cells_jobs_and_absence() -> None:
    receipt = _load(TOMBSTONES)
    generation3.validate_tombstones(receipt, ROOT)
    for field, value in (
        ("generation2_execution_id", "sha256:" + "0" * 64),
        ("generation2_job", {}),
        ("replacement", {"permitted_execution_generation": 2}),
    ):
        changed = copy.deepcopy(receipt)
        changed["models"][0][field] = value
        with pytest.raises(ValueError, match="tombstones"):
            generation3.validate_tombstones(_resign(changed), ROOT)


@pytest.mark.parametrize("model", ["qwen3.8-27b", "glm-5.3"])
def test_generation3_spec_preserves_cell_and_uses_fresh_execution(model: str) -> None:
    spec = _load(ROOT / generation3.G3_SPEC_PATHS[model])
    old = _load(ROOT / generation3.G2_SPEC_PATHS[model])
    plan = generation3.validate_spec(spec, ROOT)
    assert spec["statistical_cell"] == old["statistical_cell"]
    assert spec["execution"]["execution_generation"] == 3
    assert spec["execution"]["execution_id"] != old["execution"]["execution_id"]
    assert plan["attempts"][0]["execution_generation"] == 3
    assert plan["attempts"][0]["run_id"] != generation2.validate_spec(old, ROOT)[
        "attempts"
    ][0]["run_id"]
    assert plan["execution"]["required_priority_class"] == "fleet-serve-low"
    changed = copy.deepcopy(spec)
    changed["predecessor_generation2"]["job_uid"] = (
        "11111111-1111-4111-8111-111111111111"
    )
    changed["generation3_spec_sha256"] = generation3.digest(
        changed, "generation3_spec_sha256"
    )
    with pytest.raises(ValueError, match="specification"):
        generation3.validate_spec(changed, ROOT)


def test_generation3_held_and_manifest_are_nonlaunching() -> None:
    generation3.validate_held(_load(HELD), ROOT)
    manifest = yaml.safe_load((ROOT / generation3.MANIFEST_PATH).read_text())
    assert len(manifest["items"]) == 2
    for item in manifest["items"]:
        pod = item["spec"]["template"]["spec"]
        assert item["metadata"]["annotations"] == {
            "cyber-post-train.fleet.ai/preview-only": "true",
            "cyber-post-train.fleet.ai/launch-authorized": "false",
        }
        assert item["spec"]["backoffLimit"] == 0
        assert pod["priorityClassName"] == "fleet-serve-low"
        assert pod["preemptionPolicy"] == "Never"
    submit = (ROOT / generation3.SUBMIT_PATH).read_text()
    assert '[[ "$MODE" == preview ]]' in submit
    assert "create --dry-run=server" in submit
    assert "kubectl create -f" not in submit
    assert (ROOT / generation3.SUBMIT_PATH).stat().st_mode & stat.S_IXUSR


def test_generation3_held_rejects_resealed_priority_drift() -> None:
    changed = copy.deepcopy(_load(HELD))
    changed["priority"]["class"] = "fleet-train-high"
    with pytest.raises(ValueError, match="held"):
        generation3.validate_held(_resign(changed), ROOT)


def test_generation3_claim_is_atomic_and_terminal_binds_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _load(ROOT / generation3.G3_SPEC_PATHS["qwen3.8-27b"])
    plan = generation3.validate_spec(spec, ROOT)
    claim_root = tmp_path / "claims"
    claim_root.mkdir()
    monkeypatch.setenv("JOB_UID", "11111111-1111-4111-8111-111111111111")
    monkeypatch.setenv("POD_UID", "22222222-2222-4222-8222-222222222222")
    claim = generation3.claim_execution_generation(
        spec, plan, root=ROOT, claim_root=claim_root
    )
    generation3.validate_claim(claim, spec, plan, root=ROOT)
    with pytest.raises(RuntimeError, match="already claimed"):
        generation3.claim_execution_generation(
            spec, plan, root=ROOT, claim_root=claim_root
        )
    attempt, config = generation2._expected_cell_bindings(plan)
    terminal = generation3.terminal_receipt(
        spec,
        plan,
        claim,
        {
            "accepted": False,
            "quarantined": True,
            "claim_sha256": attempt,
            "attempt_config_sha256": config,
        },
        root=ROOT,
        terminal_at_utc="2026-09-05T05:20:00Z",
    )
    generation3.validate_terminal(terminal, spec, plan, claim, root=ROOT)
    changed = copy.deepcopy(terminal)
    changed["retry_allowed"] = True
    changed["receipt_sha256"] = generation3.digest(changed, "receipt_sha256")
    with pytest.raises(ValueError, match="terminal"):
        generation3.validate_terminal(changed, spec, plan, claim, root=ROOT)


def test_generation3_claim_requires_uid_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _load(ROOT / generation3.G3_SPEC_PATHS["glm-5.3"])
    plan = generation3.validate_spec(spec, ROOT)
    root = tmp_path / "claims"
    root.mkdir()
    monkeypatch.delenv("JOB_UID", raising=False)
    monkeypatch.setenv("POD_UID", "22222222-2222-4222-8222-222222222222")
    with pytest.raises(RuntimeError, match="downward API"):
        generation3.claim_execution_generation(spec, plan, root=ROOT, claim_root=root)
    assert os.listdir(root) == []
