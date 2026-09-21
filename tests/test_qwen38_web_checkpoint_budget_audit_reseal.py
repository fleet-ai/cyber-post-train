from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SCRIPT_PATH = ROOT / "scripts/reseal_qwen38_web_checkpoint_budget_repair_audit.py"
AUDIT_PATH = (
    ROOT
    / "docs/evidence/qwen38-web-important-checkpoint-budget-repair-prompt-preflight-20260921.json"
)


def _script_module():
    spec = importlib.util.spec_from_file_location("wbe_budget_audit_reseal", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reseal(value: dict) -> dict:
    module = _script_module()
    value["receipt_sha256"] = module._self_digest(value)
    return value


def test_current_web_budget_audit_reseal_check_is_deterministic() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--check"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "audit": str(AUDIT_PATH.relative_to(ROOT)),
        "external_mutations": 0,
        "status": "checked",
    }


def test_reseal_refuses_unknown_audit_fields(tmp_path: Path) -> None:
    module = _script_module()
    value = json.loads(AUDIT_PATH.read_bytes())
    value["unexpected"] = "not permitted"
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(_reseal(value)), encoding="utf-8")
    with pytest.raises(module.AuditError, match="audit_fields_mismatch"):
        module.build_audit(root=ROOT, audit_path=path)


def test_reseal_refuses_semantic_check_drift(tmp_path: Path) -> None:
    module = _script_module()
    value = json.loads(AUDIT_PATH.read_bytes())
    value["checks"]["campaign_is_marked_launchable"] = True
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(_reseal(value)), encoding="utf-8")
    with pytest.raises(module.AuditError, match="audit_checks_invariant_drift"):
        module.build_audit(root=ROOT, audit_path=path)
