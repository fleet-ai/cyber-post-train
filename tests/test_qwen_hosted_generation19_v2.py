from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from evals.fleet import qwen_hosted_generation19_bulk as prior
from evals.fleet import qwen_hosted_generation19_v2 as g19
from evals.fleet import qwen_hosted_generation19_v2_package as package
from evals.fleet import qwen_hosted_generation19_v2_runtime as runtime
from evals.fleet import self_hosted

ROOT = Path(__file__).parents[1]
TOMBSTONE = ROOT / (
    "docs/evidence/qwen38-study/2026-09-05-qwen38-generation19-bulk-v1-terminal.json"
)


def test_successor_preserves_cells_with_fresh_controller_identities() -> None:
    plans = g19.validate_all(ROOT)
    sources = prior.validate_all(ROOT)
    assert {len(plan["attempts"]) for plan in plans.values()} == {192}
    for controller, plan in plans.items():
        assert plan["campaign_id"].endswith("-v2")
        assert plan["sfs_root"].endswith("-v2")
        assert [
            (row["cell_id"], row["execution_id"]) for row in plan["attempts"]
        ] == [
            (row["cell_id"], row["execution_id"]) for row in sources[controller]["attempts"]
        ]


def test_package_is_closed_and_instrumented(tmp_path: Path) -> None:
    rendered = package.render(ROOT)
    assert [item["kind"] for item in rendered["items"]] == [
        "ConfigMap",
        "Job",
        "ConfigMap",
        "Job",
    ]
    for cm in rendered["items"][::2]:
        script = cm["data"]["run_qwen_hosted_generation19_v2.sh"]
        assert all(name in script for name in cm["data"] if name.endswith(".py"))
        assert "qwen_hosted_generation19_v2_runtime.py" in cm["data"]
        module_root = tmp_path / cm["metadata"]["name"] / "evals" / "fleet"
        config_root = module_root / "configs"
        config_root.mkdir(parents=True)
        (module_root.parent / "__init__.py").touch()
        (module_root / "__init__.py").touch()
        for name, value in cm["data"].items():
            if name.endswith(".py"):
                (module_root / name).write_text(value)
        (config_root / "qwen-hosted-generation19-qwen-a-v2.json").write_text(
            cm["data"]["plan-v2-a.json"]
        )
        (config_root / "qwen-hosted-generation19-qwen-b-v2.json").write_text(
            cm["data"]["plan-v2-b.json"]
        )
        subprocess.run(
            [sys.executable, "-c", "import evals.fleet.qwen_hosted_generation19_v2_runtime"],
            cwd=module_root.parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
    for job in rendered["items"][1::2]:
        env = {
            row["name"]: row.get("value")
            for row in job["spec"]["template"]["spec"]["containers"][0]["env"]
        }
        assert env["G19_DIAGNOSTIC_ROOT"].endswith("-diagnostic")
        assert env["G19_CONTROLLER"] in {"qwen-a", "qwen-b"}
        assert env["G19_PREFLIGHT_SHA256"] == package.PREFLIGHT_SHA256


def test_runtime_failure_receipt_records_last_sanitized_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = g19.validate_all(ROOT)["qwen-a"]
    item = plan["attempts"][0]

    def fail(*_args: object, **kwargs: object) -> dict:
        observer = kwargs["stage_observer"]
        observer("01-plan-rebuilt", None)
        observer("02-runtime-gate-valid", None)
        observer("03-output-root-initialized", None)
        observer("04-endpoint-lease-acquired", None)
        observer("05-run-identity-absent", item)
        observer("06-route-valid", item)
        raise RuntimeError("content-free diagnostic sentinel")

    monkeypatch.setattr(runtime.engine, "run_controller", fail)
    monkeypatch.setenv("JOB_UID", "11111111-1111-4111-8111-111111111111")
    monkeypatch.setenv("POD_UID", "22222222-2222-4222-8222-222222222222")
    diagnostic = tmp_path / "diagnostic"
    with pytest.raises(RuntimeError, match="sentinel"):
        runtime.run(
            plan,
            out=tmp_path / "out",
            proxy=tmp_path / "proxy",
            diagnostic_root=diagnostic,
        )
    receipt = json.loads((diagnostic / "FAILED.json").read_text())
    assert receipt["last_completed_stage"] == "06-route-valid"
    assert receipt["claim_written"] is False
    assert receipt["model_call_started"] is False
    assert receipt["error_type"] == "RuntimeError"
    assert receipt["receipt_sha256"] == self_hosted.digest_without(receipt, "receipt_sha256")
    assert not ({"prompt", "trace", "flag", "score"} & set(receipt))


def test_v1_terminal_tombstone_is_digest_valid_and_retry_safe() -> None:
    receipt = json.loads(TOMBSTONE.read_text())
    assert receipt["global_execution_claims"] == 0
    assert receipt["model_started_cells"] == 0
    assert receipt["scored_cells"] == 0
    assert receipt["retry_allowed_for_all_planned_cells"] is True
    assert receipt["receipt_sha256"] == self_hosted.digest_without(receipt, "receipt_sha256")
