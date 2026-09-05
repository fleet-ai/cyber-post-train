from __future__ import annotations

import copy
import json
import re
from contextlib import nullcontext
from pathlib import Path

import pytest
import yaml

from evals.fleet import autocontinue_generation2_canary_v3 as contract
from evals.fleet import autocontinue_generation2_runtime_v3 as runtime
from evals.fleet import self_hosted

ROOT = Path.cwd()
Q_SPEC = Path("evals/fleet/configs/qwen38-opencode-autocontinue-canary-generation2-v2.json")
G_SPEC = Path("evals/fleet/configs/glm53-opencode-autocontinue-canary-generation2-v2.json")
HELD = ROOT / runtime.EXECUTABLE_HELD_PATH
MANIFEST = ROOT / runtime.MANIFEST_PATH
RUN = ROOT / runtime.RUN_PATH
SUBMIT = ROOT / runtime.SUBMIT_PATH


def _load(path: Path) -> dict:
    return runtime.load(path)


def test_executable_held_receipt_binds_contract_and_complete_package() -> None:
    specs = [_load(Q_SPEC), _load(G_SPEC)]
    receipt = _load(HELD)
    runtime.validate_executable_held(receipt, specs, ROOT)
    assert receipt["launch_authorized"] is False
    assert receipt["supersedes_contract_held_receipt"] == {
        "path": runtime.CONTRACT_HELD_PATH,
        "receipt_sha256": _load(ROOT / runtime.CONTRACT_HELD_PATH)["receipt_sha256"],
    }
    for prefix in ("contract_module", "runtime_module", "manifest", "run", "submit"):
        path = ROOT / receipt["implementation"][f"{prefix}_path"]
        assert receipt["implementation"][f"{prefix}_sha256"] == runtime.file_sha256(
            path
        )


def test_executable_held_receipt_tampering_fails_closed() -> None:
    specs = [_load(Q_SPEC), _load(G_SPEC)]
    receipt = copy.deepcopy(_load(HELD))
    receipt["implementation"]["manifest_sha256"] = "sha256:" + "0" * 64
    receipt["receipt_sha256"] = runtime.digest(receipt, "receipt_sha256")
    with pytest.raises(ValueError, match="executable held receipt drifted"):
        runtime.validate_executable_held(receipt, specs, ROOT)


def test_v3_manifest_is_paired_held_create_once_shape() -> None:
    items = yaml.safe_load(MANIFEST.read_text())["items"]
    assert {item["metadata"]["name"] for item in items} == {
        "chris-q38-ac-r004-a1-g2-v1",
        "chris-glm53-ac-r013-a1-g2-v1",
    }
    for item in items:
        assert item["metadata"]["annotations"] == {
            "cyber-post-train.fleet.ai/preview-only": "true",
            "cyber-post-train.fleet.ai/launch-authorized": "false",
        }
        assert item["spec"]["backoffLimit"] == 0
        pod = item["spec"]["template"]["spec"]
        assert pod["restartPolicy"] == "Never"
        assert pod["priorityClassName"] == "fleet-train-high"
        assert pod["preemptionPolicy"] == "Never"
        assert len(pod["containers"]) == 1
        assert len(pod["initContainers"]) == 1


def test_manifest_bootstrap_files_are_all_supplied_by_submitter() -> None:
    bootstrap = yaml.safe_load(MANIFEST.read_text())["items"][0]["spec"]["template"][
        "spec"
    ]["containers"][0]["args"][0]
    submit = SUBMIT.read_text()
    literal_bootstrap_files = set(re.findall(r"/bootstrap/([A-Za-z0-9_.-]+)", bootstrap))
    supplied = set(re.findall(r"--from-file=([A-Za-z0-9_.-]+)=", submit))
    assert literal_bootstrap_files <= supplied
    assert {
        "generation2_v1.py",
        "generation2_v2.py",
        "generation2_v3.py",
        "runtime_v3.py",
        "v1-manifest.yaml",
        "v1-run.sh",
        "v1-submit.sh",
        "v2-manifest.yaml",
        "v2-run.sh",
        "v2-submit.sh",
        "v3-manifest.yaml",
        "v3-run.sh",
        "v3-submit.sh",
    } <= supplied


def test_runtime_uses_v3_release_claim_and_terminal_contracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _load(Q_SPEC)
    plan = contract.validate_spec(spec, ROOT)
    calls: list[str] = []
    monkeypatch.setenv("FLEET_API_KEY", "sealed-test-value")
    monkeypatch.setenv("JOB_UID", "11111111-1111-4111-8111-111111111111")
    monkeypatch.setenv("POD_UID", "22222222-2222-4222-8222-222222222222")
    monkeypatch.setattr(
        runtime.contract,
        "validate_release",
        lambda *args, **kwargs: calls.append("release"),
    )
    monkeypatch.setattr(runtime.hosted_runtime, "validate_live_route", lambda *a, **k: None)
    monkeypatch.setattr(runtime.hosted_runtime, "observe_live_route", lambda *a, **k: {})
    monkeypatch.setattr(
        runtime.legacy.endpoint_lease,
        "acquire_endpoint_lease",
        lambda **kwargs: nullcontext(),
    )
    monkeypatch.setattr(runtime.held_v1, "validate_preserved_claims", lambda *a: None)
    monkeypatch.setattr(runtime.hosted, "_validate_plan_identity_absence", lambda *a: None)
    monkeypatch.setattr(runtime.hosted, "_validate_inventory_for_task", lambda *a: [])
    monkeypatch.setattr(runtime.hosted, "_client", lambda key: nullcontext(object()))
    monkeypatch.setattr(
        runtime.self_hosted,
        "_request",
        lambda *a: {"team_name": "fleet", "team_id": self_hosted.FLEET_TEAM_ID},
    )
    claim = {"receipt_sha256": "sha256:" + "4" * 64}
    monkeypatch.setattr(
        runtime.contract,
        "claim_execution_generation",
        lambda *a, **k: calls.append("claim") or claim,
    )
    result = {"accepted": True, "quarantined": False}
    monkeypatch.setattr(runtime.legacy, "_run_cell", lambda *a: result)
    terminal = {
        "schema_version": contract.TERMINAL_SCHEMA,
        "generation2_spec_sha256": spec["generation2_spec_sha256"],
        "plan_sha256": plan["plan_sha256"],
    }
    monkeypatch.setattr(
        runtime.contract,
        "terminal_receipt",
        lambda *a, **k: calls.append("terminal") or terminal,
    )

    out = tmp_path / "output"
    returned = runtime.run(
        spec,
        {},
        _load(HELD),
        {},
        out,
        ROOT / "evals/fleet/fixed_proxy.py",
        ROOT,
        "1" * 40,
        authorized_at_utc="2026-09-05T05:31:00Z",
        authorization_statement="authorize exactly one paired canary release",
    )
    assert returned == terminal
    assert calls == ["release", "claim", "terminal"]
    assert json.loads((out / "PLAN.json").read_text()) == plan
    assert json.loads((out / "CANARY-TERMINAL.json").read_text()) == terminal


def test_wrappers_gate_before_paid_work_and_submit_is_held_create_once() -> None:
    run = RUN.read_text()
    assert run.index("validate-held") < run.index("validate-spec") < run.index(
        "docker info"
    ) < run.index("autocontinue_generation2_runtime_v3 run")
    submit = SUBMIT.read_text()
    assert submit.index("autocontinue_generation2_runtime_v3 preview") < submit.index(
        "create --dry-run=server"
    ) < submit.index("executable package is HELD")
    assert submit.count('kubectl -n "$NS" create -f "$MANIFEST"') == 1
    assert "kubectl delete" not in submit
    assert "kubectl apply" not in submit
