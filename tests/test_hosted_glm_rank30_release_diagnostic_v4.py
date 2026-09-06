import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from evals.fleet import hosted_glm_rank30_single_slot_release_diagnostic_package_v4 as package
from evals.fleet import hosted_glm_rank30_single_slot_release_diagnostic_v4 as diagnostic
from evals.fleet import self_hosted

ROOT = Path(__file__).resolve().parents[1]


def _materialize(tmp_path: Path) -> Path:
    rendered = package.render(ROOT)
    configmap = rendered["objects"]["items"][0]
    root = tmp_path / "materialized"
    fleet = root / "evals/fleet"
    configs = fleet / "configs"
    configs.mkdir(parents=True)
    (root / "evals/__init__.py").write_text("")
    (fleet / "__init__.py").write_text("")
    config_names = {
        "campaign.json", "selection.json", "glm-template.json", "qwen-template.json",
        "bulk-qwen-a.json", "bulk-qwen-b.json", "bulk-glm-a.json", "bulk-glm-b.json",
    }
    script = configmap["data"]["run.sh"]
    for source, target in __import__("re").findall(r"([\w.-]+):([\w.-]+)", script):
        if source not in configmap["data"]:
            continue
        destination = configs / target if source in config_names else fleet / target
        destination.write_text(configmap["data"][source])
    return root


@pytest.mark.parametrize("failed_index", range(len(diagnostic.PHASES)))
def test_materialized_package_call_through_emits_terminal_for_every_phase(
    tmp_path: Path, failed_index: int
) -> None:
    root = _materialize(tmp_path)
    output = tmp_path / f"out-{failed_index}" / "DIAGNOSTIC.json"
    code = f"""
from pathlib import Path
from evals.fleet import hosted_glm_rank30_single_slot_release_diagnostic_v4 as d
def phase(index):
    def run(_root, _state):
        if index == {failed_index}:
            raise ValueError('protected-message-must-not-appear')
    return run
phases=tuple((name, phase(index)) for index,(name,_function) in enumerate(d.PHASES))
raise SystemExit(d.run(Path({str(root)!r}), output_path=Path({str(output)!r}), phases=phases))
"""
    env = {
        **os.environ,
        "PYTHONPATH": str(root),
        "JOB_UID": "11111111-1111-4111-8111-111111111111",
        "POD_UID": "22222222-2222-4222-8222-222222222222",
    }
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, cwd=tmp_path,
        text=True, capture_output=True, check=False,
    )
    assert result.returncode == 1, result.stderr
    value = json.loads(output.read_text())
    assert value["failed_phase"] == diagnostic.PHASES[failed_index][0]
    assert value["error_type_class"] == "value-error"
    assert "protected-message-must-not-appear" not in output.read_text()
    assert value["receipt_sha256"] == self_hosted.digest_without(
        value, "receipt_sha256"
    )
    assert value["model_calls"] == value["task_calls"] == 0
    assert value["session_calls"] == value["verifier_calls"] == 0
    assert value["scoring_calls"] == value["api_mutation_calls"] == 0


def test_v4_package_is_fresh_immutable_held_and_has_no_scored_credentials() -> None:
    rendered = package.render(ROOT)
    assert rendered["launch_authorized"] is False
    assert rendered["scored_launch_authorized"] is False
    configmap, job = rendered["objects"]["items"]
    assert configmap["immutable"] is True
    assert configmap["metadata"]["name"] == diagnostic.CONFIGMAP_NAME
    assert job["metadata"]["name"] == diagnostic.JOB_NAME
    env_names = {
        row["name"]
        for row in job["spec"]["template"]["spec"]["containers"][0]["env"]
    }
    assert "FLEET_API_KEY" not in env_names
    assert "GLM_HOSTED_R30_SOURCE_SHA256" not in env_names
    assert "kubectl create" not in json.dumps(configmap["data"])
    assert "_task_sessions" not in configmap["data"]["diagnostic.py"]


def test_failed_v3_identity_is_exact_and_v4_receipt_is_sanitized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("JOB_UID", "11111111-1111-4111-8111-111111111111")
    monkeypatch.setenv("POD_UID", "22222222-2222-4222-8222-222222222222")
    phases = tuple((name, lambda _root, _state: None) for name, _fn in diagnostic.PHASES)
    output = tmp_path / "diagnostic" / "DIAGNOSTIC.json"
    assert diagnostic.run(ROOT, output_path=output, phases=phases) == 0
    value = json.loads(output.read_text())
    assert value["status"] == "PASSED_TO_SESSION_BOUNDARY"
    assert value["failed_release_job_uid"] == "9c3235a5-7919-4eae-a099-2886adad2fad"
    assert value["failed_release_pod_uid"] == "c1b44935-4eb6-40fc-88c0-d0ee7e4e4928"
    assert value["session_boundary_executed"] is False
    assert value["receipt_sha256"] == self_hosted.digest_without(
        value, "receipt_sha256"
    )


def test_v3_terminal_and_v4_held_evidence_are_self_digesting() -> None:
    evidence = ROOT / "docs/evidence/glm53-study"
    terminal = json.loads(
        (evidence / "2026-09-06-glm53-hosted-rank30-release-observer-v3-terminal.json").read_text()
    )
    held = json.loads(
        (evidence / "2026-09-06-glm53-hosted-rank30-release-diagnostic-v4-held.json").read_text()
    )
    assert terminal["receipt_sha256"] == self_hosted.digest_without(
        terminal, "receipt_sha256"
    )
    assert terminal["retry_same_identity"] is False
    assert terminal["observed_failure_stage"] == "pre-receipt-unclassified"
    assert held["receipt_sha256"] == self_hosted.digest_without(
        held, "receipt_sha256"
    )
    assert held["package_sha256"] == package.render(ROOT)["package_sha256"]
    assert held["launch_authorized"] is False
