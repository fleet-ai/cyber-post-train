import copy
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from evals.fleet import hosted_glm_rank30_single_slot_release_diagnostic_package_v5 as package
from evals.fleet import hosted_glm_rank30_single_slot_release_diagnostic_v4 as diagnostic_v4
from evals.fleet import hosted_glm_rank30_single_slot_release_diagnostic_v5 as diagnostic
from evals.fleet import hosted_glm_rank30_single_slot_v3 as broken
from evals.fleet import hosted_glm_rank30_single_slot_v4 as fixed
from evals.fleet import self_hosted

ROOT = Path(__file__).resolve().parents[1]


def _materialize(tmp_path: Path) -> Path:
    configmap = package.render(ROOT)["objects"]["items"][0]
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
    for source, target in re.findall(
        r"([\w.-]+):([\w.-]+)", configmap["data"]["run.sh"]
    ):
        if source not in configmap["data"]:
            continue
        destination = configs / target if source in config_names else fleet / target
        destination.write_text(configmap["data"][source])
    return root


def _runtime_source_plan() -> dict:
    value = copy.deepcopy(fixed.prior.prior.whole.source.validate_all(ROOT)["glm-hosted-s2"])
    value["schema_version"] = "fleet-exact-pass4-bulk-executable-plan-v3"
    value["tasks"] = [{"rank": 30}, {"rank": 31}]
    value["plan_sha256"] = self_hosted.digest_without(value, "plan_sha256")
    assert value["plan_sha256"] != fixed.prior.prior.whole.SOURCE_PLAN_SHA256
    return value


def test_v4b_failure_hash_reproduces_static_runtime_digest_confusion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime_source_plan()
    monkeypatch.setattr(
        fixed.prior.prior.whole.source, "build_runtime_plan",
        lambda *_args: runtime,
    )
    with pytest.raises(ValueError, match="source plan drifted") as caught:
        broken.build_runtime_plan(broken.CONTROLLER, {}, ROOT)
    body = {
        "phase": "05-runtime-plan",
        "type": type(caught.value).__name__,
        "message": str(caught.value),
    }
    observed = "sha256:" + hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert observed == "sha256:a4e74e385c47c00772057df3606dfa3e74307893bf32edea05563ef37118b09d"


def test_specialized_builder_preserves_static_and_runtime_authorities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime_source_plan()
    monkeypatch.setattr(
        fixed.prior.prior.whole.source, "build_runtime_plan",
        lambda *_args: runtime,
    )
    plan = fixed.build_runtime_plan(fixed.CONTROLLER, {}, ROOT)
    assert plan["static_predecessor_plan_sha256"] == fixed.prior.prior.whole.SOURCE_PLAN_SHA256
    assert plan["runtime_predecessor_plan_sha256"] == runtime["plan_sha256"]
    assert plan["plan_sha256"] == self_hosted.digest_without(plan, "plan_sha256")
    assert plan["launch_authorized"] is True
    assert [row["selection_rank"] for row in plan["attempts"]] == [30] * 4


def test_specialized_builder_rejects_runtime_source_digest_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime_source_plan()
    runtime["plan_sha256"] = "sha256:" + "0" * 64
    monkeypatch.setattr(
        fixed.prior.prior.whole.source, "build_runtime_plan",
        lambda *_args: runtime,
    )
    with pytest.raises(ValueError, match="runtime source plan digest drifted"):
        fixed.build_runtime_plan(fixed.CONTROLLER, {}, ROOT)


def test_v5_diagnostic_is_fresh_held_and_zero_mutation() -> None:
    rendered = package.render(ROOT)
    assert rendered["launch_authorized"] is False
    assert rendered["scored_launch_authorized"] is False
    assert rendered["prior_diagnostic_receipt_sha256"] == diagnostic.PRIOR_RECEIPT_SHA256
    configmap, job = rendered["objects"]["items"]
    assert configmap["immutable"] is True
    assert job["metadata"]["name"] == diagnostic.JOB_NAME
    assert len(job["metadata"]["name"]) <= 63
    env = {row["name"] for row in job["spec"]["template"]["spec"]["containers"][0]["env"]}
    assert "FLEET_API_KEY" not in env
    source = configmap["data"]["diagnostic.py"] + configmap["data"]["successor_v4.py"]
    assert "_task_sessions" not in source
    assert "_write_once" not in configmap["data"]["successor_v4.py"]
    assert "kubectl create" not in json.dumps(configmap["data"])


def test_v5_replaces_only_phase05_builder() -> None:
    assert [name for name, _fn in diagnostic.PHASES] == [
        name for name, _fn in diagnostic_v4.PHASES
    ]
    assert diagnostic.PHASES[4][1] is diagnostic._build_plan  # noqa: SLF001
    assert diagnostic.PHASES[:4] == diagnostic_v4.PHASES[:4]
    assert diagnostic.PHASES[5:] == diagnostic_v4.PHASES[5:]


def test_v5_materialized_package_calls_through_to_terminal_receipt(
    tmp_path: Path,
) -> None:
    root = _materialize(tmp_path)
    output = tmp_path / "output/DIAGNOSTIC.json"
    code = f"""
from pathlib import Path
from evals.fleet import hosted_glm_rank30_single_slot_release_diagnostic_v5 as d
phases=tuple((name, lambda _root,_state: None) for name,_function in d.PHASES)
raise SystemExit(d.run(Path({str(root)!r}), output_path=Path({str(output)!r}), phases=phases))
"""
    env = {
        **os.environ,
        "PYTHONPATH": str(root),
        "JOB_UID": "11111111-1111-4111-8111-111111111111",
        "POD_UID": "22222222-2222-4222-8222-222222222222",
    }
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=env,
        text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    value = json.loads(output.read_text())
    assert value["status"] == "PASSED_TO_SESSION_BOUNDARY"
    assert value["receipt_sha256"] == self_hosted.digest_without(
        value, "receipt_sha256"
    )
    assert value["model_calls"] == value["session_calls"] == 0
    assert value["scoring_calls"] == value["api_mutation_calls"] == 0


def test_v4b_terminal_and_v5_held_receipts_are_self_digesting() -> None:
    evidence = ROOT / "docs/evidence/glm53-study"
    terminal = json.loads(
        (
            evidence
            / "2026-09-06-glm53-hosted-rank30-release-diagnostic-v4b-terminal.json"
        ).read_text()
    )
    held = json.loads(
        (evidence / "2026-09-06-glm53-hosted-rank30-release-diagnostic-v5-held.json").read_text()
    )
    assert terminal["receipt_sha256"] == self_hosted.digest_without(
        terminal, "receipt_sha256"
    )
    assert terminal["retry_same_identity"] is False
    assert terminal["error_sha256"] == (
        "sha256:a4e74e385c47c00772057df3606dfa3e74307893bf32edea05563ef37118b09d"
    )
    assert held["receipt_sha256"] == self_hosted.digest_without(
        held, "receipt_sha256"
    )
    assert held["package_sha256"] == package.render(ROOT)["package_sha256"]
    assert held["launch_authorized"] is False
