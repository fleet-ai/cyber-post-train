import copy
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from evals.fleet import hosted_glm_rank30_single_slot_release_diagnostic_package_v6 as package
from evals.fleet import hosted_glm_rank30_single_slot_release_diagnostic_v5 as prior
from evals.fleet import hosted_glm_rank30_single_slot_release_diagnostic_v6 as diagnostic
from evals.fleet import hosted_glm_rank30_single_slot_v4 as successor
from evals.fleet import self_hosted

ROOT = Path(__file__).resolve().parents[1]


def _plan() -> dict:
    return successor.build_plan(successor.CONTROLLER, ROOT)


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


def test_v5_key_error_hash_is_exactly_reproduced() -> None:
    error = KeyError("attempts")
    body = {
        "phase": "09-release-projection",
        "type": type(error).__name__,
        "message": str(error),
    }
    observed = "sha256:" + hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert observed == "sha256:5a27ebe7ed0edfc2ef10cf4fa60f3a6b478740611c0e7e50ae0e3194f8b7504c"


def test_v6_replaces_only_phase09() -> None:
    assert diagnostic.PHASES[:8] == prior.PHASES[:8]
    assert [name for name, _fn in diagnostic.PHASES] == [
        name for name, _fn in prior.PHASES
    ]
    assert diagnostic.PHASES[8][1] is diagnostic._validate_projection  # noqa: SLF001


def test_v6_validates_exact_rank30_cells() -> None:
    state = {"plan": _plan()}
    diagnostic._validate_projection(ROOT, state)  # noqa: SLF001
    projection = successor.release_projection(state["plan"])
    assert projection["selection_rank"] == 30
    assert [row["attempt"] for row in projection["cells"]] == [1, 2, 3, 4]
    assert projection["cells"] == [
        {
            "attempt": row["attempt"],
            "cell_id": row["cell_id"],
            "execution_id": row["execution_id"],
            "run_id": row["run_id"],
        }
        for row in state["plan"]["attempts"]
    ]
    assert state["projected_attempts"] == [1, 2, 3, 4]


@pytest.mark.parametrize("mutation", ["missing", "extra", "reordered", "identity"])
def test_v6_rejects_release_projection_cell_drift(
    mutation: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()
    projection = successor.release_projection(plan)
    if mutation == "missing":
        projection["cells"].pop()
    elif mutation == "extra":
        projection["cells"].append(copy.deepcopy(projection["cells"][-1]))
    elif mutation == "reordered":
        projection["cells"].reverse()
    else:
        projection["cells"][0]["cell_id"] = "sha256:" + "0" * 64
    monkeypatch.setattr(successor, "release_projection", lambda _plan: projection)
    with pytest.raises(RuntimeError, match="projection cells drifted"):
        diagnostic._validate_projection(ROOT, {"plan": plan})  # noqa: SLF001


def test_v6_package_is_fresh_held_and_zero_mutation() -> None:
    rendered = package.render(ROOT)
    configmap, job = rendered["objects"]["items"]
    assert rendered["launch_authorized"] is False
    assert rendered["scored_launch_authorized"] is False
    assert rendered["model_calls_authorized"] is False
    assert rendered["task_session_verifier_calls_authorized"] is False
    assert rendered["prior_diagnostic_receipt_sha256"] == diagnostic.PRIOR_RECEIPT_SHA256
    assert configmap["immutable"] is True
    assert job["metadata"]["name"] == diagnostic.JOB_NAME
    assert configmap["metadata"]["name"] == diagnostic.CONFIGMAP_NAME
    assert len(job["metadata"]["name"]) <= 63
    assert len(configmap["metadata"]["name"]) <= 63
    env = {row["name"] for row in job["spec"]["template"]["spec"]["containers"][0]["env"]}
    assert "FLEET_API_KEY" not in env
    sources = "\n".join(configmap["data"].values())
    assert "_task_sessions" not in configmap["data"]["diagnostic.py"]
    assert "kubectl create" not in sources
    assert "submit" not in configmap["data"]["diagnostic.py"].lower()


def test_v6_materialized_package_reaches_score_free_terminal_boundary(
    tmp_path: Path,
) -> None:
    root = _materialize(tmp_path)
    output = tmp_path / "output/DIAGNOSTIC.json"
    code = f"""
from pathlib import Path
from evals.fleet import hosted_glm_rank30_single_slot_release_diagnostic_v6 as d
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
    assert value["model_calls"] == value["task_calls"] == 0
    assert value["session_calls"] == value["verifier_calls"] == 0
    assert value["scoring_calls"] == value["api_mutation_calls"] == 0


def test_v6_held_evidence_is_self_digesting_and_package_bound() -> None:
    evidence = ROOT / "docs/evidence/glm53-study"
    terminal = json.loads(
        (
            evidence
            / "2026-09-06-glm53-hosted-rank30-release-diagnostic-v5-terminal.json"
        ).read_text()
    )
    held = json.loads(
        (
            evidence
            / "2026-09-06-glm53-hosted-rank30-release-diagnostic-v6-held.json"
        ).read_text()
    )
    assert terminal["receipt_sha256"] == self_hosted.digest_without(
        terminal, "receipt_sha256"
    )
    assert terminal["error_sha256"] == (
        "sha256:5a27ebe7ed0edfc2ef10cf4fa60f3a6b478740611c0e7e50ae0e3194f8b7504c"
    )
    assert terminal["retry_same_identity"] is False
    assert held["receipt_sha256"] == self_hosted.digest_without(
        held, "receipt_sha256"
    )
    assert held["package_sha256"] == package.render(ROOT)["package_sha256"]
    assert held["prior_diagnostic_receipt_sha256"] == diagnostic.PRIOR_RECEIPT_SHA256
    assert held["launch_authorized"] is False
