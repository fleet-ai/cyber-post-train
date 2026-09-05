from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from evals.fleet import autocontinue_generation13_package as package
from evals.fleet import autocontinue_generation13_simple_cell as runtime
from evals.fleet import self_hosted

ROOT = Path(__file__).resolve().parents[1]
SMOKE = yaml.safe_load(
    (ROOT / "evals/fleet/cluster/opencode-train-sweep-smokes-job-v2.yaml").read_text()
)


@pytest.mark.parametrize("model", tuple(runtime.EXPECTED))
def test_exact_spec_and_package(model: str) -> None:
    spec = package.derive_spec(ROOT, model)
    runtime.validate(spec, ROOT)
    built = package.render(ROOT, model)
    assert built["configmap_json_bytes"] < 900_000
    cm, job = built["objects"]["items"]
    assert cm["metadata"]["name"] == runtime.EXPECTED[model]["configmap_name"]
    assert job["metadata"]["name"] == runtime.EXPECTED[model]["job_name"]
    assert job["metadata"]["annotations"]["cyber-post-train.fleet.ai/preview-only"] == "false"
    assert job["metadata"]["annotations"]["cyber-post-train.fleet.ai/launch-authorized"] == "true"
    assert set(cm["data"]) == {
        "Dockerfile.opencode",
        "fixed_proxy.py",
        "self_hosted.py",
        "runner.py",
        "g13.py",
        "run.sh",
        "spec.json",
        "g12-tombstone.json",
    }
    mounted = json.loads(cm["data"]["spec.json"])
    assert mounted == spec
    assert spec["attempt_config"]["execution"]["required_task_tools"] == [
        "bash",
        "submit_report",
    ]
    assert spec["attempt_config"]["harness"]["context_window_size"] == 262144
    assert self_hosted.opencode_settings(spec["attempt_config"])["compaction"] == {
        "auto": True,
        "reserved": 20000,
    }


@pytest.mark.parametrize("model", tuple(runtime.EXPECTED))
def test_tombstone_is_self_digesting_and_pre_rollout(model: str) -> None:
    row = runtime.EXPECTED[model]
    value = runtime.load(ROOT / row["tombstone_path"])
    runtime.validate_tombstone(value, row)
    assert value["model_called"] is False
    assert value["task_instance_created"] is False
    assert value["scored_outcome_created"] is False
    assert value["pod"] == {"created": False}
    assert value["admission"] == {
        "priority_class": "fleet-train-high",
        "requested_preemption_policy": "Never",
        "reason": "priority-class-preemption-policy-conflict",
    }
    assert value["kueue"]["quota_released"] is True
    assert value["fleet"]["matching_sessions"] == 0
    assert value["fleet"]["matching_verifier_records"] == 0
    assert value["sfs"] == {
        "claim_absent": True,
        "output_root_absent": True,
        "stage_root_absent": True,
    }


@pytest.mark.parametrize("model", tuple(runtime.EXPECTED))
def test_g13_uses_exact_accepted_v2_bootstrap(model: str) -> None:
    job = yaml.safe_load((ROOT / package.MANIFESTS[model]).read_text())
    smoke_pod = SMOKE["spec"]["template"]["spec"]
    pod = job["spec"]["template"]["spec"]
    smoke_dind = next(row for row in smoke_pod["initContainers"] if row["name"] == "dind")
    dind = next(row for row in pod["initContainers"] if row["name"] == "dind")
    for field in (
        "image",
        "restartPolicy",
        "args",
        "readinessProbe",
        "securityContext",
        "resources",
        "volumeMounts",
    ):
        assert dind[field] == smoke_dind[field]
    assert [row["name"] for row in pod["initContainers"]] == ["dind"]
    evaluator = pod["containers"][0]
    smoke_evaluator = smoke_pod["containers"][0]
    assert evaluator["image"] == smoke_evaluator["image"]
    assert evaluator["resources"] == smoke_evaluator["resources"]
    assert (
        "apt-get install --yes --no-install-recommends docker.io=20.10.24+dfsg1-1+deb12u1+b6"
        in evaluator["args"][0]
    )
    assert pod["preemptionPolicy"] == "Never"
    assert pod["priorityClassName"] == "fleet-serve-low"


def test_model_treatments_differ_only_as_expected() -> None:
    q = package.derive_spec(ROOT, "qwen3.8-27b")
    g = package.derive_spec(ROOT, "glm-5.3")
    for spec in (q, g):
        attempt = spec["attempt_config"]
        assert attempt["harness"]["name"] == "opencode"
        assert attempt["harness"]["version"] == "1.18.27"
        assert attempt["harness"]["max_model_requests"] == 600
        assert attempt["harness"]["timeout_seconds"] == 28800
        assert attempt["execution"]["pass_k"] == 1
        assert attempt["execution"]["planned_full_pass_k"] == 4


def test_stage_receipt_is_sanitized_and_create_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = package.derive_spec(ROOT, "qwen3.8-27b")
    spec = copy.deepcopy(spec)
    spec["stage_root"] = str(tmp_path / "stages")
    monkeypatch.setenv("JOB_UID", "11111111-1111-4111-8111-111111111111")
    monkeypatch.setenv("POD_UID", "22222222-2222-4222-8222-222222222222")
    receipt = runtime.mark_stage(spec, "01-before-static-validation")
    assert receipt["receipt_sha256"] == runtime.digest(receipt)
    assert receipt["credentials_included"] is False
    assert receipt["prompts_traces_flags_or_scores_included"] is False
    with pytest.raises(FileExistsError):
        runtime.mark_stage(spec, "01-before-static-validation")


def test_stage_sequence_present_in_runner() -> None:
    script = (ROOT / package.RUN_PATH).read_text()
    positions = [script.index(stage) for stage in runtime.STAGES[:6]]
    assert positions == sorted(positions)
    for stage in runtime.STAGES[6:15]:
        assert stage in Path(ROOT / package.RUNTIME_PATH).read_text()


def test_spec_mutation_rejected() -> None:
    spec = package.derive_spec(ROOT, "qwen3.8-27b")
    spec["execution_id"] = "sha256:" + "0" * 64
    spec["spec_sha256"] = runtime.digest(spec, "spec_sha256")
    with pytest.raises(ValueError, match="treatment drifted"):
        runtime.validate(spec, ROOT)


@pytest.mark.parametrize("model", tuple(runtime.EXPECTED))
def test_rendered_runtime_imports_in_isolation(model: str, tmp_path: Path) -> None:
    cm = package.render(ROOT, model)["objects"]["items"][0]
    fleet = tmp_path / "evals/fleet"
    fleet.mkdir(parents=True)
    (tmp_path / "evals/__init__.py").touch()
    (fleet / "__init__.py").touch()
    (fleet / "self_hosted.py").write_text(cm["data"]["self_hosted.py"])
    (fleet / "opencode_train_sweep_runner.py").write_text(cm["data"]["runner.py"])
    (fleet / "autocontinue_generation13_simple_cell.py").write_text(cm["data"]["g13.py"])
    result = subprocess.run(
        [sys.executable, "-c", "import evals.fleet.autocontinue_generation13_simple_cell"],
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
