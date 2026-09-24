from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from cyber_post_train.jobs import digest
from training import miles96_phase2_conversion as conversion


def test_exact_fti_01027_conversion_request_is_one_node_zero_step() -> None:
    plan = conversion.compile_plan()
    request = conversion.job_request(plan)
    assert plan["fti_version"] == "0.10.27"
    assert plan["fti_source_commit"] == "d23116f018cb9213f0a3ee6c228d213abcd85d80"
    assert plan["optimizer_steps"] == 0
    assert plan["model"]["root"] == conversion.MODEL_ROOT
    assert len(plan["model"]["files"]) == 28
    assert digest(plan["model"]) == conversion.MODEL_BINDING_SHA256
    assert request["image"] == conversion.IMAGE
    assert request["workers"] == 1 and request["gpus_per_worker"] == 8
    assert request["priority_class"] == "c1"
    assert request["failureAlerts"] is False
    assert request["requeueIfPreempted"] is False
    assert request["secrets"] == []
    assert request["env"]["WANDB_MODE"] == "disabled"
    assert "chris-cpt-cleanup" not in json.dumps(request)


@pytest.mark.parametrize(
    "path,value",
    [
        (("optimizer_steps",), 1),
        (("execution", "image"), "wrong"),
        (("execution", "priority"), "c0"),
        (("model", "revision"), "wrong"),
        (("model", "files"), []),
    ],
)
def test_plan_drift_is_rejected(path: tuple[str, ...], value) -> None:
    plan = conversion.compile_plan()
    target = plan
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match="plan drift"):
        conversion.job_request(plan)


def test_runtime_binds_fti_source_and_native_converter(monkeypatch, tmp_path: Path) -> None:
    plan = conversion.compile_plan()
    run_fleet = tmp_path / "run_fleet.py"
    run_fleet.write_text("exact synthetic FTI source")
    monkeypatch.setattr(conversion, "_validate_plan", lambda value: None)
    monkeypatch.setattr(conversion, "FTI_RUN_FLEET_SHA256", conversion.base._hash(run_fleet))
    monkeypatch.setitem(sys.modules, "fti", SimpleNamespace(__version__=conversion.FTI_VERSION))
    monkeypatch.setitem(sys.modules, "fti.trainers", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "fti.trainers.miles",
        SimpleNamespace(run_fleet=SimpleNamespace(__file__=str(run_fleet))),
    )
    monkeypatch.setattr(conversion.base, "check_inputs", lambda value: None)
    monkeypatch.setattr(conversion.base, "native_arguments", lambda value: ["exact-native"])
    assert conversion.validate_runtime(plan) == ["exact-native"]
    monkeypatch.setitem(sys.modules, "fti", SimpleNamespace(__version__="wrong"))
    with pytest.raises(ValueError, match="source identity"):
        conversion.validate_runtime(plan)


def test_embedded_conversion_runtime_is_exact() -> None:
    request = conversion.job_request(conversion.compile_plan())
    assert "CYBER_RUNTIME_BUNDLE" in request["env"] or any(
        key.startswith("CYBER_RUNTIME_BUNDLE_") for key in request["env"]
    )
    assert Path(conversion.__file__).read_text()
    assert request["run_dir"] == conversion.OUTPUT_ROOT
