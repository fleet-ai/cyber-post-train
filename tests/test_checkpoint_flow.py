"""No GPU, source checkpoint, or Fleet job is needed for bridge contract tests."""

import json
from pathlib import Path

import pytest

from training import checkpoint_flow as flow


def signed(value):
    return {**value, "receipt_sha256": flow._sha(flow._canonical(value))}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(signed(value), sort_keys=True) + "\n")


def test_step_flow_is_create_once_and_digest_bound(tmp_path, monkeypatch):
    root, prepared_dir = tmp_path / "run", tmp_path / "prepared"
    prepared_dir.mkdir()
    (prepared_dir / "PREPARED.json").write_text("{}\n")
    plan = {"run_name": "q38-corrected", "output_root": str(root),
            "recipe": {"max_steps": 32, "checkpoint_interval": 16}}
    request = {"name": plan["run_name"], "run_dir": str(root),
               "priority_class": "c1", "failureAlerts": False}
    prepared = {"plan_sha256": flow._sha(flow._canonical(plan)),
                "request_sha256": flow._sha(flow._canonical(request))}
    monkeypatch.setattr(flow, "_prepared", lambda _: (plan, request, prepared))

    def fake_historical(stage, value):
        output = Path(value["output"])
        if stage == "seal":
            write(output, {"schema": "cyber_skyrl_checkpoint_manifest_v1",
                           "source_plan_sha256": prepared["plan_sha256"],
                           "source_plan": plan, "optimizer_step": 16,
                           "checkpoint_path": str(root / "checkpoints/global_step_16"),
                           "gpu_reload_verified": False})
        elif stage == "export":
            seal = flow._receipt(Path(value["seal"]))
            output.mkdir(parents=True)
            (output / "model-00001.safetensors").write_bytes(b"x")
            write(output / "EXPORT.json", {
                "schema": "cyber_native_checkpoint_hf_export_v1",
                "source_checkpoint_receipt_sha256": seal["receipt_sha256"],
                "source_manifest_file_sha256": value["seal_sha256"],
                "source_plan_sha256": prepared["plan_sha256"],
                "optimizer_step": 16, "output_root": str(output), "dtype": "BF16",
                "optimizer_steps_executed": 0,
                "source_inventory_sizes_mtimes_unchanged": True,
                "all_output_tensors_reopened_equal": True,
                "files": {"model-00001.safetensors": {"bytes": 1,
                                                      "sha256": flow._sha(b"x")}}})
        else:
            exported = flow._receipt(Path(value["export"]))
            gpu = stage == "gpu"
            write(output, {"schema": "cyber_hf_export_check_v1", "status": "passed",
                           "export_sha256": value["export_sha256"],
                           "export_receipt_sha256": exported["receipt_sha256"],
                           "optimizer_steps_executed": 0, "gpus": int(gpu),
                           "gpu_reload_verified": gpu, "source_unchanged": True,
                           "serving_qualified": False, "finite_logits": gpu,
                           "generated_tokens": 2 if gpu else 0})

    monkeypatch.setattr(flow, "_historical", fake_historical)
    with pytest.raises(ValueError, match="step is not a planned checkpoint"):
        flow.run(prepared_dir, 15, "seal")
    with pytest.raises(ValueError, match="immutable receipt file missing"):
        flow.run(prepared_dir, 16, "export")
    for stage in flow.STAGES:
        flow.run(prepared_dir, 16, stage)
    proof = flow._receipt(flow._paths(plan, 16)["ready"])
    assert proof["checkpoint_sha256"] == flow._file_sha(flow._paths(plan, 16)["seal"])
    assert proof["export_sha256"] == flow._file_sha(flow._paths(plan, 16)["export"])
    assert proof["serving_qualified"] is False
    with pytest.raises(ValueError, match="create-once"):
        flow.run(prepared_dir, 16, "ready")


def test_tampered_cpu_gate_rejects_ready(tmp_path):
    path = tmp_path / "CPU_CHECK.json"
    write(path, {"schema": "cyber_hf_export_check_v1", "status": "passed",
                 "export_sha256": "1" * 64, "export_receipt_sha256": "2" * 64,
                 "optimizer_steps_executed": 0, "gpus": 1,
                 "gpu_reload_verified": True, "source_unchanged": True,
                 "serving_qualified": False})
    with pytest.raises(ValueError, match="CPU/GPU"):
        flow._check(path, {"receipt_sha256": "2" * 64}, "1" * 64, False)
    path.write_text(path.read_text().replace('"status": "passed"', '"status": "failed"'))
    with pytest.raises(ValueError, match="digest mismatch"):
        flow._receipt(path)


def test_stage_specs_require_c1_root_alert_opt_out(tmp_path, monkeypatch):
    plan = {"run_name": "q38-corrected", "output_root": "/mnt/sfs/jobs/q38-corrected",
            "recipe": {"max_steps": 32, "checkpoint_interval": 16},
            "execution": {"image": "example@sha256:" + "a" * 64},
            "runtime_sha256": flow.SOURCES["training/sft_runtime.py"]}
    request = {"name": plan["run_name"], "run_dir": plan["output_root"],
               "priority_class": "c1", "failureAlerts": False,
               "image": plan["execution"]["image"]}
    receipt = {"plan_sha256": flow._sha(flow._canonical(plan)),
               "request_sha256": flow._sha(flow._canonical(request))}
    for name, value in (("plan.json", plan), ("request.json", request),
                        ("PREPARED.json", receipt)):
        (tmp_path / name).write_text(json.dumps(value))
    monkeypatch.setattr(flow, "_prepared", lambda _: (plan, request, receipt))
    monkeypatch.setattr(flow, "_sources", lambda: {})
    cpu = flow.stage_spec(tmp_path, 16, "seal")["job"]
    assert cpu["metadata"]["annotations"]["fleet.ai/failure-alerts"] == "off"
    assert cpu["spec"]["suspend"] is True
    assert cpu["spec"]["template"]["spec"]["priorityClassName"] == "c1"
    assert "nvidia.com/gpu" not in str(cpu)
    assert flow.validate_stage_preview({"job": cpu}, cpu)["status"] == "previewed_not_created"
    with pytest.raises(ValueError, match="immutable receipt file missing"):
        flow.stage_spec(tmp_path, 16, "export")
    monkeypatch.setattr(flow, "_seal", lambda *_: ({"receipt_sha256": "0" * 64}, "1" * 64))
    monkeypatch.setattr(flow, "_export", lambda *_: ({"receipt_sha256": "2" * 64}, "3" * 64))
    gpu = flow.stage_spec(tmp_path, 16, "gpu")
    assert gpu["requires_rendered_root_annotation"] == {"fleet.ai/failure-alerts": "off"}
    assert gpu["request"]["gpus_per_worker"] == 1
    assert gpu["request"]["priority_class"] == "c1"
    assert gpu["request"]["failureAlerts"] is False


def test_all_historical_checkpoint_sources_match_pinned_bytes():
    assert set(flow._sources()) == set(flow.SOURCES)
