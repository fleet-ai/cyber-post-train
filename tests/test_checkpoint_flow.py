"""No GPU, source checkpoint, or Fleet job is needed for bridge contract tests."""

import json
from contextlib import nullcontext
from pathlib import Path

import pytest

from training import checkpoint_flow as flow, checkpoint_tick as dispatch


def signed(value):
    return {**value, "receipt_sha256": flow._sha(flow._canonical(value))}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(signed(value), sort_keys=True) + "\n")


@pytest.mark.parametrize("step,max_steps", [(16, 32), (17, 17)])
def test_step_flow_is_create_once_and_digest_bound(tmp_path, monkeypatch, step, max_steps):
    root, prepared_dir = tmp_path / "run", tmp_path / "prepared"
    prepared_dir.mkdir()
    (prepared_dir / "PREPARED.json").write_text("{}\n")
    image = "example@sha256:" + "a" * 64
    plan = {"run_name": "q38-corrected", "output_root": str(root),
            "execution": {"image": image},
            "datasets": {"dev": {"task_keys": ["task-a"]}},
            "validation_mode": "teacher_cross_entropy",
            "recipe": {"max_steps": max_steps, "checkpoint_interval": 16, "eval_interval": 16}}
    request = {"name": plan["run_name"], "run_dir": str(root),
               "priority_class": "c1", "failureAlerts": False, "image": image}
    prepared = {"plan_sha256": flow._sha(flow._canonical(plan)),
                "request_sha256": flow._sha(flow._canonical(request))}
    monkeypatch.setattr(flow, "_prepared", lambda _: (plan, request, prepared))

    def fake_historical(stage, value):
        output = Path(value["output"])
        if stage == "seal":
            write(output, {"schema": "cyber_skyrl_checkpoint_manifest_v1",
                           "source_plan_sha256": prepared["plan_sha256"],
                           "source_plan": plan, "optimizer_step": step,
                           "checkpoint_path": str(root / "checkpoints" / f"global_step_{step}"),
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
                "optimizer_step": step, "output_root": str(output), "dtype": "BF16",
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
        flow.run(prepared_dir, step, "export")
    for stage in flow.STAGES[:-1]:
        flow.run(prepared_dir, step, stage)
    with pytest.raises(ValueError, match="immutable receipt file missing"):
        flow.run(prepared_dir, step, "ready")
    with pytest.raises(ValueError, match="immutable receipt file missing"):
        flow.stage_spec(prepared_dir, step, "ready")
    dev = root / "validation" / f"step-{step:06d}.json"
    write(dev, {"optimizer_step": step - 1, "plan_sha256": prepared["plan_sha256"]})
    with pytest.raises(ValueError, match="development receipt differs"):
        flow.run(prepared_dir, step, "ready")
    with pytest.raises(ValueError, match="development receipt differs"):
        flow.stage_spec(prepared_dir, step, "ready")
    write(dev, {"optimizer_step": step, "plan_sha256": prepared["plan_sha256"]})
    with pytest.raises(ValueError, match="metrics are incomplete"):
        flow.run(prepared_dir, step, "ready")
    write(dev, {"optimizer_step": step, "plan_sha256": prepared["plan_sha256"],
                "eval_loss": 1.2, "task_macro_loss": 1.3,
                "supervised_tokens": 10, "windows": 1, "tasks": 1})
    flow.run(prepared_dir, step, "ready")
    proof = flow._receipt(flow._paths(plan, step)["ready"])
    assert proof["checkpoint_sha256"] == flow._file_sha(flow._paths(plan, step)["seal"])
    assert proof["export_sha256"] == flow._file_sha(flow._paths(plan, step)["export"])
    assert proof["teacher_loss_file_sha256"] == flow._file_sha(dev)
    assert proof["serving_qualified"] is False
    with pytest.raises(ValueError, match="create-once"):
        flow.run(prepared_dir, step, "ready")
    write(root / "checkpoint_receipts" / f"step-{step:06d}.json",
          {"optimizer_step": step, "plan_sha256": prepared["plan_sha256"],
           "checkpoint_path": str(root / "checkpoints" / f"global_step_{step}")})
    uid = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    live = {"name": request["name"] + "-deadbeef", "uid": uid, "namespace": "fleet-train-jobs",
            "status": "SUCCEEDED", "run_dir": str(root), "image": image}
    def no_stage(*_):
        pytest.fail("ready checkpoint must await route parity, not dispatch")
    state = dispatch.tick(prepared_dir, uid, live_get=lambda *_: live, lease=no_stage,
                          duplicate_get=no_stage, capacity_get=no_stage,
                          preview_get=no_stage, submit=no_stage)
    assert state["status"] == "pending_served_route_parity_and_fleet_pass4"


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


@pytest.mark.parametrize("lazy", [False, True])
def test_mechanics_checkpoint_profile_is_accepted_without_teacher_ce(tmp_path, lazy):
    plan = {"runtime_sha256": flow.SOURCES["training/sft_runtime.py"],
            "execution": {"image": "image@sha256:" + "a" * 64}}
    if lazy:
        plan.update({"runtime_sha256": flow.lazy_overlay.PATCHED["training/sft_runtime.py"],
                     "lazy_overlay_sha256": flow._sha(Path(flow.lazy_overlay.__file__).read_bytes()),
                     "datasets": {"train": {"storage_layout": flow.lazy_overlay.LAYOUT}}})
    request = {"image": plan["execution"]["image"]}
    prepared = {"schema": "qwen38_96k_fast_diagnostic_prepared_v1" if lazy else "qwen38_96k_mechanics_prepared_v1",
                "historical_commit": flow.COMMIT,
                "plan_sha256": flow._sha(flow._canonical(plan)),
                "request_sha256": flow._sha(flow._canonical(request))}
    for name, value in (("plan.json", plan), ("request.json", request),
                        ("PREPARED.json", prepared)):
        (tmp_path / name).write_text(json.dumps(value))
    assert flow._prepared(tmp_path) == (plan, request, prepared)
    if lazy:
        plan["lazy_overlay_sha256"] = "0" * 64
        (tmp_path / "plan.json").write_text(json.dumps(plan))
        with pytest.raises(ValueError, match="lazy runtime"):
            flow._prepared(tmp_path)


def test_stage_specs_require_c1_root_alert_opt_out(tmp_path, monkeypatch):
    plan = {"run_name": "q38-corrected", "output_root": str(tmp_path / "run"),
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
    write(Path(plan["output_root"]) / "checkpoint_receipts/step-000016.json",
          {"plan_sha256": receipt["plan_sha256"], "optimizer_step": 16,
           "checkpoint_path": str(Path(plan["output_root"]) / "checkpoints/global_step_16")})
    cpu = flow.stage_spec(tmp_path, 16, "seal")["job"]
    assert cpu["metadata"]["annotations"]["fleet.ai/failure-alerts"] == "off"
    assert cpu["spec"]["suspend"] is True
    assert cpu["spec"]["template"]["spec"]["priorityClassName"] == "c1"
    assert "nvidia.com/gpu" not in str(cpu)
    assert flow.validate_stage_preview({"job": cpu}, cpu)["status"] == "previewed_not_created"
    normalized = json.loads(json.dumps(cpu))
    for item in normalized["spec"]["template"]["spec"]["containers"][0]["env"]:
        if item.get("value") == "":
            item.pop("value")
    assert flow.validate_stage_preview({"job": cpu}, normalized)["status"] == "previewed_not_created"
    with pytest.raises(ValueError, match="immutable receipt file missing"):
        flow.stage_spec(tmp_path, 16, "export")
    monkeypatch.setattr(flow, "_seal", lambda *_: ({"receipt_sha256": "0" * 64}, "1" * 64))
    monkeypatch.setattr(flow, "_export", lambda *_: ({"receipt_sha256": "2" * 64}, "3" * 64))
    gpu = flow.stage_spec(tmp_path, 16, "gpu")
    assert gpu["requires_rendered_root_annotation"] == {"fleet.ai/failure-alerts": "off"}
    assert gpu["request"]["gpus_per_worker"] == 1
    assert gpu["request"]["priority_class"] == "c1"
    assert gpu["request"]["failureAlerts"] is False
    manifest = {"kind": "RayJob", "metadata": {
        "name": gpu["request"]["name"],
        "namespace": "fleet-train-jobs",
        "annotations": {"fleet.ai/failure-alerts": "off",
                        "fleet.ai/run-dir": gpu["request"]["run_dir"]},
        "labels": {"kueue.x-k8s.io/queue-name": "training-lq",
                   "kueue.x-k8s.io/priority-class": "q1"}},
        "spec": {"suspend": True, "shutdownAfterJobFinishes": True,
                 "entrypoint": gpu["request"]["command"],
                 "rayClusterSpec": {"headGroupSpec": {"template": {"spec": {
                     "priorityClassName": "c1", "containers": [{
                         "image": gpu["request"]["image"],
                         "resources": {"limits": {"nvidia.com/gpu": 1}}}]}}}}}}
    preview = {"manifest_yaml": json.dumps(manifest), "errors": [], "warnings": []}
    assert flow.validate_stage_preview(gpu, preview)["kind"] == "RayJob"
    manifest["metadata"]["annotations"]["fleet.ai/failure-alerts"] = "on"
    with pytest.raises(ValueError, match="drifted"):
        flow.validate_stage_preview(gpu, {**preview, "manifest_yaml": json.dumps(manifest)})
    uid = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    live = {"name": request["name"] + "-deadbeef", "uid": uid, "namespace": "fleet-train-jobs",
            "status": "RUNNING", "run_dir": request["run_dir"], "image": request["image"]}
    def uncertain(*_):
        raise TimeoutError("synthetic uncertain create")
    args = {"live_get": lambda *_: live, "lease": lambda: nullcontext(),
            "duplicate_get": lambda *_: None,
            "capacity_get": lambda: {"active_nodes": 1, "active_gpus": 8, "queued_jobs": 0},
            "preview_get": lambda spec: spec["job"], "submit": uncertain}
    with pytest.raises(ValueError, match="capacity exhausted"):
        dispatch.tick(tmp_path, uid, **{**args, "capacity_get": lambda: {
            "active_nodes": 1, "active_gpus": 8, "queued_jobs": 10}})
    with pytest.raises(ValueError, match="already exists remotely"):
        dispatch.tick(tmp_path, uid, **{**args, "duplicate_get": lambda *_: {"uid": "peer"}})
    with pytest.raises(TimeoutError, match="uncertain create"):
        dispatch.tick(tmp_path, uid, **args)
    assert dispatch.tick(tmp_path, uid, **args)["status"] == "stage_pending_reconcile"


def test_all_historical_checkpoint_sources_match_pinned_bytes():
    assert set(flow._sources()) == set(flow.SOURCES)
