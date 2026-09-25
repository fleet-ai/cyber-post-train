"""One stage per tick; remote I/O is injected and uncertain creates never replay."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from training import checkpoint_flow as flow

def _once(path: Path, value: dict) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or json.loads(path.read_text()) != value:
            raise ValueError("checkpoint claim or intent changed; reconcile")
        return
    with os.fdopen(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), "wb") as stream:
        stream.write(flow._canonical(value) + b"\n")
        stream.flush(); os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

def tick(directory: Path, run_uid: str, *, live_get, lease, duplicate_get,
         capacity_get, preview_get, submit) -> dict:
    """Submit at most one stage. `submit` must create/unsuspend and return name/id."""
    plan, request, prepared = flow._prepared(directory)
    live = live_get(request["name"], run_uid)
    if (not isinstance(live, dict) or not re.fullmatch(r"[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}", run_uid)
        or not re.fullmatch(re.escape(request["name"]) + r"-[a-f0-9]{8}", live.get("name", "")) or live.get("uid") != run_uid
        or live.get("namespace") != "fleet-train-jobs"
        or live.get("run_dir") != request["run_dir"] or live.get("image") != request["image"]):
        raise ValueError("live trainer identity differs from exact prepared run")
    if live.get("status") not in {"RUNNING", "SUCCEEDED"}:
        return {"status": "trainer_not_processable", "run_uid": run_uid}
    root = Path(plan["output_root"])
    source_dir = root / "checkpoint_receipts"
    if source_dir.is_symlink():
        raise ValueError("checkpoint receipt directory is symlinked")
    claims = []
    for path in sorted(source_dir.glob("step-*.json")):
        match = re.fullmatch(r"step-(\d{6}).json", path.name)
        if not match:
            raise ValueError("unexpected checkpoint receipt name")
        step = int(match[1])
        paths = flow._paths(plan, step)
        source = flow._receipt(path)
        expected = {"optimizer_step": step, "plan_sha256": prepared["plan_sha256"],
                    "checkpoint_path": str(root / "checkpoints" / f"global_step_{step}")}
        if any(source.get(k) != v for k, v in expected.items()):
            raise ValueError("saved checkpoint receipt differs from prepared run")
        if paths["slot"].is_symlink():
            raise ValueError("checkpoint stage directory is symlinked")
        paths["slot"].mkdir(parents=True, exist_ok=True)
        _once(paths["slot"] / "CLAIM.json", {
            "schema": "q38_checkpoint_claim_v1", "run_uid": run_uid,
            "source_run": request["name"], "optimizer_step": step,
            "plan_sha256": prepared["plan_sha256"], "request_sha256": prepared["request_sha256"],
            "source_receipt_file_sha256": flow._file_sha(path)})
        claims.append((step, paths))
    pending = {"status": "waiting_for_checkpoint", "run_uid": run_uid}
    for step, paths in claims:
        for stage in flow.STAGES:
            artifact = paths[stage]
            if artifact.exists() or artifact.is_symlink():
                receipt = flow._receipt(artifact)
                if stage == "ready":
                    expected = {"status": "ready_for_route_parity", "source_run": request["name"],
                                "optimizer_step": step, "plan_sha256": prepared["plan_sha256"],
                                "request_sha256": prepared["request_sha256"],
                                "checkpoint_sha256": flow._file_sha(paths["seal"]),
                                "export_sha256": flow._file_sha(paths["export"]),
                                "serving_qualified": False, "task_evaluated": False}
                    if any(receipt.get(k) != v for k, v in expected.items()):
                        raise ValueError("checkpoint ready receipt differs")
                    if plan.get("validation_mode") == "teacher_cross_entropy":
                        if any(receipt.get(k) != v for k, v in flow._teacher_ce(plan, prepared, step).items()):
                            raise ValueError("checkpoint ready teacher CE binding differs")
                    pending = {"status": "pending_served_route_parity_and_fleet_pass4",
                               "run_uid": run_uid, "step": step,
                               "checkpoint_sha256": receipt["checkpoint_sha256"]}
                continue
            intent = paths["slot"] / f"{stage.upper()}-INTENT.json"
            if intent.exists() or intent.is_symlink():
                prior = flow._receipt(intent)
                if any(prior.get(k) != v for k, v in
                       {"stage": stage, "run_uid": run_uid, "optimizer_step": step}.items()):
                    raise ValueError("checkpoint stage intent identity differs")
                pending = {"status": "stage_pending_reconcile", "step": step, "stage": stage}
                break
            if stage == "ready" and plan.get("validation_mode") == "teacher_cross_entropy":
                dev = root / "validation" / f"step-{step:06d}.json"
                if not dev.exists() and not dev.is_symlink():
                    pending = {"status": "waiting_teacher_ce", "step": step}
                    break
            spec = flow.stage_spec(directory, step, stage)
            kind = "Job" if "job" in spec else "RayJob"
            name = spec["job"]["metadata"]["name"] if kind == "Job" else spec["request"]["name"]
            with lease():
                if duplicate_get(kind, name) is not None:
                    raise ValueError("exact checkpoint stage already exists remotely")
                preview = preview_get(spec)
                proof = flow.validate_stage_preview(spec, preview)
                counts = capacity_get()
                if (any(type(counts.get(k)) is not int or counts[k] < 0
                        for k in ("active_nodes", "active_gpus", "queued_jobs"))
                    or counts["queued_jobs"] >= 10
                    or (kind == "RayJob" and (counts["active_nodes"] >= 10 or counts["active_gpus"] >= 80))):
                    raise ValueError("project-owned checkpoint capacity exhausted")
                if duplicate_get(kind, name) is not None:
                    raise ValueError("checkpoint stage appeared during preview")
                claim = {"schema": "q38_checkpoint_stage_intent_v1", "stage": stage,
                         "run_uid": run_uid, "optimizer_step": step, "kind": kind, "name": name,
                         "spec_sha256": flow._sha(flow._canonical(spec)),
                         "preview_sha256": flow._sha(flow._canonical(preview)),
                         "proof": proof}
                claim["receipt_sha256"] = flow._sha(flow._canonical(claim))
                _once(intent, claim)
                created = submit(spec, preview)
                actual = created.get("name", "") if isinstance(created, dict) else ""
                if not re.fullmatch(re.escape(name) + (r"-[a-f0-9]{8}" if kind == "RayJob" else ""), actual) or not created.get("id"):
                    raise ValueError("stage create outcome uncertain; reconcile exact intent")
                _once(paths["slot"] / f"{stage.upper()}-CREATED.json",
                      {"name": actual, "id": created["id"], "intent_sha256": claim["receipt_sha256"]})
            return {"status": "stage_submitted_once", "step": step, "stage": stage,
                    "name": actual, "id": created["id"]}
    return pending
