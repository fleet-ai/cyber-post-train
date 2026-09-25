"""Review-only builder and separate zero-step restore for the 4×8 262K canary.

No GPU submission is exposed. A CPU/SFS seal and independent review are needed
after the source reaches step 1; this is capacity proof, not scientific SFT.
"""

import argparse
import copy
import hashlib
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path

NAME = "chris-q38-t3k262-4n-reload-v2"
MANIFEST = Path("/mnt/sfs/jobs/chris-q38-t3k262-4n-can-v3/checkpoint-manifest-step1.json")
CHECKPOINT = "/mnt/sfs/jobs/chris-q38-t3k262-4n-can-v3/checkpoints/global_step_1"
SOURCE_SHA = "85cbab43a21e195e231176b3e6246dc017204c9955a27e018c8ff0a3a86977f4"
PINNED = {
    "training/checkpoints.py": "b2bfa604a45a7ea1ed1b195401ce3c489f6460b36d2cd39d68a43033d23873b7",
    "training/recovery.py": "0765eb0f09378566c68e47861f6cc3d04c245aa438353befdb1c176d8726e84a",
}
GATE = {"submission_authorized": False, "accepted": False,
        "remaining": "source step-1 seal, exact-image CPU preflight, separate root GPU review"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def build_plan(manifest):
    from training.long_context_launch import historical_request

    source, _ = historical_request(successor=True)
    unsigned = {k: v for k, v in manifest.items() if k != "receipt_sha256"}
    files = manifest.get("files", {})
    required = {"data.pt", "trainer_state.pt", "policy/fsdp_config.json",
                "policy/huggingface/config.json"} | {
                    f"policy/{kind}_world_size_32_rank_{rank}.pt"
                    for kind in ("model", "optim", "extra_state") for rank in range(32)}
    if (sha(canonical(source)) != SOURCE_SHA or
        manifest.get("schema") != "cyber_skyrl_checkpoint_manifest_v1" or
        manifest.get("receipt_sha256") != sha(canonical(unsigned)) or
        manifest.get("source_plan") != source or manifest.get("source_plan_sha256") != SOURCE_SHA or
        manifest.get("checkpoint_path") != CHECKPOINT or manifest.get("optimizer_step") != 1 or
        manifest.get("world_size") != 32 or manifest.get("sampler_batches_in_epoch") != 1 or
        manifest.get("gpu_reload_verified") is not False or not isinstance(files, dict) or
        not required <= files.keys() or any(not isinstance(v, dict) or
        type(v.get("bytes")) is not int or v["bytes"] <= 0 or
        not re.fullmatch(r"[a-f0-9]{64}", str(v.get("sha256", ""))) for v in files.values()) or
        manifest.get("total_bytes") != sum(v["bytes"] for v in files.values())):
        raise ValueError("sealed step-1 manifest is not bound to the exact 4×8 canary")
    plan = copy.deepcopy(source)
    plan.update(run_name=NAME, output_root=f"/mnt/sfs/jobs/{NAME}")
    plan["wandb"].update(run_id=NAME, name=NAME)
    plan["recovery"] = {"mode": "validate", "checkpoint": manifest,
                        "manifest_path": str(MANIFEST), "manifest_file_sha256": sha(MANIFEST.read_bytes())}
    plan["recovery_runtime_sha256"] = PINNED["training/recovery.py"]
    plan["reload_runtime_sha256"] = sha(Path(__file__).read_bytes())
    plan["reload_gate"] = GATE
    return plan


def _stage(root):
    from training.long_context_launch import REVISION, ROOT, stage_old_code

    stage_old_code(root, successor=True)
    for name, expected in PINNED.items():
        blob = subprocess.run(["git", "show", f"{REVISION}:{name}"], cwd=ROOT,
                              check=True, capture_output=True).stdout
        if sha(blob) != expected:
            raise ValueError("pinned native recovery source changed")
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)


def prepare(path=MANIFEST):
    """Build a request for review only; source files must already be CPU-sealed."""
    from training.long_context_launch import _old_python, historical_request

    if path != MANIFEST or path.is_symlink():
        raise ValueError("manifest must be the exact immutable source output")
    plan = build_plan(json.loads(path.read_text()))
    with tempfile.TemporaryDirectory(prefix="q38-262k-reload-") as tmp:
        root = Path(tmp)
        _stage(root)
        files = {name: (root / name).read_text() for name in
                 ("training/__init__.py", "training/sft_runtime.py", "training/sft_262k_runtime.py", *PINNED)}
        files["training/long_context_reload.py"] = Path(__file__).read_text()
        files["plan.json"] = canonical(plan).decode()
        _, request = historical_request(successor=True)
        request.update(name=NAME, title=NAME, run_dir=plan["output_root"])
        request["env"] = {k: v for k, v in request["env"].items() if not k.startswith("CYBER_SFT_BUNDLE")}
        request["env"].update(PYTHONPATH=plan["output_root"] + "/.runtime", WANDB_RUN_ID=NAME, WANDB_NAME=NAME)
        request = _old_python(root, """
import json,sys
from cyber_post_train.jobs import bundled_request
v=json.loads(sys.stdin.read())
print(json.dumps(bundled_request(v['request'],v['files'],'training.long_context_reload',
    ['--plan','plan.json','--plan-sha256',v['sha']]),sort_keys=True))
""", stdin=json.dumps({"request": request, "files": files, "sha": sha(canonical(plan))}))
    return plan, request


def preview(request):
    """Read-only server preview and duplicate check; no POST."""
    from training.long_context_launch import _old_python
    with tempfile.TemporaryDirectory(prefix="q38-262k-reload-") as tmp:
        root = Path(tmp)
        _stage(root)
        return _old_python(root, """
import json,os,sys
from cyber_post_train.jobs import Jobs,validate_preview
r=json.loads(sys.stdin.read())
with Jobs(os.environ.get('FLEET_API_KEY','')) as jobs:
    if any(x.get('name')==r['name'] or x.get('run_dir')==r['run_dir'] or
           x.get('title')==r['title'] for x in jobs.all_runs()):
        raise ValueError('reload identity already recorded')
    result=validate_preview(r,jobs.preview(r))
print(json.dumps(result,sort_keys=True))
""", stdin=json.dumps(request))


def check(plan):
    from training import sft_262k_runtime as long, sft_runtime as base
    long.install_runtime()
    from training import checkpoints, recovery

    source = plan["recovery"]["checkpoint"]["source_plan"]
    long.validate_plan(source, check_files=False)
    if (base._unsigned_digest(source) != SOURCE_SHA or
        plan["reload_runtime_sha256"] != base.digest(Path(__file__)) or
        plan["recovery_runtime_sha256"] != base.digest(Path(recovery.__file__)) or
        plan["recovery"]["manifest_path"] != str(MANIFEST)):
        raise ValueError("wrong frozen restore binding")
    base._checked_file(MANIFEST, plan["recovery"]["manifest_file_sha256"])
    checkpoints.verify(plan["recovery"]["checkpoint"], check_files=False)
    recovery.validate(plan, check_files=False)
    expected = {**source, "run_name": NAME, "output_root": f"/mnt/sfs/jobs/{NAME}"}
    expected["wandb"] = {**source["wandb"], "name": NAME, "run_id": NAME}
    for key in ("recovery", "recovery_runtime_sha256", "reload_runtime_sha256", "reload_gate"):
        expected[key] = plan[key]
    if plan != expected or plan["reload_gate"] != GATE:
        raise ValueError("restore-only plan changes source science or topology")


def restore(plan):
    from training import sft_262k_runtime as long, sft_runtime as base
    long.install_runtime()
    from training import recovery

    base._configure_wandb(plan)
    cfg, skyrl_cfg = base.build_runtime_configs(plan)
    skyrl_cfg.trainer.log_path = str(Path(plan["output_root"]) / "private_logs")
    trainer = long._make_trainer_class()(cfg, skyrl_cfg, plan)
    try:
        trainer.setup()
        trainer.train_dataloader = trainer.build_train_dataloader(trainer.load_dataset())
        step = recovery.load(trainer)
        if step != 1:
            raise ValueError("reload did not restore exact source step")
        return {"status": "reload_validated", "optimizer_steps_executed": 0,
                "optimizer_step": step, "world_size": 32,
                "source_manifest_sha256": plan["recovery"]["checkpoint"]["receipt_sha256"],
                "plan_sha256": plan["plan_sha256"]}
    finally:
        trainer.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    from training import sft_runtime as base

    base._checked_file(args.plan, args.plan_sha256)
    plan = json.loads(args.plan.read_text())
    check(plan)
    if not plan["reload_gate"]["submission_authorized"]:
        raise ValueError("restore-only GPU launch has not received separate root review")
    plan["plan_sha256"] = args.plan_sha256
    output = Path(plan["output_root"])
    output.mkdir(parents=True, exist_ok=True, mode=0o700)  # Bundle bootstrap made output/.runtime.
    base.write_receipt(output / "STARTED.json", {"plan_sha256": args.plan_sha256,
                                                  "started_at_unix": time.time()})
    ray = None
    try:
        import ray
        from skyrl.train.utils.utils import initialize_ray
        _, skyrl_cfg = base.build_runtime_configs(plan)
        initialize_ray(skyrl_cfg)
        task = ray.remote(num_cpus=1)(restore).remote(plan)
        result = base._wait_for_training(ray, task, output, plan=plan)
        base.write_receipt(output / "RELOAD_VALIDATED.json", result)
        print(json.dumps({"status": "reload_validated", "optimizer_steps_executed": 0}), flush=True)
    except BaseException as exc:
        base.write_receipt(output / "FAILED.json", {"status": "failed", "error_class": type(exc).__name__,
                                                    "plan_sha256": args.plan_sha256})
        raise SystemExit(1) from None
    finally:
        if ray is not None and ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()
