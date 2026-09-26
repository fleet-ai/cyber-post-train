"""Build the exact historical 57M-token four-node 262K SFT request offline.

This only prepares a request; the capacity canary must pass before submission.
"""

import copy
import json
import tempfile
from pathlib import Path

from training.long_context_launch import ROOT, _old_python, digest, stage_v4_code, v4_request, verify_bundle

SPEC = ROOT / "configs/runs/qwen38-262k-four-node-full57-v1.json"
MANIFEST = ROOT / "configs/data/qwen38-teacher3k-96k-v1.manifest.json"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def build():
    spec = json.loads(SPEC.read_text())
    canary, _ = v4_request()
    if (digest(canonical(canary)) != spec["capacity_canary_plan_sha256"] or
        spec["max_steps"] != (spec["train_rows"] + spec["batch_size"] - 1) // spec["batch_size"] * spec["epochs"] or
        (spec["submission_authorized"] and not spec["capacity_acceptance_sha256"])):
        raise ValueError("full run has no exact capacity or data binding")
    train = copy.deepcopy(json.loads(MANIFEST.read_text())["files"]["train"])
    train.update(path=spec["train_path"], sha256=spec["train_sha256"], rows=spec["train_rows"])
    if train["supervised_tokens"] != spec["supervised_tokens"] or digest(canonical({"train": train})) != spec["datasets_sha256"]:
        raise ValueError("historical 57M-token dataset metadata changed")
    plan = copy.deepcopy(canary)
    name = spec["name"]
    plan.update(run_name=name, output_root=spec["output_root"], datasets={"train": train},
                corpus_manifest_sha256=spec["corpus_manifest_sha256"],
                qualification={"schema": spec["schema"], "capacity_acceptance_sha256": spec["capacity_acceptance_sha256"],
                               "submission_authorized": spec["submission_authorized"], "purpose": "historical-corpus capacity training; not accepted lift evidence"})
    plan.pop("pause_after_step")
    plan["recipe"].update(epochs=spec["epochs"], batch_size=spec["batch_size"],
                          max_steps=spec["max_steps"], checkpoint_interval=spec["checkpoint_interval"])
    plan["wandb"].update(name=name, run_id=name, group="qwen38-teacher3k-262k-historical57m-v1",
                         tags=["qwen38", "teacher-sft", "262k", "four-node", "full-weight", "historical-57m", "not-lift-accepted"])
    science_sha = digest(canonical({k: v for k, v in plan.items() if k != "runtime_sha256"}))
    with tempfile.TemporaryDirectory(prefix="q38-262k-full-") as tmp:
        root = Path(tmp)
        stage_v4_code(root)
        path = root / "training/sft_262k_runtime.py"
        source = path.read_text()
        anchor = "    _BASE_VALIDATE_PLAN(plan, check_files=check_files)\n"
        branch = f'''    if plan.get("run_name") == {name!r}:
        if "plan_sha256" in plan and plan["plan_sha256"] != base._unsigned_digest({{k: v for k, v in plan.items() if k != "plan_sha256"}}):
            raise ValueError("runtime plan digest changed")
        if base._unsigned_digest({{k: v for k, v in plan.items() if k not in ("runtime_sha256", "plan_sha256")}}) != {science_sha!r}:
            raise ValueError("historical 57M-token full plan changed")
        if check_files:
            base._checked_file(Path(__file__), plan["runtime_sha256"])
        return
'''
        if source.count(anchor) != 1:
            raise ValueError("qualified runtime validation hook changed")
        path.write_text(source.replace(anchor, anchor + branch))
        plan["runtime_sha256"] = digest(path.read_bytes())
        request = _old_python(root, """
import json,sys
from cyber_post_train.jobs import digest
from training.sft_262k_4node_v1 import job_request
from training.sft_262k_runtime import validate_plan
plan=json.loads(sys.stdin.read())
validate_plan(plan,check_files=False)
validate_plan({**plan,'plan_sha256':digest(plan)},check_files=False)
print(json.dumps(job_request(plan),sort_keys=True))
""", stdin=json.dumps(plan))
    verify_bundle(plan, request)
    return plan, request
