"""Fresh score-blind release for GLM v19 rank-51 attempt 2."""
from __future__ import annotations
import hashlib, json, os, uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from evals.fleet import exact_pass4_bulk_runtime_v3 as engine
from evals.fleet import glm53_dedicated_v14_scored_canary_runtime_v1 as base_runtime
from evals.fleet import glm53_dedicated_v19_a2_runtime_v1 as runtime
from evals.fleet import glm53_dedicated_v19_a2_v1 as canary
from evals.fleet import glm53_dedicated_v15_canary_release_v1 as old
from evals.fleet import hosted_glm_exact_bulk_runtime_v1 as inventory
from evals.fleet import hosted_glm_exact_bulk_v1 as source
from evals.fleet import hosted_glm_exact_bulk_release_v1 as ledger
from evals.fleet import self_hosted

JOB_NAME="chris-glm53-dedicated-v19-r051-a2-release-v1"
CONFIGMAP_NAME=JOB_NAME+"-run"
OUTPUT=Path("/mnt/sfs/jobs")/JOB_NAME/"RELEASE.json"
PREDECESSOR=Path("/mnt/sfs/jobs/chris-glm53-dedicated-v18-r051-a1-canary-v2/accepted/chris-glm53-ac-bulk-b-r051-a1-g1-4283ed2c.json")
PREDECESSOR_FILE="sha256:a5cd6929cce9ca8c41c6087b3cf0fb034510d34643a53e05d3506d1ec0096e0f"
PREDECESSOR_SELF="sha256:03f7728bb2693d55264537a63f6412c936ac337d9a7accbb4344df65221142b3"

def build(root:Path, package_path:Path)->dict[str,Any]:
    key=os.environ.get("FLEET_API_KEY",""); job_uid=os.environ.get("JOB_UID",""); pod_uid=os.environ.get("POD_UID","")
    if not key or any(uuid.UUID(v).int==0 for v in (job_uid,pod_uid)): raise RuntimeError("release authority absent")
    package=json.loads(package_path.read_text())
    if package.get("status")!="READY_HELD" or package.get("launch_authorized") is not False or package.get("package_sha256")!=self_hosted.digest_without(package,"package_sha256"): raise RuntimeError("package drifted")
    raw=PREDECESSOR.read_bytes()
    if "sha256:"+hashlib.sha256(raw).hexdigest()!=PREDECESSOR_FILE: raise RuntimeError("predecessor file drifted")
    pred=json.loads(raw)
    if pred.get("receipt_sha256")!=PREDECESSOR_SELF or pred.get("receipt_sha256")!=self_hosted.digest_without(pred,"receipt_sha256") or pred.get("accepted") is not True or pred.get("credited") is not True or pred.get("selection_rank")!=51 or pred.get("attempt")!=1: raise RuntimeError("predecessor acceptance drifted")
    inv=source.load(inventory.INVENTORY_PATH)
    plan=canary.build_runtime_plan(canary.CONTROLLER,inv,root)
    item=plan["attempts"][0]
    if item["attempt"]!=2 or package.get("reserved_cell_ids",[])[1]!=item["cell_id"] or package.get("reserved_execution_ids",[])[1]!=item["execution_id"]: raise RuntimeError("attempt2 reservation drifted")
    if OUTPUT.parent.exists() or OUTPUT.parent.is_symlink() or canary.SFS_ROOT.exists() or canary.SFS_ROOT.is_symlink(): raise RuntimeError("SFS collision")
    claim=canary.CLAIM_ROOT/engine.claim_filename(item["execution_id"])
    claim_collision=int(claim.exists() or claim.is_symlink())
    kube_collision=sum(old._kube_collision(n,k) for n,k in ((canary.JOB_NAME,"jobs"),(canary.CONFIGMAP_NAME,"configmaps"),(canary.JOB_NAME+"-source","configmaps"),(canary.JOB_NAME+"-evidence","configmaps")))
    task=engine._task_for_item(plan,item); config=engine._attempt_config(plan,task,item)
    with engine._client(key) as client: sessions=self_hosted._task_sessions(client,config["task"]["key"])
    session_collision=sum(ledger._session_collides(row,config,item) for row in sessions)
    base_runtime._route_check(plan,key)
    if claim_collision or kube_collision or session_collision: raise RuntimeError("attempt2 duplicate release not clear")
    body={"schema_version":runtime.RELEASE_SCHEMA,"status":"CLEAR","plan_sha256":plan["plan_sha256"],"cell_id":item["cell_id"],"execution_id":item["execution_id"],"selection_rank":51,"attempt":2,"accepted_predecessor_attempts":[1],"predecessor_acceptance_receipt_sha256":PREDECESSOR_SELF,"fleet_session_collisions":0,"global_claim_collisions":0,"sfs_output_collisions":0,"kubernetes_object_collisions":0,"checked_immediately_before_create":True,"observer_job_uid":job_uid,"observer_pod_uid":pod_uid,"observed_at_utc":datetime.now(UTC).isoformat().replace('+00:00','Z'),"mutation_calls":0,"scores_read":False,"prompts_traces_flags_read":False}
    return {**body,"receipt_sha256":self_hosted.digest_without(body,"receipt_sha256")}
