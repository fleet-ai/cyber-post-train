"""Create-once Jobs API rail for GLM v20."""
from __future__ import annotations
import argparse,json,subprocess
from datetime import UTC,datetime
from pathlib import Path
import httpx
from evals.fleet import glm53_dedicated_v8_live as shared
from evals.fleet import glm53_dedicated_v15_live as common
from evals.fleet import glm53_dedicated_v19_live as prior
from evals.fleet import glm53_dedicated_v20 as v20
from evals.fleet import self_hosted

ALLOWED={"/mnt/sfs/jobs/chris-cyber-evalserve-q38-tp1-j-v1":{"nodes":1,"gpus":1}}

def live_gate(root:Path)->dict:
    raw,watchdog=prior._watchdog(); gate=common.live_gate(root,v20,ALLOWED); gate.pop("pre_admission",None); gate["watchdog_qualification"]={"path":v20.WATCHDOG_QUALIFICATION["receipt_path"],"file_sha256":self_hosted.sha256(raw),"receipt_sha256":watchdog["receipt_sha256"],"lifecycle_file_sha256":watchdog["lifecycle_file_sha256"],"status":watchdog["status"]}; gate["scored_create_gate"]={"status":"CLOSED","fresh_uid_bound_parity_required":True,"fresh_server_bound_release_required":True,"fresh_server_bound_preclaim_required":True}; gate["scored_tasks_launched"]=0; gate["receipt_sha256"]=self_hosted.digest_without(gate,"receipt_sha256"); return gate

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("command",choices=("preview","submit")); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    if a.output.exists() or a.output.is_symlink(): raise RuntimeError("output collision")
    if a.command=="submit" and subprocess.run(["git","status","--porcelain","--untracked-files=all"],capture_output=True,text=True,check=True).stdout: raise RuntimeError("submit requires clean worktree")
    g=live_gate(Path.cwd())
    if a.command=="preview": shared._write_once(a.output,g); return 0
    payload=v20.payload(v20.spec(Path.cwd()),Path.cwd())
    with httpx.Client(base_url=shared.BASE_URL,headers={"Authorization":f"Bearer {shared._token()}","Accept":"application/json"},timeout=60) as c: r=c.post("/v1/runs",json=payload); r.raise_for_status(); name=r.json().get("name")
    if r.status_code!=202 or not isinstance(name,str) or not name.startswith("ft-run-"): raise RuntimeError("submit identity absent")
    body={"schema_version":"fleet-glm53-dedicated-serving-v20-submission-v1","status":"SUBMITTED","submitted_at_utc":datetime.now(UTC).isoformat().replace('+00:00','Z'),"api_run_id":name,"title":v20.TITLE,"run_dir":v20.RUN_DIR,"request_sha256":g["request_sha256"],"live_gate_receipt_sha256":g["receipt_sha256"],"watchdog_qualification":g["watchdog_qualification"],"scored_create_gate":g["scored_create_gate"],"project_resource_shape":g["project_resource_shape"],"selected_priority":g["selected_priority"],"http_status":202,"scored_tasks_launched":0,"credentials_included":False,"prompts_traces_flags_or_scores_included":False}; body["receipt_sha256"]=self_hosted.digest_without(body,"receipt_sha256"); shared._write_once(a.output,body); print(json.dumps({"api_run_id":name,"status":"SUBMITTED"})); return 0
if __name__=="__main__": raise SystemExit(main())
