"""Fresh GLM TP8 v20 identity using the qualified metric watchdog."""
from __future__ import annotations
import copy, json
from pathlib import Path
from typing import Any
from evals.fleet import glm53_dedicated_v19 as v19

VERSION="v20"; TITLE="chris-cyber-evalserve-glm53-tp8-a-v20"; RUN_DIR="/mnt/sfs/jobs/chris-cyber-evalserve-glm53-tp8-a-v20"
PRE_ADMISSION=v19.PRE_ADMISSION; WATCHDOG_QUALIFICATION=v19.WATCHDOG_QUALIFICATION
PRIORITY_CLASS=v19.PRIORITY_CLASS; API_PRIORITY_CLASSES=v19.API_PRIORITY_CLASSES; QUEUE=v19.QUEUE; TOPOLOGY_MODE=v19.TOPOLOGY_MODE; TOPOLOGY_LEVEL=v19.TOPOLOGY_LEVEL

def spec(root:Path)->dict[str,Any]:
    v=copy.deepcopy(v19.spec(root)); v["schema_version"]="fleet-glm53-dedicated-serving-v20-authorized-v1"; v["title"]=TITLE; v["run_dir"]=RUN_DIR; v["request_shape_change"]={"predecessor":"v19","cpu_request":{"from":"64","to":"64"},"memory_request":{"from":"768Gi","to":"768Gi"},"reason":"fresh_identity_after_v19_preclaim_packaging_fix"}; validate(v,root); return v

def _as_v19(v:dict[str,Any])->dict[str,Any]:
    n=copy.deepcopy(v); n["schema_version"]="fleet-glm53-dedicated-serving-v19-authorized-v1"; n["title"]=v19.TITLE; n["run_dir"]=v19.RUN_DIR; n["request_shape_change"]={"predecessor":"v18","cpu_request":{"from":"64","to":"64"},"memory_request":{"from":"768Gi","to":"768Gi"},"reason":"fresh_identity_after_v18_false_idle_with_qualified_metric_watchdog"}; return n

def validate(v:dict[str,Any],root:Path)->None:
    if v.get("schema_version")!="fleet-glm53-dedicated-serving-v20-authorized-v1" or v.get("title")!=TITLE or v.get("run_dir")!=RUN_DIR: raise ValueError("v20 identity drifted")
    if v.get("request_shape_change",{}).get("reason")!="fresh_identity_after_v19_preclaim_packaging_fix": raise ValueError("v20 rationale drifted")
    v19.validate(_as_v19(v),root)

def payload(v:dict[str,Any],root:Path)->dict[str,Any]:
    validate(v,root); r=v19.payload(_as_v19(v),root); r["title"]=TITLE; r["run_dir"]=RUN_DIR; r["env"]["GLM53_RUN_DIR"]=RUN_DIR; return r
