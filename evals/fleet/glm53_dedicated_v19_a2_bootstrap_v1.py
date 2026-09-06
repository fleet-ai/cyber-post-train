"""Render exact final GLM v19 attempt-2 package CPU preclaim."""
from __future__ import annotations
import copy
from pathlib import Path
from typing import Any
from evals.fleet import glm53_dedicated_v19_a2_package_v1 as controller
from evals.fleet import self_hosted

JOB_NAME="chris-glm53-dedicated-v19-r051-a2-bootstrap-v2"
SFS_ROOT=f"/mnt/sfs/jobs/{JOB_NAME}"

def render(root:Path, release_path:Path)->dict[str,Any]:
    final=controller.render(root); source,job=copy.deepcopy(final["objects"]["items"])
    source["metadata"]["name"]=JOB_NAME+"-source"
    evidence={"apiVersion":"v1","kind":"ConfigMap","metadata":{"name":JOB_NAME+"-evidence","namespace":"fleet-train-jobs"},"immutable":True,"data":{"parity.json":(root/controller.PARITY).read_text(),"binding.json":(root/controller.BINDING).read_text(),"release.json":release_path.read_text(),"service_origin":controller.ORIGIN}}
    job["metadata"]["name"]=JOB_NAME; job["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"]=JOB_NAME; job["metadata"]["annotations"]["cyber-post-train.fleet.ai/launch-authorized"]="false"; pod=job["spec"]["template"]; pod["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"]=JOB_NAME; pod["spec"]["volumes"][0]["configMap"]["name"]=source["metadata"]["name"]
    next(v for v in pod["spec"]["volumes"] if v["name"]=="evidence")["configMap"]["name"]=evidence["metadata"]["name"]
    ev=pod["spec"]["containers"][0]["env"]
    for x in ev:
        if x["name"]=="JOB_NAME": x["value"]=JOB_NAME
        elif x["name"]=="DEDICATED_SERVICE_ORIGIN":
            x.pop("valueFrom",None)
            x["value"]=controller.ORIGIN
    ev.extend([{"name":"DEDICATED_BOOTSTRAP_ONLY","value":"1"},{"name":"DEDICATED_BOOTSTRAP_RECEIPT","value":SFS_ROOT+"/BOOTSTRAP.json"},{"name":"DEDICATED_CONTROLLER_PACKAGE_SHA256","value":final["package_sha256"]}])
    objects={"apiVersion":"v1","kind":"List","items":[source,evidence,job]}
    return {"objects":objects,"controller_package_sha256":final["package_sha256"],"package_sha256":self_hosted.sha256(self_hosted.canonical_json(objects)),"launch_authorized":False}
