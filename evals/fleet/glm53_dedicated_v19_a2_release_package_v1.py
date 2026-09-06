"""Render GLM v19 attempt-2 duplicate release observer."""
from __future__ import annotations
import copy, json
from pathlib import Path
from typing import Any
from evals.fleet import glm53_dedicated_v14_scored_canary_preflight_package_v1 as base
from evals.fleet import glm53_dedicated_v19_a2_package_v1 as controller
from evals.fleet import glm53_dedicated_v19_a2_release_v1 as release
from evals.fleet import self_hosted

def render(root:Path)->dict[str,Any]:
    package=controller.render(root); preview={k:v for k,v in package.items() if k!='objects'}
    data=copy.deepcopy(base.render(root)["objects"]["items"][0]["data"])
    paths={"base_canary.py":"evals/fleet/glm53_dedicated_v14_scored_canary_v1.py","base_runtime.py":"evals/fleet/glm53_dedicated_v14_scored_canary_runtime_v1.py","dedicated_canary.py":"evals/fleet/glm53_dedicated_v19_a2_v1.py","dedicated_runtime.py":"evals/fleet/glm53_dedicated_v19_a2_runtime_v1.py","old_release.py":"evals/fleet/glm53_dedicated_v15_canary_release_v1.py","release.py":"evals/fleet/glm53_dedicated_v19_a2_release_v1.py","run.sh":"evals/fleet/scripts/run_glm53_dedicated_v19_a2_release_v1.sh"}
    for name,path in paths.items(): data[name]=(root/path).read_text()
    data["controller-package.json"]=self_hosted.canonical_json(preview).decode(); data["parity.json"]=(root/controller.PARITY).read_text(); data["binding.json"]=(root/controller.BINDING).read_text()
    cm={"apiVersion":"v1","kind":"ConfigMap","metadata":{"name":release.CONFIGMAP_NAME,"namespace":"fleet-train-jobs"},"immutable":True,"data":data}
    job=copy.deepcopy(base.render(root)["objects"]["items"][1]); job["metadata"]["name"]=release.JOB_NAME; job["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"]=release.JOB_NAME; job["metadata"]["annotations"]["cyber-post-train.fleet.ai/launch-authorized"]="true"; pod=job["spec"]["template"]; pod["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"]=release.JOB_NAME; pod["spec"]["volumes"][0]["configMap"]["name"]=release.CONFIGMAP_NAME; pod["spec"]["containers"][0]["env"].append({"name":"DEDICATED_SERVICE_ORIGIN","value":controller.ORIGIN})
    objects={"apiVersion":"v1","kind":"List","items":[cm,job]}
    return {"objects":objects,"package_sha256":self_hosted.sha256(self_hosted.canonical_json(objects)),"controller_package_sha256":package["package_sha256"],"launch_authorized":True}
