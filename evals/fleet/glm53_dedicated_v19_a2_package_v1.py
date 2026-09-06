"""Render held GLM v19 rank-51 attempt-2 controller package."""

from __future__ import annotations
import copy, json
from pathlib import Path
from typing import Any
from evals.fleet import glm53_dedicated_v14_scored_canary_package_v1 as old
from evals.fleet import glm53_dedicated_v19_a2_v1 as canary
from evals.fleet import self_hosted

PARITY = Path("docs/evidence/glm53-study/2026-09-06-glm53-dedicated-v19-actual-opencode-parity.json")
BINDING = Path("docs/evidence/glm53-study/2026-09-06-glm53-dedicated-v19-server-binding.json")
ORIGIN = "http://ft-run-9b834930-h9v6k-head-svc.fleet-train-jobs.svc.cluster.local:8000"
SOURCE_CONFIGMAP = canary.JOB_NAME + "-source"
EVIDENCE_CONFIGMAP = canary.JOB_NAME + "-evidence"

def render(root: Path) -> dict[str, Any]:
    held = canary.build_plan(root, service_origin=ORIGIN, parity_path=root/PARITY, binding_path=root/BINDING)
    data = old._source_data(root)
    data["dedicated_canary.py"] = (root/"evals/fleet/glm53_dedicated_v19_a2_v1.py").read_text()
    data["base_canary.py"] = (root/"evals/fleet/glm53_dedicated_v14_scored_canary_v1.py").read_text()
    data["base_runtime.py"] = (root/"evals/fleet/glm53_dedicated_v14_scored_canary_runtime_v1.py").read_text()
    data["dedicated_runtime.py"] = (root/"evals/fleet/glm53_dedicated_v19_a2_runtime_v1.py").read_text()
    data["run.sh"] = (root/"evals/fleet/scripts/run_glm53_dedicated_v19_a2_v1.sh").read_text()
    source = {"apiVersion":"v1","kind":"ConfigMap","metadata":{"name":SOURCE_CONFIGMAP,"namespace":"fleet-train-jobs"},"immutable":True,"data":data}
    template = copy.deepcopy(old.base.render(root)["objects"]["items"][1])
    template["metadata"]["name"] = canary.JOB_NAME
    template["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = canary.JOB_NAME
    template["metadata"]["annotations"]["cyber-post-train.fleet.ai/launch-authorized"] = "false"
    pod=template["spec"]["template"]
    pod["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = canary.JOB_NAME
    pod["spec"]["priorityClassName"]="fleet-infra-quiet"; pod["spec"]["preemptionPolicy"]="Never"
    pod["spec"]["volumes"][0]["configMap"]["name"] = SOURCE_CONFIGMAP
    evaluator=pod["spec"]["containers"][0]
    for env in evaluator["env"]:
        if env["name"]=="JOB_NAME": env["value"]=canary.JOB_NAME
    evaluator["env"].append({"name":"DEDICATED_SERVICE_ORIGIN","valueFrom":{"configMapKeyRef":{"name":EVIDENCE_CONFIGMAP,"key":"service_origin"}}})
    evaluator["volumeMounts"].append({"name":"evidence","mountPath":"/evidence","readOnly":True})
    pod["spec"]["volumes"].append({"name":"evidence","configMap":{"name":EVIDENCE_CONFIGMAP}})
    objects={"apiVersion":"v1","kind":"List","items":[source,template]}
    body={"schema_version":"fleet-glm53-dedicated-v19-a2-package-v1","status":"READY_HELD","launch_authorized":False,"job_name":canary.JOB_NAME,"source_configmap":SOURCE_CONFIGMAP,"evidence_configmap":EVIDENCE_CONFIGMAP,"held_plan_sha256":held["plan_sha256"],"reserved_cell_ids":held["whole_task_reservation"]["cell_ids"],"reserved_execution_ids":held["whole_task_reservation"]["execution_ids"],"objects_sha256":self_hosted.sha256(self_hosted.canonical_json(objects))}
    body["package_sha256"]=self_hosted.digest_without(body,"package_sha256")
    return {**body,"objects":objects}
