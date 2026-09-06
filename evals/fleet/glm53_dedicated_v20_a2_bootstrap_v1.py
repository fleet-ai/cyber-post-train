"""Render exact final GLM v20 attempt2 CPU preclaim."""
from __future__ import annotations
import copy
from pathlib import Path
from evals.fleet import glm53_dedicated_v20_a2_package_v1 as controller
from evals.fleet import self_hosted
JOB_NAME="chris-glm53-dedicated-v20-r051-a2-bootstrap-v3"; SFS_ROOT=f"/mnt/sfs/jobs/{JOB_NAME}"
RELEASE_PATH=Path("/mnt/sfs/jobs/chris-glm53-dedicated-v20-r051-a2-release-v2/RELEASE.json")
RELEASE_FILE_SHA256="sha256:f6950e059f409b8cd7af33908bd409762668f9c07d2de557e5066eac0d5e6968"
PREDECESSOR_PATH=Path("/mnt/sfs/jobs/chris-glm53-dedicated-v18-r051-a1-canary-v2/accepted/chris-glm53-ac-bulk-b-r051-a1-g1-4283ed2c.json")
def render(root:Path):
 final=controller.render(root); source,job=copy.deepcopy(final["objects"]["items"]); source["metadata"]["name"]=JOB_NAME+"-source"; evidence={"apiVersion":"v1","kind":"ConfigMap","metadata":{"name":JOB_NAME+"-evidence-source","namespace":"fleet-train-jobs"},"immutable":True,"data":{"parity.json":(root/controller.PARITY).read_text(),"binding.json":(root/controller.BINDING).read_text(),"stage.py":(root/"evals/fleet/sfs_evidence_stage_v1.py").read_text(),"service_origin":controller.ORIGIN}}; job["metadata"]["name"]=JOB_NAME; job["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"]=JOB_NAME; job["metadata"]["annotations"]["cyber-post-train.fleet.ai/launch-authorized"]="false"; pod=job["spec"]["template"]; pod["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"]=JOB_NAME; pod["spec"]["volumes"][0]["configMap"]["name"]=source["metadata"]["name"]
 evidence_volume=next(v for v in pod["spec"]["volumes"] if v["name"]=="evidence"); evidence_volume.clear(); evidence_volume.update({"name":"evidence","emptyDir":{}}); pod["spec"]["volumes"].append({"name":"evidence-source","configMap":{"name":evidence["metadata"]["name"]}})
 stager={"name":"stage-evidence","image":"ghcr.io/astral-sh/uv:python3.12-bookworm@sha256:9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7","command":["python","-c"],"args":["import sys;sys.path.insert(0,'/evidence-source');from stage import stage;from pathlib import Path;stage(projected_root=Path('/evidence-source'),sfs_release=Path('"+str(RELEASE_PATH)+"'),predecessor=Path('"+str(PREDECESSOR_PATH)+"'),output_root=Path('/evidence'),release_file_sha256='"+RELEASE_FILE_SHA256+"')"],"securityContext":{"allowPrivilegeEscalation":False,"readOnlyRootFilesystem":True,"runAsNonRoot":False,"runAsUser":0,"runAsGroup":0},"resources":{"requests":{"cpu":"100m","memory":"128Mi"},"limits":{"cpu":"500m","memory":"512Mi"}},"volumeMounts":[{"name":"evidence-source","mountPath":"/evidence-source","readOnly":True},{"name":"evidence","mountPath":"/evidence"},{"name":"sfs","mountPath":"/mnt/sfs","readOnly":True}]}; pod["spec"]["initContainers"].insert(0,stager)
 ev=pod["spec"]["containers"][0]["env"]
 for x in ev:
  if x["name"]=="JOB_NAME": x["value"]=JOB_NAME
  elif x["name"]=="DEDICATED_SERVICE_ORIGIN": x.pop("valueFrom",None); x["value"]=controller.ORIGIN
 ev.extend([{"name":"DEDICATED_BOOTSTRAP_ONLY","value":"1"},{"name":"DEDICATED_BOOTSTRAP_RECEIPT","value":SFS_ROOT+"/BOOTSTRAP.json"},{"name":"DEDICATED_CONTROLLER_PACKAGE_SHA256","value":final["package_sha256"]}]); objects={"apiVersion":"v1","kind":"List","items":[source,evidence,job]}; return {"objects":objects,"controller_package_sha256":final["package_sha256"],"release_file_sha256":RELEASE_FILE_SHA256,"package_sha256":self_hosted.sha256(self_hosted.canonical_json(objects)),"launch_authorized":False}
