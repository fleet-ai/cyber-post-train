"""Render a non-scored exact-package GLM controller bootstrap canary."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import yaml

from evals.fleet import glm53_dedicated_v14_scored_canary_package_v1 as controller
from evals.fleet import self_hosted

JOB_NAME = "chris-glm53-dedicated-v16-r051-bootstrap-v1"
SOURCE_CONFIGMAP = JOB_NAME + "-source"
EVIDENCE_CONFIGMAP = JOB_NAME + "-evidence"
SFS_ROOT = f"/mnt/sfs/jobs/{JOB_NAME}"


def render(root: Path) -> dict[str, Any]:
    final = controller.render(root)
    final_source, final_job = final["objects"]["items"]
    source = copy.deepcopy(final_source)
    source["metadata"]["name"] = SOURCE_CONFIGMAP
    evidence = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": EVIDENCE_CONFIGMAP, "namespace": "fleet-train-jobs"},
        "immutable": True,
        "data": {
            "parity.json": "{}\n",
            "binding.json": "{}\n",
            "release.json": "{}\n",
        },
    }
    job = copy.deepcopy(final_job)
    job["metadata"]["name"] = JOB_NAME
    job["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = JOB_NAME
    job["metadata"]["annotations"]["cyber-post-train.fleet.ai/launch-authorized"] = "false"
    pod = job["spec"]["template"]
    pod["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = JOB_NAME
    pod["spec"]["volumes"][0]["configMap"]["name"] = SOURCE_CONFIGMAP
    pod["spec"]["volumes"].append(
        {"name": "evidence", "configMap": {"name": EVIDENCE_CONFIGMAP}}
    )
    evaluator = pod["spec"]["containers"][0]
    for env in evaluator["env"]:
        if env["name"] == "JOB_NAME":
            env["value"] = JOB_NAME
    evaluator["env"].extend(
        [
            {"name": "DEDICATED_BOOTSTRAP_ONLY", "value": "1"},
            {
                "name": "DEDICATED_BOOTSTRAP_RECEIPT",
                "value": SFS_ROOT + "/BOOTSTRAP.json",
            },
            {
                "name": "DEDICATED_CONTROLLER_PACKAGE_SHA256",
                "value": final["package_sha256"],
            },
            {
                "name": "DEDICATED_SERVICE_ORIGIN",
                "value": "http://bootstrap.invalid.fleet-train-jobs.svc.cluster.local:8000",
            },
        ]
    )
    evaluator["volumeMounts"].append(
        {"name": "evidence", "mountPath": "/evidence", "readOnly": True}
    )
    objects = {"apiVersion": "v1", "kind": "List", "items": [source, evidence, job]}
    body = {
        "schema_version": "fleet-glm53-dedicated-v16-controller-bootstrap-package-v1",
        "status": "READY_NON_SCORED",
        "launch_authorized": False,
        "job_name": JOB_NAME,
        "sfs_root": SFS_ROOT,
        "controller_package_sha256": final["package_sha256"],
        "source_data_sha256": self_hosted.sha256(
            self_hosted.canonical_json(final_source["data"])
        ),
        "run_sh_sha256": self_hosted.sha256(final_source["data"]["run.sh"].encode()),
        "uses_exact_controller_source_data": source["data"] == final_source["data"],
        "stops_before_claim_model_session_verifier_or_scoring": True,
        "objects": objects,
    }
    body["package_sha256"] = self_hosted.digest_without(body, "package_sha256")
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preview", "render"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    value = render(Path.cwd())
    if args.command == "preview":
        print(json.dumps({k: v for k, v in value.items() if k != "objects"}, sort_keys=True))
        return 0
    if args.output is None or args.output.exists() or args.output.is_symlink():
        parser.error("render requires an unused --output")
    args.output.write_text(yaml.safe_dump(value["objects"], sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
