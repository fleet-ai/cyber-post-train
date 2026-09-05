"""Render the held, evidence-parameterized GLM dedicated scored-canary package."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import yaml

from evals.fleet import glm53_dedicated_v14_scored_canary_v1 as canary
from evals.fleet import hosted_glm_exact_canary_package_v1 as base
from evals.fleet import self_hosted

SOURCE_CONFIGMAP = canary.JOB_NAME + "-source"
EVIDENCE_CONFIGMAP = canary.JOB_NAME + "-evidence"
PARITY = Path("docs/evidence/glm53-study/2026-09-05-glm53-dedicated-v14-actual-opencode-parity.json")
BINDING = Path("docs/evidence/glm53-study/2026-09-05-glm53-dedicated-v14-server-binding.json")
OLD_SERVICE_ORIGIN = "http://glm-v14-head-svc.fleet-train-jobs.svc.cluster.local:8000"

EXTRA_PATHS = {
    "hosted_bulk.py": "evals/fleet/hosted_glm_exact_bulk_v1.py",
    "hosted_bulk_runtime.py": "evals/fleet/hosted_glm_exact_bulk_runtime_v1.py",
    "dedicated_canary.py": "evals/fleet/glm53_dedicated_v14_scored_canary_v1.py",
    "dedicated_runtime.py": "evals/fleet/glm53_dedicated_v14_scored_canary_runtime_v1.py",
    "run.sh": "evals/fleet/scripts/run_glm53_dedicated_v14_scored_canary_v1.sh",
}
INSTALL_TARGETS = {
    "hosted_bulk.py": "evals/fleet/hosted_glm_exact_bulk_v1.py",
    "hosted_bulk_runtime.py": "evals/fleet/hosted_glm_exact_bulk_runtime_v1.py",
    "dedicated_canary.py": "evals/fleet/glm53_dedicated_v14_scored_canary_v1.py",
    "dedicated_runtime.py": "evals/fleet/glm53_dedicated_v14_scored_canary_runtime_v1.py",
}


def _source_data(root: Path) -> dict[str, str]:
    data = copy.deepcopy(base.render(root)["objects"]["items"][0]["data"])
    for name, relative in EXTRA_PATHS.items():
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"unsafe dedicated package source: {relative}")
        data[name] = path.read_text()
    run_script = data["run.sh"]
    base_required = {
        name: target
        for name, target in base.INSTALL_PATHS.items()
        if name not in {"canary.py", "parity.json"}
    }
    for name, target in {**base_required, **INSTALL_TARGETS}.items():
        if f"{name}:" not in run_script and f"/bootstrap/{name}" not in run_script:
            raise ValueError(f"dedicated packaged runtime install closure omits {name}")
        if Path(target).name not in run_script:
            raise ValueError(f"dedicated packaged runtime install closure omits {name}")
    for name, text in data.items():
        if name.endswith(".py"):
            compile(text, name, "exec")
    return data


def render(root: Path) -> dict[str, Any]:
    held = canary.build_plan(
        root,
        service_origin=OLD_SERVICE_ORIGIN,
        parity_path=root / PARITY,
        binding_path=root / BINDING,
    )
    data = _source_data(root)
    source = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": SOURCE_CONFIGMAP, "namespace": "fleet-train-jobs"},
        "immutable": True,
        "data": data,
    }
    template = copy.deepcopy(base.render(root)["objects"]["items"][1])
    template["metadata"]["name"] = canary.JOB_NAME
    template["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = canary.JOB_NAME
    template["metadata"]["annotations"]["cyber-post-train.fleet.ai/launch-authorized"] = "false"
    pod = template["spec"]["template"]
    pod["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = canary.JOB_NAME
    pod["spec"]["priorityClassName"] = "fleet-infra-quiet"
    pod["spec"]["preemptionPolicy"] = "Never"
    evaluator = pod["spec"]["containers"][0]
    for env in evaluator["env"]:
        if env["name"] == "JOB_NAME":
            env["value"] = canary.JOB_NAME
    evaluator["env"].append(
        {
            "name": "DEDICATED_SERVICE_ORIGIN",
            "valueFrom": {"configMapKeyRef": {"name": EVIDENCE_CONFIGMAP, "key": "service_origin"}},
        }
    )
    evaluator["volumeMounts"].append(
        {"name": "evidence", "mountPath": "/evidence", "readOnly": True}
    )
    pod["spec"]["volumes"][0]["configMap"]["name"] = SOURCE_CONFIGMAP
    pod["spec"]["volumes"].append(
        {"name": "evidence", "configMap": {"name": EVIDENCE_CONFIGMAP}}
    )
    objects = {"apiVersion": "v1", "kind": "List", "items": [source, template]}
    source_bytes = len(json.dumps(source).encode())
    if source_bytes >= 900_000:
        raise ValueError("dedicated canary source ConfigMap exceeds safety budget")
    body = {
        "schema_version": "fleet-glm53-dedicated-v14-scored-canary-package-v1",
        "status": "READY_HELD",
        "launch_authorized": False,
        "job_name": canary.JOB_NAME,
        "source_configmap": SOURCE_CONFIGMAP,
        "evidence_configmap": EVIDENCE_CONFIGMAP,
        "held_plan_sha256": held["plan_sha256"],
        "reserved_cell_ids": held["whole_task_reservation"]["cell_ids"],
        "reserved_execution_ids": held["whole_task_reservation"]["execution_ids"],
        "source_configmap_json_bytes": source_bytes,
        "objects_sha256": self_hosted.sha256(self_hosted.canonical_json(objects)),
        "dynamic_evidence_required": [
            "fresh_server_binding", "fresh_non_scored_parity", "fresh_duplicate_release"
        ],
    }
    body["package_sha256"] = self_hosted.digest_without(body, "package_sha256")
    return {**body, "objects": objects}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preview", "render-held"))
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    built = render(args.repo.resolve())
    if args.command == "preview":
        print(json.dumps({k: v for k, v in built.items() if k != "objects"}, sort_keys=True))
    else:
        print(yaml.safe_dump(built["objects"], sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
