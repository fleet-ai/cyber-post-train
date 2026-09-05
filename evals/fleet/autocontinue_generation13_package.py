"""Render authorized G13 Qwen/GLM ConfigMap+Job packages."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import yaml

from evals.fleet import autocontinue_generation10_glm53_v1 as glm10
from evals.fleet import autocontinue_generation10_qwen_v1 as q10
from evals.fleet import autocontinue_generation13_simple_cell as runtime
from evals.fleet import exact_pass4_universe as exact
from evals.fleet import opencode_train_sweep_runner as sweep
from evals.fleet import self_hosted

MODULE_PATH = "evals/fleet/autocontinue_generation13_package.py"
RUNTIME_PATH = "evals/fleet/autocontinue_generation13_simple_cell.py"
RUN_PATH = "evals/fleet/scripts/run_opencode_autocontinue_generation13.sh"
MANIFESTS = {
    "qwen3.8-27b": "evals/fleet/cluster/opencode-autocontinue-generation13-qwen-held-v1.yaml",
    "glm-5.3": "evals/fleet/cluster/opencode-autocontinue-generation13-glm53-held-v1.yaml",
}
G10 = {"qwen3.8-27b": q10, "glm-5.3": glm10}
PAYLOAD_PATHS = {
    "Dockerfile.opencode": "evals/fleet/Dockerfile.opencode",
    "fixed_proxy.py": "evals/fleet/fixed_proxy.py",
    "self_hosted.py": "evals/fleet/self_hosted.py",
    "runner.py": "evals/fleet/opencode_train_sweep_runner.py",
    "g13.py": RUNTIME_PATH,
    "run.sh": RUN_PATH,
}


def derive_spec(root: Path, model: str) -> dict[str, Any]:
    row = runtime.EXPECTED[model]
    spec10, plan10, _held = G10[model].static(root)
    execution = exact.execution_for(row["cell_id"], 13)
    if execution["execution_id"] != row["execution_id"]:
        raise ValueError("derived Generation-13 execution identity drifted")
    plan = copy.deepcopy(plan10)
    plan["campaign_id"] = row["job_name"]
    plan["source_job_id"] = row["job_name"]
    item = copy.deepcopy(plan["attempts"][0])
    item.update(
        execution_generation=13,
        network=row["network"],
        run_id=row["run_id"],
    )
    attempt = sweep._attempt_config(plan, plan["tasks"][0], item)
    tombstone = runtime.load(root / row["tombstone_path"])
    runtime.validate_tombstone(tombstone, row)
    body = {
        "schema_version": "fleet-opencode-generation13-simple-cell-v1",
        "model": model,
        "source_rank": row["rank"],
        "attempt": 1,
        "cell_id": row["cell_id"],
        "execution_id": row["execution_id"],
        "execution_generation": 13,
        "job_name": row["job_name"],
        "configmap_name": row["configmap_name"],
        "output_root": f"/mnt/sfs/jobs/{row['job_name']}",
        "stage_root": f"/mnt/sfs/jobs/{row['job_name']}-stages",
        "claim_root": str(runtime.CLAIM_ROOT),
        "run_id": row["run_id"],
        "network": row["network"],
        "predecessor_generation12": {
            "job_name": row["g12_job"],
            "execution_id": row["g12_execution_id"],
            "tombstone_path": row["tombstone_path"],
            "receipt_sha256": row["tombstone_sha256"],
            "job_uid": tombstone["job"]["uid"],
            "workload_uid": tombstone["kueue"]["workload_uid"],
            "configmap_uid": tombstone["configmap"]["uid"],
        },
        "attempt_config": attempt,
        "runtime": {
            "bootstrap": "accepted_v2_smoke_dind_and_apt_docker",
            "cpu_only": True,
            "create_once": True,
            "fleet_team_id": runtime.TEAM,
            "preemption_policy": "Never",
            "priority_class": "fleet-serve-low",
            "secret_name": "chris-cyber-opencode-evals-v2",
            "secret_uid": runtime.SECRET_UID,
        },
        "prompts_traces_flags_or_scores_included": False,
        "credentials_included": False,
    }
    value = {**body, "spec_sha256": runtime.digest(body, "spec_sha256")}
    runtime.validate(value, root)
    if spec10["statistical_cell"]["cell_id"] != value["cell_id"]:
        raise ValueError("Generation-10 statistical cell drifted")
    return value


def configmap(root: Path, model: str) -> dict[str, Any]:
    row = runtime.EXPECTED[model]
    spec = derive_spec(root, model)
    data: dict[str, str] = {}
    for key, source in PAYLOAD_PATHS.items():
        path = root / source
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"unsafe package source: {source}")
        data[key] = path.read_text()
    data["spec.json"] = self_hosted.canonical_json(spec).decode() + "\n"
    data["g12-tombstone.json"] = (root / row["tombstone_path"]).read_text()
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": row["configmap_name"], "namespace": "fleet-train-jobs"},
        "data": data,
    }


def render(root: Path, model: str) -> dict[str, Any]:
    cm = configmap(root, model)
    job = yaml.safe_load((root / MANIFESTS[model]).read_text())
    row = runtime.EXPECTED[model]
    pod = job.get("spec", {}).get("template", {}).get("spec", {})
    if (
        job.get("metadata", {}).get("name") != row["job_name"]
        or pod.get("priorityClassName") != "fleet-serve-low"
        or pod.get("preemptionPolicy") != "Never"
    ):
        raise ValueError("Generation-13 held Job identity drifted")
    value = {"apiVersion": "v1", "kind": "List", "items": [cm, job]}
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    if len(json.dumps(cm).encode()) >= 900_000:
        raise ValueError("Generation-13 ConfigMap exceeds safety budget")
    return {
        "objects": value,
        "package_sha256": self_hosted.sha256(encoded),
        "configmap_json_bytes": len(json.dumps(cm).encode()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preview", "render"))
    parser.add_argument("--model", choices=tuple(runtime.EXPECTED), required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    built = render(args.repo.resolve(), args.model)
    if args.command == "preview":
        print(
            json.dumps(
                {
                    "status": "READY",
                    "launch_authorized": True,
                    "objects_created": False,
                    "model": args.model,
                    "package_sha256": built["package_sha256"],
                    "configmap_json_bytes": built["configmap_json_bytes"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.output is None or args.output.exists() or args.output.is_symlink():
        parser.error("render requires an unused --output")
    args.output.write_text(yaml.safe_dump(built["objects"], sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
