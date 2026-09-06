"""Render the actual-evidence, no-live-work dedicated runtime gate observer."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

from evals.fleet import glm53_dedicated_v14_scored_canary_package_v1 as controller
from evals.fleet import hosted_glm_exact_canary_package_v1 as base

JOB = "chris-glm53-dedicated-v18-r051-runtime-gate-v2"


def render(root: Path, parity: Path, binding: Path, release: Path, origin: str) -> dict:
    source = copy.deepcopy(controller.render(root)["objects"]["items"][0])
    source["metadata"]["name"] = JOB + "-run"
    source["data"]["gate.py"] = (
        root / "evals/fleet/glm53_dedicated_runtime_gate_observer_v1.py"
    ).read_text()
    source["data"]["run.sh"] = (
        root / "evals/fleet/scripts/run_glm53_dedicated_runtime_gate_observer_v1.sh"
    ).read_text()
    source["data"]["parity.json"] = parity.read_text()
    source["data"]["binding.json"] = binding.read_text()
    source["data"]["release.json"] = release.read_text()
    job = copy.deepcopy(base.render(root)["objects"]["items"][1])
    job["metadata"]["name"] = JOB
    job["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = JOB
    job["metadata"]["annotations"]["cyber-post-train.fleet.ai/launch-authorized"] = "false"
    pod = job["spec"]["template"]
    pod["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = JOB
    pod["spec"]["volumes"][0]["configMap"]["name"] = JOB + "-run"
    evaluator = pod["spec"]["containers"][0]
    evaluator["env"].append({"name": "DEDICATED_SERVICE_ORIGIN", "value": origin})
    return {"apiVersion": "v1", "kind": "List", "items": [source, job]}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parity", type=Path, required=True)
    p.add_argument("--binding", type=Path, required=True)
    p.add_argument("--release", type=Path, required=True)
    p.add_argument("--service-origin", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or a.output.is_symlink():
        p.error("output must be unused")
    value = render(Path.cwd(), a.parity, a.binding, a.release, a.service_origin)
    a.output.write_text(yaml.safe_dump(value, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
