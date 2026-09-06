"""Create-once successor for the score-blind Qwen hosted release observer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from evals.fleet import qwen_hosted_whole_task_release_gate_package_v1 as prior

JOB_NAME = "chris-q38-hosted-r15-r16-release-gate-v2"
CONFIGMAP_NAME = JOB_NAME + "-package"
OUTPUT_ROOT = f"/mnt/sfs/jobs/{JOB_NAME}"


def render(root: Path) -> dict:
    """Render the same exact binding with a fresh object/output identity."""
    rendered = prior.render(root)
    raw = json.dumps(rendered)
    raw = raw.replace(prior.CONFIGMAP_NAME, CONFIGMAP_NAME)
    raw = raw.replace(prior.JOB_NAME, JOB_NAME)
    raw = raw.replace(prior.OUTPUT_ROOT, OUTPUT_ROOT)
    successor = json.loads(raw)
    job = successor["items"][1]
    job["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = (
        "q38-hosted-whole-task-release-gate-v2"
    )
    job["spec"]["template"]["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = (
        "q38-hosted-whole-task-release-gate-v2"
    )
    return successor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    payload = yaml.safe_dump(render(args.root.resolve(strict=True)), sort_keys=False)
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
