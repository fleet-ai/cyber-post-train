"""Generate suspended SFT/RL manifests; this command never submits workloads."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .cluster_policy import validate_training_manifest
from .io import atomic_write_text
from .job_factory import render_job_pair
from .model_adapter import load_model_adapter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    adapter = load_model_adapter(args.model_config)
    sft, rl = render_job_pair(adapter, args.run_id)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for stage, manifest in (("sft", sft), ("rl", rl)):
        validate_training_manifest(manifest)
        path = args.output_dir / f"{manifest['metadata']['name']}.yaml"
        atomic_write_text(path, yaml.safe_dump(manifest, sort_keys=False))
        print(f"prepared {stage}: {path} (suspended; not submitted)")


if __name__ == "__main__":
    main()
