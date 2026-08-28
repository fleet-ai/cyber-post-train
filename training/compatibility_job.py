"""Render the queued, self-contained GLM-5.2 B300 preflight Job."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import re
import zipfile
from pathlib import Path
from typing import Any

import yaml

from .io import atomic_write_text

IMAGE_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
JOB_NAME = "chris-cyber-glm52-preflight-b4734de4"
MODEL_ROOT = "/mnt/sfs/models/zai-org/GLM-5.2/b4734de4facf877f85769a911abafc5283eab3d9"
RESULT_ROOT = "/mnt/sfs/cyber-post-train/compatibility/b4734de4"
NEMO_RL_COMMIT = "63e620046c67f922c4a57dcb65d7e6fceb60f5d4"
IMAGE_REPOSITORY = "ghcr.io/fleet-ai/cyber-post-train-nemo-rl-glm52"


def probe_archive() -> tuple[str, str]:
    package_root = Path(__file__).resolve().parent
    files = {
        "training/__init__.py": package_root / "__init__.py",
        "training/compatibility_probe.py": package_root / "compatibility_probe.py",
        "training/io.py": package_root / "io.py",
    }
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(files.items()):
            archive.writestr(name, path.read_bytes())
    payload = stream.getvalue()
    return base64.b64encode(payload).decode(), hashlib.sha256(payload).hexdigest()


def render_preflight_job(image_digest: str) -> dict[str, Any]:
    if not IMAGE_DIGEST.fullmatch(image_digest):
        raise ValueError("image digest must be an immutable sha256 digest")
    archive, archive_sha256 = probe_archive()
    command = f"""
install -d -m 0755 {RESULT_ROOT}
printf '%s' "$PROBE_ARCHIVE_B64" | base64 -d > /tmp/cyber-post-train-probe.zip
printf '%s  %s\n' "$PROBE_ARCHIVE_SHA256" /tmp/cyber-post-train-probe.zip | sha256sum -c -
export PYTHONPATH=/tmp/cyber-post-train-probe.zip
python -m training.compatibility_probe static \\
  --model-root {MODEL_ROOT} \\
  --output {RESULT_ROOT}/environment.json
torchrun --standalone --nproc-per-node=8 -m training.compatibility_probe collective \\
  --output {RESULT_ROOT}/nccl.json
echo PREFLIGHT_COMPLETE {RESULT_ROOT}
""".strip()
    labels = {
        "kueue.x-k8s.io/queue-name": "training-lq",
        "cyber-post-train.fleet.ai/owner": "chris",
        "cyber-post-train.fleet.ai/experiment": "preflight-b4734de4",
    }
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": JOB_NAME,
            "namespace": "fleet-train-jobs",
            "labels": labels,
            "annotations": {
                "cyber-post-train.fleet.ai/probe-archive-sha256": archive_sha256,
                "cyber-post-train.fleet.ai/training-image-digest": image_digest,
            },
        },
        "spec": {
            "suspend": True,
            "backoffLimit": 0,
            "activeDeadlineSeconds": 21600,
            "ttlSecondsAfterFinished": 604800,
            "template": {
                "metadata": {"labels": {key: value for key, value in labels.items() if "/" in key}},
                "spec": {
                    "restartPolicy": "Never",
                    "nodeSelector": {"workload": "fleetai-training-ng-gpu"},
                    "tolerations": [
                        {
                            "key": "workload",
                            "operator": "Equal",
                            "value": "fleetai-training-ng-gpu",
                            "effect": "NoSchedule",
                        }
                    ],
                    "containers": [
                        {
                            "name": "preflight",
                            "image": f"{IMAGE_REPOSITORY}@{image_digest}",
                            "imagePullPolicy": "IfNotPresent",
                            "command": ["/bin/bash", "-ceu", "--"],
                            "args": [command],
                            "env": [
                                {"name": "PROBE_ARCHIVE_B64", "value": archive},
                                {"name": "PROBE_ARCHIVE_SHA256", "value": archive_sha256},
                                {"name": "TRAINING_IMAGE_DIGEST", "value": image_digest},
                                {"name": "NEMO_RL_COMMIT", "value": NEMO_RL_COMMIT},
                                {"name": "HF_HUB_OFFLINE", "value": "1"},
                                {"name": "TRANSFORMERS_OFFLINE", "value": "1"},
                            ],
                            "resources": {
                                "requests": {
                                    "cpu": "96",
                                    "memory": "1200Gi",
                                    "nvidia.com/gpu": "8",
                                },
                                "limits": {
                                    "cpu": "192",
                                    "memory": "2Ti",
                                    "nvidia.com/gpu": "8",
                                },
                            },
                            "volumeMounts": [{"name": "sfs", "mountPath": "/mnt/sfs"}],
                        }
                    ],
                    "volumes": [
                        {"name": "sfs", "persistentVolumeClaim": {"claimName": "sfs-shared"}}
                    ],
                },
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = render_preflight_job(args.image_digest)
    atomic_write_text(args.output, yaml.safe_dump(manifest, sort_keys=False))
    print(args.output)


if __name__ == "__main__":
    main()
