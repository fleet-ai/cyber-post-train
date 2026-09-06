"""Fresh score-blind release for GLM v22 rank51 attempt2."""

from pathlib import Path

from evals.fleet import glm53_dedicated_v19_a2_release_v1 as prior
from evals.fleet import glm53_dedicated_v22_a2_runtime_v1 as runtime
from evals.fleet import glm53_dedicated_v22_a2_v1 as canary

JOB_NAME = "chris-glm53-dedicated-v22-r051-a2-release-v1"
CONFIGMAP_NAME = JOB_NAME + "-run"
OUTPUT = Path("/mnt/sfs/jobs") / JOB_NAME / "RELEASE.json"
prior.canary = canary
prior.runtime = runtime
prior.JOB_NAME = JOB_NAME
prior.CONFIGMAP_NAME = CONFIGMAP_NAME
prior.OUTPUT = OUTPUT


def build(root: Path, package_path: Path):
    value = prior.build(root, package_path)
    value["schema_version"] = runtime.RELEASE_SCHEMA
    value["receipt_sha256"] = prior.self_hosted.digest_without(value, "receipt_sha256")
    return value
