"""Fresh release identity after v2 omitted immutable config inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evals.fleet import hosted_glm_s1_r2_c2_release_v2 as prior
from evals.fleet import self_hosted

SCHEMA = "fleet-hosted-glm-s1-r2-c2-successor-release-v3"
JOB_NAME = "chris-glm53-exact100-hosted-s1-r002-c2-release-v3"
CONFIGMAP_NAME = JOB_NAME + "-run"
OUTPUT_PATH = Path("/mnt/sfs/jobs") / JOB_NAME / "RELEASE.json"


def build(root: Path) -> dict[str, Any]:
    value = prior.build(root)
    value["schema_version"] = SCHEMA
    value["failed_release_v2_job_uid"] = "947503cb-83da-44b6-9e9a-678d0d930581"
    value["failed_release_v2_pod_uid"] = "2ab4e015-fdb2-4141-958f-eb4c25e3b8c9"
    value["failed_release_v2_effects"] = {
        "claims": 0,
        "model_requests": 0,
        "task_instance_session_verifier_scoring_calls": 0,
    }
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    return value
