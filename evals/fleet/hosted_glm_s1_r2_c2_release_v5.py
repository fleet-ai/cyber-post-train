"""Release the engine-complete fresh rank-2 v4 controller."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evals.fleet import hosted_glm_s1_r2_c2_release_v2 as prior
from evals.fleet import hosted_glm_s1_r2_c2_successor_v4 as successor
from evals.fleet import self_hosted

SCHEMA = "fleet-hosted-glm-s1-r2-c2-successor-release-v5"
JOB_NAME = "chris-glm53-exact100-hosted-s1-r002-c2-release-v5"
CONFIGMAP_NAME = JOB_NAME + "-run"
OUTPUT_PATH = Path("/mnt/sfs/jobs") / JOB_NAME / "RELEASE.json"


def build(root: Path) -> dict[str, Any]:
    value = prior.build(root, target=successor)
    value["schema_version"] = SCHEMA
    value["failed_controller_v3_job_uid"] = "77fd6dec-b06a-4e57-9e0a-0b48f0f51bd9"
    value["failed_controller_v3_pod_uid"] = "c99a34db-861b-45a6-b3c9-0d23842741fa"
    value["failed_controller_v3_effects"] = {"claims": 0, "model_requests": 0, "task_instance_session_verifier_scoring_calls": 0}
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    return value
