"""Release the config-complete fresh rank-2 v3 controller."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evals.fleet import hosted_glm_s1_r2_c2_release_v2 as prior
from evals.fleet import hosted_glm_s1_r2_c2_successor_v3 as successor
from evals.fleet import self_hosted

SCHEMA = "fleet-hosted-glm-s1-r2-c2-successor-release-v4"
JOB_NAME = "chris-glm53-exact100-hosted-s1-r002-c2-release-v4"
CONFIGMAP_NAME = JOB_NAME + "-run"
OUTPUT_PATH = Path("/mnt/sfs/jobs") / JOB_NAME / "RELEASE.json"
FAILED_V1_JOB_UID = prior.FAILED_V1_JOB_UID
FAILED_V1_POD_UID = prior.FAILED_V1_POD_UID


def build(root: Path) -> dict[str, Any]:
    value = prior.build(root, target=successor)
    value["schema_version"] = SCHEMA
    value["failed_controller_v2_job_uid"] = "6d8f7e0f-dc23-4cc8-8a84-997208559735"
    value["failed_controller_v2_pod_uid"] = "a9069be2-9886-4a7b-a70c-d9c349d26647"
    value["failed_controller_v2_effects"] = {"claims": 0, "model_requests": 0, "task_instance_session_verifier_scoring_calls": 0}
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    return value
