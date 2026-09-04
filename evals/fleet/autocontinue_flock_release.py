"""Validate the one-shot, non-scored shared-PVC flock preflight release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evals.fleet import self_hosted

SCHEMA = "fleet-opencode-autocontinue-flock-preflight-release-v1"
CAMPAIGN_SHA256 = "sha256:63946f224a33eb0d2c2a6fdba34358ebf9ca379e5f0f156cd137c95a9b98097e"
CAMPAIGN_FILE_SHA256 = "sha256:1dfcc6d9ddbd19ef5ff46b5d6308d62526db4d1a0ccf05c5fa3b10b176910f34"
MANIFEST_SHA256 = "sha256:d9fa9ed9b1c144977d4132e1a4b2929a1037400b7aec3fab62e047c0d825a7ce"
SOURCE_SHA256 = "sha256:dc012e6cdaf34f8297d48d43a483c1af2b2099f637403fc93d9ea45946d18088"
EXPECTED_JOBS = [
    "chris-opencode11827-ac-flock-holder-v1",
    "chris-opencode11827-ac-flock-prober-v1",
]


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def validate_release(release: dict[str, Any]) -> None:
    if (
        release.get("schema_version") != SCHEMA
        or release.get("receipt_sha256")
        != self_hosted.digest_without(release, "receipt_sha256")
        or release.get("append_only") is not True
        or release.get("authorized_by") != "/root"
        or release.get("campaign_sha256") != CAMPAIGN_SHA256
        or release.get("campaign_file_sha256") != CAMPAIGN_FILE_SHA256
        or release.get("manifest_sha256") != MANIFEST_SHA256
        or release.get("source_sha256") != SOURCE_SHA256
        or release.get("configmap_name")
        != "chris-opencode11827-ac-flock-preflight-v1"
        or release.get("job_creation_order") != EXPECTED_JOBS
        or release.get("sfs_root")
        != "/mnt/sfs/endpoint-leases/preflight-opencode11827-autocontinue-v1"
        or release.get("preflight_launch_authorized") is not True
        or release.get("scored_launch_authorized") is not False
        or release.get("model_or_fleet_api_calls") != 0
        or release.get("required_priority_class") != "fleet-train-high"
        or release.get("create_once") is not True
        or release.get("require_configmap_jobs_pods_and_sfs_absent") is not True
        or release.get("require_server_dry_run") is not True
        or release.get("cluster_objects_created_at_seal") is not False
        or release.get("terminal_receipt_sha256_at_seal") is not None
        or release.get("privacy")
        != {
            "credentials_included": False,
            "prompts_or_traces_included": False,
            "scores_included": False,
        }
    ):
        raise ValueError("flock preflight release drifted")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, required=True)
    args = parser.parse_args()
    validate_release(load_object(args.release))
    print(json.dumps({"ok": True, "scored_launch_authorized": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
