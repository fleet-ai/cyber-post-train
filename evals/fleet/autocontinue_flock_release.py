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
TERMINAL_SCHEMA = "fleet-opencode-autocontinue-flock-preflight-terminal-v1"
TERMINAL_RECEIPT_SHA256 = (
    "sha256:cf9a955fcd35111b39ea3ccbcd8987e541cdd5628d0251e81828c0640e8d050d"
)
TERMINAL_FILE_SHA256 = (
    "sha256:a5d1fb641ad0d16cdff37a279ffec41e2e1578f63adfee3889a2e77ace900b8d"
)
HOLDER_JOB_UID = "87894cba-0f94-4e28-a1f9-9e21bdd748ec"
HOLDER_POD_UID = "6316321c-08f7-4cc1-8e02-c3d899513cdc"
PROBER_JOB_UID = "fc9f51d6-03f6-4520-8819-cbfb61b93796"
PROBER_POD_UID = "5cdd6532-d1c6-4bde-bdc7-067f39924c97"
RELEASE_GATE_SCHEMA = "fleet-opencode-autocontinue-flock-release-gate-v1"
SEALED_TERMINAL_EVIDENCE_SHA256 = (
    "sha256:1975ac508b8ddab44b552dbdce630a0fd2dfb89531ceb77f8bdb386d2dca59c9"
)
SEALED_TERMINAL_EVIDENCE_PATH = (
    "docs/evidence/qwen38-study/"
    "2026-09-04-opencode-autocontinue-flock-preflight-terminal-v1.json"
)


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


def validate_terminal_evidence(evidence: dict[str, Any]) -> None:
    authorization = evidence.get("authorization")
    implementation = evidence.get("implementation")
    holder = evidence.get("holder")
    prober = evidence.get("prober")
    terminal = evidence.get("terminal")
    classification = evidence.get("classification")
    configmap = evidence.get("configmap")
    if (
        evidence.get("schema_version") != TERMINAL_SCHEMA
        or evidence.get("receipt_sha256")
        != self_hosted.digest_without(evidence, "receipt_sha256")
        or evidence.get("launch_commit")
        != "bf885b4642d0ab26db8ee1cd88d46bfd60544925"
        or not isinstance(authorization, dict)
        or authorization.get("release_receipt_sha256")
        != "sha256:e7a5fbb2ede2f972d6893af70884d50682a9b48d046cca4f4e7bc2fb6f8f24f0"
        or authorization.get("release_file_sha256")
        != "sha256:ed63b88f818a8dcd1a4d50c0cfdb54ce9e071c78334ca9311ff85bc0cbd5f23c"
        or authorization.get("campaign_sha256") != CAMPAIGN_SHA256
        or authorization.get("campaign_file_sha256") != CAMPAIGN_FILE_SHA256
        or authorization.get("preflight_launch_authorized") is not True
        or authorization.get("scored_launch_authorized") is not False
        or not isinstance(implementation, dict)
        or implementation.get("manifest_sha256") != MANIFEST_SHA256
        or implementation.get("source_sha256") != SOURCE_SHA256
        or implementation.get("submit_script_sha256")
        != "sha256:2f38635e55578a2ea340d54d272080957c1ed0b395be7f5358b876e048e3d2c0"
        or implementation.get("creation_order") != EXPECTED_JOBS
        or implementation.get("server_dry_run_passed") is not True
        or implementation.get("create_once") is not True
        or implementation.get("required_priority_class") != "fleet-train-high"
        or implementation.get("backoff_limit") != 0
        or not isinstance(configmap, dict)
        or configmap.get("name") != "chris-opencode11827-ac-flock-preflight-v1"
        or configmap.get("uid") != "9e253557-f650-4ae6-abd1-48e4d6d3b3c7"
        or configmap.get("immutable") is not True
        or not isinstance(holder, dict)
        or holder.get("job_name") != EXPECTED_JOBS[0]
        or holder.get("job_uid") != HOLDER_JOB_UID
        or holder.get("pod_name")
        != "chris-opencode11827-ac-flock-holder-v1-dzssd"
        or holder.get("pod_uid") != HOLDER_POD_UID
        or holder.get("job_succeeded") != 1
        or holder.get("job_failed") != 0
        or holder.get("pod_phase") != "Succeeded"
        or holder.get("pod_restart_count") != 0
        or holder.get("exit_code") != 0
        or not isinstance(prober, dict)
        or prober.get("job_name") != EXPECTED_JOBS[1]
        or prober.get("job_uid") != PROBER_JOB_UID
        or prober.get("pod_name")
        != "chris-opencode11827-ac-flock-prober-v1-prthr"
        or prober.get("pod_uid") != PROBER_POD_UID
        or prober.get("job_succeeded") != 1
        or prober.get("job_failed") != 0
        or prober.get("pod_phase") != "Succeeded"
        or prober.get("pod_restart_count") != 0
        or prober.get("exit_code") != 0
        or not isinstance(terminal, dict)
        or terminal.get("schema_version")
        != "fleet-endpoint-lease-cross-pod-preflight-v1"
        or terminal.get("phase") != "cross_pod_flock_passed"
        or terminal.get("terminal_file_sha256") != TERMINAL_FILE_SHA256
        or terminal.get("terminal_receipt_sha256") != TERMINAL_RECEIPT_SHA256
        or terminal.get("holder_ready_sha256")
        != "sha256:d85df0291e52e181c3298b0a0e7067c1ce26e5718e1414603fed878e640b60e5"
        or terminal.get("contention_receipt_sha256")
        != "sha256:e691f0960a0e7acdf9b4e5eebc0092d6575f8355f3cde53396b10ea690de2ac4"
        or terminal.get("holder_released_sha256")
        != "sha256:acfff8ad5182ad5bc72d8ff38e3f27caac3131ff057487942a8313863d3bb938"
        or terminal.get("uid_bindings_match_kubernetes") is not True
        or terminal.get("contention_excluded_second_pod") is not True
        or terminal.get("slot_reacquired_after_release") is not True
        or terminal.get("digest_valid") is not True
        or not isinstance(classification, dict)
        or classification.get("shared_pvc_cross_pod_flock_preflight_passed")
        is not True
        or classification.get("endpoint_lease_capacity") != 2
        or classification.get("model_or_fleet_api_calls") != 0
        or classification.get("scored_sessions_created") != 0
        or classification.get("campaign_canary_launch_authorized") is not False
        or classification.get("campaign_bulk_launch_authorized") is not False
        or evidence.get("privacy")
        != {
            "credentials_included": False,
            "prompts_or_traces_included": False,
            "scores_included": False,
            "task_content_included": False,
        }
    ):
        raise ValueError("flock preflight terminal evidence drifted")


def validate_release_gate(gate_evidence: dict[str, Any]) -> None:
    campaign = gate_evidence.get("campaign")
    terminal = gate_evidence.get("terminal_evidence")
    uids = gate_evidence.get("uid_bindings")
    gate = gate_evidence.get("gate")
    if (
        gate_evidence.get("schema_version") != RELEASE_GATE_SCHEMA
        or gate_evidence.get("append_only") is not True
        or gate_evidence.get("receipt_sha256")
        != self_hosted.digest_without(gate_evidence, "receipt_sha256")
        or not isinstance(campaign, dict)
        or campaign.get("path")
        != "evals/fleet/configs/q38-glm53-opencode-autocontinue-primary-campaign-v1.json"
        or campaign.get("campaign_sha256") != CAMPAIGN_SHA256
        or campaign.get("file_sha256") != CAMPAIGN_FILE_SHA256
        or campaign.get("immutable_bytes_unchanged") is not True
        or campaign.get("canary_launch_authorized") is not False
        or campaign.get("bulk_launch_authorized") is not False
        or not isinstance(terminal, dict)
        or terminal.get("path") != SEALED_TERMINAL_EVIDENCE_PATH
        or terminal.get("receipt_sha256") != SEALED_TERMINAL_EVIDENCE_SHA256
        or terminal.get("file_sha256")
        != "sha256:0d6497dad8e3ff49fc095d903e3ab2afeaec1928f35b7eb8b097521b9e20b485"
        or terminal.get("live_terminal_receipt_sha256") != TERMINAL_RECEIPT_SHA256
        or terminal.get("live_terminal_file_sha256") != TERMINAL_FILE_SHA256
        or uids
        != {
            "configmap_uid": "9e253557-f650-4ae6-abd1-48e4d6d3b3c7",
            "holder_job_uid": HOLDER_JOB_UID,
            "holder_pod_uid": HOLDER_POD_UID,
            "prober_job_uid": PROBER_JOB_UID,
            "prober_pod_uid": PROBER_POD_UID,
        }
        or gate
        != {
            "shared_pvc_cross_pod_flock_preflight_passed": True,
            "contention_excluded_second_pod": True,
            "slot_reacquired_after_release": True,
            "endpoint_lease_capacity": 2,
            "model_or_fleet_api_calls": 0,
            "scored_sessions_created": 0,
            "permits_scored_launch": False,
            "requires_separate_canary_authorization": True,
            "requires_separate_bulk_authorization": True,
        }
        or gate_evidence.get("privacy")
        != {
            "credentials_included": False,
            "prompts_or_traces_included": False,
            "scores_included": False,
            "task_content_included": False,
        }
    ):
        raise ValueError("flock release gate evidence drifted")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, required=True)
    args = parser.parse_args()
    validate_release(load_object(args.release))
    print(json.dumps({"ok": True, "scored_launch_authorized": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
