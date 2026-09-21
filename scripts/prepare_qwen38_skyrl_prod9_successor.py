#!/usr/bin/env python3
"""Check a freshly rebound prod9 package before any Fleet operation.

This command is deliberately offline and non-authorizing.  It accepts only a
sanitized local manifest from the restricted rebind step; it never reads task
rows, contacts Fleet/W&B/Kubernetes, stages data, or creates a workload.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from unittest import mock

from cyber_post_train.jobs import FAILURE_ALERT_ANNOTATION, FAILURE_ALERT_OFF, digest
from training import sft, skyrl_training
from training import skyrl_reward_rayjob as direct

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "configs/qualification/qwen38-rl-reward-canary-prod-v9.json"
DATA = ROOT / "configs/qualification/qwen38-rl-reward-canary-data-prod-v9.json"
IDENTITY = ROOT / "configs/qualification/qwen38-rl-reward-canary-prod9-identity-v1.json"
PREDECESSOR_MANIFEST = ROOT / "configs/qualification/qwen38-rl-reward-canary-manifest-prod-v8.json"
RECONCILIATION = ROOT / "docs/evidence/qwen38-study/2026-09-21-skyrl-prod8-reconciliation-v1.json"


def _load(path: Path) -> dict:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path} is not one JSON object") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not one JSON object")
    return value


def _source(path: Path) -> dict[str, str]:
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _seal(value: dict) -> dict:
    body = {key: item for key, item in value.items() if key != "sha256"}
    return {**body, "sha256": "sha256:" + digest(body)}


def _compile(run: dict, manifest: dict) -> tuple[dict, dict]:
    """Use the real compiler while supplying only its public manifest surface."""
    original = sft.read_mapping

    def read(path: Path) -> dict:
        if Path(path) == Path(run["data"]["manifest"]):
            return copy.deepcopy(manifest)
        return original(path)

    with mock.patch.object(sft, "read_mapping", side_effect=read):
        plan = skyrl_training.compile_rl(run, relative_to=RUN.parent)
    return plan, skyrl_training.job_request(plan)


def build(manifest_path: Path) -> dict:
    """Return a sealed, non-authorizing prod9 preparation receipt."""
    identity = direct.load_identity(IDENTITY)
    run, data, manifest = _load(RUN), _load(DATA), _load(manifest_path)
    predecessor, reconciliation = _load(PREDECESSOR_MANIFEST), _load(RECONCILIATION)
    manifest_body = {key: value for key, value in manifest.items() if key != "sha256"}
    expected_files = {"train", "dev"}
    if (
        reconciliation.get("classification") != "unknown"
        or reconciliation.get("policy", {}).get("prod8_resume_permitted") is not False
        or reconciliation.get("policy", {}).get("prod8_output_or_checkpoint_reuse_permitted")
        is not False
        or run.get("name") != identity.run_name
        or run.get("output_root") != identity.output_root
        or run.get("wandb", {}).get("run_id") != identity.wandb_run_id
        or run.get("data", {}).get("root") != identity.data_root
        or run.get("data", {}).get("manifest") != identity.data_root + "/manifest.json"
        or data.get("name") != identity.run_name
        or data.get("output") != identity.data_root
        or manifest.get("schema") != "cyber_skyrl_data_v1"
        or manifest.get("name") != identity.run_name
        or manifest.get("sha256") != "sha256:" + digest(manifest_body)
        or manifest.get("limits") != data.get("limits")
        or set(manifest.get("files", {})) != expected_files
        or {key: value.get("rows") for key, value in manifest["files"].items()}
        != {"train": 1, "dev": 1}
        or any(
            manifest.get(key) != predecessor.get(key)
            for key in ("selection_sha256", "split_sha256", "tool_catalog_sha256", "tokenizer")
        )
    ):
        raise ValueError("prod9 identity, reconciliation, or public data contract changed")

    plan, request = _compile(run, manifest)
    direct._identity_for_plan(plan, identity)
    preflight = direct.preflight_job_manifest(plan, identity=identity)
    arguments = plan["arguments"]
    if (
        request.get("workers") != 1
        or request.get("gpus_per_worker") != 8
        or request.get("priority_class") != "c1"
        or request.get("failureAlerts") is not False
        or request.get("image") != plan["execution"]["image"]
        or {
            key: arguments.get(key)
            for key in (
                "context_tokens",
                "generation_chunk_tokens",
                "compaction_trigger_tokens",
                "compaction_summary_tokens",
                "max_turns",
            )
        }
        != {
            "context_tokens": 262144,
            "generation_chunk_tokens": 4096,
            "compaction_trigger_tokens": 163840,
            "compaction_summary_tokens": 8192,
            "max_turns": 1200,
        }
        or preflight["metadata"]["name"] != identity.preflight_name
        or preflight["metadata"]["annotations"].get(FAILURE_ALERT_ANNOTATION) != FAILURE_ALERT_OFF
    ):
        raise ValueError("prod9 one-node, compact, or failure-alert contract changed")
    return _seal(
        {
            "schema": "cyber_qwen38_skyrl_prod9_offline_preparation_v1",
            "status": "prepared_not_authorized",
            "external_mutations": 0,
            "private_rows_read": False,
            "launch_authorized": False,
            "identity": identity.sealed_mapping(),
            "inputs": {
                "run": _source(RUN),
                "data": _source(DATA),
                "identity": _source(IDENTITY),
                "reconciliation": _source(RECONCILIATION),
                "rebound_manifest": {
                    "path": str(manifest_path),
                    "sha256": manifest["sha256"],
                },
            },
            "plan_sha256": "sha256:" + digest(plan),
            "request_sha256": "sha256:" + digest(request),
            "preflight_manifest_sha256": "sha256:" + digest(preflight),
            "next_live_gates": [
                "fresh absence checks for the prod9 names and destinations",
                "root-annotation server previews for stage, preflight, and RayJob",
                "create-once data stage and accepted zero-GPU release evidence",
                "exact-image CPU preflight and accepted zero-GPU release evidence",
                "arm the exact UID-bound cleanup observer before one GPU create",
            ],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    value = build(args.manifest)
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
