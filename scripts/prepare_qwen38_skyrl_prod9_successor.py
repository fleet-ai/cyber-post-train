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

from cyber_post_train.jobs import FAILURE_ALERT_OFF, digest
from training import sft, skyrl_prod9_direct, skyrl_prod9_hardening, skyrl_prod9_training
from training import skyrl_reward_rayjob as direct

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "configs/qualification/qwen38-rl-reward-canary-prod-v9.json"
DATA = ROOT / "configs/qualification/qwen38-rl-reward-canary-data-prod-v9.json"
IDENTITY = ROOT / "configs/qualification/qwen38-rl-reward-canary-prod9-identity-v1.json"
PREDECESSOR_MANIFEST = ROOT / "configs/qualification/qwen38-rl-reward-canary-manifest-prod-v8.json"
RECONCILIATION = ROOT / "docs/evidence/qwen38-study/2026-09-21-skyrl-prod8-reconciliation-v1.json"
RUNTIME_EVIDENCE = ROOT / "configs/data/qwen38-rl-reward-canary-exact-version-evidence-v10.json"


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


def _compile(
    run: dict,
    manifest: dict,
    *,
    relative_to: Path = RUN.parent,
) -> tuple[dict, dict]:
    """Use the real compiler while supplying only its public manifest surface."""
    original = sft.read_mapping

    def read(path: Path) -> dict:
        if Path(path) == Path(run["data"]["manifest"]):
            return copy.deepcopy(manifest)
        return original(path)

    with mock.patch.object(sft, "read_mapping", side_effect=read):
        plan = skyrl_prod9_training.compile_rl(run, relative_to=relative_to)
    return plan, skyrl_prod9_training.job_request(plan)


def _reviewed_config(path: Path) -> Path:
    """Limit successor selectors to reviewed, repository-owned JSON configs."""
    if path.is_symlink():
        raise ValueError("successor config must be a reviewed qualification JSON file")
    resolved = path.resolve()
    root = (ROOT / "configs/qualification").resolve()
    if resolved.parent != root or resolved.suffix != ".json" or not resolved.is_file():
        raise ValueError("successor config must be a reviewed qualification JSON file")
    return resolved


def build(
    manifest_path: Path,
    *,
    run_path: Path = RUN,
    data_path: Path = DATA,
    identity_path: Path = IDENTITY,
) -> dict:
    """Return a sealed, non-authorizing prod9 preparation receipt."""
    run_path = _reviewed_config(run_path)
    data_path = _reviewed_config(data_path)
    identity_path = _reviewed_config(identity_path)
    identity = direct.load_identity(identity_path)
    run, data, manifest = _load(run_path), _load(data_path), _load(manifest_path)
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

    hardening = skyrl_prod9_hardening.verify_source_closure(RUNTIME_EVIDENCE)
    plan, request = _compile(run, manifest, relative_to=run_path.parent)
    direct._identity_for_plan(plan, identity)
    historical_rail = skyrl_prod9_training.reject_historical_direct_rail(plan)
    arguments = plan["arguments"]
    qualified_limits = plan.get("qualification", {}).get("source_proof", {}).get("limits")
    try:
        skyrl_prod9_hardening.validate_exact_episode_limits(qualified_limits)
    except (ValueError, RuntimeError) as exc:
        raise ValueError("prod9 exact horizon contract changed") from exc
    if (
        plan.get("schema") != skyrl_prod9_training.SCHEMA
        or plan.get("prod9_runtime") != skyrl_prod9_training._binding()
        or request.get("workers") != 1
        or request.get("gpus_per_worker") != 8
        or request.get("priority_class") != "c1"
        or request.get("failureAlerts") is not False
        or request.get("image") != plan["execution"]["image"]
        or qualified_limits != skyrl_prod9_hardening.EXACT_PROD9_LIMITS
        or arguments.get("context_tokens") != qualified_limits["context_tokens"]
        or arguments.get("response_tokens") != qualified_limits["response_tokens"]
        or arguments.get("tokens_per_turn") != qualified_limits["max_tokens_per_turn"]
        or arguments.get("generation_chunk_tokens") != qualified_limits["generation_chunk_tokens"]
        or arguments.get("compaction_trigger_tokens")
        != qualified_limits["compaction_trigger_tokens"]
        or arguments.get("compaction_summary_tokens")
        != qualified_limits["compaction_summary_tokens"]
        or arguments.get("max_turns") != qualified_limits["max_turns"]
        or arguments.get("compaction_enabled") is not True
        or hardening.get("response_tokens")
        != skyrl_prod9_hardening.EXACT_PROD9_LIMITS["response_tokens"]
    ):
        raise ValueError("prod9 one-node, compact, or fresh-runtime contract changed")
    return _seal(
        {
            "schema": "cyber_qwen38_skyrl_prod9_offline_preparation_v1",
            "status": "prepared_not_authorized",
            "external_mutations": 0,
            "private_rows_read": False,
            "launch_authorized": False,
            "identity": identity.sealed_mapping(),
            "inputs": {
                "run": _source(run_path),
                "data": _source(data_path),
                "identity": _source(identity_path),
                "reconciliation": _source(RECONCILIATION),
                "rebound_manifest": {
                    "path": str(manifest_path),
                    "sha256": manifest["sha256"],
                },
                "prod9_hardening": hardening,
            },
            "runtime_sha256": "sha256:" + plan["runtime_sha256"],
            "plan_sha256": "sha256:" + digest(plan),
            "request_sha256": "sha256:" + digest(request),
            "fresh_rebind_stage": {
                "module": "training.skyrl_prod9_training",
                "function": "stage_rebind",
                "schema": "cyber_skyrl_prod9_rebind_stage_receipt_v1",
                "root_alert_annotation_required": FAILURE_ALERT_OFF,
                "bundle_module": skyrl_prod9_training.MODULE,
            },
            "fresh_cpu_preflight": {
                "module": "training.skyrl_prod9_direct",
                "function": "preflight_job_manifest",
                "schema": "cyber_skyrl_prod9_training_cpu_preflight_v1",
                "root_alert_annotation_required": FAILURE_ALERT_OFF,
                "bundle_module": skyrl_prod9_training.MODULE,
                "live_create_available": skyrl_prod9_direct.live_create_is_available(),
            },
            "historical_direct_rail": historical_rail,
            "next_live_gates": [
                "fresh dev and prod server previews proving the root alert opt-out",
                "one root-annotated zero-GPU SFS rebind create, receipt, and exact-UID release",
                "one root-annotated zero-GPU exact-image preflight create, "
                "receipt, and exact-UID release",
                "fresh Jobs-API, Kubernetes, output-root, and W&B identity absence checks",
                "fresh one-node/eight-GPU direct RayJob previews plus all-namespace capacity proof",
                "one reviewed no-retry GPU create with the pre-armed exact-UID observer",
            ],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--identity", type=Path, default=IDENTITY)
    args = parser.parse_args()
    value = build(
        args.manifest,
        run_path=args.run,
        data_path=args.data,
        identity_path=args.identity,
    )
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
