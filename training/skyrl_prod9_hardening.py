"""Fresh prod9-only safeguards for the long-horizon SkyRL canary.

Legacy SkyRL source is itself part of immutable historical source closures.
This module deliberately leaves those files untouched: fresh prod9 plans must
name this module in their runtime closure before using its recorder, direct
create rail, or terminal acceptance gate.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from cyber_post_train.gpu_capacity import ROLE_LABELS, CapacityError, live_capacity_census
from cyber_post_train.jobs import JobsError, digest

from . import skyrl_episode as legacy_episode
from . import skyrl_posttrain as legacy_posttrain
from . import skyrl_reward_rayjob as legacy_direct
from .checkpoints import receipt
from .miles_conversion import _hash
from .post_sft_artifacts import QWEN36_EXACT_MTP_OMISSION_KEYS, _safetensor_layout
from .rl_episode import EpisodeBudgetExceeded, InvalidEpisode
from .sft_runtime import _checked_file, write_receipt

CAPACITY_GATE_SCHEMA = "cyber_skyrl_prod9_direct_capacity_gate_v1"
ACCEPTANCE_SCHEMA = "cyber_skyrl_prod9_terminal_acceptance_v1"
RELOAD_OBSERVER_SCHEMA = "cyber_skyrl_prod9_reload_observer_result_v1"
PROJECT_OWNER_PREFIXES = ("chris-q38-",)
PROJECT_MAX_NODES = 8
PROJECT_MAX_GPUS = 64
CAPACITY_MAX_AGE_SECONDS = 120
PROD_CONTEXT = legacy_direct.PROD_CONTEXT
NAMESPACE = legacy_direct.NAMESPACE
SOURCE_CLOSURE_SCHEMA = "cyber_skyrl_prod9_source_closure_check_v1"


def validate_episode_limits(limits: object) -> None:
    """Validate prod9 limits without treating characters as model tokens."""
    required = {
        "context_tokens",
        "max_tokens_per_turn",
        "generation_chunk_tokens",
        "compaction_trigger_tokens",
        "compaction_summary_tokens",
        "max_turns",
        "episode_seconds",
        "tool_seconds",
        "tool_result_chars",
    }
    if (
        not isinstance(limits, dict)
        or set(limits) != required
        or any(type(limits[key]) is not int or limits[key] <= 0 for key in required)
    ):
        raise InvalidEpisode("skyrl_compaction_contract_drift")
    if not (
        limits["generation_chunk_tokens"] <= limits["max_tokens_per_turn"]
        and limits["compaction_summary_tokens"] <= limits["max_tokens_per_turn"]
        and limits["compaction_trigger_tokens"]
        + limits["max_tokens_per_turn"]
        + limits["compaction_summary_tokens"]
        < limits["context_tokens"]
    ):
        raise InvalidEpisode("invalid_compaction_limits")


def verify_source_closure(path: Path) -> dict[str, Any]:
    """Verify the public, fresh prod9 source closure without reading task data.

    Historical prod8 receipts deliberately remain frozen.  A prod9 preparation
    must instead bind the current source bytes it will require for its separate
    direct-create, token-budget, and terminal-acceptance gates.  This checks
    only repository source and a sanitized JSON evidence file; it never opens
    private rollout data, an SFS path, credentials, or an external service.
    """
    if path.is_symlink() or not path.is_file():
        raise ValueError("prod9 source closure is missing or indirect")
    try:
        value = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("prod9 source closure is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("prod9 source closure is not one JSON object")
    body = {key: item for key, item in value.items() if key != "sha256"}
    if value.get("schema") != "cyber_rl_exact_version_evidence_v1" or value.get(
        "sha256"
    ) != "sha256:" + digest(body):
        raise ValueError("prod9 source closure digest/schema changed")
    authority = value.get("tool_surface_authority")
    local_code = authority.get("local_code") if isinstance(authority, dict) else None
    if not isinstance(local_code, dict):
        raise ValueError("prod9 source closure lacks local source bindings")
    required = {
        "episode_runtime": ("training/rl_episode.py", {"collect", "_agent"}),
        "skyrl_episode_runtime": (
            "training/skyrl_episode.py",
            {"Recorder", "offline_long_horizon_probe"},
        ),
        "prod9_hardening": (
            "training/skyrl_prod9_hardening.py",
            {
                "Recorder",
                "validate_episode_limits",
                "capacity_gate",
                "accept_terminal",
            },
        ),
        "prod9_rollout_runtime": (
            "training/skyrl_prod9_rollout.py",
            {"Generator", "offline_token_safe_tool_probe", "runtime_binding"},
        ),
        "prod9_training_runtime": (
            "training/skyrl_prod9_training.py",
            {
                "compile_rl",
                "job_request",
                "preflight_request",
                "preflight",
                "stage_spec",
                "stage_request",
                "stage_rebind",
                "reject_historical_direct_rail",
                "_native",
            },
        ),
        "prod9_direct_rail": (
            "training/skyrl_prod9_direct.py",
            {
                "manifest",
                "stage_job_manifest",
                "preflight_job_manifest",
                "server_dry_run",
                "validate_cpu_preview",
                "validate_preview",
                "authorize_stage",
                "authorize_preflight",
                "authorize",
                "create_stage_once",
                "create_preflight_once",
                "create_once",
                "live_create_is_available",
            },
        ),
        "direct_rail": (
            "training/skyrl_reward_rayjob.py",
            {
                "manifest",
                "authorize",
                "duplicate_checks",
                "validate_preview",
                "rebind_private_source_for_identity",
            },
        ),
        "prod9_cleanup_observer": (
            "training/dev_cleanup_observer.py",
            {"_validated_receipt", "Observer"},
        ),
        "checkpoint_sealer": (
            "training/skyrl_posttrain.py",
            {"seal_checkpoint", "verify_manifest", "export_checkpoint"},
        ),
    }
    root = Path(__file__).resolve().parents[1]
    checked: dict[str, str] = {}
    for name, (relative, symbols) in required.items():
        binding = local_code.get(name)
        source = root / relative
        if (
            not isinstance(binding, dict)
            or binding.get("path") != "../../" + relative
            or set(binding.get("symbols", [])) != symbols
            or not source.is_file()
            or source.is_symlink()
        ):
            raise ValueError("prod9 source closure binding changed")
        raw = source.read_bytes()
        if binding.get("file_sha256") != "sha256:" + hashlib.sha256(raw).hexdigest():
            raise ValueError("prod9 source closure file digest changed")
        try:
            tree = ast.parse(raw, filename=relative)
        except SyntaxError as exc:
            raise ValueError("prod9 source closure source is unparsable") from exc
        declared = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if not symbols <= declared:
            raise ValueError("prod9 source closure symbol changed")
        checked[name] = binding["file_sha256"]
    horizon = value.get("episode_horizon", {})
    fixed = horizon.get("fixed_limits") if isinstance(horizon, dict) else None
    if not isinstance(fixed, dict):
        raise ValueError("prod9 source closure horizon is missing")
    limits = {
        key: (horizon.get("max_turns") if key == "max_turns" else fixed.get(key))
        for key in {
            "context_tokens",
            "max_tokens_per_turn",
            "generation_chunk_tokens",
            "compaction_trigger_tokens",
            "compaction_summary_tokens",
            "max_turns",
            "episode_seconds",
            "tool_seconds",
            "tool_result_chars",
        }
    }
    validate_episode_limits(limits)
    if limits["context_tokens"] != 262144 or limits["max_turns"] != 1200:
        raise ValueError("prod9 source closure long-context contract changed")
    return {
        "schema": SOURCE_CLOSURE_SCHEMA,
        "evidence_file_sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
        "evidence_self_sha256": value["sha256"],
        "sources": checked,
        "context_tokens": limits["context_tokens"],
        "max_turns": limits["max_turns"],
        "tool_result_token_safe": True,
    }


class Recorder(legacy_episode.Recorder):
    """The prod9 recorder adds exact-token prospective tool-result checks."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        validate_episode_limits(self.config["rl"])

    def append_observation(self, message, images):
        if images or self.messages is None or self.recording is not None or self.finalized:
            raise InvalidEpisode("unsupported_observation")
        self._validate_tool_observations([message])
        return message

    def append_observations(self, messages, images):
        if (
            images
            or self.messages is None
            or self.recording is not None
            or self.finalized
            or not isinstance(messages, list)
            or len(messages) < 2
        ):
            raise InvalidEpisode("unsupported_observation_group")
        self._validate_tool_observations(messages)
        return messages

    def _validate_tool_observations(self, observations: object) -> None:
        """Use tokenizer output and the full next template, never characters."""
        if not isinstance(observations, list) or not observations:
            raise InvalidEpisode("unsupported_observation")
        limits = self.config["rl"]
        for observation in observations:
            if (
                not isinstance(observation, dict)
                or observation.get("role") != "tool"
                or not isinstance(observation.get("content"), str)
                or len(observation["content"]) > limits["tool_result_chars"]
            ):
                raise InvalidEpisode("tool_result_exceeds_budget")
            try:
                tokens = list(
                    self.tokenizer.encode(observation["content"], add_special_tokens=False)
                )
            except Exception as exc:
                raise InvalidEpisode("tool_result_exceeds_budget") from exc
            if any(type(token) is not int or token < 0 for token in tokens):
                raise InvalidEpisode("tool_result_exceeds_budget")

        prospective = [*self.messages, *observations]
        prompt = self._render(prospective, self.tools)
        action_total = len(prompt) + limits["max_tokens_per_turn"]
        if action_total > limits["compaction_trigger_tokens"]:
            summary_prompt = self._render(
                [
                    *prospective,
                    {"role": "user", "content": legacy_episode.COMPACTION_PROMPT},
                ],
                [],
            )
            if len(summary_prompt) + limits["compaction_summary_tokens"] > limits["context_tokens"]:
                raise EpisodeBudgetExceeded("generation_incomplete_context_full")


def _require_root_alert_annotation(expected: dict[str, Any]) -> None:
    metadata = expected.get("metadata") if isinstance(expected, dict) else None
    annotations = metadata.get("annotations") if isinstance(metadata, dict) else None
    if not isinstance(annotations, dict) or annotations.get("fleet.ai/failure-alerts") != "off":
        raise JobsError("prod9 direct RayJob root failure-alert annotation is absent")


def capacity_gate(
    plan: dict[str, Any],
    request: dict[str, Any],
    expected: dict[str, Any],
    *,
    identity: legacy_direct.RailIdentity | None = None,
    reader: Callable[..., dict[str, Any]] = live_capacity_census,
) -> dict[str, Any]:
    """Read and seal a fresh all-namespace proof before one direct create."""
    bound = legacy_direct._identity_for_plan(plan, identity)
    _require_root_alert_annotation(expected)
    try:
        cluster = expected["spec"]["rayClusterSpec"]
        containers = cluster["headGroupSpec"]["template"]["spec"]["containers"]
        resources = containers[0]["resources"]
        direct_gpus = (
            resources["requests"]["nvidia.com/gpu"],
            resources["limits"]["nvidia.com/gpu"],
        )
    except (KeyError, TypeError, IndexError) as exc:
        raise JobsError("prod9 direct GPU capacity binding changed") from exc
    if (
        request.get("workers") != 1
        or request.get("gpus_per_worker") != 8
        or request.get("failureAlerts") is not False
        or cluster.get("workerGroupSpecs") not in (None, [])
        or len(containers) != 1
        or direct_gpus != (8, 8)
    ):
        raise JobsError("prod9 direct GPU capacity binding changed")
    try:
        census = reader(
            legacy_direct.PROD_CONTEXT,
            owner_prefixes=PROJECT_OWNER_PREFIXES,
            max_nodes=PROJECT_MAX_NODES,
            max_gpus=PROJECT_MAX_GPUS,
            planned_nodes=1,
            planned_gpus=8,
        )
    except CapacityError as exc:
        raise JobsError("prod9 direct cross-namespace GPU capacity census failed") from exc
    if not isinstance(census, dict):
        raise JobsError("prod9 direct cross-namespace GPU capacity census is invalid")
    try:
        observed_at = legacy_direct._timestamp(census.get("observed_at"))
    except JobsError as exc:
        raise JobsError("prod9 direct GPU capacity observation is invalid") from exc
    unsigned = {key: value for key, value in census.items() if key != "sha256"}
    age = (datetime.now(UTC) - observed_at).total_seconds()
    current, projected, scope = (
        census.get("current"),
        census.get("projected"),
        census.get("scope"),
    )
    if (
        census.get("sha256") != digest(unsigned)
        or census.get("schema") != "cyber_project_gpu_capacity_census_v1"
        or scope
        != {
            "kubernetes_namespaces": "all",
            "owner_prefixes": list(PROJECT_OWNER_PREFIXES),
            "ownership_labels": ROLE_LABELS,
        }
        or census.get("limits") != {"nodes": PROJECT_MAX_NODES, "gpus": PROJECT_MAX_GPUS}
        or census.get("planned") != {"nodes": 1, "gpus": 8}
        or census.get("qualified") is not True
        or census.get("problems") != []
        or not isinstance(current, dict)
        or not isinstance(projected, dict)
        or current.get("role_pod_counts", {}).get("unclassified") != 0
        or type(current.get("nodes")) is not int
        or type(current.get("gpus")) is not int
        or projected != {"nodes": current["nodes"] + 1, "gpus": current["gpus"] + 8}
        or projected["nodes"] > PROJECT_MAX_NODES
        or projected["gpus"] > PROJECT_MAX_GPUS
        or not 0 <= age <= CAPACITY_MAX_AGE_SECONDS
    ):
        raise JobsError("prod9 direct GPU capacity proof is stale or incomplete")
    return legacy_direct._seal(
        {
            "schema": CAPACITY_GATE_SCHEMA,
            "status": "passed",
            "context": legacy_direct.PROD_CONTEXT,
            "identity_sha256": bound.sealed_mapping()["sha256"],
            "plan_sha256": digest(plan),
            "request_sha256": digest(request),
            "manifest_sha256": digest(expected),
            "planned": {"nodes": 1, "gpus": 8},
            "observed_at": census["observed_at"],
            "capacity_census": census,
        }
    )


def _receipt_file(path: Path) -> tuple[dict, str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("missing or indirect acceptance receipt")
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    if any(
        getattr(before, field) != getattr(after, field)
        for field in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    ):
        raise ValueError("acceptance receipt changed while being read")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("acceptance receipt is not valid JSON") from exc
    if not isinstance(value, dict) or value.get("receipt_sha256") != digest(
        {key: item for key, item in value.items() if key != "receipt_sha256"}
    ):
        raise ValueError("acceptance receipt digest mismatch")
    return value, hashlib.sha256(payload).hexdigest()


def terminal_paths(plan: dict) -> dict[str, Path]:
    args = legacy_posttrain._validate_plan(plan)
    root = Path(plan["output_root"])
    reload_root = root.with_name(root.name + f"-p{args.steps}-reload-v1")
    return {
        "checkpoint_manifest": root / "checkpoint-seals-v1" / f"step-{args.steps}.json",
        "export": root / f"hf-export-step{args.steps}-v1" / "EXPORT.json",
        "gpu_check": reload_root / "GPU_CHECK.json",
        "reload_observer": reload_root / "OBSERVER_RESULT.json",
        "accepted": root / "ACCEPTED.json",
    }


def _inspect_export(path: Path, sha256: str) -> tuple[dict, dict]:
    _checked_file(path, sha256)
    proof = receipt(path)
    root = path.parent
    if (
        path.name != "EXPORT.json"
        or root.is_symlink()
        or proof.get("schema") != legacy_posttrain.EXPORT_SCHEMA
        or proof.get("model_repo") != "Qwen/Qwen3.8-27B"
        or proof.get("output_root") != str(root)
        or proof.get("dtype") != "BF16"
        or proof.get("optimizer_steps_executed") != 0
        or proof.get("all_output_tensors_reopened_equal") is not True
        or proof.get("source_inventory_sizes_mtimes_unchanged") is not True
        or set(proof.get("restored_base_tensors", [])) != set(QWEN36_EXACT_MTP_OMISSION_KEYS)
    ):
        raise ValueError("unsupported or incomplete prod9 export receipt")
    if {item.name for item in root.iterdir()} != set(proof["files"]) | {path.name}:
        raise ValueError("export inventory differs from receipt")
    for name, spec in proof["files"].items():
        item = root / name
        _checked_file(item, spec["sha256"])
        if item.stat().st_size != spec["bytes"]:
            raise ValueError("export payload size mismatch")
    layout, _ = _safetensor_layout(root)
    if len(layout) != proof["trained_tensors"] + len(proof["restored_base_tensors"]) or any(
        spec["dtype"] != "BF16" for spec in layout.values()
    ):
        raise ValueError("export tensor count/dtype mismatch")
    return proof, layout


def _reload_observer(path: Path, *, plan: dict, expected_name: str) -> tuple[dict, str]:
    value = legacy_posttrain._json(path)
    body = {key: item for key, item in value.items() if key != "sha256"}
    if value.get("sha256") != "sha256:" + digest(body):
        raise ValueError("reload observer receipt digest mismatch")
    absent = {
        "target_present",
        "pods_present",
        "rayjob_present",
        "workload_present",
        "raycluster_present",
    }
    try:
        UUID(value["uid"])
        UUID(value["rayjob_uid"])
        UUID(value["workload_uid"])
        UUID(value["raycluster_uid"])
        pod_uids, pod_names = value["pod_uids"], value["pod_names"]
        if (
            not isinstance(pod_uids, list)
            or not isinstance(pod_names, list)
            or len(pod_uids) != 1
            or len(pod_names) != 1
            or not isinstance(pod_names[0], str)
            or not pod_names[0]
            or not all(
                isinstance(value.get(key), str) and value[key]
                for key in ("workload_name", "raycluster_name")
            )
            or re.fullmatch(r"sha256:[0-9a-f]{64}", str(value.get("manifest_sha256"))) is None
        ):
            raise ValueError
        UUID(pod_uids[0])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("reload observer lacks exact workload identities") from exc
    if (
        value.get("schema") != RELOAD_OBSERVER_SCHEMA
        or value.get("status") != "released"
        or value.get("context") != legacy_direct.PROD_CONTEXT
        or value.get("namespace") != legacy_direct.NAMESPACE
        or value.get("kind") != "rayjob"
        or value.get("name") != expected_name
        or value.get("rayjob_name") != expected_name
        or value.get("uid") != value.get("rayjob_uid")
        or value.get("plan_sha256") != "sha256:" + digest(plan)
        or value.get("expected_gpus") != 1
        or value.get("peak_gpus") != 1
        or value.get("active_gpus") != 0
        or value.get("terminal_status") != "Succeeded"
        or value.get("restarts") != 0
        or value.get("observer_error_class") != ""
        or value.get("exit_codes") != [0]
        or not isinstance(value.get("release_observed_at"), str)
        or any(value.get(key) is not False for key in absent)
    ):
        raise ValueError("reload observer does not prove successful one-GPU release")
    return value, _hash(path)


def accept_terminal(
    plan: dict,
    *,
    checkpoint_manifest: Path,
    export: Path,
    gpu_check: Path,
    reload_observer: Path,
    output: Path,
) -> dict:
    """Create the sole prod9 completion marker after seal, reload, and release."""
    args = legacy_posttrain._validate_plan(plan)
    paths = terminal_paths(plan)
    supplied = {
        "checkpoint_manifest": checkpoint_manifest,
        "export": export,
        "gpu_check": gpu_check,
        "reload_observer": reload_observer,
        "accepted": output,
    }
    if any(Path(value) != paths[name] for name, value in supplied.items()):
        raise ValueError("terminal acceptance paths differ from the exact prod9 handoff")
    if output.exists() or output.is_symlink():
        raise FileExistsError("prod9 terminal acceptance already exists")
    manifest, manifest_file_sha256 = _receipt_file(checkpoint_manifest)
    legacy_posttrain.verify_manifest(manifest)
    if (
        manifest.get("source_plan_sha256") != digest(plan)
        or manifest.get("optimizer_step") != args.steps
        or manifest.get("optimizer_update_verified") is not True
        or manifest.get("source_inputs_unchanged") is not True
    ):
        raise ValueError("terminal checkpoint seal differs from the exact run")
    export_file_sha256 = _hash(export)
    export_receipt, _ = _inspect_export(export, export_file_sha256)
    if (
        export_receipt.get("source_checkpoint_receipt_sha256") != manifest["receipt_sha256"]
        or export_receipt.get("source_manifest_file_sha256") != manifest_file_sha256
        or export_receipt.get("source_plan_sha256") != digest(plan)
        or export_receipt.get("optimizer_step") != args.steps
        or export_receipt.get("optimizer_steps_executed") != 0
        or export_receipt.get("gpu_reload_verified") is not False
        or export_receipt.get("output_root") != str(export.parent)
    ):
        raise ValueError("BF16 export differs from the exact terminal checkpoint")
    gpu, gpu_file_sha256 = _receipt_file(gpu_check)
    if (
        gpu.get("schema") != "cyber_hf_export_check_v1"
        or gpu.get("status") != "passed"
        or gpu.get("export_sha256") != export_file_sha256
        or gpu.get("export_receipt_sha256") != export_receipt["receipt_sha256"]
        or gpu.get("optimizer_steps_executed") != 0
        or gpu.get("gpus") != 1
        or gpu.get("gpu_reload_verified") is not True
        or gpu.get("source_unchanged") is not True
        or gpu.get("finite_logits") is not True
        or gpu.get("generated_tokens") != 2
        or gpu.get("serving_qualified") is not False
    ):
        raise ValueError("one-GPU BF16 reload proof is incomplete")
    observer, observer_file_sha256 = _reload_observer(
        reload_observer,
        plan=plan,
        expected_name=plan["run_name"] + f"-p{args.steps}-reload-v1",
    )
    result = {
        "schema": ACCEPTANCE_SCHEMA,
        "status": "accepted",
        "source_plan_sha256": digest(plan),
        "checkpoint_manifest_file_sha256": manifest_file_sha256,
        "checkpoint_manifest_receipt_sha256": manifest["receipt_sha256"],
        "export_file_sha256": export_file_sha256,
        "export_receipt_sha256": export_receipt["receipt_sha256"],
        "gpu_check_file_sha256": gpu_file_sha256,
        "gpu_check_receipt_sha256": gpu["receipt_sha256"],
        "reload_observer_file_sha256": observer_file_sha256,
        "reload_observer_receipt_sha256": observer["sha256"],
        "reload_rayjob_uid": observer["rayjob_uid"],
        "reload_pod_uid": observer["pod_uids"][0],
        "optimizer_step": args.steps,
        "optimizer_updates_verified": True,
        "terminal_checkpoint_sealed": True,
        "complete_bf16_reload_verified": True,
        "gpu_resources_released": True,
        "private_payloads_included": False,
        "serving_qualified": False,
    }
    write_receipt(output, result)
    accepted, _ = _receipt_file(output)
    return accepted
