"""Fresh prod9-only safeguards for the long-horizon SkyRL canary.

Legacy SkyRL source is itself part of immutable historical source closures.
This module deliberately leaves those files untouched: fresh prod9 plans must
name this module in their runtime closure before using its recorder, direct
Jobs API submission gate, or terminal acceptance gate.
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

from cyber_post_train.jobs import JobsError, digest

from . import skyrl_episode as legacy_episode
from .miles_conversion import _hash
from .rl_episode import EpisodeBudgetExceeded, InvalidEpisode
from .sft_runtime import _checked_file, write_receipt

CAPACITY_GATE_SCHEMA = "cyber_skyrl_prod9_direct_capacity_gate_v1"
ACCEPTANCE_SCHEMA = "cyber_skyrl_prod9_terminal_acceptance_v1"
EXACT_BINDING_SCHEMA = "cyber_jobs_api_exact_rayjob_binding_v1"
EXACT_OBSERVER_SCHEMA = "cyber_jobs_api_exact_uid_observer_result_v1"
PROJECT_OWNER_PREFIXES = ("chris-q38-",)
PROJECT_MAX_NODES = 8
PROJECT_MAX_GPUS = 64
CAPACITY_MAX_AGE_SECONDS = 120
PROD_CONTEXT = "nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6"
NAMESPACE = "fleet-train-jobs"
SOURCE_CLOSURE_SCHEMA = "cyber_skyrl_prod9_source_closure_check_v1"
CREATE_ONCE_ROOT = Path("/mnt/sfs/jobs/.cyber-post-train-prod9-create-once-v1")
CREATOR_BINDING_FILES = {
    "stage": "STAGE_OBSERVER_ARMED.json.created.json",
    "preflight": "PREFLIGHT_OBSERVER_ARMED.json.created.json",
    "training": "TRAINING_OBSERVER_ARMED.json.created.json",
    "reload": "RELOAD_OBSERVER_ARMED.json.created.json",
}
EXACT_PROD9_LIMITS = {
    "context_tokens": 262144,
    "response_tokens": 4194304,
    "max_tokens_per_turn": 32768,
    "generation_chunk_tokens": 4096,
    "compaction_trigger_tokens": 163840,
    "compaction_summary_tokens": 8192,
    "max_turns": 1200,
    "episode_seconds": 14400,
    "tool_seconds": 330,
    "tool_result_chars": 50000,
}


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
        or set(limits) not in (required, required | {"response_tokens"})
        or any(type(limits[key]) is not int or limits[key] <= 0 for key in limits)
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


def validate_exact_episode_limits(limits: object) -> None:
    """Require every exact prod9 horizon limit, including response tokens."""
    validate_episode_limits(limits)
    if limits != EXACT_PROD9_LIMITS:
        raise InvalidEpisode("skyrl_exact_horizon_contract_drift")


def _canonical_operation_root(scope: str, identity: dict[str, Any]) -> Path:
    """Derive one global journal root from immutable public identity bytes."""
    if scope not in {"stage", "training", "reload"} or not isinstance(identity, dict):
        raise ValueError("prod9 operation identity is invalid")
    return CREATE_ONCE_ROOT / f"{scope}-{digest({'scope': scope, 'identity': identity})}"


def stage_operation_root(stage: dict[str, Any]) -> Path:
    body = {key: item for key, item in stage.items() if key != "sha256"}
    if stage.get("schema") != "cyber_skyrl_prod9_rebind_stage_v1" or stage.get(
        "sha256"
    ) != "sha256:" + digest(body):
        raise ValueError("prod9 stage identity is not sealed")
    return _canonical_operation_root("stage", stage)


def training_operation_root(plan: dict[str, Any]) -> Path:
    if plan.get("schema") != "cyber_skyrl_prod9_training_v1":
        raise ValueError("prod9 training operation requires the fresh plan")
    return _canonical_operation_root("training", plan)


def reload_operation_root(spec: dict[str, Any]) -> Path:
    body = {key: item for key, item in spec.items() if key != "sha256"}
    if spec.get("schema") != "cyber_skyrl_prod9_reload_spec_v1" or spec.get(
        "sha256"
    ) != "sha256:" + digest(body):
        raise ValueError("prod9 reload identity is not sealed")
    return _canonical_operation_root("reload", spec)


def creator_binding_path(root: Path, purpose: str) -> Path:
    try:
        name = CREATOR_BINDING_FILES[purpose]
    except KeyError as exc:
        raise ValueError("prod9 creator-binding purpose is invalid") from exc
    return root / name


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
        "skyrl_training_compiler": (
            "training/skyrl_training.py",
            {
                "compile_rl",
                "job_request",
                "check_artifacts",
                "native_source",
                "dataset",
                "native_result",
            },
        ),
        "skyrl_config_runtime": (
            "training/skyrl.py",
            {"SkyRLConfig", "overrides", "native_config"},
        ),
        "skyrl_rollout_runtime": (
            "training/skyrl_rollout.py",
            {"Generator"},
        ),
        "reward_qualification": (
            "training/rl_reward_canary.py",
            {
                "validate_source_package",
                "source_proof",
                "validate_run_config",
                "validate_plan_binding",
            },
        ),
        "rl_runtime": (
            "training/rl_runtime.py",
            {"native_failure", "native_rejection", "run"},
        ),
        "data_preparation": (
            "training/rl_data.py",
            {"build"},
        ),
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
                "validate_exact_episode_limits",
                "capacity_gate",
                "accept_terminal",
            },
        ),
        "prod9_rollout_runtime": (
            "training/skyrl_prod9_rollout.py",
            {"Generator", "offline_token_safe_tool_probe", "runtime_binding"},
        ),
        "sft_runtime_support": (
            "training/sft_runtime.py",
            {"write_receipt", "_checked_file"},
        ),
        "dense_runtime": (
            "training/dense.py",
            {"encode_record"},
        ),
        "io_runtime": (
            "training/io.py",
            {"canonical_json", "digest_json"},
        ),
        "sft_compiler": (
            "training/sft.py",
            {"compile_sft", "job_request"},
        ),
        "model_binding": (
            "training/models.py",
            {"bound_model"},
        ),
        "corpus_builder": (
            "training/corpus.py",
            {"build"},
        ),
        "split_runtime": (
            "training/splits.py",
            {"assign_split"},
        ),
        "qwen_tool_parser": (
            "training/qwen_tools.py",
            {"parse_tool_calls"},
        ),
        "miles_conversion_support": (
            "training/miles_conversion.py",
            {"_hash"},
        ),
        "miles_config_support": (
            "training/miles.py",
            {"MilesConfig"},
        ),
        "fleet_binding": (
            "evals/fleet/opencode_self_hosted.py",
            {"bind_task", "verify_task", "assert_required_task_tools"},
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
        "prod9_reload_rail": (
            "training/skyrl_prod9_reload.py",
            {
                "build_spec",
                "job_request",
                "manifest",
                "validate_preview",
                "capacity_gate",
                "authorize",
                "create_once",
                "validate_source",
                "run_check",
            },
        ),
        "reload_checkpoint_support": (
            "training/checkpoints.py",
            {"receipt"},
        ),
        "reload_export_check_support": (
            "training/export_check.py",
            {"enable_native_patch", "model_contract", "synthetic_forward"},
        ),
        "reload_artifact_support": (
            "training/post_sft_artifacts.py",
            {"_safetensor_layout"},
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
            {
                "_validated_receipt",
                "Observer",
                "JobsApiPrefixGuard",
                "JobsApiExactUidObserver",
            },
        ),
        "gpu_capacity_census": (
            "cyber_post_train/gpu_capacity.py",
            {"build_capacity_census", "live_capacity_census"},
        ),
        "jobs_runtime": (
            "cyber_post_train/jobs.py",
            {"JobsError", "digest", "bundled_request", "Jobs"},
        ),
        "checkpoint_sealer": (
            "training/skyrl_posttrain.py",
            {"seal_checkpoint", "verify_manifest", "export_checkpoint"},
        ),
    }
    # The reviewed evidence must cover every byte that the prod9 bundle can
    # execute.  This lazy import is safe because source verification runs only
    # after module initialization, and it makes a future RUNTIME_FILES addition
    # fail closed until the immutable evidence is extended too.
    from . import skyrl_prod9_reload, skyrl_prod9_training

    required_paths = {relative for relative, _symbols in required.values()}
    if not (
        set(skyrl_prod9_training.RUNTIME_FILES) <= required_paths
        and set(skyrl_prod9_reload.RUNTIME_FILES) <= required_paths
    ):
        raise ValueError("prod9 source closure omits a bundled runtime file")
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
            "response_tokens",
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
    validate_exact_episode_limits(limits)
    return {
        "schema": SOURCE_CLOSURE_SCHEMA,
        "evidence_file_sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
        "evidence_self_sha256": value["sha256"],
        "sources": checked,
        "context_tokens": limits["context_tokens"],
        "response_tokens": limits["response_tokens"],
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
    identity: Any | None = None,
    reader: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Read and seal a fresh all-namespace proof before one Jobs API POST."""
    from cyber_post_train.gpu_capacity import ROLE_LABELS, CapacityError, live_capacity_census

    from . import skyrl_reward_rayjob as legacy_direct

    reader = live_capacity_census if reader is None else reader
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
    from . import skyrl_posttrain as legacy_posttrain

    if plan.get("schema") != "cyber_skyrl_prod9_training_v1":
        raise ValueError("terminal acceptance requires the fresh prod9 plan")
    args = legacy_posttrain._validate_plan(plan)
    root = Path(plan["output_root"])
    reload_root = root.with_name(root.name + f"-p{args.steps}-reload-v1")
    operation_root = training_operation_root(plan)
    return {
        "checkpoint_manifest": root / "checkpoint-seals-v1" / f"step-{args.steps}.json",
        "export": root / f"hf-export-step{args.steps}-v1" / "EXPORT.json",
        "training_observer": root / "TRAINING_OBSERVER_RESULT.json",
        "training_creator_binding": creator_binding_path(operation_root, "training"),
        "training_create_journal": operation_root / "PROD9_DIRECT_RAYJOB_CREATE.jsonl",
        "gpu_check": reload_root / "GPU_CHECK.json",
        "reload_observer": reload_root / "OBSERVER_RESULT.json",
        "accepted": root / "ACCEPTED.json",
    }


def _inspect_export(path: Path, sha256: str) -> tuple[dict, dict]:
    from . import skyrl_posttrain as legacy_posttrain
    from .checkpoints import receipt
    from .post_sft_artifacts import QWEN36_EXACT_MTP_OMISSION_KEYS, _safetensor_layout

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


def _exact_creator_binding(
    path: Path,
    *,
    run_name_prefix: str,
    run_dir: str,
    expected_gpus: int,
    maximum_seconds: int,
) -> tuple[dict, str]:
    from . import skyrl_posttrain as legacy_posttrain
    from . import skyrl_reward_rayjob as legacy_direct

    value = legacy_posttrain._json(path)
    body = {key: item for key, item in value.items() if key != "sha256"}
    try:
        UUID(value["jobs_api_run_id"])
        UUID(value["rayjob_uid"])
        created_at = datetime.fromisoformat(value["rayjob_created_at"].replace("Z", "+00:00"))
        bound_at = datetime.fromisoformat(value["bound_at"].replace("Z", "+00:00"))
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("Jobs API creator binding lacks exact IDs") from exc
    if (
        value.get("schema") != EXACT_BINDING_SCHEMA
        or value.get("sha256") != "sha256:" + digest(body)
        or value.get("status") != "bound_exact_uid_cleanup_not_started"
        or value.get("context") != legacy_direct.PROD_CONTEXT
        or value.get("namespace") != legacy_direct.NAMESPACE
        or re.fullmatch(re.escape(run_name_prefix) + r"-[a-f0-9]{8}", value.get("rayjob_name", ""))
        is None
        or value.get("jobs_api_run_name") != value.get("rayjob_name")
        or value.get("run_dir") != run_dir
        or value.get("failure_alerts") != "off"
        or value.get("maximum_seconds") != maximum_seconds
        or value.get("expected_gpus") != expected_gpus
        or value.get("cleanup_started") is not False
        or created_at.tzinfo is None
        or bound_at.tzinfo is None
        or bound_at < created_at
        or re.fullmatch(r"sha256:[0-9a-f]{64}", str(value.get("prefix_guard_sha256"))) is None
    ):
        raise ValueError("Jobs API creator binding differs from the exact prod9 run")
    return value, _hash(path)


def _create_journal(
    path: Path,
    *,
    created_schema: str,
    plan_sha256: str,
    request_sha256: str,
    creator: dict,
    expected_gpus: int,
) -> tuple[dict, str, dict]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("missing or indirect Jobs API create journal")
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    if any(
        getattr(before, field) != getattr(after, field)
        for field in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    ):
        raise ValueError("Jobs API create journal changed while being read")
    try:
        rows = [json.loads(line) for line in payload.splitlines()]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Jobs API create journal is not valid JSONL") from exc
    if len(rows) != 3 or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Jobs API create journal must contain intent, response, and CREATED")
    intent, response, created = rows
    created_body = {key: item for key, item in created.items() if key != "sha256"}
    capacity = intent.get("capacity_gate")
    preview = intent.get("live_preview_proof")
    capacity_body = (
        {key: item for key, item in capacity.items() if key != "sha256"}
        if isinstance(capacity, dict)
        else {}
    )
    preview_body = (
        {key: item for key, item in preview.items() if key != "sha256"}
        if isinstance(preview, dict)
        else {}
    )
    census = capacity.get("capacity_census") if isinstance(capacity, dict) else None
    census_body = (
        {key: item for key, item in census.items() if key != "sha256"}
        if isinstance(census, dict)
        else {}
    )
    current = census.get("current") if isinstance(census, dict) else None
    projected = census.get("projected") if isinstance(census, dict) else None
    try:
        UUID(response["job_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Jobs API create journal response lacks an exact ID") from exc

    def normalized_sha256(value: object) -> str:
        if not isinstance(value, str):
            return ""
        raw = value.removeprefix("sha256:")
        return "sha256:" + raw if re.fullmatch(r"[0-9a-f]{64}", raw) else ""

    if (
        intent.get("state") != "POST_INTENT_DO_NOT_RETRY"
        or intent.get("plan_sha256") != plan_sha256
        or intent.get("request_sha256") != request_sha256
        or re.fullmatch(r"sha256:[0-9a-f]{64}", str(intent.get("manifest_sha256"))) is None
        or re.fullmatch(r"sha256:[0-9a-f]{64}", str(intent.get("authorization_sha256"))) is None
        or re.fullmatch(r"sha256:[0-9a-f]{64}", str(intent.get("live_jobs_preview_sha256"))) is None
        or response.get("state") != "POST_RESPONSE"
        or response.get("name") != creator["jobs_api_run_name"]
        or response.get("job_id") != creator["jobs_api_run_id"]
        or response.get("run_dir") != creator["run_dir"]
        or created.get("schema") != created_schema
        or created.get("sha256") != "sha256:" + digest(created_body)
        or created.get("status") != "submitted_once_and_bound_exact_uid"
        or created.get("plan_sha256") != plan_sha256
        or created.get("request_sha256") != intent.get("request_sha256")
        or created.get("manifest_sha256") != intent.get("manifest_sha256")
        or created.get("authorization_sha256") != intent.get("authorization_sha256")
        or created.get("live_jobs_preview_sha256") != intent.get("live_jobs_preview_sha256")
        or created.get("jobs_api_run_name") != creator["jobs_api_run_name"]
        or created.get("jobs_api_run_id") != creator["jobs_api_run_id"]
        or created.get("rayjob_name") != creator["rayjob_name"]
        or created.get("rayjob_uid") != creator["rayjob_uid"]
        or created.get("creator_binding_sha256") != creator["sha256"]
        or created.get("created_at") != creator["rayjob_created_at"]
        or created.get("failure_alerts") != "off"
        or created.get("priority") != "c1"
        or created.get("queue_priority") != "q1"
        or created.get("nodes") != 1
        or created.get("gpus") != expected_gpus
        or not isinstance(capacity, dict)
        or capacity.get("sha256") != "sha256:" + digest(capacity_body)
        or capacity.get("status") != "passed"
        or capacity.get("context") != PROD_CONTEXT
        or normalized_sha256(capacity.get("request_sha256")) != request_sha256
        or normalized_sha256(capacity.get("manifest_sha256")) != intent.get("manifest_sha256")
        or (
            capacity.get("plan_sha256") is not None
            and normalized_sha256(capacity.get("plan_sha256")) != plan_sha256
        )
        or capacity.get("planned") != {"nodes": 1, "gpus": expected_gpus}
        or created.get("capacity_gate_sha256") != capacity.get("sha256")
        or not isinstance(census, dict)
        or census.get("sha256") != digest(census_body)
        or census.get("qualified") is not True
        or census.get("problems") != []
        or census.get("planned") != {"nodes": 1, "gpus": expected_gpus}
        or census.get("limits") != {"nodes": PROJECT_MAX_NODES, "gpus": PROJECT_MAX_GPUS}
        or not isinstance(current, dict)
        or not isinstance(projected, dict)
        or current.get("role_pod_counts", {}).get("unclassified") != 0
        or type(current.get("nodes")) is not int
        or type(current.get("gpus")) is not int
        or projected != {"nodes": current["nodes"] + 1, "gpus": current["gpus"] + expected_gpus}
        or projected["nodes"] > PROJECT_MAX_NODES
        or projected["gpus"] > PROJECT_MAX_GPUS
        or not isinstance(preview, dict)
        or preview.get("sha256") != "sha256:" + digest(preview_body)
        or preview.get("status") != "passed"
        or preview.get("context") != PROD_CONTEXT
        or normalized_sha256(preview.get("request_sha256")) != request_sha256
        or normalized_sha256(preview.get("manifest_sha256")) != intent.get("manifest_sha256")
        or preview.get("failure_alerts") != "off"
        or preview.get("priority") != "c1"
        or preview.get("queue_priority") != "q1"
        or preview.get("nodes") != 1
        or preview.get("gpus") != expected_gpus
        or created.get("live_preview_proof_sha256") != preview.get("sha256")
    ):
        raise ValueError("Jobs API create journal/CREATED/capacity binding is incomplete")
    return created, hashlib.sha256(payload).hexdigest(), capacity


def _exact_observer(
    path: Path,
    *,
    creator: dict,
    expected_gpus: int,
    expected_receipt: dict,
) -> tuple[dict, str]:
    from . import skyrl_posttrain as legacy_posttrain
    from . import skyrl_reward_rayjob as legacy_direct

    value = legacy_posttrain._json(path)
    body = {key: item for key, item in value.items() if key != "sha256"}
    children = {kind: value.get(kind) for kind in ("workloads", "rayclusters", "pods")}
    exit_codes = value.get("exit_codes")
    try:
        for rows in children.values():
            if not isinstance(rows, list) or len(rows) != 1:
                raise ValueError
            UUID(rows[0]["uid"])
            if not isinstance(rows[0].get("name"), str) or not rows[0]["name"]:
                raise ValueError
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("exact Jobs API observer lacks owned child identities") from exc
    if (
        value.get("schema") != EXACT_OBSERVER_SCHEMA
        or value.get("sha256") != "sha256:" + digest(body)
        or value.get("status") != "released_after_terminal"
        or value.get("reason") != "exact_root_and_observed_children_absent"
        or value.get("release_confirmed") is not True
        or value.get("context") != legacy_direct.PROD_CONTEXT
        or value.get("namespace") != legacy_direct.NAMESPACE
        or value.get("binding_sha256") != creator["sha256"]
        or value.get("jobs_api_run_name") != creator["jobs_api_run_name"]
        or value.get("jobs_api_run_id") != creator["jobs_api_run_id"]
        or value.get("rayjob_name") != creator["rayjob_name"]
        or value.get("rayjob_uid") != creator["rayjob_uid"]
        or value.get("created_at") != creator["rayjob_created_at"]
        or value.get("maximum_seconds") != creator["maximum_seconds"]
        or value.get("terminal_status") != "Succeeded"
        or value.get("owned_inventory_observed") is not True
        or value.get("raycluster_identity_observed") is not True
        or value.get("peak_gpus") != expected_gpus
        or value.get("active_gpus") != 0
        or value.get("restarts") != 0
        or not isinstance(exit_codes, list)
        or not exit_codes
        or any(type(code) is not int or code != 0 for code in exit_codes)
        or value.get("receipt") != expected_receipt
        or value.get("private_logs_read") is not False
    ):
        raise ValueError("exact Jobs API observer does not prove terminal release")
    return value, _hash(path)


def accept_terminal(
    plan: dict,
    *,
    checkpoint_manifest: Path,
    export: Path,
    training_observer: Path,
    training_creator_binding: Path,
    training_create_journal: Path,
    gpu_check: Path,
    reload_observer: Path,
    reload_creator_binding: Path,
    reload_create_journal: Path,
    output: Path,
) -> dict:
    """Create the sole prod9 completion marker after seal, reload, and release."""
    from . import skyrl_posttrain as legacy_posttrain
    from . import skyrl_prod9_direct, skyrl_prod9_reload, skyrl_prod9_training

    args = legacy_posttrain._validate_plan(plan)
    paths = terminal_paths(plan)
    supplied = {
        "checkpoint_manifest": checkpoint_manifest,
        "export": export,
        "training_observer": training_observer,
        "training_creator_binding": training_creator_binding,
        "training_create_journal": training_create_journal,
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
    reload_spec = skyrl_prod9_reload.build_spec(
        plan,
        manifest,
        export_receipt,
        checkpoint_manifest_file_sha256=manifest_file_sha256,
        export_file_sha256=export_file_sha256,
    )
    reload_operation = reload_operation_root(reload_spec)
    expected_reload_binding = creator_binding_path(reload_operation, "reload")
    expected_reload_journal = reload_operation / "PROD9_RELOAD_RAYJOB_CREATE.jsonl"
    if (
        reload_creator_binding != expected_reload_binding
        or reload_create_journal != expected_reload_journal
    ):
        raise ValueError("reload create evidence paths differ from the exact prod9 handoff")
    creator, creator_file_sha256 = _exact_creator_binding(
        training_creator_binding,
        run_name_prefix=plan["run_name"],
        run_dir=plan["output_root"],
        expected_gpus=8,
        maximum_seconds=skyrl_prod9_direct.MAXIMUM_SECONDS,
    )
    training_created, training_journal_file_sha256, training_capacity = _create_journal(
        training_create_journal,
        created_schema=skyrl_prod9_direct.CREATED_SCHEMA,
        plan_sha256="sha256:" + digest(plan),
        request_sha256="sha256:" + digest(skyrl_prod9_training.job_request(plan)),
        creator=creator,
        expected_gpus=8,
    )
    training_candidate = legacy_posttrain._json(training_observer)
    training_receipt = training_candidate.get("receipt")
    training_receipt_body = (
        {key: item for key, item in training_receipt.items() if key != "sha256"}
        if isinstance(training_receipt, dict)
        else {}
    )
    if (
        not isinstance(training_receipt, dict)
        or training_receipt.get("sha256") != digest(training_receipt_body)
        or training_receipt.get("status") != "native_loop_returned"
        or training_receipt.get("plan_sha256") != digest(plan)
        or training_receipt.get("checkpoint_global_step") != manifest["optimizer_step"]
        or training_receipt.get("completed_batches")
        != len(legacy_posttrain._expected_batches(args))
        or training_receipt.get("optimizer_update_independently_verified") is not False
        or training_receipt.get("checkpoint_reload_verified") is not False
        or manifest.get("terminal_receipt_sha256") != training_receipt["sha256"]
    ):
        raise ValueError("training observer termination receipt differs from the exact run")
    training_release, training_observer_file_sha256 = _exact_observer(
        training_observer,
        creator=creator,
        expected_gpus=8,
        expected_receipt=training_receipt,
    )
    gpu, gpu_file_sha256 = _receipt_file(gpu_check)
    if (
        gpu.get("schema") != "cyber_hf_export_check_v1"
        or gpu.get("status") != "passed"
        or gpu.get("export_sha256") != export_file_sha256
        or gpu.get("export_receipt_sha256") != export_receipt["receipt_sha256"]
        or gpu.get("checkpoint_manifest_file_sha256") != manifest_file_sha256
        or gpu.get("checkpoint_receipt_sha256") != manifest["receipt_sha256"]
        or gpu.get("source_plan_sha256") != digest(plan)
        or gpu.get("model_repo") != plan["model"]["repo"]
        or gpu.get("model_revision") != plan["model"]["revision"]
        or gpu.get("checker_sha256") != _hash(Path(skyrl_prod9_reload.__file__))
        or gpu.get("optimizer_steps_executed") != 0
        or gpu.get("gpus") != 1
        or gpu.get("gpu_reload_verified") is not True
        or gpu.get("source_unchanged") is not True
        or gpu.get("finite_logits") is not True
        or gpu.get("generated_tokens") != 2
        or gpu.get("serving_qualified") is not False
    ):
        raise ValueError("one-GPU BF16 reload proof is incomplete")
    reload_request = skyrl_prod9_reload.job_request(reload_spec)
    reload_creator, reload_creator_file_sha256 = _exact_creator_binding(
        reload_creator_binding,
        run_name_prefix=reload_request["name"],
        run_dir=reload_spec["run_dir"],
        expected_gpus=1,
        maximum_seconds=skyrl_prod9_reload.MAXIMUM_SECONDS,
    )
    reload_created, reload_journal_file_sha256, reload_capacity = _create_journal(
        reload_create_journal,
        created_schema=skyrl_prod9_reload.CREATED_SCHEMA,
        plan_sha256="sha256:" + digest(plan),
        request_sha256="sha256:" + digest(reload_request),
        creator=reload_creator,
        expected_gpus=1,
    )
    if reload_created.get("spec_sha256") != reload_spec["sha256"]:
        raise ValueError("reload CREATED receipt differs from the exact reload specification")
    observer, observer_file_sha256 = _exact_observer(
        reload_observer,
        creator=reload_creator,
        expected_gpus=1,
        expected_receipt=gpu,
    )
    if observer.get("receipt") != gpu:
        raise ValueError("reload observer is not bound to the exact GPU check receipt")
    result = {
        "schema": ACCEPTANCE_SCHEMA,
        "status": "accepted",
        "source_plan_sha256": digest(plan),
        "checkpoint_manifest_file_sha256": manifest_file_sha256,
        "checkpoint_manifest_receipt_sha256": manifest["receipt_sha256"],
        "export_file_sha256": export_file_sha256,
        "export_receipt_sha256": export_receipt["receipt_sha256"],
        "training_creator_binding_file_sha256": creator_file_sha256,
        "training_creator_binding_receipt_sha256": creator["sha256"],
        "training_create_journal_file_sha256": training_journal_file_sha256,
        "training_created_receipt_sha256": training_created["sha256"],
        "training_capacity_gate_sha256": training_capacity["sha256"],
        "training_observer_file_sha256": training_observer_file_sha256,
        "training_observer_receipt_sha256": training_release["sha256"],
        "training_manifest_sha256": training_created["manifest_sha256"],
        "training_rayjob_name": training_release["rayjob_name"],
        "training_rayjob_uid": training_release["rayjob_uid"],
        "training_workload_name": training_release["workloads"][0]["name"],
        "training_workload_uid": training_release["workloads"][0]["uid"],
        "training_raycluster_name": training_release["rayclusters"][0]["name"],
        "training_raycluster_uid": training_release["rayclusters"][0]["uid"],
        "training_pod_names": [row["name"] for row in training_release["pods"]],
        "training_pod_uids": [row["uid"] for row in training_release["pods"]],
        "gpu_check_file_sha256": gpu_file_sha256,
        "gpu_check_receipt_sha256": gpu["receipt_sha256"],
        "reload_checker_file_sha256": gpu["checker_sha256"],
        "reload_observer_file_sha256": observer_file_sha256,
        "reload_observer_receipt_sha256": observer["sha256"],
        "reload_creator_binding_file_sha256": reload_creator_file_sha256,
        "reload_creator_binding_receipt_sha256": reload_creator["sha256"],
        "reload_create_journal_file_sha256": reload_journal_file_sha256,
        "reload_created_receipt_sha256": reload_created["sha256"],
        "reload_capacity_gate_sha256": reload_capacity["sha256"],
        "reload_rayjob_uid": observer["rayjob_uid"],
        "reload_pod_uid": observer["pods"][0]["uid"],
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
