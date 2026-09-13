"""Fail-closed scientific acceptance for a one-update Miles reward canary.

The native driver deliberately emits only operational completion.  This module
reopens the immutable plan, batches, private episode receipts, checkpoint seal,
scalar-only W&B observation, and UID-bound controller/release observations
before creating one sanitized acceptance receipt.  Reward values and task
content are inspected only to validate the private evidence and are never
copied into the acceptance receipt.
"""

from __future__ import annotations

import copy
import json
import math
import numbers
import re
import uuid
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from cyber_post_train.jobs import API_URLS, digest
from evals.fleet import opencode_self_hosted as fleet

from . import miles
from .miles_conversion import _hash, _write
from .miles_reload import CHECKPOINT_SCHEMA, _verify_checkpoint
from .miles_training import SCHEMA as TRAINING_SCHEMA
from .miles_training import job_request
from .rl_episode import _validate as validate_episode_config
from .rl_runtime import sealed

TERMINAL_SCHEMA = "cyber_miles_reward_canary_terminal_v1"
EPISODE_AUDIT_SCHEMA = "cyber_miles_reward_canary_episode_audit_v1"
WANDB_SCHEMA = "cyber_miles_wandb_scalar_observation_v1"
CONTROLLER_SCHEMA = "cyber_miles_controller_terminal_observation_v1"
RELEASE_SCHEMA = "cyber_miles_external_release_v1"
NAMESPACE = "fleet-train-jobs"
NAMESPACE_UID = "10394b76-e1d4-40b1-a8e2-7575e95df216"
WORLD_SIZE = 8

_SHA = re.compile(r"(?:sha256:)?[a-f0-9]{64}")
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}")
_EPISODE_FILES = frozenset(
    {
        "binding.json",
        "create-intent.json",
        "instance.json",
        "conversation.json",
        "score-intent.json",
        "reward.json",
        "cleanup.json",
        "recording.json",
    }
)
_REFERENCE_FIELDS = frozenset({"path", "file_sha256", "receipt_sha256"})
_EPISODE_AUDIT_FIELDS = frozenset(
    {
        "schema",
        "source_plan_sha256",
        "accepted_episode_count",
        "train_episode_count",
        "dev_episode_count",
        "infrastructure_invalid_count",
        "truncated_count",
        "unique_authoritative_verifier_execution_count",
        "authoritative_verifier_execution_ids_sha256",
        "authoritative_verifier_bindings_sha256",
        "train_nonzero_reward_present",
        "train_within_group_reward_variance_present",
        "reward_values_included",
        "assistant_message_count",
        "tool_message_count",
        "report_submitted_count",
        "task_content_included",
    }
)
_UPDATE_PROOF_FIELDS = frozenset(
    {"optimizer_updates", "basis", "checkpoint_payload_changed_from_base", "counter_only_claim"}
)
_TERMINAL_FIELDS = frozenset(
    {
        "schema",
        "status",
        "source_run_name",
        "source_plan_sha256",
        "source_request_sha256",
        "native_completion",
        "episode_audit",
        "optimizer_update_proof",
        "wandb_scalar_observation",
        "checkpoint_manifest",
        "controller_observation",
        "external_release",
        "external_gpu_release_verified",
        "reward_values_included",
        "task_content_included",
        "production_promotion_requires_reload_acceptance",
        "sha256",
    }
)


def _unsigned(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "sha256"}


def _read(path: Path, *, schema: str | None = None) -> tuple[dict[str, Any], str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("acceptance input must be a regular file")
    before = path.stat()
    payload = path.read_bytes()
    if path.stat() != before:
        raise ValueError("acceptance input changed while reading")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("acceptance input is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("acceptance input must be a JSON object")
    if schema is None:
        if value.get("sha256", "").removeprefix("sha256:") != digest(_unsigned(value)):
            raise ValueError("acceptance input self-digest mismatch")
    else:
        sealed(value, schema)
    return value, _hash(path)


def _reference(path: Path, value: Mapping[str, Any], file_sha256: str) -> dict[str, str]:
    return {
        "path": str(path),
        "file_sha256": "sha256:" + file_sha256.removeprefix("sha256:"),
        "receipt_sha256": "sha256:" + str(value["sha256"]).removeprefix("sha256:"),
    }


def _uuid(value: object, label: str) -> str:
    try:
        result = str(uuid.UUID(str(value)))
    except ValueError as error:
        raise ValueError(f"{label} is not a UUID") from error
    if result == "00000000-0000-0000-0000-000000000000":
        raise ValueError(f"{label} is a zero UUID")
    return result


def _time(value: object, label: str) -> float:
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        result = float(value)
    elif isinstance(value, str):
        try:
            result = datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError as error:
            raise ValueError(f"{label} is not an ISO timestamp") from error
    else:
        raise ValueError(f"{label} is not a timestamp")
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} is not a positive finite timestamp")
    return result


def _image_digest(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("runtime image ID is absent")
    match = re.fullmatch(
        r"(?:containerd|docker-pullable)://(?:[^@\s]+@)?sha256:([a-f0-9]{64})",
        value,
    )
    if match is None:
        raise ValueError("runtime image ID is not immutable")
    return match.group(1)


def _canary(plan: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    request = job_request(plan)
    args = plan.get("arguments", {})
    if (
        plan.get("schema") != TRAINING_SCHEMA
        or args.get("nodes") != 1
        or args.get("gpus_per_node") != WORLD_SIZE
        or args.get("steps") != 1
        or args.get("groups") != 1
        or args.get("samples_per_prompt") != WORLD_SIZE
        or args.get("eval_interval") != 1
        or args.get("checkpoint_interval") != 1
        or plan.get("execution", {}).get("image") != miles.IMAGE
        or plan["execution"].get("priority") != "c1"
        or request.get("workers") != 1
        or request.get("gpus_per_worker") != WORLD_SIZE
        or request.get("priority_class") != "c1"
        or request.get("requeueIfPreempted") is not False
        or request.get("secrets") != ["fleet-api", "wandb-api"]
    ):
        raise ValueError("terminal acceptance requires the exact one-update Miles canary")
    files = plan.get("data", {}).get("files", {})
    if (
        set(files) != {"train", "dev"}
        or files["train"].get("rows") != 1
        or files["dev"].get("rows") != 1
    ):
        raise ValueError("reward canary requires one frozen train and dev task")
    return args, request


def _source_configs(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    manifest_path = Path(plan["arguments"]["data_manifest"])
    manifest, _ = _read(manifest_path, schema="cyber_miles_data_v1")
    if manifest != plan["data"]:
        raise ValueError("staged Miles data manifest differs from the plan")
    for split in ("train", "dev"):
        item = manifest["files"][split]
        path = manifest_path.parent / item["path"]
        if path != Path(plan["arguments"][split + "_data"]) or _hash(path) != str(
            item["sha256"]
        ).removeprefix("sha256:"):
            raise ValueError("staged Miles task row differs from the plan")
        values = [json.loads(line) for line in path.read_bytes().splitlines()]
        if len(values) != 1 or not isinstance(values[0], dict):
            raise ValueError("reward canary task row cardinality changed")
        row = values[0]
        metadata = row.get("metadata")
        config = metadata.get("cyber_config") if isinstance(metadata, dict) else None
        if (
            not isinstance(config, dict)
            or row.get("input") is None
            or metadata.get("split") != split
            or config.get("run_id") != plan["run_name"]
            or config.get("model", {}).get("repo") != plan["model"]["repo"]
            or config.get("model", {}).get("revision") != plan["model"]["revision"]
            or config.get("model", {}).get("root") != plan["arguments"]["model_root"]
            or config.get("initial_prompt_sha256") != fleet.sha256(row["input"].encode())
            or config.get("rl")
            != {key: value for key, value in manifest["limits"].items() if key != "response_tokens"}
            or config.get("execution", {}).get("required_task_tool_catalog_sha256")
            != manifest["tool_catalog_sha256"]
        ):
            raise ValueError("reward canary task binding changed")
        validate_episode_config(config)
        result[split] = config
    return result


def _dynamic_config(binding: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(binding)
    result.pop("native_batch")
    result.pop("sampling")
    result["run_id"] = (
        result["run_id"]
        .split("-dev-baseline-r0-s", 1)[0]
        .split("-dev-after-r0-s", 1)[0]
        .split("-train-r0-s", 1)[0]
    )
    result["config_sha256"] = fleet.digest_without(result, "config_sha256")
    return result


def _reward(config: dict[str, Any], value: object) -> tuple[float, str, str]:
    if not isinstance(value, dict) or set(value) != {
        "task_key",
        "task_version_id",
        "instance_id",
        "reward",
        "verifier_execution_id",
        "direct_authority_attestation",
    }:
        raise ValueError("episode reward is not a direct-authority receipt")
    score = value.get("reward")
    execution_id = _uuid(value.get("verifier_execution_id"), "verifier execution ID")
    if (
        isinstance(score, bool)
        or not isinstance(score, numbers.Real)
        or not math.isfinite(score)
        or not 0 <= score <= 1
        or value.get("task_key") != config["task"]["key"]
        or value.get("task_version_id") != config["task"]["version_id"]
    ):
        raise ValueError("authoritative reward identity or value changed")
    attestation = value["direct_authority_attestation"]
    context = attestation.get("context") if isinstance(attestation, dict) else None
    activity = attestation.get("activity") if isinstance(attestation, dict) else None
    shadow = attestation.get("shadow") if isinstance(attestation, dict) else None
    direct = shadow.get("direct_verifier") if isinstance(shadow, dict) else None
    evidence_id = _uuid(context.get("evidence_run_id"), "evidence run ID") if context else None
    if (
        attestation.get("schema_version") != fleet.DIRECT_AUTHORITY_ATTESTATION_SCHEMA
        or context
        != {
            "task_key": config["task"]["key"],
            "task_version_id": config["task"]["version_id"],
            "instance_id": value["instance_id"],
            "evidence_run_id": evidence_id,
            "verifier_version_id": config["verifier"]["version_id"],
            "scoring_payload_mode": fleet.RUNTIME_EVIDENCE_ONLY_V3,
        }
        or activity
        != {
            "result_schema_version": "cyber_verification_result_v3",
            "reward": float(score),
            "task_version_id": config["task"]["version_id"],
            "verifier_execution_id": execution_id,
        }
        or shadow
        != {
            "mode": "authoritative",
            "status": "authoritative",
            "match": True,
            "production_execution_id": execution_id,
            "direct_verifier": direct,
        }
        or direct
        != {
            "status": "authoritative",
            "match": True,
            "execution_id": execution_id,
            "verifier_contract_version": config["authority"]["required_cyber_contract"][
                "verifier_contract"
            ],
            "context_schema_version": "cyber_verification_context_v1",
        }
        or attestation.get("data_minimization")
        != {
            "components_included": False,
            "diagnostics_included": False,
            "evidence_payloads_included": False,
            "prompts_included": False,
            "traces_included": False,
            "flags_included": False,
        }
    ):
        raise ValueError("authoritative verifier attestation changed")
    return float(score), execution_id, evidence_id


def _episode(path: Path, source: dict[str, Any], kind: str) -> dict[str, Any]:
    if (
        path.is_symlink()
        or not path.is_dir()
        or {item.name for item in path.iterdir()} != (_EPISODE_FILES | {"ACCEPTED.json"})
    ):
        raise ValueError("Miles episode package is incomplete or ambiguous")
    accepted, _ = _read(path / "ACCEPTED.json")
    if (
        set(accepted)
        != {
            "task_version_id",
            "instance_id",
            "verifier_execution_id",
            "done_reason",
            "config_sha256",
            "sample_count",
            "files",
            "sha256",
        }
        or set(accepted.get("files", {})) != _EPISODE_FILES
    ):
        raise ValueError("Miles episode acceptance fields changed")
    values = {}
    for name in _EPISODE_FILES:
        file_path = path / name
        if (
            file_path.is_symlink()
            or not file_path.is_file()
            or fleet.sha256(file_path.read_bytes()) != accepted["files"][name]
        ):
            raise ValueError("Miles episode payload differs from its acceptance receipt")
        values[name] = json.loads(file_path.read_bytes())
    binding = values["binding.json"]
    validate_episode_config(binding)
    if (
        path.name != binding.get("run_id")
        or accepted.get("config_sha256") != binding.get("config_sha256")
        or binding.get("config_sha256") != fleet.digest_without(binding, "config_sha256")
        or _dynamic_config(binding) != source
        or binding.get("native_batch") != {"kind": kind, "rollout_id": 0}
        or not path.name.startswith(source["run_id"] + f"-{kind}-r0-s")
    ):
        raise ValueError("Miles episode/source/batch identity changed")
    instance = values["instance.json"]
    cleanup = values["cleanup.json"]
    instance_id = accepted.get("instance_id")
    if (
        set(instance) != {"instance_id", "evidence_run_id"}
        or instance.get("instance_id") != instance_id
        or cleanup
        != {
            "create_attempted": True,
            "instance_created": True,
            "instance_id": instance_id,
            "instance_closed": True,
            "possible_instance_leak": False,
        }
    ):
        raise ValueError("Miles episode environment release is incomplete")
    score, execution_id, evidence_id = _reward(binding, values["reward.json"])
    if (
        instance["evidence_run_id"] != evidence_id
        or accepted.get("task_version_id") != binding["task"]["version_id"]
        or accepted.get("verifier_execution_id") != execution_id
        or values["create-intent.json"] != {"run_id": binding["run_id"]}
        or values["score-intent.json"]
        != {
            "instance_id": instance_id,
            "scoring_mode": binding["authority"]["scoring_mode"],
            "multi_app_aggregation_mode": binding["authority"]["multi_app_aggregation_mode"],
        }
    ):
        raise ValueError("Miles episode scoring identity changed")
    conversation = values["conversation.json"].get("messages")
    if not isinstance(conversation, list) or not conversation:
        raise ValueError("Miles episode has no recorded task interaction")
    assistant = sum(
        isinstance(message, dict) and message.get("role") == "assistant" for message in conversation
    )
    tools = sum(
        isinstance(message, dict) and message.get("role") == "tool" for message in conversation
    )
    if (
        assistant < 1
        or tools < 1
        or (score > 0 and accepted.get("done_reason") != "report_submitted")
    ):
        raise ValueError("Miles episode lacks genuine task interaction or terminal report")
    samples = values["recording.json"].get("samples")
    if (
        type(accepted.get("sample_count")) is not int
        or accepted["sample_count"] < 1
        or not isinstance(samples, list)
        or len(samples) != accepted["sample_count"]
    ):
        raise ValueError("Miles episode token recording is incomplete")
    for sample in samples:
        if not isinstance(sample, dict) or set(sample) != {
            "tokens",
            "response_length",
            "loss_mask",
            "rollout_log_probs",
        }:
            raise ValueError("Miles episode token recording fields changed")
        tokens, response, mask, probabilities = (
            sample["tokens"],
            sample["response_length"],
            sample["loss_mask"],
            sample["rollout_log_probs"],
        )
        if (
            not isinstance(tokens, list)
            or any(type(value) is not int or value < 0 for value in tokens)
            or type(response) is not int
            or not 0 < response < len(tokens)
            or not isinstance(mask, list)
            or not isinstance(probabilities, list)
            or len(mask) != response
            or len(probabilities) != response
            or not any(value == 1 for value in mask)
            or any(type(value) is not int or value not in (0, 1) for value in mask)
            or any(
                isinstance(value, bool)
                or not isinstance(value, numbers.Real)
                or not math.isfinite(value)
                or value > 1e-6
                for value in probabilities
            )
            or any(
                probability != 0
                for probability, enabled in zip(probabilities, mask, strict=True)
                if not enabled
            )
        ):
            raise ValueError("Miles episode token recording is invalid")
    verifier = binding["verifier"]
    if (
        set(verifier)
        not in (
            {"id", "version_id", "version", "sha256", "function_name"},
            {"id", "version_id", "version", "sha256", "code_sha256", "function_name"},
        )
        or _UUID.fullmatch(str(verifier.get("id"))) is None
        or _UUID.fullmatch(str(verifier.get("version_id"))) is None
        or _SHA.fullmatch(str(verifier.get("sha256"))) is None
        or verifier.get("function_name") != "verify"
    ):
        raise ValueError("Miles episode verifier identity is incomplete")
    return {
        "kind": kind,
        "sample_index": int(path.name.rsplit("-s", 1)[1]),
        "score": score,
        "execution_id": execution_id,
        "assistant_messages": assistant,
        "tool_messages": tools,
        "report_submitted": accepted.get("done_reason") == "report_submitted",
        "verifier": {
            "task_key": binding["task"]["key"],
            "task_version_id": binding["task"]["version_id"],
            "verifier_id": verifier["id"],
            "verifier_version_id": verifier["version_id"],
            "verifier_sha256": verifier["sha256"],
        },
    }


def episode_audit(plan: dict[str, Any]) -> dict[str, Any]:
    """Reopen all ten private episodes and return value-free scientific facts."""
    _canary(plan)
    sources = _source_configs(plan)
    root = Path(plan["output_root"]) / "episodes"
    batch_root = root / "batches"
    if root.is_symlink() or not root.is_dir() or batch_root.is_symlink() or not batch_root.is_dir():
        raise ValueError("Miles episode root is missing or indirect")
    expected_batches = {
        "dev-baseline-r0": ("dev-baseline", [0]),
        "train-r0": ("train", list(range(WORLD_SIZE))),
        "dev-after-r0": ("dev-after", [0]),
    }
    if {path.name for path in batch_root.iterdir()} != set(expected_batches):
        raise ValueError("Miles canary batch set changed")
    for name, (kind, indices) in expected_batches.items():
        directory = batch_root / name
        if directory.is_symlink() or {path.name for path in directory.iterdir()} != {
            "STARTED.json",
            "COLLECTED.json",
        }:
            raise ValueError("Miles batch evidence is incomplete or ambiguous")
        started = json.loads((directory / "STARTED.json").read_bytes())
        collected, _ = _read(directory / "COLLECTED.json", schema="cyber_miles_batch_v1")
        if (
            _unsigned(collected) != started
            or started.get("batch_id") != name
            or started.get("data_sha256") != plan["data"]["sha256"]
            or started.get("episode_indices") != indices
            or started.get("native_rollout_id") != 0
            or started.get("evaluation") is not (kind != "train")
            or started.get("optimizer_step_verified") is not False
        ):
            raise ValueError("Miles batch evidence differs from the immutable plan")
    episode_paths = [path for path in root.iterdir() if path.name != "batches"]
    if len(episode_paths) != 10:
        raise ValueError("Miles reward canary requires exactly ten episode packages")
    rows = []
    for path in episode_paths:
        matches = [
            kind for kind in ("dev-baseline", "train", "dev-after") if f"-{kind}-r0-s" in path.name
        ]
        if len(matches) != 1:
            raise ValueError("Miles episode name does not identify one batch")
        kind = matches[0]
        rows.append(_episode(path, sources["train" if kind == "train" else "dev"], kind))
    by_kind = {
        kind: sorted(row["sample_index"] for row in rows if row["kind"] == kind)
        for kind in ("dev-baseline", "train", "dev-after")
    }
    if by_kind != {
        "dev-baseline": [0],
        "train": list(range(WORLD_SIZE)),
        "dev-after": [0],
    }:
        raise ValueError("Miles episode sample identities changed")
    execution_ids = [row["execution_id"] for row in rows]
    train = [row for row in rows if row["kind"] == "train"]
    scores = [row["score"] for row in train]
    mean = math.fsum(scores) / len(scores)
    variance = math.fsum((score - mean) ** 2 for score in scores) / len(scores)
    if len(set(execution_ids)) != 10 or not any(score > 0 for score in scores) or variance <= 0:
        raise ValueError("Miles canary lacks unique authoritative reward variance")
    verifier_bindings = sorted(
        {json.dumps(row["verifier"], sort_keys=True, separators=(",", ":")) for row in rows}
    )
    return {
        "schema": EPISODE_AUDIT_SCHEMA,
        "source_plan_sha256": "sha256:" + digest(plan),
        "accepted_episode_count": 10,
        "train_episode_count": WORLD_SIZE,
        "dev_episode_count": 2,
        "infrastructure_invalid_count": 0,
        "truncated_count": 0,
        "unique_authoritative_verifier_execution_count": 10,
        "authoritative_verifier_execution_ids_sha256": "sha256:" + digest(sorted(execution_ids)),
        "authoritative_verifier_bindings_sha256": "sha256:" + digest(verifier_bindings),
        "train_nonzero_reward_present": True,
        "train_within_group_reward_variance_present": True,
        "reward_values_included": False,
        "assistant_message_count": sum(row["assistant_messages"] for row in rows),
        "tool_message_count": sum(row["tool_messages"] for row in rows),
        "report_submitted_count": sum(row["report_submitted"] for row in rows),
        "task_content_included": False,
    }


def validate_wandb_observation(value: dict[str, Any], plan: dict[str, Any]) -> None:
    sealed(value, WANDB_SCHEMA)
    args = plan["arguments"]
    expected_keys = {
        "schema",
        "source_plan_sha256",
        "source_request_sha256",
        "identity",
        "state",
        "history_rows",
        "remote_scalar_history_sha256",
        "observed_optimizer_steps",
        "step_one_metric_names",
        "step_one_metrics",
        "positive_finite_gradient_observed",
        "finite_policy_loss_observed",
        "learning_rate_matches_plan",
        "logged_artifact_count",
        "rich_payload_count",
        "reward_values_included",
        "sha256",
    }
    identity = value.get("identity")
    metrics = value.get("step_one_metric_names")
    step_one = value.get("step_one_metrics")
    if (
        set(value) != expected_keys
        or value.get("source_plan_sha256", "").removeprefix("sha256:") != digest(plan)
        or value.get("source_request_sha256", "").removeprefix("sha256:")
        != digest(job_request(plan))
        or identity
        != {
            "entity": args["wandb_entity"],
            "project": args["wandb_project"],
            "run_id": args["wandb_run_id"],
            "name": args["name"],
        }
        or value.get("state") != "finished"
        or type(value.get("history_rows")) is not int
        or not 1 <= value["history_rows"] <= 10000
        or _SHA.fullmatch(str(value.get("remote_scalar_history_sha256"))) is None
        or value.get("observed_optimizer_steps") != [0, 1]
        or not isinstance(metrics, list)
        or metrics != sorted(metrics)
        or not {
            "trainer/global_step",
            "policy/grad_norm",
            "policy/policy_loss",
            "policy/policy_lr",
        }
        <= set(metrics)
        or not isinstance(step_one, dict)
        or set(step_one)
        != {
            "trainer/global_step",
            "policy/grad_norm",
            "policy/policy_loss",
            "policy/policy_lr",
        }
        or step_one.get("trainer/global_step") != 1.0
        or isinstance(step_one.get("policy/grad_norm"), bool)
        or not isinstance(step_one.get("policy/grad_norm"), numbers.Real)
        or not math.isfinite(step_one["policy/grad_norm"])
        or step_one["policy/grad_norm"] <= 0
        or isinstance(step_one.get("policy/policy_loss"), bool)
        or not isinstance(step_one.get("policy/policy_loss"), numbers.Real)
        or not math.isfinite(step_one["policy/policy_loss"])
        or step_one.get("policy/policy_lr") != args["lr"]
        or value.get("positive_finite_gradient_observed") is not True
        or value.get("finite_policy_loss_observed") is not True
        or value.get("learning_rate_matches_plan") is not True
        or value.get("logged_artifact_count") != 0
        or value.get("rich_payload_count") != 0
        or value.get("reward_values_included") is not False
    ):
        raise ValueError("W&B observation does not prove the exact scalar-only step-one run")


def observe_wandb(plan: dict[str, Any], *, api: Any = None) -> dict[str, Any]:
    """Read one exact W&B run and return a reward-value-free sealed observation."""
    from .skyrl_training import _finite_scalar_rows, _wandb_has_rich_payloads

    _canary(plan)
    args = plan["arguments"]
    if api is None:
        import wandb

        api = wandb.Api()
    run = api.run(f"{args['wandb_entity']}/{args['wandb_project']}/{args['wandb_run_id']}")
    if (
        (run.entity, run.project, run.id, run.name)
        != (
            args["wandb_entity"],
            args["wandb_project"],
            args["wandb_run_id"],
            args["name"],
        )
        or run.state != "finished"
        or _wandb_has_rich_payloads(run)
    ):
        raise ValueError("W&B run identity, state, or scalar-only surface changed")
    rows = _finite_scalar_rows(run.scan_history(), label="W&B")
    if len(rows) > 10_000:
        raise ValueError("W&B scalar history exceeds its reviewed bound")
    by_optimizer_step: dict[int, dict[str, float]] = {}
    for row in rows:
        raw_step = row.get("trainer/global_step")
        if raw_step is None:
            continue
        if not float(raw_step).is_integer() or raw_step < 0:
            raise ValueError("W&B optimizer step is not a nonnegative integer")
        step = int(raw_step)
        target = by_optimizer_step.setdefault(step, {})
        for key, item in row.items():
            if key.startswith("_"):
                continue
            if key in target and target[key] != item:
                raise ValueError("W&B rewrites a metric within one optimizer step")
            target[key] = item
    if set(by_optimizer_step) != {0, 1}:
        raise ValueError("W&B does not prove exactly one optimizer update")
    one = by_optimizer_step[1]
    required = {
        "trainer/global_step",
        "policy/grad_norm",
        "policy/policy_loss",
        "policy/policy_lr",
    }
    if (
        not required <= set(one)
        or one["trainer/global_step"] != 1.0
        or not math.isfinite(one["policy/grad_norm"])
        or one["policy/grad_norm"] <= 0
        or not math.isfinite(one["policy/policy_loss"])
        or one["policy/policy_lr"] != args["lr"]
    ):
        raise ValueError("W&B step one lacks finite optimizer telemetry")
    value = {
        "schema": WANDB_SCHEMA,
        "source_plan_sha256": "sha256:" + digest(plan),
        "source_request_sha256": "sha256:" + digest(job_request(plan)),
        "identity": {
            "entity": run.entity,
            "project": run.project,
            "run_id": run.id,
            "name": run.name,
        },
        "state": run.state,
        "history_rows": len(rows),
        "remote_scalar_history_sha256": "sha256:" + digest(rows),
        "observed_optimizer_steps": [0, 1],
        "step_one_metric_names": sorted(one),
        "step_one_metrics": {key: one[key] for key in sorted(required)},
        "positive_finite_gradient_observed": True,
        "finite_policy_loss_observed": True,
        "learning_rate_matches_plan": True,
        "logged_artifact_count": 0,
        "rich_payload_count": 0,
        "reward_values_included": False,
    }
    value["sha256"] = "sha256:" + digest(value)
    validate_wandb_observation(value, plan)
    return value


def validate_controller_observation(value: dict[str, Any], plan: dict[str, Any]) -> None:
    sealed(value, CONTROLLER_SCHEMA)
    _, request = _canary(plan)
    api = value.get("api")
    kube = value.get("kubernetes")
    execution = value.get("execution")
    pods = kube.get("pods") if isinstance(kube, dict) else None
    rayjob = kube.get("rayjob") if isinstance(kube, dict) else None
    workload = kube.get("workload") if isinstance(kube, dict) else None
    raycluster = kube.get("raycluster") if isinstance(kube, dict) else None
    api_name = api.get("run_name") if isinstance(api, dict) else None
    expected_image = miles.IMAGE.rsplit("@sha256:", 1)[1]
    if (
        set(value)
        != {
            "schema",
            "status",
            "source_plan_sha256",
            "source_request_sha256",
            "api",
            "kubernetes",
            "execution",
            "observed_at",
            "sha256",
        }
        or value.get("status") != "succeeded"
        or value.get("source_plan_sha256", "").removeprefix("sha256:") != digest(plan)
        or value.get("source_request_sha256", "").removeprefix("sha256:") != digest(request)
        or not isinstance(api, dict)
        or set(api) != {"base_url", "run_id", "run_name", "status"}
        or api.get("base_url") != API_URLS["dev"]
        or _UUID.fullmatch(str(api.get("run_id"))) is None
        or not isinstance(api_name, str)
        or api_name != plan["run_name"] + "-" + str(api.get("run_id"))[:8]
        or api.get("status") != "SUCCEEDED"
        or not isinstance(kube, dict)
        or set(kube)
        != {"cluster", "namespace", "namespace_uid", "rayjob", "workload", "raycluster", "pods"}
        or kube.get("cluster") != "dev"
        or kube.get("namespace") != NAMESPACE
        or kube.get("namespace_uid") != NAMESPACE_UID
        or not isinstance(rayjob, dict)
        or set(rayjob) != {"name", "uid", "status"}
        or rayjob.get("name") != api_name
        or _uuid(rayjob.get("uid"), "RayJob UID") != rayjob.get("uid")
        or rayjob.get("status") != "SUCCEEDED"
        or not isinstance(workload, dict)
        or set(workload) != {"name", "uid", "owner_rayjob_uid"}
        or _uuid(workload.get("uid"), "Workload UID") != workload.get("uid")
        or workload.get("owner_rayjob_uid") != rayjob.get("uid")
        or not isinstance(workload.get("name"), str)
        or not isinstance(raycluster, dict)
        or set(raycluster) != {"name", "uid", "owner_rayjob_uid"}
        or _uuid(raycluster.get("uid"), "RayCluster UID") != raycluster.get("uid")
        or raycluster.get("owner_rayjob_uid") != rayjob.get("uid")
        or not isinstance(raycluster.get("name"), str)
        or not isinstance(pods, list)
        or len(pods) != 1
        or not isinstance(execution, dict)
        or execution
        != {
            "requested_image": miles.IMAGE,
            "priority_class": "c1",
            "effective_priority": 10000,
            "automatic_requeue": False,
            "workers": 1,
            "gpus_per_worker": WORLD_SIZE,
            "total_gpus": WORLD_SIZE,
        }
    ):
        raise ValueError("controller observation differs from the exact dev canary")
    pod = pods[0]
    if (
        set(pod)
        != {
            "name",
            "uid",
            "owner_raycluster_uid",
            "phase",
            "exit_code",
            "termination_reason",
            "runtime_image_id",
            "runtime_uid",
            "runtime_gid",
            "container_restarts",
            "gpus",
        }
        or _uuid(pod.get("uid"), "Pod UID") != pod.get("uid")
        or pod.get("owner_raycluster_uid") != raycluster["uid"]
        or pod.get("phase") != "Succeeded"
        or pod.get("exit_code") != 0
        or pod.get("termination_reason") != "Completed"
        or _image_digest(pod.get("runtime_image_id")) != expected_image
        or pod.get("runtime_uid") != 1000
        or pod.get("runtime_gid") != 100
        or pod.get("container_restarts") != 0
        or pod.get("gpus") != WORLD_SIZE
    ):
        raise ValueError("controller Pod observation is incomplete or mismatched")
    _time(value.get("observed_at"), "controller observation")


def validate_release_observation(
    value: dict[str, Any],
    plan: dict[str, Any],
    controller: dict[str, Any],
    controller_file_sha256: str,
    *,
    not_before: float,
) -> None:
    sealed(value, RELEASE_SCHEMA)
    api = controller["api"]
    kube = controller["kubernetes"]
    identities = value.get("identities")
    if (
        set(value)
        != {
            "schema",
            "status",
            "source_plan_sha256",
            "source_request_sha256",
            "controller_observation_sha256",
            "controller_observation_file_sha256",
            "api_status",
            "controller_status",
            "identities",
            "raycluster_present",
            "rayjob_present",
            "workload_present",
            "quota_reservation_present",
            "gpu_pods_present",
            "active_gpus",
            "gpu_release_proven",
            "observed_at",
            "sha256",
        }
        or value.get("status") != "released"
        or value.get("source_plan_sha256", "").removeprefix("sha256:") != digest(plan)
        or value.get("source_request_sha256", "").removeprefix("sha256:")
        != digest(job_request(plan))
        or value.get("controller_observation_sha256", "").removeprefix("sha256:")
        != controller["sha256"].removeprefix("sha256:")
        or value.get("controller_observation_file_sha256", "").removeprefix("sha256:")
        != controller_file_sha256.removeprefix("sha256:")
        or value.get("api_status") != "SUCCEEDED"
        or value.get("controller_status") != "SUCCEEDED"
        or identities
        != {
            "api_run_id": api["run_id"],
            "api_run_name": api["run_name"],
            "rayjob_uid": kube["rayjob"]["uid"],
            "workload_uid": kube["workload"]["uid"],
            "raycluster_uid": kube["raycluster"]["uid"],
            "pod_uids": [kube["pods"][0]["uid"]],
        }
        or value.get("raycluster_present") is not False
        or value.get("rayjob_present") is not False
        or value.get("workload_present") is not False
        or value.get("quota_reservation_present") is not False
        or value.get("gpu_pods_present") is not False
        or value.get("active_gpus") != 0
        or value.get("gpu_release_proven") is not True
        or _time(value.get("observed_at"), "release observation")
        < max(not_before, _time(controller["observed_at"], "controller observation"))
    ):
        raise ValueError("external release observation is incomplete or mismatched")


def _checkpoint(plan: dict[str, Any], path: Path) -> tuple[dict[str, Any], str, bool]:
    manifest, file_sha256 = _read(path, schema=CHECKPOINT_SCHEMA)
    _verify_checkpoint(manifest, hashes=True)
    source = manifest.get("source", {})
    completion, _ = _read(Path(plan["output_root"]) / "NATIVE_TRAINING_COMPLETE.json")
    if (
        manifest.get("rollout_index") != 0
        or manifest.get("next_rollout_id") != 1
        or manifest.get("world_size") != WORLD_SIZE
        or manifest.get("topology") != {"nodes": 1, "gpus_per_node": WORLD_SIZE}
        or manifest.get("model") != plan["model"]
        or source.get("run_name") != plan["run_name"]
        or source.get("output_root") != plan["output_root"]
        or source.get("plan_sha256") != digest(plan)
        or source.get("completion_sha256") != completion["sha256"].removeprefix("sha256:")
        or source.get("arguments") != plan["arguments"]
        or source.get("execution") != plan["execution"]
        or source.get("native_driver_sha256") != plan["native_driver_sha256"]
        or manifest.get("source_optimizer_update_claimed") is not False
        or manifest.get("gpu_reload_verified") is not False
        or manifest.get("optimizer_update_during_reload") is not False
    ):
        raise ValueError("Miles checkpoint seal differs from the terminal source")
    base_hashes = {
        str(item.get("sha256", "")).removeprefix("sha256:")
        for item in plan["checkpoint"].get("files", [])
    }
    trained_hashes = {
        str(item.get("sha256", "")).removeprefix("sha256:") for item in manifest["files"]
    }
    changed = bool(trained_hashes - base_hashes)
    if not changed:
        raise ValueError("sealed Miles checkpoint has no payload delta from the base")
    return manifest, file_sha256, changed


def _compile(
    plan: dict[str, Any],
    *,
    checkpoint_manifest_path: Path,
    controller_observation_path: Path,
    release_observation_path: Path,
    wandb_observation_path: Path,
) -> dict[str, Any]:
    _, request = _canary(plan)
    root = Path(plan["output_root"])
    completion_path = root / "NATIVE_TRAINING_COMPLETE.json"
    completion, completion_file_sha256 = _read(completion_path)
    if (
        set(completion)
        != {
            "status",
            "plan_sha256",
            "checkpoint_rollout_index",
            "completed_batches",
            "completed_at",
            "optimizer_update_independently_verified",
            "checkpoint_reload_verified",
            "sha256",
        }
        or completion.get("status") != "native_loop_returned"
        or completion.get("plan_sha256") != digest(plan)
        or completion.get("checkpoint_rollout_index") != 0
        or completion.get("completed_batches") != 3
        or completion.get("optimizer_update_independently_verified") is not False
        or completion.get("checkpoint_reload_verified") is not False
        or any(
            (root / name).exists()
            for name in (
                "FAILED.json",
                "REJECTED.json",
                "NATIVE_FAILURE.json",
                "NATIVE_REJECTED.json",
                "ACCEPTED.json",
            )
        )
    ):
        raise ValueError("Miles native completion is conflicting or incomplete")
    completed_at = _time(completion.get("completed_at"), "native completion")
    episodes = episode_audit(plan)
    checkpoint, checkpoint_file_sha256, checkpoint_changed = _checkpoint(
        plan, checkpoint_manifest_path
    )
    wandb, wandb_file_sha256 = _read(wandb_observation_path, schema=WANDB_SCHEMA)
    validate_wandb_observation(wandb, plan)
    controller, controller_file_sha256 = _read(
        controller_observation_path, schema=CONTROLLER_SCHEMA
    )
    validate_controller_observation(controller, plan)
    release, release_file_sha256 = _read(release_observation_path, schema=RELEASE_SCHEMA)
    validate_release_observation(
        release,
        plan,
        controller,
        controller_file_sha256,
        not_before=completed_at,
    )
    return {
        "schema": TERMINAL_SCHEMA,
        "status": "accepted",
        "source_run_name": plan["run_name"],
        "source_plan_sha256": "sha256:" + digest(plan),
        "source_request_sha256": "sha256:" + digest(request),
        "native_completion": _reference(completion_path, completion, completion_file_sha256),
        "episode_audit": episodes,
        "optimizer_update_proof": {
            "optimizer_updates": 1,
            "basis": [
                "one_native_rollout_with_one_step_per_rollout",
                "eight_accepted_train_episodes_with_positive_reward_variance",
                "positive_finite_gradient_and_finite_policy_loss_at_step_one",
                "new_all_rank_checkpoint_payload_at_rollout_index_zero",
            ],
            "checkpoint_payload_changed_from_base": checkpoint_changed,
            "counter_only_claim": False,
        },
        "wandb_scalar_observation": _reference(wandb_observation_path, wandb, wandb_file_sha256),
        "checkpoint_manifest": _reference(
            checkpoint_manifest_path, checkpoint, checkpoint_file_sha256
        ),
        "controller_observation": _reference(
            controller_observation_path, controller, controller_file_sha256
        ),
        "external_release": _reference(release_observation_path, release, release_file_sha256),
        "external_gpu_release_verified": True,
        "reward_values_included": False,
        "task_content_included": False,
        "production_promotion_requires_reload_acceptance": True,
    }


def accept_terminal(
    plan: dict[str, Any],
    *,
    checkpoint_manifest_path: Path,
    controller_observation_path: Path,
    release_observation_path: Path,
    wandb_observation_path: Path,
    output: Path,
) -> dict[str, Any]:
    """Create one sanitized terminal acceptance after all source gates pass."""
    root = Path(plan["output_root"])
    if output != root / "MILES_TERMINAL_ACCEPTED.json":
        raise ValueError("Miles terminal acceptance output path is not plan-bound")
    if output.exists() or output.is_symlink():
        raise FileExistsError("Miles terminal acceptance already exists")
    value = _compile(
        plan,
        checkpoint_manifest_path=checkpoint_manifest_path,
        controller_observation_path=controller_observation_path,
        release_observation_path=release_observation_path,
        wandb_observation_path=wandb_observation_path,
    )
    return _write(output, value)


def validate_terminal(value: dict[str, Any], *, check_files: bool = True) -> dict[str, str]:
    """Reopen every terminal prerequisite for a later production promoter."""
    sealed(value, TERMINAL_SCHEMA)
    episode = value.get("episode_audit")
    update = value.get("optimizer_update_proof")
    references = tuple(
        value.get(key)
        for key in (
            "native_completion",
            "wandb_scalar_observation",
            "checkpoint_manifest",
            "controller_observation",
            "external_release",
        )
    )
    if (
        set(value) != _TERMINAL_FIELDS
        or value.get("status") != "accepted"
        or value.get("production_promotion_requires_reload_acceptance") is not True
        or value.get("external_gpu_release_verified") is not True
        or value.get("reward_values_included") is not False
        or value.get("task_content_included") is not False
        or not isinstance(episode, dict)
        or set(episode) != _EPISODE_AUDIT_FIELDS
        or episode.get("schema") != EPISODE_AUDIT_SCHEMA
        or episode.get("accepted_episode_count") != 10
        or episode.get("train_episode_count") != WORLD_SIZE
        or episode.get("dev_episode_count") != 2
        or episode.get("infrastructure_invalid_count") != 0
        or episode.get("truncated_count") != 0
        or episode.get("unique_authoritative_verifier_execution_count") != 10
        or episode.get("train_nonzero_reward_present") is not True
        or episode.get("train_within_group_reward_variance_present") is not True
        or episode.get("reward_values_included") is not False
        or episode.get("task_content_included") is not False
        or not isinstance(update, dict)
        or set(update) != _UPDATE_PROOF_FIELDS
        or update.get("optimizer_updates") != 1
        or update.get("checkpoint_payload_changed_from_base") is not True
        or update.get("counter_only_claim") is not False
        or not all(
            isinstance(reference, dict) and set(reference) == _REFERENCE_FIELDS
            for reference in references
        )
    ):
        raise ValueError("Miles terminal acceptance is not promotion-safe")
    if not check_files:
        return {
            "source_plan_sha256": value["source_plan_sha256"],
            "terminal_receipt_sha256": value["sha256"],
        }
    checkpoint = value["checkpoint_manifest"]
    controller = value["controller_observation"]
    release = value["external_release"]
    wandb = value["wandb_scalar_observation"]
    completion = value["native_completion"]
    plan_path = Path(completion["path"]).parent / "plan.json"
    if plan_path.is_symlink() or not plan_path.is_file():
        raise ValueError("Miles terminal validation requires the exact source plan file")
    plan = json.loads(plan_path.read_bytes())
    expected = _compile(
        plan,
        checkpoint_manifest_path=Path(checkpoint["path"]),
        controller_observation_path=Path(controller["path"]),
        release_observation_path=Path(release["path"]),
        wandb_observation_path=Path(wandb["path"]),
    )
    if _unsigned(value) != expected:
        raise ValueError("Miles terminal acceptance differs from rederived evidence")
    return {
        "source_plan_sha256": value["source_plan_sha256"],
        "terminal_receipt_sha256": value["sha256"],
    }
