"""Offline qualification for fresh Qwen direct-interface self-trace sources.

This module is deliberately not a launcher.  It freezes a bounded collection
request, derives its exact attempt roster, qualifies a synthetic native-recorder
sample against the dense adapter, and turns one private accepted episode into a
content-free source receipt.  Prompt, trace, tool-result, flag, and score values
are read only inside the private review boundary and never appear in receipts.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import stat
from pathlib import Path
from typing import Any

from evals.fleet import opencode_self_hosted as fleet

from . import dense, rl_data, rl_episode, self_trace_corpus
from .io import digest_json, file_sha256

REQUEST_SCHEMA = "cyber_qwen_direct_self_trace_collection_request_v1"
PARITY_SCHEMA = "cyber_qwen_direct_recorder_dense_parity_v1"
EXPECTATION_SCHEMA = "cyber_qwen_direct_self_trace_source_expectation_v1"
SOURCE_RECEIPT_SCHEMA = "cyber_qwen_direct_self_trace_source_receipt_v1"

MODEL_REPO = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
CHAT_TEMPLATE_SHA256 = "sha256:c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"
TOOL_CATALOG_SHA256 = "sha256:85fad6bdc3a835bf52a11a99b3387740eb06eb3d1720ad9bb33f3feac215b44a"
SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
IMAGE = re.compile(r"[^\s]+@sha256:[0-9a-f]{64}\Z")

REQUEST_PREFIX_POLICY = {
    "roles": ["system", "user"],
    "tool_schema": "complete_ordered_openai_function_catalog",
    "add_generation_prompt": True,
    "compaction": "disabled",
    "tool_rewriting": "prohibited",
}


class CollectionError(ValueError):
    """A fixed, payload-free collection qualification rejection."""


def _sha(value: object) -> bool:
    return isinstance(value, str) and SHA256.fullmatch(value) is not None


def _sealed(value: dict, schema: str) -> None:
    if (
        not isinstance(value, dict)
        or value.get("schema") != schema
        or value.get("sha256")
        != digest_json({key: item for key, item in value.items() if key != "sha256"})
    ):
        raise CollectionError("sealed collection metadata differs")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        raise CollectionError("bound collection JSON is unreadable") from None


def _bound(reference: object, relative_to: Path, *, sealed: bool = False) -> tuple[Path, Any]:
    if not isinstance(reference, dict) or set(reference) != {
        "path",
        "file_sha256",
        "document_sha256",
    }:
        raise CollectionError("collection input binding fields differ")
    path_value = reference.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise CollectionError("collection input path is missing")
    path = Path(path_value)
    path = path if path.is_absolute() else relative_to / path
    if path.is_symlink() or not path.is_file() or file_sha256(path) != reference["file_sha256"]:
        raise CollectionError("collection input file digest differs")
    value = _read_json(path)
    document_sha = reference.get("document_sha256")
    observed = value.get("sha256") if isinstance(value, dict) else None
    if not _sha(document_sha) or observed not in {
        document_sha,
        document_sha.removeprefix("sha256:"),
    }:
        raise CollectionError("collection input document digest differs")
    if sealed:
        if not isinstance(value, dict):
            raise CollectionError("bound sealed collection input is not an object")
        expected = digest_json({key: item for key, item in value.items() if key != "sha256"})
        if observed not in {expected, expected.removeprefix("sha256:")}:
            raise CollectionError("bound collection input self-digest differs")
    return path, value


def _tool_catalog(reference: object, relative_to: Path) -> tuple[Path, list[dict]]:
    if not isinstance(reference, dict) or set(reference) != {
        "path",
        "file_sha256",
        "canonical_sha256",
    }:
        raise CollectionError("tool catalog binding fields differ")
    path_value = reference.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise CollectionError("tool catalog path is missing")
    path = Path(path_value)
    path = path if path.is_absolute() else relative_to / path
    if path.is_symlink() or not path.is_file() or file_sha256(path) != reference["file_sha256"]:
        raise CollectionError("tool catalog file digest differs")
    value = _read_json(path)
    if (
        not isinstance(value, list)
        or [row.get("name") for row in value if isinstance(row, dict)] != ["bash", "submit_report"]
        or any(
            set(row) != {"name", "description", "inputSchema"}
            or not isinstance(row["description"], str)
            or not isinstance(row["inputSchema"], dict)
            for row in value
        )
        or fleet.sha256(fleet.canonical_json(value)) != reference["canonical_sha256"]
        or reference["canonical_sha256"] != TOOL_CATALOG_SHA256
    ):
        raise CollectionError("tool catalog is not the exact ordered direct surface")
    return path, value


def _source_closure(value: object, relative_to: Path) -> dict[str, str]:
    names = {
        "dense.py",
        "rl_data.py",
        "rl_episode.py",
        "self_trace_collection.py",
        "self_trace_corpus.py",
        "skyrl_episode.py",
    }
    if not isinstance(value, dict) or set(value) != names:
        raise CollectionError("collector source closure fields differ")
    result = {}
    for name, reference in value.items():
        if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
            raise CollectionError("collector source binding fields differ")
        raw = reference.get("path")
        path = Path(raw) if isinstance(raw, str) else Path()
        path = path if path.is_absolute() else relative_to / path
        if (
            not isinstance(raw, str)
            or not raw
            or path.is_symlink()
            or not path.is_file()
            or not _sha(reference.get("sha256"))
            or file_sha256(path) != reference["sha256"]
            or path.name != name
        ):
            raise CollectionError("collector source file digest differs")
        result[name] = reference["sha256"]
    return result


def _train_rows(split: dict) -> set[tuple[str, str]]:
    training = split.get("training_split")
    rows = training.get("tasks") if isinstance(training, dict) else None
    if (
        not isinstance(rows, list)
        or not rows
        or any(
            not isinstance(row, dict)
            or set(row) != {"task_key", "task_version_id", "split"}
            or row.get("split") != "train"
            or not isinstance(row.get("task_key"), str)
            or not isinstance(row.get("task_version_id"), str)
            for row in rows
        )
    ):
        raise CollectionError("representative training split projection differs")
    result = {(row["task_key"], row["task_version_id"]) for row in rows}
    if len(result) != len(rows):
        raise CollectionError("representative training split is duplicated")
    return result


def _valid_runtime_inventory(row: dict) -> bool:
    environment = row.get("environment")
    verifier = row.get("verifier")
    return (
        isinstance(environment, dict)
        and set(environment)
        == {
            "id",
            "version",
            "version_id",
            "data_id",
            "data_version",
            "runtime_seed_content_sha256",
        }
        and all(
            isinstance(environment.get(name), str) and bool(environment[name])
            for name in ("id", "version", "data_id", "data_version")
        )
        and self_trace_corpus._canonical_uuid(environment.get("version_id"))
        and _sha(_normalized_sha(environment.get("runtime_seed_content_sha256")))
        and isinstance(verifier, dict)
        and set(verifier) == {"id", "version_id", "version", "sha256"}
        and self_trace_corpus._canonical_uuid(verifier.get("id"))
        and self_trace_corpus._canonical_uuid(verifier.get("version_id"))
        and isinstance(verifier.get("version"), (str, int))
        and not isinstance(verifier.get("version"), bool)
        and bool(str(verifier["version"]))
        and _sha(_normalized_sha(verifier.get("sha256")))
        and _sha(row.get("current_binding_sha256"))
    )


def derive_roster(
    tasks: set[tuple[str, str]], *, attempts_per_task_version: int, seed: str
) -> list[dict]:
    """Derive stable attempt identities without looking at outcomes or task content."""
    if (
        not tasks
        or type(attempts_per_task_version) is not int
        or attempts_per_task_version <= 0
        or not isinstance(seed, str)
        or not seed
    ):
        raise CollectionError("attempt roster policy is invalid")
    result = []
    for task_key, version_id in sorted(tasks):
        identity = hashlib.sha256(f"{seed}\0{task_key}\0{version_id}".encode()).hexdigest()[:20]
        for ordinal in range(attempts_per_task_version):
            result.append(
                {
                    "attempt_id": f"q38-self-{identity}-a{ordinal + 1}",
                    "attempt_ordinal": ordinal,
                    "task_key": task_key,
                    "task_version_id": version_id,
                }
            )
    return result


def _validate_parity_receipt(
    parity: dict, *, model: dict, interface: dict, source_closure: dict[str, str]
) -> None:
    _sealed(parity, PARITY_SCHEMA)
    native = parity.get("native")
    dense_receipt = parity.get("dense")
    observation = parity.get("first_observation")
    privacy = parity.get("privacy")
    if (
        set(parity)
        != {
            "schema",
            "status",
            "model",
            "interface",
            "source_closure_sha256",
            "fixture_sha256",
            "native",
            "dense",
            "first_observation",
            "privacy",
            "sha256",
        }
        or parity.get("status") != "qualified"
        or parity.get("model")
        != {
            "repo": model["repo"],
            "revision": model["revision"],
            "runtime_chat_template_sha256": model["runtime_chat_template_sha256"],
        }
        or parity.get("interface")
        != {
            "required_task_tools": ["bash", "submit_report"],
            "required_task_tool_catalog_sha256": interface["tool_catalog"]["canonical_sha256"],
            "compaction": "disabled",
        }
        or parity.get("source_closure_sha256") != digest_json(source_closure)
        or not _sha(parity.get("fixture_sha256"))
        or not isinstance(native, dict)
        or set(native)
        != {
            "prompt_tokens_sha256",
            "recorded_tokens_sha256",
            "recorded_loss_mask_sha256",
            "assistant_targets",
        }
        or any(
            not _sha(native.get(name))
            for name in (
                "prompt_tokens_sha256",
                "recorded_tokens_sha256",
                "recorded_loss_mask_sha256",
            )
        )
        or type(native.get("assistant_targets")) is not int
        or native["assistant_targets"] < 2
        or not isinstance(dense_receipt, dict)
        or set(dense_receipt)
        != {
            "reference_tokens_sha256",
            "reference_loss_mask_sha256",
            "trained_prefix_tokens_sha256",
            "trained_prefix_loss_mask_sha256",
            "trained_prefix_tokens",
            "supervised_tokens",
        }
        or any(
            not _sha(dense_receipt.get(name))
            for name in (
                "reference_tokens_sha256",
                "reference_loss_mask_sha256",
                "trained_prefix_tokens_sha256",
                "trained_prefix_loss_mask_sha256",
            )
        )
        or any(
            type(dense_receipt.get(name)) is not int or dense_receipt[name] <= 0
            for name in ("trained_prefix_tokens", "supervised_tokens")
        )
        or not isinstance(observation, dict)
        or observation
        != {
            "tokens": observation.get("tokens"),
            "native_dense_token_ids_equal": True,
            "native_dense_loss_masks_equal": True,
            "all_observation_tokens_masked": True,
        }
        or type(observation.get("tokens")) is not int
        or observation["tokens"] <= 0
        or privacy
        != {
            "synthetic_fixture_only": True,
            "prompt_or_trace_content_emitted": False,
            "tool_arguments_or_results_emitted": False,
            "flags_scores_or_credentials_emitted": False,
        }
    ):
        raise CollectionError("synthetic parity receipt fields differ")


def validate_request(
    request: dict, *, relative_to: Path, require_ready: bool = False
) -> dict[str, Any]:
    """Validate a sealed request and return only metadata plus private in-memory inputs."""
    _sealed(request, REQUEST_SCHEMA)
    expected = {
        "schema",
        "status",
        "producer_plan",
        "historical_audit",
        "model",
        "selection",
        "interface",
        "limits",
        "sampling",
        "runtime",
        "qualification",
        "source_closure",
        "output_policy",
        "execution",
        "sha256",
    }
    if set(request) != expected:
        raise CollectionError("collection request fields differ")
    _, plan = _bound(request["producer_plan"], relative_to, sealed=True)
    _, audit = _bound(request["historical_audit"], relative_to, sealed=True)
    if (
        plan.get("schema") != "cyber_qwen_self_sft_recollection_plan_v1"
        or plan.get("execution", {}).get("launchable") is not False
        or plan.get("historical_source_gate", {}).get("native_compatible_sessions") != 0
        or audit.get("schema") != "cyber_qwen_self_sft_interface_audit_v2"
        or audit.get("status") != "blocked_no_native_compatible_source"
        or plan.get("historical_source_gate", {}).get("audit_sha256") != audit.get("sha256")
    ):
        raise CollectionError("collection request does not bind the blocked recollection gate")
    model = request["model"]
    if (
        not isinstance(model, dict)
        or set(model)
        != {
            "repo",
            "revision",
            "root",
            "lock_file_sha256",
            "runtime_chat_template_sha256",
            "initialization",
        }
        or model.get("repo") != MODEL_REPO
        or model.get("revision") != MODEL_REVISION
        or not isinstance(model.get("root"), str)
        or not model["root"].startswith("/")
        or not _sha(model.get("lock_file_sha256"))
        or model.get("runtime_chat_template_sha256") != CHAT_TEMPLATE_SHA256
        or model.get("initialization") != "exact_fresh_base"
        or model.get("lock_file_sha256") != plan.get("model", {}).get("lock_file_sha256")
        or model.get("repo") != plan.get("model", {}).get("repo")
        or model.get("revision") != plan.get("model", {}).get("revision")
    ):
        raise CollectionError("collection request model identity differs")

    selection = request["selection"]
    if not isinstance(selection, dict) or set(selection) != {
        "eligible_inventory",
        "representative_splits",
        "mode",
        "task_versions",
        "attempts_per_task_version",
        "maximum_sessions",
        "attempt_seed",
        "attempt_roster_sha256",
    }:
        raise CollectionError("collection selection fields differ")
    _, eligible = _bound(selection["eligible_inventory"], relative_to)
    split_refs = selection["representative_splits"]
    if not isinstance(split_refs, dict) or set(split_refs) != {"a", "b"}:
        raise CollectionError("representative split bindings differ")
    splits = {name: _bound(ref, relative_to, sealed=True)[1] for name, ref in split_refs.items()}
    train_sets = {name: _train_rows(split) for name, split in splits.items()}
    tasks = train_sets["a"] | train_sets["b"]
    inventory_rows = eligible.get("task_versions") if isinstance(eligible, dict) else None
    if not isinstance(inventory_rows, list):
        raise CollectionError("eligible task inventory is missing")
    inventory: dict[tuple[str, str], dict] = {}
    for row in inventory_rows:
        if not isinstance(row, dict):
            raise CollectionError("eligible task inventory row differs")
        key = (row.get("task_key"), row.get("task_version_id"))
        if not all(isinstance(item, str) and item for item in key) or key in inventory:
            raise CollectionError("eligible task identity is invalid or duplicated")
        inventory[key] = row
    if (
        selection.get("mode") != "representative_a_b_training_union"
        or selection["eligible_inventory"]["document_sha256"]
        != "sha256:" + str(plan.get("task_universe", {}).get("sha256", "")).removeprefix("sha256:")
        or any(
            split_refs[name]["document_sha256"]
            != plan.get("splits", {}).get(name, {}).get("sha256")
            for name in ("a", "b")
        )
        or selection.get("task_versions") != len(tasks)
        or selection.get("task_versions")
        != plan.get("splits", {}).get("overlap", {}).get("train_union_task_versions")
        or not tasks.issubset(inventory)
        or any(not _valid_runtime_inventory(inventory[key]) for key in tasks)
        or type(selection.get("attempts_per_task_version")) is not int
        or selection["attempts_per_task_version"] <= 0
        or not isinstance(selection.get("attempt_seed"), str)
        or not selection["attempt_seed"]
        or selection.get("attempts_per_task_version")
        != plan.get("collection", {}).get("attempts_per_task_version")
        or selection.get("maximum_sessions") != plan.get("collection", {}).get("maximum_sessions")
    ):
        raise CollectionError("collection task union differs from the frozen study")
    roster = derive_roster(
        tasks,
        attempts_per_task_version=selection["attempts_per_task_version"],
        seed=selection["attempt_seed"],
    )
    if (
        selection.get("maximum_sessions") != len(roster)
        or selection.get("attempt_roster_sha256") != digest_json(roster)
        or len({row["attempt_id"] for row in roster}) != len(roster)
    ):
        raise CollectionError("collection attempt roster differs")

    interface = request["interface"]
    if not isinstance(interface, dict) or set(interface) != {
        "backend",
        "required_task_tools",
        "tool_catalog",
        "request_prefix_policy",
        "request_prefix_policy_sha256",
        "system_prompt_sha256",
        "runtime_chat_template_sha256",
        "native_helper_sha256",
    }:
        raise CollectionError("direct interface fields differ")
    _, catalog = _tool_catalog(interface["tool_catalog"], relative_to)
    if (
        interface.get("backend") != "skyrl_direct"
        or interface.get("required_task_tools") != ["bash", "submit_report"]
        or interface.get("request_prefix_policy") != REQUEST_PREFIX_POLICY
        or interface.get("request_prefix_policy_sha256") != digest_json(REQUEST_PREFIX_POLICY)
        or interface.get("runtime_chat_template_sha256") != CHAT_TEMPLATE_SHA256
        or interface.get("native_helper_sha256") != dense.NATIVE_HELPER_SHA
        or interface.get("system_prompt_sha256") is not None
        and not _sha(interface["system_prompt_sha256"])
    ):
        raise CollectionError("direct request-prefix contract differs")
    limits = request["limits"]
    if (
        not isinstance(limits, dict)
        or set(limits)
        != {
            "context_tokens",
            "response_tokens",
            "max_tokens_per_turn",
            "max_turns",
            "episode_seconds",
            "tool_seconds",
            "tool_result_chars",
            "ttl_seconds",
        }
        or any(type(item) is not int or item <= 0 for item in limits.values())
        or not limits["max_tokens_per_turn"]
        <= limits["response_tokens"]
        < limits["context_tokens"]
        <= 98304
        or not limits["episode_seconds"] + 240 < limits["ttl_seconds"] <= 32400
    ):
        raise CollectionError("direct collection budgets differ")
    sampling = request["sampling"]
    if (
        not isinstance(sampling, dict)
        or set(sampling) != {"max_generate_length", "temperature", "top_p", "top_k", "logprobs"}
        or sampling.get("max_generate_length") != limits["max_tokens_per_turn"]
        or type(sampling.get("temperature")) not in {int, float}
        or not math.isfinite(sampling["temperature"])
        or sampling["temperature"] < 0
        or type(sampling.get("top_p")) not in {int, float}
        or not 0 < sampling["top_p"] <= 1
        or type(sampling.get("top_k")) is not int
        or sampling["top_k"] < -1
        or sampling.get("logprobs") != 0
    ):
        raise CollectionError("direct collection sampling differs")
    qualification = request["qualification"]
    if (
        not isinstance(qualification, dict)
        or set(qualification) != {"max_length", "context_tokens", "synthetic_fixture_policy"}
        or type(qualification.get("max_length")) is not int
        or qualification["max_length"] <= 1
        or type(qualification.get("context_tokens")) is not int
        or qualification["context_tokens"] < 0
        or qualification.get("synthetic_fixture_policy")
        != "synthetic_content_only_no_task_prompt_trace_flag_score_or_credential"
    ):
        raise CollectionError("dense qualification policy differs")
    closure = _source_closure(request["source_closure"], relative_to)
    if request["output_policy"] != {
        "private": True,
        "create_once": True,
        "root_mode": "0700",
        "file_mode": "0600",
        "public_receipts": "digests_counts_and_booleans_only",
    } or request["execution"] != {
        "kind": "offline_qualification_not_a_launcher",
        "cluster_or_api_mutations_performed": False,
        "job_submission_performed": False,
        "credentials_required": False,
    }:
        raise CollectionError("collection qualification execution boundary differs")

    runtime = request["runtime"]
    if not isinstance(runtime, dict) or set(runtime) != {
        "collector_image",
        "base_route_certificate_sha256",
        "synthetic_parity_receipt",
    }:
        raise CollectionError("collector runtime binding fields differ")
    blockers = []
    image = runtime.get("collector_image")
    route = runtime.get("base_route_certificate_sha256")
    parity_ref = runtime.get("synthetic_parity_receipt")
    parity = None
    if image is None:
        blockers.append("missing_immutable_collector_image")
    elif not isinstance(image, str) or IMAGE.fullmatch(image) is None:
        raise CollectionError("collector image is not immutable")
    if route is None:
        blockers.append("missing_exact_base_route_certificate")
    elif not _sha(route):
        raise CollectionError("base route certificate digest is invalid")
    if interface["system_prompt_sha256"] is None:
        blockers.append("missing_direct_system_prompt_digest")
    if parity_ref is None:
        blockers.append("missing_synthetic_recorder_dense_parity")
    else:
        if not isinstance(parity_ref, dict) or set(parity_ref) != {"path", "sha256"}:
            raise CollectionError("synthetic parity binding fields differ")
        raw = parity_ref.get("path")
        parity_path = Path(raw) if isinstance(raw, str) else Path()
        parity_path = parity_path if parity_path.is_absolute() else relative_to / parity_path
        if (
            not isinstance(raw, str)
            or not raw
            or parity_path.is_symlink()
            or not parity_path.is_file()
            or file_sha256(parity_path) != parity_ref["sha256"]
        ):
            raise CollectionError("synthetic parity receipt file differs")
        parity = _read_json(parity_path)
        if not isinstance(parity, dict):
            raise CollectionError("synthetic parity receipt is not an object")
        _validate_parity_receipt(parity, model=model, interface=interface, source_closure=closure)
    expected_status = "qualified" if not blockers else "blocked_external_bindings"
    if request.get("status") != expected_status:
        raise CollectionError("collection request status differs from its blockers")
    if require_ready and blockers:
        raise CollectionError("collection request is not ready")
    return {
        "request": request,
        "plan": plan,
        "audit": audit,
        "tasks": tasks,
        "inventory": inventory,
        "roster": roster,
        "tool_catalog": catalog,
        "source_closure": closure,
        "parity": parity,
        "blockers": blockers,
    }


def _sample_fields(sample: object) -> tuple[list[int], int, list[int]]:
    def get(name: str) -> Any:
        return sample.get(name) if isinstance(sample, dict) else getattr(sample, name, None)

    tokens, response, mask = get("tokens"), get("response_length"), get("loss_mask")
    if (
        not isinstance(tokens, list)
        or type(response) is not int
        or not 0 < response < len(tokens)
        or not isinstance(mask, list)
        or len(mask) != response
        or any(type(token) is not int or token < 0 for token in tokens)
        or any(type(item) is not int or item not in {0, 1} for item in mask)
    ):
        raise CollectionError("synthetic native recorder sample differs")
    return tokens, response, mask


def prepare_private_request(
    *,
    system_prompt: str,
    task_prompt: str,
    tool_catalog: list[dict],
    tokenizer: Any,
    system_prompt_sha256: str,
    task_prompt_sha256: str,
    runtime_chat_template_sha256: str,
) -> dict[str, Any]:
    """Build the exact private system+user+tools prefill without logging content."""
    template = getattr(tokenizer, "chat_template", None)
    if (
        not isinstance(system_prompt, str)
        or not system_prompt
        or fleet.sha256(system_prompt.encode()) != system_prompt_sha256
        or not isinstance(task_prompt, str)
        or not task_prompt
        or fleet.sha256(task_prompt.encode()) != task_prompt_sha256
        or not isinstance(template, str)
        or fleet.sha256(template.encode()) != runtime_chat_template_sha256
        or not isinstance(tool_catalog, list)
        or fleet.sha256(fleet.canonical_json(tool_catalog)) != TOOL_CATALOG_SHA256
        or [row.get("name") for row in tool_catalog if isinstance(row, dict)]
        != ["bash", "submit_report"]
        or any(
            not isinstance(row, dict)
            or set(row) != {"name", "description", "inputSchema"}
            or not isinstance(row["description"], str)
            or not isinstance(row["inputSchema"], dict)
            for row in tool_catalog
        )
    ):
        raise CollectionError("private direct request binding differs")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_prompt},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": row["name"],
                "description": row["description"],
                "parameters": row["inputSchema"],
            },
        }
        for row in tool_catalog
    ]
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
        )
        tokens = list(
            tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
            )
        )
        if (
            not isinstance(rendered, str)
            or not tokens
            or any(type(token) is not int or token < 0 for token in tokens)
            or list(tokenizer.encode(rendered, add_special_tokens=False)) != tokens
        ):
            raise ValueError
    except Exception:
        raise CollectionError("private direct request rendering differs") from None
    # This return value is deliberately private and contains request content.  It
    # is an in-memory collector input, never a public receipt or CLI result.
    return {
        "messages": messages,
        "tools": tools,
        "rendered": rendered,
        "tokens": tokens,
        "rendered_sha256": fleet.sha256(rendered.encode()),
        "tokens_sha256": digest_json(tokens),
    }


def prepare_attempt(
    validated: dict[str, Any],
    attempt_id: str,
    *,
    task: dict,
    system_prompt: str,
    tokenizer: Any,
) -> dict[str, Any]:
    """Build one exact private config/prefix for ``rl_episode.collect``; do no I/O."""
    expectation = source_expectation(validated, attempt_id)
    attempt = expectation["attempt"]
    runtime = expectation["runtime_binding"]
    selected = {
        "task_key": attempt["task_key"],
        "task_version_id": attempt["task_version_id"],
        "env_key": runtime["environment"]["id"],
        "env_version": runtime["environment"]["version"],
        "environment_version_id": runtime["environment"]["version_id"],
        "data_key": runtime["environment"]["data_id"],
        "data_version": runtime["environment"]["data_version"],
    }
    try:
        task_binding, environment, verifier = fleet.bind_task(task, selected)
    except Exception:
        raise CollectionError("live task differs from the frozen collection attempt") from None
    expected_environment = runtime["environment"]
    expected_verifier = runtime["verifier"]
    if (
        any(
            environment.get(key) != _normalized_sha(value)
            for key, value in expected_environment.items()
        )
        or any(
            str(verifier.get(key)) != str(_normalized_sha(value))
            for key, value in expected_verifier.items()
        )
        or environment.get("ttl_seconds") != expectation["limits"]["ttl_seconds"]
        or task_binding.get("cyber_contract") != rl_data.AUTHORITY["required_cyber_contract"]
        or not isinstance(task.get("prompt"), str)
        or not task["prompt"]
    ):
        raise CollectionError("live task runtime or verifier differs from the frozen attempt")
    private_request = prepare_private_request(
        system_prompt=system_prompt,
        task_prompt=task["prompt"],
        tool_catalog=validated["tool_catalog"],
        tokenizer=tokenizer,
        system_prompt_sha256=expectation["interface"]["system_prompt_sha256"],
        task_prompt_sha256=task_binding["prompt_sha256"],
        runtime_chat_template_sha256=expectation["interface"]["runtime_chat_template_sha256"],
    )
    model = expectation["model"]
    limits = expectation["limits"]
    config = {
        "run_id": attempt["attempt_id"],
        "model": {
            "repo": model["repo"],
            "revision": model["revision"],
            "root": model["root"],
            "runtime_chat_template_sha256": model["runtime_chat_template_sha256"],
        },
        "authority": json.loads(json.dumps(rl_data.AUTHORITY)),
        "execution": {
            "required_task_tools": ["bash", "submit_report"],
            "required_task_tool_catalog_sha256": expectation["interface"][
                "required_task_tool_catalog_sha256"
            ],
        },
        "environment": environment,
        "rl": {key: limits[key] for key in limits if key not in {"response_tokens", "ttl_seconds"}},
        "task": task_binding,
        "verifier": verifier,
        "initial_prompt_sha256": private_request["rendered_sha256"],
        "initial_prompt_tokens_sha256": private_request["tokens_sha256"],
        "native_batch": {
            "kind": "self_trace_recollection",
            "attempt_id": attempt["attempt_id"],
            "attempt_ordinal": attempt["attempt_ordinal"],
            "collection_request_sha256": expectation["collection_request_sha256"],
        },
        "sampling": expectation["sampling"],
    }
    config["config_sha256"] = fleet.digest_without(config, "config_sha256")
    try:
        rl_episode._validate(config)
    except Exception:
        raise CollectionError("prepared collection attempt is not a direct RL binding") from None
    return {
        "config": config,
        "request_messages": private_request["messages"],
        "request_tools": private_request["tools"],
        "rendered_request": private_request["rendered"],
        "prompt_tokens": private_request["tokens"],
        "response_tokens": limits["response_tokens"],
        "expectation": expectation,
    }


def recorder_dense_parity(
    sample: object,
    conversation: dict,
    *,
    dense_reference_tokens: list[int],
    dense_reference_loss_mask: list[int],
    model: dict,
    interface: dict,
    source_closure: dict[str, str],
    fixture_sha256: str,
    max_length: int,
    context_tokens: int,
) -> dict:
    """Require exact native token/mask preservation through the first observation."""
    if (
        model
        != {
            "repo": MODEL_REPO,
            "revision": MODEL_REVISION,
            "runtime_chat_template_sha256": CHAT_TEMPLATE_SHA256,
        }
        or interface
        != {
            "required_task_tools": ["bash", "submit_report"],
            "required_task_tool_catalog_sha256": TOOL_CATALOG_SHA256,
            "compaction": "disabled",
        }
        or set(source_closure)
        != {
            "dense.py",
            "rl_data.py",
            "rl_episode.py",
            "self_trace_collection.py",
            "self_trace_corpus.py",
            "skyrl_episode.py",
        }
        or any(not _sha(value) for value in source_closure.values())
        or not _sha(fixture_sha256)
        or type(max_length) is not int
        or max_length <= 1
        or type(context_tokens) is not int
        or context_tokens < 0
    ):
        raise CollectionError("synthetic parity bindings differ")
    tokens, response, mask = _sample_fields(sample)
    prompt_length = len(tokens) - response
    if (
        not isinstance(dense_reference_tokens, list)
        or not isinstance(dense_reference_loss_mask, list)
        or len(dense_reference_tokens) != len(dense_reference_loss_mask)
        or any(type(token) is not int or token < 0 for token in dense_reference_tokens)
        or any(type(item) is not int or item not in {0, 1} for item in dense_reference_loss_mask)
    ):
        raise CollectionError("synthetic dense reference differs")
    reference_spans = self_trace_corpus._runs(dense_reference_loss_mask)
    try:
        self_trace_corpus._conversation_shape(
            conversation, reference_spans, required_anchor_roles=("system", "user")
        )
    except Exception:
        raise CollectionError("synthetic dense reference does not align") from None
    spans = self_trace_corpus._runs([0] * prompt_length + mask)
    try:
        shape = self_trace_corpus._conversation_shape(
            conversation, spans, required_anchor_roles=("system", "user")
        )
        rows, excluded = self_trace_corpus._segment_native(
            episode_id="synthetic-direct-parity",
            task_key="synthetic-direct-parity",
            tokens=tokens,
            response_length=response,
            response_mask=mask,
            shape=shape,
            max_length=max_length,
            context_tokens=context_tokens,
        )
    except Exception:
        raise CollectionError("synthetic recorder and dense adapter do not align") from None
    if len(spans) < 2 or len(reference_spans) < 2 or len(rows) != 1 or excluded:
        raise CollectionError("synthetic parity fixture lacks two retained direct targets")
    first_observation = (reference_spans[0][1], reference_spans[1][0])
    last_target_end = spans[-1][1]
    expected_mask = [0] * prompt_length + mask
    row = rows[0]
    if (
        first_observation[0] >= first_observation[1]
        or any(dense_reference_loss_mask[first_observation[0] : first_observation[1]])
        or len(tokens) < first_observation[1]
        or tokens[: first_observation[1]] != dense_reference_tokens[: first_observation[1]]
        or expected_mask[: first_observation[1]]
        != dense_reference_loss_mask[: first_observation[1]]
        or row["input_ids"] != tokens[:last_target_end]
        or row["loss_mask"] != expected_mask[:last_target_end]
        or row["target_token_count"] != sum(expected_mask[:last_target_end])
        or [item["tool"] for item in shape] != ["bash", "submit_report"]
    ):
        raise CollectionError("synthetic token or loss-mask parity differs")
    receipt = {
        "schema": PARITY_SCHEMA,
        "status": "qualified",
        "model": model,
        "interface": interface,
        "source_closure_sha256": digest_json(source_closure),
        "fixture_sha256": fixture_sha256,
        "native": {
            "prompt_tokens_sha256": digest_json(tokens[:prompt_length]),
            "recorded_tokens_sha256": digest_json(tokens),
            "recorded_loss_mask_sha256": digest_json([0] * prompt_length + mask),
            "assistant_targets": len(spans),
        },
        "dense": {
            "reference_tokens_sha256": digest_json(dense_reference_tokens),
            "reference_loss_mask_sha256": digest_json(dense_reference_loss_mask),
            "trained_prefix_tokens_sha256": digest_json(row["input_ids"]),
            "trained_prefix_loss_mask_sha256": digest_json(row["loss_mask"]),
            "trained_prefix_tokens": len(row["input_ids"]),
            "supervised_tokens": row["target_token_count"],
        },
        "first_observation": {
            "tokens": first_observation[1] - first_observation[0],
            "native_dense_token_ids_equal": True,
            "native_dense_loss_masks_equal": True,
            "all_observation_tokens_masked": True,
        },
        "privacy": {
            "synthetic_fixture_only": True,
            "prompt_or_trace_content_emitted": False,
            "tool_arguments_or_results_emitted": False,
            "flags_scores_or_credentials_emitted": False,
        },
    }
    return {**receipt, "sha256": digest_json(receipt)}


def source_expectation(validated: dict[str, Any], attempt_id: str) -> dict:
    """Project one ready request into a sealed, content-free episode expectation."""
    if validated.get("blockers") or not isinstance(attempt_id, str):
        raise CollectionError("source expectation requires a ready collection request")
    matches = [row for row in validated["roster"] if row["attempt_id"] == attempt_id]
    if len(matches) != 1:
        raise CollectionError("source attempt is absent or duplicated")
    attempt = matches[0]
    row = validated["inventory"][(attempt["task_key"], attempt["task_version_id"])]
    request = validated["request"]
    receipt = {
        "schema": EXPECTATION_SCHEMA,
        "collection_request_sha256": request["sha256"],
        "producer_plan_sha256": validated["plan"]["sha256"],
        "attempt": attempt,
        "model": request["model"],
        "interface": {
            "required_task_tools": request["interface"]["required_task_tools"],
            "required_task_tool_catalog_sha256": request["interface"]["tool_catalog"][
                "canonical_sha256"
            ],
            "runtime_chat_template_sha256": request["interface"]["runtime_chat_template_sha256"],
            "system_prompt_sha256": request["interface"]["system_prompt_sha256"],
            "compaction": "disabled",
        },
        "runtime_binding": {
            "environment": row["environment"],
            "verifier": row["verifier"],
            "current_binding_sha256": row["current_binding_sha256"],
        },
        "limits": request["limits"],
        "sampling": request["sampling"],
        "source_closure": validated["source_closure"],
        "qualification": request["qualification"],
    }
    return {**receipt, "sha256": digest_json(receipt)}


def _private_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file() or stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise CollectionError("private episode member mode or type differs")


def _normalized_sha(value: object) -> object:
    return (
        "sha256:" + value
        if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
        else value
    )


def _validate_expectation(expectation: dict) -> None:
    _sealed(expectation, EXPECTATION_SCHEMA)
    if set(expectation) != {
        "schema",
        "collection_request_sha256",
        "producer_plan_sha256",
        "attempt",
        "model",
        "interface",
        "runtime_binding",
        "limits",
        "sampling",
        "source_closure",
        "qualification",
        "sha256",
    }:
        raise CollectionError("source expectation fields differ")
    attempt = expectation.get("attempt")
    model = expectation.get("model")
    interface = expectation.get("interface")
    runtime = expectation.get("runtime_binding")
    limits = expectation.get("limits")
    sampling = expectation.get("sampling")
    qualification = expectation.get("qualification")
    if (
        not _sha(expectation.get("collection_request_sha256"))
        or not _sha(expectation.get("producer_plan_sha256"))
        or not isinstance(attempt, dict)
        or set(attempt) != {"attempt_id", "attempt_ordinal", "task_key", "task_version_id"}
        or not isinstance(attempt.get("attempt_id"), str)
        or re.fullmatch(r"q38-self-[0-9a-f]{20}-a[1-9][0-9]*", attempt["attempt_id"]) is None
        or type(attempt.get("attempt_ordinal")) is not int
        or attempt["attempt_ordinal"] < 0
        or not attempt["attempt_id"].endswith(f"-a{attempt['attempt_ordinal'] + 1}")
        or not isinstance(attempt.get("task_key"), str)
        or not attempt["task_key"]
        or not self_trace_corpus._canonical_uuid(attempt.get("task_version_id"))
        or not isinstance(model, dict)
        or set(model)
        != {
            "repo",
            "revision",
            "root",
            "lock_file_sha256",
            "runtime_chat_template_sha256",
            "initialization",
        }
        or model.get("repo") != MODEL_REPO
        or model.get("revision") != MODEL_REVISION
        or not isinstance(model.get("root"), str)
        or not model["root"].startswith("/")
        or not _sha(model.get("lock_file_sha256"))
        or model.get("runtime_chat_template_sha256") != CHAT_TEMPLATE_SHA256
        or model.get("initialization") != "exact_fresh_base"
        or not isinstance(interface, dict)
        or set(interface)
        != {
            "required_task_tools",
            "required_task_tool_catalog_sha256",
            "runtime_chat_template_sha256",
            "system_prompt_sha256",
            "compaction",
        }
        or interface.get("required_task_tools") != ["bash", "submit_report"]
        or interface.get("required_task_tool_catalog_sha256") != TOOL_CATALOG_SHA256
        or interface.get("runtime_chat_template_sha256") != CHAT_TEMPLATE_SHA256
        or not _sha(interface.get("system_prompt_sha256"))
        or interface.get("compaction") != "disabled"
        or not isinstance(runtime, dict)
        or set(runtime) != {"environment", "verifier", "current_binding_sha256"}
        or not isinstance(runtime.get("environment"), dict)
        or set(runtime["environment"])
        != {
            "id",
            "version",
            "version_id",
            "data_id",
            "data_version",
            "runtime_seed_content_sha256",
        }
        or any(
            not isinstance(runtime["environment"].get(name), str)
            or not runtime["environment"][name]
            for name in ("id", "version", "data_id", "data_version")
        )
        or not self_trace_corpus._canonical_uuid(runtime["environment"].get("version_id"))
        or not _sha(_normalized_sha(runtime["environment"].get("runtime_seed_content_sha256")))
        or not isinstance(runtime.get("verifier"), dict)
        or set(runtime["verifier"]) != {"id", "version_id", "version", "sha256"}
        or not self_trace_corpus._canonical_uuid(runtime["verifier"].get("id"))
        or not self_trace_corpus._canonical_uuid(runtime["verifier"].get("version_id"))
        or not isinstance(runtime["verifier"].get("version"), (str, int))
        or isinstance(runtime["verifier"].get("version"), bool)
        or not str(runtime["verifier"]["version"])
        or not _sha(_normalized_sha(runtime["verifier"].get("sha256")))
        or not _sha(runtime.get("current_binding_sha256"))
        or not isinstance(limits, dict)
        or set(limits)
        != {
            "context_tokens",
            "response_tokens",
            "max_tokens_per_turn",
            "max_turns",
            "episode_seconds",
            "tool_seconds",
            "tool_result_chars",
            "ttl_seconds",
        }
        or any(type(item) is not int or item <= 0 for item in limits.values())
        or not limits["max_tokens_per_turn"]
        <= limits["response_tokens"]
        < limits["context_tokens"]
        <= 98304
        or not limits["episode_seconds"] + 240 < limits["ttl_seconds"] <= 32400
        or not isinstance(sampling, dict)
        or set(sampling) != {"max_generate_length", "temperature", "top_p", "top_k", "logprobs"}
        or sampling.get("max_generate_length") != limits["max_tokens_per_turn"]
        or type(sampling.get("temperature")) not in {int, float}
        or isinstance(sampling.get("temperature"), bool)
        or not math.isfinite(sampling["temperature"])
        or sampling["temperature"] < 0
        or type(sampling.get("top_p")) not in {int, float}
        or isinstance(sampling.get("top_p"), bool)
        or not math.isfinite(sampling["top_p"])
        or not 0 < sampling["top_p"] <= 1
        or type(sampling.get("top_k")) is not int
        or sampling["top_k"] < -1
        or sampling.get("logprobs") != 0
        or not isinstance(qualification, dict)
        or set(qualification) != {"max_length", "context_tokens", "synthetic_fixture_policy"}
        or type(qualification.get("max_length")) is not int
        or qualification["max_length"] <= 1
        or type(qualification.get("context_tokens")) is not int
        or qualification["context_tokens"] < 0
        or qualification.get("synthetic_fixture_policy")
        != "synthetic_content_only_no_task_prompt_trace_flag_score_or_credential"
    ):
        raise CollectionError("source expectation is incomplete or incompatible")
    if set(expectation.get("source_closure", {})) != {
        "dense.py",
        "rl_data.py",
        "rl_episode.py",
        "self_trace_collection.py",
        "self_trace_corpus.py",
        "skyrl_episode.py",
    } or any(not _sha(value) for value in expectation["source_closure"].values()):
        raise CollectionError("source expectation code closure differs")
    actual_closure = {
        name: file_sha256(Path(__file__).with_name(name)) for name in expectation["source_closure"]
    }
    if expectation["source_closure"] != actual_closure:
        raise CollectionError("source expectation code bytes differ")


def review_source(
    expectation: dict,
    episode_directory: Path,
    *,
    tokenizer: Any,
    tool_catalog: list[dict],
) -> dict:
    """Return a sealed source receipt without returning any private source value."""
    _validate_expectation(expectation)
    if episode_directory.is_symlink() or not episode_directory.is_dir():
        raise CollectionError("private episode directory is missing")
    if stat.S_IMODE(episode_directory.stat().st_mode) != 0o700:
        raise CollectionError("private episode directory mode differs")
    required = {
        "ACCEPTED.json",
        "binding.json",
        "cleanup.json",
        "conversation.json",
        "create-intent.json",
        "instance.json",
        "recording.json",
        "reward.json",
        "score-intent.json",
    }
    members = {path.name for path in episode_directory.iterdir()}
    if members != required:
        raise CollectionError("private episode file inventory differs")
    for name in required:
        _private_file(episode_directory / name)
    accepted = _read_json(episode_directory / "ACCEPTED.json")
    attempt = expectation["attempt"]
    entry = {
        "episode_id": attempt["attempt_id"],
        "directory": attempt["attempt_id"],
        "task_key": attempt["task_key"],
        "task_version_id": attempt["task_version_id"],
        "accepted_sha256": file_sha256(episode_directory / "ACCEPTED.json"),
    }
    model = expectation["model"]
    interface = expectation["interface"]
    index = {
        "model": {"repo": model["repo"], "revision": model["revision"], "root": model["root"]},
        "interface": {
            "runtime_chat_template_sha256": interface["runtime_chat_template_sha256"],
            "required_task_tool_catalog_sha256": interface["required_task_tool_catalog_sha256"],
        },
    }
    # _episode needs a root plus relative directory and performs the authoritative
    # acceptance/reward/cleanup/native-token/dense checks.  The review directory
    # itself is the exact attempt directory, so its parent is the source root.
    try:
        candidate = self_trace_corpus._episode(
            episode_directory.parent,
            entry,
            index,
            {(attempt["task_key"], attempt["task_version_id"])},
            max_length=expectation["qualification"]["max_length"],
            context_tokens=expectation["qualification"]["context_tokens"],
        )
    except Exception:
        raise CollectionError("private episode failed direct-source qualification") from None
    binding = _read_json(episode_directory / "binding.json")
    conversation = _read_json(episode_directory / "conversation.json")
    create_intent = _read_json(episode_directory / "create-intent.json")
    instance = _read_json(episode_directory / "instance.json")
    score_intent = _read_json(episode_directory / "score-intent.json")
    messages = conversation.get("messages") if isinstance(conversation, dict) else None
    runtime = expectation["runtime_binding"]
    expected_environment = runtime["environment"]
    expected_verifier = runtime["verifier"]
    environment = binding.get("environment") if isinstance(binding, dict) else None
    verifier = binding.get("verifier") if isinstance(binding, dict) else None
    batch = binding.get("native_batch") if isinstance(binding, dict) else None
    expected_batch = {
        "kind": "self_trace_recollection",
        "attempt_id": attempt["attempt_id"],
        "attempt_ordinal": attempt["attempt_ordinal"],
        "collection_request_sha256": expectation["collection_request_sha256"],
    }
    try:
        expected_score_intent = fleet.build_scoring_payload(
            binding,
            instance_id=instance["instance_id"],
            final_answer="",
            messages=[],
        )
    except Exception:
        raise CollectionError("private episode scoring intent is invalid") from None
    if (
        create_intent != {"run_id": attempt["attempt_id"]}
        or score_intent != expected_score_intent
        or binding.get("sampling") != expectation["sampling"]
        or batch != expected_batch
        or binding.get("rl")
        != {
            key: expectation["limits"][key]
            for key in expectation["limits"]
            if key not in {"response_tokens", "ttl_seconds"}
        }
        or environment.get("ttl_seconds") != expectation["limits"]["ttl_seconds"]
        or any(
            environment.get(key) != _normalized_sha(value)
            for key, value in expected_environment.items()
        )
        or any(
            str(verifier.get(key)) != str(_normalized_sha(value))
            for key, value in expected_verifier.items()
        )
        or verifier.get("function_name") != "verify"
        or not isinstance(messages, list)
        or len(messages) < 2
        or [message.get("role") for message in messages[:2]] != ["system", "user"]
        or not isinstance(messages[0].get("content"), str)
        or fleet.sha256(messages[0]["content"].encode()) != interface["system_prompt_sha256"]
    ):
        raise CollectionError("private episode differs from its collection request")
    try:
        private_request = prepare_private_request(
            system_prompt=messages[0]["content"],
            task_prompt=messages[1]["content"],
            tool_catalog=tool_catalog,
            tokenizer=tokenizer,
            system_prompt_sha256=interface["system_prompt_sha256"],
            task_prompt_sha256=binding["task"]["prompt_sha256"],
            runtime_chat_template_sha256=interface["runtime_chat_template_sha256"],
        )
        recording = _read_json(episode_directory / "recording.json")
        samples = recording["samples"]
        sample = samples[0]
        prompt_length = len(sample["tokens"]) - sample["response_length"]
    except Exception:
        raise CollectionError("private episode request rendering is invalid") from None
    if (
        private_request["rendered_sha256"] != binding["initial_prompt_sha256"]
        or private_request["tokens_sha256"] != binding["initial_prompt_tokens_sha256"]
        or private_request["tokens"] != sample["tokens"][:prompt_length]
    ):
        raise CollectionError("private episode request rendering differs")
    files = {name: file_sha256(episode_directory / name) for name in sorted(required)}
    receipt = {
        "schema": SOURCE_RECEIPT_SCHEMA,
        "status": "qualified",
        "collection_request_sha256": expectation["collection_request_sha256"],
        "producer_plan_sha256": expectation["producer_plan_sha256"],
        "expectation_sha256": expectation["sha256"],
        "attempt": attempt,
        "accepted_sha256": entry["accepted_sha256"],
        "accepted_receipt_sha256": accepted.get("sha256"),
        "private_file_set_sha256": digest_json(files),
        "request": {
            "roles": ["system", "user"],
            "system_prompt_sha256": interface["system_prompt_sha256"],
            "initial_rendered_prompt_sha256": binding["initial_prompt_sha256"],
            "initial_prompt_tokens_sha256": binding["initial_prompt_tokens_sha256"],
            "ordered_tool_catalog_sha256": interface["required_task_tool_catalog_sha256"],
            "conversation_sha256": files["conversation.json"],
            "compaction": "disabled",
        },
        "outcome": {
            "authoritative_positive": True,
            "verifier_execution_bound": True,
            "environment_release_confirmed": True,
            "possible_environment_leak": False,
        },
        "targets": {
            "assistant_responses": candidate["assistant_responses"],
            "supervised_tokens": candidate["supervised_tokens"],
            "submit_report_responses": candidate["submit_report_responses"],
            "contains_retained_bash_response": True,
        },
        "source_closure_sha256": digest_json(expectation["source_closure"]),
        "privacy": {
            "prompt_or_trace_content_emitted": False,
            "tool_arguments_or_results_emitted": False,
            "flags_scores_or_credentials_emitted": False,
        },
    }
    return {**receipt, "sha256": digest_json(receipt)}


def main(argv: list[str] | None = None) -> int:
    """Validate one request without loading a prompt, trace, score, or credential."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args(argv)
    try:
        request = _read_json(args.request)
        if not isinstance(request, dict):
            raise CollectionError("collection request is not an object")
        validated = validate_request(
            request,
            relative_to=args.request.parent,
            require_ready=args.require_ready,
        )
    except Exception:
        print(json.dumps({"status": "rejected", "reason": "collection_request_not_qualified"}))
        return 2
    print(
        json.dumps(
            {
                "status": request["status"],
                "request_sha256": request["sha256"],
                "task_versions": len(validated["tasks"]),
                "maximum_sessions": len(validated["roster"]),
                "blockers": validated["blockers"],
                "cluster_or_api_mutations_performed": False,
                "job_submission_performed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
