"""Versioned user-only Qwen3.8 direct self-trace collection gate.

V1 is an immutable record of the earlier system+user proposal.  This module
binds that frozen task roster and runtime policy, but resolves the scientific
prompt question conservatively: the caller supplies no explicit system message,
matching the repository's normal ``skyrl_direct`` data path.  The pinned Qwen
chat-template bytes still render their own template-defined tool/system framing.

This remains a validator and private preparation/review library, not a launcher.
It performs no cluster, API, registry, or job mutation.
"""

from __future__ import annotations

import json
import re
import stat
from pathlib import Path
from typing import Any

from evals.fleet import opencode_self_hosted as fleet

from . import dense, rl_data, rl_episode, self_trace_corpus
from . import self_trace_collection as v1
from .io import digest_json, file_sha256

REQUEST_SCHEMA = "cyber_qwen_direct_self_trace_collection_request_v2"
STUDY_SCHEMA = "cyber_qwen_self_sft_recollection_plan_v2"
PARITY_SCHEMA = "cyber_qwen_direct_recorder_dense_parity_v2"
EXPECTATION_SCHEMA = "cyber_qwen_direct_self_trace_source_expectation_v2"
SOURCE_RECEIPT_SCHEMA = "cyber_qwen_direct_self_trace_source_receipt_v2"
COLLECTOR_QUALIFICATION_SCHEMA = "cyber_qwen_direct_collector_qualification_v1"
ROUTE_CERTIFICATE_SCHEMA = "cyber_qwen_direct_base_route_certificate_v1"

MODEL_REPO = v1.MODEL_REPO
MODEL_REVISION = v1.MODEL_REVISION
CHAT_TEMPLATE_SHA256 = v1.CHAT_TEMPLATE_SHA256
TOOL_CATALOG_SHA256 = v1.TOOL_CATALOG_SHA256
WEIGHTS_MANIFEST_SHA256 = "sha256:06c94e47c0e31fd331ed410665c830ab1b657f90f15a1b11e7bc45e2de00f352"
TOKENIZER_MANIFEST_SHA256 = (
    "sha256:3938a9a8172f2738fed1be44efc11e2562059269d50d3721213f44802b53b4e1"
)
SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
IMAGE = re.compile(r"[^\s]+@sha256:[0-9a-f]{64}\Z")

PROMPT_POLICY = {
    "version": "qwen38_skyrl_direct_user_only_v2",
    "roles": ["user"],
    "explicit_system_message": "absent",
    "task_prompt_source": "exact_live_task_bound_by_task_prompt_sha256",
    "template_defined_framing": "bound_by_runtime_chat_template_sha256",
    "compaction": "disabled",
    "tool_rewriting": "prohibited",
}
PROMPT_POLICY_SHA256 = digest_json(PROMPT_POLICY)

TEMPLATE_INVOCATIONS = {
    "rendered_text": {
        "tools_canonical_sha256": TOOL_CATALOG_SHA256,
        "tokenize": False,
        "add_generation_prompt": True,
    },
    "prompt_token_ids": {
        "tools_canonical_sha256": TOOL_CATALOG_SHA256,
        "tokenize": True,
        "return_dict": False,
        "add_generation_prompt": True,
    },
    "omitted_optional_kwargs": [
        "enable_thinking",
        "reasoning_effort",
        "preserve_thinking",
    ],
}
TEMPLATE_INVOCATIONS_SHA256 = digest_json(TEMPLATE_INVOCATIONS)

RENDERED_REQUEST_CONTRACT = {
    "task_prompt_sha256": "required_from_exact_task_binding",
    "rendered_utf8_sha256": "required_as_initial_prompt_sha256",
    "prompt_token_ids_sha256": ("required_as_initial_prompt_tokens_sha256_using_canonical_json"),
    "rendered_reencode_equals_prompt_token_ids": True,
    "recorder_prefill_equals_prompt_token_ids": True,
    "retokenization_or_text_rewrite": "prohibited",
}
RENDERED_REQUEST_CONTRACT_SHA256 = digest_json(RENDERED_REQUEST_CONTRACT)

PARITY_INTERFACE = {
    "roles": ["user"],
    "explicit_system_message": "absent",
    "required_task_tools": ["bash", "submit_report"],
    "required_task_tool_catalog_sha256": TOOL_CATALOG_SHA256,
    "runtime_chat_template_sha256": CHAT_TEMPLATE_SHA256,
    "prompt_policy_sha256": PROMPT_POLICY_SHA256,
    "template_invocations_sha256": TEMPLATE_INVOCATIONS_SHA256,
    "rendered_request_contract_sha256": RENDERED_REQUEST_CONTRACT_SHA256,
    "compaction": "disabled",
}

SOURCE_CLOSURE_PATHS = {
    **v1.SOURCE_CLOSURE_PATHS,
    "self_trace_collection_v2.py": Path(__file__),
}


class CollectionError(v1.CollectionError):
    """A fixed, payload-free v2 collection qualification rejection."""


def _sha(value: object) -> bool:
    return isinstance(value, str) and SHA256.fullmatch(value) is not None


def _sealed(value: dict, schema: str) -> None:
    if (
        not isinstance(value, dict)
        or value.get("schema") != schema
        or value.get("sha256")
        != digest_json({key: item for key, item in value.items() if key != "sha256"})
    ):
        raise CollectionError("sealed v2 collection metadata differs")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        raise CollectionError("bound v2 collection JSON is unreadable") from None


def _bound(reference: object, relative_to: Path, *, schema: str) -> tuple[Path, dict]:
    try:
        path, value = v1._bound(reference, relative_to, sealed=True)
    except Exception:
        raise CollectionError("immutable artifact binding differs") from None
    if not isinstance(value, dict):
        raise CollectionError("bound immutable artifact is not an object")
    _sealed(value, schema)
    return path, value


def _source_closure(value: object, relative_to: Path) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != set(SOURCE_CLOSURE_PATHS):
        raise CollectionError("v2 collector source closure fields differ")
    result = {}
    for name, reference in value.items():
        if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
            raise CollectionError("v2 collector source binding fields differ")
        raw = reference.get("path")
        path = Path(raw) if isinstance(raw, str) else Path()
        path = path if path.is_absolute() else relative_to / path
        if (
            not isinstance(raw, str)
            or not raw
            or path.is_symlink()
            or not path.is_file()
            or path.name != name
            or not _sha(reference.get("sha256"))
            or file_sha256(path) != reference["sha256"]
        ):
            raise CollectionError("v2 collector source file digest differs")
        result[name] = reference["sha256"]
    return result


def _validate_study(study: dict, *, base_plan_sha256: str, base_request_sha256: str) -> None:
    _sealed(study, STUDY_SCHEMA)
    if (
        set(study)
        != {
            "schema",
            "status",
            "base_plan_sha256",
            "base_collection_request_sha256",
            "scientific_treatment",
            "remaining_external_artifacts",
            "decision",
            "execution",
            "sha256",
        }
        or study.get("status") != "blocked_external_runtime_artifacts"
        or study.get("base_plan_sha256") != base_plan_sha256
        or study.get("base_collection_request_sha256") != base_request_sha256
        or study.get("scientific_treatment")
        != {
            "prompt_policy": PROMPT_POLICY,
            "prompt_policy_sha256": PROMPT_POLICY_SHA256,
            "model": {
                "repo": MODEL_REPO,
                "revision": MODEL_REVISION,
                "runtime_chat_template_sha256": CHAT_TEMPLATE_SHA256,
            },
            "ordered_tool_catalog_sha256": TOOL_CATALOG_SHA256,
            "template_invocations": TEMPLATE_INVOCATIONS,
            "template_invocations_sha256": TEMPLATE_INVOCATIONS_SHA256,
            "rendered_request_contract": RENDERED_REQUEST_CONTRACT,
            "rendered_request_contract_sha256": RENDERED_REQUEST_CONTRACT_SHA256,
        }
        or study.get("remaining_external_artifacts")
        != [
            "immutable_skyrl_direct_collector_qualification",
            "exact_collection_time_direct_base_route_certificate",
        ]
        or study.get("decision")
        != {
            "kind": "conservative_repo_defined_direct_default",
            "human_scientific_choice_required": False,
            "candidate_gpt_system_prompt_selected": False,
        }
        or study.get("execution")
        != {
            "kind": "metadata_only_not_a_job_request",
            "launchable": False,
            "cluster_or_api_mutations_performed": False,
            "job_submission_performed": False,
        }
    ):
        raise CollectionError("v2 study treatment differs")


def _validate_parity_receipt(
    parity: dict,
    *,
    model: dict,
    source_closure: dict[str, str],
    qualification: dict,
) -> None:
    _sealed(parity, PARITY_SCHEMA)
    native = parity.get("native")
    dense_receipt = parity.get("dense")
    policy = parity.get("dense_policy")
    observation = parity.get("first_observation")
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
            "dense_policy",
            "first_observation",
            "qualification_scope",
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
        or parity.get("interface") != PARITY_INTERFACE
        or parity.get("source_closure_sha256") != digest_json(source_closure)
        or not isinstance(native, dict)
        or set(native)
        != {
            "prompt_tokens_sha256",
            "recorded_tokens_sha256",
            "recorded_loss_mask_sha256",
            "assistant_targets",
        }
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
        or any(not _sha(value) for key, value in native.items() if key.endswith("sha256"))
        or any(not _sha(value) for key, value in dense_receipt.items() if key.endswith("sha256"))
        or native.get("recorded_tokens_sha256") != dense_receipt.get("reference_tokens_sha256")
        or native.get("recorded_loss_mask_sha256")
        != dense_receipt.get("reference_loss_mask_sha256")
        or type(native.get("assistant_targets")) is not int
        or native["assistant_targets"] < 2
        or type(dense_receipt.get("trained_prefix_tokens")) is not int
        or dense_receipt["trained_prefix_tokens"] <= 0
        or type(dense_receipt.get("supervised_tokens")) is not int
        or dense_receipt["supervised_tokens"] <= 0
        or policy
        != {
            "format": dense.FORMAT,
            "max_length": qualification["max_length"],
            "context_tokens": qualification["context_tokens"],
        }
        or observation
        != {
            "tokens": observation.get("tokens") if isinstance(observation, dict) else None,
            "native_dense_token_ids_equal": True,
            "native_dense_loss_masks_equal": True,
            "all_observation_tokens_masked": True,
        }
        or type(observation.get("tokens")) is not int
        or observation["tokens"] <= 0
        or parity.get("qualification_scope")
        != {
            "kind": "skyrl_recorder_adapter_to_dense_synthetic_fixture",
            "proves": "user_only_recorded_token_and_loss_mask_preservation_across_two_direct_turns",
            "target_model_weights_loaded": False,
            "target_runtime_chat_template_loaded": False,
            "pinned_native_helper_loaded": False,
            "collector_image_or_base_route_used": False,
        }
        or parity.get("privacy")
        != {
            "synthetic_fixture_only": True,
            "prompt_or_trace_content_emitted": False,
            "tool_arguments_or_results_emitted": False,
            "flags_scores_or_credentials_emitted": False,
        }
        or parity.get("fixture_sha256")
        != digest_json(
            {
                "schema": "cyber_qwen_direct_recorder_dense_fixture_v2",
                "prompt_policy_sha256": PROMPT_POLICY_SHA256,
                "native": native,
                "dense": dense_receipt,
                "dense_policy": policy,
                "first_observation": observation,
            }
        )
    ):
        raise CollectionError("v2 synthetic parity receipt fields differ")


def _validate_collector_qualification(
    qualification: dict,
    *,
    model: dict,
    source_closure: dict[str, str],
) -> str:
    _sealed(qualification, COLLECTOR_QUALIFICATION_SCHEMA)
    image = qualification.get("image")
    bindings = qualification.get("bindings")
    checks = qualification.get("qualification")
    zero_gpu = checks.get("zero_gpu_dev") if isinstance(checks, dict) else None
    if (
        set(qualification)
        != {
            "schema",
            "status",
            "observed_at",
            "image",
            "bindings",
            "qualification",
            "sha256",
        }
        or qualification.get("status") != "qualified"
        or not isinstance(qualification.get("observed_at"), str)
        or not qualification["observed_at"]
        or not isinstance(image, dict)
        or set(image) != {"reference", "manifest_digest", "runtime_image_id", "platform"}
        or IMAGE.fullmatch(str(image.get("reference", ""))) is None
        or IMAGE.fullmatch(str(image.get("runtime_image_id", ""))) is None
        or not _sha(image.get("manifest_digest"))
        or not image["reference"].endswith("@" + image["manifest_digest"])
        or not image["runtime_image_id"].endswith("@" + image["manifest_digest"])
        or image.get("platform") != "linux/amd64"
        or bindings
        != {
            "source_closure_sha256": digest_json(source_closure),
            "backend": "skyrl_direct",
            "native_helper_sha256": dense.NATIVE_HELPER_SHA,
            "model_repo": model["repo"],
            "model_revision": model["revision"],
            "runtime_chat_template_sha256": CHAT_TEMPLATE_SHA256,
            "required_task_tools": ["bash", "submit_report"],
            "required_task_tool_catalog_sha256": TOOL_CATALOG_SHA256,
            "prompt_policy_sha256": PROMPT_POLICY_SHA256,
            "template_invocations_sha256": TEMPLATE_INVOCATIONS_SHA256,
            "rendered_request_contract_sha256": RENDERED_REQUEST_CONTRACT_SHA256,
        }
        or not isinstance(checks, dict)
        or set(checks)
        != {
            "local_synthetic_exit_code",
            "clean_pull_by_digest",
            "zero_gpu_dev",
            "target_model_weights_loaded",
            "task_or_scoring_requests",
            "registry_publications",
        }
        or checks.get("local_synthetic_exit_code") != 0
        or checks.get("clean_pull_by_digest") is not True
        or checks.get("target_model_weights_loaded") is not False
        or checks.get("task_or_scoring_requests") != 0
        or checks.get("registry_publications") != 0
        or not isinstance(zero_gpu, dict)
        or set(zero_gpu)
        != {
            "pod_uid",
            "image_pull_policy",
            "runtime_image_id_matches",
            "gpu_requests",
            "exit_code",
            "restart_count",
            "deleted",
            "absence_confirmed",
        }
        or not self_trace_corpus._canonical_uuid(zero_gpu.get("pod_uid"))
        or zero_gpu.get("image_pull_policy") != "Always"
        or zero_gpu.get("runtime_image_id_matches") is not True
        or zero_gpu.get("gpu_requests") != 0
        or zero_gpu.get("exit_code") != 0
        or zero_gpu.get("restart_count") != 0
        or zero_gpu.get("deleted") is not True
        or zero_gpu.get("absence_confirmed") is not True
    ):
        raise CollectionError("collector qualification artifact differs")
    return image["reference"]


def _validate_route_certificate(
    certificate: dict,
    *,
    model: dict,
    collector_qualification_sha256: str,
    collector_image: str,
) -> None:
    _sealed(certificate, ROUTE_CERTIFICATE_SCHEMA)
    route = certificate.get("route")
    artifact = certificate.get("model_artifact")
    collector = certificate.get("collector")
    transport = certificate.get("native_transport")
    observation = certificate.get("fresh_observation")
    if (
        set(certificate)
        != {
            "schema",
            "status",
            "observed_at",
            "route",
            "model_artifact",
            "collector",
            "interface",
            "native_transport",
            "fresh_observation",
            "sha256",
        }
        or certificate.get("status") != "qualified"
        or not isinstance(certificate.get("observed_at"), str)
        or not certificate["observed_at"]
        or not isinstance(route, dict)
        or set(route)
        != {
            "served_model_id",
            "serving_block_kind",
            "object_identity_sha256",
            "normalized_server_arguments_sha256",
            "serving_image_digest",
            "ready_replicas",
        }
        or not isinstance(route.get("served_model_id"), str)
        or not route["served_model_id"]
        or route.get("serving_block_kind") not in {"shared", "dedicated"}
        or any(
            not _sha(route.get(key))
            for key in (
                "object_identity_sha256",
                "normalized_server_arguments_sha256",
                "serving_image_digest",
            )
        )
        or type(route.get("ready_replicas")) is not int
        or route["ready_replicas"] <= 0
        or not isinstance(artifact, dict)
        or set(artifact)
        != {
            "repo",
            "revision",
            "lock_file_sha256",
            "weights_manifest_sha256",
            "tokenizer_manifest_sha256",
            "chat_template_sha256",
            "payload_rehashed",
            "symlinks_absent",
        }
        or artifact.get("repo") != model["repo"]
        or artifact.get("revision") != model["revision"]
        or artifact.get("lock_file_sha256") != model["lock_file_sha256"]
        or artifact.get("chat_template_sha256") != CHAT_TEMPLATE_SHA256
        or artifact.get("weights_manifest_sha256") != WEIGHTS_MANIFEST_SHA256
        or artifact.get("tokenizer_manifest_sha256") != TOKENIZER_MANIFEST_SHA256
        or artifact.get("payload_rehashed") is not True
        or artifact.get("symlinks_absent") is not True
        or collector
        != {
            "qualification_sha256": collector_qualification_sha256,
            "image": collector_image,
        }
        or certificate.get("interface") != PARITY_INTERFACE
        or transport
        != {
            "endpoint": "/inference/v1/generate",
            "request_contract": "messages_tools_sampling_params",
            "response_contract": ("responses_response_ids_response_logprobs_stop_reasons"),
            "native_prompt_token_ids": True,
            "native_response_token_ids": True,
            "native_response_logprobs": True,
            "collector_rendered_bytes_verified": True,
            "prompt_or_tool_rewriting": False,
        }
        or observation
        != {
            "route_ready": True,
            "model_revision_readback": True,
            "runtime_identity_readback": True,
            "collector_to_route_synthetic_probe_passed": True,
            "target_task_prompt_used": False,
            "task_or_scoring_requests": 0,
        }
    ):
        raise CollectionError("direct base-route certificate artifact differs")


def validate_request(
    request: dict, *, relative_to: Path, require_ready: bool = False
) -> dict[str, Any]:
    """Validate v2 and return metadata plus immutable artifact evidence."""
    _sealed(request, REQUEST_SCHEMA)
    if set(request) != {
        "schema",
        "status",
        "study",
        "base_request",
        "model",
        "interface",
        "runtime",
        "source_closure",
        "execution",
        "sha256",
    }:
        raise CollectionError("v2 collection request fields differ")
    base_path, base_request = _bound(request["base_request"], relative_to, schema=v1.REQUEST_SCHEMA)
    try:
        base = v1.validate_request(base_request, relative_to=base_path.parent)
    except Exception:
        raise CollectionError("frozen v1 collection request differs") from None
    if base["blockers"] != [
        "missing_immutable_collector_image",
        "missing_exact_base_route_certificate",
        "missing_direct_system_prompt_digest",
    ]:
        raise CollectionError("frozen v1 blockers differ")
    _, study = _bound(request["study"], relative_to, schema=STUDY_SCHEMA)
    _validate_study(
        study,
        base_plan_sha256=base["plan"]["sha256"],
        base_request_sha256=base_request["sha256"],
    )
    model = request.get("model")
    if model != base_request["model"]:
        raise CollectionError("v2 model identity differs from frozen roster")
    interface = request.get("interface")
    expected_interface = {
        "backend": "skyrl_direct",
        "required_task_tools": ["bash", "submit_report"],
        "tool_catalog": base_request["interface"]["tool_catalog"],
        "runtime_chat_template_sha256": CHAT_TEMPLATE_SHA256,
        "native_helper_sha256": dense.NATIVE_HELPER_SHA,
        "prompt_policy": PROMPT_POLICY,
        "prompt_policy_sha256": PROMPT_POLICY_SHA256,
        "template_invocations": TEMPLATE_INVOCATIONS,
        "template_invocations_sha256": TEMPLATE_INVOCATIONS_SHA256,
        "rendered_request_contract": RENDERED_REQUEST_CONTRACT,
        "rendered_request_contract_sha256": RENDERED_REQUEST_CONTRACT_SHA256,
    }
    if interface != expected_interface:
        raise CollectionError("v2 direct prompt interface differs")
    source_closure = _source_closure(request["source_closure"], relative_to)
    runtime = request.get("runtime")
    if not isinstance(runtime, dict) or set(runtime) != {
        "collector_qualification",
        "direct_route_certificate",
        "synthetic_parity_receipt",
    }:
        raise CollectionError("v2 runtime artifact fields differ")
    _, parity = _bound(runtime["synthetic_parity_receipt"], relative_to, schema=PARITY_SCHEMA)
    _validate_parity_receipt(
        parity,
        model=model,
        source_closure=source_closure,
        qualification=base_request["qualification"],
    )

    blockers = []
    collector_reference = runtime["collector_qualification"]
    collector = collector_image = None
    if collector_reference is None:
        blockers.append("missing_immutable_collector_qualification")
    else:
        _, collector = _bound(
            collector_reference,
            relative_to,
            schema=COLLECTOR_QUALIFICATION_SCHEMA,
        )
        collector_image = _validate_collector_qualification(
            collector,
            model=model,
            source_closure=source_closure,
        )

    route_reference = runtime["direct_route_certificate"]
    route = None
    if route_reference is None:
        blockers.append("missing_exact_direct_route_certificate")
    elif collector is None or collector_image is None:
        raise CollectionError("route certificate requires collector qualification")
    else:
        _, route = _bound(
            route_reference,
            relative_to,
            schema=ROUTE_CERTIFICATE_SCHEMA,
        )
        _validate_route_certificate(
            route,
            model=model,
            collector_qualification_sha256=collector["sha256"],
            collector_image=collector_image,
        )

    if request.get("execution") != {
        "kind": "offline_qualification_not_a_launcher",
        "cluster_or_api_mutations_performed": False,
        "job_submission_performed": False,
        "credentials_required": False,
    }:
        raise CollectionError("v2 collection execution boundary differs")
    expected_status = "qualified" if not blockers else "blocked_external_bindings"
    if request.get("status") != expected_status:
        raise CollectionError("v2 collection status differs from its blockers")
    if require_ready and blockers:
        raise CollectionError("v2 collection request is not ready")
    return {
        **base,
        "request": request,
        "base_request": base_request,
        "study": study,
        "source_closure": source_closure,
        "parity": parity,
        "collector_qualification": collector,
        "collector_image": collector_image,
        "route_certificate": route,
        "blockers": blockers,
    }


def prepare_private_request(
    *,
    task_prompt: str,
    tool_catalog: list[dict],
    tokenizer: Any,
    task_prompt_sha256: str,
    runtime_chat_template_sha256: str,
) -> dict[str, Any]:
    """Render the exact user-only Qwen request in memory without emitting content."""
    template = getattr(tokenizer, "chat_template", None)
    if (
        not isinstance(task_prompt, str)
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
        raise CollectionError("private user-only request binding differs")
    messages = [{"role": "user", "content": task_prompt}]
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
        raise CollectionError("private user-only request rendering differs") from None
    return {
        "messages": messages,
        "tools": tools,
        "rendered": rendered,
        "tokens": tokens,
        "task_prompt_sha256": task_prompt_sha256,
        "rendered_sha256": fleet.sha256(rendered.encode()),
        "tokens_sha256": digest_json(tokens),
        "prompt_policy_sha256": PROMPT_POLICY_SHA256,
        "template_invocations_sha256": TEMPLATE_INVOCATIONS_SHA256,
        "rendered_request_contract_sha256": RENDERED_REQUEST_CONTRACT_SHA256,
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
    max_length: int,
    context_tokens: int,
) -> dict:
    """Require exact user-only native token/mask preservation."""
    if (
        model
        != {
            "repo": MODEL_REPO,
            "revision": MODEL_REVISION,
            "runtime_chat_template_sha256": CHAT_TEMPLATE_SHA256,
        }
        or interface != PARITY_INTERFACE
        or set(source_closure) != set(SOURCE_CLOSURE_PATHS)
        or any(not _sha(value) for value in source_closure.values())
        or type(max_length) is not int
        or max_length <= 1
        or type(context_tokens) is not int
        or context_tokens < 0
    ):
        raise CollectionError("v2 synthetic parity bindings differ")
    try:
        tokens, response, mask = v1._sample_fields(sample)
    except Exception:
        raise CollectionError("v2 synthetic native recorder sample differs") from None
    prompt_length = len(tokens) - response
    if (
        not isinstance(dense_reference_tokens, list)
        or not isinstance(dense_reference_loss_mask, list)
        or len(dense_reference_tokens) != len(dense_reference_loss_mask)
        or any(type(token) is not int or token < 0 for token in dense_reference_tokens)
        or any(type(item) is not int or item not in {0, 1} for item in dense_reference_loss_mask)
    ):
        raise CollectionError("v2 synthetic dense reference differs")
    reference_spans = self_trace_corpus._runs(dense_reference_loss_mask)
    try:
        self_trace_corpus._conversation_shape(
            conversation, reference_spans, required_anchor_roles=("user",)
        )
    except Exception:
        raise CollectionError("v2 synthetic dense reference does not align") from None
    spans = self_trace_corpus._runs([0] * prompt_length + mask)
    try:
        shape = self_trace_corpus._conversation_shape(
            conversation, spans, required_anchor_roles=("user",)
        )
        rows, excluded = self_trace_corpus._segment_native(
            episode_id="synthetic-direct-user-only-parity",
            task_key="synthetic-direct-user-only-parity",
            tokens=tokens,
            response_length=response,
            response_mask=mask,
            shape=shape,
            max_length=max_length,
            context_tokens=context_tokens,
        )
    except Exception:
        raise CollectionError("v2 synthetic recorder and dense adapter do not align") from None
    if len(spans) < 2 or len(reference_spans) < 2 or len(rows) != 1 or excluded:
        raise CollectionError("v2 synthetic parity fixture lacks two retained direct targets")
    first_observation = (reference_spans[0][1], reference_spans[1][0])
    last_target_end = spans[-1][1]
    expected_mask = [0] * prompt_length + mask
    row = rows[0]
    if (
        first_observation[0] >= first_observation[1]
        or any(dense_reference_loss_mask[first_observation[0] : first_observation[1]])
        or tokens != dense_reference_tokens
        or expected_mask != dense_reference_loss_mask
        or len(tokens) < first_observation[1]
        or tokens[: first_observation[1]] != dense_reference_tokens[: first_observation[1]]
        or expected_mask[: first_observation[1]]
        != dense_reference_loss_mask[: first_observation[1]]
        or row["input_ids"] != tokens[:last_target_end]
        or row["loss_mask"] != expected_mask[:last_target_end]
        or row["target_token_count"] != sum(expected_mask[:last_target_end])
        or [item["tool"] for item in shape] != ["bash", "submit_report"]
    ):
        raise CollectionError("v2 synthetic token or loss-mask parity differs")
    native_receipt = {
        "prompt_tokens_sha256": digest_json(tokens[:prompt_length]),
        "recorded_tokens_sha256": digest_json(tokens),
        "recorded_loss_mask_sha256": digest_json(expected_mask),
        "assistant_targets": len(spans),
    }
    dense_receipt = {
        "reference_tokens_sha256": digest_json(dense_reference_tokens),
        "reference_loss_mask_sha256": digest_json(dense_reference_loss_mask),
        "trained_prefix_tokens_sha256": digest_json(row["input_ids"]),
        "trained_prefix_loss_mask_sha256": digest_json(row["loss_mask"]),
        "trained_prefix_tokens": len(row["input_ids"]),
        "supervised_tokens": row["target_token_count"],
    }
    dense_policy = {
        "format": dense.FORMAT,
        "max_length": max_length,
        "context_tokens": context_tokens,
    }
    observation_receipt = {
        "tokens": first_observation[1] - first_observation[0],
        "native_dense_token_ids_equal": True,
        "native_dense_loss_masks_equal": True,
        "all_observation_tokens_masked": True,
    }
    fixture_sha256 = digest_json(
        {
            "schema": "cyber_qwen_direct_recorder_dense_fixture_v2",
            "prompt_policy_sha256": PROMPT_POLICY_SHA256,
            "native": native_receipt,
            "dense": dense_receipt,
            "dense_policy": dense_policy,
            "first_observation": observation_receipt,
        }
    )
    receipt = {
        "schema": PARITY_SCHEMA,
        "status": "qualified",
        "model": model,
        "interface": interface,
        "source_closure_sha256": digest_json(source_closure),
        "fixture_sha256": fixture_sha256,
        "native": native_receipt,
        "dense": dense_receipt,
        "dense_policy": dense_policy,
        "first_observation": observation_receipt,
        "qualification_scope": {
            "kind": "skyrl_recorder_adapter_to_dense_synthetic_fixture",
            "proves": (
                "user_only_recorded_token_and_loss_mask_preservation_across_two_direct_turns"
            ),
            "target_model_weights_loaded": False,
            "target_runtime_chat_template_loaded": False,
            "pinned_native_helper_loaded": False,
            "collector_image_or_base_route_used": False,
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
    """Project one ready v2 request into a sealed content-free expectation."""
    if validated.get("blockers") or not isinstance(attempt_id, str):
        raise CollectionError("v2 source expectation requires a ready request")
    matches = [row for row in validated["roster"] if row["attempt_id"] == attempt_id]
    if len(matches) != 1:
        raise CollectionError("v2 source attempt is absent or duplicated")
    attempt = matches[0]
    inventory = validated["inventory"]
    row = inventory[(attempt["task_key"], attempt["task_version_id"])]
    request = validated["request"]
    base = validated["base_request"]
    collector = validated["collector_qualification"]
    route = validated["route_certificate"]
    receipt = {
        "schema": EXPECTATION_SCHEMA,
        "collection_request_sha256": request["sha256"],
        "producer_plan_sha256": validated["plan"]["sha256"],
        "attempt": attempt,
        "model": request["model"],
        "interface": PARITY_INTERFACE,
        "runtime_artifacts": {
            "collector_qualification_sha256": collector["sha256"],
            "collector_image": validated["collector_image"],
            "direct_route_certificate_sha256": route["sha256"],
        },
        "runtime_binding": {
            "environment": row["environment"],
            "verifier": row["verifier"],
            "current_binding_sha256": row["current_binding_sha256"],
        },
        "limits": base["limits"],
        "sampling": base["sampling"],
        "source_closure": validated["source_closure"],
        "qualification": base["qualification"],
        "required_private_digests": RENDERED_REQUEST_CONTRACT,
    }
    return {**receipt, "sha256": digest_json(receipt)}


def prepare_attempt(
    validated: dict[str, Any],
    attempt_id: str,
    *,
    task: dict,
    tokenizer: Any,
) -> dict[str, Any]:
    """Build one user-only config/prefix for ``rl_episode.collect``; do no I/O."""
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
        raise CollectionError("live task differs from the frozen v2 attempt") from None
    expected_environment = runtime["environment"]
    expected_verifier = runtime["verifier"]
    if (
        any(
            environment.get(key) != v1._normalized_sha(value)
            for key, value in expected_environment.items()
        )
        or any(
            str(verifier.get(key)) != str(v1._normalized_sha(value))
            for key, value in expected_verifier.items()
        )
        or environment.get("ttl_seconds") != expectation["limits"]["ttl_seconds"]
        or task_binding.get("cyber_contract") != rl_data.AUTHORITY["required_cyber_contract"]
        or not isinstance(task.get("prompt"), str)
        or not task["prompt"]
    ):
        raise CollectionError("live task runtime or verifier differs from the v2 attempt")
    private_request = prepare_private_request(
        task_prompt=task["prompt"],
        tool_catalog=validated["tool_catalog"],
        tokenizer=tokenizer,
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
            "required_task_tool_catalog_sha256": TOOL_CATALOG_SHA256,
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
        raise CollectionError("prepared v2 attempt is not a direct RL binding") from None
    return {
        "config": config,
        # None is the existing rl_episode signal for its user-only default.
        "request_messages": None,
        "request_tools": private_request["tools"],
        "rendered_request": private_request["rendered"],
        "prompt_tokens": private_request["tokens"],
        "response_tokens": limits["response_tokens"],
        "expectation": expectation,
    }


def _private_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file() or stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise CollectionError("private v2 episode member mode or type differs")


def review_source(
    expectation: dict,
    episode_directory: Path,
    *,
    tokenizer: Any,
    tool_catalog: list[dict],
) -> dict:
    """Review one user-only accepted episode without emitting private content."""
    _sealed(expectation, EXPECTATION_SCHEMA)
    if (
        set(expectation)
        != {
            "schema",
            "collection_request_sha256",
            "producer_plan_sha256",
            "attempt",
            "model",
            "interface",
            "runtime_artifacts",
            "runtime_binding",
            "limits",
            "sampling",
            "source_closure",
            "qualification",
            "required_private_digests",
            "sha256",
        }
        or expectation.get("interface") != PARITY_INTERFACE
        or expectation.get("required_private_digests") != RENDERED_REQUEST_CONTRACT
        or expectation.get("source_closure")
        != {name: file_sha256(path) for name, path in SOURCE_CLOSURE_PATHS.items()}
    ):
        raise CollectionError("v2 source expectation differs")
    attempt = expectation.get("attempt")
    model = expectation.get("model")
    runtime = expectation.get("runtime_binding")
    artifacts = expectation.get("runtime_artifacts")
    if (
        not isinstance(attempt, dict)
        or set(attempt) != {"attempt_id", "attempt_ordinal", "task_key", "task_version_id"}
        or re.fullmatch(r"q38-self-[0-9a-f]{20}-a[1-9][0-9]*", str(attempt.get("attempt_id")))
        is None
        or not self_trace_corpus._canonical_uuid(attempt.get("task_version_id"))
        or not isinstance(model, dict)
        or model.get("repo") != MODEL_REPO
        or model.get("revision") != MODEL_REVISION
        or model.get("runtime_chat_template_sha256") != CHAT_TEMPLATE_SHA256
        or not isinstance(runtime, dict)
        or not isinstance(artifacts, dict)
        or set(artifacts)
        != {
            "collector_qualification_sha256",
            "collector_image",
            "direct_route_certificate_sha256",
        }
        or not _sha(artifacts.get("collector_qualification_sha256"))
        or IMAGE.fullmatch(str(artifacts.get("collector_image", ""))) is None
        or not _sha(artifacts.get("direct_route_certificate_sha256"))
    ):
        raise CollectionError("v2 source expectation is incomplete")
    if episode_directory.is_symlink() or not episode_directory.is_dir():
        raise CollectionError("private v2 episode directory is missing")
    if stat.S_IMODE(episode_directory.stat().st_mode) != 0o700:
        raise CollectionError("private v2 episode directory mode differs")
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
    if {path.name for path in episode_directory.iterdir()} != required:
        raise CollectionError("private v2 episode file inventory differs")
    for name in required:
        _private_file(episode_directory / name)
    entry = {
        "episode_id": attempt["attempt_id"],
        "directory": attempt["attempt_id"],
        "task_key": attempt["task_key"],
        "task_version_id": attempt["task_version_id"],
        "accepted_sha256": file_sha256(episode_directory / "ACCEPTED.json"),
    }
    index = {
        "model": {"repo": model["repo"], "revision": model["revision"], "root": model["root"]},
        "interface": {
            "runtime_chat_template_sha256": CHAT_TEMPLATE_SHA256,
            "required_task_tool_catalog_sha256": TOOL_CATALOG_SHA256,
        },
    }
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
        raise CollectionError("private v2 episode failed direct-source qualification") from None
    binding = _read_json(episode_directory / "binding.json")
    conversation = _read_json(episode_directory / "conversation.json")
    create_intent = _read_json(episode_directory / "create-intent.json")
    instance = _read_json(episode_directory / "instance.json")
    score_intent = _read_json(episode_directory / "score-intent.json")
    messages = conversation.get("messages") if isinstance(conversation, dict) else None
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
        raise CollectionError("private v2 scoring intent is invalid") from None
    expected_environment = runtime.get("environment") if isinstance(runtime, dict) else None
    expected_verifier = runtime.get("verifier") if isinstance(runtime, dict) else None
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
        or not isinstance(environment, dict)
        or not isinstance(verifier, dict)
        or not isinstance(expected_environment, dict)
        or not isinstance(expected_verifier, dict)
        or environment.get("ttl_seconds") != expectation["limits"]["ttl_seconds"]
        or any(
            environment.get(key) != v1._normalized_sha(value)
            for key, value in expected_environment.items()
        )
        or any(
            str(verifier.get(key)) != str(v1._normalized_sha(value))
            for key, value in expected_verifier.items()
        )
        or verifier.get("function_name") != "verify"
        or not isinstance(messages, list)
        or len(messages) < 1
        or messages[0].get("role") != "user"
        or any(message.get("role") == "system" for message in messages)
    ):
        raise CollectionError("private v2 episode differs from its request")
    try:
        private_request = prepare_private_request(
            task_prompt=messages[0]["content"],
            tool_catalog=tool_catalog,
            tokenizer=tokenizer,
            task_prompt_sha256=binding["task"]["prompt_sha256"],
            runtime_chat_template_sha256=CHAT_TEMPLATE_SHA256,
        )
        recording = _read_json(episode_directory / "recording.json")
        sample = recording["samples"][0]
        prompt_length = len(sample["tokens"]) - sample["response_length"]
    except Exception:
        raise CollectionError("private v2 request rendering is invalid") from None
    if (
        private_request["rendered_sha256"] != binding["initial_prompt_sha256"]
        or private_request["tokens_sha256"] != binding["initial_prompt_tokens_sha256"]
        or private_request["tokens"] != sample["tokens"][:prompt_length]
    ):
        raise CollectionError("private v2 request rendering differs")
    accepted = _read_json(episode_directory / "ACCEPTED.json")
    files = {name: file_sha256(episode_directory / name) for name in sorted(required)}
    receipt = {
        "schema": SOURCE_RECEIPT_SCHEMA,
        "status": "qualified",
        "collection_request_sha256": expectation["collection_request_sha256"],
        "producer_plan_sha256": expectation["producer_plan_sha256"],
        "expectation_sha256": expectation["sha256"],
        "attempt": attempt,
        "runtime_artifacts": artifacts,
        "accepted_sha256": entry["accepted_sha256"],
        "accepted_receipt_sha256": accepted.get("sha256"),
        "private_file_set_sha256": digest_json(files),
        "request": {
            "roles": ["user"],
            "explicit_system_message": "absent",
            "raw_task_prompt_sha256": binding["task"]["prompt_sha256"],
            "initial_rendered_prompt_sha256": binding["initial_prompt_sha256"],
            "initial_prompt_tokens_sha256": binding["initial_prompt_tokens_sha256"],
            "ordered_tool_catalog_sha256": TOOL_CATALOG_SHA256,
            "prompt_policy_sha256": PROMPT_POLICY_SHA256,
            "template_invocations_sha256": TEMPLATE_INVOCATIONS_SHA256,
            "rendered_request_contract_sha256": RENDERED_REQUEST_CONTRACT_SHA256,
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
    """Validate one v2 request without loading private task or trace content."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args(argv)
    try:
        request = _read_json(args.request)
        if not isinstance(request, dict):
            raise CollectionError("v2 collection request is not an object")
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
