"""Deterministic, read-only manifests for the post-SFT source and HF export."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import ssl
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .io import digest_json

TOKENIZER_FILES = (
    "chat_template.jinja",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)

EVIDENCE_BUNDLE_ROOT = Path("/bundle")
EVIDENCE_MOUNTED_CODE_FILES = {
    "training__init__.py": "training/__init__.py",
    "training_io.py": "training/io.py",
    "training_post_sft_artifacts.py": "training/post_sft_artifacts.py",
    "training_tokenizer_equivalence.py": "training/tokenizer_equivalence.py",
}
EVIDENCE_LOCAL_CODE_FILES = {
    **EVIDENCE_MOUNTED_CODE_FILES,
    "training__init__.py": "evals/post_sft/runtime/training__init__.py",
}
EVIDENCE_PLAN_MOUNT = "plan.json"
EVIDENCE_REGISTRATION_MOUNT = "base-registration.json"

ALLOWED_TRAINER_CONFIG_DRIFT = {
    "dtype",
    "pad_token_id",
    "text_config.dtype",
    "transformers_version",
    "use_cache",
    "vision_config.dtype",
    "vision_config.model_type",
}

EXACT_AUXILIARY_OMISSION_SCHEMA = "cyber_sft_exact_auxiliary_head_omission_v1"
QWEN36_EXACT_MTP_OMISSION_KEYS = (
    "mtp.fc.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.mlp.down_proj.weight",
    "mtp.layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
)


def _json_differences(base: Any, candidate: Any, prefix: str = "") -> list[dict[str, Any]]:
    if isinstance(base, dict) and isinstance(candidate, dict):
        rows = []
        for key in sorted(set(base) | set(candidate)):
            path = f"{prefix}.{key}" if prefix else key
            rows.extend(_json_differences(base.get(key), candidate.get(key), path))
        return rows
    if base != candidate:
        return [{"path": prefix, "base": base, "candidate": candidate}]
    return []


def compare_model_config_architecture(base_config: Path, candidate_config: Path) -> dict[str, Any]:
    """Allow known trainer/runtime metadata rewrites but no architectural drift."""

    base = json.loads(base_config.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_config.read_text(encoding="utf-8"))
    differences = _json_differences(base, candidate)
    unexpected = [row for row in differences if row["path"] not in ALLOWED_TRAINER_CONFIG_DRIFT]
    if unexpected:
        raise ValueError(
            "candidate model config contains architectural or undeclared drift: "
            + ", ".join(row["path"] for row in unexpected)
        )
    def without_allowed(document: dict[str, Any]) -> dict[str, Any]:
        result = json.loads(json.dumps(document))
        for dotted in ALLOWED_TRAINER_CONFIG_DRIFT:
            parent: Any = result
            parts = dotted.split(".")
            for part in parts[:-1]:
                parent = parent.get(part, {}) if isinstance(parent, dict) else {}
            if isinstance(parent, dict):
                parent.pop(parts[-1], None)
        return result

    normalized_base = without_allowed(base)
    normalized_candidate = without_allowed(candidate)
    if normalized_base != normalized_candidate:  # defensive: the diff check above should imply it
        raise ValueError("normalized model architectures differ")
    return {
        "schema": "cyber_sft_model_config_architecture_equivalence_v1",
        "all_architecture_and_vocab_fields_identical": True,
        "allowed_trainer_metadata_differences": differences,
        "base_config_sha256": sha256_file(base_config),
        "candidate_config_sha256": sha256_file(candidate_config),
        "normalized_architecture_sha256": digest_json(normalized_base),
    }


def _safetensor_layout(root: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - cluster image supplies this dependency
        raise ValueError("safetensors is required to inspect an HF export") from exc

    index_path = root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("HF export has no safetensors weight map")
    shard_names = sorted(set(weight_map.values()))
    if not all(isinstance(name, str) and name.endswith(".safetensors") for name in shard_names):
        raise ValueError("HF export weight map contains a non-safetensors shard")
    if sorted(path.name for path in root.glob("*.safetensors")) != shard_names:
        raise ValueError("HF export shards differ from its weight map")

    layout: dict[str, dict[str, Any]] = {}
    for name in shard_names:
        with safe_open(str(root / name), framework="pt", device="cpu") as shard:
            # Current safetensors exposes ``keys()`` but older test doubles and releases were
            # iterable. Supporting both keeps the artifact verifier version-tolerant.
            keys = shard.keys() if hasattr(shard, "keys") else iter(shard)
            for key in keys:
                if key in layout:
                    raise ValueError(f"duplicate tensor {key!r} in HF export")
                tensor_slice = shard.get_slice(key)
                shape = tensor_slice.get_shape()
                if not shape or not all(isinstance(value, int) and value > 0 for value in shape):
                    raise ValueError(f"invalid tensor shape for {key}")
                layout[key] = {
                    "shape": list(shape),
                    "dtype": tensor_slice.get_dtype(),
                    "shard": name,
                }
    if set(layout) != set(weight_map):
        raise ValueError("safetensors index keys differ from shard contents")
    for key, row in layout.items():
        if weight_map[key] != row["shard"]:
            raise ValueError(f"safetensors index points {key!r} at the wrong shard")
    return layout, shard_names


def compare_safetensor_layout(
    base_root: Path,
    candidate_root: Path,
    *,
    require_same_dtype: bool = True,
) -> dict[str, Any]:
    """Prove candidate keys/shapes, and optionally dtypes, agree with the base."""

    base, _ = _safetensor_layout(base_root.resolve(strict=True))
    candidate, _ = _safetensor_layout(candidate_root.resolve(strict=True))
    normalized_base = {
        key: {"shape": row["shape"], "dtype": row["dtype"]} for key, row in base.items()
    }
    normalized_candidate = {
        key: {"shape": row["shape"], "dtype": row["dtype"]}
        for key, row in candidate.items()
    }
    missing = sorted(set(base) - set(candidate))
    unexpected = sorted(set(candidate) - set(base))
    shape_mismatched = sorted(
        key
        for key in set(base) & set(candidate)
        if normalized_base[key]["shape"] != normalized_candidate[key]["shape"]
    )
    dtype_mismatched = sorted(
        key
        for key in set(base) & set(candidate)
        if normalized_base[key]["dtype"] != normalized_candidate[key]["dtype"]
    )
    if missing or unexpected or shape_mismatched or (require_same_dtype and dtype_mismatched):
        raise ValueError(
            "candidate safetensors layout differs from base: "
            f"missing={len(missing)}, unexpected={len(unexpected)}, "
            f"shape_mismatched={len(shape_mismatched)}, "
            f"dtype_mismatched={len(dtype_mismatched)}"
        )
    return {
        "schema": "cyber_sft_safetensors_layout_equivalence_v2",
        "tensor_count": len(candidate),
        "missing_key_count": 0,
        "unexpected_key_count": 0,
        "shape_mismatch_count": 0,
        "dtype_mismatch_count": len(dtype_mismatched),
        "dtype_match_required": require_same_dtype,
        "base_layout_sha256": digest_json(normalized_base),
        "candidate_layout_sha256": digest_json(normalized_candidate),
        "all_keys_and_shapes_match": True,
        "all_keys_shapes_and_dtypes_match": not dtype_mismatched,
    }


def compare_safetensor_layout_with_exact_auxiliary_omission(
    base_root: Path,
    candidate_root: Path,
    omission: Mapping[str, Any],
) -> dict[str, Any]:
    """Accept only one fully enumerated base-only auxiliary-head omission.

    This deliberately does not add an allowlist argument to
    :func:`compare_safetensor_layout`.  The generic comparator remains strict;
    this model-specific rail proves every missing key, shape, dtype, and element
    count against an immutable base header before accepting the known trainer
    omission.
    """

    if omission.get("schema") != EXACT_AUXILIARY_OMISSION_SCHEMA:
        raise ValueError("unsupported exact auxiliary-head omission schema")
    rows = omission.get("tensors")
    if not isinstance(rows, list) or not rows:
        raise ValueError("exact auxiliary-head omission must enumerate tensors")
    expected: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("auxiliary-head omission tensor rows must be objects")
        key = row.get("key")
        shape = row.get("shape")
        dtype = row.get("dtype")
        elements = row.get("elements")
        if (
            not isinstance(key, str)
            or not key
            or key in expected
            or not isinstance(shape, list)
            or not shape
            or any(not isinstance(size, int) or size < 1 for size in shape)
            or dtype != "BF16"
            or not isinstance(elements, int)
            or elements != math.prod(shape)
        ):
            raise ValueError("auxiliary-head omission tensor row is malformed")
        expected[key] = {"shape": shape, "dtype": dtype, "elements": elements}
    if tuple(sorted(expected)) != tuple(sorted(QWEN36_EXACT_MTP_OMISSION_KEYS)):
        raise ValueError("auxiliary-head allowlist is not the exact frozen Qwen3.6 MTP set")
    if omission.get("role") != "speculative_draft_heads":
        raise ValueError("auxiliary-head omission role is not speculative draft heads")

    base, _ = _safetensor_layout(base_root.resolve(strict=True))
    candidate, _ = _safetensor_layout(candidate_root.resolve(strict=True))
    missing = sorted(set(base) - set(candidate))
    unexpected = sorted(set(candidate) - set(base))
    shape_mismatched = sorted(
        key for key in set(base) & set(candidate) if base[key]["shape"] != candidate[key]["shape"]
    )
    candidate_wrong_dtype = sorted(
        key for key, row in candidate.items() if str(row["dtype"]).upper() != "F32"
    )
    if missing != sorted(expected):
        raise ValueError("candidate missing keys differ from the exact auxiliary-head allowlist")
    if unexpected:
        raise ValueError("candidate contains unexpected tensors outside the frozen base")
    if shape_mismatched:
        raise ValueError("candidate tensor shapes differ from the frozen base")
    if candidate_wrong_dtype:
        raise ValueError("candidate trained tensors are not uniformly F32")
    for key, declared in expected.items():
        observed = base[key]
        if observed["shape"] != declared["shape"] or observed["dtype"] != declared["dtype"]:
            raise ValueError(f"frozen base auxiliary tensor metadata differs for {key}")

    normalized_base = {
        key: {"shape": row["shape"], "dtype": row["dtype"]} for key, row in base.items()
    }
    normalized_candidate = {
        key: {"shape": row["shape"], "dtype": row["dtype"]}
        for key, row in candidate.items()
    }
    missing_parameter_count = sum(row["elements"] for row in expected.values())
    candidate_parameter_count = sum(math.prod(row["shape"]) for row in candidate.values())
    base_parameter_count = sum(math.prod(row["shape"]) for row in base.values())
    declared = {
        "base_tensor_count": len(base),
        "base_parameter_count": base_parameter_count,
        "raw_export_tensor_count": len(candidate),
        "raw_export_parameter_count": candidate_parameter_count,
        "missing_tensor_count": len(expected),
        "missing_parameter_count": missing_parameter_count,
    }
    for field, value in declared.items():
        if omission.get(field) != value:
            raise ValueError(f"exact auxiliary-head omission {field} differs from headers")
    if base_parameter_count - candidate_parameter_count != missing_parameter_count:
        raise ValueError("auxiliary-head omission parameter arithmetic does not close")
    if omission.get("serving_inference_effect") != "inert_without_speculative_decoding":
        raise ValueError("auxiliary-head omission lacks the frozen serving interpretation")
    if omission.get("restoration_policy") != "copy_exact_frozen_base_bf16_tensor_bits":
        raise ValueError("auxiliary-head restoration policy is not exact base-bit copy")

    exact_rows = [
        {"key": key, **expected[key], "base_shard": base[key]["shard"]}
        for key in sorted(expected)
    ]
    return {
        "schema": "cyber_sft_safetensors_exact_auxiliary_omission_v1",
        **declared,
        "missing_tensors": exact_rows,
        "missing_tensors_sha256": digest_json(exact_rows),
        "unexpected_key_count": 0,
        "shape_mismatch_count": 0,
        "candidate_wrong_dtype_count": 0,
        "base_layout_sha256": digest_json(normalized_base),
        "candidate_layout_sha256": digest_json(normalized_candidate),
        "exact_allowlist_match": True,
        "all_present_keys_and_shapes_match": True,
        "parameter_arithmetic_closes": True,
    }


def inspect_hf_export_with_exact_auxiliary_omission(
    root: Path,
    *,
    base_root: Path,
    omission: Mapping[str, Any],
    expected_tokenizer_manifest_sha256: str,
    expected_chat_template_sha256: str,
    expected_config_sha256: str,
    expected_sidecar_sha256: dict[str, str] | None = None,
    require_base_sidecars: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Inspect the raw F32 export after independently proving one exact omission."""

    layout = compare_safetensor_layout_with_exact_auxiliary_omission(
        base_root, root, omission
    )
    inspection = inspect_hf_export(
        root,
        expected_tokenizer_manifest_sha256=expected_tokenizer_manifest_sha256,
        expected_chat_template_sha256=expected_chat_template_sha256,
        expected_config_sha256=expected_config_sha256,
        expected_parameter_count=int(layout["raw_export_parameter_count"]),
        expected_sidecar_sha256=expected_sidecar_sha256,
        require_base_sidecars=require_base_sidecars,
        expected_dtype="F32",
    )
    inspection.update(
        {
            "parameter_count_expectation": (
                "declared_raw_export_after_exact_auxiliary_head_omission_v1"
            ),
            "base_parameter_count": layout["base_parameter_count"],
            "omitted_parameter_count": layout["missing_parameter_count"],
            "exact_auxiliary_omission_sha256": layout["missing_tensors_sha256"],
        }
    )
    return inspection, layout


def prove_speculative_decoding_disabled(
    registration: Mapping[str, Any], serving: Mapping[str, Any]
) -> dict[str, Any]:
    """Prove the exact pinned registration has no speculative/draft runtime argument."""

    speculative = serving.get("speculative_decoding")
    if not isinstance(speculative, Mapping) or speculative.get("enabled") is not False:
        raise ValueError("serving plan does not disable speculative decoding")
    prohibited = speculative.get("prohibited_runtime_args")
    if not isinstance(prohibited, list) or not prohibited or any(
        not isinstance(value, str) or not value.startswith("--") for value in prohibited
    ):
        raise ValueError("serving plan has no exact prohibited speculative argument list")
    spec = registration.get("spec")
    runtime = spec.get("runtime") if isinstance(spec, Mapping) else None
    args = runtime.get("args") if isinstance(runtime, Mapping) else None
    if not isinstance(args, list) or any(not isinstance(value, str) for value in args):
        raise ValueError("pinned serving registration runtime args are malformed")
    lowered = [value.lower().replace("_", "-") for value in args]
    forbidden = set(prohibited)
    if any(value in forbidden for value in args) or any(
        "speculative" in value or "draft-model" in value for value in lowered
    ):
        raise ValueError("pinned serving registration enables speculative decoding")
    return {
        "schema": "cyber_post_sft_no_speculative_decoding_proof_v1",
        "registration_sha256": speculative.get("registration_sha256"),
        "runtime_args_sha256": digest_json(args),
        "prohibited_runtime_args": list(prohibited),
        "prohibited_runtime_args_absent": True,
        "no_speculative_or_draft_argument": True,
    }


def _evidence_contract(plan: Mapping[str, Any]) -> Mapping[str, Any]:
    execution = plan.get("evidence_execution")
    contract = execution.get("sfs_export_inspector") if isinstance(execution, Mapping) else None
    if not isinstance(contract, Mapping):
        raise ValueError("plan lacks SFS export evidence execution contract")
    if contract.get("schema") != "cyber_sft_sfs_evidence_execution_plan_v1":
        raise ValueError("unsupported SFS export evidence execution contract")
    return contract


def validate_local_sfs_evidence_bundle(
    plan: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    """Fail closed unless every create-only ConfigMap input matches the frozen plan."""

    contract = _evidence_contract(plan)
    expected = contract.get("config_map_code_sha256")
    if not isinstance(expected, Mapping):
        raise ValueError("SFS evidence contract lacks reviewed code hashes")
    observed: dict[str, str] = {}
    for key, mounted_relative in EVIDENCE_MOUNTED_CODE_FILES.items():
        digest = sha256_file(root / EVIDENCE_LOCAL_CODE_FILES[key])
        if digest != expected.get(mounted_relative):
            raise ValueError(f"local evidence bundle differs for {mounted_relative}")
        observed[mounted_relative] = digest
    registration_path = root / str(contract.get("registration_path", ""))
    registration_digest = sha256_file(registration_path)
    if registration_digest != contract.get("registration_sha256"):
        raise ValueError("local serving registration differs from the frozen evidence contract")
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    proof = prove_speculative_decoding_disabled(registration, plan.get("serving", {}))
    if proof["registration_sha256"] != registration_digest:
        raise ValueError("no-speculative proof names a different serving registration")
    return {
        "code_sha256": observed,
        "registration_sha256": registration_digest,
        "no_speculative_decoding_proof": proof,
    }


def _kubernetes_get(path: str) -> dict[str, Any]:
    host = os.environ.get("KUBERNETES_SERVICE_HOST", "").strip()
    port = os.environ.get("KUBERNETES_SERVICE_PORT_HTTPS", "443").strip()
    if not host:
        raise ValueError("KUBERNETES_SERVICE_HOST is required for evidence provenance")
    token_path = Path("/var/run/secrets/kubernetes.io/serviceaccount/token")
    ca_path = Path("/var/run/secrets/kubernetes.io/serviceaccount/ca.crt")
    token = token_path.read_text(encoding="utf-8").strip()
    request = urllib.request.Request(
        f"https://{host}:{port}{path}", headers={"Authorization": f"Bearer {token}"}
    )
    with urllib.request.urlopen(
        request, timeout=30, context=ssl.create_default_context(cafile=str(ca_path))
    ) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise ValueError("Kubernetes API returned a non-object")
    return value


def collect_sfs_evidence_runtime_provenance(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Bind SFS evidence to exact live Job, Pod, resolved image, and immutable bundle."""

    contract = _evidence_contract(plan)
    namespace = os.environ.get("POD_NAMESPACE", "").strip()
    pod_name = os.environ.get("POD_NAME", "").strip()
    pod_uid = os.environ.get("POD_UID", "").strip()
    job_uid = os.environ.get("JOB_UID", "").strip()
    expected_namespace = str(contract.get("namespace", ""))
    job_name = str(contract.get("job_name", ""))
    config_map_name = str(contract.get("config_map_name", ""))
    container_name = str(contract.get("container_name", ""))
    if not all((namespace, pod_name, pod_uid, job_uid)) or namespace != expected_namespace:
        raise ValueError("SFS evidence downward-API identity is missing or unexpected")
    pod = _kubernetes_get(f"/api/v1/namespaces/{namespace}/pods/{pod_name}")
    job = _kubernetes_get(f"/apis/batch/v1/namespaces/{namespace}/jobs/{job_name}")
    config_map = _kubernetes_get(
        f"/api/v1/namespaces/{namespace}/configmaps/{config_map_name}"
    )
    pod_meta = pod.get("metadata")
    job_meta = job.get("metadata")
    config_meta = config_map.get("metadata")
    if not all(isinstance(value, Mapping) for value in (pod_meta, job_meta, config_meta)):
        raise ValueError("live evidence resource metadata is malformed")
    if (pod_meta.get("name"), pod_meta.get("uid")) != (pod_name, pod_uid):
        raise ValueError("live evidence Pod differs from downward-API identity")
    owners = pod_meta.get("ownerReferences")
    if not isinstance(owners, list) or not any(
        isinstance(owner, Mapping)
        and owner.get("kind") == "Job"
        and owner.get("name") == job_name
        and owner.get("uid") == job_uid
        for owner in owners
    ):
        raise ValueError("live evidence Pod is not owned by the expected Job UID")
    if (job_meta.get("name"), job_meta.get("uid")) != (job_name, job_uid):
        raise ValueError("live evidence Job differs from controller identity")
    for metadata, field in ((job_meta, "job_labels"), (pod_meta, "pod_labels")):
        labels = metadata.get("labels")
        expected_labels = contract.get(field)
        if not isinstance(labels, Mapping) or not isinstance(expected_labels, Mapping) or any(
            labels.get(key) != value for key, value in expected_labels.items()
        ):
            raise ValueError(f"live evidence {field} differ from the frozen plan")
    pod_spec = pod.get("spec")
    if not isinstance(pod_spec, Mapping):
        raise ValueError("live evidence Pod spec is malformed")
    if (
        pod_spec.get("serviceAccountName") != contract.get("service_account_name")
        or pod_spec.get("nodeSelector") != contract.get("node_selector")
    ):
        raise ValueError("live evidence Pod placement or service account differs")
    containers = pod_spec.get("containers")
    container = next(
        (
            row
            for row in containers or []
            if isinstance(row, Mapping) and row.get("name") == container_name
        ),
        None,
    )
    if not isinstance(container, Mapping):
        raise ValueError("live evidence container is absent")
    if (
        container.get("image") != contract.get("image")
        or container.get("resources") != contract.get("resources")
        or digest_json(
            {
                "command": list(container.get("command") or []),
                "args": list(container.get("args") or []),
            }
        )
        != contract.get("command_sha256")
    ):
        raise ValueError("live evidence container image/resources/command differ")
    statuses = pod.get("status", {}).get("containerStatuses")
    status = next(
        (
            row
            for row in statuses or []
            if isinstance(row, Mapping) and row.get("name") == container_name
        ),
        None,
    )
    image_id = str(status.get("imageID") if isinstance(status, Mapping) else "")
    expected_image_digest = str(contract.get("image_digest", ""))
    if expected_image_digest not in image_id:
        raise ValueError("resolved evidence imageID differs from the frozen digest")
    if (
        config_meta.get("name") != config_map_name
        or config_map.get("immutable") is not True
        or not config_meta.get("uid")
        or not config_meta.get("resourceVersion")
    ):
        raise ValueError("live evidence ConfigMap identity is incomplete or mutable")
    data = config_map.get("data")
    expected_keys = set(EVIDENCE_MOUNTED_CODE_FILES) | {
        EVIDENCE_PLAN_MOUNT,
        EVIDENCE_REGISTRATION_MOUNT,
    }
    if not isinstance(data, Mapping) or set(data) != expected_keys:
        raise ValueError("live evidence ConfigMap keys differ from the reviewed bundle")
    expected_code = contract.get("config_map_code_sha256")
    mounted: dict[str, str] = {}
    for key, relative in EVIDENCE_MOUNTED_CODE_FILES.items():
        value = data.get(key)
        if not isinstance(value, str):
            raise ValueError(f"live evidence ConfigMap is missing {key}")
        digest = "sha256:" + hashlib.sha256(value.encode()).hexdigest()
        if digest != expected_code.get(relative) or sha256_file(
            EVIDENCE_BUNDLE_ROOT / relative
        ) != digest:
            raise ValueError(f"live or mounted evidence code differs for {relative}")
        mounted[relative] = digest
    plan_bytes = data[EVIDENCE_PLAN_MOUNT].encode()
    if (EVIDENCE_BUNDLE_ROOT / EVIDENCE_PLAN_MOUNT).read_bytes() != plan_bytes:
        raise ValueError("mounted evidence plan differs from immutable ConfigMap bytes")
    if json.loads(plan_bytes) != plan:
        raise ValueError("executing evidence plan differs from immutable ConfigMap plan")
    plan_file_sha256 = "sha256:" + hashlib.sha256(plan_bytes).hexdigest()
    registration_bytes = data[EVIDENCE_REGISTRATION_MOUNT].encode()
    registration_path = EVIDENCE_BUNDLE_ROOT / EVIDENCE_REGISTRATION_MOUNT
    if registration_path.read_bytes() != registration_bytes:
        raise ValueError("mounted registration differs from immutable ConfigMap bytes")
    registration_sha256 = "sha256:" + hashlib.sha256(registration_bytes).hexdigest()
    if registration_sha256 != contract.get("registration_sha256"):
        raise ValueError("mounted registration differs from the frozen digest")
    proof = prove_speculative_decoding_disabled(
        json.loads(registration_bytes), plan.get("serving", {})
    )
    return {
        "schema": "cyber_sft_sfs_evidence_runtime_provenance_v1",
        "image": contract.get("image"),
        "image_id": image_id,
        "resolved_image_digest": expected_image_digest,
        "command_sha256": contract.get("command_sha256"),
        "service_account_name": contract.get("service_account_name"),
        "container_name": container_name,
        "plan_file_sha256": plan_file_sha256,
        "execution_plan_sha256": digest_json(contract),
        "job": {
            "namespace": namespace,
            "name": job_name,
            "uid": job_uid,
            "resource_version": str(job_meta.get("resourceVersion")),
            "spec_sha256": digest_json(job.get("spec")),
        },
        "pod": {
            "namespace": namespace,
            "name": pod_name,
            "uid": pod_uid,
            "resource_version": str(pod_meta.get("resourceVersion")),
            "spec_sha256": digest_json(pod_spec),
        },
        "config_map": {
            "namespace": namespace,
            "name": config_map_name,
            "uid": str(config_meta["uid"]),
            "resource_version": str(config_meta["resourceVersion"]),
            "immutable": True,
            "mounted_file_sha256": mounted,
            "registration_sha256": registration_sha256,
        },
        "no_speculative_decoding_proof": proof,
    }


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def structural_manifest(root: Path) -> dict[str, Any]:
    """Hash path, size, and integer mtime without reading checkpoint payload bytes."""

    resolved = root.resolve(strict=True)
    files = sorted(path for path in resolved.rglob("*") if path.is_file())
    rows = [
        {
            "path": path.relative_to(resolved).as_posix(),
            "size": path.stat().st_size,
            "mtime_seconds": int(path.stat().st_mtime),
        }
        for path in files
    ]
    return {
        "schema": "cyber_sft_structural_manifest_v1",
        "root": str(resolved),
        "file_count": len(rows),
        "files": rows,
        "manifest_sha256": digest_json(rows),
    }


def structural_tsv_sha256(root: Path) -> str:
    """Reproduce the pre-conversion BusyBox stat receipt without reading payload bytes."""

    resolved = root.resolve(strict=True)
    rows = []
    for path in sorted(item for item in resolved.rglob("*") if item.is_file()):
        stat = path.stat()
        rows.append(f"{path}\\t{stat.st_size}\\t{int(stat.st_mtime)}\n")
    return "sha256:" + hashlib.sha256("".join(rows).encode()).hexdigest()


def full_file_manifest(root: Path) -> dict[str, Any]:
    """Serialize hashing to one file at a time to limit pressure on shared SFS."""

    resolved = root.resolve(strict=True)
    files = sorted(path for path in resolved.rglob("*") if path.is_file())
    rows = []
    for path in files:
        rows.append(
            {
                "path": path.relative_to(resolved).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path).removeprefix("sha256:"),
            }
        )
    return {
        "schema": "cyber_sft_full_file_manifest_v1",
        "root": str(resolved),
        "file_count": len(rows),
        "total_bytes": sum(row["size"] for row in rows),
        "files": rows,
        "manifest_sha256": digest_json(rows),
    }


def inspect_hf_export(
    root: Path,
    *,
    expected_tokenizer_manifest_sha256: str,
    expected_chat_template_sha256: str,
    expected_config_sha256: str,
    expected_parameter_count: int,
    expected_sidecar_sha256: dict[str, str] | None = None,
    require_base_sidecars: bool = True,
    expected_dtype: str = "BF16",
) -> dict[str, Any]:
    """Hash all HF files and validate safetensors shapes without materializing tensors."""

    resolved = root.resolve(strict=True)
    layout, shard_names = _safetensor_layout(resolved)

    parameter_count = 0
    tensor_count = 0
    weight_rows = []
    for name in shard_names:
        path = resolved / name
        weight_rows.append(
            {
                "path": name,
                "size": path.stat().st_size,
                "sha256": sha256_file(path).removeprefix("sha256:"),
            }
        )
    normalized_expected_dtype = expected_dtype.upper()
    if normalized_expected_dtype not in {"BF16", "F32"}:
        raise ValueError("expected dtype must be BF16 or F32")
    observed_dtypes = sorted({str(row["dtype"]).upper() for row in layout.values()})
    if observed_dtypes != [normalized_expected_dtype]:
        raise ValueError(
            "HF export tensor dtype differs from expectation: "
            f"expected={normalized_expected_dtype}, observed={observed_dtypes}"
        )
    for row in layout.values():
        parameter_count += math.prod(row["shape"])
        tensor_count += 1
    if parameter_count != expected_parameter_count:
        raise ValueError("HF export parameter count differs from the base architecture")

    tokenizer_rows = []
    for name in TOKENIZER_FILES:
        path = resolved / name
        if path.is_file():
            tokenizer_rows.append(
                {"path": name, "sha256": sha256_file(path).removeprefix("sha256:")}
            )
        elif require_base_sidecars:
            raise ValueError(f"HF export is missing tokenizer file {name}")
    tokenizer_sha256 = digest_json(sorted(tokenizer_rows, key=lambda row: row["path"]))
    tokenizer_matches = tokenizer_sha256 == expected_tokenizer_manifest_sha256
    if require_base_sidecars and not tokenizer_matches:
        raise ValueError("HF export tokenizer manifest differs from the base checkpoint")
    chat_sha256 = sha256_file(resolved / "chat_template.jinja")
    chat_matches = chat_sha256 == expected_chat_template_sha256
    if require_base_sidecars and not chat_matches:
        raise ValueError("HF export chat template differs from the base checkpoint")
    config_sha256 = sha256_file(resolved / "config.json")
    config_matches = config_sha256 == expected_config_sha256
    if require_base_sidecars and not config_matches:
        raise ValueError("HF export model config differs from the base checkpoint")

    observed_sidecars: dict[str, str] = {}
    for path in sorted(item for item in resolved.iterdir() if item.is_file()):
        if path.name.endswith(".safetensors") or path.name == "model.safetensors.index.json":
            continue
        observed_sidecars[path.name] = sha256_file(path)
    sidecar_matches: dict[str, bool] = {}
    if expected_sidecar_sha256:
        sidecar_matches = {
            name: observed_sidecars.get(name) == digest
            for name, digest in sorted(expected_sidecar_sha256.items())
        }
        if require_base_sidecars and not all(sidecar_matches.values()):
            raise ValueError("HF export runtime sidecars differ from the base checkpoint")

    weight_hashes = {row["path"]: row["sha256"] for row in weight_rows}
    file_rows = []
    for path in sorted(item for item in resolved.rglob("*") if item.is_file()):
        relative = path.relative_to(resolved).as_posix()
        file_rows.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": weight_hashes.get(relative) or sha256_file(path).removeprefix("sha256:"),
            }
        )
    return {
        "schema": "cyber_sft_hf_output_inspection_v1",
        "root": str(resolved),
        "format": "safetensors",
        "dtype": normalized_expected_dtype.lower(),
        "shard_count": len(shard_names),
        "tensor_count": tensor_count,
        "parameter_count": parameter_count,
        "weights_manifest_sha256": digest_json(weight_rows),
        "files_manifest_sha256": digest_json(file_rows),
        "tokenizer_manifest_sha256": tokenizer_sha256,
        "tokenizer_manifest_matches_base": tokenizer_matches,
        "chat_template_sha256": chat_sha256,
        "chat_template_matches_base": chat_matches,
        "config_sha256": config_sha256,
        "config_matches_base": config_matches,
        "sidecar_sha256": observed_sidecars,
        "sidecar_matches_base": sidecar_matches,
        "base_sidecars_required": require_base_sidecars,
        "all_shards_present": True,
        "safetensors_load_passed": True,
        "parameter_count_matches": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("structural", "full"):
        child = subparsers.add_parser(command)
        child.add_argument("root", type=Path)
        child.add_argument("--output", type=Path, required=True)
    hf = subparsers.add_parser("hf")
    hf.add_argument("root", type=Path)
    hf.add_argument("--plan", type=Path, required=True)
    hf.add_argument("--output", type=Path, required=True)
    hf.add_argument("--allow-sidecar-drift", action="store_true")
    hf.add_argument("--expected-dtype", choices=("BF16", "F32"), default="BF16")
    exact = subparsers.add_parser("hf-exact-auxiliary-omission")
    exact.add_argument("root", type=Path)
    exact.add_argument("--base-root", type=Path, required=True)
    exact.add_argument("--plan", type=Path, required=True)
    exact.add_argument("--output", type=Path, required=True)
    exact.add_argument("--layout-output", type=Path, required=True)
    exact.add_argument("--allow-sidecar-drift", action="store_true")
    runtime = subparsers.add_parser("evidence-runtime-provenance")
    runtime.add_argument("--plan", type=Path, required=True)
    runtime.add_argument("--output", type=Path, required=True)
    bundle = subparsers.add_parser("validate-evidence-bundle")
    bundle.add_argument("--plan", type=Path, required=True)
    bundle.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "structural":
        result = structural_manifest(args.root)
    elif args.command == "full":
        result = full_file_manifest(args.root)
    elif args.command == "hf":
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        model = plan["base_model"]
        result = inspect_hf_export(
            args.root,
            expected_tokenizer_manifest_sha256=str(model["tokenizer_manifest_sha256"]),
            expected_chat_template_sha256=str(model["chat_template_sha256"]),
            expected_config_sha256=str(model["config_sha256"]),
            expected_parameter_count=int(model["parameter_count"]),
            expected_sidecar_sha256=model.get("runtime_sidecar_sha256"),
            require_base_sidecars=not args.allow_sidecar_drift,
            expected_dtype=args.expected_dtype,
        )
    elif args.command == "hf-exact-auxiliary-omission":
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        model = plan["base_model"]
        result, layout = inspect_hf_export_with_exact_auxiliary_omission(
            args.root,
            base_root=args.base_root,
            omission=plan["export"]["raw_export_auxiliary_head_omission"],
            expected_tokenizer_manifest_sha256=str(model["tokenizer_manifest_sha256"]),
            expected_chat_template_sha256=str(model["chat_template_sha256"]),
            expected_config_sha256=str(model["config_sha256"]),
            expected_sidecar_sha256=model.get("runtime_sidecar_sha256"),
            require_base_sidecars=not args.allow_sidecar_drift,
        )
        if os.path.lexists(args.layout_output):
            raise ValueError(f"refusing pre-existing layout output: {args.layout_output}")
        args.layout_output.write_text(
            json.dumps(layout, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    elif args.command == "validate-evidence-bundle":
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        result = validate_local_sfs_evidence_bundle(plan, args.root)
        print(json.dumps(result, sort_keys=True))
        return
    else:
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        result = collect_sfs_evidence_runtime_provenance(plan)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
