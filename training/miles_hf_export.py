"""Create-once Miles DCP -> Hugging Face export and a separate GPU load gate.

The exporter consumes the already accepted Miles checkpoint and native-reload
receipts.  It uses the exact offline converter shipped by the pinned Miles
image, restores only frozen visual/MTP tensors that the text-only Megatron
actor never owned, and publishes by a no-replace rename.  The GPU gate is a
separate process and does one fixed non-task prediction forward, but no rollout,
backward, optimizer update, verifier call, W&B event, or save.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import numbers
import os
import pickle
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from cyber_post_train.jobs import API_URLS, digest

from . import miles
from .miles_conversion import _hash, _write
from .post_sft_artifacts import _safetensor_layout
from .post_sft_cast import (
    _fsync_directory,
    _fsync_file,
    _load_tensor,
    _rename_noreplace,
    _shard_groups,
)

EXPORT_SCHEMA = "cyber_miles_native_hf_export_v2"
RELOAD_SCHEMA = "cyber_miles_hf_zero_update_reload_v1"
RELOAD_CONTROLLER_SCHEMA = "cyber_miles_hf_reload_controller_terminal_v1"
RELOAD_RELEASE_SCHEMA = "cyber_miles_hf_reload_external_release_v1"
RELOAD_ACCEPTED_SCHEMA = "cyber_miles_hf_reload_accepted_v1"
CHECKPOINT_SCHEMA = "cyber_miles_training_checkpoint_v1"
TERMINAL_SCHEMA = "cyber_miles_reward_canary_terminal_v1"
NATIVE_RELOAD_SCHEMA = "cyber_miles_rl_reload_accepted_v1"
OBSERVER_NATIVE_RELOAD_SCHEMA = "cyber_miles_policy_observer_reload_accepted_v2"
PREDICTION_SCHEMA = "cyber_miles_non_task_prediction_probe_v1"
PREDICTION_PROBE_ID = "qwen38-fixed-token-topk-v1"
PREDICTION_INPUT_IDS = tuple(range(1, 33))
PREDICTION_TOP_K = 16
PREDICTION_MARGIN = 0.01
MODEL_REPO = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
MILES_SOURCE_COMMIT = "2799fe386320c156334bf763ad4d7ca0f85dca4e"
MAX_SHARD_BYTES = 3 * 1024**3
CONVERTER_TIMEOUT_SECONDS = 4 * 60 * 60
HASH_CHUNK_BYTES = 8 * 1024**2
VOCAB_SIZE = 248_320
FROZEN_AUXILIARY_PREFIXES = ("model.visual.", "mtp.")
RELOAD_NAMESPACE = "fleet-train-jobs"

_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}")
_SHA256 = re.compile(r"[a-f0-9]{64}")
_RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,53}")
_STABLE_FILE_ATTRIBUTES = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")


def _fields(value: str) -> frozenset[str]:
    return frozenset(value.split())  # noqa: SIM905 - compact reviewed schemas


_RELOAD_WORK_FIELDS = _fields(
    "optimizer_updates rollouts verifier_calls forwards backwards checkpoint_writes wandb_events"
)
_RELOAD_WORK_EXPECTED = {key: int(key == "forwards") for key in _RELOAD_WORK_FIELDS}
_RELOAD_RESULT_FIELDS = _fields(
    "schema status run_name image export_path export_file_sha256 export_receipt_sha256 "
    "export_tensor_inventory_sha256 model_repo model_revision runtime_tensor_count "
    "artifact_only_mtp_tensor_count runtime_layout_sha256 runtime_value_inventory_sha256 "
    "all_runtime_weights_loaded all_runtime_values_finite all_runtime_values_match_export "
    "source_export_unchanged native_prediction_probe hf_prediction_probe "
    "semantic_prediction_match optimizer_updates rollouts verifier_calls forwards backwards "
    "checkpoint_writes wandb_events gpus gpu_name peak_memory_bytes "
    "external_gpu_release_verified serving_qualified completed_at sha256"
)
_CONTROLLER_FIELDS = _fields(
    "schema status cluster api_base_url kube_context namespace namespace_uid run_name "
    "source_plan_sha256 source_request_sha256 submission_binding_sha256 runtime_bundle_sha256 "
    "event_journal "
    "reload_result_path reload_result_file_sha256 reload_result_sha256 api_run_id "
    "api_run_name rayjob_name rayjob_uid workload_name workload_uid "
    "workload_owner_rayjob_uid raycluster_name raycluster_uid raycluster_owner_rayjob_uid "
    "pods api_status controller_status effective_priority automatic_requeue workers "
    "gpus_per_worker total_gpus observed_at sha256"
)
_POD_FIELDS = _fields(
    "name uid owner_raycluster_uid phase exit_code termination_reason terminated_at "
    "runtime_image_id container_restarts gpus"
)
_RELEASE_FIELDS = _fields(
    "schema status cluster api_base_url kube_context namespace namespace_uid run_name "
    "source_plan_sha256 source_request_sha256 submission_binding_sha256 runtime_bundle_sha256 "
    "reload_result_path reload_result_file_sha256 reload_result_sha256 "
    "controller_terminal_path controller_terminal_file_sha256 controller_terminal_sha256 "
    "api_run_id api_run_name rayjob_name rayjob_uid workload_name workload_uid "
    "raycluster_name raycluster_uid pod_uids api_status controller_status rayjob_present "
    "workload_present quota_reservation_present raycluster_present gpu_pods_present "
    "active_gpu_pod_uids active_gpus observed_at sha256"
)
_ACCEPTED_FIELDS = _fields(
    "schema status export_path export_file_sha256 export_receipt_sha256 "
    "reload_plan_path reload_plan_file_sha256 reload_plan_sha256 "
    "submission_binding_path submission_binding_file_sha256 submission_binding_sha256 "
    "export_tensor_inventory_sha256 reload_result_path reload_result_file_sha256 "
    "reload_result controller_terminal_path controller_terminal_file_sha256 "
    "controller_terminal external_release_path external_release_file_sha256 external_release "
    "work_executed exact_hf_reload_verified source_export_unchanged_after_release "
    "semantic_source_equivalence_verified external_gpu_release_verified "
    "post_export_promotion_requires_this_receipt "
    "serving_qualified sha256"
)
_EXPORT_FIELDS = _fields(
    "schema status source miles model output_root files tensor_inventory "
    "tensor_inventory_sha256 sidecars trained_tensor_count restored_base_tensor_count "
    "restored_base_tensor_inventory_sha256 tensor_bytes all_tensor_values_finite "
    "source_equivalence resource_guard dtype base_layout_sha256 output_layout_sha256 "
    "optimizer_updates_executed source_checkpoint_unchanged base_model_unchanged "
    "create_only gpu_reload_verified completed_at sha256"
)
_EXPORT_SOURCE_FIELDS = _fields(
    "active_canary_binding checkpoint terminal_acceptance native_reload_acceptance "
    "source_plan_sha256 prediction_probe"
)
_PREDICTION_FIELDS = _fields(
    "schema probe_id input_ids_sha256 sequence_length top_k selection_margin_threshold "
    "selection_margin_satisfied prediction_sha256 logits_included task_content_included "
    "benchmark_content_included"
)
_EXPORT_SOURCE_EQUIVALENCE_FIELDS = _fields(
    "method converter_input common_pt_sha256 dcp_metadata_sha256 "
    "trained_tensor_inventory_sha256 source_value_inventory_sha256 "
    "all_trained_values_match_source"
)
_EXPORT_RESOURCE_GUARD_FIELDS = _fields("base_model_bytes minimum_free_bytes converter_passes")
_EXPORT_MODEL_FIELDS = _fields("repo revision base_root native_parallelism")
_EXPORT_FILE_FIELDS = _fields("bytes sha256")
_EXPORT_TENSOR_FIELDS = _fields(
    "name shape dtype shard bytes sha256 value_sha256 all_values_finite source"
)


def _validate_prediction_probe(value: object) -> dict[str, Any]:
    """Validate the privacy-preserving native/HF semantic witness contract."""
    if (
        not isinstance(value, dict)
        or set(value) != _PREDICTION_FIELDS
        or value.get("schema") != PREDICTION_SCHEMA
        or value.get("probe_id") != PREDICTION_PROBE_ID
        or value.get("input_ids_sha256") != "sha256:" + digest(list(PREDICTION_INPUT_IDS))
        or value.get("sequence_length") != len(PREDICTION_INPUT_IDS)
        or value.get("top_k") != PREDICTION_TOP_K
        or value.get("selection_margin_threshold") != PREDICTION_MARGIN
        or value.get("selection_margin_satisfied") is not True
        or not isinstance(value.get("prediction_sha256"), str)
        or _SHA256.fullmatch(value["prediction_sha256"].removeprefix("sha256:")) is None
        or value.get("logits_included") is not False
        or value.get("task_content_included") is not False
        or value.get("benchmark_content_included") is not False
    ):
        raise ValueError("fixed non-task prediction probe is incomplete")
    return value


# The image digest pins third-party libraries. This is the complete Python
# source set imported by the converter's eager megatron_to_hf package plus its
# direct local helpers. _miles_root also rejects an extra source in that eager
# package, so a new import cannot silently expand the closure.
MILES_CONVERTER_SOURCES = {
    "tools/convert_torch_dist_to_hf.py": (
        "332bf9eedb5c72f83de69ac8e993d43c4ab4c2c30d4d2e60f6ba466ac5ea6390"
    ),
    "miles/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "miles/backends/__init__.py": (
        "01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b"
    ),
    "miles/backends/megatron_utils/__init__.py": (
        "9f429da0b5a97a9a81aebc5f3aa83e83a53aee2a41e6f7c3effa173c1c992730"
    ),
    "miles/backends/megatron_utils/megatron_to_hf/__init__.py": (
        "adc2f476e0285700956bf5d32db3133667182fb9a4622a7f44081f9f4126fac7"
    ),
    "miles/backends/megatron_utils/megatron_to_hf/deepseekv3.py": (
        "c1ea9c27f2cb537ec1fc43408cdfdd70862e38b2ffa1ae82ee12661f16783eac"
    ),
    "miles/backends/megatron_utils/megatron_to_hf/deepseekv4.py": (
        "6743acc1377f01b4427c049c48949f3f01a9e8b6bf978c0e1297fa8059921b27"
    ),
    "miles/backends/megatron_utils/megatron_to_hf/glm4.py": (
        "fdc2a52fa21e7f9b526b0cffeee97801cc3a28990a8ace7d33809c6044d4283e"
    ),
    "miles/backends/megatron_utils/megatron_to_hf/glm4moe.py": (
        "b6cf1014891b72c599c4c1fd779736b2356607f7d8083f0f958105341e1ad0e1"
    ),
    "miles/backends/megatron_utils/megatron_to_hf/inkling.py": (
        "b7b454ee7954139052e5cf8220209a3bfe474b544fea904d2323d4900739416a"
    ),
    "miles/backends/megatron_utils/megatron_to_hf/kimi_vl.py": (
        "ec4f39f391ab4a2edefbd85789b5319511d6edd5ad3ae622f4ba134014c8dd28"
    ),
    "miles/backends/megatron_utils/megatron_to_hf/llama.py": (
        "e9223b10d7bf62b9acb3314f909831b0b12502451595763668c541ae0cd28399"
    ),
    "miles/backends/megatron_utils/megatron_to_hf/mimo.py": (
        "97f923736c95bfa00f87a905affc228b57428c62b0534c998c709d9ff6b92254"
    ),
    "miles/backends/megatron_utils/megatron_to_hf/qwen2.py": (
        "312660a94ba4d4d390acd62f0cda2756428ca07ce72fdc65af1e2d53a4bf1904"
    ),
    "miles/backends/megatron_utils/megatron_to_hf/qwen3_5.py": (
        "0fc2b8800b606a5386c065de562ed4dbb294f52caf93ecad6bc5094b36ebb2c1"
    ),
    "miles/backends/megatron_utils/megatron_to_hf/qwen3_next.py": (
        "fb0c184194d17b04f1dad2bf3faae0d7d1a2edcda0e8145da44feb3676fc9db9"
    ),
    "miles/backends/megatron_utils/megatron_to_hf/qwen3moe.py": (
        "59aef122cc8e193e69e8bc46b2a22fbb73daa28b5f5b9da06bb53a471dbbc5b9"
    ),
    "miles/backends/megatron_utils/megatron_to_hf/processors/__init__.py": (
        "65d2515cc094a4eabd0d2b991959056f19aa401b3e175dd79f1a37640dac1fe4"
    ),
    "miles/backends/megatron_utils/megatron_to_hf/processors/padding_remover.py": (
        "d45dc9f0235b6d95f15a3e4906d3873fbf69d6b95eaf5b54157b8b070e44927c"
    ),
    "miles/backends/megatron_utils/megatron_to_hf/processors/quantizer_compressed_tensors.py": (
        "8e5988739e8b1a60aaa19b5e858e50c47de2690210178836c193751e043d23df"
    ),
    "miles/backends/megatron_utils/megatron_to_hf/processors/quantizer_fp8.py": (
        "6a27879b2d4e1b0dc7adef696ddfbe3da1eb561a13ac77f22dd659b2559e3e88"
    ),
    "miles/backends/megatron_utils/megatron_to_hf/processors/quantizer_mxfp8.py": (
        "18411f746379e9c55f46579497a0e63b9953497206914d8079e8afd820f64f5b"
    ),
    "miles/backends/megatron_utils/megatron_to_hf/processors/quantizer_nvfp4.py": (
        "45cf128d6e518fb537cc7b2359ce113b83add70085192ac8ec140fb82b862770"
    ),
    "miles/backends/megatron_utils/misc_utils.py": (
        "10566d9583fd9e593213e42b22c232264b9da70adfb9af2f7a91f08aaa695d64"
    ),
    "miles/backends/megatron_utils/sglang.py": (
        "3b531dc0f3edd6197b9fc29d71a6f1c8f95f3bdc0e2821da9919febbd0fa531a"
    ),
    "miles/backends/megatron_utils/update_weight/__init__.py": (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    ),
    "miles/backends/megatron_utils/update_weight/common.py": (
        "88b853d876ad3c36820ae7bd006a7e62a3d304f074a50b5396b06e59ef010ad9"
    ),
    "miles/backends/training_utils/__init__.py": (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    ),
    "miles/backends/training_utils/parallel.py": (
        "519ec0b0608a985c25e6e1e3211f097db3e33a408e7694f56fe6dbeaafdba14d"
    ),
    "miles/utils/__init__.py": ("a0c0be3dea3fb3d27d098fcc618db4c50c4ddbbc9df0927240fe619ab9ae4931"),
    "miles/utils/fp8_kernel.py": (
        "0c3f8121d9423de9b5109c9015f1b5e4bb0622574fa00b38c267e900ac4bd487"
    ),
    "miles/utils/ft_utils/__init__.py": (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    ),
    "miles/utils/ft_utils/process_group_utils.py": (
        "78ce6e24a9334394e8fb9fe170628d6bbff50f556605cc15a111f87d06302729"
    ),
    "miles/utils/hf_config.py": (
        "603617e36cf713e41356c0c2c20247443d87ecff7cd963480e1b4636a5a3f643"
    ),
    "miles/utils/memory_utils.py": (
        "1f316c26f18be89afffb304ea26c5646c08fe3ceb3ed95d3368ddbefdb9324e3"
    ),
    "miles/utils/mxfp8.py": ("47b13f3dc1d5e144090c45ce253a50de443f47b9a972cc97580a21975a07b8fd"),
    "miles/utils/nvfp4.py": ("c1c3f2d52862ef7adf4220721df75ade9eba8fda3f254789fcacfee15f985b77"),
    "miles/utils/reloadable_process_group.py": (
        "a667b9a60e9a8d1f72605d4b23a8e083a3a378709eded806c08e8002de809bc2"
    ),
    "miles/utils/sampling_mask.py": (
        "941ae81072b112b5f317f6596b44f500183763a241e68060a86f4c6ec36283c9"
    ),
    "miles/utils/test_utils/__init__.py": (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    ),
    "miles/utils/test_utils/det_process_group.py": (
        "62da90c7947aaddb51865839d62f4c1b1eaae2661567a22beabf7d75fc64c3c7"
    ),
    "miles/utils/types.py": ("17d6553d0be9ea4a83d5876c5ae3d49f04f6a292a569c1c62004717807cf1702"),
    "miles_plugins/__init__.py": (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    ),
    "miles_plugins/megatron_bridge/__init__.py": (
        "977a157cceee722f57a857749850b46029a68710f60f60e24eacc8767282b1f0"
    ),
    "miles_plugins/megatron_bridge/nemotron_h.py": (
        "2cddf9ed6f67306d12e6a60a661f5278f78101e08b2ad8878212329bf1fae185"
    ),
    "miles_plugins/models/__init__.py": (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    ),
    "miles_plugins/models/qwen3_vl.py": (
        "b40686ffda418972c9e33168d31d18cd5c01b930d380ace2eb0c75aa83ec8efc"
    ),
}


def _unsigned(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "sha256"}


def _sealed(value: Any, schema: str) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or value.get("schema") != schema
        or value.get("sha256", "").removeprefix("sha256:") != digest(_unsigned(value))
    ):
        raise ValueError("receipt schema or self-digest mismatch")
    return value


def _snapshot_json(path: Path, expected_sha256: object) -> tuple[dict[str, Any], str]:
    if not isinstance(expected_sha256, str):
        raise ValueError("bound JSON file digest is absent")
    expected = expected_sha256.removeprefix("sha256:")
    if _SHA256.fullmatch(expected) is None or path.is_symlink() or not path.is_file():
        raise ValueError("bound JSON file is missing or indirect")
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    attributes = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, name) != getattr(after, name) for name in attributes):
        raise ValueError("bound JSON file changed while reading")
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("bound JSON file digest mismatch")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("bound JSON file is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("bound JSON file must contain an object")
    return value, expected


def _terminal_checkpoint_reference(terminal: dict[str, Any]) -> dict[str, str]:
    _sealed(terminal, TERMINAL_SCHEMA)
    reference = terminal.get("checkpoint_manifest")
    if (
        terminal.get("status") != "accepted"
        or terminal.get("reward_values_included") is not False
        or terminal.get("task_content_included") is not False
        or terminal.get("production_promotion_requires_reload_acceptance") is not True
        or not isinstance(reference, dict)
        or set(reference) != {"path", "file_sha256", "receipt_sha256"}
    ):
        raise ValueError("Miles terminal acceptance is incomplete")
    return reference


def bind_source(
    *,
    active_canary_binding_path: Path,
    active_canary_binding_sha256: str,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    terminal_path: Path,
    terminal_sha256: str,
    native_reload_path: Path,
    native_reload_sha256: str,
    validate_historical: bool = True,
) -> dict[str, Any]:
    """Bind one accepted source without copying private terminal evidence."""
    from . import miles_promotion
    from .miles_reload import _verify_checkpoint

    checkpoint, checkpoint_file_sha256 = _snapshot_json(checkpoint_path, checkpoint_sha256)
    active_canary_binding, active_canary_binding_file_sha256 = _snapshot_json(
        active_canary_binding_path, active_canary_binding_sha256
    )
    terminal, terminal_file_sha256 = _snapshot_json(terminal_path, terminal_sha256)
    native_reload, native_reload_file_sha256 = _snapshot_json(
        native_reload_path, native_reload_sha256
    )
    _sealed(checkpoint, CHECKPOINT_SCHEMA)
    reference = _terminal_checkpoint_reference(terminal)
    miles_promotion.validate_active_canary_binding(active_canary_binding)
    miles_promotion._exact_active_canary(terminal, active_canary_binding)
    native_schema = native_reload.get("schema")
    if native_schema not in {NATIVE_RELOAD_SCHEMA, OBSERVER_NATIVE_RELOAD_SCHEMA}:
        raise ValueError("native reload acceptance schema is unsupported")
    _sealed(native_reload, native_schema)
    if validate_historical:
        from . import miles_acceptance
        from .miles_reload_acceptance import validate_accepted

        miles_acceptance.validate_terminal(terminal, check_files=True)
        validated_reload = validate_accepted(native_reload, check_files=True)
    else:
        # The immutable prepared plan was compiled and preflighted with the full
        # historical validators in the tracked checkout. The isolated Jobs API
        # bundle has no .git directory, so it reopens those exact receipt bytes
        # and the digest-bound canary identity instead of pretending it can replay
        # the historical source-commit proof from inside RUN_DIR/.runtime. The
        # digest-bound active-canary record still fixes the exact accepted run.
        validated_reload = {
            "source_manifest_sha256": native_reload.get("source_manifest_sha256"),
            "source_terminal_acceptance_sha256": native_reload.get(
                "source_terminal_acceptance_sha256"
            ),
            "prediction_probe": native_reload.get("prediction_probe"),
        }
    prediction_probe = _validate_prediction_probe(validated_reload.get("prediction_probe"))
    checkpoint_self_sha256 = checkpoint["sha256"].removeprefix("sha256:")
    if (
        Path(reference["path"]) != checkpoint_path
        or reference["file_sha256"].removeprefix("sha256:") != checkpoint_file_sha256
        or reference["receipt_sha256"].removeprefix("sha256:") != checkpoint_self_sha256
        or terminal.get("source_plan_sha256", "").removeprefix("sha256:")
        != checkpoint.get("source", {}).get("plan_sha256")
        or validated_reload.get("source_manifest_sha256", "").removeprefix("sha256:")
        != checkpoint_self_sha256
        or validated_reload.get("source_terminal_acceptance_sha256", "").removeprefix("sha256:")
        != terminal["sha256"].removeprefix("sha256:")
        or active_canary_binding.get("source_plan_sha256", "").removeprefix("sha256:")
        != checkpoint.get("source", {}).get("plan_sha256", "").removeprefix("sha256:")
    ):
        raise ValueError("terminal/reload acceptance does not bind the selected checkpoint")
    source = checkpoint.get("source", {})
    arguments = source.get("arguments", {})
    model = checkpoint.get("model", {})
    if (
        checkpoint.get("image") != miles.IMAGE
        or checkpoint.get("world_size") != 8
        or checkpoint.get("topology") != {"nodes": 1, "gpus_per_node": 8}
        or arguments.get("nodes") != 1
        or arguments.get("gpus_per_node") != 8
        or model.get("repo") != MODEL_REPO
        or model.get("revision") != MODEL_REVISION
    ):
        raise ValueError("source is not the exact Qwen3.8 TP4 x CP2 Miles profile")
    _verify_checkpoint(checkpoint, hashes=True)
    return {
        "active_canary_binding": {
            "path": str(active_canary_binding_path),
            "file_sha256": active_canary_binding_file_sha256,
            "receipt_sha256": active_canary_binding["sha256"].removeprefix("sha256:"),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "file_sha256": checkpoint_file_sha256,
            "receipt_sha256": checkpoint_self_sha256,
        },
        "terminal_acceptance": {
            "path": str(terminal_path),
            "file_sha256": terminal_file_sha256,
            "receipt_sha256": terminal["sha256"].removeprefix("sha256:"),
        },
        "native_reload_acceptance": {
            "path": str(native_reload_path),
            "file_sha256": native_reload_file_sha256,
            "receipt_sha256": native_reload["sha256"].removeprefix("sha256:"),
        },
        "source_plan_sha256": checkpoint["source"]["plan_sha256"],
        "prediction_probe": prediction_probe,
        "checkpoint_manifest": checkpoint,
    }


def _miles_root() -> Path:
    spec = importlib.util.find_spec("miles")
    if spec is None or spec.origin is None:
        raise ValueError("pinned Miles package is unavailable")
    root = Path(spec.origin).resolve().parents[1]
    eager_root = root / "miles/backends/megatron_utils/megatron_to_hf"
    expected_eager = {
        Path(name).relative_to("miles/backends/megatron_utils/megatron_to_hf")
        for name in MILES_CONVERTER_SOURCES
        if name.startswith("miles/backends/megatron_utils/megatron_to_hf/")
    }
    actual_eager = {path.relative_to(eager_root) for path in eager_root.rglob("*.py")}
    if actual_eager != expected_eager:
        raise ValueError("pinned Miles converter import closure changed")
    for name, expected in MILES_CONVERTER_SOURCES.items():
        if _hash(root / name) != expected:
            raise ValueError("pinned Miles converter source changed")
    return root


def _sidecar_names(model: dict[str, Any]) -> set[str]:
    names = {item["path"] for item in model["files"]}
    weights = {name for name in names if name.endswith(".safetensors")}
    index = "model.safetensors.index.json"
    sidecars = names - weights - {index}
    if (
        index not in names
        or not weights
        or not sidecars
        or any(Path(name).name != name for name in names)
    ):
        raise ValueError("exact base inference inventory is malformed")
    return sidecars


def _verify_base(model: dict[str, Any]) -> dict[str, tuple[int, int, int, int, int]]:
    root = Path(model["root"])
    if root.is_symlink() or not root.is_dir():
        raise ValueError("exact base model is missing or indirect")
    snapshot = {}
    for item in model["files"]:
        path = root / item["path"]
        if path.is_symlink() or not path.is_file():
            raise ValueError("exact base model contains a missing or indirect file")
        if _hash(path) != item["sha256"].removeprefix("sha256:"):
            raise ValueError("exact base model file digest mismatch")
        stat = path.stat()
        snapshot[item["path"]] = (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )
    config = json.loads((root / "config.json").read_text())
    if config.get("auto_map") or config.get("text_config", {}).get("auto_map"):
        raise ValueError("remote model code is forbidden")
    return snapshot


def _hash_range(path: Path, start: int, size: int) -> str:
    before = path.stat()
    result = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(start)
        remaining = size
        while remaining:
            chunk = stream.read(min(remaining, HASH_CHUNK_BYTES))
            if not chunk:
                raise ValueError("truncated safetensors payload")
            result.update(chunk)
            remaining -= len(chunk)
    after = path.stat()
    if any(getattr(before, name) != getattr(after, name) for name in _STABLE_FILE_ATTRIBUTES):
        raise ValueError("safetensors shard changed while hashing")
    return result.hexdigest()


def tensor_inventory(root: Path, selected_names: set[str] | None = None) -> list[dict[str, Any]]:
    """Hash and prove finite every BF16 tensor without retaining a whole shard."""
    import torch
    from safetensors import safe_open

    layout, shards = _safetensor_layout(root)
    result = []
    total_tensor_bytes = 0
    for shard_name in shards:
        path = root / shard_name
        if path.is_symlink() or not path.is_file():
            raise ValueError("safetensors shard is missing or indirect")
        before = path.stat()
        with path.open("rb") as stream:
            raw_size = stream.read(8)
            if len(raw_size) != 8:
                raise ValueError("invalid safetensors header")
            header_size = int.from_bytes(raw_size, "little")
            if not 2 <= header_size <= 64 * 1024**2:
                raise ValueError("safetensors header exceeds the bound")
            header = json.loads(stream.read(header_size))
        rows = {key: value for key, value in header.items() if key != "__metadata__"}
        expected_keys = {key for key, value in layout.items() if value["shard"] == shard_name}
        if set(rows) != expected_keys:
            raise ValueError("safetensors header/layout key mismatch")
        intervals = []
        with safe_open(path, framework="pt", device="cpu") as opened:
            if set(opened.keys()) != set(rows):
                raise ValueError("safetensors value/header key mismatch")
            for key, value in rows.items():
                offsets = value.get("data_offsets") if isinstance(value, dict) else None
                if (
                    value.get("dtype") != "BF16"
                    or value.get("shape") != layout[key]["shape"]
                    or not isinstance(offsets, list)
                    or len(offsets) != 2
                    or any(type(offset) is not int for offset in offsets)
                    or not 0 <= offsets[0] < offsets[1]
                ):
                    raise ValueError("safetensors tensor metadata is invalid")
                size = offsets[1] - offsets[0]
                if size != 2 * math.prod(value["shape"]):
                    raise ValueError("BF16 tensor byte size differs from shape")
                intervals.append((offsets[0], offsets[1], key))
                total_tensor_bytes += size
                tensor = opened.get_tensor(key)
                flat = tensor.reshape(-1)
                if any(
                    not bool(torch.isfinite(flat[start : start + 8_388_608]).all().item())
                    for start in range(0, flat.numel(), 8_388_608)
                ):
                    raise ValueError("safetensors contains a non-finite BF16 value")
                if selected_names is None or key in selected_names:
                    payload_sha256 = _hash_range(path, 8 + header_size + offsets[0], size)
                    result.append(
                        {
                            "name": key,
                            "shape": value["shape"],
                            "dtype": "BF16",
                            "shard": shard_name,
                            "bytes": size,
                            "sha256": payload_sha256,
                            "value_sha256": payload_sha256,
                            "all_values_finite": True,
                        }
                    )
        ordered = sorted(intervals)
        data_size = path.stat().st_size - 8 - header_size
        if (
            not ordered
            or ordered[0][0] != 0
            or ordered[-1][1] != data_size
            or any(left[1] != right[0] for left, right in zip(ordered, ordered[1:], strict=False))
        ):
            raise ValueError("safetensors payload has a gap or overlap")
        after = path.stat()
        if any(getattr(before, name) != getattr(after, name) for name in _STABLE_FILE_ATTRIBUTES):
            raise ValueError("safetensors shard changed during tensor inventory")
    index = json.loads((root / "model.safetensors.index.json").read_text())
    if (
        index.get("weight_map") != {key: value["shard"] for key, value in layout.items()}
        or index.get("metadata", {}).get("total_size") != total_tensor_bytes
    ):
        raise ValueError("safetensors index total size or weight map is inconsistent")
    if selected_names is not None and {row["name"] for row in result} != selected_names:
        raise ValueError("selected tensor inventory is incomplete")
    return sorted(result, key=lambda row: row["name"])


class _SourceUnpickler(pickle.Unpickler):
    """Read historical Megatron metadata without importing mutable trainer code."""

    def find_class(self, module: str, name: str) -> Any:
        if module.startswith(("megatron", "glm")):

            class MetadataOnly:
                def __init__(self, *args: Any, **kwargs: Any) -> None:
                    del args, kwargs

            return MetadataOnly
        return super().find_class(module, name)


def _source_value_inventory(generation: Path) -> list[dict[str, Any]]:
    """Independently reload DCP and map every trained value to its exact HF key."""
    import torch
    import torch.distributed.checkpoint as dist_cp
    from miles.backends.megatron_utils.megatron_to_hf import convert_to_hf, remove_padding

    _miles_root()

    class Reader(dist_cp.FileSystemReader):
        def read_metadata(self) -> Any:
            path = self.fs.concat_path(self.path, ".metadata")
            with self.fs.create_stream(path, "rb") as stream:
                metadata = _SourceUnpickler(stream).load()
            if getattr(metadata, "storage_meta", None) is None:
                metadata.storage_meta = dist_cp.StorageMeta()
            metadata.storage_meta.load_id = self.load_id
            if metadata.planner_data is None:
                metadata.planner_data = {}
            return metadata

    class Planner(dist_cp.default_planner.DefaultLoadPlanner):
        def set_up_planner(
            self,
            state_dict: Any,
            metadata: Any = None,
            is_coordinator: bool = False,
        ) -> None:
            for key, value in metadata.state_dict_metadata.items():
                if "optimizer" in key or "_state" in key:
                    continue
                if isinstance(value, dist_cp.metadata.TensorStorageMetadata):
                    value = torch.empty(value.size, dtype=value.properties.dtype)
                state_dict[key] = value
            super().set_up_planner(state_dict, metadata, is_coordinator)

    original_unpickler = pickle.Unpickler
    try:
        pickle.Unpickler = _SourceUnpickler
        arguments = torch.load(generation / "common.pt", weights_only=False)["args"]
    finally:
        pickle.Unpickler = original_unpickler
    state: dict[str, Any] = {}
    dist_cp.state_dict_loader._load_state_dict(
        state,
        storage_reader=Reader(generation),
        planner=Planner(),
        no_dist=True,
    )
    arguments.sglang_enable_ep_moe = False

    def expanded(name: str, tensor: Any) -> Any:
        layer_match = re.search(r"\.layers\.(\d+)\.", name)
        layer_rows = [(name, tensor)]
        if ".layers." in name and layer_match is None:
            if tensor.shape[0] != arguments.num_layers:
                raise ValueError("DCP layer-stacked tensor has an invalid leading dimension")
            layer_rows = [
                (name.replace(".layers.", f".layers.{index}."), tensor[index])
                for index in range(arguments.num_layers)
            ]
        for layer_name, layer_tensor in layer_rows:
            expert_match = re.search(r"mlp\.experts\.(.+)\.weight(\d+)", layer_name)
            if ".experts." in layer_name and expert_match is None:
                if layer_tensor.shape[0] != arguments.num_experts:
                    raise ValueError("DCP expert-stacked tensor has an invalid leading dimension")
                for index in range(arguments.num_experts):
                    yield (
                        layer_name.replace(".experts.experts.", ".experts.") + str(index),
                        layer_tensor[index],
                    )
            else:
                yield layer_name, layer_tensor

    rows: dict[str, dict[str, Any]] = {}
    try:
        for source_name, source_tensor in state.items():
            source_flat = source_tensor.reshape(-1)
            if any(
                not bool(torch.isfinite(source_flat[start : start + 8_388_608]).all().item())
                for start in range(0, source_flat.numel(), 8_388_608)
            ):
                raise ValueError("DCP source contains a non-finite trained value")
            for expanded_name, expanded_tensor in expanded(
                "module.module." + source_name, source_tensor
            ):
                selected = remove_padding(expanded_name, expanded_tensor, VOCAB_SIZE)
                for hf_name, hf_tensor in convert_to_hf(
                    arguments, "qwen3_5", expanded_name, selected
                ):
                    contiguous = hf_tensor.detach().to(device="cpu").contiguous()
                    if contiguous.dtype != torch.bfloat16 or hf_name in rows:
                        raise ValueError("DCP source mapping is duplicate or not BF16")
                    raw = contiguous.view(torch.uint8).numpy()
                    rows[hf_name] = {
                        "name": hf_name,
                        "shape": list(contiguous.shape),
                        "dtype": "BF16",
                        "value_sha256": hashlib.sha256(raw).hexdigest(),
                    }
    finally:
        del state
    if not rows:
        raise ValueError("DCP source value inventory is empty")
    return [rows[name] for name in sorted(rows)]


def _raw_contract(base: Path, candidate: Path) -> tuple[dict[str, Any], list[str]]:
    base_layout, _ = _safetensor_layout(base)
    candidate_layout, _ = _safetensor_layout(candidate)
    auxiliary = {key for key in base_layout if key.startswith(FROZEN_AUXILIARY_PREFIXES)}
    missing = set(base_layout) - set(candidate_layout)
    shape_drift = {
        key
        for key in set(base_layout) & set(candidate_layout)
        if base_layout[key]["shape"] != candidate_layout[key]["shape"]
    }
    if (
        not auxiliary
        or not any(key.startswith("model.visual.") for key in auxiliary)
        or not any(key.startswith("mtp.") for key in auxiliary)
        or set(candidate_layout) - set(base_layout)
        or missing != auxiliary
        or shape_drift
        or any(value["dtype"] != "BF16" for value in candidate_layout.values())
        or any(base_layout[key]["dtype"] != "BF16" for key in auxiliary)
    ):
        raise ValueError("raw Miles export differs outside frozen visual/MTP auxiliaries")
    return base_layout, sorted(auxiliary)


def _restore_auxiliary(
    base: Path, candidate: Path, base_layout: dict[str, Any], names: list[str]
) -> None:
    from safetensors.torch import save_file

    index_path = candidate / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or set(weight_map) & set(names):
        raise ValueError("raw Miles index cannot accept frozen auxiliaries")
    selected = {key: base_layout[key] for key in names}
    groups = _shard_groups(selected, MAX_SHARD_BYTES)
    restored_bytes = 0
    for number, group in enumerate(groups, 1):
        shard_name = f"model-frozen-aux-{number:05d}-of-{len(groups):05d}.safetensors"
        if (candidate / shard_name).exists():
            raise FileExistsError("frozen auxiliary shard already exists")
        tensors = {key: _load_tensor(base, base_layout[key]["shard"], key).clone() for key in group}
        if any(tensor.dtype.is_floating_point is False for tensor in tensors.values()):
            raise ValueError("frozen auxiliary tensor is not floating point")
        save_file(tensors, candidate / shard_name, metadata={"format": "pt"})
        _fsync_file(candidate / shard_name)
        for key, tensor in tensors.items():
            weight_map[key] = shard_name
            restored_bytes += tensor.numel() * tensor.element_size()
    metadata = index.get("metadata")
    if not isinstance(metadata, dict) or type(metadata.get("total_size")) is not int:
        raise ValueError("raw Miles index total size is invalid")
    metadata["total_size"] += restored_bytes
    index_path.write_text(json.dumps(index, sort_keys=True) + "\n")
    _fsync_file(index_path)


def _model_files(root: Path, sidecars: set[str]) -> dict[str, dict[str, Any]]:
    _, shards = _safetensor_layout(root)
    expected = set(shards) | sidecars | {"model.safetensors.index.json"}
    actual = {path.name for path in root.iterdir()}
    if actual != expected or any(
        path.is_symlink() or not path.is_file() for path in root.iterdir()
    ):
        raise ValueError("final HF export contains an unknown or indirect entry")
    return {
        name: {"bytes": (root / name).stat().st_size, "sha256": _hash(root / name)}
        for name in sorted(expected)
    }


def _invoke_converter(source: Path, destination: Path, metadata: Path, log: Path) -> None:
    miles_root = _miles_root()
    argv = [
        sys.executable,
        str(miles_root / "tools/convert_torch_dist_to_hf.py"),
        "--input-dir",
        str(source),
        "--output-dir",
        str(destination),
        "--origin-hf-dir",
        str(metadata),
        "--chunk-size",
        str(MAX_SHARD_BYTES),
        "--vocab-size",
        str(VOCAB_SIZE),
    ]
    passthrough = {
        name: os.environ[name]
        for name in (
            "PATH",
            "LD_LIBRARY_PATH",
            "LIBRARY_PATH",
            "CUDA_HOME",
            "CUDA_VISIBLE_DEVICES",
            "TMPDIR",
        )
        if name in os.environ
    }
    env = {
        **passthrough,
        "PYTHONPATH": str(miles_root),
        "PYTHONNOUSERSITE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "WANDB_MODE": "disabled",
        "WANDB_DISABLED": "true",
    }
    with log.open("x") as stream:
        os.chmod(log, 0o600)
        result = subprocess.run(
            argv,
            cwd=miles_root,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=CONVERTER_TIMEOUT_SECONDS,
            check=False,
        )
    if result.returncode:
        raise RuntimeError("pinned Miles conversion failed; private log preserved")


def export(
    *,
    active_canary_binding_path: Path,
    active_canary_binding_sha256: str,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    terminal_path: Path,
    terminal_sha256: str,
    native_reload_path: Path,
    native_reload_sha256: str,
    output: Path,
    prepared_source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create one exact HF directory; never reuse or replace a destination."""
    import torch

    if torch.cuda.is_available():
        raise ValueError("Miles HF export is CPU-only; GPU reload is a separate gate")
    source = bind_source(
        active_canary_binding_path=active_canary_binding_path,
        active_canary_binding_sha256=active_canary_binding_sha256,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        terminal_path=terminal_path,
        terminal_sha256=terminal_sha256,
        native_reload_path=native_reload_path,
        native_reload_sha256=native_reload_sha256,
        validate_historical=prepared_source is None,
    )
    public_source = {key: item for key, item in source.items() if key != "checkpoint_manifest"}
    if prepared_source is not None and public_source != prepared_source:
        raise ValueError("runtime source differs from the fully validated prepared binding")
    checkpoint = source["checkpoint_manifest"]
    base = Path(checkpoint["model"]["root"])
    checkpoint_root = Path(checkpoint["root"])
    generation = checkpoint_root / f"iter_{checkpoint['rollout_index']:07d}"
    attempt = output.with_name(output.name + ".partial")
    candidate = attempt / "model"
    metadata = attempt / "base-metadata"
    private_log = attempt / "private-converter.log"
    if any(path.exists() or path.is_symlink() for path in (output, attempt)):
        raise FileExistsError("export destination or attempt artifact already exists")
    resolved_output = output.resolve()
    if any(
        resolved_output == root.resolve()
        or resolved_output.is_relative_to(root.resolve())
        or root.resolve().is_relative_to(resolved_output)
        for root in (base, checkpoint_root)
    ):
        raise ValueError("export destination must be disjoint from source and exact base")
    sidecars = _sidecar_names(checkpoint["model"])
    base_before = _verify_base(checkpoint["model"])
    base_model_bytes = sum(
        item.get("size", (base / item["path"]).stat().st_size)
        for item in checkpoint["model"]["files"]
        if item["path"].endswith(".safetensors")
    )
    required_free_bytes = base_model_bytes + 8 * 1024**3
    existing = output.parent
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    if shutil.disk_usage(existing).free < required_free_bytes:
        raise ValueError("insufficient free storage for the create-once HF export")
    attempt.mkdir(mode=0o700)
    metadata.mkdir(mode=0o700)
    for name in sidecars:
        shutil.copyfile(base / name, metadata / name)
    _invoke_converter(generation, candidate, metadata, private_log)
    base_layout, restored = _raw_contract(base, candidate)
    trained_tensors = tensor_inventory(candidate)
    source_value_inventory = _source_value_inventory(generation)
    converted_value_inventory = [
        {
            "name": row["name"],
            "shape": row["shape"],
            "dtype": row["dtype"],
            "value_sha256": row["value_sha256"],
        }
        for row in trained_tensors
    ]
    if source_value_inventory != converted_value_inventory:
        raise ValueError("HF trained tensor values differ from an independent DCP reload")
    base_auxiliary_tensors = tensor_inventory(base, set(restored))
    _restore_auxiliary(base, candidate, base_layout, restored)
    final_layout, _ = _safetensor_layout(candidate)
    if {key: (value["shape"], value["dtype"]) for key, value in base_layout.items()} != {
        key: (value["shape"], value["dtype"]) for key, value in final_layout.items()
    }:
        raise ValueError("final HF tensor key/shape/BF16 parity failed")
    for name in sidecars:
        expected = next(item for item in checkpoint["model"]["files"] if item["path"] == name)
        if _hash(candidate / name) != expected["sha256"].removeprefix("sha256:"):
            raise ValueError("final HF sidecar differs from exact base")
    tensors = tensor_inventory(candidate)
    restored_set = set(restored)
    for row in tensors:
        row["source"] = "frozen_base_auxiliary" if row["name"] in restored_set else "trained"
    final_by_name = {row["name"]: row for row in tensors}
    if any(final_by_name[row["name"]]["sha256"] != row["sha256"] for row in base_auxiliary_tensors):
        raise ValueError("restored frozen auxiliary tensor differs from exact base")
    files = _model_files(candidate, sidecars)
    from .miles_reload import _verify_checkpoint

    _verify_checkpoint(checkpoint, hashes=True)
    if _verify_base(checkpoint["model"]) != base_before:
        raise ValueError("exact base changed during export")
    value = _write(
        candidate / "EXPORT.json",
        {
            "schema": EXPORT_SCHEMA,
            "status": "exported",
            "source": public_source,
            "miles": {
                "image": miles.IMAGE,
                "source_commit": MILES_SOURCE_COMMIT,
                "converter_sources": MILES_CONVERTER_SOURCES,
            },
            "model": {
                "repo": MODEL_REPO,
                "revision": MODEL_REVISION,
                "base_root": str(base),
                "native_parallelism": {"tensor": 4, "context": 2, "world_size": 8},
            },
            "output_root": str(output),
            "files": files,
            "tensor_inventory": tensors,
            "tensor_inventory_sha256": digest(tensors),
            "sidecars": {name: files[name] for name in sorted(sidecars)},
            "trained_tensor_count": len(tensors) - len(restored),
            "restored_base_tensor_count": len(restored),
            "restored_base_tensor_inventory_sha256": digest(base_auxiliary_tensors),
            "tensor_bytes": sum(row["bytes"] for row in tensors),
            "all_tensor_values_finite": all(row["all_values_finite"] for row in tensors),
            "source_equivalence": {
                "method": "independent_dcp_reload_and_pinned_mapping_value_hash_v1",
                "converter_input": str(generation),
                "common_pt_sha256": _hash(generation / "common.pt"),
                "dcp_metadata_sha256": _hash(generation / ".metadata"),
                "trained_tensor_inventory_sha256": digest(trained_tensors),
                "source_value_inventory_sha256": digest(source_value_inventory),
                "all_trained_values_match_source": True,
            },
            "resource_guard": {
                "base_model_bytes": base_model_bytes,
                "minimum_free_bytes": required_free_bytes,
                "converter_passes": 1,
            },
            "dtype": "BF16",
            "base_layout_sha256": digest(
                {
                    key: {"shape": value["shape"], "dtype": value["dtype"]}
                    for key, value in sorted(base_layout.items())
                }
            ),
            "output_layout_sha256": digest(
                {
                    key: {"shape": value["shape"], "dtype": value["dtype"]}
                    for key, value in sorted(final_layout.items())
                }
            ),
            "optimizer_updates_executed": 0,
            "source_checkpoint_unchanged": True,
            "base_model_unchanged": True,
            "create_only": True,
            "gpu_reload_verified": False,
            "completed_at": time.time(),
        },
    )
    _fsync_file(candidate / "EXPORT.json")
    _fsync_directory(candidate)
    shutil.rmtree(metadata)
    private_log.unlink()
    _rename_noreplace(candidate, output)
    attempt.rmdir()
    _fsync_directory(output.parent)
    return value


def _validate_export_source(source: Any) -> dict[str, Any]:
    """Reopen the exact historical receipts behind a published export."""
    if not isinstance(source, dict) or set(source) != _EXPORT_SOURCE_FIELDS:
        raise ValueError("Miles HF export source is not exactly bound")
    references = {}
    for name in (
        "active_canary_binding",
        "checkpoint",
        "terminal_acceptance",
        "native_reload_acceptance",
    ):
        reference = source.get(name)
        if (
            not isinstance(reference, dict)
            or set(reference) != {"path", "file_sha256", "receipt_sha256"}
            or not Path(str(reference.get("path", ""))).is_absolute()
            or any(
                _SHA256.fullmatch(str(reference.get(key, ""))) is None
                for key in ("file_sha256", "receipt_sha256")
            )
        ):
            raise ValueError("Miles HF export source reference changed")
        references[name] = reference
    rebound = bind_source(
        active_canary_binding_path=Path(references["active_canary_binding"]["path"]),
        active_canary_binding_sha256=references["active_canary_binding"]["file_sha256"],
        checkpoint_path=Path(references["checkpoint"]["path"]),
        checkpoint_sha256=references["checkpoint"]["file_sha256"],
        terminal_path=Path(references["terminal_acceptance"]["path"]),
        terminal_sha256=references["terminal_acceptance"]["file_sha256"],
        native_reload_path=Path(references["native_reload_acceptance"]["path"]),
        native_reload_sha256=references["native_reload_acceptance"]["file_sha256"],
    )
    public = {key: item for key, item in rebound.items() if key != "checkpoint_manifest"}
    if source != public:
        raise ValueError("Miles HF export source receipts changed")
    return rebound["checkpoint_manifest"]


def _validate_prepared_export(
    value: dict[str, Any], path: Path, expected_sha256: str, prepared: dict[str, Any]
) -> None:
    """Use only the exact export binding compiled by the full historical validator."""
    if (
        set(prepared)
        != {
            "path",
            "file_sha256",
            "receipt_sha256",
            "tensor_inventory_sha256",
            "active_canary_binding",
        }
        or prepared.get("path") != str(path)
        or prepared.get("file_sha256") != expected_sha256.removeprefix("sha256:")
        or prepared.get("receipt_sha256") != value["sha256"].removeprefix("sha256:")
        or prepared.get("tensor_inventory_sha256") != value.get("tensor_inventory_sha256")
        or prepared.get("active_canary_binding")
        != value.get("source", {}).get("active_canary_binding")
    ):
        raise ValueError("Miles HF runtime export differs from its fully validated plan")


def inspect_export(
    path: Path,
    expected_sha256: str,
    *,
    prepared_export: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    value, _ = _snapshot_json(path, expected_sha256)
    _sealed(value, EXPORT_SCHEMA)
    root = path.parent
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Miles HF export root is missing or indirect")
    source = value.get("source")
    if (
        not isinstance(source, dict)
        or set(source) != _EXPORT_SOURCE_FIELDS
    ):
        raise ValueError("Miles HF export source binding changed")
    _validate_prediction_probe(source.get("prediction_probe"))
    checkpoint = _validate_export_source(value.get("source")) if prepared_export is None else None
    if prepared_export is not None:
        _validate_prepared_export(value, path, expected_sha256, prepared_export)
    files = value.get("files")
    inventory = value.get("tensor_inventory")
    source_equivalence = value.get("source_equivalence")
    resource_guard = value.get("resource_guard")
    model = value.get("model")
    if (
        set(value) != _EXPORT_FIELDS
        or path.name != "EXPORT.json"
        or value.get("status") != "exported"
        or value.get("output_root") != str(root)
        or value.get("dtype") != "BF16"
        or value.get("all_tensor_values_finite") is not True
        or value.get("optimizer_updates_executed") != 0
        or value.get("source_checkpoint_unchanged") is not True
        or value.get("base_model_unchanged") is not True
        or value.get("create_only") is not True
        or value.get("gpu_reload_verified") is not False
        or not isinstance(value.get("completed_at"), numbers.Real)
        or isinstance(value.get("completed_at"), bool)
        or not math.isfinite(float(value["completed_at"]))
        or value["completed_at"] <= 0
        or value.get("miles")
        != {
            "image": miles.IMAGE,
            "source_commit": MILES_SOURCE_COMMIT,
            "converter_sources": MILES_CONVERTER_SOURCES,
        }
        or not isinstance(model, dict)
        or set(model) != _EXPORT_MODEL_FIELDS
        or model.get("repo") != MODEL_REPO
        or model.get("revision") != MODEL_REVISION
        or model.get("native_parallelism") != {"tensor": 4, "context": 2, "world_size": 8}
        or not Path(str(model.get("base_root", ""))).is_absolute()
        or not isinstance(files, dict)
        or not files
        or any(
            not isinstance(name, str)
            or Path(name).name != name
            or not isinstance(item, dict)
            or set(item) != _EXPORT_FILE_FIELDS
            or type(item.get("bytes")) is not int
            or item["bytes"] < 1
            or _SHA256.fullmatch(str(item.get("sha256", ""))) is None
            for name, item in files.items()
        )
        or not isinstance(inventory, list)
        or not inventory
        or any(not isinstance(row, dict) or set(row) != _EXPORT_TENSOR_FIELDS for row in inventory)
        or not isinstance(source_equivalence, dict)
        or set(source_equivalence) != _EXPORT_SOURCE_EQUIVALENCE_FIELDS
        or source_equivalence.get("method")
        != "independent_dcp_reload_and_pinned_mapping_value_hash_v1"
        or source_equivalence.get("all_trained_values_match_source") is not True
        or any(
            _SHA256.fullmatch(str(source_equivalence.get(key, ""))) is None
            for key in (
                "common_pt_sha256",
                "dcp_metadata_sha256",
                "trained_tensor_inventory_sha256",
                "source_value_inventory_sha256",
            )
        )
        or not isinstance(resource_guard, dict)
        or set(resource_guard) != _EXPORT_RESOURCE_GUARD_FIELDS
        or type(resource_guard.get("base_model_bytes")) is not int
        or resource_guard["base_model_bytes"] < 1
        or resource_guard.get("minimum_free_bytes")
        != resource_guard["base_model_bytes"] + 8 * 1024**3
        or resource_guard.get("converter_passes") != 1
        or _SHA256.fullmatch(str(value.get("restored_base_tensor_inventory_sha256", ""))) is None
        or _SHA256.fullmatch(str(value.get("base_layout_sha256", ""))) is None
        or value.get("base_layout_sha256") != value.get("output_layout_sha256")
        or {item.name for item in root.iterdir()} != set(files) | {"EXPORT.json"}
    ):
        raise ValueError("Miles HF export receipt is incomplete")
    if checkpoint is not None:
        generation = Path(checkpoint["root"]) / f"iter_{checkpoint['rollout_index']:07d}"
        expected_base_bytes = sum(
            item.get("size", (Path(checkpoint["model"]["root"]) / item["path"]).stat().st_size)
            for item in checkpoint["model"]["files"]
            if item["path"].endswith(".safetensors")
        )
        if (
            model["base_root"] != checkpoint["model"]["root"]
            or source_equivalence["converter_input"] != str(generation)
            or source_equivalence["common_pt_sha256"] != _hash(generation / "common.pt")
            or source_equivalence["dcp_metadata_sha256"] != _hash(generation / ".metadata")
            or resource_guard["base_model_bytes"] != expected_base_bytes
        ):
            raise ValueError("Miles HF export source-equivalence witness changed")
    for name, item in files.items():
        candidate = root / name
        if (
            candidate.is_symlink()
            or not candidate.is_file()
            or candidate.name != name
            or candidate.stat().st_size != item["bytes"]
            or _hash(candidate) != item["sha256"]
        ):
            raise ValueError("Miles HF export file inventory changed")
    tensors = tensor_inventory(root)
    for row in tensors:
        row["source"] = (
            "frozen_base_auxiliary"
            if row["name"].startswith(FROZEN_AUXILIARY_PREFIXES)
            else "trained"
        )
    restored = [row for row in tensors if row["source"] == "frozen_base_auxiliary"]
    trained = [
        {key: item for key, item in row.items() if key != "source"}
        for row in tensors
        if row["source"] == "trained"
    ]
    layout, shards = _safetensor_layout(root)
    layout_digest = digest(
        {
            key: {"shape": item["shape"], "dtype": item["dtype"]}
            for key, item in sorted(layout.items())
        }
    )
    sidecar_names = set(files) - set(shards) - {"model.safetensors.index.json"}
    if (
        tensors != inventory
        or digest(tensors) != value["tensor_inventory_sha256"]
        or value.get("trained_tensor_count") != len(trained)
        or value.get("restored_base_tensor_count") != len(restored)
        or not any(row["name"].startswith("model.visual.") for row in restored)
        or not any(row["name"].startswith("mtp.") for row in restored)
        or value.get("tensor_bytes") != sum(row["bytes"] for row in tensors)
        or value.get("all_tensor_values_finite") is not True
        or source_equivalence["trained_tensor_inventory_sha256"] != digest(trained)
        or source_equivalence["source_value_inventory_sha256"]
        != digest(
            [
                {
                    "name": row["name"],
                    "shape": row["shape"],
                    "dtype": row["dtype"],
                    "value_sha256": row["value_sha256"],
                }
                for row in trained
            ]
        )
        or value.get("output_layout_sha256") != layout_digest
        or value.get("sidecars") != {name: files[name] for name in sorted(sidecar_names)}
    ):
        raise ValueError("Miles HF tensor inventory changed")
    if checkpoint is not None:
        base_tensors = tensor_inventory(
            Path(checkpoint["model"]["root"]), {row["name"] for row in restored}
        )
        if digest(base_tensors) != value["restored_base_tensor_inventory_sha256"]:
            raise ValueError("Miles HF frozen auxiliary witness changed")
        source_values = _source_value_inventory(generation)
        exported_values = [
            {
                "name": row["name"],
                "shape": row["shape"],
                "dtype": row["dtype"],
                "value_sha256": row["value_sha256"],
            }
            for row in trained
        ]
        if (
            source_values != exported_values
            or digest(source_values) != source_equivalence["source_value_inventory_sha256"]
        ):
            raise ValueError("Miles HF source value-equivalence witness changed")
    return value, tensors


def _runtime_value_observation(tensor: Any) -> tuple[bool, str]:
    import torch

    flat = tensor.detach().reshape(-1)
    finite = all(
        bool(torch.isfinite(flat[start : start + 8_388_608]).all().item())
        for start in range(0, flat.numel(), 8_388_608)
    )
    raw = (
        tensor.detach()
        .to(device="cpu", dtype=torch.bfloat16)
        .contiguous()
        .view(torch.uint8)
        .numpy()
        .tobytes()
    )
    return finite, hashlib.sha256(raw).hexdigest()


def _prediction_receipt(last_logits: Any) -> dict[str, Any]:
    """Reduce logits to the only privacy-safe semantic witness we persist."""
    import torch

    if (
        not torch.is_tensor(last_logits)
        or last_logits.ndim != 1
        or last_logits.numel() < PREDICTION_TOP_K + 1
    ):
        raise ValueError("fixed prediction probe logits have an unexpected shape")
    values, indices = torch.topk(
        last_logits.float(),
        k=PREDICTION_TOP_K + 1,
        largest=True,
        sorted=True,
    )
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError("fixed prediction probe returned non-finite logits")
    if not bool(
        (values[PREDICTION_TOP_K - 1] - values[PREDICTION_TOP_K]).item() >= PREDICTION_MARGIN
    ):
        raise ValueError("fixed prediction top-k boundary is not numerically robust")
    predictions = [int(item) for item in indices[:PREDICTION_TOP_K].cpu().tolist()]
    return {
        "schema": PREDICTION_SCHEMA,
        "probe_id": PREDICTION_PROBE_ID,
        "input_ids_sha256": "sha256:" + digest(list(PREDICTION_INPUT_IDS)),
        "sequence_length": len(PREDICTION_INPUT_IDS),
        "top_k": PREDICTION_TOP_K,
        "selection_margin_threshold": PREDICTION_MARGIN,
        "selection_margin_satisfied": True,
        "prediction_sha256": "sha256:" + digest(predictions),
        "logits_included": False,
        "task_content_included": False,
        "benchmark_content_included": False,
    }


def _hf_prediction_probe(model: Any) -> dict[str, Any]:
    """Run the native observer's fixed task-free probe on the HF implementation."""
    import torch

    previous_training = bool(model.training)
    model.eval()
    tokens = torch.tensor(
        [PREDICTION_INPUT_IDS],
        dtype=torch.long,
        device="cuda:0",
    )
    try:
        with torch.no_grad():
            output = model(
                input_ids=tokens,
                attention_mask=None,
                position_ids=None,
                use_cache=False,
                return_dict=True,
            )
            logits = getattr(output, "logits", None)
            if (
                not torch.is_tensor(logits)
                or logits.ndim != 3
                or logits.shape[:2] != (1, len(PREDICTION_INPUT_IDS))
            ):
                raise ValueError("HF fixed prediction probe returned an unexpected shape")
            receipt = _prediction_receipt(logits[0, -1])
    finally:
        model.train(previous_training)
    return receipt


def gpu_reload(
    export_path: Path,
    export_sha256: str,
    output: Path,
    *,
    run_name: str,
    prepared_export: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load all HF values and run one fixed non-task semantic probe on one GPU."""
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText

    if _RUN_NAME.fullmatch(run_name) is None:
        raise ValueError("HF reload run name is invalid")
    if output.exists() or output.is_symlink():
        raise FileExistsError("GPU reload receipt already exists")
    if output.resolve().is_relative_to(export_path.parent.resolve()):
        raise ValueError("GPU reload receipt must be outside the immutable export")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("HF reload gate requires exactly one visible GPU")
    before, tensors = inspect_export(export_path, export_sha256, prepared_export=prepared_export)
    root = export_path.parent
    config = AutoConfig.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    if getattr(config, "auto_map", None) or getattr(config.get_text_config(), "auto_map", None):
        raise ValueError("remote model code is forbidden")
    torch.cuda.set_device(0)
    model, info = AutoModelForImageTextToText.from_pretrained(
        root,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        local_files_only=True,
        trust_remote_code=False,
        output_loading_info=True,
    )
    artifact = {row["name"]: row for row in tensors}
    state = model.state_dict()
    artifact_only = set(artifact) - set(state)
    if (
        any(not name.startswith("mtp.") for name in artifact_only)
        or set(state) - set(artifact)
        or any(list(tensor.shape) != artifact[name]["shape"] for name, tensor in state.items())
        or any(
            tensor.dtype != torch.bfloat16 or tensor.device.type != "cuda"
            for tensor in state.values()
        )
        or info.get("missing_keys")
        or info.get("mismatched_keys")
        or info.get("error_msgs")
        or set(info.get("unexpected_keys", [])) != artifact_only
    ):
        raise ValueError("HF GPU loader key/shape/BF16 contract failed")
    runtime_values = []
    for name, tensor in sorted(state.items()):
        finite, value_sha256 = _runtime_value_observation(tensor)
        if not finite or value_sha256 != artifact[name]["value_sha256"]:
            raise ValueError("HF GPU loader value/finiteness contract failed")
        runtime_values.append(
            {"name": name, "value_sha256": value_sha256, "all_values_finite": True}
        )
    runtime_layout = {
        name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        for name, tensor in sorted(state.items())
    }
    native_prediction = _validate_prediction_probe(before["source"].get("prediction_probe"))
    hf_prediction = _hf_prediction_probe(model)
    if hf_prediction != native_prediction:
        raise ValueError("HF prediction differs from the native trained-policy witness")
    peak_memory = torch.cuda.max_memory_allocated(0)
    gpu_name = torch.cuda.get_device_name(0)
    del state, model
    torch.cuda.empty_cache()
    after, _ = inspect_export(export_path, export_sha256, prepared_export=prepared_export)
    if after != before:
        raise ValueError("HF export changed during GPU reload")
    return _write(
        output,
        {
            "schema": RELOAD_SCHEMA,
            "status": "reload_validated",
            "run_name": run_name,
            "image": miles.IMAGE,
            "export_path": str(export_path),
            "export_file_sha256": export_sha256.removeprefix("sha256:"),
            "export_receipt_sha256": before["sha256"].removeprefix("sha256:"),
            "export_tensor_inventory_sha256": before["tensor_inventory_sha256"],
            "model_repo": MODEL_REPO,
            "model_revision": MODEL_REVISION,
            "runtime_tensor_count": len(runtime_layout),
            "artifact_only_mtp_tensor_count": len(artifact_only),
            "runtime_layout_sha256": digest(runtime_layout),
            "runtime_value_inventory_sha256": digest(runtime_values),
            "all_runtime_weights_loaded": True,
            "all_runtime_values_finite": True,
            "all_runtime_values_match_export": True,
            "source_export_unchanged": True,
            "native_prediction_probe": native_prediction,
            "hf_prediction_probe": hf_prediction,
            "semantic_prediction_match": True,
            "optimizer_updates": 0,
            "rollouts": 0,
            "verifier_calls": 0,
            "forwards": 1,
            "backwards": 0,
            "checkpoint_writes": 0,
            "wandb_events": 0,
            "gpus": 1,
            "gpu_name": gpu_name,
            "peak_memory_bytes": peak_memory,
            "external_gpu_release_verified": False,
            "serving_qualified": False,
            "completed_at": time.time(),
        },
    )


def _snapshot(path: Path) -> tuple[dict[str, Any], str]:
    """Read one regular JSON evidence file once and bind its exact bytes."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("HF reload evidence is missing or indirect")
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    attributes = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, key) != getattr(after, key) for key in attributes):
        raise ValueError("HF reload evidence changed while being read")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("HF reload evidence is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("HF reload evidence must contain an object")
    return value, hashlib.sha256(raw).hexdigest()


def _evidence_time(value: object) -> float:
    if isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("HF reload evidence timestamp is invalid") from exc
        if parsed.tzinfo is not None:
            return parsed.timestamp()
    raise ValueError("HF reload evidence timestamp is invalid")


def _exact_integer(value: object, expected: int) -> bool:
    return type(value) is int and value == expected


def _runtime_image_digest(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("HF reload runtime image ID is absent")
    match = re.fullmatch(
        r"(?:containerd|docker-pullable)://(?:[^@\s]+@)?sha256:([a-f0-9]{64})", value
    )
    if match is None:
        raise ValueError("HF reload runtime image ID is not immutable")
    return match.group(1)


def _validate_reload_result(value: dict[str, Any]) -> float:
    _sealed(value, RELOAD_SCHEMA)
    if (
        set(value) != _RELOAD_RESULT_FIELDS
        or value.get("status") != "reload_validated"
        or _RUN_NAME.fullmatch(str(value.get("run_name", ""))) is None
        or value.get("image") != miles.IMAGE
        or value.get("model_repo") != MODEL_REPO
        or value.get("model_revision") != MODEL_REVISION
        or _SHA256.fullmatch(str(value.get("export_file_sha256", ""))) is None
        or _SHA256.fullmatch(str(value.get("export_receipt_sha256", ""))) is None
        or _SHA256.fullmatch(str(value.get("export_tensor_inventory_sha256", ""))) is None
        or _SHA256.fullmatch(str(value.get("runtime_layout_sha256", ""))) is None
        or _SHA256.fullmatch(str(value.get("runtime_value_inventory_sha256", ""))) is None
        or type(value.get("runtime_tensor_count")) is not int
        or value["runtime_tensor_count"] < 1
        or type(value.get("artifact_only_mtp_tensor_count")) is not int
        or value["artifact_only_mtp_tensor_count"] < 0
        or value.get("all_runtime_weights_loaded") is not True
        or value.get("all_runtime_values_finite") is not True
        or value.get("all_runtime_values_match_export") is not True
        or value.get("source_export_unchanged") is not True
        or _validate_prediction_probe(value.get("native_prediction_probe"))
        != value.get("native_prediction_probe")
        or _validate_prediction_probe(value.get("hf_prediction_probe"))
        != value.get("hf_prediction_probe")
        or value.get("hf_prediction_probe") != value.get("native_prediction_probe")
        or value.get("semantic_prediction_match") is not True
        or any(
            not _exact_integer(value.get(key), expected)
            for key, expected in _RELOAD_WORK_EXPECTED.items()
        )
        or not _exact_integer(value.get("gpus"), 1)
        or not isinstance(value.get("gpu_name"), str)
        or not value["gpu_name"]
        or type(value.get("peak_memory_bytes")) is not int
        or value["peak_memory_bytes"] <= 0
        or value.get("external_gpu_release_verified") is not False
        or value.get("serving_qualified") is not False
    ):
        raise ValueError("HF reload process receipt is incomplete or conflicting")
    return _evidence_time(value.get("completed_at"))


def _validate_controller(
    value: dict[str, Any],
    result: dict[str, Any],
    result_path: Path,
    result_file_sha256: str,
    completed_at: float,
    plan: dict[str, Any],
    submission: dict[str, Any],
) -> float:
    from .miles_hf_export_job import (
        DEV_KUBE_CONTEXT,
        DEV_NAMESPACE_UID,
        validate_submission_binding,
    )

    submitted = validate_submission_binding(submission, plan, check_files=False)
    _sealed(value, RELOAD_CONTROLLER_SCHEMA)
    from .miles_event_evidence import validate_hf_event_journal

    validate_hf_event_journal(plan, submission, value.get("event_journal"), value)
    run_id = value.get("api_run_id")
    expected_name = f"{result['run_name']}-{str(run_id)[:8]}"
    pods = value.get("pods")
    observed_at = _evidence_time(value.get("observed_at"))
    expected_image = miles.IMAGE.rsplit("@sha256:", 1)[-1]
    if (
        set(value) != _CONTROLLER_FIELDS
        or value.get("status") != "succeeded"
        or value.get("cluster") != "dev"
        or value.get("api_base_url") != API_URLS["dev"]
        or value.get("kube_context") != DEV_KUBE_CONTEXT
        or value.get("namespace") != RELOAD_NAMESPACE
        or value.get("namespace_uid") != DEV_NAMESPACE_UID
        or value.get("source_plan_sha256") != "sha256:" + digest(plan)
        or value.get("source_request_sha256") != submitted["request_sha256"]
        or value.get("submission_binding_sha256") != submission["sha256"]
        or value.get("runtime_bundle_sha256") != submission["runtime_bundle_sha256"]
        or value.get("api_base_url") != submitted["api"]["base_url"]
        or value.get("run_name") != result["run_name"]
        or value.get("run_name") != plan.get("run_name")
        or value.get("reload_result_path") != str(result_path)
        or value.get("reload_result_file_sha256") != result_file_sha256
        or value.get("reload_result_sha256") != result["sha256"].removeprefix("sha256:")
        or _UUID.fullmatch(str(run_id)) is None
        or run_id != submitted["api"]["run_id"]
        or value.get("api_run_name") != expected_name
        or value.get("rayjob_name") != expected_name
        or any(
            _UUID.fullmatch(str(value.get(key))) is None
            for key in ("rayjob_uid", "workload_uid", "raycluster_uid")
        )
        or any(
            not isinstance(value.get(key), str) or not value[key]
            for key in ("workload_name", "raycluster_name")
        )
        or value.get("workload_owner_rayjob_uid") != value.get("rayjob_uid")
        or value.get("raycluster_owner_rayjob_uid") != value.get("rayjob_uid")
        or value.get("api_status") != "SUCCEEDED"
        or value.get("controller_status") != "SUCCEEDED"
        or not _exact_integer(value.get("effective_priority"), 10000)
        or value.get("automatic_requeue") is not False
        or not _exact_integer(value.get("workers"), 1)
        or not _exact_integer(value.get("gpus_per_worker"), 1)
        or not _exact_integer(value.get("total_gpus"), 1)
        or observed_at < completed_at
        or not isinstance(pods, list)
        or len(pods) != 1
    ):
        raise ValueError("HF reload controller evidence is incomplete or mismatched")
    pod = pods[0]
    if (
        not isinstance(pod, dict)
        or set(pod) != _POD_FIELDS
        or not isinstance(pod.get("name"), str)
        or not pod["name"]
        or _UUID.fullmatch(str(pod.get("uid"))) is None
        or pod.get("owner_raycluster_uid") != value["raycluster_uid"]
        or pod.get("phase") != "Succeeded"
        or not _exact_integer(pod.get("exit_code"), 0)
        or pod.get("termination_reason") != "Completed"
        or _evidence_time(pod.get("terminated_at")) < completed_at
        or _evidence_time(pod.get("terminated_at")) > observed_at
        or _runtime_image_digest(pod.get("runtime_image_id")) != expected_image
        or not _exact_integer(pod.get("container_restarts"), 0)
        or not _exact_integer(pod.get("gpus"), 1)
    ):
        raise ValueError("HF reload Pod evidence is incomplete or mismatched")
    identities = (
        run_id,
        value["rayjob_uid"],
        value["workload_uid"],
        value["raycluster_uid"],
        pod["uid"],
    )
    if len(set(identities)) != len(identities):
        raise ValueError("HF reload controller identities are not distinct")
    return observed_at


def _validate_release(
    value: dict[str, Any],
    result: dict[str, Any],
    result_path: Path,
    result_file_sha256: str,
    controller: dict[str, Any],
    controller_path: Path,
    controller_file_sha256: str,
    not_before: float,
    plan: dict[str, Any],
    submission: dict[str, Any],
) -> None:
    from .miles_event_evidence import validate_hf_release_journal

    _sealed(value, RELOAD_RELEASE_SCHEMA)
    validate_hf_release_journal(plan, submission, controller, value.get("observed_at"))
    identities = (
        "api_run_id",
        "api_run_name",
        "rayjob_name",
        "rayjob_uid",
        "workload_name",
        "workload_uid",
        "raycluster_name",
        "raycluster_uid",
    )
    if (
        set(value) != _RELEASE_FIELDS
        or value.get("status") != "released"
        or value.get("cluster") != "dev"
        or value.get("api_base_url") != API_URLS["dev"]
        or value.get("kube_context") != controller["kube_context"]
        or value.get("namespace") != RELOAD_NAMESPACE
        or value.get("namespace_uid") != controller["namespace_uid"]
        or value.get("source_plan_sha256") != controller["source_plan_sha256"]
        or value.get("source_request_sha256") != controller["source_request_sha256"]
        or value.get("submission_binding_sha256") != controller["submission_binding_sha256"]
        or value.get("runtime_bundle_sha256") != controller["runtime_bundle_sha256"]
        or value.get("source_plan_sha256") != "sha256:" + digest(plan)
        or value.get("source_request_sha256") != submission.get("source_request_sha256")
        or value.get("submission_binding_sha256") != submission.get("sha256")
        or value.get("run_name") != result["run_name"]
        or value.get("reload_result_path") != str(result_path)
        or value.get("reload_result_file_sha256") != result_file_sha256
        or value.get("reload_result_sha256") != result["sha256"].removeprefix("sha256:")
        or value.get("controller_terminal_path") != str(controller_path)
        or value.get("controller_terminal_file_sha256") != controller_file_sha256
        or value.get("controller_terminal_sha256") != controller["sha256"].removeprefix("sha256:")
        or any(value.get(key) != controller[key] for key in identities)
        or value.get("pod_uids") != [controller["pods"][0]["uid"]]
        or value.get("api_status") != "SUCCEEDED"
        or value.get("controller_status") != "SUCCEEDED"
        or value.get("rayjob_present") is not False
        or value.get("workload_present") is not False
        or value.get("quota_reservation_present") is not False
        or value.get("raycluster_present") is not False
        or value.get("gpu_pods_present") is not False
        or value.get("active_gpu_pod_uids") != []
        or not _exact_integer(value.get("active_gpus"), 0)
        or _evidence_time(value.get("observed_at")) < not_before
    ):
        raise ValueError("HF reload external release is incomplete or mismatched")


def _validate_reload_evidence(
    *,
    result: dict[str, Any],
    result_path: Path,
    result_file_sha256: str,
    controller: dict[str, Any],
    controller_path: Path,
    controller_file_sha256: str,
    release: dict[str, Any],
    plan: dict[str, Any],
    submission: dict[str, Any],
) -> dict[str, Any]:
    completed_at = _validate_reload_result(result)
    terminal_at = _validate_controller(
        controller, result, result_path, result_file_sha256, completed_at, plan, submission
    )
    _validate_release(
        release,
        result,
        result_path,
        result_file_sha256,
        controller,
        controller_path,
        controller_file_sha256,
        terminal_at,
        plan,
        submission,
    )
    export_path = Path(result["export_path"])
    if (
        not export_path.is_absolute()
        or not result_path.is_absolute()
        or result_path.resolve().is_relative_to(export_path.parent.resolve())
    ):
        raise ValueError("HF reload evidence/export paths are not disjoint absolute paths")
    prepared_export = plan.get("source", {}).get("export")
    if not isinstance(prepared_export, dict):
        raise ValueError("HF reload plan has no exact accepted export binding")
    exported, _ = inspect_export(
        export_path,
        result["export_file_sha256"],
        prepared_export=prepared_export,
    )
    if (
        exported["sha256"].removeprefix("sha256:") != result["export_receipt_sha256"]
        or exported["tensor_inventory_sha256"] != result["export_tensor_inventory_sha256"]
        or _validate_prediction_probe(exported["source"].get("prediction_probe"))
        != result["native_prediction_probe"]
        or result["hf_prediction_probe"] != result["native_prediction_probe"]
    ):
        raise ValueError("HF reload result does not bind the selected export")
    return exported


def accept_gpu_reload(
    *,
    plan_path: Path,
    submission_path: Path,
    result_path: Path,
    controller_path: Path,
    release_path: Path,
    output: Path,
) -> dict[str, Any]:
    """Join process, controller, and release evidence into one reload gate."""
    if output.exists() or output.is_symlink():
        raise FileExistsError("HF reload acceptance destination already exists")
    from .miles_hf_export_job import validate_plan, validate_submission_binding

    plan, plan_file_sha256 = _snapshot(plan_path)
    submission, submission_file_sha256 = _snapshot(submission_path)
    validate_plan(plan, check_files=True)
    validate_submission_binding(submission, plan, check_files=True)
    if plan.get("stage") != "reload":
        raise ValueError("HF reload acceptance requires the exact prepared reload plan")
    expected = {
        controller_path: result_path.parent / "HF_RELOAD_CONTROLLER_TERMINAL.json",
        release_path: result_path.parent / "HF_RELOAD_RELEASE.json",
        output: result_path.parent / "HF_RELOAD_ACCEPTED.json",
    }
    if result_path.name != "HF_RELOAD_VALIDATED.json" or any(
        actual != wanted for actual, wanted in expected.items()
    ):
        raise ValueError("HF reload acceptance evidence is outside its bound root")
    result, result_file_sha256 = _snapshot(result_path)
    controller, controller_file_sha256 = _snapshot(controller_path)
    release, release_file_sha256 = _snapshot(release_path)
    exported = _validate_reload_evidence(
        result=result,
        result_path=result_path,
        result_file_sha256=result_file_sha256,
        controller=controller,
        controller_path=controller_path,
        controller_file_sha256=controller_file_sha256,
        release=release,
        plan=plan,
        submission=submission,
    )
    work = {key: result[key] for key in _RELOAD_WORK_FIELDS}
    return _write(
        output,
        {
            "schema": RELOAD_ACCEPTED_SCHEMA,
            "status": "accepted",
            "export_path": result["export_path"],
            "export_file_sha256": result["export_file_sha256"],
            "export_receipt_sha256": result["export_receipt_sha256"],
            "export_tensor_inventory_sha256": exported["tensor_inventory_sha256"],
            "reload_plan_path": str(plan_path),
            "reload_plan_file_sha256": plan_file_sha256,
            "reload_plan_sha256": digest(plan),
            "submission_binding_path": str(submission_path),
            "submission_binding_file_sha256": submission_file_sha256,
            "submission_binding_sha256": submission["sha256"],
            "reload_result_path": str(result_path),
            "reload_result_file_sha256": result_file_sha256,
            "reload_result": result,
            "controller_terminal_path": str(controller_path),
            "controller_terminal_file_sha256": controller_file_sha256,
            "controller_terminal": controller,
            "external_release_path": str(release_path),
            "external_release_file_sha256": release_file_sha256,
            "external_release": release,
            "work_executed": work,
            "exact_hf_reload_verified": True,
            "source_export_unchanged_after_release": True,
            "semantic_source_equivalence_verified": True,
            "external_gpu_release_verified": True,
            "post_export_promotion_requires_this_receipt": True,
            "serving_qualified": False,
        },
    )


def validate_reload_accepted(value: dict[str, Any], *, check_files: bool = True) -> dict[str, str]:
    """Reopen every accepted HF reload input before a downstream consumer."""
    if not check_files:
        raise ValueError("HF reload acceptance requires reopening every evidence file")
    _sealed(value, RELOAD_ACCEPTED_SCHEMA)
    work = value.get("work_executed")
    if (
        set(value) != _ACCEPTED_FIELDS
        or value.get("status") != "accepted"
        or not isinstance(work, dict)
        or set(work) != _RELOAD_WORK_FIELDS
        or any(
            not _exact_integer(work.get(key), expected)
            for key, expected in _RELOAD_WORK_EXPECTED.items()
        )
        or value.get("exact_hf_reload_verified") is not True
        or value.get("source_export_unchanged_after_release") is not True
        or value.get("semantic_source_equivalence_verified") is not True
        or value.get("external_gpu_release_verified") is not True
        or value.get("post_export_promotion_requires_this_receipt") is not True
        or value.get("serving_qualified") is not False
    ):
        raise ValueError("accepted HF reload receipt is incomplete")
    paths = {
        "reload_result": Path(value["reload_result_path"]),
        "controller_terminal": Path(value["controller_terminal_path"]),
        "external_release": Path(value["external_release_path"]),
    }
    plan, plan_file_sha256 = _snapshot(Path(value["reload_plan_path"]))
    submission, submission_file_sha256 = _snapshot(Path(value["submission_binding_path"]))
    from .miles_hf_export_job import validate_plan, validate_submission_binding

    validate_plan(plan, check_files=True)
    validate_submission_binding(submission, plan, check_files=True)
    if (
        plan.get("stage") != "reload"
        or plan_file_sha256 != value["reload_plan_file_sha256"]
        or digest(plan) != value["reload_plan_sha256"]
        or submission_file_sha256 != value["submission_binding_file_sha256"]
        or submission["sha256"] != value["submission_binding_sha256"]
    ):
        raise ValueError("accepted HF reload plan/submission binding changed")
    reopened = {}
    for key, path in paths.items():
        observed, file_sha256 = _snapshot(path)
        if observed != value[key] or file_sha256 != value[f"{key}_file_sha256"]:
            raise ValueError("accepted HF reload evidence changed")
        reopened[key] = observed
    exported = _validate_reload_evidence(
        result=reopened["reload_result"],
        result_path=paths["reload_result"],
        result_file_sha256=value["reload_result_file_sha256"],
        controller=reopened["controller_terminal"],
        controller_path=paths["controller_terminal"],
        controller_file_sha256=value["controller_terminal_file_sha256"],
        release=reopened["external_release"],
        plan=plan,
        submission=submission,
    )
    if (
        value["export_path"] != reopened["reload_result"]["export_path"]
        or value["export_file_sha256"] != reopened["reload_result"]["export_file_sha256"]
        or value["export_receipt_sha256"] != reopened["reload_result"]["export_receipt_sha256"]
        or value["export_tensor_inventory_sha256"] != exported["tensor_inventory_sha256"]
    ):
        raise ValueError("accepted HF reload export binding changed")
    return {
        "export_receipt_sha256": value["export_receipt_sha256"],
        "prediction_sha256": reopened["reload_result"]["hf_prediction_probe"]["prediction_sha256"],
        "reload_result_sha256": reopened["reload_result"]["sha256"],
        "controller_terminal_sha256": reopened["controller_terminal"]["sha256"],
        "external_release_sha256": reopened["external_release"]["sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    export_parser = commands.add_parser("export")
    for name in ("active-canary-binding", "checkpoint", "terminal", "native-reload"):
        export_parser.add_argument(f"--{name}", type=Path, required=True)
        export_parser.add_argument(f"--{name}-sha256", required=True)
    export_parser.add_argument("--output", type=Path, required=True)
    reload_parser = commands.add_parser("reload")
    reload_parser.add_argument("--export", type=Path, required=True)
    reload_parser.add_argument("--export-sha256", required=True)
    reload_parser.add_argument("--output", type=Path, required=True)
    reload_parser.add_argument("--run-name", required=True)
    accept_parser = commands.add_parser("accept-reload")
    accept_parser.add_argument("--plan", type=Path, required=True)
    accept_parser.add_argument("--submission", type=Path, required=True)
    accept_parser.add_argument("--result", type=Path, required=True)
    accept_parser.add_argument("--controller", type=Path, required=True)
    accept_parser.add_argument("--release", type=Path, required=True)
    accept_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "export":
            result = export(
                active_canary_binding_path=args.active_canary_binding,
                active_canary_binding_sha256=args.active_canary_binding_sha256,
                checkpoint_path=args.checkpoint,
                checkpoint_sha256=args.checkpoint_sha256,
                terminal_path=args.terminal,
                terminal_sha256=args.terminal_sha256,
                native_reload_path=args.native_reload,
                native_reload_sha256=args.native_reload_sha256,
                output=args.output,
            )
        elif args.command == "reload":
            result = gpu_reload(
                args.export, args.export_sha256, args.output, run_name=args.run_name
            )
        else:
            result = accept_gpu_reload(
                plan_path=args.plan,
                submission_path=args.submission,
                result_path=args.result,
                controller_path=args.controller,
                release_path=args.release,
                output=args.output,
            )
        print(
            json.dumps(
                {
                    "schema": result["schema"],
                    "status": result["status"],
                    "sha256": result["sha256"],
                }
            )
        )
    except BaseException as exc:
        print(json.dumps({"status": "failed", "error_class": type(exc).__name__}))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
