"""Fail-closed dev-to-production gate for the exact Qwen3.8 Miles RL arm."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from cyber_post_train.jobs import API_URLS, JobsError, digest

SCHEMA = "cyber_qwen38_miles_production_promotion_v3"
ACTIVE_CANARY_SCHEMA = "cyber_qwen38_miles_active_canary_binding_v1"
PROD_NAME = "chris-q38-miles-rl-prod1"
PROD_OUTPUT = "/mnt/sfs/jobs/chris-q38-miles-rl-prod1"
PROD_DATA_ROOT = "/mnt/sfs/jobs/chris-q38-miles-rl-prod1-inputs/data"
PROD_DATA_MANIFEST = PROD_DATA_ROOT + "/manifest.json"
PROD_WANDB = {"entity": "thefleet", "project": "cyber-post-train", "run_id": PROD_NAME}
PROD_REWARD_CANARY_MODE = "reward_canary_v1"
PROD_REWARD_CANARY_OUTPUT = "/mnt/sfs/jobs/chris-q38-miles-rl-prod1-canary-v1"
PROD_REWARD_CANARY_WANDB = {
    "entity": "thefleet",
    "project": "cyber-post-train",
    "run_id": "chris-q38-miles-rl-prod1-canary-v1",
}
PROD_REWARD_CANARY_RESOURCES = {
    "cpu_request": "32",
    "cpu_limit": "32",
    "memory_request": "1800Gi",
    "memory_limit": "2400Gi",
}
PROD_REWARD_CANARY_V2_MODE = "reward_canary_v2"
PROD_REWARD_CANARY_V2_NAME = "chris-q38-miles-prod2-canary"
PROD_REWARD_CANARY_V2_OUTPUT = "/mnt/sfs/jobs/chris-q38-miles-prod2-canary-v1"
PROD_REWARD_CANARY_V2_DATA_ROOT = "/mnt/sfs/jobs/chris-q38-miles-prod2-inputs-v1/data"
PROD_REWARD_CANARY_V2_DATA_MANIFEST = PROD_REWARD_CANARY_V2_DATA_ROOT + "/manifest.json"
PROD_REWARD_CANARY_V2_WANDB = {
    "entity": "thefleet",
    "project": "cyber-post-train",
    "run_id": PROD_REWARD_CANARY_V2_NAME,
}
PROD_REWARD_CANARY_V3_MODE = "reward_canary_v3"
PROD_REWARD_CANARY_V3_NAME = "chris-q38-miles-prod3-canary"
PROD_REWARD_CANARY_V3_OUTPUT = "/mnt/sfs/jobs/chris-q38-miles-prod3-canary-v1"
PROD_REWARD_CANARY_V3_DATA_ROOT = "/mnt/sfs/jobs/chris-q38-miles-prod3-inputs-v1/data"
PROD_REWARD_CANARY_V3_DATA_MANIFEST = PROD_REWARD_CANARY_V3_DATA_ROOT + "/manifest.json"
PROD_REWARD_CANARY_V3_WANDB = {
    "entity": "thefleet",
    "project": "cyber-post-train",
    "run_id": PROD_REWARD_CANARY_V3_NAME,
}
PROD_REWARD_CANARY_LONG_V1_MODE = "reward_canary_opencode_long_v1"
PROD_REWARD_CANARY_LONG_V1_NAME = "chris-q38-miles-lc-canary1"
PROD_REWARD_CANARY_LONG_V1_OUTPUT = "/mnt/sfs/jobs/chris-q38-miles-lc-canary1"
PROD_REWARD_CANARY_LONG_V1_DATA_ROOT = (
    "/mnt/sfs/jobs/chris-q38-miles-lc-canary1-inputs/data"
)
PROD_REWARD_CANARY_LONG_V1_DATA_MANIFEST = (
    PROD_REWARD_CANARY_LONG_V1_DATA_ROOT + "/manifest.json"
)
PROD_REWARD_CANARY_LONG_V1_WANDB = {
    "entity": "thefleet",
    "project": "cyber-post-train",
    "run_id": PROD_REWARD_CANARY_LONG_V1_NAME,
}
PROD_REWARD_CANARY_LONG_V1_RESOURCES = {
    "cpu_request": "64",
    "cpu_limit": "128",
    "memory_request": "1536Gi",
    "memory_limit": "2048Gi",
}
PROD_REWARD_CANARY_LONG_V1_IMAGE = (
    "661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/miles-trainer@sha256:"
    "22c6c624ae95e1886aeab8c9921642b7753232856eb41fb618a70c52975b0a18"
)
PROD_REWARD_CANARY_LONG_V1_RUNTIME = {
    "image": PROD_REWARD_CANARY_LONG_V1_IMAGE,
    "receipt": "qwen38-miles-opencode-long-context-runtime-v1.json",
    "sha256": "sha256:a7599e67fe88e987fcfcc380a27dd2c568d329d9fcc97858d4b01fc4299bae6c",
}
PROD_REWARD_CANARY_LONG_V1_CHECKPOINT = {
    "manifest": "/mnt/sfs/jobs/chris-q38-miles-lc-base1/NATIVE_CHECKPOINT.json",
    "sha256": "sha256:7caa9fcb93a50d93f52e62d68f14e313add3e6b510d1abb42ca747dafd0923a0",
}
PROD_REWARD_CANARY_LONG_V1_CHECKPOINT_ROOT = (
    "/mnt/sfs/jobs/chris-q38-miles-lc-base1/torch-dist"
)
PROD_REWARD_CANARY_LONG_V1_CHECKPOINT_RECEIPT_SHA256 = (
    "7337a9bd828023f685c3c15870de2ce6874e156dcb7b7f92d6bd99e3722641e4"
)
PROD_REWARD_CANARY_LONG_V2_MODE = "reward_canary_opencode_long_v2"
PROD_REWARD_CANARY_LONG_V2_NAME = "chris-q38-miles-lc-canary2"
PROD_REWARD_CANARY_LONG_V2_OUTPUT = "/mnt/sfs/jobs/chris-q38-miles-lc-canary2"
PROD_REWARD_CANARY_LONG_V2_DATA_ROOT = (
    "/mnt/sfs/jobs/chris-q38-miles-lc-canary2-inputs/data"
)
PROD_REWARD_CANARY_LONG_V2_DATA_MANIFEST = (
    PROD_REWARD_CANARY_LONG_V2_DATA_ROOT + "/manifest.json"
)
PROD_REWARD_CANARY_LONG_V2_WANDB = {
    "entity": "thefleet",
    "project": "cyber-post-train",
    "run_id": PROD_REWARD_CANARY_LONG_V2_NAME,
}
PROD_REWARD_CANARY_LONG_V2_IMAGE = (
    "661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/miles-trainer@sha256:"
    "dc1a41ac386c9f7377e7a6f7b92a402830e308855aa2632413af25917f38dd93"
)
PROD_REWARD_CANARY_LONG_V2_RUNTIME = {
    "image": PROD_REWARD_CANARY_LONG_V2_IMAGE,
    "receipt": "qwen38-miles-opencode-long-context-runtime-v1.json",
    "sha256": "sha256:a2edd96163b9ebefd46f27f71d3787fb4da447ccaaa0e2b79d415895533676bc",
}
PROD_REWARD_CANARY_LONG_V2_CHECKPOINT = {
    "manifest": "/mnt/sfs/jobs/chris-q38-miles-lc-base1-prod1/NATIVE_CHECKPOINT.json",
    "sha256": "sha256:10cc7a78e6e74140858cd307feab9aa846d389cbfd91e35ed12760122c46ace2",
}
PROD_REWARD_CANARY_LONG_V2_CHECKPOINT_ROOT = (
    "/mnt/sfs/jobs/chris-q38-miles-lc-base1-prod1/torch-dist"
)
PROD_REWARD_CANARY_LONG_V2_CHECKPOINT_RECEIPT_SHA256 = (
    "5885f89feb3d3e646221c61e18713dce29b4117827306336895f5bf95073d4d0"
)
EXPERIMENT_OWNER_PREFIX = "chris-"
FLEET_RUN_NAME_LABEL = "fleet.ai/run-name"
PROD_MODEL_SHA256 = "dcfdcd6ecb6661741cd3a4b24dc5af7259642c8a6824773e0de70d55d7501179"
BASE_CHECKPOINT = {
    "root": "/mnt/sfs/jobs/chris-cpt-cleanup-q38-miles-base-v1/torch-dist",
    "receipt_sha256": "19c8e93482530170e0f648815ab74233719e6f2b3bb7879a6564b42c3abec371",
}
PROD_KUBE_CONTEXT = "nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6"
PROD_NAMESPACE_UID = "fd6d2fcd-687a-4257-9dba-a034bb381e6b"
DEV_API_BASE_URL = API_URLS["dev"]
EXPECTED_CANDIDATE_SHA256 = "7ab7d6db7141c1236884cc873341b3c149d5caa4cec45a9ceb14fe40436f836b"
EXPECTED_DEV_CANARY = {
    "source_run_name": "chris-q38-miles-rlreward-dev5",
    "source_commit": "0e6970c7f16f8199b2fa583cb19937aeecdfcfd9",
    "source_plan_sha256": "sha256:4ccc8b10e473176993b3867e4bbe3b3f1e8717cdeab77615b7493fa4513866da",
    "source_request_sha256": (
        "sha256:e56a3c25c1789526747c74353cdb4bd7963e9084ae9bf6b180be79b53a691cfb"
    ),
    "api_base_url": DEV_API_BASE_URL,
}
EXPECTED_RESOURCES = {
    "cpu_request": "64",
    "cpu_limit": "128",
    "memory_request": "1536Gi",
    "memory_limit": "2048Gi",
}
# Transitional compatibility for the separately-qualified HF export path.  The
# production-promotion gate below deliberately does not consult this retired
# identity: it consumes an explicit, digest-valid active-canary binding.
DEV3 = {
    "source_commit": "d69fd01e4b435adc5492c8aedba9cf7f3af5e40e",
    "source_plan_sha256": "245b404f9507ac2603c53969dd4506aee811e1c02ebf468543994cff76e3e95e",
    "source_request_sha256": "d2f2514a33bdf05610c5765c11efaecfcd6eea39e4297115cc39ce3c86d3f927",
    "runtime_bundle_sha256": "0e3ff1646344abb1b4be13ea064143014e94c13bc22c6c7e8c1a552f8bbe32a6",
    "api_base_url": "https://api.ft.dev.flt.build",
    "api_run_id": "6be68393-f032-40db-b640-d2d23473e85d",
    "api_run_name": "chris-q38-miles-rlreward-dev3-6be68393",
    "rayjob_uid": "803770c7-d830-4f47-8f1a-22d7e75362c3",
    "workload_uid": "a20b093e-93cd-4c26-b81c-33586d3bf6d3",
}
EXPECTED_DATA = {
    "name": PROD_NAME,
    "selection_sha256": "sha256:50d6052187ff2f5f085b9de402982d4421fba0c73bb1bd54fc4bc850516c5d68",
    "split_sha256": "sha256:c8c0083e08df55179a5acbd10602dbce5178257484b471c5ba7f4cbb04bdb35c",
    "tool_catalog_sha256": (
        "sha256:85fad6bdc3a835bf52a11a99b3387740eb06eb3d1720ad9bb33f3feac215b44a"
    ),
    "tokenizer": {
        "repo": "Qwen/Qwen3.8-27B",
        "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "backend_sha256": "ffb7a28b27dabcc333662fd3e0b0005d9e79a1c22e31453ab5a3017fbd5f25c0",
        "chat_template_sha256": "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
        "files": [
            {
                "path": "tokenizer.json",
                "sha256": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
            },
            {
                "path": "tokenizer_config.json",
                "sha256": "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27",
            },
            {
                "path": "chat_template.jinja",
                "sha256": "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
            },
            {
                "path": "merges.txt",
                "sha256": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
            },
            {
                "path": "vocab.json",
                "sha256": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
            },
        ],
    },
    "limits": {
        "context_tokens": 98304,
        "response_tokens": 81920,
        "max_tokens_per_turn": 4096,
        "max_turns": 600,
        "episode_seconds": 2400,
        "tool_seconds": 330,
        "tool_result_chars": 50000,
    },
    "rows": {"train": 59, "dev": 20},
}
EXPECTED_REWARD_CANARY_V2_DATA = {
    "name": PROD_REWARD_CANARY_V2_NAME,
    "selection_sha256": "sha256:5d562cd8c98f3a870006e4e2fb5d52e9f35e9ffd9a23a4fbf3fd1a9aa5a6e308",
    "split_sha256": "sha256:f0135e613a00c1727c63ecf0623122fdbc3bf3c8682aee5175c556e05a38d044",
    "tool_catalog_sha256": EXPECTED_DATA["tool_catalog_sha256"],
    "tokenizer": EXPECTED_DATA["tokenizer"],
    "limits": {
        "context_tokens": 98304,
        "response_tokens": 81920,
        "max_tokens_per_turn": 8192,
        "max_turns": 600,
        "episode_seconds": 2400,
        "tool_seconds": 330,
        "tool_result_chars": 4000,
    },
    "rows": {"train": 1, "dev": 1},
}
EXPECTED_REWARD_CANARY_V3_DATA = {
    **EXPECTED_REWARD_CANARY_V2_DATA,
    "name": PROD_REWARD_CANARY_V3_NAME,
    "limits": {
        **EXPECTED_REWARD_CANARY_V2_DATA["limits"],
        "max_tokens_per_turn": 32768,
    },
}
EXPECTED_REWARD_CANARY_LONG_V1_DATA = {
    "name": PROD_REWARD_CANARY_LONG_V1_NAME,
    "selection_sha256": EXPECTED_REWARD_CANARY_V2_DATA["selection_sha256"],
    "split_sha256": EXPECTED_REWARD_CANARY_V2_DATA["split_sha256"],
    "tool_catalog_sha256": EXPECTED_DATA["tool_catalog_sha256"],
    "tokenizer": EXPECTED_DATA["tokenizer"],
    "limits": {
        "context_tokens": 262144,
        "response_tokens": 245760,
        "max_tokens_per_turn": 32768,
        "max_turns": 2048,
        "episode_seconds": 28800,
        "tool_seconds": 330,
    },
    "rows": {"train": 1, "dev": 1},
    "files": {
        "train": {
            "path": "train.jsonl",
            "rows": 1,
            "max_prompt_tokens": 1261,
            "sha256": (
                "sha256:a24ec7460100723d13e86f6d57075cdb65ba8be18a6ff015d5c52b0b230dd362"
            ),
        },
        "dev": {
            "path": "dev.jsonl",
            "rows": 1,
            "max_prompt_tokens": 1276,
            "sha256": (
                "sha256:51680c07d5d977630492436946cddd86199178df80e5489ded229411270eb51e"
            ),
        },
    },
}
EXPECTED_REWARD_CANARY_LONG_V2_DATA = {
    **EXPECTED_REWARD_CANARY_LONG_V1_DATA,
    "name": PROD_REWARD_CANARY_LONG_V2_NAME,
    "files": {
        "train": {
            "path": "train.jsonl",
            "rows": 1,
            "max_prompt_tokens": 1261,
            "sha256": (
                "sha256:1175fee4acc49cdd9c83b32dc68052dd89d65d65600acf54f8b33c1c44a72ea0"
            ),
        },
        "dev": {
            "path": "dev.jsonl",
            "rows": 1,
            "max_prompt_tokens": 1276,
            "sha256": (
                "sha256:ebd9b7e22fd912cbf2c7151bcb155b7dab6ee19e406ecf493131015e9cb66b10"
            ),
        },
    },
    "manifest_file_sha256": (
        "sha256:7c726746ca1a3ec45984a24554d4efbfdc13da6fa115a13a10dfa7fecaf84975"
    ),
    "manifest_self_sha256": (
        "sha256:4ec8a1ae470c4d86728adf929cf496556f7b15bd3acdfc7bc56d57682bbd8910"
    ),
    "derivation_sha256": (
        "sha256:c7a7405f72b43a5dfd239eca3f8720fcd4117db2d7d1f53c31ed123c7d5fe9e6"
    ),
}
BENCHMARK_ISOLATION = {
    "optimizer_split": "train",
    "fleet_dev_is_evaluation_only": True,
    "final_test_rows": 0,
    "external_benchmark_training_rows": 0,
    "external_benchmark_reward_inputs": 0,
    "external_benchmark_hpo_or_checkpoint_inputs": 0,
    "external_benchmark_retry_inputs": 0,
}
LIVE_REQUIREMENTS = [
    "prod_jobs_api_name_and_output_absent",
    "prod_output_and_submission_journal_absent",
    "prod_wandb_run_id_absent",
    "active_prod_experiment_nodes_plus_candidate_at_most_eight",
    "warning_free_one_by_eight_c1_no_requeue_preview",
]
_REF_FIELDS = {"path", "file_sha256", "receipt_sha256"}
_ACTIVE_CANARY_FIELDS = {
    "schema",
    "status",
    "reward_terminal_receipt_sha256",
    "source_run_name",
    "source_commit",
    "source_plan_sha256",
    "source_request_sha256",
    "runtime_bundle_sha256",
    "api_base_url",
    "api_run_id",
    "api_run_name",
    "rayjob_uid",
    "workload_uid",
    "sha256",
}
_FIELDS = {
    "schema",
    "status",
    "candidate_run_sha256",
    "active_canary_binding",
    "reward_terminal",
    "native_reload",
    "production_data_manifest",
    "benchmark_isolation",
    "live_requirements",
    "sha256",
}
_HEX_SHA = re.compile(r"sha256:[a-f0-9]{64}")
_GIT_SHA = re.compile(r"[a-f0-9]{40}")
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}")
_RUN_NAME = re.compile(r"[a-z0-9](?:[-a-z0-9]*[a-z0-9])?")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sealed(value: Any, schema: str | None = None) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or (schema is not None and value.get("schema") != schema)
        or value.get("sha256", "").removeprefix("sha256:")
        != digest({key: item for key, item in value.items() if key != "sha256"})
    ):
        raise ValueError("Miles production evidence is not digest-valid")
    return value


def _snapshot(path: Path, expected_file_sha256: str) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("Miles production evidence path is not an exact regular file")
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, key) != getattr(after, key) for key in stable):
        raise ValueError("Miles production evidence changed while reading")
    if hashlib.sha256(payload).hexdigest() != expected_file_sha256.removeprefix("sha256:"):
        raise ValueError("Miles production evidence file digest changed")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Miles production evidence is not JSON") from error
    return _sealed(value)


def _reference(path: Path) -> tuple[dict[str, str], dict[str, Any]]:
    value = _snapshot(path, _sha256(path))
    return (
        {
            "path": str(path),
            "file_sha256": _sha256(path),
            "receipt_sha256": value["sha256"].removeprefix("sha256:"),
        },
        value,
    )


def _reopen(reference: Any, *, check_files: bool) -> dict[str, Any] | None:
    if (
        not isinstance(reference, dict)
        or set(reference) != _REF_FIELDS
        or not all(
            re.fullmatch(r"(?:sha256:)?[a-f0-9]{64}", str(reference.get(key, "")))
            for key in ("file_sha256", "receipt_sha256")
        )
    ):
        raise ValueError("Miles production evidence reference changed")
    path = reference.get("path")
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise ValueError("Miles production evidence path is not absolute")
    if not check_files:
        return None
    value = _snapshot(Path(path), reference["file_sha256"])
    if value["sha256"].removeprefix("sha256:") != reference["receipt_sha256"].removeprefix(
        "sha256:"
    ):
        raise ValueError("Miles production evidence self-digest changed")
    return value


def _exact_data(value: Any, expected: dict[str, Any] = EXPECTED_DATA) -> None:
    _sealed(value, "cyber_miles_data_v1")
    files = value.get("files", {})
    if (
        value.get("name") != expected["name"]
        or value.get("selection_sha256") != expected["selection_sha256"]
        or value.get("split_sha256") != expected["split_sha256"]
        or value.get("tool_catalog_sha256") != expected["tool_catalog_sha256"]
        or value.get("tokenizer") != expected["tokenizer"]
        or value.get("template_sha256")
        != "sha256:38d42166599348d47ded69776c5389c89924045e6827089923a031379f8a3dfe"
        or value.get("limits") != expected["limits"]
        or value.get("gpus") != 0
        or value.get("environment_creates") != 0
        or set(files) != {"train", "dev"}
        or any(
            not isinstance(files[split], dict)
            or files[split].get("path") != split + ".jsonl"
            or files[split].get("rows") != rows
            for split, rows in expected["rows"].items()
        )
    ):
        raise ValueError("Miles production data differs from the exact 59/20 split")


def _exact_long_data(
    value: Any, expected: dict[str, Any] = EXPECTED_REWARD_CANARY_LONG_V1_DATA
) -> None:
    from .miles_opencode import harness_contract

    _sealed(value, "cyber_miles_data_v2")
    if (
        value.get("name") != expected["name"]
        or value.get("selection_sha256") != expected["selection_sha256"]
        or value.get("split_sha256") != expected["split_sha256"]
        or value.get("tool_catalog_sha256") != expected["tool_catalog_sha256"]
        or value.get("tokenizer") != expected["tokenizer"]
        or value.get("template_sha256")
        != "sha256:38d42166599348d47ded69776c5389c89924045e6827089923a031379f8a3dfe"
        or value.get("limits") != expected["limits"]
        or value.get("harness") != harness_contract()
        or value.get("files") != expected["files"]
        or value.get("gpus") != 0
        or value.get("environment_creates") != 0
        or (
            expected.get("derivation_sha256") is None
            and "derivation" in value
        )
        or (
            expected.get("derivation_sha256") is not None
            and (
                not isinstance(value.get("derivation"), dict)
                or value["derivation"].get("sha256")
                != expected["derivation_sha256"]
            )
        )
    ):
        raise ValueError("Miles long-context canary data differs from the exact Fleet split")


def _candidate_without_promotion(config: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in config.items() if key != "production_promotion"}


def _is_reward_canary_marker(value: Any) -> bool:
    return value in (
        {"mode": PROD_REWARD_CANARY_MODE},
        {"mode": PROD_REWARD_CANARY_V2_MODE},
        {"mode": PROD_REWARD_CANARY_V3_MODE},
        {"mode": PROD_REWARD_CANARY_LONG_V1_MODE},
        {"mode": PROD_REWARD_CANARY_LONG_V2_MODE},
    )


def _exact_reward_canary_config(config: dict[str, Any]) -> None:
    """Admit one bounded production reward/update canary, never a full run."""
    legacy_recipe = {
        "nodes": 1,
        "gpus_per_node": 8,
        "steps": 1,
        "groups": 1,
        "samples_per_prompt": 8,
        "lr": 2e-6,
        "temperature": 0.7,
        "kl_loss_coef": 0.001,
        "max_tokens_per_gpu": 8192,
        "eval_interval": 1,
        "checkpoint_interval": 1,
        "seed": 42,
    }
    mode = config.get("production_promotion", {}).get("mode")
    if mode == PROD_REWARD_CANARY_MODE:
        expected_name = PROD_NAME
        expected_output = PROD_REWARD_CANARY_OUTPUT
        expected_data_root = PROD_DATA_ROOT
        expected_data_manifest = PROD_DATA_MANIFEST
        expected_wandb = PROD_REWARD_CANARY_WANDB
        expected_recipe = legacy_recipe
        expected_resources = PROD_REWARD_CANARY_RESOURCES
        expected_checkpoint = {
            "manifest": (
                "/mnt/sfs/jobs/chris-cpt-cleanup-q38-miles-base-v1/NATIVE_CHECKPOINT.json"
            ),
            "sha256": (
                "sha256:b3d772de9121f442ea7b9a4c9a996f2a0a99cab8c49fe3083c148fe3eebd089c"
            ),
        }
        expected_runtime = None
    elif mode == PROD_REWARD_CANARY_V2_MODE:
        expected_name = PROD_REWARD_CANARY_V2_NAME
        expected_output = PROD_REWARD_CANARY_V2_OUTPUT
        expected_data_root = PROD_REWARD_CANARY_V2_DATA_ROOT
        expected_data_manifest = PROD_REWARD_CANARY_V2_DATA_MANIFEST
        expected_wandb = PROD_REWARD_CANARY_V2_WANDB
        expected_recipe = legacy_recipe
        expected_resources = PROD_REWARD_CANARY_RESOURCES
        expected_checkpoint = {
            "manifest": (
                "/mnt/sfs/jobs/chris-cpt-cleanup-q38-miles-base-v1/NATIVE_CHECKPOINT.json"
            ),
            "sha256": (
                "sha256:b3d772de9121f442ea7b9a4c9a996f2a0a99cab8c49fe3083c148fe3eebd089c"
            ),
        }
        expected_runtime = None
    elif mode == PROD_REWARD_CANARY_V3_MODE:
        expected_name = PROD_REWARD_CANARY_V3_NAME
        expected_output = PROD_REWARD_CANARY_V3_OUTPUT
        expected_data_root = PROD_REWARD_CANARY_V3_DATA_ROOT
        expected_data_manifest = PROD_REWARD_CANARY_V3_DATA_MANIFEST
        expected_wandb = PROD_REWARD_CANARY_V3_WANDB
        expected_recipe = legacy_recipe
        expected_resources = PROD_REWARD_CANARY_RESOURCES
        expected_checkpoint = {
            "manifest": (
                "/mnt/sfs/jobs/chris-cpt-cleanup-q38-miles-base-v1/NATIVE_CHECKPOINT.json"
            ),
            "sha256": (
                "sha256:b3d772de9121f442ea7b9a4c9a996f2a0a99cab8c49fe3083c148fe3eebd089c"
            ),
        }
        expected_runtime = None
    elif mode in {
        PROD_REWARD_CANARY_LONG_V1_MODE,
        PROD_REWARD_CANARY_LONG_V2_MODE,
    }:
        successor = mode == PROD_REWARD_CANARY_LONG_V2_MODE
        expected_name = (
            PROD_REWARD_CANARY_LONG_V2_NAME
            if successor
            else PROD_REWARD_CANARY_LONG_V1_NAME
        )
        expected_output = (
            PROD_REWARD_CANARY_LONG_V2_OUTPUT
            if successor
            else PROD_REWARD_CANARY_LONG_V1_OUTPUT
        )
        expected_data_root = (
            PROD_REWARD_CANARY_LONG_V2_DATA_ROOT
            if successor
            else PROD_REWARD_CANARY_LONG_V1_DATA_ROOT
        )
        expected_data_manifest = (
            PROD_REWARD_CANARY_LONG_V2_DATA_MANIFEST
            if successor
            else PROD_REWARD_CANARY_LONG_V1_DATA_MANIFEST
        )
        expected_wandb = (
            PROD_REWARD_CANARY_LONG_V2_WANDB
            if successor
            else PROD_REWARD_CANARY_LONG_V1_WANDB
        )
        expected_recipe = {
            **legacy_recipe,
            "nodes": 4,
            "max_tokens_per_gpu": 65536,
            "native_profile": "qwen3.8-27b-256k",
            "harness": "opencode",
            "session_node_cap": 4096,
        }
        expected_resources = PROD_REWARD_CANARY_LONG_V1_RESOURCES
        expected_checkpoint = (
            PROD_REWARD_CANARY_LONG_V2_CHECKPOINT
            if successor
            else PROD_REWARD_CANARY_LONG_V1_CHECKPOINT
        )
        expected_runtime = (
            PROD_REWARD_CANARY_LONG_V2_RUNTIME
            if successor
            else PROD_REWARD_CANARY_LONG_V1_RUNTIME
        )
    else:
        raise ValueError("Miles production reward canary mode is not supported")
    if (
        config.get("name") != expected_name
        or config.get("output_root") != expected_output
        or config.get("data") != {"manifest": expected_data_manifest, "root": expected_data_root}
        or config.get("checkpoint") != expected_checkpoint
        or config.get("recipe") != expected_recipe
        or config.get("wandb") != expected_wandb
        or config.get("cluster")
        != {
            "target": "prod",
            "priority": "c1",
            "resources": expected_resources,
        }
        or config.get("runtime") != expected_runtime
        or not _is_reward_canary_marker(config.get("production_promotion"))
    ):
        raise ValueError("Miles production reward canary differs from its exact one-update arm")


def requires_production_promotion(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    cluster = value.get("cluster", {})
    execution = value.get("execution", {})
    return (isinstance(cluster, dict) and cluster.get("target") == "prod") or (
        isinstance(execution, dict) and execution.get("cluster_target") == "prod"
    )


def _exact_candidate(config: dict[str, Any]) -> None:
    if digest(_candidate_without_promotion(config)) != EXPECTED_CANDIDATE_SHA256:
        raise ValueError("Miles production candidate differs from the exact reviewed run")


def validate_active_canary_binding(value: Any) -> dict[str, Any]:
    """Validate one immutable identity extracted from an accepted dev canary."""

    _sealed(value, ACTIVE_CANARY_SCHEMA)
    sha_fields = (
        "reward_terminal_receipt_sha256",
        "source_plan_sha256",
        "source_request_sha256",
        "runtime_bundle_sha256",
    )
    uuid_fields = ("api_run_id", "rayjob_uid", "workload_uid")
    if (
        set(value) != _ACTIVE_CANARY_FIELDS
        or value.get("status") != "accepted_dev_canary"
        or value.get("api_base_url") != DEV_API_BASE_URL
        or _GIT_SHA.fullmatch(str(value.get("source_commit", ""))) is None
        or any(_HEX_SHA.fullmatch(str(value.get(key, ""))) is None for key in sha_fields)
        or any(_UUID.fullmatch(str(value.get(key, ""))) is None for key in uuid_fields)
        or _RUN_NAME.fullmatch(str(value.get("source_run_name", ""))) is None
        or value.get("api_run_name")
        != f"{value.get('source_run_name')}-{str(value.get('api_run_id'))[:8]}"
    ):
        raise ValueError("Miles active-canary binding is malformed")
    return value


def _active_canary_identity(terminal: dict[str, Any]) -> dict[str, str]:
    submission = _reopen(terminal.get("submission_binding"), check_files=True)
    controller = _reopen(terminal.get("controller_observation"), check_files=True)
    if not isinstance(submission, dict) or not isinstance(controller, dict):
        raise ValueError("Miles active-canary identity evidence is absent")
    api = submission.get("api", {})
    kube = controller.get("kubernetes", {})
    return {
        "reward_terminal_receipt_sha256": "sha256:"
        + str(terminal.get("sha256", "")).removeprefix("sha256:"),
        "source_run_name": str(terminal.get("source_run_name", "")),
        "source_commit": str(submission.get("source_commit", "")),
        "source_plan_sha256": "sha256:"
        + str(terminal.get("source_plan_sha256", "")).removeprefix("sha256:"),
        "source_request_sha256": "sha256:"
        + str(terminal.get("source_request_sha256", "")).removeprefix("sha256:"),
        "runtime_bundle_sha256": "sha256:"
        + str(submission.get("runtime_bundle_sha256", "")).removeprefix("sha256:"),
        "api_base_url": str(api.get("base_url", "")),
        "api_run_id": str(api.get("run_id", "")),
        "api_run_name": str(api.get("run_name", "")),
        "rayjob_uid": str(kube.get("rayjob", {}).get("uid", "")),
        "workload_uid": str(kube.get("workload", {}).get("uid", "")),
    }


def _exact_dev3(terminal: dict[str, Any]) -> None:
    """Retain the retired identity only for the legacy HF-export qualifier."""

    observed = _active_canary_identity(terminal)
    if (
        observed["source_run_name"] != "chris-q38-miles-rlreward-dev3"
        or observed["source_plan_sha256"].removeprefix("sha256:") != DEV3["source_plan_sha256"]
        or observed["source_request_sha256"].removeprefix("sha256:")
        != DEV3["source_request_sha256"]
        or observed["source_commit"] != DEV3["source_commit"]
        or observed["runtime_bundle_sha256"].removeprefix("sha256:")
        != DEV3["runtime_bundle_sha256"]
        or observed["api_base_url"] != DEV3["api_base_url"]
        or observed["api_run_id"] != DEV3["api_run_id"]
        or observed["api_run_name"] != DEV3["api_run_name"]
        or observed["rayjob_uid"] != DEV3["rayjob_uid"]
        or observed["workload_uid"] != DEV3["workload_uid"]
    ):
        raise ValueError("Miles HF export proof is not the retired dev3 identity")


def _exact_active_canary(terminal: dict[str, Any], binding: dict[str, Any]) -> None:
    validate_active_canary_binding(binding)
    if any(binding.get(key) != expected for key, expected in EXPECTED_DEV_CANARY.items()):
        raise ValueError("Miles production proof is not the exact dev5 canary")
    observed = {
        "schema": ACTIVE_CANARY_SCHEMA,
        "status": "accepted_dev_canary",
        **_active_canary_identity(terminal),
    }
    observed["sha256"] = digest(observed)
    if observed != binding:
        raise ValueError("Miles production proof differs from its active-canary binding")


def accept_active_canary_binding(*, reward_terminal: Path, output: Path) -> dict[str, Any]:
    """Extract one create-once, sanitized canary identity from accepted evidence."""

    if output.exists() or output.is_symlink():
        raise FileExistsError("Miles active-canary binding destination already exists")
    _, terminal = _reference(reward_terminal)
    from . import miles_acceptance

    miles_acceptance.validate_terminal(terminal, check_files=True)
    value = {
        "schema": ACTIVE_CANARY_SCHEMA,
        "status": "accepted_dev_canary",
        **_active_canary_identity(terminal),
    }
    sealed = {**value, "sha256": digest(value)}
    validate_active_canary_binding(sealed)
    if any(sealed.get(key) != expected for key, expected in EXPECTED_DEV_CANARY.items()):
        raise ValueError("Miles active-canary binding is not the exact dev5 canary")
    from .miles_conversion import _write

    return _write(output, value)


def validate_promotion(value: dict[str, Any], *, check_files: bool) -> dict[str, Any]:
    """Reopen the reward, native-reload and production-data chain."""
    _sealed(value, SCHEMA)
    if (
        set(value) != _FIELDS
        or value.get("status") != "qualified_for_fresh_live_checks"
        or value.get("candidate_run_sha256") != EXPECTED_CANDIDATE_SHA256
        or value.get("benchmark_isolation") != BENCHMARK_ISOLATION
        or value.get("live_requirements") != LIVE_REQUIREMENTS
    ):
        raise ValueError("Miles production promotion invariant changed")
    references = {
        name: _reopen(value.get(name), check_files=check_files)
        for name in (
            "active_canary_binding",
            "reward_terminal",
            "native_reload",
            "production_data_manifest",
        )
    }
    if not check_files:
        return {"candidate_run_sha256": EXPECTED_CANDIDATE_SHA256}

    from . import miles_acceptance, miles_reload_acceptance

    binding = references["active_canary_binding"]
    terminal = references["reward_terminal"]
    native_reload = references["native_reload"]
    data = references["production_data_manifest"]
    validate_active_canary_binding(binding)
    miles_acceptance.validate_terminal(terminal, check_files=True)
    miles_reload_acceptance.validate_accepted(native_reload, check_files=True)
    _exact_active_canary(terminal, binding)
    if str(native_reload.get("source_terminal_acceptance_sha256", "")).removeprefix(
        "sha256:"
    ) != str(terminal["sha256"]).removeprefix("sha256:") or str(
        native_reload.get("source_manifest_sha256", "")
    ).removeprefix("sha256:") != terminal["checkpoint_manifest"]["receipt_sha256"].removeprefix(
        "sha256:"
    ):
        raise ValueError("Miles reward and native-reload chain is not cross-bound")
    _exact_data(data)
    if value["production_data_manifest"]["path"] != PROD_DATA_MANIFEST:
        raise ValueError("Miles production data path changed")
    return {
        "candidate_run_sha256": EXPECTED_CANDIDATE_SHA256,
        "dev_source_plan_sha256": binding["source_plan_sha256"].removeprefix("sha256:"),
        "dev_checkpoint": terminal["checkpoint_manifest"],
    }


def accept_promotion(
    *,
    active_canary_binding: Path,
    reward_terminal: Path,
    native_reload: Path,
    production_data_manifest: Path,
    output: Path,
) -> dict[str, Any]:
    """Create one promotion receipt after every immutable dev gate exists."""
    if output.exists() or output.is_symlink():
        raise FileExistsError("Miles production promotion destination already exists")
    refs = {}
    for name, path in {
        "active_canary_binding": active_canary_binding,
        "reward_terminal": reward_terminal,
        "native_reload": native_reload,
        "production_data_manifest": production_data_manifest,
    }.items():
        refs[name], _ = _reference(path)
    value = {
        "schema": SCHEMA,
        "status": "qualified_for_fresh_live_checks",
        "candidate_run_sha256": EXPECTED_CANDIDATE_SHA256,
        **refs,
        "benchmark_isolation": BENCHMARK_ISOLATION,
        "live_requirements": LIVE_REQUIREMENTS,
    }
    validate_promotion({**value, "sha256": digest(value)}, check_files=True)
    from .miles_conversion import _write

    return _write(output, value)


def bind_production_promotion(config: dict[str, Any], relative_to: Path) -> dict[str, Any] | None:
    if not requires_production_promotion(config):
        if config.get("production_promotion") is not None:
            raise ValueError("Miles production promotion cannot attach to a dev run")
        return None
    reference = config.get("production_promotion")
    if _is_reward_canary_marker(reference):
        _exact_reward_canary_config(config)
        return dict(reference)
    _exact_candidate(config)
    if not isinstance(reference, dict) or set(reference) != _REF_FIELDS:
        raise ValueError("exact Miles production promotion receipt is required")
    path = Path(reference["path"])
    if not path.is_absolute():
        path = (relative_to / path).resolve()
    normalized = {**reference, "path": str(path)}
    receipt = _reopen(normalized, check_files=True)
    validate_promotion(receipt, check_files=True)
    return {**normalized, "receipt": receipt}


def _exact_plan(plan: dict[str, Any]) -> None:
    from . import miles

    args = plan.get("arguments", {})
    execution = plan.get("execution", {})
    checkpoint = plan.get("checkpoint", {})
    expected_args = {
        "name": PROD_NAME,
        "output_root": PROD_OUTPUT,
        "model_root": "/mnt/sfs/models/qwen3.8-27b-1d4bf0f2",
        "torch_dist_root": checkpoint.get("root"),
        "train_data": PROD_DATA_ROOT + "/train.jsonl",
        "dev_data": PROD_DATA_ROOT + "/dev.jsonl",
        "data_manifest": PROD_DATA_MANIFEST,
        "wandb_entity": PROD_WANDB["entity"],
        "wandb_project": PROD_WANDB["project"],
        "wandb_run_id": PROD_WANDB["run_id"],
        "model": "Qwen/Qwen3.8-27B",
        "nodes": 1,
        "gpus_per_node": 8,
        "steps": 59,
        "groups": 1,
        "samples_per_prompt": 8,
        "lr": 2e-6,
        "temperature": 0.7,
        "kl_loss_coef": 0.001,
        "max_tokens_per_gpu": 8192,
        "eval_interval": 59,
        "checkpoint_interval": 10,
        "seed": 42,
        "context_tokens": 98304,
        "response_tokens": 81920,
        "tokens_per_turn": 4096,
        "native_profile": "qwen3.8-27b",
        "harness": "direct",
        "runtime_image": miles.IMAGE,
        "session_node_cap": 1024,
    }
    # Historical base-policy plans predate the explicit scientific-policy
    # identity.  Missing/None and the exact runtime root are equivalent only
    # for this base-policy candidate; a distinct root remains a hard failure.
    policy_identity_root = args.get("policy_identity_root")
    comparable_args = {key: value for key, value in args.items() if key != "policy_identity_root"}
    if (
        plan.get("run_name") != PROD_NAME
        or plan.get("output_root") != PROD_OUTPUT
        or comparable_args != expected_args
        or policy_identity_root not in {None, expected_args["model_root"]}
        or checkpoint.get("schema") != "cyber_miles_checkpoint_v1"
        or checkpoint.get("optimizer_steps") != 0
        or checkpoint.get("root") != BASE_CHECKPOINT["root"]
        or checkpoint.get("sha256", "").removeprefix("sha256:") != BASE_CHECKPOINT["receipt_sha256"]
        or checkpoint.get("image") != miles.IMAGE
        or checkpoint.get("model") != plan.get("model")
        or digest(plan.get("model")) != PROD_MODEL_SHA256
        or execution.get("image") != miles.IMAGE
        or execution.get("cluster_target") != "prod"
        or execution.get("priority") != "c1"
        or execution.get("resources") != EXPECTED_RESOURCES
        or set(execution)
        != {"image", "priority", "resources", "cluster_target", "production_promotion"}
    ):
        raise ValueError("compiled Miles production plan differs from the exact candidate")
    _exact_data(plan.get("data"))


def _exact_reward_canary_plan(plan: dict[str, Any]) -> None:
    from . import miles

    args = plan.get("arguments", {})
    execution = plan.get("execution", {})
    checkpoint = plan.get("checkpoint", {})
    mode = execution.get("production_promotion", {}).get("mode")
    if mode == PROD_REWARD_CANARY_MODE:
        name = PROD_NAME
        output = PROD_REWARD_CANARY_OUTPUT
        data_root = PROD_DATA_ROOT
        data_manifest = PROD_DATA_MANIFEST
        wandb = PROD_REWARD_CANARY_WANDB
        tokens_per_turn = 4096
        expected_data = EXPECTED_DATA
    elif mode == PROD_REWARD_CANARY_V2_MODE:
        name = PROD_REWARD_CANARY_V2_NAME
        output = PROD_REWARD_CANARY_V2_OUTPUT
        data_root = PROD_REWARD_CANARY_V2_DATA_ROOT
        data_manifest = PROD_REWARD_CANARY_V2_DATA_MANIFEST
        wandb = PROD_REWARD_CANARY_V2_WANDB
        tokens_per_turn = 8192
        expected_data = EXPECTED_REWARD_CANARY_V2_DATA
    elif mode == PROD_REWARD_CANARY_V3_MODE:
        name = PROD_REWARD_CANARY_V3_NAME
        output = PROD_REWARD_CANARY_V3_OUTPUT
        data_root = PROD_REWARD_CANARY_V3_DATA_ROOT
        data_manifest = PROD_REWARD_CANARY_V3_DATA_MANIFEST
        wandb = PROD_REWARD_CANARY_V3_WANDB
        tokens_per_turn = 32768
        expected_data = EXPECTED_REWARD_CANARY_V3_DATA
        nodes = 1
        max_tokens_per_gpu = 8192
        context_tokens = 98304
        response_tokens = 81920
        native_profile = "qwen3.8-27b"
        harness = "direct"
        runtime_image = miles.IMAGE
        session_node_cap = 1024
        checkpoint_root = BASE_CHECKPOINT["root"]
        checkpoint_receipt_sha256 = BASE_CHECKPOINT["receipt_sha256"]
        resources = PROD_REWARD_CANARY_RESOURCES
    elif mode in {
        PROD_REWARD_CANARY_LONG_V1_MODE,
        PROD_REWARD_CANARY_LONG_V2_MODE,
    }:
        successor = mode == PROD_REWARD_CANARY_LONG_V2_MODE
        name = (
            PROD_REWARD_CANARY_LONG_V2_NAME
            if successor
            else PROD_REWARD_CANARY_LONG_V1_NAME
        )
        output = (
            PROD_REWARD_CANARY_LONG_V2_OUTPUT
            if successor
            else PROD_REWARD_CANARY_LONG_V1_OUTPUT
        )
        data_root = (
            PROD_REWARD_CANARY_LONG_V2_DATA_ROOT
            if successor
            else PROD_REWARD_CANARY_LONG_V1_DATA_ROOT
        )
        data_manifest = (
            PROD_REWARD_CANARY_LONG_V2_DATA_MANIFEST
            if successor
            else PROD_REWARD_CANARY_LONG_V1_DATA_MANIFEST
        )
        wandb = (
            PROD_REWARD_CANARY_LONG_V2_WANDB
            if successor
            else PROD_REWARD_CANARY_LONG_V1_WANDB
        )
        tokens_per_turn = 32768
        expected_data = (
            EXPECTED_REWARD_CANARY_LONG_V2_DATA
            if successor
            else EXPECTED_REWARD_CANARY_LONG_V1_DATA
        )
        nodes = 4
        max_tokens_per_gpu = 65536
        context_tokens = 262144
        response_tokens = 245760
        native_profile = "qwen3.8-27b-256k"
        harness = "opencode"
        runtime_image = (
            PROD_REWARD_CANARY_LONG_V2_IMAGE
            if successor
            else PROD_REWARD_CANARY_LONG_V1_IMAGE
        )
        session_node_cap = 4096
        checkpoint_root = (
            PROD_REWARD_CANARY_LONG_V2_CHECKPOINT_ROOT
            if successor
            else PROD_REWARD_CANARY_LONG_V1_CHECKPOINT_ROOT
        )
        checkpoint_receipt_sha256 = (
            PROD_REWARD_CANARY_LONG_V2_CHECKPOINT_RECEIPT_SHA256
            if successor
            else PROD_REWARD_CANARY_LONG_V1_CHECKPOINT_RECEIPT_SHA256
        )
        resources = PROD_REWARD_CANARY_LONG_V1_RESOURCES
    else:
        raise ValueError("compiled Miles production reward canary mode changed")
    if mode in {
        PROD_REWARD_CANARY_MODE,
        PROD_REWARD_CANARY_V2_MODE,
        PROD_REWARD_CANARY_V3_MODE,
    }:
        nodes = 1
        max_tokens_per_gpu = 8192
        context_tokens = 98304
        response_tokens = 81920
        native_profile = "qwen3.8-27b"
        harness = "direct"
        runtime_image = miles.IMAGE
        session_node_cap = 1024
        checkpoint_root = BASE_CHECKPOINT["root"]
        checkpoint_receipt_sha256 = BASE_CHECKPOINT["receipt_sha256"]
        resources = PROD_REWARD_CANARY_RESOURCES
    expected = {
        "name": name,
        "output_root": output,
        "model_root": "/mnt/sfs/models/qwen3.8-27b-1d4bf0f2",
        "policy_identity_root": "/mnt/sfs/models/qwen3.8-27b-1d4bf0f2",
        "torch_dist_root": checkpoint_root,
        "train_data": data_root + "/train.jsonl",
        "dev_data": data_root + "/dev.jsonl",
        "data_manifest": data_manifest,
        "wandb_entity": wandb["entity"],
        "wandb_project": wandb["project"],
        "wandb_run_id": wandb["run_id"],
        "model": "Qwen/Qwen3.8-27B",
        "nodes": nodes,
        "gpus_per_node": 8,
        "steps": 1,
        "groups": 1,
        "samples_per_prompt": 8,
        "lr": 2e-6,
        "temperature": 0.7,
        "kl_loss_coef": 0.001,
        "max_tokens_per_gpu": max_tokens_per_gpu,
        "eval_interval": 1,
        "checkpoint_interval": 1,
        "seed": 42,
        "context_tokens": context_tokens,
        "response_tokens": response_tokens,
        "tokens_per_turn": tokens_per_turn,
        "native_profile": native_profile,
        "harness": harness,
        "runtime_image": runtime_image,
        "session_node_cap": session_node_cap,
    }
    if (
        plan.get("run_name") != name
        or plan.get("output_root") != output
        or args != expected
        or checkpoint.get("schema") != "cyber_miles_checkpoint_v1"
        or checkpoint.get("optimizer_steps") != 0
        or checkpoint.get("root") != checkpoint_root
        or checkpoint.get("sha256", "").removeprefix("sha256:")
        != checkpoint_receipt_sha256
        or checkpoint.get("image") != runtime_image
        or checkpoint.get("model") != plan.get("model")
        or digest(plan.get("model")) != PROD_MODEL_SHA256
        or execution
        != {
            "image": runtime_image,
            "priority": "c1",
            "resources": resources,
            "cluster_target": "prod",
            "production_promotion": {"mode": mode},
        }
    ):
        raise ValueError("compiled Miles production reward canary differs from the exact arm")
    if mode in {
        PROD_REWARD_CANARY_LONG_V1_MODE,
        PROD_REWARD_CANARY_LONG_V2_MODE,
    }:
        _exact_long_data(plan.get("data"), expected_data)
    else:
        _exact_data(plan.get("data"), expected_data)


def validate_embedded_promotion(plan: dict[str, Any], *, check_files: bool = True) -> bool:
    required = requires_production_promotion(plan)
    proof = plan.get("execution", {}).get("production_promotion")
    if not required:
        if proof is not None:
            raise ValueError("Miles production promotion attached to a dev plan")
        return False
    if _is_reward_canary_marker(proof):
        _exact_reward_canary_plan(plan)
        return True
    if not isinstance(proof, dict) or set(proof) != {*_REF_FIELDS, "receipt"}:
        raise ValueError("embedded Miles production promotion is absent")
    if not isinstance(proof["receipt"], dict):
        raise ValueError("embedded Miles production promotion receipt is invalid")
    reference = {key: proof[key] for key in _REF_FIELDS}
    observed = _reopen(reference, check_files=check_files)
    if observed is not None and observed != proof["receipt"]:
        raise ValueError("embedded Miles production promotion receipt changed")
    if proof["receipt"].get("sha256", "").removeprefix("sha256:") != proof[
        "receipt_sha256"
    ].removeprefix("sha256:"):
        raise ValueError("embedded Miles production promotion digest changed")
    validate_promotion(proof["receipt"], check_files=check_files)
    _exact_plan(plan)
    return True


def validate_production_preview(
    plan: dict[str, Any], request: dict[str, Any], preview: dict[str, Any]
) -> dict[str, Any]:
    if not validate_embedded_promotion(plan, check_files=True):
        return {}
    from cyber_post_train.jobs import validate_preview

    rendered = validate_preview(request, preview)
    if _is_reward_canary_marker(plan.get("execution", {}).get("production_promotion")):
        long_horizon = (
            plan.get("execution", {}).get("production_promotion", {}).get("mode")
            in {
                PROD_REWARD_CANARY_LONG_V1_MODE,
                PROD_REWARD_CANARY_LONG_V2_MODE,
            }
        )
        workers = 4 if long_horizon else 1
        resources = (
            PROD_REWARD_CANARY_LONG_V1_RESOURCES
            if long_horizon
            else PROD_REWARD_CANARY_RESOURCES
        )
        if (
            request.get("workers") != workers
            or request.get("gpus_per_worker") != 8
            or request.get("priority_class") != "c1"
            or "priority_reason" in request
            or request.get("requeueIfPreempted") is not False
            or request.get("resources") != resources
        ):
            raise JobsError(
                f"Miles production reward canary preview is not exact {workers}x8 c1/no-requeue"
            )
        return {
            "production_reward_canary": "validated",
            "rendered_nodes": rendered["nodes"],
            "effective_priority_expected": 10000,
        }
    if (
        request.get("workers") != 1
        or request.get("gpus_per_worker") != 8
        or request.get("priority_class") != "c1"
        or request.get("requeueIfPreempted") is not False
        or request.get("resources") != EXPECTED_RESOURCES
    ):
        raise JobsError("Miles production preview request is not exact 1x8 c1/no-requeue")
    return {
        "production_promotion": "validated",
        "rendered_nodes": rendered["nodes"],
        "effective_priority_expected": 10000,
    }


def _reward_canary_live_identity(
    plan: dict[str, Any],
) -> tuple[str, str, str, dict[str, str], dict[str, Any]]:
    mode = plan.get("execution", {}).get("production_promotion", {}).get("mode")
    if mode == PROD_REWARD_CANARY_MODE:
        return (
            PROD_NAME,
            PROD_REWARD_CANARY_OUTPUT,
            PROD_DATA_MANIFEST,
            PROD_REWARD_CANARY_WANDB,
            EXPECTED_DATA,
        )
    if mode == PROD_REWARD_CANARY_V2_MODE:
        return (
            PROD_REWARD_CANARY_V2_NAME,
            PROD_REWARD_CANARY_V2_OUTPUT,
            PROD_REWARD_CANARY_V2_DATA_MANIFEST,
            PROD_REWARD_CANARY_V2_WANDB,
            EXPECTED_REWARD_CANARY_V2_DATA,
        )
    if mode == PROD_REWARD_CANARY_V3_MODE:
        return (
            PROD_REWARD_CANARY_V3_NAME,
            PROD_REWARD_CANARY_V3_OUTPUT,
            PROD_REWARD_CANARY_V3_DATA_MANIFEST,
            PROD_REWARD_CANARY_V3_WANDB,
            EXPECTED_REWARD_CANARY_V3_DATA,
        )
    if mode == PROD_REWARD_CANARY_LONG_V1_MODE:
        return (
            PROD_REWARD_CANARY_LONG_V1_NAME,
            PROD_REWARD_CANARY_LONG_V1_OUTPUT,
            PROD_REWARD_CANARY_LONG_V1_DATA_MANIFEST,
            PROD_REWARD_CANARY_LONG_V1_WANDB,
            EXPECTED_REWARD_CANARY_LONG_V1_DATA,
        )
    if mode == PROD_REWARD_CANARY_LONG_V2_MODE:
        return (
            PROD_REWARD_CANARY_LONG_V2_NAME,
            PROD_REWARD_CANARY_LONG_V2_OUTPUT,
            PROD_REWARD_CANARY_LONG_V2_DATA_MANIFEST,
            PROD_REWARD_CANARY_LONG_V2_WANDB,
            EXPECTED_REWARD_CANARY_LONG_V2_DATA,
        )
    raise JobsError("Miles production reward canary mode changed")


def require_live_files(plan: dict[str, Any], prepared_directory: Path) -> None:
    if not validate_embedded_promotion(plan, check_files=True):
        return
    if (prepared_directory / "SUBMISSION.jsonl").exists() or (
        prepared_directory / "SUBMISSION.jsonl"
    ).is_symlink():
        raise JobsError("Miles production submission journal already exists")
    reward_canary = _is_reward_canary_marker(plan.get("execution", {}).get("production_promotion"))
    if reward_canary:
        _, output, manifest_path, _, expected_data = _reward_canary_live_identity(plan)
    else:
        output, manifest_path, expected_data = PROD_OUTPUT, PROD_DATA_MANIFEST, EXPECTED_DATA
    if Path(output).exists() or Path(output).is_symlink():
        raise JobsError("Miles production output already exists")
    manifest = Path(manifest_path)
    if reward_canary:
        # The bounded canary deliberately has no dev-promotion receipt. Its
        # compiled plan already seals the exact parsed manifest, so reopen the
        # stable file and require that byte snapshot to decode to that value.
        observed = _snapshot(manifest, _sha256(manifest))
        if observed != plan.get("data"):
            raise ValueError("Miles production reward-canary data manifest changed")
    else:
        observed = _snapshot(
            manifest,
            plan["execution"]["production_promotion"]["receipt"]["production_data_manifest"][
                "file_sha256"
            ],
        )
    if observed.get("schema") == "cyber_miles_data_v2":
        _exact_long_data(observed)
    else:
        _exact_data(observed, expected_data)


def _kubectl_json(*arguments: str) -> dict[str, Any]:
    host = os.environ.get("KUBERNETES_SERVICE_HOST")
    if host:
        import httpx

        if arguments == ("get", "namespace", "fleet-train-jobs"):
            path = "/api/v1/namespaces/fleet-train-jobs"
        elif arguments[:2] == ("get", "priorityclass") and len(arguments) == 3:
            path = f"/apis/scheduling.k8s.io/v1/priorityclasses/{arguments[2]}"
        elif arguments == ("get", "pods", "-n", "fleet-train-jobs"):
            path = "/api/v1/namespaces/fleet-train-jobs/pods"
        else:
            raise JobsError("unsupported in-cluster Kubernetes read-only gate")
        token_path = Path("/var/run/secrets/kubernetes.io/serviceaccount/token")
        ca_path = Path("/var/run/secrets/kubernetes.io/serviceaccount/ca.crt")
        if not token_path.is_file() or not ca_path.is_file():
            raise JobsError("in-cluster Kubernetes identity is unavailable")
        try:
            response = httpx.get(
                f"https://{host}:{os.environ.get('KUBERNETES_SERVICE_PORT', '443')}{path}",
                headers={"Authorization": "Bearer " + token_path.read_text().strip()},
                verify=str(ca_path),
                timeout=30,
            )
            response.raise_for_status()
            value = response.json()
        except (httpx.HTTPError, ValueError):
            raise JobsError("production Kubernetes read-only gate failed") from None
        if not isinstance(value, dict):
            raise JobsError("production Kubernetes read-only result is not an object")
        return value
    result = subprocess.run(
        ["kubectl", "--context", PROD_KUBE_CONTEXT, *arguments, "-o", "json"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode:
        raise JobsError("production Kubernetes read-only gate failed")
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise JobsError("production Kubernetes read-only result is invalid") from error
    if not isinstance(value, dict):
        raise JobsError("production Kubernetes read-only result is not an object")
    return value


def _owned_experiment_pod(pod: dict[str, Any]) -> bool:
    """Select Chris's experiment Pods without charging peer work to his node cap."""
    metadata = pod.get("metadata")
    if not isinstance(metadata, dict) or not isinstance(metadata.get("name"), str):
        raise JobsError("production Pod identity is malformed")
    labels = metadata.get("labels", {})
    if not isinstance(labels, dict):
        raise JobsError("production Pod labels are malformed")
    run_name = labels.get(FLEET_RUN_NAME_LABEL)
    if run_name is not None and not isinstance(run_name, str):
        raise JobsError("production Pod run-name label is malformed")
    return metadata["name"].startswith(EXPERIMENT_OWNER_PREFIX) or (
        isinstance(run_name, str) and run_name.startswith(EXPERIMENT_OWNER_PREFIX)
    )


def _pod_gpu_count(pod: dict[str, Any]) -> int:
    spec = pod.get("spec")
    if not isinstance(spec, dict):
        raise JobsError("production Pod specification is malformed")
    containers = spec.get("containers")
    if not isinstance(containers, list):
        raise JobsError("production Pod container inventory is malformed")
    total = 0
    for container in containers:
        if not isinstance(container, dict):
            raise JobsError("production Pod container inventory is malformed")
        resources = container.get("resources", {})
        if not isinstance(resources, dict):
            raise JobsError("production Pod GPU inventory is malformed")
        observed = []
        for field in ("requests", "limits"):
            quantities = resources.get(field, {})
            if not isinstance(quantities, dict):
                raise JobsError("production Pod GPU inventory is malformed")
            try:
                observed.append(int(quantities.get("nvidia.com/gpu", 0)))
            except (TypeError, ValueError) as error:
                raise JobsError("production Pod GPU inventory is malformed") from error
        total += max(observed)
    return total


def require_live_external(plan: dict[str, Any], client, *, wandb_api=None) -> dict[str, Any]:
    """Recheck Jobs, W&B and the eight-node ceiling immediately before POST."""
    if not validate_embedded_promotion(plan, check_files=True):
        return {}
    reward_canary = _is_reward_canary_marker(plan.get("execution", {}).get("production_promotion"))
    if reward_canary:
        run_name, output, _, wandb_identity, _ = _reward_canary_live_identity(plan)
    else:
        run_name, output, wandb_identity = PROD_NAME, PROD_OUTPUT, PROD_WANDB
    if any(
        not isinstance(row, dict)
        or row.get("name") == run_name
        or str(row.get("name", "")).startswith(run_name + "-")
        or row.get("run_dir") == output
        for row in client.all_runs()
    ):
        raise JobsError("Miles production Jobs API identity/output exists or is malformed")
    if wandb_api is None:
        import wandb

        wandb_api = wandb.Api()
    try:
        observed = wandb_api.run(
            f"{wandb_identity['entity']}/{wandb_identity['project']}/{wandb_identity['run_id']}"
        )
    except Exception as error:  # W&B has no stable not-found exception across pins.
        message = str(error).lower()
        if "not found" not in message and "could not find run" not in message:
            raise JobsError("Miles production W&B duplicate check failed closed") from None
    else:
        if observed is not None:
            raise JobsError("Miles production W&B run ID already exists")
    namespace = _kubectl_json("get", "namespace", "fleet-train-jobs")
    if namespace.get("metadata", {}).get("uid") != PROD_NAMESPACE_UID:
        raise JobsError("production namespace identity changed")
    priority_name = "c1"
    priority_value = 10000
    priority = _kubectl_json("get", "priorityclass", priority_name)
    if (
        priority.get("metadata", {}).get("name") != priority_name
        or priority.get("value") != priority_value
        or priority.get("preemptionPolicy") != "PreemptLowerPriority"
    ):
        raise JobsError("production c1 effective priority changed")
    pods = _kubectl_json("get", "pods", "-n", "fleet-train-jobs").get("items")
    if not isinstance(pods, list):
        raise JobsError("production Pod inventory is malformed")
    active_nodes, unscheduled = set(), 0
    for pod in pods:
        if not isinstance(pod, dict):
            raise JobsError("production Pod inventory is malformed")
        if pod.get("status", {}).get("phase") not in {"Pending", "Running"}:
            continue
        if not _owned_experiment_pod(pod):
            continue
        gpu = _pod_gpu_count(pod)
        if gpu:
            node = pod.get("spec", {}).get("nodeName")
            if node:
                active_nodes.add(node)
            else:
                unscheduled += 1
    active = len(active_nodes) + unscheduled
    candidate_nodes = (
        4
        if plan.get("execution", {}).get("production_promotion", {}).get("mode")
        in {
            PROD_REWARD_CANARY_LONG_V1_MODE,
            PROD_REWARD_CANARY_LONG_V2_MODE,
        }
        else 1
    )
    if active + candidate_nodes > 8:
        raise JobsError("Miles production node budget would exceed eight")
    return {
        "jobs_api_absent": True,
        "wandb_absent": True,
        "active_experiment_nodes": active,
        "candidate_nodes": candidate_nodes,
        "node_limit": 8,
        "effective_priority": priority["value"],
    }
