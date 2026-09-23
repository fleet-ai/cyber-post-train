"""Append-only Fast3 qualification layered on the frozen v8 reward canary."""

from __future__ import annotations

import copy
import json
from pathlib import Path, PurePosixPath

from cyber_post_train.jobs import digest
from evals.fleet import opencode_self_hosted as fleet

from . import rl_reward_canary as historical
from . import skyrl_prod9_training

ROOT = historical.ROOT
QUALIFICATION_PATH = historical.QUALIFICATION_PATH
QUALIFICATION_FILE_SHA256 = historical.QUALIFICATION_FILE_SHA256
QUALIFICATION_SELF_SHA256 = historical.QUALIFICATION_SELF_SHA256
RUNTIME_SOURCES = historical.RUNTIME_SOURCES
TASK_SET_SELF_SHA256 = historical.TASK_SET_SELF_SHA256
SPLIT_SELF_SHA256 = historical.SPLIT_SELF_SHA256
TOOL_CATALOG_SHA256 = historical.TOOL_CATALOG_SHA256
LIMITS = historical.LIMITS
RECIPE = historical.RECIPE
MODEL = historical.MODEL
TASKS = historical.TASKS
IMAGE = historical.IMAGE
FAST3_SCIENCE_RECIPE = {**RECIPE, "eval_before_train": True}
FAST3_BASE_COMMIT = "cc07933546abb82023cf0413b4538dcf9d992ed5"
FAST3_RUN_PATH = "configs/qualification/qwen38-rl-reward-canary-prod-v11-fast3.json"
FAST3_DATA_PATH = "configs/qualification/qwen38-rl-reward-canary-data-prod11-fast3.json"
FAST3_IDENTITY_PATH = "configs/qualification/qwen38-rl-reward-canary-prod11-fast3-identity-v1.json"
FAST3_POLICY_PATH = (
    "configs/qualification/qwen38-rl-reward-canary-prod11-fast3-generation-retry-v1.json"
)
FAST3_SCIENCE_PATH = (
    "configs/qualification/qwen38-rl-reward-canary-prod11-fast3-predecessor-science-v1.json"
)
FAST3_QUALIFICATION_PATH = "configs/qualification/qwen38-rl-reward-canary-port-v9.json"
FAST3_RUNTIME_EVIDENCE_PATH = "configs/data/qwen38-rl-reward-canary-exact-version-evidence-v9.json"
PROD11_FAILURE_DIAGNOSTIC_PATH = (
    "docs/evidence/qwen38-study/2026-09-23-skyrl-prod11-generation-failure-diagnostic-v1.json"
)
FAST2_RETIREMENT_PATH = (
    "docs/evidence/qwen38-study/2026-09-23-skyrl-prod11-fast2-retirement-v1.json"
)
PROD11_FAILURE_DIAGNOSTIC_FILE_SHA256 = (
    "sha256:69ced0087abecd437308c2ffb66e8a0167bf56f7a3ba3ad22f3b1c18a481c2f0"
)
PROD11_FAILURE_DIAGNOSTIC_SELF_SHA256 = (
    "200d51089c155cac2fa777db4f84287c50d14bdb23c86e60e8e47c2d3d2fd2fe"
)
FAST2_RETIREMENT_FILE_SHA256 = (
    "sha256:a99d7f43e2a7f202404a9903f38f5f19ed388dcd9f870bcaef5dc85649e0cf39"
)
FAST2_RETIREMENT_SELF_SHA256 = (
    "sha256:f8cded5039d649de08db82ff5f21deb21979ca9dade7d431c0550dc268e5552d"
)
FAST3_IDENTITY = {
    "run_name": "chris-q38-rlreward-prod11-fast3",
    "output_root": "/mnt/sfs/jobs/chris-q38-rlreward-prod11-fast3",
    "data_root": "/mnt/sfs/jobs/chris-q38-study-corpora-v1/rlreward-inputs-prod11-fast3-v1/data",
    "wandb_run_id": "chris-q38-rlreward-prod11-fast3",
}
FAST3_PORT_FILES = tuple(
    dict.fromkeys(
        (
            *skyrl_prod9_training.RUNTIME_FILES,
            "training/checkpoints.py",
            "training/export.py",
            "training/export_check.py",
            "training/post_sft_artifacts.py",
            "training/post_sft_base_surface.py",
            "training/post_sft_cast.py",
            "training/skyrl_posttrain.py",
            "training/skyrl_prod9_reload.py",
            "training/rl_reward_fast3.py",
            "training/skyrl_fast3_posttrain.py",
            "training/skyrl_fast3_reload.py",
            "training/skyrl_fast3_retry.py",
            "training/skyrl_fast3_rollout.py",
            "training/skyrl_fast3_training.py",
        )
    )
)
FAST3_PROFILE_FILE_SHA256 = {
    "run": "sha256:a9f642b167731cb30391cb09b4fdc90e2cfd10fc82b3be8ceaefd068456c9844",
    "data": "sha256:84fe83fe3444837a36e90a8dc367531014316f5b8bf0b9c12f4d73fca995bf6c",
    "identity": "sha256:d77c51b8d54ce537321ff3ed366390bdb13418eb24a17ae02e4ff83d8c32a67a",
    "policy": "sha256:3ead20ef5330303687a377560dff0b097bef6f481194542503978b07788e577f",
}

_path = historical._path
_bound_json = historical._bound_json
_sealed = historical._sealed


def _fast3_port_successor_files() -> dict[str, str]:
    result = {}
    for path in FAST3_PORT_FILES:
        source = _path(ROOT / path)
        before = source.stat()
        payload = source.read_bytes()
        after = source.stat()
        if any(
            getattr(before, key) != getattr(after, key)
            for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        ):
            raise ValueError("Fast3 port source changed while being read")
        result[path] = fleet.sha256(payload)
    return result


def fast3_port_successor_sha256() -> str:
    """Bind the small current-main-native Fast3 source delta as one digest."""
    return "sha256:" + digest(_fast3_port_successor_files())


def _fast3_expected_runtime_evidence(*, files: dict[str, str] | None = None) -> dict:
    selected_files = _fast3_port_successor_files() if files is None else files
    body = {
        "schema": "cyber_rl_fast3_current_main_runtime_evidence_v1",
        "base_commit": FAST3_BASE_COMMIT,
        "legacy_no_policy": {
            "adapter": "training.skyrl_episode.single_attempt_engine",
            "rl_episode_file_sha256": RUNTIME_SOURCES["training/rl_episode.py"],
            "skyrl_episode_file_sha256": RUNTIME_SOURCES["training/skyrl_episode.py"],
            "unchanged": True,
        },
        "eval_schedule": {
            "eval_before_train": True,
            "post_update_eval_required": True,
        },
        "generation_retry_policy": {
            "path": "../qualification/" + Path(FAST3_POLICY_PATH).name,
            "file_sha256": FAST3_PROFILE_FILE_SHA256["policy"],
            "self_sha256": (
                "sha256:3df3abe9fbadcf4cbc89d842a857d68b2510ef36f919bc035427fd85db5b2c28"
            ),
        },
        "port_successor_files": selected_files,
        "port_successor_sha256": "sha256:" + digest(selected_files),
    }
    return {**body, "sha256": "sha256:" + digest(body)}


def _fast3_expected_qualification(
    *, science: dict, science_file_sha256: str, runtime: dict, runtime_file_sha256: str
) -> dict:
    body = {
        "schema": "cyber_qwen38_skyrl_reward_canary_port_v9",
        "parent_qualification": {
            "path": Path(QUALIFICATION_PATH).name,
            "file_sha256": QUALIFICATION_FILE_SHA256,
            "self_sha256": QUALIFICATION_SELF_SHA256,
        },
        "runtime_evidence": {
            "path": "../data/" + Path(FAST3_RUNTIME_EVIDENCE_PATH).name,
            "file_sha256": runtime_file_sha256,
            "self_sha256": runtime["sha256"],
        },
        "predecessor_science": {
            "path": Path(FAST3_SCIENCE_PATH).name,
            "file_sha256": science_file_sha256,
            "self_sha256": science["sha256"],
        },
        "source": {
            "base_commit": FAST3_BASE_COMMIT,
            "port_successor_sha256": runtime["port_successor_sha256"],
        },
        "execution": {
            "cluster_target": "prod",
            "jobs_api_base_url": "https://api.ft.flt.build",
            "image": IMAGE,
            "environment": {"VLLM_USE_FLASHINFER_SAMPLER": "0"},
        },
        "submission_gate": {
            "preview_authorized": False,
            "submission_authorized": False,
            "blockers": [
                "fast3_source_pr_not_merged",
                "fast3_launch_chain_not_separately_bound",
            ],
        },
    }
    return {**body, "sha256": "sha256:" + digest(body)}


def _fast3_profile(path: str) -> tuple[dict, str]:
    source = _path(ROOT / path)
    before = source.stat()
    payload = source.read_bytes()
    after = source.stat()
    if any(
        getattr(before, key) != getattr(after, key)
        for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    ):
        raise ValueError("Fast3 profile changed while being read")
    try:
        value = json.loads(payload)
    except (TypeError, ValueError) as error:
        raise ValueError("Fast3 profile is not valid JSON") from error
    if type(value) is not dict:
        raise ValueError("Fast3 profile must be an object")
    return value, fleet.sha256(payload)


def _fast3_historical_evidence() -> None:
    diagnostic = _bound_json(
        ROOT / PROD11_FAILURE_DIAGNOSTIC_PATH,
        PROD11_FAILURE_DIAGNOSTIC_FILE_SHA256,
    )
    retirement = _bound_json(
        ROOT / FAST2_RETIREMENT_PATH,
        FAST2_RETIREMENT_FILE_SHA256,
    )
    diagnostic_body = {key: item for key, item in diagnostic.items() if key != "sha256"}
    retirement_body = {key: item for key, item in retirement.items() if key != "sha256"}
    causes = diagnostic.get("episode_failure", {}).get("causes", [])
    if (
        diagnostic.get("schema") != "cyber_rl_prod11_failure_diagnostic_v1"
        or diagnostic.get("sha256") != PROD11_FAILURE_DIAGNOSTIC_SELF_SHA256
        or diagnostic.get("sha256") != digest(diagnostic_body)
        or diagnostic.get("status") != "completed"
        or diagnostic.get("sfs_read_only") is not True
        or len(causes) != 3
        or causes[-1].get("error_type") != "InvalidEpisode"
        or any("http_status" in cause for cause in causes)
        or diagnostic.get("source", {}).get("plan_sha256")
        != "f86ca0c93754def88d2f9053fb7fa012af3b6d29046846fa5ad229972ce9910f"
        or retirement.get("schema") != "cyber_skyrl_prod11_fast2_retirement_reconciliation_v1"
        or retirement.get("sha256") != FAST2_RETIREMENT_SELF_SHA256
        or retirement.get("sha256") != "sha256:" + digest(retirement_body)
        or retirement.get("source_head") != "58f81904a7478fd90dfe988d97b8fa831fbeadc1"
        or retirement.get("run_name") != "chris-q38-rlreward-prod11-fast2"
        or retirement.get("status") != "retired_without_gpu_launch_identity_and_output_absent"
        or retirement.get("gpu_launch", {}).get("jobs_post_attempted") is not False
        or retirement.get("gpu_launch", {}).get("gpus_requested") != 0
        or retirement.get("output_absence", {}).get("fresh_sfs_lstat_in_this_receipt") is not False
    ):
        raise ValueError("Fast3 historical evidence changed")

    forbidden = {"authorization", "api_key", "secret", "prompt", "trace", "conversation"}

    def keys(value: object):
        if isinstance(value, dict):
            for key, item in value.items():
                yield str(key).lower()
                yield from keys(item)
        elif isinstance(value, list):
            for item in value:
                yield from keys(item)

    if forbidden & set(keys(diagnostic)) or forbidden & set(keys(retirement)):
        raise ValueError("Fast3 historical evidence contains private fields")


def _fast3_normalized_science() -> dict:
    return {
        "model": MODEL,
        "data": {
            "task_set_self_sha256": TASK_SET_SELF_SHA256,
            "split_self_sha256": SPLIT_SELF_SHA256,
            "tool_catalog_sha256": TOOL_CATALOG_SHA256,
        },
        "tasks": [
            {
                key: task[key]
                for key in (
                    "task_key",
                    "task_version_id",
                    "environment_version_id",
                    "verifier_version_id",
                    "split",
                )
            }
            for task in TASKS
        ],
        "horizon": LIMITS,
        "recipe": FAST3_SCIENCE_RECIPE,
    }


def _fast3_profile_bindings(identity_sha256: str, policy_sha256: str) -> dict:
    return {
        "run": {"path": FAST3_RUN_PATH, "file_sha256": FAST3_PROFILE_FILE_SHA256["run"]},
        "data": {
            "path": FAST3_DATA_PATH,
            "file_sha256": FAST3_PROFILE_FILE_SHA256["data"],
        },
        "identity": {
            "path": FAST3_IDENTITY_PATH,
            "file_sha256": FAST3_PROFILE_FILE_SHA256["identity"],
            "self_sha256": identity_sha256,
        },
        "policy": {
            "path": FAST3_POLICY_PATH,
            "file_sha256": FAST3_PROFILE_FILE_SHA256["policy"],
            "self_sha256": policy_sha256,
        },
    }


def _fast3_expected_science(
    *, port_successor_sha256: str, identity_sha256: str, policy_sha256: str
) -> dict:
    body = {
        "schema": "cyber_skyrl_fast3_predecessor_science_v1",
        "scientific_predecessor": {
            "source_commit": "f7452d7eafb33f2d84b724f07480b58c91826207",
            "run_name": "chris-q38-rlreward-prod11",
            "run_config": {
                "path": "configs/qualification/qwen38-rl-reward-canary-prod-v11.json",
                "file_sha256": (
                    "sha256:a7f34702b4c19df80128ed1fd8e0114502605eafd3dd63daf3105f171092038c"
                ),
            },
            "data_config": {
                "path": "configs/qualification/qwen38-rl-reward-canary-data-prod-v11.json",
                "file_sha256": (
                    "sha256:28e30df00464fd7c8413d8ad2bd7078d3ab352adfcc5b4bac4dd6e6d91e639e9"
                ),
            },
            "identity": {
                "path": "configs/qualification/qwen38-rl-reward-canary-prod11-identity-v1.json",
                "file_sha256": (
                    "sha256:5aa3fa64fe91e9b51a920020d6c64302224cbf60a3a82c5dc6855b20139de7b3"
                ),
                "self_sha256": (
                    "sha256:74e756a0929488b8cd596c0a28ed0c58c6f6abe254e1b0a78b9a4f0e783145d0"
                ),
            },
            "eval_before_train": True,
            "failure_diagnostic": {
                "path": PROD11_FAILURE_DIAGNOSTIC_PATH,
                "file_sha256": PROD11_FAILURE_DIAGNOSTIC_FILE_SHA256,
                "self_sha256": PROD11_FAILURE_DIAGNOSTIC_SELF_SHA256,
                "plan_sha256": ("f86ca0c93754def88d2f9053fb7fa012af3b6d29046846fa5ad229972ce9910f"),
                "http_status_known": False,
                "retryability_proven": False,
            },
        },
        "operational_predecessor": {
            "source_commit": "58f81904a7478fd90dfe988d97b8fa831fbeadc1",
            "run_name": "chris-q38-rlreward-prod11-fast2",
            "identity_file_sha256": (
                "sha256:d1ae811bae6e0b9ce94f8c108c7264dcea4e2c9c50bda2d6bde8a872f2d18a5f"
            ),
            "data_config_file_sha256": (
                "sha256:2bc7c34a6877c1025310c5b9b85b0f8a7e3dacc361f8b5a6c3a103c6b803f5e7"
            ),
            "run_config_file_sha256": (
                "sha256:3e8f078b97e7fbfad3c1428fb26a2c5b0284402545b1b54bf3417610a9dcff0b"
            ),
            "retirement": {
                "path": FAST2_RETIREMENT_PATH,
                "file_sha256": FAST2_RETIREMENT_FILE_SHA256,
                "self_sha256": FAST2_RETIREMENT_SELF_SHA256,
                "fresh_sfs_lstat": False,
                "gpu_launch_attempted": False,
            },
            "optimizer_updates": 0,
            "reward_observations": 0,
            "scientific_parity_claim": False,
        },
        "normalized_science": _fast3_normalized_science(),
        "fast3_identity": FAST3_IDENTITY,
        "fast3_profiles": _fast3_profile_bindings(identity_sha256, policy_sha256),
        "historical_qualification": {
            "path": QUALIFICATION_PATH,
            "file_sha256": QUALIFICATION_FILE_SHA256,
            "self_sha256": QUALIFICATION_SELF_SHA256,
        },
        "retry_evidence": {
            "historical_http_status_known": False,
            "retryability_proven": False,
        },
        "retry_policy_sha256": policy_sha256,
        "port_successor_sha256": port_successor_sha256,
        "permitted_deltas": [
            "fresh_identity_output_data_root",
            "port_successor_sha256",
            "generation_retry_policy",
        ],
    }
    return {**body, "sha256": "sha256:" + digest(body)}


def _fast3_binding(config: dict) -> dict | None:
    selected = {
        "run_name": config.get("name"),
        "output_root": config.get("output_root"),
        "data_root": config.get("data", {}).get("root"),
        "wandb_run_id": config.get("wandb", {}).get("run_id"),
    }
    if selected != FAST3_IDENTITY:
        if FAST3_IDENTITY["run_name"] in selected.values():
            raise ValueError("Fast3 identity is incomplete")
        return None

    from . import skyrl_fast3_retry

    _fast3_historical_evidence()
    run_profile, run_file_sha256 = _fast3_profile(FAST3_RUN_PATH)
    data_profile, data_file_sha256 = _fast3_profile(FAST3_DATA_PATH)
    identity, identity_file_sha256 = _fast3_profile(FAST3_IDENTITY_PATH)
    policy, policy_file_sha256 = _fast3_profile(FAST3_POLICY_PATH)
    science, science_file_sha256 = _fast3_profile(FAST3_SCIENCE_PATH)
    runtime, runtime_file_sha256 = _fast3_profile(FAST3_RUNTIME_EVIDENCE_PATH)
    qualification, qualification_file_sha256 = _fast3_profile(FAST3_QUALIFICATION_PATH)
    _sealed(identity, "cyber_skyrl_reward_direct_identity_v1")
    _sealed(science, "cyber_skyrl_fast3_predecessor_science_v1")
    checked_policy = skyrl_fast3_retry.validate_policy(policy)
    expected_identity = {
        **FAST3_IDENTITY,
        "schema": "cyber_skyrl_reward_direct_identity_v1",
        "stage_name": "chris-q38-prod11-fast3-data-v1",
        "preflight_name": "chris-q38-prod11-fast3-preflight-v1",
        "predecessor_run_name": "chris-q38-rlreward-prod11-fast2",
        "predecessor_data_root": (
            "/mnt/sfs/jobs/chris-q38-study-corpora-v1/rlreward-inputs-prod11-fast2-v1/data"
        ),
    }
    expected_identity["sha256"] = "sha256:" + digest(expected_identity)
    expected_science = _fast3_expected_science(
        port_successor_sha256=fast3_port_successor_sha256(),
        identity_sha256=identity["sha256"],
        policy_sha256=policy["sha256"],
    )
    expected_runtime = _fast3_expected_runtime_evidence()
    expected_qualification = _fast3_expected_qualification(
        science=science,
        science_file_sha256=science_file_sha256,
        runtime=runtime,
        runtime_file_sha256=runtime_file_sha256,
    )
    if (
        config != run_profile
        or config.get("recipe") != RECIPE
        or config.get("qualification") != Path(FAST3_QUALIFICATION_PATH).name
        or {
            "run": run_file_sha256,
            "data": data_file_sha256,
            "identity": identity_file_sha256,
            "policy": policy_file_sha256,
        }
        != FAST3_PROFILE_FILE_SHA256
        or data_profile
        != {
            "backend": "skyrl",
            "name": FAST3_IDENTITY["run_name"],
            "output": FAST3_IDENTITY["data_root"],
            "model_lock": "../models/qwen38-27b-1d4bf0f2.lock.json",
            "model_root": MODEL["root"],
            "task_set": "../data/qwen38-rl-reward-canary-task-set-v3.json",
            "split": "../data/qwen38-rl-reward-canary-split-v1.json",
            "tool_catalog": "../data/qwen38-rl-filtered-canary-tool-catalog-v1.json",
            "limits": LIMITS,
        }
        or identity != expected_identity
        or checked_policy != policy
        or science != expected_science
        or runtime != expected_runtime
        or qualification != expected_qualification
    ):
        raise ValueError("Fast3 predecessor-science profile changed")
    body = {
        "schema": "cyber_qwen38_skyrl_fast3_plan_binding_v1",
        "eval_before_train": True,
        "identity_sha256": identity["sha256"],
        "qualification_file_sha256": qualification_file_sha256,
        "qualification": qualification,
        "runtime_evidence": runtime,
        "predecessor_science_file_sha256": science_file_sha256,
        "predecessor_science": science,
        "port_successor_sha256": science["port_successor_sha256"],
        "generation_retry_policy": checked_policy,
    }
    return {**body, "sha256": "sha256:" + digest(body)}


def _validate_fast3_plan_binding(value: object, metadata: dict, arguments: dict) -> bool:
    """Validate Fast3 against the exact source bytes executing this check."""
    selected = {
        "run_name": arguments.get("name"),
        "output_root": arguments.get("output_root"),
        "data_root": str(PurePosixPath(arguments.get("data_manifest", "")).parent),
        "wandb_run_id": arguments.get("wandb_run_id"),
    }
    if value is None:
        if selected == FAST3_IDENTITY or metadata.get("name") == FAST3_IDENTITY["run_name"]:
            raise ValueError("Fast3 plan lacks its sealed binding")
        return False
    if type(value) is not dict:
        raise ValueError("Fast3 plan binding is invalid")
    if set(value) != {
        "schema",
        "eval_before_train",
        "identity_sha256",
        "qualification_file_sha256",
        "qualification",
        "runtime_evidence",
        "predecessor_science_file_sha256",
        "predecessor_science",
        "port_successor_sha256",
        "generation_retry_policy",
        "sha256",
    }:
        raise ValueError("Fast3 plan binding fields changed")
    from . import skyrl_fast3_retry

    body = {key: item for key, item in value.items() if key != "sha256"}
    science = value.get("predecessor_science")
    runtime = value.get("runtime_evidence")
    qualification = value.get("qualification")
    policy = skyrl_fast3_retry.validate_policy(value.get("generation_retry_policy"))
    identity_body = {
        **FAST3_IDENTITY,
        "schema": "cyber_skyrl_reward_direct_identity_v1",
        "stage_name": "chris-q38-prod11-fast3-data-v1",
        "preflight_name": "chris-q38-prod11-fast3-preflight-v1",
        "predecessor_run_name": "chris-q38-rlreward-prod11-fast2",
        "predecessor_data_root": (
            "/mnt/sfs/jobs/chris-q38-study-corpora-v1/rlreward-inputs-prod11-fast2-v1/data"
        ),
    }
    identity_sha256 = "sha256:" + digest(identity_body)
    expected_science = _fast3_expected_science(
        port_successor_sha256=value.get("port_successor_sha256"),
        identity_sha256=identity_sha256,
        policy_sha256=policy["sha256"],
    )
    trusted_port_files = _fast3_port_successor_files()
    if type(runtime) is not dict or runtime.get("port_successor_files") != trusted_port_files:
        raise ValueError("Fast3 embedded port source closure changed")
    expected_runtime = _fast3_expected_runtime_evidence(files=trusted_port_files)
    science_file_sha256 = fleet.sha256(
        json.dumps(science, indent=2, sort_keys=True).encode() + b"\n"
    )
    runtime_file_sha256 = fleet.sha256(
        json.dumps(runtime, indent=2, sort_keys=True).encode() + b"\n"
    )
    expected_qualification = _fast3_expected_qualification(
        science=science,
        science_file_sha256=science_file_sha256,
        runtime=runtime,
        runtime_file_sha256=runtime_file_sha256,
    )
    if (
        value.get("schema") != "cyber_qwen38_skyrl_fast3_plan_binding_v1"
        or value.get("sha256") != "sha256:" + digest(body)
        or value.get("eval_before_train") is not True
        or selected != FAST3_IDENTITY
        or metadata.get("name") != FAST3_IDENTITY["run_name"]
        or value.get("identity_sha256") != identity_sha256
        or value.get("port_successor_sha256") != runtime.get("port_successor_sha256")
        or value.get("generation_retry_policy") != policy
        or science != expected_science
        or runtime != expected_runtime
        or qualification != expected_qualification
        or value.get("qualification_file_sha256")
        != fleet.sha256(json.dumps(qualification, indent=2, sort_keys=True).encode() + b"\n")
        or value.get("predecessor_science_file_sha256") != science_file_sha256
    ):
        raise ValueError("Fast3 plan binding changed")
    return True


def normalized_historical_config(config: dict) -> dict:
    """Select the unchanged v8 validator while preserving all Fast3 science."""
    result = copy.deepcopy(config)
    result["qualification"] = Path(QUALIFICATION_PATH).name
    return result


def validate_run_config(
    config: dict, metadata: dict, bound_model: dict, *, relative_to: Path
) -> dict:
    fast3 = _fast3_binding(config)
    if fast3 is None:
        raise ValueError("not the exact Fast3 profile")
    legacy = historical.validate_run_config(
        normalized_historical_config(config), metadata, bound_model, relative_to=relative_to
    )
    qualification = fast3["qualification"]
    body = {
        "schema": "cyber_qwen38_skyrl_reward_canary_plan_binding_v9",
        "profile": "qwen38_skyrl_reward_canary_fast3_v1",
        "historical_binding": legacy,
        "source_proof": legacy["source_proof"],
        "qualification_file_sha256": fast3["qualification_file_sha256"],
        "qualification_self_sha256": qualification["sha256"],
        "image": qualification["execution"]["image"],
        "environment": qualification["execution"]["environment"],
        "cluster_target": qualification["execution"]["cluster_target"],
        "jobs_api_base_url": qualification["execution"]["jobs_api_base_url"],
        "submission_gate": qualification["submission_gate"],
        "fast3": fast3,
    }
    return {**body, "sha256": "sha256:" + digest(body)}


def validate_plan_binding(binding: object, metadata: dict, arguments: dict) -> dict:
    if type(binding) is not dict:
        raise ValueError("Fast3 plan lacks its source binding")
    if set(binding) != {
        "schema",
        "profile",
        "historical_binding",
        "source_proof",
        "qualification_file_sha256",
        "qualification_self_sha256",
        "image",
        "environment",
        "cluster_target",
        "jobs_api_base_url",
        "submission_gate",
        "fast3",
        "sha256",
    }:
        raise ValueError("Fast3 reward-canary plan binding fields changed")
    historical_binding = binding.get("historical_binding")
    if type(historical_binding) is not dict or set(historical_binding) != {
        "schema",
        "profile",
        "source_proof",
        "qualification_file_sha256",
        "qualification_self_sha256",
        "image",
        "environment",
        "cluster_target",
        "jobs_api_base_url",
        "submission_gate",
        "sha256",
    }:
        raise ValueError("Fast3 historical plan binding fields changed")
    body = {key: item for key, item in binding.items() if key != "sha256"}
    legacy = historical.validate_plan_binding(historical_binding, metadata, arguments)
    _validate_fast3_plan_binding(binding.get("fast3"), metadata, arguments)
    fast3 = binding["fast3"]
    qualification = fast3["qualification"]
    if (
        binding.get("schema") != "cyber_qwen38_skyrl_reward_canary_plan_binding_v9"
        or binding.get("sha256") != "sha256:" + digest(body)
        or binding.get("profile") != "qwen38_skyrl_reward_canary_fast3_v1"
        or binding.get("source_proof") != legacy["source_proof"]
        or binding.get("qualification_file_sha256") != fast3["qualification_file_sha256"]
        or binding.get("qualification_self_sha256") != qualification["sha256"]
        or binding.get("image") != qualification["execution"]["image"]
        or binding.get("environment") != qualification["execution"]["environment"]
        or binding.get("cluster_target") != qualification["execution"]["cluster_target"]
        or binding.get("jobs_api_base_url") != qualification["execution"]["jobs_api_base_url"]
        or binding.get("submission_gate") != qualification["submission_gate"]
    ):
        raise ValueError("Fast3 reward-canary plan binding changed")
    return binding
