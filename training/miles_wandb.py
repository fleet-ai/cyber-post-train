"""Strict W&B adapter for future comparative Miles studies.

This module is additive: qualified ``miles_training`` bundles do not include it.
It must be installed by a newly qualified runtime before Miles initializes its
tracking manager.  Only a small scalar schema reaches W&B; task text, rewards,
scores, responses, artifacts, and arbitrary native metrics are discarded.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any

from cyber_post_train.jobs import digest

SCHEMA = "cyber_miles_wandb_contract_v2"
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
TAG = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,63}")
TRAIN_KEYS = ("train/step", "train/loss", "train/grad_norm", "train/lr-pg_0")
TOKEN_KEYS = (
    "tokens/rollout_step",
    "tokens/response_tokens",
    "tokens/total_response_tokens",
)
HISTORY_KEYS = TRAIN_KEYS + TOKEN_KEYS
_RAW_TOKEN_KEYS = ("rollout/step", "rollout/episode_response_length/mean")
_contract: dict | None = None


def _identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise ValueError(f"invalid W&B {field}")
    return value


def build_contract(wandb: dict, plan: dict, plan_sha256: str) -> dict:
    """Return the immutable, public comparison contract for one fresh run."""
    if set(wandb) != {"entity", "project", "group", "run_id", "name", "job_type", "tags"}:
        raise ValueError("W&B study fields changed")
    identity = {
        key: _identifier(wandb[key], key)
        for key in ("entity", "project", "group", "run_id", "name", "job_type")
    }
    if identity["group"] in {identity["run_id"], identity["name"]}:
        raise ValueError("comparative W&B group must be stable, not a run identity")
    tags = wandb["tags"]
    if (
        not isinstance(tags, list)
        or not 1 <= len(tags) <= 16
        or tags != sorted(set(tags))
        or any(not isinstance(tag, str) or not TAG.fullmatch(tag) for tag in tags)
    ):
        raise ValueError("W&B tags must be sorted, unique, and safe")
    if not re.fullmatch(r"sha256:[a-f0-9]{64}", plan_sha256):
        raise ValueError("plan digest must be explicit")
    arguments = plan["arguments"]
    public_config = {
        "experiment_plan_sha256": plan_sha256,
        "run_name": plan["run_name"],
        "model_repo": plan["model"]["repo"],
        "model_revision": plan["model"]["revision"],
        "data_sha256": plan["data"]["sha256"],
        "checkpoint_sha256": plan["checkpoint"]["sha256"],
        **{
            key: arguments[key]
            for key in (
                "nodes",
                "gpus_per_node",
                "steps",
                "groups",
                "samples_per_prompt",
                "lr",
                "temperature",
                "kl_loss_coef",
                "seed",
            )
        },
    }
    value = {
        "schema": SCHEMA,
        **identity,
        "tags": tags,
        "resume": "never",
        "history_keys": list(HISTORY_KEYS),
        "axes": {
            "optimizer": "train/step",
            "token_exposure": "tokens/total_response_tokens",
        },
        "public_config": public_config,
    }
    value["sha256"] = "sha256:" + digest(value)
    return value


def request_env(contract: dict) -> dict[str, str]:
    validate_contract(contract)
    return {
        "WANDB_RUN_ID": contract["run_id"],
        "WANDB_NAME": contract["name"],
        "WANDB_RUN_GROUP": contract["group"],
        "WANDB_JOB_TYPE": contract["job_type"],
        "WANDB_TAGS": json.dumps(contract["tags"], separators=(",", ":")),
        "WANDB_RESUME": "never",
        "WANDB_DISABLE_CODE": "true",
        "WANDB_SAVE_CODE": "false",
        "WANDB_DISABLE_GIT": "true",
        "WANDB_CONSOLE": "off",
        "WANDB__DISABLE_STATS": "true",
        "CYBER_EXPERIMENT_PLAN_SHA256": contract["public_config"][
            "experiment_plan_sha256"
        ],
        "CYBER_WANDB_CONTRACT_SHA256": contract["sha256"],
    }


def validate_contract(contract: dict) -> None:
    if contract.get("schema") != SCHEMA or contract.get("sha256") != "sha256:" + digest(
        {key: value for key, value in contract.items() if key != "sha256"}
    ):
        raise ValueError("W&B contract digest mismatch")
    rebuilt = build_contract(
        {
            key: contract[key]
            for key in ("entity", "project", "group", "run_id", "name", "job_type", "tags")
        },
        {
            "run_name": contract["public_config"]["run_name"],
            "model": {
                "repo": contract["public_config"]["model_repo"],
                "revision": contract["public_config"]["model_revision"],
            },
            "data": {"sha256": contract["public_config"]["data_sha256"]},
            "checkpoint": {"sha256": contract["public_config"]["checkpoint_sha256"]},
            "arguments": {
                key: contract["public_config"][key]
                for key in (
                    "nodes",
                    "gpus_per_node",
                    "steps",
                    "groups",
                    "samples_per_prompt",
                    "lr",
                    "temperature",
                    "kl_loss_coef",
                    "seed",
                )
            },
        },
        contract["public_config"]["experiment_plan_sha256"],
    )
    if rebuilt != contract:
        raise ValueError("W&B contract fields changed")


def validate_environment(contract: dict) -> None:
    expected = request_env(contract)
    if os.environ.get("WANDB_RESUME") != "never" or any(
        os.environ.get(name) != expected[name] for name in expected
    ):
        raise ValueError("W&B process environment differs from immutable contract")


def install(contract: dict) -> None:
    """Install exact native tracking hooks before Miles creates a backend."""
    global _contract
    validate_contract(contract)
    validate_environment(contract)
    _contract = contract
    from miles.utils.tracking_utils.base import WandbBackend

    WandbBackend.init = _init
    WandbBackend.log = _log
    WandbBackend.finish = _finish
    WandbBackend.define_step_key_metric_group = _ignore_native_definitions


def install_worker() -> None:
    """Ray worker hook; the complete public contract is passed as safe JSON."""
    raw = os.environ.get("CYBER_WANDB_CONTRACT_JSON")
    if not raw:
        raise ValueError("worker W&B contract is absent")
    install(json.loads(raw))


def worker_env(contract: dict) -> dict[str, str]:
    value = request_env(contract)
    value["CYBER_WANDB_CONTRACT_JSON"] = json.dumps(
        contract, sort_keys=True, separators=(",", ":")
    )
    return value


def _settings(wandb, primary: bool):
    return wandb.Settings(
        init_timeout=300.0,
        mode="shared",
        x_primary=primary,
        x_disable_stats=True,
        **({} if primary else {"x_update_finish_state": False}),
    )


def _init(self, args, *, primary: bool = True, **_kwargs) -> None:
    if _contract is None:
        raise ValueError("W&B comparison contract was not installed")
    import wandb

    kwargs = {
        "id": _contract["run_id"],
        "entity": _contract["entity"],
        "project": _contract["project"],
        "group": _contract["group"],
        "name": _contract["name"],
        "job_type": _contract["job_type"],
        "tags": _contract["tags"],
        "config": {
            **_contract["public_config"],
            "wandb_contract_sha256": _contract["sha256"],
        },
        "resume": "never" if primary else "allow",
        "reinit": not primary,
        "settings": _settings(wandb, primary),
    }
    if args.wandb_dir:
        Path(args.wandb_dir).mkdir(parents=True, exist_ok=True)
        kwargs["dir"] = args.wandb_dir
    run = wandb.init(**kwargs)
    observed = {
        "run_id": run.id,
        "name": run.name,
        "group": run.group,
        "job_type": run.job_type,
        "tags": sorted(run.tags),
        "entity": run.entity,
        "project": run.project,
    }
    expected = {
        key: _contract[key]
        for key in ("run_id", "name", "group", "job_type", "tags", "entity", "project")
    }
    if observed != expected:
        raise ValueError("W&B identity readback differs from immutable contract")
    args.wandb_run_id = run.id
    self._cyber_primary = primary
    self._cyber_total_response_tokens = 0
    self._cyber_global_batch_size = args.global_batch_size
    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("tokens/total_response_tokens")
    wandb.define_metric("tokens/*", step_metric="tokens/total_response_tokens")
    if primary:
        _write_receipt(Path(args.wandb_dir).parent)


def _scalar(value: Any) -> float | int:
    if hasattr(value, "item"):
        value = value.item()
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("W&B public history accepts finite scalars only")
    return value


def _log(self, metrics: dict[str, Any], step: int | None = None, **_kwargs) -> None:
    del step
    import wandb

    clean = {key: _scalar(metrics[key]) for key in TRAIN_KEYS if key in metrics}
    if all(key in metrics for key in _RAW_TOKEN_KEYS):
        rollout_step = _scalar(metrics[_RAW_TOKEN_KEYS[0]])
        mean = _scalar(metrics[_RAW_TOKEN_KEYS[1]])
        batch = _scalar(self._cyber_global_batch_size)
        response_tokens = mean * batch
        rounded = round(response_tokens)
        if response_tokens < 0 or not math.isclose(response_tokens, rounded, abs_tol=1e-6):
            raise ValueError("derived response-token exposure is invalid")
        response_tokens = int(rounded)
        self._cyber_total_response_tokens += response_tokens
        clean.update(
            {
                "tokens/rollout_step": rollout_step,
                "tokens/response_tokens": response_tokens,
                "tokens/total_response_tokens": self._cyber_total_response_tokens,
            }
        )
    if clean:
        if not set(clean).issubset(HISTORY_KEYS):
            raise ValueError("W&B history allowlist drift")
        wandb.log(clean)


def _ignore_native_definitions(self, prefix: str, step_key: str) -> None:
    del self, prefix, step_key


def _finish(self) -> None:
    del self
    import wandb

    wandb.finish()


def _write_receipt(root: Path) -> None:
    if _contract is None:
        raise ValueError("W&B contract absent")
    value = {
        "schema": "cyber_miles_wandb_initialized_v2",
        "status": "initialized",
        "contract_sha256": _contract["sha256"],
        "experiment_plan_sha256": _contract["public_config"]["experiment_plan_sha256"],
        "identity": {
            key: _contract[key]
            for key in ("entity", "project", "group", "run_id", "name", "job_type", "tags")
        },
        "resume": "never",
        "history_keys": list(HISTORY_KEYS),
        "axes": _contract["axes"],
    }
    value["sha256"] = "sha256:" + digest(value)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "WANDB_CONTRACT.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, separators=(",", ":"))
