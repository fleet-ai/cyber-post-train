"""Prepare private Miles prompts from reviewed Fleet task versions. CPU/GET only.

No task selection by score, random holdout, environment creation or optimizer.
All versions of one family stay together; test/reserved tasks are never fetched.
Native prompt filtering must retain every train/dev row, not silently skip one.
"""

from __future__ import annotations

import copy
import json
import os
import re
import uuid
from pathlib import Path

from cyber_post_train.jobs import digest
from evals.fleet import opencode_self_hosted as fleet

from . import miles, rl_episode
from .corpus import local_tokenizer
from .sft import _known, _sfs_root, read_mapping
from .splits import split_key

AUTHORITY = {
    "multi_app_aggregation_mode": "fractional",
    "scoring_mode": "partial",
    "scoring_payload_mode": fleet.RUNTIME_EVIDENCE_ONLY_V3,
    "provisioning_route_template": (
        "/v1/rollout-rewards/{task_key}/versions/{task_version_id}/instances"
    ),
    "scoring_route_template": "/v1/rollout-rewards/{task_key}/versions/{task_version_id}",
    "required_cyber_contract": {
        "evidence_schema": "1.0.0",
        "submission_protocol": "2.0.0",
        "verifier_contract": "3.0.0",
    },
}


def _sealed(value, schema):
    if value.get("schema") != schema or value.get("sha256") != "sha256:" + digest(
        {k: v for k, v in value.items() if k != "sha256"}
    ):
        raise ValueError("reviewed task-set/split digest mismatch")


def selection(task_set: dict, split: dict) -> list[dict]:
    _sealed(task_set, "cyber_rl_task_set_v1")
    _sealed(split, "cyber_task_split_v1")
    if task_set.get("training_data_eligible") is not True:
        raise ValueError("task set is not approved for training")
    fields = {
        "task_key",
        "task_version_id",
        "env_key",
        "env_version",
        "environment_version_id",
        "data_key",
        "data_version",
        "lineage",
    }
    assigned, families, task_families, versions, rows = {}, {}, {}, set(), []
    for row in split["tasks"]:
        key = (row["task_key"], row["task_version_id"])
        if key in assigned or row["split"] not in {"train", "dev", "test", "reserved_dev"}:
            raise ValueError("duplicate or invalid frozen split")
        assigned[key] = row["split"]
    for row in task_set["tasks"]:
        if set(row) != fields or any(
            not isinstance(row[k], str) or not row[k] for k in fields - {"lineage"}
        ):
            raise ValueError("task set needs exact runtime tuple and reviewed family lineage")
        key = (row["task_key"], row["task_version_id"])
        if key in versions or key not in assigned:
            raise ValueError("duplicate task or task absent from frozen split")
        versions.add(key)
        for field in ("task_version_id", "environment_version_id"):
            if str(uuid.UUID(row[field])) != row[field]:
                raise ValueError("task/environment versions must be exact canonical UUIDs")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", row["task_key"]):
            raise ValueError("unsafe Fleet task key")
        family = split_key(row)
        if task_families.setdefault(row["task_key"], family) != family:
            raise ValueError("task family changes across versions")
        group = assigned[key]
        if families.setdefault(family, group) != group:
            raise ValueError("task family crosses train/dev/test partitions")
        if group in {"train", "dev"}:
            rows.append({**row, "split": group})
    if versions != assigned.keys() or {r["split"] for r in rows} != {"train", "dev"}:
        raise ValueError("complete split lineage plus nonempty train/dev sets required")
    return sorted(rows, key=lambda r: (r["split"], r["task_key"], r["task_version_id"]))


def _native(lock, root):
    import torch
    from fti.trainers.miles.run_fleet import TEMPLATES
    from miles.utils.chat_template_utils.tito_tokenizer import get_tito_tokenizer
    from miles.utils.data import Dataset

    if torch.cuda.is_available():
        raise ValueError("RL data preparation must not hold a GPU")
    tokenizer, identity = local_tokenizer(lock, Path(root))
    template = (TEMPLATES / "qwen3.8_fixed.jinja").read_bytes()
    if fleet.sha256(template) != "sha256:" + miles.TEMPLATE_SHA256:
        raise ValueError("native Qwen template drift")
    tokenizer.chat_template = template.decode()
    return tokenizer, get_tito_tokenizer(tokenizer, "qwen35"), Dataset, identity


def build(config: dict, *, relative_to: Path, client) -> dict:
    """Return counts/digests only. The supplied client is authenticated; GET only."""
    _known(
        config,
        {
            "name",
            "task_set",
            "split",
            "tool_catalog",
            "model_lock",
            "model_root",
            "limits",
            "output",
        },
        "RL data",
    )
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,30}", config["name"]):
        raise ValueError("invalid RL run name")
    output = relative_to / config["output"]
    if output.exists() or output.is_symlink():
        raise FileExistsError("create-once RL data destination exists")
    task_set = read_mapping(relative_to / config["task_set"])
    split = read_mapping(relative_to / config["split"])
    selected = selection(task_set, split)
    catalog = json.loads((relative_to / config["tool_catalog"]).read_text())
    if fleet.sha256(fleet.canonical_json(catalog)) != task_set["tool_catalog_sha256"] or (
        not isinstance(catalog, list)
        or [t.get("name") for t in catalog] != ["bash", "submit_report"]
        or any(not isinstance(t.get("inputSchema"), dict) for t in catalog)
    ):
        raise ValueError("tool catalog must match the reviewed exact task tool surface")
    tools = [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t["inputSchema"],
            },
        }
        for t in catalog
    ]
    lock = read_mapping(relative_to / config["model_lock"])
    if lock["repo"] != "Qwen/Qwen3.8-27B":
        raise ValueError("native Miles data profile currently targets Qwen3.8-27B")
    root = _sfs_root(config["model_root"], "model root")
    limits = copy.deepcopy(config["limits"])
    response_tokens = limits.pop("response_tokens")
    episode = {
        "run_id": config["name"],
        "model": {
            "repo": lock["repo"],
            "revision": lock["revision"],
            "root": root,
            "tito_family": "qwen35",
            "runtime_chat_template_sha256": "sha256:" + miles.TEMPLATE_SHA256,
        },
        "authority": AUTHORITY,
        "execution": {
            "required_task_tools": ["bash", "submit_report"],
            "required_task_tool_catalog_sha256": task_set["tool_catalog_sha256"],
        },
        "environment": {"ttl_seconds": 32400},
        "rl": limits,
    }
    episode["config_sha256"] = fleet.digest_without(episode, "config_sha256")
    rl_episode._validate(episode)
    if type(response_tokens) is not int or not (
        limits["max_tokens_per_turn"] <= response_tokens < limits["context_tokens"] <= 98304
    ):
        raise ValueError("response/context budget outside native Qwen profile")
    prompt_budget = limits["context_tokens"] - response_tokens
    tokenizer, tito, Dataset, tokenizer_identity = _native(lock, root)
    account = fleet._request(client, "GET", "/v1/account")
    if account.get("team_id") != fleet.FLEET_TEAM_ID or account.get("team_name") != "fleet":
        raise ValueError("RL data requires Fleet team identity")
    rows, lengths = {"train": [], "dev": []}, {"train": [], "dev": []}
    for selected_task in selected:
        task = fleet._request(
            client,
            "GET",
            f"/v1/tasks/{selected_task['task_key']}",
            params={"version_id": selected_task["task_version_id"]},
        )
        binding, environment, verifier = fleet.bind_task(task, selected_task)
        if binding["cyber_contract"] != AUTHORITY["required_cyber_contract"]:
            raise ValueError("task lacks the reviewed cyber verifier contract")
        frozen = {
            **copy.deepcopy(episode),
            "task": binding,
            "environment": environment,
            "verifier": verifier,
        }
        fleet.verify_task(frozen, task)
        if not isinstance(task.get("prompt"), str) or not task["prompt"].strip():
            raise ValueError("task has no prompt")
        messages = [{"role": "user", "content": task["prompt"]}]
        rendered = tito.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=True
        )
        tokens = list(
            tito.apply_chat_template(
                messages, tools=tools, tokenize=True, add_generation_prompt=True
            )
        )
        if tokenizer.encode(rendered, add_special_tokens=False) != tokens:
            raise ValueError("native dataset rendering differs from recorder prefill tokens")
        if not 0 < len(tokens) <= prompt_budget:
            raise ValueError("task prompt/tools exceed budget; no silent filtering")
        frozen["initial_prompt_sha256"] = fleet.sha256(rendered.encode())
        frozen["config_sha256"] = fleet.digest_without(frozen, "config_sha256")
        row = {
            "input": rendered,
            "metadata": {
                "split": selected_task["split"],
                "lineage": selected_task["lineage"],
                "cyber_config": frozen,
            },
        }
        rows[selected_task["split"]].append(row)
        lengths[selected_task["split"]].append(len(tokens))
    # No output before every task validates. A partial publication is never
    # resumable by rerunning this command; preserve it and use a reviewed new root.
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    files = {}
    for group, values in rows.items():
        path = output / (group + ".jsonl")
        payload = b"".join(fleet.canonical_json(row) + b"\n" for row in values)
        with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        native = Dataset(
            str(path),
            tokenizer,
            None,
            prompt_budget,
            prompt_key="input",
            metadata_key="metadata",
            apply_chat_template=False,
        )
        if len(native) != len(values) or any(
            s.prompt != row["input"] or s.metadata != row["metadata"]
            for s, row in zip(native, values, strict=True)
        ):
            raise ValueError("native Miles dataset dropped or changed a selected task")
        files[group] = {
            "path": path.name,
            "sha256": fleet.sha256(payload),
            "rows": len(values),
            "max_prompt_tokens": max(lengths[group]),
        }
    manifest = {
        "schema": "cyber_miles_data_v1",
        "name": config["name"],
        "selection_sha256": task_set["sha256"],
        "split_sha256": split["sha256"],
        "tokenizer": tokenizer_identity,
        "template_sha256": "sha256:" + miles.TEMPLATE_SHA256,
        "tool_catalog_sha256": task_set["tool_catalog_sha256"],
        "limits": config["limits"],
        "files": files,
        "gpus": 0,
        "environment_creates": 0,
    }
    manifest["sha256"] = "sha256:" + digest(manifest)
    fleet.write_json_once(output / "split.json", split)
    fleet.write_json_once(output / "task-set.json", task_set)
    fleet.write_json_once(output / "manifest.json", manifest)
    return {"manifest_sha256": manifest["sha256"], "files": files, "submitted": False}
