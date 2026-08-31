"""Freeze Fleet task versions and build a secret-free multi-environment RL config."""

from __future__ import annotations

import http.client
import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .io import atomic_write_json, digest_json, iter_jsonl

SPLITS = ("train", "dev", "test")
SAFE_TASK_FIELDS = (
    "task_key",
    "task_version_id",
    "task_version",
    "environment_version_id",
    "env_key",
    "env_version",
    "data_key",
    "data_version",
)


class TaskCatalogError(ValueError):
    """Raised when the authoritative task catalog contradicts the frozen dataset."""


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise TaskCatalogError(f"missing {field}")
    return value


def split_bindings(trajectories: Iterable[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Reduce private trajectories to one safe, exact binding per task version."""
    observed: dict[str, dict[str, str]] = {}
    for row in trajectories:
        lineage = row.get("lineage")
        environment = row.get("environment")
        if not isinstance(lineage, Mapping) or not isinstance(environment, Mapping):
            raise TaskCatalogError("trajectory omitted lineage or environment")
        task_version_id = _required_string(
            lineage.get("eval_task_version_id"), "lineage.eval_task_version_id"
        )
        split = _required_string(row.get("split"), "split")
        if split not in SPLITS:
            raise TaskCatalogError(f"unsupported split {split!r}")
        binding = {
            "task_key": _required_string(lineage.get("task_key"), "lineage.task_key"),
            "task_version_id": task_version_id,
            "task_version": _required_string(lineage.get("task_version"), "lineage.task_version"),
            "split": split,
            "env_key": _required_string(environment.get("env_key"), "environment.env_key"),
            "env_version": _required_string(
                environment.get("version"), "environment.version"
            ),
            "data_key": _required_string(
                environment.get("data_key"), "environment.data_key"
            ),
            "data_version": _required_string(
                environment.get("data_version"), "environment.data_version"
            ),
        }
        previous = observed.setdefault(task_version_id, binding)
        if previous != binding:
            raise TaskCatalogError(f"contradictory trajectory binding for {task_version_id}")
    return sorted(observed.values(), key=lambda item: (item["split"], item["task_key"]))


def _safe_catalog_match(binding: Mapping[str, str], options: Iterable[Mapping[str, Any]]) -> dict:
    task_version_id = binding["task_version_id"]
    matches = [option for option in options if option.get("task_version_id") == task_version_id]
    if len(matches) != 1:
        raise TaskCatalogError(
            f"expected one catalog match for {binding['task_key']}@{task_version_id}, "
            f"found {len(matches)}"
        )
    match = matches[0]
    comparisons = {
        "key": "task_key",
        "env_key": "env_key",
        "env_version": "env_version",
        "data_key": "data_key",
        "data_version": "data_version",
    }
    for catalog_field, binding_field in comparisons.items():
        if match.get(catalog_field) != binding[binding_field]:
            raise TaskCatalogError(
                f"catalog {catalog_field} contradicts frozen data for {task_version_id}"
            )
    safe = {
        "task_key": _required_string(match.get("key"), "catalog.key"),
        "task_version_id": task_version_id,
        "task_version": _required_string(match.get("task_version"), "catalog.task_version"),
        "environment_version_id": _required_string(
            match.get("environment_version_id"), "catalog.environment_version_id"
        ),
        "env_key": binding["env_key"],
        "env_version": binding["env_version"],
        "data_key": binding["data_key"],
        "data_version": binding["data_version"],
        "split": binding["split"],
        "resolution_authority": "training_task_catalog",
    }
    # Deliberately do not copy catalog env_variables, image URLs, prompts, or verifier data.
    return safe


def _safe_roster_environment_match(
    binding: Mapping[str, str], environment_versions: Iterable[Mapping[str, Any]]
) -> dict[str, str]:
    """Resolve an archived roster task whose task-picker row is no longer listed."""
    matches = [
        version
        for version in environment_versions
        if version.get("env_key") == binding["env_key"]
        and version.get("version") == binding["env_version"]
        and version.get("data_key") == binding["data_key"]
        and version.get("data_version") == binding["data_version"]
    ]
    if len(matches) != 1:
        raise TaskCatalogError(
            f"expected one environment catalog match for {binding['task_key']}, "
            f"found {len(matches)}"
        )
    environment_version_id = _required_string(
        matches[0].get("id"), "environment_catalog.id"
    )
    return {
        "task_key": binding["task_key"],
        "task_version_id": binding["task_version_id"],
        "task_version": binding["task_version"],
        "environment_version_id": environment_version_id,
        "env_key": binding["env_key"],
        "env_version": binding["env_version"],
        "data_key": binding["data_key"],
        "data_version": binding["data_version"],
        "split": binding["split"],
        "resolution_authority": "fleet_job_roster+training_environment_catalog",
    }


def freeze_task_split(
    trajectories: Iterable[Mapping[str, Any]],
    fetch_options: Callable[[str], Iterable[Mapping[str, Any]]],
    *,
    source_job_id: str,
    dataset_manifest_digest: str,
    workers: int = 8,
    fetch_environment_versions: Callable[[str], Iterable[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Resolve exact task/environment IDs and emit only a safe split lock."""
    if workers < 1 or workers > 32:
        raise ValueError("workers must be in 1..32")
    bindings = split_bindings(trajectories)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        option_sets = executor.map(lambda item: fetch_options(item["task_key"]), bindings)
        tasks = []
        for binding, options in zip(bindings, option_sets, strict=True):
            options = list(options)
            if any(
                option.get("task_version_id") == binding["task_version_id"]
                for option in options
            ):
                tasks.append(_safe_catalog_match(binding, options))
            elif fetch_environment_versions is not None:
                tasks.append(
                    _safe_roster_environment_match(
                        binding, fetch_environment_versions(binding["env_key"])
                    )
                )
            else:
                tasks.append(_safe_catalog_match(binding, options))
    counts = dict(sorted(Counter(item["split"] for item in tasks).items()))
    snapshot: dict[str, Any] = {
        "schema": "fleet_rl_task_split_v1",
        "source": {
            "job_id": source_job_id,
            "dataset_manifest_digest": dataset_manifest_digest,
        },
        "policy": {
            "split_seed": "fleet-cyber-split-v1",
            "split_unit": "application + vulnerability family + exact task lineage",
            "catalog_authority": "https://ft.flt.build/api/v1/tasks",
            "archived_task_authority": "exact source-job roster + environment catalog",
            "secret_fields_retained": [],
        },
        "counts": {"tasks": len(tasks), "splits": counts},
        "tasks": sorted(tasks, key=lambda item: (item["split"], item["task_key"])),
    }
    snapshot["manifest_digest"] = digest_json(snapshot)
    return snapshot


def verify_task_split(snapshot: Mapping[str, Any]) -> None:
    if snapshot.get("schema") != "fleet_rl_task_split_v1":
        raise TaskCatalogError("unsupported task split schema")
    unsigned = {key: value for key, value in snapshot.items() if key != "manifest_digest"}
    if snapshot.get("manifest_digest") != digest_json(unsigned):
        raise TaskCatalogError("task split manifest digest mismatch")
    tasks = snapshot.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise TaskCatalogError("task split has no tasks")
    ids: set[str] = set()
    keys: set[str] = set()
    counts: Counter[str] = Counter()
    for task in tasks:
        if not isinstance(task, Mapping):
            raise TaskCatalogError("task split contains a non-object task")
        unexpected = set(task) - {*SAFE_TASK_FIELDS, "split", "resolution_authority"}
        if unexpected:
            raise TaskCatalogError(f"unsafe or unexpected task fields: {sorted(unexpected)}")
        for field in SAFE_TASK_FIELDS:
            _required_string(task.get(field), field)
        _required_string(task.get("resolution_authority"), "resolution_authority")
        split = _required_string(task.get("split"), "split")
        if split not in SPLITS:
            raise TaskCatalogError(f"unsupported split {split!r}")
        counts[split] += 1
        ids.add(str(task["task_version_id"]))
        keys.add(str(task["task_key"]))
    if len(ids) != len(tasks) or len(keys) != len(tasks):
        raise TaskCatalogError("task split contains duplicate task versions or keys")
    declared = snapshot.get("counts")
    actual = {"tasks": len(tasks), "splits": dict(sorted(counts.items()))}
    if declared != actual:
        raise TaskCatalogError("task split counts do not match tasks")


def build_treatment_set(
    snapshot: Mapping[str, Any], exclusions: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    """Record an explicit as-treated set when an intent-to-treat task cannot run."""
    verify_task_split(snapshot)
    by_id = {str(task["task_version_id"]): task for task in snapshot["tasks"]}
    excluded: list[dict[str, str]] = []
    seen: set[str] = set()
    for exclusion in exclusions:
        task_version_id = _required_string(
            exclusion.get("task_version_id"), "exclusion.task_version_id"
        )
        reason = _required_string(exclusion.get("reason"), "exclusion.reason")
        evidence = _required_string(exclusion.get("evidence"), "exclusion.evidence")
        if task_version_id in seen:
            raise TaskCatalogError(f"duplicate treatment exclusion {task_version_id}")
        seen.add(task_version_id)
        task = by_id.get(task_version_id)
        if task is None:
            raise TaskCatalogError(
                f"treatment exclusion is not in the frozen split: {task_version_id}"
            )
        if task["split"] != "train":
            raise TaskCatalogError("only train tasks may be excluded from the treatment set")
        excluded.append(
            {
                "task_key": str(task["task_key"]),
                "task_version_id": task_version_id,
                "reason": reason,
                "evidence": evidence,
            }
        )
    train_count = sum(task["split"] == "train" for task in snapshot["tasks"])
    treatment: dict[str, Any] = {
        "schema": "fleet_rl_treatment_set_v1",
        "intent_to_treat_manifest_digest": snapshot["manifest_digest"],
        "counts": {
            "intent_to_treat_train_tasks": train_count,
            "as_treated_train_tasks": train_count - len(excluded),
            "excluded_train_tasks": len(excluded),
        },
        "exclusions": sorted(excluded, key=lambda item: item["task_version_id"]),
    }
    treatment["manifest_digest"] = digest_json(treatment)
    return treatment


def verify_treatment_set(
    treatment: Mapping[str, Any], snapshot: Mapping[str, Any]
) -> set[str]:
    verify_task_split(snapshot)
    if treatment.get("schema") != "fleet_rl_treatment_set_v1":
        raise TaskCatalogError("unsupported treatment-set schema")
    unsigned = {key: value for key, value in treatment.items() if key != "manifest_digest"}
    if treatment.get("manifest_digest") != digest_json(unsigned):
        raise TaskCatalogError("treatment-set manifest digest mismatch")
    if treatment.get("intent_to_treat_manifest_digest") != snapshot["manifest_digest"]:
        raise TaskCatalogError("treatment set does not bind the frozen task split")
    exclusions = treatment.get("exclusions")
    if not isinstance(exclusions, list):
        raise TaskCatalogError("treatment-set exclusions must be a list")
    rebuilt = build_treatment_set(snapshot, exclusions)
    if rebuilt != treatment:
        raise TaskCatalogError("treatment-set contents or counts are inconsistent")
    return {str(item["task_version_id"]) for item in exclusions}


def _runtime_task(task: Mapping[str, Any]) -> dict[str, Any]:
    runtime = {field: task[field] for field in SAFE_TASK_FIELDS}
    runtime["env_variables"] = {}
    runtime["env_variable_deletions"] = []
    return runtime


def build_full_rl_config(
    template: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    treatment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind the train/dev partitions to a typed Fleet RL request."""
    verify_task_split(snapshot)
    config = json.loads(json.dumps(template))
    tasks = config.get("tasks")
    eval_spec = config.get("eval")
    if not isinstance(tasks, dict) or not isinstance(eval_spec, dict):
        raise TaskCatalogError("RL template must contain tasks and eval objects")
    if tasks.get("selection_mode") != "tasks":
        raise TaskCatalogError("full multi-environment RL requires selection_mode=tasks")
    if config.get("fleet_env") is not None:
        raise TaskCatalogError("multi-environment task-first RL requires fleet_env=null")

    excluded = verify_treatment_set(treatment, snapshot) if treatment is not None else set()
    if treatment is not None:
        counts = treatment["counts"]
        config["title"] = (
            f"{config.get('title', 'Fleet RL run')} "
            f"({counts['as_treated_train_tasks']} runnable; "
            f"{counts['excluded_train_tasks']} archived exclusion)"
        )
    locked_tasks = snapshot["tasks"]
    tasks["task_keys"] = []
    tasks["task_versions"] = [
        _runtime_task(task)
        for task in locked_tasks
        if task["split"] == "train" and task["task_version_id"] not in excluded
    ]
    eval_spec["task_keys"] = []
    eval_spec["task_versions"] = [
        _runtime_task(task) for task in locked_tasks if task["split"] == "dev"
    ]
    if not tasks["task_versions"] or not eval_spec["task_versions"]:
        raise TaskCatalogError("both train and dev task partitions are required")
    return config


class FleetTrainingTaskCatalog:
    """Read-only client for the training API's authoritative task picker."""

    def __init__(self, bearer: str, *, base_url: str = "https://api.ft.flt.build"):
        if not bearer:
            raise ValueError("a Fleet Training API bearer is required")
        self._bearer = bearer
        self.base_url = base_url.rstrip("/")

    def options(self, task_key: str) -> list[dict[str, Any]]:
        query = urllib.parse.urlencode({"task_key": task_key})
        request = urllib.request.Request(
            f"{self.base_url}/v1/tasks?{query}",
            headers={"Accept": "application/json", "Authorization": f"Bearer {self._bearer}"},
        )
        last_error: BaseException | None = None
        for attempt in range(5):
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    value = json.load(response)
                break
            except (http.client.HTTPException, OSError, ValueError, urllib.error.URLError) as exc:
                last_error = exc
                if attempt == 4:
                    raise TaskCatalogError(
                        f"Fleet task catalog failed after retries for {task_key}: "
                        f"{type(last_error).__name__}"
                    ) from exc
                time.sleep(min(8.0, 0.5 * 2**attempt) + random.random() * 0.2)
        else:  # pragma: no cover - the final retry always raises
            raise TaskCatalogError(f"Fleet task catalog failed for {task_key}")
        if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
            raise TaskCatalogError("Fleet task catalog returned a non-list payload")
        return value

    def environment_versions(self, env_key: str) -> list[dict[str, Any]]:
        query = urllib.parse.urlencode({"env_key": env_key})
        request = urllib.request.Request(
            f"{self.base_url}/v1/rl/run/env-versions?{query}",
            headers={"Accept": "application/json", "Authorization": f"Bearer {self._bearer}"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            value = json.load(response)
        versions = value.get("versions") if isinstance(value, dict) else None
        if not isinstance(versions, list) or not all(isinstance(item, dict) for item in versions):
            raise TaskCatalogError("Fleet environment catalog returned a malformed payload")
        return versions


def write_task_snapshot(path: Path, snapshot: Mapping[str, Any]) -> None:
    atomic_write_json(path, snapshot)


def write_full_rl_config(path: Path, config: Mapping[str, Any]) -> None:
    atomic_write_json(path, config)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TaskCatalogError(f"{path} must contain a JSON object")
    return value


def read_trajectories(path: Path) -> Iterable[dict[str, Any]]:
    return iter_jsonl(path)
