"""Small resumable controller above benchmark-specific evaluation drivers."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

SCHEMA = "cyber_eval_campaign_v1"
RECEIPT_SCHEMA = "cyber_eval_campaign_driver_receipt_v1"
PHASES = ("rollout", "score")
ACTIONS = ("preview", "ready", "launch", "observe")
TERMINAL = {"accepted", "infrastructure_invalid", "failed"}
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{1,127}")
_PLACEHOLDERS = {
    "packet",
    "receipt",
    "preview_receipt",
    "readiness_receipt",
    "launch_receipt",
    "terminal_receipt",
}
CANONICAL_REGISTRY = Path.home() / ".local/state/cyber-post-train/evaluation-registry-v1"
CONFIG_KEYS = set(
    [
        "schema",
        "campaign_id",
        "scheduler",
        "pass_k",
        "budgets_sha256",
        "matrix_sha256",
        "models",
        "benchmarks",
    ]
)
MODEL_KEYS = set(
    [
        "id",
        "checkpoint_id",
        "weights_sha256",
        "matched_treatment_receipt_sha256",
        "serving_route_receipt_sha256",
        "live_parity_receipt_sha256",
    ]
)
BENCHMARK_KEYS = set(
    [
        "id",
        "task_set_sha256",
        "harness",
        "scoring_protocol_sha256",
        "sampling",
        "targets",
        "rollout_driver",
        "score_driver",
    ]
)
HARNESS_KEYS = {"name", "identity_receipt_sha256"}
TARGET_KEYS = {"id", "identity_receipt_sha256", "canary"}


class CampaignError(ValueError):
    """The campaign or driver evidence is incomplete or inconsistent."""


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical(value)).hexdigest()


def _write_once(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CampaignError("invalid_json_file")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CampaignError("invalid_json_file") from exc
    if not isinstance(value, dict) or path.read_bytes() != raw:
        raise CampaignError("invalid_json_file")
    return value


def _exact_id(value: object, label: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise CampaignError(f"invalid_{label}")
    return value


def _exact_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise CampaignError(f"invalid_{label}")
    return value


def _mapping(value: object, keys: set[str], error: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise CampaignError(error)
    return value


def _digests(value: dict[str, Any], fields: tuple[str, ...]) -> None:
    for field in fields:
        _exact_digest(value[field], field)


def _validate_sampling(value: object) -> dict[str, Any]:
    sampling = _mapping(value, {"temperature", "top_p", "attempt_seeds"}, "invalid_sampling")
    seeds = sampling["attempt_seeds"]
    if (
        not isinstance(seeds, list)
        or len(seeds) != 4
        or any(
            seed is not None and (type(seed) is not int or not 0 <= seed < 2**31) for seed in seeds
        )
        or (any(seed is None for seed in seeds) and any(seed is not None for seed in seeds))
        or len({seed for seed in seeds if seed is not None})
        != len([seed for seed in seeds if seed is not None])
        or type(sampling["temperature"]) not in (int, float)
        or type(sampling["top_p"]) not in (int, float)
    ):
        raise CampaignError("invalid_sampling")
    return sampling


def _validate_driver(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "provider",
        "submission",
        "source_sha256",
        "timeout_seconds",
        "commands",
    }:
        raise CampaignError(f"invalid_{label}_driver")
    provider, submission = value["provider"], value["submission"]
    if provider not in {"fleet", "tensorlake", "local"}:
        raise CampaignError(f"invalid_{label}_provider")
    if submission not in {"jobs_api", "kubernetes", "none"} or (
        provider == "fleet" and submission == "none"
    ):
        raise CampaignError(f"invalid_{label}_submission")
    if provider != "fleet" and submission != "none":
        raise CampaignError(f"invalid_{label}_submission")
    _exact_digest(value["source_sha256"], f"{label}_source_sha256")
    timeout = value["timeout_seconds"]
    commands = value["commands"]
    if type(timeout) is not int or not 1 <= timeout <= 600 or not isinstance(commands, dict):
        raise CampaignError(f"invalid_{label}_driver")
    if set(commands) != set(ACTIONS):
        raise CampaignError(f"invalid_{label}_commands")
    required = {
        "preview": {"packet", "receipt"},
        "ready": {"packet", "receipt", "preview_receipt"},
        "launch": {"packet", "receipt", "preview_receipt", "readiness_receipt"},
        "observe": {"packet", "receipt", "launch_receipt"},
    }
    if label == "score":
        required["preview"].add("terminal_receipt")
    for action, argv in commands.items():
        if not isinstance(argv, list) or not argv or any(not isinstance(arg, str) for arg in argv):
            raise CampaignError(f"invalid_{label}_{action}_command")
        fields: set[str] = set()
        for arg in argv:
            fields.update(re.findall(r"\{([a-z_]+)\}", arg))
        if not required[action] <= fields or not fields <= _PLACEHOLDERS:
            raise CampaignError(f"invalid_{label}_{action}_command")
    return value


def _validate_config(value: dict[str, Any]) -> dict[str, Any]:
    if set(value) != CONFIG_KEYS or value["schema"] != SCHEMA or value["pass_k"] != 4:
        raise CampaignError("invalid_campaign")
    _exact_id(value["campaign_id"], "campaign_id")
    _digests(value, ("budgets_sha256", "matrix_sha256"))
    scheduler = _mapping(
        value["scheduler"],
        {"max_launches_per_step", "serial_canaries"},
        "invalid_campaign_scheduler",
    )
    if (
        type(scheduler["max_launches_per_step"]) is not int
        or not 1 <= scheduler["max_launches_per_step"] <= 100
        or type(scheduler["serial_canaries"]) is not bool
    ):
        raise CampaignError("invalid_campaign_scheduler")
    models = value["models"]
    if not isinstance(models, list) or len(models) < 2:
        raise CampaignError("matched_models_required")
    model_ids: set[str] = set()
    parity: set[tuple[object, ...]] = set()
    for model in models:
        model = _mapping(model, MODEL_KEYS, "invalid_model")
        model_id = _exact_id(model["id"], "model_id")
        if (
            model_id in model_ids
            or not isinstance(model["checkpoint_id"], str)
            or not model["checkpoint_id"]
        ):
            raise CampaignError("invalid_model")
        model_ids.add(model_id)
        _digests(
            model,
            (
                "weights_sha256",
                "matched_treatment_receipt_sha256",
                "serving_route_receipt_sha256",
                "live_parity_receipt_sha256",
            ),
        )
        parity.add((model["matched_treatment_receipt_sha256"],))
    if len(parity) != 1:
        raise CampaignError("model_parity_mismatch")
    benchmarks = value["benchmarks"]
    if not isinstance(benchmarks, list) or not benchmarks:
        raise CampaignError("benchmarks_required")
    benchmark_ids: set[str] = set()
    for benchmark in benchmarks:
        benchmark = _mapping(benchmark, BENCHMARK_KEYS, "invalid_benchmark")
        benchmark_id = _exact_id(benchmark["id"], "benchmark_id")
        if benchmark_id in benchmark_ids:
            raise CampaignError("duplicate_benchmark_id")
        benchmark_ids.add(benchmark_id)
        _digests(benchmark, ("task_set_sha256", "scoring_protocol_sha256"))
        _validate_sampling(benchmark["sampling"])
        harness = _mapping(benchmark["harness"], HARNESS_KEYS, "invalid_harness")
        _exact_id(harness["name"], "harness_name")
        _digests(harness, ("identity_receipt_sha256",))
        targets = benchmark["targets"]
        if not isinstance(targets, list) or not targets:
            raise CampaignError("benchmark_targets_required")
        target_ids: set[str] = set()
        for target in targets:
            target = _mapping(target, TARGET_KEYS, "invalid_benchmark_target")
            target_id = _exact_id(target["id"], "target_id")
            if target_id in target_ids:
                raise CampaignError("duplicate_benchmark_target")
            target_ids.add(target_id)
            if type(target["canary"]) is not bool:
                raise CampaignError("invalid_benchmark_target")
            _digests(target, ("identity_receipt_sha256",))
        _validate_driver(benchmark["rollout_driver"], "rollout")
        _validate_driver(benchmark["score_driver"], "score")
    return value


def build_plan(config: dict[str, Any]) -> dict[str, Any]:
    """Validate and deterministically expand models × targets × four attempts."""
    _validate_config(config)
    queues: list[list[dict[str, Any]]] = []
    for benchmark in sorted(config["benchmarks"], key=lambda item: item["id"]):
        queue: list[dict[str, Any]] = []
        benchmark_identity = {
            key: benchmark[key]
            for key in ("id", "task_set_sha256", "harness", "scoring_protocol_sha256")
        }
        benchmark_targets = sorted(
            benchmark["targets"], key=lambda item: (not item["canary"], item["id"])
        )
        models = sorted(config["models"], key=lambda item: item["id"])
        for attempt in range(1, 5):
            for target_index, target in enumerate(benchmark_targets):
                offset = (target_index + attempt - 1) % len(models)
                for model in models[offset:] + models[:offset]:
                    statistical_target = {
                        key: target[key] for key in ("id", "identity_receipt_sha256")
                    }
                    statistical_model = {
                        key: model[key]
                        for key in (
                            "id",
                            "checkpoint_id",
                            "weights_sha256",
                            "matched_treatment_receipt_sha256",
                        )
                    }
                    identity = {
                        "model": statistical_model,
                        "benchmark": benchmark_identity,
                        "target": statistical_target,
                        "attempt": attempt,
                        "seed": benchmark["sampling"]["attempt_seeds"][attempt - 1],
                        "sampling": {
                            key: benchmark["sampling"][key] for key in ("temperature", "top_p")
                        },
                        "budgets_sha256": config["budgets_sha256"],
                    }
                    queue.append(
                        {
                            "experiment_key": digest(identity),
                            "identity": identity,
                            "matrix_sha256": config["matrix_sha256"],
                            "serving_evidence": {
                                "serving_route_receipt_sha256": model[
                                    "serving_route_receipt_sha256"
                                ],
                                "live_parity_receipt_sha256": model["live_parity_receipt_sha256"],
                            },
                            "canary": target["canary"],
                            "drivers": {
                                "rollout": benchmark["rollout_driver"],
                                "score": benchmark["score_driver"],
                            },
                        }
                    )
        queues.append(queue)
    targets: list[dict[str, Any]] = []
    while any(queues):
        for queue in queues:
            if queue:
                targets.append(queue.pop(0))
    keys = [target["experiment_key"] for target in targets]
    if len(keys) != len(set(keys)):
        raise CampaignError("duplicate_experiment_key")
    unsigned = {
        "schema": SCHEMA,
        "campaign_id": config["campaign_id"],
        "matrix_sha256": config["matrix_sha256"],
        "registry_path": str(CANONICAL_REGISTRY),
        "scheduler": config["scheduler"],
        "pass_k": 4,
        "targets": targets,
    }
    return {**unsigned, "plan_sha256": digest(unsigned)}


def prepare(config_path: Path, state: Path) -> dict[str, Any]:
    plan = build_plan(_read(config_path))
    state.mkdir(mode=0o700, parents=True, exist_ok=False)
    _write_once(state / "plan.json", plan)
    for target in plan["targets"]:
        _write_once(state / "targets" / target["experiment_key"] / "packet.json", target)
    return plan


def load_plan(state: Path) -> dict[str, Any]:
    value = _read(state / "plan.json")
    unsigned = {key: item for key, item in value.items() if key != "plan_sha256"}
    if (
        value.get("schema") != SCHEMA
        or value.get("plan_sha256") != digest(unsigned)
        or value.get("registry_path") != str(CANONICAL_REGISTRY)
    ):
        raise CampaignError("invalid_campaign_plan")
    for target in value.get("targets", []):
        if _read(state / "targets" / target["experiment_key"] / "packet.json") != target:
            raise CampaignError("campaign_target_changed")
    return value


def _receipt_path(state: Path, key: str, phase: str, action: str) -> Path:
    return state / "targets" / key / phase / f"{action}.json"


def _validate_fleet_preview(receipt: dict[str, Any], driver: dict[str, Any]) -> None:
    objects = receipt.get("rendered_objects")
    if not isinstance(objects, list):
        raise CampaignError("fleet_preview_missing_rendered_objects")
    roots = [
        item for item in objects if isinstance(item, dict) and item.get("kind") in {"Job", "RayJob"}
    ]
    if not roots or any(
        item.get("metadata", {}).get("annotations", {}).get("fleet.ai/failure-alerts") != "off"
        for item in roots
    ):
        raise CampaignError("fleet_preview_failure_alerts_not_off")
    if (
        driver["submission"] == "jobs_api"
        and receipt.get("request", {}).get("failureAlerts") is not False
    ):
        raise CampaignError("fleet_jobs_api_failure_alerts_not_off")


def _validate_tensorlake_preflight(receipt: dict[str, Any]) -> dict[str, Any]:
    gate = receipt.get("provider_preflight")
    if not isinstance(gate, dict) or set(gate) != {
        "shared_capacity_receipt_sha256",
        "provider_inventory_receipt_sha256",
        "create_claim_absent",
        "start_claim_absent",
        "remote_name_absent",
        "output_root_absent",
        "checked_at_epoch",
    }:
        raise CampaignError("tensorlake_provider_preflight_missing")
    _exact_digest(gate["shared_capacity_receipt_sha256"], "shared_capacity_receipt_sha256")
    _exact_digest(gate["provider_inventory_receipt_sha256"], "provider_inventory_sha256")
    if (
        gate["create_claim_absent"] is not True
        or gate["start_claim_absent"] is not True
        or gate["remote_name_absent"] is not True
        or gate["output_root_absent"] is not True
        or type(gate["checked_at_epoch"]) not in (int, float)
        or gate["checked_at_epoch"] <= 0
    ):
        raise CampaignError("tensorlake_provider_preflight_failed")
    return gate


def validate_receipt(
    receipt: dict[str, Any],
    target: dict[str, Any],
    phase: str,
    action: str,
    state: Path,
) -> dict[str, Any]:
    driver = target["drivers"][phase]
    unsigned = {key: item for key, item in receipt.items() if key != "receipt_sha256"}
    expected_status = {
        "preview": {"accepted"},
        "ready": {"ready", "deferred_not_ready"},
        "launch": {"created"},
        "observe": {"running", *TERMINAL},
    }
    if (
        receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("receipt_sha256") != digest(unsigned)
        or receipt.get("experiment_key") != target["experiment_key"]
        or receipt.get("phase") != phase
        or receipt.get("action") != action
        or receipt.get("provider") != driver["provider"]
        or receipt.get("status") not in expected_status[action]
    ):
        raise CampaignError("driver_receipt_identity_mismatch")
    if action == "preview" and driver["provider"] == "fleet":
        _validate_fleet_preview(receipt, driver)
    if action in {"preview", "ready", "launch"} and driver["provider"] == "tensorlake":
        gate = _validate_tensorlake_preflight(receipt)
        remote_name = receipt.get("remote_name")
        key_fragment = target["experiment_key"].removeprefix("sha256:")[:12]
        if not isinstance(remote_name, str) or key_fragment not in remote_name:
            raise CampaignError("tensorlake_remote_name_not_attempt_unique")
        if action == "launch":
            preview = _read(_receipt_path(state, target["experiment_key"], phase, "preview"))
            previous = _validate_tensorlake_preflight(preview)
            if (
                receipt["remote_name"] != preview["remote_name"]
                or gate["checked_at_epoch"] < previous["checked_at_epoch"]
            ):
                raise CampaignError("tensorlake_precreate_revalidation_stale")
    if phase == "score":
        collection = _read(_receipt_path(state, target["experiment_key"], "rollout", "observe"))
        if collection.get("status") != "accepted" or receipt.get(
            "collection_terminal_receipt_sha256"
        ) != collection.get("receipt_sha256"):
            raise CampaignError("score_collection_binding_mismatch")
    if action == "ready":
        preview = _read(_receipt_path(state, target["experiment_key"], phase, "preview"))
        if receipt.get("preview_receipt_sha256") != preview["receipt_sha256"]:
            raise CampaignError("readiness_preview_mismatch")
        reason = receipt.get("defer_reason_code")
        if (receipt["status"] == "ready" and reason is not None) or (
            receipt["status"] == "deferred_not_ready"
            and reason not in {"capacity_unavailable", "dependency_not_ready"}
        ):
            raise CampaignError("invalid_readiness_outcome")
    if action == "launch":
        preview = _read(_receipt_path(state, target["experiment_key"], phase, "preview"))
        if receipt.get("preview_receipt_sha256") != preview["receipt_sha256"]:
            raise CampaignError("launch_preview_mismatch")
        readiness_path = Path(receipt.get("readiness_receipt_path", ""))
        readiness_digest = _exact_digest(
            receipt.get("readiness_receipt_sha256"), "readiness_receipt_sha256"
        )
        expected_readiness_path = (
            state
            / "targets"
            / target["experiment_key"]
            / phase
            / "observations"
            / f"{readiness_digest.removeprefix('sha256:')}.json"
        )
        if readiness_path != expected_readiness_path:
            raise CampaignError("launch_readiness_mismatch")
        readiness = _read(expected_readiness_path)
        if (
            readiness.get("action") != "ready"
            or readiness.get("status") != "ready"
            or readiness_digest != readiness.get("receipt_sha256")
        ):
            raise CampaignError("launch_readiness_mismatch")
        validate_receipt(readiness, target, phase, "ready", state)
        if driver["provider"] == "tensorlake":
            ready_gate = _validate_tensorlake_preflight(readiness)
            launch_gate = _validate_tensorlake_preflight(receipt)
            if (
                receipt.get("remote_name") != readiness.get("remote_name")
                or launch_gate["checked_at_epoch"] < ready_gate["checked_at_epoch"]
            ):
                raise CampaignError("tensorlake_precreate_revalidation_stale")
        if not isinstance(receipt.get("remote_id"), str) or not receipt["remote_id"]:
            raise CampaignError("launch_remote_id_missing")
    if action == "observe":
        launch = _read(_receipt_path(state, target["experiment_key"], phase, "launch"))
        if (
            receipt.get("launch_receipt_sha256") != launch["receipt_sha256"]
            or receipt.get("remote_id") != launch["remote_id"]
        ):
            raise CampaignError("observation_launch_mismatch")
        if receipt["status"] in TERMINAL:
            _exact_digest(receipt.get("terminal_evidence_sha256"), "terminal_evidence_sha256")
    return receipt


def record(state: Path, key: str, phase: str, action: str, receipt_path: Path) -> dict[str, Any]:
    plan = load_plan(state)
    target = next((item for item in plan["targets"] if item["experiment_key"] == key), None)
    if target is None or phase not in PHASES or action not in ACTIONS:
        raise CampaignError("unknown_campaign_target")
    receipt = _read(receipt_path)
    validate_receipt(receipt, target, phase, action, state)
    destination = (
        state
        / "targets"
        / key
        / phase
        / "observations"
        / f"{receipt['receipt_sha256'].removeprefix('sha256:')}.json"
        if (action == "observe" and receipt["status"] == "running") or action == "ready"
        else _receipt_path(state, key, phase, action)
    )
    existing = _read(destination) if destination.exists() else None
    if existing is not None:
        if existing != receipt:
            raise CampaignError("campaign_receipt_already_exists")
        return existing
    _write_once(destination, receipt)
    return receipt


def _target_status(state: Path, target: dict[str, Any]) -> tuple[str, str]:
    key = target["experiment_key"]
    for phase in PHASES:
        observe = _receipt_path(state, key, phase, "observe")
        if observe.exists():
            status = validate_receipt(_read(observe), target, phase, "observe", state)["status"]
            if status == "accepted":
                if phase == "score":
                    return "complete", "none"
                continue
            if status in TERMINAL:
                return f"{phase}_{status}", "none"
            return f"{phase}_running", f"{phase}_observe"
        if (
            _receipt_path(state, key, phase, "launch-intent").exists()
            and not _receipt_path(state, key, phase, "launch").exists()
        ):
            return f"{phase}_launch_uncertain", "none"
        preview = _receipt_path(state, key, phase, "preview")
        if not preview.exists():
            return f"{phase}_pending", f"{phase}_preview"
        validate_receipt(_read(preview), target, phase, "preview", state)
        launch = _receipt_path(state, key, phase, "launch")
        if not launch.exists():
            return f"{phase}_ready", f"{phase}_launch"
        validate_receipt(_read(launch), target, phase, "launch", state)
        return f"{phase}_created", f"{phase}_observe"
    raise CampaignError("invalid_target_state")


def status(state: Path) -> dict[str, Any]:
    plan = load_plan(state)
    rows = [
        {"experiment_key": target["experiment_key"], "state": _target_status(state, target)[0]}
        for target in plan["targets"]
    ]
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["state"]] = counts.get(row["state"], 0) + 1
    return {
        "campaign_id": plan["campaign_id"],
        "plan_sha256": plan["plan_sha256"],
        "counts": counts,
        "targets": rows,
    }


def _claim(registry: Path, state: Path, plan: dict[str, Any], key: str, phase: str) -> None:
    if (
        registry != CANONICAL_REGISTRY
        or registry.is_symlink()
        or registry.resolve(strict=False) != registry
    ):
        raise CampaignError("invalid_campaign_registry")
    registry.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock = (registry / ".lock").open("a", encoding="utf-8")
    with lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = registry / "claims" / f"{key}.{phase}.json"
        value = {
            "schema": "cyber_eval_campaign_claim_v1",
            "experiment_key": key,
            "phase": phase,
            "campaign_state": str(state.resolve()),
            "plan_sha256": plan["plan_sha256"],
        }
        if path.exists():
            if _read(path) != value:
                raise CampaignError("experiment_already_claimed")
        else:
            _write_once(path, value)


def _run_driver(
    state: Path,
    registry: Path,
    plan: dict[str, Any],
    target: dict[str, Any],
    phase: str,
    action: str,
    execute: bool,
) -> None:
    if action == "launch" and not execute:
        return
    key = target["experiment_key"]
    driver = target["drivers"][phase]
    packet = state / "targets" / key / "packet.json"
    values = {
        "packet": str(packet),
        "preview_receipt": str(_receipt_path(state, key, phase, "preview")),
        "launch_receipt": str(_receipt_path(state, key, phase, "launch")),
        "terminal_receipt": str(_receipt_path(state, key, "rollout", "observe")),
    }

    def invoke(driver_action: str) -> tuple[dict[str, Any], Path]:
        with tempfile.NamedTemporaryFile(dir=state, prefix=".driver-", delete=False) as stream:
            temporary = Path(stream.name)
        temporary.unlink()
        values["receipt"] = str(temporary)
        argv = [arg.format(**values) for arg in driver["commands"][driver_action]]
        try:
            result = subprocess.run(argv, capture_output=True, timeout=driver["timeout_seconds"])
            if result.returncode or not temporary.exists():
                raise CampaignError(f"{phase}_{driver_action}_driver_failed")
            receipt = record(state, key, phase, driver_action, temporary)
            saved = (
                state
                / "targets"
                / key
                / phase
                / "observations"
                / f"{receipt['receipt_sha256'].removeprefix('sha256:')}.json"
                if driver_action == "ready"
                else _receipt_path(state, key, phase, driver_action)
            )
            return receipt, saved
        finally:
            temporary.unlink(missing_ok=True)

    if action == "launch":
        readiness, readiness_path = invoke("ready")
        if readiness["status"] == "deferred_not_ready":
            return
        values["readiness_receipt"] = str(readiness_path)
        _claim(registry, state, plan, key, phase)
        _write_once(
            _receipt_path(state, key, phase, "launch-intent"),
            {
                "schema": "cyber_eval_campaign_launch_intent_v1",
                "experiment_key": key,
                "phase": phase,
                "preview_receipt_sha256": _read(Path(values["preview_receipt"]))["receipt_sha256"],
                "readiness_receipt_sha256": readiness["receipt_sha256"],
            },
        )
    invoke(action)


def step(state: Path, *, execute: bool = False) -> dict[str, Any]:
    """Advance every independent target once; one failure never stops siblings."""
    plan = load_plan(state)
    registry = Path(plan["registry_path"])
    errors: list[dict[str, str]] = []
    advanced = 0
    launches = 0
    canary_hold = plan["scheduler"]["serial_canaries"] and any(
        target["canary"] and _target_status(state, target)[0] != "complete"
        for target in plan["targets"]
    )
    launch_limit = 1 if canary_hold else plan["scheduler"]["max_launches_per_step"]
    for target in plan["targets"]:
        _state, next_action = _target_status(state, target)
        if next_action == "none":
            continue
        phase, action = next_action.split("_", 1)
        if action == "launch" and (
            launches >= launch_limit or canary_hold and not target["canary"]
        ):
            continue
        try:
            before = _target_status(state, target)
            _run_driver(state, registry, plan, target, phase, action, execute)
            if _target_status(state, target) != before:
                advanced += 1
                if action == "launch":
                    launches += 1
        except Exception as exc:
            errors.append({"experiment_key": target["experiment_key"], "error": type(exc).__name__})
    return {"advanced": advanced, "errors": errors, **status(state)}


def run(
    state: Path,
    *,
    poll_seconds: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Run score-blind campaign steps until every target is terminal or held."""

    if type(poll_seconds) not in (int, float) or not 0 < poll_seconds < float("inf"):
        raise CampaignError("invalid_poll_seconds")
    plan = load_plan(state)
    targets = {target["experiment_key"]: target for target in plan["targets"]}
    while True:
        result = step(state, execute=True)
        states = {
            target["experiment_key"]: _target_status(state, target) for target in plan["targets"]
        }
        held = [
            {"experiment_key": key, "reason": "launch_uncertain"}
            for key, (target_state, _action) in states.items()
            if target_state.endswith("_launch_uncertain")
        ]
        active = {key for key, (_target_state, action) in states.items() if action != "none"}
        canary_barrier = plan["scheduler"]["serial_canaries"] and any(
            targets[key]["canary"] and action == "none" and target_state != "complete"
            for key, (target_state, action) in states.items()
        )
        if canary_barrier:
            serial_holds = {key for key in active if not targets[key]["canary"]}
            held.extend(
                {"experiment_key": key, "reason": "serial_canary_not_accepted"}
                for key in sorted(serial_holds)
            )
            active -= serial_holds
        if not active:
            return {
                **result,
                "run_status": "held" if held else "terminal",
                "held_targets": held,
            }
        if result["advanced"] == 0:
            sleep(float(poll_seconds))


def main(argv: list[str] | None = None) -> None:
    """Run the small campaign controller without changing the shared project CLI."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("config", type=Path)
    prepare_parser.add_argument("--output", required=True, type=Path)
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("directory", type=Path)
    step_parser = subparsers.add_parser("step")
    step_parser.add_argument("directory", type=Path)
    step_parser.add_argument("--execute", action="store_true")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("directory", type=Path)
    run_parser.add_argument("--execute", action="store_true", required=True)
    run_parser.add_argument("--poll-seconds", type=float, default=30.0)
    record_parser = subparsers.add_parser("record")
    record_parser.add_argument("directory", type=Path)
    record_parser.add_argument("--experiment-key", required=True)
    record_parser.add_argument("--phase", choices=PHASES, required=True)
    record_parser.add_argument("--action", choices=ACTIONS, required=True)
    record_parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.config, args.output)
    elif args.command == "status":
        result = status(args.directory)
    elif args.command == "step":
        result = step(args.directory, execute=args.execute)
    elif args.command == "run":
        result = run(args.directory, poll_seconds=args.poll_seconds)
    else:
        result = record(
            args.directory,
            args.experiment_key,
            args.phase,
            args.action,
            args.receipt,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
