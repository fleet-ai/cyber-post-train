"""Held DP6-m score-free server and qualification authority.

This successor is append-only relative to the consumed DP6-l identities.  It
contains no scored-cell selector and exposes no live create operation.
"""

from __future__ import annotations

import json
import shlex
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from evals.fleet import glm53_dedicated_v7 as jobs_api
from evals.fleet import qwen38_dedicated_dp6_v1 as runtime
from evals.fleet import qwen38_dp6_l_scorefree_v1 as predecessor
from evals.fleet import qwen38_dp6_qualification_guard_v3 as lifecycle_guard
from evals.fleet import self_hosted

CONFIG_PATH = Path("evals/fleet/configs/qwen38-dedicated-dp6-m-scorefree-v1-held.json")
PLAN_PATH = Path("docs/evidence/qwen38-study/2026-09-06-qwen38-dp6-m-scorefree-held-plan-v1.json")
PREVIEW_PATH = Path("docs/evidence/qwen38-study/2026-09-06-qwen38-dp6-m-jobs-preview-v1.json")
INVENTORY_PATH = Path("docs/evidence/qwen38-study/2026-09-06-qwen38-dp6-m-review-inventory-v1.json")
RELEASE_PATH = Path("docs/evidence/qwen38-study/2026-09-06-qwen38-dp6-m-held-release-v1.json")
SUCCESSOR_AUTHORITY_PATH = Path(
    "docs/evidence/qwen38-study/2026-09-06-qwen38-dp6-l-scorefree-held-plan-v1.json"
)
SUCCESSOR_AUTHORITY_RECEIPT_SHA256 = (
    "sha256:1dfb18260d44cc7e282c93232b6622a3822db0698fd168e807be50446546b581"
)

CONFIG_SCHEMA = "fleet-qwen38-dp6-m-scorefree-config-v1"
PLAN_SCHEMA = "fleet-qwen38-dp6-m-scorefree-held-plan-v1"
PREVIEW_SCHEMA = "fleet-qwen38-dp6-m-jobs-preview-v1"
INVENTORY_SCHEMA = "fleet-qwen38-dp6-m-review-inventory-v1"
RELEASE_SCHEMA = "fleet-qwen38-dp6-m-held-release-v1"
TITLE = "chris-cyber-evalserve-q38-dp6-m-v1"
RUN_DIR = "/mnt/sfs/jobs/chris-cyber-evalserve-q38-dp6-m-v1"
SERVING_BLOCK = "dedicated-qwen-dp6-m-v1"
QUALIFIER_JOB = "chris-cyber-q38-dp6-m-qualifier-v13"
QUALIFIER_OUTPUT_ROOT = f"/mnt/sfs/jobs/{QUALIFIER_JOB}"
SERVER_PRIORITY_CLASS = "fleet-infra-quiet"
SERVER_PRIORITY_VALUE = -1000
QUALIFIER_PRIORITY_CLASS = "fleet-serve-low"
QUALIFIER_PRIORITY_VALUE = 100
MAX_PROJECT_NODES = 2
MAX_PROJECT_GPUS = 16
IDLE_SECONDS = 600
SERVER_PRE_READY_TIMEOUT_SECONDS = 600
COUNTER_PRODUCER_STARTUP_GRACE_SECONDS = 120
MAX_JSON_BYTES = 16 * 1024 * 1024
PREVIEW_KEYS = {
    "schema_version",
    "status",
    "http_status",
    "title",
    "run_dir",
    "priority_class",
    "config_sha256",
    "request_sha256",
    "api_mutations",
    "scored_calls",
    "prompts_traces_flags_or_scores_included",
    "receipt_sha256",
}
INVENTORY_KEYS = {
    "schema_version",
    "status",
    "active_project_serving_runs",
    "active_project_gpu_nodes",
    "active_project_gpus",
    "target_identity_matches",
    "capacity",
    "api_mutations",
    "scored_calls",
    "prompts_traces_flags_or_scores_included",
    "receipt_sha256",
}

LIFECYCLE_PATH = Path("evals/fleet/scripts/qwen38_dedicated_dp6_lifecycle_v2.sh")
OBSERVER_V1_PATH = Path("evals/fleet/qwen38_dp6_metric_observer_v1.py")
OBSERVER_V2_PATH = Path("evals/fleet/qwen38_dp6_metric_observer_v2.py")
OBSERVER_V3_PATH = Path("evals/fleet/qwen38_dp6_metric_observer_v3.py")
OBSERVER_V4_PATH = Path("evals/fleet/qwen38_dp6_metric_observer_v4.py")
OBSERVER_V5_PATH = Path("evals/fleet/qwen38_dp6_metric_observer_v5.py")
QUALIFIER_RUNTIME_PATH = Path("evals/fleet/qwen38_dp6_m_qualifier_runtime_v1.py")
QUALIFIER_PACKAGE_PATH = Path("evals/fleet/qwen38_dp6_m_qualifier_package_v1.py")
LIVE_GATE_PATH = Path("evals/fleet/qwen38_dp6_m_live_v1.py")

RUNTIME_OBSERVER_PATH = "/tmp/qwen38_dp6_metric_observer_v5.py"
RUNTIME_DEPENDENCY_DIR = "/tmp/evals/fleet"


def _load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError(f"{path} is not a safe bounded JSON input")

    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in rows:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = item
        return result

    value = json.loads(path.read_bytes(), object_pairs_hook=pairs)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _receipt_valid(value: Mapping[str, Any], field: str = "receipt_sha256") -> bool:
    return value.get(field) == self_hosted.digest_without(dict(value), field)


def _validate_successor_authority(value: Mapping[str, Any]) -> None:
    # The predecessor plan is immutable input authority.  Pinning its canonical
    # self-digest rejects both re-signed extra fields and any historical-plan
    # rewrite without recursively depending on the predecessor's mutable source
    # reconstruction code.
    if (
        not _receipt_valid(value)
        or value.get("receipt_sha256") != SUCCESSOR_AUTHORITY_RECEIPT_SHA256
    ):
        raise ValueError("DP6-m successor authority drifted")


def jobs_payload(root: Path) -> dict[str, Any]:
    lifecycle = (root / LIFECYCLE_PATH).read_text()
    expected_pre_ready = f"PRE_READY_TIMEOUT_SECONDS={SERVER_PRE_READY_TIMEOUT_SECONDS}"
    if expected_pre_ready not in lifecycle:
        raise ValueError("DP6-m lifecycle pre-ready timeout drifted")
    sources = {
        "qwen38_dp6_metric_observer_v1.py": (root / OBSERVER_V1_PATH).read_text(),
        "qwen38_dp6_metric_observer_v2.py": (root / OBSERVER_V2_PATH).read_text(),
        "qwen38_dp6_metric_observer_v3.py": (root / OBSERVER_V3_PATH).read_text(),
        "qwen38_dp6_metric_observer_v4.py": (root / OBSERVER_V4_PATH).read_text(),
        "qwen38_dp6_metric_observer_v5.py": (root / OBSERVER_V5_PATH).read_text(),
    }
    writes = [
        "from pathlib import Path",
        f"base=Path({RUNTIME_DEPENDENCY_DIR!r})",
        "base.mkdir(parents=True,exist_ok=True)",
        "(base.parent/'__init__.py').touch()",
        "(base/'__init__.py').touch()",
    ]
    for name, source in sources.items():
        target = (
            RUNTIME_OBSERVER_PATH if name.endswith("v5.py") else f"{RUNTIME_DEPENDENCY_DIR}/{name}"
        )
        writes.extend(
            (
                f"Path({target!r}).write_text({source!r})",
                f"Path({target!r}).chmod(0o500)",
            )
        )
    bootstrap = "python3 - <<'PY'\n" + "\n".join(writes) + "\nPY\n" + lifecycle
    payload = {
        "image": runtime.IMAGE,
        "command": (
            "bash -lc " + shlex.quote(bootstrap) + " -- " + shlex.join(runtime.SERVER_ARGUMENTS)
        ),
        "workers": 1,
        "gpus_per_worker": 6,
        "env": {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONPATH": lifecycle_guard.observer_pythonpath(RUNTIME_DEPENDENCY_DIR),
            "QWEN38_RUN_DIR": RUN_DIR,
            "QWEN38_DP6_OBSERVER_SCRIPT": RUNTIME_OBSERVER_PATH,
        },
        "secrets": [],
        "resources": {
            "cpu_request": "32",
            "cpu_limit": "96",
            "memory_request": "256Gi",
            "memory_limit": "768Gi",
        },
        "priority_class": SERVER_PRIORITY_CLASS,
        "privileged": False,
        "run_dir": RUN_DIR,
        "title": TITLE,
    }
    if set(payload) != jobs_api.EXPECTED_API_FIELDS:
        raise AssertionError("DP6-m Jobs API payload shape drifted")
    return payload


def preview_identity(manifest_yaml: str, root: Path) -> dict[str, Any]:
    payload = jobs_payload(root)
    value = yaml.safe_load(manifest_yaml)
    if not isinstance(value, dict) or value.get("kind") != "RayJob":
        raise ValueError("DP6-m preview is not one RayJob")
    spec = value.get("spec") or {}
    pod = (
        ((spec.get("rayClusterSpec") or {}).get("headGroupSpec") or {}).get("template") or {}
    ).get("spec") or {}
    containers = pod.get("containers") or []
    if len(containers) != 1:
        raise ValueError("DP6-m preview head container cardinality drifted")
    container = containers[0]
    requests = (container.get("resources") or {}).get("requests") or {}
    limits = (container.get("resources") or {}).get("limits") or {}
    env = {row.get("name"): row.get("value") for row in container.get("env") or []}
    result = {
        "kind": "RayJob",
        "suspended": spec.get("suspend"),
        "queue": (value.get("metadata", {}).get("labels") or {}).get("kueue.x-k8s.io/queue-name"),
        "priority_class": pod.get("priorityClassName"),
        "preemption_policy": pod.get("preemptionPolicy"),
        "image": container.get("image"),
        "run_dir": env.get("RUN_DIR"),
        "gpu_request": int(requests.get("nvidia.com/gpu", 0)),
        "gpu_limit": int(limits.get("nvidia.com/gpu", 0)),
        "command_sha256": self_hosted.sha256(str(spec.get("entrypoint") or "").encode()),
    }
    if result != {
        "kind": "RayJob",
        "suspended": True,
        "queue": "training-lq",
        "priority_class": SERVER_PRIORITY_CLASS,
        "preemption_policy": None,
        "image": payload["image"],
        "run_dir": RUN_DIR,
        "gpu_request": 6,
        "gpu_limit": 6,
        "command_sha256": self_hosted.sha256(payload["command"].encode()),
    }:
        raise ValueError("DP6-m preview contract drifted")
    return result


def config(root: Path) -> dict[str, Any]:
    payload = jobs_payload(root)
    value = {
        "schema_version": CONFIG_SCHEMA,
        "status": "HELD_NO_LAUNCH",
        "launch_authorized": False,
        "scoring_authorized": False,
        "title": TITLE,
        "run_dir": RUN_DIR,
        "serving_block": SERVING_BLOCK,
        "create_once": True,
        "request_sha256": self_hosted.sha256(self_hosted.canonical_json(payload)),
        "request_binding": {
            "image": payload["image"],
            "command_sha256": self_hosted.sha256(payload["command"].encode()),
            "workers": 1,
            "gpus_per_worker": 6,
            "resources": payload["resources"],
            "privileged": False,
        },
        "priority_contract": {
            "class": SERVER_PRIORITY_CLASS,
            "value": SERVER_PRIORITY_VALUE,
            "preemption_policy": "Never",
            "jobs_api_allowed_classes": ["fleet-train-high", "fleet-infra-quiet"],
            "fleet_serve_low_preview_http_status": 422,
            "fleet_train_high_disallowed_as_preempting": True,
            "highest_jobs_api_admitted_nonpreempting": True,
            "fresh_jobs_api_preview_acceptance_required": True,
        },
        "side_effects": {"api_mutations": 0, "scored_calls": 0},
        "prompts_traces_flags_or_scores_included": False,
    }
    value["config_sha256"] = self_hosted.digest_without(value, "config_sha256")
    return value


def validate_config(value: Mapping[str, Any], root: Path) -> None:
    if dict(value) != config(root):
        raise ValueError("DP6-m held config drifted")


def plan(root: Path) -> dict[str, Any]:
    successor = _load(root / SUCCESSOR_AUTHORITY_PATH)
    _validate_successor_authority(successor)
    value = {
        "schema_version": PLAN_SCHEMA,
        "status": "HELD_PENDING_INDEPENDENT_REVIEW",
        "launch_authorized": False,
        "scoring_authorized": False,
        "config_path": str(CONFIG_PATH),
        "config_sha256": config(root)["config_sha256"],
        "successor_authority": {
            "path": str(SUCCESSOR_AUTHORITY_PATH),
            "receipt_sha256": successor["receipt_sha256"],
            "fresh_server_title": TITLE,
            "fresh_server_run_dir": RUN_DIR,
            "fresh_qualifier": QUALIFIER_JOB,
        },
        "superseded_plan": {
            "path": str(predecessor.PLAN_PATH),
            "receipt_sha256": _load(root / predecessor.PLAN_PATH)["receipt_sha256"],
            "rewritten": False,
        },
        "release_first_gate": {
            "active_project_serving_runs_required": 0,
            "active_project_gpu_nodes_required": 0,
            "active_project_gpus_required": 0,
            "unknown_or_orphan_project_objects_fail_closed": True,
            "target_jobs_api_kubernetes_sfs_matches_required": 0,
            "projected_gpu_nodes": 1,
            "projected_gpus": 6,
            "maximum_project_gpu_nodes": MAX_PROJECT_NODES,
            "maximum_project_gpus": MAX_PROJECT_GPUS,
            "minimum_schedulable_six_gpu_fit_nodes": 1,
            "minimum_nominal_quota_headroom_gpus": 6,
        },
        "server": {
            "title": TITLE,
            "run_dir": RUN_DIR,
            "serving_block": SERVING_BLOCK,
            "image": runtime.IMAGE,
            "model_revision": runtime.MODEL_REVISION,
            "server_arguments_sha256": runtime.SERVER_ARGUMENTS_SHA256,
            "tensor_parallel_size": 1,
            "data_parallel_size": 6,
            "priority_class": SERVER_PRIORITY_CLASS,
            "priority_value": SERVER_PRIORITY_VALUE,
            "preemption_policy": "Never",
            "post_ready_idle_seconds": IDLE_SECONDS,
            "metric_observer": str(OBSERVER_V5_PATH),
        },
        "qualifier_controller": {
            "job_name": QUALIFIER_JOB,
            "configmap_name": QUALIFIER_JOB,
            "output_root": QUALIFIER_OUTPUT_ROOT,
            "priority_class": QUALIFIER_PRIORITY_CLASS,
            "priority_value": QUALIFIER_PRIORITY_VALUE,
            "preemption_policy": "Never",
            "task_instance_session_verifier_scoring_calls": 0,
        },
        "qualification": {
            "concurrency_ladder": [1, 2, 4, 6],
            "strictly_ascending_stop_before_next_on_failure": True,
            "actual_opencode_version": "1.18.27",
            "context_management": "opencode_1.18.27_native_compaction_autocontinue_v1",
            "compaction_headroom_tokens": 20000,
            "max_output_tokens": 32768,
            "max_model_requests": 600,
            "tools": ["bash", "submit_report"],
            "tool_catalog_sha256": (
                "sha256:85fad6bdc3a835bf52a11a99b3387740eb06eb3d1720ad9bb33f3feac215b44a"
            ),
            "exact_tool_names_order_and_arguments_required": True,
            "stable_post_binding_counter_baseline_required_before_every_wave": True,
            "initial_counter_baseline_preserved_create_once": True,
            "immutable_exact_counter_snapshots_required_before_and_after_every_wave": True,
            "sampler_state_forbidden_as_wave_counter_authority": True,
            "server_health_precedes_counter_producer": True,
            "server_pre_ready_timeout_seconds": SERVER_PRE_READY_TIMEOUT_SECONDS,
            "counter_producer_startup_grace_seconds": (COUNTER_PRODUCER_STARTUP_GRACE_SECONDS),
            "counter_producer_wait_must_cover_pre_ready_plus_grace": True,
            "controller_resource_sample_wait_seconds": 30,
            "controller_resource_sample_ready_before_probe": True,
            "controller_resource_cgroup_membership_source": "/proc/self/cgroup",
            "controller_resource_exact_cgroup_v2_leaf_required": True,
            "controller_resource_shared_ready_marker_required": True,
            "controller_resource_sample_max_age_seconds": 5,
            "controller_resource_sample_advance_wait_seconds": 5,
            "controller_resource_producer_consumer_end_to_end_tested": True,
            "http_failure_method_path_status_required": True,
            "http_failure_structured_code_fixed_allowlist_only": True,
            "http_failure_response_shape_and_class_hash_required": True,
            "http_failure_free_text_forbidden": True,
            "observer_sfs_mount_path_must_be_derived_from_pod_spec": True,
            "server_binding_projected_create_once_before_counter_producer": True,
            "server_binding_projection_uid_and_mount_bound": True,
            "server_binding_projection_exact_readback_required": True,
            "fresh_sanitized_failed_observer_status_releases_immediately": True,
            "server_request_delta_must_equal_observed_model_requests": True,
            "short_wave_instantaneous_gpu_utilization_is_descriptive": True,
            "all_six_devices_active_at_c6_required": True,
            "failed_wave_sanitized_evidence_preserved": True,
            "failure_codes_fixed_allowlist_only": True,
            "exception_derived_text_type_or_hash_forbidden": True,
            "minimum_latency_headroom_milliseconds": 120000,
            "minimum_latency_headroom_fraction": 0.2,
            "task_instance_session_verifier_scoring_calls": 0,
            "statistical_cells_selected": 0,
        },
        "lifecycle": {
            "idle_seconds": IDLE_SECONDS,
            "completed_request_delta_or_active_request_or_gpu_activity_only": True,
            "health_or_clock_heartbeat_forbidden": True,
            "release_on_idle_or_failure": True,
        },
        "source_digests": {
            str(path): self_hosted.sha256((root / path).read_bytes())
            for path in (
                LIFECYCLE_PATH,
                OBSERVER_V1_PATH,
                OBSERVER_V2_PATH,
                OBSERVER_V3_PATH,
                OBSERVER_V4_PATH,
                OBSERVER_V5_PATH,
                QUALIFIER_RUNTIME_PATH,
                QUALIFIER_PACKAGE_PATH,
                LIVE_GATE_PATH,
                Path("evals/fleet/qwen38_dp6_qualification_guard_v3.py"),
            )
        },
        "statistical_cells_selected": 0,
        "scored_calls": 0,
        "prompts_traces_flags_or_scores_included": False,
    }
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    return value


def validate_plan(value: Mapping[str, Any], root: Path) -> None:
    if dict(value) != plan(root):
        raise ValueError("DP6-m held plan drifted")


def validate_preview(value: Mapping[str, Any], root: Path) -> None:
    if (
        set(value) != PREVIEW_KEYS
        or not _receipt_valid(value)
        or (
            value.get("schema_version") != PREVIEW_SCHEMA
            or value.get("status") != "HELD_NOT_RUN"
            or value.get("http_status") is not None
            or value.get("title") != TITLE
            or value.get("run_dir") != RUN_DIR
            or value.get("priority_class") != SERVER_PRIORITY_CLASS
            or value.get("config_sha256") != config(root)["config_sha256"]
            or value.get("request_sha256") != config(root)["request_sha256"]
            or value.get("api_mutations") != 0
            or value.get("scored_calls") != 0
            or value.get("prompts_traces_flags_or_scores_included") is not False
        )
    ):
        raise ValueError("DP6-m preview receipt drifted")


def validate_inventory(value: Mapping[str, Any]) -> None:
    if (
        set(value) != INVENTORY_KEYS
        or not _receipt_valid(value)
        or (
            value.get("schema_version") != INVENTORY_SCHEMA
            or value.get("status") != "HELD_NOT_QUERIED"
            or value.get("active_project_serving_runs") is not None
            or value.get("active_project_gpu_nodes") is not None
            or value.get("active_project_gpus") is not None
            or value.get("target_identity_matches") is not None
            or value.get("capacity") is not None
            or value.get("api_mutations") != 0
            or value.get("scored_calls") != 0
            or value.get("prompts_traces_flags_or_scores_included") is not False
        )
    ):
        raise ValueError("DP6-m review inventory is not clear")


def validate_held_release(value: Mapping[str, Any], root: Path) -> None:
    status = "HELD_CODE_REVIEW_ONLY"
    expected = {
        "schema_version": RELEASE_SCHEMA,
        "status": status,
        "launch_authorized": False,
        "scoring_authorized": False,
        "title": TITLE,
        "run_dir": RUN_DIR,
        "serving_block": SERVING_BLOCK,
        "plan_receipt_sha256": plan(root)["receipt_sha256"],
        "config_sha256": config(root)["config_sha256"],
        "preview_receipt_sha256": _load(root / PREVIEW_PATH)["receipt_sha256"],
        "inventory_receipt_sha256": _load(root / INVENTORY_PATH)["receipt_sha256"],
        "fresh_live_gate_required_immediately_before_create": True,
        "fresh_preview_and_inventory_required_before_release": True,
        "server_create_limit": 1,
        "qualifier_create_limit": 1,
        "release_first_required": True,
        "statistical_cells_selected": 0,
        "scored_calls": 0,
        "api_mutations": 0,
        "prompts_traces_flags_or_scores_included": False,
    }
    if (
        not _receipt_valid(value)
        or {k: v for k, v in value.items() if k != "receipt_sha256"} != expected
    ):
        raise ValueError("DP6-m held release drifted")


def load_all(root: Path) -> tuple[dict[str, Any], ...]:
    paths = (CONFIG_PATH, PLAN_PATH, PREVIEW_PATH, INVENTORY_PATH, RELEASE_PATH)
    values = tuple(_load(root / path) for path in paths)
    validate_config(values[0], root)
    validate_plan(values[1], root)
    validate_preview(values[2], root)
    validate_inventory(values[3])
    validate_held_release(values[4], root)
    return values
