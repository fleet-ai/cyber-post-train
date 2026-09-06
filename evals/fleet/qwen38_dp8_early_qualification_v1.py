"""Held, score-free Qwen DP8 qualification while TP1 remains productive."""

from __future__ import annotations

import json
import shlex
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from evals.fleet import glm53_dedicated_v7 as jobs_api
from evals.fleet import qwen38_dedicated_dp8_v1 as runtime
from evals.fleet import qwen38_dp8_post_rank99_launch_v1 as ladder
from evals.fleet import qwen38_dp8_post_rank99_plan_v1 as predecessor
from evals.fleet import self_hosted

PLAN_PATH = Path(
    "docs/evidence/qwen38-study/2026-09-06-qwen38-dp8-early-qualification-held-v2.json"
)
CONFIG_PATH = Path("evals/fleet/configs/qwen38-dedicated-dp8-early-qualification-v2-held.json")
PREVIEW_PATH = Path(
    "docs/evidence/qwen38-study/2026-09-06-qwen38-dp8-early-qualification-preview-v2.json"
)
INVENTORY_PATH = Path(
    "docs/evidence/qwen38-study/2026-09-06-qwen38-dp8-early-qualification-node-inventory-v2.json"
)
RELEASE_PATH = Path(
    "docs/evidence/qwen38-study/2026-09-06-qwen38-dp8-early-qualification-held-release-v2.json"
)
V1_INCIDENT_PATH = Path(
    "docs/evidence/qwen38-study/"
    "2026-09-06-qwen38-dp8-early-v1-infrastructure-failure-release-v1.json"
)
PRIORITY_CONTRACT_PATH = Path(
    "docs/evidence/qwen38-study/2026-09-06-qwen38-dp8-priority-contract-v1.json"
)
SCHEMA = "fleet-qwen38-dp8-early-qualification-held-v2"
CONFIG_SCHEMA = "fleet-qwen38-dp8-early-qualification-config-v2"
PREVIEW_SCHEMA = "fleet-qwen38-dp8-early-qualification-preview-v2"
INVENTORY_SCHEMA = "fleet-qwen38-dp8-early-qualification-node-inventory-v2"
RELEASE_SCHEMA = "fleet-qwen38-dp8-early-qualification-held-release-v2"
TITLE = "chris-cyber-evalserve-q38-dp8-c-v2"
RUN_DIR = "/mnt/sfs/jobs/chris-cyber-evalserve-q38-dp8-c-v2"
SERVING_BLOCK = "dedicated-qwen-dp8-c-v2"
SERVER_PRIORITY_CLASS = "fleet-infra-quiet"
QUALIFIER_PRIORITY_CLASS = "fleet-serve-low"
QUALIFIER_PRIORITY_VALUE = 100
LOWER_NONPREEMPTING_PRIORITY_CLASS = "fleet-infra-quiet"
LOWER_NONPREEMPTING_PRIORITY_VALUE = -1000
LIFECYCLE_V2_PATH = Path("evals/fleet/scripts/qwen38_dedicated_dp8_lifecycle_v2.sh")
OBSERVER_V2_PATH = Path("evals/fleet/qwen38_dp8_metric_observer_v2.py")
QUALIFIER_RUNTIME_PATH = Path("evals/fleet/qwen38_dp8_early_qualifier_runtime_v1.py")
QUALIFIER_PACKAGE_PATH = Path("evals/fleet/qwen38_dp8_early_qualifier_package_v1.py")
LIVE_SUBMITTER_PATH = Path("evals/fleet/qwen38_dp8_early_live_v1.py")
RUNTIME_OBSERVER_PATH = "/tmp/qwen38_dp8_metric_observer_v2.py"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def jobs_payload(root: Path) -> dict[str, Any]:
    lifecycle = (root / LIFECYCLE_V2_PATH).read_text()
    observer_source = (root / OBSERVER_V2_PATH).read_text()
    bootstrap = (
        "python3 - <<'PY'\n"
        "from pathlib import Path\n"
        f"Path({RUNTIME_OBSERVER_PATH!r}).write_text({observer_source!r})\n"
        f"Path({RUNTIME_OBSERVER_PATH!r}).chmod(0o500)\n"
        "PY\n" + lifecycle
    )
    payload = {
        "image": runtime.IMAGE,
        "command": (
            "bash -lc " + shlex.quote(bootstrap) + " -- " + shlex.join(runtime.SERVER_ARGUMENTS)
        ),
        "workers": 1,
        "gpus_per_worker": 8,
        "env": {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "QWEN38_RUN_DIR": RUN_DIR,
            "QWEN38_OBSERVER_SCRIPT": RUNTIME_OBSERVER_PATH,
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
        raise AssertionError("early DP8 Jobs API payload shape drifted")
    return payload


def validate_config(value: Mapping[str, Any], root: Path) -> None:
    if value.get("config_sha256") != self_hosted.digest_without(dict(value), "config_sha256"):
        raise ValueError("early DP8 config digest drifted")
    payload = jobs_payload(root)
    if {key: item for key, item in value.items() if key != "config_sha256"} != {
        "schema_version": CONFIG_SCHEMA,
        "status": "HELD_NO_LAUNCH",
        "launch_authorized": False,
        "scoring_authorized": False,
        "title": TITLE,
        "run_dir": RUN_DIR,
        "serving_block": SERVING_BLOCK,
        "create_once": True,
        "request": payload,
        "request_sha256": self_hosted.sha256(self_hosted.canonical_json(payload)),
        "priority_contract": {
            "requested": SERVER_PRIORITY_CLASS,
            "preemption_policy": "Never",
            "jobs_api_allowed": ["fleet-train-high", "fleet-infra-quiet"],
            "higher_nonpreempting_class_rejected": QUALIFIER_PRIORITY_CLASS,
            "rejection_http_status": 422,
            "fleet_train_high_disallowed_as_preempting": True,
        },
        "privacy": {
            "credentials_included": False,
            "prompts_traces_flags_or_scores_included": False,
        },
    }:
        raise ValueError("early DP8 config contract drifted")


def validate_priority_contract(value: Mapping[str, Any]) -> None:
    if value.get("receipt_sha256") != self_hosted.digest_without(
        dict(value), "receipt_sha256"
    ):
        raise ValueError("early DP8 priority contract digest drifted")
    if {key: item for key, item in value.items() if key != "receipt_sha256"} != {
        "schema_version": "fleet-qwen38-dp8-priority-contract-v1",
        "status": "VALIDATED_NON_MUTATING",
        "observed_at": value.get("observed_at"),
        "route": "POST /v1/runs/preview",
        "server_preview_attempts": [
            {
                "priority_class": QUALIFIER_PRIORITY_CLASS,
                "http_status": 422,
                "error_type": "validation_error",
                "allowed_classes_extracted": ["fleet-infra-quiet", "fleet-train-high"],
            },
            {
                "priority_class": SERVER_PRIORITY_CLASS,
                "http_status": 200,
                "rendered_kind_count": 1,
            },
        ],
        "server_selection": {
            "selected": SERVER_PRIORITY_CLASS,
            "preemption_policy": "Never",
            "fleet_train_high_rejected_by_project_policy_as_preempting": True,
        },
        "qualifier_cpu_priority": {
            "selected": QUALIFIER_PRIORITY_CLASS,
            "priority_value": QUALIFIER_PRIORITY_VALUE,
            "preemption_policy": "Never",
            "submission_path": "kubernetes_create_once_not_jobs_api",
        },
        "api_mutations": 0,
        "credentials_included": False,
        "response_bodies_included": False,
        "prompts_traces_flags_or_scores_included": False,
    }:
        raise ValueError("early DP8 priority contract drifted")


def validate_plan(value: Mapping[str, Any], root: Path) -> None:
    if value.get("receipt_sha256") != self_hosted.digest_without(dict(value), "receipt_sha256"):
        raise ValueError("early DP8 plan digest drifted")
    expected = {
        "schema_version": SCHEMA,
        "status": "HELD_PENDING_ROOT_REVIEW",
        "launch_authorized": False,
        "scoring_authorized": False,
        "purpose": "parallel_score_free_dp8_capacity_qualification",
        "jobs_api_config": {
            "path": str(CONFIG_PATH),
            "config_sha256": _load(root / CONFIG_PATH)["config_sha256"],
        },
        "predecessor": {
            "immutable_plan_path": str(predecessor.PLAN_PATH),
            "immutable_plan_receipt_sha256": predecessor.load_held(root)["receipt_sha256"],
            "changed_contract": "qualification_may_coexist_with_productive_tp1",
            "predecessor_rewritten": False,
        },
        "superseded_v1_incident": {
            "path": str(V1_INCIDENT_PATH),
            "receipt_sha256": _load(root / V1_INCIDENT_PATH)["receipt_sha256"],
            "v1_identity_reusable": False,
            "qualification_or_scoring_occurred": False,
        },
        "priority_contract_evidence": {
            "path": str(PRIORITY_CONTRACT_PATH),
            "receipt_sha256": _load(root / PRIORITY_CONTRACT_PATH)["receipt_sha256"],
            "api_mutations": 0,
        },
        "coexistence_gate": {
            "max_project_gpu_nodes": 2,
            "max_project_gpus": 16,
            "required_gpu_nodes_before_create": 1,
            "maximum_gpus_before_create": 8,
            "allowed_existing_serving_block": "dedicated-qwen-tp1-j-v1",
            "existing_tp1_must_be_running_ready_restart0_and_productive": True,
            "fresh_jobs_api_kubernetes_gpu_inventory_required": True,
            "unknown_or_preempting_gpu_workload_fails_closed": True,
            "peer_workload_admitted_quota_reserved_no_preemption_required": True,
            "fresh_uid_bound_tp1_traffic_max_age_seconds": 300,
            "one_schedulable_b300_node_with_eight_free_gpus_required": True,
            "b300_training_free_quota_gpus_required": 8,
            "live_priority_class_value_and_policy_required": True,
        },
        "server": {
            "title": TITLE,
            "run_dir": RUN_DIR,
            "serving_block": SERVING_BLOCK,
            "create_once": True,
            "image": runtime.IMAGE,
            "model_repository": "Qwen/Qwen3.8-27B",
            "model_revision": runtime.MODEL_REVISION,
            "model_path": runtime.MODEL_PATH,
            "served_id": "qwen3.8-27b",
            "context_length": 262144,
            "tensor_parallel_size": 1,
            "data_parallel_size": 8,
            "load_balance_method": "total_tokens",
            "server_arguments_sha256": runtime.SERVER_ARGUMENTS_SHA256,
            "workers": 1,
            "gpus_per_worker": 8,
            "priority_class": SERVER_PRIORITY_CLASS,
            "preemption_policy": "Never",
            "privileged": False,
            "jobs_api_payload_sha256": self_hosted.sha256(
                self_hosted.canonical_json(jobs_payload(root))
            ),
        },
        "qualifier_controller": {
            "placement": "fleet-train-jobs_cpu_job",
            "namespace": "fleet-train-jobs",
            "job_name": "chris-cyber-q38-dp8-c-qualifier-v2",
            "configmap_name": "chris-cyber-q38-dp8-c-qualifier-v2",
            "output_root": "/mnt/sfs/jobs/chris-cyber-q38-dp8-c-qualifier-v2",
            "priority_class": QUALIFIER_PRIORITY_CLASS,
            "priority_value": QUALIFIER_PRIORITY_VALUE,
            "preemption_policy": "Never",
            "priority_selection": {
                "selected": QUALIFIER_PRIORITY_CLASS,
                "selected_value": QUALIFIER_PRIORITY_VALUE,
                "lower_nonpreempting_alternative": LOWER_NONPREEMPTING_PRIORITY_CLASS,
                "lower_nonpreempting_value": LOWER_NONPREEMPTING_PRIORITY_VALUE,
                "reason": "highest_reviewed_nonpreempting_class_avoids_qualification_starvation",
            },
            "evaluator_image": (
                "ghcr.io/astral-sh/uv:python3.12-bookworm@sha256:"
                "9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7"
            ),
            "docker_image": (
                "docker.io/library/docker@sha256:"
                "f649ef046008ca7f926a2571c32b0ac22e5c59eb61b959617f9acc2a4c638cf5"
            ),
            "docker_cli_sha256": (
                "sha256:242c7a8de606afba2acada7c7af00d77f92c3601678b2f3a60911b49a892c722"
            ),
            "docker_buildx_sha256": (
                "sha256:8c38f60308a895fa570f1410e453c5de11aafd65a99fa99965d96d24b6225a78"
            ),
            "docker_cli_total_bytes": 105594160,
            "docker_cli_emptydir_size": "256Mi",
            "docker_build_pull_exact_digest_intended": True,
            "shared_workspace_between_evaluator_and_dind": True,
            "tmpdir": "/workspace/tmp",
            "cluster_dind_host_gateway_required": True,
            "cluster_proxy_bind_host": "0.0.0.0",
            "dind_resources": {
                "requests": {"cpu": "4", "memory": "8Gi", "ephemeral-storage": "40Gi"},
                "limits": {"cpu": "10", "memory": "24Gi", "ephemeral-storage": "80Gi"},
            },
            "evaluator_resources": {
                "requests": {"cpu": "1", "memory": "2Gi", "ephemeral-storage": "10Gi"},
                "limits": {"cpu": "4", "memory": "8Gi", "ephemeral-storage": "40Gi"},
            },
            "controller_resource_observer": "dind_cgroup_v2_per_second_v1",
            "zero_dind_oom_throttle_and_protocol_errors_required": True,
            "actual_opencode_runs_in_pinned_dind": True,
            "service_origin_from_uid_bound_binding_only": True,
            "cluster_dns_only_no_port_forward": True,
            "sfs_claim_and_evidence_mount": "sfs-shared",
            "kubectl_access_required": False,
            "gpu_distribution_source": "server_local_signed_sfs_observation_v2",
            "server_pod_name_and_uid_cross_binding_required": True,
            "create_only_renderer_required_before_launch": True,
            "create_only_renderer_path": str(QUALIFIER_PACKAGE_PATH),
            "create_only_renderer_file_sha256": self_hosted.sha256(
                (root / QUALIFIER_PACKAGE_PATH).read_bytes()
            ),
            "runtime_path": str(QUALIFIER_RUNTIME_PATH),
            "runtime_file_sha256": self_hosted.sha256((root / QUALIFIER_RUNTIME_PATH).read_bytes()),
            "separate_uid_bound_qualifier_release_required": True,
            "controller_package_ready": True,
            "server_submitter_path": str(LIVE_SUBMITTER_PATH),
            "server_submitter_file_sha256": self_hosted.sha256(
                (root / LIVE_SUBMITTER_PATH).read_bytes()
            ),
            "server_submitter_requires_clean_exact_commit": True,
            "server_submitter_create_once_post_limit": 1,
            "server_submitter_allowed_peer": "dedicated-qwen-tp1-j-v1",
        },
        "qualification": {
            "existing_runner_path": str(QUALIFIER_RUNTIME_PATH),
            "existing_runner_file_sha256": self_hosted.sha256(
                (root / QUALIFIER_RUNTIME_PATH).read_bytes()
            ),
            "lifecycle_file_sha256": self_hosted.sha256((root / LIFECYCLE_V2_PATH).read_bytes()),
            "metric_observer_file_sha256": self_hosted.sha256(
                (root / OBSERVER_V2_PATH).read_bytes()
            ),
            "stable_post_binding_counter_baseline_required_before_every_wave": True,
            "historical_request_counters_must_not_refresh_idle": True,
            "concurrency_ladder": [1, 2, 4, 8],
            "strictly_ascending_stop_before_next_on_failure": True,
            "actual_opencode_version": "1.18.27",
            "context_management": ("opencode_1.18.27_native_compaction_autocontinue_v1"),
            "compaction_headroom_tokens": 20000,
            "max_output_tokens": 32768,
            "max_model_requests": 600,
            "tools": ["bash", "submit_report"],
            "tool_catalog_sha256": (
                "sha256:85fad6bdc3a835bf52a11a99b3387740eb06eb3d1720ad9bb33f3feac215b44a"
            ),
            "exact_tool_names_order_and_arguments_required": True,
            "uid_bound_request_and_gpu_distribution_required": True,
            "all_eight_devices_active_at_c8_required": True,
            "model_request_count_derived_from_stream_receipts": True,
            "server_request_delta_must_equal_observed_model_requests": True,
            "per_wave_latency_and_timeout_headroom_recorded": True,
            "minimum_latency_headroom_milliseconds": 120000,
            "minimum_latency_headroom_fraction": 0.2,
            "task_instance_session_verifier_scoring_calls": 0,
            "statistical_cells_selected": 0,
        },
        "post_qualification_gate": {
            "passing_capacity_receipt_does_not_authorize_scoring": True,
            "fresh_wholly_unstarted_task_required": True,
            "all_four_statistical_cells_reserved_atomically": True,
            "canonical_claims_before_model_call": True,
            "g19_hosted_controllers_must_skip_all_four": True,
            "one_scored_a1_canary_before_bulk": True,
            "separate_server_bound_release_required": True,
        },
        "lifecycle": {
            "model_loading_counts_as_productive": True,
            "post_ready_idle_seconds": 600,
            "real_traffic_only_refreshes_idle_deadline": True,
            "release_on_failure_or_idle": True,
            "artificial_heartbeat_forbidden": True,
        },
        "privacy": {
            "credentials_included": False,
            "request_or_response_bodies_included": False,
            "prompts_traces_flags_or_scores_included": False,
        },
    }
    if {key: item for key, item in value.items() if key != "receipt_sha256"} != expected:
        raise ValueError("early DP8 plan contract drifted")
    if value["qualification"]["existing_runner_file_sha256"] != self_hosted.sha256(
        (root / value["qualification"]["existing_runner_path"]).read_bytes()
    ):
        raise ValueError("early DP8 qualification runner bytes drifted")
    if value["qualification"]["lifecycle_file_sha256"] != self_hosted.sha256(
        (root / LIFECYCLE_V2_PATH).read_bytes()
    ):
        raise ValueError("early DP8 lifecycle bytes drifted")
    if value["qualification"]["metric_observer_file_sha256"] != self_hosted.sha256(
        (root / OBSERVER_V2_PATH).read_bytes()
    ):
        raise ValueError("early DP8 metric observer bytes drifted")


def validate_preview(value: Mapping[str, Any], root: Path) -> None:
    if value.get("receipt_sha256") != self_hosted.digest_without(dict(value), "receipt_sha256"):
        raise ValueError("early DP8 preview digest drifted")
    if set(value) != {
        "schema_version",
        "status",
        "observed_at",
        "route",
        "config_path",
        "config_sha256",
        "request_sha256",
        "title",
        "run_dir",
        "serving_block",
        "rendered",
        "api_mutations",
        "launch_authorized",
        "scoring_authorized",
        "prompts_traces_flags_or_scores_included",
        "credentials_included",
        "response_body_included",
        "receipt_sha256",
    } or (
        value.get("schema_version") != PREVIEW_SCHEMA
        or value.get("status") != "PASSED_NON_MUTATING"
        or value.get("config_path") != str(CONFIG_PATH)
        or value.get("config_sha256") != _load(root / CONFIG_PATH).get("config_sha256")
        or value.get("request_sha256")
        != self_hosted.sha256(self_hosted.canonical_json(jobs_payload(root)))
        or value.get("title") != TITLE
        or value.get("run_dir") != RUN_DIR
        or value.get("serving_block") != SERVING_BLOCK
        or value.get("api_mutations") != 0
        or value.get("route") != "POST /v1/runs/preview"
        or value.get("launch_authorized") is not False
        or value.get("scoring_authorized") is not False
        or value.get("prompts_traces_flags_or_scores_included") is not False
        or value.get("credentials_included") is not False
        or value.get("response_body_included") is not False
    ):
        raise ValueError("early DP8 preview authority drifted")
    rendered = value.get("rendered")
    if not isinstance(rendered, dict) or rendered != {
        "image": jobs_payload(root)["image"],
        "command_sha256": self_hosted.sha256(jobs_payload(root)["command"].encode()),
        "gpus": 8,
        "priority_class": SERVER_PRIORITY_CLASS,
        "privileged": False,
        "run_dir": RUN_DIR,
        "queue": "training-lq",
        "suspended": True,
        "preferred_topology": "topology.nebius.com/tier-1",
    }:
        raise ValueError("early DP8 preview rendered identity drifted")


def preview_identity(manifest_yaml: str, root: Path) -> dict[str, Any]:
    payload = jobs_payload(root)
    manifests = [row for row in yaml.safe_load_all(manifest_yaml) if isinstance(row, dict)]
    if len(manifests) != 1 or manifests[0].get("kind") != "RayJob":
        raise RuntimeError("preview did not render exactly one RayJob")
    rayjob = manifests[0]
    spec = rayjob.get("spec") or {}
    head = ((spec.get("rayClusterSpec") or {}).get("headGroupSpec") or {}).get("template") or {}
    pod = head.get("spec") or {}
    containers = pod.get("containers") or []
    if len(containers) != 1:
        raise RuntimeError("preview rendered unexpected container count")
    container = containers[0]
    security = container.get("securityContext") or {}
    resources = container.get("resources") or {}
    request = (resources.get("requests") or {}).get("nvidia.com/gpu")
    limit = (resources.get("limits") or {}).get("nvidia.com/gpu")
    env = {row.get("name"): row.get("value") for row in container.get("env") or []}
    annotations = (head.get("metadata") or {}).get("annotations") or {}
    actual = {
        "image": container.get("image"),
        "command_sha256": self_hosted.sha256(str(spec.get("entrypoint") or "").encode()),
        "gpus": request if request == limit else None,
        "priority_class": pod.get("priorityClassName"),
        "privileged": security.get("privileged", False),
        "run_dir": env.get("RUN_DIR"),
        "queue": (rayjob.get("metadata", {}).get("labels") or {}).get("kueue.x-k8s.io/queue-name"),
        "suspended": spec.get("suspend"),
        "preferred_topology": annotations.get("kueue.x-k8s.io/podset-preferred-topology"),
    }
    expected = {
        "image": payload["image"],
        "command_sha256": self_hosted.sha256(payload["command"].encode()),
        "gpus": 8,
        "priority_class": SERVER_PRIORITY_CLASS,
        "privileged": False,
        "run_dir": RUN_DIR,
        "queue": "training-lq",
        "suspended": True,
        "preferred_topology": "topology.nebius.com/tier-1",
    }
    if actual != expected:
        raise RuntimeError("preview rendered early DP8 identity drifted")
    return actual


def validate_inventory(value: Mapping[str, Any]) -> None:
    if value.get("receipt_sha256") != self_hosted.digest_without(dict(value), "receipt_sha256"):
        raise ValueError("early DP8 inventory digest drifted")
    if set(value) != {
        "schema_version",
        "status",
        "observed_at",
        "jobs_api_inventory_complete",
        "kubernetes_inventory_complete",
        "gpu_nodes",
        "gpus",
        "gpu_cluster_capacity",
        "projected_gpu_nodes_after_create",
        "projected_gpus_after_create",
        "unknown_active_dedicated_runs",
        "project_serving_inventory",
        "priority_class_contract",
        "active_peer",
        "cpu_qualifier_placement",
        "title_matches",
        "run_dir_matches",
        "kubernetes_identity_matches",
        "sfs_run_dir_exists",
        "api_mutations",
        "launch_authorized",
        "scoring_authorized",
        "prompts_traces_flags_or_scores_included",
        "receipt_sha256",
    } or (
        value.get("schema_version") != INVENTORY_SCHEMA
        or value.get("status") != "HOLD_NO_SCHEDULABLE_EIGHT_GPU_NODE"
        or value.get("gpu_nodes") != 1
        or value.get("gpus") != 1
        or value.get("projected_gpu_nodes_after_create") != 2
        or value.get("projected_gpus_after_create") != 9
        or value.get("unknown_active_dedicated_runs") != 0
        or value.get("jobs_api_inventory_complete") is not True
        or value.get("kubernetes_inventory_complete") is not True
        or value.get("title_matches") != 0
        or value.get("run_dir_matches") != 0
        or value.get("kubernetes_identity_matches") != 0
        or value.get("sfs_run_dir_exists") is not False
        or value.get("api_mutations") != 0
        or value.get("launch_authorized") is not False
        or value.get("scoring_authorized") is not False
        or value.get("prompts_traces_flags_or_scores_included") is not False
    ):
        raise ValueError("early DP8 inventory is not safely held")
    capacity = value.get("gpu_cluster_capacity")
    if not isinstance(capacity, dict) or (
        capacity.get("all_namespaces_gpu_requests_included") is not True
        or capacity.get("b300_nodes") != 24
        or capacity.get("schedulable_nodes") != 23
        or capacity.get("unschedulable_nodes") != 1
        or capacity.get("eligible_schedulable_nodes_with_eight_free_gpus") != 0
        or capacity.get("schedulable_free_gpu_histogram") != {"0": 22, "6": 1}
        or capacity.get("tp1_node")
        != {
            "name": "computeinstance-e04nnbyjvf87b5q53j",
            "allocatable_gpus": 8,
            "requested_gpus": 2,
            "free_gpus": 6,
        }
    ):
        raise ValueError("early DP8 full-node capacity evidence drifted")
    project = value.get("project_serving_inventory")
    if not isinstance(project, dict) or (
        project.get("active_jobs_api_run_ids") != ["ft-run-e87e2bd4"]
        or project.get("active_run_dirs")
        != ["/mnt/sfs/jobs/chris-cyber-evalserve-q38-tp1-j-v1"]
        or project.get("gpu_requests") != 1
        or project.get("gpu_nodes") != 1
        or project.get("orphan_project_rayjobs") != 0
        or project.get("orphan_project_gpu_pods") != 0
        or project.get("superseded_v1_jobs_api_http_status") != 404
        or project.get("superseded_v1_kubernetes_object_matches") != 0
    ):
        raise ValueError("early DP8 project serving inventory drifted")
    priority = value.get("priority_class_contract")
    if priority != {
        "server_priority_class": SERVER_PRIORITY_CLASS,
        "server_preemption_policy": "Never",
        "jobs_api_preview_http_status": 200,
        "jobs_api_allowed_priority_classes": ["fleet-train-high", "fleet-infra-quiet"],
        "fleet_serve_low_preview_http_status": 422,
        "fleet_train_high_disallowed_as_preempting": True,
        "qualifier_priority_class": QUALIFIER_PRIORITY_CLASS,
        "qualifier_priority_value": QUALIFIER_PRIORITY_VALUE,
        "qualifier_preemption_policy": "Never",
        "fleet_infra_quiet_priority_value": LOWER_NONPREEMPTING_PRIORITY_VALUE,
    }:
        raise ValueError("early DP8 priority contract drifted")
    peer = value.get("active_peer")
    if not isinstance(peer, dict) or (
        peer.get("serving_block") != "dedicated-qwen-tp1-j-v1"
        or peer.get("gpu_nodes") != 1
        or peer.get("gpus") != 1
        or peer.get("running_ready_restart0") is not True
        or peer.get("productive_traffic") is not True
    ):
        raise ValueError("early DP8 inventory peer drifted")
    placement = value.get("cpu_qualifier_placement")
    if not isinstance(placement, dict) or (
        placement.get("node_pinning") is not False
        or placement.get("node_selector") != "workload=fleetai-training-ng-cpu"
        or placement.get("qualifier_requests")
        != {"cpu_millicores": 5000, "memory_mib": 10240}
        or placement.get("qualifier_limits")
        != {"cpu_millicores": 14000, "memory_mib": 32768}
        or placement.get("request_fit_node_count", 0) < 1
    ):
        raise ValueError("early DP8 CPU qualifier placement is not schedulable")
    nodes = placement.get("nodes")
    if not isinstance(nodes, list) or len(nodes) != 3:
        raise ValueError("early DP8 CPU qualifier node inventory is incomplete")
    required = placement["qualifier_requests"]
    fit = sum(
        type(row.get("allocatable_cpu_millicores")) is int
        and type(row.get("observed_cpu_used_millicores")) is int
        and row["allocatable_cpu_millicores"] - row["observed_cpu_used_millicores"]
        >= required["cpu_millicores"]
        and type(row.get("allocatable_memory_mib")) is int
        and type(row.get("observed_memory_used_mib")) is int
        and row["allocatable_memory_mib"] - row["observed_memory_used_mib"]
        >= required["memory_mib"]
        for row in nodes
        if isinstance(row, dict)
    )
    if fit != placement["request_fit_node_count"]:
        raise ValueError("early DP8 CPU qualifier placement fit count drifted")


def validate_held_release(value: Mapping[str, Any], root: Path) -> None:
    if value.get("receipt_sha256") != self_hosted.digest_without(dict(value), "receipt_sha256"):
        raise ValueError("early DP8 held release digest drifted")
    if set(value) != {
        "schema_version",
        "status",
        "launch_authorized",
        "scoring_authorized",
        "title",
        "run_dir",
        "serving_block",
        "config_sha256",
        "plan_receipt_sha256",
        "preview_receipt_sha256",
        "inventory_receipt_sha256",
        "server_create_limit",
        "statistical_cells_selected",
        "qualification_only",
        "explicit_root_review_required",
        "fresh_live_gate_required_immediately_before_any_create",
        "api_mutations",
        "prompts_traces_flags_or_scores_included",
        "receipt_sha256",
    } or (
        value.get("schema_version") != RELEASE_SCHEMA
        or value.get("status") != "HELD_FOR_ROOT_REVIEW"
        or value.get("launch_authorized") is not False
        or value.get("scoring_authorized") is not False
        or value.get("title") != TITLE
        or value.get("run_dir") != RUN_DIR
        or value.get("serving_block") != SERVING_BLOCK
        or value.get("config_sha256") != _load(root / CONFIG_PATH).get("config_sha256")
        or value.get("plan_receipt_sha256") != _load(root / PLAN_PATH).get("receipt_sha256")
        or value.get("preview_receipt_sha256") != _load(root / PREVIEW_PATH).get("receipt_sha256")
        or value.get("inventory_receipt_sha256")
        != _load(root / INVENTORY_PATH).get("receipt_sha256")
        or value.get("server_create_limit") != 1
        or value.get("statistical_cells_selected") != 0
        or value.get("api_mutations") != 0
        or value.get("explicit_root_review_required") is not True
        or value.get("qualification_only") is not True
        or value.get("fresh_live_gate_required_immediately_before_any_create") is not True
        or value.get("prompts_traces_flags_or_scores_included") is not False
    ):
        raise ValueError("early DP8 release is not held")


def load_all(root: Path) -> tuple[dict[str, Any], ...]:
    config = _load(root / CONFIG_PATH)
    plan = _load(root / PLAN_PATH)
    preview = _load(root / PREVIEW_PATH)
    inventory = _load(root / INVENTORY_PATH)
    release = _load(root / RELEASE_PATH)
    priority = _load(root / PRIORITY_CONTRACT_PATH)
    validate_config(config, root)
    validate_priority_contract(priority)
    validate_plan(plan, root)
    validate_preview(preview, root)
    validate_inventory(inventory)
    validate_held_release(release, root)
    return config, plan, preview, inventory, release


QUALIFICATION_LEVELS = ladder.LEVELS
