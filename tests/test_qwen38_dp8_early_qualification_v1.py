from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from evals.fleet import qwen38_dp8_early_qualification_v1 as early
from evals.fleet import self_hosted

ROOT = Path(__file__).resolve().parents[1]


def _resign(value: dict[str, object], field: str = "receipt_sha256") -> None:
    value[field] = self_hosted.digest_without(value, field)


def test_held_packet_is_valid_and_authorizes_nothing() -> None:
    config, plan, preview, inventory, release = early.load_all(ROOT)
    assert config["launch_authorized"] is False
    assert plan["scoring_authorized"] is False
    assert release["launch_authorized"] is False
    assert release["statistical_cells_selected"] == 0
    assert preview["api_mutations"] == 0
    assert inventory["gpu_nodes"] == 1
    assert inventory["projected_gpu_nodes_after_create"] == 2
    assert early.QUALIFICATION_LEVELS == (1, 2, 4, 8)
    assert plan["qualifier_controller"]["kubectl_access_required"] is False
    assert plan["qualifier_controller"]["controller_package_ready"] is True
    assert plan["qualifier_controller"]["kubectl_access_required"] is False
    assert plan["qualifier_controller"]["separate_uid_bound_qualifier_release_required"] is True


def test_jobs_payload_is_exact_one_node_eight_gpu_nonpreempting_server() -> None:
    payload = early.jobs_payload(ROOT)
    assert payload["title"] == "chris-cyber-evalserve-q38-dp8-c-v2"
    assert payload["run_dir"] == "/mnt/sfs/jobs/chris-cyber-evalserve-q38-dp8-c-v2"
    assert payload["workers"] == 1
    assert payload["gpus_per_worker"] == 8
    assert payload["priority_class"] == "fleet-infra-quiet"
    assert payload["privileged"] is False
    assert "--tp-size 1" in payload["command"]
    assert "--dp-size 8" in payload["command"]
    assert early.QUALIFIER_PRIORITY_CLASS == "fleet-serve-low"
    assert early.QUALIFIER_PRIORITY_VALUE == 100
    plan = early.load_all(ROOT)[1]
    assert plan["qualifier_controller"]["preemption_policy"] == "Never"
    priority = early._load(ROOT / early.PRIORITY_CONTRACT_PATH)
    early.validate_priority_contract(priority)
    assert priority["server_preview_attempts"][0]["http_status"] == 422
    assert priority["server_preview_attempts"][1]["http_status"] == 200
    assert priority["api_mutations"] == 0


def test_preview_parser_binds_rendered_identity() -> None:
    payload = early.jobs_payload(ROOT)
    manifest = {
        "kind": "RayJob",
        "metadata": {"labels": {"kueue.x-k8s.io/queue-name": "training-lq"}},
        "spec": {
            "suspend": True,
            "entrypoint": payload["command"],
            "rayClusterSpec": {
                "headGroupSpec": {
                    "template": {
                        "metadata": {
                            "annotations": {
                                "kueue.x-k8s.io/podset-preferred-topology": (
                                    "topology.nebius.com/tier-1"
                                )
                            }
                        },
                        "spec": {
                            "priorityClassName": "fleet-infra-quiet",
                            "containers": [
                                {
                                    "image": payload["image"],
                                    "env": [
                                        {"name": "RUN_DIR", "value": payload["run_dir"]}
                                    ],
                                    "resources": {
                                        "requests": {"nvidia.com/gpu": 8},
                                        "limits": {"nvidia.com/gpu": 8},
                                    },
                                }
                            ],
                        },
                    }
                }
            },
        },
    }
    assert early.preview_identity(yaml.safe_dump(manifest), ROOT)["gpus"] == 8
    manifest["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"][
        "containers"
    ][0]["resources"]["requests"]["nvidia.com/gpu"] = 4
    with pytest.raises(RuntimeError):
        early.preview_identity(yaml.safe_dump(manifest), ROOT)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("launch_authorized",), True),
        (("scoring_authorized",), True),
        (("coexistence_gate", "max_project_gpu_nodes"), 3),
        (("server", "model_revision"), "drifted"),
        (("server", "data_parallel_size"), 4),
        (("qualifier_controller", "cluster_dns_only_no_port_forward"), False),
        (("qualifier_controller", "kubectl_access_required"), True),
        (("qualification", "concurrency_ladder"), [8]),
        (("qualification", "statistical_cells_selected"), 1),
        (("post_qualification_gate", "all_four_statistical_cells_reserved_atomically"), False),
        (("lifecycle", "post_ready_idle_seconds"), 3600),
    ],
)
def test_held_plan_fails_closed(path: tuple[str, ...], replacement: object) -> None:
    value = copy.deepcopy(early._load(ROOT / early.PLAN_PATH))
    cursor = value
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = replacement
    _resign(value)
    with pytest.raises(ValueError):
        early.validate_plan(value, ROOT)


def test_inventory_and_release_fail_closed_on_authority_drift() -> None:
    inventory = copy.deepcopy(early._load(ROOT / early.INVENTORY_PATH))
    inventory["unknown_active_dedicated_runs"] = 1
    _resign(inventory)
    with pytest.raises(ValueError):
        early.validate_inventory(inventory)

    release = copy.deepcopy(early._load(ROOT / early.RELEASE_PATH))
    release["launch_authorized"] = True
    _resign(release)
    with pytest.raises(ValueError):
        early.validate_held_release(release, ROOT)


def test_score_free_packet_never_exposes_a_submit_verb() -> None:
    assert not hasattr(early, "submit")
    assert not hasattr(early, "submit_create_once")
