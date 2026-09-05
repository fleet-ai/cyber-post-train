from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from evals.fleet import qwen38_dedicated_dp8_v1 as dp8
from evals.fleet import qwen38_dedicated_dp8_v1_live as live
from evals.fleet import self_hosted

ROOT = Path(__file__).resolve().parents[1]


def test_dp8_preserves_exact_model_and_harness_parsers() -> None:
    value = dp8.spec(ROOT)
    payload = dp8.payload(value, ROOT)
    assert value["model"]["revision"] == "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    assert value["model"]["context_length"] == 262144
    assert value["runtime"]["reasoning_parser"] == "qwen3"
    assert value["runtime"]["tool_call_parser"] == "qwen3_coder"
    assert "--tp-size 1" in payload["command"]
    assert "--dp-size 8" in payload["command"]
    assert "--load-balance-method total_tokens" in payload["command"]
    assert "--enable-dp-attention" not in payload["command"]


def test_dp8_uses_one_full_node_and_never_preempts() -> None:
    payload = dp8.payload(dp8.spec(ROOT), ROOT)
    assert payload["workers"] == 1
    assert payload["gpus_per_worker"] == 8
    assert payload["priority_class"] == "fleet-infra-quiet"
    assert payload["privileged"] is False
    assert dp8.spec(ROOT)["resources"]["preemption_policy"] == "Never"


def test_dp8_scope_forbids_scoring_before_separate_release() -> None:
    evaluation = dp8.spec(ROOT)["evaluation"]
    assert evaluation["fresh_actual_opencode_parity_required_before_scoring"] is True
    assert evaluation["scored_tasks_allowed"] is False
    assert evaluation["scored_successor_requires_separate_release"] is True


def test_project_shape_counts_active_tp1_peer_and_rejects_unknown() -> None:
    shape = live._project_shape(
        [{
            "name": "ft-run-peer",
            "run_dir": "/mnt/sfs/jobs/chris-cyber-evalserve-q38-tp1-b-v2",
            "status": "RUNNING",
        }]
    )
    assert shape["planned_nodes"] == 2
    assert shape["planned_gpus"] == 9
    with pytest.raises(RuntimeError, match="unknown active"):
        live._project_shape(
            [
                {
                    "name": "ft-run-other",
                    "run_dir": "/mnt/sfs/jobs/chris-cyber-evalserve-other",
                    "status": "Running",
                }
            ]
        )


def test_live_serving_inventory_rejects_stale_history_rows() -> None:
    client = Mock()
    gone = Mock(status_code=404)
    live_response = Mock(status_code=200)
    live_response.raise_for_status.return_value = None
    live_response.json.return_value = {
        "name": "ft-run-live",
        "run_dir": "/mnt/sfs/jobs/chris-cyber-evalserve-q38-tp1-b-v2",
        "status": "RUNNING",
    }
    client.get.side_effect = [gone, live_response]
    rows = [
        {
            "name": "ft-run-gone",
            "run_dir": "/mnt/sfs/jobs/chris-cyber-evalserve-q38-tp1-a-v3",
            "status": "submitted",
        },
        {
            "name": "ft-run-live",
            "run_dir": "/mnt/sfs/jobs/chris-cyber-evalserve-q38-tp1-b-v2",
            "status": "RUNNING",
        },
    ]
    assert live._live_serving_runs(client, rows) == [
        {
            "name": "ft-run-live",
            "run_dir": "/mnt/sfs/jobs/chris-cyber-evalserve-q38-tp1-b-v2",
            "status": "RUNNING",
        }
    ]


def test_preview_requires_exact_full_node_render() -> None:
    payload = dp8.payload(dp8.spec(ROOT), ROOT)
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
                            "imagePullSecrets": [{"name": "ghcr-pull"}],
                            "priorityClassName": "fleet-infra-quiet",
                            "containers": [{
                                "image": payload["image"],
                                "env": [{"name": "RUN_DIR", "value": payload["run_dir"]}],
                                "resources": {
                                    "requests": {"nvidia.com/gpu": 8},
                                    "limits": {"nvidia.com/gpu": 8},
                                },
                            }],
                        },
                    }
                }
            },
        },
    }
    rendered = live._preview_identity(yaml.safe_dump(manifest), payload)
    assert rendered["gpus"] == 8
    assert rendered["command_sha256"] == self_hosted.sha256(payload["command"].encode())


def test_lifecycle_requires_all_eight_exact_b300s() -> None:
    source = (ROOT / dp8.LIFECYCLE_PATH).read_text()
    assert "len(rows) == 8" in source
    assert "list(range(8))" in source
    assert "NVIDIA B300 SXM6 PC" in source
    assert "memory_mib\"] == 275040" in source
    assert "IDLE_SECONDS=600" in source
