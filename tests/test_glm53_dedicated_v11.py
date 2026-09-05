from pathlib import Path

import pytest
import yaml

from evals.fleet import glm53_dedicated_v11 as v11
from evals.fleet import glm53_dedicated_v11_preview as preview
from evals.fleet import self_hosted

ROOT = Path(__file__).resolve().parents[1]


def test_v11_is_held_and_uses_exact_required_topology() -> None:
    value = v11.spec(ROOT)
    request = v11.payload(value, ROOT)
    assert value["status"] == "PREVIEW_ONLY"
    assert value["launch_authorized"] is False
    assert request["topology_mode"] == "required"
    assert "topology_level" not in request
    assert request["workers"] == 1
    assert request["gpus_per_worker"] == 8
    assert request["priority_class"] == "fleet-infra-quiet"
    assert request["privileged"] is True
    assert set(request) == v11.EXPECTED_API_FIELDS


def test_v11_preserves_exact_model_image_and_tp8_arguments() -> None:
    value = v11.spec(ROOT)
    assert value["model"]["revision"] == v11.MODEL_REVISION
    assert value["model"]["context_length"] == 262144
    assert value["runtime"]["image"] == v11.IMAGE
    assert value["server_arguments_sha256"] == v11.SERVER_ARGUMENTS_SHA256
    assert "--tp-size" in v11.SERVER_ARGUMENTS
    assert v11.SERVER_ARGUMENTS[v11.SERVER_ARGUMENTS.index("--tp-size") + 1] == "8"


def test_v11_rejects_two_node_shape_under_single_node_entrypoint() -> None:
    value = v11.spec(ROOT)
    value["resources"]["workers"] = 2
    with pytest.raises(ValueError, match="resource policy drifted"):
        v11.validate(value, ROOT)
    assert v11.spec(ROOT)["two_node_tp16"]["allowed"] is False


def test_v11_identity_is_fresh_from_v8() -> None:
    assert v11.TITLE.endswith("-v11")
    assert v11.RUN_DIR.endswith("-v11")
    assert "-v8" not in v11.TITLE
    assert "-v8" not in v11.RUN_DIR


def _manifest(request: dict, *, mode: str = "required", workers: int = 1) -> str:
    annotations = {f"kueue.x-k8s.io/podset-{mode}-topology": v11.TOPOLOGY_LEVEL}
    pod = {
        "metadata": {"annotations": annotations},
        "spec": {
            "priorityClassName": "fleet-infra-quiet",
            "imagePullSecrets": [{"name": "ghcr-pull"}],
            "containers": [
                {
                    "image": v11.IMAGE,
                    "env": [{"name": "RUN_DIR", "value": v11.RUN_DIR}],
                    "securityContext": {"privileged": True},
                    "resources": {
                        "requests": {"nvidia.com/gpu": 8},
                        "limits": {"nvidia.com/gpu": 8},
                    },
                }
            ],
        },
    }
    cluster = {"headGroupSpec": {"template": pod}, "workerGroupSpecs": []}
    if workers == 2:
        cluster["workerGroupSpecs"] = [{"template": pod}]
    return yaml.safe_dump(
        {
            "apiVersion": "ray.io/v1",
            "kind": "RayJob",
            "metadata": {"labels": {"kueue.x-k8s.io/queue-name": v11.QUEUE}},
            "spec": {
                "entrypoint": request["command"],
                "suspend": True,
                "rayClusterSpec": cluster,
            },
        }
    )


def test_preview_identity_requires_exact_rendered_tas_and_single_node() -> None:
    request = v11.payload(v11.spec(ROOT), ROOT)
    assert preview.preview_identity(_manifest(request), request) == {
        "image": v11.IMAGE,
        "priority_class": "fleet-infra-quiet",
        "privileged": True,
        "run_dir": v11.RUN_DIR,
        "gpus": 8,
        "image_pull_secrets": ["ghcr-pull"],
        "queue": "training-lq",
        "suspended_for_admission": True,
        "topology_mode": "required",
        "topology_level": v11.TOPOLOGY_LEVEL,
        "command_sha256": self_hosted.sha256(request["command"].encode()),
    }
    with pytest.raises(RuntimeError, match="rendered identity drifted"):
        preview.preview_identity(_manifest(request, mode="preferred"), request)
    with pytest.raises(RuntimeError, match="unexpectedly rendered a worker group"):
        preview.preview_identity(_manifest(request, workers=2), request)


def test_contract_has_topology_fields_but_no_queue_or_flavor_selector() -> None:
    openapi = {
        "paths": {
            "/v1/runs": {},
            "/v1/runs/preview": {},
            "/v1/runs/{name}": {"delete": {}},
        },
        "components": {
            "schemas": {
                "RLJobConfig": {
                    "properties": {
                        "topology_mode": {
                            "anyOf": [{"enum": ["required", "preferred"]}, {"type": "null"}]
                        },
                        "topology_level": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    }
                }
            }
        },
    }
    preview.require_contract(openapi)
    openapi["components"]["schemas"]["RLJobConfig"]["properties"]["queue"] = {"type": "string"}
    with pytest.raises(RuntimeError, match="newly exposed queue"):
        preview.require_contract(openapi)
