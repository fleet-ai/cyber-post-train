from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


_SCRIPT = (
    Path(__file__).parents[1]
    / "skills"
    / "cyber-cluster-jobs-operator"
    / "scripts"
    / "check_contract.py"
)
_SPEC = importlib.util.spec_from_file_location("cluster_jobs_contract", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
validate = _MODULE.validate


def _document() -> dict:
    fields = {
        name: {"type": "string"}
        for name in (
            "image",
            "command",
            "env",
            "secrets",
            "resources",
            "priority_class",
            "privileged",
            "run_dir",
            "title",
        )
    }
    fields["workers"] = {"type": "integer", "minimum": 1}
    fields["gpus_per_worker"] = {"type": "integer", "minimum": 1, "maximum": 8}
    return {
        "paths": {
            "/v1/runs": {
                "post": {
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/RLJobConfig"}
                            }
                        }
                    }
                }
            }
        },
        "components": {
            "schemas": {
                "RLJobConfig": {
                    "description": "A generic job: an image, a command, and its shape.",
                    "required": ["image", "command"],
                    "properties": fields,
                }
            }
        },
    }


def test_accepts_general_live_contract() -> None:
    result = validate(_document())
    assert result["status"] == "PASSED"
    assert result["minimum_gpus_per_worker"] == 1


@pytest.mark.parametrize("missing", ["image", "command", "workers", "resources"])
def test_rejects_missing_general_job_fields(missing: str) -> None:
    document = _document()
    del document["components"]["schemas"]["RLJobConfig"]["properties"][missing]
    with pytest.raises(ValueError, match="missing general job fields"):
        validate(document)


def test_rejects_cpu_only_contract_change() -> None:
    document = _document()
    document["components"]["schemas"]["RLJobConfig"]["properties"]["gpus_per_worker"][
        "minimum"
    ] = 0
    with pytest.raises(ValueError, match="minimum changed"):
        validate(document)
