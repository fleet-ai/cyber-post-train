import json
from pathlib import Path

import pytest

from evals.fleet import projected_runtime_plan as projected
from evals.fleet import self_hosted

SCHEMA = "fleet-qwen-generation17-bulk-executable-plan-v1"


def _plan() -> dict:
    body = {"schema_version": SCHEMA, "controller": "qwen-a"}
    return {**body, "plan_sha256": self_hosted.digest_without(body, "plan_sha256")}


def test_projected_symlink_is_materialized_as_regular_file(tmp_path: Path) -> None:
    backing = tmp_path / "..data" / "runtime-plan.json"
    backing.parent.mkdir()
    backing.write_text(json.dumps(_plan()) + "\n")
    source = tmp_path / "runtime-plan.json"
    source.symlink_to(backing)
    destination = tmp_path / "workspace" / "runtime-plan.json"

    assert projected.materialize(source, destination, schema=SCHEMA) == _plan()
    assert destination.is_file()
    assert not destination.is_symlink()
    assert json.loads(destination.read_text()) == _plan()


def test_materialization_rejects_digest_drift(tmp_path: Path) -> None:
    source = tmp_path / "runtime-plan.json"
    value = _plan()
    value["controller"] = "drifted"
    source.write_text(json.dumps(value))

    with pytest.raises(ValueError, match="self-digest drifted"):
        projected.materialize(source, tmp_path / "out.json", schema=SCHEMA)


def test_materialization_is_create_once(tmp_path: Path) -> None:
    source = tmp_path / "runtime-plan.json"
    source.write_text(json.dumps(_plan()))
    destination = tmp_path / "out.json"
    destination.write_text("preserved")

    with pytest.raises(FileExistsError):
        projected.materialize(source, destination, schema=SCHEMA)
    assert destination.read_text() == "preserved"

