from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from cyber_post_train.cli import app
from cyber_post_train.spec import (
    build_source_spec,
    compile_plan,
    file_digest,
    load_locked_spec,
    lock_spec,
)


def _components(root: Path) -> dict[str, dict[str, str]]:
    result = {}
    for name in ("model", "serving", "harness", "dataset", "protocol"):
        path = root / f"{name}.json"
        path.write_text(json.dumps({"kind": name}) + "\n")
        result[name] = {"path": path.name}
    return result


def _source(root: Path) -> Path:
    source = root / "source.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "schema": "cyber_experiment_spec_v1",
                "experiment_id": "test-evaluation-v1",
                "kind": "evaluation",
                "adapter": "fleet",
                "components": _components(root),
                "execution": {
                    "backend": "fleet",
                    "output_root": "output/test-evaluation-v1",
                    "max_concurrency": 2,
                    "serving_block": "hosted",
                },
            },
            sort_keys=False,
        )
    )
    return source


def test_lock_validate_and_compile_are_deterministic(tmp_path: Path) -> None:
    source = _source(tmp_path)
    locked = lock_spec(source, root=tmp_path)
    locked_path = tmp_path / "locked.yaml"
    locked_path.write_text(yaml.safe_dump(locked, sort_keys=False))

    spec = load_locked_spec(locked_path, root=tmp_path)
    first = compile_plan(locked_path, root=tmp_path)
    second = compile_plan(locked_path, root=tmp_path)

    assert spec.experiment_id == "test-evaluation-v1"
    assert first == second
    assert first["plan_sha256"].startswith("sha256:")


def test_validation_detects_component_drift(tmp_path: Path) -> None:
    source = _source(tmp_path)
    locked = lock_spec(source, root=tmp_path)
    locked_path = tmp_path / "locked.yaml"
    locked_path.write_text(yaml.safe_dump(locked, sort_keys=False))
    (tmp_path / "model.json").write_text('{"changed":true}\n')

    with pytest.raises(ValueError, match="digest mismatch"):
        load_locked_spec(locked_path, root=tmp_path)


def test_training_rejects_external_benchmark_dataset(tmp_path: Path) -> None:
    for name in ("model", "trainer", "protocol"):
        (tmp_path / f"{name}.json").write_text("{}\n")
    benchmark = tmp_path / "webexploitbench-dataset.json"
    benchmark.write_text("{}\n")
    locked = {
        "schema": "cyber_experiment_spec_v1",
        "experiment_id": "bad-training-v1",
        "kind": "training",
        "adapter": "fleet-sft",
        "components": {
            name: {"path": path.name, "sha256": file_digest(path)}
            for name, path in {
                "model": tmp_path / "model.json",
                "trainer": tmp_path / "trainer.json",
                "protocol": tmp_path / "protocol.json",
                "dataset": benchmark,
            }.items()
        },
        "execution": {
            "backend": "training-jobs-api",
            "output_root": "output/bad-training-v1",
            "max_concurrency": 1,
        },
    }
    path = tmp_path / "locked.yaml"
    path.write_text(yaml.safe_dump(locked, sort_keys=False))

    with pytest.raises(ValueError, match="external benchmark"):
        load_locked_spec(path, root=tmp_path)


def test_cli_lock_is_create_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _source(tmp_path)
    output = tmp_path / "locked.yaml"
    monkeypatch.setattr(
        "cyber_post_train.cli.lock_spec", lambda path: lock_spec(path, root=tmp_path)
    )

    first = CliRunner().invoke(app, ["experiment", "lock", str(source), "--output", str(output)])
    second = CliRunner().invoke(app, ["experiment", "lock", str(source), "--output", str(output)])

    assert first.exit_code == 0
    assert second.exit_code != 0


def test_build_source_spec_selects_components_by_adapter() -> None:
    evaluation = build_source_spec(
        experiment_id="qwen38-fleet-p4-v1",
        adapter="fleet",
        component_paths={
            "model": "configs/components/model.json",
            "serving": "configs/components/serving.json",
            "harness": "configs/components/harness.json",
            "dataset": "configs/components/dataset.json",
            "protocol": "configs/components/protocol.json",
            "trainer": None,
        },
        backend="fleet",
        output_root="output/qwen38-fleet-p4-v1",
        max_concurrency=2,
        serving_block="hosted",
    )
    assert evaluation["kind"] == "evaluation"
    assert set(evaluation["components"]) == {
        "model",
        "serving",
        "harness",
        "dataset",
        "protocol",
    }


def test_cli_init_is_create_once(tmp_path: Path) -> None:
    output = tmp_path / "experiment.yaml"
    args = [
        "experiment",
        "init",
        "qwen38-fleet-p4-v1",
        "--output",
        str(output),
        "--adapter",
        "fleet",
        "--model",
        "configs/components/model.json",
        "--serving",
        "configs/components/serving.json",
        "--harness",
        "configs/components/harness.json",
        "--dataset",
        "configs/components/dataset.json",
        "--protocol",
        "configs/components/protocol.json",
        "--serving-block",
        "hosted",
    ]
    first = CliRunner().invoke(app, args)
    second = CliRunner().invoke(app, args)

    assert first.exit_code == 0
    assert second.exit_code != 0
    value = yaml.safe_load(output.read_text())
    assert value["adapter"] == "fleet"
    assert value["execution"]["serving_block"] == "hosted"
