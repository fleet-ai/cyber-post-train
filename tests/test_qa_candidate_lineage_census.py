import hashlib
import json
from pathlib import Path

from evals.fleet import qa_candidate_lineage_census as census
from training import task_family_split

ROOT = Path(__file__).resolve().parents[1]


def selected(key: str = "task") -> dict:
    return {
        "task_id": "task-id",
        "task_key": key,
        "task_version_id": "version-id",
        "qa_status": "clean",
    }


def live_task(key: str = "task") -> dict:
    return {
        "key": key,
        "eval_task_version_id": "version-id",
        "metadata": {
            "projection_id": "blackbox_ctf_v1",
            "task_graph_band": "medium",
            "cyber_subject": {
                "task_graph_id": "composite-a-b",
                "source_locator": ("cyber/task-graphs/composite-a-b@4:task_graph_source"),
                "atom_sources": [
                    {
                        "artifact_key": "cyber/atoms/app/a",
                        "locator": "cyber/atoms/app/a@1:atom_source",
                        "version_index": 1,
                    },
                    {
                        "artifact_key": "cyber/atoms/app/b",
                        "locator": "cyber/atoms/app/b@8:atom_source",
                        "version_index": 8,
                    },
                ],
            },
        },
        "prompt": "must never enter the projected census",
    }


def test_projection_keeps_exact_lineage_and_drops_task_content() -> None:
    value = census.project_lineage(live_task(), selected())
    assert value["atom_artifact_keys"] == ["cyber/atoms/app/a", "cyber/atoms/app/b"]
    assert value["atom_source_locators"] == [
        "cyber/atoms/app/a@1:atom_source",
        "cyber/atoms/app/b@8:atom_source",
    ]
    assert value["lineage"]["task_family"] == "cyber/task-graphs/composite-a-b"
    assert "prompt" not in str(value).lower()
    assert value["lineage_binding_sha256"] == task_family_split.canonical_digest(
        {key: item for key, item in value.items() if key != "lineage_binding_sha256"}
    )


def test_projection_rejects_exact_version_drift() -> None:
    task = live_task()
    task["eval_task_version_id"] = "different"
    try:
        census.project_lineage(task, selected())
    except ValueError as error:
        assert "version differs" in str(error)
    else:
        raise AssertionError("exact version drift was accepted")


def test_projection_rejects_task_graph_prefix_confusion() -> None:
    task = live_task()
    task["metadata"]["cyber_subject"]["source_locator"] = (
        "cyber/task-graphs/composite-a-b@alias@4:task_graph_source"
    )
    try:
        census.project_lineage(task, selected())
    except ValueError as error:
        assert "graph identity is inconsistent" in str(error)
    else:
        raise AssertionError("a prefixed but different task-graph key was accepted")


def test_projection_rejects_noncanonical_source_version() -> None:
    task = live_task()
    task["metadata"]["cyber_subject"]["atom_sources"][0]["locator"] = (
        "cyber/atoms/app/a@01:atom_source"
    )
    try:
        census.project_lineage(task, selected())
    except ValueError as error:
        assert "not canonical" in str(error)
    else:
        raise AssertionError("a noncanonical Registry locator was accepted")


def test_checked_in_live_census_binds_the_exact_projection_source() -> None:
    path = ROOT / "configs/data/fleet-blackbox-qa33-live-lineage-20260924-v1.json"
    value = json.loads(path.read_text())
    controller = ROOT / "evals/fleet/qa_candidate_lineage_census.py"
    assert value["source"]["controller_file_sha256"] == (
        "sha256:" + hashlib.sha256(controller.read_bytes()).hexdigest()
    )
    assert value["counts"] == {
        "candidate_task_versions": 33,
        "exact_lineage_bound": 33,
        "runtime_receipt_qualified": 0,
    }
    assert value["sha256"] == task_family_split.canonical_digest(
        {key: item for key, item in value.items() if key != "sha256"}
    )
