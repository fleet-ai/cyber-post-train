from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from evals.fleet import exact_pass4_universe as exact
from evals.fleet import self_hosted

CAMPAIGN = Path("evals/fleet/configs/q38-glm53-exact-easiest100-pass4-campaign-v1.json")


def _campaign() -> dict:
    return exact.read_object(CAMPAIGN)


def _universe() -> dict:
    return exact.build_universe(_campaign(), Path.cwd())


def _tombstone(cell: dict, generation: int = 1) -> dict:
    execution = exact.execution_for(cell["cell_id"], generation)
    value = {
        "schema_version": exact.TOMBSTONE_SCHEMA,
        "cell_id": cell["cell_id"],
        "execution_id": execution["execution_id"],
        "execution_generation": generation,
        "classification": "pre_model_infrastructure_failure",
        "terminal": True,
        "model_called": False,
        "verifier_called": False,
        "session_created": False,
        "authoritative_outcome_created": False,
        "retry_allowed": True,
        "evidence_receipt_sha256": "sha256:" + "a" * 64,
    }
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    return value


def test_exact_universe_is_100_tasks_per_model_pass4_and_disjoint() -> None:
    universe = _universe()
    assert universe["cell_count"] == 800
    assert universe["model_counts"] == {"qwen3.8-27b": 400, "glm-5.3": 400}
    assert universe["task_counts"] == {"qwen3.8-27b": 100, "glm-5.3": 100}
    assert universe["attempts"] == [1, 2, 3, 4]
    assert {row["treatment"]["compaction_headroom_tokens"] for row in universe["cells"]} == {20000}
    assert {row["selection_rank"] for row in universe["cells"]} == set(range(1, 101))
    assert all("source_rank" not in row for row in universe["cells"])
    assert len({row["cell_id"] for row in universe["cells"]}) == 800
    assert (
        len({(row["model"], row["task_version_id"], row["attempt"]) for row in universe["cells"]})
        == 800
    )
    qwen_versions = {
        row["task_version_id"] for row in universe["cells"] if row["model"] == "qwen3.8-27b"
    }
    glm_versions = {
        row["task_version_id"] for row in universe["cells"] if row["model"] == "glm-5.3"
    }
    assert qwen_versions == glm_versions


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    [
        (("selection", "source_job_id"), "wrong", "selection provenance"),
        (("selection", "ranking"), ["task_key_asc"], "selection provenance"),
        (("selection", "source_aggregate"), {"task_count": 1}, "selection provenance"),
        (("selection", "eligibility"), {}, "selection provenance"),
        (("models", "qwen3.8-27b", "revision"), "main", "model bindings"),
        (("treatment", "tools"), ["bash"], "treatment binding"),
    ],
)
def test_campaign_provenance_and_treatment_drift_fail_closed(
    path: tuple[str, ...], replacement: object, message: str
) -> None:
    campaign = copy.deepcopy(_campaign())
    target = campaign
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    with pytest.raises(ValueError, match=message):
        exact.build_universe(campaign, Path.cwd())


def test_unknown_campaign_fields_fail_closed() -> None:
    campaign = _campaign()
    campaign["replacement_tasks"] = []
    with pytest.raises(ValueError, match="campaign fields drifted"):
        exact.build_universe(campaign, Path.cwd())


def test_selection_file_digest_and_semantic_digest_are_both_enforced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = copy.deepcopy(_campaign())
    selection = exact.read_object(Path(campaign["selection"]["path"]))
    selection["tasks"][0]["task_version"] = "drift"
    copied = tmp_path / "selection.json"
    copied.write_text(json.dumps(selection))
    copied_sha = "sha256:" + hashlib.sha256(copied.read_bytes()).hexdigest()
    monkeypatch.setattr(exact, "EXPECTED_SELECTION_PATH", "selection.json")
    monkeypatch.setattr(exact, "EXPECTED_SELECTION_FILE_SHA256", copied_sha)
    campaign["selection"]["path"] = "selection.json"
    campaign["selection"]["file_sha256"] = copied_sha
    with pytest.raises(ValueError, match="semantic digest mismatch"):
        exact.build_universe(campaign, tmp_path)


def test_execution_generation_does_not_change_statistical_cell_identity() -> None:
    cell = _universe()["cells"][0]
    generation1 = exact.next_execution(cell, [])
    tombstone1 = _tombstone(cell)
    generation2 = exact.next_execution(cell, [tombstone1])
    assert generation1["cell_id"] == generation2["cell_id"] == cell["cell_id"]
    assert generation1["execution_generation"] == 1
    assert generation2["execution_generation"] == 2
    assert generation1["execution_id"] != generation2["execution_id"]


@pytest.mark.parametrize(
    "field",
    [
        "model_called",
        "verifier_called",
        "session_created",
        "authoritative_outcome_created",
    ],
)
def test_any_stochastic_or_authoritative_side_effect_forbids_retry(field: str) -> None:
    cell = _universe()["cells"][0]
    tombstone = _tombstone(cell)
    tombstone[field] = True
    tombstone["receipt_sha256"] = self_hosted.digest_without(tombstone, "receipt_sha256")
    with pytest.raises(ValueError, match="not a retry-safe"):
        exact.next_execution(cell, [tombstone])


def test_retry_requires_valid_digest_exact_cell_and_contiguous_generations() -> None:
    cell = _universe()["cells"][0]
    bad_digest = _tombstone(cell)
    bad_digest["receipt_sha256"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="digest mismatch"):
        exact.next_execution(cell, [bad_digest])

    wrong_cell = _tombstone(cell)
    wrong_cell["cell_id"] = "sha256:" + "b" * 64
    wrong_cell["receipt_sha256"] = self_hosted.digest_without(wrong_cell, "receipt_sha256")
    with pytest.raises(ValueError, match="not a retry-safe"):
        exact.next_execution(cell, [wrong_cell])

    generation2 = _tombstone(cell, generation=2)
    with pytest.raises(ValueError, match="not contiguous"):
        exact.next_execution(cell, [generation2])
