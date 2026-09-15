from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from evals.platform_v2.legacy_import import (
    DEFAULT_NAMESPACE,
    DEFAULT_REPOSITORY,
    FLEET_TEAM_ID,
    canonical_json,
    load_selection,
    task_tag,
)

SELECTION = Path("configs/data/fleet-a62-task-split-v1.json")
SPEC = Path("evals/platform_v2/fleet_a62_compositions_v1.json")
TAG_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")


def _load_spec() -> dict:
    return json.loads(SPEC.read_text())


def _membership(selection: dict, included_splits: set[str]) -> list[dict[str, str]]:
    seen: Counter[str] = Counter()
    rows = []
    for row in selection["tasks"]:
        seen[row["split"]] += 1
        if row["split"] in included_splits:
            rows.append(
                {
                    "split": row["split"],
                    "task_version_id": row["task_version_id"],
                    "source_tag": task_tag(row, seen[row["split"]]),
                }
            )
    return rows


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value)).hexdigest()


def test_composition_spec_binds_the_frozen_selection_and_controller_destination() -> None:
    selection = load_selection(SELECTION)
    spec = _load_spec()

    assert spec["schema"] == "fleet_platform_v2_cyber_compositions_v1"
    assert spec["source_selection"] == {
        "path": str(SELECTION),
        "file_sha256": "sha256:" + hashlib.sha256(SELECTION.read_bytes()).hexdigest(),
        "manifest_digest": selection["manifest_digest"],
        "source_job_id": selection["source"]["job_id"],
        "task_count": 160,
        "split_counts": {"train": 130, "dev": 10, "test": 20},
    }
    assert spec["source_imports"]["source_team_id"] == FLEET_TEAM_ID
    assert spec["source_imports"]["namespace"] == DEFAULT_NAMESPACE
    assert spec["source_imports"]["repository"] == DEFAULT_REPOSITORY
    assert spec["source_imports"]["required_state"] == "published"
    assert spec["destination"] == {
        "namespace": DEFAULT_NAMESPACE,
        "repository": DEFAULT_REPOSITORY,
        "create_destination": False,
        "advance_latest": False,
        "conflict_mode": "auto",
    }


def test_composition_identities_are_exact_disjoint_and_content_free() -> None:
    selection = load_selection(SELECTION)
    spec = _load_spec()
    compositions = {row["composition_id"]: row for row in spec["compositions"]}

    assert set(compositions) == {"all160", "train130", "dev10", "test20"}
    assert {
        key: (row["included_splits"], row["task_count"], row["version_tag"])
        for key, row in compositions.items()
    } == {
        "all160": (["train", "dev", "test"], 160, "all160-fb09668f-v1"),
        "train130": (["train"], 130, "train130-fb09668f-v1"),
        "dev10": (["dev"], 10, "dev10-fb09668f-v1"),
        "test20": (["test"], 20, "test20-fb09668f-v1"),
    }
    assert len({row["version_tag"] for row in compositions.values()}) == 4

    split_versions: dict[str, set[str]] = {}
    for split in ("train", "dev", "test"):
        split_versions[split] = {
            row["task_version_id"] for row in selection["tasks"] if row["split"] == split
        }
    assert not (split_versions["train"] & split_versions["dev"])
    assert not (split_versions["train"] & split_versions["test"])
    assert not (split_versions["dev"] & split_versions["test"])
    assert set().union(*split_versions.values()) == {
        row["task_version_id"] for row in selection["tasks"]
    }

    for row in compositions.values():
        members = _membership(selection, set(row["included_splits"]))
        assert len(members) == row["task_count"]
        assert _digest(members) == row["membership_sha256"]
        assert TAG_RE.fullmatch(row["version_tag"])
        assert row["version_tag"] != "latest"
        assert row["published_reference"] == (
            f"{DEFAULT_NAMESPACE}/{DEFAULT_REPOSITORY}:{row['version_tag']}"
        )

    public_identity = dict(spec)
    public_identity.pop("privacy")
    serialized = json.dumps(public_identity).lower()
    for forbidden in ("prompt", "trace", "score", "flag", "credential"):
        assert forbidden not in serialized


def test_composition_publication_contract_is_fail_closed() -> None:
    spec = _load_spec()
    assert spec["publication_protocol"] == {
        "preview_then_publish": True,
        "publish_only_previewed_request": True,
        "require_expected_root": True,
        "require_expected_revision": True,
        "require_expected_task_digest": True,
        "require_all_selected_imports_published": True,
        "allow_partial_composition": False,
    }
    assert spec["privacy"] == {
        "task_content_included": False,
        "prompts_included": False,
        "traces_included": False,
        "scores_included": False,
        "credentials_included": False,
    }


def test_composition_spec_self_digest_verifies() -> None:
    spec = _load_spec()
    expected = spec.pop("spec_sha256")
    assert expected == _digest(spec)
