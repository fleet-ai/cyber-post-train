"""Offline conversion of the frozen study split into a reusable role anchor."""

from __future__ import annotations

from pathlib import Path

import pytest

from training import fleet_collection_anchor as anchor
from training.io import file_sha256

ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "configs/data/fleet-blackbox-current-study-split-20260914-v2.json"
INVENTORY = ROOT / "configs/data/fleet-blackbox-current-high-quality-20260914-v1.json"


def _config(output: Path) -> dict:
    return {
        "schema": anchor.REQUEST_SCHEMA,
        "legacy_split": {
            "path": SPLIT.relative_to(ROOT).as_posix(),
            "sha256": file_sha256(SPLIT),
        },
        "legacy_inventory": {
            "path": INVENTORY.relative_to(ROOT).as_posix(),
            "sha256": file_sha256(INVENTORY),
        },
        "output": str(output),
    }


def test_freezes_exact_legacy_roles_without_rebalancing(tmp_path: Path) -> None:
    output = tmp_path / "role-anchor.json"
    result = anchor.build(_config(output), relative_to=ROOT)
    assert result["submitted"] is False
    assert result["artifact_kind"] == "immutable_family_role_anchor"
    assert result["inherited_family_count"] == 75
    assert result["heldout_family_count"] == 25
    assert output.is_file()
    with pytest.raises(FileExistsError, match="already exists"):
        anchor.build(_config(output), relative_to=ROOT)


def test_rejects_inventory_reference_that_does_not_match_frozen_split(tmp_path: Path) -> None:
    config = _config(tmp_path / "role-anchor.json")
    config["legacy_inventory"] = dict(config["legacy_split"])
    with pytest.raises(ValueError, match="immutable inventory path|digest mismatch"):
        anchor.build(config, relative_to=ROOT)
