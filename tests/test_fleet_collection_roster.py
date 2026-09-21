"""Source-only regression tests for broad, anchored Fleet collection rosters."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training import collection_campaign as campaign
from training import fleet_collection_roster as roster
from training import task_family_split
from training.io import file_sha256


def _sha(character: str) -> str:
    return "sha256:" + character * 64


def _task(index: int, *, version: str = "a") -> dict:
    task_key = f"task-{index:03d}"
    version_id = f"00000000-0000-4000-8000-{index * 10 + (version == 'b'):012d}"
    return {
        "task_key": task_key,
        "task_version_id": version_id,
        "lineage": {
            "application": f"app-{index % 3}",
            "environment": f"env-{index % 3}",
            "difficulty": f"difficulty-{index % 3}",
            "vulnerability_family": [f"vulnerability-{index % 3}"],
            "task_family": f"family-{index:03d}",
        },
        "runtime": {
            "task_key": task_key,
            "task_version_id": version_id,
            "env_key": f"env-{index % 3}",
            "env_version": "v1",
            "environment_version_id": "00000000-0000-4001-8000-000000000001",
            "data_key": "commercial",
            "data_version": "v1",
        },
        "qualification_receipt_sha256": _sha("d"),
    }


def _write(path: Path, value: dict) -> dict:
    path.write_text(json.dumps(value, sort_keys=True))
    return {"path": path.name, "sha256": file_sha256(path)}


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict, dict, dict]:
    base = [_task(index) for index in range(12)]
    base_rows = [
        {
            "task_key": row["task_key"],
            "task_version_id": row["task_version_id"],
            "lineage": row["lineage"],
        }
        for row in base
    ]
    base_split = task_family_split.build(
        base_rows,
        inventory_sha256=_sha("1"),
        seed="base-roster-v1",
        ratios={"train": 0.5, "dev": 0.25, "final_test": 0.25},
        max_group_task_version_fraction=0.6,
    )
    anchor = task_family_split.freeze_role_anchor(base_split, base_rows)
    monkeypatch.setattr(
        task_family_split,
        "trusted_fleet_collection_root_anchor",
        lambda: json.loads(json.dumps(anchor)),
    )
    # Keep a second exact version for an old training family and add twelve
    # genuinely new families.  The old family must retain its original role.
    expanded = [*base, _task(0, version="b"), *[_task(index) for index in range(12, 24)]]
    supply = campaign.sealed(
        {
            "schema": roster.SUPPLY_SCHEMA,
            "catalog_snapshot_sha256": _sha("c"),
            "task_versions": [
                {"task_key": row["task_key"], "task_version_id": row["task_version_id"]}
                for row in expanded
            ],
        }
    )
    qualified = campaign.sealed(
        {
            "schema": roster.QUALIFIED_SCHEMA,
            "supply_catalog_sha256": supply["sha256"],
            "catalog_snapshot_sha256": supply["catalog_snapshot_sha256"],
            "task_validity_receipt_sha256": _sha("a"),
            "task_versions": expanded,
        }
    )
    config = {
        "schema": roster.REQUEST_SCHEMA,
        "supply_catalog": _write(tmp_path / "supply.json", supply),
        "qualified_catalog": _write(tmp_path / "qualified.json", qualified),
        "role_anchor": _write(tmp_path / "anchor.json", anchor),
        "seed": "broad-roster-v1",
        "ratios": {"train": 0.5, "dev": 0.25, "final_test": 0.25},
        "max_group_task_version_fraction": 0.6,
        "output": "roster-output",
    }
    return config, base_split, anchor


def test_render_preserves_old_roles_and_emits_generic_collection_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, base_split, _anchor = _fixture(tmp_path, monkeypatch)
    rendered = roster.render(config, relative_to=tmp_path)

    inventory = rendered["metadata-inventory.json"]
    bindings = rendered["runtime-bindings.json"]
    split = rendered["family-split.json"]
    lock = rendered["protected-family-lock.json"]
    receipt = rendered["ROSTER.json"]
    assert inventory["schema"] == campaign.INVENTORY_SCHEMA
    assert bindings["schema"] == campaign.RUNTIME_BINDINGS_SCHEMA
    assert split["schema"] == task_family_split.ANCHORED_SCHEMA
    assert receipt["outcome_blind"] is True
    assert receipt["external_actions_submitted"] is False
    assert receipt["family_split_sha256"] == split["sha256"]
    assert receipt["catalog_snapshot_sha256"] == _sha("c")
    assert lock["source_split_sha256"] == split["sha256"]
    assert rendered["role-anchor.json"]["sha256"] == receipt["role_anchor_sha256"]

    previous = {row["group_id"]: row["split"] for row in base_split["tasks"]}
    current = {row["group_id"]: row["split"] for row in split["tasks"]}
    assert {group: current[group] for group in previous} == previous
    assert len(inventory["task_versions"]) == 25
    assert not (
        {"prompt", "trace", "reasoning", "outcome"} & set(json.dumps(rendered).lower().split('"'))
    )


def test_build_is_create_once_and_returns_aggregate_only_handoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _base_split, _anchor = _fixture(tmp_path, monkeypatch)
    result = roster.build(config, relative_to=tmp_path)
    assert result["submitted"] is False
    assert result["artifact_kind"] == "metadata_roster_handoff_only"
    assert result["task_versions"] == 25
    assert (tmp_path / "roster-output" / "family-split.json").is_file()
    with pytest.raises(FileExistsError, match="already exists"):
        roster.build(config, relative_to=tmp_path)


def test_rejects_unqualified_or_outcome_bearing_catalog_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _base_split, _anchor = _fixture(tmp_path, monkeypatch)
    qualified_path = tmp_path / "qualified.json"
    qualified = json.loads(qualified_path.read_text())
    qualified["task_versions"][0]["task_version_id"] = "00000000-0000-4000-8000-999999999999"
    qualified = campaign.sealed({key: value for key, value in qualified.items() if key != "sha256"})
    config["qualified_catalog"] = _write(qualified_path, qualified)
    with pytest.raises(ValueError, match="absent from the supply"):
        roster.render(config, relative_to=tmp_path)

    config, _base_split, _anchor = _fixture(tmp_path, monkeypatch)
    qualified_path = tmp_path / "qualified.json"
    qualified = json.loads(qualified_path.read_text())
    qualified["task_versions"][0]["outcome"] = "success"
    qualified = campaign.sealed({key: value for key, value in qualified.items() if key != "sha256"})
    config["qualified_catalog"] = _write(qualified_path, qualified)
    with pytest.raises(ValueError, match="content or rollout outcomes|unknown or missing"):
        roster.render(config, relative_to=tmp_path)


def test_rejects_catalog_snapshot_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, _base_split, _anchor = _fixture(tmp_path, monkeypatch)
    qualified_path = tmp_path / "qualified.json"
    qualified = json.loads(qualified_path.read_text())
    qualified["catalog_snapshot_sha256"] = _sha("e")
    qualified = campaign.sealed({key: value for key, value in qualified.items() if key != "sha256"})
    config["qualified_catalog"] = _write(qualified_path, qualified)
    with pytest.raises(ValueError, match="different source catalog snapshot"):
        roster.render(config, relative_to=tmp_path)


def test_anchored_roster_can_render_only_train_tasks_for_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _base_split, _anchor = _fixture(tmp_path, monkeypatch)
    rendered = roster.render(config, relative_to=tmp_path)
    inventory = rendered["metadata-inventory.json"]
    split = rendered["family-split.json"]
    bindings = rendered["runtime-bindings.json"]
    request = {
        "schema": campaign.REQUEST_SCHEMA,
        "campaign_name": "q38-broad-self-v1",
        "source_kind": "self",
        "source_model": {
            "repository": "Qwen/Qwen3.8-27B",
            "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
            "session_model": "qwen/qwen3.8-27b",
        },
        "template_sha256": _sha("8"),
        "source_authorization_receipt_sha256": _sha("b"),
        "route": {
            "name": "source",
            "served_id": "qwen3.8-27b",
            "catalog": {"engine": "sglang", "precision": "bf16", "tensor_parallel_size": 1},
            "model_info": {
                "model_path": "/models/qwen",
                "model_type": "qwen3_5",
                "architectures": ["Qwen3_5ForConditionalGeneration"],
            },
            "server_info": {
                "model_path": "/models/qwen",
                "context_length": 262144,
                "tp_size": 1,
                "dp_size": 8,
                "load_balance_method": "total_tokens",
                "quantization": None,
                "kv_cache_dtype": "fp8_e4m3",
                "reasoning_parser": "qwen3",
                "tool_call_parser": "qwen3_coder",
            },
            "endpoint_origin": "https://inference.flt.build",
        },
        "harness": {
            "harness": "opencode",
            "harness_version": "1.18.27",
            "release_asset_sha256": _sha("c"),
            "provider_adapter": "@ai-sdk/openai-compatible",
            "context_management": campaign.ONLINE_COMPACTION,
            "context_window_size": 262144,
            "compaction_headroom_tokens": 20000,
            "max_output_tokens": 32768,
            "max_model_requests": 600,
            "timeout_seconds": 28800,
            "tools": ["bash", "submit_report"],
            "tool_catalog_sha256": _sha("d"),
        },
        "images": {"agent": _sha("e"), "proxy": _sha("f")},
        "sampling": {"temperature": 0.6, "top_p": 0.95, "seed": 42},
        "concurrency": 4,
        "attempts_per_task": 4,
        "target_unique_visible_action_tokens": campaign.MINIMUM_VISIBLE_TARGET_TOKENS,
        "reasoning_policy": campaign.VISIBLE_ACTIONS_ONLY,
        "offline_compaction_policy": campaign.OPAQUE_COMPACTION_REJECT,
    }
    collection = campaign.render(
        request,
        inventory,
        split,
        bindings,
        role_anchor=rendered["role-anchor.json"],
    )
    roles = {(row["task_key"], row["task_version_id"]): row["split"] for row in split["tasks"]}
    assert all(
        roles[(row["task_key"], row["task_version_id"])] == "train"
        for row in collection["task-selection.json"]["tasks"]
    )

    with pytest.raises(ValueError, match="anchored split and trusted role anchor"):
        campaign.render(request, inventory, split, bindings)


def test_rejects_a_forged_resealed_root_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _base_split, anchor = _fixture(tmp_path, monkeypatch)
    forged = json.loads(json.dumps(anchor))
    original = forged["roles"][0]["split"]
    forged["roles"][0]["split"] = next(
        role for role in ("train", "dev", "final_test") if role != original
    )
    forged["sha256"] = task_family_split.canonical_digest(
        {key: value for key, value in forged.items() if key != "sha256"}
    )
    config["role_anchor"] = _write(tmp_path / "forged-anchor.json", forged)
    with pytest.raises(ValueError, match="not the trusted root"):
        roster.render(config, relative_to=tmp_path)
