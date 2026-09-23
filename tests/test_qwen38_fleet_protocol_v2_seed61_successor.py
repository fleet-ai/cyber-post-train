"""Regressions for the score-blind seed-61 whole-pair successor."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from evals.fleet import heldout_launch
from scripts import prepare_qwen38_fleet_protocol_v2_replacements as protocol_v2
from scripts import prepare_qwen38_fleet_protocol_v2_seed61_successor as successor
from scripts import prepare_qwen38_fleet_seed46_pass8_packets as source


def _canonical(value: dict) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    )


def _predecessor() -> tuple[dict, dict]:
    mapping = [
        {"invalid_original_seed": seed, "replacement_seed": replacement}
        for seed, replacement in zip(protocol_v2.FROZEN_INVALID_SEEDS, range(54, 61), strict=True)
    ]
    definition = {
        "schema": protocol_v2.COMPARISON_DEFINITION_SCHEMA,
        "protocol_study_id": protocol_v2.PROTOCOL_V2_STUDY_ID,
        "replacement_mapping": mapping,
        "included_seeds": [48, 54, 55, 56, 57, 58, 59, 60],
        "replica_protocols": [
            {
                "seed": seed,
                "origin": "retained_original" if seed == 48 else "whole_pair_replacement",
                "protocol_id": f"protocol-{seed}",
                "comparison_protocol_sha256": "sha256:" + f"{seed:064x}",
            }
            for seed in (48, 54, 55, 56, 57, 58, 59, 60)
        ],
        "sessions_per_arm": 136,
        "total_sessions": 272,
        "later_invalid_seed_policy": "versioned successor required",
    }
    definition["sha256"] = _canonical(definition)
    receipt = {
        "sha256": successor.PREDECESSOR_RECEIPT_SHA256,
        "included_seeds": definition["included_seeds"],
        "excluded_original_seeds": list(protocol_v2.FROZEN_INVALID_SEEDS),
        "capacity": {
            "actual_started_rollouts_today": 112,
            "new_replacement_rollouts": 238,
            "projected_rollouts_after_reservation": 350,
            "daily_rollout_cap": 500,
        },
    }
    return receipt, definition


def test_checked_in_seed56_evidence_is_exact_and_score_blind() -> None:
    evidence = successor._evidence()  # noqa: SLF001
    assert evidence == {
        "pre_snapshot": {
            "path": (
                "docs/evidence/qwen38-study/"
                "2026-09-23-q38-dev17-seed56-protocol-v2-retirement-pre.json"
            ),
            "sha256": successor.RETIREMENT_PRE_SHA256,
            "file_sha256": successor.RETIREMENT_PRE_FILE_SHA256,
        },
        "post_receipt": {
            "path": (
                "docs/evidence/qwen38-study/"
                "2026-09-23-q38-dev17-seed56-protocol-v2-retirement-receipt.json"
            ),
            "sha256": successor.RETIREMENT_POST_SHA256,
            "file_sha256": successor.RETIREMENT_POST_FILE_SHA256,
        },
        "reason_class": "post_claim.connecterror_without_local_result_or_session",
    }


def test_prepare_replaces_only_complete_seed56_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    predecessor_receipt, predecessor_definition = _predecessor()
    monkeypatch.setattr(
        successor,
        "_predecessor",
        lambda _path: (predecessor_receipt, predecessor_definition),
    )
    monkeypatch.setattr(
        source.shared,
        "_live_parity",
        lambda *_args, **_kwargs: {
            "receipt_sha256": "sha256:" + "a" * 64,
            "observed_at": "2026-09-23T09:00:00Z",
        },
    )
    live = tmp_path / "live-parity.json"
    live.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "seed61"
    receipt = successor.prepare(
        predecessor_packets=tmp_path / "unused-predecessor",
        live_parity=live,
        output=output,
        now=datetime(2026, 9, 23, 9, tzinfo=UTC),
    )

    assert receipt["sha256"] == _canonical(
        {key: item for key, item in receipt.items() if key != "sha256"}
    )
    assert receipt["included_seeds"] == list(successor.INCLUDED_SEEDS)
    assert receipt["lineage"] == {
        "invalid_original_seed": 49,
        "superseded_replacement_seed": 56,
        "successor_seed": 61,
        "whole_pair_excluded": True,
        "cell_level_replacement_forbidden": True,
    }
    assert receipt["capacity"] == {
        "daily_rollout_cap": 500,
        "predecessor_actual_started_at_census": 112,
        "predecessor_reserved_rollouts": 238,
        "predecessor_projected_after_reservation": 350,
        "predecessor_reservation_already_includes_seed56_pair": True,
        "excluded_seed56_reservation_credit_claimed": 0,
        "new_seed61_rollouts": 34,
        "projected_after_seed61_reservation": 384,
        "remaining_after_seed61_reservation": 116,
        "within_daily_cap": True,
    }
    assert receipt["privacy"]["score_values_read"] is False
    assert receipt["external_mutations"] == 0
    assert receipt["server_preview_performed"] is False
    assert receipt["launch_performed"] is False
    assert {row["arm_id"] for row in receipt["replacement_arms"]} == {
        "base",
        "candidate",
    }

    definition = receipt["comparison_definition"]
    assert definition["schema"] == successor.COMPARISON_SCHEMA
    assert definition["included_seeds"] == list(successor.INCLUDED_SEEDS)
    assert definition["excluded_protocol_v2_seeds"] == [56]
    assert definition["superseded_replacement"]["successor_seed"] == 61
    assert [row["seed"] for row in definition["replica_protocols"]] == list(
        successor.INCLUDED_SEEDS
    )

    task_rosters = []
    protocol_digests = set()
    for arm in protocol_v2.ARMS:
        package = heldout_launch.build_package(output / "seed61" / arm / "LAUNCH_PACKET.json")
        task_rosters.append(
            next(iter(package.evaluation_config["routes"].values()))["task_versions"]
        )
        protocol_digests.add(package.packet.identity["comparison_protocol_sha256"])
        assert package.packet.identity["sampling_seed"] == 61
        assert package.packet.identity["pass_k"] == 1
        assert package.packet.identity["retry_limit"] == 0
        assert package.evaluation_config["max_reviewed_infrastructure_retries"] == 0
        assert package.job["metadata"]["annotations"]["fleet.ai/failure-alerts"] == "off"
        pod = package.job["spec"]["template"]["spec"]
        assert pod["automountServiceAccountToken"] is False
        assert pod["priorityClassName"] == "c1"
        assert (
            package.job["spec"]["template"]["metadata"]["labels"][
                "cyber-post-train.fleet.ai/postgres-client"
            ]
            == "true"
        )
        assert "nvidia.com/gpu" not in json.dumps(package.job)
    assert task_rosters[0] == task_rosters[1]
    assert len(task_rosters[0]) == 17
    assert len(protocol_digests) == 1


def test_definition_rejects_non_pass8_predecessor() -> None:
    _receipt, definition = _predecessor()
    definition["sessions_per_arm"] = 119
    protocol = {"protocol_id": "seed61", "sha256": "sha256:" + "a" * 64}
    with pytest.raises(ValueError, match="exact pass@8"):
        successor._definition(definition, protocol)  # noqa: SLF001
