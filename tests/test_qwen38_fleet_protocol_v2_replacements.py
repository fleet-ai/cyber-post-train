"""Regressions for score-blind whole-replica protocol-v2 replacements."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from evals.fleet import heldout_launch
from scripts import prepare_qwen38_fleet_protocol_v2_replacements as replacements
from scripts import prepare_qwen38_fleet_seed46_pass8_packets as source
from scripts import render_qwen38_fleet_protocol_v2_replacement_launchers as launchers


def _canonical(value: dict) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    )


def _source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    live = tmp_path / "source-live-parity.json"
    live.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        source.shared,
        "_live_parity",
        lambda *_args, **_kwargs: {
            "receipt_sha256": "sha256:" + "a" * 64,
            "observed_at": "2026-09-23T00:00:00Z",
        },
    )
    output = tmp_path / "source-packets"
    source.prepare(
        output=output,
        live_parity=live,
        now=datetime(2026, 9, 23, tzinfo=UTC),
    )
    return output


def _intent(
    tmp_path: Path,
    source_packets: Path,
    *,
    seeds: tuple[int, ...] = (47, 51, 52, 53),
) -> Path:
    receipt_path = source_packets / "PREPARATION_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    reasons = {
        47: "terminal_replica_incomplete",
        51: "mixed_infrastructure_invalid_replica",
        52: "mixed_infrastructure_invalid_replica",
        53: "replica_not_started",
    }
    body = {
        "schema": replacements.INTENT_SCHEMA,
        "source_protocol_study_id": source.PROTOCOL_STUDY_ID,
        "source_preparation_receipt_sha256": receipt["sha256"],
        "source_preparation_receipt_file_sha256": replacements._file_sha(receipt_path),  # noqa: SLF001
        "invalid_original_replicas": [
            {
                "seed": seed,
                "reason_class": reasons.get(seed, "terminal_replica_incomplete"),
                "evidence_receipt_sha256s": [
                    (
                        json.loads(
                            replacements.SEED51_INVALID_EVIDENCE.read_text(encoding="utf-8")
                        )["sha256"]
                        if seed == 51
                        else "sha256:" + f"{seed:064x}"
                    )
                ],
            }
            for seed in seeds
        ],
        "score_unsealed": False,
        "daily_rollout_cap": replacements.DAILY_ROLLOUT_CAP,
    }
    value = {**body, "sha256": _canonical(body)}
    path = tmp_path / "migration-intent.json"
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, dict]:
    source_packets = _source(tmp_path, monkeypatch)
    intent = _intent(tmp_path, source_packets)
    live = tmp_path / "replacement-live-parity.json"
    live.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "replacement-packets"
    receipt = replacements.prepare(
        source_packets=source_packets,
        migration_intent=intent,
        live_parity=live,
        output=output,
        now=datetime(2026, 9, 23, 1, tzinfo=UTC),
    )
    return source_packets, output, receipt


def test_mapping_is_deterministic_and_never_reuses_source_seeds() -> None:
    assert replacements.deterministic_mapping([47, 51, 52, 53]) == [
        {"invalid_original_seed": 47, "replacement_seed": 54},
        {"invalid_original_seed": 51, "replacement_seed": 55},
        {"invalid_original_seed": 52, "replacement_seed": 56},
        {"invalid_original_seed": 53, "replacement_seed": 57},
    ]
    assert replacements.deterministic_mapping([47, 50, 52, 53]) == [
        {"invalid_original_seed": 47, "replacement_seed": 54},
        {"invalid_original_seed": 50, "replacement_seed": 55},
        {"invalid_original_seed": 52, "replacement_seed": 56},
        {"invalid_original_seed": 53, "replacement_seed": 57},
    ]
    with pytest.raises(ValueError, match="canonical"):
        replacements.deterministic_mapping([52, 47])
    with pytest.raises(ValueError, match="canonical"):
        replacements.deterministic_mapping([47, 47])
    with pytest.raises(ValueError, match="canonical"):
        replacements.deterministic_mapping([47, 54])


def test_preparer_replaces_complete_pairs_and_binds_new_protocols(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_packets, output, receipt = _prepare(tmp_path, monkeypatch)

    assert receipt["schema"] == replacements.RECEIPT_SCHEMA
    assert receipt["sha256"] == _canonical(
        {key: item for key, item in receipt.items() if key != "sha256"}
    )
    assert [
        (row["invalid_original_seed"], row["replacement_seed"]) for row in receipt["migrations"]
    ] == [(47, 54), (51, 55), (52, 56), (53, 57)]
    assert all(row["whole_pair_excluded"] is True for row in receipt["migrations"])
    assert receipt["excluded_original_seeds"] == [47, 51, 52, 53]
    assert receipt["included_seeds"] == [46, 48, 49, 50, 54, 55, 56, 57]
    assert receipt["capacity"] == {
        "new_replacement_rollouts": 136,
        "final_comparison_rollouts": 272,
        "original_base_rollouts": 136,
        "original_candidate_started_seeds": [46, 47, 48, 49, 50, 51],
        "original_candidate_started_seed_rollouts": 102,
        "original_candidate_retired_before_start_seeds": [52, 53],
        "original_candidate_retired_before_start_rollouts": 0,
        "scoring_or_metadata_cpu_model_rollouts": 0,
        "cumulative_model_rollouts_consumed_or_planned_today": 374,
        "daily_rollout_cap": 500,
        "within_daily_cap": True,
    }
    assert receipt["privacy"] == {
        "score_values_read": False,
        "prompts_responses_flags_rewards_or_trace_content_read": False,
        "infrastructure_reason_classes_only": True,
    }
    assert receipt["sanitized_invalid_replica_evidence"] == [
        {
            "seed": 51,
            "arm_id": "base",
            "path": ("docs/evidence/qwen38-fleet-dev17-seed51-base-invalid-replica-20260923.json"),
            "file_sha256": (
                "sha256:525885e6650d6d742144b12de2ac1807a77cd25f23e07cec58d2832551150497"
            ),
            "receipt_sha256": (
                "sha256:572e330d1340e83d2aaf188665c0fd99b401e33eeec5f858717e2db60952bfa3"
            ),
            "reason_class": "mixed_infrastructure_invalid_replica",
        }
    ]
    assert len(receipt["replacement_arms"]) == 8
    assert {(row["replacement_seed"], row["arm_id"]) for row in receipt["replacement_arms"]} == {
        (seed, arm) for seed in (54, 55, 56, 57) for arm in replacements.ARMS
    }
    definition = receipt["comparison_definition"]
    assert definition["schema"] == replacements.COMPARISON_DEFINITION_SCHEMA
    assert definition["protocol_study_id"] == replacements.PROTOCOL_V2_STUDY_ID
    assert definition["sha256"] != replacements.PREDECESSOR_COMPARISON_DEFINITION_SHA256
    assert definition["sha256"] == _canonical(
        {key: item for key, item in definition.items() if key != "sha256"}
    )
    assert definition["included_seeds"] == receipt["included_seeds"]
    assert definition["sessions_per_arm"] == 136
    assert definition["total_sessions"] == 272
    assert [row["seed"] for row in definition["replica_protocols"]] == receipt["included_seeds"]
    assert (
        json.loads((output / "COMPARISON_DEFINITION.json").read_text(encoding="utf-8"))
        == definition
    )

    retirement = json.loads((output / "RETIREMENT_EVIDENCE.json").read_text(encoding="utf-8"))
    assert retirement["sha256"] == receipt["retirement_evidence"]["sha256"]
    assert [row["seed"] for row in retirement["targets"]] == [52, 53]
    assert retirement["model_rollouts"] == 0
    assert retirement["outputs_or_databases_deleted"] is False
    assert retirement["preserved_immutable_config_maps"] == 2

    original = heldout_launch.build_package(
        source_packets / "seed46" / "base" / "LAUNCH_PACKET.json"
    )
    original_tasks = next(iter(original.evaluation_config["routes"].values()))["task_versions"]
    protocol_digests = set()
    for seed in (54, 55, 56, 57):
        pair = []
        for arm in replacements.ARMS:
            package = heldout_launch.build_package(
                output / f"seed{seed}" / arm / "LAUNCH_PACKET.json"
            )
            pair.append(package)
            route = next(iter(package.evaluation_config["routes"].values()))
            assert route["task_versions"] == original_tasks
            assert len(route["task_versions"]) == 17
            assert package.packet.identity["sampling_seed"] == seed
            assert package.packet.identity["pass_k"] == 1
            assert package.packet.identity["retry_limit"] == 0
            assert package.evaluation_config["concurrency"] == 4
            assert package.evaluation_config["max_reviewed_infrastructure_retries"] == 0
            assert package.evaluation_config["training_data_eligible"] is False
            assert package.job["metadata"]["annotations"]["fleet.ai/failure-alerts"] == "off"
            assert package.job["spec"]["template"]["spec"]["priorityClassName"] == "c1"
            assert "nvidia.com/gpu" not in json.dumps(package.job)
            protocol_digests.add(package.packet.identity["comparison_protocol_sha256"])
        assert (
            pair[0].packet.identity["comparison_protocol_sha256"]
            == pair[1].packet.identity["comparison_protocol_sha256"]
        )
        assert pair[0].packet.identity["protocol_id"].endswith("replacement-p1-v2")
        source_seed = {54: 47, 55: 51, 56: 52, 57: 53}[seed]
        source_protocol = receipt["migrations"][(47, 51, 52, 53).index(source_seed)][
            "excluded_source_arms"
        ]["base"]["comparison_protocol_sha256"]
        assert pair[0].packet.identity["comparison_protocol_sha256"] != source_protocol
    assert len(protocol_digests) == 4


def test_replacement_changes_only_declared_identity_axes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_packets, output, _receipt = _prepare(tmp_path, monkeypatch)
    old = heldout_launch.build_package(source_packets / "seed47" / "base" / "LAUNCH_PACKET.json")
    new = heldout_launch.build_package(output / "seed54" / "base" / "LAUNCH_PACKET.json")
    old_config = copy.deepcopy(old.evaluation_config)
    new_config = copy.deepcopy(new.evaluation_config)
    old_config.pop("name")
    new_config.pop("name")
    assert old_config["sampling"].pop("seed") == 47
    assert new_config["sampling"].pop("seed") == 54
    assert new_config == old_config
    assert new.packet.identity["model_revision"] == old.packet.identity["model_revision"]
    assert new.packet.identity["harness"] == old.packet.identity["harness"] == "opencode"
    assert new.packet.identity["context_management"] == old.packet.identity["context_management"]


def test_intent_fails_closed_on_score_unseal_reason_or_source_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_packets = _source(tmp_path, monkeypatch)
    intent_path = _intent(tmp_path, source_packets)
    source_receipt_path = source_packets / "PREPARATION_RECEIPT.json"
    source_receipt = json.loads(source_receipt_path.read_text(encoding="utf-8"))
    value = json.loads(intent_path.read_text(encoding="utf-8"))
    for mutation in ("score", "reason", "source"):
        changed = copy.deepcopy(value)
        if mutation == "score":
            changed["score_unsealed"] = True
        elif mutation == "reason":
            changed["invalid_original_replicas"][0]["reason_class"] = "capability_result"
        else:
            changed["source_preparation_receipt_sha256"] = "sha256:" + "f" * 64
        changed["sha256"] = _canonical(
            {key: item for key, item in changed.items() if key != "sha256"}
        )
        path = tmp_path / f"bad-{mutation}.json"
        path.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(ValueError):
            replacements._load_intent(path, source_receipt, source_receipt_path)  # noqa: SLF001

    stale_roster = _intent(tmp_path, source_packets, seeds=(47, 52, 53))
    with pytest.raises(ValueError, match="frozen invalid-replica roster"):
        replacements._load_intent(  # noqa: SLF001
            stale_roster, source_receipt, source_receipt_path
        )


def test_retirement_evidence_rejects_resigned_post_state_disagreement(
    tmp_path: Path,
) -> None:
    preflight = tmp_path / "preflight.json"
    preflight.write_bytes(replacements.RETIREMENT_PREFLIGHT.read_bytes())
    post = json.loads(replacements.RETIREMENT_POST.read_text(encoding="utf-8"))
    post["post_verification"]["exact_jobs_absent"] = 1
    post_path = tmp_path / "post.json"
    post["source_preflight"] = {
        "path": str(preflight.resolve()),
        "file_sha256": replacements._file_sha(preflight),  # noqa: SLF001
        "self_sha256": json.loads(preflight.read_text(encoding="utf-8"))["sha256"],
    }
    post["sha256"] = _canonical({key: item for key, item in post.items() if key != "sha256"})
    post_path.write_text(json.dumps(post, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="post-verification"):
        replacements._retirement_evidence(preflight, post_path)  # noqa: SLF001


def test_renderer_binds_all_eight_packets_and_launch_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _source_packets, packets, migration = _prepare(tmp_path, monkeypatch)
    output = tmp_path / "launchers"
    receipt = launchers.render(packets=packets, output=output)

    assert receipt["schema"] == launchers.RENDER_SCHEMA
    assert receipt["migration_receipt_sha256"] == migration["sha256"]
    assert receipt["comparison_definition_sha256"] == migration["comparison_definition"]["sha256"]
    assert receipt["included_seeds"] == [46, 48, 49, 50, 54, 55, 56, 57]
    assert receipt["sha256"] == _canonical(
        {key: item for key, item in receipt.items() if key != "sha256"}
    )
    assert len(receipt["arms"]) == 8
    assert receipt["create_gates"] == {
        "identical_server_previews_required": 2,
        "duplicate_census_before_first_preview": True,
        "duplicate_census_after_second_preview": True,
        "output_database_kubernetes_and_ledger_absence_required": True,
        "authoritative_completed_session_required_before_cell_acceptance": True,
        "one_create_call_per_arm": True,
    }
    assert receipt["runtime_module_sha256"]["evals/fleet/heldout_launch.py"].startswith("sha256:")
    bundle = yaml.safe_load((output / "launchers.yaml").read_text(encoding="utf-8"))
    jobs = [item for item in bundle["items"] if item["kind"] == "Job"]
    maps = [item for item in bundle["items"] if item["kind"] == "ConfigMap"]
    assert len(jobs) == len(maps) == 8
    assert all(job["metadata"]["annotations"]["fleet.ai/failure-alerts"] == "off" for job in jobs)
    assert all(job["spec"]["template"]["spec"]["priorityClassName"] == "c1" for job in jobs)
    assert all("nvidia.com/gpu" not in json.dumps(job) for job in jobs)


def test_renderer_rejects_one_packet_byte_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _source_packets, packets, _migration = _prepare(tmp_path, monkeypatch)
    packet = packets / "seed54" / "base" / "LAUNCH_PACKET.json"
    value = json.loads(packet.read_text(encoding="utf-8"))
    value["database"] = "q38_dev17_s54_base_repl_p1_v2_drift"
    packet.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises((ValueError, heldout_launch.HeldoutLaunchError)):
        launchers.render(packets=packets, output=tmp_path / "bad-launchers")


def test_comparison_definition_forbids_cell_level_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _source_packets, _output, receipt = _prepare(tmp_path, monkeypatch)
    definition = receipt["comparison_definition"]
    assert definition["whole_replica_pairs_only"] is True
    assert definition["cell_level_replacement_forbidden"] is True
    assert len(definition["included_seeds"]) == 8
    assert len(definition["replica_protocols"]) == 8
    assert len(receipt["migrations"]) == 4
    assert all(
        set(row["excluded_source_arms"]) == set(replacements.ARMS) for row in receipt["migrations"]
    )
    assert all(
        len(
            next(
                iter(
                    heldout_launch.build_package(
                        _output / f"seed{row['replacement_seed']}" / arm / "LAUNCH_PACKET.json"
                    )
                    .evaluation_config["routes"]
                    .values()
                )
            )["task_versions"]
        )
        == replacements.TASKS_PER_ARM
        for row in receipt["migrations"]
        for arm in replacements.ARMS
    )
