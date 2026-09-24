from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "configs/qualification/qwen38-miles96-signal-candidates-20260924-v2.json"
CYBER_CONTRACT = {
    "evidence_schema": "1.0.0",
    "submission_protocol": "2.0.0",
    "verifier_contract": "3.0.0",
}
TOOL_TRANSFORM_SOURCE_SHA256 = (
    "sha256:0524f19dcc886b20d17b39c21bd6417f537423eec6ef2359487fc3911dad441c"
)
LIVE_BINDING_SHA256 = {
    "7317": "sha256:af41b09f5523214d153208e5a76b2b91caeebd9e6b3e49906d39de49844c7541",
    "0756": "sha256:200074c210072b1691927db9eef0d53af300c3a492380dc16dff5835323de84b",
    "f294": "sha256:08990cecfa0f6f5ff7183533e957fe2049357d67baffb29e22afed855a379874",
    "2c50": "sha256:5a5a7303dd594575b107c16e75025275560e710373f051a22b3375859e4d9d8f",
}


def _load(path: str | Path):
    resolved = ROOT / path if isinstance(path, str) else path
    return json.loads(resolved.read_text())


def _canonical_sha(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(raw).hexdigest()}"


def _file_sha(path: str) -> str:
    return f"sha256:{hashlib.sha256((ROOT / path).read_bytes()).hexdigest()}"


def _one(rows, *, key: str, value: str):
    matches = [row for row in rows if row.get(key) == value]
    assert len(matches) == 1
    return matches[0]


def test_packet_and_source_authorities_are_sealed() -> None:
    packet = _load(PACKET)
    body = dict(packet)
    body.pop("sha256")
    assert packet["sha256"] == _canonical_sha(body)
    assert packet["launchable"] is False

    for name, authority in packet["authorities"].items():
        path = authority.get("path")
        if not path or name == "0756_safe_summary_prior":
            continue
        assert authority["file_sha256"] == _file_sha(path)
        source = _load(path)
        if name == "tool_catalog":
            assert authority["raw_catalog_sha256"] == _canonical_sha(source)
            projected = [
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["inputSchema"],
                    },
                }
                for tool in source
            ]
            assert authority["openai_projection_sha256"] == _canonical_sha(projected)
            assert authority["transform_source_sha256"] == TOOL_TRANSFORM_SOURCE_SHA256
            assert authority["ordered_tools"] == [tool["name"] for tool in source]
        elif "self_sha256" in authority:
            source_body = dict(source)
            source_body.pop("sha256")
            assert authority["self_sha256"].removeprefix("sha256:") == _canonical_sha(
                source_body
            ).removeprefix("sha256:")


def test_candidate_order_and_exact_current_bindings() -> None:
    packet = _load(PACKET)
    candidates = packet["candidates"]
    assert [row["rank"] for row in candidates] == [1, 2, 3, 4]
    assert (
        [row["id"] for row in candidates]
        == packet["deterministic_order"]
        == [
            "7317",
            "0756",
            "f294",
            "2c50",
        ]
    )

    task_set = _load(packet["authorities"]["task_set"]["path"])["tasks"]
    production = _load(packet["authorities"]["current_production_inventory"]["path"])["tasks"]
    proven = _load(packet["authorities"]["receipt_proven_inventory"]["path"])["task_versions"]
    eligible = _load(packet["authorities"]["historical_runtime_inventory"]["path"])["task_versions"]
    split_rows = _load(packet["authorities"]["current_lineage_split"]["path"])["tasks"]
    components = _load(packet["authorities"]["lineage_census"]["path"])["components"]
    roles = _load(packet["authorities"]["lineage_role_anchor"]["path"])["roles"]
    teacher_rows = _load(packet["authorities"]["teacher3k_training_lineage"]["path"])[
        "training_task_keys"
    ]

    for candidate in candidates:
        task = candidate["task"]
        env = candidate["environment"]
        verifier = candidate["verifier"]
        lineage = candidate["lineage"]
        key = task["key"]
        version = task["version_id"]

        current = _one(production, key="task_version_id", value=version)
        assert current == {
            "qa_status": "not_analyzed",
            "task_id": task["id"],
            "task_key": key,
            "task_shape": "blackbox",
            "task_version_id": version,
        }

        task_set_row = _one(task_set, key="task_version_id", value=version)
        assert task_set_row["task_key"] == key
        assert task_set_row["env_key"] == env["id"]
        assert task_set_row["env_version"] == env["version"]
        assert task_set_row["environment_version_id"] == env["version_id"]
        assert task_set_row["data_key"] == env["data_id"]
        assert task_set_row["data_version"] == env["data_version"]
        assert task_set_row["lineage"] == {
            "application": lineage["application"],
            "task_family": lineage["task_family"],
        }

        runtime = _one(eligible, key="task_version_id", value=version)
        assert runtime["task_key"] == key
        assert runtime["environment"] == {
            field: value for field, value in env.items() if field != "ttl_seconds"
        }
        assert runtime["verifier"] == {
            field: value for field, value in verifier.items() if field != "function_name"
        }
        assert (
            runtime["current_binding_sha256"]
            == candidate["static_binding"]["current_inventory_binding_sha256"]
        )
        assert all(
            runtime["execution_evidence"][flag]
            for flag in (
                "execution_digest_valid",
                "config_reconstructed_to_receipt_digest",
                "exact_current_environment_and_verifier_match",
            )
        )

        proven_row = _one(proven, key="task_version_id", value=version)
        assert proven_row["task_id"] == task["id"]
        assert proven_row["task_key"] == key
        assert proven_row["lineage"]["task_family"] == lineage["task_family"]
        assert proven_row["provenance"]["evidence_class"] == candidate["quality"]["evidence_class"]
        assert proven_row["provenance"]["certification"] == {
            "receipt_sha256": candidate["quality"]["certification_receipt_sha256"],
            "status": "accepted",
        }

        split = _one(split_rows, key="task_version_id", value=version)
        assert split == {
            "group_id": lineage["group_id"],
            "shared_atom_component_id": lineage["component_id"],
            "split": "train",
            "task_key": key,
            "task_version_id": version,
        }
        assert _one(roles, key="group_id", value=lineage["group_id"])["split"] == "train"

        component = _one(components, key="component_id", value=lineage["component_id"])
        assert component["inherited_role"] == "train"
        assert component["task_versions"] == [
            {"source": "receipt_proven", "task_key": key, "task_version_id": version}
        ]
        assert lineage["component_task_version_count"] == 1

        teacher = _one(teacher_rows, key="task_key", value=key)
        assert version in {row["task_version_id"] for row in teacher["versions"]}
        assert lineage["teacher3k_training_exposed"] is True


def test_all_candidates_reconstruct_exact_phase1_live_binding() -> None:
    packet = _load(PACKET)
    assert packet["fleet_team"] == {
        "id": "a1025f0b-ad67-49fc-a023-51800ab43e84",
        "name": "fleet",
    }

    for candidate in packet["candidates"]:
        task = {
            field: candidate["task"][field]
            for field in (
                "key",
                "version_id",
                "prompt_sha256",
                "env_variables_sha256",
                "output_json_schema_sha256",
            )
        }
        task["cyber_contract"] = CYBER_CONTRACT
        live_binding = {
            "task": task,
            "environment": candidate["environment"],
            "verifier": candidate["verifier"],
        }

        assert candidate["task"]["cyber_contract"] == CYBER_CONTRACT
        assert candidate["task"]["cyber_contract_sha256"] == _canonical_sha(CYBER_CONTRACT)
        assert candidate["environment"]["ttl_seconds"] == 32400
        assert candidate["verifier"]["function_name"] == "verify"
        expected = LIVE_BINDING_SHA256[candidate["id"]]
        assert candidate["static_binding"]["expected_live_binding_sha256"] == expected
        assert _canonical_sha(live_binding) == expected


def test_jit_gate_requires_every_phase1_live_binding_field() -> None:
    gate = _load(PACKET)["fresh_live_jit_gate"]
    required = set(gate["required_exact_fields"])
    assert {
        "fleet_team_id",
        "task.cyber_contract",
        "environment.ttl_seconds",
        "verifier.function_name",
    } <= required
    assert "task.cyber_contract_sha256" not in required


def test_no_selected_group_or_component_crosses_split_roles() -> None:
    packet = _load(PACKET)
    rows = _load(packet["authorities"]["current_lineage_split"]["path"])["tasks"]
    for candidate in packet["candidates"]:
        lineage = candidate["lineage"]
        group_roles = {row["split"] for row in rows if row["group_id"] == lineage["group_id"]}
        component_roles = {
            row["split"]
            for row in rows
            if row["shared_atom_component_id"] == lineage["component_id"]
        }
        assert group_roles == component_roles == {"train"}
        assert lineage["dev_or_final_test_overlap_count"] == 0

    excluded = packet["excluded_candidates"]
    assert len(excluded) == 1
    old = excluded[0]
    row = _one(rows, key="task_version_id", value=old["task_version_id"])
    assert old["id"] == "8095"
    assert row["split"] == old["current_split"] == "dev"
    assert row["group_id"] == old["group_id"]
    assert row["shared_atom_component_id"] == old["component_id"]


def test_signal_priors_and_zero_update_acceptance_are_fail_closed() -> None:
    packet = _load(PACKET)
    for candidate in packet["candidates"]:
        prior = candidate["quality"]["selection_prior"]
        assert prior["completed_accepted_episodes"] == 4
        assert prior["passes"] in {2, 3}
        assert prior["mixed_binary_outcomes"] is True

    contract = packet["phase1_contract"]
    assert contract["samples_per_candidate"] == 8
    assert contract["max_concurrent_episodes"] == 2
    assert contract["outer_retry_or_replacement"] is False
    assert contract["optimizer_steps"] == 0
    assert contract["checkpoint_allowed"] is False
    assert contract["all_slots_terminally_accounted"] is True
    assert contract["acceptance"] == {
        "minimum_normally_completed_gradeable_episodes": 2,
        "minimum_distinct_finite_rewards": 2,
        "unique_verifier_execution_id_per_grade": True,
        "all_instances_released": True,
        "runtime_preflight_passed": True,
        "optimizer_updates": 0,
        "checkpoint_files": 0,
    }
    assert packet["fresh_live_jit_gate"]["status"] == (
        "required_not_performed_by_this_read_only_packet"
    )


def test_packet_contains_no_private_content() -> None:
    packet = _load(PACKET)
    assert all(value is False for value in packet["privacy"].values())
    serialized = json.dumps(packet, sort_keys=True)
    for forbidden in (
        '"prompt_text"',
        '"reward_values"',
        '"trace"',
        '"flag"',
        '"answer"',
        '"credential"',
    ):
        assert forbidden not in serialized
