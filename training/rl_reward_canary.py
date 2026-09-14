"""Immutable source/evidence gate for the Qwen3.8 reward-acquisition canary.

This module is deliberately separate from :mod:`training.rl_data`. Historical
self-trace receipts freeze that generic builder byte-for-byte; a new experiment
must not rewrite an already accepted source closure to add its own gate.
"""

from __future__ import annotations

import json
from pathlib import Path

from cyber_post_train.jobs import digest
from evals.fleet import opencode_self_hosted as fleet

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PROOF_SCHEMA = "cyber_rl_reward_canary_source_closure_v1"
VERSION_RECEIPT_SCHEMA = "cyber_rl_reward_canary_task_version_receipt_v2"
VERSION_OBSERVATION_SCHEMA = "cyber_rl_reward_canary_task_version_get_observation_v1"
VERSION_SANITIZED_RESPONSE_SCHEMA = "cyber_rl_reward_canary_sanitized_task_version_response_v1"
VERSION_ACQUISITION_JOURNAL_SCHEMA = "cyber_rl_reward_canary_task_version_get_journal_v1"
TASK_SET_PATH = "configs/data/qwen38-rl-reward-canary-task-set-v2.json"
SPLIT_PATH = "configs/data/qwen38-rl-reward-canary-split-v1.json"
EXACT_VERSION_EVIDENCE_PATH = "configs/data/qwen38-rl-reward-canary-exact-version-evidence-v2.json"
SOURCE_IDENTITY = {
    "task_set_file_sha256": (
        "sha256:d4419253598fbcaf693a3c28f131bdbe852452bf60ce055386b4698d6e52aea8"
    ),
    "task_set_self_sha256": (
        "sha256:ecb13f7b488a3c94a6c06068508f4e2d2a527340171eea0a9670c1bf79686b31"
    ),
    "split_file_sha256": (
        "sha256:c3a34993a932c2b29e2c70f45682ce03ee02d03a9c4493edbab45eb349cb64fa"
    ),
    "split_self_sha256": (
        "sha256:8279ea19808ad1accb00d3f3145c3ec087030677786e0188251cfc197f306adf"
    ),
    "exact_version_evidence_file_sha256": (
        "sha256:fcfdf85272c6ef399b19b260fb89ea5785efbafb85e1c233b3d4265fd517f64f"
    ),
    "exact_version_evidence_self_sha256": (
        "sha256:25a095c90c2c3fa2b7ce7c5f016b7cf997cded21e188736d199970f544db29bd"
    ),
    # The source package authorizes the direct selection; the successor package
    # proves why the non-current clone is rejected instead of treated as equal.
    "eligible_source_observation_file_sha256": (
        "sha256:2b02d2972463921db79949bd280569155f5c818fc91d8621f69e281ca0dcdebe"
    ),
    "eligible_source_response_file_sha256": (
        "sha256:d575753138844437da893ef9a8c14f88afac922292e5486a24114b89edc55e09"
    ),
    "eligible_source_journal_file_sha256": (
        "sha256:18903bbe35b7cba655efeea6134d8f61300a98a53d804101c54ef7cd86b6ca55"
    ),
    "metadata_only_successor_observation_file_sha256": (
        "sha256:196bc04eedb7aef49faf61c03c17f5ab7ff6c3fd9eb14e2ab697fa7dea043ad6"
    ),
    "metadata_only_successor_response_file_sha256": (
        "sha256:cf9b6e5f8f6fcd7ab75df88fc22049f02348b2744af53d7d540ea62083f70a2b"
    ),
    "metadata_only_successor_journal_file_sha256": (
        "sha256:dc12912799ef5e5420e35c53a1eb9af6bb1a1ea7dcb3f9d1d71530219c697552"
    ),
    "data_preparation_file_sha256": (
        "sha256:f0c327d2ecc464610a5fd86e11b112870d28feb43d9155d7507d8fedbd87632b"
    ),
    "episode_runtime_file_sha256": (
        "sha256:faa8aa3d75d0f1c18473dc92952148441893dcd9e99a40d9d8d76e9e8e6ae100"
    ),
    "fleet_binding_file_sha256": (
        "sha256:b28e267d02024ac979f4a0f39ee778e68331da490f5084c3e5a34136eff04707"
    ),
}
LIMITS = {
    "context_tokens": 98304,
    "response_tokens": 81920,
    "max_tokens_per_turn": 4096,
    "max_turns": 600,
    "episode_seconds": 2400,
    "tool_seconds": 330,
    "tool_result_chars": 50000,
}
_HORIZON_FIXED_LIMIT_KEYS = frozenset(LIMITS) - {"max_turns"}
_VERSION_RECEIPT_FIELDS = frozenset(
    {
        "authority",
        "authority_payload_sha256",
        "authority_observation",
        "environment_version_id",
        "field_sha256",
        "metadata_tools",
        "metadata_without_tools_sha256",
        "role",
        "schema",
        "sha256",
        "task_key",
        "task_version_id",
        "verifier_version_id",
    }
)
_VERSION_OBSERVATION_FIELDS = frozenset(
    {
        "schema",
        "authority",
        "request",
        "status_code",
        "response_body_sha256",
        "observed_at",
        "task_key",
        "task_version_id",
        "sanitized_response",
        "acquisition_journal",
        "sha256",
    }
)
_VERSION_SANITIZED_RESPONSE_FIELDS = frozenset(
    {
        "schema",
        "task_key",
        "task_version_id",
        "environment_version_id",
        "verifier_version_id",
        "metadata_tools",
        "metadata_without_tools_sha256",
        "fields",
        "sha256",
    }
)
_VERSION_ACQUISITION_JOURNAL_FIELDS = frozenset(
    {
        "schema",
        "authority",
        "request",
        "status_code",
        "response_body_sha256",
        "sanitized_response_path",
        "sanitized_response_file_sha256",
        "sanitized_response_self_sha256",
        "observed_at",
        "sha256",
    }
)
_DIRECT_IDENTITY_FIELDS = frozenset(
    {
        "data_id",
        "data_version",
        "environment_id",
        "environment_version_id",
        "key",
        "task_lifecycle_status",
        "task_modality",
        "team_id",
        "verifier_id",
        "version",
    }
)
_VERSION_FIELD_NAMES = frozenset(
    {
        "attachments",
        "created_at",
        "data_id",
        "data_version",
        "env_variables",
        "environment_id",
        "environment_version_id",
        "factual_answer",
        "graded",
        "key",
        "multi_app_seed_bindings",
        "multi_app_seed_versions",
        "output_json_schema",
        "prompt",
        "seed_config",
        "task_lifecycle_status",
        "task_modality",
        "task_scenario_id",
        "team_id",
        "verifier",
        "verifier_id",
        "version",
        "warnings",
    }
)


def _sealed(value: dict, schema: str) -> None:
    if value.get("schema") != schema or value.get("sha256") != "sha256:" + digest(
        {key: item for key, item in value.items() if key != "sha256"}
    ):
        raise ValueError("reviewed reward source digest mismatch")


def _path(path: Path) -> Path:
    resolved = path.resolve()
    if path.is_symlink() or not resolved.is_relative_to(ROOT) or not resolved.is_file():
        raise ValueError("reward evidence path is unavailable or escapes the source tree")
    return resolved


def _bound_json(path: Path, expected_file_sha256: str) -> dict:
    source = _path(path)
    before = source.stat()
    payload = source.read_bytes()
    after = source.stat()
    if any(
        getattr(before, key) != getattr(after, key)
        for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    ):
        raise ValueError("exact-version evidence changed while it was read")
    if fleet.sha256(payload) != expected_file_sha256:
        raise ValueError("exact-version evidence file digest mismatch")
    try:
        value = json.loads(payload)
    except (TypeError, ValueError) as error:
        raise ValueError("exact-version evidence is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("exact-version evidence must be a JSON object")
    return value


def _bound_bytes(path: Path, expected_file_sha256: str) -> bytes:
    source = _path(path)
    before = source.stat()
    payload = source.read_bytes()
    after = source.stat()
    if any(
        getattr(before, key) != getattr(after, key)
        for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    ):
        raise ValueError("reward evidence changed while it was read")
    if fleet.sha256(payload) != expected_file_sha256:
        raise ValueError("reward evidence file digest mismatch")
    return payload


def _validate_source_identity(task_set: dict, split: dict, limits: dict) -> None:
    if (
        task_set.get("sha256") != SOURCE_IDENTITY["task_set_self_sha256"]
        or split.get("sha256") != SOURCE_IDENTITY["split_self_sha256"]
        or limits != LIMITS
    ):
        raise ValueError("reward canary differs from its exact immutable source identity")


def _version_receipt(
    binding: object,
    *,
    evidence_path: Path,
    role: str,
    task_key: str,
    task_version_id: str,
) -> dict:
    """Open one exact sanitized receipt derived from the authoritative task GET."""
    if not isinstance(binding, dict) or set(binding) != {
        "path",
        "file_sha256",
        "self_sha256",
        "authority_payload_sha256",
        "authority_observation",
    }:
        raise ValueError(
            "retained sanitized authoritative GET observation binding is missing or incomplete"
        )
    receipt = _bound_json(evidence_path.parent / binding["path"], binding["file_sha256"])
    _sealed(receipt, VERSION_RECEIPT_SCHEMA)
    observation_binding = binding["authority_observation"]
    if not isinstance(observation_binding, dict) or set(observation_binding) != {
        "path",
        "file_sha256",
        "self_sha256",
    }:
        raise ValueError("sanitized authoritative GET observation binding is incomplete")
    reviewed = {
        "observation": SOURCE_IDENTITY.get(f"{role}_observation_file_sha256"),
        "response": SOURCE_IDENTITY.get(f"{role}_response_file_sha256"),
        "journal": SOURCE_IDENTITY.get(f"{role}_journal_file_sha256"),
    }
    if (
        any(
            not isinstance(value, str) or not value.startswith("sha256:")
            for value in reviewed.values()
        )
        or observation_binding.get("file_sha256") != reviewed["observation"]
    ):
        raise ValueError("reviewed retained authoritative GET observation is unavailable")
    observation = _bound_json(
        evidence_path.parent / observation_binding["path"],
        observation_binding["file_sha256"],
    )
    _sealed(observation, VERSION_OBSERVATION_SCHEMA)
    response_binding = observation.get("sanitized_response")
    journal_binding = observation.get("acquisition_journal")
    expected_reference_fields = {"path", "file_sha256", "self_sha256"}
    if (
        not isinstance(response_binding, dict)
        or set(response_binding) != expected_reference_fields
        or not isinstance(journal_binding, dict)
        or set(journal_binding) != expected_reference_fields
    ):
        raise ValueError("authoritative GET observation lacks retained response bytes or journal")
    if (
        response_binding.get("file_sha256") != reviewed["response"]
        or journal_binding.get("file_sha256") != reviewed["journal"]
    ):
        raise ValueError("retained authoritative response or journal is not the reviewed package")
    response_payload = _bound_bytes(
        evidence_path.parent / response_binding["path"], response_binding["file_sha256"]
    )
    journal_payload = _bound_bytes(
        evidence_path.parent / journal_binding["path"], journal_binding["file_sha256"]
    )
    try:
        response = json.loads(response_payload)
        journal = json.loads(journal_payload)
    except (TypeError, ValueError) as error:
        raise ValueError("retained authoritative response or journal is not valid JSON") from error
    if (
        not isinstance(response, dict)
        or not isinstance(journal, dict)
        or response_payload != fleet.canonical_json(response)
        or journal_payload != fleet.canonical_json(journal)
    ):
        raise ValueError("retained authoritative response and journal must be canonical JSON bytes")
    _sealed(response, VERSION_SANITIZED_RESPONSE_SCHEMA)
    _sealed(journal, VERSION_ACQUISITION_JOURNAL_SCHEMA)
    digests = receipt.get("field_sha256")
    sanitized_fields = response.get("fields")
    if not isinstance(sanitized_fields, dict) or set(sanitized_fields) != _VERSION_FIELD_NAMES:
        raise ValueError("sanitized authoritative response fields are incomplete")
    derived_digests = {}
    for name, entry in sanitized_fields.items():
        if not isinstance(entry, dict) or set(entry) not in ({"value"}, {"redacted_sha256"}):
            raise ValueError("sanitized authoritative response field representation changed")
        if name in _DIRECT_IDENTITY_FIELDS and set(entry) != {"value"}:
            raise ValueError("authoritative runtime identity may not be replaced by an opaque hash")
        if "value" in entry:
            derived_digests[name] = fleet.sha256(fleet.canonical_json(entry["value"]))
        else:
            value = entry["redacted_sha256"]
            if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
                raise ValueError("sanitized response redaction digest is malformed")
            derived_digests[name] = value
    expected_request = {
        "method": "GET",
        "url": f"https://orchestrator.fleetai.com/v1/tasks/{task_key}",
        "query": {"version_id": task_version_id},
    }
    if (
        set(receipt) != _VERSION_RECEIPT_FIELDS
        or receipt.get("authority") != "https://orchestrator.fleetai.com/v1/tasks/{task_key}"
        or receipt.get("role") != role
        or receipt.get("task_key") != task_key
        or receipt.get("task_version_id") != task_version_id
        or receipt.get("sha256") != binding["self_sha256"]
        or receipt.get("authority_payload_sha256") != binding["authority_payload_sha256"]
        or receipt.get("authority_observation") != observation_binding
        or set(observation) != _VERSION_OBSERVATION_FIELDS
        or observation.get("sha256") != observation_binding["self_sha256"]
        or observation.get("authority") != "https://orchestrator.fleetai.com"
        or observation.get("request") != expected_request
        or observation.get("status_code") != 200
        or observation.get("task_key") != task_key
        or observation.get("task_version_id") != task_version_id
        or not isinstance(observation.get("observed_at"), str)
        or not observation["observed_at"].endswith("Z")
        or observation.get("sanitized_response") != response_binding
        or observation.get("acquisition_journal") != journal_binding
        or set(response) != _VERSION_SANITIZED_RESPONSE_FIELDS
        or response.get("sha256") != response_binding["self_sha256"]
        or response.get("task_key") != task_key
        or response.get("task_version_id") != task_version_id
        or response.get("environment_version_id") != receipt.get("environment_version_id")
        or response.get("verifier_version_id") != receipt.get("verifier_version_id")
        or response.get("metadata_tools") != receipt.get("metadata_tools")
        or response.get("metadata_without_tools_sha256")
        != receipt.get("metadata_without_tools_sha256")
        or derived_digests != digests
        or response.get("fields", {}).get("key", {}).get("value") != task_key
        or response.get("fields", {}).get("environment_version_id", {}).get("value")
        != receipt.get("environment_version_id")
        or set(journal) != _VERSION_ACQUISITION_JOURNAL_FIELDS
        or journal.get("sha256") != journal_binding["self_sha256"]
        or journal.get("authority") != observation.get("authority")
        or journal.get("request") != expected_request
        or journal.get("status_code") != 200
        or journal.get("response_body_sha256") != observation.get("response_body_sha256")
        or journal.get("sanitized_response_path") != response_binding["path"]
        or journal.get("sanitized_response_file_sha256") != response_binding["file_sha256"]
        or journal.get("sanitized_response_self_sha256") != response_binding["self_sha256"]
        or journal.get("observed_at") != observation.get("observed_at")
        or observation.get("response_body_sha256") != fleet.sha256(response_payload)
        or receipt.get("authority_payload_sha256") != fleet.sha256(response_payload)
        or not isinstance(digests, dict)
        or set(digests) != _VERSION_FIELD_NAMES
        or any(
            not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71
            for value in (
                receipt.get("authority_payload_sha256"),
                receipt.get("metadata_without_tools_sha256"),
                observation.get("response_body_sha256"),
                *digests.values(),
            )
        )
    ):
        raise ValueError("authoritative version receipt is malformed or mismatched")
    return receipt


def _validate_tool_surface_authority(
    authority: object,
    *,
    evidence_path: Path,
    task_set: dict,
    source_receipt: dict,
) -> None:
    """Prove that task metadata is not the canary's tool-catalog authority."""
    if not isinstance(authority, dict):
        raise ValueError("reward canary tool-surface authority is missing")
    expected_code = {
        "data_preparation": {
            "path": "../../training/rl_data.py",
            "file_sha256": SOURCE_IDENTITY["data_preparation_file_sha256"],
            "symbols": ["build"],
        },
        "episode_runtime": {
            "path": "../../training/rl_episode.py",
            "file_sha256": SOURCE_IDENTITY["episode_runtime_file_sha256"],
            "symbols": ["collect", "_agent"],
        },
        "fleet_binding": {
            "path": "../../evals/fleet/opencode_self_hosted.py",
            "file_sha256": SOURCE_IDENTITY["fleet_binding_file_sha256"],
            "symbols": ["bind_task", "verify_task", "assert_required_task_tools"],
        },
    }
    expected_route = {
        "theseus_commit": "75228148c57e72fd054007a2593a324c5d6db97f",
        "path": "orchestrator/public_api/rollout_rewards.py",
        "exact_version_hydration": True,
        "metadata_tools_read": False,
        "provisioning_route": (
            "/v1/rollout-rewards/{task_key}/versions/{task_version_id}/instances"
        ),
        "scoring_route": "/v1/rollout-rewards/{task_key}/versions/{task_version_id}",
    }
    if authority != {
        "mode": "trainer_injected_reviewed_catalog_plus_runtime_mcp_exact_match",
        "task_metadata_tools_required": False,
        "source_metadata_tools": None,
        "ordered_tools": ["bash", "submit_report"],
        "tool_catalog_sha256": task_set.get("tool_catalog_sha256"),
        "local_code": expected_code,
        "reviewed_server_route": expected_route,
    } or source_receipt.get("metadata_tools") is not None:
        raise ValueError("reward canary tool-surface authority changed")

    markers = {
        "data_preparation": (
            b'[t.get("name") for t in catalog] != ["bash", "submit_report"]',
            b"messages, tools=tools, tokenize=False, add_generation_prompt=True",
        ),
        "episode_runtime": (
            b"catalog = (await mcp.list_tools()).tools",
            b"fleet.assert_required_task_tools(",
            b"raw = json.loads(fleet.canonical_json(raw))",
            b'for name in config["execution"]["required_task_tools"]',
        ),
        "fleet_binding": (
            b"def bind_task(",
            b"def verify_task(",
            b"def assert_required_task_tools(",
            b"if tool_names != required:",
        ),
    }
    for name, binding in expected_code.items():
        payload = _bound_bytes(
            evidence_path.parent / binding["path"], binding["file_sha256"]
        )
        if any(marker not in payload for marker in markers[name]):
            raise ValueError("reward canary tool enforcement source changed")


def validate_exact_version_evidence(
    task_set: dict,
    split: dict,
    limits: dict,
    *,
    task_set_dir: Path,
) -> None:
    """Validate the exact source-direct eligibility and 600-turn-only horizon."""
    _sealed(task_set, "cyber_rl_task_set_v1")
    _sealed(split, "cyber_task_split_v1")
    _validate_source_identity(task_set, split, limits)
    provenance = task_set.get("reward_signal_provenance")
    binding = provenance.get("exact_version_evidence") if isinstance(provenance, dict) else None
    if not isinstance(binding, dict) or set(binding) != {"path", "file_sha256", "self_sha256"}:
        raise ValueError("reward task set lacks exact-version evidence")
    if (
        binding.get("file_sha256") != SOURCE_IDENTITY["exact_version_evidence_file_sha256"]
        or binding.get("self_sha256") != SOURCE_IDENTITY["exact_version_evidence_self_sha256"]
    ):
        raise ValueError("exact-version evidence identity mismatch")
    evidence_path = task_set_dir / binding["path"]
    evidence = _bound_json(evidence_path, binding["file_sha256"])
    _sealed(evidence, "cyber_rl_exact_version_evidence_v1")
    if evidence["sha256"] != binding["self_sha256"]:
        raise ValueError("exact-version evidence identity mismatch")

    selected = evidence.get("selected_version", {})
    selected_key = (selected.get("task_key"), selected.get("task_version_id"))
    assignments = {
        (row["task_key"], row["task_version_id"]): row["split"] for row in split["tasks"]
    }
    selected_rows = [
        row
        for row in task_set["tasks"]
        if (row["task_key"], row["task_version_id"]) == selected_key
    ]
    if len(selected_rows) != 1 or assignments.get(selected_key) != "train":
        raise ValueError("selected reward version evidence is not exact and train-only")
    selected_row = selected_rows[0]

    source = evidence.get("eligible_source", {})
    eligible = _bound_json(
        evidence_path.parent / source.get("manifest_path", ""),
        source.get("manifest_file_sha256"),
    )
    if "sha256:" + eligible.get("sha256", "") != source.get("manifest_self_sha256"):
        raise ValueError("eligible-source manifest identity mismatch")
    source_rows = [
        row
        for row in eligible.get("task_versions", [])
        if (row.get("task_key"), row.get("task_version_id"))
        == (source.get("task_key"), source.get("task_version_id"))
    ]
    if len(source_rows) != 1 or "sha256:" + digest(source_rows[0]) != source.get("row_sha256"):
        raise ValueError("eligible-source row is absent or changed")

    source_receipt = _version_receipt(
        source.get("version_receipt"),
        evidence_path=evidence_path,
        role="eligible_source",
        task_key=source.get("task_key"),
        task_version_id=source.get("task_version_id"),
    )
    successor = evidence.get("rejected_metadata_only_successor", {})
    successor_receipt = _version_receipt(
        successor.get("version_receipt"),
        evidence_path=evidence_path,
        role="metadata_only_successor",
        task_key=source.get("task_key"),
        task_version_id=successor.get("successor_task_version_id"),
    )
    if (
        selected_key != (source.get("task_key"), source.get("task_version_id"))
        or successor.get("source_task_version_id") != source.get("task_version_id")
        or successor.get("successor_task_version_id") == selected_key[1]
        or successor.get("only_semantic_diff_path") != "/metadata/tools"
        or successor.get("eligibility_status") != "rejected_missing_versioned_starting_data"
        or successor.get("ordered_tools") != ["bash", "submit_report"]
        or source_receipt.get("metadata_tools") is not None
        or successor_receipt.get("metadata_tools") != ["bash", "submit_report"]
    ):
        raise ValueError("authoritative metadata-only successor binding is absent or changed")
    source_environment = source_rows[0].get("environment", {})
    expected_runtime = {
        "environment_id": selected_row.get("env_key"),
        "environment_version_id": selected_row.get("environment_version_id"),
        "version": selected_row.get("env_version"),
        "data_id": selected_row.get("data_key"),
        "data_version": selected_row.get("data_version"),
    }
    if (
        source_environment
        != {
            **source_environment,
            "id": expected_runtime["environment_id"],
            "version_id": expected_runtime["environment_version_id"],
            "version": expected_runtime["version"],
            "data_id": expected_runtime["data_id"],
            "data_version": expected_runtime["data_version"],
        }
        or source_receipt.get("environment_version_id")
        != expected_runtime["environment_version_id"]
        or successor_receipt.get("environment_version_id")
        != expected_runtime["environment_version_id"]
        or source_receipt.get("verifier_version_id")
        != source_rows[0].get("verifier", {}).get("version_id")
        or successor_receipt.get("verifier_version_id") != source_receipt.get("verifier_version_id")
    ):
        raise ValueError("authoritative version receipt runtime identity changed")
    expected_field_digests = {
        key: fleet.sha256(fleet.canonical_json(value)) for key, value in expected_runtime.items()
    }
    if any(
        source_receipt["field_sha256"].get(key) != value
        for key, value in expected_field_digests.items()
    ):
        raise ValueError("eligible source receipt differs from reviewed runtime identity")

    successor_differing_fields = sorted(
        key
        for key in _VERSION_FIELD_NAMES
        if source_receipt["field_sha256"][key] != successor_receipt["field_sha256"][key]
    )
    if (
        successor_differing_fields != ["data_id", "data_version"]
        or successor.get("authoritative_differing_fields") != successor_differing_fields
        or source_receipt["metadata_without_tools_sha256"]
        != successor_receipt["metadata_without_tools_sha256"]
    ):
        raise ValueError("rejected metadata-only successor evidence changed")

    _validate_tool_surface_authority(
        evidence.get("tool_surface_authority"),
        evidence_path=evidence_path,
        task_set=task_set,
        source_receipt=source_receipt,
    )
    if (
        selected.get("eligibility_status") != "eligible_authoritative_source_direct"
        or task_set.get("training_data_eligible") is not True
    ):
        raise ValueError("source-direct reward-canary eligibility is not receipt-derived")

    horizon = evidence.get("episode_horizon", {})
    fixed_limits = horizon.get("fixed_limits")
    if (
        horizon.get("max_turns") != 600
        or limits.get("max_turns") != horizon.get("max_turns")
        or set(limits) != _HORIZON_FIXED_LIMIT_KEYS | {"max_turns"}
        or not isinstance(fixed_limits, dict)
        or set(fixed_limits) != _HORIZON_FIXED_LIMIT_KEYS
        or {key: limits.get(key) for key in _HORIZON_FIXED_LIMIT_KEYS} != fixed_limits
    ):
        raise ValueError("episode horizon differs from its fixed token/time bounds")
    task_contract = _bound_json(
        evidence_path.parent / horizon.get("task_specific_contract_path", ""),
        horizon.get("task_specific_contract_file_sha256"),
    )
    contract_rows = [
        row
        for row in task_contract.get("tasks", {}).get("task_versions", [])
        if (row.get("task_key"), row.get("task_version_id")) == selected_key
    ]
    rollout = task_contract.get("rollout", {})
    if (
        len(contract_rows) != 1
        or "sha256:" + digest(contract_rows[0]) != horizon.get("selected_task_row_sha256")
        or "sha256:" + digest(rollout) != horizon.get("rollout_contract_sha256")
        or rollout.get("max_turns") != 600
        or rollout.get("required_task_tools") != ["bash", "submit_report"]
    ):
        raise ValueError("task-specific episode-horizon contract changed")
    aggregate = _bound_bytes(
        evidence_path.parent / horizon.get("aggregate_evidence_path", ""),
        horizon.get("aggregate_evidence_file_sha256"),
    )
    if b"source Fleet job allowed 600 agent steps" not in aggregate or b"p90 154" not in aggregate:
        raise ValueError("aggregate episode-horizon evidence changed")


def validate_config(config: dict, *, relative_to: Path) -> dict | None:
    """Gate reward-marked data inputs before any authenticated Fleet GET."""
    task_set_path = relative_to / config["task_set"]
    if task_set_path.resolve() == (ROOT / TASK_SET_PATH).resolve():
        task_set = _bound_json(task_set_path, SOURCE_IDENTITY["task_set_file_sha256"])
    else:
        try:
            task_set = json.loads(task_set_path.read_bytes())
        except (TypeError, ValueError) as error:
            raise ValueError("RL task set is not valid JSON") from error
        if not isinstance(task_set, dict) or task_set.get("reward_signal_provenance") is None:
            return None
        task_set = _bound_json(task_set_path, SOURCE_IDENTITY["task_set_file_sha256"])
    split = _bound_json(relative_to / config["split"], SOURCE_IDENTITY["split_file_sha256"])
    validate_exact_version_evidence(
        task_set,
        split,
        config.get("limits"),
        task_set_dir=task_set_path.parent,
    )
    return source_proof()


def source_proof() -> dict:
    proof = {
        "schema": SOURCE_PROOF_SCHEMA,
        "task_set_path": TASK_SET_PATH,
        "task_set_file_sha256": SOURCE_IDENTITY["task_set_file_sha256"],
        "task_set_self_sha256": SOURCE_IDENTITY["task_set_self_sha256"],
        "split_path": SPLIT_PATH,
        "split_file_sha256": SOURCE_IDENTITY["split_file_sha256"],
        "split_self_sha256": SOURCE_IDENTITY["split_self_sha256"],
        "exact_version_evidence_path": EXACT_VERSION_EVIDENCE_PATH,
        "exact_version_evidence_file_sha256": SOURCE_IDENTITY["exact_version_evidence_file_sha256"],
        "exact_version_evidence_self_sha256": SOURCE_IDENTITY["exact_version_evidence_self_sha256"],
        "max_turns": 600,
        "required_task_tools": ["bash", "submit_report"],
    }
    proof["sha256"] = "sha256:" + digest(proof)
    return proof


def validate_repository_source(metadata: dict) -> dict:
    """Resolve and validate the exact reviewed source before plan preparation."""
    task_set_path, split_path = ROOT / TASK_SET_PATH, ROOT / SPLIT_PATH
    task_set = _bound_json(task_set_path, SOURCE_IDENTITY["task_set_file_sha256"])
    split = _bound_json(split_path, SOURCE_IDENTITY["split_file_sha256"])
    if (
        metadata.get("selection_sha256") != task_set.get("sha256")
        or metadata.get("split_sha256") != split.get("sha256")
        or metadata.get("limits") != LIMITS
    ):
        raise ValueError("staged reward data does not bind the reviewed source closure")
    validate_exact_version_evidence(
        task_set,
        split,
        metadata["limits"],
        task_set_dir=task_set_path.parent,
    )
    return source_proof()


def validate_source_proof(proof: object, metadata: dict) -> None:
    """Validate the compiler-produced proof without mutable filesystem lookups."""
    if proof != source_proof():
        raise ValueError("reward canary source closure proof changed")
    if (
        metadata.get("selection_sha256") != SOURCE_IDENTITY["task_set_self_sha256"]
        or metadata.get("split_sha256") != SOURCE_IDENTITY["split_self_sha256"]
        or metadata.get("limits") != LIMITS
    ):
        raise ValueError("reward canary data differs from its source closure proof")


def build(config: dict, *, relative_to: Path, client) -> dict:
    """Validate the source-direct/horizon closure, then use the frozen generic builder."""
    # The generic builder owns required-field errors. This conditional keeps
    # its long-standing test/substitution seam while every real task set is
    # inspected before any authenticated Fleet GET.
    if "task_set" in config:
        validate_config(config, relative_to=relative_to)
    from .rl_data import build as generic_build

    return generic_build(config, relative_to=relative_to, client=client)
