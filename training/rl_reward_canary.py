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
TASK_SET_PATH = "configs/data/qwen38-rl-reward-canary-task-set-v1.json"
SPLIT_PATH = "configs/data/qwen38-rl-reward-canary-split-v1.json"
EXACT_VERSION_EVIDENCE_PATH = "configs/data/qwen38-rl-reward-canary-exact-version-evidence-v1.json"
SOURCE_IDENTITY = {
    "task_set_file_sha256": (
        "sha256:113e561e2fa3339670a846e7e76985a0f5d09ce250ebedddbac8408c2c9c8a79"
    ),
    "task_set_self_sha256": (
        "sha256:608b8c47790fd6e9fef6e86115a7b624273589b8b11e920224db54b7b848e965"
    ),
    "split_file_sha256": (
        "sha256:bcdc058dcb5039c5e68781fea939eaaa928a77f0698d5e8802f457b291f4278c"
    ),
    "split_self_sha256": (
        "sha256:fe77cff7256c8c7554eceaf228b855ccc506c2035882d459499fe9b5505ea7db"
    ),
    "exact_version_evidence_file_sha256": (
        "sha256:f9f9594d36a6bab11b83d2c1c4e213b1ca88fa98b8d42885cbd83c4f7b80851b"
    ),
    "exact_version_evidence_self_sha256": (
        "sha256:d83d8ff2ae66ecc945a439c2a6f47f118a19b928d13721faec4cd28d5b87ee77"
    ),
    # No authoritative v2 GET packages have been reviewed. These stay null so
    # a newly self-digested repo package cannot silently authorize training.
    "eligible_source_observation_file_sha256": None,
    "eligible_source_response_file_sha256": None,
    "eligible_source_journal_file_sha256": None,
    "metadata_only_successor_observation_file_sha256": None,
    "metadata_only_successor_response_file_sha256": None,
    "metadata_only_successor_journal_file_sha256": None,
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


def validate_exact_version_evidence(
    task_set: dict,
    split: dict,
    limits: dict,
    *,
    task_set_dir: Path,
) -> None:
    """Validate non-content successor eligibility and the 600-turn-only horizon."""
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
    successor = evidence.get("metadata_only_successor", {})
    successor_receipt = _version_receipt(
        successor.get("version_receipt"),
        evidence_path=evidence_path,
        role="metadata_only_successor",
        task_key=selected_key[0],
        task_version_id=selected_key[1],
    )
    if (
        successor.get("source_task_version_id") != source.get("task_version_id")
        or successor.get("successor_task_version_id") != selected_key[1]
        or successor.get("only_semantic_diff_path") != "/metadata/tools"
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

    differing_fields = sorted(
        key
        for key in _VERSION_FIELD_NAMES
        if source_receipt["field_sha256"][key] != successor_receipt["field_sha256"][key]
    )
    receipt_equivalent = (
        not differing_fields
        and source_receipt["metadata_without_tools_sha256"]
        == successor_receipt["metadata_without_tools_sha256"]
    )
    derived_status = (
        "eligible_authoritative_metadata_only_successor"
        if receipt_equivalent
        else "blocked_authoritative_receipt_mismatch"
    )
    if (
        selected.get("eligibility_status") != derived_status
        or task_set.get("training_data_eligible") is not receipt_equivalent
    ):
        raise ValueError("reward-canary eligibility assertion is not receipt-derived")

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
    if not receipt_equivalent:
        raise ValueError(
            "authoritative source/successor receipts are not metadata-only equivalent: "
            + ",".join(differing_fields)
        )


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
    """Validate the successor/horizon closure, then use the frozen generic builder."""
    # The generic builder owns required-field errors. This conditional keeps
    # its long-standing test/substitution seam while every real task set is
    # inspected before any authenticated Fleet GET.
    if "task_set" in config:
        validate_config(config, relative_to=relative_to)
    from .rl_data import build as generic_build

    return generic_build(config, relative_to=relative_to, client=client)
