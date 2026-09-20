"""Independently verify a final-test-family-free dense SFT corpus."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

REQUEST_SCHEMA = "cyber_dense_sft_corpus_final_lock_filter_request_v1"
SELECTION_SCHEMA = "cyber_task_family_exclusion_selection_v1"
RECEIPT_SCHEMA = "cyber_dense_sft_corpus_final_lock_filter_receipt_v1"
VERIFICATION_SCHEMA = "cyber_dense_sft_corpus_final_lock_verification_v1"
DENSE_FORMAT = "pretokenized_assistant_segments_v1"
FINAL_LIMITATION = (
    "Every complete source session sharing an application/task-family unit "
    "with the frozen final test lock is excluded."
)


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def read_mapping(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("configuration/manifest must be a mapping")
    return value


def _path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _sealed(value: dict, schema: str) -> None:
    expected = "sha256:" + digest({key: item for key, item in value.items() if key != "sha256"})
    if value.get("schema") != schema or value.get("sha256") != expected:
        raise ValueError(f"invalid {schema} receipt")


def _bound_file(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or file_sha256(path) != expected:
        raise ValueError(f"{label} file digest mismatch")


def _taxonomy_unit(row: dict) -> tuple[str, str]:
    taxonomy = row.get("taxonomy", {})
    application = taxonomy.get("application", {})
    family = taxonomy.get("task_family", {})
    if (
        application.get("status") != "verified"
        or family.get("status") != "verified"
        or not isinstance(application.get("value"), str)
        or not application["value"]
        or not isinstance(family.get("value"), str)
        or not family["value"]
    ):
        raise ValueError("inventory task has unknown or ambiguous split identity")
    return application["value"], family["value"]


def _inventory_index(inventory: dict) -> dict[tuple[str, str], tuple[str, str]]:
    result: dict[tuple[str, str], tuple[str, str]] = {}
    tasks = inventory.get("tasks", [])
    for row in tasks:
        identity = row.get("task_key"), row.get("task_version_id")
        if not all(isinstance(value, str) and value for value in identity):
            raise ValueError("inventory task identity is incomplete")
        unit = _taxonomy_unit(row)
        if identity in result and result[identity] != unit:
            raise ValueError("inventory task identity is ambiguous")
        result[identity] = unit
    if len(result) != len(tasks) or not result:
        raise ValueError("inventory contains duplicate or empty task identities")
    return result


def _counts(rows: list[dict]) -> dict:
    first: dict[str, dict] = {}
    for row in rows:
        first.setdefault(row["source_session_id"], row)
    return {
        "rows": len(rows),
        "task_keys": sorted({row["task_key"] for row in rows}),
        "format": DENSE_FORMAT,
        "source_sessions": len(first),
        "supervised_tokens": sum(row["target_token_count"] for row in rows),
        "assistant_responses": sum(len(row["target_spans"]) for row in rows),
        "source_total_assistant_responses": sum(
            row["source_assistant_count"] for row in first.values()
        ),
        "excluded_assistant_responses": sum(
            len(row["excluded_assistant_targets"]) for row in first.values()
        ),
    }


def _dense_rows(table: object) -> list[dict]:
    rows = table.to_pylist()
    if any(
        not isinstance(row.get("input_ids"), list)
        or not row["input_ids"]
        or not isinstance(row.get("loss_mask"), list)
        or len(row["input_ids"]) != len(row["loss_mask"])
        or row.get("token_count") != len(row["input_ids"])
        or not isinstance(row.get("target_token_count"), int)
        or row["target_token_count"] < 1
        for row in rows
    ):
        raise ValueError("dense Parquet row structure is invalid")
    return rows


def verify(config: dict, *, relative_to: Path) -> dict:
    """Verify a materialized successor without changing it or its source."""
    import pyarrow.parquet as pq

    if config.get("schema") != REQUEST_SCHEMA:
        raise ValueError("unexpected holdout-filter request schema")
    _sealed(config, REQUEST_SCHEMA)
    selection = config.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("filter request lacks a selection")
    _sealed(selection, SELECTION_SCHEMA)

    source = config.get("source", {})
    protection = config.get("protection", {})
    destination = config.get("destination", {})
    materialization = config.get("materialization", {})
    if (
        destination.get("create_once") is not True
        or destination.get("publication") != "atomic_no_replace"
    ):
        raise ValueError("filter destination is not create-once")

    manifest_path = _path(relative_to, source["manifest_path"])
    inventory_path = _path(relative_to, protection["inventory_path"])
    final_path = _path(relative_to, protection["final_test_lock_path"])
    bound_inputs = (
        (manifest_path, source["manifest_file_sha256"], "source manifest"),
        (inventory_path, protection["inventory_file_sha256"], "inventory"),
        (final_path, protection["final_test_lock_file_sha256"], "final lock"),
    )
    for path, expected, label in bound_inputs:
        _bound_file(path, expected, label)

    source_manifest = read_mapping(manifest_path)
    inventory = read_mapping(inventory_path)
    final = read_mapping(final_path)
    _sealed(source_manifest, "cyber_dense_sft_corpus_v1")
    _sealed(inventory, "cyber_study_inventory_v1")
    _sealed(final, "cyber_final_test_lock_v1")
    if (
        source_manifest["sha256"] != source["manifest_sha256"]
        or source_manifest["source_sha256"] != source["source_selection_sha256"]
        or source_manifest["split_sha256"] != source["source_split_sha256"]
        or inventory["sha256"] != protection["inventory_sha256"]
        or final["sha256"] != protection["final_test_lock_sha256"]
        or final["inventory_sha256"] != inventory["sha256"]
        or selection["mapping_inventory_sha256"] != inventory["sha256"]
        or selection["protected_final_test_sha256"] != final["sha256"]
        or selection["source_split_sha256"] != source_manifest["split_sha256"]
    ):
        raise ValueError("filter request cross-field binding differs")

    source_spec = source_manifest.get("files", {}).get("train")
    if (
        set(source_manifest.get("files", {})) != {"train"}
        or not isinstance(source_spec, dict)
        or source_spec.get("format") != DENSE_FORMAT
        or source_spec.get("path") != source["train_parquet_path"]
        or source_spec.get("sha256") != source["train_parquet_sha256"]
    ):
        raise ValueError("source corpus file binding differs")

    index = _inventory_index(inventory)
    final_identities = {(row["task_key"], row["task_version_id"]) for row in final.get("tasks", [])}
    if len(final_identities) != len(final.get("tasks", [])) or not final_identities:
        raise ValueError("final lock contains duplicate or empty identities")
    try:
        protected_units = {index[identity] for identity in final_identities}
    except KeyError as exc:
        raise ValueError("final lock task is absent from inventory") from exc
    if len(protected_units) != selection["protected_task_families"]:
        raise ValueError("protected family count differs from selection")

    source_path = Path(source["data_root"]) / source["train_parquet_path"]
    source_before = file_sha256(source_path)
    if source_before != source["train_parquet_sha256"]:
        raise ValueError("source train Parquet digest mismatch")
    source_table = pq.read_table(source_path)
    source_rows = _dense_rows(source_table)
    source_counts = _counts(source_rows)
    source_identities: set[tuple[str, str]] = set()
    session_units: dict[str, set[tuple[str, str]]] = {}
    session_actions: dict[str, bool] = {}
    for row in source_rows:
        identity = row.get("task_key"), row.get("source_task_version_id")
        if identity not in index:
            raise ValueError("source row is absent from inventory")
        session = row.get("source_session_id")
        if not isinstance(session, str) or not session:
            raise ValueError("source row lacks a session identity")
        unit = index[identity]
        session_units.setdefault(session, set()).add(unit)
        session_actions.setdefault(session, unit in protected_units)
        if session_actions[session] != (unit in protected_units):
            raise ValueError("partial source-session removal would be required")
        source_identities.add(identity)
    if any(len(units) != 1 for units in session_units.values()):
        raise ValueError("one source session spans multiple task families")
    expected_source_spec = {
        "path": source["train_parquet_path"],
        "sha256": source_before,
        **source_counts,
    }
    if (
        expected_source_spec != source_spec
        or len(source_identities) != source["task_versions"]
        or len(source_rows) != source["rows"]
        or source_counts["source_sessions"] != source["source_sessions"]
        or source_counts["supervised_tokens"] != source["supervised_tokens"]
    ):
        raise ValueError("source Parquet counts differ from its sealed summary")

    excluded = sorted(
        (identity for identity in source_identities if index[identity] in protected_units),
        key=lambda item: (item[0], item[1]),
    )
    expected_excluded = [
        (row["task_key"], row["task_version_id"])
        for row in selection["excluded_source_task_versions"]
    ]
    if (
        excluded != expected_excluded
        or len(excluded) != selection["excluded_source_task_version_count"]
    ):
        raise ValueError("computed family exclusions differ from selection")
    expected_rows = [row for row in source_rows if not session_actions[row["source_session_id"]]]

    output = Path(destination["data_root"])
    expected_names = {
        destination["train_parquet_path"],
        destination["private_selection_path"],
        destination["manifest_path"],
        destination["receipt_path"],
    }
    if (
        not output.is_dir()
        or {path.name for path in output.iterdir()} != expected_names
        or not all((output / name).is_file() for name in expected_names)
    ):
        raise ValueError("successor corpus is absent, partial, or contains extras")

    data_path = output / destination["train_parquet_path"]
    private_path = output / destination["private_selection_path"]
    successor_path = output / destination["manifest_path"]
    receipt_path = output / destination["receipt_path"]
    data_before = file_sha256(data_path)
    private_digest = file_sha256(private_path)
    successor_file_before = file_sha256(successor_path)
    receipt_file_before = file_sha256(receipt_path)
    successor = read_mapping(successor_path)
    receipt = read_mapping(receipt_path)
    _sealed(successor, "cyber_dense_sft_corpus_v1")
    _sealed(receipt, RECEIPT_SCHEMA)

    train_spec = successor.get("files", {}).get("train")
    if (
        set(successor.get("files", {})) != {"train"}
        or not isinstance(train_spec, dict)
        or train_spec.get("path") != destination["train_parquet_path"]
        or train_spec.get("format") != DENSE_FORMAT
        or train_spec.get("sha256") != data_before
        or receipt.get("manifest_file_sha256") != successor_file_before
        or receipt.get("manifest_sha256") != successor["sha256"]
        or receipt.get("train_parquet_sha256") != data_before
    ):
        raise ValueError("successor manifest/receipt file binding differs")

    output_table = pq.read_table(data_path)
    output_rows = _dense_rows(output_table)
    if output_table.schema != source_table.schema or output_rows != expected_rows:
        raise ValueError("successor rows differ from the expected source subset")
    counts = _counts(output_rows)
    retained_identities = {(row["task_key"], row["source_task_version_id"]) for row in output_rows}
    if any(identity not in index for identity in retained_identities):
        raise ValueError("successor row is absent from inventory")
    exact_overlap = retained_identities & final_identities
    family_overlap = {index[identity] for identity in retained_identities} & protected_units
    acceptance = materialization.get("acceptance", {})
    if (
        exact_overlap
        or family_overlap
        or len(retained_identities) != 35
        or len(retained_identities) != selection["retained_source_task_versions"]
        or len(retained_identities) != acceptance.get("retained_task_versions")
        or acceptance.get("final_test_exact_identity_overlap") != 0
        or acceptance.get("final_test_task_family_overlap") != 0
        or acceptance.get("all_output_task_identities_known_to_inventory") is not True
    ):
        raise ValueError("successor corpus violates final-test isolation")

    expected_train_spec = {
        "path": destination["train_parquet_path"],
        "sha256": data_before,
        **counts,
    }
    if train_spec != expected_train_spec:
        raise ValueError("successor Parquet counts differ from its manifest")

    builder_path = Path(__file__).with_name("holdout_filter.py")
    if not builder_path.is_file():
        raise ValueError("materializer source is absent from verifier bundle")
    expected_successor = copy.deepcopy(source_manifest)
    expected_successor.pop("sha256", None)
    expected_successor["source_sha256"] = private_digest
    expected_successor["split_sha256"] = selection["sha256"]
    expected_successor["files"] = {"train": expected_train_spec}
    expected_successor["split_exclusions"] = {
        "source_split_exclusions": source_manifest.get("split_exclusions", {}),
        "frozen_final_task_families": len(protected_units),
        "excluded_source_task_versions": len(excluded),
        "excluded_source_sessions": sum(session_actions.values()),
    }
    expected_successor["holdout_filter_provenance"] = {
        "request_sha256": config["sha256"],
        "selection_sha256": selection["sha256"],
        "source_manifest_file_sha256": source["manifest_file_sha256"],
        "source_manifest_sha256": source_manifest["sha256"],
        "source_train_parquet_sha256": source_before,
        "source_selection_sha256": source["source_selection_sha256"],
        "final_test_lock_file_sha256": protection["final_test_lock_file_sha256"],
        "final_test_lock_sha256": final["sha256"],
        "inventory_file_sha256": protection["inventory_file_sha256"],
        "inventory_sha256": inventory["sha256"],
        "private_selection_file_sha256": private_digest,
        "excluded_source_task_versions": selection["excluded_source_task_versions"],
    }
    expected_successor.setdefault("builder_sha256", {})["holdout_filter.py"] = file_sha256(
        builder_path
    )
    expected_successor.setdefault("limitations", []).append(FINAL_LIMITATION)
    expected_successor["sha256"] = "sha256:" + digest(expected_successor)
    if successor != expected_successor:
        raise ValueError("public successor manifest differs from reconstruction")

    expected_receipt = {
        "schema": RECEIPT_SCHEMA,
        "request_sha256": config["sha256"],
        "selection_sha256": selection["sha256"],
        "source_manifest_file_sha256": source["manifest_file_sha256"],
        "source_manifest_sha256": source_manifest["sha256"],
        "source_train_parquet_sha256": source_before,
        "source_stable": True,
        "final_test_lock_file_sha256": protection["final_test_lock_file_sha256"],
        "final_test_lock_sha256": final["sha256"],
        "inventory_file_sha256": protection["inventory_file_sha256"],
        "inventory_sha256": inventory["sha256"],
        "manifest_file_sha256": successor_file_before,
        "manifest_sha256": successor["sha256"],
        "train_parquet_sha256": data_before,
        "rows": counts["rows"],
        "source_sessions": counts["source_sessions"],
        "task_versions": len(retained_identities),
        "supervised_tokens": counts["supervised_tokens"],
        "task_keys": counts["task_keys"],
        "excluded_source_task_versions": selection["excluded_source_task_versions"],
        "final_test_exact_identity_overlap": 0,
        "final_test_task_family_overlap": 0,
        "create_once": True,
        "public_manifest_handoff": destination["public_manifest_path"],
    }
    expected_receipt["sha256"] = "sha256:" + digest(expected_receipt)
    if receipt != expected_receipt:
        raise ValueError("successor receipt differs from reconstruction")

    if (
        file_sha256(source_path) != source_before
        or any(file_sha256(path) != expected for path, expected, _ in bound_inputs)
        or file_sha256(data_path) != data_before
        or file_sha256(private_path) != private_digest
        or file_sha256(successor_path) != successor_file_before
        or file_sha256(receipt_path) != receipt_file_before
    ):
        raise ValueError("source or successor changed during verification")

    return {
        "schema": VERIFICATION_SCHEMA,
        "status": "verified",
        "output": str(output),
        "request_sha256": config["sha256"],
        "selection_sha256": selection["sha256"],
        "source_train_parquet_sha256": source_before,
        "source_stable": True,
        "manifest_file_sha256": successor_file_before,
        "manifest_sha256": successor["sha256"],
        "receipt_file_sha256": receipt_file_before,
        "receipt_sha256": receipt["sha256"],
        "train_parquet_sha256": data_before,
        "train": {
            "rows": counts["rows"],
            "source_sessions": counts["source_sessions"],
            "task_versions": len(retained_identities),
            "supervised_tokens": counts["supervised_tokens"],
        },
        "final_test_exact_identity_overlap": 0,
        "final_test_task_family_overlap": 0,
        "public_manifest": successor,
        "submitted": False,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    request_path = Path(args.request).resolve()
    result = verify(read_mapping(request_path), relative_to=request_path.parent.parent.parent)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
