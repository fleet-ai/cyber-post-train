"""Create a train corpus that excludes every family in a frozen final-test lock."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Iterable
from pathlib import Path

REQUEST_SCHEMA = "cyber_dense_sft_corpus_final_lock_filter_request_v1"
SELECTION_SCHEMA = "cyber_task_family_exclusion_selection_v1"
RECEIPT_SCHEMA = "cyber_dense_sft_corpus_final_lock_filter_receipt_v1"
DENSE_FORMAT = "pretokenized_assistant_segments_v1"


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


def _known(value: dict, names: set[str], label: str) -> None:
    if not isinstance(value, dict) or value.keys() - names:
        raise ValueError(f"{label} contains unsupported fields")


def atomic_write_json(path: Path, value: object, *, private: bool = False) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    if private:
        os.chmod(temporary, 0o600)
    os.rename(temporary, path)


def atomic_write_jsonl(path: Path, rows: Iterable[dict], *, private: bool = False) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    if private:
        os.chmod(temporary, 0o600)
    os.rename(temporary, path)


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
    for row in inventory.get("tasks", []):
        identity = row.get("task_key"), row.get("task_version_id")
        if not all(isinstance(value, str) and value for value in identity):
            raise ValueError("inventory task identity is incomplete")
        unit = _taxonomy_unit(row)
        if identity in result and result[identity] != unit:
            raise ValueError("inventory task identity is ambiguous")
        result[identity] = unit
    if len(result) != len(inventory.get("tasks", [])) or not result:
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


def build(config: dict, *, relative_to: Path) -> dict:
    """Filter a sealed dense corpus and publish the successor create-once."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    _known(
        config,
        {
            "schema",
            "name",
            "source",
            "protection",
            "selection",
            "destination",
            "materialization",
            "sha256",
        },
        "final-lock corpus filter request",
    )
    _sealed(config, REQUEST_SCHEMA)
    selection = config.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("filter request lacks a selection")
    _sealed(selection, SELECTION_SCHEMA)

    source = config["source"]
    protection = config["protection"]
    destination = config["destination"]
    if (
        destination.get("create_once") is not True
        or destination.get("publication") != "atomic_no_replace"
    ):
        raise ValueError("filter destination must be create-once")

    manifest_path = _path(relative_to, source["manifest_path"])
    inventory_path = _path(relative_to, protection["inventory_path"])
    final_path = _path(relative_to, protection["final_test_lock_path"])
    _bound_file(manifest_path, source["manifest_file_sha256"], "source manifest")
    _bound_file(inventory_path, protection["inventory_file_sha256"], "inventory")
    _bound_file(final_path, protection["final_test_lock_file_sha256"], "final lock")

    manifest = read_mapping(manifest_path)
    inventory = read_mapping(inventory_path)
    final = read_mapping(final_path)
    _sealed(manifest, "cyber_dense_sft_corpus_v1")
    _sealed(inventory, "cyber_study_inventory_v1")
    _sealed(final, "cyber_final_test_lock_v1")
    if (
        manifest["sha256"] != source["manifest_sha256"]
        or manifest["source_sha256"] != source["source_selection_sha256"]
        or manifest["split_sha256"] != source["source_split_sha256"]
        or inventory["sha256"] != protection["inventory_sha256"]
        or final["sha256"] != protection["final_test_lock_sha256"]
        or final["inventory_sha256"] != inventory["sha256"]
        or selection["mapping_inventory_sha256"] != inventory["sha256"]
        or selection["protected_final_test_sha256"] != final["sha256"]
        or selection["source_split_sha256"] != manifest["split_sha256"]
    ):
        raise ValueError("filter request cross-field binding differs")

    train_spec = manifest.get("files", {}).get("train")
    if (
        set(manifest.get("files", {})) != {"train"}
        or not isinstance(train_spec, dict)
        or train_spec.get("format") != DENSE_FORMAT
        or train_spec.get("path") != source["train_parquet_path"]
        or train_spec.get("sha256") != source["train_parquet_sha256"]
        or train_spec.get("rows") != source["rows"]
        or train_spec.get("source_sessions") != source["source_sessions"]
        or train_spec.get("supervised_tokens") != source["supervised_tokens"]
        or len(train_spec.get("task_keys", [])) != source["task_versions"]
    ):
        raise ValueError("source corpus summary differs from filter request")

    index = _inventory_index(inventory)
    final_identities = {(row["task_key"], row["task_version_id"]) for row in final.get("tasks", [])}
    if len(final_identities) != len(final.get("tasks", [])) or not final_identities:
        raise ValueError("final lock contains duplicate or empty identities")
    try:
        protected_units = {index[identity] for identity in final_identities}
    except KeyError as exc:
        raise ValueError("final lock task is absent from the bound inventory") from exc
    if len(protected_units) != selection["protected_task_families"]:
        raise ValueError("protected family count differs from selection")

    source_path = Path(source["data_root"]) / source["train_parquet_path"]
    before = file_sha256(source_path)
    if before != source["train_parquet_sha256"]:
        raise ValueError("source train Parquet digest mismatch")
    source_table = pq.read_table(source_path)
    rows = source_table.to_pylist()
    if len(rows) != source["rows"]:
        raise ValueError("source train Parquet row count differs")

    session_units: dict[str, set[tuple[str, str]]] = {}
    session_actions: dict[str, bool] = {}
    source_identities: set[tuple[str, str]] = set()
    for row in rows:
        identity = row.get("task_key"), row.get("source_task_version_id")
        if identity not in index:
            raise ValueError("source row is absent from the bound inventory")
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
    if {identity[0] for identity in source_identities} != set(train_spec["task_keys"]):
        raise ValueError("source manifest task keys differ from Parquet identities")

    excluded = sorted(source_identities, key=lambda item: (item[0], item[1]))
    excluded = [identity for identity in excluded if index[identity] in protected_units]
    expected_excluded = [
        (row["task_key"], row["task_version_id"])
        for row in selection["excluded_source_task_versions"]
    ]
    if (
        excluded != expected_excluded
        or len(excluded) != selection["excluded_source_task_version_count"]
    ):
        raise ValueError("computed task-family exclusions differ from selection")

    retained = [row for row in rows if not session_actions[row["source_session_id"]]]
    counts = _counts(retained)
    retained_identities = {(row["task_key"], row["source_task_version_id"]) for row in retained}
    acceptance = config["materialization"]["acceptance"]
    if (
        not retained
        or len(retained_identities) != selection["retained_source_task_versions"]
        or len(retained_identities) != acceptance["retained_task_versions"]
        or retained_identities & final_identities
        or any(index[identity] in protected_units for identity in retained_identities)
    ):
        raise ValueError("filtered corpus does not satisfy final-lock isolation")
    # The source Parquet is already tokenizer/mask-qualified and bound by its
    # exact digest. Filtering must preserve each retained row byte-for-byte in
    # logical Arrow form; it must not reinterpret or retokenize the examples.
    if any(
        not isinstance(row.get("input_ids"), list)
        or not row["input_ids"]
        or not isinstance(row.get("loss_mask"), list)
        or len(row["input_ids"]) != len(row["loss_mask"])
        or row.get("token_count") != len(row["input_ids"])
        for row in retained
    ):
        raise ValueError("retained dense row structure is invalid")

    output = Path(destination["data_root"])
    if output.exists():
        raise FileExistsError("create-once corpus destination exists")
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    # mkdir is the no-replace publication claim. If this process stops before
    # RECEIPT.json is written, the claimed directory remains visibly partial
    # and a successor cannot silently reuse or overwrite it.
    output.mkdir(mode=0o700)
    data_path = output / destination["train_parquet_path"]
    temporary_data = data_path.with_name(data_path.name + ".partial")
    pq.write_table(
        pa.Table.from_pylist(retained, schema=source_table.schema),
        temporary_data,
        compression="zstd",
    )
    os.rename(temporary_data, data_path)
    os.chmod(data_path, 0o600)
    if pq.read_table(data_path).to_pylist() != retained:
        raise ValueError("filtered Parquet readback differs")
    if file_sha256(source_path) != before:
        raise ValueError("source train Parquet changed during filtering")

    private_rows = [
        {
            "source_session_id": session,
            "action": "exclude" if excluded_action else "retain",
            "application": next(iter(session_units[session]))[0],
            "task_family": next(iter(session_units[session]))[1],
        }
        for session, excluded_action in sorted(session_actions.items())
    ]
    private_path = output / destination["private_selection_path"]
    atomic_write_jsonl(private_path, private_rows, private=True)

    successor = copy.deepcopy(manifest)
    successor.pop("sha256", None)
    successor["source_sha256"] = file_sha256(private_path)
    successor["split_sha256"] = selection["sha256"]
    successor["files"] = {
        "train": {
            "path": destination["train_parquet_path"],
            "sha256": file_sha256(data_path),
            **counts,
        }
    }
    successor["split_exclusions"] = {
        "source_split_exclusions": manifest.get("split_exclusions", {}),
        "frozen_final_task_families": len(protected_units),
        "excluded_source_task_versions": len(excluded),
        "excluded_source_sessions": sum(session_actions.values()),
    }
    successor["holdout_filter_provenance"] = {
        "request_sha256": config["sha256"],
        "selection_sha256": selection["sha256"],
        "source_manifest_file_sha256": source["manifest_file_sha256"],
        "source_manifest_sha256": manifest["sha256"],
        "source_train_parquet_sha256": before,
        "source_selection_sha256": source["source_selection_sha256"],
        "final_test_lock_file_sha256": protection["final_test_lock_file_sha256"],
        "final_test_lock_sha256": final["sha256"],
        "inventory_file_sha256": protection["inventory_file_sha256"],
        "inventory_sha256": inventory["sha256"],
        "private_selection_file_sha256": file_sha256(private_path),
        "excluded_source_task_versions": selection["excluded_source_task_versions"],
    }
    successor.setdefault("builder_sha256", {})["holdout_filter.py"] = file_sha256(Path(__file__))
    successor.setdefault("limitations", []).append(
        "Every complete source session sharing an application/task-family unit "
        "with the frozen final test lock is excluded."
    )
    successor["sha256"] = "sha256:" + digest(successor)
    manifest_output = output / destination["manifest_path"]
    atomic_write_json(manifest_output, successor, private=True)

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "request_sha256": config["sha256"],
        "selection_sha256": selection["sha256"],
        "source_manifest_file_sha256": source["manifest_file_sha256"],
        "source_manifest_sha256": manifest["sha256"],
        "source_train_parquet_sha256": before,
        "source_stable": file_sha256(source_path) == before,
        "final_test_lock_file_sha256": protection["final_test_lock_file_sha256"],
        "final_test_lock_sha256": final["sha256"],
        "inventory_file_sha256": protection["inventory_file_sha256"],
        "inventory_sha256": inventory["sha256"],
        "manifest_file_sha256": file_sha256(manifest_output),
        "manifest_sha256": successor["sha256"],
        "train_parquet_sha256": successor["files"]["train"]["sha256"],
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
    receipt["sha256"] = "sha256:" + digest(receipt)
    # The receipt is written last. Its absence means this claimed output is
    # incomplete and must never be consumed as a corpus.
    atomic_write_json(output / destination["receipt_path"], receipt, private=True)

    return {
        "output": str(output),
        "manifest_sha256": successor["sha256"],
        "receipt_sha256": receipt["sha256"],
        "train": {
            "rows": counts["rows"],
            "source_sessions": counts["source_sessions"],
            "task_versions": len(retained_identities),
            "supervised_tokens": counts["supervised_tokens"],
        },
        "submitted": False,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    request_path = Path(args.request).resolve()
    result = build(read_mapping(request_path), relative_to=request_path.parent.parent.parent)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
