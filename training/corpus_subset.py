"""Create a train-only dense SFT corpus from immutable, already-qualified corpora."""

from __future__ import annotations

import collections
import os
from pathlib import Path

from cyber_post_train.jobs import digest

from .corpus import local_tokenizer
from .io import atomic_write_json, file_sha256, iter_jsonl
from .sft import _known, read_mapping
from .sft_runtime import DENSE_FORMAT, dense_rows
from .study_split_v2 import SCHEMA as SPLIT_SCHEMA
from .study_split_v2 import validate as validate_split


def _path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _sealed(value: dict, schema: str) -> None:
    if value.get("schema") != schema or value.get("sha256") != "sha256:" + digest(
        {key: item for key, item in value.items() if key != "sha256"}
    ):
        raise ValueError(f"invalid {schema} receipt")


def _qualified(episode: dict, *, model: str, policy: str) -> bool:
    coverage = episode.get("coverage", {})
    assistant = coverage.get("assistant_responses")
    submits = coverage.get("submit_report_responses")
    tokens = coverage.get("supervised_tokens")
    submit_tokens = coverage.get("submit_report_tokens")
    return bool(
        episode.get("schema") == "cyber_sft_episode_metadata_v1"
        and episode.get("source_kind") == "teacher"
        and episode.get("model_id") == model
        and episode.get("validity") == "valid"
        and episode.get("verified_success") is True
        and coverage.get("status") == "certified"
        and coverage.get("target_policy_sha256") == policy
        and type(coverage.get("non_submit_tool_responses")) is int
        and coverage["non_submit_tool_responses"] >= 1
        and type(coverage.get("completed_non_submit_tool_rounds")) is int
        and coverage["completed_non_submit_tool_rounds"] >= 1
        and type(assistant) is int
        and type(submits) is int
        and assistant > submits >= 0
        and submits / assistant <= 0.5
        and type(tokens) is int
        and type(submit_tokens) is int
        and tokens > 0
        and 0 <= submit_tokens / tokens <= 0.5
    )


def build(config: dict, *, relative_to: Path) -> dict:
    """Filter exact-version train rows, deduplicate them, and write create-once output."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    _known(
        config,
        {
            "schema",
            "inventory",
            "study_split",
            "model_lock",
            "tokenizer_root",
            "source_kind",
            "source_model",
            "parents",
            "output",
        },
        "dense subset",
    )
    if config.get("schema") != "cyber_dense_sft_subset_request_v1":
        raise ValueError("unsupported dense subset request")
    if config.get("source_kind") != "teacher" or not isinstance(config.get("source_model"), str):
        raise ValueError("this path requires one explicit teacher source model")
    output = _path(relative_to, config["output"])
    if output.exists():
        raise FileExistsError("create-once corpus destination exists")

    inventory_path = _path(relative_to, config["inventory"])
    split_path = _path(relative_to, config["study_split"])
    split = read_mapping(split_path)
    if split.get("schema") != SPLIT_SCHEMA:
        raise ValueError("expected the reviewed representative study split")
    recorded_inventory = Path(split["inventory"]["path"])
    if recorded_inventory.resolve() != inventory_path.resolve():
        raise ValueError("configured inventory differs from the study split binding")
    validate_split(split, inventory_path=recorded_inventory)
    train = {
        (row["task_key"], row["task_version_id"]): row["group_id"]
        for row in split["training_split"]["tasks"]
    }
    held_out = {
        (row["task_key"], row["task_version_id"])
        for name in ("dev", "final_test")
        for row in split["evaluation"][name]["tasks"]
    }

    lock_path = _path(relative_to, config["model_lock"])
    lock = read_mapping(lock_path)
    tokenizer, tokenizer_identity = local_tokenizer(
        lock, _path(relative_to, config["tokenizer_root"])
    )
    parents = config.get("parents")
    if not isinstance(parents, list) or not parents:
        raise ValueError("at least one immutable parent corpus is required")

    rows: dict[str, dict] = {}
    sources: dict[str, dict] = {}
    selected_ids: set[str] = set()
    provenance = []
    target_policy = None
    maximum = context = None
    role_ids: dict[str, set[str]] = collections.defaultdict(set)
    for parent in parents:
        _known(
            parent,
            {"corpus_manifest", "source_selection", "coverage_manifest"},
            "parent corpus",
        )
        manifest_path = _path(relative_to, parent["corpus_manifest"])
        selection_path = _path(relative_to, parent["source_selection"])
        coverage_path = _path(relative_to, parent["coverage_manifest"])
        manifest, selection, coverage_manifest = map(
            read_mapping, (manifest_path, selection_path, coverage_path)
        )
        _sealed(manifest, "cyber_dense_sft_corpus_v1")
        _sealed(selection, "cyber_sft_source_selection_v1")
        _sealed(coverage_manifest, "cyber_sft_coverage_extraction_v1")
        if manifest.get("validation_mode") != "task_outcomes_only" or set(manifest["files"]) != {
            "train"
        }:
            raise ValueError("parent must be a train-only task-outcome corpus")
        if manifest["tokenizer"] != tokenizer_identity:
            raise ValueError("parent tokenizer identity differs")
        if selection.get("status") != "ready" or selection.get("models") != [
            config["source_model"]
        ]:
            raise ValueError("parent source selection is not ready for the requested model")
        if manifest["source_selection"] != {
            "study_split_sha256": selection["study_split_sha256"],
            "source_selection_sha256": selection["sha256"],
            "target_policy_sha256": selection["target_policy_sha256"],
            "selected_episode_count": len(selection["selected_episode_ids"]),
        }:
            raise ValueError("parent manifest/source-selection binding differs")
        if coverage_manifest["target_policy_sha256"] != selection["target_policy_sha256"]:
            raise ValueError("coverage and source-selection target policies differ")
        for name, expected in coverage_manifest["files"].items():
            if file_sha256(coverage_path.parent / name) != expected:
                raise ValueError("coverage file digest mismatch")
        target_policy = target_policy or selection["target_policy_sha256"]
        maximum = maximum or manifest["max_length"]
        context = context if context is not None else manifest["context_tokens"]
        if (
            target_policy != selection["target_policy_sha256"]
            or maximum != manifest["max_length"]
            or context != manifest["context_tokens"]
        ):
            raise ValueError("parent target/window policies differ")

        episodes = {}
        for episode in iter_jsonl(coverage_path.parent / "episodes.jsonl"):
            _sealed(episode, "cyber_sft_episode_metadata_v1")
            episode_id = episode["episode_id"]
            if episode_id in episodes:
                raise ValueError("duplicate coverage episode")
            episodes[episode_id] = episode
        evidence = {item["episode_id"]: item for item in selection["selected_evidence"]}
        chosen = set(selection["selected_episode_ids"])
        if chosen != set(evidence) or not chosen <= set(episodes):
            raise ValueError("selected episode evidence is incomplete")
        for episode_id in chosen:
            episode = episodes[episode_id]
            expected = {
                "episode_id": episode_id,
                "metadata_sha256": episode["sha256"],
                "trace_sha256": episode["trace_sha256"],
                "acceptance_sha256": episode["acceptance_sha256"],
                "normalized_record_sha256": episode["normalized_record_sha256"],
            }
            if evidence[episode_id] != expected:
                raise ValueError("selected evidence differs from coverage metadata")
            identity = (episode["task_key"], episode["task_version_id"])
            role = (
                "train" if identity in train else "held_out" if identity in held_out else "outside"
            )
            role_ids[role].add(episode_id)
            if identity in train and _qualified(
                episode, model=config["source_model"], policy=target_policy
            ):
                previous = sources.setdefault(episode_id, episode)
                if previous != episode:
                    raise ValueError("episode metadata differs between parent corpora")
                selected_ids.add(episode_id)

        train_spec = manifest["files"]["train"]
        parquet_path = manifest_path.parent / train_spec["path"]
        if file_sha256(parquet_path) != train_spec["sha256"]:
            raise ValueError("parent Parquet digest mismatch")
        parent_rows = pq.read_table(parquet_path).to_pylist()
        if len(parent_rows) != train_spec["rows"]:
            raise ValueError("parent Parquet row count differs")
        present = set()
        for row in parent_rows:
            episode_id = row["source_session_id"]
            if episode_id not in selected_ids or episode_id not in chosen:
                continue
            episode = sources[episode_id]
            if row["task_key"] != episode["task_key"] or row["source_model"] != episode["model_id"]:
                raise ValueError("dense row source provenance differs from certified metadata")
            present.add(episode_id)
            prior = rows.setdefault(row["window_id"], row)
            if prior != row:
                raise ValueError("duplicate window identity has different payload")
        expected_present = chosen & selected_ids
        if present != expected_present:
            raise ValueError("a selected qualified episode lacks dense rows")
        provenance.append(
            {
                "corpus_manifest_file_sha256": file_sha256(manifest_path),
                "corpus_manifest_sha256": manifest["sha256"],
                "parent_train_sha256": train_spec["sha256"],
                "source_sha256": manifest["source_sha256"],
                "source_selection_file_sha256": file_sha256(selection_path),
                "source_selection_sha256": selection["sha256"],
                "coverage_manifest_file_sha256": file_sha256(coverage_path),
                "coverage_manifest_sha256": coverage_manifest["sha256"],
            }
        )

    values = sorted(
        rows.values(),
        key=lambda row: (
            row["task_key"],
            row["source_session_id"],
            row["segment_id"],
            row["window_id"],
        ),
    )
    present_ids = {row["source_session_id"] for row in values}
    if present_ids != selected_ids or not values:
        raise ValueError("selected source/window coverage is incomplete")
    first = {}
    for row in values:
        first.setdefault(row["source_session_id"], row)
    counts = {
        "rows": len(values),
        "task_keys": sorted({row["task_key"] for row in values}),
        "format": DENSE_FORMAT,
        "source_sessions": len(first),
        "supervised_tokens": sum(row["target_token_count"] for row in values),
        "assistant_responses": sum(len(row["target_spans"]) for row in values),
        "source_total_assistant_responses": sum(
            row["source_assistant_count"] for row in first.values()
        ),
        "excluded_assistant_responses": sum(
            len(row["excluded_assistant_targets"]) for row in first.values()
        ),
    }
    dense_rows(values, counts, max_length=maximum, vocab_size=len(tokenizer))
    selected_tasks = {
        (sources[episode_id]["task_key"], sources[episode_id]["task_version_id"])
        for episode_id in selected_ids
    }
    if not selected_tasks <= set(train) or selected_tasks & held_out:
        raise ValueError("selected corpus crosses the frozen train boundary")

    output.mkdir(parents=True, mode=0o700)
    data_path = output / "train.parquet"
    pq.write_table(pa.Table.from_pylist(values), data_path, compression="zstd")
    os.chmod(data_path, 0o600)
    if pq.read_table(data_path).to_pylist() != values:
        raise ValueError("Parquet readback differs")
    source_selection = {
        "schema": "cyber_dense_sft_subset_selection_v1",
        "study_split_sha256": split["sha256"],
        "target_policy_sha256": target_policy,
        "source_kind": config["source_kind"],
        "source_model": config["source_model"],
        "selected_episode_ids": sorted(selected_ids),
        "selected_evidence": [
            {
                "episode_id": episode_id,
                "metadata_sha256": sources[episode_id]["sha256"],
                "trace_sha256": sources[episode_id]["trace_sha256"],
                "acceptance_sha256": sources[episode_id]["acceptance_sha256"],
                "normalized_record_sha256": sources[episode_id]["normalized_record_sha256"],
            }
            for episode_id in sorted(selected_ids)
        ],
    }
    source_selection["sha256"] = "sha256:" + digest(source_selection)
    atomic_write_json(output / "source-selection.json", source_selection, private=True)
    manifest = {
        "schema": "cyber_dense_sft_corpus_v1",
        "source_sha256": source_selection["sha256"],
        "split_sha256": split["sha256"],
        "tokenizer": tokenizer_identity,
        "files": {"train": {"path": "train.parquet", "sha256": file_sha256(data_path), **counts}},
        "train_models": [config["source_model"]],
        "max_length": maximum,
        "context_tokens": context,
        "dev_windows": 0,
        "validation_mode": "task_outcomes_only",
        "split_exclusions": {
            "held_out_parent_sources": len(role_ids["held_out"]),
            "outside_parent_sources": len(role_ids["outside"]),
        },
        "whole_source_exclusions": {},
        "subset_provenance": {
            "inventory_file_sha256": file_sha256(inventory_path),
            "inventory_sha256": split["inventory"]["logical_sha256"],
            "study_split_file_sha256": file_sha256(split_path),
            "selected_task_versions": len(selected_tasks),
            "selected_source_sessions": len(selected_ids),
            "selected_group_ids_sha256": "sha256:"
            + digest(sorted(train[identity] for identity in selected_tasks)),
            "source_selection_file_sha256": file_sha256(output / "source-selection.json"),
            "parents": provenance,
            "harness_provenance": (
                "Bound by immutable trace/normalized-record digests; the parent metadata does "
                "not expose a separate harness label, so no harness is inferred."
            ),
        },
        "builder_sha256": {"corpus_subset.py": file_sha256(Path(__file__))},
        "limitations": [
            "Visible assistant actions only; private reasoning is omitted.",
            "The 50-task train split has certified teacher coverage on only the selected versions.",
            "Fresh Fleet dev outcomes, not teacher-token loss, select checkpoints.",
        ],
    }
    manifest["sha256"] = "sha256:" + digest(manifest)
    atomic_write_json(output / "manifest.json", manifest, private=True)
    return {
        "output": str(output),
        "manifest_sha256": manifest["sha256"],
        "train": {
            "rows": counts["rows"],
            "source_sessions": counts["source_sessions"],
            "task_versions": len(selected_tasks),
            "supervised_tokens": counts["supervised_tokens"],
        },
        "submitted": False,
    }
