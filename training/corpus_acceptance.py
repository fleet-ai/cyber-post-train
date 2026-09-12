"""Verify a private dense corpus after materialization without decoding it.

The Parquet payload is read only as opaque bytes to recompute its SHA-256.  No
rows, token arrays, prompts, traces, answers, flags, scores, or outcomes are
decoded or inspected.  This complements metadata-only source selection: the
builder owns semantic construction, while this verifier proves that the exact
sealed bytes and private layout survived materialization.
"""

from __future__ import annotations

import json
import re
import stat
from pathlib import Path

from .io import digest_json, file_sha256
from .study_data import check_seal, seal

SCHEMA = "cyber_dense_sft_corpus_acceptance_v1"
EXPECTED_KEYS = {
    "manifest_sha256",
    "train_sha256",
    "source_selection_sha256",
    "study_split_sha256",
    "training_split_sha256",
    "target_policy_sha256",
    "episodes",
    "segments",
    "supervised_tokens",
    "model_repo",
    "model_revision",
    "chat_template_sha256",
    "tokenizer_backend_sha256",
    "max_length",
    "context_tokens",
    "builder_components_sha256",
    "frozen_materialization_runtime_sha256",
}


def _digest(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value) is not None


def _private_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError("corpus members must be regular non-symlink files")
    if stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise ValueError("corpus members must have exact mode 0600")


def _sealed_private_json(path: Path, schema: str | None = None) -> dict:
    _private_file(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or not _digest(value.get("sha256")):
        raise ValueError("private JSON member is not self-digested")
    if schema is None:
        if value["sha256"] != digest_json(
            {key: row for key, row in value.items() if key != "sha256"}
        ):
            raise ValueError("private JSON member digest differs")
    else:
        check_seal(value, schema)
    return value


def accept(corpus_dir: Path, expected: dict) -> dict:
    """Return a sanitized receipt for an exact private, train-only corpus."""
    if set(expected) != EXPECTED_KEYS:
        raise ValueError("corpus acceptance expectation keys differ")
    if any(
        not _digest(expected[key])
        for key in (
            "manifest_sha256",
            "train_sha256",
            "source_selection_sha256",
            "study_split_sha256",
            "training_split_sha256",
            "target_policy_sha256",
            "chat_template_sha256",
            "tokenizer_backend_sha256",
            "builder_components_sha256",
            "frozen_materialization_runtime_sha256",
        )
    ):
        raise ValueError("corpus acceptance requires exact SHA-256 bindings")
    if any(
        type(expected[key]) is not int or expected[key] <= 0
        for key in ("episodes", "segments", "supervised_tokens", "max_length", "context_tokens")
    ):
        raise ValueError("corpus acceptance requires positive exact counts")
    if (
        not isinstance(expected["model_repo"], str)
        or not expected["model_repo"]
        or not isinstance(expected["model_revision"], str)
        or re.fullmatch(r"[0-9a-f]{40}", expected["model_revision"]) is None
    ):
        raise ValueError("corpus acceptance requires exact model identity")
    if corpus_dir.is_symlink() or not corpus_dir.is_dir():
        raise ValueError("corpus root must be a regular non-symlink directory")
    if stat.S_IMODE(corpus_dir.stat().st_mode) != 0o700:
        raise ValueError("corpus root must have exact mode 0700")
    required = {
        "manifest.json",
        "source-selection.json",
        "split.json",
        "study-split.json",
        "train.parquet",
    }
    if {path.name for path in corpus_dir.iterdir()} != required:
        raise ValueError("private corpus file inventory differs")

    manifest = _sealed_private_json(corpus_dir / "manifest.json", "cyber_dense_sft_corpus_v1")
    selection = _sealed_private_json(
        corpus_dir / "source-selection.json", "cyber_sft_source_selection_v1"
    )
    split = _sealed_private_json(corpus_dir / "split.json")
    study_split = _sealed_private_json(corpus_dir / "study-split.json")
    train_path = corpus_dir / "train.parquet"
    _private_file(train_path)

    train = manifest.get("files", {}).get("train")
    tokenizer = manifest.get("tokenizer", {})
    source = manifest.get("source_selection", {})
    if (
        manifest.get("sha256") != expected["manifest_sha256"]
        or set(manifest.get("files", {})) != {"train"}
        or not isinstance(train, dict)
        or train.get("path") != "train.parquet"
        or train.get("format") != "pretokenized_assistant_segments_v1"
        or manifest.get("dev_windows") != 0
        or manifest.get("validation_mode") != "task_outcomes_only"
        or manifest.get("max_length") != expected["max_length"]
        or manifest.get("context_tokens") != expected["context_tokens"]
        or tokenizer.get("repo") != expected["model_repo"]
        or tokenizer.get("revision") != expected["model_revision"]
        or tokenizer.get("chat_template_sha256")
        != expected["chat_template_sha256"].removeprefix("sha256:")
        or tokenizer.get("backend_sha256")
        != expected["tokenizer_backend_sha256"].removeprefix("sha256:")
        or digest_json(manifest.get("builder_sha256")) != expected["builder_components_sha256"]
    ):
        raise ValueError("corpus manifest differs from the frozen interface")
    if (
        source.get("source_selection_sha256") != expected["source_selection_sha256"]
        or source.get("study_split_sha256") != expected["study_split_sha256"]
        or source.get("target_policy_sha256") != expected["target_policy_sha256"]
        or source.get("selected_episode_count") != expected["episodes"]
        or selection.get("sha256") != expected["source_selection_sha256"]
        or selection.get("study_split_sha256") != expected["study_split_sha256"]
        or selection.get("training_split_sha256") != expected["training_split_sha256"]
        or selection.get("target_policy_sha256") != expected["target_policy_sha256"]
        or selection.get("status") != "ready"
        or selection.get("blockers") != []
        or selection.get("totals", {}).get("episodes") != expected["episodes"]
        or selection.get("totals", {}).get("supervised_tokens") != expected["supervised_tokens"]
        or len(selection.get("selected_episode_ids", [])) != expected["episodes"]
        or study_split.get("sha256") != expected["study_split_sha256"]
        or split.get("sha256") != expected["training_split_sha256"]
    ):
        raise ValueError("corpus source/split bindings differ")
    if (
        train.get("source_sessions") != expected["episodes"]
        or train.get("rows") != expected["segments"]
        or train.get("supervised_tokens") != expected["supervised_tokens"]
        or train.get("sha256") != expected["train_sha256"]
        or file_sha256(train_path) != expected["train_sha256"]
    ):
        raise ValueError("opaque train payload identity or aggregate counts differ")

    return seal(
        {
            "schema": SCHEMA,
            "status": "accepted",
            "validator_code_sha256": file_sha256(Path(__file__)),
            "read_policy": "opaque_sha256_only_no_parquet_or_token_decoding",
            "layout": "private_train_only",
            "directory_mode": "0700",
            "file_mode": "0600",
            "manifest_sha256": manifest["sha256"],
            "train_sha256": train["sha256"],
            "source_selection_sha256": selection["sha256"],
            "study_split_sha256": study_split["sha256"],
            "training_split_sha256": split["sha256"],
            "target_policy_sha256": expected["target_policy_sha256"],
            "builder_components_sha256": expected["builder_components_sha256"],
            "frozen_materialization_runtime_sha256": expected[
                "frozen_materialization_runtime_sha256"
            ],
            "model": {
                "repo": expected["model_repo"],
                "revision": expected["model_revision"],
            },
            "counts": {
                "episodes": expected["episodes"],
                "segments": expected["segments"],
                "supervised_tokens": expected["supervised_tokens"],
            },
        }
    )
