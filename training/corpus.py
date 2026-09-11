"""Create private dense SFT data from an explicit frozen Fleet split.

This does not choose new held-out tasks, download a model, or submit a job.
Teacher and self-distillation use the same path with different source filters.
"""

from __future__ import annotations

import collections
import hashlib
import os
import re
from pathlib import Path

from cyber_post_train.jobs import digest

from .dense import Excluded, clean_dev_windows, compatible_messages, encode_record, native_helper
from .dense import segment_record as segments
from .io import atomic_write_json, file_sha256, iter_jsonl
from .sft import _known, read_mapping
from .sft_runtime import DENSE_FORMAT, dense_rows
from .splits import split_key
from .stage_sft_corpus import _eligible


def select_sources(records: list[dict], split: dict, models: list[str]) -> tuple[list, list, dict]:
    """Honor the frozen reference session and group every family across versions."""
    if split.get("schema") != "cyber_task_split_v1" or split.get("sha256") != "sha256:" + digest(
        {k: v for k, v in split.items() if k != "sha256"}
    ):
        raise ValueError("invalid frozen task split")
    tasks = {(t["task_key"], t["task_version_id"]): t for t in split["tasks"]}
    if not tasks or len(tasks) != len(split["tasks"]):
        raise ValueError("empty or duplicate split bindings")
    if any(t["split"] not in {"train", "dev", "test", "reserved_dev"} for t in tasks.values()):
        raise ValueError("unknown split assignment")
    refs = {t["reference_session_id"]: key for key, t in tasks.items() if t["split"] == "dev"}
    if not refs or len(refs) != sum(t["split"] == "dev" for t in tasks.values()) or not all(refs):
        raise ValueError("dev tasks require unique frozen reference sessions")
    if (
        not isinstance(models, list)
        or not models
        or any(not isinstance(m, str) or not m for m in models)
    ):
        raise ValueError("explicit nonempty training source model filter required")
    train, dev, seen, families, skipped = [], {}, set(), {}, collections.Counter()
    for record in records:
        sid = record.get("record_id")
        if not isinstance(sid, str) or not sid or sid in seen:
            raise ValueError("missing or duplicate source session identity")
        seen.add(sid)
        lineage = record["lineage"]
        key = (lineage["task_key"], lineage["eval_task_version_id"])
        task = tasks.get(key)
        if task is None:
            if sid in refs:
                raise ValueError("reference task version differs from frozen split")
            skipped["outside_exact_split"] += 1
            continue
        assigned = task["split"]
        family = split_key(record)
        if families.setdefault(family, assigned) != assigned:
            raise ValueError("one task family crosses frozen splits")
        if sid in refs:
            if refs[sid] != key or not _eligible(record):
                raise ValueError("frozen dev reference is not the expected verified success")
            dev[sid] = record
        elif assigned != "train":
            skipped["heldout_or_reserved"] += 1
        elif record["source"]["model"] not in models:
            skipped["other_source_model"] += 1
        elif not _eligible(record):
            skipped["not_verified_success"] += 1
        else:
            train.append(record)
    if set(dev) != set(refs) or not train:
        raise ValueError("missing held-out reference or empty eligible training set")
    return sorted(train, key=lambda r: r["record_id"]), [dev[k] for k in sorted(dev)], dict(skipped)


def local_tokenizer(lock: dict, root: Path):
    """Load only the exact local tokenizer bytes; no network or remote code."""
    from transformers import AutoTokenizer

    if not re.fullmatch(r"[0-9a-f]{40}", lock["revision"]):
        raise ValueError("tokenizer requires exact revision")
    entries = lock["tokenizer"]["files"]
    if not entries or len({entry["path"] for entry in entries}) != len(entries):
        raise ValueError("tokenizer inventory is missing or duplicated")
    for entry in lock["tokenizer"]["files"]:
        relative = Path(entry["path"])
        if not relative.parts or relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe tokenizer inventory path")
        path = root / relative
        if path.is_symlink() or file_sha256(path) != "sha256:" + entry["sha256"].removeprefix(
            "sha256:"
        ):
            raise ValueError("local tokenizer differs from model lock")
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    identity = {
        "repo": lock["repo"],
        "revision": lock["revision"],
        "files": lock["tokenizer"]["files"],
        "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        "backend_sha256": hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
    }
    return tokenizer, identity


def build(config: dict, *, relative_to: Path) -> dict:
    """Produce create-once Parquet/manifest files; return only aggregate metadata."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    _known(
        config,
        {
            "source",
            "split",
            "model_lock",
            "tokenizer_root",
            "native_helper",
            "train_models",
            "max_length",
            "context_tokens",
            "dev_windows",
            "output",
        },
        "data",
    )
    _known(config["source"], {"path", "sha256"}, "source")
    source = relative_to / config["source"]["path"]
    output = relative_to / config["output"]
    if output.exists():
        raise FileExistsError("create-once corpus destination exists")
    source_sha = file_sha256(source)
    if source_sha != "sha256:" + config["source"]["sha256"].removeprefix("sha256:"):
        raise ValueError("source digest mismatch")
    split = read_mapping(relative_to / config["split"])
    train, dev, split_exclusions = select_sources(
        list(iter_jsonl(source)), split, config["train_models"]
    )
    tokenizer, identity = local_tokenizer(
        read_mapping(relative_to / config["model_lock"]), relative_to / config["tokenizer_root"]
    )
    helper = native_helper(relative_to / config["native_helper"])
    maximum, context, targets = (
        config.get("max_length", 16384),
        config.get("context_tokens", 4096),
        config.get("dev_windows", 5),
    )
    if (
        any(type(v) is not int for v in (maximum, context, targets))
        or maximum < 2
        or context < 0
        or targets < 1
    ):
        raise ValueError("invalid data window bounds")
    rows, included, exclusions = [], [], collections.Counter()
    for record in train:
        try:
            messages, _ = compatible_messages(record)
            anchor, chunks = encode_record(messages, tokenizer, helper)
            packed = segments(record, anchor, chunks, max_tokens=maximum, context_budget=context)
        except Excluded as exc:
            exclusions[exc.reason] += 1
            continue
        rows.extend(packed)
        included.append(packed[0])
    # A bad dev reference is a blocker, not permission to substitute another task.
    dev_rows = [
        w for r in dev for w in clean_dev_windows(r, tokenizer, max_tokens=maximum, targets=targets)
    ]
    if not rows:
        raise ValueError("no compatible training sources")
    counts = {
        "rows": len(rows),
        "task_keys": sorted({r["task_key"] for r in rows}),
        "format": DENSE_FORMAT,
        "source_sessions": len(included),
        "supervised_tokens": sum(r["target_token_count"] for r in rows),
        "assistant_responses": sum(len(r["target_spans"]) for r in rows),
        "source_total_assistant_responses": sum(r["source_assistant_count"] for r in included),
        "excluded_assistant_responses": sum(len(r["excluded_assistant_targets"]) for r in included),
    }
    dense_rows(rows, counts, max_length=maximum, vocab_size=len(tokenizer))
    if set(counts["task_keys"]) & {r["task_key"] for r in dev_rows}:
        raise ValueError("train/dev task key overlap")
    if file_sha256(source) != source_sha:
        raise ValueError("source changed during preparation")
    output.mkdir(parents=True, mode=0o700)
    files = {}
    for name, values, metadata in (
        ("train", rows, counts),
        (
            "dev",
            dev_rows,
            {
                "rows": len(dev_rows),
                "task_keys": sorted({r["task_key"] for r in dev_rows}),
                "format": "chat_messages_last_assistant_v2",
            },
        ),
    ):
        path = output / f"{name}.parquet"
        pq.write_table(pa.Table.from_pylist(values), path, compression="zstd")
        os.chmod(path, 0o600)
        if pq.read_table(path).to_pylist() != values:
            raise ValueError("Parquet readback differs")
        files[name] = {"path": path.name, "sha256": file_sha256(path), **metadata}
    manifest = {
        "schema": "cyber_dense_sft_corpus_v1",
        "source_sha256": source_sha,
        "split_sha256": split["sha256"],
        "tokenizer": identity,
        "files": files,
        "train_models": config["train_models"],
        "max_length": maximum,
        "context_tokens": context,
        "dev_windows": targets,
        "split_exclusions": split_exclusions,
        "whole_source_exclusions": dict(exclusions),
        "builder_sha256": {
            name: file_sha256(Path(__file__).with_name(name))
            for name in ("corpus.py", "dense.py", "stage_sft_corpus.py", "splits.py")
        },
        "limitations": [
            "Visible actions only; private reasoning is omitted.",
            "Complete recent rounds do not guarantee semantically sufficient context.",
            "Family-held-out tasks can share applications.",
        ],
    }
    manifest["sha256"] = "sha256:" + digest(manifest)
    atomic_write_json(output / "manifest.json", manifest, private=True)
    atomic_write_json(output / "split.json", split, private=True)
    return {
        "output": str(output),
        "manifest_sha256": manifest["sha256"],
        "train": {k: counts[k] for k in ("rows", "source_sessions", "supervised_tokens")},
        "dev": {"rows": len(dev_rows), "tasks": len(dev)},
        "whole_source_exclusions": dict(exclusions),
    }
