"""Convert reviewed message windows to the pinned SkyRL last-assistant format.

Run this with the exact trainer image and local model tokenizer, off GPU. The
source JSONL contains private task text; only the aggregate manifest is public.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import copy
import sys
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Callable

from .corpus import sha256

FORMAT = "chat_messages_last_assistant_v2"
MODEL = ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")
TOKENIZER_SHA = "sha256:3938a9a8172f2738fed1be44efc11e2562059269d50d3721213f44802b53b4e1"
TOKENIZER_FILES = {
    "tokenizer.json": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
    "tokenizer_config.json": "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27",
    "chat_template.jinja": "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
    "merges.txt": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
    "vocab.json": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
}
BACKEND_SHA = "ffb7a28b27dabcc333662fd3e0b0005d9e79a1c22e31453ab5a3017fbd5f25c0"
SKYRL_FILES = {
    "sft_trainer.py": "a5ef8a2e22de785b6760abffdd9353f1246a5898983b4d27b4aadd8089a3579a",
    "generators/utils.py": "55c15b660067749febda00d4fb1c2110ff436717bbd4b73bf66055a73d0b87d5",
}


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _ids(value: object) -> list[int]:
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        raise ValueError("tokenizer did not return one integer sequence")
    return value


def load_pinned_tokenizer(root: Path):
    """Reopen only the frozen local Qwen tokenizer; never fetch or run remote code."""
    from transformers import AutoTokenizer

    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("frozen tokenizer root is missing or linked")
    for relative, digest in TOKENIZER_FILES.items():
        path = root / relative
        if path.is_symlink() or not path.is_file() or _file_sha(path) != "sha256:" + digest:
            raise ValueError("frozen tokenizer sidecar differs")
    tokenizer = AutoTokenizer.from_pretrained(
        root, local_files_only=True, trust_remote_code=False, use_fast=True,
    )
    if (hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
            != TOKENIZER_FILES["chat_template.jinja"]
            or hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()
            != BACKEND_SHA):
        raise ValueError("loaded tokenizer identity differs")
    return tokenizer


@cache
def _pinned_skyrl():
    from skyrl.train import sft_trainer

    root = Path(sft_trainer.__file__).resolve().parent
    for relative, expected in SKYRL_FILES.items():
        source = root / relative
        if _file_sha(source) != "sha256:" + expected:
            raise ValueError("SkyRL tokenization source differs from qualified image")
    return sft_trainer


def pinned_tokenize(row: dict, tokenizer) -> tuple[list[int], int]:
    """Use only the image-proven native mask, including each row's tool schema."""
    sft_trainer = _pinned_skyrl()
    tools = row["tools"]
    if isinstance(tools, str):
        tools = json.loads(tools)
    messages = sft_trainer._normalize_chat_messages(row["messages"])
    prompt = _ids(tokenizer.apply_chat_template(
        messages[:-1], tools=tools, add_generation_prompt=True, tokenize=True,
        return_dict=False,
    ))
    full = _ids(tokenizer.apply_chat_template(
        messages, tools=tools, add_generation_prompt=False, tokenize=True,
        return_dict=False,
    ))
    native = sft_trainer.tokenize_chat_example(row, tokenizer, max_length=None)
    # Qwen's chat template merges the prompt's last newline with the opening
    # no-thinking separator in the full response. The pinned native loss mask
    # deliberately begins after that one token; accept no wider mismatch.
    prefix_ok = bool(prompt) and (
        full[:len(prompt)] == prompt or (
            full[:len(prompt) - 1] == prompt[:-1]
            and tokenizer.decode(full[:len(prompt) + 1]).startswith(tokenizer.decode(prompt))
        )
    )
    if (native is None or not prefix_ok
            or native["input_ids"] != full
            or native["num_actions"] != len(full) - len(prompt)
            or native["loss_mask"] != [1] * native["num_actions"]):
        raise ValueError("native assistant mask differs from full tool-aware render")
    return full, native["num_actions"]


def pinned_count_tokens(messages: list[dict], tools: list[dict], target_index: int,
                        tokenizer) -> tuple[int, int]:
    """Supply this callback to ``corpus.build_corpus`` with the exact tokenizer."""
    if target_index != len(messages) - 1:
        raise ValueError("only last-assistant targets are supported")
    ids, actions = pinned_tokenize({"messages": messages, "tools": tools}, tokenizer)
    return len(ids), actions


def _source_partition(source: Path, split: str, receipt: dict) -> list[dict]:
    item = receipt["partitions"][split]
    if item["path"] != f"{split}.jsonl":
        raise ValueError("unexpected private partition path")
    path = source / item["path"]
    if path.is_symlink() or path.stat().st_size != item["bytes"] or _file_sha(path) != item["sha256"]:
        raise ValueError("private partition bytes differ from receipt")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    proof = item["receipt"]
    if (sha256({k: v for k, v in proof.items() if k != "sha256"}) != proof["sha256"]
            or proof["split"] != split or proof["rows"] != len(rows)
            or proof["rows_sha256"] != sha256(rows)):
        raise ValueError("private partition selection proof differs")
    return rows


def materialize_parquet(source: Path, destination: Path, *, tokenizer_root: Path,
                        _test_tokenize: Callable[[dict], tuple[list[int], int]] | None = None,
                        _test_tokenizer_sha256: str | None = None) -> dict:
    """Create train/dev Parquet and a digest-valid manifest, receipt last.

    Production always loads the exact local tokenizer and image-pinned SkyRL.
    The private test seam deliberately cannot produce a trainer-ready manifest.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    production = _test_tokenize is None
    tokenizer_sha256 = TOKENIZER_SHA if production else _test_tokenizer_sha256
    if production:
        tokenizer = load_pinned_tokenizer(tokenizer_root)
        tokenize = lambda row: pinned_tokenize(row, tokenizer)
    else:
        tokenize = _test_tokenize
    if not isinstance(tokenizer_sha256, str):
        raise ValueError("tokenizer identity is missing")
    source, destination = Path(source), Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("trainer-ready destination already exists")
    receipt_path = source / "RECEIPT.json"
    if receipt_path.is_symlink():
        raise ValueError("source receipt is a symlink")
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("format") != "structured_message_windows_v1"
            or receipt.get("trainer_ready") is not False
            or sha256({k: v for k, v in receipt.items() if k != "sha256"}) != receipt.get("sha256")):
        raise ValueError("source receipt is not the reviewed structured corpus")
    inputs = [receipt["partitions"][s]["receipt"]["inputs_sha256"] for s in ("train", "dev")]
    if inputs[0] != inputs[1] or any(
        receipt["partitions"][s]["receipt"]["tokenizer_sha256"] != tokenizer_sha256
        for s in ("train", "dev")
    ):
        raise ValueError("source splits or tokenizer differ")
    prepared: dict[str, list[dict]] = {}
    families: dict[str, set[str]] = {}
    all_targets: set[str] = set()
    for split in ("train", "dev"):
        values = []
        families[split] = set()
        for row in _source_partition(source, split, receipt):
            if (row.get("target_message_index") != len(row["messages"]) - 1
                    or row["messages"][-1].get("role") != "assistant"
                    or sha256(row["tools"]) != inputs[0]["tools"]
                    or row["target_id"] in all_targets):
                raise ValueError("window target or tool contract differs")
            all_targets.add(row["target_id"])
            families[split].add(row["family_id"])
            value = {**copy.deepcopy(row), "task_key": row["task_version_id"],
                     "window_id": row["target_id"], "token_count": row["total_tokens"],
                     "tools": json.dumps(row["tools"], sort_keys=True, separators=(",", ":"))}
            for message in value["messages"]:
                if "tool_calls" in message:
                    message["tool_calls"] = json.dumps(
                        message["tool_calls"], sort_keys=True, separators=(",", ":"))
            try:
                before, actions = tokenize(row)
                after, after_actions = tokenize(value)
            except Exception:
                raise ValueError("qualified tokenizer rejected a private window") from None
            if (before != after or actions != after_actions or len(after) != row["total_tokens"]
                    or actions != row["supervised_tokens"]
                    or not 0 < actions < len(after)):
                raise ValueError("private window tokenization changed")
            values.append(value)
        if not values:
            raise ValueError("empty trainer split")
        prepared[split] = values
    if families["train"] & families["dev"]:
        raise ValueError("train/dev family overlap")
    destination.mkdir(mode=0o700)
    files = {}
    for split, values in prepared.items():
        path = destination / f"{split}.parquet"
        pq.write_table(pa.Table.from_pylist(values), path, compression="zstd")
        os.chmod(path, 0o600)
        reopened = pq.read_table(path).to_pylist()
        if len(reopened) != len(values):
            raise ValueError("Parquet row count changed on readback")
        for original, decoded in zip(values, reopened, strict=True):
            try:
                a, n = tokenize(original)
                b, m = tokenize(decoded)
            except Exception:
                raise ValueError("qualified tokenizer rejected a Parquet row") from None
            if a != b or n != m or decoded["task_key"] != original["task_key"]:
                raise ValueError("Parquet readback changes model-visible bytes")
        files[split] = {"path": path.name, "sha256": _file_sha(path), "rows": len(values),
                        "task_keys": sorted({r["task_key"] for r in values}), "format": FORMAT}
    manifest = {"schema": "qwen38_tool_aware_parquet_v1", "trainer_ready": production,
                "validation_mode": "teacher_cross_entropy", "source_receipt_sha256": receipt["sha256"],
                "split_sha256": inputs[0]["roster"],
                "tokenizer": {"repo": MODEL[0], "revision": MODEL[1], "sha256": tokenizer_sha256},
                "files": files}
    manifest["sha256"] = sha256(manifest)
    path = destination / "manifest.json"
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as output:
        json.dump(manifest, output, sort_keys=True, separators=(",", ":"))
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--tokenizer-root", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        manifest = materialize_parquet(
            args.source, args.destination, tokenizer_root=args.tokenizer_root,
        )
    except Exception as error:
        # Parser, tokenizer and Arrow errors may quote private messages or
        # schema values. Only the error class may leave the preparation host.
        print(f"corpus conversion rejected: {type(error).__name__}", file=sys.stderr)
        return 2
    print(json.dumps({"manifest_sha256": manifest["sha256"],
                      "rows": {k: v["rows"] for k, v in manifest["files"].items()},
                      "status": "trainer_ready"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
