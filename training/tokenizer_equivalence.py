"""Fail-closed tokenizer equivalence checks for a composed inference bundle.

Receipts deliberately contain only hashes and counts. Benchmark and Fleet prompt
contents must remain outside the repository and outside durable receipts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from .io import digest_json

EXPECTED_APPEND_ONLY_TOKENS = {
    "<|audio_start|>": 248070,
    "<|audio_end|>": 248071,
    "<tts_pad>": 248072,
    "<tts_text_bos>": 248073,
    "<tts_text_eod>": 248074,
    "<tts_text_bos_single>": 248075,
    "<|audio_pad|>": 248076,
}


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _added_token_map(document: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = document.get("added_tokens")
    if not isinstance(rows, list):
        raise ValueError("tokenizer.json has no added_tokens list")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("content"), str):
            raise ValueError("tokenizer.json contains an invalid added token")
        content = row["content"]
        if content in result:
            raise ValueError(f"duplicate added token {content!r}")
        result[content] = row
    return result


def compare_tokenizer_mappings(
    base_tokenizer_json: Path,
    candidate_tokenizer_json: Path,
    *,
    expected_append_only_tokens: Mapping[str, int] = EXPECTED_APPEND_ONLY_TOKENS,
    base_tokenizer_config: Path | None = None,
) -> dict[str, Any]:
    """Prove that candidate adds only declared IDs and remaps no base token."""

    base_document = _json(base_tokenizer_json)
    candidate_document = _json(candidate_tokenizer_json)
    base_model_vocab = base_document.get("model", {}).get("vocab")
    candidate_model_vocab = candidate_document.get("model", {}).get("vocab")
    if not isinstance(base_model_vocab, dict) or not isinstance(candidate_model_vocab, dict):
        raise ValueError("tokenizer model vocab is missing")
    if base_model_vocab != candidate_model_vocab:
        raise ValueError("candidate tokenizer remaps the base model vocabulary")

    base_added = _added_token_map(base_document)
    candidate_added = _added_token_map(candidate_document)
    for content, base_row in base_added.items():
        if candidate_added.get(content) != base_row:
            raise ValueError(f"candidate tokenizer remaps base added token {content!r}")
    candidate_only = {
        content: row for content, row in candidate_added.items() if content not in base_added
    }
    observed_additions = {content: row.get("id") for content, row in candidate_only.items()}
    if observed_additions != dict(expected_append_only_tokens):
        raise ValueError("candidate tokenizer additions differ from the declared append-only set")
    if not all(row.get("special") is True for row in candidate_only.values()):
        raise ValueError("candidate tokenizer additions are not all special tokens")

    base_ids = set(base_model_vocab.values()) | {row.get("id") for row in base_added.values()}
    if not base_ids or not all(isinstance(value, int) for value in base_ids):
        raise ValueError("base tokenizer contains invalid token IDs")
    added_ids = sorted(observed_additions.values())
    expected_ids = list(range(max(base_ids) + 1, max(base_ids) + 1 + len(added_ids)))
    if added_ids != expected_ids or base_ids.intersection(added_ids):
        raise ValueError("candidate tokenizer additions are not append-only contiguous IDs")

    additions_already_declared_by_base = None
    if base_tokenizer_config is not None:
        decoder = _json(base_tokenizer_config).get("added_tokens_decoder")
        if not isinstance(decoder, dict):
            raise ValueError("base tokenizer config has no added_tokens_decoder")
        declared = {
            str(row.get("content")): int(token_id)
            for token_id, row in decoder.items()
            if isinstance(row, dict) and str(row.get("content")) in observed_additions
        }
        if declared != observed_additions:
            raise ValueError(
                "candidate tokenizer JSON additions differ from the base effective tokenizer"
            )
        additions_already_declared_by_base = True

    return {
        "schema": "cyber_sft_tokenizer_mapping_equivalence_v1",
        "base_model_vocab_size": len(base_model_vocab),
        "base_added_token_count": len(base_added),
        "candidate_added_token_count": len(candidate_added),
        "common_token_to_id_mapping_identical": True,
        "common_added_token_definitions_identical": True,
        "candidate_only_tokens": [
            {"content_sha256": _sha256_text(content), "id": token_id}
            for content, token_id in sorted(observed_additions.items(), key=lambda item: item[1])
        ],
        "candidate_additions_append_only": True,
        "additions_already_declared_by_base_config": additions_already_declared_by_base,
        "mapping_sha256": digest_json(
            {
                "model_vocab": base_model_vocab,
                "base_added_tokens": base_added,
                "candidate_only": candidate_only,
            }
        ),
    }


def compare_effective_tokenizers(base_root: Path, candidate_root: Path) -> dict[str, Any]:
    """Prove the Transformers-loaded runtime token-to-ID mapping is exactly identical."""

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - train image supplies transformers
        raise ValueError("transformers is required for effective tokenizer equivalence") from exc
    base = AutoTokenizer.from_pretrained(base_root, local_files_only=True)
    candidate = AutoTokenizer.from_pretrained(candidate_root, local_files_only=True)
    base_vocab = base.get_vocab()
    candidate_vocab = candidate.get_vocab()
    if base_vocab != candidate_vocab:
        raise ValueError("effective runtime token-to-ID mappings differ")
    special_ids = {}
    for field in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        base_value = getattr(base, field)
        candidate_value = getattr(candidate, field)
        if base_value != candidate_value:
            raise ValueError(f"effective runtime tokenizer changes {field}")
        special_ids[field] = base_value
    return {
        "schema": "cyber_sft_effective_tokenizer_equivalence_v1",
        "token_count": len(base_vocab),
        "token_to_id_mapping_identical": True,
        "special_token_ids_identical": True,
        "special_token_ids": special_ids,
        "mapping_sha256": digest_json(base_vocab),
    }


def compare_special_token_ids(base_config: Path, candidate_config: Path) -> dict[str, Any]:
    """Compare every shared named special token without relying on a short field list."""

    base = _json(base_config)
    candidate = _json(candidate_config)
    if not isinstance(base, dict) or not isinstance(candidate, dict):
        raise ValueError("tokenizer config must be a JSON object")
    base_fields = {
        key: value
        for key, value in base.items()
        if (key.endswith("_token") or key.endswith("_token_id")) and not key.startswith("add_")
    }
    candidate_fields = {
        key: value
        for key, value in candidate.items()
        if (key.endswith("_token") or key.endswith("_token_id")) and not key.startswith("add_")
    }
    common_fields = sorted(set(base_fields) & set(candidate_fields))
    missing = sorted(set(base_fields) - set(candidate_fields))
    mismatched = [
        field for field in common_fields if base_fields[field] != candidate_fields[field]
    ]
    if missing:
        raise ValueError(f"candidate tokenizer removes special-token fields: {missing}")
    if mismatched:
        raise ValueError(f"candidate tokenizer changes special-token fields: {mismatched}")
    candidate_only = {
        field: candidate_fields[field] for field in sorted(set(candidate_fields) - set(base_fields))
    }
    return {
        "schema": "cyber_sft_special_token_equivalence_v1",
        "common_fields_compared": common_fields,
        "common_special_token_fields_identical": True,
        "common_fields_sha256": digest_json(
            {field: base_fields[field] for field in common_fields}
        ),
        "candidate_only_named_fields": [
            {"field": field, "value_sha256": _sha256_text(str(value))}
            for field, value in candidate_only.items()
        ],
    }


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key in sorted(value, key=str):
            yield from _strings(value[key])
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for item in value:
            yield from _strings(item)


def iter_corpus_strings(path: Path) -> Iterator[str]:
    """Yield strings deterministically from JSON, JSONL, Parquet, or plain text."""

    suffix = path.suffix.lower()
    if suffix == ".json":
        yield from _strings(_json(path))
    elif suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    try:
                        yield from _strings(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    elif suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - train image supplies pyarrow
            raise ValueError("pyarrow is required to inspect a Parquet corpus") from exc
        table = pq.read_table(path)
        for row in table.to_pylist():
            yield from _strings(row)
    else:
        yield path.read_text(encoding="utf-8")


def compare_corpora(
    base_tokenizer: Path,
    candidate_tokenizer: Path,
    corpora: Iterable[tuple[str, Iterable[str]]],
) -> dict[str, Any]:
    """Require exact encode and decode parity for all supplied corpus strings."""

    if base_tokenizer.is_dir() and candidate_tokenizer.is_dir():
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - train image supplies transformers
            raise ValueError("transformers is required for corpus equivalence") from exc
        base = AutoTokenizer.from_pretrained(base_tokenizer, local_files_only=True)
        candidate = AutoTokenizer.from_pretrained(candidate_tokenizer, local_files_only=True)

        def encode(tokenizer: Any, value: str) -> list[int]:
            return list(tokenizer.encode(value, add_special_tokens=False))

        def decode(tokenizer: Any, ids: list[int]) -> str:
            return tokenizer.decode(ids, skip_special_tokens=False)

    elif base_tokenizer.is_file() and candidate_tokenizer.is_file():
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - train image supplies tokenizers
            raise ValueError("tokenizers is required for corpus equivalence") from exc
        base = Tokenizer.from_file(str(base_tokenizer))
        candidate = Tokenizer.from_file(str(candidate_tokenizer))

        def encode(tokenizer: Any, value: str) -> list[int]:
            return tokenizer.encode(value, add_special_tokens=False).ids

        def decode(tokenizer: Any, ids: list[int]) -> str:
            return tokenizer.decode(ids, skip_special_tokens=False)

    else:
        raise ValueError("base and candidate tokenizers must both be files or both be directories")
    corpus_receipts = []
    total_strings = 0
    total_tokens = 0
    for label, strings in corpora:
        input_rows = []
        encoding_rows = []
        count = 0
        tokens = 0
        for ordinal, value in enumerate(strings):
            if not isinstance(value, str):
                raise ValueError(f"corpus {label!r} contains a non-string value")
            base_ids = encode(base, value)
            candidate_ids = encode(candidate, value)
            value_sha = _sha256_text(value)
            if base_ids != candidate_ids:
                raise ValueError(
                    f"tokenization differs in corpus {label!r} ordinal {ordinal} "
                    f"input {value_sha}"
                )
            base_decoded = decode(base, base_ids)
            candidate_decoded = decode(candidate, candidate_ids)
            if base_decoded != candidate_decoded:
                raise ValueError(
                    f"decode differs in corpus {label!r} ordinal {ordinal} input {value_sha}"
                )
            input_rows.append(
                {"ordinal": ordinal, "utf8_bytes": len(value.encode()), "sha256": value_sha}
            )
            encoding_rows.append({"ordinal": ordinal, "ids": base_ids})
            count += 1
            tokens += len(base_ids)
        if count == 0:
            raise ValueError(f"corpus {label!r} contains no strings")
        corpus_receipts.append(
            {
                "label": label,
                "string_count": count,
                "token_count": tokens,
                "input_manifest_sha256": digest_json(input_rows),
                "encoding_manifest_sha256": digest_json(encoding_rows),
                "encode_parity": True,
                "decode_parity": True,
            }
        )
        total_strings += count
        total_tokens += tokens
    return {
        "schema": "cyber_sft_tokenizer_corpus_equivalence_v1",
        "corpora": corpus_receipts,
        "corpus_count": len(corpus_receipts),
        "total_string_count": total_strings,
        "total_token_count": total_tokens,
        "all_encode_parity": True,
        "all_decode_parity": True,
        "receipt_sha256": digest_json(corpus_receipts),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--candidate-config", type=Path, required=True)
    parser.add_argument("--corpus", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    corpus_specs = []
    for item in args.corpus:
        label, separator, raw_path = item.partition("=")
        if not separator or not label or not raw_path:
            raise ValueError("--corpus must use LABEL=PATH")
        corpus_specs.append((label, iter_corpus_strings(Path(raw_path))))
    result = {
        "schema": "cyber_sft_tokenizer_equivalence_v1",
        "mapping": compare_tokenizer_mappings(
            args.base / "tokenizer.json" if args.base.is_dir() else args.base,
            args.candidate / "tokenizer.json" if args.candidate.is_dir() else args.candidate,
            base_tokenizer_config=args.base_config,
        ),
        "effective_mapping": compare_effective_tokenizers(args.base, args.candidate)
        if args.base.is_dir() and args.candidate.is_dir()
        else None,
        "special_tokens": compare_special_token_ids(args.base_config, args.candidate_config),
        "corpus": compare_corpora(args.base, args.candidate, corpus_specs),
    }
    result["equivalence_receipt_sha256"] = digest_json(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
