import json
from pathlib import Path

import pytest

from training.tokenizer_equivalence import (
    compare_corpora,
    compare_special_token_ids,
    compare_tokenizer_mappings,
    iter_corpus_strings,
)


def _tokenizer(path: Path, additions: list[dict] | None = None) -> None:
    document = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": 3,
                "content": "<special>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
            *(additions or []),
        ],
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "WordLevel",
            "vocab": {"hello": 0, "world": 1, "<unk>": 2},
            "unk_token": "<unk>",
        },
    }
    path.write_text(json.dumps(document))


def _addition(content: str, token_id: int) -> dict:
    return {
        "id": token_id,
        "content": content,
        "single_word": False,
        "lstrip": False,
        "rstrip": False,
        "normalized": False,
        "special": True,
    }


def test_mapping_requires_declared_append_only_tokens(tmp_path: Path):
    base = tmp_path / "base.json"
    candidate = tmp_path / "candidate.json"
    _tokenizer(base)
    _tokenizer(candidate, [_addition("<new>", 4)])
    receipt = compare_tokenizer_mappings(base, candidate, expected_append_only_tokens={"<new>": 4})
    assert receipt["candidate_additions_append_only"] is True
    assert "content" not in receipt["candidate_only_tokens"][0]

    changed = json.loads(candidate.read_text())
    changed["model"]["vocab"]["hello"] = 1
    candidate.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="remaps"):
        compare_tokenizer_mappings(base, candidate, expected_append_only_tokens={"<new>": 4})


def test_mapping_rejects_undeclared_or_non_contiguous_additions(tmp_path: Path):
    base = tmp_path / "base.json"
    candidate = tmp_path / "candidate.json"
    _tokenizer(base)
    _tokenizer(candidate, [_addition("<new>", 5)])
    with pytest.raises(ValueError, match="append-only contiguous"):
        compare_tokenizer_mappings(base, candidate, expected_append_only_tokens={"<new>": 5})
    with pytest.raises(ValueError, match="declared append-only set"):
        compare_tokenizer_mappings(base, candidate, expected_append_only_tokens={"<other>": 5})


def test_special_token_fields_must_match(tmp_path: Path):
    base = tmp_path / "base-config.json"
    candidate = tmp_path / "candidate-config.json"
    base.write_text(json.dumps({"eos_token": "<special>", "pad_token_id": 2}))
    candidate.write_text(json.dumps({"eos_token": "<special>", "pad_token_id": 2}))
    assert compare_special_token_ids(base, candidate)["common_special_token_fields_identical"]
    candidate.write_text(json.dumps({"eos_token": "changed", "pad_token_id": 2}))
    with pytest.raises(ValueError, match="special-token"):
        compare_special_token_ids(base, candidate)


def test_corpus_parity_receipt_omits_text_and_fails_closed(tmp_path: Path):
    base = tmp_path / "base.json"
    candidate = tmp_path / "candidate.json"
    _tokenizer(base)
    _tokenizer(candidate)
    receipt = compare_corpora(base, candidate, [("test", ["hello world", "hello"])])
    assert receipt["total_string_count"] == 2
    assert "hello" not in json.dumps(receipt)

    changed = json.loads(candidate.read_text())
    changed["pre_tokenizer"] = {"type": "WhitespaceSplit"}
    candidate.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="tokenization differs"):
        compare_corpora(base, candidate, [("test", ["hello,world"])])


def test_iter_corpus_strings_is_deterministic_and_recursive(tmp_path: Path):
    source = tmp_path / "corpus.jsonl"
    source.write_text('{"z":"last","a":["first",{"x":"middle"}]}\n')
    assert list(iter_corpus_strings(source)) == ["first", "middle", "last"]
