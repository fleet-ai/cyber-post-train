"""Synthetic-only tests: no Fleet transcripts, tokens, or credentials."""

import json
import shutil
from collections import UserDict
from pathlib import Path
from types import SimpleNamespace

import pytest

from training import runtime
from training.runtime import TOKENIZER_FILES, _ids, load_pinned_tokenizer


def test_tokenizer_mapping_result_is_one_sequence():
    # Transformers 5 returns BatchEncoding, a Mapping that is not a dict.
    assert _ids(UserDict(input_ids=[1, 2, 3])) == [1, 2, 3]


def test_exact_qwen_tokenizer_batch_encoding(tmp_path):
    pytest.importorskip("transformers")
    source = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/"
              "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")
    if not all((source / name).is_file() for name in TOKENIZER_FILES):
        pytest.skip("pinned tokenizer is not available locally")
    for name in (*TOKENIZER_FILES, "config.json"):
        shutil.copyfile(source / name, tmp_path / name)
    tokenizer = load_pinned_tokenizer(tmp_path)
    value = tokenizer.apply_chat_template([{"role": "user", "content": "Synthetic task."}],
                                          tokenize=True, add_generation_prompt=False)
    assert len(_ids(value)) > 0


def test_native_tool_aware_loss_mask(monkeypatch):
    row = {"messages": [{"role": "user", "content": "Synthetic task"},
                        {"role": "assistant", "content": "Synthetic answer"}],
           "tools": json.dumps([{"type": "function", "function": {"name": "fleet_bash"}}])}
    native = {"input_ids": [1, 2, 3, 4], "num_actions": 2, "loss_mask": [1, 1]}
    monkeypatch.setattr(runtime, "_pinned_skyrl", lambda: SimpleNamespace(
        _normalize_chat_messages=lambda messages: messages,
        tokenize_chat_example=lambda *_args, **_kwargs: native,
    ))

    class Tokenizer:
        def apply_chat_template(self, *_args, add_generation_prompt, **_kwargs):
            return [1, 2] if add_generation_prompt else [1, 2, 3, 4]

    assert runtime.pinned_tokenize(row, Tokenizer()) == ([1, 2, 3, 4], 2)
    native["loss_mask"] = [1, 0]
    with pytest.raises(ValueError, match="native assistant mask"):
        runtime.pinned_tokenize(row, Tokenizer())
