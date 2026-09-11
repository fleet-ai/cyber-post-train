"""Rendered prompts must use native cursor semantics without vision coercion."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


@pytest.fixture
def loader(monkeypatch):
    calls = []

    class Base:
        def __init__(self, args):
            assert not args.rollout_global_dataset
            self.args = args
            self.epoch_id = 0
            self.sample_index = self.sample_offset = 0

        def save(self, rollout_id):
            return rollout_id

        def load(self, rollout_id):
            return rollout_id

    class Dataset:
        def __init__(self, path, **kwargs):
            calls.append((path, kwargs))
            self.shuffles = []

        def shuffle(self, epoch):
            self.shuffles.append(epoch)

    tokenizer = object()
    monkeypatch.setitem(sys.modules, "miles.rollout.data_source", NS(RolloutDataSource=Base))
    monkeypatch.setitem(sys.modules, "miles.utils.data", NS(Dataset=Dataset))
    monkeypatch.setitem(
        sys.modules,
        "miles.utils.processing_utils",
        NS(load_tokenizer=lambda *a, **k: tokenizer),
    )
    path = Path(__file__).resolve().parents[1] / "training/miles_text.py"
    spec = importlib.util.spec_from_file_location("synthetic_text_source", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = NS(
        rollout_global_dataset=True,
        apply_chat_template=False,
        multimodal_keys=None,
        tool_key=None,
        label_key=None,
        input_key="input",
        metadata_key="metadata",
        hf_checkpoint="/synthetic/model",
        chat_template_path="/synthetic/template",
        prompt_data="/synthetic/train.jsonl",
        rollout_max_prompt_len=1024,
        rollout_seed=42,
        rollout_shuffle=True,
    )
    return module.TextDataSource, Base, args, calls, tokenizer


@pytest.mark.parametrize("shuffle", [False, True])
def test_only_loading_changes_native_save_and_load_are_inherited(loader, shuffle):
    source, base, args, calls, tokenizer = loader
    args.rollout_shuffle = shuffle
    instance = source(args)
    assert args.rollout_global_dataset and instance.args is args
    assert instance.sample_index == instance.sample_offset == 0
    assert source.save is base.save and source.load is base.load
    assert instance.dataset.shuffles == ([0] if shuffle else [])
    assert calls == [
        (
            args.prompt_data,
            {
                "tokenizer": tokenizer,
                "processor": None,
                "max_length": 1024,
                "prompt_key": "input",
                "metadata_key": "metadata",
                "apply_chat_template": False,
                "seed": 42,
            },
        )
    ]


@pytest.mark.parametrize(
    "field,value",
    [
        ("rollout_global_dataset", False),
        ("apply_chat_template", True),
        ("multimodal_keys", {}),
        ("tool_key", "tools"),
        ("label_key", "label"),
        ("input_key", "prompt"),
        ("metadata_key", "other"),
    ],
)
def test_rejects_a_different_data_contract_before_loading(loader, field, value):
    source, _, args, calls, _ = loader
    setattr(args, field, value)
    with pytest.raises(ValueError, match="rendered text"):
        source(args)
    assert not calls
