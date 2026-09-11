"""Rendered prompts must use native cursor semantics without vision coercion."""

import copy
import importlib.util
import pickle
import random
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


@pytest.fixture(params=["local", "native"])
def loader(monkeypatch, tmp_path, request):
    calls = []

    class Base:
        def __init__(self, args):
            assert not args.rollout_global_dataset
            self.args = args
            self.epoch_id = 0
            self.sample_index = self.sample_offset = self.sample_group_index = 0
            self.metadata = {}

        def save(self, rollout_id):
            torch = pytest.importorskip("torch")
            path = Path(self.args.save) / f"rollout/global_dataset_state_dict_{rollout_id}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    k: getattr(self, k)
                    for k in (
                        "sample_offset",
                        "epoch_id",
                        "sample_group_index",
                        "sample_index",
                        "metadata",
                    )
                },
                path,
            )

        def load(self, rollout_id):
            torch = pytest.importorskip("torch")
            path = Path(self.args.load) / f"rollout/global_dataset_state_dict_{rollout_id}.pt"
            # Match the native silent fallback being guarded, not a safer double.
            if not path.exists():
                return
            values = torch.load(path)
            for key in ("sample_offset", "epoch_id", "sample_group_index", "sample_index"):
                setattr(self, key, values.get(key, 0))
            self.metadata = values.get("metadata", {})
            if self.args.rollout_shuffle:
                self.dataset.shuffle(self.epoch_id)

    class Dataset:
        def __init__(self, path, **kwargs):
            calls.append((path, kwargs))
            self.shuffles = []
            self.seed = kwargs["seed"]
            self.origin_samples = [NS(prompt=f"synthetic {i}") for i in range(3)]
            self.samples = self.origin_samples

        def shuffle(self, epoch):
            self.shuffles.append(epoch)
            indices = list(range(len(self)))
            random.Random(self.seed + epoch).shuffle(indices)
            self.samples = [self.origin_samples[i] for i in indices]

        def __len__(self):
            return len(self.origin_samples)

    tokenizer = object()
    if request.param == "native":
        Base = pytest.importorskip("miles.rollout.data_source").RolloutDataSource
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
        rollout_batch_size=2,
        n_samples_per_prompt=2,
        save=str(tmp_path / "saved"),
        load=str(tmp_path / "saved"),
        start_rollout_id=0,
        finetune=True,
        no_load_optim=True,
        no_load_rng=True,
    )
    return module.TextDataSource, Base, args, calls, tokenizer


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("tool_key", [None, "tools"])
def test_text_loader_keeps_native_cursor_owner(loader, shuffle, tool_key):
    source, base, args, calls, tokenizer = loader
    args.rollout_shuffle = shuffle
    args.tool_key = tool_key
    instance = source(args)
    assert args.rollout_global_dataset and instance.args is args
    assert instance.sample_index == instance.sample_offset == 0
    assert issubclass(source, base)
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
        ("tool_key", "other"),
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


def set_cursor(instance, rollout_id):
    expected = instance._expected_cursor(rollout_id)
    for key, value in expected.items():
        setattr(instance, key, copy.deepcopy(value))
    instance.dataset.shuffle(instance.epoch_id)
    return expected


def recovery_args(args):
    args = copy.copy(args)
    args.finetune = args.no_load_optim = args.no_load_rng = False
    args.start_rollout_id = None
    return args


@pytest.mark.parametrize("rid,groups", [(0, 1), (1, 2), (2, 1), (2, 3), (9, 2)])
def test_native_cursor_roundtrip_covers_cross_epoch_and_exact_end(loader, rid, groups):
    pytest.importorskip("torch")
    source, _, args, _, _ = loader
    args.rollout_batch_size = groups
    original = source(args)
    if hasattr(original, "get_samples"):
        # Advance the real native cursor rather than seeding it with the same
        # arithmetic as our guard: this independently checks epoch/end semantics.
        for _ in range(rid + 1):
            original.get_samples(groups)
        expected = {
            k: copy.deepcopy(getattr(original, k))
            for k in ("sample_offset", "epoch_id", "sample_group_index", "sample_index", "metadata")
        }
    else:
        expected = set_cursor(original, rid)
    original.save(rid)
    restored = source(recovery_args(args))
    restored.load(rid)
    assert {k: getattr(restored, k) for k in expected} == expected
    assert restored.dataset.samples == original.dataset.samples
    if hasattr(original, "get_samples"):
        # Real Miles, not the local stand-in: the next prompt and attempt IDs
        # must equal uninterrupted execution, including after a shuffle.
        assert original.get_samples(groups) == restored.get_samples(groups)


def test_only_fresh_base_can_omit_cursor(loader):
    source, _, args, _, _ = loader
    instance = source(args)
    instance.load(-1)
    assert instance.sample_index == 0
    with pytest.raises(ValueError, match="initial base"):
        source(recovery_args(args)).load(-1)
    instance.sample_index = 2
    with pytest.raises(ValueError, match="completed rollout"):
        instance.load(-1)


@pytest.mark.parametrize("rid", [None, True, -2, 0.0])
def test_invalid_checkpoint_index_never_falls_back_to_zero(loader, rid):
    source, _, args, _, _ = loader
    with pytest.raises(ValueError, match="dimensions"):
        source(recovery_args(args)).load(rid)


@pytest.mark.parametrize("key", ["finetune", "no_load_optim", "no_load_rng"])
def test_weights_only_load_is_not_recovery(loader, key):
    source, _, args, _, _ = loader
    args = recovery_args(args)
    setattr(args, key, True)
    with pytest.raises(ValueError, match="optimizer and RNG"):
        source(args).load(0)


@pytest.mark.parametrize(
    "fault",
    [
        "absent",
        "empty",
        "truncated",
        "oversize",
        "symlink",
        "parent_symlink",
        "wrong_epoch",
        "wrong_offset",
        "wrong_group",
        "wrong_sample",
        "bool",
        "missing_field",
        "extra_field",
        "metadata",
        "not_dict",
    ],
)
def test_missing_or_damaged_cursor_never_resets_or_mutates_current_state(loader, fault):
    torch = pytest.importorskip("torch")
    source, _, args, _, _ = loader
    instance = source(recovery_args(args))
    path = Path(args.load) / "rollout/global_dataset_state_dict_1.pt"
    path.parent.mkdir(parents=True)
    state = instance._expected_cursor(1)
    if fault != "absent":
        if fault == "empty":
            path.touch()
        elif fault == "truncated":
            path.write_bytes(b"not a checkpoint")
        elif fault == "oversize":
            with path.open("wb") as handle:
                handle.truncate(1048577)
        else:
            if fault.startswith("wrong_"):
                key = {
                    "wrong_epoch": "epoch_id",
                    "wrong_offset": "sample_offset",
                    "wrong_group": "sample_group_index",
                    "wrong_sample": "sample_index",
                }[fault]
                state[key] += 1
            elif fault == "bool":
                state["epoch_id"] = True  # True == 1; equality alone is unsafe.
            elif fault == "missing_field":
                del state["metadata"]
            elif fault == "extra_field":
                state["unexpected"] = 0
            elif fault == "metadata":
                state["metadata"] = {"unreviewed": "synthetic"}
            elif fault == "not_dict":
                state = []
            torch.save(state, path)
            if fault == "symlink":
                target = path.with_suffix(".real")
                path.rename(target)
                path.symlink_to(target)
            elif fault == "parent_symlink":
                target = path.parent.with_name("real-rollout")
                path.parent.rename(target)
                path.parent.symlink_to(target, target_is_directory=True)
    before = (instance.sample_index, instance.sample_offset, instance.epoch_id)
    with pytest.raises((ValueError, pickle.UnpicklingError)):
        instance.load(1)
    assert (instance.sample_index, instance.sample_offset, instance.epoch_id) == before


def test_save_reopens_native_file_and_rejects_in_memory_counter_drift(loader, monkeypatch):
    source, base, args, _, _ = loader
    instance = source(args)
    with pytest.raises(ValueError, match="completed rollout"):
        instance.save(0)
    set_cursor(instance, 0)
    monkeypatch.setattr(base, "save", lambda *a: None)
    with pytest.raises(ValueError, match="missing"):
        instance.save(0)


def test_restore_checks_native_readback_not_just_input_file(loader, monkeypatch):
    pytest.importorskip("torch")
    source, base, args, _, _ = loader
    saved = source(args)
    set_cursor(saved, 0)
    saved.save(0)
    monkeypatch.setattr(base, "load", lambda *a: None)
    with pytest.raises(ValueError, match="completed rollout"):
        source(recovery_args(args)).load(0)
