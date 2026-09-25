"""Verify the frozen Qwen tokenizer and SkyRL's native assistant loss mask."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from functools import cache
from pathlib import Path

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
