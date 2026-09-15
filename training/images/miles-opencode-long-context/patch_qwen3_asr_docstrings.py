"""Patch two missing Qwen3-ASR argument docs in the pinned Bridge source."""

# The long literal lines below are exact anchors from an immutable upstream file.
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

SOURCE_SHA256 = "bfdb9bff47ea68a1e72924d7f7ea83a16bde921c731298f1cd346d99c4b46b40"
PATCHED_SHA256 = "90ebb373c06195ad4a8d117e17ed154ffc67c78ccc370fe7f5dcab78ae864355"

_MODEL_FORWARD = """    ) -> Union[tuple, BaseModelOutputWithPast]:
        if (input_ids is None) ^ (inputs_embeds is not None):"""
_MODEL_FORWARD_PATCHED = """    ) -> Union[tuple, BaseModelOutputWithPast]:
        r\"\"\"
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of input sequence tokens in the sequence.
        \"\"\"
        if (input_ids is None) ^ (inputs_embeds is not None):"""
_CONDITIONAL_FORWARD = """        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        \"\"\""""
_CONDITIONAL_FORWARD_PATCHED = """        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of input sequence tokens in the sequence.
        \"\"\""""


def patch(raw: bytes) -> bytes:
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("unexpected Megatron Bridge Qwen3-ASR source")
    source = raw.decode()
    if source.count(_MODEL_FORWARD) != 1 or source.count(_CONDITIONAL_FORWARD) != 1:
        raise ValueError("Qwen3-ASR docstring patch anchors changed")
    result = source.replace(_MODEL_FORWARD, _MODEL_FORWARD_PATCHED).replace(
        _CONDITIONAL_FORWARD, _CONDITIONAL_FORWARD_PATCHED
    ).encode()
    if hashlib.sha256(result).hexdigest() != PATCHED_SHA256:
        raise ValueError("Qwen3-ASR docstring patch output changed")
    return result


def main() -> None:
    path = Path(sys.argv[1])
    path.write_bytes(patch(path.read_bytes()))


if __name__ == "__main__":
    main()
