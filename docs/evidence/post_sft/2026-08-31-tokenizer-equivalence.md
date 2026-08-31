# Post-SFT tokenizer equivalence gate

The step-318 trainer sidecars are not byte-identical to the pinned Qwen3.6-27B base, so the raw
export must not be served directly. The difference is a Transformers serialization change rather
than an effective vocabulary change: both tokenizers load to the same 248,077 token→ID mapping,
the core special-token IDs match, and IDs 248070–248076 were already declared in the base
`tokenizer_config.json` before appearing in the trainer's `tokenizer.json`.

Exact encode and decode parity passed over 170,225 strings and 48,468,947 tokens, including all
2,540 rendered SFT training windows and their raw message/tool fields, all ten lineage-held-out
Fleet development prompts, all 15 WebExploitBench prompts plus frozen harness inputs, the fixed
five ExploitGym Qwen Code chats, and explicit tool/control strings. Receipts contain hashes and
counts only; no benchmark or Fleet prompt content is committed.

The accepted serving artifact is therefore a weights-only composition:

- safetensors shards and index must be byte-identical to the completed raw step-318 export;
- every runtime sidecar must be byte-identical to base revision `6a9e13bd...`;
- every post-training tensor key, shape, and dtype must match the base layout; and
- the live SGLang route must still pass tokenizer, tool-call, runtime, and checkpoint parity.

This is pre-export evidence from the exact selected source checkpoint. The completed export is
still required to repeat these checks and bind all output hashes before either benchmark launches.
The machine-readable receipt is
[`2026-08-31-tokenizer-equivalence.json`](./2026-08-31-tokenizer-equivalence.json).
