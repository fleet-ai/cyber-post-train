# Exact pass@4 ledger v50

V50 resolves the last cell previously classified as active. Qwen rank 99,
attempt 4 has an exact immutable claim, a full trace manifest, a scoring intent,
closed-instance evidence, and a terminal failed Kubernetes Job/Pod. It is
therefore **blocked/non-repeatable**, not accepted and not eligible to rerun.

The strict score-blind tally is now:

- Qwen3.8-27B: 42 accepted, 0 active, 9 blocked/non-repeatable, 349 unstarted.
- GLM-5.3: 24 accepted, 0 active, 4 blocked/non-repeatable, 372 unstarted.
- Total: 66 accepted, 0 active, 13 blocked/non-repeatable, 721 unstarted.

The machine-readable authority is
`2026-09-06-exact-pass4-ledger-v50.json`. It contains no prompt, trace content,
flag, answer, reward, score, transcript, or credential. Two older Qwen aggregate
claims remain unresolved because they still lack exact cell-level authority;
V50 does not project either claim onto the experiment cells.
