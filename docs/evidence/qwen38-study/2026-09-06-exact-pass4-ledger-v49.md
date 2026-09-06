# Exact pass@4 ledger v49

This is the score-blind, fully enumerated audit of the 800 statistical cells in
`chris-cyber-q38-glm53-exact-easiest100-pass4-v1`. The machine-readable receipt
is `2026-09-06-exact-pass4-ledger-v49.json`.

## Strict cell-authoritative tally

| Model | Accepted | Active | Blocked/nonrepeatable | Unstarted | Total |
|---|---:|---:|---:|---:|---:|
| Qwen3.8-27B | 42 | 1 | 8 | 349 | 400 |
| GLM-5.3 | 24 | 0 | 4 | 372 | 400 |
| Total | 66 | 1 | 12 | 721 | 800 |

The tally reproduces the v48 manifest and its successful live validation. Every
cell appears exactly once with immutable model, rank, attempt, cell ID, task
version ID, state, and any v48 evidence receipt reference.

## Why the later 51/0/9/340 Qwen tally is not adopted

The later Qwen lineage carries an aggregate claim of 51 accepted, zero active,
nine blocked, and 340 unstarted. It does not contain the nine exact acceptance
receipts needed to make that projection:

- rank 15 and rank 16 controller Jobs both succeeded, but controller exit zero
  also permits quarantine or preserved-claim outcomes; the eight cell receipts
  are not in the merged lineage;
- rank 99 attempt 4 was active in v48, but no later exact accepted receipt is
  present in the inspected lineage;
- the rank 18 terminal receipt increments the blocked count by one without
  naming which attempt/cell, while also recording zero model calls.

The selector subsequently classified all 100 task identities as ambiguous. Its
reported tally is carried input metadata, not a tally derived from the selector
rows. Therefore v49 records the later evidence and its exact file/receipt hashes
but does not silently project aggregate assertions onto cells.

## Resolution needed

Promote the later claim only after obtaining and validating:

1. one exact terminal receipt for each rank 15/16 attempt;
2. the exact rank 99 attempt 4 terminal receipt; and
3. an exact rank 18 cell ID plus a valid nonrepeatable claim, if one cell truly
   became blocked.

No workload, Fleet API, model, verifier, or scoring mutation was performed to
produce this audit. No prompts, traces, flags, answers, scores, or credentials
were read.
