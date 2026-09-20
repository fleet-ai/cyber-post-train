# Qwen3.8 Fresh75 v2 export/reload/staging packet preparation — 2026-09-20

## Outcome

Seven deterministic, no-submit packets are frozen in
`configs/qualification/qwen38-fresh75-v2-export-reload-stage-packets-v1.json`.
They cover every terminal accepted SFT arm and deliberately exclude the still-running
`b8-lr1e5-e4` duration/overfit arm. The packets execute no optimizer step, expose no
external benchmark content or outcomes, make no API or cluster call, and perform no
SFS write.

The packet-set digest is
`sha256:da5330cb79bc1afca96dfb3ab57bea731ea317edf5434dbdba2c84fcb36f3401`;
the JSON file digest is
`sha256:05c744a687fedd4d447ca60641480f1c942b3ec7c8d651569483ddbd14bfde00`.
It binds the frozen heldout queue at
`sha256:942a2a0e19df7d68b0066928b1c6bbeb5121d934a79825354015de272c6245f6`.

This is preparation, not execution. The explicit `10/10` failure stop remains in force,
so every packet is `launchable: false`, `submitted: false`, and
`external_mutation_authorized: false`.

## Exact source bindings and reserved destinations

| Priority | Arm | Step | Saved receipt path | Saved receipt file SHA-256 | Embedded receipt SHA-256 | Reserved served id |
|---:|---|---:|---|---|---|---|
| 1 | `b32-lr1e5-e2` | 58 | `/mnt/sfs/jobs/chris-q38-f75-b32-v2/checkpoint_receipts/step-000058.json` | `sha256:d42aa664fb99704bd21cd663107f7fa275b0209f9bd3476c8a2c8341bedcc2a4` | `sha256:f18f2e62669a9b0ae41742d3cbd995053432e57f565a551a8324092ac703bffa` | `chris-q38-f75v2-b32-lr1e5-e2-wbe` |
| 2 | `b16-lr1e5-e2` | 116 | `/mnt/sfs/jobs/chris-q38-f75-b16-v2/checkpoint_receipts/step-000116.json` | `sha256:b920774f52af5a3c4384275ed33d82f9f7b0c4e6579c85f2526b74a2e8c3c062` | `sha256:28a93a72375e9bfd513564f50d808fdbb7b3ec02c60e8df0569c8e05a2e11b8f` | `chris-q38-f75v2-b16-lr1e5-e2-wbe` |
| 3 | `b8-lr1e5-e2` | 230 | `/mnt/sfs/jobs/chris-q38-f75-base-v2/checkpoint_receipts/step-000230.json` | `sha256:b3dc481ed81d513961f1e4717837aed14e0cbcc5c33269d34595de01ca3b62b0` | `sha256:a9fe223effaac6670f2084a2c85751539d60368a41d4efb3a47ce45e00258811` | `chris-q38-f75v2-b8-lr1e5-e2-wbe` |
| 4 | `b8-lr3e6-e2` | 230 | `/mnt/sfs/jobs/chris-q38-f75-lr3-v2/checkpoint_receipts/step-000230.json` | `sha256:9f8a834d8ac97e0df0c20d2a8501bdc2c50895a58bbc34ff2de237fad0a795f5` | `sha256:06369b3366b07332246ff4d7115a04ade4f01f795ad0c2a65a57da0c953c7146` | `chris-q38-f75v2-b8-lr3e6-e2-wbe` |
| 5 | `b64-lr1e5-e2` | 30 | `/mnt/sfs/jobs/chris-q38-f75-b64-v2/checkpoint_receipts/step-000030.json` | `sha256:50d3d1d058d09f3137334146fd7a44157a3d65d472b56b02013ed74fe43fc0dd` | `sha256:8a842953561f892d4d4450ca057cd1ac09125e616bdc6396e5d16ee16c7f0975` | `chris-q38-f75v2-b64-lr1e5-e2-wbe` |
| 6 | `b8-lr1e6-e2` | 230 | `/mnt/sfs/jobs/chris-q38-f75-lr1-v2/checkpoint_receipts/step-000230.json` | `sha256:4945ae4f47ab9b8d7039de856c7dd0c301f57f9dcb54bfaccdc61448b1cbd8c7` | `sha256:5972ba9ce617f2f1c043e56c1a32b5a7f2b7fe186dfe3a3bf8abd7a8fd901f7a` | `chris-q38-f75v2-b8-lr1e6-e2-wbe` |
| 7 | `b8-lr1e5-e1` | 115 | `/mnt/sfs/jobs/chris-q38-f75-e1-v2/checkpoint_receipts/step-000115.json` | `sha256:fd92b96e9bb8f843eca6c3cde9f155d6a99c96faf3f89ce52dc602c99ff7a319` | `sha256:17eb31f0dc0e371471965037b6bdc812328df71eea94db63e553af3a0c15f269` | `chris-q38-f75v2-b8-lr1e5-e1-wbe` |

Each packet additionally freezes the exact accepted handoff, reconstructed source plan,
runtime digest, native checkpoint path, create-once seal/export/check paths, one-GPU
request/root, future stage-plan path, inference destination, and acceptance marker.
Destinations are unique across all seven arms.

The priority order is operational only: terminal-arm recent-decile train loss and
relative drop were respectively `0.2303 / 56.1%`, `0.2342 / 53.3%`,
`0.2348 / 52.8%`, `0.2554 / 48.7%`, `0.2604 / 48.4%`, `0.3423 / 36.0%`, and
`0.4047 / 20.2%` in the table order. These finite training metrics prioritize handoff
work; they are not capability evidence and must not select the evaluation winner.

## Frozen zero-step contract

The generator binds the exact Qwen base revision and model/tokenizer/template manifests,
the training and staging image digests, and SHA-256s for every implementation file used
by sealing, export, checking, and staging. The export contract requires:

- CPU-only create-once BF16 reconstruction with `optimizer_steps_executed: 0`;
- 1,184 trained tensors plus the exact 15 restored MTP tensor names, producing 1,199
  tensors and 55,562,855,904 bytes;
- unchanged source sizes/mtimes and byte-equal reopen checks;
- a CPU integrity/meta-layout reload, followed only after separate authorization by one
  c1 GPU finite two-token synthetic reload; and
- zero-GPU, receipt-first, no-replace staging of exactly 29 payload files into an absent
  inference destination, with registration explicitly out of scope.

The final stage plans cannot yet be materialized honestly: their required export and
GPU-check file/embedded digests and exact payload manifest do not exist. The packets
reserve their paths and enforce the complete input contract without inventing values.

## Read-only SFS audit and present blocker

`training/fresh75_export_packets.py audit-sfs` is a local-filesystem-only auditor. On an
already-mounted `/mnt/sfs`, it opens each exact receipt with `O_NOFOLLOW`, checks both
the frozen file digest and embedded receipt digest, rechecks plan/step/path bindings,
and validates the exact native checkpoint layout without writing source bytes or
creating a reader workload.

This worktree host did not expose `/mnt/sfs`; therefore no live SFS receipt/layout audit
was claimed. The next operator gate is to run this read-only command in an environment
where the existing mount is already available:

```sh
uv run python -m training.fresh75_export_packets audit-sfs
```

Only after that passes, and only after a renewed failure budget and explicit mutation
authorization, may an operator create the checkpoint payload seals. The subsequent
gates remain strictly ordered: freeze seal digests; CPU export; CPU check; separately
authorized one-GPU reload plus UID-bound terminal/release proof; freeze export/GPU
receipt digests into the generated stage plan; destination-absence GET; zero-GPU
atomic staging; paused new-UID registration; and fresh live base/candidate parity.

## Verification performed

```text
uv run ruff check training/fresh75_export_packets.py tests/test_fresh75_export_packets.py
All checks passed!

uv run pytest -q tests/test_fresh75_export_packets.py
8 passed

uv run pytest -q tests/test_model_stage.py tests/test_fresh75_export_packets.py
23 passed

uv run pytest -q tests/test_fresh75_holdout_queue.py tests/test_model_stage.py tests/test_fresh75_export_packets.py
31 passed

uv run python -m training.fresh75_export_packets verify
status: valid_no_submit_packet_set
packets: 7
launchable: false
```

The test suite proves deterministic regeneration, all-seven coverage, distinct
create-once destinations, fail-closed zero-step gates, exact handoff/receipt bindings,
tamper rejection, nofollow SFS reads, exact native layout validation, and absence of an
external control-plane client. No launch, stage, registration, POST, cancellation, or
other external mutation occurred.

The broader `test_checkpoints.py`, `test_export.py`, and `test_export_check.py` suites
could not be collected on this host because its local `uv` environment does not include
PyTorch. That is an environment limitation, not a passing result; the packet-specific
tests exercise the read-only bindings, and the PyTorch-dependent suites remain required
in the pinned training image before any later authorization to execute these operations.
