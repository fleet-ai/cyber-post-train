# Qwen3.8 SFT checkpoint flow

Only new corrected-corpus runs qualify. Old 96K Teacher3K native steps
150/200/250 were pruned; receipts cannot recreate them.

For `N` steps saving every `I`, use `keep_checkpoints=N`, `eval_interval=I`.
Pinned SkyRL evaluates an off-interval final step after saving it: its receipt
names step `N` although W&B logs at `N+1`. Require digest-valid save and
same-step teacher-CE receipts for every checkpoint.

Budget ~303 GiB/native checkpoint and ~55.6 GB/BF16 export: 856 steps/every
50 means 18 native (~5.33 TiB) plus ~1 TB exports. Shared SFS had ~421 TiB
free on 2026-09-25, not reserved; recheck quota/free space before admission.
Never prune an unevaluated checkpoint.

`training.checkpoint_tick.tick(prepared_dir, rayjob_uid, ...)` handles RUNNING
or SUCCEEDED trainers and initiates at most one seal→BF16 export→CPU check→
one-GPU zero-optimizer reload→ready stage. Fsynced claims/intents prevent
replay after uncertain creation. Caller supplies UID-bound live reads, shared
capacity lease, duplicate/capacity checks, server preview and create/unsuspend
callbacks; **no implicit live adapter exists**. Inspect with
`python3 -m training.checkpoint_flow spec <prepared-dir> <step> --for-stage <stage>`.

CPU Jobs use zero GPUs; GPU reload uses one. Require rendered c1/q1 and root
`fleet.ai/failure-alerts: "off"`. `CHECKPOINT_READY.json` proves artifacts and
same-step CE, **not** task success; Fleet pass@4 awaits served-route parity.

If storage forbids keep-all, pause at each interval, seal/export, then exactly
resume. Asynchronous export with keep=2 races pruning and is not sufficient.
