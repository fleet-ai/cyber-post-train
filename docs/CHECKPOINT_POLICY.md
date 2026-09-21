# Checkpoint policy

This page records when the current Qwen3.8 training plans save progress, how
many saves they keep, and what has actually been proven about restarting from
those saves. A configured save is not automatically a proven recovery point.

## SFT cadence and storage

The current broad-teacher SFT successors use the following policy:

| Treatment | Total optimizer steps | Save every | Saves over the run | Kept live |
|---|---:|---:|---:|---:|
| 32K, batch 8, learning rate 3e-6 | 1,837 | 100 steps | 19 | latest 2 |
| 32K, batch 8, learning rate 1e-6 | 1,837 | 100 steps | 19 | latest 2 |
| 32K, batch 16, learning rate 3e-6 | 919 | 50 steps | 19 | latest 2 |
| 64K, batch 8, learning rate 3e-6 | 1,120 | 15 steps | 75 | latest 2 |

The save count includes the forced final save when the last step is not already
an interval boundary. The exact settings live in
`qwen38-teacher3k-32k-full-b8-lr3e6-v3.json`,
`qwen38-teacher3k-32k-full-b8-lr1e6-v3.json`,
`qwen38-teacher3k-32k-full-b16-lr3e6-v3.json`, and
`qwen38-teacher3k-64k-full-b8-lr3e6-v5.json` under `configs/runs/`. They are
locked by
`tests/test_teacher3k_successor_checkpoint_cadence.py` and
`tests/test_teacher3k_successor_checkpoint_retention.py`.

One measured full-weight checkpoint is about 303 GiB. Keeping two therefore
uses about 606 GiB after pruning. Because the new save is written before the
oldest retained save is removed, allow at least about 909 GiB during a save.
The 32K plans write about 5.62 TiB cumulatively over 19 saves; the 64K plan
writes about 22.19 TiB over 75 saves. Those cumulative figures estimate storage
traffic, not simultaneous storage. The source measurement is
`docs/evidence/qwen38-full-checkpoint-size-audit-20260920.json`.

The target cadence and the watchdog horizon are different:

- The **target cadence** is the configured interval above. Historical step
  rates put each first save at roughly 2.9 to 3.25 hours.
- The **recovery watchdog horizon** is a conservative upper bound used to avoid
  killing slow but healthy work before its next save. It is 53,300 seconds
  (about 14.8 hours) for the 32K plans and 17,460 seconds (about 4.85 hours) for
  the 64K plan. It is not the expected time between saves.
- A separate 20-minute no-progress check still detects a genuinely stalled
  allocation. Increasing the recovery horizon did not relax that check.

SFT has a tested resume mechanism and small-fixture GPU proof. A newly produced
broad checkpoint still needs its own exact reload and continuation check before
it should be called a qualified recovery point.

## RL cadence and recovery boundary

The one-step Qwen3.8 RL reward canary saves at step 1 and keeps at most two
checkpoints. The prepared 10-step and 50-step SkyRL treatments save every 10
steps and keep the latest two. Their code checks the expected files, counters,
sampler cursor, optimizer state, random-number state, and policy-shard hashes.
The exact files are
`configs/qualification/qwen38-rl-reward-canary-prod-v8.json`,
`configs/qualification/qwen38-skyrl-production-plan-a1-v1.json`, and
`configs/qualification/qwen38-skyrl-production-plan-dose50-v1.json`.

Those RL plans currently set `trainer.resume_mode` to `none`, and their result
explicitly reports that GPU reload has not been verified. No live broad RL
checkpoint has yet passed a recovery qualification. The current RL files
therefore prove checkpoint structure and contents, not recoverability.

RL recovery needs two separate checks. First, a fresh exact-image canary must
reload step 1 without an optimizer update and prove the saved policy,
optimizer, random-number state, sampler cursor, and trainer global step were all
restored. Second, a create-once continuation must execute exactly one finite
optimizer update to step 2, seal and reload step 2, and prove the source step-1
checkpoint stayed byte-identical. Only that complete step-1-to-step-2 chain may
qualify a separately reviewed successor to enable RL resume; a zero-update load
by itself is not sufficient.

## Save telemetry successor

Current SFT plans do not yet publish checkpoint-save duration or byte counts to
W&B. Adding that instrumentation directly to `training/sft_runtime.py` changes
the exact runtime digest bound into prepared plans, one-off qualification
requests, and historical receipts. Rewriting those sealed artifacts would make
their earlier evidence untrue, so telemetry must ship as a separately versioned
runtime successor. It must not delay the already prepared broad successors.

That successor should report only scalar values to its local stream and W&B:

- time spent inside the native `save_checkpoint` call;
- bytes and files visible when that call returns; and
- the count of filesystem-stat errors during that observation.

The distributed writer may still be flushing after the native call returns, so
those metrics must be named `native_save_return`, not final durable size,
end-to-end write time, or write throughput. The terminal checkpoint needs an
explicit flush path so its metrics cannot be lost merely because no later
training log event occurs. Filesystem observation must remain non-fatal, and
tests must cover both ordinary periodic saves and the final-only save.

Landing the successor requires new run/plan identities, fresh runtime and
request digests, CPU preflight, and a bounded checkpoint/reload qualification.
It must not rewrite any historical sealed plan. Fully durable size and
reloadability still require the separate checkpoint finalization and reload
gates.
