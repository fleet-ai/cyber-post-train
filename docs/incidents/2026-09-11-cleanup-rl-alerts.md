# Cleanup RL failures and alerts — 2026-09-11

Observed through 15:40 UTC. New cluster submissions and successors are paused
following Chris's complaint, pending his explicit resumption. No monitor,
failure record, peer workload, or healthy active workload was changed.

## What is established

Seven `chris-cpt-*` RayJobs from this cleanup task failed today between
10:16:26 and 12:26:55 UTC (03:16–05:26 PDT). All seven exact UIDs are present in
the live `fleet-train-control-plane/ft-idle-monitor-status` ConfigMap's
`data.state.failed` list. This proves the monitor recorded them as processed;
it is not an independent per-message Slack delivery audit. Other historical
`chris-*` jobs are excluded from this incident's count.

| Failed run | Recorded cause and confidence | Evidence |
| --- | --- | --- |
| `chris-cpt-q38-miles-rl-v1-52144863` | Confirmed packaged launcher defect: `runpy` did not register `__main__` before wrapper dispatch. | [Receipt](../evidence/cleanup-rl-real-preparation-20260911.json) |
| `chris-cpt-q38-miles-rl-v2-6ec354b0` | Actor died; original underlying cause was not retained. CPU reproduction separately found unwanted vision-processor selection for text inputs. Do not present the reproduction as recovered original cause. | [Terminal](../evidence/cleanup-rl-real-preparation-20260911.json), [reproduction](../evidence/cleanup-miles-text-loader-20260911.json) |
| `chris-cpt-q38-miles-rl-v3-9e31cbcd` | Actor initialization failed; original nested cause was not retained. CPU reproduction proved our loader rejected the actual native `tool_key=tools` default, unlike the test fixture. | [Receipt](../evidence/cleanup-miles-rl-v3-20260911.json) |
| `chris-cpt-q38-miles-rl-v4-68547176` | Confirmed missing `megatron.post_training` import path during native trainer initialization. | [Receipt](../evidence/cleanup-miles-rl-v4-20260911.json) |
| `chris-cpt-q38-miles-rl-v5-f81a2a88` | Confirmed host-RAM OOM, not GPU-memory exhaustion. The inherited 768-GiB limit was inadequate for RL initialization. | [Receipt](../evidence/cleanup-miles-rl-v5-20260911.json) |
| `chris-cpt-q38-skyrl-rl-v3-f948d864` | Confirmed permission error: root-run preparation/preflight hid that the actual UID 1000 trainer could not read root-owned 0700 inputs. | [Receipt](../evidence/cleanup-skyrl-rl-v3-20260911.json) |
| `chris-cpt-q38-miles-rl-v6-daa26380` | Real episode reached 46 assistant messages, then failed at a tool boundary before grading. The old receipt lost nested errors; the exact cause remains unknown. A separately fixed tool deadline is not proof of this cause. | [Receipt](../evidence/cleanup-miles-rl-v6-20260911.json) |

All seven have recorded GPU/quota release. None establishes an accepted RL
reward, optimizer update, or RL checkpoint. These were real failed jobs, not
seven repeat notifications about one failed object.

## Why the alerts occurred

Live `fleet-train-control-plane/ft-idle-monitor-status` runs every five minutes,
with Slack and Better Stack enabled, using image `fleet/ftl:91212928`.
The inspected [matching source](https://github.com/fleet-ai/theseus/blob/91212928e107ff949899f37a563d7cd0b9c9d123/services/ftl/src/ftl/report_status.py)
selects failed Jobs/RayJobs and deduplicates by immutable UID. Each successor is
a new object, so each new failure is independently eligible. A high-priority
request does not change this rule. No evidence in this review attributes an
idle-GPU alert to these seven runs; their observed lifetimes were below one hour.

## Operational mistake

The agent kept qualifying newly integrated runtime stages through successive
live GPU jobs. Tests covered components but did not initially exercise the exact
packaged entrypoint, real native defaults/imports, actual runtime user, or
full-scale host-memory demand. Several avoidable integration errors therefore
escaped. Passing CPU tests was treated as sufficient to proceed despite the
remaining GPU-only gaps, and incomplete failure diagnostics did not stop the
successor cycle soon enough. This violated the user's stated operating constraint.

Corrective code/tests already exist for the bootstrap, text-loader defaults,
import path, runtime-user preflight, RAM reservation and sanitized nested error
capture. Their existence is not proof that RL now works. No further live retry
is authorized by this document. Further proposals must state remaining untested
boundaries honestly; do not promise zero failures, suppress genuine alerts,
force success exits, or delete evidence to avoid notification.

## Progress, kept separate from the failures

- Qwen SFT, held-out loss logging, W&B, native checkpoint restore, one resumed
  optimizer update, BF16 export and separate synthetic GPU reload/generation
  were verified. These small operational fixtures are not a capability study.
  [Progress index](../CONSOLIDATION.md)
- At 15:39 UTC, Miles V7 remained Running/Ready with zero restarts, active GPU
  computation and one isolated Fleet episode. No reward, checkpoint or completed
  RL update was present. It was not cancelled or modified during this review.
- At 15:38 UTC, the local evaluation V8 had 149 completed model turns and no
  terminal result. This is not an accepted rollout yet.
- GLM's CPU-only staging pilot, submitted before the complaint, started at
  15:37:04 UTC with zero restarts and no GPU reservation. Full GLM training is
  still unqualified. No new submission was made after the complaint.

## Local follow-up at 15:46 UTC

Four new regression cases reproduced loss of nested `ExceptionGroup` causes at
the shared native trainer error boundary. The small recorder fix preserves leaf
types and code locations, caps the receipt at eight distinct exceptions, handles
cycles and still excludes private messages. The real MCP library's synthetic
timeout test now exercises this boundary as well as episode diagnostics. This
does not recover missing historical causes or change any frozen running bundle.

Full local suite: 1,780 passed, 157 explicitly skipped, seven subtests passed.
The shared RL runtime has 82/82 statements and 24/24 branches covered; this is
not complete repository coverage or GPU qualification. Ruff and formatting pass.
A prior test invocation exposed a pinned-image GLM test placed outside the
native test group; it is now collected under the existing pinned-runtime gate.
No production GLM source guard was weakened, and no new native-image job was run.
