# Cleanup RL failures and alerts — 2026-09-11

Initial audit through 15:40 UTC; follow-up through 15:57 UTC. Chris subsequently
reauthorized submissions only after checking the intended path, then reported
another alert. No new cluster job was created during this investigation.
No monitor, failure record, peer workload, or healthy active workload was changed.

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

## Eighth failure: the already-running Miles V7

The exact V7 RayJob failed at 15:46:54 UTC (08:46:54 PDT). Its UID is also in
the status monitor's processed-failure list. This was another real failure, not
a duplicate of an earlier alert or a new submission after the complaint.

The digest-checked native receipts and hashed episode records locate the failure
in the initial `dev-baseline` episode, after 1,377.21 seconds, before grading or
optimization. The specific recorded reason is `generation_incomplete` from
`rl_episode._agent`. That branch combines a per-turn length stop, full context,
aborted generation and missing text; the frozen bundle did not retain which one.
Do not claim it was definitely a token limit, tool timeout or cluster fault.
There is no accepted reward, optimizer update or checkpoint from V7.

The Pod and RayCluster are gone; the exact Workload has `admission: null`,
`QuotaReserved=False` and `Admitted=False`, with reason `Finished`. Its eight
GPUs are released. The cleanup receipt records the owned Fleet instance closed,
and a separate authenticated GET confirms that same instance is `stopped`.
[Terminal evidence](../evidence/cleanup-miles-rl-v7-terminal-20260911.json)

The next RL GPU submission remains held on an unresolved generation-stop
contract, not on missing user authorization. Test that boundary before another
run; do not submit a replacement just to learn which stop occurred. Distinguish
an expected bounded/incomplete canary from a real runtime defect without
pretending either produced a valid reward or update. Any clean-rejection design
must remain explicit in receipts and must not swallow unexpected errors.

A local regression reproduced the diagnostic ambiguity before the correction.
The runner now preserves separate allowlisted reasons for length, context,
abort and non-text stops, without exposing output text or arbitrary finish values.
All retain the same no-scoring/no-optimization rejection; no error exit was
silenced, budget changed or live bundle patched. This is a diagnostic repair,
not proof that the incomplete-generation path is resolved.

## Follow-up: source verification and the pre-existing GLM CPU pilot

At 16:10:04Z the already-running `chris-cpt-glm-native-stage-v1` container
(Pod UID `ae29a922-9aba-4c58-9f5b-cce12fea665a`) exited 1 with zero restarts.
Job UID `9289aa28-bb9e-4bc3-a6ae-aa2302cef3ff` became `Failed=True` at
16:10:07Z. The Workload is absent; the CPU/RAM reservation is released, and
the Job requested no GPUs. The last process sample observed `FAILED.json`
present, `MODEL_LOADED.json` and `STAGED.json` absent, and zero cgroup OOM or
OOM-kill events. Do not infer the application cause from memory usage: its
receipt was not retrieved before the container exited. No successor was submitted.
Chris supplied the 09:10 PDT Slack notification, independently establishing
delivery for this failure. This is a ninth failed Job in this incident, not an
additional RL optimizer failure.

The live CronJobs still select `fleet/ftl:91212928`: failure polling every five
minutes and idle checking every thirty minutes, with the 1%/30-minute/one-hour
idle policy. The exact revision's `failed_jobs` and `pending_failures` functions
were executed locally against synthetic objects, without any delivery function.
CPU Jobs with `fleet-infra-quiet` or `c1` and a GPU RayJob all qualify when failed;
the same UID is deduplicated. The source SHA-256 is
`d1272982db8cece4c58cb72ecef4d1d106be42a91f387717ba8275c5aa8e89bc`.
There is no current per-job quiet annotation. The root operational mistake
remains live qualification on monitored Jobs, not ignorance of a priority flag.

New cluster submissions are held while the development/reporting path is
resolved. Work can continue locally. A supported owner-only development route
requires platform-maintainer approval; never evade the monitor through another
namespace/resource kind, altered deduplication state, or false success.

## Local stop-contract correction, not deployed

The exact installed FTI recorder source was matched by SHA-256 to Theseus
`a170ed0d5cff13de3f0abe56445d5cdd4bba696f`:
`services/fti/src/fti/trainers/miles/recording.py`, SHA-256
`593698b7e12ad97eee37804d5dba7f36ad9d73c8cb9e8c06ae20328d3bd18b48`.
Its agent treats known horizon limits as explicit episode endings. Our adapter
had converted them into job-fatal generic exceptions. This does not establish
which stop occurred in the old V7 bundle.

The local correction distinguishes explicitly typed known budget stops from
aborts, unknown finishes, malformed responses and runtime faults. Budget-limited
batches stop without scoring incomplete episodes, refilling samples or feeding
them to optimization. They record `REJECTED`, never `ACCEPTED` or training
completion. Mixed failures, uncertain instance release and process teardown
failure stay hard failures. Native and supervisor receipts must match the plan
and digest; a nonzero child exit cannot be reclassified as rejection.

Regression tests caught and fixed two additional boundaries before deployment:
Ray replaces exception arguments during transport, and native Miles group
cleanup discards sibling exceptions after awaiting them. The adapter preserves
typed reasons through real Ray 2.58 serialization and observes each episode
error so a sibling cleanup failure cannot disappear. Process teardown now
precedes any supervisor completion/rejection marker.

Focused local tests with Ray 2.58: 415 passed, 45 explicitly skipped. The broad
local suite without optional services/native runtimes: 1,790 passed, 189 skipped,
seven subtests passed. Native Miles image/GPU paths remain unqualified by these
new tests; these results do not authorize a replacement launch. No live bundle,
alert monitor, failure state, credential, peer or rollout campaign was modified.
