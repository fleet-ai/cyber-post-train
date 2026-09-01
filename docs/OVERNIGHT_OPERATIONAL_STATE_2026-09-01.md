# Cyber post-training overnight operational state — 2026-09-01

Last reconciled: **2026-09-01 13:41 UTC**. This is a sanitized operational
handoff. It contains no prompts, traces, flags, benchmark answers, credentials,
or unsealed scores. Live observations are time-bound; immutable receipts remain
the authority for terminal claims.

## Evidence vocabulary

- **Proven** — validated immutable repository evidence establishes the claim.
- **Measured** — a read-only observation established the claim at the stated
  time; it may later change.
- **Inferred** — a conclusion follows from proven or measured facts but is not
  itself a terminal receipt.
- **Pending** — the required authority or terminal evidence does not exist yet.

## Current decisions

| Stream | Classification | Current state |
|---|---|---|
| Qwen3.8 Fleet calibration v2 | Proven terminal infrastructure incident | One valid authoritative zero was retained, three attempts are unresolved, sixteen never launched, and no positive reward was observed. The reward-release gate remains closed; v3 was not submitted. |
| Qwen3.8 training qualification | Proven blocked before submission | The train-only corpus is locally verified but unstaged. The training catalog has no Qwen3.8 row, the exact model bytes are not staged on the training volume, and the candidate trainer has only static/same-architecture support—not Qwen3.8 forward/backward/optimizer/checkpoint proof. |
| Qwen3.8 WebExploitBench v2 | Proven infrastructure-interrupted and unscored | The controller disappeared at the unchanged memory gate before any trial, model call, event, resource, or terminal outcome. The v2 root is immutable and must not be resumed or scored. |
| Qwen3.8 WebExploitBench v3 | Pending explicit launch approval | The single-use successor and durable supervisor/runtime-root gates are merged, but only unpaid preview is authorized. No v3 launch is claimed. |
| Qwen3.6 SFT serving | Proven live parity | The post-SFT registration and Ready/live parity gates completed and are bound by sanitized receipt digests. This proves served artifact parity, not benchmark capability. |
| Qwen3.6 WebExploitBench post-SFT v1 | Proven infrastructure-invalid | One trial completed, one was orphaned, and thirteen never started before controller loss. No score or trace was inspected; the run is not accepted as an evaluation. Exact-run orphan cleanup completed without touching peer workloads. |
| Qwen3.6 WebExploitBench post-SFT v2 | Pending credential rotation and explicit approval | The successor changes only its single-use run ID and derived config digest. Its merged preview reports `launch_authorized=false` and `resources_created=false`; it has no execute path. |
| Qwen3.6 ExploitGym paired run | Measured active and healthy at 13:41 UTC | Job `chris-cyber-exploitgym-q36-sft-paired-v1`, UID `f5fdf341-329e-4674-9768-a20d417f74b5`, had one ready/running Pod, UID `48c62e27-a86c-4b74-b88a-d3ceaa360443`, with zero restarts; Job counts were active 1, succeeded 0, failed 0. No log, prompt, trace, answer, or score was read. |
| Qwen3.6 prompt curriculum | Pending shared API deployment | Owned consumer validation is merged, but shared Theseus PR #28555 remains an open draft. Creation stays blocked until the exact-version response behavior is merged, deployed, and proven against the public endpoint. |

## Qwen3.8 evidence

The minimized Fleet calibration incident is
[`2026-09-01-fleet-calibration-v2-infrastructure-incident.json`](evidence/qwen38-study/2026-09-01-fleet-calibration-v2-infrastructure-incident.json).
It explicitly records the source receipt's non-reproducing embedded digest,
preserves the incident as descriptive only, and refuses cleanup, score recovery,
rerun, or v3-launch claims. Therefore the absence of a positive reward is
**proven for the retained valid outcome**, while model capability across the
unresolved or never-launched attempts is unknown.

The read-only qualification receipt
[`qwen38-27b-readiness-2026-09-01-v2.json`](../configs/qualification/qwen38-27b-readiness-2026-09-01-v2.json)
proves the following without staging or submission:

- the exact 18-shard BF16 model lock and 55,563,006,776-byte weight set;
- a train-only local corpus of 508 successful sessions from 86 lineages,
  yielding 2,540 windows with no dev, test, or external-benchmark rows;
- no Qwen3.8 Training API catalog row or immutable training-volume path;
- a Ready Qwen3.6-proven trainer candidate whose Qwen3.8 execution
  compatibility remains unproven; and
- no model staging, corpus staging, image build/push, or training submission.

The trainer compatibility receipt
[`qwen38-27b-trainer-compatibility-2026-09-01-v1.json`](../configs/qualification/qwen38-27b-trainer-compatibility-2026-09-01-v1.json)
additionally proves that the serving and training model roots are on distinct
storage. Reusing the inference path as if it were staged training data would be
false evidence. Draft PR [#78](https://github.com/fleet-ai/cyber-post-train/pull/78)
contains a dry-run-only, create-once rematerialization bridge. It remains
unmerged and blocked on explicit approval for the 55.6 GB transfer and a
positive reward gate; no staging resource was created.

The WebExploitBench v2 incident is frozen in
[`2026-09-01-qwen38-v2-interrupted-unscored.json`](evidence/webexploitbench/2026-09-01-qwen38-v2-interrupted-unscored.json).
The v3 successor changes only the run identity and resulting digest. Merged
durability gates seal the complete evaluator source, bind the actual CAGE
interpreter/modules/repository/benchmark/run/claim roots, and create
owner/start/exit receipts with file-and-directory durability. Those gates make
a future launch safer; they do not authorize one.

## Qwen3.6 evidence

The Qwen3.6 post-SFT incident receipt
[`2026-09-01-qwen36-post-sft-v1-infrastructure-incident.json`](evidence/webexploitbench/2026-09-01-qwen36-post-sft-v1-infrastructure-incident.json)
binds the exact post-SFT model revision, Qwen Code 0.22.3, CAGE protocol,
paired-identity receipt, completed registration receipt, and live-parity
receipt. The latter two validated digests prove serving parity was complete
before the evaluation-controller incident. They do not turn the partial run
into a score.

The exact orphan cleanup is frozen in
[`2026-09-01-qwen36-post-sft-v1-orphan-cleanup.json`](evidence/webexploitbench/2026-09-01-qwen36-post-sft-v1-orphan-cleanup.json).
It proves the run-owned container and network were removed, the shared agent
network and peer ExploitGym workload were preserved, and neither scores nor
traces were inspected. The preview-only successor is documented in
[`QWEN36_WEBEXPLOIT_RECOVERY.md`](QWEN36_WEBEXPLOIT_RECOVERY.md).

The curriculum consumer gates from PRs #75 and #79 are merged. They require
canonical exact environment-version UUIDs on source and all created members,
reject mutable-parent fallback, and keep task-group creation separate from a
later paid job. Shared Theseus PR
[#28555](https://github.com/fleet-ai/theseus/pull/28555) is still an open draft;
even after merge, a deployed behavioral probe must prove the no-fallback
contract before any creation action.

## Owned PR disposition, #65–#80

Repository state was read from GitHub at 2026-09-01 13:41 UTC.

| PR | Disposition | Durable contribution |
|---:|---|---|
| [#65](https://github.com/fleet-ai/cyber-post-train/pull/65) | Merged | Frozen Qwen3.8 model/serving locks and matched external-evaluation controls. |
| [#66](https://github.com/fleet-ai/cyber-post-train/pull/66) | Merged | Qwen3.8 Fleet reward-calibration protocol and fail-closed successor. |
| [#67](https://github.com/fleet-ai/cyber-post-train/pull/67) | Merged | Initial Qwen3.8 training qualification and one-step gate design. |
| [#68](https://github.com/fleet-ai/cyber-post-train/pull/68) | Merged | Read-only, sealed-safe WebExploit lifecycle monitor. |
| [#69](https://github.com/fleet-ai/cyber-post-train/pull/69) | Merged | Train-only prompt-curriculum dry run. |
| [#70](https://github.com/fleet-ai/cyber-post-train/pull/70) | Merged | Fail-closed reconciliation of interrupted self-hosted evaluations. |
| [#71](https://github.com/fleet-ai/cyber-post-train/pull/71) | Merged | Corrected live base-artifact evidence revalidation. |
| [#72](https://github.com/fleet-ai/cyber-post-train/pull/72) | Merged | Single-use Qwen3.8 Web recovery with durable supervision receipts. |
| [#73](https://github.com/fleet-ai/cyber-post-train/pull/73) | Merged | Pre-paid-launch CAGE interpreter/module/repository/output-root identity gate. |
| [#74](https://github.com/fleet-ai/cyber-post-train/pull/74) | Merged | Sanitized Qwen3.6 curriculum review plan. |
| [#75](https://github.com/fleet-ai/cyber-post-train/pull/75) | Merged | GET-only curriculum creation preflight and exhaustive duplicate detection. |
| [#76](https://github.com/fleet-ai/cyber-post-train/pull/76) | Merged | Exact Qwen3.8 train-only corpus and model-staging integrity gates. |
| [#77](https://github.com/fleet-ai/cyber-post-train/pull/77) | Merged | Qwen3.8 storage-topology and trainer-compatibility evidence. |
| [#78](https://github.com/fleet-ai/cyber-post-train/pull/78) | Open draft | Dry-run-only Qwen3.8 model rematerialization/staging bridge; no execution approval. |
| [#79](https://github.com/fleet-ai/cyber-post-train/pull/79) | Merged | Exact curriculum source/member environment-version identity enforcement. |
| [#80](https://github.com/fleet-ai/cyber-post-train/pull/80) | Merged | Blocked, preview-only Qwen3.6 Web successor bound to incident and cleanup evidence. |

## Operator decision checklist

1. **Protect the active ExploitGym run.** Observe only aggregate lifecycle and
   UID-bound health until its terminal receipt and downstream acceptance exist;
   do not inspect sealed outputs or alter the Job.
2. **Do not launch Qwen3.6 Web v2.** First rotate the exposed credential,
   record sanitized rotation evidence, obtain explicit successor approval, and
   freshly prove absent run/claim roots plus exact runtime identity.
3. **Do not launch Qwen3.8 Web v3.** Preserve v2; require explicit approval and
   rerun all create-once/runtime-root preflights immediately before launch.
4. **Do not stage or register Qwen3.8 yet.** Require explicit 55.6 GB transfer
   approval, a positive authoritative reward receipt with valid cleanup, the
   exact training-volume destination, and a reviewed create-once plan.
5. **Do not start Qwen3.8 SFT or RL.** Stage and verify the exact model and
   train-only corpus, register the catalog row, bind an exact Ready trainer,
   pass an unpaid server preview, then run only the one-step capability gate.
6. **Do not create the Qwen3.6 curriculum task groups.** Wait for shared PR
   #28555 to merge and deploy, prove exact-version/no-parent-fallback behavior,
   then rerun the GET-only preflight and duplicate scan.
7. **Keep classifications separate.** Serving parity and plumbing are
   operational gates; interrupted runs are not zeros; only accepted terminal
   receipts support capability or learning claims.
