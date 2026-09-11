# Qwen post-SFT evaluation readiness

Audit time: **2026-09-11T20:38:11Z**. Scope: source, documentation and isolated
synthetic tests only. No model request, benchmark attempt, credential read,
cluster mutation or Tensorlake API mutation was performed. No private prompts,
traces, reports, answers or scores were inspected. This document is not a launch
receipt, a live resource census or permission to replay an existing campaign.

## Decision

The intended pipeline is identifiable, but it is **not an already-qualified,
one-command checkpoint-to-benchmark path**. The generic Qwen checkpoint tools and
Fleet evaluation CLI are present. The Tensorlake/OpenCode path is in an open PR,
and a newly selected checkpoint still needs its own export, serving and matched
evaluation bindings. Do not reuse the old fixed-checkpoint registration wrapper
or historical Qwen Code benchmark campaign as the new treatment.

### Where the Tensorlake instructions actually are

| Source | Exact state verified in this audit |
| --- | --- |
| Freshly fetched `origin/main` | `2b775b827afc87daf01170fd031c5834337aa9e5`; no Tensorlake matches in the relevant tracked README/docs/evals/training/scripts/skills paths |
| [PR 209: Add Tensorlake WebExploitBench OpenCode runner](https://github.com/fleet-ai/cyber-post-train/pull/209) | **OPEN**, not draft, not merged; head `dc650af596eb693d75a95324eeb1066b8933ff15`, base equal to the main SHA above |
| Actual instructions | [`evals/webexploitbench/tensorlake/README.md` at that PR head](https://github.com/fleet-ai/cyber-post-train/blob/dc650af596eb693d75a95324eeb1066b8933ff15/evals/webexploitbench/tensorlake/README.md) |
| PR software check | GitHub [Repository tests](https://github.com/fleet-ai/cyber-post-train/actions/runs/34637626653/job/103389283007) succeeded at 19:14:41Z; local focused checks below also pass |

The current worktree additionally contains the September 11 dev-first policy at
`6014cc3ee5e68add317ef8ebe844042de0213db9`; those policy additions are not yet in
the fetched main above. Follow the current approved policy, but do not describe
either those additions or PR 209 as already merged. Source status is time-bound;
recheck the exact deployed/selected revision before execution.

## The supported sequence and its evidence gates

```text
Frozen Fleet dev selection
  -> complete native checkpoint + seal
  -> create-once BF16 export + exact reload
  -> new immutable matched serving registration + dev qualification
  -> frozen base/post protocol
       |-- new Tensorlake WebExploitBench campaign
       `-- new untouched Fleet test campaign
  -> independent terminal acceptance + private archive + resource release
```

### 1. Select and seal the checkpoint

Select the training arm and optimizer step using the predeclared **Fleet dev**
metric only. Never use WebExploitBench outcomes to choose a checkpoint, tune
hyperparameters, repair prompts or design rewards. Declining imitation loss is
not itself evidence of improved task success. Multiple rollout seeds are not
independent training replicates.

For a trusted owned run, use the maintained public CLI:

```bash
cyber-post-train checkpoint-seal <prepared-run> <step> --output <new-manifest>
cyber-post-train checkpoint-export <manifest> --sha256 <manifest-file-sha256> --output <new-export>
cyber-post-train checkpoint-check <export>/EXPORT.json --sha256 <export-receipt-file-sha256> --output <new-check>
```

These are a recipe, **not commands executed by this audit**. Bind the source
run/plan, completed optimizer step, every rank's model/optimizer files,
sampler/cursor state, exact base/tokenizer/chat template and native runtime.
Source checkpoints must not still be changing. The seal is CPU-only and does not
prove successful distributed restore. The exporter reconstructs the Qwen BF16
layout and approved exact-base runtime tensors/files; merely renaming FP32
weights is invalid. A weights-only HF export cannot resume training.

`EXPORT.json` and the CPU/meta check do not prove a real GPU reload. A separately
authorized bounded dev `checkpoint-check --gpu` exercises complete BF16 loading,
finite synthetic forward/generation and source stability. Native zero-step
optimizer/sampler restore is a separate evidence gate. Neither check qualifies
the production serving engine, tool parser or full evaluation lifecycle. See
[artifact handoff](POST_SFT_EVALUATION.md) and [training commands](TRAINING.md).

### 2. Stage and qualify a new serving identity

Use Fleet's persistent **InferenceModel control plane**, not an indefinitely
running training Job. The inference cache and training SFS are distinct storage
contracts; create-once staging must rehash the exact accepted export and its
sidecars at the serving destination. Do not assume a path exists on another
cluster or namespace.

Freeze the new model registration and the matched base registration. Prove
registration completion, live InferenceModel UID, Deployment/Pod identities,
actual image digest and mounted model bytes. Pin tokenizer/template, engine
version/arguments, BF16/quantization, KV-cache dtype, context size, TP/DP layout,
reasoning/tool parser, speculative-decoding treatment and gateway model ID.
Run non-scored chat, tool-format, context/continuation, concurrency and cleanup
checks through the **exact route used by the evaluator**. Ready alone is not
proof of loaded weights or successful tool execution.

**Current integration gap:** [`training/register_post_sft.py`](../training/register_post_sft.py)
is hardcoded to `ft-run-574bd7b3-step-318`, its historical source path and a fixed
older SGLang image. It intentionally rejects other checkpoint identities. It is
not a generic registration command for this Qwen sweep. The historical
[serving runbook](../evals/webexploitbench/serving/README.md) explicitly says not
to replay its launchers. A newly reviewed control-plane payload and its
supported staging/registration procedure are still required; no new payload was
invented here.

Match a fresh exact Qwen base and the selected post-SFT checkpoint. An opaque
shared alias can support a descriptive serving block, not an exact causal
base/post comparison without matching weight/runtime proof. Keep hosted and
dedicated blocks explicit and partition complete tasks before execution.

### 3. Freeze the WebExploitBench Tensorlake protocol

PR 209 implements this topology:

- A small persistent **operator-host controller** owns a local, private,
  single-writer claim journal.
- Each Tensorlake sandbox hosts Docker, the pinned CAGE evaluator, the assigned
  benchmark application containers and the OpenCode agent containers.
- The sandbox calls an existing OpenAI-compatible inference route. Tensorlake
  does not provision or register the cluster model endpoint.
- Private results stay under the sandbox's declared persistent result root;
  controller claims stay local. This is **not automatically a cluster-CPU/SFS or
  PostgreSQL deployment**. Backup, immutable archival and release must be planned.

The inherited source protocol documents CAGE
`09a191c565230cebb8255899d622d23c7ddeff33`, its content-addressed compatibility
patch, gated WebExploitBench revision
`7f97d87fa8ab728260c0ba9b09b9c8f00bb82ad5`, Level 0, and linux/amd64. Those are
documented reference pins, **not a verified new Tensorlake snapshot inventory**.
The new private qualification receipt must attest the actual authorized gated
pack hashes, exact task set, benchmark/verifier/judge definitions, prompt hashes,
target images, CAGE source/patch and evaluator image. No dataset access was needed
for this audit. See the [benchmark guide](../evals/webexploitbench/README.md).

The PR pins OpenCode **1.18.27** and its release hash. Its MCP tools are
`web_bash` and `web_submit_findings`, preserving benchmark report/PoC submission.
Fleet uses `fleet_bash` and `fleet_submit_report` for a different task contract.
Do not rename benchmark tools to Fleet tools to claim superficial parity;
require identical tools and semantics **between base and post within each
benchmark**. Submission syntax validation must not expose grader feedback or
repair the model's answer.

Before any scored sandbox is created, qualify a clean filesystem snapshot with
the exact Docker/Compose runtime, cached digest-pinned evaluator/target/agent/
netproxy images, one exact task per project and non-inference
`cage --preflight-only` success. Require no prior `.cage_runs`, output roots,
claims or running application containers, and a quiescent Docker state before
snapshotting. Preserve its immutable snapshot ID and qualification receipt hash.

Seal a new `webexploitbench_tensorlake_launch_v1` plan using
`python -m evals.webexploitbench.tensorlake.seal_plan`, then validate with
`python -m evals.webexploitbench.tensorlake.controller --plan <plan> --validate-only`.
Both modules require the selected PR revision; main does not currently provide
them. Bind campaign/run/partition identities, project hashes, runner bytes,
complete protocol bytes/self-digest, snapshot and qualification receipt,
resources, timeouts, attempt budgets and any same-ID resume policy. Freeze the
controller's own source/package revision too: the plan's runner hash alone does
not pin all controller code.

The load-plan validator verifies operational bindings, but does **not** itself
require every scientific weight/tokenizer/judge/task/decoding field in an
arbitrary protocol. The focused tests deliberately accept a synthetic protocol
without those fields. Operator review of the complete scientific protocol is
therefore mandatory; a valid plan digest is not automatic scientific acceptance.
The sample `models.yml` contains shared aliases and `max_retries: 2`, not a
checkpoint-specific live identity proof. Freeze and qualify both student/judge
selection and request-retry semantics; registry presence alone does not select
the intended CAGE judge.

### 4. Keep the Fleet test campaign separate and genuinely held out

The maintained [`cyber-post-train eval` path](../evals/fleet/README.md) provides
`prepare`, read-only `preflight`, `init`, bounded `run` workers and score-blind
`status`. `init` and `run` are mutations and were **not** executed. Use a fresh
private PostgreSQL database and immutable prepared directory, not a reset of the
historical rollout campaign. A CPU Docker worker drives the task/harness; GPU
serving remains a separately qualified resource.

Freeze every `task_key`, task-version UUID, environment key/version/UUID,
starting-data key/version, verifier identity and harness/tool digest. Missing
starting-data bindings are a hold, not permission to borrow the latest version.
Preflight proves Fleet-team account, exact task/runtime metadata, route metadata,
required Docker images and actual runtime UID/HOME/startup behavior. Bind model
weights/route, decoding and seeds, pass@k, turns, context/output, deadlines,
request/tool budgets, no-retry treatment and cleanup in the prepared plan.

An evaluation manifest must prove task-family non-overlap with **all training
data used by every compared checkpoint**, including earlier teacher/self runs
and retained versions/sessions. A newly declared split does not make an already
exposed checkpoint held out. The approved task-family split may share
applications: describe it as **task-held-out, not application-held-out**. Dev is
for selection; keep the final Fleet test set untouched and
`training_data_eligible: false`. Allocate every model/task's attempts to one
predeclared serving block and report block-specific results before pooling.

## Concrete gaps to close before launch

| Gate | Finding / required next evidence |
| --- | --- |
| Executable revision | Tensorlake source is open PR 209, not main. Select/review that exact revision or its reviewed successor; do not assume `git pull main` installs it. |
| Selected artifact | No new sweep checkpoint selection, complete seal, export/check receipts or GPU-reload identity was supplied or audited here. Existing qualification runs prove mechanisms, not this artifact. |
| Serving | New create-once staging and registration payload/UIDs plus matched live base/post routes are missing from this audit. The historical fixed-checkpoint helper cannot supply them. |
| Context policy | Tensorlake renderer uses `reserved=20000`; current Fleet v2 uses `reserved=32768+20000=52768` for the documented 32,768-token output limit. With context 262,144, native thresholds are respectively 209,376 and 176,608 tokens. This is a real behavioral difference, not merely a name. Qualify exact-binary long-response/compaction continuation; a change needs a new adapter image/protocol, never an in-flight patch. |
| Snapshot and protocol | Need a real clean snapshot ID, reviewed non-scored qualification, all task/project/image/model/judge pins and endpoint access. PR examples and unit fixtures are not those receipts. |
| Outcome acceptance | Process exit is not benchmark acceptance. Tensorlake runner's error trap records `pre_scoring_gate_rejected` or `scored_controller_failed` and exits zero. A monitor must read truthful lifecycle status and preserve faults; `scored_controller_exited` also does not prove authoritative grade, complete results or cleanup. Never translate these states into success/zero capability. |
| Untouched Fleet comparison | Need an explicit task-family lineage intersection check against each chosen checkpoint's entire training history and a frozen final-test/serving-block plan. No private task contents or scores are needed for that check. |
| Capacity and release | Need fresh exact ownership/admission/serving-consumer census, Tensorlake quota and resource budget, endpoint load bound, access expiry, private archival and cleanup/release procedure. No live capacity was claimed by this audit. |

The context difference is acceptable as two separately named benchmark
treatments if each base/post pair is internally matched; it does not justify
claiming identical Fleet/Tensorlake harness behavior. The PR's tests do not
exercise real OpenCode long-context inference, CAGE grading, Docker isolation,
Tensorlake suspension or the complete selected-model serving path.

## Duplicate, resource and secrecy controls

The Tensorlake controller takes a single-writer file lock, uses create-once
fsynced claims before sandbox creation and scored-process dispatch, checks exact
names/IDs and resources, and holds uncertain operations for reconciliation.
Never repeat an ambiguous create or scored start. Same-ID resume requires an
explicit protocol permission, a named saved nonterminal lifecycle digest and
the exact suspended owned sandbox; it resumes the existing process and must not
launch CAGE again. Missing/unreadable process state conservatively consumes
attempt slots. Preserve claims, receipts and original attempts.

The example 15 tasks × pass@4 is **60 attempts per student arm**, hence 120 for
a matched two-arm study. Runtime hard ceilings are 50 sandboxes and 48 active
attempt slots, not permission or an inference-throughput recommendation. Each
active partition consumes its declared pass@k slots; at k=4 a 48-slot ceiling
allows at most 12 active scored partitions. The README's 8 CPU / 32,768 MB RAM /
102,400 MB disk per sandbox is an example shape, not measured need. Aggregate
resources must include paused sandboxes, controller durability, student/judge
requests and archive storage. Benchmark rollout seeds do not increase the
number of independent tasks.

The user's **eight actively allocated owned-node** cap still covers cluster
training, admitted startup, dedicated serving and conservatively CPU allocations;
unallocated queue entries do not count. Tensorlake resource quotas and costs are
a separate explicit bound, not an inferred exemption. Count all owned endpoints
and their real consumers before adding capacity. Do not alter peers or the
existing rollout campaign. Follow [resource lifecycle](GPU_RESOURCE_LIFECYCLE.md).

Follow the current [dev-first cluster policy](CLUSTER_ALERTS_AND_INFERENCE_SERVING.md)
for changed serving/evaluation executable behavior. A new dev GPU/serving test
must check its target-local mounts/Secrets, actual image and request/tool/cleanup
path before production. Tensorlake also needs its own non-scored snapshot and
end-to-end synthetic qualification; CPU unit tests are not substitutes for
either. Choosing Tensorlake is an execution-topology decision, not a way to
suppress cluster alerts. Report real failures, preserve evidence and release
only exact owned resources under the authorized bounds.

Use credentials only through approved secret handling. Keep private benchmark
artifacts, lifecycle details beyond the allowed projection and graded outputs
out of training/W&B/code/reporting. The controller exposes an eight-field
score-blind lifecycle projection; independent accepted-result verification and
authorized aggregate reporting must remain separate. Do not download raw logs
to diagnose progress. A final outcome needs normal terminal harness events,
exact authoritative grading, complete private results and environment cleanup;
interrupted, malformed, output-limited or unknown outcomes remain held rather
than fabricated model failures.

## Checks performed and limits

- Fetched main twice and read PR metadata; the exact main/head/status above were
  unchanged on the final readback. No branch merge/rebase was performed.
- Inspected the current artifact/Fleet/cluster guides and source, plus the exact
  PR controller, sealer, runner, adapter and focused tests. No training runtime
  files or live workloads were modified.
- Extracted only the PR's relevant source and synthetic test files into an
  isolated temporary directory. Under Python 3.12, its focused controller and
  adapter tests passed: **18 passed, zero skipped**. Initial setup attempts lacked
  pytest/a required package file; neither ran workload code. The final run used
  the complete relevant package files and explicit test dependencies.
- `bash -n` passed for both shell entrypoints; `node --check` passed for all three
  adapter JavaScript modules. These are syntax checks, not image/runtime proof.
- No Docker build, real MCP server, model inference, Tensorlake operation,
  benchmark grading, GPU test or production route qualification was performed.

No new eval script/config was added: without the missing immutable checkpoint,
route, snapshot and task bindings, a runnable-looking config would be misleading.
The next deliverable is a reviewed exact plan and qualification receipts, not a
duplicate campaign or a replay of historical launch commands.
