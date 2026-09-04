# Repository design: one experiment interface, strict backend adapters

## Purpose

`cyber-post-train` should make a controlled cyber experiment easy to describe,
preview, launch, monitor, and reproduce. It should not hide scientific choices or
turn historical one-off scripts into implicit defaults.

The public interface should support four common actions:

1. evaluate an exact model on Fleet cyber tasks;
2. evaluate an exact model on WebExploitBench;
3. evaluate an exact model on ExploitGym; and
4. train an exact model with Fleet SFT or verifiable-reward RL.

Every action must be preview-first, create-once, and evidence-producing. The same
interface should work when a model is served by an existing endpoint or by a
dedicated immutable serving job, while recording those as different treatment
blocks.

## What belongs here

- Typed, versioned experiment specifications.
- Exact model, tokenizer, serving, harness, task, verifier, and dataset locks.
- Backend adapters for Fleet, WebExploitBench, ExploitGym, and the Training Jobs API.
- Preview, duplicate detection, launch, monitoring, acceptance, and receipt code.
- Reusable cluster manifests and immutable image build inputs.
- Sanitized evidence, protocol documents, and regression tests.
- Operator skills that explain decisions which cannot be enforced completely in code.

## What does not belong here

- Credentials or copied secret values.
- External benchmark prompts, traces, answers, applications, flags, or exploit hints.
- Mutable `current` pointers presented as reproducible model or task identity.
- Raw local outputs, temporary worktrees, downloaded checkpoints, or cluster caches.
- A second implementation of Fleet task authoring or Pipeline Lanes.
- Model-specific behavior embedded in a supposedly generic module.
- Historical failed manifests in the primary happy-path interface.

Historical evidence remains immutable. Superseded operational manifests should move
to a clearly indexed archive only after references and tests prove that no live path
imports them.

## Target user interface

The stable command should be `cyber-post-train` with five top-level groups:

```text
cyber-post-train catalog       # models, harnesses, benchmarks, trainers
cyber-post-train experiment    # validate and compile a frozen experiment
cyber-post-train eval          # preview/launch/status/accept evaluations
cyber-post-train train         # preview/launch/status/export training
cyber-post-train evidence      # reconcile and verify immutable receipts
```

Examples of the intended workflow:

```bash
uv run cyber-post-train experiment validate configs/experiments/qwen38-fleet-p4.yaml
uv run cyber-post-train eval preview configs/experiments/qwen38-fleet-p4.yaml
uv run cyber-post-train eval launch configs/experiments/qwen38-fleet-p4.yaml --execute
uv run cyber-post-train eval status --run-receipt output/qwen38-fleet-p4/LAUNCHED.json
```

`validate` is offline. `preview` may use authenticated read-only APIs. `launch` is a
no-op without `--execute`. `status` never mutates. `accept` verifies terminal state,
authoritative grading, cleanup, artifact digests, and the exact launch identity.

## Specification model

Do not create one unstructured mega-config. Compile an experiment from small immutable
components:

- **model lock**: repository, revision, weight manifest, tokenizer, chat template;
- **serving profile**: endpoint or dedicated image, engine arguments, precision,
  quantization, context window, resource shape;
- **harness profile**: harness name/version/image, system prompt, ordered tool schema,
  step and token semantics;
- **dataset lock**: source, split, exact task/version manifest, lineage constraints;
- **protocol**: sampling, pass@k, time/step/token budgets, retry policy, sealing policy;
- **execution profile**: backend, queue, priority, concurrency, output root, owner;
- **training profile**: objective, trainer/image, optimizer, checkpoint and reward gates.

The compiled plan resolves all references to content digests and validates cross-field
identity. A launch consumes only that compiled plan, never the mutable source files.

## Adapter boundary

The common lifecycle is:

```text
validate -> compile -> preview -> claim -> launch -> monitor -> accept
```

Each backend adapter implements that lifecycle without changing its scientific
meaning:

| Adapter | Transport | Required authority |
|---|---|---|
| `fleet` | Fleet Jobs/session APIs plus controlled agent runtime | exact task-version, model route, harness and verifier identities |
| `webexploitbench` | pinned CAGE + Qwen Code runtime | exact benchmark snapshot, CAGE/harness images and sealed result root |
| `exploitgym` | pinned control image + dynamic graders | exact task images, graders, arm order and single-use root |
| `fleet-sft` | Training Jobs API | exact corpus/model/trainer/runtime and optimizer-step evidence |
| `fleet-rl` | Training Jobs API | all SFT controls plus task tool allowlists and authoritative reward IDs |

Adapters may expose backend-specific options, but must return the same receipt envelope
and lifecycle states. They must not silently translate an unsupported option.

## Harness and serving rules

- A harness is part of the treatment, not a cosmetic CLI choice.
- Ordered tools, system prompt, step semantics, context policy, timeout, and retry rules
  are immutable harness fields.
- Hosted and dedicated serving are explicit treatment blocks. They may run concurrently
  only on disjoint complete-task partitions.
- A model name is insufficient identity. Exact revision, weight/tokenizer/template
  manifests, serving image, and live-parity receipt are required where the platform
  exposes them.
- Throughput controllers may change concurrency only from score-blind health metrics.
  They may never duplicate an accepted, active, or fully persisted attempt.

## Repository layout

The target layout is:

```text
cyber_post_train/       stable CLI, component schemas, compiler, receipts
adapters/               Fleet, WebExploitBench, ExploitGym, SFT, RL adapters
configs/components/     reusable immutable model/harness/serving/data profiles
configs/experiments/    small user-authored experiment compositions
evals/                  evaluator-specific implementation details
training/               trainer-specific implementation details
cluster/                reusable build/serve/stage manifests, not run history
docs/evidence/          sanitized immutable evidence
archive/                indexed historical manifests after migration proof
skills/                 focused decision guidance only
tests/                  unit, rendered-boundary, duplicate and receipt tests
```

Existing `evals/` and `training/` code should move behind adapters incrementally. Avoid
a flag-day rewrite.

## Required invariants

The compiler and adapters must fail closed on:

- mutable or missing model/task/image revisions;
- a model/harness/serving identity disagreement;
- unsupported model-harness or trainer-model compatibility;
- external benchmark data referenced by training, reward, retrieval, or selection;
- duplicate task-attempt cells across active or accepted runs;
- overlapping hosted/dedicated partitions or split pass@k tasks;
- output roots or create-once claims that already exist;
- missing authoritative verifier IDs or cleanup evidence;
- context, step, token, timeout, or tool-schema drift;
- more sessions or GPUs than the reviewed plan;
- credentials appearing in configs, commands, receipts, or logs.

## Migration plan

1. **Inventory and facade.** Add the stable top-level CLI, a machine-readable catalog,
   and a doctor command. Keep existing implementations unchanged behind adapters.
2. **Component schemas.** Introduce versioned component locks and a deterministic
   experiment compiler with cross-field tests.
3. **Evaluation adapters.** Move Fleet, WebExploitBench, and ExploitGym onto the common
   lifecycle and receipt envelope. Preserve benchmark containment.
4. **Training adapters.** Move SFT and RL preview/launch/status/export behind the same
   component locks. Keep reward-acquisition and checkpoint gates explicit.
5. **Archive.** Trace imports and documentation references, then move superseded
   manifests and model-specific status files to an indexed archive. Never delete
   immutable evidence.
6. **Examples and skills.** Ship one minimal example per adapter and a focused operator
   skill. Validate skills and examples in CI.

Each phase should be a narrow reviewed change. Live jobs must continue using their
frozen source commits until their terminal receipts are accepted.
