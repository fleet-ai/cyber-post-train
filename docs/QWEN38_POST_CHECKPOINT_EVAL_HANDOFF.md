# Qwen3.8 post-checkpoint evaluation handoff

The [handoff template](../configs/evaluation/qwen38-post-checkpoint-eval-handoff-v1.template.json)
records the two evaluations that follow an accepted Qwen3.8-27B checkpoint. It
does not launch an evaluation and it authorizes no paid work.

## What happens next

1. **Fleet development tasks choose the model.** Compare the exact base model
   and candidate on the 20 task families and four fixed attempts already bound
   by either split A or split B. The training plan must have chosen that split
   before anyone sees results. The only selection signal is the paired change
   in full-task success on the same task version and attempt seed.
2. **WebExploitBench reports external performance.** Run a new matched base and
   candidate pair on the pinned benchmark. Its results, validity information,
   and failure analysis stay sealed until the Fleet development selection
   decision is frozen. They never choose a learning rate, checkpoint, stopping
   point, retry, or tie-break.

Training loss is a fitting diagnostic only. Fleet final-test outcomes and
WebExploitBench are not model-selection signals.

## When a checkpoint is eligible

Every item below must be backed by an exact, accepted receipt:

1. Training ended normally with finite real updates and an immutable plan.
2. The stable checkpoint was fully hashed and reloaded on its original ranks
   without taking another optimizer step.
3. The model was exported deterministically to BF16, including exact weight,
   tokenizer, and chat-template manifests, and the export reloaded successfully.
4. The export was staged once and both base and candidate serving registrations
   identify immutable model bytes and runtime images.
5. A fresh probe through the evaluator's real route proves base/candidate parity
   for tokenizer, template, precision, quantization, engine arguments, parsers,
   context limits, and harness images. Only the model weights may differ.
6. The training arm proves that its agent harness is OpenCode 1.18.27 with the
   pinned release digest. An unknown harness or Qwen Code arm needs a separate
   reviewed evaluation protocol; it cannot be relabeled as compatible.
7. The training plan already binds exactly one of the two frozen Fleet
   development protocols and has no task-family overlap with its training data.
8. Fresh preflight confirms the Fleet team, exact task/runtime/verifier tuples,
   serving route, images, and unused result and claim identities.

The template deliberately keeps every candidate-specific field `null`. It is a
checklist, not a wildcard. A candidate-specific private child must fill each
field from accepted evidence and pass the existing validators.

## What “same harness” means

Both surfaces use the same pinned OpenCode 1.18.27 binary and release digest,
OpenAI-compatible reasoning and tool transport, 262,144-token context, 32,768
maximum output tokens, native compaction/continuation, and disabled built-in
tools.

The task-specific tools correctly differ. Fleet tasks use `bash` and
`submit_report`; WebExploitBench uses `web_bash` and `web_submit_findings`
because it has a different reporting and grading contract. Pretending those
tool schemas are identical would change the benchmark. Within each surface,
the base and candidate must use identical harness bytes and settings.

## Fleet development evaluation

Use the exact split named by the training plan:

- [split A protocol](../configs/evaluation/qwen38-blackbox-fleet-dev-a-outcome-protocol-v2.json),
  [task set](../configs/evaluation/qwen38-blackbox-fleet-dev-a-task-set-v1.json),
  and [base control](../configs/evaluation/qwen38-blackbox-fleet-dev-a-base-control-v2.json)
- [split B protocol](../configs/evaluation/qwen38-blackbox-fleet-dev-b-outcome-protocol-v2.json),
  [task set](../configs/evaluation/qwen38-blackbox-fleet-dev-b-task-set-v1.json),
  and [base control](../configs/evaluation/qwen38-blackbox-fleet-dev-b-base-control-v2.json)

The existing `cyber-post-train eval` workflow prepares and preflights the exact
task and serving bindings. Its controller belongs on an authorized CPU worker;
do not allocate a GPU merely to run the controller. Any separate new or changed
cluster executable must pass development qualification first and may use at
most `c1` / derived `q1` (priority 10,000), never `c0` / `q0`.

Keep outcomes in private storage and outside W&B. All 80 base/candidate
task-and-seed pairs must be authoritative and valid. A genuine zero remains a
zero; an infrastructure failure makes the comparison incomplete and never
becomes a model failure.

## WebExploitBench report

Use the [post-training parent](../configs/evaluation/qwen38-post-training-webexploitbench-tensorlake-parent-v1.json)
and [paired-plan template](../configs/evaluation/qwen38-post-training-webexploitbench-tensorlake-paired-plan-v1.template.json).
The pair must use one qualified Tensorlake snapshot with separate base and
candidate projects and prove that only the served model changes. A
candidate-only run is not a base-versus-candidate comparison.

Tensorlake scored sandboxes have no Kubernetes priority field. Any separately
authorized cluster helper remains capped at `c1` / `q1`.

The current paired template remains blocked. Before dispatch, the repository
still needs independently validated two-arm snapshot receipts, project parity,
a fresh complete duplicate inventory, both controller previews, and a paired
controller that enforces and records the counterbalanced schedule. Unit tests,
a Ready endpoint, or a filled template do not satisfy these gates.

After execution, accept results only from complete authoritative benchmark
records and confirm every exact owned sandbox was terminated. Process exit by
itself proves neither a valid result nor resource cleanup.

## Offline verification

The focused contract test reopens every referenced file, validates the Fleet
and WebExploitBench parents with their existing validators, checks the priority
ceiling and selection firewall, and confirms this handoff is still inert:

```sh
uv run pytest -q tests/test_qwen38_post_checkpoint_eval_handoff.py
```
