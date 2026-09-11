# Qwen3.8 blackbox SFT study

This is the execution contract for the next Qwen3.8-27B teacher- and self-SFT
study. It does not treat a published task, a tokenizer-ready transcript, or a
numeric model outcome as proof that a task is runnable.

## Eligible task universe

The frozen input is
[`qwen-blackbox-eligible-v1.json`](../configs/data/qwen-blackbox-eligible-v1.json).
It contains **89 exact task versions** with both:

1. a digest-valid completed execution receipt proving the exact environment,
   verifier, scoring intent, ingest and cleanup path; and
2. a current exact-version read proving that the same starting data, runtime
   seed, environment version and verifier version can still be resolved.

The other 71 members of the historical 160-version roster remain excluded. Nine
of them have clean historical executions but no longer expose a reproducible
starting-data binding. The allowlist is task validity evidence, not SFT-source
eligibility and not authority to launch a job.

The study inventory joins those 89 versions to each task's exact immutable atom
source metadata. It has five applications, six environment labels, reviewed
vulnerability class/category labels and easy/medium/hard difficulty labels. The
splitter uses the exact atom locator as the family boundary; it does not infer a
family from model success or a task-name hash.

## Outcome-blind representative splits

The common final-test lock contains 10 task families. Each of split A and split B
contains 59 train, 20 Fleet development and the same 10 final-test tasks. The two
development sets overlap on only 5/20 tasks; the two train sets overlap on 44/59.
All choices use taxonomy only—no model outcome, transcript, loss or score enters
split selection.

For both development variants, the maximum absolute group-share gap from the
89-task population is below 0.03 for application and environment, below 0.06 for
vulnerability class/category, and below 0.04 for difficulty. The exact manifests
are:

- [`qwen-blackbox-study-final-test-v1.json`](../configs/data/qwen-blackbox-study-final-test-v1.json)
- [`qwen-blackbox-study-split-a-v1.json`](../configs/data/qwen-blackbox-study-split-a-v1.json)
- [`qwen-blackbox-study-split-b-v1.json`](../configs/data/qwen-blackbox-study-split-b-v1.json)

The 20-task Fleet development set is the per-arm model-selection signal. The
10-task final set stays sealed until a configuration is selected. WebExploitBench
is run after each completed arm as requested, but it is never used to rank or
retune arms; repeated inspection would otherwise turn it into another development
set.

## Source policy

Teacher and self sources are separate treatments. Each source must be a verified
successful session on an exact train task version, with acceptance, trace and
normalized-record digests. Held-out identities are rejected before their message
payloads are accessed.

The dense compiler supervises every compatible visible assistant response once.
Earlier five-ending-window corpora overweighted late submission behavior. The new
coverage gate requires at least one completed non-submission tool round and one
non-submission tool or decision response, enforces per-family episode/token caps,
and blocks a corpus if `submit_report` exceeds 50% of either supervised responses
or supervised tokens. These are structural coverage checks, not a claim that an
individual shell command was useful.

Teacher and self arms are not interpreted as a matched comparison when their task
or token coverage differs. Each source treatment gets its own baseline and HPO
curve.

## Training and evaluation protocol

- Exact initialization: `Qwen/Qwen3.8-27B` at
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- W&B logs training loss and global optimizer step for fitting diagnostics. It
  receives no prompts, messages, traces, flags, token IDs or scores.
- Teacher-reference validation CE is not computed or used for checkpoint choice.
- Checkpoints are written at the configured interval and retained by recency.
- Arm selection uses fresh Fleet development outcomes only. Ties are retained for
  a fresh confirmation rather than broken with training loss or teacher CE.
- The selected configuration is evaluated on the common Fleet final set. Each
  completed checkpoint also runs the requested WebExploitBench protocol through
  the separately qualified Tensorlake adapter, but those results do not tune HPO.

The first wave is explicit, not an accidental Cartesian product: broad learning
rates and epochs are varied within teacher and self treatments, then the promising
region is narrowed in a later wave. Split A/B is a separate robustness factor.
Every arm has unique output and W&B identities and binds exact corpus, source,
split, model and evaluation digests.

## Operational gates

New or changed training, serving and Tensorlake paths qualify on the dev cluster
first. Production submission is allowed only after a clean dev receipt, checkpoint
reload and exact duplicate checks. At most eight nodes from this study may be
active at once; the four pre-existing inference endpoints are outside that cap.
Queued work with no allocation consumes zero nodes. Failed, ambiguous or stalled
work is never blindly repeated, and a broken owned allocation is released before
off-node repair.

No paid study arm had been submitted when these manifests were frozen. Source
coverage, train-only runtime qualification, dev checkpoint serving and the real
Tensorlake lifecycle remain explicit pre-production gates.
