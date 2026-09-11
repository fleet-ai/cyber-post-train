# Scientific controls

## Comparison

Freeze the question, exact base model, training data, recipe, checkpoint-selection
rule and evaluation protocol before unsealing external results. Typical arms are
base, SFT-only, RL-from-base and SFT→RL; choose only the arms needed for the
question. Multiple rollout seeds are not independent training replicates.

Teacher SFT imitates verified-success teacher actions with next-token
cross-entropy. Self-SFT uses verified-success student actions under the same
quality gates. RL obtains fresh isolated episodes and authoritative verifier
rewards. Do not confuse declining imitation loss with improved task success.

## Data

Split before windowing. Every version and session of a task family belongs to
one split. The default generic splitter is application×task-family grouped:
it can share applications across splits and is NOT application-disjoint.
For an application-held-out study, freeze an explicit app-disjoint manifest.

Keep dev/test out of SFT, preference pairs and RL prompt sets. Do not silently
rewrite existing frozen splits after a split-policy fix. Choose hyperparameters
and checkpoints on Fleet dev only; preserve a final untouched test set.

WebExploitBench, ExploitGym and other external benchmarks are evaluation-only.
Their prompts, traces, applications, answers, outcomes and derived hints must
not enter training, retrieval, reward design or checkpoint selection.

For SFT, count both total context and supervised assistant tokens. Mask tool
observations and copied context. Preserve target coverage and exclusion counts;
never hide overlength examples by silent truncation. Keep validation data and
tokenization fixed across measurements; report token-weighted and task-macro
loss separately.

## Matched evaluation

Bind weights, tokenizer/chat template, numerical precision, quantization,
serving image/arguments, agent harness, prompt and ordered tool schemas, task and
environment versions, verifier, decoding, pass@k and all time/token/turn limits.
Change only the intervention checkpoint for a matched base/post comparison.

Hosted and dedicated endpoints are separate serving blocks. Partition complete
tasks before execution and report each block before pooling. An opaque shared
endpoint can support descriptive results, not an exact causal control when its
bytes/runtime cannot be matched. Extra capacity does not authorize duplicate
attempts or a silent harness/precision change.

## Acceptance and reporting

Require finite optimization metrics, changed weights, complete recoverable
checkpoints and exact reload evidence for a usable training artifact. Run a
reward-acquisition canary before scaling RL: real verifier IDs, useful rewards,
a real parameter update and checkpoint—not just a loop returning zeros.

Require exact task/harness identity, authoritative grading, complete private
results and environment cleanup for a valid evaluation outcome. Distinguish
genuine model failures from infrastructure-invalid, interrupted and unknown
outcomes. Preserve originals and follow the predeclared retry policy.

Report paired task-level changes and uncertainty over tasks. Report training-seed
variance separately. Pre-register exclusions, the primary metric and stopping
rule. Never reinterpret a timeout or missing reward as model incapability.
