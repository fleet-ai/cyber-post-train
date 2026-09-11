# Representative study splits and train-only source selection

`training/study_data.py` is a metadata-only planning layer over the existing
`training/corpus.py` and `training/dense.py` compiler. It does not fetch tasks,
read transcripts, infer vulnerability labels, score models, tokenize new data,
change target masks, or launch work.

The new study protocol uses **fresh task outcomes**, not teacher-reference CE,
for development and final evaluation. No development reference session is
required or emitted. Legacy `cyber_task_split_v1` remains a historical CE input;
new studies emit a train-only `cyber_task_split_v2` for
`validation_mode: task_outcomes_only`.

## Required inventory and its present limitation

The self-digested `cyber_study_inventory_v1` has `origin: fleet_cyber` and `tasks`.
Each task row contains exactly:

| Field | Required meaning |
| --- | --- |
| `task_key`, `task_version_id` | Exact authorized Fleet task and immutable version |
| `lineage_id` | Reviewed identity joining aliases and versions |
| `certification` | `status: accepted` plus immutable `receipt_sha256`; other statuses are quarantined |
| `exposure` | `unexposed`, `exposed`, or `unknown`; unexposed requires `receipt_sha256` |
| `taxonomy.task_family` | A verified grouping label and `evidence_sha256`; never a task-key-derived guess |
| `taxonomy.application/environment/vulnerability_family/difficulty` | Verified labels with evidence, or explicit `missing`/`unverified` |

A verified taxonomy field has `status: verified`, `value` (one label or a
nonempty list of labels), and `evidence_sha256`. `task_family` must have one
reviewed label. Environment and application are distinct axes; neither is
inferred from a task name or deployment URL. Difficulty is predeclared reviewed
metadata, not computed from this study's development/final outcomes. Missing
axes remain visible in audit counts and `__missing__`/`__unverified__` buckets.
Missing family evidence quarantines the task and any known unresolved aliases.

An exposure receipt must cover all known family aliases/versions and establish
that the final-test group was not used in student training or recipe selection.
One exposed or unknown version disqualifies the entire family from final-test
eligibility, even if that version is rejected or no longer runnable. These
checks bind **claims to evidence digests**; they do not re-certify an unavailable
upstream receipt or prove completeness of an incomplete inventory.

As of the September 11 repository census, the inspected structured inputs do
**not** provide a complete accepted-task/version join to independently verified
family, vulnerability, difficulty and untouched-exposure evidence. Historical
task-key-derived families must not be relabelled as verified vulnerability
families. This implementation has synthetic tests, not a newly certified real
representative split. The missing upstream join is a data blocker, not a reason
to fabricate metadata or relax final-test controls.

## Freeze one final test, then create split variants

1. Audit the inventory with `audit_inventory(inventory)`.
2. Call `freeze_final_test(inventory, groups_count=..., seed=...,
   protected_task_versions=existing_sealed_exact_tuples)` **once**, before any
   model outcomes. Existing sealed test identities are mandatory inputs when
   continuing an existing study; the helper never invents or rediscovers them.
   A protected exact version protects its entire reviewed family. Missing,
   exposed or uncertified protected identities fail closed.
3. Persist that `cyber_final_test_lock_v1` as a create-once reviewed artifact.
   Do not call the freezer again to obtain a more favorable test set.
4. Call `split_variant(inventory, final_test_lock, dev_groups=..., seed=...)`
   with the same inventory and lock for each predeclared ablation seed.

The deterministic greedy selector balances group-weighted marginal shares
across the four taxonomy axes, including missing buckets. Each axis has equal
weight; an immutable seed breaks ties. Protected test groups are fixed before
balancing remaining choices. This is not a global optimizer or a guarantee of
identical difficulty. Every split reports population/split group counts and
the maximum absolute marginal share gap per axis.

Counts are **families**, not tasks: a family with several task versions stays
together. Shared applications are permitted; label results as reviewed
task-family-held-out, **not unseen-application or unseen-vulnerability-family
generalization**. Vulnerability categories are balancing labels, not necessarily
the grouping boundary. Inventory changes require a separately reviewed study,
not silently reusing a test lock against changed tasks.

`cyber_study_split_v1` contains the full assignments and three independent
compiler/evaluator inputs:

- `training_split`: self-digested `cyber_task_split_v2`, containing only exact
  train `task_key`, `task_version_id`, `split` triples. No reference sessions.
- `evaluation.dev`: `cyber_eval_task_selection_v1`, development tasks for later
  fresh matched task-outcome evaluations.
- `evaluation.final_test`: the same schema, fixed final-test task set and lock
  digest across all variants. Do not use its outcomes for checkpoint or recipe
  selection; evaluate only after the development decision is frozen.

## Train-only, evidence-bound SFT source selection

`select_sources(episodes, study_split, models=[...], target_policy_sha256=...,
max_episodes_per_family=..., max_supervised_tokens_per_family=...,
source_seed=...)` consumes sanitized
`cyber_sft_episode_metadata_v1` objects. Non-train rows are skipped after reading
only their exact task/version identity; held-out outcomes, coverage and model
fields are not read. Prefer constructing this metadata input from train sessions
only in the first place.

Each training metadata object contains exactly `schema`, `sha256`, `task_key`,
`task_version_id`, `episode_id`, exact `model_id`, `source_kind: teacher|self`,
`validity: valid`, `verified_success: true`, `acceptance_sha256`, `trace_sha256`,
`normalized_record_sha256`, and `coverage`. No score values or source payloads.
`normalized_record_sha256` is `training.io.digest_json(complete_normalized_record)`;
it is not an assumption that a legacy `content_digest` covers fields added later.

Coverage must have a certification receipt, `scope: eligible_dense_targets`, and
the caller's exact `target_policy_sha256`. That policy binds tokenizer/template,
tool normalization and context-eligibility rules. Counts are for targets that
actually survive that policy, **not all raw trajectory responses**: otherwise
overlength exclusion could leave a report-only corpus while a pre-exclusion
audit incorrectly claims good action coverage. A compiler change invalidates
the coverage receipt and requires re-auditing before reuse.

The exact nonnegative counts are:

- `assistant_responses`, `supervised_tokens`;
- mutually exclusive `submit_report_responses`, `non_submit_tool_responses`,
  `decision_responses`, `other_responses`, summing to assistant responses;
- `submit_report_tokens`, counting supervised tokens of submission-class
  responses, including supervised terminators but never copied context;
- `completed_non_submit_tool_rounds`, at most one per non-submission assistant
  response with the required observed tool results.

The upstream audited classifier must use conservative, declared categories:
any response containing `submit_report` is submission-class (including mixed
calls); remaining tool-call responses are non-submission tool responses;
remaining visible assistant decisions are decision responses. Do not infer
semantic exploit progress from these counts. They are useful coverage proxies,
not evidence that the model learned a capability.

Sources without a non-submission decision or a completed non-submission tool
round are excluded. Then mandatory, predeclared family-level episode and
supervised-token ceilings prevent high-volume or long families from contributing
unlimited training. All task versions share one family budget. A deterministic
round robin across exact source models promotes teacher-model diversity; seeded
episode identities break ties without reading scores or losses. A whole episode
that cannot fit the remaining family token budget is skipped, not truncated or
partially target-masked. The defaults are intentionally not "all successes"—the
caller must choose and bind both budgets before selection. Apply the same
ceilings to separately compiled teacher/self arms when comparing them.

A selection with more than 50% submission tokens or more
than 50% submission responses is blocked by default. These explicit review
thresholds are heuristics, not research-derived optima, and should be frozen
before model-result-based selection. Reports are **not** silently dropped,
oversampled or downweighted to pass the gate. Retained dense assistant targets
still use the existing exact-once/masked-context compiler.

The selection receipt reports selected immutable episode IDs, policy/record
digests, candidates before caps, assistant/target-token counts, per-task,
per-family and teacher/self/model coverage, cap/quality exclusion reasons,
submission shares, and maximum family episode/token exposure. The budgets are
ceilings, not a guarantee of equal exposure when source availability is sparse.
Teacher/self source counts alone do not make a matched comparison:
compare shared task coverage and supervised-token budgets as well.

An explicit teacher-availability ablation may additionally set
`balance_target_supervised_tokens_per_family`. The deterministic selector then
chooses complete certified episodes toward that target, adding an episode only
when it strictly reduces the family's distance to the target. The existing
episode and hard token ceilings still apply; source-model diversity breaks only
equal-distance ties. This never slices an episode, duplicates a token target, or
changes its native messages/masks. Because it can lower total token exposure,
label and interpret it as a teacher-availability treatment rather than a pure
importance-weighting estimate. See the frozen Qwen audit in
[`QWEN_TEACHER_EXPOSURE_BALANCE_2026-09-11.md`](QWEN_TEACHER_EXPOSURE_BALANCE_2026-09-11.md).

Pass the ready selection through
`filter_records(private_records, selection, study_split)`. It validates the
complete chosen set and each canonical record digest before returning any
records, preserves their private contents unchanged, and rejects duplicates,
missing sources, altered records or train leakage. Feed these records and
`study_split["training_split"]` to the existing corpus selector with
`reference_validation=False`; its independent verified-success and dense
exact-once gates remain active. Bind the selection receipt and target-policy
digest alongside the compiled corpus in qualification evidence. The compiler's
actual coverage must agree with its certified eligible-target counts; this
module does not certify private data it has never inspected.

## Verification

`tests/test_study_data.py` uses only synthetic metadata. It covers deterministic
grouped splits and final-test protection, missing taxonomy, rejected/exposed
versions, family conflicts, schema/digest drift, held-out-field noninspection,
teacher/self coverage, report-only and dominance rejection, target-policy drift,
private-record digest binding, and the train-only existing-corpus adapter.

## Metadata-only historical cross-check, September 11

Two bounded reads confirmed source identity availability, not new corpus
eligibility or outcome quality:

- Dense-v4 `source-coverage.private.json`: 186 source sessions, 8,156 original
  assistant responses, 7,641 eligible and 515 excluded for
  `overlength_required_previous_round`. File SHA-256:
  `4eeeb8da51e22d7dc287700606bab3ef53813d347cc661634baae7ca9b4975e5`.
  Its only fields are source identity, original count, eligible ordinals and
  excluded ordinals/reasons. It lacks task/family joins, source-model categories
  and eligible submit-token counts needed for the new representative/capped gate.
- Frozen old Qwen-self `train.parquet`: read **only** `source_session_id` and
  `task_key` columns. Its 175 windows map to 35 sessions and 20 task keys. File
  SHA-256: `0796cc615b8060adc4ad21a797e4021f456d165d5b41f0c92738b6d928276ab4`;
  canonical sorted session/task-pair digest:
  `91cd224a68968154cc8d9c5132d96aac0508e1a4e3ac6dde2bb6fc6b15203e45`.
  The projection does not establish exact task-version, model, success-receipt or
  compiled-target-category bindings. No message column was loaded.

Neither metadata source alone satisfies the new inventory or source-selection
contract. These historical artifacts stay unchanged. A live candidate count
from another audit is not silently substituted for this frozen training set.
