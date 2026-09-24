# Qwen3.8 Teacher3K step-1000 corpus audit

Date: 2026-09-24
Status: confirmed data-contract defects; replacement corpus remains blocked

## Outcome

The step-1000 checkpoint optimized a heavily concentrated, visible-action
objective whose dominant rows did not preserve a valid OpenCode interaction
boundary. The two strongest defects are independently sufficient to invalidate a
clean capability interpretation: 80.27% of derived rows were raw-token clipped,
and the training tool names differed from the model-facing evaluation tool names.

The immutable aggregate receipt is
[`qwen38-teacher3k-step1000-corpus-audit-20260924.json`](evidence/qwen38-teacher3k-step1000-corpus-audit-20260924.json).
It contains no task or session identifiers, prompts, traces, private reasoning,
raw scores, credentials, or external benchmark content. No API, cluster,
evaluation, model, scoring, or private-data-materialization operation was run for
this audit.

## Bound corpus

The digest chain binds a private aggregate selection (`441f489c...`) through the
source manifest (`d0641cc7...`), 32K manifest (`a8d08609...`), final Parquet
(`bf245db0...`), and materialization receipt (`af6a686b...`). The bound corpus has:

- 2,886 source sessions across 496 task keys and 1,176 exact versions;
- 3,679 source windows rechunked into 14,693 training rows;
- 176,654 supervised assistant targets and 57,384,881 supervised tokens; and
- a maximum single target of 4,700 tokens, so target length was not the reason
  for raw-token clipping.

Step 1000 consumed 8,000 distinct rows and 31,035,607 supervised tokens, 54.08%
of the planned epoch. The permitted aggregate metadata does not expose the
per-final-row source mapping needed to calculate the exact task, session, or
teacher mix of that prefix.

## Concentration

| Unit | Count | Largest token share | Top-five share | Top-ten share |
|---|---:|---:|---:|---:|
| Teacher model | 10 | 49.14% | 90.10% | 100.00% |
| Exact task key | 496 | 7.04% | 23.22% | 32.13% |
| Exact task version | 1,176 | 3.68% | 11.56% | 18.16% |
| Source session | 2,886 | 0.66% | 2.81% | 4.83% |
| Transitive reviewed-atom component | 370 | 13.27% | 32.17% | 40.11% |

The task-key view understates family concentration. The lineage graph contains
389 reviewed atom lineages and 370 transitive components; 59 components join
multiple task keys, and the largest joins six keys and seven atom lineages.
Fractionally assigning a multi-application task across its reviewed atom
applications gives 44.21% of tokens to the largest application, followed by
15.57%, 14.39%, 10.67%, 9.87%, 4.71%, and 0.59%.

## Target and success provenance

The exact response-type census is 166,434 non-submit tool actions (94.21%),
2,908 visible decision/reasoning responses (1.65%), 2,691 final-submit responses
(1.52%), and 4,621 other responses (2.62%). Final submissions account for
1,320,887 tokens, or 2.30% of the supervised objective. Private reasoning was
omitted by design. The sealed selection does not publish token counts separately
for visible decisions and non-submit tool actions, so their token split must not
be inferred from response counts.

Every selected source passed the historical eligibility, infrastructure,
authoritative-success, and completed-verifier gates, and all 2,886 session,
acceptance, trace, and normalized-record digests are unique. That whole-session
evidence is weaker than success inside the retained fragment: 149 malformed
lifecycle records were prefix-salvaged, dropping 13,295 messages and 5,878
assistant responses. Those prefixes retain 1,650,605 supervised tokens, but 148
of 149 contain no retained final submission.

A separate 419-row accepted-metadata projection has complete self-digest,
trace, verifier, cleanup, and ingest checks, but shares zero session IDs,
acceptance digests, or trace digests with Teacher3K. It therefore does not
corroborate these source sessions.

## Duplicate and held-out findings

The source builder removed three exact window-payload duplicates. The retained
selection has no duplicate session, acceptance, trace, or normalized-record
digests. There are 449 legitimate multi-window sessions, producing 793 windows
beyond one-per-session. The aggregate receipt asserts that supervised target
occurrences were preserved during rechunking, but it exposes no digest inventory
for the 14,693 derived rows. An independent zero-duplicate claim for those rows
is therefore unavailable from the permitted metadata.

All 1,176 training versions now have reviewed lineage resolution: 1,153 directly
from exact metadata, one from an exact-key sibling, and 22 through reviewed
legacy aliases. Exact key and version overlap with the proposed 25-family
heldout was zero, yet reviewed atom lineage proves that five families were
exposed through six aliases. The largest supported Teacher3K-clean set is 20
families: 13 development and seven final-test. This is specific to the bound
Teacher3K corpus, not a claim that those families are globally untouched.

## Confirmed defects

1. **Anchorless raw-token rechunking.** 11,794 of 14,693 rows place their first
   target at the clip offset. In a fixed sample, 254 of 256 clipped rows lacked
   the expected chat delimiter while all 256 unclipped controls had it. The
   step-1000 prefix contained 6,451 clipped rows out of 8,000.
2. **Tool-contract mismatch.** Training targets used bare `bash` and
   `submit_report`; evaluation exposed `fleet_bash` and `fleet_submit_report`.
   Across 136 verified evaluation traces, the candidate produced 32 invalid
   bare-`bash` events in 24 attempts and no valid report submission.
3. **Success inherited across destructive salvage.** Whole-session success did
   not prove that the retained prefix contained the successful report.
4. **Task-key-only leakage control.** Exact-string exclusion missed five
   heldout families represented by six aliases.
5. **Uneven token weighting.** One teacher supplied 49.14% of tokens, and the
   largest transitive lineage component supplied 13.27%.
6. **False historical holdout field.** The immutable manifests' claim that all
   25 heldout families were excluded is provenance only; the proven clean count
   is 20.

## Unproven boundaries

- Distinct per-session harness digests do not prove distinct harness builds;
  they may have session-scoped semantics.
- Full-corpus concentration must not be asserted as the exact step-1000 prefix
  mix without final-row source identities and sampler order.
- Target-occurrence preservation is not an independently recomputed
  zero-duplicate proof for derived rows.
- The aggressive full-weight optimization recipe may have amplified forgetting,
  but it does not displace the proven context and tool-contract defects.
- The 20-family corrected set is Teacher3K-lineage clean, not globally pristine.

## Replacement boundary

The message-aligned replacement plan remains
`blocked_pending_exact_source_evidence`: launch is unauthorized, no private data
has been materialized, and no GPU job has been submitted. The completed lineage
map may satisfy part of the roster requirement, but the plan has not been
resealed. It still requires retained successful-report binding, captured proof
of the exact provider-facing tool contract, and private-source digest readback
before create-once materialization.

Validate the receipt and repository-available bindings with:

```sh
uv run pytest -q tests/test_qwen38_teacher3k_step1000_corpus_audit.py
```
