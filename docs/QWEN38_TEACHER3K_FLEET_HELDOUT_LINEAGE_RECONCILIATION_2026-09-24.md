# Teacher3k Fleet heldout lineage reconciliation

Date: 2026-09-24

## Decision

The proposed 25-task holdout was wrong and must not be used as a
training-heldout comparison for the teacher3k checkpoints.

It checked exact task keys. The training corpus, however, contains composite
and hinted task keys that point to the same underlying vulnerability as five of
the 25 heldout tasks. Those five tasks are training-exposed even though their
canonical task keys and exact version IDs never appeared in the corpus.

The largest lineage-clean subset supported by the exact source evidence has 20
task families:

- 13 from the frozen `dev` partition;
- 7 from the frozen `final_test` partition;
- 4 attempts per task, or 80 attempts per model/checkpoint arm.

This is a correction, not a reinterpretation. The uncommitted proposal
`configs/evaluation/qwen38-teacher3k-fleet-heldout25-proposal-v1.json`
(`sha256:4f9b8357679d39aed8383429491dcaece10fa13aad4a690d6a177ebbe01c2750`)
was rejected before activation and removed. It is superseded by
[`qwen38-teacher3k-fleet-heldout20-pass4-protocol-v2.json`](../configs/evaluation/qwen38-teacher3k-fleet-heldout20-pass4-protocol-v2.json).

## What was audited

The 32K, 64K, and 96K teacher3k manifests all bind the same private source
selection:

- 496 training task keys;
- 1,176 exact training task versions;
- 2,886 selected source sessions;
- source-selection file SHA-256
  `441f489c11e2f775bca81d98b7f578e0993a459375bf07f9e1828532ae3d91b6`.

For every exact training version, the audit made a read-only Fleet task-metadata
request and retained only safe identity data: task key, task version, task-graph
locator, and atom locator. It did not retain prompts, answers, traces, flags,
session IDs, or model capability results.

The resulting authoritative map is
[`qwen38-teacher3k-training-lineage-map-20260924-v1.json`](../configs/data/qwen38-teacher3k-training-lineage-map-20260924-v1.json):

- logical SHA-256
  `6e067606fc04162c8765b288982651773aa28c6f72390fe83428a20b9b39e75f`;
- file SHA-256
  `49be490fed7726e49b92197853f3b2a5504f08dee1617ac6e2cdbfa69d02e5ea`;
- 1,153 versions resolved directly from exact Fleet atom metadata;
- 1 legacy version resolved from another version of the same exact task key;
- 22 legacy WIZ versions resolved through three reviewed aliases already present
  in the corpus;
- 0 unresolved training versions and 0 unresolved training task keys.

The frozen split remains the source of the 25 candidate identities. Its file
SHA-256 is
`28a3dcaf31f14d724def9023d9435681772d5b8b3a8647cacc7f72ea8fc8adcb`
and its logical SHA-256 is
`05b3a8dc90ca93adc9671942d75ecb54ff8d0951b0dd48a087e29ec1e641840c`.
The reviewed quality inventory has file SHA-256
`6c211259a29376de195411e5cc182bb6f26c5b8027ab86fec855b44f064b0fd0`
and logical SHA-256
`23cfc25af183ff474e914443682e4707162110ddc2aa16103a13d984fe05b83c`.

All three historical corpus manifests contain the field
`held_out_task_families_excluded_across_all_versions: 25` in both their catalog
and rechunk provenance. This audit falsifies that specific field for the bound
source selection. The manifests remain immutable checkpoint provenance, but
that field must no longer be cited as leakage evidence. The replacement fact is
20 lineage-clean tasks and 5 exposed tasks through 6 alias keys.

## Exact contamination findings

Five heldout families occur in six training aliases:

| Frozen role | Heldout exact task key | Reviewed family | Training alias category | Training aliases |
|---|---|---|---|---:|
| final | `cysec1-2-current-gen_blackbox-bc363f8239aeeaa3f648046a__blackbox_ctf_v1` | `cyber/atoms/current/csrf-token-credential-hash-leak@0` | composite | 1 |
| dev | `cysec1-2-fakelook-gen_blackbox-8095d44c3871865064a0702c__blackbox_ctf_v1` | `cyber/atoms/fakelook/signature-merge-second-pass@0` | composite | 1 |
| dev | `cysec1-2-fentry-gen_blackbox-54997075dbb8019cd6e2e0ba__blackbox_ctf_v1` | `cyber/atoms/fentry/envelope-clte-ingest-desync@0` | hinted | 1 |
| dev | `cysec1-2-fira-gen_blackbox-5b51b6dcb4193e59b96b1b13__blackbox_ctf_v1` | `cyber/atoms/fira/restricted-comment-listing-skip@0` | composite | 1 |
| dev | `cysec1-2-fubspot-gen_blackbox-438e3127ab9af7efad692398__blackbox_ctf_v1` | `cyber/atoms/fubspot/settings-users-totp-secret-disclosure@0` | composite | 2 |

The corrected protocol contains the six exact alias task keys and version IDs.
They are excluded from the clean evaluation roster.

## Why the earlier check failed

The source selection used `group_id: task-key:<exact-task-key>` for all 496
training keys. A composite key and its canonical key therefore had different
group IDs even when both referenced the same atom. The earlier proposal compared
only exact key strings and saw zero overlap. That claim was literally true but
scientifically insufficient.

The corrected check compares the reviewed atom locator for each candidate task
against the union of atom locators from all 1,176 training versions. This catches
canonical, composite, hinted, and legacy aliases.

## Corrected pass@4 protocol

Use the exact 20-task roster in the corrected protocol for checkpoints trained
only from the bound 32K, 64K, or 96K teacher3k corpora.

For every model/checkpoint arm:

1. Run four independent attempts on each exact task version.
2. Keep the task roster, harness, prompt contract, tool and token budgets,
   sampling recipe, infrastructure, and scoring path identical between arms.
3. Mark a task as pass@4 when at least one of its four attempts is a verified
   success.
4. Report the 13 dev tasks and 7 final tasks separately. The combined 20-task
   number is descriptive and must not hide those two strata.
5. Bind the exact checkpoint and serving revision in a separate immutable launch
   plan. This reconciliation does not authorize or launch an evaluation.

With only 20 task families, uncertainty remains wide: the worst-case 95% Wilson
half-width is about 20.1 percentage points. The corresponding figures are about
23.9 points for dev13 and 29.8 points for final7. A zero-success result has a
95% upper bound of about 16.1% for union20, 22.8% for dev13, and 35.4% for
final7.

## Interpretation boundary

“Clean” here means no reviewed vulnerability lineage in this exact teacher3k
training corpus. It does **not** mean globally untouched. These task versions
have prior exact-execution certification, and earlier repository evidence shows
historical use for some of the frozen final tasks. The corrected subset supports
a teacher3k training-lineage comparison, not a claim of a pristine global test
set.

## Effects

This work made no API mutation, cluster mutation, model call, scoring call, or
evaluation launch. All Fleet access used read-only task metadata.
