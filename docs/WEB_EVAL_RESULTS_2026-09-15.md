# WebExploitBench evaluation history — 15 September 2026

This is an evidence review, not a live provider status page. The public report
reads `site/web-evals.json`. It distinguishes complete results, partial results,
invalid scoring, small system tests, incomplete campaigns and unlaunched plans.
Raw reports, task answers, exploit files, judge requests and credentials remain
private. Unknown settings are marked unknown rather than reconstructed by guess.

## Standard result format and the paper's reference results

The public page now uses the original paper's three Level-0 columns for every
campaign: Pass@1, Pass@3 (Avg.), and Pass@3 (Max). A blank source record is shown
as “not available,” never as zero. Partial values state their exact coverage out
of the benchmark's 110 known weaknesses.

The paper's published Level-0 web-exploitation results are:

| Model | Agent program | Pass@1 | Pass@3 (Avg.) | Pass@3 (Max) |
| --- | --- | ---: | ---: | ---: |
| GPT-5.5 | Codex 0.133.0 | 19.09% | 16.06% | 28.18% |
| Claude-Opus-4.7 | Claude Code 2.1.150 | 16.36% | 14.55% | 26.36% |
| GLM-5.1 | Claude Code 2.1.150 | 11.82% | 8.18% | 15.45% |
| DeepSeek-V4-Pro | Claude Code 2.1.150 | 10.00% | 8.18% | 18.18% |
| Qwen-3.7-Max | Qwen Code 0.16.1 | 10.91% | 12.42% | 20.91% |
| Kimi-2.6 | Kimi Code 1.44.0 | 3.64% | 3.03% | 8.18% |

Those results cover all 110 Level-0 weaknesses with three independent attempts.
They are reference context, not matched controls for Fleet's OpenCode campaigns:
the agent programs, model versions, budgets and scoring setups differ.

## Qwen3.8 OpenCode baseline, rescored with GPT

The source campaign is
`chris-web-qwen38-opencode11827-tl-l0-p4-20260911-v12`.
It intended four attempts on each of 15 websites: 60 attempts total. Its original
GLM scoring was invalid because the configured judge was unavailable. **That did
not erase the collected work.** A separate GPT scoring pass reused the stored
reports and automatic-check evidence without asking Qwen to solve anything again.

| Quantity | Verified value |
| --- | --- |
| Student | Qwen3.8-27B, base revision `1d4bf0f2` |
| Agent program | OpenCode 1.18.27 |
| New judge | `gpt-5.5-2026-04-23`, medium reasoning, 32,768 output-token limit |
| Judge temperature | Not sent; unsupported for this model |
| Usable attempts | 44/60, spanning 13/15 websites |
| Websites with all four usable attempts | 7 |
| Excluded attempts | 8 unavailable targets, 4 authentication failures, 4 agent-process failures |
| Passed known-weakness checks | 45/278 = 16.19% |
| Mean per-attempt score | 16.2055% |
| Judge calls | 199 complete replies; zero scoring errors |

The two percentages weight attempts differently: 45/278 pools individual
weakness checks, while the mean first computes a fraction for each attempt and
then averages those fractions. Neither is the percentage of websites solved.
These are **partial-coverage results**, not a complete baseline or full pass@4.
Missing attempts are not model failures. Do not pool the original scoring and
the rescore as independent attempts.

### Evidence checked

The private audit at `/private/tmp/wbe-gpt-final-audit.hrsQ5r` contains the
preflight, accepted small test and final completion records. Their self-digests
were checked, as were all 44 score-file hashes and the source archive hashes.
The final completion digest is
`sha256:b25f4701aae5e1dc1e7188aaaf405284e6a49e9f9c02b6e7d9de920b1b47e1da`.

The 243 stored final-report/automatic-check files matched the original bytes.
All 199 judge replies were complete and parseable, with no error, truncation or
duplicate call. One early test's GPT reply is stored under a misleading legacy
filename ending in `original-glm-failed.jsonl`; its contents identify GPT. It was
included in the verified total rather than mistaken for a missing judge call.

The official benchmark implementation performed scoring. Benchmark source hash:
`sha256:746f15315bb0981bbde9756a39581a6067967a62795f29a2908806f68b6b48e3`.
CAGE revision: `09a191c565230cebb8255899d622d23c7ddeff33`.
Dataset revision: `7f97d87fa8ab728260c0ba9b09b9c8f00bb82ad5`.

## Other known evaluations and preparations

| Evaluation | What can be claimed |
| --- | --- |
| Qwen3.8 / Qwen Code 0.22.3 / GLM | 48/60 usable attempts, 51/317 checks; partial only. Different budgets prevent a harness-only comparison. Historical; no new Qwen Code runs. |
| OpenCode V6 and V7 repair tests | One accepted attempt each; system checks, not full evaluations. |
| OpenCode V12 repair test | Four accepted attempts on one website, 13 validated judge calls. |
| Qwen Code V6/V7 controls | V6 had an invalid judge reply; V7 was accepted. Historical only. |
| Wider V8/V10/V11/V14/V17/V21/V24 reruns | No accepted full-benchmark result established. Preserve each campaign separately; never pool repair attempts opportunistically. |
| Teacher-SFT step186 matched V28 test | One controller per model exited and resources were released. Score acceptance has not been established; no improvement claim. |
| Teacher V30/V31 preparations | Prepared plans do not establish launched or completed evaluations. |
| Fresh75's first WEB launch | Campaign `fresh75-step230-opencode-web-p1-v3` is infrastructure-invalid. It produced no model attempts. A later repaired pass@1 campaign is reported separately below. |
| Self-SFT WEB | No accepted complete campaign established in this audit. |
| Qwen3.6 / Qwen Code | Historical 15-target pass@1 result: 10/110 checks. One allowed model timeout; no infrastructure-invalid target. |
| Qwen3.6 / Claude Code corrected result | Historical 15-target selected result: 16/110 after documented replacements for broken target runs. Earlier 14/94 was not final. |
| GLM5.3 / Qwen Code | Interim records only; no validated final full result established. |

The September 13 baseline audit predates the GPT rescore. Its claim that the
original OpenCode scoring was unusable remains true; it should not be read as
saying the later, separately checked GPT scores are also unusable.

Supporting repository records:

- [Earlier baseline audit](evidence/webexploitbench/2026-09-13-qwen38-harness-baseline-audit.md)
- [Wider-run infrastructure audit](evidence/post_sft/2026-09-14-webexploitbench-v11-infrastructure-audit.md)
- [Fresh75 export and reload](evidence/qwen38-fresh75-step230-reload-accepted-20260915.json)
- [Fresh75 staged weights and paused registration](evidence/qwen38-fresh75-step230-inference-stage-v2-accepted-20260915.json)

## Fresh75 step 230, complete pass@1 evaluation

On September 16, the repaired two-stage pipeline completed one OpenCode attempt
on every WebExploitBench website and then scored the 15 saved attempts with
`gpt-5.5-2026-04-23`. Collection and scoring were separate: a scoring failure
could not erase or rerun the model's work.

| Quantity | Verified value |
| --- | --- |
| Student | Qwen3.8-27B, Fresh75 final step 230 |
| Agent program | OpenCode 1.18.27 |
| Attempts | 15/15 usable; one per website |
| Judge | `gpt-5.5-2026-04-23` |
| Scored attempts | 15/15; no scoring error |
| Passed known-weakness checks | 3/110 = 2.73% |
| Mean attempt score | 2.0208% |
| Attempts with a non-zero score | 3/15 |

The collection-set receipt is
`sha256:673b3a311c3ee7d85d5f76d5453953ad6a7ad84c9248ce0addac2206c200fab5`.
The scoring completion receipt is
`sha256:2c4790458c67cd076d5339919b03ca9b0b22170f779a6d4dbd41fb0b3d7d0c72`.
All 15 score bundles were reopened and independently checked against their
sealed scoring plans after completion.

This is a complete pass@1 measurement, but it does **not** establish training
improvement. The historical base-model result used pass@4 and came from a
different execution campaign. A matched base-model control using the same
collection and scoring settings is required for a lift claim.

## Fresh75 step 230, provisional Pass@3 from the pass@8 campaign

The wider campaign `fresh75-step230-opencode-web-p8-v3` has produced 104 usable
attempts across 13 of 15 websites. All 104 saved attempts were scored separately
with `gpt-5.5-2026-04-23`, and all 13 scoring results passed an independent
integrity check with zero scoring failures.

| Quantity | Provisional verified value |
| --- | --- |
| Student | Qwen3.8-27B, Fresh75 final step 230 |
| Agent program | OpenCode 1.18.27 |
| Attempts | 104/120 usable; eight on each of 13 websites |
| Judge | `gpt-5.5-2026-04-23` |
| Benchmark coverage | 100/110 known weaknesses; 13/15 websites |
| Pass@1 | 2.000%; first designated attempt, partial 100/110 coverage |
| Pass@3 (Avg.) | 3.125% |
| Pass@3 (Max) | 4.9643% |

These are **partial-coverage, provisional numbers**, not a full benchmark
result. The calculation follows the paper's two Pass@3 views. Pass@3 (Avg.) is
the average success rate across attempts. Pass@3 (Max) counts a known weakness
as found when at least one of three attempts finds it. Because this campaign has
eight attempts per website rather than exactly three, the calculation averages
over every possible three-attempt group among those eight attempts. This avoids
hand-picking a favorable trio. The numbers will be recomputed after the final 16
attempts from the remaining two websites are available and scored.

## What remains

The current pass@8 run still needs the final 16 attempts from two websites. The
next scientific comparison should run a base-model control with
the same OpenCode, task, attempt-budget and scoring settings. The historical
GPT-rescored partial baseline is useful history, but it is not a matched control
for this run.

The one-command controller is described in
[WEB_EVAL_RUNNER.md](WEB_EVAL_RUNNER.md). Its no-model environment qualification
passed all 15 sites only in the source machine. That was insufficient: all nine
machines restored from the resulting snapshot failed
`snapshot_runtime_readiness` before any model call. The failure is preserved in
[the Fresh75 infrastructure incident](evidence/webexploitbench/2026-09-15-fresh75-p1-v3-infrastructure-invalid.json).
The preparation path had explicitly made a filesystem-only Tensorlake snapshot,
which cold-boots without preserving the Docker daemon's running state. The
successful successor binds a memory snapshot, inherits its exact machine
resources and checks each restored machine before any model call. The pass@8
campaign repeats the same safety gate before wider collection.
