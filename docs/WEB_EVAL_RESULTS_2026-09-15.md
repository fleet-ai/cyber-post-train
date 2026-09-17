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
| Pass@1 | 9.2593% on 54/110 weaknesses with a designated first attempt |
| Pass@3 (Avg.) | 14.6667% on the 71/110 weaknesses with at least three usable attempts |
| Pass@3 (Max) | 24.0000% on the same partial 71/110 coverage |
| Passed known-weakness checks | 45/278 = 16.19% |
| Mean per-attempt score | 16.2055% |
| Judge calls | 199 complete replies; zero scoring errors |

Pass@1 uses the designated `pass_1` attempt where it exists. Pass@3 uses every
possible group of three from each website with at least three usable attempts;
no favorable group was selected. The 45/278 figure pools all 44 attempts and is
not one of the paper's headline measures. These are **partial-coverage
results**, not complete benchmark scores. Missing attempts are not model
failures. Do not pool the original scoring and the rescore as independent
attempts.

### Evidence checked

The private audit at `/private/tmp/wbe-gpt-final-audit.hrsQ5r` contains the
preflight, accepted small test and final completion records. Their self-digests
were checked, as were all 44 GPT-specific score-file hashes and the source
archive hashes.
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

## Smoke test 1

Smoke test 1 has produced 104 usable attempts across 13 of 15 websites. All 104
saved attempts were scored separately
with `gpt-5.5-2026-04-23`, and all 13 scoring results passed an independent
integrity check with zero scoring failures.

| Quantity | Provisional verified value |
| --- | --- |
| Student | Qwen3.8-27B SFT checkpoint at optimizer step 230 |
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
[the earlier infrastructure incident](evidence/webexploitbench/2026-09-15-fresh75-p1-v3-infrastructure-invalid.json).
The preparation path had explicitly made a filesystem-only Tensorlake snapshot,
which cold-boots without preserving the Docker daemon's running state. The
successful successor binds a memory snapshot, inherits its exact machine
resources and checks each restored machine before any model call. Smoke test 1
repeats the same safety gate before wider collection.
