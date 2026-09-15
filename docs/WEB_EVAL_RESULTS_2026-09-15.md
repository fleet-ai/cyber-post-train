# WebExploitBench evaluation history — 15 September 2026

This is an evidence review, not a live provider status page. The public report
reads `site/web-evals.json`. It distinguishes complete results, partial results,
invalid scoring, small system tests, incomplete campaigns and unlaunched plans.
Raw reports, task answers, exploit files, judge requests and credentials remain
private. Unknown settings are marked unknown rather than reconstructed by guess.

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
| Fresh75 final step230 | Training, export, GPU reload and live serving accepted. Campaign `fresh75-step230-opencode-web-p1-v3` is collecting one OpenCode attempt on each of all 15 sites with scoring disabled until the saved work is downloaded. No performance result exists yet. |
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

## What the next run tests

Use Fresh75's **predeclared final step230**, not a checkpoint selected using WEB
results. Collect with OpenCode, preserve and download the unscored work, then
score a separate copy with a frozen judge. A judge outage must not cause new
student attempts. Compare with a newly matched base control before claiming
training improvement: the historical GPT-rescored partial baseline is useful
history, not automatically a matched control for a changed execution setup.

The one-command controller is described in
[WEB_EVAL_RUNNER.md](WEB_EVAL_RUNNER.md). Its no-model environment qualification
passed all 15 sites before the live Fresh75 launch. Unit tests and environment
qualification still do not establish unattended production reliability: the
active Fresh75 run must demonstrate collection, preservation, download and
release before the whole workflow is described as operationally proven.
