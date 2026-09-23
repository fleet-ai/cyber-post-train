# Public research report

Static source for the public Fleet Cyber research report. It intentionally
contains only aggregate results and plain-language methods. Raw benchmark
prompts, traces, exploit payloads, answers, flags, and credentials do not belong
here.

Preview locally:

```sh
python3 -m http.server 8080 --directory site
```

The deployed site is served from the repository's `gh-pages` branch. Update
`report-data.js`, review the rendered pages, then publish the exact static files.

## Importing final matched evaluations

`evaluation-results.json` is generated from two final, sanitized numerical
summaries. The importer fails closed unless both the WebExploitBench and Fleet
development studies are complete pass@8 comparisons, all task rows account for
eight valid or technical outcomes per model, and the exact evaluator receipt
schemas and evidence digests are present. It computes the task changes, headline
values, and 95% confidence intervals itself.

```sh
python -m cyber_post_train.public_eval_import \
  --web /absolute/private/path/web-final-sanitized.json \
  --web-file-sha256 sha256:<reviewed-file-digest> \
  --fleet /absolute/private/path/fleet-final-sanitized.json \
  --fleet-file-sha256 sha256:<reviewed-file-digest> \
  --output site/evaluation-results.json
```

Running the same command with the same inputs is a no-op. The generated public
file contains randomly reordered task numbers, final percentages, valid and
technical attempt counts, paired changes, confidence intervals, and evidence digests. It
cannot contain task names or IDs, prompts, responses, traces, answers, flags,
per-attempt scores, or credentials. Review the rendered page before publishing.

The command also requires the exact reviewed SHA-256 of each source file. Each
source file must be a
`cyber_sanitized_matched_pass8_aggregate_v1` self-digesting record. Both need
the exact comparison definition, terminal-index, and scored-outcome-index
digests. The WebExploitBench record must name the current replica-set, arm-
terminal, and deferred-score-acceptance schemas. The Fleet record must name the
held-out launch observation, accepted-cell, and controller-terminal schemas.
The exact comparison-definition digest binds the base and step-1,000 revisions,
OpenCode version, saved task list, and all eight attempt identities.
It recomputes task percentages, paired totals, changes, and the deterministic
10,000-resample 95% confidence interval. Source rows must be reordered using a
private random permutation; only its receipt digest reaches the public file.
The private source paths and mapping are never copied into the public file.

The Fleet comparison uses the 17-task development split. Those tasks were kept
out of training, but they may guide experiment choices; they are not the final
untouched Fleet test. Qwen3.8-27B was released after WebExploitBench, so the
matched before-and-after change is useful while the absolute score is not a
temporally clean generalization claim.

## Writing standard

Write for a reader who knows neither Fleet nor machine learning. Prefer the
ordinary explanation over an internal name. For example:

| Do not write | Write instead |
| --- | --- |
| roster | saved list of tasks |
| binding | the exact files and settings needed to run the task |
| runtime seed | hidden setup used to create and score the task |
| run receipt | independently checked record of what ran |
| verifier | automatic answer check |
| harness | program that lets the model use tools |

When a technical term is necessary, define it before using its abbreviation.
The public-language test blocks several internal phrases that previously made
the report hard to read.
