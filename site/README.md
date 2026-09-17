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

## Experiment planner

`training-decision-space.json` holds the curated literature and candidate rows.
`experiment-plan.js` is shared by the browser and the read-only Node CLI;
`experiment-view.js` renders the compact tables, explanations and downloads.
See [the method and budget format](../docs/TRAINING_DECISION_SPACE.md).
Run `node scripts/plan-experiments.cjs --help` from the repository root (Node 18+).

For a plan-only release, fetch the latest `gh-pages` first and preserve all
non-plan sections, scripts and result data. The deployed evaluation dashboard
can be ahead of main; copying the entire main-branch `site/` would roll it back.
Publish only the reviewed plan section, planner assets, and the corresponding
script/CSS changes, then verify the live page and unchanged result-file hashes.

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
