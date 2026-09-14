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
