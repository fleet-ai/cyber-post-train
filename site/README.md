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
