# Working in cyber-post-train

Keep this repository small and understandable. `main` is the one persistent
local and remote branch. Remove temporary checkouts after use. Every change
must leave fewer than 10,000 tracked text lines; run
`python3 scripts/check_size.py` and the relevant tests.

The retained JSON files are historical observations. Do not turn a dated
checkpoint path, task count, or candidate qualification into a claim about
current availability or scientific validity without checking the source system.
Group task versions by reviewed application and task family before splitting;
the split helper alone does not establish that lineage.

Keep scripts focused, standard-library-only when practical, and tested at
their actual input/output boundary. Remove obsolete versions instead of
stacking wrappers. Do not add generated outputs, virtual environments, raw
prompts/traces/answers/flags, weights, or credentials to Git.

For any Fleet Kubernetes `Job` or `RayJob` created for Chris, the rendered
root object **must** contain this exact annotation before submission:

```yaml
metadata:
  annotations:
    fleet.ai/failure-alerts: "off"
```

If a launcher creates that object indirectly, verify the server-rendered
preview. A request flag, Pod annotation, or patch after creation is not enough.
This suppresses failed-job notifications only; it does not excuse idle GPUs or
failure cleanup.
