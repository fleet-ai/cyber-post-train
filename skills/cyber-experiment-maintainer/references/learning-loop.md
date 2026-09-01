# Experiment learning decision matrix

| Observation | Durable response |
|---|---|
| A request field renders with different semantics | Validate the rendered output and add a regression at the API boundary. |
| Two plan fields identify different artifacts | Add a cross-field invariant; update the stale plan only from accepted immutable evidence. |
| YAML is valid but deployed render is rejected | Test the real rendered manifests, not source text alone. |
| A private image cannot pull | Prove image digest and pull authorization before submission; do not mutate the failed immutable Job. |
| API and Kubernetes status disagree | Improve UID-bound evidence collection and terminal reconciliation; do not guess. |
| Zero reward coincides with truncation | Fix and canary the harness horizon/tool contract before scaling RL. |
| A valid evaluation outcome is inconvenient | Preserve it; do not rerun outside the frozen retry policy. |
| A procedure recurs across runs | Encode the decision guidance in a focused skill and deterministic mechanics in code. |

## Promotion test for a skill rule

Add guidance to a skill only when all are true:

1. The lesson changes a future agent decision rather than merely documenting history.
2. It applies across multiple runs or models.
3. It cannot be enforced completely by code alone.
4. It does not depend on a current job ID, mutable endpoint state, secret, or benchmark content.
5. Its scope and stopping condition are clear.

Otherwise keep it in a receipt, config, regression test, or living evidence report.
