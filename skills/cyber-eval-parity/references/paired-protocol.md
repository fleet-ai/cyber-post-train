# Paired evaluation protocol checklist

## Identity held constant

- Base checkpoint origin and intervention lineage are explicit.
- Tokenizer, chat template, non-weight serving files, precision, quantization, inference engine, and server arguments match.
- Harness image, adapter version, system prompt, tool schemas, context policy, and agent budgets match.
- Hosted catalog aliases are backed by immutable model-repository, checkpoint,
  provider, endpoint-route, and deployment-revision evidence. An alias routed
  through a different provider remains a separate treatment even when its
  marketing model name matches.
- Tasks, task images, prompts, verifiers/graders, flags, seeds, timeouts, concurrency, and retry rules match.
- The only unexplained serving-artifact difference is the intended weight manifest.

## Evidence before launch

- Training/export receipt is terminal and accepted.
- Staging receipt proves atomic publication and complete payload hashes.
- Registration receipt binds exact Job, Pod, ConfigMap, image ID, returned model spec, and weight manifest.
- Live-parity receipt binds both serving objects, model/server info, structured tool calls, and deterministic finite logit probes.
- Output root and launch claim are absent before the one allowed create-only launch.
- For local execution, OCI inspection proves every evaluator image matches the
  host architecture before any model call or scored task begins. Cross-platform
  emulation is an infrastructure experiment, not an interchangeable scored arm.

## Outcome handling

- A successful verifier/grader under the exact protocol is a valid outcome.
- A deterministic valid zero remains zero; do not rerun it to improve pass@1.
- Infrastructure failures are retained and classified without looking at arm identity.
- Retry only under the frozen symmetric policy and use a new reviewed run identity when the output lifecycle requires one.
- Preserve exact per-task results and report paired statistics; do not substitute aggregate provider dashboards for evidence.

## Benchmark containment

Never copy external benchmark content or observations into training, retrieval, prompts, rewards, model selection, or skills. Skills may describe the evaluation procedure, but must not contain task content, solutions, scores used for tuning, or failure-derived exploit hints.
