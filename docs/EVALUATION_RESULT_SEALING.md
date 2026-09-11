# HPO evaluation result sealing

Fleet-dev, Fleet-final, and Tensorlake WebExploitBench evaluations use one
offline boundary: `evals/study_sealing.py`. It neither launches work nor reads
raw results. Every study launcher must call its `check-child` command as its
last local preflight and stop on any nonzero exit.

## Launch gate

An immutable `cyber_study_evaluation_child_v1` references the exact frozen
parent protocol by repository path, logical SHA-256, and file SHA-256. The
child remains `launchable: false` while any binding is null. It can become
launchable only when all of these are exact:

- accepted checkpoint and export receipts plus weight, tokenizer, and chat
  template manifests;
- create-once staging and serving-registration receipts;
- live base/post parity and immutable harness/runtime identities;
- for Tensorlake, the qualified source, task, grader, image, snapshot,
  controller, and sandbox-resource bindings;
- a pre-created absolute private result root with mode `0700`, containing the
  claim-journal path.

The Fleet base and post-SFT arms must differ only in their weight identity.
Tokenizer, template, harness images, serving runtime, and live parity must
match. An exact child marked nonlaunchable, or an incomplete child marked
launchable, is rejected.

```sh
python -m evals.study_sealing check-child /private/path/child.json
```

## Private completion and blinded receipt

The evaluator owns raw outcomes under the child's private result root. It must
not send outcomes, validity, or scores to W&B; W&B is training telemetry only.
After resources are released, it may write a strict
`cyber_private_evaluation_completion_v1` beside those raw artifacts. That
receipt contains classifications and artifact digests but no scores. The
sealer reads only this sanitized receipt and stats the private directory; it
does not enumerate or open raw result files.

```sh
python -m evals.study_sealing seal-completion \
  --child /private/path/child.json \
  --completion /private/results/completion.json \
  --output /private/receipts/blinded.json
```

The create-once public projection reveals only that the child terminated, its
surface and parent identity, private receipt/result-manifest digests, and the
remaining unseal gate. It deliberately omits the arm, terminal
classification, validity, and outcomes.

## Outcome-access gate

Before anyone opens private results, freeze a sealed
`cyber_study_outcome_access_freeze_v1` and validate it:

```sh
python -m evals.study_sealing check-outcome-access \
  --receipt /private/receipts/blinded.json \
  --freeze /private/study/outcome-access-freeze.json
```

For Fleet-dev, the full blinded arm set and selection-rule digest must be
frozen before dev outcomes are read for model selection. For Fleet-final and
WebExploitBench, the selected checkpoint seal must already be frozen. Final and
Web outcomes therefore cannot influence HPO or checkpoint selection.

This gate is intentionally small: it validates and projects artifacts only.
It contains no API, Kubernetes, Tensorlake, model, or W&B client.
