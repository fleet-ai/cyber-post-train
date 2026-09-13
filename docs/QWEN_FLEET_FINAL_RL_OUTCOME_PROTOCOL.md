# Qwen Fleet final RL outcome protocol

[`qwen38-blackbox-fleet-final-rl-outcome-protocol-v1.json`](../configs/evaluation/qwen38-blackbox-fleet-final-rl-outcome-protocol-v1.json)
freezes the matched base-versus-RL comparison for the Qwen3.8-27B Miles arm.
It is score-blind, non-launchable, and does not authorize an endpoint, Fleet
session, paid job, or evaluation attempt. The existing post-SFT parent remains
unchanged for historical and SFT-specific use.

## Exact final set

The RL parent reuses the existing immutable ten-family task set:
[`qwen38-blackbox-fleet-final-task-set-v1.json`](../configs/evaluation/qwen38-blackbox-fleet-final-task-set-v1.json).
The validator pins its self-digest, file digest, and final-test lock digest. It
also reuses the existing structural validator for every exact task,
environment, starting-data, and verifier binding. A replacement or re-sealed
task set therefore requires a new parent version.

The final set remains ineligible for training, preferences, rewards,
hyperparameter selection, or checkpoint selection. WebExploitBench remains a
separate sealed external evaluation and may not contribute inputs, outputs,
outcomes, derived hints, retry decisions, or stopping signals to the RL arm.

## Matched arm contract

The parent fixes OpenCode 1.18.27, ordered `bash` and `submit_report`, context
policy, sampling, attempt seeds, pass@4, exact task/version pairing, outcome
authority, and retry policy. The child requires complete base and candidate
bindings for:

- checkpoint acceptance, HF export, weight, tokenizer, and chat-template
  manifests;
- create-once staging and serving registration;
- served-model and route-profile identities;
- normalized serving execution contract and immutable serving image;
- one shared live base/candidate parity receipt; and
- immutable agent and proxy images.

Tokenizer, chat template, serving execution contract, live-parity receipt, and
all three runtime images must match exactly. The weight manifest must differ.
Checkpoint, export, staging, registration, route, and served-model identities
may differ only as provenance or nonsemantic labels for those distinct bytes.

The RL training boundary additionally binds the exact arm, plan, terminal
acceptance, checkpoint-freeze, and training-input receipts. All four external
benchmark-use declarations must be `false`.

## Offline child binding

Use a private binding file only after the selected RL checkpoint, export,
staging, registration, and live-parity receipts are accepted. Choose fresh
absent prepared-output and result roots; the result root must be inside the
prepared-output root.

```sh
python3 -m evals.fleet.final_rl_outcome_protocol validate-parent \
  --parent configs/evaluation/qwen38-blackbox-fleet-final-rl-outcome-protocol-v1.json

python3 -m evals.fleet.final_rl_outcome_protocol materialize-child \
  --parent configs/evaluation/qwen38-blackbox-fleet-final-rl-outcome-protocol-v1.json \
  --bindings /private/path/fleet-final-rl-bindings.json \
  --output /private/path/fleet-final-rl-child.json

python3 -m evals.fleet.final_rl_outcome_protocol validate-child \
  --parent configs/evaluation/qwen38-blackbox-fleet-final-rl-outcome-protocol-v1.json \
  --child /private/path/fleet-final-rl-child.json
```

Materialization is create-once and mode `0600`. The child remains
`bound_nonlaunchable`, even when every digest is present. Before any later
reviewed execution, the evaluator must reopen the referenced receipts, repeat
the exact live task and route preflight, prove the result roots are still
absent, and create a separate execution claim. Neither the parent nor child is
accepted directly by the Fleet evaluation launcher.

## Focused verification

```sh
python -m pytest -q tests/test_fleet_final_rl_outcome_protocol.py
```

The tests cover the frozen task-set identity, legacy post-SFT compatibility,
complete receipt binding, base/candidate parity, benchmark isolation,
create-once output, fresh result roots, immutable identities, and tamper
rejection.
