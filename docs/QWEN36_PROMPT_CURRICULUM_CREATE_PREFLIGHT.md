# Qwen3.6 prompt-curriculum creation preflight

This is the mutation boundary for the train-only Qwen3.6 reward-acquisition
pilot. It consumes the exact prompt-free review plan produced by the merged
preparation workflow. It does not publish source, create a task group, submit a
job, or touch a Pipeline Lane.

## Frozen experiment shape

- exactly two train-only task families: `current` and `fakelook`;
- one shared Fleet task key per family;
- exactly four cumulative prompt-only members (`level-0` through `level-3`);
- one Qwen3.6-27B model and Qwen Code 0.22.3;
- `pass_k=1`, four planned sessions per job, and the six-session cap;
- 600 model requests and a 120-minute session duration;
- exact source task, environment, data, runtime seed, verifier, and Registry
  task-graph source identities from the reviewed plan.

The preflight reads the Fleet account and both exact source task versions, then
pages the complete active-and-archived job inventory for the exact Fleet
project. It rejects deterministic-name duplicates and a conservative superset
of semantic duplicates based on model, harness, budget, project, shared task
key, and four member labels. It refuses any other or mutating request. Source
prompts are used only in memory to regenerate and hash the proposed group
payload; the receipt contains only hashes and non-secret identities.

## Create-once boundary

Fleet's public task-group API currently has no list-by-name endpoint and no
documented create-idempotency key. Job idempotency cannot repair an ambiguous
task-group creation. Therefore the later mutation procedure must:

1. durably save the reviewed creation claim before the first POST;
2. issue exactly one task-group POST for that claim;
3. durably save the returned group ID before any next action;
4. stop without retry if the POST result is ambiguous;
5. hydrate the exact group and all four returned member versions;
6. prove the shared task identity, unique member versions, exact prompt hashes,
   and unchanged non-prompt bindings;
7. obtain an authoritative server-side read of every member's exact
   environment-version UUID;
8. only then create one idempotent `pass_k=1` job bound to the hydrated group.

Until Fleet adds server-side task-group idempotency or list-by-name lookup, an
ambiguous task-group POST requires manual server-side reconciliation. It is not
safe to infer that creation failed and retry.

The current public task-version response exposes the environment key and
version label but omits the concrete environment-version UUID. The hydration
validator therefore remains launch-blocking even when every public field
matches. It must not claim the environment is unchanged or unblock a paid job
until a separate authoritative server-side member-version receipt proves the
UUID for all four members.

Prompt-mode task-group creation also accepts a verifier lineage, not a concrete
verifier-version ID. If the active verifier changes, creation can therefore
mint members with a newer verifier. The post-create hydration check compares
the concrete served verifier ID, version, and hash and blocks the job on any
drift. No paid evaluation may proceed merely because the task-group POST
succeeded.

## Running the read-only preflight

The output path must not already exist. The command reads the Fleet credential
from the environment and never includes it in output.

```sh
uv run python -m evals.fleet.prompt_curriculum_create \
  --config evals/fleet/configs/qwen36-27b-prompt-curriculum-pilot-v1.json \
  --split configs/data/fleet-a62-task-split-v1.json \
  --review-plan /absolute/private/path/review-plan.json \
  --out /absolute/private/path/create-preflight.json
```

This command is preparation only. A successful receipt still does not
authorize task-group creation or paid evaluation.
