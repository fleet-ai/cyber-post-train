# Cyber post-training

This repository was reset on 2026-09-24. It now holds a small, dated research
record and one reusable data-splitting module. The previous training launchers,
evaluation campaigns, dashboard, generated job configurations, and local model
copies were retired. This checkout does not submit jobs or serve models.

## What remains

- [`training/splits.py`](training/splits.py) assigns all versions and attempts
  from the same application/task family to the same train, development, or test
  split. The assignment is deterministic. Callers must provide reviewed family
  identity; the code cannot decide whether two different names describe the
  same underlying challenge.
- [`configs/data/`](configs/data/) contains dated, read-only task coverage,
  qualification, split, and teacher-corpus records. Their counts are explained
  in [`docs/RESEARCH_SNAPSHOT.md`](docs/RESEARCH_SNAPSHOT.md).
- [`docs/evidence/qwen38-sft-checkpoint-inventory-20260924.json`](docs/evidence/qwen38-sft-checkpoint-inventory-20260924.json)
  records checkpoint and model-export identities as observed on 2026-09-24.
  It is a historical inventory, not a current availability check.
- [`docs/RESET.md`](docs/RESET.md) explains what was removed and where the old
  code can be found in Git history.

Run the checks with Python 3; no packages need to be installed:

```sh
python3 -m unittest discover -s tests
python3 scripts/check_size.py
```

The repository limit is fewer than 10,000 physical lines across **all tracked
text files**, including records, tests, and documentation. The check runs on
every push and pull request.

## Working from this starting point

Treat the retained JSON files as dated evidence. Confirm current task quality,
model availability, and evaluation validity from their original systems before
using them in a new study. A new training or evaluation tool should start with
one clear use case, a small test, and the exact job/evaluation record needed to
interpret its result. Do not copy old launchers forward just because they exist
in Git history.

Never commit credentials, raw task prompts, private traces, flags, answers,
model weights, or generated runtime folders. Store large artifacts in their
designated external stores and keep only verified references here.
