# Qwen3.8 production LoRA export acceptance

The create-once production LoRA merge/export run completed successfully. Its
public receipt was read independently, its file digest and embedded self-digest
were recomputed, and its training resources were absent after completion.

The machine-readable handoff is
[`qwen38-lora-prod-step1-export-acceptance-v1.json`](../../configs/qualification/qwen38-lora-prod-step1-export-acceptance-v1.json).
It is fail-closed under
[`qwen38_lora_export_acceptance.py`](../../training/qwen38_lora_export_acceptance.py).

## What completed

- RayJob: `chris-q38-lora-prod-exp-v1-34e3dcf5`
- RayJob UID: `30279c0f-2ef5-4003-9045-eb318961c4aa`
- Workload UID: `e5ad6724-379f-4bff-9992-7dca246b1a49`
- RayCluster UID: `f140b798-ca93-4151-b027-af49f8cf65c3`
- Pod UID: `c621bd6c-cefd-4f4a-b206-df50c9515a84`
- Terminal time: `2026-09-20T15:40:48Z`
- Export root: `/mnt/sfs/jobs/chris-q38-lora-prod-exp-v1/merged-hf`
- Export receipt file SHA-256:
  `bfeedcac2a82f25696e7d102de3b5f509ded9539a9a07d65c39a94a1c58bc7a2`
- Export receipt self SHA-256:
  `237d90b34377599284c2358cd2611dfe2487a812a7179fc7460ed6ff3541ca76`

The receipt proves zero optimizer updates during export, deterministic BF16
merge, 1,199 of 1,199 tensors reopened, 55,562,855,904 tensor bytes, finite
logits after a GPU reload, exact adapter and optimizer reload, and unchanged
base and source checkpoint inputs. The exact Pod and RayCluster were absent in
the terminal check, so the run held zero GPUs.

This closes the one-step LoRA operational gate and permits preparation of the
broad LoRA run. It does not authorize an external submission while the global
cluster failure budget is being reconciled.

## What remains before evaluation

The merged artifact is not yet a served or evaluated model. A later operator
must reopen the remote public receipt, verify the same file and self digests
with the strict receipt validator, stage the full payload immutably, and create
one new paused `c1` route with a new UID. Live base/candidate parity and a fresh
duplicate census must pass before a matched score-free OpenCode
WebExploitBench task-0 canary or the frozen Fleet development evaluation.
Scoring remains a separate resumable stage. The Fleet final-test set stays
closed until development evidence selects one checkpoint.

Validate the local handoff with:

```sh
uv run python -m training.qwen38_lora_export_acceptance \
  --handoff configs/qualification/qwen38-lora-prod-step1-export-acceptance-v1.json
```
