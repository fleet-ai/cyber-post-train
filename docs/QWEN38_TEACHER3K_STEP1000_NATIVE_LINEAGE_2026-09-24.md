# Qwen3.8 Teacher3K step-1000 native lineage

The exact native FP32 step-1000 checkpoint is no longer discoverable through
the authorized shared filesystem, Kubernetes control plane, or repository
evidence. Its self-digesting checkpoint receipt and 33-file seal still exist,
as does the accepted BF16 export. Those surviving records prove historical
identity; they do not contain the deleted FP32 or optimizer bytes and cannot
recreate them.

The machine-readable evidence is
[`2026-09-24-q38-step1000-native-lineage-absence-v1.json`](evidence/qwen38-study/2026-09-24-q38-step1000-native-lineage-absence-v1.json).
It contains only sanitized paths, timestamps, UIDs, counts, hashes and
classifications. It contains no task text, prompts, traces, outputs, tensor
values, flags, scores or credentials.

## What remains

- The step-1000 checkpoint receipt is present and self-valid: file SHA-256
  `378011c8…`, receipt SHA-256 `bae4b721…`.
- The step-1000 seal is present and self-valid: file SHA-256 `b639cd04…`,
  receipt SHA-256 `685af3a1…`. It records 33 native files totaling
  324,627,486,795 bytes, including eight FP32 model shards and eight optimizer
  shards.
- The accepted BF16 export is present and self-valid: receipt SHA-256
  `8a1d9e09…`, 29 files totaling 55,586,032,099 bytes, payload manifest
  `023c5f8b…`.

At `2026-09-24T11:26:14Z`, the UID-bound storage observer saw native
`global_step_1000` absent and native `global_step_1800` and
`global_step_1837` present. This agrees with the immutable run configuration:
save every 100 steps and keep the latest two. The runtime implements that
contract by recursively deleting older checkpoint directories while leaving
receipts outside them.

## Bounded absence, not an absolute cloud-backup claim

The search covered the canonical run root, all accessible Teacher3K/step-1000
training roots, the generic checkpoint root, likely export/model/cache roots,
small JSON manifests keyed by the exact source path and shard hashes, Git
history, Kubernetes workload specifications, and Kubernetes snapshot
capabilities. It found no second native manifest, copy locator, preservation
receipt or upload path. The staged serving artifact contains BF16 safetensors
and no native `.pt` files.

Some unrelated private and evaluation-output directories reject the observer
UID. An administrator-only storage-provider backup that is not surfaced in the
mounted filesystem, Kubernetes API, repository or receipts cannot be ruled out.
No evidence says such a backup exists. If one is produced, all 33 files must
match the surviving seal before it can be accepted as the native checkpoint.

## Replay classification

An exact FP32-to-BF16 replay is not currently possible. The exporter requires
the eight native CPU-FP32 model shards, reconstructs their distributed tensors,
and performs a lossy BF16 cast. The accepted BF16 payload cannot recover
discarded FP32 low bits, optimizer shards or complete native resume state.
Rehashing or reusing that accepted BF16 payload preserves the served artifact,
but it is not a replay of the original conversion.
