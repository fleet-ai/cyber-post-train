# Qwen3.8 training-model staging bridge

## Decision

Do not try to mount `inference/hf-cache-shared` from a Pod in
`fleet-train-jobs`. PersistentVolumeClaims are namespace scoped, and the live
inference cache and training SFS are distinct claims bound to distinct volumes.
Creating a duplicate claim, rebinding a PV, or exposing the live inference
filesystem through a temporary service would add a larger operational and trust
boundary than this model transfer needs.

The selected bridge is **immutable upstream rematerialization**. One queued,
CPU-only Job in `fleet-train-jobs` downloads only the allowlisted files from
`Qwen/Qwen3.8-27B` at exact commit
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. It then recomputes every one of
the 18 weight-shard hashes and all 11 runtime/license sidecar hashes against the
same committed model lock used to identify the inference checkpoint. This
establishes byte equality without a direct filesystem relay.

The frozen dry-run plan is
[`qwen38-27b-stage-bridge-v1.json`](../configs/qualification/qwen38-27b-stage-bridge-v1.json).
It binds the observed PVC/PV identities, exact inference model UID and source
path, destination, queue, zero-GPU resources, digest-pinned image, command,
model lock, and the approvals required before execution.

## Transaction and evidence contract

The destination is the exact revision directory:

```text
/mnt/sfs/models/Qwen/Qwen3.8-27B/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
```

The executor:

1. verifies the raw committed model-lock bytes and the self-digested plan;
2. reads the Hugging Face tree at the exact commit and reproduces the frozen
   canonical 18-shard manifest;
3. downloads into a plan-digest-specific, single-use transaction directory;
4. removes downloader cache metadata fail-closed, then uses `lstat` over the
   complete recursive tree to reject every symlink, non-regular file,
   unexpected directory, empty directory, or residual cache tree;
5. hashes every model, tokenizer, configuration, preprocessing, and license
   file;
6. writes exact `source-tree.json` and `.cyber-post-train-lock.json` files;
7. writes a self-digested `.fleet-stage-receipt.json` sidecar binding the
   source, payload manifest, destination, image, command, queue, and resources;
8. publishes the directory with Linux `renameat2(RENAME_NOREPLACE)`; and
9. treats an existing destination as idempotent only when all bytes, locks,
   manifests, and the receipt revalidate exactly.

The sidecar is staging acceptance, not complete runtime evidence. After a
future approved execution, a read-only terminal collector must additionally
bind the exact Job and Pod UIDs, resolved image ID, terminal exit code, and
Kueue admission before any Training API catalog row is registered.

## Dry-run

The checked-in launcher intentionally has no execution mode:

```bash
bash scripts/preview_qwen38_stage_bridge.sh preview
```

It validates the lock and plan, checks the live queue and destination claim,
revalidates both exact PVC UIDs and `volumeName` values plus both exact live PV
UIDs and their complete PVC `claimRef` bindings, refuses duplicate Job or
ConfigMap names, then performs Kubernetes server-side dry runs for an immutable
ConfigMap and the suspended Job. It creates no
resource and copies no bytes. As a second fail-safe, the executor refuses the
committed `dry_run_only_unapproved` plan before any remote manifest request or
filesystem write. A reviewed follow-up must freeze a new
`approved_for_execution` plan digest before this Job can materialize bytes.

## Execution prerequisites

Execution remains blocked until all of the following are independently
reviewed:

- explicit approval for the approximately 55.6 GB materialization;
- positive authoritative Qwen3.8 Fleet reward with valid cleanup;
- exact Job and ConfigMap names remain absent;
- the immutable destination and single-use transaction path remain absent, or
  the destination already carries an exactly valid receipt;
- `training-lq` remains Active and the CPU node pool is available;
- the remote commit still reproduces the frozen shard manifest; and
- a narrow follow-up change adds an explicit create-only execution path and a
  UID-bound terminal collector.

No PVC/PV creation, inference filesystem mutation, model catalog registration,
GPU request, or scheduler bypass is part of this bridge.
