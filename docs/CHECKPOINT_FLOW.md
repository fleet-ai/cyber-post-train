# Qwen3.8 SFT: preserve and evaluate every planned checkpoint

This applies to a **new corrected-corpus run**, not the already-running 96K
Teacher3K job. The latter uses `keep_checkpoints: 2`: its receipts at steps
150/200/250 survived, but the native weights were pruned before export. A
receipt alone cannot be evaluated or used to reconstruct the weights.

## Simple first-study policy

Freeze a finite `max_steps = N` and checkpoint interval `I`. Set
`keep_checkpoints >= ceil(N/I)` in the immutable plan, including the final
tail checkpoint. Set `eval_interval = I` when using a separate teacher-text
development set. This makes retention independent of export/evaluation speed;
an exporter failure cannot erase an unevaluated checkpoint. Training can
continue while CPU export and task evaluation run on separate allocations.

Budget storage before submit. A measured full native Qwen3.8 checkpoint is
~303 GiB; one BF16 export is ~55.6 GB. At 856 steps and interval 50, expect
18 native checkpoints (~5.33 TiB) plus ~1.0 TB of exports, not the old
two-checkpoint ~606 GiB. On 2026-09-25 the live `sfs-shared` PVC was 1 PiB,
`df` showed ~421 TiB free, and `fleet-train-jobs` exposed no ResourceQuota.
This is shared free space, **not** a per-project reservation or proof that no
other storage policy exists. Recheck live free space and any project cap before
admission; do not accumulate unlimited study runs or delete another run's data.

For each saved step `S`, bind the exact run ID/UID, image, plan and corpus
digests, then require all of these distinct gates:

1. Digest-valid `checkpoint_receipts/step-<S:06>.json` and, when teacher CE is
   enabled, `validation/step-<S:06>.json`, both matching the plan and step.
   The CPU sealer must inventory and hash all native rank files, verify the
   trainer/sampler cursor, and reject files changing during its read.
2. CPU-only seal and BF16 export to *new* step-specific paths; export must
   reopen every tensor, match exact key/shape/dtype and base sidecars, preserve
   source hashes, and publish atomically with `EXPORT.json`.
3. CPU export check, then a bounded zero-optimizer GPU reload/finite-forward
   check on the exact BF16 export. These are artifact gates, not task success.
4. Stage create-once, register the route paused at zero replicas, then resume
   only after a fresh GPU-capacity check. Verify live model identity and exact
   base-versus-candidate harness parity before Fleet development pass@4.
   Pause the route and release its GPUs after use. Keep the final test set
   sealed for the predeclared checkpoint/protocol; repeatedly inspecting it
   would invalidate a confirmatory lift claim.

Historical proven CLI forms (from `c908d3a8`, **not present in the slim
September-2026 checkout until a small verified bridge restores them**) are:

```sh
cyber-post-train checkpoint-seal <exact-prepared-dir> <S> --output <run>/checkpoint-seals-v1/step-<S>.json
cyber-post-train checkpoint-export <seal.json> --sha256 <seal-file-sha256> --output <run>/hf-export-step<S>-v1
cyber-post-train checkpoint-check <export>/EXPORT.json --sha256 <export-file-sha256> --output <new-cpu-check.json>
cyber-post-train checkpoint-check <export>/EXPORT.json --sha256 <export-file-sha256> --output <new-gpu-check.json> --gpu
```

CPU seal/export/check run in bounded zero-GPU c1 Pods with the SFS mount.
GPU check and trainer use the Jobs API: duplicate census, `POST /v1/runs/preview`,
verify rendered root `RayJob` has `fleet.ai/failure-alerts: "off"`, c1/q1,
exact image/resources/output, then one create-only `POST /v1/runs` with a
durable intent journal. Serving registration uses the inference control plane
`POST /fleet/v1/models` paused; resume/pause are resource-version-bound.
The old staging module is Fresh75-specific and must not be reused unchanged.

If a real project storage cap rules out keep-all, the *fallback* is a planned
pause at each interval, CPU seal/export while the source run is stopped, then
an exact native resume in a fresh output/W&B run while task eval proceeds.
Qwen full-weight step-1-to-step-2 pause/resume was proven on 2026-09-11, but
repeating it 18 times adds launch/reload overhead and more failure surfaces.
Do not merely run an asynchronous exporter beside `keep_checkpoints: 2`:
pruning can race it, and no current acknowledgment gate prevents deletion.
