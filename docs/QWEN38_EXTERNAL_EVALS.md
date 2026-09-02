# Qwen3.8-27B external-evaluation track

## State

**Current operational classification:** v2 is proven infrastructure-interrupted
and unscored; all 15 trials remained planned and no model call occurred. V3 is
a frozen, single-use successor with merged durability and runtime-root gates,
but only unpaid preview is authorized. It has not launched. The time-stamped,
sanitized consolidation is
[`OVERNIGHT_OPERATIONAL_STATE_2026-09-01.md`](OVERNIGHT_OPERATIONAL_STATE_2026-09-01.md).

The exact base is `Qwen/Qwen3.8-27B` at revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. The Fleet route
`qwen3.8-27b` was observed Ready 1/1 with BF16 weights, tensor parallelism 1,
the same digest-pinned SGLang image used by Qwen3.6, 262,144-token context,
FP8 KV cache, TensorRT-LLM MHA, `qwen3` reasoning parsing, and
`qwen3_coder` tool parsing. A forced structured-tool request returned exactly
one valid call.

The following are frozen:

- the official model revision, 18-shard weight manifest, tokenizer manifest,
  chat template, and live serving-spec digest;
- WebExploitBench Level 0 with CAGE
  `09a191c565230cebb8255899d622d23c7ddeff33`, Qwen Code 0.22.3, the same
  linux/amd64 harness image, 15 targets / 110 vulnerabilities, pass@1,
  one-target concurrency, 262,144-token student context, 65,536-token GLM-5.3
  judge context, and the existing grading and retry policy;
- ExploitGym's exact five task identities through the immutable Qwen3.6 source
  protocol digest, together with the independently verified control image
  `ghcr.io/fleet-ai/cyber-post-train-exploitgym-control@sha256:466027a5b270e822acbba676ccd775f443e7ed897fdf2d5a3f824f776141e7a0`,
  dynamic graders, firewall, pass@1, single-worker, and 3,600-second budget.

## Interpretation boundary

Qwen3.8-27B was released on 2026-08-14, after WebExploitBench. This track can
measure a controlled delta between exact Qwen3.8 base and post-training
checkpoints. It is not a temporally clean holdout claim, because benchmark
contamination cannot be ruled out. External scores and traces remain sealed and
evaluation-only; none may influence training, rewards, checkpoint selection,
or prompts.

## Launch gates

Run the static control check, then the inexpensive live route/chat/tool probe:

```sh
uv run python -m evals.qwen38_study validate
uv run python -m evals.qwen38_study probe --output /restricted/qwen38-live-preflight.json
```

The WebExploitBench scored base arm may launch only from its single-use absent
run root with the pinned local harness image. ExploitGym remains prepared but
unlaunched until a post-training checkpoint exists, so its two arms can be
counterbalanced and run close in time under one final protocol. A successful
probe is an operational gate, not a capability result.

The first WebExploitBench launch root is preserved as interrupted and unscored.
Review found that the model lock read a nonexistent Hugging Face `lfs.sha256`
field instead of the authoritative `lfs.oid`, so the launch was stopped before
any terminal task result. The corrected weight-manifest digest is
`sha256:06c94e47c0e31fd331ed410665c830ab1b657f90f15a1b11e7bc45e2de00f352`;
the successor used the v2 config, protocol, and run id.

The v2 controller was launched under a transient terminal session and later
disappeared while all 15 trials were still `planned` at CAGE's unchanged host
memory gate. It produced zero model calls, events, resources, or terminal
outcomes and had no durable exit receipt, so its exact terminating signal or
actor cannot be reconstructed. It is permanently preserved as interrupted and
unscored by
`docs/evidence/webexploitbench/2026-09-01-qwen38-v2-interrupted-unscored.json`.
It must never be resumed, scored, or reused.

The v3 recovery changes only the single-use `run_id`; its protocol changes only
the resulting experiment-config digest. `launch-qwen-recovery` revalidates that
identity, the byte-identical v2 evidence, CAGE checkout, and local harness image
before creating a detached supervisor. Recovery evidence is create-only. The
supervisor first acquires a permanent single-owner receipt, then writes separate
start and exit receipts; even a zero process exit
is classified as unscored process evidence until the normal sealed evaluation
result path proves a scientific outcome.

The launch claim seals a deterministic manifest of every Python source file in
the WebExploitBench evaluator package, including the module entry point, CLI,
monitor, and recovery implementation. The detached child recomputes the full
manifest before starting CAGE. Owner, start, and exit receipts carry validated
self-digests and form a digest chain back to the launch claim. Every create-only
write is synced in file-then-parent-directory order; a sync failure is a launch
failure, not a warning.

Dry-run preparation is unpaid and creates nothing:

```bash
python -m evals.webexploitbench \
  --config evals/webexploitbench/configs/qwen38-27b-1d4bf0f2-level0-qwen-code-full-v3.json \
  launch-qwen-recovery \
  --cage-dir evals/webexploitbench/.workbench/CAGE \
  --predecessor-config evals/webexploitbench/configs/qwen38-27b-1d4bf0f2-level0-qwen-code-full-v2.json \
  --predecessor-protocol evals/webexploitbench/manifests/qwen38-27b-qwen-code-protocol-v2.json \
  --protocol-manifest evals/webexploitbench/manifests/qwen38-27b-qwen-code-protocol-v3.json \
  --incident-receipt docs/evidence/webexploitbench/2026-09-01-qwen38-v2-interrupted-unscored.json
```

The command above is a static preview and cannot execute. A credential-ready
preview additionally supplies the same sanitized, self-digested rotation
receipt used by the Qwen3.8 Fleet-50 launch:

```bash
python -m evals.webexploitbench \
  --config evals/webexploitbench/configs/qwen38-27b-1d4bf0f2-level0-qwen-code-full-v3.json \
  launch-qwen-recovery \
  --cage-dir evals/webexploitbench/.workbench/CAGE \
  --predecessor-config evals/webexploitbench/configs/qwen38-27b-1d4bf0f2-level0-qwen-code-full-v2.json \
  --predecessor-protocol evals/webexploitbench/manifests/qwen38-27b-qwen-code-protocol-v2.json \
  --protocol-manifest evals/webexploitbench/manifests/qwen38-27b-qwen-code-protocol-v3.json \
  --incident-receipt docs/evidence/webexploitbench/2026-09-01-qwen38-v2-interrupted-unscored.json \
  --credential-rotation-receipt /restricted/rotation-receipt.json
```

The bridge reads only the live Secret UID, resourceVersion, identity, and
expected-key presence for preview. It requires an exact match to the receipt's
post-rotation metadata and requires the confirmed rotation time to postdate
`2026-09-01T23:39:05Z`. On an approved `--execute`, it rejects an ambient
`FLEET_API_KEY`, atomically reads the same metadata plus the encoded value from
`fleet-train-jobs/fleet-api`, and decodes the value only in process memory. The
detached supervisor receives it through an explicit allowlisted environment;
ambient cloud, source-control, Python-path, and other credential variables are
not forwarded. The hidden supervisor entry point requires a random one-time
parent capability, represented only by its digest in the launch claim and
receipt chain. The CAGE child receives the Fleet
key but not that supervisor capability. The execute function has no ambient-key
fallback and refuses missing bound rotation evidence. The value is never
accepted as an argument, printed, or written into a claim, command, log setup,
or receipt; the durable launch claim binds the rotation receipt digest.

An authorized operator may add `--execute` only after the credential owner has
actually rotated the Secret, the reusable receipt validates, explicit launch
approval is present, and the credential-ready preview has been reviewed.
Existing v3 run or recovery paths, any Secret metadata drift, any v2 evidence
drift, and any config/protocol/image drift all fail closed. Do not invoke CAGE's
raw `--resume` path.
