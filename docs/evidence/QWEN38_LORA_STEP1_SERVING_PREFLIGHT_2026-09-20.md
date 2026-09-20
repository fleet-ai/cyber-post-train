# Qwen3.8 LoRA step-1 serving preflight

This handoff prepares the accepted one-step LoRA model for later matched
evaluation without registering, resuming, serving, or evaluating it. The sealed
machine-readable plan is
[`configs/qualification/qwen38-lora-step1-serving-preflight-v1.json`](../../configs/qualification/qwen38-lora-step1-serving-preflight-v1.json).

## What was checked read-only

The accepted export receipt was read from the development SFS path
`/jobs/chris-q38-lora-export-c1-v5/QWEN38_LORA_MERGED_HF_EXPORT.json`. Its file
SHA-256 is
`3da93b3304e284205ba8a8939d1dcb8cc4fe749b0f255bbbe35376af5e7c5ff2`,
matching the accepted evaluation packet. It names 28 model files totaling
55,586,036,499 bytes. Their canonical manifest SHA-256 is
`f08cdb03fa6ef79228028417bec978a509275bbb1ebe7d99d5db4a00ae9fad84`.
The receipt says that the merge was deterministic, all tensors reopened equal,
the complete model and tokenizer reloaded on GPU, logits were finite, no
optimizer step ran during export, and both the source checkpoint and base model
remained unchanged. This check did not read prompts, traces, flags, answers, or
scores.

The safest current Qwen3.8 serving template is the paused c1 route
`chris-q38-fresh75-step230-wbe-v1`, UID
`fa7dafb6-a700-4eb7-b955-e40255420617`. A read-only control-plane observation
found it paused with zero requested replicas and exact spec digest
`sha256:cdad3122886c0670f841169d2b2f69eb67d35a03b0c781a7f13847c8557eac19`.
That source must be read again immediately before any later registration; this
document is not permission to reuse a changed template.

## Exact intended lifecycle

1. Re-read and fully hash the accepted export and its independent validation
   receipt.
2. Prove `/models/chris-q38-lora-step1-wbe-v1` does not exist.
3. Stream the 28 exact files into a create-once transaction using a bounded,
   zero-GPU c1 Pod, atomically publish the destination, and verify every byte
   after publication.
4. Re-read the paused c1 Qwen3.8 template and require its UID and full spec to
   match the sealed source observation.
5. Create exactly one new route named `chris-q38-lora-step1-wbe-v1`, initially
   paused with zero replicas. Seal the returned new UID. Never fall back to an
   older or differently named route.
6. Only after one project node is available, resume one 8-GPU replica, bind its
   exact Pod/Workload identity, prove Ready with zero restarts, and capture
   content-free live parity against base `qwen3.8-27b`. Generated text is checked
   in memory and is not written to evidence.
7. Start a useful matched evaluation consumer promptly. Pause the exact route
   and prove its GPU release when the consumer completes, fails, or stops making
   progress.

## Why it is not launchable yet

The current production streaming stage accepts the older Fresh75 full-export
receipt schema, not the LoRA merged-export schema. Reusing it would fail closed
before staging, so that contract must be extended and tested first. No LoRA
stage receipt or candidate route exists yet, no fresh live-parity receipt exists,
and the production project is currently using all eight allowed GPU nodes. The
plan records all five facts explicitly; removing any of them without new
evidence fails validation.

No external mutation was performed while producing this preflight.
