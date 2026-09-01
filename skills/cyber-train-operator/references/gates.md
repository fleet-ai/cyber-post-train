# Training gate checklist

Use this checklist at the transition named in the request. Do not execute later stages merely because an earlier stage passed.

## Plan and data

- Exact base model revision, shard manifest, tokenizer, and chat template are immutable.
- Train/dev/test split is lineage-safe; external benchmarks are absent from training and recipe selection.
- Run config, dataset, trainer, reward, evaluation protocol, and expected output identities have digests.
- Model adapter proves architecture, precision, topology, and export compatibility.

## Preview and admission

- Authenticated server preview is HTTP-successful and has no errors.
- Preview binds exact task versions, environment versions, verifiers, ordered tools, step semantics, resources, queue, image, and output roots.
- Planned session and GPU cost match the requested canary or run.
- Duplicate title and create-only destination checks are clean.
- Scheduler admission is allowed to wait; never bypass or displace peer jobs.

## Runtime

- Record controller and Pod UIDs before cleanup.
- Record requested image and resolved image ID, ConfigMap/Secret references without secret values, queue, resources, and ownership labels.
- Prove dataset/task chunks independently in head and worker when transport transforms them.
- Observe real rollouts, verifier calls, optimizer events, and checkpoints separately.

## Checkpoint handoff

1. Prove source checkpoint identity and optimizer-step count.
2. Export without training.
3. Inspect actual tensor dtype, shape, count, and hashes.
4. Convert deterministically if required; never relabel bytes.
5. Assemble exact runtime sidecars from the frozen base.
6. Stage with atomic no-replace semantics and validate the embedded acceptance receipt.
7. Register idempotently using a pullable digest and reviewed image-pull configuration.
8. Require registration-completion and live-parity receipts before evaluation.

## RL reward-acquisition canary

- Use exact authoritative task versions with a known nonzero historical signal.
- Match the task-facing tool contract and enforce it before tool execution.
- Give the student enough trajectory and turn budget to reach report submission.
- Retain exact authoritative verifier execution IDs per episode.
- Require real rollouts, at least one optimizer step, and a durable checkpoint.
- Interpret all-zero valid rewards as a canary result; interpret truncation or evidence failure as a harness/configuration defect.
