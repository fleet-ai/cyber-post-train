# Qwen3.8-27B Fleet reward-calibration track

## Purpose

Qwen3.6-27B produced too little reward under the first native RL harness to
justify interpreting an all-zero training run as a learning result. This track
tests a stronger, newer 27B model on Fleet-owned, non-test blackbox tasks before
any full training allocation. It is a model/harness calibration, not yet a
post-training result.

## Frozen identity and protocol

- Served model: `qwen3.8-27b`.
- Underlying revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Runtime: SGLang, BF16 weights, TP=1, FP8 KV cache, 262,144-token context.
- Harness: Qwen Code 0.22.3 at source commit
  `09825973e7d3c3fd07e17909c396aa62f48ce51f`.
- Agent budget: at most 600 model requests and 10,800 seconds per task.
- Task tools: exactly `bash`, `submit_report`, in that order.
- Tool schema catalog:
  `sha256:85fad6bdc3a835bf52a11a99b3387740eb06eb3d1720ad9bb33f3feac215b44a`.
- Scoring: exact task-version deterministic verifier through the server-owned
  rollout-reward authority, binary aggregation, pass@1.
- Data: twenty exact `train`/`dev` versions across all six app families; zero
  rows from the sealed test20 split.

The gateway model path, catalog revision, server configuration, and synthetic
structured-tool call must all agree before task execution. The first selected
task is an in-protocol canary. Only a positive authoritative reward with a
verified cleanup receipt opens the remaining calibration. A valid zero or an
infrastructure error leaves the remaining bank untouched.

The create-only v1 canary stopped before agent execution or scoring because a
non-root macOS controller cannot change Docker Desktop bind-mount ownership to
uid 1000. It verified the exact task runtime and tool-schema digest, then closed
the environment and removed all containers. Commit `d29ced4` preserves that
plan. V2 runs the container as the invoking uid/gid on non-root Docker Desktop
controllers; root/cluster execution retains the image's uid 1000 behavior.

V2 was preregistered with an infrastructure-only canary gate: any authoritative
model outcome, including a valid zero, released the remaining bank. Its first
attempt returned an authoritative zero with verifier execution and complete
cleanup, after which the controller began later waves exactly as its code said.
That criterion conflicts with the subsequently merged Qwen3.8 training
qualification protocol, which requires positive reward before releasing a
larger calibration bank. V2 is therefore preserved as an as-treated descriptive
calibration, not hidden or rerun and not eligible to satisfy the training gate.
Stopping its controller would mutate live execution, so it was left untouched.

V3 makes the positive-reward criterion an exact, required configuration field.
Missing, altered, zero-valued, non-authoritative, or cleanup-incomplete outcomes
all fail closed. Its twenty versions are disjoint from V2 so already valid
pass@1 attempts cannot be silently repeated. V3 is a prepared create-only plan;
this change does not launch it.

## What is and is not matched

The task versions, environments, data, prompts, runtime seeds, two-tool surface,
deterministic verifiers, long interaction ceiling, and Qwen Code context policy
are pinned. Qwen Code's own automatic compaction is used. Its system prompt and
compaction implementation are not claimed to be byte-identical to the original
Fleet Agent Runtime, so this run estimates reward acquisition under the Qwen
Code 0.22.3 harness and does not directly estimate the native RL rollout rate.

Qwen3.6 attempt-1 outcomes may be compared only where the independently frozen
campaign has the exact same `task_version_id`. Missing matches remain missing
comparison evidence. Infrastructure-invalid outcomes are not converted to zeros
and valid pass@1 outcomes are never silently rerun.

## Launch boundary

The campaign consumes the existing routed inference endpoint and local harness
containers. It creates no training job, uses no GPU queue allocation, and does
not alter any peer workload. A full SFT or RL run remains separately gated on a
terminal calibration with useful nonzero reward and an exact trainable model
artifact/recipe.
