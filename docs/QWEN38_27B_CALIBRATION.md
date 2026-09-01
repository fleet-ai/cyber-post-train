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
task is an in-protocol canary. A valid zero counts as a valid model outcome and
opens the remaining calibration; an infrastructure error stops the campaign.

The create-only v1 canary stopped before agent execution or scoring because a
non-root macOS controller cannot change Docker Desktop bind-mount ownership to
uid 1000. It verified the exact task runtime and tool-schema digest, then closed
the environment and removed all containers. Commit `d29ced4` preserves that
plan. V2 runs the container as the invoking uid/gid on non-root Docker Desktop
controllers; root/cluster execution retains the image's uid 1000 behavior.

## What is and is not matched

The task versions, environments, data, prompts, runtime seeds, two-tool surface,
deterministic verifiers, long interaction ceiling, and Qwen Code context policy
are pinned. Qwen Code's own automatic compaction is used. Its system prompt and
compaction implementation are not claimed to be byte-identical to the original
Fleet Agent Runtime, so this run estimates reward acquisition under the Qwen
Code 0.22.3 harness and does not directly estimate the native RL rollout rate.

The selected versions are shared with the independently frozen Qwen3.6
calibration. Once that campaign is terminal, attempt-1 outcomes may be compared
by exact `task_version_id`. Infrastructure-invalid outcomes are not converted
to zeros and valid pass@1 outcomes are never silently rerun.

## Launch boundary

The campaign consumes the existing routed inference endpoint and local harness
containers. It creates no training job, uses no GPU queue allocation, and does
not alter any peer workload. A full SFT or RL run remains separately gated on a
terminal calibration with useful nonzero reward and an exact trainable model
artifact/recipe.
