# Qwen3.8 long-horizon Miles/OpenCode implementation

Status: implemented and offline-tested; image build and cluster qualification have not run.
This record is not an RL-success or production-promotion receipt.

## Why the earlier canary cannot be promoted

The earlier production canary is configured by
`configs/qualification/qwen38-miles-rl-reward-canary-prod-v3.json`. It uses the
legacy direct agent in `training/rl_episode.py`, one 8-GPU node, a 98,304-token
context ceiling, and no context compaction. It can validate reward and optimizer
plumbing, but it cannot represent a long black-box exploit episode. Its output
must not qualify a full RL run.

## Qualified design

The successor uses the native FTI `qwen3.8-27b-256k` profile: 262,144 context
tokens, 245,760 response tokens, tensor parallel 8, context parallel 4, and four
8-GPU nodes. It runs exact OpenCode 1.18.27 through Miles session server v2 for
up to 2,048 model calls and eight hours. The Fleet environment lifetime is nine
hours, which remains the platform ceiling. Its job watchdog is separately bound
to nine hours (plus the existing five-minute checkpoint drain), so the generic
eight-hour RL watchdog cannot terminate a still-valid eight-hour episode.

The native profile keeps its measured radix cache enabled. The successor binds
`consistent_hashing`, so every turn reaches the engine holding that episode's
cached prefix; round-robin routing would repeatedly prefill long histories and
does not match the qualified FTI profile.

OpenCode compacts before overflow at 176,608 tokens. This follows from
262,144 context tokens minus a 32,768-token next response and 52,768 tokens of
reserved headroom. Summaries are limited to 4,096 tokens and preserve 15,000
recent tokens.

Every post-compaction primary-model history is recorded as a separate Miles
segment under the same rollout id and receives the one authoritative episode
reward. The default Miles postprocessor masks shared completion tokens so a
policy token is optimized once. Compaction summaries use a separate Miles
session: their text is visible to later primary-model requests, but the summary
tokens are deliberately excluded from the training objective. This prevents a
synthetic summary from being flattened behind a false old prefix or optimized
twice.

OpenCode receives normal MCP tool results. The legacy always-on 4,000-character
tool-result prefix truncation is absent. The OpenCode child process receives
only its local session credential and runner token; Fleet, W&B, and cloud
credentials are not copied into it. The task prompt is supplied over standard
input rather than exposed in the process command line.

FTI 0.8.4 still labels this native profile's TITO family as `qwen35`. The
profile's vendored `qwen3.8_fixed.jinja` bytes exactly match the template that
Miles resolves for `qwen38small`. The long-horizon path therefore binds
`qwen38small`, `preserve_thinking=true`, and `reasoning_effort=xhigh`, and
rejects a template or cross-field mismatch instead of trusting either name.
Image qualification also binds the inherited `qwen3` reasoning parser and
`qwen3_coder` tool parser.

Miles' default 1,024-node session limit is unchanged for existing jobs. The
derived image accepts an explicit bounded `MILES_SESSION_MAX_NODES=4096` value,
and rejects malformed or out-of-range values. Offline tests force repeated
compactions across more than 1,024 nodes.

## Exact source bindings

- FTI image base: `sha256:b713f93d8da719a08aa20f1c45752d32a40d72b95159f410cdb6046d5ed1cf5d`
- FTI version/profile: `0.8.4`, `qwen3.8-27b-256k`
- Miles commit: `2799fe386320c156334bf763ad4d7ca0f85dca4e`
- Original Miles session-tree file: `sha256:fd978a1ef2617f4bf30850fedd197e546cdc9c6542b00b03df502cbb285fc732`
- Installed native Miles driver: `sha256:34693c11173e69a402228bd1d732122eef9fdb6e5d105660dfdd5eae0b1b954c`
- Installed native checkpoint converter: `sha256:b5faeaf7bc80b7c98534ff763f43ae45768f6f93425f9cd536551a48ab462425`
- Patched installed session-tree file: `sha256:59bed80a62a8ab94e0bb9012f4f9f0290245a5c8db6feadd0997a7bfb57025ee`
- OpenCode tag/commit: `v1.18.27`, `4b7e19e315cca414121ba1d61523fef74bb3ae8b`
- OpenCode Linux binary: `sha256:bddf894e5c2bc3d8cf452bd6e5ab2273bbe4a37eeeb9aec848d3d7d20db1f256`
- Qwen3.8 fixed chat template: `sha256:38d42166599348d47ded69776c5389c89924045e6827089923a031379f8a3dfe`
- Derived-image source bundle: `sha256:56077adb7471a36022f5fe543830bbeab2265604079916614370d333d5fa018e`

## Remaining gates

1. Build and publish the derived image by immutable digest.
2. Inside that image, verify installed source/binary/template hashes, parse the
   native four-node profile, and repeat the greater-than-1,024-node compaction
   tests.
3. Create an exact zero-step native checkpoint using that same image.
4. Run one c1 dev canary on the exact four-node topology. It must prove repeated
   compaction, one authoritative reward per rollout, reward variance, a finite
   nonzero update, a durable checkpoint, and clean release.
5. Reload the checkpoint with zero optimizer updates. Only then may a fresh c1
   production canary be prepared. Full RL remains closed until that canary passes.

The machine-readable, deliberately unaccepted qualification template is
`configs/qualification/qwen38-miles-opencode-long-context-runtime-v1.template.json`.
