# Qwen3.8 long-horizon Miles/OpenCode implementation

Status: runtime-qualified. The corrected immutable image passed both a
zero-GPU full-driver-import gate and a bounded real-CUDA native-parser gate.
Both dev probes were deleted immediately after their terminal receipts and no
dev GPU remains allocated. This record is not an RL-success or
production-promotion receipt: real reward, an optimizer update, checkpoint
creation, and checkpoint reload remain separate gates.

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

FTI 0.8.4 labels this native profile's TITO family as `qwen35` but selects its
distinct `qwen3.8_fixed.jinja` template. Installed Miles correctly rejects that
combination because a named TITO family owns exactly one fixed template. The
base image's Miles checkout (`9e178ca1`) missed upstream commit `257992eb`,
which adds `qwen38small`; that upstream template is byte-for-byte identical to
FTI's Qwen3.8 template. The derived image backports those two exact upstream
files and binds `preserve_thinking=true`, `reasoning_effort=xhigh`, the `qwen3`
reasoning parser, and the `qwen3_coder` tool parser.

Miles' default 1,024-node session limit is unchanged for existing jobs. The
derived image accepts an explicit bounded `MILES_SESSION_MAX_NODES=4096` value,
and rejects malformed or out-of-range values. Offline tests force repeated
compactions across more than 1,024 nodes.

## Exact source bindings

- FTI image base: `sha256:b713f93d8da719a08aa20f1c45752d32a40d72b95159f410cdb6046d5ed1cf5d`
- FTI version/profile: `0.8.4`, `qwen3.8-27b-256k`
- Installed Miles commit: `9e178ca16839b0600155f3927f57ce0670b8f453`
- Qwen3.8 TITO backport: `257992eb52bfa1f5248b5a5ae8f5a959be500788`
- Original Miles session-tree file: `sha256:fd978a1ef2617f4bf30850fedd197e546cdc9c6542b00b03df502cbb285fc732`
- Installed native Miles driver: `sha256:85dbfd31d41a84f9c2e79a2918583851fb53925630afa229e9cd0a154b170f46`
- Installed native checkpoint converter: `sha256:0c2541d30073777a30344273a3773844a70ca1961287520c0496a1cec18d43f6`
- Patched installed session-tree file: `sha256:59bed80a62a8ab94e0bb9012f4f9f0290245a5c8db6feadd0997a7bfb57025ee`
- Patched Megatron Qwen3-ASR source: `sha256:90ebb373c06195ad4a8d117e17ed154ffc67c78ccc370fe7f5dcab78ae864355`
- Superseded pre-backport image: `sha256:04c1b4ed1c6faebba80fb0220496af0994f52a89a702638083420905a15724bf`
- Qualified derived image: `sha256:dc1a41ac386c9f7377e7a6f7b92a402830e308855aa2632413af25917f38dd93`
- Image source commit: `3d97b6bbb30fefa97f08959b0ce9b50c86cd947f`
- Qualification source commit: `75afd1241d1042855f793a4069c321d6ee9e452b`
- OpenCode tag/commit: `v1.18.27`, `4b7e19e315cca414121ba1d61523fef74bb3ae8b`
- OpenCode Linux binary: `sha256:bddf894e5c2bc3d8cf452bd6e5ab2273bbe4a37eeeb9aec848d3d7d20db1f256`
- Qwen3.8 fixed chat template: `sha256:38d42166599348d47ded69776c5389c89924045e6827089923a031379f8a3dfe`
- Corrected derived-image source bundle: `sha256:441853f27031f2e9c60c70a04e862080bc815c5eaf191c82f183f9824d469a72`

## Qualification evidence before the backport

These probes are defect-discovery evidence, not successful runtime
qualification. They used no Fleet task, generated no reward, and performed no
optimizer update.

- BuildKit gate `q38-miles-runtime-qual-a3c379c0` (Job UID
  `cdce05af-ad51-49af-bdcc-337c81d0d656`) cleanly proved that the installed
  enum did not contain `qwen38small`.
- BuildKit gate `q38-miles-runtime-qual-b4fb7162` (Job UID
  `1dd1f70e-9860-40e5-a937-ddb59dffde48`) reached the native Megatron import
  and then proved that its full parser requires a real CUDA runtime. This is why
  the immutable image gate is zero-GPU source validation and the parser gate is
  a separate bounded GPU probe.
- Direct clean-exit parser Pod `q38-miles-runtime-gpu-qual-d43e57f2` (UID
  `2427d850-c81a-4daa-960c-4e2f3ce5b2a6`) used one GPU and rejected a missing
  exact model path. Direct clean-exit parser Pod
  `q38-miles-runtime-gpu-qual-7cc25f64` (UID
  `ad1b6e01-9c0a-423b-9748-f8de7f0d1393`) used the exact model config and
  rejected the invalid `qwen35` plus Qwen3.8-template combination. Both Pods
  exited `Succeeded`, restarted zero times, and released their GPU; neither was
  a training Job or Slack-alerting failed Job.
- Zero-GPU source-inspection Pods bound the base checkout and source files:
  `q38-miles-tito-contract-inspect-v2` (UID
  `f5bd23b7-c543-49e3-8098-1ee761ce9ebe`),
  `q38-miles-template-contract-inspect-v2` (UID
  `460cb09d-6fd2-47a0-9b7f-138e5e5a9b34`), and
  `q38-miles-source-hash-inspect-v1` (UID
  `df6492e1-5db7-4169-af3e-0d646dc0515b`). Each succeeded with zero restarts.

The minimal correction is exact upstream Miles commit `257992eb`: install its
`qwen38small` tokenizer registration and byte-identical Qwen3.8 template, pass
`--tito-model qwen38small`, and omit `--chat-template-path`. The named family
then owns and automatically resolves the template, `preserve_thinking=true`,
`reasoning_effort=xhigh`, reasoning parser `qwen3`, and tool parser
`qwen3_coder`. No local model family or alternate template is invented.

## Accepted runtime qualification

The immutable BuildKit image completed at 2026-09-15T06:24:26Z with exit zero,
zero restarts, and no GPU. Its exact Job and Pod were deleted and their absence
was verified. A zero-GPU CPU probe then imported the complete native driver and
checked the installed hashes and compaction behavior. It deliberately did not
claim the native 256K parser, because that parser queries a real CUDA device.

The bounded CUDA probe `q38-miles-runtime-gpu-qual-75afd124` (Pod UID
`bd9d6722-d6a5-44b1-aa17-27fbde972203`) used one dev GPU under c1 with a
600-second hard deadline. It ran for 34 seconds, exited zero with no restart,
and validated the real-CUDA native driver, the complete four-node 256K parser,
the Qwen3.8 TITO/template binding, exact OpenCode binary, repeated compaction
through more than 1,024 nodes, one episode reward across compaction segments,
summary-token exclusion, and untruncated primary tool results. Its receipt
digest is
`sha256:0597e05ee1d6557fe5f3ebc2cda80bcb4cf92db4ac93863b767df2084d7dac43`.
The exact Pod and immutable ConfigMap were deleted immediately; a readback
found zero active `q38-miles` dev Pods or GPUs.

The machine-readable receipt is
`configs/qualification/qwen38-miles-opencode-long-context-runtime-v1.json`.
Exact operational identities and release evidence are in
`docs/evidence/qwen38-study/2026-09-15-miles-long-context-runtime-qualification-v1.json`.

## Remaining gates

1. Complete exact output, duplicate, and aggregate node-cap checks for one c1
   four-node reward-acquisition canary. Dev may only be used for bounded tests;
   it must never host a durable service or an unattended long run.
2. Run the canary against real Fleet tasks and the authoritative verifier. It
   must show non-constant real reward before a finite nonzero optimizer update,
   then write a durable checkpoint and release all four nodes.
3. Reload that exact checkpoint with zero optimizer updates. Only after the
   reload gate passes may a longer production arm be prepared. Full RL remains
   closed until these scientific gates pass.

The original qualification plan remains in
`configs/qualification/qwen38-miles-opencode-long-context-runtime-v1.template.json`.
