# Miles + OpenCode long-context image

This is a deliberately small derived image for long-horizon Qwen3.8 RL. It
starts from the immutable FTI 0.8.4 Miles image, installs the exact Linux
OpenCode 1.18.27 executable, and raises Miles session-v2's node cap only when
`MILES_SESSION_MAX_NODES` is explicitly set. The old default remains 1024.

The build fails unless the installed Miles tree, native training driver,
distributed-initialization sources, Megatron optimizer, and native checkpoint
converter have the exact reviewed Git/SHA-256 identities. It fetches
the official OpenCode release archive and checks both the archive and extracted
binary digests. This keeps Git-based BuildKit builds reproducible without
committing the binary or relying on a local-context upload. Publish the resulting
image by digest, then record an exact zero-GPU source check and a bounded dev
CUDA parser/qualification receipt before any production configuration may refer
to it.

Megatron Bridge 0.5.0 eagerly imports its vendored Qwen3-ASR model registry even
for text-only Qwen training. Transformers 5.12.1 reports two missing
`cache_position` argument docs in that vendored file. The image applies an
exact-source, exact-output two-docstring patch so the normal validator remains
enabled; it does not disable or monkeypatch the validator.

Required qualification:

1. default import reports 1024 nodes; `MILES_SESSION_MAX_NODES=4096` reports
   4096; malformed/out-of-range values fail at import;
2. a forced offline session creates more than 1024 nodes and at least three
   disjoint compaction segments without truncated nodes;
3. the exact `qwen3.8-27b-256k` arguments parse as TP8 x CP4 on 4 x 8 GPUs,
   262144 context, 245760 response, 65536 train tokens per GPU, radix-cache
   affinity via `consistent_hashing`, and the upstream `qwen38small` TITO
   family with the exact Qwen3.8 profile template and parsers;
4. the complete native driver imports through a CUDA-stub process with no
   undocumented-signature findings before any GPU allocation;
5. an exact 4 x 8, 32-rank clean-exit qualification uses the hybrid
   `cpu:gloo,cuda:nccl` world, initializes the complete Qwen model and Megatron
   optimizer through its parameter-group object collective, performs no
   rollout/reward/update/checkpoint write, and releases all 32 GPUs;
6. only then may a reward canary prove repeated native OpenCode compaction, one
   authoritative reward per original rollout, finite nonzero update, durable
   checkpoint, release, and zero-update reload.

The base image's installed Miles checkout is `9e178ca1`; its Fleet branch
missed upstream Qwen3.8 TITO support even though FTI already ships the Qwen3.8
profile. The build backports exact upstream commit `257992eb` and refuses any
different checkout or source-file hash.

The 32-rank gate is required because a one-process CUDA parser check cannot
exercise distributed optimizer construction. The first 4 x 8 canary reached
that stage with a CUDA-only default world and failed its Python-object gather
before any rollout. The proposed hybrid default leaves CUDA tensor collectives
on NCCL while routing default-world object metadata over Gloo; it remains
unqualified until the exact 32-rank gate passes.
