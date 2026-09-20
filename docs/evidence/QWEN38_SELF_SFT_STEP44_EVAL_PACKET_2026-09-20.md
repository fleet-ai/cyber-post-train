# Qwen3.8 self-SFT step-44 evaluation packet

Status: **checkpoint accepted; serving and evaluation not launched**.

The repository evidence census contains one accepted self-SFT checkpoint:
`q38-self-sft-step44`. Training completed 44 optimizer steps from 35 verified
successful Qwen sessions spanning 20 Fleet tasks. The accepted source contains
175 supervised windows and 45,676 supervised assistant tokens. The checkpoint
payload was independently rehashed, its BF16 export was produced without an
optimizer update, and a one-GPU synthetic forward produced finite output while
leaving the source unchanged.

The checkpoint is not yet ready for a capability claim. Its old GPU check was a
synthetic reload, not an inference-serving qualification. No current serving
route, base-versus-candidate live-parity receipt, matched WebExploitBench
comparison, or accepted Fleet held-out result exists.

## Sealed packet

[`qwen38-self-sft-step44-eval-packet-v1.json`](../../configs/evaluation/qwen38-self-sft-step44-eval-packet-v1.json)
binds:

- the exact training run, checkpoint, export, checkpoint-seal receipt, zero-step
  reload, and independent verifier identities;
- the exact Qwen3.8 base lock and currently sealed base route UID;
- one new create-once candidate route name,
  `chris-q38-self-sft-step44-wbe-v1`, initially paused at `c1` with zero
  replicas;
- the same pinned OpenCode 1.18.27 WebExploitBench benchmark, CAGE revision,
  accepted 15-task TensorLake snapshot, task order, and score-free collection
  runner used by the Fresh75 matched study;
- a one-task paired canary followed, only after acceptance, by four independent
  15-task pass-at-one replicas per arm. Together those replicas form logical
  pass@4 without coupling rollout collection to scoring;
- all 17 exact Fleet development task-version IDs at pass@1; and
- all eight final-test task-version IDs in a closed state. The final set may be
  opened once, only after one checkpoint has been selected on development data.

The common base collection may be reused only when an immutable accepted bundle
proves exact equality of every non-weight control. Otherwise the packet requires
a fresh duplicate census and one create-once base collection. Similar names or
historical aggregate scores are not enough.

## Exact remaining blocker

The immediate blocker is serving qualification, not checkpoint recovery. Before
any rollout can start, the operator must:

1. materialize and rehash the exact checkpoint manifest and export receipt;
2. prove the candidate route name is absent, then create it once in a paused
   state and preserve the returned UID;
3. wait until admitting its one eight-GPU replica keeps total project use at or
   below eight active nodes;
4. reload the export and produce a fresh base-versus-candidate live-parity
   receipt covering tokenizer, chat template, precision, serving image and
   arguments, structured tools, and deterministic finite probes; and
5. run a fresh TensorLake duplicate census immediately before the paired canary.

No route, TensorLake sandbox, Fleet session, or scoring process was created by
this packet. It is an inert, fail-closed plan.
