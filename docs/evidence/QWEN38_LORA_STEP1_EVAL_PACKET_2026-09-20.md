# Qwen3.8 LoRA step-1 evaluation packet

Status: **training and merged export accepted on development; serving and
evaluation not launched**.

The one-step LoRA run `chris-q38-lora-sft-c1-v10` finished successfully on the
development cluster, produced optimizer step 1, and released its eight GPUs.
Its checkpoint receipt was independently validated. A later zero-update export
merged the adapter into the exact Qwen3.8-27B base layout, reloaded the complete
model and tokenizer in a fresh process, checked finite output, and passed an
independent exact-image verifier. Development GPUs were released after both
runs.

The accepted merged model is at
`/mnt/sfs/jobs/chris-q38-lora-export-c1-v5/merged-hf`. This is an accepted model
artifact, not yet an accepted serving deployment or a capability result.

## Sealed evaluation packet

[`qwen38-lora-step1-eval-packet-v1.json`](../../configs/evaluation/qwen38-lora-step1-eval-packet-v1.json)
binds:

- the exact training Pods, pinned training image, request and plan digests,
  checkpoint receipt, merged export manifests, and independent validation;
- the exact Qwen3.8-27B base revision, tokenizer, chat template, weight
  manifest, and current base route UID;
- one unused create-once candidate route name,
  `chris-q38-lora-step1-wbe-v1`, initially paused at `c1` with zero replicas;
- the same pinned OpenCode 1.18.27 WebExploitBench tasks and score-free
  collection protocol as the Fresh75 study;
- one matched task-0 pass@1 canary, followed only after acceptance by four
  independent 15-task pass@1 replicas per arm (logical pass@4); and
- the frozen Fleet development 17-task pass@1 set while keeping the final
  eight-task set closed until a single checkpoint is selected.

Collection and scoring stay separate. The rollout bundles are immutable and
resumable; judge failures therefore do not require another model rollout. A
historical base bundle may be reused only when its immutable receipt proves
every non-weight control is identical. Similar names are not proof.

## Exact remaining gate

Before any rollout starts, the operator must:

1. re-materialize and rehash the accepted checkpoint and merged-export
   receipts;
2. prove the reserved route name is absent, create it once in a paused state,
   and seal the returned UID;
3. admit its one eight-GPU replica only when total project use remains at or
   below eight active nodes;
4. stage and reload the merged model, then prove live base-versus-candidate
   equality for tokenizer, chat template, precision, serving image and
   arguments, structured tools, and deterministic probes;
5. run a fresh TensorLake duplicate census; and
6. accept the matched task-0 score-free canary before any 15-task expansion.

No serving route, TensorLake sandbox, Fleet session, or scoring job was created
while preparing this packet.
