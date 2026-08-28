# GLM-5.2 cyber post-training stream

This directory implements the reproducible boundary around training: read-only
Fleet export, lossless tool-trajectory normalization, lineage/leakage tracking,
SFT and preference materialization, online-RL prompts, reward composition, and
a fail-closed model-compatibility/run-plan gate.

It does **not** claim that a served FP8 endpoint is a trainable checkpoint.
GLM-5.2's official BF16 checkpoint is the optimizer source; the exact revision
is pinned at commit `b4734de4facf877f85769a911abafc5283eab3d9`.
NVIDIA merged native GLM-5.2 Megatron/vLLM GRPO support in NeMo RL commit
`63e620046c67f922c4a57dcb65d7e6fceb60f5d4`, including its required vLLM 0.25.1
compatibility patch. That upstream recipe was validated at much larger H100
topologies than our presently free capacity, so the compatibility receipt still
requires an independent B300 topology proof before training.

## Data flow

```text
Fleet completed jobs
  -> private raw export (transcript route only; no grader stdout/source)
  -> normalized immutable trajectories
     -> verified successes -> leakage-safe SFT train/dev/test
     -> matched success/failure -> optional preference data
     -> every valid task -> fresh online GRPO environment prompt
  -> manifest + exact digests
  -> compatibility-gated SFT -> pre-RL eval -> GRPO -> post-RL eval
```

Use credentials only through the environment or the cluster secret store:

```bash
python -m training export \
  --job-id a62dd51f-a52b-4941-8207-4679e4b25b51 \
  --output /secure/fleet/raw-sessions.jsonl

python -m training normalize \
  --input /secure/fleet/raw-sessions.jsonl \
  --output-dir /secure/fleet/glm52-v1

python -m training plan \
  --config training/configs/glm52_all_fleet.json \
  --manifest /secure/fleet/glm52-v1/manifest.json \
  --compatibility-receipt /secure/receipts/glm52-nebius.json \
  --evaluation-protocol /secure/protocols/glm52-cyber-v1.json \
  --output /secure/plans/glm52-fleet-v1.json

# Dry-run is the default; no cluster process starts.
python -m training run \
  --plan /secure/plans/glm52-fleet-v1.json \
  --work-dir /secure/runs/glm52-fleet-v1
```

For full historical discovery, use `--discover-completed-cyber
--created-after <ISO timestamp>`. The discovery path fetches completed jobs,
then retains only jobs containing a blackbox task key. An explicit lower bound
is required so a typo cannot issue an unbounded historical export.

## Training choices

- Start with shared-expert LoRA on attention and expert MLP projections. It is
  recoverable, small to checkpoint, and permits fast rank/target ablations.
- Run one epoch of verified-success SFT. A deterministic split unit binds the
  application, vulnerability family and exact task lineage, so sessions from a
  lineage cannot cross train/dev/test. All exported data remains accounted for.
- Use failures as matched preference negatives and, most importantly, as fresh
  online-RL prompts. Never imitate an unverified failing action sequence.
- GRPO reward is `verifier_score × behavior_score × integrity_gate`.
  Infrastructure failures and grader/reward hacking are always zeroed.
- Keep WebExploitBench outside this pipeline. Its terms make it an evaluation
  holdout, not training or reward-design data.

The causal ablation is base, SFT-only, RL-from-base, and SFT→RL when compute
allows. This separates tool-format adaptation from verifier-grounded capability
learning. Recipe selection uses lineage-held-out Fleet tasks; after the recipe is
frozen, a final all-data fit may be measured only on untouched external
benchmarks. Training-distribution scores are reported separately and are never
presented as generalization.

## Exact pre/post identity

`training.science` makes the evaluation protocol a hard input to the training
plan. Both arms must share one protocol digest binding the base checkpoint shard
manifest, tokenizer files, chat template, inference image and engine commit,
precision/quantization recipe, harness image and commit, tool schema, system
prompt, benchmark tasks/environments/prompts/verifiers, sampling parameters,
budgets and seeds. The comparison receipt permits exactly one changed variable:
checkpoint or adapter bytes.

Current hosted Fleet GLM-5.2 jobs are operational shakedowns because the hosted
route does not expose those exact artifact identities. A publishable experiment
will self-host both base and intervention arms from the same pinned checkpoint
and serving image; it will not compare a hosted FP8 base to a differently served
post-trained model.

## Live-launch gate

Copy `compatibility_receipt.example.json` and prove every check on the exact
Nebius topology. The planner refuses a missing check, mutable model revision,
or mismatched dataset digest. After the gate passes, bind the sealed plan to a
pinned NeMo RL/Megatron container and site-specific Slurm or Kubernetes job.
The cluster injects `SFT_TRAIN_COMMAND`, `RL_TRAIN_COMMAND` and
`CYBER_EVAL_HOOK` as argv strings. Add `--execute` only inside the approved
scheduler allocation. Successful stage receipts make requeues idempotent.

Useful primary documentation:

- GLM-5.2 BF16 checkpoint: https://huggingface.co/zai-org/GLM-5.2
- NeMo RL model extension: https://docs.nvidia.com/nemo/rl/latest/guides/add-new-model.html
- NeMo RL LoRA: https://docs.nvidia.com/nemo/rl/latest/guides/lora.html
- Megatron Bridge MoE LoRA controls: https://docs.nvidia.com/nemo/megatron-bridge/latest/training/peft.html
