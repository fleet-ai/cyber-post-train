# Student-model selection — 2026-08-28

## Decision

Use `Qwen/Qwen3.6-27B` at immutable Hugging Face revision
`6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` as the primary student.

This is a dense 27B model with a native 262,144-token context, an Apache-2.0
license, and the `qwen3_5` architecture. The exact checkpoint was published in
April 2026, before the WebExploitBench dataset was published in June 2026. That
ordering does not prove absence of semantic contamination, but it rules out
direct pretraining on the released benchmark bytes and prompts.

The model is deliberately not selected because it tops a general leaderboard.
It is selected to maximize the validity and throughput of the causal
post-training experiment:

1. strong enough agentic coding and terminal performance to reduce floor risk;
2. enough expected headroom for a measurable post-training delta;
3. exact open weights, tokenizer and chat template;
4. a dense architecture that avoids MoE router and rollout/training log-prob
   consistency confounds;
5. one-node eight-B300 SFT and RL feasibility;
6. Qwen tool/reasoning parser compatibility and an existing Agent Runtime route;
7. publication before the sealed external benchmark.

The immutable local descriptor is
`configs/models/qwen36-27b-6a9e13bd.lock.json`.

## Alternatives considered

| Candidate | Decision |
|---|---|
| Qwen3.6-35B-A3B | Operational fallback. Already served by Fleet and much faster per generated token, but MoE router/recompute consistency is an extra RL confound. |
| Qwen3.5-27B | Training fallback. Same architecture and mature recipes, but weaker and not currently self-hosted. |
| Qwen3.5-9B | Infrastructure canary only; too much blackbox-exploitation floor risk for the primary causal claim. |
| GPT-OSS-20B | Compute-efficient, but Harmony formatting creates avoidable harness-parity differences. |
| Qwen3-Coder-30B-A3B-Instruct | Mature tool use, but its non-thinking behavior is less attractive for exploratory exploitation. |
| Qwen3.8-27B | Teacher/control only. It is served and strong, but was released after WebExploitBench and likely leaves less measurable headroom. |
| GLM-5.2 | Frontier-scale comparison only. Its 1.5 TB checkpoint makes seeds and ablations unnecessarily expensive. |

## Empirical gate

Before treating the full baseline as scientifically useful, run a stratified
20–30-task Fleet canary with the frozen harness:

- below 5% pass@1: investigate tool/harness validity, then consider a stronger
  model only if the harness is sound;
- 5–40%: desired learning band;
- above 50%: retain the result but consider a harder holdout or the 35B-A3B
  operational fallback to preserve measurable headroom.

Tool-call validity, refusals, infrastructure failures and verifier failures are
reported separately. A zero verifier score is not automatically interpreted as
insufficient cyber reasoning.

## Primary sources

- <https://huggingface.co/Qwen/Qwen3.6-27B>
- <https://huggingface.co/api/models/Qwen/Qwen3.6-27B>
- <https://github.com/verl-project/verl/blob/ea53291385ce764019a2b40733605f21d8317583/examples/grpo_trainer/run_qwen3_5_27b_fsdp.sh>
- <https://github.com/AgentCyberRange/WebExploitBench/releases/tag/v1>
