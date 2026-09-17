# Research-plan checks and starting references

Reviewed 16 September 2026. These are pointers to methods, not evergreen numeric
defaults. Re-open the source and inspect the closest actual configuration.

| Question | Primary starting point | Transfer warning |
|---|---|---|
| Successful security demonstrations on Qwen | [CTF-Dojo v1 §3.1](https://arxiv.org/html/2508.18370v1) | Retained training set differs from collection; much shorter context |
| Similar-size agents and data curation | [OpenThinkerAgent-32B](https://huggingface.co/open-thoughts/OpenThinkerAgent-32B), [author study](https://www.openthoughts.ai/blog/openthoughts-agent) | More data and different tasks; final model settings differ from small-model ablations |
| An explicit same-size full-SFT recipe | [OpenThinker-32B YAML](https://github.com/open-thoughts/open-thoughts/blob/main/train/OpenThinker-32B.yaml) | Earlier Qwen generation and noninteractive reasoning |
| Full tuning versus adapters | [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/), [LoRA Learns Less and Forgets Less](https://arxiv.org/html/2405.09673v2) | Tune LR independently; rank, target modules and retention matter |
| Staged resource allocation | [Hyperband](https://www.jmlr.org/papers/v18/16-558.html) | An inspired staged plan is not the published algorithm; early proxy scores can misrank |
| Group-based RL | [DeepSeekMath](https://arxiv.org/abs/2402.03300) | Short math responses do not establish budgets for long tool episodes |

## Practical controls

- Separate source, coverage and training amount. Compare sources on an
  intersection while separately testing maximum usable data. A stronger teacher
  may solve harder families; that is useful practically but confounds a pure
  imitation-source comparison.
- Use an identical per-family target-token budget for controlled source arms.
  Matching only total sessions, windows or epochs is insufficient.
- A data-poor regime may favor smaller batches/more updates. Large-data recipes
  do not establish that a 100-example corpus wants the same batch.
- Full-model and LoRA experiments need independently selected learning rates.
  Include all intended linear/MLP modules and verify actual trainable parameters.
  If low-rank underfits, test capacity before dismissing the method.
- Context cap is not average example length or episode horizon. A shorter cap
  changes both compute and information. Preserve target coverage and disclose
  lost history; never claim an isolated speed optimization if data also changed.
- Warmup length, schedule horizon, batch size and epochs interact. Compare
  equal-token exposure when that is the question; compare equal-compute when
  that is the question. These are different experiments.
- Report task-level uncertainty, invalid episodes and training-seed variation.
  Prefer task success over teacher-token imitation loss for capability claims.
- A one-update test qualifies machinery, not a hypothesis about capability.
  Conversely, an operationally valid full run with no gain is still information.
- Predeclare a maximum study size and holdout. Refinement uses development
  results only; external test outcomes cannot flow back into the plan.
