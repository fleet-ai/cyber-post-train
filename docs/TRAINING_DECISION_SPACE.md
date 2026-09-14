# Qwen3.8-27B cyber training decision space

Last updated: 2026-09-13

## The question

The goal is to make one exact Qwen3.8-27B model better at Fleet's blackbox
cyber tasks. Here, **lift** means the trained model solves a larger share of
previously unseen tasks than the unchanged base model when everything else is
the same.

That last clause matters. A new agent program, a longer time limit, a different
grader, or an easier test set can raise a score without the model learning
anything. Those changes can be useful product experiments, but they are not
evidence of training lift.

This page maps the practical choices worth studying. It does not propose one
giant grid containing every combination. We should first hold the measurement
system fixed, test the few choices most likely to matter, and only then refine
the winner.

## Plain-language key

- **Base model:** the unchanged Qwen3.8-27B model from which training starts.
- **Blackbox cyber task:** a challenge where the agent can interact with a
  running application but cannot inspect its source code.
- **Token:** a small unit of text used by the model; a word may contain one or
  several tokens.
- **Tokenizer:** the fixed set of rules that converts text and tool calls into
  tokens and back again.
- **Model weights:** the learned numbers inside a model that training changes.
- **Checkpoint:** a saved copy of the model and the state needed to continue
  training.
- **Agent software (also called the harness):** the program around the model. It formats the conversation,
  exposes tools, manages long conversations, and decides when the run ends.
- **Task family:** all versions of the same underlying challenge. Every version
  of one family must stay in the same data split.
- **Grader:** the program that checks what happened in a task and assigns the
  official result.
- **Training split:** tasks the model may learn from.
- **Development split:** unseen tasks used to choose settings and checkpoints.
- **Final test split:** unseen tasks opened only after the recipe is frozen.
- **Teacher trace:** a successful attempt produced by a stronger model.
- **Self trace:** a successful attempt produced by Qwen itself.
- **SFT (supervised fine-tuning):** train Qwen to predict the successful actions
  in saved traces.
- **RL (reinforcement learning):** let Qwen try fresh tasks, score each attempt
  with the real task grader, and make actions from better attempts more likely.
- **Training example (also called a training window):** one bounded slice of a long trace. It contains earlier
  context plus one or more assistant actions that the model must learn to
  predict.
- **Batch:** the examples combined into one model update.
- **Supervised token:** a token in a saved successful action that SFT is asked
  to predict. Context can be shown without being supervised.
- **Learning rate:** how large each model update is.
- **Epoch:** one pass over the SFT training rows.
- **Rollout:** one fresh RL attempt at a task.
- **Reward:** the numeric result used for RL. A binary reward is 1 for a verified
  solve and 0 for a genuine, correctly graded failure.
- **GRPO:** an RL method that compares several attempts at the same task. It can
  train without learning a separate estimate of expected future reward.
- **PPO:** an older RL method that normally learns an extra estimate of expected
  future reward. That estimate may be a small added output on the main model or
  a separate model, and it adds memory and complexity.
- **Dr. GRPO:** a version of GRPO designed to stop long answers from receiving
  unintended extra influence.
- **DAPO-style GRPO:** a GRPO variant that seeks task groups with useful reward
  differences and treats long responses more carefully.
- **GSPO:** an RL method that limits changes to a whole generated answer rather
  than limiting each generated token separately.
- **KL penalty:** a cost for moving too far from the model that training started
  from.
- **Clipping:** a limit on how much one update can change the probability of an
  action.
- **Temperature:** how varied the model's fresh attempts are. Higher values make
  attempts less predictable.
- **pass@1:** the share of tasks solved with one attempt.
- **pass@4:** the share of tasks solved at least once in four attempts.
- **Seed:** a recorded random-number starting point. Repeating a recipe with
  different seeds shows how much luck affects it.
- **BF16:** a common 16-bit number format used to train large models while using
  less memory than ordinary 32-bit numbers.
- **Quantization:** storing model numbers with fewer bits. It saves memory but
  can change model behavior, so it is a separate experiment.
- **Loss:** the model's prediction error. Lower training loss means better
  imitation of the saved examples, not necessarily better task solving.
- **Optimizer:** the calculation that turns prediction or reward feedback into
  a model-weight update.
- **Gradient:** the numerical signal showing how each model weight should move
  to reduce loss.
- **Canary:** a deliberately tiny run used to prove a setup works before an
  expensive run.
- **Confidence interval:** a range showing how much a measured result could
  change if a different but comparable set of tasks had been sampled.
- **Hash:** a short digital fingerprint of file contents. Matching hashes are
  used to prove two runs used the same bytes.
- **Low-rank adapter:** small trainable matrices attached to a frozen model so
  that only a small share of its numbers must be updated.
- **Ablation:** a controlled comparison in which one planned choice changes and
  the rest stay fixed.
- **Data curriculum:** a planned order for showing training examples, such as
  basic investigation actions before complete solutions.
- **Experiment arm:** one version of an experiment, such as teacher-SFT at one
  learning rate.
- **Model service (sometimes called an endpoint):** a running copy of the model
  that accepts requests and returns generated output.
- **Packaged software image (sometimes called a container image):** a fixed
  bundle of code and software libraries used to run a job the same way again.
- **GPU-hour:** one graphics processor used for one hour.
- **Environment-hour:** one task website kept running for one hour.
- **Miles:** one of the repository's two software systems for running RL. It
  generates task attempts, gets scores from the task grader, and uses those
  scores to update the model.
- **SkyRL:** the repository's other software system for running RL. It serves
  the same broad purpose as Miles, so the first comparison should give both
  systems the same model, tasks, budgets, and update settings.
- **Number notation such as 1e-6:** a compact way to write a small number;
  `1e-6` means `0.000001`.

## Three levels of decisions

| Level | Meaning | Rule |
|---|---|---|
| **Must control** | A change here can make a comparison unfair or unsafe. | Freeze it inside every base-versus-trained comparison. If it changes, run a new matched base result. |
| **Test now** | A choice likely to change learning or generalization. | Include it in the first controlled search waves. |
| **Test later** | A useful refinement with more implementation cost or weaker evidence. | Study it only after the basic SFT and reward-bearing RL paths work. |

## 1. Measurement and agent controls

These variables do not normally belong in the training search. They define what
the experiment means.

| Variable | What it means | Values worth using | Level and decision |
|---|---|---|---|
| Exact base weights | The exact bytes of Qwen3.8-27B before training. | Exact saved model version `1d4bf0f2...` and its locked file hashes. | **Must control.** Never compare against a model that merely has the same display name. |
| Tokenizer and conversation format | The rules that turn text and tool calls into model tokens. | The exact files locked with the base model. | **Must control.** A formatting change can alter tool use by itself. |
| Numerical format | How precisely weights are stored and computed. | BF16 (the defined 16-bit format) for the Qwen runs described here; do not silently store weights with fewer bits. | **Must control.** A smaller numerical format is a separate experiment. |
| Harness | The agent program around Qwen. | Qwen Code and OpenCode are two separate experiment blocks. Choose one primary harness; never pool them. | **Test now, but match within each block.** A harness comparison requires a fresh base and trained result under each harness. |
| Harness version | The exact source and packaged software image for the agent program. | One version that cannot change after publication per experiment block. | **Must control.** “OpenCode” or “Qwen Code” without a version is not enough. |
| System instructions and tools | The instructions and actions available to the agent. | The task's exact instructions, command tool, and final-report tool (internally named `bash` and `submit_report`). | **Must control.** Tool definitions must be enforced by the running system, not only described in text. |
| Conversation shortening | Whether an overlong conversation is summarized and then continued. | Off, or one proven summarize-and-continue policy. | **Test later.** First prove that a shortened run really continues. Never call a context failure a model failure. |
| Attempt budget | Maximum time, turns, and tokens allowed for an evaluation attempt. | One frozen budget derived from successful training-only traces; apply it to base and trained models. | **Must control.** Longer budgets often improve scores without training. |
| Model sampling during evaluation | How random evaluation output is. | One frozen temperature, probability cutoff, and seed policy. | **Must control.** Use the same values for both arms. |
| Task and grader versions | The exact challenge, its starting state, and the program that scores it. | Version identifiers that cannot change after publication, plus file hashes. | **Must control.** Missing or mismatched grading is an infrastructure failure, not a zero. |
| Broken-infrastructure rule | What happens when a task website, model service, or grader breaks. | Preserve the attempt, mark it invalid, and retry only under a rule written before results are seen. | **Must control.** Apply the same rule to every experiment arm. |

## 2. Task set, split, and sampling

The current strict public manifest retains 89 runnable, previously validated
task versions from a historical set of 160. Two representative split variants
each contain 59 training tasks, 20 development tasks, and the same 10 final-test
tasks. Ten final tasks can reveal a large change but cannot support a precise
claim; increasing the high-quality task pool remains important.

| Variable | What it means | Values worth testing | Level and decision |
|---|---|---|---|
| Task quality gate | Which challenges count as real tasks rather than broken infrastructure. | Only tasks with a demonstrated working environment, working grader, clean cleanup, and complete current launch information. | **Must control.** A genuine failed solve is eligible; an infrastructure failure is not. |
| Split unit | What stays together when tasks are divided. | Entire task family, including all versions and sessions. | **Must control.** Never split individual windows or attempts at random. |
| Split variant | A different representative assignment of task families. | Existing split A and split B, each 59/20/10. | **Test now.** The final 10 stay untouched; compare sensitivity across the two train/development assignments. |
| Application balance | The mix of products or web applications. | Match the available application proportions, or give each application equal total weight. | **Test now.** Report both task-level and application-level results. |
| Difficulty balance | The mix of easier and harder tasks. | Representative natural mix versus equalized easy/medium/hard weight, but only when difficulty labels are independently verified. | **Test later.** Do not invent difficulty from test outcomes. |
| Vulnerability-family balance | The mix of underlying bug types. | Representative mix versus equal family weight, once reliable labels exist. | **Test later.** Current opaque names are not a trustworthy classification of security weaknesses. |
| Per-task sampling weight | How often each training task is selected. | Uniform by task family; equal weight by application; favor tasks that are neither almost always solved nor almost always failed, based only on training results. | **Test now.** Uniform-by-family is the clean baseline. |
| Repetition cap | Whether tasks with many saved successes dominate SFT. | At most 1, 2, or 4 traces per task family. | **Test now.** Measure the mix by supervised tokens as well as trace count. |
| Curriculum order | The order in which training examples are shown. | Fully shuffled; initial exploration and tool-use actions before complete solution chains. | **Test now.** The current pool has 82 medium, 6 hard, and 1 easy task, so it cannot support a meaningful easy-to-hard comparison. |
| Curriculum pace | How quickly complete solutions enter after action-focused examples. | All at once; two equal stages; three stages using 25%/35%/40% of updates. | **Test later.** Keep total examples and update count matched. |
| Source age | Whether old but still valid task versions remain in training. | Latest valid version only versus all version-compatible training traces. | **Test later.** Never let versions cross task-family splits. |
| Data amount | How much approved training signal is used. | 25%, 50%, and 100% of task families, selected with the same representative sampler. | **Test now.** This reveals whether quality, diversity, or raw quantity limits lift. |

## 3. Teacher traces, Qwen traces, and their mixture

Teacher and self traces answer different questions. Teacher traces test whether
Qwen can imitate strategies it did not reliably discover. Self traces preserve
Qwen's own style and test whether verified self-generated successes can be
amplified. They should be compared at matched task coverage and matched numbers
of supervised tokens.

| Variable | What it means | Values worth testing | Level and decision |
|---|---|---|---|
| Source model | Which model produced the successful trace. | Strong-teacher only; Qwen-only; mixed teachers; mixed teacher plus Qwen. | **Test now.** Record exact source-model identity. |
| Teacher identity | Whether different strong models teach different behavior. | GPT-only; Grok-only when enough valid examples exist; balanced GPT/Grok mixture. | **Test now.** Do not let one teacher win merely because it contributed more tokens. |
| Teacher:self ratio | The share of SFT target tokens from strong models versus Qwen. | 100:0, 75:25, 50:50, 25:75, 0:100. Start with 100:0, 50:50, 0:100. | **Test now.** Define the ratio by supervised tokens, not file rows. |
| Success requirement | Which saved attempts can teach the model. | Verified successes only. | **Must control.** Never train SFT on a broken, ambiguous, truncated, or merely “completed” attempt. |
| Trace diversity | How similar accepted examples may be. | Remove exact copies only; also remove very similar copies; choose a varied capped subset per task. | **Test now.** Preserve uncommon successful strategies while stopping one repeated pattern from dominating. |
| Efficiency preference | Whether shorter successful solutions get extra weight. | No length preference; mild preference among equally correct traces. | **Test later.** A strong shortest-only filter can teach brittle shortcuts. |
| Iterative self-training | Generate new Qwen attempts, keep verified successes, fine-tune, then repeat. | Zero, one, or two generate-filter-train rounds. | **Test later.** Freeze each round and evaluate before using its outputs in the next. |
| Failed self traces | Whether genuine failures are used as negative examples. | Exclude from ordinary SFT; consider only in a separately defined preference or RL objective. | **Must control.** Next-token imitation of failures teaches failure. |

## 4. Turning long traces into SFT examples

This is likely one of the highest-value parts of the search. A trace may contain
hundreds of actions. Training only on the final report mostly teaches formatting;
training every assistant action can teach investigation, hypothesis changes,
tool choice, exploitation, verification, and reporting.

| Variable | What it means | Values worth testing | Level and decision |
|---|---|---|---|
| Action coverage | Which assistant actions contribute to the loss that updates the model. | Every eligible assistant action exactly once; final action only as a health-check comparison; key actions selected by rules written before training. | **Test now.** Training every eligible action is the recommended baseline. |
| Context-only text | Earlier text shown to the model but not learned as a target. | User text, tool output, and copied earlier assistant text receive zero training loss. | **Must control.** The model should learn its actions, not copy server output or secrets. |
| Window length | The maximum tokens in one SFT row. | 8,192; 16,384; 32,768 when memory permits. Start with 16,384. | **Test now.** Re-tokenize with the exact Qwen tokenizer. |
| Context allowance | How much of the window is reserved for information before the target action. | 2,048; 4,096; 8,192 tokens. Start with 4,096. | **Test now.** Longer context may help planning but reduces the number of target tokens per batch. |
| Number of targets per window | Whether a row teaches one action or several nearby actions. | One target per row; all fitting consecutive targets once. | **Test now.** Total target-token exposure must remain equal. |
| Target placement | Where the action appears in the window. | Place the target action at the end; retain complete tool calls and their boundaries. | **Test now.** Never split a tool call or its immediate result. |
| Early/middle/final weighting | Whether discovery, exploitation, and final reporting receive equal influence. | 1:1:1; 2:2:1; equal total weight per task. | **Test now.** Avoid a training set dominated by the short final submission step. |
| Long-trace weighting | Whether a long trace receives more total influence simply because it has more actions. | Equal per target; equal total per trace; equal total per task family. | **Test now.** Equal total weight per task family is the safest primary baseline. |
| Repeated-action handling | How loops and repeated commands are treated. | Keep all valid actions; collapse exact repeated no-progress runs; cap consecutive duplicates at 1 or 2. | **Test later.** Preserve recovery behavior, not mechanical loops. |
| Overlength behavior | What happens when a complete action and required context do not fit. | Count and exclude it; use a larger window in a separate experiment arm. | **Must control.** Never silently cut tokens and pretend the example is intact. |
| Automatically summarized traces | Whether attempts containing an automatic conversation summary may enter SFT. | Exclude unless the conversation before and after the summary can be reconstructed exactly; then compare as a separate data group. | **Must control now; test later.** A summary is not the original conversation. |
| Tool-result size | How much command output is retained as context. | 16,000 or 50,000 characters per tool result, always keeping the beginning and end and recording every cut. | **Test later.** No hidden cutting. |
| Loss weighting by source | How strongly each source type affects the update. | Equal per supervised token; equal per task; teacher:self weights matching the declared mixture. | **Test now.** Log the realized weights, not only requested ratios. |

## 5. SFT optimization

The ranges below are intentionally geometric: each value is roughly three times
the previous one. This explores the space faster than tiny adjacent changes.
One extreme value may be used only in a short canary if its stability is unknown.

| Variable | What it means | Values worth testing | Level and decision |
|---|---|---|---|
| Learning rate | Update size. Too low learns little; too high damages useful behavior. | 3e-7, 1e-6, 3e-6; 1e-5 only as a brief high-risk test. | **Test now.** The repository default is 3e-6; the successful historical run used 1e-6. |
| Global batch size | Number of SFT windows combined per update across all GPUs. | 8, 32, 64. | **Test now.** Also log supervised target tokens per update because windows have different lengths. |
| Microbatch per GPU | Rows processed at once by each GPU before update signals are combined. | 1 first; 2 only if memory allows. | **Test later.** This should not change the scientific result when global batch stays fixed, but it can cause small rounding differences. |
| Epoch count | How many complete passes are made over SFT rows. | 1, 2, 4. | **Test now.** Evaluate saved checkpoints inside each run; do not assume the last epoch is best. |
| Total supervised tokens | The actual amount of assistant output learned from. | Match across data-mixture comparisons; report exact count per epoch and overall. | **Must control.** Equal row counts are not equal training doses. |
| Learning-rate schedule | Whether update size stays constant or decreases. | Constant; a smooth decrease from the peak to 10% of the peak. | **Test later.** Current supported path is constant. |
| Warmup | A gradual rise from a tiny update size at the start. | 0%, 3%, 10% of updates. | **Test later.** Current supported path uses 0%; the alternatives are SFT hypotheses, not values copied from an RL study. |
| Weight decay | A small pressure against very large weight changes. | 0, 0.01, 0.1. | **Test later.** First expose and lock the value the training program actually uses. |
| Gradient clipping | A limit on unusually large update signals. | Maximum update-signal size 0.5 or 1.0; disabled only as a health check. | **Test later.** Log how often the limit activates. |
| Full-weight versus adapters | Update all model weights or only small attached trainable matrices. | Update all weights as primary; low-rank adapters as a lower-cost comparison. | **Test later.** The two methods have different learning capacity and model-saving paths. |
| Example order seed | Random order of SFT rows. | At least 3 recorded seeds for finalists. | **Test now for finalists.** One seed is enough for coarse rejection, not a final claim. |
| Checkpoint interval | How often recoverable model state is saved. | Every 10, 25, or 50 updates, plus final. | **Must control operationally.** It should not change learning, but enables fair checkpoint selection and recovery. |
| Checkpoint choice | Which saved point becomes the candidate model. | Highest success on the Fleet development tasks; use training loss only to detect broken or unstable learning. | **Must control.** Do not choose from prediction error on held-out teacher text, the final Fleet tasks, or WebExploitBench. |

Training loss answers “is the model fitting these saved examples?” It does not
answer “is the model better at new cyber tasks?” Development task success is the
selection measure that matters.

## 6. RL task and reward choices

RL is useful because it trains from fresh interaction and the real grader rather
than copying a fixed trace. It is also easier to fool. A reward-bearing canary
must prove that the model can finish an attempt, the official grader runs,
different attempts can receive different rewards, one real model update occurs,
and a recoverable checkpoint is saved.

| Variable | What it means | Values worth testing | Level and decision |
|---|---|---|---|
| Starting checkpoint | The model RL begins from. | Base Qwen; best teacher-SFT; best self-SFT; best mixed-SFT. | **Test now.** This directly tests whether SFT is a useful RL warm start. |
| RL task split | Tasks on which fresh RL attempts may run. | The same training families used by the corresponding SFT arm; development and final test excluded. | **Must control.** Bind exact task and grader versions. |
| Official reward source | Which system decides success. | The real grader locked to that exact task version, plus a unique ID for each scoring run. | **Must control.** A missing scoring ID makes the attempt invalid, not zero reward. |
| Outcome reward | Reward for the final task result. | Binary 0/1 first; verified partial progress in a separate later arm. | **Test now.** Binary reward is easiest to audit. |
| Partial-progress reward | Credit for verified intermediate objectives. | Off; exact fraction provided by the grader; a separately studied blend only after the fraction is audited. | **Test later.** Use only progress confirmed by the grader, never a model's self-claim. |
| Report-format reward | Extra credit for calling the final report tool correctly. | None first; at most a small separately logged bonus later. | **Test later.** A large format bonus can teach reporting without exploitation. |
| Invalid-attempt handling | What happens when the environment, model service, or grader breaks. | Exclude from model updates and preserve the reason. | **Must control.** Do not turn infrastructure failures into negative model feedback. |
| Time-limit handling | What happens when an otherwise healthy attempt reaches its declared time, turn, or token limit. | Count it as a genuine model failure under the prewritten rule; mark it invalid only when infrastructure, rather than the model, caused the timeout. | **Must control.** Track time-limit rate alongside reward. |
| Group reward variation | Whether attempts at one task include both successes and failures. | Require and log mixed-reward groups in the canary. | **Must control before scale.** Ordinary GRPO learns nothing from all-zero or all-one groups. |
| Zero-variation groups | How groups with identical rewards are handled. | Keep them and produce zero update as the baseline; later compare a fixed, prewritten replacement cap that seeks useful reward differences. | **Test now after baseline works.** Record every replacement because it changes which tasks training sees. |
| Task sampling for RL | Which training task is attempted next. | Uniform by family; application-balanced; favor tasks that are neither near 0% nor near 100% solved. | **Test now.** Use training outcomes only. |
| Replay of old attempts | Whether previously generated attempts are reused. | No replay for the primary fresh-attempt run; one bounded replay study later. | **Test later.** Old attempts were generated by an older model and can bias updates. |

## 7. RL attempts, model updates, and running system

The repository has two implementation paths, Miles and SkyRL. They are training
systems, not scientific algorithms by themselves. First compare them with the
same model, data, reward, group shape, and one-step canary. Scale only a path
that produces a real nonzero reward, update, checkpoint, and successful reload.

| Variable | What it means | Values worth testing | Level and decision |
|---|---|---|---|
| RL implementation | The training system that coordinates generation and updates. | Miles and SkyRL, first as matched one-step canaries. | **Test now operationally.** Do not treat one system merely starting as evidence of learning. |
| RL algorithm | The rule that converts rewards into weight updates. | GRPO first; Dr. GRPO, DAPO-style GRPO, and GSPO later; PPO only if its extra expected-reward estimate and added complexity are justified. | **Test later after GRPO works.** Each algorithm needs its own verified implementation. |
| Attempts per task | Number of fresh responses compared for the same task. | 8 and 16; 4 only when enough tasks combine into a total batch that divides evenly across all training GPUs. | **Test now.** Larger groups improve the chance of mixed rewards but cost more website time. |
| Different tasks per update | Number of distinct task prompts contributing to one update. | 1, 2, 4, 8. Start with 1×8 for the canary, then 4×8 if speed allows. | **Test now.** More tasks reduce dependence on one challenge. |
| RL learning rate | Update size for reward-based training. | 1e-7, 3e-7, 1e-6; 3e-6 only as a brief high-risk test. | **Test now.** Several open GRPO recipes use 1e-6, but math tasks are not cyber agents. |
| Number of optimizer updates | How long RL runs. | 1 reward canary; 10 short search; 50 medium; 100–150 long only after development lift. | **Test now in stages.** More updates are not automatically better. |
| Updates per batch of attempts | How many times the same newly generated attempts are reused immediately. | 1 first; 2 later. | **Test later.** Reuse may learn more from each attempt but moves training farther from the model that generated the data. |
| Reference model | The fixed model used to measure how far RL moves. | The RL starting checkpoint; periodic reference reset only in a later long-run study. | **Must control.** Log the exact reference checkpoint. |
| KL penalty size | Strength of the cost for moving away from the reference model. | 0, 0.001, 0.01, 0.04. Start with 0.001 and compare 0. | **Test now.** Track the measured change, not only the requested number. |
| Clipping range | Maximum change in action probability trusted in one update. | Equal lower and upper limits of 0.2; DAPO-style lower limit 0.2 and upper limit 0.28. | **Test now.** The repository's Miles path already names the 0.2/0.28 shape. |
| Loss averaging | Whether long and short attempts receive equal total weight or equal weight per token. | Equal per attempt; equal per trained token; Dr. GRPO adjustment that removes unintended extra weight from long answers. | **Test now.** Track response length because ordinary GRPO can favor long incorrect output. |
| Entropy bonus | An optional reward for keeping output varied. | 0; 1e-4; 1e-3. | **Test later.** Use only if diversity collapses; too much produces noise. |
| Rollout temperature | Randomness while collecting training attempts. | 0.7, 1.0, 1.2. Start with 1.0. | **Test now.** Evaluation temperature remains fixed separately. |
| Probability cutoff | How much of the model's token distribution remains available. | Top-p 0.9, 0.95, 1.0. | **Test later.** “Top-p 0.9” keeps the smallest token set whose probabilities add to 90%. |
| Total context budget | Maximum prompt plus generated tokens in one attempt. | 65,536 and 98,304. Start with the reviewed 98,304-token limit. | **Test now carefully.** Any cutting caused by the limit must be reported. |
| Response budget | Portion of total context available for model generation. | 49,152 and 81,920 tokens. | **Test now carefully.** It must remain below the total budget. |
| Tokens per turn | Maximum generated tokens before the next tool call or observation. | 2,048; 4,096; 8,192. Start with 4,096. | **Test now.** A too-small turn cap can prevent complete commands or reports. |
| Turn count | Maximum model/tool cycles in one attempt. | 64 and 80 first; 120 only if successful training-only traces justify it. | **Test now.** Use enough room to finish, then study efficiency. |
| Attempt time | Real-time deadline for one full attempt. | 1,800; 2,400; 3,600 seconds. Start with 2,400. | **Test now operationally.** Base the choice on valid training-only completion times. |
| Tool deadline | Maximum time for one command. | At least 330 seconds for a five-minute server limit. | **Must control.** Client timeout must exceed the advertised tool timeout. |
| Tool-output allowance | Maximum command-result text returned to the model. | 16,000 and 50,000 characters. | **Test now.** Mark any cut clearly and use the same rule in every attempt. |
| Weight freshness | How current the attempt-producing model is relative to the training model. | Refresh it after every model update first; allow at most one-update delay later. | **Test later.** An unlimited delay trains on behavior from an old model. |
| Attempt generation and training placement | Whether fresh attempts and model training share GPUs. | Share the same GPUs first; separate them only for a matched speed study. | **Test later.** This should mainly affect computing speed, but confirm that rounding and model-update timing remain comparable. |
| Checkpoint and evaluation interval | How often RL saves and checks development tasks. | Every 1 step for canary; every 10 or 25 for longer runs; always final. | **Must control operationally.** Keep the best and latest recoverable states. |
| Resume behavior | Whether a stopped run continues from the exact next task/update. | One planned stop-and-resume test before a long run. | **Must control.** Restore weights, optimizer, random state, and data position together. |

Why these anchors? DeepSeekMath used GRPO with learning rate 1e-6, 64 attempts
per math question, a 0.04 penalty for moving away from the starting model, and
one model update per newly generated batch. DAPO used learning rate 1e-6, 16
attempts per question, update limits of 0.2/0.28, replacement of groups with no
reward differences, and a gradual penalty for overlong answers. These are
useful scale references, not recipes to copy: a forty-minute tool-using cyber
attempt is very different from a math answer.

## 8. Combined training schedules

| Variable | What it means | Values worth testing | Level and decision |
|---|---|---|---|
| Training path | The ordered stages applied to Qwen. | Base only; teacher-SFT; self-SFT; mixed-SFT; RL from base; teacher-SFT→RL; self-SFT→RL; mixed-SFT→RL. | **Test now.** These are the core scientific arms. |
| SFT before RL | Whether imitation gives RL a better starting success rate. | No SFT; best teacher-SFT; best self-SFT; best mixed-SFT. | **Test now.** Match the RL task mix, number of attempts, and number of model updates. |
| SFT after RL | Whether a final imitation pass restores format or tool behavior. | None first; very small mixed-SFT “recovery” pass later. | **Test later.** It can erase RL gains, so evaluate both sides. |
| Alternating stages | Switch between SFT and RL more than once. | One SFT→RL cycle first; two cycles later. | **Test later.** Freeze and evaluate every boundary. |
| General-data preservation | Mix non-cyber instruction data to reduce forgetting. | 0% inside the current cyber-only project; 5% or 10% only in a clearly separate follow-up study. | **Test later.** Non-cyber data is outside the current goal and must not be mixed into its main result. |
| Reference reset during long RL | Move the KL reference to a newer checkpoint. | Never; once at the midpoint; every fixed 50 updates. | **Test later.** ProRL motivates studying this for long runs, not using it by default. |

## 9. Evaluation and model selection

| Variable | What it means | Values worth using | Level and decision |
|---|---|---|---|
| Primary development measure | The result used to choose recipes. | Fleet development pass@1, averaged equally across task families. | **Must control.** Training loss and prediction error on teacher-written text are health checks, not the target. |
| Secondary development measures | Other useful views of capability. | pass@4, verified partial progress, application-average success, completion time, turns, and tokens. | **Must control.** Label them secondary before running. |
| Final measure | The untouched score reported after selection. | Fleet final-test pass@1; pass@4 as secondary; raw per-task outcomes. | **Must control.** Open once per frozen finalist family. With only 10 tasks, treat small differences as uncertain. |
| External benchmark | A separate test of improvement beyond Fleet tasks. | A newly validated, fair WebExploitBench comparison used only for reporting; reserve another untouched benchmark if WebExploitBench results influence choices. | **Must control.** If its results help choose data, settings, or checkpoints, it is a development test—not independent final evidence. |
| Harness transfer | Whether lift survives a different agent program. | Primary harness result, then a separately matched Qwen Code/OpenCode pair. | **Test later.** A trained-versus-base comparison is required inside each harness. |
| Evaluation attempts | Number of tries per task. | pass@1 for all search decisions; pass@4 for finalists. | **Must control.** Never rerun a valid pass@1 zero. |
| Checkpoint evaluation cadence | How often training pauses for task evaluation. | Track training loss continuously; run the full Fleet development test after each completed training run; for very long runs, optionally use one fixed small development subset at epoch or stage boundaries. | **Test now operationally.** The development tasks must never enter training, and frequent repeated checks can overfit decisions to them. |
| Training repeats | Independent training runs with different seeds. | One for coarse screening; three for finalists. | **Must control for claims.** Evaluation attempts are not substitutes for training repeats. |
| Paired task analysis | Compare base and trained outcomes on the same task versions. | Per-task win/loss/tie table and a paired resampling confidence interval: repeatedly resample whole task families and recalculate the lift. | **Must control.** Do not compare only two pooled percentages. |
| Confidence interval | A range showing sampling uncertainty. | Repeatedly resample whole task families, not individual attempts; report 95% intervals. | **Must control.** Attempts on one task are related, not independent data points. |
| Guard against a lucky search winner | Avoid declaring the luckiest of many recipes a discovery. | Write down the search before it starts, use development data for ranking, and reserve the final test for a small frozen set. | **Must control.** Report the number of experiment arms tried. |
| Minimum useful lift | Smallest improvement worth the cost. | Write down the minimum absolute increase in task success before running the search. | **Must control.** With only 10 final tasks, only very large changes will be clear. |
| Cost and efficiency | Resources needed for each gain. | GPU-hours, environment-hours, generated tokens, and solved tasks per cost. | **Test now.** The best model is not necessarily the most expensive winner. |
| Safety and general behavior | Check whether cyber lift damages basic tool use or reporting. | Fixed small test set for non-secret tool formatting and general instructions. | **Must control.** Keep this separate from the cyber success score. |

## 10. Repeatability and run health

Every run should record the following before it starts:

| Item | Plain-language requirement |
|---|---|
| Model identity | Hash every model, tokenizer, and conversation-format file. |
| Data identity | Hash the task list, split, list of saved source attempts, prepared rows, and source weights actually used. |
| Training identity | Record the exact saved code version, packaged software image, full settings, seed, GPU layout, and intended update count. |
| Reward identity | Record exact task, environment, grader, tools, and one unique grader execution ID per RL attempt. |
| Evaluation identity | Record checkpoint, model-service image, agent software, task and grader versions, budgets, seeds, and retry policy. |
| Progress | Send training loss, reward, update size, output length, invalid-attempt rate, task mix, and checkpoint details to Weights & Biases, the experiment dashboard. Never upload prompts or traces. |
| Recovery | Save model, optimizer, learning-rate state, the exact state of the random sequence, and the exact next data position. Test one stop and resume. |
| Completion | Reload the exported model and prove it can produce numeric predictions without invalid numbers before using it for evaluation. |
| Run status | Separate “program started,” “model updated,” “valid score produced,” and “model improved.” They are different claims. |

## Recommended search order

The order below learns the most while limiting expensive combinations.

1. **Freeze measurement.** Choose one primary harness and prove a valid matched
   Qwen base evaluation. Lock the 89-task eligible set, split A/B, exact grader,
   budgets, and invalid-run rule.
2. **Find the SFT example design.** At learning rate 1e-6, batch 32, and one
   epoch, compare teacher traces, Qwen traces, and a 50:50 mixture measured by
   trained tokens; train every eligible action versus a task-balanced set of key
   actions; and compare split A with split B.
3. **Tune SFT update settings.** For the best two data approaches, combine each
   learning rate (3e-7, 1e-6, 3e-6) with each batch size (8, 32, 64), and evaluate checkpoints from epochs
   1/2/4. Reject unstable settings early. Repeat finalists with three seeds.
4. **Acquire real RL reward.** For both Miles and SkyRL, run a matched one-update
   1-task × 8-attempt canary. Advance only a path with real mixed rewards, a
   nonzero update, a recoverable checkpoint, and a successful reload.
5. **Find an RL starting point.** Compare base, best teacher-SFT, best self-SFT,
   and best mixed-SFT at 10 updates with identical tasks and rollout settings.
6. **Tune the working RL path.** Search learning rate, attempts per task, KL
   penalty, clipping, and task sampling in that order. Use 50 updates only for
   settings with development lift; use 100–150 only after a stable medium run.
7. **Test the core finalists.** Compare base, SFT-only, RL-from-base, and SFT→RL
   with three training seeds. Freeze winners before opening the final Fleet set
   and WebExploitBench.

Use **successive halving** for large searches: run every setting for a small,
equal budget, keep only settings that pass health checks and show development
promise, then give survivors more budget. This avoids spending a full run's computing time
on obviously unstable settings while preserving a written selection rule.

## What not to combine into one conclusion

- A Qwen Code result and an OpenCode result.
- A shared model service and a separately hosted service unless they are proven
  to use the same model bytes and generation settings.
- A binary-reward run and a partial-credit run.
- An SFT arm trained on different task families from its comparator.
- Training loss and held-out task success.
- Evaluation attempts and independent training repeats.
- A valid model failure and an environment, model-service, or grader failure.

## Evidence and research anchors

Project evidence:

- [Scientific controls](SCIENTIFIC_PROTOCOL.md)
- [Training guide](TRAINING.md)
- [Current 160-to-89 task filtering report](QWEN_BLACKBOX_TASK_FILTERING_REPORT_2026-09-12.md)
- [Task and source eligibility audit](QWEN_BLACKBOX_TASK_ELIGIBILITY_2026-09-11.md)
- [Qwen3.6 study evidence, including SFT and RL failure modes](QWEN36_STUDY_EVIDENCE.md)
- [Repository consolidation and backend qualification](CONSOLIDATION.md)

Primary research sources:

- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388): a staged
  SFT→reasoning-RL→mixed-SFT→general-RL recipe, plus strong-to-weak
  distillation. Its claim that distillation beat RL applies to its reported
  models and tasks, not automatically to Fleet cyber tasks.
- [STaR](https://arxiv.org/abs/2203.14465) and
  [ReST-EM](https://arxiv.org/abs/2312.06585): generate, verify, and learn from
  successful self-produced reasoning, which motivates a Qwen self-training arm.
- [DeepSeekMath](https://arxiv.org/abs/2402.03300): the original GRPO method and
  one concrete learning-rate, group-size, and KL reference point.
- [DAPO](https://arxiv.org/abs/2503.14476): seeking task groups with useful
  reward differences, using unequal lower and upper update limits, averaging
  loss over trained tokens, and softly penalizing overlong responses.
- [Understanding R1-Zero-Like Training](https://arxiv.org/abs/2503.20783): the
  response-length bias in ordinary GRPO and the Dr. GRPO correction.
- [GSPO](https://arxiv.org/abs/2507.18071): comparing probability changes for a
  whole generated answer rather than for each token separately.
- [Tulu 3](https://arxiv.org/abs/2411.15124): staged SFT and learning from
  verifiable rewards with development and unseen evaluations.
- [ProRL](https://arxiv.org/abs/2505.24864): prolonged RL, task diversity,
  limits on movement away from the starting model, and resetting that starting
  comparison point. It supports a later long-run study only
  after short and medium runs are stable.

These papers are anchors, not proof that their math or coding settings transfer
unchanged to long, interactive cyber attempts. Every recommended range above is
a hypothesis to test against Fleet development tasks.
