# Operational lessons

These are narrow, reusable rules learned from real training failures. They are
not claims about model quality. Exact job identities and sanitized receipts live
in the linked evidence records; prompts and private episode content do not.

## SkyRL cyber episodes: continue length-limited chunks, compact changed prompts

The `chris-q38-rlreward-prod5` run loaded successfully and began its first
before-training evaluation, but collected zero training episodes and performed
zero optimizer updates. The model filled the 4,096-token generation request
without emitting an end marker or a valid tool call. The runtime treated that
request boundary as the end of the episode, so it truthfully rejected the run as
`generation_incomplete_length`. All eight GPUs were then released. The exact
sanitized receipts and release proof are recorded in
[`docs/evidence/qwen38-study/2026-09-20-skyrl-prod4-direct-rayjob-qualification-v1.md`](evidence/qwen38-study/2026-09-20-skyrl-prod4-direct-rayjob-qualification-v1.md).

The durable rule is:

1. A generation request ending because it reached its token cap is a chunk
   boundary, not an episode outcome. Continue from the exact sampled token IDs.
2. Keep one declared maximum for a single assistant turn. If the model reaches
   that maximum without ending the turn, reject the episode explicitly; never
   invent a reward or silently truncate it.
3. When conversation history is replaced by a compact working memory, preserve
   the summary generation and the next action as separate policy steps with
   their own exact prompts, sampled IDs, masks, and log probabilities.
4. Execute every valid tool call from one assistant turn in order. A successful
   report submission may end execution of later calls, but multiple valid calls
   must not be rejected merely because there is more than one.
5. Before allocating a GPU, run the production recorder with the installed Qwen
   tokenizer and a deterministic in-memory engine. The test must cross at least
   one length boundary, compact history, execute ordered `bash` and
   `submit_report` calls, assign reward only after authoritative grading, and
   verify that changed prompts remain distinct training samples.

Scale-up is closed until this zero-GPU probe and the exact-image CPU preflight
both pass for the immutable successor configuration.
