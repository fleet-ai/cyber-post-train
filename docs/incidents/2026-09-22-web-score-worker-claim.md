# WebExploitBench score-worker claim rule

On 2026-09-22, two local watcher processes and a one-off recovery process were
allowed to target the same already-exported evaluation arm. The scorer itself
failed closed: its first execution wrote the permanent score claim, later
workers refused that ambiguous claim, and no incomplete output was published as
a score. However, terminating the competing workers left that first claim
without a completion receipt.

The durable rule is simple:

1. Exactly one local process may own an arm's score transaction at a time.
2. It must take `exclusive_scoring_arm` before dry validation and hold the lock
   until the terminal score receipt is written.
3. A claim without a completion receipt is preserved and never retried under
   the same identity. It requires an explicitly reviewed scoring-only successor
   if the frozen evaluation policy permits one.
4. The collection export, rollout, judge settings, and model identity remain
   immutable; recovery must never rerun the model rollout.

The regression test proves that a second process cannot enter the same arm lock
and that the lock becomes available after the owner exits. Existing deferred
score tests separately prove that a stale permanent claim never invokes the
judge a second time.
