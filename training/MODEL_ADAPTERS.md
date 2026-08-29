# Model adapters and training jobs

Model adapters contain only immutable model/runtime facts and trainer argv. The sealed Fleet
dataset digest, split products, reward contract, and evaluation contract live in
`training/job_factory.py`; model configuration cannot override them.

Generate the Qwen 3.6 27B pair without submitting it:

```sh
uv run python -m training.job_cli \
  --model-config training/configs/models/qwen36-27b.json \
  --run-id fleet-all-v1 \
  --output-dir cluster/jobs
```

Both direct Jobs are emitted suspended, in `fleet-train-jobs`, on `training-lq`, with no priority
class (priority 0). Do not set pod `preemptionPolicy`: the live cluster admission controller
computes it and rejects an explicit value. Equal-priority jobs cannot preempt each other under the
live ClusterQueue's `LowerPriority`-only policy.

The SFT Job verifies the exact model checkpoint lock and a successful model/image/driver
compatibility receipt before invoking the configured driver. It hashes every output checkpoint
file and atomically publishes an SFT receipt. The online-RL Job independently validates that
receipt and every checkpoint digest in both an init container and its main process. It therefore
cannot train from base weights or start before the exact SFT checkpoint exists.

This direct manifest backend is preparation scaffolding, not the active training backend: its
example adapter intentionally names drivers that must be supplied and proved by a compatibility
receipt. It remains fail-closed for private trainer images.

The active runnable backend is Fleet Training's typed SFT/RL API. Its immutable run requests live
under `configs/runs/` and bind a registered staged model plus an exact ready trainer-version UUID.
The API renders a RayJob preview and submits that same shape through Kueue. The one-step SFT and
RL-from-base compatibility arms must turn green before the full one-epoch SFT or SFT-to-RL arm is
submitted.
