# Nebius access and launch checklist

## Observed access state (2026-08-28)

- User CLI profile and kubeconfig are configured for `fleeta-bic-prod`.
- Main cluster: `fleetai-training`, 192 NVIDIA B300 GPUs, 24 × 8-GPU NVLink
  nodes on the `eu-west2-a` InfiniBand fabric, plus three CPU nodes.
- Dev cluster: `fleetai-training-dev`, 2 × one-GPU B300 preemptible nodes and
  one CPU node.
- Kubernetes authorization permits pod creation. The existing main cluster uses
  namespace `fleet-train-jobs`, Kueue local queue `training-lq`, cluster queue
  `training-cq`, and a 1 PiB RWX shared filesystem.
- The access check found 184 requested GPUs already active. Resource admission,
  checkpoint location and image ownership remain coordination gates; browser or
  kubeconfig access alone is not permission to displace those workloads.
- Every workload from this repository must be named `chris-cyber-glm52-*`, use
  `fleet-train-jobs` and `training-lq`, and retain default priority 0.
  `training.cluster_policy` enforces these fields. The cluster's admission
  controller forbids an explicit pod `preemptionPolicy: Never`; safety instead
  follows from equal priority 0 plus `training-cq`'s `LowerPriority`-only
  within-queue preemption policy, which cannot evict an equal-priority job.
  Never unsuspend an unadmitted job by hand and never delete, suspend, scale or
  otherwise alter workloads not bearing our ownership label.

Submit new manifests only through the dry-run-by-default guard:

```bash
python -m training.cluster_submit cluster/jobs/chris-cyber-glm52-<stage>.yaml
python -m training.cluster_submit cluster/jobs/chris-cyber-glm52-<stage>.yaml --execute
```

It rechecks the exact production context, live Kueue preemption contract,
namespace, queue, owner/name labels, absence of priority escalation, and server
admission before it can call `kubectl apply`. It deliberately has no cancel,
delete, suspend, scale or queue-bypass operation.

The invitation alone is not sufficient to launch training. No secrets should
be pasted into this repository or shell history. The following access can be
granted independently and minimally.

## Cluster identity and scheduler

- Nebius project/tenant identifier and the intended training cluster name.
- VPN/bastion instructions if required, plus SSH certificate or SSO flow.
- Scheduler type and login endpoint: Slurm (`sbatch`/`srun`) or Kubernetes.
- Namespace/account/partition, QoS, maximum wall time, and submit permission.
- GPU type, node count/quota, GPUs per node, local NVMe, RAM, and exact
  NVLink/InfiniBand topology. GLM-5.2 is a 753B-parameter MoE, so this determines
  whether adapter training is practical.

## Storage and images

- Read access to a private, immutable copy of the pinned GLM-5.2 BF16 weights.
- Read/write checkpoint object-store URI and short-lived workload identity.
- Shared filesystem paths and quotas for datasets, tokenizer cache and logs.
- Container registry pull/push access; approved CUDA/NCCL/Transformer Engine
  base image. Pin the eventual training image by digest.
- Whether compute nodes have internet egress. If not, mirror Python wheels,
  NeMo RL/Megatron sources, the model and tokenizer in advance.

## Fleet and reward service

- A Fleet **team `fleet`** service credential with read-only access to the
  chosen historical cyber jobs. Inject it as `FLEET_API_KEY` through the
  cluster secret manager.
- For online RL, a scoped reward-broker endpoint capable of launching a fresh
  isolated challenge and returning verifier score by opaque episode id. It
  must not return verifier source, expected flags, grader stdout or environment
  credentials to the trainer.
- Network policy allowing challenge gateways and the reward broker while
  denying cloud metadata, the Fleet control plane beyond required routes, and
  arbitrary public targets.
- A separate evaluation credential/hook so pre/post checkpoints can run the
  frozen Fleet and WebExploitBench protocols.

## Operations

- Weights & Biases or internal metrics project and secret injection method.
- Log retention/redaction requirements and who can view cyber transcripts.
- Checkpoint retention budget, failure/requeue policy and spot/preemptible use.
- Named owner for emergency job cancellation and cost alerts.
- Approval to use the GLM-5.2 MIT-licensed checkpoint and Fleet trajectories
  for this internal post-training purpose.

## First cluster action

Run only the compatibility receipt workload: load the pinned BF16 checkpoint,
round-trip the tokenizer/chat template, execute one BF16 forward/backward and
LoRA optimizer step with the real MoE partition, save/resume, export to the
inference representation, and compare fixed-prompt logits within a documented
tolerance. Do not launch SFT until that receipt is green.
