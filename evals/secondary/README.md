# Secondary black-box benchmark stream

This directory keeps work moving while gated WebExploitBench access is pending.
It supports only public, official challenge containers and treats all benchmark
materials and outputs as **evaluation-only**.

## What is prepared

| Benchmark | Immutable source | Use | Important limitation |
|---|---|---|---|
| XBEN | `xbow-engineering/validation-benchmarks` at `a16cb1ba0701b20d63913846536fc7b0672082af` | Closest public structural smoke: 104 black-box web CTFs with injected flags | XBOW now calls it outdated; modern agents report near-saturation. It also carries an explicit do-not-train canary. |
| CVE-Bench | official `v2.1.0`, peeled commit `332f0ec0fc95c4a4d35fe7458a73969aad1aa1d4` | Harder secondary transfer test on 40 real web CVEs; use `zero_day` | Prompts enumerate impact objectives and the apps are known vulnerable versions, so it is less open-ended than Fleet/WebExploitBench. |

XBEN is an engineering and harness sanity check, not the headline capability
number. CVE-Bench zero-day is the better temporary scientific signal. Neither
benchmark may be used to tune prompts, rewards, training examples, retrieval or
post-training data selection.

Official sources:

- <https://github.com/xbow-engineering/validation-benchmarks>
- <https://github.com/uiuc-kang-lab/cve-bench/tree/v2.1.0>
- <https://inspect.aisi.org.uk/providers.html#openai-compatible>

## Dry-run plans

From the repository root:

```bash
python -m evals.secondary plan \
  --config evals/secondary/configs/xben-smoke.json \
  --checkout /tmp/cyber-evals/xben

python -m evals.secondary plan \
  --config evals/secondary/configs/cvebench-smoke.json \
  --checkout /tmp/cyber-evals/cvebench
```

The plans contain only environment-variable **names**, never values. Both pin
Fleet's OpenAI-compatible gateway model to `glm-5.2-fp8`, use pass@1 and bound
concurrency to one. The runner maps `FLEET_API_KEY` to Inspect's
`OPENAI_API_KEY` only in the child process; it never writes or prints the value.
It refuses a non-empty result directory, preventing an accidental duplicate
paid run. Use a new `run_id` for a changed protocol; use Inspect's `eval-retry`
for a resumable infrastructure failure.

## Bootstrap and smoke

Use an isolated Linux/amd64 worker for the real run. CVE-Bench explicitly calls
its arm64 support experimental. Keep checkouts and logs in access-controlled,
evaluation-only storage outside the training repository.

```bash
export FLEET_API_KEY=...  # shell or secret manager only

python -m evals.secondary bootstrap \
  --config evals/secondary/configs/cvebench-smoke.json \
  --checkout /secure/evals/cvebench --execute

python -m evals.secondary run \
  --config evals/secondary/configs/cvebench-smoke.json \
  --checkout /secure/evals/cvebench --execute
```

For XBEN, the official repository supplies applications but no model runner.
The included Inspect task supplies a minimal ReAct shell agent and exact flag
scoring. Before invoking it, build the selected application with an ephemeral
flag and ensure its Compose network is isolated from non-challenge targets:

```bash
export XBEN_FLAG="FLAG{$(openssl rand -hex 20)}"
docker compose \
  -f /secure/evals/xben/benchmarks/XBEN-019-24/docker-compose.yml \
  build --build-arg "FLAG=$XBEN_FLAG"

python -m evals.secondary run \
  --config evals/secondary/configs/xben-smoke.json \
  --checkout /secure/evals/xben --execute
```

The XBEN execution path deliberately requires `XBEN_FLAG` from the environment.
It never generates or serializes it. For production, run the Compose workload
in an outbound-denied worker namespace; the task prompt additionally confines
the agent to the selected target.

## Fleet Agent Runtime

Fleet Agent Runtime can run the same model, but public XBEN/CVE-Bench containers
are not registered Fleet task keys. Registration must preserve the official
container bytes, hidden verifier and outbound-denied network policy. Until that
registration path is reviewed, the official Inspect harness is the
reproducibility authority; do not upload a rewritten challenge merely to obtain
a Fleet dashboard link.

## Why this is not a gate workaround

This stream does not scrape, mirror or reconstruct gated WebExploitBench data.
It uses separately licensed public benchmarks while the individual Hugging Face
request follows its normal approval process.
