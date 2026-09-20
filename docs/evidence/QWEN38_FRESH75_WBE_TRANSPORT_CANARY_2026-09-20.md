# Fresh75 WebExploitBench transport canary — 2026-09-20

This note records the score-free OpenCode collection gate for the sealed
Qwen3.8-27B base-versus-Fresh75 comparison. It contains no task text, model
output, flag, answer, score, or credential.

## Scientific boundary

- Baseline route: `qwen3.8-27b`, inference-model UID
  `d06c0531-7181-41ad-a3fd-cd8e3e774ab9`.
- Candidate route: `chris-q38-fresh75-step230-wbe-v1`, inference-model UID
  `fa7dafb6-a700-4eb7-b955-e40255420617`.
- The candidate remains paused while the eight-node production allowance is
  fully occupied by training. Its retired predecessor remains forbidden.
- Rollout collection is score-free. Scoring is a later, independent action.
- A baseline-only collection may qualify transport while candidate capacity is
  unavailable, but it is not a matched capability result. A comparison is
  reportable only after both sealed arms complete and the separately sealed
  scoring stage accepts both.

## V5 terminal result

The first repaired-path baseline canary used sandbox
`1b51dhjuqbqdiv4iqber2`. It did not begin a benchmark attempt and did not start
scoring. It stopped at CAGE's fixed preflight request, preserved filesystem
snapshot `nsk098xawc7eqr41rydun`, and released the sandbox.

- terminal status: `failure_preserved_and_released`
- failed gate: `cage_preflight`
- failure receipt:
  `sha256:c3c8198f82bae7b8a6fb2f8c65ef09cf4fccff8f2c17bfea45033115cfa5985c`
- snapshot acceptance receipt:
  `sha256:e15d51e37024fdf87dd0d626cfa702e2b143e77adec4348adc00ac8af4a7b66e`
- release receipt:
  `sha256:66c9be143b698536bbd7247c6692bfa2b9ab750afc391ea51e7dbe608096f6ef`

Two short no-network diagnostic restores inspected only the private preflight
failure class and file digest. They did not expose or persist the private log.
Both were released:

- `qi9lhtckz05jnoyxdt8tu`
- `ywgjc5thsjymzd0w97olc`

The sanitized result was unambiguous: all three bounded preflight attempts
ended in DNS resolution failure and `PreflightError`. There was no path,
configuration, authorization, model-name, HTTP-status, Docker, or permission
failure signature.

## Repair

Commit `4ea37a35` extends only the DNS-proven retry window from three attempts
at 15 seconds to twelve attempts at 30 seconds. The retry classifier must see
an explicit DNS-resolution failure. Any other error still stops after its first
attempt. Therefore this repair cannot replay an accepted model request or a
benchmark attempt. Focused readiness/supervisor tests and Ruff pass.

## V6 terminal bootstrap stall

The fresh V6 base plan is
`sha256:65dace903c2eb0d9ac81ba0267d2150e8f1749c434c7c63c64cd4c8220abf5b7`.
Its paired candidate plan is
`sha256:bfdf3699214edc6c02a587caebf23e09fe388cedefe6bff220c0c4f03ab23e6f`,
and the pair receipt is
`sha256:c1f82f7a835b6f4795dd06ebe49c6fda3735e12f311504527c4674bda9858535`.

At launch, a fresh provider census found the create-once name absent and the
shared base route was independently Ready with two replicas in both the Fleet
API and Kubernetes. The score-free collection was:

- sandbox: `8b990zf3pca8rpek6qkfj`
- process: `1510`
- campaign: `q38-base-f75p230-oc-wbe-c1-v2`
- terminal supervisor plan:
  `sha256:ee6f8d863e7909ee30169a96d36849f47b85d917fb2fdbd83aac627da270931e`

The only sanitized phase indicator available while V6 was live reported
`bootstrap` / `initializing`, and the provider process remained active. Separate
read-only existence probes found no CAGE-preflight, benchmark-validation, or
collection-acceptance receipt, so no benchmark attempt or scoring stage is
known to have started. A process-tree probe found the controller waiting on a
Docker child. The runner used by V6 did not place a deadline around each CAGE
preflight Docker call; therefore one stuck call could outlive the otherwise
bounded DNS retry schedule.

After more than twenty minutes with the same evidence, the stall met the fixed
watchdog. A write-ahead release claim bound the exact sandbox and process before
one provider delete. The provider then reported the sandbox terminated, and a
fresh inventory contained no live row with its ID or name.

- sanitized stall evidence receipt:
  `sha256:e4455c5c0c32d8de45137632db88a86ed7980937af71735cd6633ce7c4c0e1db`
- release receipt:
  `sha256:aa1fab471b8592f18b4233de20d66938452886c99538741a4dc66b0dd2c9e9f9`
- terminal classification: bootstrap transport stall; zero accepted benchmark
  attempts and zero scoring attempts

The successor runner now gives every CAGE preflight Docker call a strict
five-minute deadline. A timed-out call is not treated as DNS and cannot be
retried: it produces a terminal score-free collection failure for preservation
and release. This hardening did not alter V6 or its sealed runner digest. Any
successor must rebuild and reseal its plans against the new runner digest.

## V7 bounded successor

Exactly one successor was sealed against the bounded runner. A fresh provider
inventory of 339 sandboxes found both create-once V7 names absent and no live
V6 predecessor. The duplicate-census receipt is
`sha256:0f3842425c3fa034065b7889e05e53b1eb08d8b6e52c5796a9663a70b648b009`.

- base plan:
  `sha256:46f8ae5480f4aaf1fb331d41ed0d0349509b67ba56ae6c0aef081b1ff8e86522`
- candidate plan:
  `sha256:753d26e02d69fc23b83c677508c356276826e57ce7f12b72e6a5e24a43b1f66a`
- pair receipt:
  `sha256:dde4772aeba1b749fb8c9a3c4afb01c2e4f96422d7d8ae81c59ac57155e6c2a1`
- bounded runner:
  `sha256:bd4aa3c1cf27f63ae151c818873e6070a0a23f55b34934457338c357aafb3742`
- baseline sandbox: `udrfx303h58nqrl6ipil7`
- baseline process: `1684`
- campaign: `q38-base-f75p230-oc-wbe-c1-v3`

V7 starts only the baseline task-0 score-free transport canary. Its terminal
supervisor will preserve an accepted or sanitized-failure snapshot and release
the exact sandbox. No full expansion, candidate collection, or scoring is
authorized until this canary accepts.

## Prepared expansion

Four sealed V6 replicas contain 15 distinct benchmark tasks per arm. They are
historical and cannot be launched against the changed runner digest. Together
they represent 60 baseline and 60 candidate score-free attempts. The baseline
60 must be rebuilt and resealed only after the V7 task-0 transport canary
accepts. Candidate work
also requires a newly Ready exact candidate route and fresh live-parity and
duplicate checks. No expansion was launched when this note was written.

| Replica | Base plan | Candidate plan | Pair receipt |
|---|---|---|---|
| 0 | `sha256:8787b0a4ca1ca2789b9db3fa0d7be1d3e33db0408d8573871f02499e45be055f` | `sha256:acf32c2bb8b2cc897e10d988a3cbde5be867f26c7565f2286ededc4eed7cef9d` | `sha256:69b991e43f291126b49f51bfc5a889a847e0679d9e2389602de0079064b65832` |
| 1 | `sha256:29f0eb7c376df66523599a4004ee8c2a8bf782b45c170a59f0bc171f2fda0ca0` | `sha256:247137ccef47f6f21682f9a8809ec4c2f1cf2f568b9971ebb1dfcff3d773ce50` | `sha256:cbb47c8d7534ef69a95cb433609006c08228cc577f579d08803dbda9f81a659b` |
| 2 | `sha256:8c5de3aeba33560786e04d76a8f14948699890c018951750b2a863a8e4d29568` | `sha256:0548cb7881edd2dc55d9fa0b4f46868e09c68fbf4bfdeb7005105e7567196364` | `sha256:ea8338f1a9d72fee1480a3a3d22bb3a27b9351d6ab2713408e7c695cea09fbaf` |
| 3 | `sha256:daba55960be71ac13a45499c63f830e17749482a7fb198b1cc14bc89d52a7275` | `sha256:e1405d1f43de910c300422469d8ab93d5e876f538ac7572abf3c5eb94297dba4` | `sha256:8b4457d1a459790d098a7dba7ae439c02bdfd1452fc44bec175a6338d4e79fc3` |
