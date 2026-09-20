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

## V7 bounded successor and terminal result

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

V7 started only the baseline task-0 score-free transport canary. After more
than 2,200 seconds, the provider still reported process 1684 as running with no
exit code or signal. Direct bounded reads could not retrieve the sanitized
lifecycle, benchmark-validation, or collection-acceptance receipts. The
supervisor itself was repeatedly waiting on the sandbox management origin.
No private process output, task text, model output, flag, answer, or score was
read.

This crossed the same fixed 1,200-second no-receipt watchdog used for V6. A
write-ahead claim bound the exact sandbox and process before one provider
delete. The provider then reported the sandbox terminated, and the fresh
inventory contained no active row with its ID or name.

- terminal evidence:
  [`qwen38-fresh75-wbe-v7-stalled-release-20260920.json`](qwen38-fresh75-wbe-v7-stalled-release-20260920.json)
- evidence receipt:
  `sha256:e74457b7a8ce8603c64f284ebab029236d0d572ff4b1f2b8fca21331e29edc22`
- release receipt:
  `sha256:776f210fd8e41a2b1c7aed8a6369c57e646ff22aba5a0548665c7d0af6e2b6fb`
- provider state after release: `terminated`
- active matching sandboxes after release: 0

The truthful classification is narrower than “the rollout failed”: the run
did not produce a sanitized receipt that proves the benchmark started, and the
management path could not provide usable lifecycle evidence. Full expansion,
candidate collection, and scoring therefore remained closed.

## V8 management-readback repair

The collection launcher now reopens one immutable qualification receipt from
the live sandbox filesystem and requires exact byte equality before it writes
the collection-process claim or dispatches the rollout. An unavailable file
API or changed receipt stops before any model process can start. A successful
check writes a local, sandbox-bound management-readback receipt.

This changes the launcher's source digest, so the V7 canary and four prepared
expansion replicas are historical evidence only. They must never be launched
under the repaired code. A successor requires a freshly qualified shared
snapshot, freshly sealed launch and pair receipts, and a fresh duplicate
census. The first successor remains one baseline task-0 canary; there is no
fanout until its early lifecycle and terminal collection receipts are accepted.

### V8 live result and bounded startup repair

V8 used a freshly qualified shared snapshot and a fresh duplicate census. The
base task-0 sandbox was created exactly once, but the first immutable-file
readback immediately after creation returned HTTP 500. The launcher failed
closed before writing a collection-process claim, so no model process, rollout,
or score started. The exact owned sandbox was released once and the provider
reported it terminated; a later inventory contained zero active rows with its
ID or name.

This was a startup-readiness defect in our launcher rather than benchmark or
model evidence. The snapshot qualification already retries transient receipt
reads, while the collection launcher had performed only one immediate read.
The launcher now retries this read-only byte-equality check for at most 30
attempts, two seconds apart. It still fails closed before dispatch if the file
cannot be reopened or its bytes differ. Focused launcher tests cover both
eventual success and bounded failure.

- sandbox: `p7hs0a56ocmglguu4adcx`
- name: `q38-b-f75-c1-v8-t00`
- result: no collection and no model call; released and verified terminated
- sanitized evidence:
  [`qwen38-fresh75-wbe-v8-management-readback-release-20260920.json`](qwen38-fresh75-wbe-v8-management-readback-release-20260920.json)
- evidence receipt:
  `sha256:236e53f47b899a59992f68fe9be812d9be81216080ada9db124fb74cf0b41a5b`

Because the repair changes source bytes bound by qualification, the V8 launch
receipts are retired. A successor must use a newly qualified snapshot, newly
sealed plans, and another exact duplicate census. Fanout remains closed until
that successor's task-0 collection evidence is accepted.

## Prepared expansion

Four sealed V7 replicas contain 15 distinct benchmark tasks per arm. Together
they represent 60 baseline and 60 candidate score-free attempts. They all bind
the bounded runner. The baseline 60 may start only after the V7 task-0
transport canary accepts and a fresh duplicate census remains clean. Candidate
work also requires a newly Ready exact candidate route and fresh live-parity
and duplicate checks. No expansion was launched when this note was written.

These V7 plans are now retired because the management-readback repair changes
the launcher's sealed source digest. The table remains as historical evidence;
none of these exact plans may be launched. A fresh shared-snapshot
qualification and new launch receipts are required first.

| Replica | Base plan | Candidate plan | Pair receipt |
|---|---|---|---|
| 0 | `sha256:888f10e160b3dd59235169f6ca81dfdf9e5196e848f3248674ea2cfa0d8b1258` | `sha256:8d44bf3aeab5f668acd75db54d7d1e1a054a915a4a9dc835b465ce8cdec66050` | `sha256:35550529bbf4240aa14b3efc8d0011f60bd57b1a3e7b8de5708927189b0ac6b6` |
| 1 | `sha256:8d89242ec2e9a7cfbf5c8756c2ee491d405f0c485f07b092f43550b301974329` | `sha256:4aad348843755d337eb2bc949e0f6cdf87829b0af21a2f3c939c2437594db102` | `sha256:5c5c11df2ede02444217b447913d3ce03d069c5f885f3f4eaa5a11a21fc311ef` |
| 2 | `sha256:9cea7b2d5a068ba40ff4b9167e6cf4b2a457b59b0173136534cbfa4c56c248c3` | `sha256:f50207816bcec4971e2110323bb04669d74b595bbcb9573155ceb63f014e25fc` | `sha256:accb20b4498d0aad391b80f274bc554b97d1131afb11aaad8a3248d3f7193448` |
| 3 | `sha256:8df367a41275c8945119a57739560931ea15c59179da3f623e58493d487c8e64` | `sha256:9dd416b740a342fc010b125a47cab9ad899c5456624db935445d1448a7b21322` | `sha256:740ba340ab891c1fa4ef94e6dc15786621cb4d91492a49050480e55eb0cc7c4c` |

## V17 shared snapshot and the real-collection network gate

The current shared snapshot qualification completed successfully and released
its qualification sandbox. It made no model, benchmark, or judge request. Its
purpose was only to prove that the exact source files, images, benchmark task
order, one-task projects, and 15-task projects can be reopened and validated.

- qualification transaction: `wbe-fresh75-pair-v17`
- snapshot: `mp3wre17q6fwxbndexd50`
- qualification plan:
  `sha256:87bea7994e03a23dfbc6d8629cdc558510f4ca86cf4d9ef01304dd1e50369aba`
- completion receipt:
  `sha256:b7001d59237534d8fa581102be882c5e18455ee7fdd492036b3605633b1c3bc3`
- one-task receipt:
  `sha256:81995f671c586319334cdf80553980c116f925869c01e96e5f7e3ba92ad30480`
- 15-task receipt:
  `sha256:dcd5cf7df15f601ceb563346e207fb44ea885730d5c9fe60ce3794c5aca39e4d`
- release receipt:
  `sha256:b85197014906f63c4b47b39c45acf6c7e5176156a3127b1c0aade0abbee0bae3`

The qualification sandbox correctly had no internet access because it never
needed to contact a model. A real rollout sandbox does need to reach the frozen
Fleet inference address. A later task-0 rollout read back a provider network
policy with internet access disabled. Its bounded CAGE preflight then produced
only DNS-resolution retries, before OpenCode, the benchmark, any model request,
or scoring. That is a launcher/network-contract defect, not model or benchmark
evidence.

Commit `48044a92` fixes the boundary explicitly. Every real collection plan must
request internet access; the provider must read back the same policy before the
launcher writes a process claim or dispatches the collection. Snapshot
qualification remains offline. Focused tests cover both a plan that omits this
policy and a provider readback that does not match it.

Exactly one corrected task-0 baseline plan is sealed but remains unlaunched
until the known-invalid predecessor reaches its bounded terminal state and its
exact sandbox is released:

- corrected task-0 plan:
  `sha256:6012b00c189d49ea2679c137f1aab1e2296427fd6d992bf743fa0b0640d4e416`
- corrected protocol:
  `sha256:3a83c9034c61c89e63de4a44a9531787632cc38e79efd43bd2ff07289093cd39`
- corrected fixed-control view:
  `sha256:cce74a1d66f407728ab0c1a6491fb585ce25f0bc71dbdf3549527356314e4555`

Four new baseline expansion plans are also sealed against the accepted 15-task
qualification. Each contains 15 distinct, one-attempt, score-free collections.
They share fixed-control digest
`sha256:dc03c1012abc1c36702fc470b0b149a9cb1e02a797bf7f66c905dc1bb963681c`.
They must remain unlaunched until the corrected task-0 collection produces a
valid terminal rollout artifact.

| Replica | Prepared baseline plan |
|---|---|
| 0 | `sha256:47831532020ec76e0678e6b817823247e679642512576c4a8d8039df13f8bc8d` |
| 1 | `sha256:0ad55f3b8eb49698acc07d06ea99ccd0a57c08158341072d62f95190949d5120` |
| 2 | `sha256:6d17aedd63212072cd766eb07ceea2e779e3ff1f23bf61f45ad3fcd8af3e6075` |
| 3 | `sha256:0dfa004885b53d11c2ef89cedc58c78e8f68cac04f59aabaa448db2440fa64d7` |

## V18 deterministic qualification repair

The private V17 execution directory was no longer present locally, so its
qualification files could not be reopened by path. The accepted V17 filesystem
snapshot remained available. Four uniquely named, no-internet metadata-only
restores were used to recover its immutable qualification bundle. They made no
model, benchmark, or judge request. Every restore was terminated after use and
reconciled to zero active inventory matches:

| Recovery sandbox | Provider ID | Release receipt |
|---|---|---|
| `wbe-recover-f75-pair-v17-meta-v1` | `j089d4afh720vbs305zyb` | `sha256:7d3722a21b15d5985a6f4d2fe8eb71b97bd76b52850fd586d368f1679f1980fb` |
| `wbe-recover-f75-pair-v17-meta-v2` | `af0dibija9jjbwlf2tb0x` | `sha256:ec6fe7c774b717c3ffb019926e317757166b7d46456eed9878136dee60688d81` |
| `wbe-recover-f75-pair-v17-meta-v3` | `eszdiulat9rip4hoduqpc` | `sha256:f0b3a2d74050dc094854f632a8c5a747053f151e3c0ee27d7b987bf912cc0965` |
| `wbe-recover-f75-pair-v17-meta-v4` | `tbbk8z1q398212mofuu26` | `sha256:fc62ce6aab23da1f29d51da7ddecd46b8d61709fde7ebe3b4d6facebda734dd6` |

The final restore used the repository's bounded small-receipt reader, including
its captured-output fallback when TensorLake's direct file endpoint returned a
server error. The recovered bundle self-digest is
`sha256:185f247093f2f9a2f147179c38189161da0d2989364a734a22f6121852b4de3a`.
It recovered the exact project template, base and candidate model registries,
and these previously qualified runtime images:

- evaluator: `cage/wbe-evaluator:matched-v3` at
  `sha256:eb5f1948ee3f7d16f66a01c86c4c1cd8950f912c91060e0848ce92942568322a`;
- OpenCode agent: `cage/opencode-webexploitbench:1.18.27-v12-4b39ba8f-d142a7ce`
  at `sha256:7976411d3b5b8eacbb887b8e6148fc14680513b0d35f202fbd7a6205080a329c`;
- network proxy: `cage/netproxy:v1.1` at
  `sha256:0eae9d69a8c092e1b1fbe8a26bacfcce505892b99783f554c7a969adb2080c45`.

Those exact inputs and the reviewed CAGE commit were resealed with the current
qualification, launcher, and benchmark-source bytes as transaction
`wbe-fresh75-pair-v18`. Its source snapshot remains the accepted V17 snapshot
`mp3wre17q6fwxbndexd50`. The new local qualification plan self-digest is
`sha256:bfa3ccd46a9208254c78cc39356f42d62c29c0856338e64b528e241722418df5`
and its file SHA-256 is
`sha256:ef408727275d4975f6df7125812ecbd69ad3c25112a05031f9624fee2e91977f`.
Focused qualification, collection-launcher, supervisor, rollout-bundle, and
deferred-scoring tests passed before execution. V18 qualification execution,
one base-only score-free task-0 canary, and separate deferred scoring are the
only authorized next actions. Candidate activation and wider fanout remain
closed until the base canary is accepted.
