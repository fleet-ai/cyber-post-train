# Qwen3.8 checkpoint serving parity — 2026-09-20

This record contains serving identities and content-free probe results only. It
contains no benchmark task, prompt, generated answer, score, or credential.

## Fresh75 step 230

The previously live route `chris-q38-fresh75-step230-web-v1` is **not valid for
matched evaluation**. Its checkpoint bytes were correct, but it used an older
SGLang image, the plain SGLang entrypoint, and omitted a current base-model
runtime flag. It was paused and released before the replacement started:

- InferenceModel UID: `d048104e-ade6-49d2-b088-d3819e606061`
- terminal retained phase: `paused`
- ready replicas: `0`
- retained Pods: `0`

The create-once replacement is `chris-q38-fresh75-step230-web-v2`:

- InferenceModel UID: `e624a6f0-8748-433c-973e-aefc7f22ddf6`
- accepted API resource version: `31081480`
- accepted Kubernetes resource version: `31081593`
- staged model root: `/models/chris-q38-fresh75-step230-v1`
- serving model root: `/scratch/models/chris-q38-fresh75-step230-web-v2`
- payload revision: `sha256:36eec01f1dd3f0d47f6b099e79970d9533cf59c37d8e15478907413dd162e029`
- exact runtime image: `sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9`
- normalized base/candidate execution contract: `sha256:d82d78721f4ec8d4b0d6228242df4838b38086a48108e57fc6fe4e7be1fa3662`
- model-info projection: `sha256:27ddee5a2cc8fe14b426d93ec8a64ebc52aba812a954fb665851e0c3a1dfa5b4`
- server-info projection: `sha256:1a083cc36a5272b16128c0c80976346856bfe7f6bb2d1a82f528624aef818086`
- live parity receipt: `sha256:8ccabd95d993cbf73ce8be1ad80a4673c4c18d7fa1aa79c46cbda0ecf467a5d1`
- live parity receipt file SHA-256: `98166ec8a39ff43462fbd0d744ee32e1d03095deb63693d54afaf9e4ca13c0e8`
- create/readback receipt: `sha256:985bba9fce3977b43efe3650bcec7208a816afe0bc6f610bb607a8e9c3a86118`
- create/readback receipt file SHA-256: `42d474e473cf517b8e11435f7a95afa1c8f6efb673e62f9649109e83b4be3d8e`

The accepted route is BF16 with no weight quantization, tensor parallel 1,
data parallel 8, a 262,144-token context, FP8 KV cache, Qwen3 reasoning, and
Qwen3-Coder tool parsing. The current base and candidate matched on the exact
image, compatibility entrypoint, runtime flags, resources, model metadata, and
server metadata. Its single serving Pod had zero restarts and the exact image.
The fixed structured-tool probe passed. Two fixed logit observations had
finite scores and identical greedy token identity within each route. Exact
floating-point equality and maximum score delta are recorded rather than used
as the gate because the shared base can route identical requests to different
healthy replicas. The candidate's logit projection differed from the base
route, confirming that the probe reached different checkpoint weights.

The full prompt-free receipt was captured locally at:

`/private/tmp/cpt-checkpoint-promotion-live-v1/.private/fresh75-v2/live-parity-v3.json`

The ignored local path is not the durable authority; the identities and hashes
above are the durable, reviewable evidence. The old route must remain frozen
historical evidence and must never be reused as a serving template.

## Teacher-v5 step 186

The create-once matched route is `chris-q38-teacher-v5-step186-web-v1`:

- InferenceModel UID: `10b1b0ce-7b02-429e-8361-084a3039f248`
- accepted API and Kubernetes resource version: `31080713`
- staged model root: `/models/chris-q38-teacher-sft-v5-step186-v1`
- serving model root: `/scratch/models/chris-q38-teacher-v5-step186-web-v1`
- payload revision: `sha256:e6d59f5a58ce54b913d775cd71b81c64bd54637b53795df82baec0f71efefaf9`
- exact runtime image: `sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9`
- normalized base/candidate execution contract: `sha256:d82d78721f4ec8d4b0d6228242df4838b38086a48108e57fc6fe4e7be1fa3662`
- model-info projection: `sha256:27ddee5a2cc8fe14b426d93ec8a64ebc52aba812a954fb665851e0c3a1dfa5b4`
- server-info projection: `sha256:1a083cc36a5272b16128c0c80976346856bfe7f6bb2d1a82f528624aef818086`
- live parity receipt: `sha256:cf2922f723f01b62d4d5e71215aec7931b0e88ccfad4b09df526729f9b3ed946`
- live parity receipt file SHA-256: `082fbe2bd7f13e315c45491930a77934a48ddace462e2d7a88ea65d79830ef64`
- create/readback receipt: `sha256:125b701333fe1f94394ccec54a3203082571e265a95212d9f0ddd61756e42fe5`
- create/readback receipt file SHA-256: `92ba5b0e19f5b7328ba5c0b0902902740b75c8abada23784d617daf5c625b1cf`

The route has the same accepted serving profile as Fresh75. Its serving Pod
UID is `2ef5ab8c-d460-4c41-852d-222118556e0d`; it was Ready with zero restarts
and the exact image at capture. Catalog routing, model and server metadata,
the structured-tool probe, and the two finite greedy-logit observations all
passed. Its token identity and logit projection differed from the base on the
fixed content-free probe, which is consistent with the candidate weights being
used.

The full prompt-free receipt was captured locally at:

`/private/tmp/cpt-checkpoint-promotion-live-v1/.private/teacher-v1/live-parity.json`

## Self-SFT step 44

The staged, create-once route `chris-q38-self-sft-step44-v1` was resumed only
after its exact consumer slots were available:

- InferenceModel UID: `05057134-546f-484e-9672-3f9ca50cec77`
- accepted API resource version: `31092393`
- accepted Kubernetes resource version: `31092820`
- staged model root: `/models/chris-q38-self-sft-step44-v1`
- serving model root: `/scratch/models/chris-q38-self-sft-step44-v1`
- payload revision: `sha256:a0e55bebe9d78ac76c012446ae0147a421cd333e2a6bf2bb6f4fb3450fa776fe`
- exact runtime image: `sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9`
- normalized base/candidate execution contract: `sha256:d82d78721f4ec8d4b0d6228242df4838b38086a48108e57fc6fe4e7be1fa3662`
- model-info projection: `sha256:27ddee5a2cc8fe14b426d93ec8a64ebc52aba812a954fb665851e0c3a1dfa5b4`
- server-info projection: `sha256:1a083cc36a5272b16128c0c80976346856bfe7f6bb2d1a82f528624aef818086`
- accepted staging receipt: `sha256:20460ed56f955b7054d60d944aa9e0e1a95a4eef53907df3a0dc2b345c3e87a8`
- live parity receipt: `sha256:66992f74ac736422e0b724681bd7e0cab126a7d90c19a99598e182c8677b3572`
- live parity receipt file SHA-256: `676a89af48d7802dd32eb0c3864e6d4ffe05029619f1c8c67f464949a33f93bc`

The serving Pod UID is `c1271f58-ca3e-41e7-8a01-1a3603cb9619`. It was Ready
with zero restarts and the exact image at capture. The route matches the same
BF16, unquantized, 262,144-token current-base serving profile as Fresh75 and
teacher-v5. Catalog routing, model and server metadata, the structured-tool
probe, and two finite greedy-logit observations all passed. The candidate and
base produced different stable token identities on the fixed content-free
probe, confirming that the self-SFT checkpoint rather than base weights served
the request.

The full prompt-free receipt was captured locally at:

`/private/tmp/cpt-checkpoint-promotion-live-v1/.private/self-v1/live-parity.json`

## Released stale teacher3k step-10 route

The capacity census also found an older project-owned route,
`chris-q38-t3k262-s10-web-v1`, still serving without an active consumer:

- InferenceModel UID: `33c6ede6-1646-4f3a-a2c5-da6a34009ffc`
- last serving Pod UID: `420e82c9-d0df-4f14-b5fd-915c6b1f61a1`
- model revision: `sha256:d6301dba2cc0e6bc05ea48d2793584d4ca725cba8863dc4ac6ef75bdbbfa668b`
- checkpoint identity: teacher3k SFT step 10
- API resource version before pause: `31040514`
- last recorded useful request time: `2026-09-18T14:13:05Z`

A score-blind WebExploitBench census proved that no active or planned campaign
consumed this route. The full V23 campaign is base-only and the checkpoint
backlog contains Fresh75, teacher-v5, and self-SFT only. Historical step-10
campaign evidence remains preserved and was not changed.

Exactly one supported pause request used the immediately preceding resource
version. The API accepted resource version `31104916`; the same route UID then
reached `paused` at resource version `31105742` with routing disabled, zero
desired replicas, zero ready replicas, and zero active Pods. Kubernetes
readback proved the exact former Pod UID absent and found no remaining Pod for
the model label. This released its eight GPUs without deleting historical
route or evaluation evidence.
