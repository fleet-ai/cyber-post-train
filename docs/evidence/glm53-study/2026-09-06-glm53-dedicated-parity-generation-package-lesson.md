# Dedicated parity generation-package lesson

The v31 server and request-counter watchdog reached digest-valid READY and ACTIVE state, but the score-free parity Pod exited before issuing a model request. The materialized runtime package omitted the module that supplied the pinned DinD and UV image constants. Its rendered `run.sh` also named the v30 parity module while mounting a v31 authorization receipt. Fixing only the later schema mismatch would therefore leave the earlier import failure intact.

Every fresh generation must now prove all of the following from the exact rendered ConfigMap rather than from mocked helpers:

1. Materialize every packaged file into an otherwise isolated directory and import the executable parity module there.
2. Require `run.sh` to name the current generation's module exactly once and reject inherited executable v30/v31 identities.
3. Require the mounted authorization schema, package schema, server title, run directory, Job name, and result root to use the same generation.
4. Recompute `run_sha256` and `package_sha256` after any executable rewrite.
5. Keep Fleet task, session, verifier, and scoring calls at zero; release the exact server through the UID-bound rail on any pre-request failure.

The v31 identity is frozen and must never be retried. The v32 package remains held until independent review.
