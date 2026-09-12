# OpenCode agent-image publication

Fresh Fleet base/post-training comparisons require a pullable immutable agent
image, not a local Docker image or an SFS archive. The exact target is OpenCode
1.18.27 on linux/amd64. Its current create-once plan is
[`opencode11827-agent-image-publication-plan-v2.json`](../configs/evaluation/opencode11827-agent-image-publication-plan-v2.json).

## Current gate

The source is ready but publication is intentionally blocked. The dev BuildKit
role can push only `fleet/miles-trainer`, which is a trainer-catalog repository
and must not be repurposed. The live local-build API also has no
`rewrite_timestamp` field and renders image output without BuildKit's
`rewrite-timestamp=true`. Submitting now would either be rejected before a build
or create a predictably failing build Job.

The required Theseus changes are small and explicit. They span two independently
applied Terraform states plus the Training API deployment; merging source alone
does not open this gate:

1. Provision `fleet/cyber-post-train-opencode-runtime` in the account's core ECR
   state with immutable tags. The dev `ecr-pull` role already grants read access
   to `fleet/*`; verify that live role after apply rather than adding a second
   pull credential.
2. Add `fleet/cyber-post-train-opencode-runtime` to
   `k8s/training/terraform/buildkit-role/fleetai-training-dev.tfvars` and apply
   the dev BuildKit-role state. This is the repository-scoped push grant.
3. Add `rewrite_timestamp: bool = false` to `LocalImageBuildRequest` in
   `services/fleet-train-api/src/fleet_train_api/local_image_build.py`. When
   true, render the image output as
   `type=image,name=<image>,push=true,rewrite-timestamp=true`. Include the field
   in the existing recipe fingerprint only when true; the default false path
   must retain byte-for-byte legacy fingerprints so existing Miles catalog rows
   do not conflict. Test both values, build the Training API, and move the dev
   `image-build-api` pin to that tested image.

The image-build OpenAPI is served at
`https://api.ft.dev.flt.build/v1/images/openapi.json`; the generic
`/v1/openapi.json` is a different service and cannot establish this gate. A
read-only live check on 2026-09-11 found OpenAPI SHA-256
`44d1488cb0b1573ab446ed6c882dcde5d3d73460a1404790596d546ac625e48f`,
no `rewrite_timestamp` property, and the same source bytes on current Theseus
main as the blocked plan. Do not upload a context or submit a build merely to
reconfirm that schema result.

Do not submit until this command exits zero against a fresh live OpenAPI copy:

```bash
uv run python -m evals.fleet.opencode_image_context validate-openapi \
  --openapi /path/to/dev-image-openapi.json
```

Also require a reviewed Terraform plan for both infrastructure states and a
live readback of the dedicated repository plus builder push policy. OpenAPI
success proves only the request/rendering half, not ECR authorization.

## Recreate the exact context

Download the official `opencode-linux-x64.tar.gz` release asset through the
normal authenticated developer workflow. The preparer rejects any asset or
binary whose digest differs, rejects unexpected archive layout, and writes a
deterministic create-once context containing only the Dockerfile, binary and
empty workspace directory:

```bash
uv run python -m evals.fleet.opencode_image_context prepare \
  --asset /path/to/opencode-linux-x64.tar.gz \
  --output /new/path/opencode11827-context.tar.gz
```

The plan binds the expected archive digest and byte count. Upload those exact
bytes with `POST /v1/images/contexts` and its returned presigned PUT. Keep the
presigned URL in memory and never save it; attach the Fleet bearer only to the
Fleet API calls, never to S3. Pass the server-returned context ID to the request
renderer:

```bash
uv run python -m evals.fleet.opencode_image_context request \
  --context-id <server-returned-id> \
  --context-sha256 e1d143706870c0a283c4d816d6f36c30095cc7dba08d20a90a9dbf09d27ed27e \
  --source-commit 43c6a22e14fc09c9f108cd4efea350323198c0f4
```

Review the rendered JSON byte-for-byte against the plan, then submit it once to
`POST /v1/images/build/local`. A timeout is ambiguous: observe the deterministic
build name; never repeat the POST blindly.

## Acceptance

Terminal BuildKit success is only publication. Bind its returned
`repository@sha256` and qualify that exact digest in one direct, server-dry-run
checked, zero-GPU dev Pod using `imagePullPolicy: Always`. Acceptance requires
the labels, platform, non-root user, private HOME, tool order, OpenCode version,
and both native compaction/autocontinuation treatments to match. The Pod must
exit zero with no restarts, then be deleted and confirmed absent. Only that
receipt may fill an evaluation plan's `agent_image_digest`.
