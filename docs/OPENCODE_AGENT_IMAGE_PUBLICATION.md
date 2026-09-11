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

The required Theseus changes are small and explicit:

1. Provision `fleet/cyber-post-train-opencode-runtime` with immutable tags and
   dev-node pull access, then add it to the dev BuildKit role's
   `push_repository_prefixes`.
2. Add `rewrite_timestamp: bool = false` to `LocalImageBuildRequest`. When true,
   render the image output as
   `type=image,name=<image>,push=true,rewrite-timestamp=true`. Include the field
   in the existing recipe fingerprint and deploy the API.

Do not submit until this command exits zero against a fresh live OpenAPI copy:

```bash
uv run python -m evals.fleet.opencode_image_context validate-openapi \
  --openapi /path/to/dev-image-openapi.json
```

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
