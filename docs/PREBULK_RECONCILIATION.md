# Exact pass@4 pre-bulk reconciliation

The bulk rollout remains held until two independent CPU Jobs finish. The source
observer validates the exact 100-task inventory, both accepted generation-5
canaries, their global execution claims, their authoritative session metadata,
and the absence of every planned bulk Job, output root, claim, and Fleet run
identity. It stores only sanitized identities and digests.

The source Job cannot truthfully claim its own eventual success. Therefore a
separate create-once acceptance Job runs only after the source Job is exclusively
Complete. It validates that exact source Job and Pod by UID, repeats all live
duplicate checks, and writes the final digest-valid `TERMINAL.json`. The final
receipt enumerates the 798 remaining statistical cells without task content,
transcripts, flags, or scores. Infrastructure failure preserves all artifacts
and requires a reviewed successor generation; neither Job retries automatically.

The acceptance Job also copies the exact raw bytes of both projected G5 scoring
release receipts, without reformatting, to:

- `/mnt/sfs/jobs/chris-cyber-exact100-prebulk-reconcile-v1/qwen3.8-27b-generation5-release.json`
- `/mnt/sfs/jobs/chris-cyber-exact100-prebulk-reconcile-v1/glm-5.3-generation5-release.json`

Their original file and receipt digests are frozen in `OBSERVATION.json`, and
each canary-gate receipt points to its corresponding SFS copy. This makes the
terminal bundle portable to the desktop bulk validator; the projected runtime
ConfigMaps are not assumed to exist in the later immutable bulk snapshot.

The observer also authenticates inventory v2 as a producer, rather than trusting
its terminal file alone. It requires the exact Job and Pod UIDs recorded by the
terminal to be exclusively Complete/Succeeded with zero restarts and exit code
zero. With GET-only ConfigMap access it validates that the inventory bootstrap
and submission-intent ConfigMaps are immutable, their package commit and package
digest agree, and every bootstrap payload matches the package manifest. The
acceptance Job then raw-copies `data.package.json` write-once to
`/mnt/sfs/jobs/chris-cyber-exact100-prebulk-reconcile-v1/exact100-inventory-package.json`.
`OBSERVATION.json` and `TERMINAL.json` bind that copy’s file digest, package
digest, package commit, ConfigMap UIDs, and producer Job/Pod UIDs. This is the
sanitized `exact100_inventory_package` authority consumed by bulk release.

`TERMINAL.json.runtime` and `observer_job_succeeded` describe the source Job.
`TERMINAL.json.collector_runtime` binds the acceptance Job that wrote the
terminal. Because a process cannot prove its own future Kubernetes completion,
the bulk submit gate must additionally observe that exact acceptance Job and Pod
as exclusively Complete/Succeeded, with the UIDs from `collector_runtime`, zero
restarts, and zero container exit codes.

The Fleet duplicate check queries only session-list metadata for each of the 100
frozen task keys. Bulk executors must persist `run_id`, `execution_id`, and
`cell_id` in session metadata; those are the identities the observer checks for
active or accepted planned cells. The two generation-5 sessions are additionally
bound to their exact model, task version (when returned), run ID, completed state,
and verifier execution ID. No transcript or score endpoint is called.

Release order is deliberately mechanical:

1. Commit this package while it is held and keep the release receipt absent.
2. After the inventory and both generation-5 chains are terminal, render the
   observer-only release from an immutable snapshot of that exact package commit.
   Set an absent absolute `PREBULK_RELEASE_OUTPUT` outside the repository and run
   the wrapper's `prepare-release` mode; it renders twice and creates that output
   once with mode `0400`.
3. Commit the append-only release receipt and require a clean descendant checkout.
4. Run the wrapper in `preview`, then `submit-source` exactly once.
5. After the exact source Job is exclusively Complete, run `submit-accept` exactly
   once.
6. After the exact acceptance Job is exclusively Complete, validate the four
   bulk consumers against the digest-valid terminal and canary-gate receipts.

Desktop rendering never assumes a local SFS mount. The wrapper binds the exact
running, zero-restart `allie-dev` Pod UID, mirrors only the fixed inventory
terminal plus the two fixed G5 terminal/claim pairs, and verifies each remote
digest before and after copying. The renderer receives that private mirror via
`--evidence-root`; absolute logical `/mnt/sfs/...` paths remain unchanged in
receipts. Source and acceptance Jobs use `--evidence-root /mnt/sfs`, so the same
mapping code validates the live PVC. No prompt, transcript, flag, or score path
is in the mirror allowlist.

Before either create, the wrapper reads the committed release once into an
exclusive mode-`0400` temporary snapshot. Validation, hashing, ConfigMap
rendering, and live equality checks all use those same bytes. Immediately before
each create it rechecks the clean Git head, rendered-object digests, release
snapshot digest, observer UID, and every remote evidence digest. A changing
source, mirror, or render fails closed and leaves any already-created immutable
intent available for manual reconciliation.

The observer also cross-checks each executable template's model repository and
revision, hosted endpoint origin, served/session identities, serving block,
OpenCode version and provider adapter, context/compaction treatment, tool set,
and settings digests. The live hosted roster does not currently publish context
metadata, so each bulk cell revalidates the immutable local 262,144-token
OpenCode limit immediately before claiming it and rejects any contradictory
context value if the roster begins publishing one.

Both submission modes create an immutable intent ConfigMap before their Job.
Every create is fail-closed; partial objects are preserved for reconciliation,
never deleted or overwritten. The Jobs use the highest compatible non-preempting
CPU observer priority (`fleet-serve-low`), request no GPU, set
`preemptionPolicy: Never`, and have no automatic retry.

The resulting terminal and two sanitized canary-gate receipts are inputs to the
existing four-controller bulk release validator. They do not themselves launch
or authorize any scored rollout.
