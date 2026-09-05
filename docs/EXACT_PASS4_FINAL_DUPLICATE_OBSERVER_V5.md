# Exact pass@4 final-v5 duplicate observer

This held observer is the final, score-blind absence gate immediately before a
final-v5 controller group is released. It is deliberately split into two
create-once CPU Jobs. The source Job exhaustively checks Fleet session metadata,
the exact Kubernetes Job/Pod/ConfigMap names, every selected SFS output root,
and all generation claim JSON. The accept Job requires that exact source Job to
have succeeded, repeats the complete observation, and freezes `ACCEPTED.json`.

The six groups are `hosted-qwen`, `hosted-glm`, `dedicated-a-canary`,
`dedicated-a-bulk`, `dedicated-b-canary`, and `dedicated-b-bulk`. Each receipt
binds the exact final-v5 cells, controller Jobs, ConfigMaps, output roots,
package commit, Fleet team, credential Secret UID, observer Job/Pod UIDs, and
GET-only request counts. A final-v5 release must call
`validate_accepted_for_release`; structural validation alone is insufficient.
That live validation enforces the 15-minute freshness limit and exclusive
success of both observer Jobs.

The wrapper supports `preview`, `prepare-release`, `submit-source`, and
`submit-accept`. `prepare-release` creates the append-only release only after
the implementation commit. Each submit requires an unused absolute
`DUPLICATE_CREATE_RECEIPT` and uses the bounded Kubernetes create relay, which
retains local access when valid and the exact in-cluster fallback otherwise.
The accept Job should be submitted only after the source Job is terminally
successful; it independently fails closed if this ordering is violated.

All observer Jobs are CPU-only at `fleet-serve-low`, use
`preemptionPolicy: Never`, have no automatic retry, and create no scored
session. The observer reads no prompt, trace, transcript, flag, solution, score,
or Secret data. Its PartialObjectMetadata request observes only Secret identity.
Package ConfigMaps are immutable and group-specific so partial work for one
group cannot collide with another group. Nothing in the held manifest launches
or authorizes an object by itself.
