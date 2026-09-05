# Jobs API request: per-pod-set topology scope

The general `POST /v1/runs` contract currently exposes only job-wide
`topology_mode` and `topology_level` fields. The authenticated preview applies
that annotation to every rendered Ray pod set. It exposes no queue,
ResourceFlavor, head-topology, or worker-topology selector.

For a one-pod, eight-GPU serving run, the Ray head is itself the sole GPU
worker. A selective worker-only field therefore cannot repair admission for
that shape. For a multi-pod run, the smallest safe platform extension is to
support distinct topology settings for the head and worker templates, with
preview returning the exact annotations for each pod set. The API must reject a
request unless every rendered pod set has at least one compatible live flavor.

Until that exists, operators must treat a post-submit Kueue condition that can
only name non-topology fallback flavors as an inadmissible generation, release
it through `DELETE /v1/runs/{name}`, and use a fresh identity after a reviewed
request-shape change. A static pre-submit quota snapshot is not a reservation
and cannot substitute for the authoritative Workload admission result.

The GLM TP8 scale gate remains closed on the current head-wide required-TAS
shape. Do not submit another identical successor.

A two-node/16-GPU request does not avoid this boundary. It renders one
eight-GPU Ray head and one eight-GPU worker, applies the same required-TAS
annotation to both, and therefore needs topology-compatible admission for both
pod sets. It would also consume the entire approved 16-GPU ceiling and is a
different TP/runtime contract from the validated TP8 server command. It must
not be used as an admission workaround.
