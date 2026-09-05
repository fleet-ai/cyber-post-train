# Jobs API priority request for dedicated GLM serving

## Current constraint

The live Kubernetes cluster has two `PriorityClass` objects whose
`preemptionPolicy` is `Never`: `fleet-infra-quiet` at value `-1000` and
`fleet-serve-low` at value `100`. The general cluster Jobs API deployed at
`POST /v1/runs` accepts only `fleet-train-high` and `fleet-infra-quiet`, even
though its published `RLJobConfig.priority_class` OpenAPI field is an
unconstrained string. An authenticated preview using `fleet-serve-low` returns
HTTP 422 before any run is created.

This means the highest Jobs-API-supported class that cannot preempt peer work
is currently `fleet-infra-quiet`. It can sit behind a large queue even when the
cluster has a better-purpose `fleet-serve-low` class for best-effort GPU model
serving.

## Requested platform change

Allow `fleet-serve-low` in the `/v1/runs` priority validator and expose the
accepted priority names as an enum or a read-only capability field in OpenAPI.
Keep server-side validation that the referenced live PriorityClass exists.
After deployment, clients can intersect the advertised allowlist with the live
PriorityClass inventory and select the unique highest class whose
`preemptionPolicy` is `Never`.

No cluster deployment or shared service was changed as part of this study.
Until the API change is deployed and behaviorally verified, dedicated GLM jobs
must use `fleet-infra-quiet`; `fleet-train-high` remains prohibited because it
can preempt lower-priority peer workloads.
