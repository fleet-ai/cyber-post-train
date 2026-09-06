# Qwen3.8 hosted whole-task release gate

The generation-19 rank-15/rank-16 Jobs failed before publishing claims or
starting model work. Their Kubernetes object identities are nevertheless
consumed and must not be retried. The successor keeps the same eight
statistical cells, model revision, OpenCode 1.18.27 harness, `bash` plus
`submit_report` tools, 262,144-token context, and native compaction with
autocontinue. It changes only execution generation and operational identities.

The two generation-20 scored Jobs are held. Rendering them without a release
receipt produces `launch-authorized=false`; that output is for review only.

Before release, render and review the score-free observer:

```sh
python -m evals.fleet.qwen_hosted_whole_task_release_gate_package_v1 > /tmp/q38-release-gate.yaml
```

The observer is a create-once, non-preempting CPU Job. It performs GET-only
checks for the exact failed predecessor Job/Pod/ConfigMap identities, absence
of all fresh Job/Pod/ConfigMap and SFS roots, global claim and accepted-receipt
collisions, authoritative session collisions, and simultaneous availability of
both hosted-Qwen endpoint lease slots. It emits only aggregate counts, binding
hashes, its Job/Pod UIDs, and a self-digest. It does not request task bodies,
transcripts, verifier output, or scores, and it performs no model, task,
session, verifier, or scoring mutation.

Submission is intentionally not automated by this change. After a reviewer
creates the manifest once, the exact observer Job must become exclusively
`Complete` with zero restarts. An independent operator must validate the
digest-valid `OBSERVATION.json` against the bound Job and Pod UIDs before
authoring a scored release receipt. A failed, active, stale, or missing
observer never authorizes launch.

The held scored manifest can be inspected with:

```sh
python - <<'PY'
import yaml
from pathlib import Path
from evals.fleet import qwen_hosted_whole_task_successor_v2_package as package

print(yaml.safe_dump(package.render(Path.cwd()), sort_keys=False))
PY
```

Do not submit that held output. A later reviewed release must bind the accepted
observer receipt and terminal UID evidence before the renderer can mark either
scored Job launch-authorized.
