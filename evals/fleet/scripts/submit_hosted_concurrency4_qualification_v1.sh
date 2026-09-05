#!/usr/bin/env bash
set -euo pipefail

mode=${1:-preview}
if [[ "$mode" != preview ]]; then
  printf '%s\n' 'HELD: hosted concurrency-4 qualification has no launch authorization' >&2
  exit 2
fi
root=$(git rev-parse --show-toplevel)
commit=$(git -C "$root" rev-parse HEAD)
builder=$root/evals/fleet/hosted_concurrency4_qualification_package_v1.py
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
git -C "$root" show "$commit:evals/fleet/hosted_concurrency4_qualification_package_v1.py" >"$work/package.py"
uv run --project "$root" python "$work/package.py" render-configmap --repo "$root" --package-commit "$commit" >"$work/configmap.json"
uv run --project "$root" python "$work/package.py" render-intent --configmap "$work/configmap.json" >"$work/intent.json"
uv run --project "$root" python "$work/package.py" render-job --repo "$root" --package-commit "$commit" >"$work/job.yaml"
test "$(jq -r '.data["package.json"]|fromjson|.job_manifest.sha256' "$work/configmap.json")" = \
  "sha256:$(sha256sum "$work/job.yaml" | awk '{print $1}')"
jq -e '.metadata.annotations["cyber-post-train.fleet.ai/launch-authorized"]=="false" and .immutable==true' \
  "$work/configmap.json" "$work/intent.json" >/dev/null
uv run --project "$root" python - "$work/job.yaml" <<'PY'
import sys, yaml
manifest = yaml.safe_load(open(sys.argv[1]))
pod = manifest["spec"]["template"]["spec"]
assert pod["priorityClassName"] == "fleet-serve-low"
assert pod["preemptionPolicy"] == "Never"
assert not any("gpu" in key.lower() for c in pod["containers"] for key in c.get("resources", {}).get("requests", {}))
PY
if command -v kubectl >/dev/null && kubectl config current-context >/dev/null 2>&1; then
  kubectl -n fleet-train-jobs create --dry-run=server -f "$work/configmap.json" -o name >/dev/null
  kubectl -n fleet-train-jobs create --dry-run=server -f "$work/intent.json" -o name >/dev/null
  kubectl -n fleet-train-jobs create --dry-run=server -f "$work/job.yaml" -o name >/dev/null
fi
jq -n --arg commit "$commit" --arg package "$(jq -r '.data.package_sha256' "$work/configmap.json")" \
  '{ok:true,status:"HELD",package_commit:$commit,package_sha256:$package,objects_created:false,chat_completion_requests_if_released:24,task_instance_session_scoring_or_verifier_calls:0,launch_authorized:false,scored_bulk_launch_authorized:false}'
