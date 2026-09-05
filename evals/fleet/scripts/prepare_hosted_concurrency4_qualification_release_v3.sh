#!/usr/bin/env bash
set -euo pipefail
umask 077

mode=${1:-preview}
[[ "$mode" == preview || "$mode" == render-release ]] || {
  printf '%s\n' 'usage: prepare_hosted_concurrency4_qualification_release_v3.sh preview|render-release' >&2
  exit 2
}

root=$(git rev-parse --show-toplevel)
release_rel=docs/evidence/qwen38-study/2026-09-05-hosted-concurrency4-qualification-launch-release-v3.json
release="$root/$release_rel"
work=$(mktemp -d /private/tmp/hosted-c4-v3-gather.XXXXXX)
trap 'rm -rf "$work"' EXIT

test -z "$(git -C "$root" status --porcelain=v1 --untracked-files=all)"
implementation_commit=$(git -C "$root" rev-parse HEAD)
git -C "$root" cat-file -e "$implementation_commit^{commit}"
if git -C "$root" cat-file -e "$implementation_commit:$release_rel" 2>/dev/null; then
  printf '%s\n' 'release must be append-only after the implementation commit' >&2
  exit 1
fi
test ! -e "$release" && test ! -L "$release"

uv run --project "$root" python \
  -m evals.fleet.hosted_concurrency4_qualification_release_v3_gather gather \
  --repo "$root" --output-dir "$work/evidence" >"$work/gather-summary.json"

if [[ "$mode" == preview ]]; then
  uv run --project "$root" python \
    -m evals.fleet.hosted_concurrency4_qualification_release_v3 render-acceptance \
    --repo "$root" --evidence-dir "$work/evidence" --output "$work/acceptance.json"
  jq -n \
    --arg implementation_commit "$implementation_commit" \
    --arg allie_dev_uid "$(jq -er .allie_dev_uid "$work/gather-summary.json")" \
    --arg binding_sha256 "$(jq -er .binding_sha256 "$work/acceptance.json")" \
    '{status:"VALIDATED_PREVIEW",implementation_commit:$implementation_commit,allie_dev_uid:$allie_dev_uid,generation7_acceptance_binding_sha256:$binding_sha256,release_created:false,prompts_traces_flags_or_scores_included:false}'
  exit 0
fi

uv run --project "$root" python \
  -m evals.fleet.hosted_concurrency4_qualification_release_v3 render-release \
  --repo "$root" --implementation-commit "$implementation_commit" \
  --evidence-dir "$work/evidence" --output "$release"
chmod 0400 "$release"
uv run --project "$root" python \
  -m evals.fleet.hosted_concurrency4_qualification_release_v3 validate-release \
  --repo "$root" --release "$release"
uv run --project "$root" python \
  -m evals.fleet.hosted_concurrency4_qualification_release_v3 validate-live-binding \
  --repo "$root" --release "$release" --evidence-dir "$work/evidence"

jq -n \
  --arg implementation_commit "$implementation_commit" \
  --arg path "$release_rel" \
  --arg receipt_sha256 "$(jq -er .receipt_sha256 "$release")" \
  '{status:"RELEASE_RENDERED_APPEND_ONLY",implementation_commit:$implementation_commit,path:$path,receipt_sha256:$receipt_sha256,objects_created:false,prompts_traces_flags_or_scores_included:false}'
