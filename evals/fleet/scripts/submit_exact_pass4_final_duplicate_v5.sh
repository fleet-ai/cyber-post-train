#!/usr/bin/env bash
set -euo pipefail
umask 077

mode=${1:-preview}
group=${2:-}
case "${mode}" in
  preview|prepare-release|submit-source|submit-accept) ;;
  *) printf '%s\n' 'usage: submit_exact_pass4_final_duplicate_v5.sh preview|prepare-release|submit-source|submit-accept [group]' >&2; exit 2 ;;
esac

root=$(git rev-parse --show-toplevel)
release_rel=docs/evidence/qwen38-study/2026-09-05-opencode-exact-pass4-final-duplicate-observer-release-v5.json
release="${root}/${release_rel}"
package_commit=$(git -C "${root}" rev-parse HEAD)
work=$(mktemp -d /private/tmp/final-v5-duplicate.XXXXXX)
trap 'rm -rf "${work}"' EXIT

test -z "$(git -C "${root}" status --porcelain=v1 --untracked-files=all)"
git -C "${root}" cat-file -e "${package_commit}^{commit}"

render_release() {
  local target=$1
  set -o noclobber
  uv run --project "${root}" python \
    -m evals.fleet.exact_pass4_final_duplicate_observer_v5 render-release \
    --root "${root}" --package-commit "${package_commit}" >"${target}"
  set +o noclobber
}

if [[ "${mode}" == prepare-release ]]; then
  if git -C "${root}" cat-file -e "${package_commit}:${release_rel}" 2>/dev/null; then
    printf '%s\n' 'release must be append-only after the implementation commit' >&2
    exit 1
  fi
  test ! -e "${release}" && test ! -L "${release}"
  render_release "${release}"
  chmod 0400 "${release}"
  uv run --project "${root}" python \
    -m evals.fleet.exact_pass4_final_duplicate_observer_v5 validate-release \
    --root "${root}" --release "${release}"
  jq -n --arg path "${release_rel}" --arg commit "${package_commit}" \
    --arg receipt_sha256 "$(jq -er .receipt_sha256 "${release}")" \
    '{status:"RELEASE_RENDERED_APPEND_ONLY",path:$path,package_commit:$commit,receipt_sha256:$receipt_sha256,objects_created:false}'
  exit 0
fi

case "${group}" in
  hosted-qwen|hosted-glm|dedicated-a-canary|dedicated-a-bulk|dedicated-b-canary|dedicated-b-bulk) ;;
  *) printf '%s\n' 'an exact final-v5 group is required' >&2; exit 2 ;;
esac

release_input=${release}
if [[ ! -f "${release_input}" || -L "${release_input}" ]]; then
  release_input="${work}/release.json"
  render_release "${release_input}"
fi
uv run --project "${root}" python \
  -m evals.fleet.exact_pass4_final_duplicate_observer_v5 validate-release \
  --root "${root}" --release "${release_input}"

render_mode=source
[[ "${mode}" == submit-accept ]] && render_mode=accept
uv run --project "${root}" python \
  -m evals.fleet.exact_pass4_final_duplicate_renderer_v5 render \
  --root "${root}" --release "${release_input}" --group "${group}" \
  --mode "${render_mode}" --output-dir "${work}/rendered"
uv run --project "${root}" python \
  -m evals.fleet.kubernetes_create_relay validate \
  --manifest "${work}/rendered/manifest.json" \
  --allowlist "${work}/rendered/allowlist.json"

if [[ "${mode}" == preview ]]; then
  jq -n --arg group "${group}" --arg mode "${render_mode}" \
    '{status:"VALIDATED_PREVIEW",group:$group,mode:$mode,objects_created:false,launch_authorized:false}'
  exit 0
fi

: "${DUPLICATE_CREATE_RECEIPT:?DUPLICATE_CREATE_RECEIPT must be an unused absolute receipt path}"
[[ "${DUPLICATE_CREATE_RECEIPT}" == /* ]] || {
  printf '%s\n' 'DUPLICATE_CREATE_RECEIPT must be absolute' >&2
  exit 2
}
test ! -e "${DUPLICATE_CREATE_RECEIPT}" && test ! -L "${DUPLICATE_CREATE_RECEIPT}"
uv run --project "${root}" python \
  -m evals.fleet.kubernetes_create_relay create \
  --manifest "${work}/rendered/manifest.json" \
  --allowlist "${work}/rendered/allowlist.json" \
  --transport auto --receipt "${DUPLICATE_CREATE_RECEIPT}"
