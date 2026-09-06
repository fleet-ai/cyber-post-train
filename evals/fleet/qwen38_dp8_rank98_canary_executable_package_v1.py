"""Render the executable rank-98/a1 controller only after explicit root release."""

# ruff: noqa: E501 -- immutable bootstrap destinations are intentionally explicit.

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from evals.fleet import qwen38_dp8_rank98_canary_controller_v1 as controller
from evals.fleet import qwen38_dp8_rank98_canary_package_v1 as held_package
from evals.fleet import self_hosted

SCHEMA = "fleet-qwen38-dp8-r98-a1-executable-controller-package-v1"

BOOTSTRAP = """root=/workspace/cyber-post-train
mkdir -p "$root/evals/fleet/configs" "$root/evals/fleet/scripts"
touch "$root/evals/__init__.py" "$root/evals/fleet/__init__.py"
install -m 0644 /bootstrap/self_hosted.py "$root/evals/fleet/self_hosted.py"
install -m 0644 /bootstrap/exact_pass4_crypto.py "$root/evals/fleet/exact_pass4_crypto.py"
install -m 0644 /bootstrap/exact_pass4_universe.py "$root/evals/fleet/exact_pass4_universe.py"
install -m 0644 /bootstrap/legacy.py "$root/evals/fleet/qwen38_dedicated_scored_canary_v1.py"
install -m 0644 /bootstrap/controller.py "$root/evals/fleet/qwen38_dp8_rank98_canary_controller_v1.py"
install -m 0644 /bootstrap/server-binding.json "$root/server-binding.json"
install -m 0644 /bootstrap/qualification-result.json "$root/qualification-result.json"
install -m 0644 /bootstrap/release.json "$root/release.json"
install -m 0644 /bootstrap/root-release.json "$root/root-release.json"
install -m 0644 /bootstrap/parity.json "$root/parity.json"
install -m 0644 /bootstrap/campaign.json "$root/evals/fleet/configs/q38-glm53-exact-easiest100-pass4-campaign-v1.json"
install -m 0644 /bootstrap/selection.json "$root/evals/fleet/configs/opencode-easiest-train100-selection-v2.json"
install -m 0644 /bootstrap/source.json "$root/evals/fleet/configs/qwen-hosted-generation19-qwen-b-v4.json"
install -m 0644 /bootstrap/Dockerfile.opencode "$root/evals/fleet/Dockerfile.opencode"
install -m 0644 /bootstrap/fixed_proxy.py "$root/evals/fleet/fixed_proxy.py"
install -m 0755 /bootstrap/run.sh "$root/evals/fleet/scripts/run-qwen-dp8-r98-a1.sh"
exec "$root/evals/fleet/scripts/run-qwen-dp8-r98-a1.sh"
"""

RUN_SCRIPT = """#!/usr/bin/env bash
set -euo pipefail
root=${CYBER_ROOT:-/workspace/cyber-post-train}
: "${JOB_UID:?JOB_UID is required}" "${POD_UID:?POD_UID is required}"
: "${FLEET_API_KEY:?FLEET_API_KEY is required}"
: "${QWEN_DP8_HELD_PACKAGE_SHA256:?held package digest is required}"
test "${DOCKER_HOST:-}" = unix:///var/run/docker.sock
for _ in $(seq 1 120); do docker info >/dev/null 2>&1 && break; sleep 1; done
docker info >/dev/null
docker build --pull --platform linux/amd64 --tag chris/opencode:1.18.27-cyber-v1 \
  --file "$root/evals/fleet/Dockerfile.opencode" "$root/evals/fleet"
test "$(docker run --rm chris/opencode:1.18.27-cyber-v1 opencode --version)" = 1.18.27
export AGENT_HARNESS_IMAGE=chris/opencode:1.18.27-cyber-v1
export FIXED_PROXY_IMAGE=ghcr.io/astral-sh/uv:python3.12-bookworm@sha256:9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7
cd "$root"
exec uv run --no-project --with httpx==0.28.1 python -m evals.fleet.qwen38_dp8_rank98_canary_controller_v1
"""


class ExecutablePackageError(RuntimeError):
    """The final executable package is not root-released."""


def render_bundle(
    root: Path,
    binding: Mapping[str, Any],
    qualification_result: Mapping[str, Any],
    held_release: Mapping[str, Any],
    root_release: Mapping[str, Any],
) -> dict[str, Any]:
    """Render one create-once bundle; this function never submits it."""

    held = held_package.render_bundle(root, binding, qualification_result, held_release)
    held_sha = held["package_sha256"]
    partition = held_release["held_partition"]
    controller.validate_root_release(
        root_release, binding, qualification_result, partition, held_sha
    )
    held_configmap, held_job = held["objects"]
    data = copy.deepcopy(held_configmap["data"])
    paths = {
        "exact_pass4_crypto.py": root / "evals/fleet/exact_pass4_crypto.py",
        "exact_pass4_universe.py": root / "evals/fleet/exact_pass4_universe.py",
        "controller.py": root / "evals/fleet/qwen38_dp8_rank98_canary_controller_v1.py",
    }
    if any(path.is_symlink() or not path.is_file() for path in paths.values()):
        raise ExecutablePackageError("executable controller dependency is absent")
    data.update({name: path.read_text() for name, path in paths.items()})
    data["root-release.json"] = self_hosted.canonical_json(dict(root_release)).decode()
    data["run.sh"] = RUN_SCRIPT
    package_sha = held_package.package_sha256(data)
    annotations = {
        "cyber-post-train.fleet.ai/launch-authorized": "true",
        "cyber-post-train.fleet.ai/scoring-authorized": "true",
        "cyber-post-train.fleet.ai/package-sha256": package_sha,
        "cyber-post-train.fleet.ai/held-package-sha256": held_sha,
        "cyber-post-train.fleet.ai/server-binding-sha256": binding["receipt_sha256"],
        "cyber-post-train.fleet.ai/qualification-result-sha256": qualification_result[
            "receipt_sha256"
        ],
        "cyber-post-train.fleet.ai/held-partition-sha256": partition["receipt_sha256"],
        "cyber-post-train.fleet.ai/root-release-sha256": root_release["receipt_sha256"],
    }
    configmap = copy.deepcopy(held_configmap)
    configmap["data"] = data
    configmap["metadata"]["annotations"] = copy.deepcopy(annotations)
    job = copy.deepcopy(held_job)
    job["spec"]["suspend"] = False
    job["metadata"]["annotations"] = copy.deepcopy(annotations)
    job["spec"]["template"]["metadata"]["annotations"] = copy.deepcopy(annotations)
    container = job["spec"]["template"]["spec"]["containers"][0]
    container["args"] = [BOOTSTRAP]
    env = {row["name"]: row for row in container.get("env") or []}
    env["QWEN_DP8_CANARY_LAUNCH_AUTHORIZED"] = {
        "name": "QWEN_DP8_CANARY_LAUNCH_AUTHORIZED",
        "value": "true",
    }
    env["QWEN_DP8_CANARY_SCORING_AUTHORIZED"] = {
        "name": "QWEN_DP8_CANARY_SCORING_AUTHORIZED",
        "value": "true",
    }
    env["QWEN_DP8_HELD_PACKAGE_SHA256"] = {
        "name": "QWEN_DP8_HELD_PACKAGE_SHA256",
        "value": held_sha,
    }
    container["env"] = list(env.values())
    result = {
        "schema_version": SCHEMA,
        "status": "ROOT_RELEASED_CREATE_ONCE_RENDER_ONLY",
        "package_sha256": package_sha,
        "held_package_sha256": held_sha,
        "root_release_receipt_sha256": root_release["receipt_sha256"],
        "objects": [configmap, job],
        "submit_calls": 0,
        "kubernetes_mutation_calls": 0,
        "prompts_traces_flags_scores_or_model_outputs_included": False,
    }
    result["receipt_sha256"] = self_hosted.digest_without(result, "receipt_sha256")
    return result
