"""Render the held rank-98 DP8 scored-canary ConfigMap and Job.

The rendered Job is suspended and carries explicit false launch/scoring
annotations.  Its package can validate exact OpenCode/DinD materialization but
contains no claim, model, task, session, verifier, or scoring execution rail.
"""

# ruff: noqa: E501 -- immutable bootstrap destinations are intentionally explicit.

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from evals.fleet import qwen38_dedicated_rank2_v3_job as source_job
from evals.fleet import qwen38_dp8_rank98_canary_partition_v1 as partition
from evals.fleet import self_hosted

SCHEMA = "fleet-qwen38-dp8-rank98-canary-controller-package-held-v1"
RUNNER_IMAGE = (
    "ghcr.io/astral-sh/uv:python3.12-bookworm@"
    "sha256:9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7"
)
OPEN_CODE_IMAGE = "chris/opencode:1.18.27-cyber-v1"

BOOTSTRAP = """root=/workspace/cyber-post-train
mkdir -p "$root/evals/fleet/configs" "$root/evals/fleet/scripts" "$root/docs/evidence/qwen38-study"
touch "$root/evals/__init__.py" "$root/evals/fleet/__init__.py"
install -m 0644 /bootstrap/self_hosted.py "$root/evals/fleet/self_hosted.py"
install -m 0644 /bootstrap/legacy.py "$root/evals/fleet/qwen38_dedicated_scored_canary_v1.py"
install -m 0644 /bootstrap/server-binding.json "$root/server-binding.json"
install -m 0644 /bootstrap/qualification-result.json "$root/qualification-result.json"
install -m 0644 /bootstrap/release.json "$root/release.json"
install -m 0644 /bootstrap/parity.json "$root/parity.json"
install -m 0644 /bootstrap/controller-contract.json "$root/controller-contract.json"
install -m 0644 /bootstrap/campaign.json "$root/evals/fleet/configs/q38-glm53-exact-easiest100-pass4-campaign-v1.json"
install -m 0644 /bootstrap/selection.json "$root/evals/fleet/configs/opencode-easiest-train100-selection-v2.json"
install -m 0644 /bootstrap/source.json "$root/evals/fleet/configs/qwen-hosted-generation19-qwen-b-v4.json"
install -m 0644 /bootstrap/Dockerfile.opencode "$root/evals/fleet/Dockerfile.opencode"
install -m 0644 /bootstrap/fixed_proxy.py "$root/evals/fleet/fixed_proxy.py"
install -m 0755 /bootstrap/run.sh "$root/evals/fleet/scripts/run-qwen-dp8-r98-held.sh"
exec "$root/evals/fleet/scripts/run-qwen-dp8-r98-held.sh"
"""

RUN_SCRIPT = """#!/usr/bin/env bash
set -euo pipefail
root=${CYBER_ROOT:-/workspace/cyber-post-train}
test "${QWEN_DP8_CANARY_LAUNCH_AUTHORIZED:-}" = false
test "${QWEN_DP8_CANARY_SCORING_AUTHORIZED:-}" = false
test "${QWEN_DP8_CANARY_ACTIVE_ATTEMPT:-}" = 1
test "${QWEN_DP8_CANARY_HELD_ATTEMPTS:-}" = 2,3,4
test "${DOCKER_HOST:-}" = unix:///var/run/docker.sock
for _ in $(seq 1 120); do docker info >/dev/null 2>&1 && break; sleep 1; done
docker info >/dev/null
docker build --pull --platform linux/amd64 --tag chris/opencode:1.18.27-cyber-v1 \
  --file "$root/evals/fleet/Dockerfile.opencode" "$root/evals/fleet"
test "$(docker run --rm chris/opencode:1.18.27-cyber-v1 opencode --version)" = 1.18.27
python3 - "$root/release.json" "$root/controller-contract.json" <<'PY'
import json, sys
release=json.load(open(sys.argv[1], encoding="utf-8"))
contract=json.load(open(sys.argv[2], encoding="utf-8"))
assert release["launch_authorized"] is False
assert release["scoring_authorized"] is False
assert release["controller_create_permitted"] is False
assert contract["active_attempt"] == 1
assert contract["held_attempts"] == [2, 3, 4]
assert contract["claim_or_model_calls"] == 0
PY
"""


class PackageError(RuntimeError):
    """The held controller package is incomplete or unsafe."""


def _read_text(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise PackageError(f"package input is absent or unsafe: {path}")
    return path.read_text()


def _validate_parity(root: Path, binding: Mapping[str, Any]) -> str:
    path = root / str(binding["parity_path"])
    raw = path.read_bytes()
    value = json.loads(raw)
    if (
        self_hosted.sha256(raw) != binding.get("parity_file_sha256")
        or value.get("receipt_sha256") != binding.get("parity_receipt_sha256")
        or value.get("receipt_sha256")
        != self_hosted.digest_without(value, "receipt_sha256")
    ):
        raise PackageError("future server parity evidence drifted")
    return raw.decode()


def configmap_data(
    root: Path,
    binding: Mapping[str, Any],
    qualification_result: Mapping[str, Any],
    release: Mapping[str, Any],
) -> dict[str, str]:
    expected = partition.build_held(qualification_result, binding, release["live_scan"])
    if release.get("held_partition") != expected:
        raise PackageError("fresh held partition release drifted")
    contract = {
        "schema_version": "fleet-qwen38-dp8-r98-held-controller-contract-v1",
        "active_attempt": 1,
        "held_attempts": [2, 3, 4],
        "claim_or_model_calls": 0,
        "same_task_max_inflight": 1,
        "future_append_only_root_release_required": True,
        "same_server_and_controller_uid_continuation_required": True,
    }
    return {
        "self_hosted.py": _read_text(root / "evals/fleet/self_hosted.py"),
        "legacy.py": _read_text(root / "evals/fleet/qwen38_dedicated_scored_canary_v1.py"),
        "server-binding.json": self_hosted.canonical_json(dict(binding)).decode(),
        "qualification-result.json": self_hosted.canonical_json(
            dict(qualification_result)
        ).decode(),
        "release.json": self_hosted.canonical_json(dict(release["held_partition"])).decode(),
        "parity.json": _validate_parity(root, binding),
        "controller-contract.json": self_hosted.canonical_json(contract).decode(),
        "campaign.json": _read_text(
            root / "evals/fleet/configs/q38-glm53-exact-easiest100-pass4-campaign-v1.json"
        ),
        "selection.json": _read_text(
            root / "evals/fleet/configs/opencode-easiest-train100-selection-v2.json"
        ),
        "source.json": _read_text(
            root / "evals/fleet/configs/qwen-hosted-generation19-qwen-b-v4.json"
        ),
        "Dockerfile.opencode": _read_text(root / "evals/fleet/Dockerfile.opencode"),
        "fixed_proxy.py": _read_text(root / "evals/fleet/fixed_proxy.py"),
        "run.sh": RUN_SCRIPT,
    }


def package_sha256(data: Mapping[str, str]) -> str:
    return self_hosted.sha256(self_hosted.canonical_json(dict(data)))


def render_bundle(
    root: Path,
    binding: Mapping[str, Any],
    qualification_result: Mapping[str, Any],
    release: Mapping[str, Any],
) -> dict[str, Any]:
    """Render an immutable suspended bundle; never create it."""

    data = configmap_data(root, binding, qualification_result, release)
    package_sha = package_sha256(data)
    identity = release["held_partition"]["held_controller_identity"]
    name = identity["job_name"]
    source = root / source_job.SOURCE
    if self_hosted.sha256(source.read_bytes()) != source_job.SOURCE_SHA256:
        raise PackageError("proven CPU/DinD source Job drifted")
    job = copy.deepcopy(yaml.safe_load(source.read_text()))
    job["metadata"]["name"] = name
    job["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = name
    job["metadata"]["annotations"] = {
        "cyber-post-train.fleet.ai/launch-authorized": "false",
        "cyber-post-train.fleet.ai/scoring-authorized": "false",
        "cyber-post-train.fleet.ai/serving-block": binding["serving_block"],
        "cyber-post-train.fleet.ai/package-sha256": package_sha,
        "cyber-post-train.fleet.ai/server-binding-sha256": binding["receipt_sha256"],
        "cyber-post-train.fleet.ai/qualification-result-sha256": qualification_result[
            "receipt_sha256"
        ],
        "cyber-post-train.fleet.ai/release-sha256": release["held_partition"][
            "receipt_sha256"
        ],
    }
    job["spec"]["suspend"] = True
    template = job["spec"]["template"]
    template["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = name
    template["metadata"]["annotations"] = copy.deepcopy(job["metadata"]["annotations"])
    template["spec"]["priorityClassName"] = "fleet-infra-quiet"
    template["spec"]["preemptionPolicy"] = "Never"
    template["spec"]["volumes"][0]["configMap"]["name"] = identity["configmap_name"]
    container = template["spec"]["containers"][0]
    if container.get("image") != RUNNER_IMAGE:
        raise PackageError("proven evaluator image drifted")
    container["args"] = [BOOTSTRAP]
    container.setdefault("env", []).extend(
        [
            {"name": "QWEN_DP8_CANARY_LAUNCH_AUTHORIZED", "value": "false"},
            {"name": "QWEN_DP8_CANARY_SCORING_AUTHORIZED", "value": "false"},
            {"name": "QWEN_DP8_CANARY_ACTIVE_ATTEMPT", "value": "1"},
            {"name": "QWEN_DP8_CANARY_HELD_ATTEMPTS", "value": "2,3,4"},
        ]
    )
    configmap = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": identity["configmap_name"],
            "namespace": "fleet-train-jobs",
            "annotations": copy.deepcopy(job["metadata"]["annotations"]),
        },
        "immutable": True,
        "data": data,
    }
    result = {
        "schema_version": SCHEMA,
        "status": "RENDERED_HELD_NO_CREATE",
        "package_sha256": package_sha,
        "server_binding_receipt_sha256": binding["receipt_sha256"],
        "qualification_result_sha256": qualification_result["receipt_sha256"],
        "held_partition_receipt_sha256": release["held_partition"]["receipt_sha256"],
        "objects": [configmap, job],
        "launch_authorized": False,
        "scoring_authorized": False,
        "controller_create_permitted": False,
        "api_mutation_calls": 0,
        "prompts_traces_flags_scores_or_model_outputs_included": False,
    }
    result["receipt_sha256"] = self_hosted.digest_without(result, "receipt_sha256")
    return result
