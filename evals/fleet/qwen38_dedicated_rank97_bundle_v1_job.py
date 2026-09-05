"""Render the one-Job four-claim rank97 dedicated Qwen controller."""

# ruff: noqa: E501 -- immutable bootstrap paths are intentionally explicit.

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from evals.fleet import qwen38_dedicated_rank2_v3_job as base
from evals.fleet import self_hosted

NAME = "chris-cyber-q38-opencode11827-ded-tp1-r097-g19-d-bundle-v2"
EXPERIMENT = "q38-ded-tp1-r097-g19-d-bundle-v2"
SERVING_BLOCK = "dedicated-qwen-tp1-d-v1"
SOURCE = base.SOURCE
SOURCE_SHA256 = base.SOURCE_SHA256
RUN_SCRIPT = Path("evals/fleet/scripts/run_qwen38_dedicated_rank97_bundle_v1.sh")

BOOTSTRAP = """root=/workspace/cyber-post-train
mkdir -p "$root/evals/fleet/configs" "$root/evals/fleet/scripts" "$root/docs/evidence/qwen38-study"
touch "$root/evals/__init__.py" "$root/evals/fleet/__init__.py"
install -m 0644 /bootstrap/self_hosted.py "$root/evals/fleet/self_hosted.py"
install -m 0644 /bootstrap/exact_pass4_crypto.py "$root/evals/fleet/exact_pass4_crypto.py"
install -m 0644 /bootstrap/exact_pass4_universe.py "$root/evals/fleet/exact_pass4_universe.py"
install -m 0644 /bootstrap/legacy.py "$root/evals/fleet/qwen38_dedicated_scored_canary_v1.py"
install -m 0644 /bootstrap/lane.py "$root/evals/fleet/qwen38_dedicated_rank97_bundle_v1.py"
install -m 0644 /bootstrap/parity.json "$root/docs/evidence/qwen38-study/2026-09-05-qwen38-dedicated-tp1-d-v1-actual-opencode-parity.json"
install -m 0644 /bootstrap/campaign.json "$root/evals/fleet/configs/q38-glm53-exact-easiest100-pass4-campaign-v1.json"
install -m 0644 /bootstrap/selection.json "$root/evals/fleet/configs/opencode-easiest-train100-selection-v2.json"
install -m 0644 /bootstrap/source.json "$root/evals/fleet/configs/qwen-hosted-generation19-qwen-a-v4.json"
install -m 0644 /bootstrap/Dockerfile.opencode "$root/evals/fleet/Dockerfile.opencode"
install -m 0644 /bootstrap/fixed_proxy.py "$root/evals/fleet/fixed_proxy.py"
install -m 0755 /bootstrap/run.sh "$root/evals/fleet/scripts/run_qwen38_dedicated_rank97_bundle_v1.sh"
exec "$root/evals/fleet/scripts/run_qwen38_dedicated_rank97_bundle_v1.sh"
"""


def render(root: Path) -> dict[str, Any]:
    source = root / SOURCE
    if self_hosted.sha256(source.read_bytes()) != SOURCE_SHA256:
        raise ValueError("source evaluator Job template drifted")
    value = copy.deepcopy(yaml.safe_load(source.read_text()))
    value["metadata"]["name"] = NAME
    value["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = EXPERIMENT
    value["metadata"]["annotations"]["cyber-post-train.fleet.ai/serving-block"] = SERVING_BLOCK
    template = value["spec"]["template"]
    template["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = EXPERIMENT
    pod = template["spec"]
    pod["volumes"][0]["configMap"]["name"] = NAME
    container = pod["containers"][0]
    container["args"] = [BOOTSTRAP]
    container.setdefault("env", []).append(
        {"name": "QWEN_RANK97_RELEASE_PATH", "value": "/bootstrap/release.json"}
    )
    return value


def configmap_data(root: Path, release_path: Path) -> dict[str, str]:
    paths = {
        "self_hosted.py": root / "evals/fleet/self_hosted.py",
        "exact_pass4_crypto.py": root / "evals/fleet/exact_pass4_crypto.py",
        "exact_pass4_universe.py": root / "evals/fleet/exact_pass4_universe.py",
        "legacy.py": root / "evals/fleet/qwen38_dedicated_scored_canary_v1.py",
        "lane.py": root / "evals/fleet/qwen38_dedicated_rank97_bundle_v1.py",
        "parity.json": root
        / (
            "docs/evidence/qwen38-study/"
            "2026-09-05-qwen38-dedicated-tp1-d-v1-actual-opencode-parity.json"
        ),
        "campaign.json": root
        / ("evals/fleet/configs/q38-glm53-exact-easiest100-pass4-campaign-v1.json"),
        "selection.json": root
        / "evals/fleet/configs/opencode-easiest-train100-selection-v2.json",
        "source.json": root / "evals/fleet/configs/qwen-hosted-generation19-qwen-a-v4.json",
        "Dockerfile.opencode": root / "evals/fleet/Dockerfile.opencode",
        "fixed_proxy.py": root / "evals/fleet/fixed_proxy.py",
        "run.sh": root / RUN_SCRIPT,
        "release.json": release_path,
    }
    if any(path.is_symlink() or not path.is_file() for path in paths.values()):
        raise ValueError("rank97 package input is absent or unsafe")
    return {name: path.read_text() for name, path in paths.items()}
