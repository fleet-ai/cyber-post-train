"""Render/package the fresh rank3/a2 generation-22 successor."""

# ruff: noqa: E501

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from evals.fleet import qwen38_dedicated_rank2_v3_job as base
from evals.fleet import qwen38_dedicated_rank3_v3 as prior
from evals.fleet import self_hosted

NAME = "chris-cyber-q38-opencode11827-ded-tp1-r003-a2-g22-v1"
EXPERIMENT = "q38-ded-tp1-r003-a2-g22-v1"
RUN_SCRIPT = Path("evals/fleet/scripts/run_qwen38_dedicated_rank3_a2_g22_v1.sh")
BOOTSTRAP = """root=/workspace/cyber-post-train
mkdir -p "$root/evals/fleet/configs" "$root/evals/fleet/scripts" "$root/docs/evidence/qwen38-study"
touch "$root/evals/__init__.py" "$root/evals/fleet/__init__.py"
install -m 0644 /bootstrap/self_hosted.py "$root/evals/fleet/self_hosted.py"
install -m 0644 /bootstrap/exact_pass4_crypto.py "$root/evals/fleet/exact_pass4_crypto.py"
install -m 0644 /bootstrap/exact_pass4_universe.py "$root/evals/fleet/exact_pass4_universe.py"
install -m 0644 /bootstrap/legacy.py "$root/evals/fleet/qwen38_dedicated_scored_canary_v1.py"
install -m 0644 /bootstrap/prior.py "$root/evals/fleet/qwen38_dedicated_rank3_v3.py"
install -m 0644 /bootstrap/lane.py "$root/evals/fleet/qwen38_dedicated_rank3_a2_g22_v1.py"
install -m 0644 /bootstrap/parity.json "$root/docs/evidence/qwen38-study/2026-09-05-qwen38-dedicated-tp1-b-v2-actual-opencode-parity.json"
install -m 0644 /bootstrap/rank3-release.json "$root/docs/evidence/qwen38-study/2026-09-05-qwen38-dedicated-rank3-g21-b-v2-release-v3.json"
install -m 0644 /bootstrap/campaign.json "$root/evals/fleet/configs/q38-glm53-exact-easiest100-pass4-campaign-v1.json"
install -m 0644 /bootstrap/selection.json "$root/evals/fleet/configs/opencode-easiest-train100-selection-v2.json"
install -m 0644 /bootstrap/source.json "$root/evals/fleet/configs/qwen38-opencode-train50-pass4-v3.json"
install -m 0644 /bootstrap/Dockerfile.opencode "$root/evals/fleet/Dockerfile.opencode"
install -m 0644 /bootstrap/fixed_proxy.py "$root/evals/fleet/fixed_proxy.py"
install -m 0755 /bootstrap/run.sh "$root/evals/fleet/scripts/run_qwen38_dedicated_rank3_a2_g22_v1.sh"
exec "$root/evals/fleet/scripts/run_qwen38_dedicated_rank3_a2_g22_v1.sh"
"""


def render(root: Path) -> dict[str, Any]:
    source = root / base.SOURCE
    if self_hosted.sha256(source.read_bytes()) != base.SOURCE_SHA256:
        raise ValueError("source evaluator Job template drifted")
    value = copy.deepcopy(yaml.safe_load(source.read_text()))
    value["metadata"]["name"] = NAME
    value["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = EXPERIMENT
    value["metadata"]["annotations"]["cyber-post-train.fleet.ai/serving-block"] = (
        "dedicated-qwen-tp1-b-v2"
    )
    template = value["spec"]["template"]
    template["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = EXPERIMENT
    template["spec"]["volumes"][0]["configMap"]["name"] = NAME
    container = template["spec"]["containers"][0]
    container["args"] = [BOOTSTRAP]
    container.setdefault("env", []).append(
        {"name": "QWEN_RANK3_G22_RELEASE_PATH", "value": "/bootstrap/release.json"}
    )
    return value


def configmap_data(root: Path, release_path: Path) -> dict[str, str]:
    paths = {
        "self_hosted.py": root / "evals/fleet/self_hosted.py",
        "exact_pass4_crypto.py": root / "evals/fleet/exact_pass4_crypto.py",
        "exact_pass4_universe.py": root / "evals/fleet/exact_pass4_universe.py",
        "legacy.py": root / "evals/fleet/qwen38_dedicated_scored_canary_v1.py",
        "prior.py": root / "evals/fleet/qwen38_dedicated_rank3_v3.py",
        "lane.py": root / "evals/fleet/qwen38_dedicated_rank3_a2_g22_v1.py",
        "parity.json": root / prior.PARITY,
        "rank3-release.json": root
        / ("docs/evidence/qwen38-study/2026-09-05-qwen38-dedicated-rank3-g21-b-v2-release-v3.json"),
        "campaign.json": root
        / "evals/fleet/configs/q38-glm53-exact-easiest100-pass4-campaign-v1.json",
        "selection.json": root / "evals/fleet/configs/opencode-easiest-train100-selection-v2.json",
        "source.json": root / "evals/fleet/configs/qwen38-opencode-train50-pass4-v3.json",
        "Dockerfile.opencode": root / "evals/fleet/Dockerfile.opencode",
        "fixed_proxy.py": root / "evals/fleet/fixed_proxy.py",
        "run.sh": root / RUN_SCRIPT,
        "release.json": release_path,
    }
    if any(path.is_symlink() or not path.is_file() for path in paths.values()):
        raise ValueError("rank3/a2 g22 package input is absent or unsafe")
    return {name: path.read_text() for name, path in paths.items()}
