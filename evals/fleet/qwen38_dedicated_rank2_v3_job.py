"""Render the create-once Kubernetes evaluator Job for dedicated Qwen rank2/a2."""

# ruff: noqa: E501 -- shell bootstrap paths are intentionally exact.

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from evals.fleet import self_hosted

SOURCE = Path("evals/fleet/cluster/qwen38-dedicated-tp1-scored-canary-v2.yaml")
SOURCE_SHA256 = "sha256:aa299342269f5d7f1ef34fad35965904e1c9349e596444a2269ddd55cd3d89c8"
NAME = "chris-cyber-q38-opencode11827-ded-tp1-r002-a2-v3"
EXPERIMENT = "q38-ded-tp1-r002-a2-v3"
SERVING_BLOCK = "dedicated-qwen-tp1-v3"

BOOTSTRAP = """root=/workspace/cyber-post-train
mkdir -p "$root/evals/fleet/configs" "$root/evals/fleet/scripts" "$root/docs/evidence/qwen38-study"
touch "$root/evals/__init__.py" "$root/evals/fleet/__init__.py"
install -m 0644 /bootstrap/self_hosted.py "$root/evals/fleet/self_hosted.py"
install -m 0644 /bootstrap/exact_pass4_crypto.py "$root/evals/fleet/exact_pass4_crypto.py"
install -m 0644 /bootstrap/exact_pass4_universe.py "$root/evals/fleet/exact_pass4_universe.py"
install -m 0644 /bootstrap/legacy.py "$root/evals/fleet/qwen38_dedicated_scored_canary_v1.py"
install -m 0644 /bootstrap/lane.py "$root/evals/fleet/qwen38_dedicated_rank2_v3.py"
install -m 0644 /bootstrap/parity.json "$root/docs/evidence/qwen38-study/2026-09-05-qwen38-dedicated-v3-actual-opencode-parity.json"
install -m 0644 /bootstrap/campaign.json "$root/evals/fleet/configs/q38-glm53-exact-easiest100-pass4-campaign-v1.json"
install -m 0644 /bootstrap/selection.json "$root/evals/fleet/configs/opencode-easiest-train100-selection-v2.json"
install -m 0644 /bootstrap/source.json "$root/evals/fleet/configs/qwen38-opencode-train50-pass4-v3.json"
install -m 0644 /bootstrap/Dockerfile.opencode "$root/evals/fleet/Dockerfile.opencode"
install -m 0644 /bootstrap/fixed_proxy.py "$root/evals/fleet/fixed_proxy.py"
install -m 0755 /bootstrap/run.sh "$root/evals/fleet/scripts/run.sh"
exec "$root/evals/fleet/scripts/run.sh"
"""


def render(root: Path) -> dict[str, Any]:
    source = root / SOURCE
    if self_hosted.sha256(source.read_bytes()) != SOURCE_SHA256:
        raise ValueError("source evaluator Job template drifted")
    value = yaml.safe_load(source.read_text())
    result = copy.deepcopy(value)
    result["metadata"]["name"] = NAME
    result["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = EXPERIMENT
    result["metadata"]["annotations"]["cyber-post-train.fleet.ai/serving-block"] = SERVING_BLOCK
    template = result["spec"]["template"]
    template["metadata"]["labels"]["cyber-post-train.fleet.ai/experiment"] = EXPERIMENT
    template["spec"]["volumes"][0]["configMap"]["name"] = NAME
    template["spec"]["containers"][0]["args"] = [BOOTSTRAP]
    return result


def main() -> int:
    print(yaml.safe_dump(render(Path.cwd()), sort_keys=False), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
