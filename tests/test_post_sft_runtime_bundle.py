import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKER = ROOT / "evals/post_sft/runtime/training__init__.py"
PLAN = ROOT / "configs/evaluation/qwen36-27b-ft-run-574bd7b3-post-sft.json"


def test_isolated_training_package_marker_has_no_transitive_imports():
    tree = ast.parse(MARKER.read_text())
    assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree))


def test_every_post_sft_bundle_uses_and_freezes_the_minimal_marker():
    expected = "sha256:" + hashlib.sha256(MARKER.read_bytes()).hexdigest()
    plan = json.loads(PLAN.read_text())
    contracts = (
        plan["staging_execution"],
        plan["cast_execution"],
        plan["evidence_execution"]["base_artifact_inspector"],
        plan["evidence_execution"]["registration"],
    )
    for item in contracts:
        assert item["config_map_code_sha256"]["training/__init__.py"] == expected

    for relative in (
        "evals/post_sft/scripts/submit_evidence.sh",
        "evals/post_sft/scripts/submit_bf16_cast.sh",
        "evals/post_sft/scripts/submit_inference_stage.sh",
        "evals/post_sft/scripts/submit_base_artifact_inspection.sh",
        "evals/post_sft/scripts/submit_registration.sh",
    ):
        script = (ROOT / relative).read_text()
        assert 'evals/post_sft/runtime/training__init__.py' in script
        assert '--from-file=training__init__.py="$ROOT/training/__init__.py"' not in script
