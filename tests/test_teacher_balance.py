import json
import re
from pathlib import Path

from training.io import digest_json

CONFIG = (
    Path(__file__).parents[1]
    / "configs"
    / "studies"
    / "qwen-blackbox-teacher-balanced-exposure-v1.json"
)


def test_teacher_balance_lock_is_self_digested_and_nonlaunchable():
    value = json.loads(CONFIG.read_text())
    assert value["sha256"] == digest_json({k: v for k, v in value.items() if k != "sha256"})
    assert value["treatment"]["role"] == "teacher_availability_ablation"
    assert value["treatment"]["unit"] == "whole_certified_native_episode"
    assert value["launch_status"].startswith("blocked_until_private_balanced_corpora")
    assert "partial episode truncation" in value["treatment"]["prohibited"]
    assert "duplicated token targets" in value["treatment"]["prohibited"]


def test_teacher_balance_preserves_coverage_while_reducing_family_inequality():
    value = json.loads(CONFIG.read_text())
    for variant in value["variants"].values():
        available, balanced = variant["available"], variant["balanced"]
        assert balanced["families"] == available["families"]
        assert balanced["episodes"] < available["episodes"]
        assert balanced["supervised_tokens"] < available["supervised_tokens"]
        assert balanced["family_token_gini"] < available["family_token_gini"]
        assert (
            balanced["family_token_coefficient_of_variation"]
            < available["family_token_coefficient_of_variation"]
        )
        for key, digest in variant.items():
            if key.endswith("sha256"):
                assert re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
