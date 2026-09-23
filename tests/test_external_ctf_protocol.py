import copy
import json
from pathlib import Path

import pytest

from evals.external_ctf.protocol import build_plan, digest, load_protocol, validate_protocol

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/evaluation/qwen38-external-ctf-paired-v1.json"


def test_protocol_binds_exact_benchmarks_models_and_eval_only_boundary() -> None:
    value = load_protocol(PROTOCOL)
    assert set(value["benchmarks"]) == {
        "cvebench_zero_day",
        "nyu_ctf_web_test",
        "cybench_web",
    }
    assert [value["benchmarks"][name]["task_count"] for name in value["benchmarks"]] == [
        40,
        19,
        6,
    ]
    assert value["data_policy"] == "evaluation_only_never_training_or_tuning"
    assert value["arms"]["base"]["served_model"] == "qwen3.8-27b"
    assert value["arms"]["step_1000"]["served_model"] == "chris-q38-t3k32-s1000-v1"


def test_plan_is_paired_counterbalanced_and_excludes_only_missing_source() -> None:
    value = load_protocol(PROTOCOL)
    cve = build_plan(value, "cvebench_zero_day")
    assert len(cve["cells"]) == 80
    assert cve["cells"][:4] == [
        {"task_id": "CVE-2023-37999", "attempt": 0, "arm": "base"},
        {"task_id": "CVE-2023-37999", "attempt": 0, "arm": "step_1000"},
        {"task_id": "CVE-2023-51483", "attempt": 0, "arm": "step_1000"},
        {"task_id": "CVE-2023-51483", "attempt": 0, "arm": "base"},
    ]
    cybench = build_plan(value, "cybench_web")
    assert len(cybench["cells"]) == 10
    assert all("GlacierExchange" not in row["task_id"] for row in cybench["cells"])
    assert cybench["plan_sha256"] == digest(
        {key: item for key, item in cybench.items() if key != "plan_sha256"}
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(data_policy="training_allowed"),
        lambda value: value["arms"]["step_1000"].update(max_context_size=32768),
        lambda value: value["benchmarks"]["nyu_ctf_web_test"].update(scoring="llm_judge"),
    ],
)
def test_protocol_fails_closed_on_scientific_drift(mutation) -> None:
    value = json.loads(PROTOCOL.read_text())
    mutation(value)
    value["protocol_sha256"] = digest(
        {key: item for key, item in value.items() if key != "protocol_sha256"}
    )
    with pytest.raises(ValueError):
        validate_protocol(copy.deepcopy(value))
