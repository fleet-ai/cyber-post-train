import copy
import json
from pathlib import Path

import pytest

from evals.external_ctf.protocol import build_plan, digest, load_protocol, validate_protocol
from evals.external_ctf.tensorlake import active_project_count, cell_name, external_names
from evals.external_ctf.worker import main as worker_main

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
        {
            "task_id": "CVE-2023-37999",
            "attempt": 0,
            "arm": "base",
            "launchable": True,
            "preflight_state": "ready",
        },
        {
            "task_id": "CVE-2023-37999",
            "attempt": 0,
            "arm": "step_1000",
            "launchable": True,
            "preflight_state": "ready",
        },
        {
            "task_id": "CVE-2023-51483",
            "attempt": 0,
            "arm": "step_1000",
            "launchable": True,
            "preflight_state": "ready",
        },
        {
            "task_id": "CVE-2023-51483",
            "attempt": 0,
            "arm": "base",
            "launchable": True,
            "preflight_state": "ready",
        },
    ]
    cybench = build_plan(value, "cybench_web")
    assert len(cybench["cells"]) == 12
    unavailable = [row for row in cybench["cells"] if not row["launchable"]]
    assert len(unavailable) == 2
    assert all("GlacierExchange" in row["task_id"] for row in unavailable)
    assert cybench["official_task_count"] == 6
    assert cybench["executable_task_count"] == 5
    assert cybench["infrastructure_invalid_task_count"] == 1
    assert cybench["plan_sha256"] == digest(
        {key: item for key, item in cybench.items() if key != "plan_sha256"}
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(data_policy="training_allowed"),
        lambda value: value["arms"]["step_1000"].update(max_context_size=32768),
        lambda value: value["benchmarks"]["nyu_ctf_web_test"].update(scoring="llm_judge"),
        lambda value: value["benchmarks"]["cybench_web"]["harness"].update(platform="linux/arm64"),
        lambda value: value["execution"]["sampling"].update(temperature=0.6),
        lambda value: value["execution"]["retry"].update(automatic_model_retry=True),
        lambda value: value["benchmarks"]["cvebench_zero_day"]["bindings"].update(
            grader_sha256="sha256:" + "0" * 64
        ),
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


def test_shared_capacity_counts_only_exact_live_project_names() -> None:
    protocol = load_protocol(PROTOCOL)
    names = external_names(protocol)
    assert len(names) == (40 + 19 + 6) * 2
    target = cell_name("cvebench_zero_day", 0, "base")
    rows = [
        {"name": target, "status": "running"},
        {"name": "unrelated-running-sandbox", "status": "running"},
        {"name": cell_name("cvebench_zero_day", 1, "base"), "status": "terminated"},
    ]
    assert active_project_count(rows, names) == 1
    with pytest.raises(RuntimeError, match="inventory_conflict"):
        active_project_count([rows[0], rows[0]], names)


def test_remote_worker_fails_closed_off_linux_amd64(monkeypatch) -> None:
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr("platform.machine", lambda: "arm64")
    with pytest.raises(RuntimeError, match="linux_amd64_required"):
        worker_main()
