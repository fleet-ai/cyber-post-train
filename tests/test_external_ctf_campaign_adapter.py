from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from evals import campaign
from evals.external_ctf import campaign_adapter
from evals.external_ctf.protocol import DEFAULT_PROTOCOL, canonical, file_digest, load_protocol

SHA = "sha256:" + "1" * 64
ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "configs/evaluation/qwen38-top5-multibench-pass4-matrix-20260923-v1.json"


def _models() -> list[dict]:
    matrix = json.loads(MATRIX.read_bytes())
    return [
        {
            "id": arm["arm_id"],
            "checkpoint_id": arm["artifact_id"],
            "matrix_arm_sha256": campaign.digest(arm),
            "weights_sha256": "sha256:" + str(index) * 64,
            "matched_treatment_receipt_sha256": SHA,
            "serving_route_receipt_sha256": "sha256:" + "7" * 64,
            "live_parity_receipt_sha256": "sha256:" + "8" * 64,
            "served_model": "served-" + arm["arm_id"],
        }
        for index, arm in enumerate(matrix["arms"], start=1)
    ]


def _bindings() -> dict:
    matrix = json.loads(MATRIX.read_bytes())
    return {
        "budgets_sha256": "sha256:" + "2" * 64,
        "matrix_sha256": "sha256:" + matrix["sha256"],
        "harness_receipt_sha256": {
            benchmark: "sha256:" + "4" * 64 for benchmark in campaign_adapter.BENCHMARKS
        },
        "scoring_protocol_sha256": {
            benchmark: "sha256:" + "5" * 64 for benchmark in campaign_adapter.BENCHMARKS
        },
        "models_loaded": _models(),
    }


def _write_signed(path: Path, value: dict) -> dict:
    signed = campaign_adapter._signed(value)  # noqa: SLF001
    path.write_bytes(canonical(signed) + b"\n")
    return signed


def _reference(path: Path, receipt: str) -> dict:
    return {
        "path": str(path.resolve()),
        "file_sha256": file_digest(path.read_bytes()),
        "receipt_sha256": receipt,
    }


def test_binding_loader_accepts_formatted_protocol_and_exact_signed_evidence(
    tmp_path: Path,
) -> None:
    protocol = load_protocol()
    retry_path = tmp_path / "retry.json"
    retry = _write_signed(retry_path, {"schema": "test_retry"})
    qualification_path = tmp_path / "qualification.json"
    qualification = _write_signed(qualification_path, {"schema": "test_qualification"})
    summary_references = {}
    for benchmark in campaign_adapter.BENCHMARKS:
        summary_path = tmp_path / f"{benchmark}.json"
        summary = _write_signed(
            summary_path,
            {
                "schema": "external_ctf_qualification_summary_v1",
                "benchmark": benchmark,
                "protocol_sha256": protocol["protocol_sha256"],
                "qualification_contract_sha256": protocol["benchmarks"][benchmark][
                    "runtime_qualification"
                ]["contract_sha256"],
                "qualification_packet_receipt_sha256": qualification["receipt_sha256"],
                "score_reads": 0,
                "model_requests": 0,
                "rows": [],
            },
        )
        summary_references[benchmark] = _reference(summary_path, summary["receipt_sha256"])
    models = []
    for model in _models():
        route_path = tmp_path / f"{model['id']}.json"
        route = _write_signed(
            route_path,
            {
                "status": "accepted",
                "served_model": model["served_model"],
                "weights_sha256": model["weights_sha256"],
                "serving_route_receipt_sha256": model["serving_route_receipt_sha256"],
                "live_parity_receipt_sha256": model["live_parity_receipt_sha256"],
            },
        )
        models.append(
            {
                **model,
                "route_preflight": _reference(route_path, route["receipt_sha256"]),
            }
        )
    bindings_path = tmp_path / "bindings.json"
    bindings = _write_signed(
        bindings_path,
        {
            "schema": campaign_adapter.SCHEMA,
            "matrix": _reference(
                MATRIX, "sha256:" + json.loads(MATRIX.read_bytes())["sha256"]
            ),
            "budgets_sha256": "sha256:" + "2" * 64,
            "protocol": _reference(DEFAULT_PROTOCOL, protocol["protocol_sha256"]),
            "web_retry_execution": _reference(retry_path, retry["receipt_sha256"]),
            "qualification_packet": _reference(qualification_path, qualification["receipt_sha256"]),
            "qualification_summaries": summary_references,
            "harness_receipt_sha256": {
                benchmark: "sha256:" + "4" * 64 for benchmark in campaign_adapter.BENCHMARKS
            },
            "scoring_protocol_sha256": {
                benchmark: "sha256:" + "5" * 64 for benchmark in campaign_adapter.BENCHMARKS
            },
            "models": models,
        },
    )

    loaded, observed_protocol = campaign_adapter.load_bindings(bindings_path)

    assert loaded["receipt_sha256"] == bindings["receipt_sha256"]
    assert loaded["matrix_sha256"] == "sha256:" + json.loads(MATRIX.read_bytes())["sha256"]
    assert observed_protocol["protocol_sha256"] == protocol["protocol_sha256"]
    assert [row["id"] for row in loaded["models_loaded"]] == [row["id"] for row in models]


def test_rendered_config_is_exact_six_arm_pass4_native_campaign(monkeypatch) -> None:
    protocol = load_protocol()
    bindings = _bindings()
    monkeypatch.setattr(campaign_adapter, "load_bindings", lambda _path: (bindings, protocol))

    config = campaign_adapter.controller_config(Path("/private/tmp/bindings.json"))
    plan = campaign.build_plan(config)

    assert len(config["models"]) == 6
    assert len(plan["targets"]) == 1464
    assert config["pass_k"] == 4
    assert config["scheduler"] == {"max_launches_per_step": 4, "serial_canaries": True}
    assert [len(row["targets"]) for row in config["benchmarks"]] == [40, 16, 5]
    assert all(
        row["sampling"]["attempt_seeds"] == [None, None, None, None] for row in config["benchmarks"]
    )
    assert all(row["score_driver"]["provider"] == "local" for row in config["benchmarks"])
    assert all(row["rollout_driver"]["provider"] == "tensorlake" for row in config["benchmarks"])
    assert "gpt" not in json.dumps(config).lower()
    assert sum(target["canary"] for target in plan["targets"]) == 6 * 4
    assert not any(
        target["canary"]
        for target in plan["targets"]
        if target["identity"]["benchmark"]["id"] != "cvebench_zero_day"
    )


def test_remote_names_bind_model_attempt_and_experiment_key(monkeypatch) -> None:
    protocol = load_protocol()
    bindings = _bindings()
    monkeypatch.setattr(campaign_adapter, "load_bindings", lambda _path: (bindings, protocol))
    plan = campaign.build_plan(
        campaign_adapter.controller_config(Path("/private/tmp/bindings.json"))
    )
    first = plan["targets"][0]
    second_model = next(
        row
        for row in plan["targets"]
        if row["identity"]["target"] == first["identity"]["target"]
        and row["identity"]["attempt"] == first["identity"]["attempt"]
        and row["identity"]["model"] != first["identity"]["model"]
    )
    second_attempt = next(
        row
        for row in plan["targets"]
        if row["identity"]["target"] == first["identity"]["target"]
        and row["identity"]["model"] == first["identity"]["model"]
        and row["identity"]["attempt"] != first["identity"]["attempt"]
    )

    names = {
        campaign_adapter.remote_name(first),
        campaign_adapter.remote_name(second_model),
        campaign_adapter.remote_name(second_attempt),
    }
    assert len(names) == 3
    assert campaign_adapter.remote_name(first) == campaign_adapter.remote_name(copy.deepcopy(first))
    assert first["experiment_key"].removeprefix("sha256:")[:12] in campaign_adapter.remote_name(
        first
    )


def test_packet_rejects_driver_source_drift(tmp_path: Path, monkeypatch) -> None:
    protocol = load_protocol()
    bindings = _bindings()
    monkeypatch.setattr(campaign_adapter, "load_bindings", lambda _path: (bindings, protocol))
    packet = campaign.build_plan(
        campaign_adapter.controller_config(Path("/private/tmp/bindings.json"))
    )["targets"][0]
    packet_path = tmp_path / "packet.json"
    packet_path.write_bytes(canonical(packet) + b"\n")

    observed, _cell = campaign_adapter._packet(packet_path, bindings, protocol)  # noqa: SLF001
    assert observed["experiment_key"] == packet["experiment_key"]

    packet["drivers"]["rollout"]["source_sha256"] = "sha256:" + "0" * 64
    packet_path.write_bytes(canonical(packet) + b"\n")
    with pytest.raises(campaign_adapter.ExternalCampaignError, match="campaign_packet_invalid"):
        campaign_adapter._packet(packet_path, bindings, protocol)  # noqa: SLF001


def test_current_adapter_gates_enable_cve_and_hold_nyu_and_cybench() -> None:
    protocol = load_protocol()
    summaries = {}
    cells = {}
    for benchmark in campaign_adapter.BENCHMARKS:
        index = campaign_adapter.CANARIES[benchmark]
        task_id = protocol["benchmarks"][benchmark]["task_ids"][index]
        summaries[benchmark] = {
            "rows": [
                {
                    "task_index": index,
                    "task_id": task_id,
                    "outcome": "runtime_preflight_passed",
                    "terminal_receipt_sha256": SHA,
                    "release_receipt_sha256": SHA,
                }
            ]
        }
        cells[benchmark] = {"benchmark": benchmark, "task_index": index, "task_id": task_id}
    bindings = {"qualification_summaries_loaded": summaries}

    assert campaign_adapter._qualification_ready(  # noqa: SLF001
        bindings, protocol, cells["cvebench_zero_day"]
    )
    assert not campaign_adapter._qualification_ready(  # noqa: SLF001
        bindings, protocol, cells["nyu_ctf_web_test"]
    )
    assert not campaign_adapter._qualification_ready(  # noqa: SLF001
        bindings, protocol, cells["cybench_web"]
    )


def test_campaign_result_is_exact_and_native() -> None:
    protocol = load_protocol()
    model = _models()[1]
    cell = {
        "benchmark": "cvebench_zero_day",
        "task_id": protocol["benchmarks"]["cvebench_zero_day"]["task_ids"][5],
        "model": model,
        "experiment_key": "sha256:" + "9" * 64,
        "attempt": 3,
    }
    value = {
        "schema": "external_ctf_campaign_cell_result_v1",
        "protocol_sha256": protocol["protocol_sha256"],
        "benchmark": cell["benchmark"],
        "task_id": cell["task_id"],
        "arm": model["id"],
        "model_id": model["id"],
        "weights_sha256": model["weights_sha256"],
        "experiment_key": cell["experiment_key"],
        "attempt": 3,
        "status": "scored",
        "score": 1,
        "grader": protocol["benchmarks"]["cvebench_zero_day"]["scoring"],
    }
    raw = canonical(value) + b"\n"

    assert campaign_adapter._campaign_result(raw, protocol=protocol, cell=cell) == value  # noqa: SLF001
    tampered = {**value, "weights_sha256": "sha256:" + "0" * 64}
    with pytest.raises(campaign_adapter.ExternalCampaignError, match="campaign_result_invalid"):
        campaign_adapter._campaign_result(  # noqa: SLF001
            canonical(tampered) + b"\n", protocol=protocol, cell=cell
        )


def test_closed_adapter_launch_fails_before_provider_access() -> None:
    with pytest.raises(
        campaign_adapter.ExternalCampaignError, match="scored_adapter_not_qualified"
    ):
        campaign_adapter._launch_cell(  # noqa: SLF001
            protocol={},
            bindings={},
            cell={"benchmark": "nyu_ctf_web_test"},
            authority={},
            output=Path("/not/reached"),
        )


@pytest.mark.parametrize(
    ("context", "canary", "expected"),
    [
        ({"active": 99, "campaign_active": 0}, True, True),
        ({"active": 100, "campaign_active": 0}, True, False),
        ({"active": 99, "campaign_active": 1}, True, False),
        ({"active": 99, "campaign_active": 3}, False, True),
        ({"active": 99, "campaign_active": 4}, False, False),
    ],
)
def test_capacity_gate_honors_shared_project_and_campaign_limits(
    context: dict, canary: bool, expected: bool
) -> None:
    assert campaign_adapter._capacity_ready(context, {"canary": canary}) is expected  # noqa: SLF001
