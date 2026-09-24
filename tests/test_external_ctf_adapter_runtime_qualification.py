from __future__ import annotations

from pathlib import Path

from evals.external_ctf import adapter_runtime_qualification as qualification
from evals.external_ctf.protocol import digest, file_digest, load_protocol


def _signed(value: dict) -> dict:
    return {**value, "receipt_sha256": digest(value)}


def _runtime_receipt(protocol: dict, benchmark: str, task_index: int) -> dict:
    value = {
        "status": "runtime_qualified",
        "error_code": None,
        "qualification_contract_sha256": protocol["benchmarks"][benchmark]["runtime_qualification"][
            "contract_sha256"
        ],
        "source_commit": protocol["benchmarks"][benchmark]["source"]["commit"],
        "task_index": task_index,
        "task_id_sha256": file_digest(
            protocol["benchmarks"][benchmark]["task_ids"][task_index].encode()
        ),
        "provider_calls": 0,
        "model_requests": 0,
        "scored_attempts": 0,
        "contains_prompts_flags_solutions_traces_or_scores": False,
    }
    return _signed(value)


def _credential_receipt() -> dict:
    value = {
        "status": "qualified",
        "adapter_source_sha256": qualification.opencode_scored.source_sha256(),
        "credential_environment_names_checked": sorted(qualification.SENSITIVE_ENVIRONMENT_NAMES),
        "provider_calls": 0,
        "model_requests": 0,
        "scoring_calls": 0,
        "provider_credential_location": "fixed_proxy_only",
        "ambient_real_credential_present": False,
        "agent_real_model_or_scoring_credential_present": False,
        "challenge_real_model_or_scoring_credential_present": False,
        "tensorlake_management_credential_forwarded": False,
        "agent_docker_socket_present": False,
        "challenge_docker_socket_present": False,
        "cleanup_verified": True,
        "contains_credentials_prompts_flags_solutions_traces_or_scores": False,
    }
    return _signed(value)


def _model_free_environment(monkeypatch) -> None:
    monkeypatch.setattr(qualification.platform, "system", lambda: "Linux")
    monkeypatch.setattr(qualification.platform, "machine", lambda: "x86_64")
    for name in qualification.SENSITIVE_ENVIRONMENT_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(qualification, "_docker_ids", lambda _kind: set())
    monkeypatch.setattr(qualification, "_clone", lambda *_args: None)


def test_combined_qualification_is_model_free_and_exact(monkeypatch) -> None:
    protocol = load_protocol()
    _model_free_environment(monkeypatch)
    monkeypatch.setattr(
        qualification.opencode_scored,
        "qualify_credential_boundary",
        lambda _protocol: _credential_receipt(),
    )
    monkeypatch.setattr(
        qualification.nyu,
        "qualify",
        lambda _checkout, task_index, _protocol: _runtime_receipt(
            protocol, qualification.nyu.BENCHMARK, task_index
        ),
    )
    monkeypatch.setattr(
        qualification.cybench,
        "qualify",
        lambda _checkout, task_index, _image, _protocol: _runtime_receipt(
            protocol, qualification.cybench.BENCHMARK, task_index
        ),
    )

    receipt = qualification.qualify()

    assert receipt["status"] == "runtime_qualified"
    assert receipt["provider_calls"] == receipt["model_requests"] == 0
    assert receipt["scoring_calls"] == receipt["scored_attempts"] == 0
    assert [row["phase"] for row in receipt["phases"]] == [
        "credential_isolation",
        "nyu_ctf_web_test",
        "cybench_web",
    ]
    assert all(row["cleanup_verified"] for row in receipt["phases"])
    assert receipt["receipt_sha256"] == digest(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    assert qualification.validate_receipt(receipt, protocol) == receipt


def test_phase_failure_does_not_suppress_later_phase(monkeypatch) -> None:
    protocol = load_protocol()
    _model_free_environment(monkeypatch)
    called: list[str] = []
    monkeypatch.setattr(
        qualification.opencode_scored,
        "qualify_credential_boundary",
        lambda _protocol: _credential_receipt(),
    )

    def fail_nyu(*_args) -> dict:
        called.append("nyu")
        raise RuntimeError("not exposed in receipt")

    def pass_cybench(_checkout: Path, task_index: int, _image: str, _protocol: Path) -> dict:
        called.append("cybench")
        return _runtime_receipt(protocol, qualification.cybench.BENCHMARK, task_index)

    monkeypatch.setattr(qualification.nyu, "qualify", fail_nyu)
    monkeypatch.setattr(qualification.cybench, "qualify", pass_cybench)

    receipt = qualification.qualify()

    assert called == ["nyu", "cybench"]
    assert receipt["status"] == "infrastructure_invalid"
    assert receipt["phases"][1]["status"] == "infrastructure_invalid"
    assert receipt["phases"][2]["status"] == "qualified"
    assert "not exposed in receipt" not in str(receipt)


def test_worker_rejects_any_real_credential(monkeypatch) -> None:
    monkeypatch.setattr(qualification.platform, "system", lambda: "Linux")
    monkeypatch.setattr(qualification.platform, "machine", lambda: "x86_64")
    monkeypatch.setenv("TENSORLAKE_API_KEY", "secret")

    try:
        qualification.qualify()
    except qualification.AdapterQualificationError as error:
        assert str(error) == "real_credential_present_in_worker_environment"
    else:
        raise AssertionError("credential-bearing worker environment was accepted")
