import copy
import json
import subprocess
from pathlib import Path

import pytest

from evals.external_ctf import tensorlake
from evals.external_ctf.protocol import (
    build_plan,
    canonical,
    digest,
    file_digest,
    load_protocol,
    observed_source,
    validate_protocol,
)
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
    assert (
        value["arms"]["base"]["serving_runtime_sha256"]
        == value["arms"]["step_1000"]["serving_runtime_sha256"]
    )
    assert value["arms"]["step_1000"]["provenance"]["kind"] == "post_checkpoint"
    assert value["arms"]["step_1000"]["provenance"]["live_parity_receipt_sha256"].startswith(
        "sha256:"
    )


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
    nyu = build_plan(value, "nyu_ctf_web_test")
    unavailable_nyu = [row for row in nyu["cells"] if not row["launchable"]]
    assert len(unavailable_nyu) == 6
    assert {row["task_id"] for row in unavailable_nyu} == {
        "2021q-web-scp_terminal",
        "2020f-web-snailrace1",
        "2020f-web-sharkfacts",
    }
    assert {row["preflight_state"] for row in unavailable_nyu} == {
        "infra_invalid_no_reproducible_runtime"
    }
    assert nyu["official_task_count"] == 19
    assert nyu["executable_task_count"] == 16
    assert nyu["infrastructure_invalid_task_count"] == 3


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(data_policy="training_allowed"),
        lambda value: value["arms"]["step_1000"].update(max_context_size=32768),
        lambda value: value["benchmarks"]["nyu_ctf_web_test"].update(scoring="llm_judge"),
        lambda value: value["benchmarks"]["cybench_web"]["harness"].update(platform="linux/arm64"),
        lambda value: value["execution"]["sampling"].update(temperature=0.6),
        lambda value: value["execution"]["retry"].update(automatic_model_retry=True),
        lambda value: value["arms"]["base"].update(serving_runtime_sha256="sha256:" + "0" * 64),
        lambda value: value["arms"]["step_1000"]["provenance"].update(
            live_parity_receipt_sha256="sha256:" + "0" * 64
        ),
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


def test_nyu_availability_reads_pinned_commit_not_mutable_checkout(tmp_path: Path) -> None:
    checkout = tmp_path / "nyu"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.email", "tests@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.name", "External CTF Tests"],
        check=True,
    )
    repository = "https://example.invalid/nyu-ctf-bench"
    subprocess.run(["git", "-C", str(checkout), "remote", "add", "origin", repository], check=True)
    task_root = checkout / "benchmark" / "synthetic-web"
    task_root.mkdir(parents=True)
    dataset = {"synthetic-web": {"category": "web", "path": "benchmark/synthetic-web"}}
    manifest = json.dumps(dataset, sort_keys=True).encode()
    (checkout / "test_dataset.json").write_bytes(manifest)
    (checkout / "LICENSE").write_text("synthetic-license\n")
    (task_root / "challenge.json").write_text('{"compose": true}\n')
    (task_root / "docker-compose.yml").write_text("services: {}\n")
    subprocess.run(["git", "-C", str(checkout), "add", "."], check=True)
    subprocess.run(["git", "-C", str(checkout), "commit", "-qm", "pinned"], check=True)
    commit = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    protocol = {
        "benchmarks": {
            "nyu_ctf_web_test": {
                "source": {
                    "repository": repository,
                    "commit": commit,
                    "license": "synthetic",
                    "license_sha256": file_digest(b"synthetic-license\n"),
                },
                "source_manifest_sha256": file_digest(manifest),
                "task_ids": ["synthetic-web"],
                "task_ids_sha256": file_digest(b"synthetic-web\n"),
                "source_unavailable_task_ids": [],
                "execution_unavailable_task_ids": [],
            }
        }
    }

    # Simulate a collection-to-plan migration in the mutable checkout. The
    # frozen source observation must still read only the exact committed tree.
    (task_root / "challenge.json").write_text('{"compose": false}\n')
    (task_root / "docker-compose.yml").unlink()
    (task_root / "current-source-only.yml").write_text("not frozen\n")

    observed = observed_source(protocol, "nyu_ctf_web_test", checkout)
    assert observed["execution_unavailable_task_count"] == 0
    assert observed["verified"] is True


def _write_created(
    state: Path,
    protocol: dict,
    *,
    task_index: int,
    arm: str,
    sandbox_id: str,
) -> str:
    name = cell_name("cvebench_zero_day", task_index, arm)
    task_id = protocol["benchmarks"]["cvebench_zero_day"]["task_ids"][task_index]
    value = {
        "schema": "external_ctf_sandbox_created_v1",
        "benchmark": "cvebench_zero_day",
        "task_index": task_index,
        "task_id": task_id,
        "arm": arm,
        "name": name,
        "sandbox_id": sandbox_id,
        "status": "running",
        "protocol_sha256": protocol["protocol_sha256"],
        "spec_sha256": "sha256:" + "0" * 64,
    }
    (state / f"{name}.created.json").write_bytes(canonical(value) + b"\n")
    return name


def test_repeated_start_after_ambiguous_post_never_dispatches_again(
    tmp_path: Path, monkeypatch
) -> None:
    protocol = json.loads(PROTOCOL.read_text())
    state = tmp_path / "state"
    state.mkdir(mode=0o700)
    name = _write_created(state, protocol, task_index=0, arm="base", sandbox_id="sandbox-1")

    class Client:
        def __init__(self) -> None:
            self.posts = 0
            self.gets = 0

        def request(self, method, url, _payload=None, **_kwargs):
            if method == "GET":
                self.gets += 1
                return {"status": "running", "sandbox_url": "https://sandbox.invalid"}
            self.posts += 1
            raise TimeoutError("ambiguous provider response")

    client = Client()
    monkeypatch.setattr(tensorlake, "load_protocol", lambda _path: protocol)
    monkeypatch.setattr(tensorlake, "_client", lambda: client)
    monkeypatch.setenv("FLEET_API_KEY", "test-only-key")
    arguments = {
        "protocol_path": PROTOCOL,
        "state": state,
        "benchmark": "cvebench_zero_day",
        "task_index": 0,
        "arm": "base",
    }
    with pytest.raises(TimeoutError, match="ambiguous"):
        tensorlake.start(**arguments)
    assert (state / f"{name}.process-claim.json").is_file()
    assert not (state / f"{name}.process.json").exists()

    with pytest.raises(tensorlake.ExternalCtfError, match="cell_start_already_claimed"):
        tensorlake.start(**arguments)
    assert client.posts == 1
    assert client.gets == 1


def test_max_parallel_one_blocks_a_second_cell_before_provider_access(
    tmp_path: Path, monkeypatch
) -> None:
    protocol = json.loads(PROTOCOL.read_text())
    state = tmp_path / "state"
    state.mkdir(mode=0o700)
    first = _write_created(state, protocol, task_index=0, arm="base", sandbox_id="sandbox-1")
    _write_created(state, protocol, task_index=0, arm="step_1000", sandbox_id="sandbox-2")
    (state / f"{first}.create-claim.json").write_bytes(b"{}\n")
    (state / f"{first}.process-claim.json").write_bytes(b"{}\n")

    monkeypatch.setattr(tensorlake, "load_protocol", lambda _path: protocol)
    monkeypatch.setattr(
        tensorlake,
        "_client",
        lambda: (_ for _ in ()).throw(AssertionError("provider must not be accessed")),
    )
    monkeypatch.setenv("FLEET_API_KEY", "test-only-key")
    with pytest.raises(tensorlake.ExternalCtfError, match="max_parallel_cells_exceeded"):
        tensorlake.start(
            protocol_path=PROTOCOL,
            state=state,
            benchmark="cvebench_zero_day",
            task_index=0,
            arm="step_1000",
        )
