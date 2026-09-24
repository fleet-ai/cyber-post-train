"""Bridge external CTF cells into the resumable evaluation campaign controller.

The benchmark runtimes and native graders remain in ``external_ctf.tensorlake``
and ``external_ctf.worker``.  This module only freezes the six-arm/pass@4
universe, translates campaign cells to attempt-unique provider identities, and
emits the controller's small receipts.
"""

from __future__ import annotations

import argparse
import base64
import json
import lzma
import os
import re
import time
import urllib.parse
from pathlib import Path
from typing import Any

from evals.campaign import RECEIPT_SCHEMA, canonical, digest
from evals.webexploitbench.tensorlake import collection_replica_set as replica_set

from . import opencode_scored, tensorlake
from .execution_packet import load as load_execution_packet
from .protocol import file_digest, load_protocol

SCHEMA = "external_ctf_campaign_bindings_v1"
BENCHMARKS = ("cvebench_zero_day", "nyu_ctf_web_test", "cybench_web")
CANARIES = {"cvebench_zero_day": 5, "nyu_ctf_web_test": 6, "cybench_web": 2}
PREFIXES = {"cvebench_zero_day": "cve", "nyu_ctf_web_test": "nyu", "cybench_web": "cyb"}
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{1,127}")
_RESULT_RECONCILE_ATTEMPTS = 6
_RESULT_RECONCILE_DELAY_SECONDS = 2
_CAMPAIGN_WORKER_BOOTSTRAP = r"""
import base64
import hashlib
import importlib.util
import json
import lzma
import os
import platform
import pathlib
import subprocess
import sys

bundle = os.environ.pop("CAMPAIGN_BUNDLE_B64", None)
if bundle is not None:
    compressed = base64.b64decode(bundle, validate=True)
    if "sha256:" + hashlib.sha256(compressed).hexdigest() != os.environ.pop(
        "CAMPAIGN_BUNDLE_SHA256"
    ):
        raise RuntimeError("campaign_bundle_digest_mismatch")
    files = json.loads(lzma.decompress(compressed))
    root = pathlib.Path("/workspace/external_ctf_campaign")
    if root.exists() or root.is_symlink():
        raise RuntimeError("campaign_bundle_root_exists")
    for package in (root / "evals", root / "evals/external_ctf"):
        package.mkdir(mode=0o700, parents=True)
        (package / "__init__.py").write_bytes(b"")
    for relative, encoded in files.items():
        relative_path = pathlib.PurePosixPath(relative)
        if (
            relative_path.is_absolute()
            or not relative_path.parts
            or any(part in {"", ".", ".."} for part in relative_path.parts)
        ):
            raise RuntimeError("campaign_bundle_path_invalid")
        target = root.joinpath(*relative_path.parts)
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        target.write_bytes(base64.b64decode(encoded, validate=True))
    sys.path.insert(0, str(root))

worker = base64.b64decode(os.environ.pop("WORKER_B64"), validate=True)
if "sha256:" + hashlib.sha256(worker).hexdigest() != os.environ.pop("WORKER_SHA256"):
    raise RuntimeError("worker_digest_mismatch")
path = pathlib.Path("/workspace/external_ctf_worker.py")
path.write_bytes(worker)
spec = importlib.util.spec_from_file_location("external_ctf_worker", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
model_raw = base64.b64decode(os.environ.pop("CAMPAIGN_MODEL_B64"), validate=True)
if "sha256:" + hashlib.sha256(model_raw).hexdigest() != os.environ.pop(
    "CAMPAIGN_MODEL_SHA256"
):
    raise RuntimeError("campaign_model_digest_mismatch")
model = json.loads(model_raw)
original = module._run_cvebench

def run(protocol, task_id, arm, checkout, qualification):
    protocol["arms"][arm] = {**protocol["arms"][arm], "served_model": model["served_model"]}
    return original(protocol, task_id, arm, checkout, qualification)

module._run_cvebench = run

def external_qualification(protocol, benchmark, task_id):
    raw = base64.b64decode(os.environ.pop("EXTERNAL_CTF_QUALIFICATION_B64"), validate=True)
    if module._sha(raw) != os.environ.pop("EXTERNAL_CTF_QUALIFICATION_SHA256"):
        raise ValueError("runtime_qualification_file_digest_mismatch")
    terminal = json.loads(raw)
    result = terminal.get("result") if isinstance(terminal, dict) else None
    unsigned_terminal = {
        key: item for key, item in terminal.items() if key != "receipt_sha256"
    }
    unsigned_result = (
        {key: item for key, item in result.items() if key != "receipt_sha256"}
        if isinstance(result, dict)
        else {}
    )
    task_index = protocol["benchmarks"][benchmark]["task_ids"].index(task_id)
    if (
        raw != module._canonical(terminal) + b"\n"
        or terminal.get("receipt_sha256") != module._sha(module._canonical(unsigned_terminal))
        or terminal.get("schema") != "external_ctf_cell_terminal_v1"
        or terminal.get("protocol_sha256") != protocol["protocol_sha256"]
        or terminal.get("benchmark") != benchmark
        or terminal.get("task_index") != task_index
        or terminal.get("task_id") != task_id
        or terminal.get("arm") != "qualification"
        or terminal.get("outcome") != "runtime_preflight_passed"
        or not isinstance(result, dict)
        or result.get("receipt_sha256") != module._sha(module._canonical(unsigned_result))
        or result.get("protocol_sha256") != protocol["protocol_sha256"]
        or result.get("benchmark") != benchmark
        or result.get("task_index") != task_index
        or result.get("status") != "runtime_qualified"
        or result.get("model_requests") != 0
    ):
        raise ValueError("runtime_qualification_binding_mismatch")
    if benchmark == "nyu_ctf_web_test":
        qualification = result.get("qualification")
        if not isinstance(qualification, dict):
            raise ValueError("runtime_qualification_binding_mismatch")
        return qualification
    if benchmark == "cybench_web" and isinstance(result.get("task_evidence"), dict):
        return result
    raise ValueError("runtime_qualification_binding_mismatch")

def run_external():
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("linux_amd64_required")
    protocol = module._protocol()
    benchmark = os.environ.pop("EXTERNAL_CTF_BENCHMARK")
    task_id = os.environ.pop("EXTERNAL_CTF_TASK_ID")
    arm = os.environ.pop("EXTERNAL_CTF_ARM")
    mode = os.environ.pop("EXTERNAL_CTF_MODE")
    source = protocol["benchmarks"][benchmark]
    if (
        benchmark not in {"nyu_ctf_web_test", "cybench_web"}
        or mode != "scored"
        or arm != "base"
        or task_id not in source["task_ids"]
        or task_id in source.get("source_unavailable_task_ids", [])
        or task_id in source.get("execution_unavailable_task_ids", [])
    ):
        raise ValueError("cell_identity_invalid")
    module._CONTEXT.update(
        {
            "protocol_sha256": protocol["protocol_sha256"],
            "benchmark": benchmark,
            "task_id": task_id,
            "arm": arm,
        }
    )
    protocol["arms"][arm] = {
        **protocol["arms"][arm],
        "served_model": model["served_model"],
    }
    checkout = pathlib.Path("/workspace/external-ctf-source")
    subprocess.run(
        ["git", "clone", "--quiet", source["source"]["repository"], str(checkout)],
        check=True,
    )
    subprocess.run(
        ["git", "checkout", "--quiet", source["source"]["commit"]],
        cwd=checkout,
        check=True,
    )
    observed = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
    ).strip()
    if observed != source["source"]["commit"]:
        raise ValueError("source_commit_mismatch")
    qualification = external_qualification(protocol, benchmark, task_id)
    from evals.external_ctf import opencode_scored

    runner = (
        opencode_scored.run_nyu
        if benchmark == "nyu_ctf_web_test"
        else opencode_scored.run_cybench
    )
    passed = runner(protocol, checkout, task_id, arm, qualification)
    module._write(
        {
            "schema": "external_ctf_cell_result_v1",
            "protocol_sha256": protocol["protocol_sha256"],
            "benchmark": benchmark,
            "task_id": task_id,
            "arm": arm,
            "status": "scored",
            "score": int(passed),
            "grader": source["scoring"],
        }
    )

try:
    if os.environ.get("EXTERNAL_CTF_BENCHMARK") == "cvebench_zero_day":
        module.main()
    else:
        run_external()
    result = json.loads(module.RESULT.read_bytes())
    if (
        result.get("schema") != "external_ctf_cell_result_v1"
        or result.get("status") != "scored"
        or result.get("arm") != "base"
    ):
        raise RuntimeError("legacy_worker_result_invalid")
    result.update(
        {
            "schema": "external_ctf_campaign_cell_result_v1",
            "arm": model["id"],
            "model_id": model["id"],
            "weights_sha256": model["weights_sha256"],
            "experiment_key": model["experiment_key"],
            "attempt": model["attempt"],
        }
    )
    module.RESULT.write_bytes(module._canonical(result) + b"\n")
except Exception as error:
    module._write(
        {
            "schema": "external_ctf_campaign_cell_result_v1",
            **module._CONTEXT,
            "arm": model["id"],
            "model_id": model["id"],
            "weights_sha256": model["weights_sha256"],
            "experiment_key": model["experiment_key"],
            "attempt": model["attempt"],
            "status": "infrastructure_invalid",
            "error_class": type(error).__name__,
        }
    )
    raise
"""


class ExternalCampaignError(RuntimeError):
    """A campaign cell is not exactly bound or is unsafe to advance."""


def _signed(value: dict[str, Any]) -> dict[str, Any]:
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    return {**unsigned, "receipt_sha256": digest(unsigned)}


def _scored_bundle(protocol: dict[str, Any], benchmark: str) -> bytes:
    """Extend the reviewed model-free runtime bundle with the shared scorer."""
    files = json.loads(lzma.decompress(tensorlake._qualification_bundle(protocol, benchmark)))  # noqa: SLF001
    paths = {
        "evals/external_ctf/opencode_scored.py": Path(__file__).with_name("opencode_scored.py"),
        "evals/external_ctf/external_proxy.py": Path(__file__).with_name("external_proxy.py"),
        "evals/external_ctf/fixed_proxy.py": Path(__file__).parents[1] / "fleet/fixed_proxy.py",
    }
    files.update(
        {relative: base64.b64encode(path.read_bytes()).decode() for relative, path in paths.items()}
    )
    return lzma.compress(canonical(files), preset=9)


def _write_once(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    raw = canonical(value) + b"\n"
    if path.exists():
        if path.is_symlink() or path.read_bytes() != raw:
            raise ExternalCampaignError("output_collision")
        return
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _write_signed_once(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    signed = _signed(value)
    _write_once(path, signed)
    return signed


def _read(path: Path, label: str, *, signed: bool = False) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ExternalCampaignError(label + "_file_invalid")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExternalCampaignError(label + "_file_invalid") from exc
    if (
        not isinstance(value, dict)
        or path.read_bytes() != raw
        or signed
        and raw != canonical(value) + b"\n"
    ):
        raise ExternalCampaignError(label + "_file_invalid")
    if signed and value.get("receipt_sha256") != digest(
        {key: item for key, item in value.items() if key != "receipt_sha256"}
    ):
        raise ExternalCampaignError(label + "_receipt_invalid")
    return value


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ExternalCampaignError(label + "_invalid")
    return value


def _file_binding(
    value: object,
    label: str,
    *,
    signed: bool = False,
    receipt_field: str = "receipt_sha256",
) -> tuple[Path, dict]:
    if not isinstance(value, dict) or set(value) != {
        "path",
        "file_sha256",
        "receipt_sha256",
    }:
        raise ExternalCampaignError(label + "_binding_invalid")
    path = Path(value["path"])
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ExternalCampaignError(label + "_binding_invalid") from exc
    if not path.is_absolute() or file_digest(raw) != _sha(
        value["file_sha256"], label + "_file_sha256"
    ):
        raise ExternalCampaignError(label + "_binding_invalid")
    loaded = _read(path, label, signed=signed)
    if loaded.get(receipt_field) != _sha(value["receipt_sha256"], label + "_receipt_sha256"):
        raise ExternalCampaignError(label + "_binding_invalid")
    return path, loaded


def _matrix(value: object) -> tuple[Path, dict[str, Any], str]:
    if not isinstance(value, dict) or set(value) != {
        "path",
        "file_sha256",
        "receipt_sha256",
    }:
        raise ExternalCampaignError("matrix_binding_invalid")
    path = Path(value["path"])
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ExternalCampaignError("matrix_binding_invalid") from exc
    matrix = _read(path, "matrix")
    unsigned = {key: item for key, item in matrix.items() if key != "sha256"}
    receipt = digest(unsigned)
    arms = matrix.get("arms")
    arm_ids = (
        [row.get("arm_id") for row in arms]
        if isinstance(arms, list) and all(isinstance(row, dict) for row in arms)
        else []
    )
    operation = matrix.get("operation")
    if (
        not path.is_absolute()
        or file_digest(raw) != _sha(value["file_sha256"], "matrix_file_sha256")
        or value["receipt_sha256"] != receipt
        or matrix.get("sha256") != receipt.removeprefix("sha256:")
        or matrix.get("schema") != "cyber_qwen38_top5_multibench_pass4_matrix_v1"
        or matrix.get("status") != "prepared_no_launch"
        or matrix.get("launchable") is not False
        or not isinstance(arms, list)
        or len(arms) != 6
        or [row.get("rank") for row in arms] != list(range(6))
        or len(set(arm_ids)) != 6
        or matrix.get("evaluation_protocol", {}).get("arms") != arm_ids
        or matrix.get("evaluation_protocol", {}).get("pass_k") != 4
        or matrix.get("evaluation_protocol", {}).get("independent_pass1_attempt_indices")
        != [0, 1, 2, 3]
        or not isinstance(operation, dict)
        or set(operation)
        != {
            "checkpoint_promotions_started",
            "routes_created_or_mutated",
            "jobs_created",
            "evaluation_cells_created",
            "fleet_confirmation_campaigns_created",
        }
        or any(item != 0 for item in operation.values())
    ):
        raise ExternalCampaignError("matrix_binding_invalid")
    return path, matrix, receipt


def _models(value: object, matrix: dict[str, Any]) -> list[dict[str, Any]]:
    required = {
        "id",
        "checkpoint_id",
        "matrix_arm_sha256",
        "weights_sha256",
        "matched_treatment_receipt_sha256",
        "serving_route_receipt_sha256",
        "live_parity_receipt_sha256",
        "served_model",
        "route_preflight",
    }
    if not isinstance(value, list) or len(value) != 6:
        raise ExternalCampaignError("six_model_bindings_required")
    matrix_arms = {row["arm_id"]: row for row in matrix["arms"]}
    result: list[dict[str, Any]] = []
    ids: set[str] = set()
    for row in value:
        if not isinstance(row, dict) or set(row) != required:
            raise ExternalCampaignError("model_binding_invalid")
        model_id = row["id"]
        if (
            not isinstance(model_id, str)
            or _ID.fullmatch(model_id) is None
            or model_id in ids
            or model_id not in matrix_arms
            or row["checkpoint_id"] != matrix_arms[model_id].get("artifact_id")
            or row["matrix_arm_sha256"] != digest(matrix_arms[model_id])
            or not isinstance(row["served_model"], str)
            or not row["served_model"]
        ):
            raise ExternalCampaignError("model_binding_invalid")
        ids.add(model_id)
        for field in (
            "matrix_arm_sha256",
            "weights_sha256",
            "matched_treatment_receipt_sha256",
            "serving_route_receipt_sha256",
            "live_parity_receipt_sha256",
        ):
            _sha(row[field], field)
        preflight_path, preflight = _file_binding(
            row["route_preflight"], "route_preflight", signed=True
        )
        if (
            preflight.get("status") != "accepted"
            or preflight.get("served_model") != row["served_model"]
            or preflight.get("weights_sha256") != row["weights_sha256"]
            or preflight.get("serving_route_receipt_sha256") != row["serving_route_receipt_sha256"]
            or preflight.get("live_parity_receipt_sha256") != row["live_parity_receipt_sha256"]
        ):
            raise ExternalCampaignError("route_preflight_binding_invalid")
        result.append({**row, "route_preflight_path": str(preflight_path)})
    if (
        ids != set(matrix_arms)
        or len({row["matched_treatment_receipt_sha256"] for row in result}) != 1
    ):
        raise ExternalCampaignError("model_treatment_mismatch")
    return result


def load_bindings(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    value = _read(path, "campaign_bindings", signed=True)
    expected = {
        "schema",
        "matrix",
        "budgets_sha256",
        "protocol",
        "web_retry_execution",
        "qualification_packet",
        "qualification_summaries",
        "scored_adapter_qualification",
        "harness_receipt_sha256",
        "scoring_protocol_sha256",
        "models",
        "receipt_sha256",
    }
    if value.get("schema") != SCHEMA or set(value) != expected:
        raise ExternalCampaignError("campaign_bindings_invalid")
    _sha(value["budgets_sha256"], "budgets_sha256")
    matrix_path, matrix, matrix_sha256 = _matrix(value["matrix"])
    protocol_path, protocol_file = _file_binding(
        value["protocol"], "protocol", signed=False, receipt_field="protocol_sha256"
    )
    protocol = load_protocol(protocol_path)
    if protocol_file.get("protocol_sha256") != value["protocol"]["receipt_sha256"]:
        raise ExternalCampaignError("protocol_binding_invalid")
    retry_path, _retry = _file_binding(
        value["web_retry_execution"], "web_retry_execution", signed=True
    )
    qualification_path, _qualification = _file_binding(
        value["qualification_packet"], "qualification_packet", signed=True
    )
    adapter = None
    if value["scored_adapter_qualification"] is not None:
        _adapter_path, adapter = _file_binding(
            value["scored_adapter_qualification"],
            "scored_adapter_qualification",
            signed=True,
        )
    harness_ids = {
        protocol["benchmarks"][benchmark]["harness"]["image_id"]
        for benchmark in ("nyu_ctf_web_test", "cybench_web")
    }
    if adapter is not None and (
        set(adapter)
        != {
            "schema",
            "protocol_sha256",
            "adapter_source_sha256",
            "status",
            "platform",
            "proxy_image",
            "harness_image_id",
            "provider_calls",
            "model_requests",
            "provider_credential_location",
            "agent_provider_credential_present",
            "challenge_provider_credential_present",
            "agent_docker_socket_present",
            "cleanup_verified",
            "contains_credentials_prompts_flags_solutions_traces_or_scores",
            "receipt_sha256",
        }
        or adapter.get("schema") != "external_ctf_scored_adapter_qualification_v1"
        or adapter.get("protocol_sha256") != protocol["protocol_sha256"]
        or adapter.get("adapter_source_sha256") != opencode_scored.source_sha256()
        or adapter.get("status") != "qualified"
        or adapter.get("platform") != "linux/amd64"
        or adapter.get("proxy_image") != opencode_scored.PROXY_IMAGE
        or harness_ids != {adapter.get("harness_image_id")}
        or adapter.get("provider_calls") != 0
        or adapter.get("model_requests") != 0
        or adapter.get("provider_credential_location") != "fixed_proxy_only"
        or adapter.get("agent_provider_credential_present") is not False
        or adapter.get("challenge_provider_credential_present") is not False
        or adapter.get("agent_docker_socket_present") is not False
        or adapter.get("cleanup_verified") is not True
        or adapter.get("contains_credentials_prompts_flags_solutions_traces_or_scores") is not False
    ):
        raise ExternalCampaignError("scored_adapter_qualification_invalid")
    summaries = value["qualification_summaries"]
    if not isinstance(summaries, dict) or set(summaries) != set(BENCHMARKS):
        raise ExternalCampaignError("qualification_summaries_invalid")
    loaded_summaries = {}
    for benchmark in BENCHMARKS:
        _summary_path, summary = _file_binding(
            summaries[benchmark], benchmark + "_qualification_summary", signed=True
        )
        if (
            summary.get("schema") != "external_ctf_qualification_summary_v1"
            or summary.get("benchmark") != benchmark
            or summary.get("protocol_sha256") != protocol["protocol_sha256"]
            or summary.get("qualification_contract_sha256")
            != protocol["benchmarks"][benchmark]["runtime_qualification"]["contract_sha256"]
            or summary.get("qualification_packet_receipt_sha256")
            != value["qualification_packet"]["receipt_sha256"]
            or summary.get("score_reads") != 0
            or summary.get("model_requests") != 0
            or not isinstance(summary.get("rows"), list)
        ):
            raise ExternalCampaignError("qualification_summary_binding_invalid")
        loaded_summaries[benchmark] = summary
    if not isinstance(value["harness_receipt_sha256"], dict) or set(
        value["harness_receipt_sha256"]
    ) != set(BENCHMARKS):
        raise ExternalCampaignError("harness_bindings_invalid")
    if not isinstance(value["scoring_protocol_sha256"], dict) or set(
        value["scoring_protocol_sha256"]
    ) != set(BENCHMARKS):
        raise ExternalCampaignError("scoring_bindings_invalid")
    for field in ("harness_receipt_sha256", "scoring_protocol_sha256"):
        for benchmark, item in value[field].items():
            _sha(item, field + "_" + benchmark)
    models = _models(value["models"], matrix)
    return {
        **value,
        "matrix_path": matrix_path,
        "matrix_loaded": matrix,
        "matrix_sha256": matrix_sha256,
        "protocol_path": protocol_path,
        "web_retry_execution_path": retry_path,
        "qualification_packet_path": qualification_path,
        "scored_adapter_qualification_loaded": adapter,
        "qualification_summaries_loaded": loaded_summaries,
        "models_loaded": models,
    }, protocol


def _target_identity(protocol: dict[str, Any], benchmark: str, index: int) -> dict[str, Any]:
    row = protocol["benchmarks"][benchmark]
    return {
        "benchmark": benchmark,
        "official_task_index": index,
        "task_id_sha256": file_digest(row["task_ids"][index].encode()),
        "source_commit": row["source"]["commit"],
        "qualification_contract_sha256": row["runtime_qualification"]["contract_sha256"],
        "grader_sha256": row["bindings"]["grader_sha256"],
    }


def _harness_name(benchmark: str) -> str:
    return "cvebench-native-inspect" if benchmark == "cvebench_zero_day" else "opencode-native-ctf"


def controller_config(bindings_path: Path) -> dict[str, Any]:
    """Render the exact generic-controller config from reviewed private bindings."""

    bindings, protocol = load_bindings(bindings_path)
    source_sha256 = file_digest(Path(__file__).read_bytes())

    def commands(phase: str) -> dict[str, list[str]]:
        return {
            "preview": [
                "uv",
                "run",
                "python",
                "-m",
                "evals.external_ctf.campaign_adapter",
                "--bindings",
                str(bindings_path.resolve()),
                "preview",
                phase,
                "{packet}",
                "{receipt}",
                *(["--terminal", "{terminal_receipt}"] if phase == "score" else []),
            ],
            "ready": [
                "uv",
                "run",
                "python",
                "-m",
                "evals.external_ctf.campaign_adapter",
                "--bindings",
                str(bindings_path.resolve()),
                "ready",
                phase,
                "{packet}",
                "{receipt}",
                "--previous",
                "{preview_receipt}",
            ],
            "launch": [
                "uv",
                "run",
                "python",
                "-m",
                "evals.external_ctf.campaign_adapter",
                "--bindings",
                str(bindings_path.resolve()),
                "launch",
                phase,
                "{packet}",
                "{receipt}",
                "--previous",
                "{preview_receipt}",
                "--readiness",
                "{readiness_receipt}",
            ],
            "observe": [
                "uv",
                "run",
                "python",
                "-m",
                "evals.external_ctf.campaign_adapter",
                "--bindings",
                str(bindings_path.resolve()),
                "observe",
                phase,
                "{packet}",
                "{receipt}",
                "--previous",
                "{launch_receipt}",
            ],
        }

    benchmarks = []
    for benchmark in BENCHMARKS:
        source = protocol["benchmarks"][benchmark]
        targets = []
        for index in source["runtime_qualification"]["task_indices"]:
            identity = _target_identity(protocol, benchmark, index)
            targets.append(
                {
                    "id": f"{PREFIXES[benchmark]}-t{index:02d}",
                    "identity_receipt_sha256": digest(identity),
                    "canary": index == CANARIES[benchmark],
                }
            )
        benchmarks.append(
            {
                "id": benchmark,
                "task_set_sha256": source["task_ids_sha256"],
                "harness": {
                    "name": _harness_name(benchmark),
                    "identity_receipt_sha256": bindings["harness_receipt_sha256"][benchmark],
                },
                "scoring_protocol_sha256": bindings["scoring_protocol_sha256"][benchmark],
                "sampling": {"temperature": 1.0, "top_p": 0.95, "attempt_seeds": [None] * 4},
                "targets": targets,
                "rollout_driver": {
                    "provider": "tensorlake",
                    "submission": "none",
                    "source_sha256": source_sha256,
                    "timeout_seconds": 600,
                    "commands": commands("rollout"),
                },
                "score_driver": {
                    "provider": "local",
                    "submission": "none",
                    "source_sha256": source_sha256,
                    "timeout_seconds": 600,
                    "commands": commands("score"),
                },
            }
        )
    return {
        "schema": "cyber_eval_campaign_v1",
        "campaign_id": "q38-six-arm-external-ctf-pass4-v1",
        "scheduler": {"max_launches_per_step": 4, "serial_canaries": True},
        "pass_k": 4,
        "budgets_sha256": bindings["budgets_sha256"],
        "matrix_sha256": bindings["matrix_sha256"],
        "models": [
            {
                key: row[key]
                for key in (
                    "id",
                    "checkpoint_id",
                    "weights_sha256",
                    "matched_treatment_receipt_sha256",
                    "serving_route_receipt_sha256",
                    "live_parity_receipt_sha256",
                )
            }
            for row in bindings["models_loaded"]
        ],
        "benchmarks": benchmarks,
    }


def remote_name(packet: dict[str, Any]) -> str:
    identity = packet["identity"]
    benchmark = identity["benchmark"]["id"]
    target = identity["target"]["id"]
    model = identity["model"]["id"]
    attempt = identity["attempt"]
    key = packet["experiment_key"].removeprefix("sha256:")[:12]
    slug = re.sub(r"[^a-z0-9-]", "-", model.lower()).strip("-")[:24]
    return f"extctf-{PREFIXES[benchmark]}-{target}-{slug}-r{attempt - 1:02d}-{key}"


def _packet(
    packet_path: Path, bindings: dict[str, Any], protocol: dict[str, Any]
) -> tuple[dict, dict]:
    packet = _read(packet_path, "campaign_packet")
    identity = packet.get("identity")
    drivers = packet.get("drivers")
    source_sha256 = file_digest(Path(__file__).read_bytes())
    if (
        not isinstance(identity, dict)
        or not isinstance(drivers, dict)
        or any(
            not isinstance(drivers.get(phase), dict)
            or drivers[phase].get("source_sha256") != source_sha256
            for phase in ("rollout", "score")
        )
    ):
        raise ExternalCampaignError("campaign_packet_invalid")
    benchmark = identity.get("benchmark", {}).get("id")
    target = identity.get("target", {}).get("id")
    model_id = identity.get("model", {}).get("id")
    attempt = identity.get("attempt")
    if (
        benchmark not in BENCHMARKS
        or not isinstance(target, str)
        or re.fullmatch(PREFIXES[benchmark] + r"-t\d{2}", target) is None
        or type(attempt) is not int
        or attempt not in {1, 2, 3, 4}
        or identity.get("seed") is not None
        or identity.get("sampling") != {"temperature": 1.0, "top_p": 0.95}
        or packet.get("matrix_sha256") != bindings["matrix_sha256"]
        or identity.get("budgets_sha256") != bindings["budgets_sha256"]
    ):
        raise ExternalCampaignError("campaign_packet_invalid")
    index = int(target[-2:])
    source = protocol["benchmarks"][benchmark]
    if index not in source["runtime_qualification"]["task_indices"]:
        raise ExternalCampaignError("campaign_target_not_launchable")
    target_identity = _target_identity(protocol, benchmark, index)
    expected_benchmark = {
        "id": benchmark,
        "task_set_sha256": source["task_ids_sha256"],
        "harness": {
            "name": _harness_name(benchmark),
            "identity_receipt_sha256": bindings["harness_receipt_sha256"][benchmark],
        },
        "scoring_protocol_sha256": bindings["scoring_protocol_sha256"][benchmark],
    }
    if (
        packet.get("experiment_key") != digest(identity)
        or identity.get("benchmark") != expected_benchmark
        or packet.get("canary") is not (index == CANARIES[benchmark])
        or identity["target"].get("identity_receipt_sha256") != digest(target_identity)
    ):
        raise ExternalCampaignError("campaign_target_binding_mismatch")
    model = next((row for row in bindings["models_loaded"] if row["id"] == model_id), None)
    if (
        model is None
        or any(
            identity["model"].get(field) != model[field]
            for field in (
                "id",
                "checkpoint_id",
                "weights_sha256",
                "matched_treatment_receipt_sha256",
            )
        )
        or packet.get("serving_evidence")
        != {
            "serving_route_receipt_sha256": model["serving_route_receipt_sha256"],
            "live_parity_receipt_sha256": model["live_parity_receipt_sha256"],
        }
    ):
        raise ExternalCampaignError("campaign_model_binding_mismatch")
    return packet, {
        "benchmark": benchmark,
        "task_index": index,
        "task_id": source["task_ids"][index],
        "model": model,
        "attempt": attempt,
        "experiment_key": packet["experiment_key"],
        "name": remote_name(packet),
        "canary": packet.get("canary") is True,
    }


def _qualification_ready(bindings: dict[str, Any], protocol: dict[str, Any], cell: dict) -> bool:
    benchmark = cell["benchmark"]
    summaries = bindings.get("qualification_summaries_loaded")
    if not isinstance(summaries, dict) or benchmark not in summaries:
        return False
    rows = summaries[benchmark].get("rows")
    if not isinstance(rows, list):
        return False
    row = next(
        (
            item
            for item in rows
            if isinstance(item, dict) and item.get("task_index") == cell["task_index"]
        ),
        None,
    )
    return (
        isinstance(row, dict)
        and row.get("task_id") == cell["task_id"]
        and row.get("outcome") == "runtime_preflight_passed"
        and _DIGEST.fullmatch(str(row.get("terminal_receipt_sha256"))) is not None
        and _DIGEST.fullmatch(str(row.get("release_receipt_sha256"))) is not None
        and (
            benchmark == "cvebench_zero_day"
            or bindings.get("scored_adapter_qualification_loaded") is not None
        )
    )


def _all_names(packet_path: Path) -> set[str]:
    plan = _read(packet_path.parents[2] / "plan.json", "campaign_plan")
    return {remote_name(row) for row in plan.get("targets", [])}


def _provider_preflight(
    *,
    packet_path: Path,
    bindings: dict[str, Any],
    protocol: dict[str, Any],
    cell: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    authority = tensorlake.capacity_authority(
        protocol, bindings["web_retry_execution_path"], require_live_owner=True
    )
    qualification, _qualification_state = load_execution_packet(
        bindings["qualification_packet_path"],
        protocol=protocol,
        authority=authority,
        require_current_source=False,
    )
    if qualification.get("receipt_sha256") != bindings["qualification_packet"]["receipt_sha256"]:
        raise ExternalCampaignError("qualification_packet_binding_invalid")
    rows = tensorlake._client().inventory()  # noqa: SLF001
    names = _all_names(packet_path)
    owned = set(authority["owned_names"]) | names
    try:
        active = replica_set.shared_project_capacity_count(rows, owned, authority["state"])
        pending = replica_set._shared_capacity_reservations(  # noqa: SLF001
            authority["state"], owned
        )
    except replica_set.CollectionReplicaSetError as exc:
        raise ExternalCampaignError(str(exc)) from exc
    campaign_active_names = {
        str(row.get("name", ""))
        for row in rows
        if str(row.get("name", "")) in names and row.get("status") != "terminated"
    } | (set(pending) & names)
    output = packet_path.parent / "external-provider"
    preflight = {
        "shared_capacity_receipt_sha256": authority["capacity_successor_state_receipt_sha256"],
        "provider_inventory_receipt_sha256": file_digest(canonical(rows)),
        "create_claim_absent": not (output / "create-claim.json").exists(),
        "start_claim_absent": not (output / "start-claim.json").exists(),
        "remote_name_absent": not any(row.get("name") == cell["name"] for row in rows),
        "output_root_absent": not output.exists(),
        "checked_at_epoch": time.time_ns() / 1_000_000_000,
    }
    return preflight, {
        **authority,
        "owned_names": owned,
        "campaign_names": names,
        "active": active,
        "campaign_active": len(campaign_active_names),
        "output": output,
    }


def _capacity_ready(context: dict[str, Any], cell: dict[str, Any]) -> bool:
    return context["active"] < replica_set.PROJECT_ACTIVE_SANDBOX_LIMIT and context[
        "campaign_active"
    ] < (1 if cell["canary"] else 4)


def _campaign_result(raw: bytes, *, protocol: dict[str, Any], cell: dict[str, Any]) -> dict:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExternalCampaignError("campaign_result_invalid") from exc
    common = {
        "schema",
        "protocol_sha256",
        "benchmark",
        "task_id",
        "arm",
        "model_id",
        "weights_sha256",
        "experiment_key",
        "attempt",
        "status",
    }
    if (
        not isinstance(value, dict)
        or raw != canonical(value) + b"\n"
        or value.get("schema") != "external_ctf_campaign_cell_result_v1"
        or value.get("protocol_sha256") != protocol["protocol_sha256"]
        or value.get("benchmark") != cell["benchmark"]
        or value.get("task_id") != cell["task_id"]
        or value.get("arm") != cell["model"]["id"]
        or value.get("model_id") != cell["model"]["id"]
        or value.get("weights_sha256") != cell["model"]["weights_sha256"]
        or value.get("experiment_key") != cell["experiment_key"]
        or value.get("attempt") != cell["attempt"]
        or value.get("status") not in {"scored", "infrastructure_invalid"}
    ):
        raise ExternalCampaignError("campaign_result_invalid")
    if value["status"] == "scored":
        if set(value) != common | {"score", "grader"} or value.get("score") not in {0, 1}:
            raise ExternalCampaignError("campaign_result_invalid")
    elif set(value) != common | {"error_class"} or not isinstance(value.get("error_class"), str):
        raise ExternalCampaignError("campaign_result_invalid")
    return value


def _qualification_terminal(
    *, protocol: dict[str, Any], bindings: dict[str, Any], cell: dict, authority: dict
) -> bytes:
    packet, state = load_execution_packet(
        bindings["qualification_packet_path"],
        protocol=protocol,
        authority=authority,
        require_current_source=False,
    )
    name = tensorlake.cell_name(cell["benchmark"], cell["task_index"], "qualification")
    path = state / f"{name}.terminal.json"
    raw = path.read_bytes()
    terminal = _read(path, "qualification_terminal", signed=True)
    summary = bindings["qualification_summaries_loaded"][cell["benchmark"]]
    row = next(
        (item for item in summary["rows"] if item.get("task_index") == cell["task_index"]),
        None,
    )
    if (
        not isinstance(row, dict)
        or row.get("outcome") != "runtime_preflight_passed"
        or row.get("terminal_receipt_sha256") != terminal.get("receipt_sha256")
        or terminal.get("execution_packet_receipt_sha256") != packet["receipt_sha256"]
        or terminal.get("benchmark") != cell["benchmark"]
        or terminal.get("task_id") != cell["task_id"]
        or terminal.get("arm") != "qualification"
    ):
        raise ExternalCampaignError("campaign_qualification_binding_invalid")
    return raw


def _launch_cell(
    *,
    protocol: dict[str, Any],
    bindings: dict[str, Any],
    cell: dict[str, Any],
    authority: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    if not _qualification_ready(bindings, protocol, cell):
        raise ExternalCampaignError("scored_adapter_not_qualified")
    if output.exists() or output.is_symlink():
        raise ExternalCampaignError("campaign_output_root_exists")
    spec = tensorlake._sandbox_spec(cell["name"], authority["snapshot_id"])  # noqa: SLF001
    client = tensorlake._client()  # noqa: SLF001
    with replica_set._shared_tensorlake_create_lock(authority["state"]):  # noqa: SLF001
        rows = client.inventory()
        if any(row.get("name") == cell["name"] for row in rows):
            raise ExternalCampaignError("duplicate_sandbox_name")
        try:
            pending = replica_set._shared_capacity_reservations(  # noqa: SLF001
                authority["state"], authority["owned_names"]
            )
        except replica_set.CollectionReplicaSetError as exc:
            raise ExternalCampaignError(str(exc)) from exc
        if cell["name"] in pending:
            raise ExternalCampaignError("campaign_capacity_reservation_exists")
        campaign_names = authority.get("campaign_names")
        if not isinstance(campaign_names, set) or cell["name"] not in campaign_names:
            raise ExternalCampaignError("campaign_capacity_names_invalid")
        campaign_active = {
            str(row.get("name", ""))
            for row in rows
            if str(row.get("name", "")) in campaign_names and row.get("status") != "terminated"
        } | (set(pending) & campaign_names)
        if len(campaign_active) >= (1 if cell["canary"] else 4):
            raise ExternalCampaignError("campaign_parallelism_limit")
        output.mkdir(mode=0o700, parents=False)
        try:
            reservation, active = replica_set.reserve_shared_capacity_slot(
                state=authority["state"],
                rows=rows,
                owned_names=authority["owned_names"],
                sandbox_name=cell["name"],
                creator="external_ctf",
                authority_receipt_sha256=authority["capacity_successor_receipt_sha256"],
                spec_sha256=file_digest(canonical(spec)),
            )
        except replica_set.CollectionReplicaSetError as exc:
            output.rmdir()
            raise ExternalCampaignError(str(exc)) from exc
        claim = _write_signed_once(
            output / "create-claim.json",
            {
                "schema": "external_ctf_campaign_create_claim_v1",
                "experiment_key": cell["experiment_key"],
                "name": cell["name"],
                "spec_sha256": file_digest(canonical(spec)),
                "active_before_create": active,
                "capacity_reservation_receipt_sha256": reservation["receipt_sha256"],
            },
        )
        response = client.request("POST", tensorlake.API + "/sandboxes", spec)
    sandbox_id = response.get("sandbox_id")
    if not isinstance(sandbox_id, str) or re.fullmatch(r"[A-Za-z0-9._:-]+", sandbox_id) is None:
        raise ExternalCampaignError("sandbox_create_response_invalid")
    created = _write_signed_once(
        output / "created.json",
        {
            "schema": "external_ctf_campaign_created_v1",
            "experiment_key": cell["experiment_key"],
            "name": cell["name"],
            "sandbox_id": sandbox_id,
            "create_claim_receipt_sha256": claim["receipt_sha256"],
            "capacity_reservation_receipt_sha256": reservation["receipt_sha256"],
        },
    )
    detail = client.request("GET", tensorlake.API + "/sandboxes/" + sandbox_id)
    origin = detail.get("sandbox_url")
    if detail.get("status") != "running" or not isinstance(origin, str):
        raise ExternalCampaignError("sandbox_not_runnable")
    qualification = _qualification_terminal(
        protocol=protocol, bindings=bindings, cell=cell, authority=authority
    )
    worker = tensorlake.WORKER.read_bytes()
    bundle = (
        None
        if cell["benchmark"] == "cvebench_zero_day"
        else _scored_bundle(protocol, cell["benchmark"])
    )
    model = canonical(
        {
            "id": cell["model"]["id"],
            "served_model": cell["model"]["served_model"],
            "weights_sha256": cell["model"]["weights_sha256"],
            "experiment_key": cell["experiment_key"],
            "attempt": cell["attempt"],
        }
    )
    fleet_key = os.environ.get("FLEET_API_KEY", "")
    if not fleet_key or fleet_key.strip() != fleet_key or any(ch.isspace() for ch in fleet_key):
        raise ExternalCampaignError("fleet_credential_missing_or_invalid")
    environment = {
        "EXTERNAL_CTF_PROTOCOL_B64": base64.b64encode(
            bindings["protocol_path"].read_bytes()
        ).decode(),
        "EXTERNAL_CTF_BENCHMARK": cell["benchmark"],
        "EXTERNAL_CTF_TASK_ID": cell["task_id"],
        "EXTERNAL_CTF_ARM": "base",
        "EXTERNAL_CTF_MODE": "scored",
        "EXTERNAL_CTF_QUALIFICATION_B64": base64.b64encode(qualification).decode(),
        "EXTERNAL_CTF_QUALIFICATION_SHA256": file_digest(qualification),
        "CAMPAIGN_MODEL_B64": base64.b64encode(model).decode(),
        "CAMPAIGN_MODEL_SHA256": file_digest(model),
        "WORKER_B64": base64.b64encode(worker).decode(),
        "WORKER_SHA256": file_digest(worker),
        "FLEET_API_KEY": fleet_key,
    }
    if bundle is not None:
        environment.update(
            {
                "CAMPAIGN_BUNDLE_B64": base64.b64encode(bundle).decode(),
                "CAMPAIGN_BUNDLE_SHA256": file_digest(bundle),
            }
        )
    process_spec = {
        "command": "/usr/bin/python3",
        "args": ["-c", _CAMPAIGN_WORKER_BOOTSTRAP],
        "user": "root",
        "env": environment,
        "stdin_mode": "closed",
        "stdout_mode": "discard",
        "stderr_mode": "discard",
    }
    _write_signed_once(
        output / "start-claim.json",
        {
            "schema": "external_ctf_campaign_start_claim_v1",
            "experiment_key": cell["experiment_key"],
            "sandbox_id": sandbox_id,
            "created_receipt_sha256": created["receipt_sha256"],
            "worker_sha256": file_digest(worker),
            "bundle_sha256": None if bundle is None else file_digest(bundle),
            "model_sha256": file_digest(model),
        },
    )
    process = client.request("POST", origin + "/api/v1/processes", process_spec)
    pid = process.get("pid")
    if type(pid) is not int or pid < 1:
        raise ExternalCampaignError("process_create_response_invalid")
    launched = _write_signed_once(
        output / "launched.json",
        {
            "schema": "external_ctf_campaign_launched_v1",
            "experiment_key": cell["experiment_key"],
            "name": cell["name"],
            "sandbox_id": sandbox_id,
            "pid": pid,
            "capacity_reservation_receipt_sha256": reservation["receipt_sha256"],
        },
    )
    return {"remote_id": f"{sandbox_id}:{pid}", "receipt_sha256": launched["receipt_sha256"]}


def _observe_cell(
    *,
    protocol: dict[str, Any],
    bindings: dict[str, Any],
    cell: dict[str, Any],
    output: Path,
    owned_names: set[str],
) -> dict[str, Any]:
    launched = _read(output / "launched.json", "campaign_launched", signed=True)
    if (
        launched.get("experiment_key") != cell["experiment_key"]
        or launched.get("name") != cell["name"]
    ):
        raise ExternalCampaignError("campaign_launch_binding_mismatch")
    client = tensorlake._client()  # noqa: SLF001
    terminal_path = output / "terminal.json"
    if terminal_path.exists():
        terminal = _read(terminal_path, "campaign_terminal", signed=True)
    else:
        detail = client.request("GET", tensorlake.API + "/sandboxes/" + launched["sandbox_id"])
        origin = detail.get("sandbox_url")
        if not isinstance(origin, str):
            raise ExternalCampaignError("sandbox_process_status_ambiguous")
        rows = client.request("GET", origin + "/api/v1/processes").get("processes")
        matches = (
            [row for row in rows if isinstance(row, dict) and row.get("pid") == launched["pid"]]
            if isinstance(rows, list)
            else []
        )
        if len(matches) != 1:
            raise ExternalCampaignError("sandbox_process_status_ambiguous")
        if matches[0].get("status") == "running":
            return {"status": "running"}
        if matches[0].get("status") != "exited":
            raise ExternalCampaignError("sandbox_process_status_ambiguous")
        url = (
            origin
            + "/api/v1/files?"
            + urllib.parse.urlencode({"path": "/workspace/external-ctf-result.json"})
        )
        result = None
        result_raw = None
        for attempt in range(_RESULT_RECONCILE_ATTEMPTS):
            try:
                candidate = client.request("GET", url, raw=True, max_response_bytes=1048576)
                if isinstance(candidate, bytes):
                    result_raw = candidate
                    result = _campaign_result(candidate, protocol=protocol, cell=cell)
                    break
            except Exception:
                pass
            if attempt + 1 < _RESULT_RECONCILE_ATTEMPTS:
                time.sleep(_RESULT_RECONCILE_DELAY_SECONDS)
        accepted = result is not None and result.get("status") == "scored"
        terminal = _write_signed_once(
            terminal_path,
            {
                "schema": "external_ctf_campaign_terminal_v1",
                "experiment_key": cell["experiment_key"],
                "name": cell["name"],
                "sandbox_id": launched["sandbox_id"],
                "pid": launched["pid"],
                "benchmark": cell["benchmark"],
                "task_index": cell["task_index"],
                "model_id": cell["model"]["id"],
                "attempt": cell["attempt"],
                "result_file_sha256": None if result_raw is None else file_digest(result_raw),
                "result": result,
                "outcome": "accepted_model_outcome" if accepted else "infrastructure_invalid",
                "infrastructure_error_class": None if accepted else "worker_or_result_invalid",
            },
        )
    release_path = output / "released.json"
    if release_path.exists():
        release = _read(release_path, "campaign_release", signed=True)
        if (
            release.get("status") != "terminated"
            or release.get("terminal_receipt_sha256") != terminal["receipt_sha256"]
        ):
            raise ExternalCampaignError("campaign_release_binding_invalid")
        return terminal
    client.request("DELETE", tensorlake.API + "/sandboxes/" + launched["sandbox_id"], raw=True)
    for _ in range(60):
        detail = client.request("GET", tensorlake.API + "/sandboxes/" + launched["sandbox_id"])
        if detail.get("status") == "terminated":
            release = _write_signed_once(
                release_path,
                {
                    "schema": "external_ctf_campaign_release_v1",
                    "experiment_key": cell["experiment_key"],
                    "sandbox_id": launched["sandbox_id"],
                    "status": "terminated",
                    "terminal_receipt_sha256": terminal["receipt_sha256"],
                },
            )
            authority = tensorlake.capacity_authority(
                protocol, bindings["web_retry_execution_path"], require_live_owner=False
            )
            try:
                replica_set.release_shared_capacity_slot(
                    state=authority["state"],
                    owned_names=set(authority["owned_names"]) | owned_names,
                    sandbox_name=cell["name"],
                    reservation_receipt_sha256=launched["capacity_reservation_receipt_sha256"],
                    provider_state="terminated",
                    terminal_receipt_sha256=release["receipt_sha256"],
                )
            except replica_set.CollectionReplicaSetError as exc:
                raise ExternalCampaignError(str(exc)) from exc
            return terminal
        time.sleep(2)
    raise ExternalCampaignError("sandbox_release_not_confirmed")


def _common(packet: dict, phase: str, action: str, provider: str) -> dict[str, Any]:
    return {
        "schema": RECEIPT_SCHEMA,
        "experiment_key": packet["experiment_key"],
        "phase": phase,
        "action": action,
        "provider": provider,
    }


def _previous(path: Path | None, label: str) -> dict[str, Any]:
    if path is None:
        raise ExternalCampaignError(label + "_required")
    return _read(path, label, signed=True)


def run_action(
    *,
    bindings_path: Path,
    action: str,
    phase: str,
    packet_path: Path,
    receipt_path: Path,
    previous_path: Path | None = None,
    readiness_path: Path | None = None,
    terminal_path: Path | None = None,
) -> dict[str, Any]:
    bindings, protocol = load_bindings(bindings_path)
    packet, cell = _packet(packet_path, bindings, protocol)
    provider = "tensorlake" if phase == "rollout" else "local"
    value = _common(packet, phase, action, provider)
    if phase == "score":
        collection = (
            _previous(terminal_path, "collection_terminal")
            if action == "preview"
            else _previous(previous_path, "previous")
        )
        binding = (
            collection.get("receipt_sha256")
            if action == "preview"
            else collection.get("collection_terminal_receipt_sha256")
        )
        value["collection_terminal_receipt_sha256"] = binding
    if action == "preview":
        value["status"] = "accepted"
        if phase == "rollout":
            gate, _context = _provider_preflight(
                packet_path=packet_path, bindings=bindings, protocol=protocol, cell=cell
            )
            value.update({"remote_name": cell["name"], "provider_preflight": gate})
    elif action == "ready":
        previous = _previous(previous_path, "preview")
        value["preview_receipt_sha256"] = previous["receipt_sha256"]
        if phase == "rollout":
            gate, context = _provider_preflight(
                packet_path=packet_path, bindings=bindings, protocol=protocol, cell=cell
            )
            value.update({"remote_name": cell["name"], "provider_preflight": gate})
            dependency_ready = _qualification_ready(bindings, protocol, cell)
            capacity_ready = _capacity_ready(context, cell)
            if dependency_ready and capacity_ready:
                value["status"] = "ready"
            else:
                value.update(
                    {
                        "status": "deferred_not_ready",
                        "defer_reason_code": (
                            "dependency_not_ready"
                            if not dependency_ready
                            else "capacity_unavailable"
                        ),
                    }
                )
        else:
            value["status"] = "ready"
    elif action == "launch":
        previous = _previous(previous_path, "preview")
        readiness = _previous(readiness_path, "readiness")
        if (
            readiness.get("status") != "ready"
            or readiness.get("phase") != phase
            or readiness.get("experiment_key") != packet["experiment_key"]
            or readiness.get("preview_receipt_sha256") != previous["receipt_sha256"]
        ):
            raise ExternalCampaignError("readiness_binding_invalid")
        value.update(
            {
                "status": "created",
                "preview_receipt_sha256": previous["receipt_sha256"],
                "readiness_receipt_path": str(readiness_path),
                "readiness_receipt_sha256": readiness["receipt_sha256"],
            }
        )
        if phase == "rollout":
            gate, context = _provider_preflight(
                packet_path=packet_path, bindings=bindings, protocol=protocol, cell=cell
            )
            if (
                not all(
                    gate[key]
                    for key in (
                        "create_claim_absent",
                        "start_claim_absent",
                        "remote_name_absent",
                        "output_root_absent",
                    )
                )
                or not _qualification_ready(bindings, protocol, cell)
                or not _capacity_ready(context, cell)
            ):
                raise ExternalCampaignError("cell_not_ready_for_create")
            remote = _launch_cell(
                protocol=protocol,
                bindings=bindings,
                cell=cell,
                authority=context,
                output=context["output"],
            )
            value.update(
                {
                    "remote_name": cell["name"],
                    "remote_id": remote["remote_id"],
                    "provider_preflight": gate,
                }
            )
        else:
            output = packet_path.parent / "external-provider"
            terminal = _read(output / "terminal.json", "external_terminal", signed=True)
            marker = _signed(
                {
                    "schema": "external_ctf_native_score_claim_v1",
                    "experiment_key": packet["experiment_key"],
                    "terminal_receipt_sha256": terminal["receipt_sha256"],
                }
            )
            _write_once(output / "score-claim.json", marker)
            value["remote_id"] = "native:" + terminal["receipt_sha256"]
    elif action == "observe":
        launch = _previous(previous_path, "launch")
        value.update(
            {
                "remote_id": launch["remote_id"],
                "launch_receipt_sha256": launch["receipt_sha256"],
            }
        )
        output = packet_path.parent / "external-provider"
        if phase == "rollout":
            observed = _observe_cell(
                protocol=protocol,
                bindings=bindings,
                cell=cell,
                output=output,
                owned_names=_all_names(packet_path),
            )
            if observed["status"] == "running":
                value["status"] = "running"
            else:
                value["status"] = (
                    "accepted"
                    if observed["outcome"] == "accepted_model_outcome"
                    else "infrastructure_invalid"
                )
                value["terminal_evidence_sha256"] = observed["receipt_sha256"]
        else:
            terminal = _read(output / "terminal.json", "external_terminal", signed=True)
            value.update(
                {"status": "accepted", "terminal_evidence_sha256": terminal["receipt_sha256"]}
            )
    else:
        raise ExternalCampaignError("action_invalid")
    signed = _signed(value)
    _write_once(receipt_path, signed)
    return signed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bindings", type=Path, required=True)
    sub = parser.add_subparsers(dest="action", required=True)
    render = sub.add_parser("render")
    render.add_argument("--output", type=Path, required=True)
    for action in ("preview", "ready", "launch", "observe"):
        command = sub.add_parser(action)
        command.add_argument("phase", choices=("rollout", "score"))
        command.add_argument("packet", type=Path)
        command.add_argument("receipt", type=Path)
        command.add_argument("--previous", type=Path)
        command.add_argument("--readiness", type=Path)
        command.add_argument("--terminal", type=Path)
    args = parser.parse_args()
    if args.action == "render":
        value = controller_config(args.bindings)
        _write_once(args.output, value)
    else:
        value = run_action(
            bindings_path=args.bindings,
            action=args.action,
            phase=args.phase,
            packet_path=args.packet,
            receipt_path=args.receipt,
            previous_path=args.previous,
            readiness_path=args.readiness,
            terminal_path=args.terminal,
        )
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
