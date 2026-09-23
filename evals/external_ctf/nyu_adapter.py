"""Pure contract for matched NYU CTF Bench web evaluations.

This module deliberately does not create sandboxes or call a model.  It binds
the pinned benchmark bytes, keeps the hidden flag outside the agent contract,
validates a model-free runtime qualification, and seals one private result.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import stat
import subprocess
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from .protocol import DEFAULT_PROTOCOL, load_protocol

BENCHMARK = "nyu_ctf_web_test"
TASK_COUNT = 19
SOURCE_REPOSITORY = "https://github.com/NYU-LLM-CTF/NYU_CTF_Bench"
SOURCE_RELEASE = "v20250206"
UNAVAILABLE = frozenset({"2021q-web-scp_terminal", "2020f-web-snailrace1", "2020f-web-sharkfacts"})
ARMS = ("base", "step_1000")
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")


class NyuAdapterError(RuntimeError):
    """The frozen source, runtime, trace, or result contract is invalid."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _digest_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _digest(value: object) -> str:
    return _digest_bytes(_canonical(value))


def _safe_path(value: object) -> str:
    if not isinstance(value, str):
        raise NyuAdapterError("source_path_invalid")
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise NyuAdapterError("source_path_invalid")
    return path.as_posix()


def _blob(checkout: Path, commit: str, path: str) -> bytes:
    if _COMMIT.fullmatch(commit) is None:
        raise NyuAdapterError("source_commit_invalid")
    path = _safe_path(path)
    try:
        return subprocess.run(
            ["git", "show", f"{commit}:{path}"],
            cwd=checkout,
            check=True,
            capture_output=True,
        ).stdout
    except subprocess.CalledProcessError as exc:
        raise NyuAdapterError("pinned_source_blob_missing") from exc


def _benchmark(protocol: dict[str, Any]) -> dict[str, Any]:
    try:
        benchmark = protocol["benchmarks"][BENCHMARK]
    except (KeyError, TypeError) as exc:
        raise NyuAdapterError("nyu_protocol_missing") from exc
    task_ids = benchmark.get("task_ids")
    unavailable = benchmark.get("execution_unavailable_task_ids")
    harness = benchmark.get("harness", {})
    bindings = benchmark.get("bindings", {})
    if (
        benchmark.get("task_count") != TASK_COUNT
        or not isinstance(task_ids, list)
        or len(task_ids) != TASK_COUNT
        or len(set(task_ids)) != TASK_COUNT
        or _digest_bytes(("\n".join(task_ids) + "\n").encode()) != benchmark.get("task_ids_sha256")
        or benchmark.get("source_unavailable_task_ids") != []
        or not isinstance(unavailable, list)
        or set(unavailable) != UNAVAILABLE
        or len(unavailable) != len(UNAVAILABLE)
        or benchmark.get("selection") != "category_equals_web_in_official_test_dataset"
        or benchmark.get("scoring") != "pinned_manifest_exact_hidden_flag"
        or harness.get("name") != "opencode"
        or harness.get("version") != "1.18.27"
        or harness.get("platform") != "linux/amd64"
        or _SHA256.fullmatch(str(harness.get("image_id", ""))) is None
        or set(bindings)
        != {
            "system_prompt_sha256",
            "tool_schema_sha256",
            "runner_sha256",
            "grader_sha256",
            "adapter_sha256",
        }
        or any(_SHA256.fullmatch(str(value)) is None for value in bindings.values())
    ):
        raise NyuAdapterError("nyu_protocol_contract_invalid")
    source = benchmark.get("source", {})
    if (
        source.get("repository") != SOURCE_REPOSITORY
        or source.get("release") != SOURCE_RELEASE
        or _COMMIT.fullmatch(str(source.get("commit", ""))) is None
        or _SHA256.fullmatch(str(source.get("license_sha256", ""))) is None
        or _SHA256.fullmatch(str(benchmark.get("source_manifest_sha256", ""))) is None
    ):
        raise NyuAdapterError("nyu_source_binding_invalid")
    arms = protocol.get("arms", {})
    if set(arms) != set(ARMS):
        raise NyuAdapterError("nyu_arm_binding_invalid")
    common = (
        "endpoint_origin_sha256",
        "tokenizer_sha256",
        "chat_template_sha256",
        "serving_runtime_sha256",
        "max_context_size",
        "inference_precision",
        "quantization",
    )
    if any(arms["base"].get(key) != arms["step_1000"].get(key) for key in common):
        raise NyuAdapterError("nyu_non_weight_arm_drift")
    return benchmark


@dataclass(frozen=True, repr=False)
class NyuTask:
    task_id: str
    source_commit: str
    challenge_sha256: str
    compose_sha256: str
    attachment_manifest_sha256: str
    target_host: str
    target_port: int
    _prompt: str = field(repr=False)
    _flag: str = field(repr=False)
    _attachments: tuple[tuple[str, bytes], ...] = field(repr=False)

    def agent_inputs(self) -> dict[str, object]:
        """Return only official player-visible bytes; never the hidden flag."""
        return {
            "prompt": self._prompt,
            "attachments": [
                {"path": path, "bytes": content} for path, content in self._attachments
            ],
        }

    def score(self, answer: str) -> int:
        if not isinstance(answer, str):
            raise NyuAdapterError("terminal_answer_invalid")
        return int(hmac.compare_digest(answer.encode(), self._flag.encode()))


def load_task(protocol: dict[str, Any], checkout: Path, task_id: str) -> NyuTask:
    """Load one task from exact Git bytes, immune to mutable checkout changes."""
    benchmark = _benchmark(protocol)
    if task_id not in benchmark["task_ids"]:
        raise NyuAdapterError("nyu_task_identity_invalid")
    if task_id in UNAVAILABLE:
        raise NyuAdapterError("nyu_task_runtime_unavailable")
    commit = benchmark["source"]["commit"]
    if _digest_bytes(_blob(checkout, commit, "LICENSE")) != benchmark["source"]["license_sha256"]:
        raise NyuAdapterError("nyu_license_digest_mismatch")
    manifest_raw = _blob(checkout, commit, "test_dataset.json")
    if _digest_bytes(manifest_raw) != benchmark["source_manifest_sha256"]:
        raise NyuAdapterError("nyu_manifest_digest_mismatch")
    manifest = json.loads(manifest_raw)
    if not isinstance(manifest, dict):
        raise NyuAdapterError("nyu_manifest_invalid")
    official = [key for key, value in manifest.items() if value.get("category") == "web"]
    if official != benchmark["task_ids"]:
        raise NyuAdapterError("nyu_official_roster_drift")
    root = _safe_path(manifest[task_id].get("path"))
    challenge_raw = _blob(checkout, commit, f"{root}/challenge.json")
    compose_raw = _blob(checkout, commit, f"{root}/docker-compose.yml")
    challenge = json.loads(challenge_raw)
    if challenge.get("compose") is not True or challenge.get("category") != "web":
        raise NyuAdapterError("nyu_compose_contract_invalid")
    flag = challenge.get("flag")
    description = challenge.get("description")
    host = challenge.get("box")
    port = challenge.get("internal_port", challenge.get("port"))
    if isinstance(port, str) and port.isdecimal():
        port = int(port)
    if (
        not isinstance(flag, str)
        or not flag
        or not isinstance(description, str)
        or not description
        or not isinstance(host, str)
        or not host
        or type(port) is not int
        or not 0 < port < 65536
    ):
        raise NyuAdapterError("nyu_challenge_contract_invalid")
    try:
        prompt = description.format_map({"box": host, "port": port})
    except (KeyError, ValueError) as exc:
        raise NyuAdapterError("nyu_prompt_format_invalid") from exc
    prompt += "\n\nReturn only the exact flag."
    if flag in prompt:
        raise NyuAdapterError("nyu_hidden_flag_visible_in_prompt")
    paths = challenge.get("files", [])
    if not isinstance(paths, list) or len(paths) != len(set(paths)):
        raise NyuAdapterError("nyu_attachment_manifest_invalid")
    attachments: list[tuple[str, bytes]] = []
    for item in paths:
        path = _safe_path(item)
        content = _blob(checkout, commit, f"{root}/{path}")
        if flag.encode() in content:
            raise NyuAdapterError("nyu_hidden_flag_visible_in_attachment")
        attachments.append((path, content))
    attachment_manifest = [
        {"path": path, "sha256": _digest_bytes(content), "size": len(content)}
        for path, content in attachments
    ]
    return NyuTask(
        task_id=task_id,
        source_commit=commit,
        challenge_sha256=_digest_bytes(challenge_raw),
        compose_sha256=_digest_bytes(compose_raw),
        attachment_manifest_sha256=_digest(attachment_manifest),
        target_host=host,
        target_port=port,
        _prompt=prompt,
        _flag=flag,
        _attachments=tuple(attachments),
    )


def _signed(value: dict[str, Any], signature: str = "receipt_sha256") -> bool:
    observed = value.get(signature)
    unsigned = {key: item for key, item in value.items() if key != signature}
    return _SHA256.fullmatch(str(observed or "")) is not None and observed == _digest(unsigned)


def _validate_qualification(
    protocol: dict[str, Any], task: NyuTask, qualification: dict[str, Any]
) -> None:
    benchmark = _benchmark(protocol)
    images = qualification.get("images")
    expected = {
        "schema",
        "protocol_sha256",
        "benchmark",
        "task_id",
        "source_commit",
        "challenge_sha256",
        "compose_sha256",
        "attachment_manifest_sha256",
        "target_host",
        "target_port",
        "status",
        "model_requests",
        "startup_healthy",
        "target_reachable",
        "grader_negative_control",
        "grader_positive_control",
        "agent_image_id",
        "opencode_version",
        "platform",
        "isolated_task_network",
        "images",
        "receipt_sha256",
    }
    if (
        set(qualification) != expected
        or not _signed(qualification)
        or qualification.get("schema") != "external_ctf_nyu_runtime_qualification_v1"
        or qualification.get("protocol_sha256") != protocol.get("protocol_sha256")
        or qualification.get("benchmark") != BENCHMARK
        or qualification.get("task_id") != task.task_id
        or qualification.get("source_commit") != task.source_commit
        or qualification.get("challenge_sha256") != task.challenge_sha256
        or qualification.get("compose_sha256") != task.compose_sha256
        or qualification.get("attachment_manifest_sha256") != task.attachment_manifest_sha256
        or qualification.get("target_host") != task.target_host
        or qualification.get("target_port") != task.target_port
        or qualification.get("status") != "runtime_preflight_passed"
        or qualification.get("model_requests") != 0
        or qualification.get("startup_healthy") is not True
        or qualification.get("target_reachable") is not True
        or qualification.get("grader_negative_control") is not False
        or qualification.get("grader_positive_control") is not True
        or qualification.get("agent_image_id") != benchmark["harness"]["image_id"]
        or qualification.get("opencode_version") != benchmark["harness"]["version"]
        or qualification.get("platform") != "linux/amd64"
        or qualification.get("isolated_task_network") is not True
        or not isinstance(images, list)
        or not images
        or images != sorted(images, key=lambda row: str(row.get("reference")))
        or len({row.get("reference") for row in images}) != len(images)
        or any(
            not isinstance(row, dict)
            or set(row) != {"reference", "image_id", "repository_digest"}
            or not isinstance(row.get("reference"), str)
            or _SHA256.fullmatch(str(row.get("image_id", ""))) is None
            or "@sha256:" not in str(row.get("repository_digest", ""))
            for row in images
        )
    ):
        raise NyuAdapterError("nyu_runtime_qualification_invalid")


def agent_contract(
    protocol: dict[str, Any],
    task: NyuTask,
    arm: str,
    qualification: dict[str, Any],
) -> dict[str, Any]:
    """Build a launch-neutral OpenCode contract after model-free qualification."""
    benchmark = _benchmark(protocol)
    if arm not in ARMS:
        raise NyuAdapterError("nyu_arm_invalid")
    _validate_qualification(protocol, task, qualification)
    model = protocol["arms"][arm]
    common = {
        "schema": "external_ctf_nyu_agent_contract_v1",
        "benchmark": BENCHMARK,
        "protocol_sha256": protocol["protocol_sha256"],
        "task_id": task.task_id,
        "task_binding": {
            "source_commit": task.source_commit,
            "challenge_sha256": task.challenge_sha256,
            "compose_sha256": task.compose_sha256,
            "attachment_manifest_sha256": task.attachment_manifest_sha256,
            "prompt_sha256": _digest_bytes(task._prompt.encode()),
            "target_host": task.target_host,
            "target_port": task.target_port,
        },
        "harness": {
            "name": "opencode",
            "version": benchmark["harness"]["version"],
            "image_id": benchmark["harness"]["image_id"],
            "platform": "linux/amd64",
            "context": model["max_context_size"],
            "output": benchmark["budget"]["max_output_tokens"],
            "max_model_requests": benchmark["budget"]["max_model_requests"],
            "native_compaction": True,
            "allowed_tools": ["bash", "edit", "read", "glob", "grep", "list"],
        },
        "bindings": dict(benchmark["bindings"]),
        "isolation": {
            "agent_user": "1000:1000",
            "read_only_root": True,
            "privileged": False,
            "cap_drop": ["ALL"],
            "no_new_privileges": True,
            "host_network": False,
            "docker_socket_mounted": False,
            "benchmark_checkout_mounted": False,
            "provider_credentials_in_agent": False,
            "task_network": "per_cell_internal",
            "model_network": "per_cell_internal_proxy_only",
            "agent_mounts": [
                {"name": "task-input", "mode": "ro"},
                {"name": "workspace", "mode": "rw"},
                {"name": "home", "mode": "rw"},
                {"name": "sealed-output", "mode": "rw"},
            ],
        },
        "budget": dict(benchmark["budget"]),
        "sampling": dict(protocol["execution"]["sampling"]),
        "qualification_receipt_sha256": qualification["receipt_sha256"],
    }
    contract = {
        **common,
        "arm": arm,
        "model_binding": {
            key: model[key] for key in ("served_model", "model_revision", "model_artifact_sha256")
        },
        "common_contract_sha256": _digest(common),
    }
    contract["contract_sha256"] = _digest(contract)
    return contract


def runtime_qualification_contract(protocol: dict[str, Any], task: NyuTask) -> dict[str, Any]:
    """Return a content-free contract for one future Linux runtime qualification."""
    benchmark = _benchmark(protocol)
    if (
        task.task_id not in benchmark["task_ids"]
        or task.task_id in UNAVAILABLE
        or task.source_commit != benchmark["source"]["commit"]
    ):
        raise NyuAdapterError("nyu_task_identity_invalid")
    unsigned = {
        "schema": "external_ctf_nyu_runtime_qualification_contract_v1",
        "protocol_sha256": protocol["protocol_sha256"],
        "benchmark": BENCHMARK,
        "task_index": benchmark["task_ids"].index(task.task_id),
        "task_id_sha256": _digest_bytes(task.task_id.encode()),
        "source_commit": task.source_commit,
        "task_binding_sha256": _digest(
            {
                "challenge_sha256": task.challenge_sha256,
                "compose_sha256": task.compose_sha256,
                "attachment_manifest_sha256": task.attachment_manifest_sha256,
                "target_host": task.target_host,
                "target_port": task.target_port,
            }
        ),
        "platform": "linux/amd64",
        "agent_image_id": benchmark["harness"]["image_id"],
        "opencode_version": benchmark["harness"]["version"],
        "model_requests": 0,
        "required_controls": {
            "startup_healthy": True,
            "target_reachable": True,
            "grader_negative_control": False,
            "grader_positive_control": True,
            "isolated_task_network": True,
            "resolved_image_digests_required": True,
        },
    }
    return {**unsigned, "contract_sha256": _digest(unsigned)}


def source_qualification_manifest(protocol: dict[str, Any], checkout: Path) -> dict[str, Any]:
    """Qualify exact source bytes and predeclare every task without starting a runtime."""
    benchmark = _benchmark(protocol)
    if benchmark.get("adapter_qualified") is not False:
        raise NyuAdapterError("nyu_qualification_requires_closed_adapter_gate")
    rows: list[dict[str, Any]] = []
    contracts: list[str] = []
    for task_index, task_id in enumerate(benchmark["task_ids"]):
        identity = {
            "task_index": task_index,
            "task_id_sha256": _digest_bytes(task_id.encode()),
        }
        if task_id in UNAVAILABLE:
            rows.append(
                {
                    **identity,
                    "source_state": "present_runtime_unavailable",
                    "runtime_state": "infra_invalid_no_reproducible_runtime",
                    "runtime_qualification_contract_sha256": None,
                }
            )
            continue
        task = load_task(protocol, checkout, task_id)
        contract = runtime_qualification_contract(protocol, task)
        contracts.append(contract["contract_sha256"])
        rows.append(
            {
                **identity,
                "source_state": "present",
                "runtime_state": "pending_model_free_linux_amd64_qualification",
                "runtime_qualification_contract_sha256": contract["contract_sha256"],
            }
        )
    if (
        len(rows) != TASK_COUNT
        or sum(row["source_state"] == "present" for row in rows) != 16
        or sum(row["source_state"] == "present_runtime_unavailable" for row in rows) != 3
    ):
        raise NyuAdapterError("nyu_source_qualification_count_mismatch")
    unsigned = {
        "schema": "external_ctf_nyu_source_qualification_v1",
        "protocol_sha256": protocol["protocol_sha256"],
        "benchmark": BENCHMARK,
        "source_commit": benchmark["source"]["commit"],
        "official_task_count": TASK_COUNT,
        "runtime_candidate_task_count": 16,
        "runtime_unavailable_task_count": 3,
        "task_rows": rows,
        "runtime_qualification_contracts_sha256": _digest(contracts),
        "status": "source_qualified_runtime_pending",
        "adapter_qualified": False,
        "runtime_qualified": False,
        "provider_calls": 0,
        "model_requests": 0,
        "challenge_containers_started": 0,
        "contains_prompts_flags_solutions_traces_or_scores": False,
    }
    return {**unsigned, "receipt_sha256": _digest(unsigned)}


def assert_matched_pair(base: dict[str, Any], candidate: dict[str, Any]) -> None:
    if base.get("arm") != "base" or candidate.get("arm") != "step_1000":
        raise NyuAdapterError("nyu_pair_arm_invalid")
    for contract in (base, candidate):
        observed = contract.get("contract_sha256")
        unsigned = {key: item for key, item in contract.items() if key != "contract_sha256"}
        if observed != _digest(unsigned):
            raise NyuAdapterError("nyu_agent_contract_digest_invalid")
    excluded = {"arm", "model_binding", "contract_sha256"}

    def common(value: dict[str, Any]) -> dict[str, Any]:
        return {key: item for key, item in value.items() if key not in excluded}

    if (
        common(base) != common(candidate)
        or base.get("common_contract_sha256") != candidate.get("common_contract_sha256")
        or base.get("model_binding") == candidate.get("model_binding")
    ):
        raise NyuAdapterError("nyu_pair_non_weight_drift")


def terminal_answer(trace: bytes, *, exit_code: int, timed_out: bool) -> str:
    """Accept only a complete, natural OpenCode 1.18.27 terminal step."""
    if timed_out:
        raise NyuAdapterError("opencode_execution_timeout")
    if exit_code != 0:
        raise NyuAdapterError("opencode_process_error")
    events: list[dict[str, Any]] = []
    try:
        lines = trace.decode(errors="strict").splitlines()
    except UnicodeDecodeError as exc:
        raise NyuAdapterError("opencode_trace_malformed") from exc
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise NyuAdapterError("opencode_trace_malformed") from exc
        if not isinstance(event, dict):
            raise NyuAdapterError("opencode_trace_malformed")
        events.append(event)
    if not events:
        raise NyuAdapterError("opencode_trace_empty")
    if any(event.get("type") == "error" for event in events):
        raise NyuAdapterError("opencode_harness_error")
    finishes = [
        (index, event)
        for index, event in enumerate(events)
        if event.get("type") == "step_finish" and isinstance(event.get("part"), dict)
    ]
    if not finishes:
        raise NyuAdapterError("opencode_trace_unfinished")
    final_index, final = finishes[-1]
    if any(event.get("type") == "step_start" for event in events[final_index + 1 :]):
        raise NyuAdapterError("opencode_trace_unfinished")
    reason = final["part"].get("reason")
    if reason in {"length", "max_tokens", "max_output_tokens"}:
        raise NyuAdapterError("opencode_output_limit")
    if reason != "stop":
        raise NyuAdapterError("opencode_trace_not_natural")
    for event in reversed(events[: final_index + 1]):
        part = event.get("part")
        if not isinstance(part, dict):
            continue
        kind = str(part.get("type") or event.get("type") or "").lower()
        text = part.get("text", part.get("content"))
        if kind == "text" and isinstance(text, str) and text:
            return text
    return ""


def _write_once(path: Path, content: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    os.fchmod(descriptor, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def seal_result_once(
    output: Path,
    *,
    protocol: dict[str, Any],
    task: NyuTask,
    arm: str,
    qualification: dict[str, Any],
    trace: bytes,
    exit_code: int,
    timed_out: bool,
) -> dict[str, Any]:
    """Privately preserve the full trace and one score receipt without replacement."""
    benchmark = _benchmark(protocol)
    _validate_qualification(protocol, task, qualification)
    if arm not in ARMS:
        raise NyuAdapterError("nyu_arm_invalid")
    answer = terminal_answer(trace, exit_code=exit_code, timed_out=timed_out)
    score = task.score(answer)
    task_index = benchmark["task_ids"].index(task.task_id)
    unsigned = {
        "schema": "external_ctf_nyu_sealed_result_v1",
        "protocol_sha256": protocol["protocol_sha256"],
        "benchmark": BENCHMARK,
        "task_index": task_index,
        "task_id": task.task_id,
        "arm": arm,
        "outcome": "accepted_model_outcome",
        "status": "scored",
        "score": score,
        "grader": benchmark["scoring"],
        "termination": "natural_stop",
        "trace_sha256": _digest_bytes(trace),
        "qualification_receipt_sha256": qualification["receipt_sha256"],
        "preservation_state": "sandbox_private_complete_controller_export_required",
    }
    receipt = {**unsigned, "receipt_sha256": _digest(unsigned)}
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    output.chmod(0o700)
    _write_once(output / "trace.jsonl", trace)
    _write_once(output / "terminal.json", _canonical(receipt) + b"\n")
    directory = os.open(output, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    if any(
        stat.S_IMODE(path.stat().st_mode) != mode
        for path, mode in (
            (output, 0o700),
            (output / "trace.jsonl", 0o600),
            (output / "terminal.json", 0o600),
        )
    ):
        raise NyuAdapterError("nyu_sealed_result_permissions_invalid")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seal the provider-free NYU web source and runtime-qualification manifest."
    )
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = source_qualification_manifest(load_protocol(args.protocol), args.checkout)
    _write_once(args.output, _canonical(receipt) + b"\n")
    print(
        json.dumps(
            {"receipt_sha256": receipt["receipt_sha256"], "status": receipt["status"]},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
