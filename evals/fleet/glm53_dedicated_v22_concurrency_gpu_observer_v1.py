"""UID-bound, content-free GPU observer for the GLM v22 concurrency ladder."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from evals.fleet import glm53_dedicated_v22_concurrency_qualification_v1 as qualification
from evals.fleet import self_hosted

NAMESPACE = "fleet-train-jobs"
SAMPLE_SECONDS = 240
POLL_SECONDS = 1
UUID_KEYS = ("server_rayjob_uid", "server_head_pod_uid", "qualifier_job_uid", "qualifier_pod_uid")


class ObserverError(RuntimeError):
    """Stable content-free observer failure."""


CommandRunner = Callable[..., subprocess.CompletedProcess[bytes]]


def _command(
    args: list[str],
    *,
    runner: CommandRunner = subprocess.run,
    input_bytes: bytes | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    return runner(
        args,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )


def _json(args: list[str], *, runner: CommandRunner = subprocess.run) -> dict[str, Any]:
    try:
        value = json.loads(_command(args, runner=runner).stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise ObserverError("kubernetes_identity_read_failed") from exc
    if not isinstance(value, dict):
        raise ObserverError("kubernetes_identity_shape_invalid")
    return value


def _ready(pod: dict[str, Any], container: str) -> bool:
    statuses = pod.get("status", {}).get("containerStatuses", [])
    row = next((value for value in statuses if value.get("name") == container), None)
    return (
        pod.get("status", {}).get("phase") == "Running"
        and row is not None
        and row.get("ready") is True
        and row.get("restartCount") == 0
    )


def resolve_identity(
    root: Path, *, runner: CommandRunner = subprocess.run
) -> tuple[dict[str, str], str, str]:
    plan = qualification.render(root)
    server = plan["server"]
    rayjob = _json(
        ["kubectl", "-n", NAMESPACE, "get", "rayjob", server["api_run_id"], "-o", "json"],
        runner=runner,
    )
    pods = _json(["kubectl", "-n", NAMESPACE, "get", "pods", "-o", "json"], runner=runner)
    server_rows = [
        row
        for row in pods.get("items", [])
        if row.get("metadata", {}).get("uid") == server["head_pod_uid"]
    ]
    qualifier_rows = [
        row
        for row in pods.get("items", [])
        if row.get("metadata", {}).get("labels", {}).get("job-name") == qualification.JOB_NAME
    ]
    qualifier_job = _json(
        ["kubectl", "-n", NAMESPACE, "get", "job", qualification.JOB_NAME, "-o", "json"],
        runner=runner,
    )
    if (
        rayjob.get("metadata", {}).get("uid") != server["rayjob_uid"]
        or len(server_rows) != 1
        or len(qualifier_rows) != 1
        or not _ready(server_rows[0], "ray-head")
        or not _ready(qualifier_rows[0], "qualifier")
        or qualifier_job.get("status", {}).get("active") != 1
    ):
        raise ObserverError("kubernetes_identity_or_health_mismatch")
    server_pod = server_rows[0]["metadata"]["name"]
    qualifier_pod = qualifier_rows[0]["metadata"]["name"]
    identity = {
        "server_rayjob_uid": rayjob["metadata"]["uid"],
        "server_head_pod_uid": server_rows[0]["metadata"]["uid"],
        "qualifier_job_uid": qualifier_job["metadata"]["uid"],
        "qualifier_pod_uid": qualifier_rows[0]["metadata"]["uid"],
    }
    return identity, server_pod, qualifier_pod


def parse_gpu_sample(raw: bytes) -> dict[int, tuple[float, int, int]]:
    rows: dict[int, tuple[float, int, int]] = {}
    try:
        for line in raw.decode("utf-8", errors="strict").splitlines():
            index, utilization, memory_used, memory_total = [
                part.strip() for part in line.split(",")
            ]
            rows[int(index)] = (float(utilization), int(memory_used), int(memory_total))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ObserverError("gpu_sample_invalid") from exc
    if sorted(rows) != list(range(8)) or any(
        total <= 0 or used < 0 or used > total for _, used, total in rows.values()
    ):
        raise ObserverError("gpu_device_roster_invalid")
    return rows


def build_receipt(
    *,
    concurrency: int,
    server: dict[str, Any],
    identity_before: dict[str, str],
    identity_after: dict[str, str],
    samples: list[dict[int, tuple[float, int, int]]],
) -> dict[str, Any]:
    if (
        concurrency not in qualification.CONCURRENCY
        or identity_before != identity_after
        or set(identity_before) != set(UUID_KEYS)
        or not samples
    ):
        raise ObserverError("observer_identity_or_samples_invalid")
    max_utilization = [max(sample[index][0] for sample in samples) for index in range(8)]
    max_memory = [max(sample[index][1] for sample in samples) for index in range(8)]
    if any(value <= 0 or value > 100 for value in max_utilization):
        raise ObserverError("not_all_expected_gpus_active_in_wave")
    body: dict[str, Any] = {
        "schema_version": qualification.GPU_OBSERVER_SCHEMA,
        "status": "OBSERVED_SCORE_FREE_WAVE",
        "concurrency": concurrency,
        "server": server,
        "identity": identity_before,
        "devices_seen": 8,
        "samples_per_device": len(samples),
        "max_utilization_percent_by_device": max_utilization,
        "max_memory_mib_by_device": max_memory,
        "server_identity_unchanged": True,
        "qualifier_identity_unchanged": True,
        "phase_observed_by_existence_only": True,
        "prompts_responses_traces_tool_arguments_scores_read_or_persisted": False,
    }
    body["receipt_sha256"] = self_hosted.digest_without(body, "receipt_sha256")
    return body


def _phase_exists(qualifier_pod: str, concurrency: int, *, runner: CommandRunner) -> bool:
    path = qualification.RESULT_ROOT / f"PHASE-c{concurrency}.json"
    result = _command(
        [
            "kubectl",
            "-n",
            NAMESPACE,
            "exec",
            qualifier_pod,
            "-c",
            "qualifier",
            "--",
            "test",
            "-f",
            str(path),
        ],
        runner=runner,
        check=False,
    )
    return result.returncode == 0


def atomic_write_command(qualifier_pod: str, destination: Path) -> list[str]:
    writer = (
        "import os,sys; p=sys.argv[1]; data=sys.stdin.buffer.read(); "
        "fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o444); "
        "os.write(fd,data); os.fsync(fd); os.close(fd)"
    )
    return [
        "kubectl",
        "-n",
        NAMESPACE,
        "exec",
        "-i",
        qualifier_pod,
        "-c",
        "qualifier",
        "--",
        "python",
        "-c",
        writer,
        str(destination),
    ]


def observe(
    root: Path,
    *,
    runner: CommandRunner = subprocess.run,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict[str, Any]]:
    plan = qualification.render(root)
    receipts: list[dict[str, Any]] = []
    identity, server_pod, qualifier_pod = resolve_identity(root, runner=runner)
    for concurrency in qualification.CONCURRENCY:
        deadline = monotonic() + qualification.GPU_OBSERVER_WAIT_SECONDS - 30
        while monotonic() < deadline and not _phase_exists(
            qualifier_pod, concurrency, runner=runner
        ):
            sleep(POLL_SECONDS)
        if not _phase_exists(qualifier_pod, concurrency, runner=runner):
            raise ObserverError(f"c{concurrency}_phase_timeout")
        samples: list[dict[int, tuple[float, int, int]]] = []
        sample_deadline = min(deadline, monotonic() + SAMPLE_SECONDS)
        while monotonic() < sample_deadline:
            raw = _command(
                [
                    "kubectl",
                    "-n",
                    NAMESPACE,
                    "exec",
                    server_pod,
                    "-c",
                    "ray-head",
                    "--",
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                runner=runner,
            ).stdout
            samples.append(parse_gpu_sample(raw))
            if all(max(sample[index][0] for sample in samples) > 0 for index in range(8)):
                break
            sleep(POLL_SECONDS)
        identity_after, current_server_pod, current_qualifier_pod = resolve_identity(
            root, runner=runner
        )
        if current_server_pod != server_pod or current_qualifier_pod != qualifier_pod:
            raise ObserverError("pod_name_changed")
        receipt = build_receipt(
            concurrency=concurrency,
            server=plan["server"],
            identity_before=identity,
            identity_after=identity_after,
            samples=samples,
        )
        destination = qualification.RESULT_ROOT / "gpu-observer" / f"c{concurrency}.json"
        payload = (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode()
        _command(
            atomic_write_command(qualifier_pod, destination),
            runner=runner,
            input_bytes=payload,
        )
        receipts.append(receipt)
    return receipts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    observe(args.root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
