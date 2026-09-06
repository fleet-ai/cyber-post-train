"""UID-bound, content-free GPU observer for the GLM v23 score-free qualifier."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import glm53_dedicated_v22_concurrency_gpu_observer_v1 as support
from evals.fleet import glm53_dedicated_v23_scorefree_qualifier_v1 as qualifier

NAMESPACE = "fleet-train-jobs"
SAMPLE_SECONDS = 240
POLL_SECONDS = 1


class ObserverError(RuntimeError):
    """The UID-bound observer failed closed."""


CommandRunner = Callable[..., subprocess.CompletedProcess[bytes]]


def resolve_identity(
    authorization: dict[str, Any], *, runner: CommandRunner = subprocess.run
) -> tuple[dict[str, str], str, str]:
    binding = authorization.get("server_binding") or {}
    rayjob = support._json(  # noqa: SLF001
        ["kubectl", "-n", NAMESPACE, "get", "rayjob", binding["api_run_id"], "-o", "json"],
        runner=runner,
    )
    pods = support._json(  # noqa: SLF001
        ["kubectl", "-n", NAMESPACE, "get", "pods", "-o", "json"], runner=runner
    )
    server_rows = [
        row
        for row in pods.get("items", [])
        if row.get("metadata", {}).get("uid") == binding.get("head_pod_uid")
    ]
    qualifier_rows = [
        row
        for row in pods.get("items", [])
        if row.get("metadata", {}).get("labels", {}).get("job-name") == qualifier.JOB_NAME
    ]
    job = support._json(  # noqa: SLF001
        ["kubectl", "-n", NAMESPACE, "get", "job", qualifier.JOB_NAME, "-o", "json"],
        runner=runner,
    )
    if (
        rayjob.get("metadata", {}).get("uid") != binding.get("rayjob_uid")
        or len(server_rows) != 1
        or len(qualifier_rows) != 1
        or not support._ready(server_rows[0], "ray-head")  # noqa: SLF001
        or not support._ready(qualifier_rows[0], "qualifier")  # noqa: SLF001
        or job.get("status", {}).get("active") != 1
    ):
        raise ObserverError("v23_kubernetes_identity_or_health_mismatch")
    identity = {
        "server_rayjob_uid": rayjob["metadata"]["uid"],
        "server_head_pod_uid": server_rows[0]["metadata"]["uid"],
        "qualifier_job_uid": job["metadata"]["uid"],
        "qualifier_pod_uid": qualifier_rows[0]["metadata"]["uid"],
    }
    return identity, server_rows[0]["metadata"]["name"], qualifier_rows[0]["metadata"]["name"]


def build_receipt(
    *,
    concurrency: int,
    identity_before: dict[str, str],
    identity_after: dict[str, str],
    samples: list[dict[int, tuple[float, int, int]]],
) -> dict[str, Any]:
    if identity_before != identity_after or not samples:
        raise ObserverError("v23_observer_identity_or_samples_invalid")
    utilization = [max(sample[index][0] for sample in samples) for index in range(8)]
    if any(value <= 0 or value > 100 for value in utilization):
        raise ObserverError("v23_not_all_expected_gpus_active")
    body: dict[str, Any] = {
        "schema_version": qualifier.GPU_OBSERVER_SCHEMA,
        "status": "OBSERVED_SCORE_FREE_WAVE",
        "concurrency": concurrency,
        "server": qualifier.build_held()["server"],
        "identity": identity_before,
        "devices_seen": 8,
        "samples_per_device": len(samples),
        "max_utilization_percent_by_device": utilization,
        "server_identity_unchanged": True,
        "qualifier_identity_unchanged": True,
        "prompts_responses_traces_tool_arguments_scores_read_or_persisted": False,
    }
    body["receipt_sha256"] = crypto.digest_without(body, "receipt_sha256")
    return body


def observe(
    authorization: dict[str, Any],
    *,
    runner: CommandRunner = subprocess.run,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict[str, Any]]:
    identity, server_pod, qualifier_pod = resolve_identity(authorization, runner=runner)
    receipts: list[dict[str, Any]] = []
    for concurrency in qualifier.CONCURRENCY:
        phase = qualifier.RESULT_ROOT / f"PHASE-c{concurrency}.json"
        for _ in range(qualifier.GPU_OBSERVER_WAIT_SECONDS):
            result = support._command(  # noqa: SLF001
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
                    str(phase),
                ],
                runner=runner,
                check=False,
            )
            if result.returncode == 0:
                break
            sleep(POLL_SECONDS)
        else:
            raise ObserverError(f"v23_c{concurrency}_phase_timeout")
        samples = []
        for _ in range(SAMPLE_SECONDS):
            raw = support._command(  # noqa: SLF001
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
            samples.append(support.parse_gpu_sample(raw))
            if all(max(sample[index][0] for sample in samples) > 0 for index in range(8)):
                break
            sleep(POLL_SECONDS)
        after, current_server, current_qualifier = resolve_identity(authorization, runner=runner)
        if current_server != server_pod or current_qualifier != qualifier_pod:
            raise ObserverError("v23_pod_name_changed")
        receipt = build_receipt(
            concurrency=concurrency,
            identity_before=identity,
            identity_after=after,
            samples=samples,
        )
        destination = qualifier.RESULT_ROOT / "gpu-observer" / f"c{concurrency}.json"
        payload = (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode()
        support._command(  # noqa: SLF001
            support.atomic_write_command(qualifier_pod, destination),
            runner=runner,
            input_bytes=payload,
        )
        receipts.append(receipt)
    return receipts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    args = parser.parse_args()
    observe(qualifier.load(args.authorization))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
