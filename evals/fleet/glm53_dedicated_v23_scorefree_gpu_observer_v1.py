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
RELEASE_CONFIRMATION_SCHEMA = "fleet-glm53-dedicated-v23-watchdog-release-confirmed-uid-absent-v1"


class ObserverError(RuntimeError):
    """The UID-bound observer failed closed."""


CommandRunner = Callable[..., subprocess.CompletedProcess[bytes]]


def qualifier_pod_owned_by_job(pod: dict[str, Any], job: dict[str, Any]) -> bool:
    owners = pod.get("metadata", {}).get("ownerReferences") or []
    return owners == [
        {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "name": job.get("metadata", {}).get("name"),
            "uid": job.get("metadata", {}).get("uid"),
            "controller": True,
            "blockOwnerDeletion": True,
        }
    ]


def _is_v23_remnant(row: dict[str, Any], binding: dict[str, Any]) -> bool:
    """Match exact UIDs, owner UIDs, and name/label remnants of one v23 server."""

    metadata = row.get("metadata") or {}
    expected_uids = {
        binding["rayjob_uid"],
        binding["workload_uid"],
        binding["head_pod_uid"],
        binding["service_uid"],
    }
    if metadata.get("uid") in expected_uids or any(
        owner.get("uid") in expected_uids for owner in metadata.get("ownerReferences") or []
    ):
        return True
    identity_tokens = (
        binding["api_run_id"],
        binding["server_title"],
        binding["server_run_dir"],
    )
    safe_metadata_strings = [metadata.get("name", "")]
    safe_metadata_strings.extend(
        str(owner.get("name", "")) for owner in metadata.get("ownerReferences") or []
    )
    safe_metadata_strings.extend(
        str(value)
        for field in ("labels", "annotations")
        for value in (metadata.get(field) or {}).values()
    )
    return any(
        token in candidate
        for token in identity_tokens
        for candidate in safe_metadata_strings
        if candidate
    )


def build_release_confirmation(
    binding: dict[str, Any], terminal: dict[str, Any], inventory: dict[str, Any]
) -> dict[str, Any]:
    """Distinguish accepted API deletion from exact cluster UID absence."""

    expected_uids = {
        binding.get("rayjob_uid"),
        binding.get("workload_uid"),
        binding.get("head_pod_uid"),
        binding.get("service_uid"),
    }
    rows = inventory.get("items")
    released_keys = {
        "schema_version",
        "status",
        "server_binding_sha256",
        "active_receipt_sha256",
        "release_route",
        "fleet_task_instance_calls",
        "fleet_session_calls",
        "verifier_calls",
        "scoring_calls",
        "protected_content_included",
        "receipt_sha256",
    }
    failed_keys = (released_keys - {"active_receipt_sha256"}) | {"reason"}
    owned_rows = [row for row in rows or [] if _is_v23_remnant(row, binding)]
    if (
        len(expected_uids) != 4
        or any(not isinstance(value, str) or not value for value in expected_uids)
        or set(terminal) not in (released_keys, failed_keys)
        or terminal.get("receipt_sha256") != crypto.digest_without(terminal, "receipt_sha256")
        or terminal.get("status")
        not in {"RELEASED_IDLE_API_ABSENT", "FAILED_CLOSED_RELEASED_API_ABSENT"}
        or terminal.get("schema_version") != "fleet-glm53-dedicated-v23-watchdog-terminal-v1"
        or terminal.get("release_route") != "DELETE /v1/runs/{api_run_id}"
        or any(
            terminal.get(field) != 0
            for field in (
                "fleet_task_instance_calls",
                "fleet_session_calls",
                "verifier_calls",
                "scoring_calls",
            )
        )
        or terminal.get("protected_content_included") is not False
        or terminal.get("server_binding_sha256") != crypto.sha256(crypto.canonical_json(binding))
        or not isinstance(rows, list)
        or owned_rows
    ):
        raise ObserverError("v23_release_uid_absence_unconfirmed")
    body: dict[str, Any] = {
        "schema_version": RELEASE_CONFIRMATION_SCHEMA,
        "status": "CONFIRMED_UID_ABSENT",
        "server_binding_sha256": crypto.sha256(crypto.canonical_json(binding)),
        "watchdog_terminal_receipt_sha256": terminal["receipt_sha256"],
        "absent_uids": sorted(str(value) for value in expected_uids),
        "resource_kinds_checked": ["Pod", "RayCluster", "RayJob", "Service", "Workload"],
        "fleet_task_instance_calls": 0,
        "fleet_session_calls": 0,
        "verifier_calls": 0,
        "scoring_calls": 0,
        "protected_content_included": False,
    }
    body["receipt_sha256"] = crypto.digest_without(body, "receipt_sha256")
    return body


def confirm_release(
    binding: dict[str, Any], terminal: dict[str, Any], *, runner: CommandRunner = subprocess.run
) -> dict[str, Any]:
    inventory = support._json(  # noqa: SLF001
        [
            "kubectl",
            "-n",
            NAMESPACE,
            "get",
            "rayjobs.ray.io,rayclusters.ray.io,workloads.kueue.x-k8s.io,pods,services",
            "-o",
            "json",
        ],
        runner=runner,
    )
    return build_release_confirmation(binding, terminal, inventory)


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
        or not qualifier_pod_owned_by_job(qualifier_rows[0], job)
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
    parser.add_argument(
        "command", nargs="?", choices=("observe", "confirm-release"), default="observe"
    )
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--binding", type=Path)
    parser.add_argument("--terminal", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.command == "observe":
        if args.authorization is None:
            parser.error("observe requires --authorization")
        observe(qualifier.load(args.authorization))
    else:
        if args.binding is None or args.terminal is None or args.out is None:
            parser.error("confirm-release requires --binding, --terminal, and --out")
        receipt = confirm_release(
            qualifier.load(args.binding),
            qualifier.load(args.terminal),
        )
        qualifier.engine.write_once(args.out, receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
