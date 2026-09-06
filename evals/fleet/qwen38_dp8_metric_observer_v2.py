"""Content-free Qwen DP8 request/GPU observer for the server idle rail."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import socket
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

RANKS = 8
METRIC_RE = re.compile(r"^(?P<head>[^\s{]+)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>[0-9.eE+-]+)$")
LABEL_RE = re.compile(r'(\w+)="([^"]*)"')


def request_counters(metrics: str) -> list[int]:
    families: dict[str, dict[int, int]] = {}
    for raw in metrics.splitlines():
        match = METRIC_RE.match(raw.strip())
        if match is None or "requests_total" not in match.group("head"):
            continue
        labels = dict(LABEL_RE.findall(match.group("labels") or ""))
        rank_text = labels.get("dp_rank") or labels.get("data_parallel_rank")
        if rank_text is None or not rank_text.isdigit():
            continue
        rank = int(rank_text)
        value = float(match.group("value"))
        if 0 <= rank < RANKS and value >= 0 and value.is_integer():
            families.setdefault(match.group("head"), {})[rank] = int(value)
    complete = [row for row in families.values() if set(row) == set(range(RANKS))]
    if len(complete) != 1:
        raise ValueError("metrics did not expose one exact eight-rank request family")
    return [complete[0][rank] for rank in range(RANKS)]


def gpu_sample(text: str) -> tuple[list[int], list[int]]:
    memory = [0] * RANKS
    utilization = [0] * RANKS
    seen: set[int] = set()
    for raw in csv.reader(text.splitlines()):
        if len(raw) != 3:
            raise ValueError("GPU sample row shape drifted")
        index, used, active = (item.strip() for item in raw)
        rank = int(index)
        if rank not in range(RANKS) or rank in seen:
            raise ValueError("GPU sample index drifted")
        seen.add(rank)
        memory[rank] = int(used)
        utilization[rank] = int(active)
    if seen != set(range(RANKS)) or any(value < 0 for value in memory) or any(
        not 0 <= value <= 100 for value in utilization
    ):
        raise ValueError("GPU sample is incomplete")
    return memory, utilization


def observation(
    before: list[int],
    after: list[int],
    *,
    memory_mib: list[int],
    utilization_percent: list[int],
    server_run_dir: str,
    pod_name: str,
    pod_uid: str,
    api_run_id: str,
    service_uid: str,
    server_binding_receipt_sha256: str,
    observed_at_epoch: int,
) -> dict[str, Any] | None:
    if len(before) != RANKS or len(after) != RANKS:
        raise ValueError("request counter shape drifted")
    if any(type(value) is not int or value < 0 for value in [*before, *after]):
        raise ValueError("request counter value drifted")
    if any(after[index] < before[index] for index in range(RANKS)):
        raise ValueError("request counters decreased")
    delta = [after[index] - before[index] for index in range(RANKS)]
    if sum(delta) == 0:
        return None
    if not server_run_dir.startswith("/mnt/sfs/jobs/chris-cyber-evalserve-q38-dp8-"):
        raise ValueError("server run directory is outside the Qwen DP8 rail")
    for value in (pod_uid, service_uid):
        if not re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", value
        ):
            raise ValueError("server binding UID is invalid")
    if not api_run_id.startswith("ft-run-") or not re.fullmatch(
        r"sha256:[0-9a-f]{64}", server_binding_receipt_sha256
    ):
        raise ValueError("server binding identity is invalid")
    body = {
        "schema_version": "fleet-qwen38-dp8-real-traffic-observation-v2",
        "status": "REAL_REQUEST_COUNTER_INCREASED",
        "server_run_dir": server_run_dir,
        "api_run_id": api_run_id,
        "pod_name": pod_name,
        "pod_uid": pod_uid,
        "service_uid": service_uid,
        "server_binding_receipt_sha256": server_binding_receipt_sha256,
        "observed_at_epoch": observed_at_epoch,
        "request_counters_before_by_rank": before,
        "request_counters_after_by_rank": after,
        "request_deltas_by_rank": delta,
        "gpu_memory_used_mib_by_rank": memory_mib,
        "gpu_utilization_percent_by_rank": utilization_percent,
        "prompts_traces_flags_or_scores_included": False,
    }
    body["receipt_sha256"] = "sha256:" + hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return body


def server_binding(path: Path, server_run_dir: str, pod_name: str) -> dict[str, str]:
    """Load the create-once UID binding projected by the qualifier controller."""
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError("server binding is not projected")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("server binding receipt is malformed")
    expected = "sha256:" + hashlib.sha256(
        json.dumps(
            {key: item for key, item in value.items() if key != "receipt_sha256"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    if value.get("receipt_sha256") != expected:
        raise ValueError("server binding receipt digest drifted")
    required = {
        "api_run_id",
        "head_pod_name",
        "head_pod_uid",
        "service_uid",
        "run_dir",
        "receipt_sha256",
    }
    if (
        not required.issubset(value)
        or value.get("run_dir") != server_run_dir
        or value.get("head_pod_name") != pod_name
    ):
        raise ValueError("server binding receipt identity drifted")
    return {key: str(value[key]) for key in required}


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
    temporary.chmod(0o600)
    os.replace(temporary, path)


def _create_json_once(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-url", default="http://127.0.0.1:8000/metrics")
    parser.add_argument("--state-path", type=Path, required=True)
    parser.add_argument("--receipt-path", type=Path, required=True)
    parser.add_argument("--traffic-path", type=Path, required=True)
    parser.add_argument("--binding-path", type=Path, required=True)
    parser.add_argument("--event-dir", type=Path, required=True)
    parser.add_argument("--server-run-dir", required=True)
    args = parser.parse_args()

    binding = server_binding(args.binding_path, args.server_run_dir, socket.gethostname())

    with urllib.request.urlopen(args.metrics_url, timeout=3) as response:
        metrics = response.read().decode("utf-8")
    after = request_counters(metrics)
    before = [0] * RANKS
    if args.state_path.is_file() and not args.state_path.is_symlink():
        previous = json.loads(args.state_path.read_text())
        if isinstance(previous, dict) and previous.get("request_counters_by_rank") == [
            int(value) for value in previous.get("request_counters_by_rank", [])
        ]:
            before = previous["request_counters_by_rank"]
    if any(after[index] < before[index] for index in range(RANKS)):
        raise SystemExit("request counters decreased")
    _atomic_json(args.state_path, {"request_counters_by_rank": after})
    if after == before:
        return
    raw = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    memory, utilization = gpu_sample(raw)
    now = int(__import__("time").time())
    receipt = observation(
        before,
        after,
        memory_mib=memory,
        utilization_percent=utilization,
        server_run_dir=args.server_run_dir,
        pod_name=socket.gethostname(),
        pod_uid=binding["head_pod_uid"],
        api_run_id=binding["api_run_id"],
        service_uid=binding["service_uid"],
        server_binding_receipt_sha256=binding["receipt_sha256"],
        observed_at_epoch=now,
    )
    if receipt is None:
        return
    event = args.event_dir / f"{now}-{receipt['receipt_sha256'][7:23]}.json"
    _create_json_once(event, receipt)
    _atomic_json(args.receipt_path, receipt)
    args.traffic_path.touch(exist_ok=True)
    os.utime(args.traffic_path, (now, now))


if __name__ == "__main__":
    main()
