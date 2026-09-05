"""Execute one released final-v5 controller through the reviewed v3 engine."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import httpx

from evals.fleet import exact_pass4_bulk_runtime_v3 as engine
from evals.fleet import exact_pass4_final_bulk_v5 as bulk


def _runtime_release_gate(plan: dict) -> None:
    path = Path(os.environ.get("FINAL_BULK_RELEASE_PATH", ""))
    expected_sha = os.environ.get("FINAL_BULK_RELEASE_FILE_SHA256", "")
    package_commit = os.environ.get("FINAL_BULK_PACKAGE_COMMIT", "")
    if (
        path != Path("/bootstrap/release.json")
        or bulk.SHA_RE.fullmatch(expected_sha) is None
        or bulk.COMMIT_RE.fullmatch(package_commit) is None
        or path.is_symlink()
        or not path.is_file()
        or bulk.sha256(path.read_bytes()) != expected_sha
    ):
        raise RuntimeError("final v5 release environment drifted")
    release = bulk.load(path)
    if release.get("package_commit") != package_commit:
        raise RuntimeError("final v5 release package commit drifted")
    bulk.validate_runtime_release(release, plan, Path(plan["repo_root"]))


def _route_check(plan: dict, key: str) -> None:
    if plan["serving_kind"] == "hosted":
        engine._fresh_route_check(plan, key)  # noqa: SLF001 - unchanged reviewed check
        return
    origin = plan["model"]["endpoint_origin"]
    with httpx.Client(base_url=origin, timeout=30) as client:
        response = client.get("/v1/models")
        response.raise_for_status()
        value = response.json()
    rows = value.get("data") if isinstance(value, dict) else None
    matches = [row for row in rows or [] if isinstance(row, dict) and row.get("id") == "glm-5.3"]
    if len(matches) != 1:
        raise RuntimeError("final v5 dedicated model identity drifted")
    context = [
        matches[0].get(field)
        for field in ("context_length", "max_model_len", "max_context_length")
        if matches[0].get(field) is not None
    ]
    if context and any(type(value) is not int or value != 262144 for value in context):
        raise RuntimeError("final v5 dedicated context drifted")


def run(plan: dict, *, out: Path, proxy: Path) -> dict:
    engine.bulk = bulk
    # No lifecycle heartbeat or model traffic is allowed until the immutable
    # group release has been revalidated inside the controller Pod.
    _runtime_release_gate(plan)
    heartbeat: subprocess.Popen | None = None
    if plan["serving_kind"] == "dedicated":
        stream = plan.get("stream") or 1
        heartbeat = subprocess.Popen(
            [
                str(Path(plan["repo_root"]) / bulk.dedicated.HEARTBEAT_PATH),
                "watch",
                plan["replica"],
                str(stream),
                str(os.getpid()),
            ]
        )
    try:
        return engine.run_controller(
            plan,
            out=out,
            proxy=proxy,
            runtime_gate_check=lambda _plan: None,
            route_check=_route_check,
        )
    finally:
        if heartbeat is not None:
            heartbeat.terminate()
            heartbeat.wait(timeout=10)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--proxy", type=Path, required=True)
    args = parser.parse_args()
    run(bulk.load(args.plan), out=args.out, proxy=args.proxy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
