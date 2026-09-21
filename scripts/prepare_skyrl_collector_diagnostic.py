"""Prepare, but never preview or create, a dev SkyRL collector diagnostic.

This command intentionally has no Fleet client, kubectl invocation, credentials,
preview switch, or create switch.  It produces the immutable local packet a
separate reviewed one-shot creator must consume only after it obtains a fresh
server preview and arms an exact-identity cleanup observer.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from cyber_post_train.jobs import JobsError, digest
from training import skyrl_collector_diagnostic as diagnostic


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise JobsError("collector diagnostic config is unreadable") from exc
    if not isinstance(value, dict):
        raise JobsError("collector diagnostic config must be an object")
    return value


def _write_once(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def prepare(config_path: Path, output: Path) -> dict[str, Any]:
    """Create one immutable local-only packet, refusing replacement."""
    if output.exists() or output.is_symlink():
        raise JobsError("collector diagnostic packet destination already exists")
    config = _read_object(config_path)
    plan = diagnostic.compile_diagnostic(config, relative_to=config_path.parent)
    request = diagnostic.job_request(plan)
    offline = diagnostic.offline_preview(plan, request)
    observer = diagnostic.release_observer_contract(plan, request)
    output.mkdir(parents=True, mode=0o700)
    try:
        _write_once(output / "PLAN.json", plan)
        _write_once(output / "REQUEST.json", request)
        _write_once(output / "OFFLINE_PREVIEW.json", offline)
        _write_once(output / "RELEASE_OBSERVER_CONTRACT.json", observer)
        manifest = diagnostic._seal(
            {
                "schema": "cyber_skyrl_collector_diagnostic_packet_v1",
                "status": "prepared_locally_not_server_previewed",
                "plan_sha256": "sha256:" + digest(plan),
                "request_sha256": "sha256:" + digest(request),
                "offline_preview_sha256": offline["sha256"],
                "release_observer_contract_sha256": observer["sha256"],
                "preview_authorized": False,
                "create_authorized": False,
                "external_reads": 0,
                "external_mutations": 0,
            }
        )
        _write_once(output / "PACKET.json", manifest)
    except BaseException:
        # Do not attempt cleanup/replacement: a partial packet is evidence that
        # an operator must inspect, never permission to regenerate a request.
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        packet = prepare(args.config, args.output)
        print(json.dumps({"status": packet["status"], "sha256": packet["sha256"]}))
    except BaseException:
        # The config can contain non-public source locations.  Keep CLI errors
        # deliberately generic; the caller can inspect local validation itself.
        print(json.dumps({"status": "failed"}))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
