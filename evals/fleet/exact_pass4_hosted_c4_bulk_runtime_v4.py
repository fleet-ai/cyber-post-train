"""Run one held hosted-c4 controller through the reviewed v3 runtime engine."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from evals.fleet import exact_pass4_bulk_runtime_v3 as runtime
from evals.fleet import exact_pass4_hosted_c4_bulk_v4 as bulk


def _runtime_release_gate_check(plan: dict) -> None:
    path_value = os.environ.get("HOSTED_C4_BULK_RELEASE_PATH")
    expected_file_sha = os.environ.get("HOSTED_C4_BULK_RELEASE_FILE_SHA256")
    package_commit = os.environ.get("HOSTED_C4_BULK_PACKAGE_COMMIT")
    if (
        not path_value
        or not expected_file_sha
        or bulk.SHA_RE.fullmatch(expected_file_sha) is None
        or not package_commit
        or bulk.COMMIT_RE.fullmatch(package_commit) is None
    ):
        raise RuntimeError("hosted-c4 runtime release environment is incomplete")
    path = Path(path_value)
    if not path.is_absolute() or not path.is_relative_to(Path("/mnt/sfs/jobs")):
        raise RuntimeError("hosted-c4 runtime release path is unsafe")
    release = bulk.load(path)
    if bulk.sha256(path.read_bytes()) != expected_file_sha:
        raise RuntimeError("hosted-c4 runtime release file drifted")
    bulk.validate_runtime_release(
        release,
        plan,
        Path(plan["repo_root"]),
        package_commit=package_commit,
    )


def run(plan: dict, *, out: Path, proxy: Path) -> dict:
    # The v3 engine resolves its plan authority through this module-global
    # binding.  Replace it once before any validation, claim, or model call;
    # all runtime mechanics remain byte-identical to the reviewed engine.
    runtime.bulk = bulk
    return runtime.run_controller(
        plan,
        out=out,
        proxy=proxy,
        runtime_gate_check=_runtime_release_gate_check,
    )


def main() -> int:
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
