"""Entrypoint for the instrumented Qwen G19 v4 successors."""

from __future__ import annotations

import argparse
from pathlib import Path

from evals.fleet import qwen_hosted_generation19_v2_runtime as runtime
from evals.fleet import qwen_hosted_generation19_v4 as g19


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--proxy", type=Path, required=True)
    parser.add_argument("--diagnostic-root", type=Path, required=True)
    args = parser.parse_args()
    plan = g19.load(args.plan)
    runtime.run(
        plan,
        out=args.out,
        proxy=args.proxy,
        diagnostic_root=args.diagnostic_root,
        bulk_module=g19,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
