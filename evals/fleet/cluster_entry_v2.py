"""Versioned cluster entrypoint for the Fleet evaluator v2 runtime closure."""

from __future__ import annotations

import argparse
import json
from typing import Any

from evals.fleet import cluster_entry as v1
from evals.fleet import evaluate_v2


def execute(args: argparse.Namespace) -> dict[str, Any]:
    """Run the v1 lifecycle with the exact v2 runtime identity and image probe."""

    previous_runtime_identity = v1.evaluate.runtime_identity
    previous_check_images = v1.evaluate.check_images
    v1.evaluate.runtime_identity = evaluate_v2.runtime_identity
    v1.evaluate.check_images = evaluate_v2.check_images
    try:
        return v1.execute(args)
    finally:
        v1.evaluate.runtime_identity = previous_runtime_identity
        v1.evaluate.check_images = previous_check_images


def parse_args() -> argparse.Namespace:
    return v1.parse_args()


if __name__ == "__main__":
    print(json.dumps(execute(parse_args()), sort_keys=True))
