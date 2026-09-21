#!/usr/bin/env python3
"""Print a no-credential, no-workload Jobs API alert-opt-out rollout check."""

from __future__ import annotations

import argparse
import json

from cyber_post_train.jobs import API_URLS
from cyber_post_train.jobs_api_readiness import inspect_failure_alert_rollout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster", choices=sorted(API_URLS), default="prod")
    args = parser.parse_args()
    print(json.dumps(inspect_failure_alert_rollout(API_URLS[args.cluster]), sort_keys=True))


if __name__ == "__main__":
    main()
