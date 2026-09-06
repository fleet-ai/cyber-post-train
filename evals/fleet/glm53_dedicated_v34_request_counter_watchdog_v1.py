"""Generation-exact runtime entrypoint for the GLM v34 request watchdog."""

from __future__ import annotations

import re

from evals.fleet import glm53_dedicated_v23_request_counter_watchdog_v1 as engine

API_RUN_ID_RE = re.compile(r"glm53-tp8-v34-[0-9a-f]{8}")


def main() -> int:
    engine.API_RUN_ID_RE = API_RUN_ID_RE
    return engine.main()


if __name__ == "__main__":
    raise SystemExit(main())
