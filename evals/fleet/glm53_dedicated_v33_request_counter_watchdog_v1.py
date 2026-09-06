"""Generation-exact runtime entrypoint for the GLM v33 request watchdog."""

from __future__ import annotations

from evals.fleet import glm53_dedicated_v23_request_counter_watchdog_v1 as engine
from evals.fleet import glm53_dedicated_v33_create_v1 as server

def main() -> int:
    engine.API_RUN_ID_RE = server.API_RUN_ID_RE
    return engine.main()


if __name__ == "__main__":
    raise SystemExit(main())
