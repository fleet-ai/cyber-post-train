"""Generation-correct CLI adapter for the reviewed score-free GPU observer."""

from __future__ import annotations

from evals.fleet import glm53_dedicated_v23_scorefree_gpu_observer_v1 as observer
from evals.fleet import glm53_dedicated_v29_scorefree_qualifier_v1 as qualifier


def main() -> int:
    original = observer.qualifier
    with qualifier.bound_engine():
        observer.qualifier = qualifier
        try:
            return observer.main()
        finally:
            observer.qualifier = original


if __name__ == "__main__":
    raise SystemExit(main())
