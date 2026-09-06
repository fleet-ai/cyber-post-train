"""UID-bound GPU observer adapter for the GLM v23 score-free qualifier."""

from evals.fleet import glm53_dedicated_v22_concurrency_gpu_observer_v1 as observer
from evals.fleet import glm53_dedicated_v23_scorefree_qualifier_v1 as qualifier

# The reviewed observer implementation is generation-neutral apart from this
# contract module. Bind it explicitly rather than copying operational logic.
observer.qualification = qualifier


def main() -> int:
    return observer.main()


if __name__ == "__main__":
    raise SystemExit(main())
