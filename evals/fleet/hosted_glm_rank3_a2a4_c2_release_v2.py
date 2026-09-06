"""Fresh release identity for the corrected rank-3 tail successor."""

from pathlib import Path

from evals.fleet import exact_pass4_bulk_runtime_v3 as engine
from evals.fleet import hosted_glm_rank3_a2a4_c2_release_v1 as prior
from evals.fleet import hosted_glm_rank3_a2a4_c2_runtime_v2 as runtime
from evals.fleet import hosted_glm_rank3_a2a4_c2_successor_v2 as successor

JOB_NAME = "chris-glm53-exact100-hosted-r003-a2a4-release-v2"
CONFIGMAP_NAME = JOB_NAME + "-run"
OUTPUT_PATH = Path("/mnt/sfs/jobs") / JOB_NAME / "RELEASE.json"


def main() -> int:
    receipt = prior.build(
        Path.cwd(),
        target=successor,
        runtime_module=runtime,
        failed_identity=runtime.FAILED_V1,
    )
    OUTPUT_PATH.parent.mkdir(mode=0o700, parents=True, exist_ok=False)
    engine._write_once(OUTPUT_PATH, receipt)  # noqa: SLF001
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
