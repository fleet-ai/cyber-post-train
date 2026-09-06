"""Fresh rank-30 observer identity with the reviewed v1 gate unchanged."""

from __future__ import annotations

from pathlib import Path

from evals.fleet import hosted_glm_rank30_single_slot_release_v1 as release

JOB_NAME = "chris-glm53-exact100-hosted-r030-single-slot-release-v2"
CONFIGMAP_NAME = JOB_NAME + "-run"
OUTPUT_PATH = Path("/mnt/sfs/jobs") / JOB_NAME / "RELEASE.json"


def main() -> int:
    receipt = release.build(Path.cwd())
    OUTPUT_PATH.parent.mkdir(mode=0o700, parents=True, exist_ok=False)
    release.engine._write_once(OUTPUT_PATH, receipt)  # noqa: SLF001
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
