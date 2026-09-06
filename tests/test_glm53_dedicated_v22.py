from pathlib import Path

from evals.fleet import glm53_dedicated_v21 as v21
from evals.fleet import glm53_dedicated_v22 as v22
from evals.fleet import glm53_dedicated_v22_live as live

ROOT = Path(__file__).resolve().parents[1]


def test_v22_changes_only_identity_and_adds_generation_qualification() -> None:
    old = v21.payload(v21.spec(ROOT), ROOT)
    new = v22.payload(v22.spec(ROOT), ROOT)
    assert new["title"] == v22.TITLE
    assert new["run_dir"] == v22.RUN_DIR
    assert new["env"] == {**old["env"], "GLM53_RUN_DIR": v22.RUN_DIR}
    for key in set(old) - {"title", "run_dir", "env"}:
        assert new[key] == old[key]
    assert v22.spec(ROOT)["generation_qualification"] == v22.GENERATION_QUALIFICATION


def test_generation_qualification_is_byte_and_self_digest_valid() -> None:
    value = live._generation_qualification(ROOT)
    assert value["receipt_sha256"] == v22.GENERATION_QUALIFICATION["receipt_sha256"]
