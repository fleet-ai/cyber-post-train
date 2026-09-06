from pathlib import Path

from evals.fleet import glm53_dedicated_v20 as v20
from evals.fleet import glm53_dedicated_v21 as v21

ROOT = Path(__file__).resolve().parents[1]


def test_v21_changes_only_identity_and_adds_package_canary():
    old = v20.payload(v20.spec(ROOT), ROOT)
    new = v21.payload(v21.spec(ROOT), ROOT)
    assert new["title"] == v21.TITLE
    assert new["run_dir"] == v21.RUN_DIR
    assert new["env"] == {**old["env"], "GLM53_RUN_DIR": v21.RUN_DIR}
    for key in set(old) - {"title", "run_dir", "env"}:
        assert new[key] == old[key]
    assert v21.spec(ROOT)["package_canary"] == v21.PACKAGE_CANARY
