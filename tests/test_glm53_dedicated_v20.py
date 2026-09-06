from pathlib import Path
from evals.fleet import glm53_dedicated_v19 as v19
from evals.fleet import glm53_dedicated_v20 as v20
ROOT=Path(__file__).resolve().parents[1]
def test_v20_changes_only_identity_from_v19():
 old=v19.payload(v19.spec(ROOT),ROOT); new=v20.payload(v20.spec(ROOT),ROOT)
 assert new["title"]==v20.TITLE and new["run_dir"]==v20.RUN_DIR
 assert new["env"]=={**old["env"],"GLM53_RUN_DIR":v20.RUN_DIR}
 for k in set(old)-{"title","run_dir","env"}: assert new[k]==old[k]
