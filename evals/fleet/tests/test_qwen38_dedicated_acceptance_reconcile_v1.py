from pathlib import Path

from evals.fleet import qwen38_dedicated_acceptance_reconcile_v1 as reconcile
from evals.fleet import qwen38_dedicated_scored_canary_v1 as canary

ROOT = Path(__file__).parents[3]


def test_reconciler_is_exactly_read_only_against_fleet() -> None:
    source = Path(reconcile.__file__).read_text()
    assert "canary._classify" in source
    assert '"fleet_api_mutations": 0' in source
    assert '"ACCEPTED.json"' in source and '"TERMINAL.json"' in source
    assert "legacy_list_fields_may_be_null_but_never_mismatched_v1" in source
    assert "self_hosted._request" not in source
    assert "POST" not in source and "DELETE" not in source and "PATCH" not in source


def test_historical_source_plan_does_not_mutate_current_plan_authority() -> None:
    historical = reconcile.historical_source_plan(ROOT)
    current = canary.build_plan(ROOT, 1)
    assert historical["config"]["execution"]["network"] == reconcile.HISTORICAL_NETWORK
    assert historical["plan_sha256"] == (
        "sha256:5358ae8d0c81fd18d815f5289eabf771274101e099c799c49a85d0713687aa67"
    )
    assert historical["plan_sha256"] != current["plan_sha256"]
    assert current["config"]["execution"]["network"] == current["item"]["run_id"]
