from pathlib import Path

from evals.fleet import qwen_hosted_generation19_bulk as g19

ROOT = Path(__file__).resolve().parents[1]


def test_partition_shape_is_whole_task_and_held() -> None:
    assert len(g19.ALLOWED_RANKS) == 96
    assert not set(g19.ALLOWED_RANKS) & {2, 3, 4, 5}
    assert {len(row["ranks"]) for row in g19.CONTROLLERS.values()} == {48}
    owners = {
        rank: controller
        for controller, authority in g19.CONTROLLERS.items()
        for rank in authority["ranks"]
    }
    assert set(owners) == set(g19.ALLOWED_RANKS)


def test_plan_builder_requires_live_inventory_fixture() -> None:
    # The paid renderer consumes the accepted SFS inventory, while this unit
    # test fixes the immutable source campaign and partition at review time.
    assert (
        ROOT / "evals/fleet/configs/q38-glm53-exact-easiest100-pass4-campaign-v1.json"
    ).is_file()
