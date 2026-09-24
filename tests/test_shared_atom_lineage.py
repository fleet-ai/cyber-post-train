import pytest

from training import shared_atom_lineage as lineage


def row(key: str, version: str, *atoms: str) -> dict:
    return {"task_key": key, "task_version_id": version, "atom_artifact_keys": list(atoms)}


def test_successor_versions_and_composites_form_transitive_components() -> None:
    result = lineage.build_components(
        [
            row("single-a", "v1", "cyber/atoms/app/a@0:atom_source"),
            row("single-a", "v2", "cyber/atoms/app/a@3:atom_source"),
            row("composite", "v1", "cyber/atoms/app/a", "cyber/atoms/app/b"),
            row("single-b", "v1", "cyber/atoms/app/b@9:atom_source"),
            row("separate", "v1", "cyber/atoms/app/c"),
        ]
    )
    assert len(result["components"]) == 2
    joined = next(
        component for component in result["components"] if len(component["task_versions"]) == 4
    )
    assert joined["atom_artifact_keys"] == ["cyber/atoms/app/a", "cyber/atoms/app/b"]


def test_same_task_key_versions_cannot_escape_when_atom_metadata_changes() -> None:
    result = lineage.build_components(
        [row("same", "v1", "cyber/atoms/app/a"), row("same", "v2", "cyber/atoms/app/b")]
    )
    assert len(result["components"]) == 1


def test_component_roles_reject_a_train_heldout_bridge() -> None:
    components = lineage.build_components(
        [
            row("train", "v1", "cyber/atoms/app/a"),
            row("bridge", "v1", "cyber/atoms/app/a", "cyber/atoms/app/b"),
            row("test", "v1", "cyber/atoms/app/b"),
        ]
    )
    with pytest.raises(ValueError, match="crosses frozen split roles"):
        lineage.component_roles(
            components, {("train", "v1"): "train", ("test", "v1"): "final_test"}
        )


@pytest.mark.parametrize(
    "value",
    [
        "",
        "cyber/atoms/app",
        "cyber/atoms/app/a@x:atom_source",
        "cyber/atoms/app/a@0",
        "cyber/atoms/app/a@junk@1:atom_source",
        "cyber/atoms/app/a:bad@1:atom_source",
    ],
)
def test_invalid_atom_identity_is_rejected(value: str) -> None:
    with pytest.raises(ValueError):
        lineage.atom_artifact_key(value)
