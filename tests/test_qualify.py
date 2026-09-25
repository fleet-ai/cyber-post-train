"""Transitive task/atom lineage is the split boundary, not a task-key label."""

import unittest

from training.qualify import components, metadata_only


def row(key, version, *atoms):
    return {"task_key": key, "task_version_id": version,
            "atom_artifact_keys": [f"cyber/atoms/app/{atom}" for atom in atoms]}


class LineageTests(unittest.TestCase):
    def test_transitive_atoms_and_same_key_versions_are_one_family(self):
        document = {"task_versions": [row("a", "v1", "x"), row("a", "v2", "y"),
                                       row("b", "v1", "y", "z"), row("c", "v1", "z")]}
        labels, known = components(document)
        self.assertEqual(len(known), 4)
        self.assertEqual(len(set(labels.values())), 1)

    def test_atom_versions_share_family_but_conflicting_exact_version_rejects(self):
        left = {"task_versions": [row("a", "v1", "x@1"), row("b", "v1", "x@2")]}
        labels, _ = components(left)
        self.assertEqual(labels["a", "v1"], labels["b", "v1"])
        with self.assertRaisesRegex(ValueError, "conflicting atom lineage"):
            components(left, {"task_versions": [row("a", "v1", "other")]})

    def test_private_fields_and_unknown_atoms_reject(self):
        with self.assertRaisesRegex(ValueError, "private task"):
            metadata_only({"nested": [{"prompt": "must stay private"}]})
        with self.assertRaisesRegex(ValueError, "reviewed atom"):
            components({"task_versions": [row("a", "v1", "unbound").__or__({
                "atom_artifact_keys": ["not-an-atom"]})]})


if __name__ == "__main__":
    unittest.main()
