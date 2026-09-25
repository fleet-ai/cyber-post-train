import hashlib
import json
import unittest

from training.family_roles import ROOT_ID, legacy_projection, roles, seal


def source():
    return {"training_task_keys": [
        {"task_key": "a", "versions": [
            {"task_version_id": "1", "atom_lineages": ["cyber/atoms/app/x@1"],
             "source_sessions": 2, "supervised_tokens": 20},
            {"task_version_id": "2", "atom_lineages": ["cyber/atoms/app/x@2"],
             "source_sessions": 1, "supervised_tokens": 10}]},
        {"task_key": "b", "versions": [
            {"task_version_id": "3", "atom_lineages": ["cyber/atoms/app/x@3"],
             "source_sessions": 1, "supervised_tokens": 10}]},
        {"task_key": "c", "versions": [
            {"task_version_id": "4", "atom_lineages": ["cyber/atoms/app/y@1"],
             "source_sessions": 1, "supervised_tokens": 10}]},
        {"task_key": "d", "versions": [
            {"task_version_id": "5", "atom_lineages": ["cyber/atoms/app/z@1"],
             "source_sessions": 1, "supervised_tokens": 10}]},
    ]}


def protected(atom="cyber/atoms/app/unrelated"):
    return (
        {"tasks": [{"task_key": "heldout", "task_version_id": "old", "split": "final_test"}]},
        {"task_versions": [{"task_key": "heldout", "task_version_id": "old",
                            "lineage": {"task_family": atom + "@0"}}]},
    )


class FamilyRolesTest(unittest.TestCase):
    def test_aliases_stay_in_one_split_and_digest_is_canonical(self):
        split, receipts = protected()
        result = roles(source(), protected_split=split,
                       protected_receipts=receipts, validation_families=1)
        rows = result["identities"]
        self.assertEqual(len(rows), 5)
        self.assertEqual(len({row["family_id"] for row in rows[:3]}), 1)
        self.assertEqual(len({row["split"] for row in rows[:3]}), 1)
        self.assertEqual({row["split"] for row in rows}, {"train", "dev"})
        body = {key: value for key, value in result.items() if key != "sha256"}
        expected = hashlib.sha256(json.dumps(body, sort_keys=True,
                                            separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
        self.assertEqual(result["sha256"], "sha256:" + expected)

    def test_missing_reviewed_lineage_fails(self):
        value = source()
        value["training_task_keys"][0]["versions"][0]["atom_lineages"] = []
        split, receipts = protected()
        with self.assertRaises(ValueError):
            roles(value, protected_split=split,
                  protected_receipts=receipts, validation_families=1)

    def test_exposed_family_is_quarantined_entirely(self):
        split, receipts = protected("cyber/atoms/app/x")
        result = roles(source(), protected_split=split,
                       protected_receipts=receipts, validation_families=1)
        self.assertEqual({row["split"] for row in result["identities"]
                          if row["task_key"] in {"a", "b"}}, {"test"})

    def test_legacy_projection_has_new_bound_root(self):
        source_map = seal(source())
        split, receipts = (seal(value) for value in protected("cyber/atoms/app/x"))
        reviewed = roles(source_map, protected_split=split,
                         protected_receipts=receipts, validation_families=1)
        anchor, roster = legacy_projection(source_map, split, receipts,
                                            reviewed, validation_families=1)
        self.assertEqual(roster["root_role_anchor_id"], ROOT_ID)
        self.assertEqual(roster["family_role_anchor_sha256"], anchor["sha256"])
        self.assertEqual(len(roster["identities"]), 5)
        self.assertEqual({row["split"] for row in roster["identities"]},
                         {"train", "dev", "final_test"})
        self.assertEqual(len(roster["heldout_group_ids"]), 2)
        corrupted = {**reviewed, "identities": reviewed["identities"][:-1]}
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            legacy_projection(source_map, split, receipts, corrupted,
                              validation_families=1)


if __name__ == "__main__":
    unittest.main()
