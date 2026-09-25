import hashlib
import json
import unittest

from training.family_roles import roles


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
    ]}


class FamilyRolesTest(unittest.TestCase):
    def test_aliases_stay_in_one_split_and_digest_is_canonical(self):
        result = roles(source(), validation_families=1)
        rows = result["identities"]
        self.assertEqual(len(rows), 4)
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
        with self.assertRaises(ValueError):
            roles(value, validation_families=1)


if __name__ == "__main__":
    unittest.main()
