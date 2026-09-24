"""Check that related task versions cannot leak across data splits."""

from copy import deepcopy
import unittest

from training.splits import apply_splits, assign_split, split_key


def example(version=1, *, app='fira', family='demo', registry='cyber/task-graphs/demo'):
    return {'lineage': {'application': app, 'task_family': family,
                        'lineage_key': f'registry:{registry}@{version}'}}


class SplitTests(unittest.TestCase):
    def test_versions_and_attempts_stay_together(self):
        records = [example(version) for version in range(1, 101)]
        for index, row in enumerate(records):
            row.update(record_id=f'session-{index}', prompt_sha256=str(index))
        apply_splits(records)
        self.assertEqual(len({row['split_unit'] for row in records}), 1)
        self.assertEqual(len({row['split'] for row in records}), 1)

    def test_order_and_subset_do_not_change_assignment(self):
        records = [example(family=f'task-{index}', registry=f'cyber/task-graphs/{index}')
                   for index in range(100)]
        reversed_records = deepcopy(records[::-1])
        subset = deepcopy(records[::3])
        for group in (records, reversed_records, subset):
            apply_splits(group)
        by_unit = {row['split_unit']: row['split'] for row in records}
        self.assertEqual(set(by_unit.values()), {'train', 'dev', 'test'})
        for row in reversed_records + subset:
            self.assertEqual(by_unit[row['split_unit']], row['split'])

    def test_conflicting_family_fails_before_mutation(self):
        records = [example(1), example(2, family='renamed')]
        before = deepcopy(records)
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            apply_splits(records)
        self.assertEqual(records, before)

    def test_missing_identity_and_invalid_ratios_fail(self):
        for field in ('application', 'task_family', 'lineage_key'):
            row = example()
            row['lineage'][field] = None
            with self.assertRaises(ValueError):
                apply_splits([row])
            self.assertNotIn('split', row)
        for train, dev in ((0, 0.1), (0.8, 0.2), (0.8, -0.1), (float('nan'), 0)):
            with self.assertRaises(ValueError):
                assign_split('unit', train=train, dev=dev)

    def test_family_key_has_no_separator_collision(self):
        self.assertNotEqual(split_key(example(app='a|b', family='c')),
                            split_key(example(app='a', family='b|c')))


if __name__ == '__main__':
    unittest.main()
