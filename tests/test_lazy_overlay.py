"""Synthetic-only proof of the optional dense row-group reader."""

import hashlib
import json
import runpy
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from training import launch, lazy_overlay
from training.dense_bridge import LAYOUT_BUILDER_SHA, stage_historical
from training.lazy_overlay import (DenseRowGroups, FROZEN, LAYOUT, PATCHED,
                                   patch_frozen, verify_complete_sessions)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _rows():
    rows = []
    for source in range(2):
        for target in range(2):
            ids = [10, 11, 12 + source, 14 + target]
            start = 2 + target
            mask = [int(i == start) for i in range(len(ids))]
            rows.append({"source_session_id": f"source-{source}", "task_key": f"task-{source}",
                         "window_id": f"window-{source}-{target}", "input_ids": ids,
                         "loss_mask": mask, "token_count": len(ids), "target_token_count": 1,
                         "source_assistant_count": 2, "eligible_assistant_indices": [0, 1],
                         "excluded_assistant_targets": [], "copied_context_assistant_indices": [],
                         "target_spans": [{"assistant_index": target,
                                           "source_message_index": 2 + 2 * target,
                                           "token_start": start, "token_end": start + 1,
                                           "source_target_sha256": "sha256:" + _digest(ids[start:start + 1])}]})
    return rows


def _spec(rows):
    return {"storage_layout": LAYOUT, "format": "pretokenized_assistant_segments_v1",
            "rows": len(rows), "task_keys": ["task-0", "task-1"], "source_sessions": 2,
            "supervised_tokens": len(rows), "assistant_responses": len(rows),
            "source_total_assistant_responses": 4, "excluded_assistant_responses": 0}


class LazyOverlayTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        stage_historical(self.root / "frozen")
        frozen = runpy.run_path(str(self.root / "frozen/training/sft_runtime.py"))
        self.dense_rows, self.prepare_rows = frozen["dense_rows"], frozen["prepare_rows"]

    def test_exact_source_patch_and_single_row_group_builder_are_distinct(self):
        for name in FROZEN:
            original = launch.historical_source(name)
            patched = launch.historical_source(name, lazy=True)
            self.assertEqual(patched, patch_frozen(name, original))
            self.assertEqual(hashlib.sha256(patched).hexdigest(), PATCHED[name])
            self.assertNotEqual(patched, original)
            compile(patched, name, "exec")
            with self.assertRaises(ValueError):
                patch_frozen(name, original + b"\n")
        self.assertEqual(stage_historical(self.root / "layout", target_names=True,
                                          single_row_groups=True), LAYOUT_BUILDER_SHA)
        self.assertNotEqual(stage_historical(self.root / "ordinary"), LAYOUT_BUILDER_SHA)

    def test_synthetic_compiler_embeds_bound_lazy_runtime(self):
        fixture = runpy.run_path(str(Path(__file__).with_name("test_launch.py")))
        manifest = fixture["fake_full_manifest"]()
        manifest["files"]["train"]["storage_layout"] = LAYOUT
        manifest["sha256"] = "sha256:" + launch.sha(launch.canonical(
            {key: value for key, value in manifest.items() if key != "sha256"}))
        config = json.loads((Path(__file__).parents[1] / "configs/runs/qwen38-96k-full-v1.json").read_text())
        config["model"]["lock"] = "../models/qwen38-27b-1d4bf0f2.lock.json"
        config["model"]["weights"] = "../models/qwen38-27b-1d4bf0f2.weights.json"
        config["data"]["manifest"] = "../data/corpus.json"
        config["recipe"]["keep_checkpoints"] = 102
        result = launch._legacy("compile", {"config": config},
                                manifest=launch.canonical(manifest), lazy=True)
        self.assertEqual(result["plan"]["runtime_sha256"], PATCHED["training/sft_runtime.py"])
        self.assertEqual(result["plan"]["lazy_overlay_sha256"], hashlib.sha256(
            Path(lazy_overlay.__file__).read_bytes()).hexdigest())
        self.assertFalse(result["request"]["failureAlerts"])
        runtime_path = self.root / "patched_runtime.py"
        runtime_path.write_bytes(launch.historical_source("training/sft_runtime.py", lazy=True))
        options = runpy.run_path(str(runtime_path))["sft_overrides"](result["plan"])
        self.assertEqual(options["dataloader_num_workers"], 0)

    def write(self, rows, *, group_size=1):
        path = self.root / "train.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=group_size)
        return path

    def test_streamed_complete_sessions_match_frozen_dense_rows(self):
        rows = _rows()
        data = DenseRowGroups(self.write(rows), _spec(rows), max_length=8)
        self.assertEqual(verify_complete_sessions(data, self.dense_rows, 100)["supervised_tokens"], 4)
        self.assertEqual(data.sequence_lengths, [4] * 4)
        self.assertEqual([data[i]["window_id"] for i in range(len(data))],
                         [row["window_id"] for row in rows])
        self.assertEqual([data[i]["loss_mask"] for i in range(len(data))],
                         [[1, 0], [1], [1, 0], [1]])
        self.assertEqual([data[i] for i in range(len(data))], self.prepare_rows(
            rows, _spec(rows), list(range(100)), None, max_length=8))

    def test_corruption_or_incomplete_session_fails_closed(self):
        rows = _rows()
        rows[1]["loss_mask"] = [0, 0, 1, 0]
        data = DenseRowGroups(self.write(rows), _spec(rows), 8)
        with self.assertRaises(ValueError):
            verify_complete_sessions(data, self.dense_rows, 100)
        oversized = [{**_rows()[0], "window_id": f"synthetic-{i}"} for i in range(129)]
        spec = {**_spec(oversized), "source_sessions": 1, "task_keys": ["task-0"]}
        with self.assertRaisesRegex(ValueError, "memory envelope"):
            verify_complete_sessions(DenseRowGroups(self.write(oversized), spec, 8), self.dense_rows, 100)
        rows = _rows()[:-1]
        data = DenseRowGroups(self.write(rows), _spec(rows), 8)
        with self.assertRaises(ValueError):
            verify_complete_sessions(data, self.dense_rows, 100)

    def test_multirow_group_rejected_and_seeded_resume_stable(self):
        rows = _rows()
        with self.assertRaisesRegex(ValueError, "layout"):
            DenseRowGroups(self.write(rows, group_size=2), _spec(rows), 8)
        data = DenseRowGroups(self.write(rows), _spec(rows), 8)
        try:
            import torch
            from torchdata.stateful_dataloader import StatefulDataLoader
        except ImportError:
            self.skipTest("pinned torchdata is unavailable locally")

        def loader():
            generator = torch.Generator().manual_seed(17)
            return StatefulDataLoader(data, batch_size=1, shuffle=True, generator=generator,
                                      num_workers=0, collate_fn=lambda batch: batch[0]["window_id"])

        first = loader()
        iterator = iter(first)
        next(iterator)
        state = first.state_dict()
        remaining = list(iterator)
        resumed = loader()
        resumed.load_state_dict(state)
        self.assertEqual(list(resumed), remaining)


if __name__ == "__main__":
    unittest.main()
