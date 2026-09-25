"""Optional, one-row-group-per-window reader for sealed dense SFT data.

The ordinary eager loader remains the default.  A CPU preflight must call
``verify_complete_sessions`` before this map dataset is used for training.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

LAYOUT = "dense_single_row_group_v1"
FROZEN = {"training/sft.py": "447dcaac2b610c1b6c124a13e7d541edc4c3145d26d8ff31577d75b37d8dd67d",
          "training/sft_runtime.py": "8cb671f377d089e1303248e237f386c256b212b15e41dadc07ac49cf2695ee17"}
PATCHED = {"training/sft.py": "dd16b7084e0a0c9c6fe1b3aad909f13de12fb4e8b33979b713a6f9a11f152722",
           "training/sft_runtime.py": "25c9c7150b9ccc3806d71ec00a22adc909c6dc843ecc3bd39b31aba3167798e3"}


def require_pin(plan: dict) -> None:
    if (plan["datasets"]["train"].get("storage_layout") != LAYOUT
            or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != plan.get("lazy_overlay_sha256")):
        raise ValueError("lazy runtime bytes or layout differ from compiled plan")


def patch_frozen(name: str, payload: bytes) -> bytes:
    """Exact-source, exact-result patch; ordinary staged runtime is unchanged."""
    if name not in FROZEN or hashlib.sha256(payload).hexdigest() != FROZEN[name]:
        raise ValueError("unreviewed frozen SFT source")
    if name.endswith("/sft.py"):
        edits = (
            ('    if "checkpoint_recovery_horizon_seconds" in config:',
             '    if datasets["train"].get("storage_layout") == "dense_single_row_group_v1":\n'
             '        plan["lazy_overlay_sha256"] = hashlib.sha256(\n'
             '            Path(__file__).with_name("lazy_overlay.py").read_bytes()).hexdigest()\n'
             '    if "checkpoint_recovery_horizon_seconds" in config:'),
            ('    if plan.get("model", {}).get("repo") == "zai-org/GLM-5.3" and "lora" in plan:',
             '    if plan["datasets"]["train"].get("storage_layout") == "dense_single_row_group_v1":\n'
             '        helper = Path(__file__).with_name("lazy_overlay.py").read_bytes()\n'
             '        if hashlib.sha256(helper).hexdigest() != plan.get("lazy_overlay_sha256"):\n'
             '            raise ValueError("lazy helper differs from compiled plan")\n'
             '        contents.setdefault("extra_files", {}).update({\n'
             '            "training/__init__.py": "", "training/lazy_overlay.py": helper.decode()})\n'
             '    if plan.get("model", {}).get("repo") == "zai-org/GLM-5.3" and "lora" in plan:'),
            ('        prepare_rows,\n        validate_runtime_sources,',
             '        prepare_rows,\n        dense_rows,\n        validate_runtime_sources,'),
            ('        rows = prepare_rows(\n            pq.read_table(spec["path"]).to_pylist(),',
             '        if split == "train" and spec.get("storage_layout") == "dense_single_row_group_v1":\n'
             '            from .lazy_overlay import DenseRowGroups, require_pin, verify_complete_sessions\n'
             '            require_pin(plan)\n'
             '            rows = DenseRowGroups(Path(spec["path"]), spec, plan["recipe"]["max_length"])\n'
             '            counts[split] = verify_complete_sessions(rows, dense_rows, len(tokenizer))\n'
             '            prepared_rows[split] = rows\n'
             '            continue\n'
             '        rows = prepare_rows(\n            pq.read_table(spec["path"]).to_pylist(),'),
        )
    else:
        edits = (
            ('    train = plan["datasets"]["train"]\n    dev = plan["datasets"].get("dev")',
             '    train = plan["datasets"]["train"]\n    dev = plan["datasets"].get("dev")\n'
             '    if (train.get("storage_layout") not in (None, "dense_single_row_group_v1")\n'
             '            or (dev and dev.get("storage_layout") is not None)\n'
             '            or (train.get("storage_layout") == "dense_single_row_group_v1" and not\n'
             '                re.fullmatch(r"[a-f0-9]{64}", plan.get("lazy_overlay_sha256", "")))):\n'
             '        raise ValueError("unreviewed dense storage layout")'),
            ('            rows = pq.read_table(spec["path"]).to_pylist()\n            return prepare_rows(',
             '            if split == "train" and spec.get("storage_layout") == "dense_single_row_group_v1":\n'
             '                from training.lazy_overlay import DenseRowGroups, require_pin\n'
             '                require_pin(self.plan)\n'
             '                return DenseRowGroups(Path(spec["path"]), spec, self.sft_cfg.max_length)\n'
             '            rows = pq.read_table(spec["path"]).to_pylist()\n            return prepare_rows('),
            ('        def load_dataset(self):',
             '        def _log_dataset_stats(self, tokenized):\n'
             '            if self.plan["datasets"]["train"].get("storage_layout") == "dense_single_row_group_v1":\n'
             '                from training.lazy_overlay import DenseRowGroups\n'
             '                if not isinstance(tokenized, DenseRowGroups):\n'
             '                    raise ValueError("lazy train loader returned another dataset")\n'
             '                tokenized = ({"input_ids": range(n)} for n in tokenized.sequence_lengths)\n'
             '            return super()._log_dataset_stats(tokenized)\n\n'
             '        def load_dataset(self):'),
        )
    text = payload.decode()
    for before, after in edits:
        if text.count(before) != 1:
            raise ValueError("frozen SFT source lacks unique lazy overlay patch point")
        text = text.replace(before, after, 1)
    result = text.encode()
    if hashlib.sha256(result).hexdigest() != PATCHED[name]:
        raise ValueError("lazy overlay source patch digest differs")
    return result


class DenseRowGroups:
    """Map-style dataset; SkyRL's seeded StatefulDataLoader owns ordering/resume."""

    def __init__(self, path: Path, spec: dict, max_length: int):
        import pyarrow.parquet as pq

        self.file = pq.ParquetFile(path)
        self.spec = spec
        self.max_length = max_length
        if (spec.get("storage_layout") != LAYOUT
                or self.file.metadata.num_rows != spec["rows"]
                or self.file.num_row_groups != spec["rows"]
                or any(self.file.metadata.row_group(i).num_rows != 1
                       for i in range(self.file.num_row_groups))):
            raise ValueError("dense single-row-group layout differs")
        self.sequence_lengths = pq.read_table(path, columns=["token_count"])["token_count"].to_pylist()
        if (len(self.sequence_lengths) != spec["rows"]
                or any(type(n) is not int or not 1 < n <= max_length for n in self.sequence_lengths)):
            raise ValueError("dense scalar token inventory differs")

    def __len__(self):
        return self.spec["rows"]

    def raw(self, index: int) -> dict:
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError(index)
        rows = self.file.read_row_group(index).to_pylist()
        if len(rows) != 1 or rows[0]["token_count"] != self.sequence_lengths[index]:
            raise ValueError("dense row-group payload differs")
        return rows[0]

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        row = self.raw(index)
        ids, mask = row["input_ids"], row["loss_mask"]
        if (len(ids) != len(mask) or len(ids) != row["token_count"]
                or not mask or mask[0] != 0 or 1 not in mask
                or sum(mask) != row["target_token_count"]):
            raise ValueError("dense row-group target mask differs")
        first = mask.index(1)
        return {"input_ids": ids, "attention_mask": [1] * len(ids),
                "num_actions": len(ids) - first, "loss_mask": mask[first:],
                "task_key": row["task_key"], "window_id": row["window_id"]}


def verify_complete_sessions(dataset: DenseRowGroups, dense_rows, vocab_size: int) -> dict:
    """Stream bounded complete source sessions through the frozen exact validator."""
    seen_sources, seen_windows, task_keys = set(), set(), set()
    totals = {key: 0 for key in ("rows", "supervised_tokens", "assistant_responses",
                                   "source_sessions", "source_total_assistant_responses",
                                   "excluded_assistant_responses")}
    group: list[dict] = []

    def complete() -> None:
        if not group:
            return
        first = group[0]
        local = {"format": dataset.spec["format"], "rows": len(group),
                 "task_keys": [first["task_key"]], "source_sessions": 1,
                 "supervised_tokens": sum(row["target_token_count"] for row in group),
                 "assistant_responses": sum(len(row["target_spans"]) for row in group),
                 "source_total_assistant_responses": first["source_assistant_count"],
                 "excluded_assistant_responses": len(first["excluded_assistant_targets"])}
        native = dense_rows(group, local, max_length=dataset.max_length, vocab_size=vocab_size)
        if len(native) != len(group) or any(dataset[i] != item for i, item in
                                            zip(range(totals["rows"], totals["rows"] + len(group)), native)):
            raise ValueError("lazy row adaptation differs from frozen dense validator")
        for key in totals:
            totals[key] += local[key]
        task_keys.add(first["task_key"])
        seen_sources.add(first["source_session_id"])

    for index in range(len(dataset)):
        row = dataset.raw(index)
        source = row["source_session_id"]
        if group and source != group[0]["source_session_id"]:
            complete()
            group.clear()
        if source in seen_sources or row["window_id"] in seen_windows:
            raise ValueError("dense source or window is repeated non-contiguously")
        if len(group) >= 128 or sum(r["token_count"] for r in group) + row["token_count"] > 12_582_912:
            raise ValueError("source session exceeds bounded CPU preflight memory envelope")
        seen_windows.add(row["window_id"])
        group.append(row)
    complete()
    if (any(totals[key] != dataset.spec[key] for key in totals)
            or task_keys != set(dataset.spec["task_keys"])):
        raise ValueError("streamed dense source inventory differs")
    return {"rows": totals["rows"], "tasks": len(task_keys),
            "supervised_tokens": totals["supervised_tokens"]}
