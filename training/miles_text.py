"""Native Miles cursor/checkpoints, with an explicitly text-only input loader.

Qwen's AutoProcessor is multimodal even for our text-only checkpoint recipe.
Its presence must not reformat already-rendered prompts or add vision inputs.
"""

import copy
from pathlib import Path

from miles.rollout.data_source import RolloutDataSource
from miles.utils.data import Dataset
from miles.utils.processing_utils import load_tokenizer


class TextDataSource(RolloutDataSource):
    def __init__(self, args):
        if (
            not args.rollout_global_dataset
            or args.apply_chat_template
            or args.multimodal_keys is not None
            or args.tool_key not in (None, "tools")
            or args.label_key is not None
            or args.input_key != "input"
            or args.metadata_key != "metadata"
        ):
            raise ValueError("Fleet RL requires rendered text and separate task metadata")
        # Let native Miles initialize and own its cursor/save/load implementation.
        # Native tool_key defaults to "tools" but only affects chat templating,
        # which is disabled here; already-rendered inputs need no tools column.
        empty = copy.copy(args)
        empty.rollout_global_dataset = False
        super().__init__(empty)
        self.args = args
        self.tokenizer = load_tokenizer(
            args.hf_checkpoint, chat_template_path=args.chat_template_path, trust_remote_code=False
        )
        self.dataset = Dataset(
            args.prompt_data,
            tokenizer=self.tokenizer,
            processor=None,
            max_length=args.rollout_max_prompt_len,
            prompt_key=args.input_key,
            metadata_key=args.metadata_key,
            apply_chat_template=False,
            seed=args.rollout_seed,
        )
        if args.rollout_shuffle:
            self.dataset.shuffle(self.epoch_id)

    def _expected_cursor(self, rollout_id):
        groups, samples, rows = (
            self.args.rollout_batch_size,
            self.args.n_samples_per_prompt,
            len(self.dataset),
        )
        if (
            type(rollout_id) is not int
            or rollout_id < 0
            or type(groups) is not int
            or not 0 < groups <= rows
            or type(samples) is not int
            or samples < 2
        ):
            raise ValueError("invalid native checkpoint cursor dimensions")
        consumed = (rollout_id + 1) * groups
        # Native Miles advances epochs on the *next* read after an exact end.
        return {
            "sample_offset": (consumed - 1) % rows + 1,
            "epoch_id": (consumed - 1) // rows,
            "sample_group_index": consumed,
            "sample_index": consumed * samples,
            "metadata": {},
        }

    @staticmethod
    def _check_cursor(actual, expected):
        if (
            not isinstance(actual, dict)
            or actual != expected
            or any(type(actual.get(k)) is not int for k in expected if k != "metadata")
        ):
            raise ValueError("native checkpoint cursor does not match completed rollout batches")

    def _read_cursor(self, root, rollout_id):
        import torch

        path = Path(root) / f"rollout/global_dataset_state_dict_{rollout_id}.pt"
        if (
            path.resolve() != path.absolute()
            or not path.is_file()
            or not 0 < path.stat().st_size <= 1048576
        ):
            raise ValueError("native checkpoint cursor is missing, unsafe or incomplete")
        return torch.load(path, map_location="cpu", weights_only=True)

    def save(self, rollout_id):
        expected = self._expected_cursor(rollout_id)
        self._check_cursor({k: getattr(self, k) for k in expected}, expected)
        super().save(rollout_id)
        self._check_cursor(self._read_cursor(self.args.save, rollout_id), expected)

    def load(self, rollout_id=None):
        if type(rollout_id) is int and rollout_id == -1:
            # The only legal missing cursor is the reviewed initial base load.
            if (
                type(self.args.start_rollout_id) is not int
                or self.args.start_rollout_id != 0
                or not all(
                    getattr(self.args, k, False) is True
                    for k in ("finetune", "no_load_optim", "no_load_rng")
                )
            ):
                raise ValueError("missing cursor allowed only for an explicit initial base load")
            self._check_cursor(
                {
                    k: getattr(self, k)
                    for k in (
                        "sample_offset",
                        "epoch_id",
                        "sample_group_index",
                        "sample_index",
                        "metadata",
                    )
                },
                {
                    "sample_offset": 0,
                    "epoch_id": 0,
                    "sample_group_index": 0,
                    "sample_index": 0,
                    "metadata": {},
                },
            )
            return
        expected = self._expected_cursor(rollout_id)
        if any(getattr(self.args, k, False) for k in ("finetune", "no_load_optim", "no_load_rng")):
            raise ValueError("cursor recovery requires optimizer and RNG recovery, not finetuning")
        self._check_cursor(self._read_cursor(self.args.load, rollout_id), expected)
        super().load(rollout_id)
        self._check_cursor({k: getattr(self, k) for k in expected}, expected)
