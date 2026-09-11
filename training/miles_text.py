"""Native Miles cursor/checkpoints, with an explicitly text-only input loader.

Qwen's AutoProcessor is multimodal even for our text-only checkpoint recipe.
Its presence must not reformat already-rendered prompts or add vision inputs.
"""

import copy

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
