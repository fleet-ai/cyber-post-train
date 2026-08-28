"""Fleet cyber post-training data and launch tooling.

The package deliberately keeps data preparation independent of a particular
trainer.  Fleet trajectories are immutable inputs; framework-specific launch
configuration is generated only after dataset and model compatibility gates
pass.
"""

from .normalize import normalize_export_row
from .rewards import compute_reward

__all__ = ["compute_reward", "normalize_export_row"]
