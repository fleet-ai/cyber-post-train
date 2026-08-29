"""Immutable, model-only configuration for queued training workloads.

Dataset selection, split policy, reward composition, and evaluation remain outside
this adapter so changing a model cannot silently change the experiment science.
"""

from __future__ import annotations

import json
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import digest_json

SCHEMA = "cyber_post_train_model_adapter_v1"
_DNS = re.compile(r"^[a-z0-9](?:[-a-z0-9]*[a-z0-9])?$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_IMAGE = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")
_FIELDS = {
    "model_root",
    "checkpoint_dir",
    "sft_train",
    "sft_dev",
    "rl_prompts",
    "reward_contract",
}


@dataclass(frozen=True)
class Resources:
    gpus: int
    cpu_request: str
    cpu_limit: str
    memory_request: str
    memory_limit: str


@dataclass(frozen=True)
class ModelAdapter:
    slug: str
    model_id: str
    revision: str
    architecture: str
    model_root: str
    checkpoint_lock: str
    weights_manifest_sha256: str
    compatibility_receipt: str
    image: str
    python: str
    working_dir: str
    sft_argv: tuple[str, ...]
    rl_argv: tuple[str, ...]
    driver_contract: str
    resources: Resources
    digest: str

    def format_argv(self, stage: str, values: dict[str, str]) -> list[str]:
        template = self.sft_argv if stage == "sft" else self.rl_argv
        unknown = set(values) - _FIELDS
        if unknown:
            raise ValueError(f"unknown model adapter template values: {sorted(unknown)}")
        return [item.format_map(values) for item in template]


def _argv(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(x, str) and x for x in value):
        raise ValueError(f"{field} must be a non-empty argv array")
    for item in value:
        fields = {name for _, name, _, _ in string.Formatter().parse(item) if name}
        if not fields <= _FIELDS:
            unsupported = sorted(fields - _FIELDS)
            raise ValueError(f"{field} uses unsupported template fields: {unsupported}")
    return tuple(value)


def load_model_adapter(path: Path) -> ModelAdapter:
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or raw.get("schema") != SCHEMA:
        raise ValueError("invalid model adapter schema")
    allowed = {"schema", "model", "runtime", "stages", "resources", "adapter_digest"}
    if unknown := set(raw) - allowed:
        raise ValueError(f"model adapter contains forbidden top-level fields: {sorted(unknown)}")
    model = raw.get("model") or {}
    runtime = raw.get("runtime") or {}
    stages = raw.get("stages") or {}
    resources = raw.get("resources") or {}
    slug = model.get("slug")
    revision = model.get("revision")
    image = runtime.get("image")
    model_root = model.get("root")
    receipt = model.get("compatibility_receipt")
    checkpoint_lock = model.get("checkpoint_lock")
    weights_manifest = model.get("weights_manifest_sha256")
    if not isinstance(slug, str) or not _DNS.fullmatch(slug):
        raise ValueError("model.slug must be a DNS label")
    if not isinstance(revision, str) or not _REVISION.fullmatch(revision):
        raise ValueError("model.revision must be a full 40-character git revision")
    if not isinstance(image, str) or not _IMAGE.fullmatch(image):
        raise ValueError("runtime.image must be pinned by sha256 digest")
    expected_root = f"/mnt/sfs/models/{model.get('id')}/{revision}"
    if model_root != expected_root:
        raise ValueError(f"model.root must equal {expected_root}")
    if checkpoint_lock != f"{model_root}/.cyber-post-train-lock.json":
        raise ValueError("checkpoint lock must be the immutable model root lock")
    valid_manifest = isinstance(weights_manifest, str) and re.fullmatch(
        r"sha256:[0-9a-f]{64}", weights_manifest
    )
    if not valid_manifest:
        raise ValueError("model.weights_manifest_sha256 must be a sha256 digest")
    compatibility_root = "/mnt/sfs/cyber-post-train/compatibility/"
    if not isinstance(receipt, str) or not receipt.startswith(compatibility_root):
        raise ValueError("compatibility receipt must live under the shared compatibility root")
    gpus = resources.get("gpus")
    if not isinstance(gpus, int) or not 1 <= gpus <= 8:
        raise ValueError("resources.gpus must fit one B300 node (1..8)")
    driver_contract = stages.get("driver_contract")
    valid_driver = isinstance(driver_contract, str) and _DNS.fullmatch(
        driver_contract.replace("_", "-")
    )
    if not valid_driver:
        raise ValueError("stages.driver_contract must be a stable identifier")
    canonical = {key: value for key, value in raw.items() if key != "adapter_digest"}
    actual_digest = digest_json(canonical)
    declared_digest = raw.get("adapter_digest")
    if declared_digest is not None and declared_digest != actual_digest:
        raise ValueError("model adapter digest mismatch")
    request = Resources(
        gpus=gpus,
        cpu_request=str(resources.get("cpu_request")),
        cpu_limit=str(resources.get("cpu_limit")),
        memory_request=str(resources.get("memory_request")),
        memory_limit=str(resources.get("memory_limit")),
    )
    return ModelAdapter(
        slug=slug,
        model_id=str(model.get("id")),
        revision=revision,
        architecture=str(model.get("architecture")),
        model_root=model_root,
        checkpoint_lock=checkpoint_lock,
        weights_manifest_sha256=weights_manifest,
        compatibility_receipt=receipt,
        image=image,
        python=str(runtime.get("python", "python")),
        working_dir=str(runtime.get("working_dir", "/workspace")),
        sft_argv=_argv(stages.get("sft_argv"), "stages.sft_argv"),
        rl_argv=_argv(stages.get("rl_argv"), "stages.rl_argv"),
        driver_contract=driver_contract,
        resources=request,
        digest=actual_digest,
    )
