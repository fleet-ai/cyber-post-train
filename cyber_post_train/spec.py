"""Typed, immutable experiment composition and plan compilation."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .catalog import ROOT

SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
EXPERIMENT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,79}$")

ADAPTER_KIND = {
    "fleet": "evaluation",
    "webexploitbench": "evaluation",
    "exploitgym": "evaluation",
    "fleet-sft": "training",
    "fleet-rl": "training",
}
REQUIRED_COMPONENTS = {
    "evaluation": frozenset({"model", "serving", "harness", "dataset", "protocol"}),
    "training": frozenset({"model", "dataset", "trainer", "protocol"}),
}


def canonical_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def safe_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError("component paths must be repository-relative and may not contain '..'")
    return path


class ComponentRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    sha256: str

    @model_validator(mode="after")
    def validate_ref(self) -> ComponentRef:
        safe_relative_path(self.path)
        if not SHA256_RE.fullmatch(self.sha256):
            raise ValueError("component sha256 must use sha256:<64 lowercase hex>")
        return self


class ExecutionProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    backend: str = Field(min_length=1)
    output_root: str
    max_concurrency: int = Field(ge=1)
    serving_block: Literal["hosted", "dedicated", "matched-self-hosted"] | None = None

    @model_validator(mode="after")
    def validate_output(self) -> ExecutionProfile:
        path = safe_relative_path(self.output_root)
        if path.parts[0] != "output":
            raise ValueError("execution.output_root must be under output/")
        return self


class ExperimentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    schema_version: Literal["cyber_experiment_spec_v1"] = Field(alias="schema")
    experiment_id: str
    kind: Literal["evaluation", "training"]
    adapter: Literal["fleet", "webexploitbench", "exploitgym", "fleet-sft", "fleet-rl"]
    components: dict[str, ComponentRef]
    execution: ExecutionProfile

    @model_validator(mode="after")
    def validate_composition(self) -> ExperimentSpec:
        if not EXPERIMENT_ID_RE.fullmatch(self.experiment_id):
            raise ValueError("experiment_id must be a lowercase hyphenated identifier")
        if ADAPTER_KIND[self.adapter] != self.kind:
            raise ValueError(f"adapter {self.adapter!r} is not a {self.kind} adapter")
        missing = REQUIRED_COMPONENTS[self.kind] - self.components.keys()
        if missing:
            raise ValueError("missing components: " + ", ".join(sorted(missing)))
        if self.kind == "evaluation" and self.execution.serving_block is None:
            raise ValueError("evaluation execution requires an explicit serving_block")
        if self.kind == "training":
            dataset_path = self.components["dataset"].path.lower()
            if "webexploitbench" in dataset_path or "exploitgym" in dataset_path:
                raise ValueError("external benchmark data may not be a training dataset")
        return self


def _load_mapping(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a mapping")
    return value


def lock_spec(source_path: Path, *, root: Path = ROOT) -> dict[str, Any]:
    """Resolve component file digests in an unlocked source spec."""

    value = _load_mapping(source_path)
    components = value.get("components")
    if not isinstance(components, dict):
        raise ValueError("components must be a mapping")
    locked_components: dict[str, dict[str, str]] = {}
    for name, raw_ref in sorted(components.items()):
        if not isinstance(name, str) or not isinstance(raw_ref, dict):
            raise ValueError("each component must be a named mapping")
        component_path = raw_ref.get("path")
        if not isinstance(component_path, str):
            raise ValueError(f"component {name!r} is missing path")
        relative = safe_relative_path(component_path)
        resolved = root.joinpath(*relative.parts)
        if not resolved.is_file():
            raise ValueError(f"component {name!r} does not exist: {component_path}")
        locked_components[name] = {"path": component_path, "sha256": file_digest(resolved)}
    value["components"] = locked_components
    ExperimentSpec.model_validate(value)
    return value


def load_locked_spec(path: Path, *, root: Path = ROOT) -> ExperimentSpec:
    spec = ExperimentSpec.model_validate(_load_mapping(path))
    for name, component in spec.components.items():
        relative = safe_relative_path(component.path)
        resolved = root.joinpath(*relative.parts)
        if not resolved.is_file():
            raise ValueError(f"component {name!r} does not exist: {component.path}")
        actual = file_digest(resolved)
        if actual != component.sha256:
            raise ValueError(
                f"component {name!r} digest mismatch: expected {component.sha256}, got {actual}"
            )
    return spec


def compile_plan(path: Path, *, root: Path = ROOT) -> dict[str, Any]:
    spec = load_locked_spec(path, root=root)
    spec_value = spec.model_dump(mode="json", by_alias=True)
    plan: dict[str, Any] = {
        "schema": "cyber_experiment_plan_v1",
        "experiment_id": spec.experiment_id,
        "kind": spec.kind,
        "adapter": spec.adapter,
        "components": spec_value["components"],
        "execution": spec_value["execution"],
        "source_spec_sha256": canonical_digest(spec_value),
        "lifecycle": ["validate", "preview", "claim", "launch", "status", "accept"],
    }
    plan["plan_sha256"] = canonical_digest(plan)
    return plan


def write_new(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
