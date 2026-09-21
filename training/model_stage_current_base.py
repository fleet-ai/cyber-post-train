"""Stage one accepted export and bind registration to the current Qwen3.8 runtime.

The original :mod:`training.model_stage` source is immutable evidence for the
accepted Fresh75 transfer.  This small versioned wrapper keeps those transfer
bytes unchanged while replacing only the registration-clone validator and the
rendered entrypoint.  Both source files are digest-bound by the plan.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

try:
    from . import model_stage as base
except ImportError:  # pragma: no cover - used by the rendered standalone Pod
    import model_stage as base  # type: ignore[no-redef]

CURRENT_BASE_ID = "qwen3.8-27b"
CURRENT_BASE_SPEC_SHA256 = (
    "sha256:935294640178869a97ecacb2ac7aa7853a6a7d742f493acc918e0ae589652f0d"
)


def _replace_flag(args: list[Any], flag: str, value: str) -> None:
    if args.count(flag) != 1:
        raise ValueError(f"source registration must contain exactly one {flag}")
    index = args.index(flag)
    if index + 1 >= len(args):
        raise ValueError(f"source registration has no value for {flag}")
    args[index + 1] = value


def _validate_registration_clone(plan: Mapping[str, Any]) -> None:
    source = base._mapping(plan.get("registration_source"), "registration source")
    desired = base._mapping(plan.get("desired_registration"), "desired registration")
    source_spec = base._mapping(source.get("spec"), "source registration spec")
    desired_spec = base._mapping(desired.get("spec"), "desired registration spec")
    if source.get("id") != CURRENT_BASE_ID:
        raise ValueError("registration source is not the reviewed current Qwen3.8 base")
    if source.get("spec_sha256") != CURRENT_BASE_SPEC_SHA256:
        raise ValueError("registration source digest is not the reviewed current base digest")
    if base.digest_json(source_spec) != CURRENT_BASE_SPEC_SHA256:
        raise ValueError("source registration spec digest does not validate")
    resource_version = source.get("observed_resource_version")
    if not isinstance(resource_version, str) or re.fullmatch(r"[0-9]+", resource_version) is None:
        raise ValueError("registration source has no exact resource version")
    if source.get("observed_phase") != "ready":
        raise ValueError("registration source was not ready when observed")

    destination = base._mapping(plan.get("destination"), "destination")
    model_id, staged = destination["model_id"], destination["path"]
    if desired.get("id") != model_id:
        raise ValueError("desired registration id differs from the staged model id")
    display_name = desired_spec.get("displayName")
    if not isinstance(display_name, str) or not display_name.strip():
        raise ValueError("desired registration displayName must be non-empty")

    expected = json.loads(json.dumps(source_spec))
    expected["displayName"] = display_name
    expected["desiredState"] = "paused"
    expected["placement"]["priorityClassName"] = "c1"
    expected["scaling"] = {"minReplicas": 0}
    expected["model"]["sourcePath"] = staged
    expected["model"]["path"] = "/scratch/models/" + model_id
    expected["model"]["revision"] = plan["source"]["payload"]["manifest_sha256"]
    _replace_flag(expected["runtime"]["args"], "--model-path", expected["model"]["path"])
    _replace_flag(expected["runtime"]["args"], "--served-model-name", model_id)
    if dict(desired_spec) != expected:
        raise ValueError("desired registration changes fields outside the reviewed clone set")
    if (
        desired_spec.get("desiredState") != "paused"
        or desired_spec.get("scaling") != {"minReplicas": 0}
        or desired_spec.get("placement", {}).get("priorityClassName") != "c1"
    ):
        raise ValueError("desired registration must begin paused at zero replicas on c1")


@contextlib.contextmanager
def _current_base_validator() -> Iterator[None]:
    original = base._validate_registration_clone
    base._validate_registration_clone = _validate_registration_clone
    try:
        yield
    finally:
        base._validate_registration_clone = original


def _verify_wrapper(plan: Mapping[str, Any], wrapper_path: Path) -> None:
    expected = plan.get("execution", {}).get("runtime_wrapper_sha256")
    if wrapper_path.is_symlink() or base._digest_bytes(wrapper_path.read_bytes()) != expected:
        raise ValueError("current-base wrapper bytes differ from the stage plan")


def read_plan(path: Path, wrapper_path: Path = Path(__file__)) -> dict[str, Any]:
    with _current_base_validator():
        plan = base.read_plan(path)
    _verify_wrapper(plan, wrapper_path)
    return plan


def validate_plan(plan: Mapping[str, Any], wrapper_path: Path = Path(__file__)) -> None:
    with _current_base_validator():
        base.validate_plan(plan)
    _verify_wrapper(plan, wrapper_path)


def execute_stage(
    plan: Mapping[str, Any],
    *,
    open_source: base.OpenSource,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    _verify_wrapper(plan, Path(__file__))
    with _current_base_validator():
        return base.execute_stage(plan, open_source=open_source, environment=environment)


def render_manifest(
    plan: Mapping[str, Any], base_path: Path, wrapper_path: Path, plan_path: Path
) -> str:
    import yaml

    _verify_wrapper(plan, wrapper_path)
    with _current_base_validator():
        rendered = base.render_manifest(plan, base_path, plan_path)
    config_map, pod = list(yaml.safe_load_all(rendered))
    config_map["data"]["model_stage_current_base.py"] = wrapper_path.read_text(
        encoding="utf-8"
    )
    container = pod["spec"]["containers"][0]
    container["command"] = ["python3", "/bundle/model_stage_current_base.py"]
    container["volumeMounts"].insert(
        2,
        {
            "name": "bundle",
            "mountPath": "/bundle/model_stage_current_base.py",
            "subPath": "model_stage_current_base.py",
            "readOnly": True,
        },
    )
    return yaml.safe_dump_all([config_map, pod], sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("execute").add_argument("--plan", type=Path, required=True)
    render = commands.add_parser("render")
    render.add_argument("--plan", type=Path, required=True)
    render.add_argument("--base-source", type=Path, default=Path(base.__file__))
    render.add_argument("--wrapper-source", type=Path, default=Path(__file__))
    args = parser.parse_args()
    plan = read_plan(args.plan, getattr(args, "wrapper_source", Path(__file__)))
    if args.command == "render":
        print(
            render_manifest(
                plan,
                args.base_source,
                args.wrapper_source,
                args.plan,
            ),
            end="",
        )
        return
    user = os.environ.get("FILEBROWSER_USER", "")

    def opener(path: str) -> contextlib.AbstractContextManager[Any]:
        return base._filebrowser_stream(path, user)

    result = execute_stage(plan, open_source=opener, environment=os.environ)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
