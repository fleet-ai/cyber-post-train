"""Isolated compiler and request path for SFT checkpoint telemetry v1.

The legacy :mod:`training.sft` compiler is itself sealed into RL runtime
digests.  This successor therefore delegates unchanged scientific-plan
construction to that compiler, then binds a new runtime and transport here.
Nothing in this module is imported by an existing plan or portable bundle.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import shlex
from pathlib import Path

from cyber_post_train.jobs import canonical_gzip, digest, validate_request

from . import sft
from . import sft_runtime as base_runtime
from . import sft_runtime_checkpoint_telemetry_v1 as telemetry_runtime


def _source_paths() -> tuple[Path, Path]:
    return Path(telemetry_runtime.__file__), Path(base_runtime.__file__)


def _base_plan(plan: dict, *, check_files: bool) -> dict:
    """Validate the successor and derive the exact legacy delegate plan."""
    wrapper_path, base_path = _source_paths()
    telemetry_runtime.validate_runtime_binding(
        plan,
        runtime_path=wrapper_path,
        base_runtime_path=base_path,
    )
    if "recovery" in plan:
        raise ValueError("checkpoint telemetry v1 is not a recovery-plan runtime")
    if plan.get("model", {}).get("repo") != "Qwen/Qwen3.8-27B":
        raise ValueError("checkpoint telemetry v1 is qualified only for full-weight Qwen3.8 SFT")

    delegated = copy.deepcopy(plan)
    delegated.pop("runtime_variant")
    delegated["runtime_sha256"] = hashlib.sha256(base_path.read_bytes()).hexdigest()
    base_runtime.validate_plan(delegated, check_files=check_files)
    return delegated


def compile_sft(config: dict, *, relative_to: Path) -> dict:
    """Compile a fresh plan that explicitly selects this successor."""
    if not isinstance(config, dict):
        raise ValueError("SFT telemetry configuration must be a mapping")
    if config.get("runtime_variant") != telemetry_runtime.RUNTIME_VARIANT:
        raise ValueError("SFT telemetry configuration must select native_save_return_v1")
    if "lora" in config or "recovery" in config:
        raise ValueError("SFT checkpoint telemetry v1 requires a fresh full-weight plan")

    delegated = copy.deepcopy(config)
    delegated.pop("runtime_variant")
    plan = sft.compile_sft(delegated, relative_to=relative_to)
    wrapper_path, base_path = _source_paths()
    plan["runtime_sha256"] = hashlib.sha256(wrapper_path.read_bytes()).hexdigest()
    plan["runtime_variant"] = telemetry_runtime.runtime_binding(base_path)
    _base_plan(plan, check_files=False)
    validate_request(job_request(plan))
    return plan


def _bundle_transport(plan: dict) -> tuple[dict[str, str], str]:
    wrapper_path, base_path = _source_paths()
    plan_bytes = json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    contents = {
        "runtime": wrapper_path.read_text(),
        "plan": plan_bytes.decode(),
        "extra_files": {
            "training/__init__.py": Path(__file__).with_name("__init__.py").read_text(),
            "training/sft_runtime.py": base_path.read_text(),
        },
    }
    compressed = canonical_gzip(
        json.dumps(contents, sort_keys=True, separators=(",", ":")).encode()
    )
    bundle_sha256 = hashlib.sha256(compressed).hexdigest()
    encoded = base64.b64encode(compressed).decode()
    transport = {"CYBER_SFT_BUNDLE": encoded}
    bundle_expression = "os.environ.pop('CYBER_SFT_BUNDLE')"
    if len(encoded) > 120000:
        parts = [encoded[index : index + 48000] for index in range(0, len(encoded), 48000)]
        transport = {f"CYBER_SFT_BUNDLE_{index}": value for index, value in enumerate(parts)}
        bundle_expression = (
            f"''.join(os.environ.pop('CYBER_SFT_BUNDLE_'+str(i)) for i in range({len(parts)}))"
        )
    bootstrap = (
        "import base64,gzip,hashlib,importlib,json,os,pathlib,runpy,sys;"
        f"b=base64.b64decode({bundle_expression},validate=True);"
        f"assert hashlib.sha256(b).hexdigest()=={bundle_sha256!r};"
        "v=json.loads(gzip.decompress(b));"
        "p=pathlib.Path(os.environ['RUN_DIR'])/'.runtime';p.mkdir(parents=True,mode=0o700);"
        "(p/'sft_runtime.py').write_text(v['runtime']);(p/'plan.json').write_text(v['plan']);"
        "[((p/n).parent.mkdir(parents=True,exist_ok=True),(p/n).write_text(t)) "
        "for n,t in v.get('extra_files',{}).items()];"
        "sys.path.insert(0,str(p));importlib.invalidate_caches();"
        f"sys.argv=['sft_runtime','--plan',str(p/'plan.json'),'--plan-sha256',{hashlib.sha256(plan_bytes).hexdigest()!r}];"
        "runpy.run_path(str(p/'sft_runtime.py'),run_name='__main__')"
    )
    return transport, "python -c " + shlex.quote(bootstrap)


def job_request(plan: dict) -> dict:
    """Render this successor without changing the legacy request renderer."""
    delegated = _base_plan(plan, check_files=False)
    request = sft.job_request(delegated)
    transport, command = _bundle_transport(plan)
    environment = {
        key: value
        for key, value in request["env"].items()
        if not key.startswith("CYBER_SFT_BUNDLE")
    }
    environment.update(transport)
    environment["PYTHONPATH"] = str(Path(plan["output_root"]) / ".runtime")
    request["command"] = command
    request["env"] = environment
    validate_request(request)
    return request


def preflight(plan: dict) -> dict:
    """Run the legacy CPU gates plus the successor's exact trainer boundary."""
    delegated = _base_plan(plan, check_files=True)
    result = sft.preflight(delegated)
    telemetry_runtime.telemetry_trainer_class(base_runtime._make_trainer_class())
    result["schema"] = "cyber_sft_checkpoint_telemetry_cpu_preflight_v1"
    result["request_sha256"] = digest(job_request(plan))
    result["plan_sha256"] = digest(plan)
    result["checked"] = [*result["checked"], "native_save_return_telemetry_boundary"]
    return result
