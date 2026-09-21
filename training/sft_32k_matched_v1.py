"""Compiler and transport for one matched 32K dense-versus-LoRA pair."""

from __future__ import annotations

import base64
import contextlib
import copy
import hashlib
import json
import shlex
from pathlib import Path

from cyber_post_train.jobs import canonical_gzip, digest, validate_request

from . import sft
from . import sft_runtime as base_runtime
from . import sft_runtime_32k_matched_v1 as runtime

_BASE_JOB_REQUEST = sft.job_request
_BASE_VALIDATE_PLAN = base_runtime.validate_plan


def _source_paths() -> tuple[Path, Path]:
    return Path(runtime.__file__), Path(base_runtime.__file__)


@contextlib.contextmanager
def _base_plan_context():
    """Keep the compiler's imported LoRA map aligned with the runtime."""
    previous = sft.QWEN38_LORA_BROAD_FULL_PLANS
    with runtime.base_plan_context() as plans:
        sft.QWEN38_LORA_BROAD_FULL_PLANS = plans
        try:
            yield
        finally:
            sft.QWEN38_LORA_BROAD_FULL_PLANS = previous


def _base_plan(plan: dict, *, check_files: bool) -> dict:
    wrapper_path, base_path = _source_paths()
    runtime.validate_runtime_binding(
        plan,
        runtime_path=wrapper_path,
        base_runtime_path=base_path,
        check_files=check_files,
    )
    delegated = runtime.delegated_plan(plan)
    with _base_plan_context():
        _BASE_VALIDATE_PLAN(delegated, check_files=check_files)
    return delegated


def compile_sft(config: dict, *, relative_to: Path) -> dict:
    """Compile exactly one of the two immutable matched configurations."""
    if (
        not isinstance(config, dict)
        or config.get("runtime_variant") != runtime.RUNTIME_VARIANT
        or config.get("name") not in {runtime.DENSE_RUN_NAME, runtime.LORA_RUN_NAME}
    ):
        raise ValueError("matched SFT configuration must select its exact versioned runtime")
    runtime._validate_optimizer(config.get("optimizer"))
    delegated = copy.deepcopy(config)
    optimizer = delegated.pop("optimizer")
    delegated.pop("runtime_variant")
    with _base_plan_context():
        plan = sft.compile_sft(delegated, relative_to=relative_to)
    wrapper_path, base_path = _source_paths()
    plan["runtime_sha256"] = hashlib.sha256(wrapper_path.read_bytes()).hexdigest()
    plan["runtime_variant"] = runtime.runtime_binding(base_path)
    plan["optimizer"] = copy.deepcopy(optimizer)
    _base_plan(plan, check_files=False)
    validate_request(job_request(plan))
    return plan


def _bundle_transport(plan: dict) -> tuple[dict[str, str], str]:
    wrapper_path, base_path = _source_paths()
    plan_bytes = json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    extras = {
        "training/__init__.py": Path(__file__).with_name("__init__.py").read_text(),
        "training/sft_runtime.py": base_path.read_text(),
    }
    if "lora" in plan:
        extras.update(
            {
                "training/io.py": Path(__file__).with_name("io.py").read_text(),
                "training/qwen38_lora_artifacts.py": Path(__file__)
                .with_name("qwen38_lora_artifacts.py")
                .read_text(),
            }
        )
    contents = {
        "runtime": wrapper_path.read_text(),
        "plan": plan_bytes.decode(),
        "extra_files": extras,
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
    """Render the base request while replacing only versioned runtime transport."""
    delegated = _base_plan(plan, check_files=False)
    with _base_plan_context():
        request = _BASE_JOB_REQUEST(delegated)
    transport, command = _bundle_transport(plan)
    request["command"] = command
    request["env"] = {
        key: value
        for key, value in request["env"].items()
        if not key.startswith("CYBER_SFT_BUNDLE")
    }
    request["env"].update(transport)
    pythonpath = str(Path(plan["output_root"]) / ".runtime")
    if "lora" in plan:
        pythonpath += ":/opt/skyrl"
    request["env"]["PYTHONPATH"] = pythonpath
    validate_request(request)
    return request


def _preflight_validate_plan(plan: dict, *, check_files: bool = True) -> None:
    """Validate wrapper plans and delegated base renders through the right rail."""
    if "runtime_variant" in plan:
        runtime.validate_plan(plan, check_files=check_files)
    else:
        _BASE_VALIDATE_PLAN(plan, check_files=check_files)


@contextlib.contextmanager
def _preflight_context():
    """Route base CPU checks through the versioned plan and optimizer hooks."""
    with runtime.execution_context():
        previous_validate = sft.validate_plan
        previous_request = sft.job_request
        sft.validate_plan = _preflight_validate_plan
        sft.job_request = job_request
        try:
            yield
        finally:
            sft.validate_plan = previous_validate
            sft.job_request = previous_request


def preflight(plan: dict) -> dict:
    """Run unchanged CPU checks against the explicit matched optimizer render."""
    _base_plan(plan, check_files=True)
    with _preflight_context():
        result = sft.preflight(plan)
    result["request_sha256"] = digest(job_request(plan))
    result["plan_sha256"] = digest(plan)
    result["checked"] = [*result["checked"], "matched_32k_optimizer_and_identity"]
    return result
