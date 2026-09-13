"""Prepare and register one exact Qwen export. No serving readiness is inferred.

The only external mutation is the explicit ``execute`` command. Preview performs
local validation plus authenticated GETs; no server dry-run API is assumed.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from .io import canonical_json, digest_json, file_sha256

API = "https://inference.flt.build/fleet/v1/models"
ACCOUNT = "https://orchestrator.fleetai.com/v1/account"
TEAM_ID = "a1025f0b-ad67-49fc-a023-51800ab43e84"
SCHEMA = "cyber_exact_serving_registration_v2"
MILES_SCHEMA = "cyber_exact_miles_serving_registration_v1"
DEV_SCHEMA = "cyber_serving_dev_qualification_v1"
MILES_UPDATE_SCHEMA = "cyber_miles_source_update_identity_v1"
HEX = re.compile(r"[0-9a-f]{64}")
MODEL_ID = re.compile(r"[a-z0-9][a-z0-9-]{1,61}[a-z0-9]")
DEV_CHECKS = {
    "exact_model_identity",
    "full_model_reload",
    "finite_forward",
    "structured_tool_call",
    "context_continuation",
    "cleanup_verified",
}


def _sha(value: object) -> str:
    if not isinstance(value, str) or not HEX.fullmatch(value.removeprefix("sha256:")):
        raise ValueError("expected exact SHA-256")
    return "sha256:" + value.removeprefix("sha256:")


def _signed(value: dict) -> dict:
    return {**value, "receipt_sha256": digest_json(value)}


def _read(path: Path, expected: str | None = None, *, signed: bool = False) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError("evidence must be a regular nonsymlink file")
    if expected is not None and file_sha256(path) != _sha(expected):
        raise ValueError("evidence file SHA-256 mismatch")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError("expected an evidence object")
    if signed:
        actual = digest_json({k: v for k, v in value.items() if k != "receipt_sha256"})
        if _sha(value.get("receipt_sha256")) != actual:
            raise ValueError("evidence self-digest mismatch")
    return value


def _once(path: Path, value: dict) -> None:
    """An interrupted write remains a claim; no overwrite or silent retry."""
    raw = (canonical_json(value) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _absolute(value: object) -> Path:
    if not isinstance(value, str):
        raise ValueError("expected absolute directory")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or len(path.parts) < 3:
        raise ValueError("unsafe directory")
    return path


def _arg(args: list[str], flag: str) -> str:
    for arg in args:
        if not arg.startswith("--"):
            continue
        name = arg.split("=", 1)[0]
        if (
            (name == flag and arg != flag)
            or (name != flag and flag.startswith(name))
            or name == flag.replace("-", "_").replace("__", "--", 1)
        ):
            raise ValueError("alternate/abbreviated serving identity argument is forbidden")
    if args.count(flag) != 1:
        raise ValueError("missing or duplicate serving identity argument")
    index = args.index(flag) + 1
    if index >= len(args) or args[index].startswith("--"):
        raise ValueError("missing serving argument value")
    return args[index]


def _registration(value: dict) -> dict:
    """Keep the complete desired spec; never accept unpinned executable bytes."""
    if not MODEL_ID.fullmatch(str(value.get("id", ""))):
        raise ValueError("model ID must be a unique DNS label")
    if not isinstance(value.get("spec"), dict):
        raise ValueError("missing serving spec")
    result = {"id": value["id"], "spec": copy.deepcopy(value["spec"])}
    spec = result["spec"]
    if set(spec) - {
        "cache",
        "capabilities",
        "desiredState",
        "displayName",
        "model",
        "placement",
        "resources",
        "routing",
        "runtime",
        "scaling",
    }:
        raise ValueError("unrecognized serving spec fields")
    if spec.get("desiredState", "serving") != "serving":
        raise ValueError("baseline serving spec must not be paused/retired")
    model, runtime = spec["model"], spec["runtime"]
    if set(model) - {
        "dataParallelAttention",
        "dataParallelSize",
        "path",
        "precision",
        "revision",
        "sourcePath",
        "tensorParallelSize",
    } or set(runtime) - {
        "args",
        "command",
        "engine",
        "env",
        "healthPath",
        "image",
        "lockMemory",
        "port",
    }:
        raise ValueError("unrecognized model/runtime fields")
    _absolute(model["sourcePath"])
    _absolute(model["path"])
    if not isinstance(model.get("revision"), str) or not model["revision"]:
        raise ValueError("missing immutable model revision")
    if model.get("precision") != "bf16" or type(model.get("tensorParallelSize")) is not int:
        raise ValueError("expected exact BF16 tensor parallelism")
    if not 1 <= model["tensorParallelSize"] <= 8:
        raise ValueError("unsupported tensor parallel shape")
    image = runtime["image"]
    _sha(image["digest"])
    if (
        runtime.get("engine") != "sglang"
        or not isinstance(image.get("repository"), str)
        or not image["repository"]
        or "@" in image["repository"]
        or not isinstance(runtime.get("command"), list)
        or not runtime["command"]
    ):
        raise ValueError("unsupported serving engine/image/command")
    if runtime["command"] not in (
        ["python3", "-m", "sglang.launch_server"],
        ["python", "-m", "sglang.launch_server"],
    ):
        raise ValueError("only the directly invoked SGLang entrypoint is supported")
    args = runtime["args"]
    if not isinstance(args, list) or not all(isinstance(v, str) for v in args):
        raise ValueError("serving args must be strings")
    if "--" in args:
        raise ValueError("serving argument end-of-options sentinel is unsupported")
    if any(
        re.match(r"--(?:api[-_]key|token|password|credential|secret)(?:=|$)", arg, re.I)
        for arg in args
    ):
        raise ValueError("credential-bearing serving arguments are prohibited")
    if (
        _arg(args, "--model-path") != model["path"]
        or _arg(args, "--served-model-name") != value["id"]
    ):
        raise ValueError("serving argument identity differs from model spec")
    # Do not permit literal credential-bearing environment values in a receipt.
    for env in runtime.get("env", []):
        if (
            not isinstance(env, dict)
            or env.get("name")
            not in {
                "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
                "HF_DATASETS_OFFLINE",
            }
            or env.get("value") != "1"
            or set(env) != {"name", "value"}
        ):
            raise ValueError("only offline-mode literal environment entries are supported")
    if spec.get("scaling", {}).get("minReplicas") != 1:
        raise ValueError("one separately accounted serving replica is required")
    if spec["scaling"].get("maxReplicas", 1) != 1:
        raise ValueError("automatic replica expansion is not allowed")
    gpu_count = spec["resources"]["limits"].get("nvidia.com/gpu")
    if type(gpu_count) is not int or not 1 <= gpu_count <= 8:
        raise ValueError("explicit bounded GPU allocation is required")
    return result


def _canonical_spec(spec: dict) -> dict:
    value = copy.deepcopy(spec)
    capabilities = value.get("capabilities")
    if not isinstance(capabilities, list) or not all(isinstance(x, str) for x in capabilities):
        raise ValueError("capabilities must be explicit")
    value["capabilities"] = sorted(capabilities)
    return value


def execution_contract(registration: dict) -> dict:
    """Only path/name/revision and placement differ between qualified dev/prod.

    Device count, resource shape, image, runtime, precision and scaling remain
    exact. Dev/prod storage/placement are independently observed by the operator.
    """
    value = _registration(registration)
    model = value["spec"]["model"]
    replacements = {
        value["id"]: "MODEL_ID",
        model["path"]: "MODEL_PATH",
        model["sourcePath"]: "SOURCE_PATH",
    }
    value["spec"]["runtime"]["args"] = [
        replacements.get(arg, arg) for arg in value["spec"]["runtime"]["args"]
    ]
    model.update(sourcePath="SOURCE_PATH", path="MODEL_PATH", revision="CHECKPOINT_REVISION")
    value["spec"].pop("displayName", None)
    value["spec"].pop("placement", None)
    return _canonical_spec(value["spec"])


def _stage(config: dict, export: dict, export_receipt_sha256: str) -> None:
    """Reopen one generic HF payload at its measured inference-PVC destination."""
    storage = config["storage"]
    staging = _read(Path(config["staging"]["path"]), config["staging"]["sha256"], signed=True)
    if storage.get("namespace") != "inference" or not storage.get("pvc_name"):
        raise ValueError("explicit inference cache mount identity is required")
    if str(uuid.UUID(storage["pvc_uid"])) != storage["pvc_uid"]:
        raise ValueError("immutable inference PVC UID is required")
    staged = _absolute(storage["staged_root"])
    if staged.is_symlink() or not staged.is_dir() or staged.resolve() != staged:
        raise ValueError("staged root must be an exact nonsymlink directory")
    _absolute(storage["source_path"])
    if (
        staging.get("schema") != "cyber_inference_staging_check_v1"
        or staging.get("status") != "passed"
        or any(
            staging.get(key) != storage[key]
            for key in ("namespace", "pvc_name", "pvc_uid", "source_path", "staged_root")
        )
        or staging.get("destination_create_once") is not True
        or staging.get("payload_rehashed") is not True
        or _sha(staging.get("export_receipt_sha256")) != _sha(export_receipt_sha256)
        or _sha(staging.get("export_file_sha256")) != _sha(config["export"]["sha256"])
        or _sha(staging.get("files_manifest_sha256")) != digest_json(export["files"])
    ):
        raise ValueError("exact inference staging/mount receipt is required")
    if str(uuid.UUID(staging["observer_pod_uid"])) != staging["observer_pod_uid"]:
        raise ValueError("staging observer Pod UID is not canonical")
    _sha(staging.get("mount_evidence_sha256"))
    mount = _absolute(staging["observed_mount_root"])
    catalog = Path(staging["catalog_root"])
    if catalog != Path("/models") or not staged.is_relative_to(mount) or staged == mount:
        raise ValueError("staging mount/catalog mapping is invalid")
    if str(catalog / staged.relative_to(mount)) != storage["source_path"]:
        raise ValueError("staged directory does not map to the registered source path")
    files = export.get("files")
    if (
        not isinstance(files, dict)
        or not files
        or {p.name for p in staged.iterdir()} != set(files) | {"EXPORT.json"}
    ):
        raise ValueError("staged file inventory differs from export")
    if (staged / "EXPORT.json").is_symlink() or file_sha256(staged / "EXPORT.json") != _sha(
        config["export"]["sha256"]
    ):
        raise ValueError("staged export receipt mismatch")
    for name, info in files.items():
        if not isinstance(name, str) or Path(name).name != name or name in {".", ".."}:
            raise ValueError("unsafe export payload name")
        path = staged / name
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != info["bytes"]
            or file_sha256(path) != _sha(info["sha256"])
        ):
            raise ValueError("staged payload size/hash mismatch")
    if not {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "model.safetensors.index.json",
    } <= set(files):
        raise ValueError("staged tokenizer/runtime/index evidence is incomplete")
    if not any(name.endswith(".safetensors") for name in files):
        raise ValueError("staged weights are absent")


def _export_and_stage(config: dict) -> tuple[dict, dict]:
    export = _read(Path(config["export"]["path"]), config["export"]["sha256"], signed=True)
    gpu = _read(Path(config["gpu_check"]["path"]), config["gpu_check"]["sha256"], signed=True)
    if (
        export.get("schema") != "cyber_native_checkpoint_hf_export_v1"
        or export.get("model_repo") != "Qwen/Qwen3.8-27B"
        or export.get("dtype") != "BF16"
        or export.get("optimizer_steps_executed") != 0
        or type(export.get("optimizer_step")) is not int
        or export["optimizer_step"] < 1
        or export.get("all_output_tensors_reopened_equal") is not True
        or export.get("source_inventory_sizes_mtimes_unchanged") is not True
    ):
        raise ValueError("export is not a complete zero-step Qwen BF16 handoff")
    for key in ("source_checkpoint_receipt_sha256", "source_plan_sha256"):
        _sha(export.get(key))
    if (
        gpu.get("schema") != "cyber_hf_export_check_v1"
        or gpu.get("status") != "passed"
        or gpu.get("gpu_reload_verified") is not True
        or gpu.get("gpus") != 1
        or gpu.get("source_unchanged") is not True
        or gpu.get("finite_logits") is not True
        or gpu.get("optimizer_steps_executed") != 0
        or gpu.get("synthetic_only") is not True
        or _sha(gpu.get("export_sha256")) != _sha(config["export"]["sha256"])
        or _sha(gpu.get("export_receipt_sha256")) != _sha(export["receipt_sha256"])
    ):
        raise ValueError("exact GPU export check is missing or inconsistent")
    _stage(config, export, export["receipt_sha256"])
    return export, gpu


def _miles_update_identity(export: dict) -> dict:
    """Name the trained source without mislabeling export/reload zero-work as training."""
    source = export.get("source")
    checkpoint = source.get("checkpoint") if isinstance(source, dict) else None
    if not isinstance(checkpoint, dict):
        raise ValueError("validated Miles export has no source checkpoint identity")
    return {
        "schema": MILES_UPDATE_SCHEMA,
        "source_plan_sha256": _sha(source.get("source_plan_sha256")),
        "source_checkpoint_receipt_sha256": _sha(checkpoint.get("receipt_sha256")),
        "export_tensor_inventory_sha256": _sha(export.get("tensor_inventory_sha256")),
    }


def _miles_export_and_stage(config: dict) -> tuple[dict, dict, dict, dict]:
    """Reopen the public Miles export and accepted zero-update reload validators."""
    from . import miles_hf_export

    export_path = Path(config["export"]["path"])
    export, _ = miles_hf_export.inspect_export(export_path, config["export"]["sha256"])
    export_receipt_sha256 = _sha(export.get("sha256"))
    reload_receipt = _read(
        Path(config["reload_acceptance"]["path"]), config["reload_acceptance"]["sha256"]
    )
    reload_validation = {
        key: _sha(value)
        for key, value in miles_hf_export.validate_reload_accepted(
            reload_receipt, check_files=True
        ).items()
    }
    if (
        reload_receipt.get("serving_qualified") is not False
        or str(export_path) != reload_receipt.get("export_path")
        or _sha(reload_receipt.get("export_file_sha256")) != _sha(config["export"]["sha256"])
        or _sha(reload_receipt.get("export_receipt_sha256")) != export_receipt_sha256
        or _sha(reload_validation.get("export_receipt_sha256")) != export_receipt_sha256
        or _sha(reload_receipt.get("export_tensor_inventory_sha256"))
        != _sha(export.get("tensor_inventory_sha256"))
    ):
        raise ValueError("accepted Miles zero-update reload does not bind this export")
    _stage(config, export, export_receipt_sha256)
    return export, reload_receipt, reload_validation, _miles_update_identity(export)


def _candidate(config: dict, base: dict, export: dict, *, revision: str | None = None) -> dict:
    candidate = copy.deepcopy(base)
    if config["model_id"] == base["id"] or not MODEL_ID.fullmatch(config["model_id"]):
        raise ValueError("new safe model identity is required")
    if not isinstance(config["display_name"], str) or not config["display_name"]:
        raise ValueError("explicit display name is required")
    model = candidate["spec"]["model"]
    source = str(_absolute(config["storage"]["source_path"]))
    if not source.startswith("/models/") or source == model["sourcePath"]:
        raise ValueError("new create-once inference cache path is required")
    if _overlap(source, model["sourcePath"]):
        raise ValueError("new source path overlaps the baseline")
    serving = "/scratch/models/" + source.removeprefix("/models/")
    replacements = {
        candidate["id"]: config["model_id"],
        model["sourcePath"]: source,
        model["path"]: serving,
    }
    candidate["spec"]["runtime"]["args"] = [
        replacements.get(arg, arg) for arg in candidate["spec"]["runtime"]["args"]
    ]
    candidate["id"] = config["model_id"]
    candidate["spec"]["displayName"] = config["display_name"]
    model.update(
        sourcePath=source,
        path=serving,
        revision=_sha(export["receipt_sha256"] if revision is None else revision),
    )
    _registration(candidate)
    if execution_contract(base) != execution_contract(candidate):
        raise ValueError("non-checkpoint serving configuration drift")
    return candidate


def _config(value: dict) -> dict:
    schema = value.get("schema")
    fields = {
        SCHEMA: {
            "schema",
            "base_registration",
            "export",
            "gpu_check",
            "staging",
            "storage",
            "model_id",
            "display_name",
        },
        MILES_SCHEMA: {
            "schema",
            "base_registration",
            "export",
            "reload_acceptance",
            "staging",
            "storage",
            "model_id",
            "display_name",
        },
    }
    if schema not in fields or set(value) != fields[schema]:
        raise ValueError("unexpected registration configuration schema/fields")
    config = copy.deepcopy(value)
    if not isinstance(config["storage"], dict) or set(config["storage"]) != {
        "namespace",
        "pvc_name",
        "pvc_uid",
        "staged_root",
        "source_path",
    }:
        raise ValueError("unexpected storage configuration fields")
    references = {"base_registration", "export", "staging"}
    references.add("gpu_check" if schema == SCHEMA else "reload_acceptance")
    for name in references:
        ref = config[name]
        if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
            raise ValueError("unexpected evidence reference fields")
        path = _absolute(ref["path"])
        if path.is_symlink() or path.resolve() != path:
            raise ValueError("evidence reference must not traverse a symlink")
        ref["path"] = str(path)
        ref["sha256"] = _sha(ref["sha256"])
    return config


def _artifact(config: dict) -> tuple[dict, str, dict]:
    if config["schema"] == SCHEMA:
        export, _ = _export_and_stage(config)
        receipt_sha256 = _sha(export["receipt_sha256"])
        return (
            export,
            receipt_sha256,
            {
                "export_receipt_sha256": receipt_sha256,
                "optimizer_step": export["optimizer_step"],
                "staged_manifest_sha256": digest_json(export["files"]),
            },
        )
    export, reload_receipt, reload_validation, update_identity = _miles_export_and_stage(config)
    receipt_sha256 = _sha(export["sha256"])
    return (
        export,
        receipt_sha256,
        {
            "artifact_kind": "miles_rl",
            "export_file_sha256": _sha(config["export"]["sha256"]),
            "export_receipt_sha256": receipt_sha256,
            "reload_acceptance_file_sha256": _sha(config["reload_acceptance"]["sha256"]),
            "reload_acceptance_receipt_sha256": _sha(reload_receipt["sha256"]),
            "reload_validation": reload_validation,
            "source_update_identity": update_identity,
            "source_update_identity_sha256": digest_json(update_identity),
            "staged_manifest_sha256": digest_json(export["files"]),
        },
    )


def prepare(config: dict, output: Path) -> dict:
    config = _config(config)
    base = _registration(
        _read(Path(config["base_registration"]["path"]), config["base_registration"]["sha256"])
    )
    export, revision, artifact = _artifact(config)
    candidate = _candidate(config, base, export, revision=revision)
    plan = _signed(
        {
            "schema": config["schema"],
            "api": API,
            "config": config,
            "base_registration": base,
            "registration": candidate,
            "registration_sha256": digest_json(candidate),
            "execution_contract_sha256": digest_json(execution_contract(candidate)),
            **artifact,
            "registration_code_sha256": _code(config["schema"]),
            "created_at": datetime.now(UTC).isoformat(),
        }
    )
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    _once(output / "plan.json", plan)
    return summary(plan)


def load(directory: Path) -> dict:
    plan = _read(directory / "plan.json", signed=True)
    schema = plan.get("schema")
    if (
        schema not in {SCHEMA, MILES_SCHEMA}
        or plan.get("api") != API
        or plan.get("registration_code_sha256") != _code(schema)
    ):
        raise ValueError("prepared registration schema/API/source drift")
    config = _config(plan["config"])
    if config != plan["config"] or config["schema"] != schema:
        raise ValueError("prepared registration configuration drift")
    base = _registration(
        _read(Path(config["base_registration"]["path"]), config["base_registration"]["sha256"])
    )
    export, revision, artifact = _artifact(config)
    registration = _registration(plan["registration"])
    common_fields = {
        "schema",
        "api",
        "config",
        "base_registration",
        "registration",
        "registration_sha256",
        "execution_contract_sha256",
        "registration_code_sha256",
        "created_at",
        "receipt_sha256",
    }
    if (
        (schema == MILES_SCHEMA and set(plan) != common_fields | set(artifact))
        or base != plan["base_registration"]
        or registration != _candidate(config, base, export, revision=revision)
        or digest_json(registration) != plan["registration_sha256"]
        or digest_json(execution_contract(registration)) != plan["execution_contract_sha256"]
        or execution_contract(base) != execution_contract(registration)
        or registration["spec"]["model"]["revision"] != revision
        or registration["spec"]["model"]["sourcePath"] != config["storage"]["source_path"]
        or registration["id"] != config["model_id"]
        or any(plan.get(key) != value for key, value in artifact.items())
    ):
        raise ValueError("prepared registration evidence mismatch")
    return plan


def summary(plan: dict) -> dict:
    result = {
        "model_id": plan["registration"]["id"],
        "api": API,
        "plan_sha256": plan["receipt_sha256"],
        "registration_sha256": plan["registration_sha256"],
        "execution_contract_sha256": plan["execution_contract_sha256"],
        "serving_ready": False,
    }
    if plan["schema"] == SCHEMA:
        result["optimizer_step"] = plan["optimizer_step"]
    else:
        result.update(
            artifact_kind="miles_rl",
            source_update_identity_sha256=plan["source_update_identity_sha256"],
        )
    return result


def _code(schema: str = SCHEMA) -> dict:
    names = [
        "serving_registration.py",
        "register_post_sft.py",
        "io.py",
    ]
    if schema == MILES_SCHEMA:
        names.append("miles_hf_export.py")
    return {name: file_sha256(Path(__file__).with_name(name)) for name in names}


def _overlap(left: str, right: str) -> bool:
    a, b = Path(left), Path(right)
    return a == b or a.is_relative_to(b) or b.is_relative_to(a)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("control-plane redirect refused")


class Client:
    def __init__(self, key: str):
        if not key:
            raise ValueError("FLEET_API_KEY is required")
        self.key = key
        self.opener = urllib.request.build_opener(_NoRedirect)

    def request(
        self, method: str, url: str, body: dict | None = None, *, idempotency: str | None = None
    ) -> dict:
        if method not in {"GET", "POST"} or (
            method == "POST" and (url != API or body is None or not idempotency)
        ):
            raise ValueError("only GET or explicit idempotency-bound model POST is supported")
        if (
            url != ACCOUNT
            and url != API
            and not url.startswith(API + "?")
            and not url.startswith(API + "/")
        ):
            raise ValueError("unapproved control-plane URL")
        headers = {"Authorization": "Bearer " + self.key}
        if idempotency:
            headers["Idempotency-Key"] = idempotency
        if body is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            url,
            method=method,
            headers=headers,
            data=None if body is None else canonical_json(body).encode(),
        )
        try:
            with self.opener.open(request, timeout=60) as response:
                value = json.load(response)
        except urllib.error.HTTPError as exc:
            # Never include a server body, URL query, environment or key.
            raise RuntimeError(f"control-plane HTTP {exc.code}") from None
        if not isinstance(value, dict):
            raise ValueError("unknown control-plane response schema")
        return value

    def inventory(self) -> list[dict]:
        rows, url, seen = [], API, set()
        while url:
            if url in seen or len(seen) >= 1000:
                raise ValueError("model pagination cycle/limit")
            seen.add(url)
            page = self.request("GET", url)
            if (
                set(page) - {"object", "data", "has_more", "next_cursor"}
                or page.get("object") != "list"
                or not isinstance(page.get("data"), list)
            ):
                raise ValueError("unknown model-list schema")
            if any(
                not isinstance(row, dict)
                or not isinstance(row.get("id"), str)
                or not isinstance(row.get("spec"), dict)
                for row in page["data"]
            ):
                raise ValueError("model-list row lacks exact desired spec")
            rows.extend(page["data"])
            more, cursor = page.get("has_more", False), page.get("next_cursor")
            if (
                type(more) is not bool
                or bool(cursor) != more
                or (more and not isinstance(cursor, str))
            ):
                raise ValueError("ambiguous model pagination")
            url = API + "?" + urllib.parse.urlencode({"cursor": cursor}) if more else None
        if len({row["id"] for row in rows}) != len(rows):
            raise ValueError("duplicate model catalog identities")
        return rows


def preview(directory: Path, client: Client) -> dict:
    plan = load(directory)
    account = client.request("GET", ACCOUNT)
    team = account.get("team_id") or account.get("team", {}).get("id") or account.get("id")
    if team != TEAM_ID:
        raise ValueError("Fleet account identity mismatch")
    rows = client.inventory()
    base = plan["base_registration"]
    matches = [row for row in rows if row["id"] == base["id"]]
    if len(matches) != 1 or _canonical_spec(matches[0]["spec"]) != _canonical_spec(base["spec"]):
        raise ValueError("live baseline missing or desired spec drifted")
    candidate = plan["registration"]
    for row in rows:
        model = row["spec"].get("model")
        if not isinstance(model, dict) or any(
            not isinstance(model.get(field), str) or not model[field]
            for field in ("sourcePath", "path", "revision")
        ):
            raise ValueError("model catalog lacks complete duplicate-check identities")
        _absolute(model["sourcePath"])
        _absolute(model["path"])
        if (
            row["id"] == candidate["id"]
            or model.get("revision") == candidate["spec"]["model"]["revision"]
            or any(
                isinstance(model.get(field), str)
                and _overlap(model[field], candidate["spec"]["model"][field])
                for field in ("sourcePath", "path")
            )
        ):
            raise FileExistsError(
                "model identity/source/revision already claimed; reconcile, do not POST"
            )
    return {
        **summary(plan),
        "preview": "local_and_authenticated_GETs",
        "model_count": len(rows),
        "duplicates": 0,
        "server_dry_run": False,
    }


def _qualification(plan: dict, path: Path, expected: str) -> dict:
    proof = _read(path, expected, signed=True)
    if (
        proof.get("schema") != DEV_SCHEMA
        or proof.get("status") != "passed"
        or proof.get("cluster") != "dev"
        or proof.get("api_base_url") != "https://api.ft.dev.flt.build"
        or _sha(proof.get("execution_contract_sha256")) != plan["execution_contract_sha256"]
        or _sha(proof.get("export_receipt_sha256")) != plan["export_receipt_sha256"]
        or set(proof.get("checks", {})) != DEV_CHECKS
        or any(value is not True for value in proof["checks"].values())
        or not proof.get("controller_uid")
        or not proof.get("pod_uid")
        or not proof.get("observed_at")
    ):
        raise ValueError("reviewed exact dev serving qualification is required")
    if plan["schema"] == MILES_SCHEMA and (
        _sha(proof.get("source_update_identity_sha256")) != plan["source_update_identity_sha256"]
        or _sha(proof.get("reload_acceptance_receipt_sha256"))
        != plan["reload_acceptance_receipt_sha256"]
        or _sha(proof.get("staged_manifest_sha256")) != plan["staged_manifest_sha256"]
    ):
        raise ValueError("Miles dev qualification does not bind the exact staged update")
    for key in ("controller_uid", "pod_uid"):
        if str(uuid.UUID(proof[key])) != proof[key]:
            raise ValueError("dev object UID is not canonical")
    if datetime.fromisoformat(proof["observed_at"].replace("Z", "+00:00")).tzinfo is None:
        raise ValueError("dev evidence observation must have a timezone")
    image = plan["registration"]["spec"]["runtime"]["image"]
    if proof.get("runtime_image_id") != image["repository"] + "@" + _sha(image["digest"]):
        raise ValueError("dev runtime image differs from registration")
    _sha(proof.get("evidence_manifest_sha256"))
    return proof


@contextmanager
def _lock(directory: Path):
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("prepared directory must be real")
    fd = os.open(directory / "owner.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)


def _matching_result(plan: dict, result: dict) -> dict:
    if result.get("id") != plan["registration"]["id"] or _canonical_spec(
        result.get("spec", {})
    ) != _canonical_spec(plan["registration"]["spec"]):
        raise ValueError("registration readback differs from full immutable desired spec")
    version = result.get("resource_version")
    if not (
        (isinstance(version, str) and version.strip() == version and bool(version))
        or (type(version) is int and version >= 0)
    ):
        raise ValueError("registration has no resource-version identity")
    return _signed(
        {
            "schema": "cyber_serving_registration_result_v2",
            "plan_sha256": plan["receipt_sha256"],
            "registration_sha256": plan["registration_sha256"],
            "model_id": result["id"],
            "resource_version": version,
            "registered": True,
            "serving_ready": False,
            "live_parity_verified": False,
            "observed_at": datetime.now(UTC).isoformat(),
        }
    )


def execute(directory: Path, client: Client, *, dev_evidence: Path, dev_sha256: str) -> dict:
    with _lock(directory):
        plan = load(directory)
        proof = _qualification(plan, dev_evidence, dev_sha256)
        if any((directory / name).exists() for name in ("intent.json", "registered.json")):
            raise FileExistsError("registration already claimed; use read-only reconcile")
        checked = preview(directory, client)
        if checked["plan_sha256"] != plan["receipt_sha256"]:
            raise ValueError("plan changed during final registration preview")
        intent = _signed(
            {
                "schema": "cyber_serving_registration_intent_v2",
                "api": API,
                "plan_sha256": plan["receipt_sha256"],
                "registration_sha256": plan["registration_sha256"],
                "dev_qualification_sha256": _sha(proof["receipt_sha256"]),
                "preview": checked,
                "created_at": datetime.now(UTC).isoformat(),
            }
        )
        _once(directory / "intent.json", intent)
        # Exactly one HTTP attempt. Any exception leaves the intent permanently.
        result = client.request(
            "POST",
            API,
            plan["registration"],
            idempotency="exact-serving-" + plan["receipt_sha256"].removeprefix("sha256:"),
        )
        accepted = _matching_result(plan, result)
        _once(directory / "registered.json", accepted)
        return accepted


def reconcile(directory: Path, client: Client) -> dict:
    with _lock(directory):
        plan = load(directory)
        intent = _read(directory / "intent.json", signed=True)
        if intent.get("plan_sha256") != plan["receipt_sha256"] or intent.get("api") != API:
            raise ValueError("intent does not belong to this exact plan/API")
        account = client.request("GET", ACCOUNT)
        if (
            account.get("team_id") or account.get("team", {}).get("id") or account.get("id")
        ) != TEAM_ID:
            raise ValueError("Fleet account identity mismatch")
        result = client.request(
            "GET", API + "/" + urllib.parse.quote(plan["registration"]["id"], safe="")
        )
        accepted = _matching_result(plan, result)
        path = directory / "registered.json"
        if path.exists():
            existing = _read(path, signed=True)
            if (
                existing.get("plan_sha256") != plan["receipt_sha256"]
                or existing.get("registration_sha256") != plan["registration_sha256"]
            ):
                raise ValueError("existing registration receipt differs")
            return existing
        _once(path, accepted)
        return accepted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--config", required=True, type=Path)
    prepare_parser.add_argument("--output", required=True, type=Path)
    for name in ("preview", "execute", "reconcile"):
        sub = commands.add_parser(name)
        sub.add_argument("directory", type=Path)
        if name == "execute":
            sub.add_argument("--dev-evidence", type=Path, required=True)
            sub.add_argument("--dev-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        if args.action == "prepare":
            result = prepare(_read(args.config), args.output)
        else:
            client = Client(os.environ.get("FLEET_API_KEY", ""))
            if args.action == "execute":
                result = execute(
                    args.directory,
                    client,
                    dev_evidence=args.dev_evidence,
                    dev_sha256=args.dev_sha256,
                )
            elif args.action == "preview":
                result = preview(args.directory, client)
            else:
                result = reconcile(args.directory, client)
        print(canonical_json(result))
        return 0
    except Exception as exc:
        print(
            canonical_json({"error_type": type(exc).__name__, "action": args.action}),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
