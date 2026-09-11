"""Synthetic-only registration transaction tests: no model or network access."""

import copy
import json
from pathlib import Path

import pytest

from training import serving_registration as s
from training.io import digest_json, file_sha256
from training.register_post_sft import register


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return {"path": str(path), "sha256": file_sha256(path)}


@pytest.fixture
def prepared(tmp_path):
    tmp_path = tmp_path.resolve()
    stage = tmp_path / "cache" / "cyber-sft" / "selected-step50-v1"
    stage.mkdir(parents=True)
    names = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "model.safetensors.index.json",
        "model.safetensors",
    ]
    for name in names:
        (stage / name).write_bytes(b"synthetic fixture, not model bytes\n")
    export = s._signed(
        {
            "schema": "cyber_native_checkpoint_hf_export_v1",
            "model_repo": "Qwen/Qwen3.8-27B",
            "dtype": "BF16",
            "optimizer_steps_executed": 0,
            "optimizer_step": 50,
            "source_checkpoint_receipt_sha256": "sha256:" + "a" * 64,
            "source_plan_sha256": "b" * 64,
            "all_output_tensors_reopened_equal": True,
            "source_inventory_sizes_mtimes_unchanged": True,
            "files": {
                name: {
                    "bytes": (stage / name).stat().st_size,
                    "sha256": file_sha256(stage / name).removeprefix("sha256:"),
                }
                for name in names
            },
        }
    )
    export_ref = write(stage / "EXPORT.json", export)
    gpu_ref = write(
        tmp_path / "GPU.json",
        s._signed(
            {
                "schema": "cyber_hf_export_check_v1",
                "status": "passed",
                "gpus": 1,
                "gpu_reload_verified": True,
                "source_unchanged": True,
                "finite_logits": True,
                "synthetic_only": True,
                "optimizer_steps_executed": 0,
                "export_sha256": export_ref["sha256"].removeprefix("sha256:"),
                "export_receipt_sha256": export["receipt_sha256"],
            }
        ),
    )
    base = {
        "id": "qwen-base-v1",
        "spec": {
            "displayName": "Qwen base",
            "capabilities": ["chat_completions", "tool_calling"],
            "model": {
                "sourcePath": "/models/qwen-base/revision",
                "path": "/scratch/models/qwen-base/revision",
                "revision": "f" * 40,
                "precision": "bf16",
                "tensorParallelSize": 1,
            },
            "runtime": {
                "engine": "sglang",
                "image": {"repository": "example/sglang", "digest": "sha256:" + "c" * 64},
                "command": ["python", "-m", "sglang.launch_server"],
                "args": [
                    "--model-path",
                    "/scratch/models/qwen-base/revision",
                    "--served-model-name",
                    "qwen-base-v1",
                    "--context-length",
                    "262144",
                ],
                "env": [{"name": "HF_HUB_OFFLINE", "value": "1"}],
            },
            "resources": {
                "requests": {"cpu": "8", "memory": "64Gi"},
                "limits": {"cpu": "8", "memory": "64Gi", "nvidia.com/gpu": 1},
            },
            "scaling": {"minReplicas": 1},
            "placement": {"nodeGroup": "reviewed-prod"},
        },
    }
    config = {
        "schema": s.SCHEMA,
        "base_registration": write(tmp_path / "base.json", base),
        "export": export_ref,
        "gpu_check": gpu_ref,
        "model_id": "chris-qwen-sweep-step50-v1",
        "display_name": "Qwen selected dev checkpoint",
        "storage": {
            "namespace": "inference",
            "pvc_name": "hf-cache-shared",
            "pvc_uid": "11111111-2222-3333-4444-555555555555",
            "staged_root": str(stage),
            "source_path": "/models/cyber-sft/selected-step50-v1",
        },
    }
    config["staging"] = write(
        tmp_path / "staging.json",
        s._signed(
            {
                "schema": "cyber_inference_staging_check_v1",
                "status": "passed",
                **config["storage"],
                "observer_pod_uid": "66666666-7777-8888-9999-aaaaaaaaaaaa",
                "observed_mount_root": str(tmp_path / "cache"),
                "catalog_root": "/models",
                "destination_create_once": True,
                "payload_rehashed": True,
                "mount_evidence_sha256": "sha256:" + "d" * 64,
                "export_receipt_sha256": export["receipt_sha256"],
                "export_file_sha256": export_ref["sha256"],
                "files_manifest_sha256": digest_json(export["files"]),
            }
        ),
    )
    directory = tmp_path / "prepared"
    s.prepare(config, directory)
    plan = s.load(directory)
    dev = s._signed(
        {
            "schema": s.DEV_SCHEMA,
            "status": "passed",
            "cluster": "dev",
            "api_base_url": "https://api.ft.dev.flt.build",
            "execution_contract_sha256": plan["execution_contract_sha256"],
            "export_receipt_sha256": plan["export_receipt_sha256"],
            "checks": {key: True for key in s.DEV_CHECKS},
            "controller_uid": "11111111-2222-3333-4444-555555555555",
            "pod_uid": "66666666-7777-8888-9999-aaaaaaaaaaaa",
            "observed_at": "2026-09-11T00:00:00Z",
            "runtime_image_id": "example/sglang@sha256:" + "c" * 64,
            "evidence_manifest_sha256": "sha256:" + "e" * 64,
        }
    )
    dev_ref = write(tmp_path / "dev.json", dev)
    return config, directory, base, dev_ref


class FakeClient:
    def __init__(self, base):
        self.rows = [copy.deepcopy(base)]
        self.calls = []
        self.failure = None
        self.team = s.TEAM_ID

    def inventory(self):
        self.calls.append(("GET", s.API))
        return copy.deepcopy(self.rows)

    def request(self, method, url, body=None, **kwargs):
        self.calls.append((method, url))
        if url == s.ACCOUNT:
            return {"team_id": self.team}
        if method == "POST":
            assert kwargs["idempotency"].startswith("exact-serving-")
            result = {**copy.deepcopy(body), "resource_version": "42"}
            if self.failure == "timeout_after_create":
                self.rows.append(result)
                raise TimeoutError("private response must not escape")
            if self.failure == "spec_drift":
                result["spec"]["model"]["precision"] = "fp8"
            self.rows.append(result)
            return result
        return next(row for row in self.rows if url.endswith("/" + row["id"]))


def execute(directory, client, dev):
    return s.execute(directory, client, dev_evidence=Path(dev["path"]), dev_sha256=dev["sha256"])


def test_configurable_checkpoint_changes_only_identity(prepared):
    config, directory, base, _ = prepared
    plan = s.load(directory)
    candidate = plan["registration"]
    assert candidate["id"] == config["model_id"]
    assert candidate["spec"]["model"]["revision"] == plan["export_receipt_sha256"]
    assert s.execution_contract(candidate) == s.execution_contract(base)
    assert s.preview(directory, FakeClient(base))["server_dry_run"] is False
    with pytest.raises(FileExistsError):
        s.prepare(config, directory)


@pytest.mark.parametrize("defect", ["reference_symlink", "reference_extra", "storage_extra"])
def test_prepare_rejects_unreviewed_reference_shapes(prepared, tmp_path, defect):
    config, _, _, _ = prepared
    config = copy.deepcopy(config)
    if defect == "reference_symlink":
        link = tmp_path / "export-reference.json"
        link.symlink_to(config["export"]["path"])
        config["export"]["path"] = str(link)
    elif defect == "reference_extra":
        config["export"]["unexpected"] = "not approved"
    else:
        config["storage"]["unexpected"] = "not approved"
    with pytest.raises(ValueError):
        s.prepare(config, tmp_path / "not-prepared")
    assert not (tmp_path / "not-prepared").exists()


def test_execute_is_one_post_and_never_claims_serving_ready(prepared):
    _, directory, base, dev = prepared
    client = FakeClient(base)
    result = execute(directory, client, dev)
    assert (
        result["registered"] and not result["serving_ready"] and not result["live_parity_verified"]
    )
    assert (directory / "intent.json").stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        execute(directory, client, dev)
    assert sum(method == "POST" for method, _ in client.calls) == 1
    assert s.reconcile(directory, client) == result


def test_uncertain_create_can_only_reconcile_without_post(prepared):
    _, directory, base, dev = prepared
    client = FakeClient(base)
    client.failure = "timeout_after_create"
    with pytest.raises(TimeoutError):
        execute(directory, client, dev)
    assert (directory / "intent.json").exists() and not (directory / "registered.json").exists()
    with pytest.raises(FileExistsError):
        execute(directory, client, dev)
    result = s.reconcile(directory, client)
    assert result["registered"]
    assert sum(method == "POST" for method, _ in client.calls) == 1


@pytest.mark.parametrize("field", ["id", "sourcePath", "path", "revision"])
def test_catalog_collision_prevents_mutation(prepared, field):
    _, directory, base, dev = prepared
    client = FakeClient(base)
    candidate = s.load(directory)["registration"]
    other = copy.deepcopy(base)
    other["id"] = "other-model"
    if field == "id":
        other["id"] = candidate["id"]
    else:
        other["spec"]["model"][field] = candidate["spec"]["model"][field]
    client.rows.append(other)
    with pytest.raises(FileExistsError):
        execute(directory, client, dev)
    assert not (directory / "intent.json").exists()
    assert not any(method == "POST" for method, _ in client.calls)


@pytest.mark.parametrize(
    "defect",
    [
        "team",
        "baseline",
        "payload",
        "extra",
        "symlink",
        "receipt",
        "dev",
        "dev_image",
        "dev_checkpoint",
    ],
)
def test_bad_evidence_stops_before_post(prepared, defect):
    config, directory, base, dev = prepared
    client = FakeClient(base)
    stage = Path(config["storage"]["staged_root"])
    if defect == "team":
        client.team = "wrong-team"
    elif defect == "baseline":
        client.rows[0]["spec"]["model"]["precision"] = "fp8"
    elif defect == "payload":
        (stage / "model.safetensors").write_bytes(b"modified")
    elif defect == "extra":
        (stage / "unbound").touch()
    elif defect == "symlink":
        (stage / "model.safetensors").unlink()
        (stage / "model.safetensors").symlink_to(stage / "tokenizer.json")
    elif defect == "receipt":
        Path(config["gpu_check"]["path"]).write_text("{}")
    else:
        path = Path(dev["path"])
        proof = json.loads(path.read_text())
        proof.pop("receipt_sha256")
        if defect == "dev":
            proof["checks"]["structured_tool_call"] = False
        elif defect == "dev_image":
            proof["runtime_image_id"] = "different@sha256:" + "f" * 64
        else:
            proof["export_receipt_sha256"] = "sha256:" + "f" * 64
        dev = write(path, s._signed(proof))
    with pytest.raises((ValueError, FileExistsError)):
        execute(directory, client, dev)
    assert not any(method == "POST" for method, _ in client.calls)


def test_changed_reply_is_held_and_not_retried(prepared):
    _, directory, base, dev = prepared
    client = FakeClient(base)
    client.failure = "spec_drift"
    with pytest.raises(ValueError, match="readback"):
        execute(directory, client, dev)
    assert (directory / "intent.json").exists() and not (directory / "registered.json").exists()
    with pytest.raises(FileExistsError):
        execute(directory, client, dev)


@pytest.mark.parametrize("version", [None, "", " ", "42 ", -1, True])
def test_result_requires_meaningful_resource_version(prepared, version):
    _, directory, _, _ = prepared
    plan = s.load(directory)
    result = {**copy.deepcopy(plan["registration"]), "resource_version": version}
    with pytest.raises(ValueError, match="resource-version"):
        s._matching_result(plan, result)


@pytest.mark.parametrize("version", ["42", "version-42", 0])
def test_result_accepts_observed_resource_version_types(prepared, version):
    _, directory, _, _ = prepared
    plan = s.load(directory)
    result = {**copy.deepcopy(plan["registration"]), "resource_version": version}
    assert s._matching_result(plan, result)["resource_version"] == version


def test_changed_final_preview_plan_stops_before_intent_or_post(prepared, monkeypatch):
    _, directory, base, dev = prepared
    original = s.preview

    def changed_preview(directory, client):
        result = original(directory, client)
        result["plan_sha256"] = "sha256:" + "f" * 64
        return result

    monkeypatch.setattr(s, "preview", changed_preview)
    client = FakeClient(base)
    with pytest.raises(ValueError, match="changed during"):
        execute(directory, client, dev)
    assert not (directory / "intent.json").exists()
    assert not any(method == "POST" for method, _ in client.calls)


def test_plan_cannot_change_placement_by_rehashing(prepared):
    _, directory, _, _ = prepared
    path = directory / "plan.json"
    plan = json.loads(path.read_text())
    plan.pop("receipt_sha256")
    plan["registration"]["spec"]["placement"] = {"nodeGroup": "unreviewed"}
    plan["registration_sha256"] = digest_json(plan["registration"])
    write(path, s._signed(plan))
    with pytest.raises(ValueError):
        s.load(directory)


def test_plan_cannot_misreport_export_optimizer_step(prepared):
    _, directory, _, _ = prepared
    path = directory / "plan.json"
    plan = json.loads(path.read_text())
    plan.pop("receipt_sha256")
    plan["optimizer_step"] += 1
    write(path, s._signed(plan))
    with pytest.raises(ValueError):
        s.load(directory)


def test_incomplete_catalog_identity_stops_before_post(prepared):
    _, directory, base, dev = prepared
    client = FakeClient(base)
    incomplete = copy.deepcopy(base)
    incomplete["id"] = "another-model"
    incomplete["spec"]["model"].pop("sourcePath")
    client.rows.append(incomplete)
    with pytest.raises(ValueError, match="duplicate-check identities"):
        execute(directory, client, dev)
    assert not (directory / "intent.json").exists()
    assert not any(method == "POST" for method, _ in client.calls)


def test_client_paginates_and_rejects_unknown_partial_schema():
    client = object.__new__(s.Client)
    calls = []

    def request(method, url):
        calls.append(url)
        if len(calls) == 1:
            return {
                "object": "list",
                "data": [{"id": "a", "spec": {}}],
                "has_more": True,
                "next_cursor": "next",
            }
        return {
            "object": "list",
            "data": [{"id": "b", "spec": {}}],
            "has_more": False,
            "next_cursor": None,
        }

    client.request = request
    assert [x["id"] for x in client.inventory()] == ["a", "b"]
    assert calls[-1] == s.API + "?cursor=next"
    for page in [
        {"object": "list", "data": [], "next": "unrecognized"},
        {"object": "list", "data": [], "has_more": True},
        {"object": "list", "data": [], "next_cursor": "dangling"},
        {"object": "list", "data": [{"id": "incomplete"}]},
    ]:
        client.request = lambda *_, p=page: p
        with pytest.raises(ValueError):
            client.inventory()


def test_legacy_registration_cannot_mutate():
    with pytest.raises(ValueError, match="retired"):
        register({}, "not-a-real-key")


def test_no_redirect_or_unapproved_url():
    client = s.Client("synthetic-key")
    with pytest.raises(ValueError, match="unapproved"):
        client.request("GET", "https://unapproved.invalid/")
    with pytest.raises(ValueError, match="redirect"):
        s._NoRedirect().redirect_request(None, None, 302, None, None, "https://other.invalid")
    with pytest.raises(ValueError, match="only GET"):
        client.request("DELETE", s.API + "/model")
    with pytest.raises(ValueError, match="only GET"):
        client.request("POST", s.API, {})


@pytest.mark.parametrize("defect", ["credential", "shell", "paused", "unknown_field", "autoscale"])
def test_unsafe_baseline_is_not_serialized(prepared, tmp_path, defect):
    config, _, base, _ = prepared
    if defect == "credential":
        base["spec"]["runtime"]["args"].extend(["--api-key", "synthetic-key"])
    elif defect == "shell":
        base["spec"]["runtime"]["command"] = ["bash", "-c", "unreviewed shell"]
    elif defect == "paused":
        base["spec"]["desiredState"] = "paused"
    elif defect == "unknown_field":
        base["spec"]["credential"] = "must-not-be-persisted"
    else:
        base["spec"]["scaling"]["maxReplicas"] = 3
    config["base_registration"] = write(Path(config["base_registration"]["path"]), base)
    output = tmp_path / "rejected"
    with pytest.raises(ValueError):
        s.prepare(config, output)
    assert not output.exists()


def test_overlapping_cache_path_is_a_duplicate(prepared):
    _, directory, base, dev = prepared
    other = copy.deepcopy(base)
    other["id"] = "other-serving-model"
    other["spec"]["model"]["sourcePath"] = "/models/cyber-sft"
    client = FakeClient(base)
    client.rows.append(other)
    with pytest.raises(FileExistsError):
        execute(directory, client, dev)
    assert not any(method == "POST" for method, _ in client.calls)


@pytest.mark.parametrize(
    "extra",
    [
        "--model-path=/different",
        "--model",
        "--model_path",
        "--model-p",
        "--served-model-name=different",
        "--served-model",
        "--served_model_name",
        "--",
    ],
)
def test_alternate_identity_arguments_are_rejected(prepared, tmp_path, extra):
    config, _, base, _ = prepared
    base["spec"]["runtime"]["args"].append(extra)
    config["base_registration"] = write(Path(config["base_registration"]["path"]), base)
    with pytest.raises(ValueError):
        s.prepare(config, tmp_path / "alternate")


@pytest.mark.parametrize("defect", ["source_path", "mount", "manifest", "pvc_uid", "missing"])
def test_stage_mount_receipt_must_bind_real_destination(prepared, tmp_path, defect):
    config, _, _, _ = prepared
    if defect == "missing":
        config.pop("staging")
    else:
        path = Path(config["staging"]["path"])
        proof = json.loads(path.read_text())
        proof.pop("receipt_sha256")
        if defect == "source_path":
            proof["source_path"] = "/models/other-file-tree"
        elif defect == "mount":
            proof["observed_mount_root"] = str(tmp_path / "cache" / "cyber-sft")
        elif defect == "manifest":
            proof["files_manifest_sha256"] = "sha256:" + "f" * 64
        else:
            proof["pvc_uid"] = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        config["staging"] = write(path, s._signed(proof))
    with pytest.raises(ValueError):
        s.prepare(config, tmp_path / "invalid-stage")


def test_cli_failure_is_sanitized(tmp_path, capsys):
    config = tmp_path / "config.json"
    config.write_text('{"private-server-body":"not emitted"}')
    assert s.main(["prepare", "--config", str(config), "--output", str(tmp_path / "out")]) == 1
    output = capsys.readouterr()
    assert "private-server-body" not in output.err and "not emitted" not in output.err
    assert json.loads(output.err) == {"action": "prepare", "error_type": "ValueError"}
