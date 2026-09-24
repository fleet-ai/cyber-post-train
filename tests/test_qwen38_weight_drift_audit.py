import importlib.util
import json
import struct
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_qwen38_weight_drift.py"
SPEC = importlib.util.spec_from_file_location("weight_drift", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def write_model(root: Path, values: dict[str, list[int]]) -> None:
    root.mkdir()
    offset = 0
    header = {}
    payload = bytearray()
    for key, tensor in values.items():
        raw = struct.pack(f"<{len(tensor)}H", *tensor)
        header[key] = {
            "dtype": "BF16",
            "shape": [len(tensor)],
            "data_offsets": [offset, offset + len(raw)],
        }
        payload.extend(raw)
        offset += len(raw)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    shard = "model-00001-of-00001.safetensors"
    (root / shard).write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": len(payload)},
                "weight_map": {key: shard for key in values},
            }
        )
    )


def test_layout_and_drift_separate_trained_from_frozen(tmp_path):
    names = {
        "mtp.fc.weight": [0x3F80, 0x4000],
        "model.visual.proj.weight": [0x3F80, 0x4000],
        "model.layers.0.self_attn.q_proj.weight": [0x3F80, 0x4000],
        "lm_head.weight": [0x3F80, 0x4000],
    }
    base, step1, step2 = tmp_path / "base", tmp_path / "s1", tmp_path / "s2"
    write_model(base, names)
    write_model(
        step1,
        {
            **names,
            "model.layers.0.self_attn.q_proj.weight": [0x3F81, 0x4000],
        },
    )
    write_model(
        step2,
        {
            **names,
            "model.layers.0.self_attn.q_proj.weight": [0x3F82, 0x4000],
            "lm_head.weight": [0x3F80, 0x4001],
        },
    )
    base_layout, _ = audit.layout(base)
    s1_layout, _ = audit.layout(step1)
    s2_layout, _ = audit.layout(step2)
    result = audit.audit_drift(base_layout, [("s1", s1_layout), ("s2", s2_layout)])
    assert result["base_vs_s1"]["global"]["changed_tensors"] == 1
    assert result["base_vs_s2"]["global"]["changed_tensors"] == 2
    assert result["base_vs_s2"]["categories"]["mtp"]["changed_tensors"] == 0
    assert result["base_vs_s2"]["categories"]["visual"]["changed_tensors"] == 0
    assert result["s1_vs_s2"]["global"]["changed_tensors"] == 2
    assert result["base_vs_s2"]["layer_summary"]["layers_with_changed_tensors"] == 1
    assert result["base_vs_s2"]["global"]["candidate_all_zero_tensors"] == 0
    interpretation = audit.interpret_drift(result, ["s1", "s2"])
    assert interpretation["all_structural_weight_checks_pass"]
    assert interpretation["checkpoints"]["s2"]["changed_mtp_tensors_vs_base"] == 0


def test_layout_rejects_index_header_disagreement(tmp_path):
    root = tmp_path / "model"
    write_model(root, {"model.weight": [0x3F80]})
    index = json.loads((root / "model.safetensors.index.json").read_text())
    index["weight_map"] = {"other.weight": "model-00001-of-00001.safetensors"}
    (root / "model.safetensors.index.json").write_text(json.dumps(index))
    try:
        audit.layout(root)
    except ValueError as exc:
        assert "index/header key mismatch" in str(exc)
    else:
        raise AssertionError("layout mismatch was accepted")


def test_export_validation_binds_exact_identity_and_payload(tmp_path):
    base, export = tmp_path / "base", tmp_path / "export"
    base.mkdir()
    export.mkdir()
    for name in audit.SIDECARS:
        (base / name).write_text(name)
        (export / name).write_text(name)
    (export / "payload.bin").write_bytes(b"payload")
    files = {
        path.name: {"bytes": path.stat().st_size, "sha256": audit.sha(path)}
        for path in export.iterdir()
    }
    receipt = {
        "schema": "cyber_native_checkpoint_hf_export_v1",
        "optimizer_step": 1000,
        "optimizer_steps_executed": 0,
        "dtype": "BF16",
        "all_output_tensors_reopened_equal": True,
        "source_inventory_sizes_mtimes_unchanged": True,
        "trained_tensors": 1184,
        "restored_base_tensors": sorted(audit.FROZEN_MTP_KEYS),
        "sidecars": {name: audit.sha(export / name) for name in audit.SIDECARS},
        "files": files,
    }
    receipt["receipt_sha256"] = audit.hashlib.sha256(audit.canonical(receipt)).hexdigest()
    receipt_path = export / "EXPORT.json"
    receipt_path.write_text(json.dumps(receipt))
    identity = (
        audit.sha(receipt_path),
        receipt["receipt_sha256"],
        audit.hashlib.sha256(audit.canonical(files)).hexdigest(),
    )
    result = audit.validate_export(export, base, "step1000", identity)
    assert result["payload_files_verified"] == len(files)
    (export / "payload.bin").write_bytes(b"tampered")
    try:
        audit.validate_export(export, base, "step1000", identity)
    except ValueError as exc:
        assert "payload digest mismatch" in str(exc)
    else:
        raise AssertionError("tampered payload was accepted")
