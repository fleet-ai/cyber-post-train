import importlib.util
from pathlib import Path

ROOT = Path(__file__).parents[1]
QUALIFIER = (
    ROOT / "configs/qualification/qwen38_miles_runtime_qualification_v1.py"
)
SPEC = importlib.util.spec_from_file_location("miles_runtime_qualification", QUALIFIER)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_native_parser_requires_real_cuda_not_the_driver_stub() -> None:
    assert MODULE.native_parser_required("cuda") is True
    assert MODULE.native_parser_required("cuda_stub") is False
    assert MODULE.native_parser_required(None) is False


def test_cpu_import_receipt_cannot_be_mistaken_for_cuda_acceptance() -> None:
    source = QUALIFIER.read_text()
    assert 'else "image_qualified_cpu"' in source
    assert (
        'checks["native_256k_parser_checked"] = parse_native and '
        "native_parser_required(" in source
    )
    assert (
        "if native_parser_required(driver_import_mode):\n        "
        "miles_training._validate_long_runtime_receipt"
    ) in source


def test_runtime_gate_binds_hybrid_world_and_installed_distributed_sources() -> None:
    source = QUALIFIER.read_text()
    assert '"distributed-backend": miles.LONG_DISTRIBUTED_BACKEND' in source
    assert "parsed.distributed_backend == miles.LONG_DISTRIBUTED_BACKEND" in source
    assert "backend_config.device_backend_map.items()" in source
    assert '{"cpu": "gloo", "cuda": "nccl"}' in source
    assert '"megatron_optimizer_sha256"' in source
    assert '"miles_distributed_source_sha256"' in source
