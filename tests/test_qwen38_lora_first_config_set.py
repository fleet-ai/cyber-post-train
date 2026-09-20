import hashlib
import json
from pathlib import Path

import pytest

from training.science import validate_eval_protocol

ROOT = Path(__file__).parents[1]
RUNS = ROOT / "configs" / "runs"
EVALS = ROOT / "configs" / "evaluation"
DATA = ROOT / "configs" / "data"
FILTER_REQUEST = DATA / "qwen38-fresh75-teacher-sft-final-lock-filter-v2.request.json"
FILTERED_MANIFEST = DATA / "qwen38-fresh75-teacher-sft-final-lock-free-v2.manifest.json"
QUALIFICATION = (
    ROOT / "configs" / "qualification" / "qwen38-lora-megatron-trainer-image-2026-09-20-v1.json"
)
CORPUS_QUALIFICATION = (
    ROOT / "configs" / "qualification" / "qwen38-lora-one-step-corpus-2026-09-20-v1.json"
)
QUALIFIED_SOURCE = "7e9356c8e02e7382e84b8484638baccdd1bbf680"
QUALIFIED_IMAGE = (
    "ghcr.io/fleet-ai/skyrl-fleet-v2/trainer@sha256:"
    "7da4adba80d032509dba69fb4dd23bedca17fde3e2b88643815f80d6ee6c5317"
)
GATE_TEMPLATE = RUNS / "qwen38-27b-lora-sft-r64-a32-one-step-v4.template.json"
RETIRED_GATE_TEMPLATES = [
    RUNS / "qwen38-27b-lora-sft-r64-a32-one-step-v1.template.json",
    RUNS / "qwen38-27b-lora-sft-r64-a32-one-step-v2.template.json",
    RUNS / "qwen38-27b-lora-sft-r64-a32-one-step-v3.template.json",
]


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_first_qwen38_lora_sft_templates_freeze_supported_surface():
    gate = read(GATE_TEMPLATE)
    anchor = read(RUNS / "qwen38-27b-lora-sft-r64-a32-anchor-v1.template.json")

    allowed = {
        "backend",
        "name",
        "output_root",
        "model",
        "data",
        "recipe",
        "wandb",
        "cluster",
        "lora",
        "runtime",
        "pause_after_step",
    }
    for value in (gate, anchor):
        assert set(value) <= allowed
        assert value["backend"] == "skyrl"
        assert value["model"]["lock"].endswith("qwen38-27b-1d4bf0f2.lock.json")
        assert value["model"]["weights"].endswith("qwen38-27b-1d4bf0f2.weights.json")
        assert value["lora"] == {
            "type": "lora",
            "target_modules": "all-linear",
            "rank": 64,
            "alpha": 32,
            "init_method": "kaiming",
            "dropout": 0.0,
        }
        assert value["recipe"]["lr"] == 3e-5
        assert value["recipe"]["nodes"] == 1
        assert value["recipe"]["gpus_per_node"] == 8
        assert value["cluster"]["priority"] == "c1"
        assert value["wandb"]["entity"] == "thefleet"
        assert value["wandb"]["project"] == "cyber-post-train"
        assert value["wandb"]["run_id"] == value["name"]
        assert len(value["name"]) <= 31

    assert gate["pause_after_step"] == 1
    assert gate["name"] == "chris-q38-lora-sft-c1-v4"
    assert gate["output_root"] == "/mnt/sfs/jobs/chris-q38-lora-sft-c1-v4"
    assert gate["recipe"]["batch_size"] == 1
    assert gate["recipe"]["max_length"] == 16384
    assert gate["data"] == {
        "manifest": "../data/qwen38-fresh75-teacher-sft-final-lock-free-v2.manifest.json",
        "root": (
            "/mnt/sfs/jobs/chris-q38-study-corpora-v1/fresh75-teacher-final-lock-free-v2/data"
        ),
    }
    assert gate["runtime"] == {
        "skyrl_source_commit": QUALIFIED_SOURCE,
        "image": QUALIFIED_IMAGE,
    }
    assert anchor.get("pause_after_step") is None
    assert anchor["runtime"] == {"skyrl_source_commit": None, "image": None}
    assert anchor["recipe"]["batch_size"] == 8
    assert anchor["recipe"]["max_length"] == 32768
    assert "broad-teacher-actions" in anchor["data"]["manifest"]


@pytest.mark.parametrize("retired", RETIRED_GATE_TEMPLATES)
def test_failed_operational_identities_are_retired(retired):
    from training import sft

    with pytest.raises(ValueError, match="exact digest-bound one-step"):
        sft.compile_sft(read(retired), relative_to=RUNS)


def test_one_step_template_binds_independently_verified_leak_free_corpus():
    from training import sft, sft_runtime

    value = read(GATE_TEMPLATE)
    manifest = read(FILTERED_MANIFEST)

    assert sft_runtime.QWEN38_MEGATRON_SKYRL_REVISION == QUALIFIED_SOURCE
    assert sft_runtime.QWEN38_MEGATRON_IMAGE == QUALIFIED_IMAGE
    assert (
        "sha256:" + hashlib.sha256(FILTERED_MANIFEST.read_bytes()).hexdigest()
        == "sha256:9e144c94b6ad85dc715e100ac5ae6689d6d385972fdfa378d39d8da1b0ccbed5"
    )
    assert manifest["sha256"] == (
        "sha256:5db5600ac9f403fd147b2ecbf6068b514d075d07ac683d97bc83319c6e89d149"
    )
    assert manifest["split_sha256"] == (
        "sha256:b7b536940995a0d4b8674a4bdadd6240ef88dc1c07200a93f27b592ac8c046ea"
    )
    train = manifest["files"]["train"]
    assert train["sha256"] == (
        "sha256:9bac7eef01ff7dfff5de82f139a3ccaaabacf26fe070c713396874355b8ecfbc"
    )
    assert train["rows"] == 866
    assert train["source_sessions"] == 108
    assert train["supervised_tokens"] == 998652
    assert len(train["task_keys"]) == 35

    plan = sft.compile_sft(value, relative_to=RUNS)
    assert sft_runtime._qwen38_lora_one_step_identity(plan) == (
        sft_runtime.qwen38_lora_one_step_plan_binding()
    )
    assert plan["recipe"]["max_steps"] == 866
    assert plan["pause_after_step"] == 1
    assert plan["datasets"]["train"]["path"] == (
        "/mnt/sfs/jobs/chris-q38-study-corpora-v1/"
        "fresh75-teacher-final-lock-free-v2/data/train.parquet"
    )
    request = sft.job_request(plan)
    assert request["workers"] == 1
    assert request["gpus_per_worker"] == 8
    assert request["priority_class"] == "c1"
    assert request["requeueIfPreempted"] is False


def test_one_step_runtime_accepts_only_digest_enrichment_of_the_exact_plan():
    from training import sft, sft_runtime

    value = read(GATE_TEMPLATE)
    source_plan = sft.compile_sft(value, relative_to=RUNS)
    plan_sha256 = sft_runtime._unsigned_digest(source_plan)
    runtime_plan = {**source_plan, "plan_sha256": plan_sha256}

    # This is the exact transition performed by the packaged entrypoint after
    # it verifies plan.json against --plan-sha256.  It must not reject its own
    # runtime-only evidence field before model setup.
    sft_runtime._validate_entrypoint_sources(runtime_plan, verify_qwen_files=False)
    assert sft_runtime._qwen38_lora_one_step_identity(runtime_plan) == (
        sft_runtime.qwen38_lora_one_step_plan_binding()
    )

    # Conversely, that runtime-only field may never be serialized back into a
    # prepared Jobs API request.
    with pytest.raises(ValueError, match="runtime-only evidence"):
        sft.job_request(runtime_plan)


def test_corpus_qualification_receipt_binds_manifest_template_and_runtime():
    from training import sft, sft_runtime

    receipt = read(CORPUS_QUALIFICATION)
    embedded = receipt.pop("receipt_sha256")
    expected = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    assert embedded == expected
    assert receipt["schema"] == "qwen38_lora_one_step_corpus_qualification_v1"
    assert receipt["status"] == "qualified_corpus"
    assert receipt["training_submitted"] is False

    corpus = receipt["corpus"]
    manifest_path = ROOT / corpus["public_manifest_path"]
    assert manifest_path == FILTERED_MANIFEST
    assert corpus["manifest_file_sha256"] == (
        "sha256:" + hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    )
    manifest = read(manifest_path)
    train = manifest["files"]["train"]
    assert corpus["manifest_sha256"] == manifest["sha256"]
    assert corpus["train_parquet_sha256"] == train["sha256"]
    assert corpus["rows"] == train["rows"]
    assert corpus["source_sessions"] == train["source_sessions"]
    assert corpus["supervised_tokens"] == train["supervised_tokens"]
    assert corpus["task_versions"] == len(train["task_keys"])

    request = read(FILTER_REQUEST)
    assert receipt["request"] == {
        "file_sha256": "sha256:" + hashlib.sha256(FILTER_REQUEST.read_bytes()).hexdigest(),
        "final_test_lock_file_sha256": request["protection"]["final_test_lock_file_sha256"],
        "final_test_lock_sha256": request["protection"]["final_test_lock_sha256"],
        "inventory_file_sha256": request["protection"]["inventory_file_sha256"],
        "inventory_sha256": request["protection"]["inventory_sha256"],
        "path": str(FILTER_REQUEST.relative_to(ROOT)),
        "request_sha256": request["sha256"],
        "selection_sha256": request["selection"]["sha256"],
        "source_manifest_file_sha256": request["source"]["manifest_file_sha256"],
        "source_manifest_sha256": request["source"]["manifest_sha256"],
        "source_selection_sha256": request["source"]["source_selection_sha256"],
        "source_train_parquet_sha256": request["source"]["train_parquet_sha256"],
    }
    assert receipt["request"]["selection_sha256"] == manifest["split_sha256"]

    gate = read(GATE_TEMPLATE)
    assert gate["data"] == {
        "manifest": "../data/" + manifest_path.name,
        "root": corpus["data_root"],
    }
    plan = sft.compile_sft(gate, relative_to=RUNS)
    runtime = sft_runtime.qwen38_lora_one_step_plan_binding()
    assert sft_runtime._qwen38_lora_one_step_identity(plan) == runtime
    assert runtime["corpus_manifest_sha256"] == corpus["manifest_sha256"]
    assert runtime["split_manifest_sha256"] == receipt["request"]["selection_sha256"]
    assert runtime["datasets_sha256"] == sft_runtime._unsigned_digest(plan["datasets"])
    assert plan["datasets"]["train"]["sha256"] == corpus["train_parquet_sha256"]
    assert plan["datasets"]["train"]["rows"] == corpus["rows"]
    assert runtime["recipe"]["max_steps"] == corpus["rows"]


def test_final_lock_filter_request_is_sealed_family_aware_and_zero_overlap():
    request = read(FILTER_REQUEST)
    source = read(ROOT / request["source"]["manifest_path"])
    final_lock = read(ROOT / request["protection"]["final_test_lock_path"])
    inventory = read(ROOT / request["protection"]["inventory_path"])

    def logical_sha256(value: dict) -> str:
        unsigned = {key: item for key, item in value.items() if key != "sha256"}
        return (
            "sha256:"
            + hashlib.sha256(
                json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        )

    def file_sha256(path: Path) -> str:
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()

    assert request["schema"] == "cyber_dense_sft_corpus_final_lock_filter_request_v1"
    assert request["sha256"] == logical_sha256(request)
    assert request["selection"]["sha256"] == logical_sha256(request["selection"])
    assert request["protection"]["split_unit"] == ["application", "task_family"]
    assert request["protection"]["join_identity"] == [
        "task_key",
        "task_version_id",
    ]
    assert request["destination"]["create_once"] is True
    assert request["destination"]["publication"] == "atomic_no_replace"
    assert request["materialization"]["partial_source_session_removal"] == "reject"
    assert request["materialization"]["unknown_or_ambiguous_source_row"] == "reject"
    assert request["materialization"]["predicted_rows_sessions_tokens_or_file_sha256"] is False

    source_path = ROOT / request["source"]["manifest_path"]
    lock_path = ROOT / request["protection"]["final_test_lock_path"]
    inventory_path = ROOT / request["protection"]["inventory_path"]
    assert request["source"]["manifest_file_sha256"] == file_sha256(source_path)
    assert request["source"]["manifest_sha256"] == source["sha256"] == logical_sha256(source)
    assert request["source"]["train_parquet_sha256"] == source["files"]["train"]["sha256"]
    assert request["source"]["source_selection_sha256"] == source["source_sha256"]
    assert request["protection"]["final_test_lock_file_sha256"] == file_sha256(lock_path)
    assert (
        request["protection"]["final_test_lock_sha256"]
        == final_lock["sha256"]
        == logical_sha256(final_lock)
    )
    assert request["protection"]["inventory_file_sha256"] == file_sha256(inventory_path)
    assert (
        request["protection"]["inventory_sha256"]
        == inventory["sha256"]
        == logical_sha256(inventory)
    )

    inventory_by_identity = {
        (row["task_key"], row["task_version_id"]): row for row in inventory["tasks"]
    }
    inventory_by_key = {}
    for row in inventory["tasks"]:
        inventory_by_key.setdefault(row["task_key"], []).append(row)
    final_rows = [
        inventory_by_identity[(row["task_key"], row["task_version_id"])]
        for row in final_lock["tasks"]
    ]

    def unit(row: dict) -> tuple[str, str]:
        taxonomy = row["taxonomy"]
        return taxonomy["application"]["value"], taxonomy["task_family"]["value"]

    protected_units = {unit(row) for row in final_rows}
    assert len(protected_units) == request["selection"]["protected_task_families"] == 10
    source_rows = []
    for task_key in source["files"]["train"]["task_keys"]:
        candidates = inventory_by_key[task_key]
        assert len(candidates) == 1
        source_rows.append(candidates[0])
    excluded_rows = [row for row in source_rows if unit(row) in protected_units]
    excluded = request["selection"]["excluded_source_task_versions"]
    assert [
        {
            "application": unit(row)[0],
            "task_family": unit(row)[1],
            "task_key": row["task_key"],
            "task_version_id": row["task_version_id"],
        }
        for row in excluded_rows
    ] == excluded
    assert len(excluded_rows) == request["selection"]["excluded_source_task_version_count"] == 2
    retained_rows = [row for row in source_rows if unit(row) not in protected_units]
    assert len(retained_rows) == request["selection"]["retained_source_task_versions"] == 35
    assert not {unit(row) for row in retained_rows} & protected_units
    assert not {(row["task_key"], row["task_version_id"]) for row in retained_rows} & {
        (row["task_key"], row["task_version_id"]) for row in final_rows
    }
    filtered = read(FILTERED_MANIFEST)
    assert filtered["split_sha256"] == request["selection"]["sha256"]
    assert set(filtered["files"]["train"]["task_keys"]) == {
        row["task_key"] for row in retained_rows
    }


def test_image_qualification_receipt_binds_code_template_and_census():
    from training import sft_runtime

    receipt = read(QUALIFICATION)
    embedded = receipt.pop("receipt_sha256")
    expected = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )

    gate = read(GATE_TEMPLATE)
    anchor = read(RUNS / "qwen38-27b-lora-sft-r64-a32-anchor-v1.template.json")
    assert embedded == expected
    assert receipt["status"] == "qualified_for_cpu_preflight"
    assert receipt["paid_training_authorized"] is False
    assert receipt["source"]["revision"] == QUALIFIED_SOURCE
    assert receipt["image"]["immutable_ref"] == QUALIFIED_IMAGE
    assert receipt["source"]["source_files_sha256"] == (sft_runtime.QWEN38_MEGATRON_SOURCE_SHA256)
    assert receipt["source"]["source_file_count"] == len(receipt["source"]["source_files_sha256"])
    assert receipt["source"]["source_file_count"] == receipt["verification"]["source_file_count"]
    assert receipt["source"]["source_census_sha256"] == (
        "sha256:" + sft_runtime.QWEN38_MEGATRON_SOURCE_CENSUS_SHA256
    )
    assert (
        receipt["verification"]["source_census_sha256"] == receipt["source"]["source_census_sha256"]
    )
    assert receipt["image"]["immutable_ref"] == (
        "ghcr.io/fleet-ai/skyrl-fleet-v2/trainer@" + receipt["image"]["digest"]
    )
    assert receipt["verification"]["pod"]["requested_image"] == receipt["image"]["immutable_ref"]
    assert receipt["verification"]["pod"]["runtime_image_id"] == receipt["image"]["immutable_ref"]
    assert receipt["build_job"]["gpu_requests"] == 0
    assert receipt["verification"]["pod"]["gpu_requests"] == 0
    assert receipt["build_job"]["object_deleted"] is True
    assert receipt["build_job"]["pod_deleted"] is True
    assert receipt["build_job"]["gpu_release_verified"] is True
    assert receipt["verification"]["object_deleted"] is True
    assert receipt["verification"]["matching_pods_after_cleanup"] == 0
    assert receipt["verification"]["gpu_requests_after_cleanup"] == 0
    assert receipt["verification"]["training_submitted"] is False
    assert gate["runtime"] == {
        "skyrl_source_commit": receipt["source"]["revision"],
        "image": receipt["image"]["immutable_ref"],
    }
    assert anchor["runtime"] == {"skyrl_source_commit": None, "image": None}


def test_production_anchor_stays_fail_closed_after_one_step_binding():
    from training import sft

    value = read(RUNS / "qwen38-27b-lora-sft-r64-a32-anchor-v1.template.json")
    assert value["runtime"]["skyrl_source_commit"] is None
    assert value["runtime"]["image"] is None
    assert value.get("pause_after_step") is None
    # Give the anchor an existing manifest so this checks its unresolved
    # runtime/qualification binding rather than stopping at a missing future
    # broad-corpus file.
    value["data"]["manifest"] = "../data/qwen38-fresh75-teacher-sft-train-v1.manifest.json"
    value["data"]["root"] = "/mnt/sfs/jobs/chris-q38-study-corpora-v1/fresh75-teacher-max-v1/data"
    with pytest.raises(ValueError, match="source commit differs"):
        sft.compile_sft(value, relative_to=RUNS)


@pytest.mark.parametrize(
    "name",
    [
        "qwen38-lora-web-l0-opencode-pass1-v1.template.json",
        "qwen38-lora-fleet-final-opencode-pass1-v1.template.json",
    ],
)
def test_first_qwen38_lora_eval_templates_are_explicitly_unresolved(name):
    value = read(EVALS / name)
    assert value["schema"] == "cyber_prepost_eval_protocol_v1"
    assert value["model"]["repo"] == "Qwen/Qwen3.8-27B"
    assert value["harness"]["source_revision"] == "4b7e19e315cca414121ba1d61523fef74bb3ae8b"
    assert value["sampling"] == {
        "temperature": 0.6,
        "top_p": 0.95,
        "max_output_tokens": 32768,
    }
    assert value["context"]["max_context_tokens"] == 262144
    assert value["context"]["max_input_tokens"] == 229376
    assert value["context"]["automatic_compaction"] is True
    assert value["context"]["automatic_continuation"] is True
    assert value["context"]["compaction_headroom_tokens"] == 20000
    assert value["retry"]["scientific_attempt"] == {
        "automatic_retry": False,
        "max_retries": 0,
    }
    request_retry = value["retry"]["model_request"]
    if name.startswith("qwen38-lora-web-"):
        assert value["context"]["compaction_reserved_tokens"] == 20000
        assert request_retry["maximum_attempts"] == 5
        assert request_retry["retryable_http_statuses"] == [429, 502, 503, 504]
        assert request_retry["retry_transport_errors"] is True
        assert (
            value["context"]["renderer_source_sha256"]
            == "sha256:"
            + hashlib.sha256(
                (ROOT / "evals/webexploitbench/opencode_adapter/render_config.mjs").read_bytes()
            ).hexdigest()
        )
        assert (
            request_retry["implementation_source_sha256"]
            == "sha256:"
            + hashlib.sha256(
                (ROOT / "evals/webexploitbench/opencode_adapter/retry_proxy.mjs").read_bytes()
            ).hexdigest()
        )
    else:
        assert value["context"]["compaction_reserved_tokens"] == 52768
        assert request_retry["maximum_attempts"] == 1
        assert request_retry["retryable_http_statuses"] == []
        assert request_retry["retry_transport_errors"] is False
        assert (
            value["context"]["renderer_source_sha256"]
            == "sha256:"
            + hashlib.sha256(
                (ROOT / "evals/fleet/opencode_self_hosted.py").read_bytes()
            ).hexdigest()
        )
        assert (
            request_retry["implementation_source_sha256"]
            == "sha256:"
            + hashlib.sha256((ROOT / "evals/fleet/fixed_proxy.py").read_bytes()).hexdigest()
        )
    assert value["random_seeds"] == [42]
    with pytest.raises(ValueError, match="unresolved"):
        validate_eval_protocol(value)
