from __future__ import annotations

import ast
import copy
import gzip
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from cyber_post_train import public_eval_import
from evals.fleet import final_pass8_aggregate as final
from scripts import render_qwen38_fleet_pass8_final_aggregate as renderer

ROOT = Path(__file__).resolve().parents[1]
_TEST_BOUND_PATHS: dict[str, Path] = {}


@pytest.fixture(autouse=True)
def _isolated_bound_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    _TEST_BOUND_PATHS.clear()
    # Unit fixtures preserve the exact scientific definition but use synthetic
    # operational arm identities. Production defaults remain bound to the
    # authoritative provider-free receipt and are asserted separately below.
    monkeypatch.setattr(
        final,
        "FROZEN_MIGRATION_RECEIPT_SHA256",
        "sha256:af22ddc22e7494d19a0439e105dfe77a39fe016b99f696ceb740090ce012b87d",
    )
    monkeypatch.setattr(
        final,
        "FROZEN_MIGRATION_RECEIPT_FILE_SHA256",
        "sha256:b015df295968ce638204aa57181b2954ecd4988fe58b0342e4d8b176b88be8df",
    )
    monkeypatch.setattr(
        final,
        "FROZEN_COMPARISON_DEFINITION_FILE_SHA256",
        "sha256:c5d29211fa9ce5361d5483861e995bb45cf694f7fd2bd27806da6f6fa87d565a",
    )
    monkeypatch.setattr(
        final,
        "_bound_path",
        lambda value: _TEST_BOUND_PATHS.get(str(value), Path(str(value))),
    )


def test_production_source_binds_authoritative_protocol_v2_receipts() -> None:
    source = Path(final.__file__).read_text()
    assert "9813713ef2ab023ac6c64df494ca7cbf39b920e8fba2f05f5949daaa006479f1" in source
    assert "1356a652628619020ab7cbdd6b599a5b162dc99b47ee71b957f8428bb49227c6" in source
    assert "9fa3f39f9030dede9efaf38d8e0dfa218a36db885f8f6c72695ba5aa5ec13643" in source
    assert "fcee992ba4bb81527c5a63774fa76e9334163f7d583824ddb4b9f3b46c1c84d4" in source


class FakeSnapshot:
    def __init__(self, gate: final.GateSnapshot, scored: list[dict[str, Any]]) -> None:
        self.value = gate
        self.scored = scored
        self.score_reads = 0

    def gate(self) -> final.GateSnapshot:
        return self.value

    def scored_results(self) -> list[dict[str, Any]]:
        self.score_reads += 1
        return copy.deepcopy(self.scored)


class CallbackSnapshot(FakeSnapshot):
    def __init__(
        self,
        gate: final.GateSnapshot,
        scored: list[dict[str, Any]],
        callback: Any,
    ) -> None:
        super().__init__(gate, scored)
        self.callback = callback

    def scored_results(self) -> list[dict[str, Any]]:
        result = super().scored_results()
        self.callback()
        return result


def _signed(value: dict[str, Any], field: str = "sha256") -> dict[str, Any]:
    return {**value, field: final._digest(value)}  # noqa: SLF001


def _protocol(seed: int, *, replacement: bool) -> dict[str, Any]:
    protocol_sha256s = {
        46: "sha256:fc4c31caddf382f064bf4301f7ca88bcbf744207190386c491e055df23d30899",
        48: "sha256:6e8a15d372bda78978ea48c1889b7c5ef4a1f902e3dd85d534ede185cd5a646a",
        49: "sha256:29d56d78a5c2a2ea6f1763c8cc0fdda9bf7018088f71061986b92a27fdcd2315",
        50: "sha256:82f3f5fa3e41f854ef60a555a7bff2f373d5233cafa6802db18cb3347e09361c",
        54: "sha256:8355cf7a9682b96e2d83cdc5de5098f25954f3d21f85706fcdfe59c36073a319",
        55: "sha256:46775efd79f44e0c692be974ee7f48d270a43c6da25be90f60e56ffc20ce340d",
        56: "sha256:2792a3808fe69d14cb6cfe7a38c8860ba20a183f259505215a64eea2bdbdd086",
        57: "sha256:22a256cf7fb861e1720d6f80c2ea9b8bada730f1da6806f997fbd4e0dc9806b7",
        58: "sha256:fd645d784d659b0931881827cfd9ef0141c90213058e234eaa06225766e7521a",
        59: "sha256:1a79e7403620b7958e74b57fb362edc9435bc571a467ed2520b7096aef92c356",
        60: "sha256:a93ec807668d4734b7e65b00aa147684342a04654cb51947caf1be15d18c6b96",
    }
    suffix = "replacement-p1-v2" if replacement else "p1-v1"
    return {
        "seed": seed,
        "origin": "whole_pair_replacement" if replacement else "retained_original",
        "protocol_id": f"q38-dev17-s{seed}-base-t3k32s1000-{suffix}",
        "comparison_protocol_sha256": protocol_sha256s[seed],
    }


RUNTIME_FILES = {
    "evaluate.py": "33b88cf783bffdfa7fc194e8107afbd6721e762cb5d7020da9057c8d797e761e",
    "exact_pass4_crypto.py": "477b0531fbc34103ed91ff03668616a56f05687ede8c2c9ebb4e9dd2106337d5",
    "fixed_proxy.py": "a65270cb021cbc8e2fe9b00d901b4b70e9ddc98dd58d93e9f695b1768dfb984d",
    "opencode_self_hosted.py": "428e9f2e4d4c758c4f682cbed314cf96b866d051b34fa0eea8757a3aad97aa1d",
    "rollout_ledger.py": "633914e92d6a0fde6f7040f1fddf9a056c43ed9a41301f0baab1ae856397518c",
    "rollout_postgres.py": "74b946005267efa6611861135f135558ac32e10383e7c0619d278af8d8b1ccd9",
    "rollout_worker.py": "6782d014143a430c2c5523a2161f453a804e7203a07ee9e24792d705be0c224e",
}


def _arm_values(seed: int, arm: str, *, replacement: bool) -> tuple[str, str, str]:
    if replacement:
        model = "base" if arm == "base" else "t3k32s1000"
        experiment = f"q38-s{seed}-{model}-repl-p1-v2"
        short = f"{model}-repl-p1-v2"
        return (
            experiment,
            f"/mnt/sfs/jobs/chris-q38-fleet-dev17-s{seed}-{short}",
            f"q38_dev17_s{seed}_{model}_repl_p1_v2",
        )
    suffix = "base-p1-v1" if arm == "base" else "t3k32s1000-p1-v2"
    experiment = f"q38-dev17-s{seed}-{suffix}"
    return (
        experiment,
        f"/mnt/sfs/jobs/chris-q38-fleet-dev17-s{seed}-{suffix}",
        experiment.replace("-", "_"),
    )


def _evaluation_value(seed: int, arm: str, *, replacement: bool) -> dict[str, Any]:
    base = json.loads(renderer.BASE_CONFIG.read_text())
    tasks = json.loads(renderer.TASK_SET.read_text())["tasks"]
    tasks = sorted(tasks, key=lambda row: row["task_version_id"])
    route = copy.deepcopy(base["routes"]["base"])
    experiment, _output_root, _database = _arm_values(seed, arm, replacement=replacement)
    if arm == "base":
        model_id = "qwen3.8-27b-base"
        models = {model_id: base["models"][model_id]}
        routes = {"base": route}
    else:
        model_id = "teacher3k32-step1000"
        models = {
            model_id: {
                "repository": "/mnt/sfs/jobs/chris-q38-t3k32-b8-v3/hf-export-step1000-v1",
                "revision": (
                    "sha256:023c5f8b0559ba050f0d672a6bc27aabecec7d5837595f8ea5bc914446d26db5"
                ),
                "session_model": "qwen/chris-q38-t3k32-s1000-v1",
            }
        }
        route.update(model=model_id, served_id="chris-q38-t3k32-s1000-v1")
        route["model_info"]["model_path"] = "/scratch/models/chris-q38-t3k32-s1000-v1"
        route["server_info"]["model_path"] = "/scratch/models/chris-q38-t3k32-s1000-v1"
        routes = {"teacher3k32": route}
    value = {
        "schema": "cyber_fleet_eval_v1",
        "campaign_id": experiment,
        "run_prefix": experiment,
        "selection": {"source_job_id": final.SOURCE_TASK_JOB_ID},
        "tasks": tasks,
        "models": models,
        "routes": routes,
        "treatment": base["harness"],
        "images": base["images"],
        "pass_k": 1,
        "concurrency": 4,
        "training_data_eligible": False,
        "automatic_retry": False,
        "max_reviewed_infrastructure_retries": 0,
        "runtime_files": RUNTIME_FILES,
        "sampling": {"temperature": 0.6, "top_p": 0.95, "seed": seed},
        "interpretation": "serving-block descriptive evaluation",
    }
    return {**value, "sha256": final._plain_digest(value)}  # noqa: SLF001


RETAINED_IDENTITIES = {
    "base": {
        "protocol_id": "q38-dev17-s48-base-t3k32s1000-p1-v1",
        "comparison_arms": ["base", "candidate"],
        "arm_id": "base",
        "evaluation_config_name": "q38-dev17-s48-base-p1-v1",
        "evaluation_config_sha256": (
            "sha256:487c418ba8608435ac6b3d8ac79c10495bbe39e1a7f48a7c3f9d787a8f2be3db"
        ),
        "task_selection_sha256": (
            "sha256:79c834e739246da29aca9513965ecfc7032f8df7744eb2245ce0303aba1b97c5"
        ),
        "split_manifest_file_sha256": (
            "sha256:28a3dcaf31f14d724def9023d9435681772d5b8b3a8647cacc7f72ea8fc8adcb"
        ),
        "split_manifest_sha256": (
            "sha256:05b3a8dc90ca93adc9671942d75ecb54ff8d0951b0dd48a087e29ec1e641840c"
        ),
        "comparison_protocol_file_sha256": (
            "sha256:e6cc19ae4620d0c2071ddeb11d2285fd77195e5309e9c97cb884b6446b26b603"
        ),
        "comparison_protocol_sha256": (
            "sha256:6e8a15d372bda78978ea48c1889b7c5ef4a1f902e3dd85d534ede185cd5a646a"
        ),
        "checkpoint_provenance_sha256": (
            "sha256:03cdc5ffa45980d094674f6ea1aa6cbb77b91541681ffb286a74ca4c34d2511d"
        ),
        "serving_route_proof_sha256": (
            "sha256:084414df2084dc1da1ccb320f6bd64a6e804d8bb8adaedca9a1a57d9cff68297"
        ),
        "model_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "harness": "opencode",
        "harness_version": "1.18.27",
        "context_management": "opencode_1.18.27_native_compaction_autocontinue_v2",
        "sampling_seed": 48,
        "pass_k": 1,
        "retry_limit": 0,
        "output_root": "/mnt/sfs/jobs/chris-q38-fleet-dev17-s48-base-p1-v1",
        "database": "q38_dev17_s48_base_p1_v1",
    },
    "candidate": {
        "protocol_id": "q38-dev17-s48-base-t3k32s1000-p1-v1",
        "comparison_arms": ["base", "candidate"],
        "arm_id": "candidate",
        "evaluation_config_name": "q38-dev17-s48-t3k32s1000-p1-v2",
        "evaluation_config_sha256": (
            "sha256:4a0842d75cfc0fef658c1b69b81bcbf0c1bf65c348ee4d5d71cbc4f2c88783bb"
        ),
        "task_selection_sha256": (
            "sha256:79c834e739246da29aca9513965ecfc7032f8df7744eb2245ce0303aba1b97c5"
        ),
        "split_manifest_file_sha256": (
            "sha256:28a3dcaf31f14d724def9023d9435681772d5b8b3a8647cacc7f72ea8fc8adcb"
        ),
        "split_manifest_sha256": (
            "sha256:05b3a8dc90ca93adc9671942d75ecb54ff8d0951b0dd48a087e29ec1e641840c"
        ),
        "comparison_protocol_file_sha256": (
            "sha256:e6cc19ae4620d0c2071ddeb11d2285fd77195e5309e9c97cb884b6446b26b603"
        ),
        "comparison_protocol_sha256": (
            "sha256:6e8a15d372bda78978ea48c1889b7c5ef4a1f902e3dd85d534ede185cd5a646a"
        ),
        "checkpoint_provenance_sha256": (
            "sha256:b04620005175c01f5155689f2897efbfc39abc684561c2a3d0d294fc4f17b580"
        ),
        "serving_route_proof_sha256": (
            "sha256:1be6c1650c6d49ba5fd68e285206936fbd807003637cecbeb84ca8343eed9706"
        ),
        "model_revision": (
            "sha256:023c5f8b0559ba050f0d672a6bc27aabecec7d5837595f8ea5bc914446d26db5"
        ),
        "harness": "opencode",
        "harness_version": "1.18.27",
        "context_management": "opencode_1.18.27_native_compaction_autocontinue_v2",
        "sampling_seed": 48,
        "pass_k": 1,
        "retry_limit": 0,
        "output_root": "/mnt/sfs/jobs/chris-q38-fleet-dev17-s48-t3k32s1000-p1-v2",
        "database": "q38_dev17_s48_t3k32s1000_p1_v2",
    },
}


def _arm_evidence(
    seed: int, arm: str, protocol: dict[str, Any], *, replacement: bool
) -> dict[str, Any]:
    experiment, output_root, database = _arm_values(seed, arm, replacement=replacement)
    if not replacement:
        identity = copy.deepcopy(RETAINED_IDENTITIES[arm])
        packet_sha256 = {
            "base": "sha256:ef004b3b69017b4fd15f938a35761fd25a6a3ffebbbe767cf416c6125a7c5431",
            "candidate": (
                "sha256:b4f0163672d791b5d85a09bef1c0d8d435ef69e1a7c499a1c4edf4a30f7cb449"
            ),
        }[arm]
    else:
        identity = {
            "protocol_id": protocol["protocol_id"],
            "comparison_arms": list(final.ARMS),
            "arm_id": arm,
            "evaluation_config_name": experiment,
            "evaluation_config_sha256": "sha256:" + f"{seed + (0 if arm == 'base' else 1):064x}",
            "task_selection_sha256": (
                "sha256:79c834e739246da29aca9513965ecfc7032f8df7744eb2245ce0303aba1b97c5"
            ),
            "split_manifest_file_sha256": (
                "sha256:28a3dcaf31f14d724def9023d9435681772d5b8b3a8647cacc7f72ea8fc8adcb"
            ),
            "split_manifest_sha256": (
                "sha256:05b3a8dc90ca93adc9671942d75ecb54ff8d0951b0dd48a087e29ec1e641840c"
            ),
            "comparison_protocol_file_sha256": "sha256:" + f"{seed + 1000:064x}",
            "comparison_protocol_sha256": protocol["comparison_protocol_sha256"],
            "checkpoint_provenance_sha256": "sha256:" + f"{seed + 2000:064x}",
            "serving_route_proof_sha256": (
                "sha256:6c31daa69a834ea53a1b0fe2cb6938498d686499c522192b1dde0d7599684449"
            ),
            "model_revision": (
                "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
                if arm == "base"
                else "sha256:023c5f8b0559ba050f0d672a6bc27aabecec7d5837595f8ea5bc914446d26db5"
            ),
            "harness": "opencode",
            "harness_version": "1.18.27",
            "context_management": "opencode_1.18.27_native_compaction_autocontinue_v2",
            "sampling_seed": seed,
            "pass_k": 1,
            "retry_limit": 0,
            "output_root": output_root,
            "database": database,
        }
        packet_sha256 = "sha256:" + f"{seed + (4000 if arm == 'base' else 4001):064x}"
    evaluation = _evaluation_value(seed, arm, replacement=replacement)
    return {
        "packet_file_sha256": packet_sha256,
        "evaluation_identity": identity,
        "evaluation_identity_sha256": final._digest(identity),  # noqa: SLF001
        "evaluation_plan_sha256": "sha256:" + evaluation["sha256"],
        "runtime_files_sha256": RUNTIME_FILES,
    }


def _migration(
    tmp_path: Path,
    excluded: tuple[int, ...] = (46, 47, 49, 50, 51, 52, 53),
) -> Path:
    migration_root = tmp_path / "migration"
    migration_root.mkdir()
    mapping = [
        {"invalid_original_seed": seed, "replacement_seed": 54 + index}
        for index, seed in enumerate(excluded)
    ]
    included = sorted(
        [
            *(seed for seed in final.SOURCE_SEEDS if seed not in excluded),
            *(row["replacement_seed"] for row in mapping),
        ]
    )
    replacement_seeds = {row["replacement_seed"] for row in mapping}
    protocols = [_protocol(seed, replacement=seed in replacement_seeds) for seed in included]
    base = json.loads(renderer.BASE_CONFIG.read_text())
    comparison = _signed(
        {
            "schema": final.COMPARISON_DEFINITION_SCHEMA,
            "protocol_study_id": "q38-dev17-base-step1000-p8-v2",
            "predecessor_launch_receipt_sha256": (final.PREDECESSOR_COMPARISON_DEFINITION_SHA256),
            "aggregation": "eight_predeclared_pass1_replicas_per_task_and_arm",
            "original_seeds": list(final.SOURCE_SEEDS),
            "excluded_original_seeds": list(excluded),
            "replacement_mapping": mapping,
            "included_seeds": included,
            "replica_protocols": protocols,
            "task_count": final.TASK_COUNT,
            "sessions_per_arm": final.TASK_COUNT * final.PASS_K,
            "total_sessions": final.TASK_COUNT * final.PASS_K * len(final.ARMS),
            "comparison_arms": list(final.ARMS),
            "models": {
                "base": {
                    "model_id": "qwen3.8-27b",
                    "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
                },
                "candidate": {
                    "model_id": "chris-q38-t3k32-s1000-v1",
                    "revision": (
                        "sha256:023c5f8b0559ba050f0d672a6bc27aabecec7d5837595f8ea5bc914446d26db5"
                    ),
                },
            },
            "task_selection_sha256": (
                "sha256:38ea6686afa068c19e69fea2493e01027fdf99f7e01b20376e107b1a0cfa0b68"
            ),
            "split_manifest_sha256": (
                "sha256:05b3a8dc90ca93adc9671942d75ecb54ff8d0951b0dd48a087e29ec1e641840c"
            ),
            "binding_roster_sha256": (
                "sha256:39ae49c2db322725800b15347d5da5d171e7e03b21dec5b570683414d70e83c5"
            ),
            "harness": base["harness"],
            "images": base["images"],
            "sampling_without_seed": {"temperature": 0.6, "top_p": 0.95},
            "pass_k_per_replica": 1,
            "retry_limit": 0,
            "training_data_eligible": False,
            "whole_replica_pairs_only": True,
            "cell_level_replacement_forbidden": True,
            "included_seed_status": "provisional_until_all_valid8_terminal_gates_pass",
            "later_invalid_seed_policy": (
                "create a versioned successor intent, definition, and receipt before score "
                "unseal; never edit this definition in place"
            ),
        }
    )
    assert comparison["sha256"] == final.FROZEN_COMPARISON_DEFINITION_SHA256
    definition_path = migration_root / "COMPARISON_DEFINITION.json"
    definition_path.write_text(json.dumps(comparison, sort_keys=True) + "\n")
    protocol_by_seed = {row["seed"]: row for row in protocols}
    migrations = []
    replacement_protocols = []
    replacement_arms = []
    retained_arms = []
    for seed in included:
        if seed in replacement_seeds:
            continue
        for arm in final.ARMS:
            retained_arms.append(
                {
                    "seed": seed,
                    "arm_id": arm,
                    **_arm_evidence(seed, arm, protocol_by_seed[seed], replacement=False),
                }
            )
    for row in mapping:
        original = row["invalid_original_seed"]
        seed = row["replacement_seed"]
        excluded_arms = {}
        for arm_index, arm in enumerate(final.ARMS):
            excluded_arms[arm] = {
                "packet_file_sha256": "sha256:" + f"{original + arm_index + 2000:064x}",
                "evaluation_identity_sha256": ("sha256:" + f"{original + arm_index + 3000:064x}"),
                "evaluation_config_sha256": ("sha256:" + f"{original + arm_index + 4000:064x}"),
                "comparison_protocol_file_sha256": (
                    "sha256:" + f"{original + arm_index + 5000:064x}"
                ),
                "comparison_protocol_sha256": ("sha256:" + f"{original + arm_index + 6000:064x}"),
                "protocol_id": f"source-{original}",
                "job_name": f"source-{original}-{arm}",
                "config_map_name": f"source-{original}-{arm}-code",
                "output_root": f"/private/source/{original}/{arm}",
                "database": f"source_{original}_{arm}",
            }
        migrations.append(
            {
                **row,
                **copy.deepcopy(final.FROZEN_INVALID_REPLICA_EVIDENCE[original]),
                "excluded_source_arms": excluded_arms,
                "whole_pair_excluded": True,
            }
        )
        replacement_protocols.append(
            {
                **row,
                "protocol_id": protocol_by_seed[seed]["protocol_id"],
                "sha256": protocol_by_seed[seed]["comparison_protocol_sha256"],
                "file_sha256": "sha256:" + f"{seed + 1000:064x}",
            }
        )
        for arm in final.ARMS:
            arm_evidence = _arm_evidence(seed, arm, protocol_by_seed[seed], replacement=True)
            replacement_arms.append(
                {
                    "arm_id": arm,
                    **row,
                    **arm_evidence,
                    "serving_proof_file_sha256": arm_evidence["evaluation_identity"][
                        "serving_route_proof_sha256"
                    ],
                    "comparison_protocol_file_sha256": arm_evidence["evaluation_identity"][
                        "comparison_protocol_file_sha256"
                    ],
                    "comparison_protocol_sha256": protocol_by_seed[seed][
                        "comparison_protocol_sha256"
                    ],
                    "packet_path": f"seed{seed}/{arm}/LAUNCH_PACKET.json",
                }
            )
    receipt = _signed(
        {
            "schema": final.MIGRATION_SCHEMA,
            "source": {
                "protocol_study_id": "q38-dev17-seeds46to53-base-step1000-p8-v1",
                "preparation_receipt_sha256": (
                    "sha256:22b59a3e9e1ea2ed8e8b92920304f27bbd953925fe0ecf709c7308433db8f0d3"
                ),
                "preparation_receipt_file_sha256": (
                    "sha256:2827250c73d024cab06906d393eba9270784f99b70bb3812d9893c53b1f84859"
                ),
                "seeds": list(final.SOURCE_SEEDS),
                "arm_count": len(final.SOURCE_SEEDS) * len(final.ARMS),
                "retained_candidate_successor_preparation": {
                    "path": "/private/retained/PREPARATION_RECEIPT.json",
                    "file_sha256": (
                        "sha256:5605623bf5f6205a70a6bad0f02985f985858904badfa7be413a1cead1cbe67a"
                    ),
                    "receipt_sha256": (
                        "sha256:d3624db05e0665cf0483d802a49c54be601293e5267b5475143a282791269189"
                    ),
                    "successor_generation": 2,
                },
            },
            "migration_intent_sha256": (
                "sha256:0e32d785e8b3570b2e972389b93d70334e9889b8a3d2b9230f0d13ce6137c79c"
            ),
            "sanitized_invalid_replica_evidence": [
                {"seed": seed, **copy.deepcopy(final.FROZEN_INVALID_REPLICA_EVIDENCE[seed])}
                for seed in final.FROZEN_INVALID_SEEDS
            ],
            "checked_in_seed51_invalid_evidence": {
                "path": (
                    "docs/evidence/qwen38-fleet-dev17-seed51-base-invalid-replica-20260923.json"
                ),
                "file_sha256": (
                    "sha256:525885e6650d6d742144b12de2ac1807a77cd25f23e07cec58d2832551150497"
                ),
                "receipt_sha256": (
                    "sha256:572e330d1340e83d2aaf188665c0fd99b401e33eeec5f858717e2db60952bfa3"
                ),
            },
            "comparison_definition": comparison,
            "comparison_definition_file_sha256": final._file_digest(  # noqa: SLF001
                definition_path
            ),
            "mapping_rule": (
                "invalid original seeds sorted ascending map to the smallest unused seeds "
                "greater than 53, ascending"
            ),
            "migrations": migrations,
            "excluded_original_seeds": list(excluded),
            "included_seeds": included,
            "replacement_protocols": replacement_protocols,
            "replacement_arms": replacement_arms,
            "retained_arms": retained_arms,
            "checked_in_partial_recovery_hold_evidence": {
                "path": (
                    "docs/evidence/qwen38-fleet-dev17-base-prefix-recovery-hold-20260923.json"
                ),
                "file_sha256": (
                    "sha256:d20b4fe2310de95cfd5561877f6d93133dcf7e5ed903f425addfb28bda2583ae"
                ),
                "receipt_sha256": (
                    "sha256:e7b0031010f097f66dd3c87de94ecbc0d1505de97167c47d0a2d7615a0164e52"
                ),
                "source_private_receipt_sha256": (
                    "sha256:45893e380d1d9755cbe514ec85628b7fec930d9e2bf5e61920967155112bda09"
                ),
            },
            "retirement_evidence": {
                "sha256": (
                    "sha256:81ff2a02299d3ec5aeed76e9b4284cceacd286029e08db2e5654d8eed53c04f2"
                ),
                "file_sha256": (
                    "sha256:ca06ff95e38c6b6da95ab3052596fa84cc3216757188d395dfb0aa49be0656cb"
                ),
                "targets": 2,
                "model_rollouts": 0,
                "outputs_or_databases_deleted": False,
            },
            "scientific_identity": {
                "task_count_per_arm": final.TASK_COUNT,
                "comparison_arms": list(final.ARMS),
                "pass_k": 1,
                "retry_limit": 0,
                "same_task_versions_models_harness_budgets_and_sampling_recipe": True,
                "whole_replica_pairs_only": True,
                "cell_level_replacement_forbidden": True,
                "seed_reuse_forbidden": True,
                "later_invalid_seed_requires_versioned_successor_before_score_unseal": True,
            },
            "capacity": {
                "new_replacement_rollouts": len(mapping) * len(final.ARMS) * final.TASK_COUNT,
                "final_comparison_rollouts": final.TASK_COUNT * final.PASS_K * len(final.ARMS),
                "actual_started_rollouts_today": 112,
                "projected_rollouts_after_reservation": 350,
                "remaining_after_reservation": 150,
                "daily_rollout_cap": 500,
                "within_daily_cap": True,
            },
            "global_daily_budget_evidence": {
                "path": "docs/evidence/qwen38-fleet-global-daily-rollout-budget-20260923.json",
                "sha256": (
                    "sha256:85c5a4f38943abab21ae60db1791fbc6ab005560b665e6680dfaa7f5889819f2"
                ),
                "file_sha256": (
                    "sha256:064bad27391869368ce2c14853663e7a181020b0ef6f73a4325c5697c1b46e15"
                ),
                "window_utc": {
                    "start_inclusive": "2026-09-23T00:00:00Z",
                    "end_exclusive": "2026-09-24T00:00:00Z",
                },
            },
            "selection": {
                "selection_sha256": (
                    "sha256:38ea6686afa068c19e69fea2493e01027fdf99f7e01b20376e107b1a0cfa0b68"
                ),
                "split_sha256": (
                    "sha256:05b3a8dc90ca93adc9671942d75ecb54ff8d0951b0dd48a087e29ec1e641840c"
                ),
                "corpus_dev_windows": 0,
                "binding_roster_sha256": (
                    "sha256:39ae49c2db322725800b15347d5da5d171e7e03b21dec5b570683414d70e83c5"
                ),
            },
            "live_parity_file_sha256": (
                "sha256:6c31daa69a834ea53a1b0fe2cb6938498d686499c522192b1dde0d7599684449"
            ),
            "live_parity_receipt_sha256": (
                "sha256:20357189cde921fa28a7dce9bb8b99654914557a6b490f43005f6f1bde416f05"
            ),
            "privacy": {
                "score_values_read": False,
                "prompts_responses_flags_rewards_or_trace_content_read": False,
                "infrastructure_reason_classes_only": True,
            },
            "external_mutations": 0,
            "launch_performed": False,
        }
    )
    receipt_path = migration_root / "MIGRATION_RECEIPT.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
    return receipt_path


def _plan(tmp_path: Path) -> dict[str, Any]:
    migration = _migration(tmp_path)
    plan = final.build_current_study_plan(
        task_set_path=renderer.TASK_SET,
        roster_path=renderer.ROSTER,
        base_config_path=renderer.BASE_CONFIG,
        migration_receipt_path=migration,
    )
    for replica in plan["replicas"]:
        root = tmp_path / f"s{replica['seed']}-{replica['arm']}"
        root.mkdir()
        _TEST_BOUND_PATHS[replica["output_root"]] = root
        _TEST_BOUND_PATHS[replica["terminal_receipt_path"]] = root / "TERMINAL_OBSERVATION.json"
    _TEST_BOUND_PATHS[plan["private_output_root"]] = tmp_path / "final"
    return plan


def _test_path(value: str) -> Path:
    return final._bound_path(value)  # noqa: SLF001


def _evaluation(plan: dict[str, Any], replica: dict[str, Any]) -> dict[str, Any]:
    arm = plan["arms"][replica["arm"]]
    value = {
        "schema": "cyber_fleet_eval_v1",
        "campaign_id": replica["experiment_id"],
        "run_prefix": replica["experiment_id"],
        "selection": {"source_job_id": final.SOURCE_TASK_JOB_ID},
        "tasks": sorted(plan["tasks"], key=lambda row: row["task_version_id"]),
        "models": {arm["model_id"]: arm["model"]},
        "routes": {arm["serving_block"]: arm["route"]},
        "treatment": plan["harness"],
        "images": plan["images"],
        "sampling": {**plan["sampling"], "seed": replica["seed"]},
        "pass_k": 1,
        "concurrency": 4,
        "automatic_retry": False,
        "max_reviewed_infrastructure_retries": 0,
        "training_data_eligible": False,
        "runtime_files": replica["runtime_files_sha256"],
        "interpretation": "serving-block descriptive evaluation",
    }
    value["sha256"] = final._plain_digest(value)  # noqa: SLF001
    return value


def _local_record(cell_id: str, session_id: str, score: float, index: int) -> dict[str, Any]:
    digest = f"{index + 1:064x}"[-64:]
    value = {
        "execution_id": "sha256:" + f"{index + 300:064x}"[-64:],
        "cell_id": cell_id,
        "execution_generation": 1,
        "run_id": f"run-{index}",
        "session_id": session_id,
        "verifier_execution_id": f"verifier-{index}",
        "score": score,
        "config_sha256": digest,
        "artifact_directory": f"attempts/{index}",
        "trace_path": "trace.json",
        "trace_sha256": digest,
        "result_path": "result.json",
        "result_sha256": digest,
        "reward_path": "reward-result.json",
        "reward_sha256": digest,
        "session_ingest_path": "session-ingest.json",
        "session_ingest_sha256": digest,
        "cleanup_path": "cleanup.json",
        "cleanup_sha256": digest,
        "session_ingest_status": "completed",
        "agent_exit_code": 0,
        "agent_termination": "completed",
        "elapsed_seconds": 10.0,
    }
    return {**value, "record_sha256": final._plain_digest(value)}  # noqa: SLF001


def _study(tmp_path: Path) -> tuple[dict[str, Any], dict[tuple[int, str], FakeSnapshot]]:
    plan = _plan(tmp_path)
    snapshots: dict[tuple[int, str], FakeSnapshot] = {}
    for replica in plan["replicas"]:
        evaluation = _evaluation(plan, replica)
        root = _test_path(replica["output_root"])
        (root / "EVAL.json").write_text(json.dumps(evaluation, sort_keys=True) + "\n")
        arm = plan["arms"][replica["arm"]]
        stored_plan = []
        for task in sorted(plan["tasks"], key=lambda row: row["task_version_id"]):
            row = {
                "experiment_id": replica["experiment_id"],
                "task_key": task["task_key"],
                "task_version_id": task["task_version_id"],
                "model_id": arm["model_id"],
                "model_revision": arm["model"]["revision"],
                "serving_block": arm["serving_block"],
                "endpoint_model_id": arm["route"]["served_id"],
                "harness_id": "protocol-" + evaluation["sha256"],
                "attempt": 1,
                "max_retries": 0,
            }
            stored_plan.append({"cell_id": final._cell_id(row), **row})  # noqa: SLF001
        plan_digest = final._plain_digest(stored_plan)  # noqa: SLF001
        terminal = _signed(
            {
                "schema": final.TERMINAL_SCHEMA,
                "observed_at": "2026-09-23T00:00:00+00:00",
                "evaluation_identity_sha256": replica["evaluation_identity_sha256"],
                "comparison_protocol_sha256": replica["comparison_protocol_sha256"],
                "protocol_id": replica["protocol_id"],
                "arm_id": replica["arm"],
                "job": {
                    "name": replica["job_name"],
                    "uid": "11111111-1111-4111-8111-111111111111",
                    "terminal_condition": "Complete",
                    "succeeded": 1,
                    "failed": 0,
                },
                "config_map": {
                    "name": replica["config_map_name"],
                    "uid": "22222222-2222-4222-8222-222222222222",
                },
                "workloads": [],
                "pods": [],
                "database": {
                    "name": replica["database"],
                    "summary": {
                        "total": final.TASK_COUNT,
                        "local_results": final.TASK_COUNT,
                        "by_state": {"accepted": final.TASK_COUNT},
                        "by_serving_block": [
                            {
                                "serving_block": arm["serving_block"],
                                "count": final.TASK_COUNT,
                            }
                        ],
                        "stale_active": 0,
                        "plan_sha256": plan_digest,
                    },
                },
                "output_root": {"path": replica["output_root"], "exists": True},
                "decision": {
                    "capability_result_status": "not_interpreted",
                    "score_blind_reconciliation_required": False,
                    "unresolved_cells": 0,
                    "rollout_retry_performed": False,
                    "score_read_or_generated": False,
                },
                "privacy": {
                    "prompts_responses_flags_rewards_or_trace_content_included": False,
                    "score_values_included": False,
                    "credentials_included": False,
                },
            }
        )
        _test_path(replica["terminal_receipt_path"]).write_text(
            json.dumps(terminal, sort_keys=True) + "\n"
        )
        cells = []
        local = []
        events = []
        planned_by_task = {row["task_version_id"]: row for row in stored_plan}
        for index, task in enumerate(plan["tasks"]):
            planned = planned_by_task[task["task_version_id"]]
            cell_id = planned["cell_id"]
            session_id = f"private-session-{replica['seed']}-{replica['arm']}-{index}"
            record = _local_record(
                cell_id,
                session_id,
                float(index == 0 and replica["arm"] == "candidate"),
                index,
            )
            accepted = _signed(
                {
                    "schema_version": final.ACCEPTED_SCHEMA,
                    "accepted": True,
                    "campaign_id": replica["experiment_id"],
                    "cell_id": cell_id,
                    "execution_id": record["execution_id"],
                    "ledger_cell_id": cell_id,
                    "run_id": record["run_id"],
                    "serving_block": planned["serving_block"],
                    "session_id": session_id,
                    "verifier_execution_id": record["verifier_execution_id"],
                    "config_sha256": record["config_sha256"],
                    "session_ingest_completed": True,
                    "cleanup_completed": True,
                    "score_persisted_privately": True,
                    "scores_included": False,
                    "prompts_or_traces_included": False,
                },
                "receipt_sha256",
            )
            receipt = accepted["receipt_sha256"]
            accepted_path = root / record["artifact_directory"] / "ACCEPTED.json"
            accepted_path.parent.mkdir(parents=True, exist_ok=True)
            accepted_path.write_text(json.dumps(accepted, sort_keys=True) + "\n")
            cells.append(
                {
                    "cell_id": cell_id,
                    **planned,
                    "state": "accepted",
                    "session_id": session_id,
                    "lease_expires_at": None,
                    "retry_count": 0,
                    "max_retries": 0,
                    "result_class": "valid",
                    "receipt_digest": receipt,
                    "failure_code": None,
                    "reconciliation_digest": None,
                }
            )
            local.append(record)
            events.append(
                {
                    "cell_id": cell_id,
                    "event": "accepted",
                    "to_state": "accepted",
                    "detail_json": json.dumps({"receipt_digest": receipt}),
                }
            )
        gate = final.GateSnapshot(
            plan_sha256=plan_digest,
            cells=cells,
            local_metadata=[
                {key: row[key] for key in final.SCORE_BLIND_LOCAL_FIELDS} for row in local
            ],
            events=events,
            reconciliations=[],
        )
        snapshots[(replica["seed"], replica["arm"])] = FakeSnapshot(gate, local)
    return plan, snapshots


def test_final_gate_opens_scores_only_after_all_replicas_and_emits_safe_public_input(
    tmp_path: Path,
) -> None:
    plan, snapshots = _study(tmp_path)

    result = final.finalize(plan, snapshots, output_root=_test_path(plan["private_output_root"]))

    assert result["status"] == "final"
    assert result["valid_outcomes_per_task_arm"] == 8
    assert all(snapshot.score_reads == 1 for snapshot in snapshots.values())
    public_path = _test_path(plan["private_output_root"]) / "SANITIZED_AGGREGATE.json"
    public = json.loads(public_path.read_text())
    validated = public_eval_import._validate(public, final._file_digest(public_path))  # noqa: SLF001
    assert validated["summary"]["paired_valid_tasks"] == 17
    assert all(
        row[arm]["valid_attempts"] == 8 and row[arm]["infrastructure_invalid_attempts"] == 0
        for row in public["task_rows"]
        for arm in final.ARMS
    )
    public_text = public_path.read_text().lower()
    for forbidden in (
        "task_key",
        "task_version_id",
        "session_id",
        "cell_id",
        "prompt",
        "trace",
        "verifier",
        "private-session",
    ):
        assert forbidden not in public_text
    assert oct(public_path.stat().st_mode & 0o777) == "0o600"


@pytest.mark.parametrize(
    ("state", "result_class"),
    (
        ("running", None),
        ("retry_review", "infrastructure_invalid"),
        ("terminal", "infrastructure_invalid"),
    ),
)
def test_one_unready_cell_prevents_every_score_read(
    tmp_path: Path, state: str, result_class: str | None
) -> None:
    plan, snapshots = _study(tmp_path)
    first = snapshots[(48, "base")]
    first.value.cells[0]["state"] = state
    first.value.cells[0]["result_class"] = result_class

    with pytest.raises(final.FinalAggregateError, match="non-authoritative"):
        final.finalize(plan, snapshots, output_root=_test_path(plan["private_output_root"]))

    assert all(snapshot.score_reads == 0 for snapshot in snapshots.values())
    assert not _test_path(plan["private_output_root"]).exists()


def test_database_blind_query_has_no_score_or_score_commitment() -> None:
    class Result:
        def __init__(self, rows: list[dict[str, Any]]) -> None:
            self.rows = rows

        def fetchall(self) -> list[dict[str, Any]]:
            return self.rows

    class Connection:
        def __init__(self) -> None:
            self.queries: list[str] = []

        def execute(self, query: str) -> Result:
            self.queries.append(query)
            return Result([{"value": "plan"}] if "ledger_metadata" in query else [])

    connection = Connection()
    final._PostgresSnapshot(connection).gate()  # noqa: SLF001
    query = next(
        value
        for value in connection.queries
        if "FROM rollout_local_results ORDER BY cell_id" in value
    )
    selected = query.partition("SELECT ")[2].partition(" FROM")[0].split(", ")
    assert tuple(selected) == final.SCORE_BLIND_LOCAL_FIELDS
    assert not {
        "score",
        "record_sha256",
        "trace_sha256",
        "result_sha256",
        "reward_sha256",
        "session_ingest_sha256",
        "cleanup_sha256",
        "trace_path",
        "result_path",
        "reward_path",
        "session_ingest_path",
        "cleanup_path",
        "elapsed_seconds",
    } & set(selected)


def test_database_dsn_preserves_root_authority_and_selects_only_database() -> None:
    assert final._database_dsn(  # noqa: SLF001
        "postgresql://reader@example.internal:5432/postgres?sslmode=require",
        "q38_dev17_s48_base_p1_v1",
    ) == ("postgresql://reader@example.internal:5432/q38_dev17_s48_base_p1_v1?sslmode=require")


@pytest.mark.parametrize("field", ("database", "dbname", "host", "hostaddr", "port", "service"))
def test_database_dsn_rejects_query_authority_overrides(field: str) -> None:
    with pytest.raises(final.FinalAggregateError, match="root connection"):
        final._database_dsn(  # noqa: SLF001
            f"postgresql://reader@example.internal/postgres?{field}=override",
            "q38_dev17_s48_base_p1_v1",
        )


@pytest.mark.parametrize("drift", ("missing", "extra_private_field", "identity"))
def test_worker_accepted_receipt_is_exact_before_score_open(tmp_path: Path, drift: str) -> None:
    plan, snapshots = _study(tmp_path)
    replica = next(row for row in plan["replicas"] if row["seed"] == 48 and row["arm"] == "base")
    metadata = snapshots[(48, "base")].value.local_metadata[0]
    path = _test_path(replica["output_root"]) / metadata["artifact_directory"] / "ACCEPTED.json"
    if drift == "missing":
        path.unlink()
    else:
        receipt = json.loads(path.read_text())
        receipt.pop("receipt_sha256")
        if drift == "extra_private_field":
            receipt["prompt"] = "must never be parsed"
        else:
            receipt["session_id"] = "different-session"
        receipt["receipt_sha256"] = final._digest(receipt)  # noqa: SLF001
        path.write_text(json.dumps(receipt, sort_keys=True) + "\n")

    with pytest.raises(final.FinalAggregateError, match="accepted cell receipt"):
        final.finalize(plan, snapshots, output_root=_test_path(plan["private_output_root"]))
    assert all(value.score_reads == 0 for value in snapshots.values())


def test_worker_accepted_receipt_parent_symlink_is_rejected(tmp_path: Path) -> None:
    plan, snapshots = _study(tmp_path)
    replica = next(row for row in plan["replicas"] if row["seed"] == 48 and row["arm"] == "base")
    metadata = snapshots[(48, "base")].value.local_metadata[0]
    artifact = _test_path(replica["output_root"]) / metadata["artifact_directory"]
    relocated = artifact.with_name("relocated")
    artifact.rename(relocated)
    artifact.symlink_to(relocated, target_is_directory=True)

    with pytest.raises(final.FinalAggregateError, match="parent is not an exact directory"):
        final.finalize(plan, snapshots, output_root=_test_path(plan["private_output_root"]))
    assert all(value.score_reads == 0 for value in snapshots.values())


@pytest.mark.parametrize("drift", ("extra_terminal_field", "retry", "event_detail"))
def test_terminal_and_event_privacy_shapes_are_exact_before_score_open(
    tmp_path: Path, drift: str
) -> None:
    plan, snapshots = _study(tmp_path)
    replica = next(row for row in plan["replicas"] if row["seed"] == 48 and row["arm"] == "base")
    if drift == "event_detail":
        event = snapshots[(48, "base")].value.events[0]
        detail = json.loads(event["detail_json"])
        detail["trace"] = "must never be parsed"
        event["detail_json"] = json.dumps(detail)
    else:
        path = _test_path(replica["terminal_receipt_path"])
        receipt = json.loads(path.read_text())
        receipt.pop("sha256")
        if drift == "extra_terminal_field":
            receipt["prompt"] = "must never be parsed"
        else:
            receipt["decision"]["rollout_retry_performed"] = True
        receipt["sha256"] = final._digest(receipt)  # noqa: SLF001
        path.write_text(json.dumps(receipt, sort_keys=True) + "\n")

    with pytest.raises(final.FinalAggregateError):
        final.finalize(plan, snapshots, output_root=_test_path(plan["private_output_root"]))
    assert all(value.score_reads == 0 for value in snapshots.values())


def test_orphan_accepted_event_prevents_every_score_read(tmp_path: Path) -> None:
    plan, snapshots = _study(tmp_path)
    event = copy.deepcopy(snapshots[(48, "base")].value.events[0])
    event["cell_id"] = "00000000-0000-4000-8000-000000000000"
    snapshots[(48, "base")].value.events.append(event)

    with pytest.raises(final.FinalAggregateError, match="accepted event roster differs"):
        final.finalize(plan, snapshots, output_root=_test_path(plan["private_output_root"]))

    assert all(value.score_reads == 0 for value in snapshots.values())


def test_unused_reconciliation_receipt_prevents_every_score_read(tmp_path: Path) -> None:
    plan, snapshots = _study(tmp_path)
    evidence = _signed(
        {
            "schema_version": "fleet-stored-session-reconciliation-v2",
            "reviewed_intent_sha256": "a" * 64,
            "evaluation_plan_sha256": "sha256:" + "b" * 64,
            "source_job_uid_sha256": "sha256:" + "c" * 64,
            "source_job_terminal_receipt_sha256": "sha256:" + "d" * 64,
            "selected_cell_count": 1,
            "prior_retry_review_count": 1,
            "accepted_existing_completed_session_count": 1,
            "source_agent_exit_code": 1,
            "source_agent_termination": "process_error",
            "source_failure_code_sha256": "sha256:" + "e" * 64,
            "action": "accept_existing_scored_session",
            "model_generation_performed": False,
            "scoring_call_performed": False,
            "score_values_included": False,
            "prompt_response_flag_reward_or_trace_content_included": False,
            "cell_task_session_or_trace_identifiers_included": False,
        },
        "receipt_sha256",
    )
    snapshots[(48, "base")].value.reconciliations.append(evidence)

    with pytest.raises(final.FinalAggregateError, match="reconciliation receipt roster differs"):
        final.finalize(plan, snapshots, output_root=_test_path(plan["private_output_root"]))

    assert all(value.score_reads == 0 for value in snapshots.values())


@pytest.mark.parametrize("reconciled", (False, True))
def test_local_lifecycle_must_match_acceptance_path_before_score_open(
    tmp_path: Path, reconciled: bool
) -> None:
    plan, snapshots = _study(tmp_path)
    if reconciled:
        _add_reconciliation(plan, snapshots)
        snapshots[(48, "base")].value.local_metadata[0]["agent_exit_code"] = 0
    else:
        snapshots[(48, "base")].value.local_metadata[0]["agent_termination"] = "process_error"

    with pytest.raises(final.FinalAggregateError, match="lifecycle differs"):
        final.finalize(plan, snapshots, output_root=_test_path(plan["private_output_root"]))
    assert all(value.score_reads == 0 for value in snapshots.values())


def test_protocol_v2_uses_only_eight_complete_whole_replica_pairs(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    included = set(plan["included_seeds"])
    excluded = set(plan["excluded_original_seeds"])
    replacements = {row["replacement_seed"] for row in plan["replacement_mapping"]}

    assert len(included) == final.PASS_K
    assert included.isdisjoint(excluded)
    assert replacements <= included
    assert {(row["seed"], row["arm"]) for row in plan["replicas"]} == {
        (seed, arm) for seed in included for arm in final.ARMS
    }
    assert all(
        row["origin"] == "whole_pair_replacement"
        for row in plan["replicas"]
        if row["seed"] in replacements
    )


def test_plan_cannot_mix_an_excluded_original_with_its_replacement(tmp_path: Path) -> None:
    plan, snapshots = _study(tmp_path)
    replacement = plan["replacement_mapping"][0]
    row = next(
        item
        for item in plan["replicas"]
        if item["seed"] == replacement["replacement_seed"] and item["arm"] == "candidate"
    )
    row["seed"] = replacement["invalid_original_seed"]
    plan.pop("sha256")
    plan.update(sha256=final._digest(plan))  # noqa: SLF001

    with pytest.raises(final.FinalAggregateError, match="eight included pairs"):
        final.finalize(plan, snapshots, output_root=_test_path(plan["private_output_root"]))

    assert all(snapshot.score_reads == 0 for snapshot in snapshots.values())


def test_migration_receipt_tamper_is_rejected_before_plan_build(tmp_path: Path) -> None:
    receipt_path = _migration(tmp_path)
    receipt = json.loads(receipt_path.read_text())
    receipt["included_seeds"][0] = 47
    receipt_path.write_text(json.dumps(receipt) + "\n")

    with pytest.raises(final.FinalAggregateError, match="self digest"):
        final.build_current_study_plan(
            task_set_path=renderer.TASK_SET,
            roster_path=renderer.ROSTER,
            base_config_path=renderer.BASE_CONFIG,
            migration_receipt_path=receipt_path,
        )


def test_resigned_migration_requires_full_retained_arm_evidence(tmp_path: Path) -> None:
    receipt_path = _migration(tmp_path)
    receipt = json.loads(receipt_path.read_text())
    receipt.pop("sha256")
    receipt.pop("retained_arms")
    receipt["sha256"] = final._digest(receipt)  # noqa: SLF001
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")

    with pytest.raises(final.FinalAggregateError, match="differs from the freeze"):
        final.build_current_study_plan(
            task_set_path=renderer.TASK_SET,
            roster_path=renderer.ROSTER,
            base_config_path=renderer.BASE_CONFIG,
            migration_receipt_path=receipt_path,
        )


def test_resigned_migration_cannot_add_private_content_field(tmp_path: Path) -> None:
    receipt_path = _migration(tmp_path)
    receipt = json.loads(receipt_path.read_text())
    receipt.pop("sha256")
    receipt["migrations"][0]["prompt"] = "private content must never enter the plan"
    receipt["sha256"] = final._digest(receipt)  # noqa: SLF001
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")

    with pytest.raises(final.FinalAggregateError, match="differs from the freeze"):
        final.build_current_study_plan(
            task_set_path=renderer.TASK_SET,
            roster_path=renderer.ROSTER,
            base_config_path=renderer.BASE_CONFIG,
            migration_receipt_path=receipt_path,
        )


def test_resigned_replacement_identity_drift_is_rejected(tmp_path: Path) -> None:
    receipt_path = _migration(tmp_path)
    receipt = json.loads(receipt_path.read_text())
    receipt.pop("sha256")
    arm = receipt["replacement_arms"][0]
    arm["evaluation_identity"]["harness_version"] = "unreviewed"
    arm["evaluation_identity_sha256"] = final._digest(arm["evaluation_identity"])  # noqa: SLF001
    receipt["sha256"] = final._digest(receipt)  # noqa: SLF001
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")

    with pytest.raises(final.FinalAggregateError, match="differs from the freeze"):
        final.build_current_study_plan(
            task_set_path=renderer.TASK_SET,
            roster_path=renderer.ROSTER,
            base_config_path=renderer.BASE_CONFIG,
            migration_receipt_path=receipt_path,
        )


def test_resigned_replacement_protocol_file_drift_is_rejected(tmp_path: Path) -> None:
    receipt_path = _migration(tmp_path)
    receipt = json.loads(receipt_path.read_text())
    receipt.pop("sha256")
    receipt["replacement_protocols"][0]["file_sha256"] = "sha256:" + "f" * 64
    receipt["sha256"] = final._digest(receipt)  # noqa: SLF001
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")

    with pytest.raises(final.FinalAggregateError, match="differs from the freeze"):
        final.build_current_study_plan(
            task_set_path=renderer.TASK_SET,
            roster_path=renderer.ROSTER,
            base_config_path=renderer.BASE_CONFIG,
            migration_receipt_path=receipt_path,
        )


def test_resigned_replacement_route_proof_drift_is_rejected(tmp_path: Path) -> None:
    receipt_path = _migration(tmp_path)
    receipt = json.loads(receipt_path.read_text())
    receipt.pop("sha256")
    arm = receipt["replacement_arms"][0]
    arm["evaluation_identity"]["serving_route_proof_sha256"] = "sha256:" + "f" * 64
    arm["evaluation_identity_sha256"] = final._digest(arm["evaluation_identity"])  # noqa: SLF001
    arm["serving_proof_file_sha256"] = "sha256:" + "f" * 64
    receipt["sha256"] = final._digest(receipt)  # noqa: SLF001
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")

    with pytest.raises(final.FinalAggregateError, match="differs from the freeze"):
        final.build_current_study_plan(
            task_set_path=renderer.TASK_SET,
            roster_path=renderer.ROSTER,
            base_config_path=renderer.BASE_CONFIG,
            migration_receipt_path=receipt_path,
        )


@pytest.mark.parametrize("field", ("packet_file_sha256", "evaluation_plan_sha256"))
def test_resigned_retained_packet_evidence_drift_is_rejected(tmp_path: Path, field: str) -> None:
    receipt_path = _migration(tmp_path)
    receipt = json.loads(receipt_path.read_text())
    receipt.pop("sha256")
    receipt["retained_arms"][0][field] = "sha256:" + "f" * 64
    receipt["sha256"] = final._digest(receipt)  # noqa: SLF001
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")

    with pytest.raises(final.FinalAggregateError, match="differs from the freeze"):
        final.build_current_study_plan(
            task_set_path=renderer.TASK_SET,
            roster_path=renderer.ROSTER,
            base_config_path=renderer.BASE_CONFIG,
            migration_receipt_path=receipt_path,
        )


def test_resigned_alternate_exclusion_roster_is_rejected(tmp_path: Path) -> None:
    receipt_path = _migration(tmp_path)
    receipt = json.loads(receipt_path.read_text())
    comparison = receipt["comparison_definition"]
    comparison.pop("sha256")
    comparison["excluded_original_seeds"] = [47, 52, 53]
    comparison["sha256"] = final._digest(comparison)  # noqa: SLF001
    definition_path = receipt_path.parent / "COMPARISON_DEFINITION.json"
    definition_path.write_text(json.dumps(comparison, sort_keys=True) + "\n")
    receipt.pop("sha256")
    receipt["comparison_definition"] = comparison
    receipt["comparison_definition_file_sha256"] = final._file_digest(  # noqa: SLF001
        definition_path
    )
    receipt["sha256"] = final._digest(receipt)  # noqa: SLF001
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")

    with pytest.raises(final.FinalAggregateError, match="differs from the freeze"):
        final.build_current_study_plan(
            task_set_path=renderer.TASK_SET,
            roster_path=renderer.ROSTER,
            base_config_path=renderer.BASE_CONFIG,
            migration_receipt_path=receipt_path,
        )


def _rewrite_terminal(replica: dict[str, Any], mutate: Any) -> None:
    path = _test_path(replica["terminal_receipt_path"])
    receipt = json.loads(path.read_text())
    receipt.pop("sha256")
    mutate(receipt)
    path.write_text(json.dumps(_signed(receipt), sort_keys=True) + "\n")


def _rewrite_evaluation(replica: dict[str, Any], mutate: Any) -> None:
    path = _test_path(replica["output_root"]) / "EVAL.json"
    evaluation = json.loads(path.read_text())
    evaluation.pop("sha256")
    mutate(evaluation)
    evaluation["sha256"] = final._plain_digest(evaluation)  # noqa: SLF001
    path.write_text(json.dumps(evaluation, sort_keys=True) + "\n")


def test_wrong_protocol_prevents_every_score_read(tmp_path: Path) -> None:
    plan, snapshots = _study(tmp_path)
    replica = next(row for row in plan["replicas"] if row["seed"] == 48 and row["arm"] == "base")
    _rewrite_terminal(replica, lambda receipt: receipt.update(protocol_id="wrong"))

    with pytest.raises(final.FinalAggregateError, match="terminal receipt identity"):
        final.finalize(plan, snapshots, output_root=_test_path(plan["private_output_root"]))

    assert all(snapshot.score_reads == 0 for snapshot in snapshots.values())


@pytest.mark.parametrize("drift", ("ignored_field", "runtime"))
def test_evaluation_schema_or_runtime_drift_prevents_every_score_read(
    tmp_path: Path, drift: str
) -> None:
    plan, snapshots = _study(tmp_path)
    replica = next(row for row in plan["replicas"] if row["seed"] == 48 and row["arm"] == "base")

    def mutate(evaluation: dict[str, Any]) -> None:
        if drift == "ignored_field":
            evaluation["unreviewed_behavior"] = True
        else:
            evaluation["runtime_files"]["evaluate.py"] = "f" * 64

    _rewrite_evaluation(replica, mutate)

    with pytest.raises(final.FinalAggregateError):
        final.finalize(plan, snapshots, output_root=_test_path(plan["private_output_root"]))

    assert all(snapshot.score_reads == 0 for snapshot in snapshots.values())


@pytest.mark.parametrize("field", ("evaluation_plan_sha256", "runtime_files_sha256"))
def test_sealed_plan_evidence_drift_prevents_every_score_read(tmp_path: Path, field: str) -> None:
    plan, snapshots = _study(tmp_path)
    replica = next(row for row in plan["replicas"] if row["seed"] == 48 and row["arm"] == "base")
    if field == "evaluation_plan_sha256":
        replica[field] = "sha256:" + "f" * 64
    else:
        replica[field]["evaluate.py"] = "f" * 64
    plan.pop("sha256")
    plan["sha256"] = final._digest(plan)  # noqa: SLF001

    with pytest.raises(final.FinalAggregateError, match="evaluation plan or runtime differs"):
        final.finalize(plan, snapshots, output_root=_test_path(plan["private_output_root"]))

    assert all(snapshot.score_reads == 0 for snapshot in snapshots.values())


def test_live_ledger_plan_must_recompute_from_exact_cells(tmp_path: Path) -> None:
    plan, snapshots = _study(tmp_path)
    identity = (48, "base")
    snapshot = snapshots[identity]
    wrong = "f" * 64
    snapshot.value = final.GateSnapshot(
        plan_sha256=wrong,
        cells=snapshot.value.cells,
        local_metadata=snapshot.value.local_metadata,
        events=snapshot.value.events,
        reconciliations=snapshot.value.reconciliations,
    )
    replica = next(row for row in plan["replicas"] if (row["seed"], row["arm"]) == identity)
    _rewrite_terminal(
        replica,
        lambda receipt: receipt["database"]["summary"].update(plan_sha256=wrong),
    )

    with pytest.raises(final.FinalAggregateError, match="immutable ledger plan"):
        final.finalize(plan, snapshots, output_root=_test_path(plan["private_output_root"]))

    assert all(value.score_reads == 0 for value in snapshots.values())


@pytest.mark.parametrize(
    "drift",
    ("evaluation_identity", "job_uid", "config_map_name", "config_map_uid", "privacy"),
)
def test_terminal_identity_drift_prevents_every_score_read(tmp_path: Path, drift: str) -> None:
    plan, snapshots = _study(tmp_path)
    replica = next(row for row in plan["replicas"] if row["seed"] == 48 and row["arm"] == "base")

    def mutate(receipt: dict[str, Any]) -> None:
        if drift == "evaluation_identity":
            receipt["evaluation_identity_sha256"] = "sha256:" + "0" * 64
        elif drift == "job_uid":
            receipt["job"]["uid"] = "not-a-uid"
        elif drift == "config_map_name":
            receipt["config_map"]["name"] = "wrong-config-map"
        elif drift == "config_map_uid":
            receipt["config_map"]["uid"] = "not-a-uid"
        else:
            receipt["privacy"]["prompts_responses_flags_rewards_or_trace_content_included"] = True

    _rewrite_terminal(replica, mutate)

    with pytest.raises(final.FinalAggregateError, match="terminal receipt identity"):
        final.finalize(plan, snapshots, output_root=_test_path(plan["private_output_root"]))

    assert all(value.score_reads == 0 for value in snapshots.values())


def test_scored_query_cannot_change_score_blind_metadata(tmp_path: Path) -> None:
    plan, snapshots = _study(tmp_path)
    snapshot = snapshots[(48, "base")]
    snapshot.scored[0]["config_sha256"] = "a" * 64

    with pytest.raises(final.FinalAggregateError, match="score-blind gate"):
        final.finalize(plan, snapshots, output_root=_test_path(plan["private_output_root"]))

    assert snapshot.score_reads == 1
    assert not _test_path(plan["private_output_root"]).exists()


def test_terminal_evidence_digest_is_from_the_bytes_gated_before_score_open(
    tmp_path: Path,
) -> None:
    plan, snapshots = _study(tmp_path)
    replica = next(row for row in plan["replicas"] if row["seed"] == 48 and row["arm"] == "base")
    terminal_path = _test_path(replica["terminal_receipt_path"])
    original_file_sha256 = final._file_digest(terminal_path)  # noqa: SLF001
    prior = snapshots[(48, "base")]

    def mutate_after_gate() -> None:
        terminal_path.write_bytes(terminal_path.read_bytes() + b"\n")

    snapshots[(48, "base")] = CallbackSnapshot(prior.value, prior.scored, mutate_after_gate)
    final.finalize(plan, snapshots, output_root=_test_path(plan["private_output_root"]))

    terminal_index = json.loads(
        (_test_path(plan["private_output_root"]) / "PRIVATE_TERMINAL_INDEX.json").read_text()
    )
    row = next(
        item for item in terminal_index["rows"] if item["seed"] == 48 and item["arm"] == "base"
    )
    assert row["terminal_receipt_file_sha256"] == original_file_sha256
    assert final._file_digest(terminal_path) != original_file_sha256  # noqa: SLF001


def test_json_digest_is_computed_from_the_same_single_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "evidence.json"
    original = b'{"value":"original"}\n'
    source.write_bytes(original)
    real_read = final._read_regular_once  # noqa: SLF001
    calls = 0

    def mutate_after_read(path: Path, label: str) -> bytes:
        nonlocal calls
        calls += 1
        raw = real_read(path, label)
        path.write_text('{"value":"replacement"}\n')
        return raw

    monkeypatch.setattr(final, "_read_regular_once", mutate_after_read)
    value, digest = final._read_json_and_digest(source, "test evidence")  # noqa: SLF001

    assert calls == 1
    assert value == {"value": "original"}
    assert digest == "sha256:" + hashlib.sha256(original).hexdigest()


def test_private_random_nonce_changes_public_commitment_for_identical_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class NoShuffle:
        @staticmethod
        def shuffle(_values: list[int]) -> None:
            return None

    nonces = iter(("11" * 32, "22" * 32))
    monkeypatch.setattr(final.secrets, "SystemRandom", NoShuffle)
    monkeypatch.setattr(final.secrets, "token_hex", lambda _length: next(nonces))
    receipts = []
    for name in ("first", "second"):
        study_root = tmp_path / name
        study_root.mkdir()
        plan, snapshots = _study(study_root)
        final.finalize(plan, snapshots, output_root=_test_path(plan["private_output_root"]))
        public_path = _test_path(plan["private_output_root"]) / "SANITIZED_AGGREGATE.json"
        public = json.loads(public_path.read_text())
        receipts.append(public["anonymization_receipt_sha256"])
        assert "private_nonce" not in public_path.read_text()

    assert receipts[0] != receipts[1]


def test_atomic_publication_never_replaces_a_concurrent_claimant(tmp_path: Path) -> None:
    plan, snapshots = _study(tmp_path)
    output = _test_path(plan["private_output_root"])
    prior = snapshots[(48, "base")]

    def claim_output_after_initial_absence_check() -> None:
        output.mkdir()
        (output / "claimant.txt").write_text("preserve me\n")

    snapshots[(48, "base")] = CallbackSnapshot(
        prior.value,
        prior.scored,
        claim_output_after_initial_absence_check,
    )
    with pytest.raises(final.FinalAggregateError, match="claimed concurrently"):
        final.finalize(plan, snapshots, output_root=output)

    assert (output / "claimant.txt").read_text() == "preserve me\n"
    assert not (output / "FINAL.json").exists()


def test_resigned_replica_output_redirection_is_rejected(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    replica = plan["replicas"][0]
    replica["output_root"] = "/mnt/sfs/jobs/unreviewed-replica-output"
    replica["terminal_receipt_path"] = replica["output_root"] + "/TERMINAL_OBSERVATION.json"
    plan.pop("sha256")
    plan["sha256"] = final._digest(plan)  # noqa: SLF001

    with pytest.raises(final.FinalAggregateError, match="replica identity differs"):
        final.validate_plan(plan)


def _add_reconciliation(
    plan: dict[str, Any], snapshots: dict[tuple[int, str], FakeSnapshot]
) -> dict[str, Any]:
    replica = next(row for row in plan["replicas"] if row["seed"] == 48 and row["arm"] == "base")
    evaluation = json.loads((_test_path(replica["output_root"]) / "EVAL.json").read_text())
    terminal_path = _test_path(replica["terminal_receipt_path"])
    terminal = json.loads(terminal_path.read_text())
    terminal.pop("sha256")
    terminal["decision"]["score_blind_reconciliation_required"] = True
    terminal["decision"]["unresolved_cells"] = 1
    terminal["sha256"] = final._digest(terminal)  # noqa: SLF001
    terminal_path.write_text(json.dumps(terminal, sort_keys=True) + "\n")
    snapshot = snapshots[(48, "base")]
    cell = snapshot.value.cells[0]
    metadata = next(
        row for row in snapshot.value.local_metadata if row["cell_id"] == cell["cell_id"]
    )
    (_test_path(replica["output_root"]) / metadata["artifact_directory"] / "ACCEPTED.json").unlink()
    metadata.update(agent_exit_code=1, agent_termination="process_error")
    scored = next(row for row in snapshot.scored if row["cell_id"] == cell["cell_id"])
    scored.update(agent_exit_code=1, agent_termination="process_error")
    scored["record_sha256"] = final._plain_digest(  # noqa: SLF001
        {field: scored[field] for field in final.LOCAL_RECORD_FIELDS}
    )
    intent = "9" * 64
    cell["reconciliation_digest"] = intent
    snapshot.value.events[0] = {
        "cell_id": cell["cell_id"],
        "event": "stored_scored_session_reconciled",
        "to_state": "accepted",
        "detail_json": json.dumps(
            {
                "reviewed_intent_sha256": intent,
                "cell_receipt_sha256": cell["receipt_digest"],
                "source_job_terminal_receipt_sha256": terminal["sha256"],
                "action": "accept_existing_scored_session",
            }
        ),
    }
    evidence = _signed(
        {
            "schema_version": "fleet-stored-session-reconciliation-v2",
            "reviewed_intent_sha256": intent,
            "evaluation_plan_sha256": evaluation["sha256"],
            "source_job_uid_sha256": (
                "sha256:" + hashlib.sha256(terminal["job"]["uid"].encode()).hexdigest()
            ),
            "source_job_terminal_receipt_sha256": terminal["sha256"],
            "selected_cell_count": 1,
            "prior_retry_review_count": 1,
            "accepted_existing_completed_session_count": 1,
            "source_agent_exit_code": 1,
            "source_agent_termination": "process_error",
            "source_failure_code_sha256": "sha256:" + "f" * 64,
            "action": "accept_existing_scored_session",
            "model_generation_performed": False,
            "scoring_call_performed": False,
            "score_values_included": False,
            "prompt_response_flag_reward_or_trace_content_included": False,
            "cell_task_session_or_trace_identifiers_included": False,
        },
        "receipt_sha256",
    )
    snapshot.value.reconciliations.append(evidence)
    return evidence


def test_reviewed_stored_session_needs_matching_private_reconciliation(tmp_path: Path) -> None:
    plan, snapshots = _study(tmp_path)
    _add_reconciliation(plan, snapshots)

    with pytest.raises(final.FinalAggregateError, match="held or infrastructure-invalid"):
        final.finalize(plan, snapshots, output_root=_test_path(plan["private_output_root"]))
    assert all(value.score_reads == 0 for value in snapshots.values())


def test_legacy_reconciliation_receipt_is_fully_bound_before_score_open(tmp_path: Path) -> None:
    plan, snapshots = _study(tmp_path)
    evidence = _add_reconciliation(plan, snapshots)
    evidence.pop("receipt_sha256")
    evidence["schema_version"] = "fleet-stored-session-reconciliation-v1"
    evidence["prior_stale_active_count"] = 0
    for field in (
        "source_agent_exit_code",
        "source_agent_termination",
        "source_failure_code_sha256",
        "action",
    ):
        evidence.pop(field)
    evidence["receipt_sha256"] = final._digest(evidence)  # noqa: SLF001
    event = snapshots[(48, "base")].value.events[0]
    event["event"] = "stored_session_reconciled"
    detail = json.loads(event["detail_json"])
    detail.pop("action")
    detail["failure_code"] = "stored_session.owner_terminal"
    event["detail_json"] = json.dumps(detail)

    with pytest.raises(final.FinalAggregateError, match="held or infrastructure-invalid"):
        final.finalize(plan, snapshots, output_root=_test_path(plan["private_output_root"]))
    assert all(value.score_reads == 0 for value in snapshots.values())


def _as_subset_receipt(evidence: dict[str, Any]) -> None:
    evidence.pop("receipt_sha256")
    evidence.update(
        source_total_cell_count=17,
        prior_arm_state_counts={
            "pending": 0,
            "claimed": 0,
            "running": 0,
            "grading": 0,
            "accepted": 16,
            "retry_review": 1,
            "terminal": 0,
        },
        post_arm_state_counts={
            "pending": 0,
            "claimed": 0,
            "running": 0,
            "grading": 0,
            "accepted": 17,
            "retry_review": 0,
            "terminal": 0,
        },
        nonselected_cell_count=16,
        nonselected_cells_preserved=True,
    )
    evidence["receipt_sha256"] = final._digest(evidence)  # noqa: SLF001


def test_subset_reconciliation_state_transition_is_fully_bound(tmp_path: Path) -> None:
    plan, snapshots = _study(tmp_path)
    evidence = _add_reconciliation(plan, snapshots)
    _as_subset_receipt(evidence)

    with pytest.raises(final.FinalAggregateError, match="held or infrastructure-invalid"):
        final.finalize(plan, snapshots, output_root=_test_path(plan["private_output_root"]))
    assert all(value.score_reads == 0 for value in snapshots.values())


@pytest.mark.parametrize("drift", ("total", "states", "local_counts"))
def test_subset_reconciliation_count_drift_prevents_score_open(tmp_path: Path, drift: str) -> None:
    plan, snapshots = _study(tmp_path)
    evidence = _add_reconciliation(plan, snapshots)
    _as_subset_receipt(evidence)
    evidence.pop("receipt_sha256")
    if drift == "total":
        evidence["source_total_cell_count"] = 18
    elif drift == "states":
        evidence["post_arm_state_counts"]["accepted"] = 16
        evidence["post_arm_state_counts"]["retry_review"] = 1
    else:
        evidence.update(
            source_local_result_count=15,
            missing_local_result_count=1,
            missing_local_results_are_unselected=True,
            selected_cells_have_local_results=True,
            missing_local_result_cells_preserved=True,
            missing_local_result_failure_code_sha256="sha256:" + "a" * 64,
        )
    evidence["receipt_sha256"] = final._digest(evidence)  # noqa: SLF001

    with pytest.raises(final.FinalAggregateError, match="reconciliation receipt is invalid"):
        final.finalize(plan, snapshots, output_root=_test_path(plan["private_output_root"]))

    assert all(value.score_reads == 0 for value in snapshots.values())


@pytest.mark.parametrize("drift", ("plan", "source", "action", "count"))
def test_reconciliation_identity_drift_prevents_score_open(tmp_path: Path, drift: str) -> None:
    plan, snapshots = _study(tmp_path)
    evidence = _add_reconciliation(plan, snapshots)
    evidence.pop("receipt_sha256")
    if drift == "plan":
        evidence["evaluation_plan_sha256"] = "sha256:" + "0" * 64
    elif drift == "source":
        evidence["source_job_terminal_receipt_sha256"] = "sha256:" + "0" * 64
    elif drift == "action":
        evidence["action"] = "generate_more_model_output"
    else:
        evidence["selected_cell_count"] = 2
        evidence["accepted_existing_completed_session_count"] = 2
    evidence["receipt_sha256"] = final._digest(evidence)  # noqa: SLF001

    with pytest.raises(final.FinalAggregateError, match="reconciliation receipt is invalid"):
        final.finalize(plan, snapshots, output_root=_test_path(plan["private_output_root"]))

    assert all(value.score_reads == 0 for value in snapshots.values())


def test_renderer_is_cpu_only_c1_alert_suppressed_and_module_bound(tmp_path: Path) -> None:
    root = tmp_path / "render"

    receipt = renderer.render(output=root, migration_receipt=_migration(tmp_path))

    bundle = yaml.safe_load((root / "final-aggregate.yaml").read_text())
    config_map, job = bundle["items"]
    assert receipt["external_mutations"] == 0
    assert receipt["launch_performed"] is False
    assert receipt["two_server_previews_required_before_create"] is True
    assert receipt["source_reader_identity"] == {
        "uid": 0,
        "gid": 100,
        "sfs_access": "read_only",
    }
    assert receipt["publisher_identity"] == {
        "uid": 1000,
        "gid": 100,
        "database_credential": False,
    }
    assert job["metadata"]["annotations"] == {
        "fleet.ai/failure-alerts": "off",
        "cyber-post-train.fleet.ai/create-once": "true",
    }
    assert job["spec"]["template"]["spec"]["priorityClassName"] == "c1"
    pod = job["spec"]["template"]["spec"]
    assert pod["automountServiceAccountToken"] is False
    assert pod["securityContext"] == {"seccompProfile": {"type": "RuntimeDefault"}}
    aggregate_container, publisher_container = pod["containers"]
    assert aggregate_container["securityContext"] == {
        "allowPrivilegeEscalation": False,
        "capabilities": {"drop": ["ALL"]},
        "readOnlyRootFilesystem": True,
        "runAsNonRoot": False,
        "runAsUser": 0,
        "runAsGroup": 100,
    }
    assert publisher_container["securityContext"] == {
        "allowPrivilegeEscalation": False,
        "capabilities": {"drop": ["ALL"]},
        "readOnlyRootFilesystem": True,
        "runAsNonRoot": True,
        "runAsUser": 1000,
        "runAsGroup": 100,
    }
    assert "nvidia.com/gpu" not in json.dumps(job)
    files = json.loads(gzip.decompress(base64_decode(config_map["binaryData"]["bundle.json.gz"])))
    assert set(files) == {
        "aggregate.py",
        "publish.py",
        "study.json",
        "run.py",
        "wheel-lock.json",
    }
    assert receipt["source_files"]["aggregate.py"] == (
        "sha256:" + hashlib.sha256(files["aggregate.py"].encode()).hexdigest()
    )
    wheels = json.loads(files["wheel-lock.json"])
    assert wheels == renderer.WHEEL_LOCK
    assert all(
        item["url"].startswith("https://files.pythonhosted.org/packages/")
        and len(item["sha256"]) == 64
        and item["size"] > 0
        for item in wheels
    )
    command = aggregate_container["args"][0]
    assert "uv run" not in command and "--with" not in command
    assert "hashlib.sha256(wheel).hexdigest()==item['sha256']" in command
    assert "python run.py" in command
    assert "(os.geteuid(),os.getegid())==(0,100)" in command
    assert "pass8-staging-probe" in command
    assert "for replica in plan['replicas']" in command
    assert "source/'EVAL.json'" in command
    assert "trap - EXIT" in command
    assert "[ ! -f /result-staging/READY ]" in command
    assert ": > /result-staging/FAILED || true" in command
    assert "assert " not in command
    assert (
        subprocess.run(
            ["bash", "-n"],
            input=command,
            text=True,
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )
    bootstrap = command.split("python - <<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    ast.parse(bootstrap)
    assert files["run.py"].index("print(json.dumps") < files["run.py"].index(
        'ready = staging.parent / "READY"'
    )
    assert "sort_keys=True), flush=True)" in files["run.py"]
    mounts = aggregate_container["volumeMounts"]
    assert next(row for row in mounts if row["name"] == "sfs-readonly")["readOnly"] is True
    assert not any(row["name"] == "sfs-control" for row in mounts)
    assert (
        next(row for row in mounts if row["name"] == "result-staging").get("readOnly", False)
        is False
    )
    aggregate_env = {row["name"] for row in aggregate_container["env"]}
    assert "ROLLOUT_DATABASE_URL" in aggregate_env
    publisher_mounts = publisher_container["volumeMounts"]
    assert not any(row["name"] == "sfs-readonly" for row in publisher_mounts)
    assert (
        next(row for row in publisher_mounts if row["name"] == "result-staging")["readOnly"] is True
    )
    control = next(row for row in publisher_mounts if row["name"] == "sfs-control")
    assert control["subPath"] == "jobs/chris-q38-study-corpora-v1/launch-controls"
    publisher_env = {row["name"] for row in publisher_container["env"]}
    assert "ROLLOUT_DATABASE_URL" not in publisher_env
    publisher_command = publisher_container["args"][0]
    assert "FINAL_PUBLISHER_MODULE_SHA256" in publisher_command
    assert "assert " not in publisher_command
    assert (
        subprocess.run(
            ["bash", "-n"],
            input=publisher_command,
            text=True,
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )
    ast.parse(renderer.PUBLISHER)


def base64_decode(value: str) -> bytes:
    import base64

    return base64.b64decode(value, validate=True)


def _server_preview(bundle: dict[str, Any], uid: str) -> dict[str, Any]:
    value = copy.deepcopy(bundle)
    for item in value["items"]:
        item["metadata"].update(
            uid=uid,
            creationTimestamp="2026-09-23T00:00:00Z",
            resourceVersion="1",
        )
        if item["kind"] == "Job":
            item["status"] = {"active": 0}
            item["spec"]["selector"] = {"matchLabels": {"controller-uid": uid}}
            item["spec"]["template"]["metadata"]["labels"].update(
                {
                    "controller-uid": uid,
                    "batch.kubernetes.io/controller-uid": uid,
                }
            )
    return value


def test_two_preview_validator_normalizes_only_server_job_identity(tmp_path: Path) -> None:
    root = tmp_path / "render"
    migration = _migration(tmp_path)
    renderer.render(output=root, migration_receipt=migration)
    bundle = yaml.safe_load((root / "final-aggregate.yaml").read_text())
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(_server_preview(bundle, "11111111-1111-4111-8111-111111111111")))
    second.write_text(json.dumps(_server_preview(bundle, "22222222-2222-4222-8222-222222222222")))

    receipt = renderer.validate_previews(
        render_root=root,
        migration_receipt=migration,
        first=first,
        second=second,
        output=tmp_path / "previews.json",
    )

    assert receipt["root_failure_alerts"] == "off"
    assert receipt["priority_class"] == "c1"
    assert receipt["gpu_requests"] == 0
    assert receipt["create_performed"] is False


def test_two_preview_validator_rejects_render_receipt_drift(tmp_path: Path) -> None:
    root = tmp_path / "render"
    migration = _migration(tmp_path)
    renderer.render(output=root, migration_receipt=migration)
    bundle = yaml.safe_load((root / "final-aggregate.yaml").read_text())
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(_server_preview(bundle, "11111111-1111-4111-8111-111111111111")))
    second.write_text(json.dumps(_server_preview(bundle, "22222222-2222-4222-8222-222222222222")))
    receipt = json.loads((root / "RENDER.json").read_text())
    receipt["gpu_requests"] = 1
    (root / "RENDER.json").write_text(json.dumps(receipt))

    with pytest.raises(renderer.RenderError, match="self digest"):
        renderer.validate_previews(
            render_root=root,
            migration_receipt=migration,
            first=first,
            second=second,
            output=tmp_path / "previews.json",
        )


def test_two_preview_validator_rejects_admission_added_gpu(tmp_path: Path) -> None:
    root = tmp_path / "render"
    migration = _migration(tmp_path)
    renderer.render(output=root, migration_receipt=migration)
    bundle = yaml.safe_load((root / "final-aggregate.yaml").read_text())
    first_value = _server_preview(bundle, "11111111-1111-4111-8111-111111111111")
    second_value = _server_preview(bundle, "22222222-2222-4222-8222-222222222222")
    for value in (first_value, second_value):
        job = next(item for item in value["items"] if item["kind"] == "Job")
        job["spec"]["template"]["spec"]["initContainers"] = [
            {
                "name": "admission-added",
                "image": "example.invalid/image@sha256:" + "0" * 64,
                "resources": {"limits": {"nvidia.com/mig-1g.10gb": 1}},
            }
        ]
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(first_value))
    second.write_text(json.dumps(second_value))

    with pytest.raises(renderer.RenderError, match="unexpectedly requests an accelerator"):
        renderer.validate_previews(
            render_root=root,
            migration_receipt=migration,
            first=first,
            second=second,
            output=tmp_path / "previews.json",
        )


def test_two_preview_validator_rejects_one_preview_replayed_twice(tmp_path: Path) -> None:
    root = tmp_path / "render"
    migration = _migration(tmp_path)
    renderer.render(output=root, migration_receipt=migration)
    bundle = yaml.safe_load((root / "final-aggregate.yaml").read_text())
    preview = tmp_path / "preview.json"
    preview.write_text(json.dumps(_server_preview(bundle, "11111111-1111-4111-8111-111111111111")))

    with pytest.raises(renderer.RenderError, match="distinct regular files"):
        renderer.validate_previews(
            render_root=root,
            migration_receipt=migration,
            first=preview,
            second=preview,
            output=tmp_path / "previews.json",
        )


@pytest.mark.parametrize(
    "drift",
    (
        "host_network",
        "privileged",
        "env_from",
        "parallelism",
        "manual_selector",
        "termination_message",
        "priority",
        "automount",
        "unreviewed_uid",
    ),
)
def test_two_preview_validator_rejects_unsafe_admission_fields(tmp_path: Path, drift: str) -> None:
    root = tmp_path / "render"
    migration = _migration(tmp_path)
    renderer.render(output=root, migration_receipt=migration)
    bundle = yaml.safe_load((root / "final-aggregate.yaml").read_text())
    previews = [
        _server_preview(bundle, "11111111-1111-4111-8111-111111111111"),
        _server_preview(bundle, "22222222-2222-4222-8222-222222222222"),
    ]
    for value in previews:
        job = next(item for item in value["items"] if item["kind"] == "Job")
        pod = job["spec"]["template"]["spec"]
        if drift == "host_network":
            pod["hostNetwork"] = True
        elif drift == "privileged":
            pod["containers"][0]["securityContext"] = {"privileged": True}
        elif drift == "env_from":
            pod["containers"][0]["envFrom"] = [{"secretRef": {"name": "unreviewed-secret"}}]
        elif drift == "parallelism":
            job["spec"]["parallelism"] = 2
        elif drift == "manual_selector":
            job["spec"]["manualSelector"] = True
            job["spec"]["selector"] = {"matchLabels": {"attacker.example/selected": "true"}}
        elif drift == "termination_message":
            pod["containers"][0]["terminationMessagePath"] = (
                "/mnt/sfs/private/PRIVATE_SCORED_OUTCOME_INDEX.json"
            )
        elif drift == "automount":
            pod["automountServiceAccountToken"] = True
        elif drift == "unreviewed_uid":
            pod["securityContext"]["runAsUser"] = 1000
        else:
            pod["priority"] = 1_000_000
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(previews[0]))
    second.write_text(json.dumps(previews[1]))

    with pytest.raises(renderer.RenderError):
        renderer.validate_previews(
            render_root=root,
            migration_receipt=migration,
            first=first,
            second=second,
            output=tmp_path / "previews.json",
        )


@pytest.mark.parametrize("drift", ("extra_secret", "config_map_data", "config_map_finalizer"))
def test_two_preview_validator_rejects_extra_objects_or_config_map_fields(
    tmp_path: Path, drift: str
) -> None:
    root = tmp_path / "render"
    migration = _migration(tmp_path)
    renderer.render(output=root, migration_receipt=migration)
    bundle = yaml.safe_load((root / "final-aggregate.yaml").read_text())
    previews = [
        _server_preview(bundle, "11111111-1111-4111-8111-111111111111"),
        _server_preview(bundle, "22222222-2222-4222-8222-222222222222"),
    ]
    for value in previews:
        config_map = next(item for item in value["items"] if item["kind"] == "ConfigMap")
        if drift == "extra_secret":
            value["items"].append(
                {
                    "apiVersion": "v1",
                    "kind": "Secret",
                    "metadata": {"name": "unreviewed", "namespace": "fleet-train-jobs"},
                    "stringData": {"private": "content"},
                }
            )
        elif drift == "config_map_data":
            config_map["data"] = {"unreviewed": "executable"}
        else:
            config_map["metadata"]["finalizers"] = ["attacker.example/hold"]
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(previews[0]))
    second.write_text(json.dumps(previews[1]))

    with pytest.raises(renderer.RenderError):
        renderer.validate_previews(
            render_root=root,
            migration_receipt=migration,
            first=first,
            second=second,
            output=tmp_path / "previews.json",
        )


def test_preview_validator_rebuilds_render_policy_after_resigning_tamper(
    tmp_path: Path,
) -> None:
    root = tmp_path / "render"
    migration = _migration(tmp_path)
    renderer.render(output=root, migration_receipt=migration)
    bundle_path = root / "final-aggregate.yaml"
    bundle = yaml.safe_load(bundle_path.read_text())
    job = next(item for item in bundle["items"] if item["kind"] == "Job")
    job["spec"]["template"]["spec"]["priorityClassName"] = "c0"
    bundle_path.write_text(yaml.safe_dump(bundle, sort_keys=False))
    receipt_path = root / "RENDER.json"
    receipt = json.loads(receipt_path.read_text())
    receipt.pop("sha256")
    receipt["rendered_bundle_file_sha256"] = final._file_digest(bundle_path)  # noqa: SLF001
    receipt["sha256"] = renderer._canonical_digest(receipt)  # noqa: SLF001
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(_server_preview(bundle, "11111111-1111-4111-8111-111111111111")))
    second.write_text(json.dumps(_server_preview(bundle, "22222222-2222-4222-8222-222222222222")))

    with pytest.raises(renderer.RenderError, match="Kubernetes objects differ"):
        renderer.validate_previews(
            render_root=root,
            migration_receipt=migration,
            first=first,
            second=second,
            output=tmp_path / "previews.json",
        )


def test_preview_validator_rejects_resigned_executable_bundle_tamper(tmp_path: Path) -> None:
    root = tmp_path / "render"
    migration = _migration(tmp_path)
    renderer.render(output=root, migration_receipt=migration)
    bundle_path = root / "final-aggregate.yaml"
    original = yaml.safe_load(bundle_path.read_text())
    config_map = next(item for item in original["items"] if item["kind"] == "ConfigMap")
    files = json.loads(gzip.decompress(base64_decode(config_map["binaryData"]["bundle.json.gz"])))
    files["aggregate.py"] += "\nraise RuntimeError('unreviewed executable')\n"
    compressed = gzip.compress(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode(),
        compresslevel=9,
        mtime=0,
    )
    compressed = compressed[:9] + b"\xff" + compressed[10:]
    digests = {
        name: "sha256:" + hashlib.sha256(value.encode()).hexdigest()
        for name, value in files.items()
    }
    plan = json.loads(files["study.json"])
    changed_map, changed_job = renderer._objects(plan, compressed, digests)  # noqa: SLF001
    changed_bundle = {
        "apiVersion": "v1",
        "kind": "List",
        "items": [changed_map, changed_job],
    }
    bundle_path.write_text(yaml.safe_dump(changed_bundle, sort_keys=False))
    changed_receipt = renderer._render_receipt(  # noqa: SLF001
        plan=plan,
        compressed=compressed,
        digests=digests,
        rendered_bundle_file_sha256=final._file_digest(bundle_path),  # noqa: SLF001
    )
    (root / "RENDER.json").write_text(json.dumps(changed_receipt, sort_keys=True) + "\n")
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(
        json.dumps(_server_preview(changed_bundle, "11111111-1111-4111-8111-111111111111"))
    )
    second.write_text(
        json.dumps(_server_preview(changed_bundle, "22222222-2222-4222-8222-222222222222"))
    )

    with pytest.raises(renderer.RenderError, match="executable bytes differ"):
        renderer.validate_previews(
            render_root=root,
            migration_receipt=migration,
            first=first,
            second=second,
            output=tmp_path / "previews.json",
        )


@pytest.mark.parametrize("drift", ("output_root", "replacement_identity"))
def test_preview_validator_rejects_resigned_study_plan_tamper(tmp_path: Path, drift: str) -> None:
    root = tmp_path / "render"
    migration = _migration(tmp_path)
    renderer.render(output=root, migration_receipt=migration)
    bundle_path = root / "final-aggregate.yaml"
    original = yaml.safe_load(bundle_path.read_text())
    config_map = next(item for item in original["items"] if item["kind"] == "ConfigMap")
    files = json.loads(gzip.decompress(base64_decode(config_map["binaryData"]["bundle.json.gz"])))
    plan = json.loads(files["study.json"])
    plan.pop("sha256")
    if drift == "output_root":
        plan["private_output_root"] = "/mnt/sfs/jobs/unreviewed-private-output"
    else:
        replica = next(row for row in plan["replicas"] if row["seed"] == 54)
        replica["evaluation_identity_sha256"] = "sha256:" + "0" * 64
    plan["sha256"] = final._digest(plan)  # noqa: SLF001
    files["study.json"] = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    compressed = gzip.compress(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode(),
        compresslevel=9,
        mtime=0,
    )
    compressed = compressed[:9] + b"\xff" + compressed[10:]
    digests = {
        name: "sha256:" + hashlib.sha256(value.encode()).hexdigest()
        for name, value in files.items()
    }
    changed_map, changed_job = renderer._objects(plan, compressed, digests)  # noqa: SLF001
    changed_bundle = {
        "apiVersion": "v1",
        "kind": "List",
        "items": [changed_map, changed_job],
    }
    bundle_path.write_text(yaml.safe_dump(changed_bundle, sort_keys=False))
    changed_receipt = renderer._render_receipt(  # noqa: SLF001
        plan=plan,
        compressed=compressed,
        digests=digests,
        rendered_bundle_file_sha256=final._file_digest(bundle_path),  # noqa: SLF001
    )
    (root / "RENDER.json").write_text(json.dumps(changed_receipt, sort_keys=True) + "\n")
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(
        json.dumps(_server_preview(changed_bundle, "11111111-1111-4111-8111-111111111111"))
    )
    second.write_text(
        json.dumps(_server_preview(changed_bundle, "22222222-2222-4222-8222-222222222222"))
    )

    with pytest.raises(renderer.RenderError, match="authoritative migration evidence"):
        renderer.validate_previews(
            render_root=root,
            migration_receipt=migration,
            first=first,
            second=second,
            output=tmp_path / "previews.json",
        )
