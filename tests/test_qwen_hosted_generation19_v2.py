from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from evals.fleet import qwen_hosted_generation19_bulk as prior
from evals.fleet import qwen_hosted_generation19_v2 as g19
from evals.fleet import qwen_hosted_generation19_v2_package as package
from evals.fleet import qwen_hosted_generation19_v2_runtime as runtime
from evals.fleet import qwen_hosted_generation19_v3 as g19_v3
from evals.fleet import qwen_hosted_generation19_v3_package as package_v3
from evals.fleet import qwen_hosted_generation19_v4 as g19_v4
from evals.fleet import qwen_hosted_generation19_v4_package as package_v4
from evals.fleet import self_hosted

ROOT = Path(__file__).parents[1]
TOMBSTONE = ROOT / (
    "docs/evidence/qwen38-study/2026-09-05-qwen38-generation19-bulk-v1-terminal.json"
)


def test_successor_preserves_cells_with_fresh_controller_identities() -> None:
    plans = g19.validate_all(ROOT)
    sources = prior.validate_all(ROOT)
    assert {len(plan["attempts"]) for plan in plans.values()} == {192}
    for controller, plan in plans.items():
        assert plan["campaign_id"].endswith("-v2")
        assert plan["sfs_root"].endswith("-v2")
        assert [
            (row["cell_id"], row["execution_id"]) for row in plan["attempts"]
        ] == [
            (row["cell_id"], row["execution_id"]) for row in sources[controller]["attempts"]
        ]


def test_package_is_closed_and_instrumented(tmp_path: Path) -> None:
    rendered = package.render(ROOT)
    assert [item["kind"] for item in rendered["items"]] == [
        "ConfigMap",
        "Job",
        "ConfigMap",
        "Job",
    ]
    for cm in rendered["items"][::2]:
        script = cm["data"]["run_qwen_hosted_generation19_v2.sh"]
        assert all(name in script for name in cm["data"] if name.endswith(".py"))
        assert "qwen_hosted_generation19_v2_runtime.py" in cm["data"]
        module_root = tmp_path / cm["metadata"]["name"] / "evals" / "fleet"
        config_root = module_root / "configs"
        config_root.mkdir(parents=True)
        (module_root.parent / "__init__.py").touch()
        (module_root / "__init__.py").touch()
        for name, value in cm["data"].items():
            if name.endswith(".py"):
                (module_root / name).write_text(value)
        (config_root / "qwen-hosted-generation19-qwen-a-v2.json").write_text(
            cm["data"]["plan-v2-a.json"]
        )
        (config_root / "qwen-hosted-generation19-qwen-b-v2.json").write_text(
            cm["data"]["plan-v2-b.json"]
        )
        subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; "
                    "from evals.fleet import qwen_hosted_generation19_v2 as g; "
                    "from evals.fleet import qwen_hosted_generation19_v2_runtime; "
                    "g.validate_all(Path.cwd())"
                ),
            ],
            cwd=module_root.parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
    for job in rendered["items"][1::2]:
        env = {
            row["name"]: row.get("value")
            for row in job["spec"]["template"]["spec"]["containers"][0]["env"]
        }
        assert env["G19_DIAGNOSTIC_ROOT"].endswith("-diagnostic")
        assert env["G19_CONTROLLER"] in {"qwen-a", "qwen-b"}
        assert env["G19_PREFLIGHT_SHA256"] == package.PREFLIGHT_SHA256


def test_runtime_failure_receipt_records_last_sanitized_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = g19.validate_all(ROOT)["qwen-a"]
    item = plan["attempts"][0]

    def fail(*_args: object, **kwargs: object) -> dict:
        observer = kwargs["stage_observer"]
        observer("01-plan-rebuilt", None)
        observer("02-runtime-gate-valid", None)
        observer("03-output-root-initialized", None)
        observer("04-endpoint-lease-acquired", None)
        observer("05-run-identity-absent", item)
        observer("06-route-valid", item)
        raise RuntimeError("content-free diagnostic sentinel")

    monkeypatch.setattr(runtime.engine, "run_controller", fail)
    monkeypatch.setenv("JOB_UID", "11111111-1111-4111-8111-111111111111")
    monkeypatch.setenv("POD_UID", "22222222-2222-4222-8222-222222222222")
    diagnostic = tmp_path / "diagnostic"
    with pytest.raises(RuntimeError, match="sentinel"):
        runtime.run(
            plan,
            out=tmp_path / "out",
            proxy=tmp_path / "proxy",
            diagnostic_root=diagnostic,
        )
    receipt = json.loads((diagnostic / "FAILED.json").read_text())
    assert receipt["last_completed_stage"] == "06-route-valid"
    assert receipt["claim_written"] is False
    assert receipt["model_call_started"] is False
    assert receipt["error_type"] == "RuntimeError"
    assert receipt["receipt_sha256"] == self_hosted.digest_without(receipt, "receipt_sha256")
    assert not ({"prompt", "trace", "flag", "score"} & set(receipt))


def test_v1_terminal_tombstone_is_digest_valid_and_retry_safe() -> None:
    receipt = json.loads(TOMBSTONE.read_text())
    assert receipt["global_execution_claims"] == 0
    assert receipt["model_started_cells"] == 0
    assert receipt["scored_cells"] == 0
    assert receipt["retry_allowed_for_all_planned_cells"] is True
    assert receipt["receipt_sha256"] == self_hosted.digest_without(receipt, "receipt_sha256")


def test_v3_materialized_package_executes_validate_all(tmp_path: Path) -> None:
    plans = g19_v3.validate_all(ROOT)
    assert {len(plan["attempts"]) for plan in plans.values()} == {192}
    rendered = package_v3.render(ROOT)
    for cm in rendered["items"][::2]:
        module_root = tmp_path / cm["metadata"]["name"] / "evals" / "fleet"
        config_root = module_root / "configs"
        config_root.mkdir(parents=True)
        (module_root.parent / "__init__.py").touch()
        (module_root / "__init__.py").touch()
        for name, value in cm["data"].items():
            if name.endswith(".py"):
                (module_root / name).write_text(value)
        for controller in ("a", "b"):
            (config_root / f"qwen-hosted-generation19-qwen-{controller}-v3.json").write_text(
                cm["data"][f"plan-v3-{controller}.json"]
            )
        subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; "
                    "from evals.fleet import qwen_hosted_generation19_v3 as g; "
                    "g.validate_all(Path.cwd())"
                ),
            ],
            cwd=module_root.parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
        probe = textwrap.dedent(
            """
            import copy
            import os
            from pathlib import Path
            from evals.fleet import exact_pass4_bulk_runtime_v3 as engine
            from evals.fleet import qwen_hosted_generation19_v3 as g

            root = Path.cwd()
            plan = copy.deepcopy(g.validate_all(root)["qwen-a"])
            plan["repo_root"] = str(root)
            plan["execution"]["claim_root"] = str(root / "claims")
            plan["execution"]["endpoint_lease"]["lease_root"] = str(root / "leases")
            g.build_runtime_plan = lambda *_args: plan
            engine.bulk = g
            os.environ["FLEET_API_KEY"] = "test-only"
            os.environ["JOB_UID"] = "11111111-1111-4111-8111-111111111111"
            os.environ["POD_UID"] = "22222222-2222-4222-8222-222222222222"
            try:
                engine.run_controller(
                    plan,
                    out=root / "output",
                    proxy=root / "proxy.py",
                    model_runner=lambda *_args: (_ for _ in ()).throw(
                        SystemExit("stop before external model mutation")
                    ),
                    classifier=lambda *_args: {},
                    check_run_absent=lambda *_args: None,
                    route_check=lambda *_args: None,
                    runtime_gate_check=lambda *_args: None,
                )
            except SystemExit:
                pass
            else:
                raise AssertionError("packaged boundary probe did not stop")
            assert len(list((root / "claims").glob("*.json"))) == 1
            """
        )
        subprocess.run(
            [sys.executable, "-c", probe],
            cwd=module_root.parents[1],
            check=True,
            capture_output=True,
            text=True,
        )


def test_v2_terminal_tombstone_is_digest_valid_and_retry_safe() -> None:
    path = ROOT / (
        "docs/evidence/qwen38-study/"
        "2026-09-05-qwen38-generation19-bulk-v2-terminal.json"
    )
    receipt = json.loads(path.read_text())
    assert receipt["error_type"] == "FileNotFoundError"
    assert receipt["global_execution_claims"] == 0
    assert receipt["model_started_cells"] == 0
    assert receipt["receipt_sha256"] == self_hosted.digest_without(receipt, "receipt_sha256")


def test_v4_materialized_entrypoint_reaches_atomic_claim_without_external_mutation(
    tmp_path: Path,
) -> None:
    rendered = package_v4.render(ROOT)
    cm = rendered["items"][0]
    module_root = tmp_path / "materialized" / "evals" / "fleet"
    config_root = module_root / "configs"
    config_root.mkdir(parents=True)
    (module_root.parent / "__init__.py").touch()
    (module_root / "__init__.py").touch()
    for name, value in cm["data"].items():
        if name.endswith(".py"):
            (module_root / name).write_text(value)
    for controller in ("a", "b"):
        (config_root / f"qwen-hosted-generation19-qwen-{controller}-v4.json").write_text(
            cm["data"][f"plan-v4-{controller}.json"]
        )
    probe = textwrap.dedent(
        """
        import copy
        import os
        from pathlib import Path
        from evals.fleet import qwen_hosted_generation19_v2_runtime as shared
        from evals.fleet import qwen_hosted_generation19_v4 as g
        from evals.fleet import qwen_hosted_generation19_v4_runtime
        from evals.fleet import self_hosted

        root = Path.cwd()
        plans = g.validate_all(root)
        plan = copy.deepcopy(plans["qwen-a"])
        plan["repo_root"] = str(root)
        plan["execution"]["claim_root"] = str(root / "claims")
        plan["execution"]["endpoint_lease"]["lease_root"] = str(root / "leases")
        g.build_runtime_plan = lambda *_args: plan
        execution_ids = sorted(
            row["execution_id"] for value in plans.values() for row in value["attempts"]
        )
        body = {
            "schema_version": "fleet-qwen-generation19-hosted-bulk-preflight-v1",
            "status": "CLEAR",
            "planned_cells": 384,
            "planned_execution_ids_sha256": self_hosted.sha256(
                self_hosted.canonical_json(execution_ids)
            ),
            "mutation_calls": 0,
        }
        receipt = {**body, "receipt_sha256": self_hosted.digest_without(body, "receipt_sha256")}
        self_hosted.write_json_once(root / "CLEAR.json", receipt)
        os.environ["G19_PREFLIGHT_PATH"] = str(root / "CLEAR.json")
        os.environ["G19_PREFLIGHT_SHA256"] = receipt["receipt_sha256"]
        os.environ["FLEET_API_KEY"] = "test-only"
        os.environ["JOB_UID"] = "11111111-1111-4111-8111-111111111111"
        os.environ["POD_UID"] = "22222222-2222-4222-8222-222222222222"
        try:
            shared.run(
                plan,
                out=root / "output",
                proxy=root / "proxy.py",
                diagnostic_root=root / "diagnostic",
                bulk_module=g,
                engine_kwargs={
                    "model_runner": lambda *_args: (_ for _ in ()).throw(
                        SystemExit("stop before external model mutation")
                    ),
                    "classifier": lambda *_args: {},
                    "check_run_absent": lambda *_args: None,
                    "route_check": lambda *_args: None,
                },
            )
        except SystemExit:
            pass
        else:
            raise AssertionError("packaged v4 entrypoint did not stop")
        assert len(list((root / "claims").glob("*.json"))) == 1
        assert (root / "diagnostic" / "STAGE-r001-a1-08-model-runner-entered.json").is_file()
        """
    )
    subprocess.run(
        [sys.executable, "-c", probe],
        cwd=module_root.parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    assert {len(plan["attempts"]) for plan in g19_v4.validate_all(ROOT).values()} == {192}
