import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import glm53_dedicated_v24_watchdog_live_release_v1 as release_engine
from evals.fleet import glm53_dedicated_v33_controller_package_v1 as package
from evals.fleet import glm53_dedicated_v33_create_v1 as server
from evals.fleet import glm53_dedicated_v33_incluster_parity_v1 as parity
from evals.fleet import glm53_dedicated_v33_watchdog_live_release_v1 as adapter
from evals.fleet import glm53_dedicated_v33_watchdog_package_v1 as watchdog

ROOT = Path(__file__).resolve().parents[1]
OBSERVED_RUN_ID = "glm53-tp8-v33-e4888aa7"
PACKAGE_COMMIT = "5087b2754be38b64aa403df1b6689218ebe018ae"


class FakeResponse:
    status = 202

    def __init__(self, body: dict[str, object]) -> None:
        self.body = body

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.body).encode()


def authorization() -> dict[str, object]:
    value: dict[str, object] = {
        "receipt_sha256": "authorization-digest",
    }
    return value


def binding(api_run_id: str = OBSERVED_RUN_ID) -> dict[str, object]:
    return {
        "server_title": server.TITLE,
        "server_run_dir": server.RUN_DIR,
        "api_run_id": api_run_id,
        "rayjob_uid": "11111111-1111-4111-8111-111111111111",
        "workload_uid": "22222222-2222-4222-8222-222222222222",
        "head_pod_uid": "33333333-3333-4333-8333-333333333333",
        "service_uid": "44444444-4444-4444-8444-444444444444",
        "service_origin": "http://glm53-tp8-v33-e4888aa7-head.svc:8000",
        "served_id": server.SERVED_ID,
        "model_revision": server.MODEL_REVISION,
        "context_length": server.CONTEXT_LENGTH,
    }


def test_v33_create_accepts_only_requested_prefix_plus_eight_hex(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server, "validate_authorization", lambda _value: None)
    monkeypatch.setenv("FLEET_API_KEY", "test-only")
    result = server.create_once(
        authorization(),
        result_path=tmp_path / "CREATED.json",
        opener=lambda *_args, **_kwargs: FakeResponse({"id": OBSERVED_RUN_ID}),
    )
    assert result["api_run_id"] == OBSERVED_RUN_ID
    assert result["server_api_name"] == server.API_NAME
    assert result["server_run_dir"] == server.RUN_DIR
    assert result["receipt_sha256"] == crypto.digest_without(result, "receipt_sha256")


@pytest.mark.parametrize(
    "api_run_id",
    [
        "ft-run-e4888aa7",
        "glm53-tp8-v32-e4888aa7",
        "glm53-tp8-v33-e4888aa",
        "glm53-tp8-v33-e4888aa70",
        "glm53-tp8-v33-E4888AA7",
        "prefix-glm53-tp8-v33-e4888aa7",
    ],
)
def test_v33_create_rejects_aliases_and_malformed_ids(
    api_run_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server, "validate_authorization", lambda _value: None)
    monkeypatch.setenv("FLEET_API_KEY", "test-only")
    with pytest.raises(server.CreateError, match="identity_invalid_reconcile_do_not_retry"):
        server.create_once(
            authorization(),
            result_path=tmp_path / "CREATED.json",
            opener=lambda *_args, **_kwargs: FakeResponse({"id": api_run_id}),
        )
    assert not (tmp_path / "CREATED.json").exists()


def test_v33_create_rejects_ambiguous_matching_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server, "validate_authorization", lambda _value: None)
    monkeypatch.setenv("FLEET_API_KEY", "test-only")
    with pytest.raises(server.CreateError, match="identity_invalid_reconcile_do_not_retry"):
        server.create_once(
            authorization(),
            result_path=tmp_path / "CREATED.json",
            opener=lambda *_args, **_kwargs: FakeResponse(
                {"id": OBSERVED_RUN_ID, "run": {"id": "glm53-tp8-v33-deadbeef"}}
            ),
        )

    with pytest.raises(server.CreateError, match="identity_invalid_reconcile_do_not_retry"):
        server.create_once(
            authorization(),
            result_path=tmp_path / "CREATED-ALIAS.json",
            opener=lambda *_args, **_kwargs: FakeResponse(
                {"id": OBSERVED_RUN_ID, "run_id": "ft-run-e4888aa7"}
            ),
        )


def test_v33_binding_watchdog_probe_and_release_use_same_exact_contract() -> None:
    server.validate_binding(binding())
    for api_run_id in ("ft-run-e4888aa7", "glm53-tp8-v32-e4888aa7"):
        with pytest.raises(server.CreateError, match="binding_invalid"):
            server.validate_binding(binding(api_run_id))

    with adapter.bound_engine():
        probe = release_engine._api_probe_source(OBSERVED_RUN_ID)  # noqa: SLF001
        assert server.API_RUN_ID_RE.pattern in probe
        assert "ft-run-[0-9a-f]" not in probe
        release_engine._api_delete_source(OBSERVED_RUN_ID)  # noqa: SLF001
        with pytest.raises(release_engine.LiveReleaseError, match="api_run_id_invalid"):
            release_engine._api_delete_source("ft-run-e4888aa7")  # noqa: SLF001


def test_v33_generation_identities_are_fresh_and_score_free() -> None:
    assert server.payload()["name"] == server.API_NAME == "glm53-tp8-v33"
    assert server.TITLE.endswith("-v33")
    assert server.RUN_DIR.endswith("-v33")
    assert watchdog.JOB_NAME.endswith("v33-request-watchdog-v1")
    assert parity.JOB_NAME.endswith("v33-incluster-actual-parity-v1")
    assert package.JOB_NAME.endswith("v33-create-watchdog-parity-controller-v1")
    assert all("v33" in path for path in package.FILES if "glm53_dedicated_v33" in path)
    held = adapter.build_held("a" * 40)
    assert held["api_run_id_pattern"] == server.API_RUN_ID_RE.pattern
    assert held["scored_launch_authorized"] is False
    assert held["api_mutation_calls"] == 0


def test_v33_watchdog_package_runs_generation_exact_runtime() -> None:
    classes = [
        {
            "metadata": {"name": "fleet-infra-quiet"},
            "value": -1000,
            "preemptionPolicy": "Never",
        },
        {
            "metadata": {"name": "fleet-serve-low"},
            "value": 100,
            "preemptionPolicy": "Never",
        },
    ]
    rendered = watchdog.render(
        ROOT,
        PACKAGE_COMMIT,
        binding(),
        ready_at_epoch=1.0,
        priority_classes=classes,
    )
    configmap, job = rendered["objects"]["items"]
    manifest = json.loads(configmap["data"]["package.json"])
    runtime_path = "evals/fleet/glm53_dedicated_v33_request_counter_watchdog_v1.py"
    assert runtime_path in manifest["files"]
    assert manifest["package_sha256"] == crypto.digest_without(manifest, "package_sha256")
    command = job["spec"]["template"]["spec"]["containers"][0]["command"][-1]
    assert "glm53_dedicated_v33_request_counter_watchdog_v1" in command
    assert "glm53_dedicated_v23_request_counter_watchdog_v1 watch" not in command


def test_v33_watchdog_projected_configmap_has_complete_import_closure(
    tmp_path: Path,
) -> None:
    classes = [
        {
            "metadata": {"name": "fleet-infra-quiet"},
            "value": -1000,
            "preemptionPolicy": "Never",
        },
        {
            "metadata": {"name": "fleet-serve-low"},
            "value": 100,
            "preemptionPolicy": "Never",
        },
    ]
    rendered = watchdog.render(
        ROOT,
        PACKAGE_COMMIT,
        binding(),
        ready_at_epoch=1.0,
        priority_classes=classes,
    )
    data = rendered["objects"]["items"][0]["data"]
    projected = tmp_path / "work"
    for key, source in data.items():
        if "__SLASH__" not in key:
            continue
        target = projected / key.replace("__SLASH__", "/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from evals.fleet import "
                "glm53_dedicated_v33_request_counter_watchdog_v1 as runtime; "
                "assert runtime.API_RUN_ID_RE.fullmatch('glm53-tp8-v33-e4888aa7'); "
                "assert not runtime.API_RUN_ID_RE.fullmatch('ft-run-e4888aa7')"
            ),
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(projected)},
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


def test_v32_incident_and_release_receipts_are_sanitized_and_digest_valid() -> None:
    evidence = ROOT / "docs/evidence/glm53-study"
    for name in (
        "2026-09-06-glm53-dedicated-v32-custom-api-id-incident-v1.json",
        "2026-09-06-glm53-dedicated-v32-custom-api-id-release-v1.json",
        "2026-09-06-glm53-dedicated-v33-custom-api-id-lifecycle-held-v1.json",
    ):
        value = json.loads((evidence / name).read_text())
        assert value["receipt_sha256"] == crypto.digest_without(
            copy.deepcopy(value), "receipt_sha256"
        )
        assert value["protected_content_included"] is False
        assert value["fleet_task_instance_calls"] == 0
        assert value["fleet_session_calls"] == 0
        assert value["verifier_calls"] == 0
        assert value["scoring_calls"] == 0
