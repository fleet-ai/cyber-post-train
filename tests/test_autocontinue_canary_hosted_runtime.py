from __future__ import annotations

import copy
import shutil
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from evals.fleet import autocontinue_canary_controller as canary
from evals.fleet import autocontinue_canary_hosted_release as phase_c
from evals.fleet import autocontinue_canary_hosted_runtime as runtime
from evals.fleet import autocontinue_hosted_health as health

ROOT = Path(__file__).parents[1]
Q_PLAN = ROOT / "evals/fleet/configs/qwen38-opencode-autocontinue-canary1-v1.json"
Q_SUCCESSOR_PLAN = ROOT / "evals/fleet/configs/qwen38-opencode-autocontinue-canary1-v2.json"
G_SUCCESSOR_PLAN = ROOT / "evals/fleet/configs/glm53-opencode-autocontinue-canary1-v2.json"


class _Response:
    def __init__(self, payload: bytes) -> None:
        self.status = 200
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self, _size: int) -> bytes:
        return self.payload


def _opener(request, timeout):
    del timeout
    if request.full_url.endswith("/v1/account"):
        return _Response(b'{"team_name":"fleet","team_id":"a1025f0b-ad67-49fc-a023-51800ab43e84"}')
    return _Response(b'{"data":[{"id":"qwen3.8-27b"},{"id":"glm-5.3"}]}')


def _bad_opener(case):
    def opener(request, timeout):
        del timeout
        if case == "transport":
            raise OSError("sanitized transport failure")
        if request.full_url.endswith("/v1/account"):
            if case == "team":
                return _Response(b'{"team_name":"other","team_id":"wrong"}')
            return _Response(
                b'{"team_name":"fleet","team_id":"a1025f0b-ad67-49fc-a023-51800ab43e84"}'
            )
        payload = {
            "malformed": b'{"data":{}}',
            "missing": b'{"data":[{"id":"glm-5.3"}]}',
            "duplicate": (b'{"data":[{"id":"qwen3.8-27b"},{"id":"qwen3.8-27b"},{"id":"glm-5.3"}]}'),
        }[case]
        return _Response(payload)

    return opener


def _route(caller=runtime.LAUNCHER_CALLER, **kwargs):
    return runtime.observe_live_route(
        "redacted",
        caller=caller,
        opener=_opener,
        now=datetime(2026, 9, 4, 21, 45, tzinfo=UTC),
        **kwargs,
    )


def test_live_route_observation_is_sanitized_and_exact() -> None:
    receipt = _route()
    runtime.validate_live_route(
        receipt,
        caller=runtime.LAUNCHER_CALLER,
        maximum_age_seconds=120,
        now=datetime(2026, 9, 4, 21, 46, tzinfo=UTC),
    )
    assert receipt["hosted_route"]["served_ids_present_exactly_once"] == [
        "qwen3.8-27b",
        "glm-5.3",
    ]
    assert receipt["request_counts"]["chat_completions"] == 0
    assert receipt["request_counts"]["fleet_task_or_scoring"] == 0
    assert "redacted" not in repr(receipt)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["fleet_account"].__setitem__("team_name", "other"),
        lambda value: value["hosted_route"].__setitem__(
            "served_ids_present_exactly_once", ["qwen3.8-27b"]
        ),
        lambda value: value["hosted_route"].__setitem__("dedicated_serving_state", "RUNNING"),
        lambda value: value["request_counts"].__setitem__("chat_completions", 1),
        lambda value: value["privacy"].__setitem__("response_bodies_included", True),
        lambda value: value.__setitem__("unknown_authority", True),
    ],
)
def test_live_route_rejects_resealed_semantic_tampering(mutate) -> None:
    receipt = _route()
    mutate(receipt)
    receipt["receipt_sha256"] = canary.digest_without(receipt, "receipt_sha256")
    with pytest.raises(ValueError, match="not authoritative"):
        runtime.validate_live_route(
            receipt,
            caller=runtime.LAUNCHER_CALLER,
            maximum_age_seconds=120,
            now=datetime(2026, 9, 4, 21, 46, tzinfo=UTC),
        )


def test_live_route_rejects_stale_observation() -> None:
    receipt = _route()
    with pytest.raises(ValueError, match="not authoritative"):
        runtime.validate_live_route(
            receipt,
            caller=runtime.LAUNCHER_CALLER,
            maximum_age_seconds=120,
            now=datetime(2026, 9, 4, 21, 45, tzinfo=UTC) + timedelta(seconds=121),
        )


def test_runtime_route_failure_prevents_permanent_claim(monkeypatch, tmp_path: Path) -> None:
    plan = canary.load_object(Q_PLAN)
    launch_receipt = _route()
    called = False
    lease_entered = False
    validations = 0

    monkeypatch.setattr(runtime, "validate_hosted_release", lambda *_: None)

    def validate(*_args, **_kwargs):
        nonlocal validations
        validations += 1
        if validations == 2:
            raise ValueError("route down")

    monkeypatch.setattr(runtime, "validate_live_route", validate)
    monkeypatch.setenv("FLEET_API_KEY", "redacted")

    @contextmanager
    def lease(**_kwargs):
        nonlocal lease_entered
        lease_entered = True
        yield

    monkeypatch.setattr(canary.endpoint_lease, "acquire_endpoint_lease", lease)
    monkeypatch.setattr(
        runtime,
        "observe_live_route",
        lambda *_args, **_kwargs: {"status": "FAILED"},
    )

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(canary, "claim_global_cell", forbidden)
    with pytest.raises(ValueError, match="route down"):
        runtime.run_hosted(
            plan,
            {},
            launch_receipt,
            tmp_path / "out",
            tmp_path / "proxy.py",
            ROOT,
            "16e25f1f127dcf76018a223ad8ac57e2117f83d7",
        )
    assert called is False
    assert lease_entered is True
    assert validations == 2
    assert not (tmp_path / "out").exists()


def test_runtime_live_route_check_is_under_lease_and_before_all_claims() -> None:
    source = (ROOT / "evals/fleet/autocontinue_canary_hosted_runtime.py").read_text()
    run = source.split("def run_hosted(", 1)[1].split("def main()", 1)[0]
    lease = run.index("acquire_endpoint_lease")
    route = run.index("runtime_route = observe_live_route")
    global_claim = run.index("global_claim = canary.claim_global_cell")
    task_claim = run.index("result = canary._run_cell")
    assert lease < route < global_claim < task_claim


@pytest.mark.parametrize("case", ["team", "malformed", "missing", "duplicate", "transport"])
def test_runtime_route_observer_failures_leave_all_claims_and_execution_untouched(
    case: str, monkeypatch, tmp_path: Path
) -> None:
    plan = canary.load_object(Q_PLAN)
    launch_receipt = _route()
    entered = False
    forbidden_calls = []
    monkeypatch.setattr(runtime, "validate_hosted_release", lambda *_: None)
    monkeypatch.setenv("FLEET_API_KEY", "redacted")
    monkeypatch.setenv("JOB_UID", "11111111-1111-4111-8111-111111111111")
    monkeypatch.setenv("POD_UID", "22222222-2222-4222-8222-222222222222")

    @contextmanager
    def lease(**_kwargs):
        nonlocal entered
        entered = True
        yield

    monkeypatch.setattr(canary.endpoint_lease, "acquire_endpoint_lease", lease)
    real_observe = runtime.observe_live_route

    def failing_observe(*args, **kwargs):
        kwargs["opener"] = _bad_opener(case)
        return real_observe(*args, **kwargs)

    monkeypatch.setattr(runtime, "observe_live_route", failing_observe)

    def forbidden(*_args, **_kwargs):
        forbidden_calls.append(True)
        raise AssertionError("claim or model execution must not run")

    monkeypatch.setattr(canary, "claim_global_cell", forbidden)
    monkeypatch.setattr(canary, "_run_cell", forbidden)
    with pytest.raises((health.GateError, ValueError)):
        runtime.run_hosted(
            plan,
            {},
            launch_receipt,
            tmp_path / "out",
            tmp_path / "proxy.py",
            ROOT,
            "16e25f1f127dcf76018a223ad8ac57e2117f83d7",
        )
    assert entered is True
    assert forbidden_calls == []
    assert not (tmp_path / "out").exists()


def _release(plan: dict, root: Path = ROOT) -> dict:
    bundle = phase_c.BUNDLES[plan["shard_key"]]
    successor = bundle.get("successor") is True
    compatibility_sha = (
        bundle["compatibility_sha"]
        if successor
        else canary._compatibility(root)["receipt_sha256"]
    )
    evidence = {
        "fresh_inventory_receipt_sha256": canary.TASK_INVENTORY_SHA,
        "fresh_inventory_execution_sha256": canary.TASK_INVENTORY_EXECUTION_SHA,
        "fresh_duplicate_inventory_receipt_path": bundle["duplicate_path"],
        "fresh_duplicate_inventory_receipt_sha256": bundle["duplicate_sha"],
        "preflight_authorization_receipt_path": bundle["preauth_path"],
        "preflight_authorization_receipt_sha256": bundle["preauth_sha"],
        "preflight_receipt_path": bundle["preflight_path"],
        "preflight_receipt_sha256": bundle["preflight_sha"],
        "preflight_post_exit_receipt_path": bundle["post_exit_path"],
        "preflight_post_exit_receipt_sha256": bundle["post_exit_sha"],
        "dedicated_parity_receipt_sha256": canary.DEDICATED_PARITY_SHA,
        "dedicated_parity_role": "historical_compatibility_only_not_live_route_authority",
        "hosted_health_receipt_sha256": canary.HOSTED_HEALTH_SHA,
        "shared_pvc_flock_receipt_sha256": canary.FLOCK_GATE_SHA,
        "controller_compatibility_receipt_sha256": compatibility_sha,
    }
    if not successor:
        evidence.update(
            {
                "hosted_only_route_receipt_path": phase_c.ROUTE_PATH,
                "hosted_only_route_receipt_sha256": phase_c.ROUTE_SHA,
            }
        )
    release = {
        "schema_version": canary.RELEASE_SCHEMA,
        "append_only": True,
        "status": "RELEASED",
        "released_at_utc": "2026-09-04T21:45:00Z",
        "campaign_sha256": canary.CAMPAIGN_SHA,
        "plan_sha256": plan["plan_sha256"],
        "cell": {
            "source_rank": plan["tasks"][0]["source_rank"],
            "attempt": 1,
            "task_version_id": plan["tasks"][0]["task"]["version_id"],
        },
        "evidence": evidence,
        "implementation": {
            "package_commit": "16e25f1f127dcf76018a223ad8ac57e2117f83d7",
            "plan_sha256": plan["plan_sha256"],
            "controller_sha256": canary._sha(
                root / "evals/fleet/autocontinue_canary_controller.py"
            ),
            "frozen_controller_sha256": canary._sha(
                root / "evals/fleet/hosted_sweep_controller.py"
            ),
            "phase_c_validator_sha256": canary._sha(
                root / "evals/fleet/autocontinue_canary_hosted_release.py"
            ),
            "hosted_runtime_sha256": canary._sha(
                root / "evals/fleet/autocontinue_canary_hosted_runtime.py"
            ),
            "hosted_health_sha256": canary._sha(root / "evals/fleet/autocontinue_hosted_health.py"),
            "self_hosted_sha256": canary._sha(root / "evals/fleet/self_hosted.py"),
            "runner_sha256": canary._sha(root / "evals/fleet/opencode_train_sweep_runner.py"),
            "endpoint_lease_sha256": canary._sha(root / "evals/fleet/endpoint_lease.py"),
            "fixed_proxy_sha256": canary._sha(root / "evals/fleet/fixed_proxy.py"),
            "dockerfile_sha256": canary._sha(root / "evals/fleet/Dockerfile.opencode"),
            "campaign_file_sha256": canary._sha(
                root
                / "evals/fleet/configs/q38-glm53-opencode-autocontinue-primary-campaign-v1.json"
            ),
            "compatibility_file_sha256": canary._sha(
                root
                / "docs/evidence/qwen38-study/"
                / (
                    "2026-09-04-opencode-autocontinue-canary-controller-compatibility-v3.json"
                    if successor
                    else "2026-09-04-opencode-autocontinue-canary-controller-compatibility-v2.json"
                )
            ),
            "preflight_manifest_sha256": canary._sha(
                root
                / "evals/fleet/cluster/"
                / (
                    "opencode-autocontinue-canary-preflights-v3.yaml"
                    if successor
                    else "opencode-autocontinue-canary-preflights-v2.yaml"
                )
            ),
            "scored_manifest_sha256": canary._sha(
                root
                / "evals/fleet/cluster/"
                / (
                    "opencode-autocontinue-canary-successor-scored-v4.yaml"
                    if successor
                    else "opencode-autocontinue-canary-scored-v2.yaml"
                )
            ),
            "run_script_sha256": canary._sha(
                root / "evals/fleet/scripts/run_opencode_autocontinue_canary.sh"
            ),
            "launcher_sha256": canary._sha(
                root
                / "evals/fleet/scripts/"
                / (
                    "submit_opencode_autocontinue_canaries_v2.sh"
                    if successor
                    else "submit_opencode_autocontinue_canaries_v1.sh"
                )
            ),
        },
        "authorization": {
            "launch_authorized": True,
            "create_once": True,
            "required_priority_class": "fleet-train-high",
            "author": "/root",
            "authorized_at_utc": "2026-09-04T21:45:00Z",
            "statement": "future exact hosted-only authorization",
        },
        "terminal_contract": {
            "downward_job_uid_required": True,
            "downward_pod_uid_required": True,
            "post_exit_k8s_observer_required": True,
            "terminal_schema_version": canary.TERMINAL_SCHEMA,
            "post_exit_schema_version": canary.POST_EXIT_SCHEMA,
        },
        "hosted_only_route": {
            "serving_route": "HOSTED_ONLY",
            "dedicated_serving_state": "USER_STOPPED_UNAVAILABLE",
            "dedicated_recreation_authorized": False,
            "fresh_authenticated_models_get_required_before_scored_job_create": True,
            "fresh_authenticated_models_get_required_before_global_cell_claim": True,
            "required_served_ids": ["qwen3.8-27b", "glm-5.3"],
            "launcher_precreate_maximum_route_age_seconds": 120,
            "runtime_does_not_rely_on_launcher_receipt_freshness": True,
            "runtime_maximum_route_age_seconds": 30,
        },
        "privacy": phase_c.PRIVACY,
    }
    if successor:
        release["implementation"]["manifest_authorization_sha256"] = canary._sha(
            root / "evals/fleet/autocontinue_successor_manifest_authorization.py"
        )
    release["receipt_sha256"] = canary.digest_without(release, "receipt_sha256")
    return release


@pytest.mark.parametrize(
    "plan_path",
    [
        Q_PLAN,
        ROOT / "evals/fleet/configs/glm53-opencode-autocontinue-canary1-v1.json",
    ],
)
def test_hosted_release_real_chain_is_exact(plan_path: Path) -> None:
    plan = canary.load_object(plan_path)
    release = _release(plan)
    runtime.validate_hosted_release(release, plan, ROOT, "16e25f1f127dcf76018a223ad8ac57e2117f83d7")
    changed = copy.deepcopy(release)
    changed["hosted_only_route"]["dedicated_recreation_authorized"] = True
    changed["receipt_sha256"] = canary.digest_without(changed, "receipt_sha256")
    with pytest.raises(ValueError, match="hosted-only"):
        runtime.validate_hosted_release(
            changed, plan, ROOT, "16e25f1f127dcf76018a223ad8ac57e2117f83d7"
        )


@pytest.mark.parametrize("plan_path", [Q_SUCCESSOR_PLAN, G_SUCCESSOR_PLAN])
def test_successor_hosted_release_real_chain_is_executable(plan_path: Path) -> None:
    plan = canary.load_object(plan_path)
    release = _release(plan)
    runtime.validate_hosted_release(
        release, plan, ROOT, "16e25f1f127dcf76018a223ad8ac57e2117f83d7"
    )


@pytest.mark.parametrize("plan_path", [Q_SUCCESSOR_PLAN, G_SUCCESSOR_PLAN])
def test_rendered_successor_workspace_validates_release_and_requires_canonical_paths(
    plan_path: Path, tmp_path: Path
) -> None:
    plan = canary.load_object(plan_path)
    bundle = phase_c.BUNDLES[plan["shard_key"]]
    reconstructed = tmp_path / plan["shard_key"] / "workspace/cyber-post-train"
    common = {
        "evals/fleet/autocontinue_canary_controller.py": (
            ROOT / "evals/fleet/autocontinue_canary_controller.py"
        ),
        "evals/fleet/hosted_sweep_controller.py": (
            ROOT / "evals/fleet/hosted_sweep_controller.py"
        ),
        "evals/fleet/autocontinue_canary_hosted_release.py": (
            ROOT / "evals/fleet/autocontinue_canary_hosted_release.py"
        ),
        "evals/fleet/autocontinue_canary_hosted_runtime.py": (
            ROOT / "evals/fleet/autocontinue_canary_hosted_runtime.py"
        ),
        "evals/fleet/autocontinue_hosted_health.py": (
            ROOT / "evals/fleet/autocontinue_hosted_health.py"
        ),
        "evals/fleet/autocontinue_successor_manifest_authorization.py": (
            ROOT / "evals/fleet/autocontinue_successor_manifest_authorization.py"
        ),
        "evals/fleet/self_hosted.py": ROOT / "evals/fleet/self_hosted.py",
        "evals/fleet/opencode_train_sweep_runner.py": (
            ROOT / "evals/fleet/opencode_train_sweep_runner.py"
        ),
        "evals/fleet/endpoint_lease.py": ROOT / "evals/fleet/endpoint_lease.py",
        "evals/fleet/fixed_proxy.py": ROOT / "evals/fleet/fixed_proxy.py",
        "evals/fleet/Dockerfile.opencode": ROOT / "evals/fleet/Dockerfile.opencode",
        "evals/fleet/configs/q38-glm53-opencode-autocontinue-primary-campaign-v1.json": (
            ROOT
            / "evals/fleet/configs/q38-glm53-opencode-autocontinue-primary-campaign-v1.json"
        ),
        "evals/fleet/cluster/opencode-autocontinue-canary-preflights-v3.yaml": (
            ROOT / "evals/fleet/cluster/opencode-autocontinue-canary-preflights-v3.yaml"
        ),
        "evals/fleet/cluster/opencode-autocontinue-canary-successor-scored-v4.yaml": (
            ROOT
            / "evals/fleet/cluster/opencode-autocontinue-canary-successor-scored-v4.yaml"
        ),
        "evals/fleet/scripts/run_opencode_autocontinue_canary.sh": (
            ROOT / "evals/fleet/scripts/run_opencode_autocontinue_canary.sh"
        ),
        "evals/fleet/scripts/submit_opencode_autocontinue_canaries_v2.sh": (
            ROOT / "evals/fleet/scripts/submit_opencode_autocontinue_canaries_v2.sh"
        ),
        (
            "docs/evidence/qwen38-study/"
            "2026-09-04-opencode-autocontinue-canary-controller-compatibility-v3.json"
        ): (
            ROOT
            / "docs/evidence/qwen38-study/"
            "2026-09-04-opencode-autocontinue-canary-controller-compatibility-v3.json"
        ),
    }
    for evidence_key in ("preauth_path", "preflight_path", "post_exit_path", "duplicate_path"):
        relative = bundle[evidence_key]
        common[relative] = ROOT / relative
    for relative, source in common.items():
        destination = reconstructed / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    release = _release(plan, reconstructed)
    runtime.validate_hosted_release(
        release,
        plan,
        reconstructed,
        "16e25f1f127dcf76018a223ad8ac57e2117f83d7",
    )

    scored = (
        reconstructed
        / "evals/fleet/cluster/opencode-autocontinue-canary-successor-scored-v4.yaml"
    )
    scored.rename(scored.with_name("opencode-autocontinue-canary-scored-v3.yaml"))
    with pytest.raises(FileNotFoundError):
        runtime.validate_hosted_release(
            release,
            plan,
            reconstructed,
            "16e25f1f127dcf76018a223ad8ac57e2117f83d7",
        )
    scored.with_name("opencode-autocontinue-canary-scored-v3.yaml").rename(scored)

    manifest_authorization = (
        reconstructed / "evals/fleet/autocontinue_successor_manifest_authorization.py"
    )
    manifest_authorization.rename(
        manifest_authorization.with_name("scored_manifest_authorization.py")
    )
    with pytest.raises(FileNotFoundError):
        runtime.validate_hosted_release(
            release,
            plan,
            reconstructed,
            "16e25f1f127dcf76018a223ad8ac57e2117f83d7",
        )


def test_hosted_release_separates_preflight_and_final_package_commits() -> None:
    plan = canary.load_object(Q_PLAN)
    release = _release(plan)
    # The immutable Phase-C preauth remains bound to Phase-A3, while the final
    # implementation package is caller-bound independently.
    runtime.validate_hosted_release(release, plan, ROOT, "16e25f1f127dcf76018a223ad8ac57e2117f83d7")
    with pytest.raises(ValueError, match="hosted-only"):
        runtime.validate_hosted_release(
            release, plan, ROOT, "9c93095f60d627c239fe2d01f35efd918f328fe6"
        )


def test_hosted_release_rejects_wrong_phase_a3_preflight_commit(monkeypatch) -> None:
    plan = canary.load_object(Q_PLAN)
    release = _release(plan)
    monkeypatch.setattr(phase_c, "PACKAGE_COMMIT", "16e25f1f127dcf76018a223ad8ac57e2117f83d7")
    with pytest.raises(ValueError, match="preflight authorization"):
        runtime.validate_hosted_release(
            release, plan, ROOT, "16e25f1f127dcf76018a223ad8ac57e2117f83d7"
        )


def test_hosted_release_rejects_relocated_phase_c_reconciliation(monkeypatch) -> None:
    plan = canary.load_object(Q_PLAN)
    release = _release(plan)
    bundle = phase_c.BUNDLES[plan["shard_key"]]
    paths = {
        bundle["preauth_path"]: canary.load_object(ROOT / bundle["preauth_path"]),
        bundle["preflight_path"]: canary.load_object(ROOT / bundle["preflight_path"]),
        bundle["post_exit_path"]: canary.load_object(ROOT / bundle["post_exit_path"]),
        bundle["duplicate_path"]: canary.load_object(ROOT / bundle["duplicate_path"]),
        phase_c.ROUTE_PATH: canary.load_object(ROOT / phase_c.ROUTE_PATH),
    }
    changed = copy.deepcopy(paths[bundle["post_exit_path"]])
    changed["scored_job_created"] = changed["reconciliation"].pop("scored_job_created")
    changed["scored_sfs_root_absent"] = changed["reconciliation"].pop("scored_sfs_root_absent")
    changed["receipt_sha256"] = canary.digest_without(changed, "receipt_sha256")

    def load(_root: Path, path: str, _sha: str):
        return changed if path == bundle["post_exit_path"] else paths[path]

    monkeypatch.setattr(phase_c, "_load_exact", load)
    with pytest.raises(ValueError, match="preflight bundle"):
        runtime.validate_hosted_release(
            release, plan, ROOT, "16e25f1f127dcf76018a223ad8ac57e2117f83d7"
        )
