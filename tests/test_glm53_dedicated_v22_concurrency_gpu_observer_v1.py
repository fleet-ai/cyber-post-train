import json
import subprocess
from pathlib import Path

import pytest

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import glm53_dedicated_v22_concurrency_gpu_observer_v1 as observer
from evals.fleet import glm53_dedicated_v22_concurrency_qualification_v1 as qualification

ROOT = Path(__file__).resolve().parents[1]


def _identity(suffix: str = "1") -> dict[str, str]:
    return {
        "server_rayjob_uid": f"11111111-1111-4111-8111-11111111111{suffix}",
        "server_head_pod_uid": f"22222222-2222-4222-8222-22222222222{suffix}",
        "qualifier_job_uid": f"33333333-3333-4333-8333-33333333333{suffix}",
        "qualifier_pod_uid": f"44444444-4444-4444-8444-44444444444{suffix}",
    }


def _sample(utilization: float = 50) -> dict[int, tuple[float, int, int]]:
    return {index: (utilization, 250_000 + index, 275_040) for index in range(8)}


def test_observer_receipt_binds_uids_all_devices_and_no_content() -> None:
    server = qualification.render(ROOT)["server"]
    value = observer.build_receipt(
        concurrency=2,
        server=server,
        identity_before=_identity(),
        identity_after=_identity(),
        samples=[_sample()],
    )
    assert value["identity"] == _identity()
    assert value["devices_seen"] == 8
    assert value["max_utilization_percent_by_device"] == [50] * 8
    assert value["prompts_responses_traces_tool_arguments_scores_read_or_persisted"] is False
    assert value["receipt_sha256"] == crypto.digest_without(value, "receipt_sha256")


def test_observer_rejects_uid_rotation_or_any_inactive_device() -> None:
    server = qualification.render(ROOT)["server"]
    with pytest.raises(observer.ObserverError, match="observer_identity_or_samples_invalid"):
        observer.build_receipt(
            concurrency=1,
            server=server,
            identity_before=_identity("1"),
            identity_after=_identity("2"),
            samples=[_sample()],
        )
    sample = _sample()
    sample[7] = (0, 250_007, 275_040)
    with pytest.raises(observer.ObserverError, match="not_all_expected_gpus_active_in_wave"):
        observer.build_receipt(
            concurrency=1,
            server=server,
            identity_before=_identity(),
            identity_after=_identity(),
            samples=[sample],
        )


def test_atomic_writer_is_uid_bound_stdin_o_excl_and_content_free() -> None:
    command = observer.atomic_write_command(
        "qualifier-pod", Path("/mnt/sfs/jobs/qualifier/gpu-observer/c2.json")
    )
    assert command[:8] == [
        "kubectl",
        "-n",
        observer.NAMESPACE,
        "exec",
        "-i",
        "qualifier-pod",
        "-c",
        "qualifier",
    ]
    assert "O_EXCL" in command[-2]
    assert command[-1].endswith("/gpu-observer/c2.json")


def test_plan_names_explicit_external_producer_and_next_wave_gate() -> None:
    value = qualification.render(ROOT)
    assert value["gpu_observer_execution"] == {
        "mode": "local_operator_uid_bound_kubectl",
        "module": "evals.fleet.glm53_dedicated_v22_concurrency_gpu_observer_v1",
        "phase_files_observed_by_existence_only": True,
        "server_rayjob_and_head_pod_uids_revalidated": True,
        "qualifier_job_and_pod_uids_revalidated": True,
        "atomic_receipt_write_via_stdin": True,
        "response_prompt_trace_tool_or_score_content_read": False,
    }
    assert value["fail_closed_ramp"]["gpu_observer_receipt_required_before_next_wave"] is True
    run = (
        ROOT / "evals/fleet/scripts/run_glm53_dedicated_v22_concurrency_qualification_v1.sh"
    ).read_text()
    assert 'mkdir -p /work/evals/fleet/configs "$QUALIFICATION_OUTPUT_ROOT/gpu-observer"' in run
    assert "--gpu-observer-root" in run


def test_parse_gpu_sample_requires_exact_eight_device_roster() -> None:
    raw = "\n".join(f"{index}, 75, {250000 + index}, 275040" for index in range(8)).encode()
    assert observer.parse_gpu_sample(raw)[7] == (75.0, 250007, 275040)
    with pytest.raises(observer.ObserverError, match="gpu_device_roster_invalid"):
        observer.parse_gpu_sample(raw.rsplit(b"\n", 1)[0])


def test_identity_resolver_binds_exact_server_and_qualifier_uids() -> None:
    server = qualification.render(ROOT)["server"]
    server_pod_uid = server["head_pod_uid"]
    objects = {
        "rayjob": {"metadata": {"uid": server["rayjob_uid"]}},
        "pods": {
            "items": [
                {
                    "metadata": {"name": "server-pod", "uid": server_pod_uid, "labels": {}},
                    "status": {
                        "phase": "Running",
                        "containerStatuses": [
                            {"name": "ray-head", "ready": True, "restartCount": 0}
                        ],
                    },
                },
                {
                    "metadata": {
                        "name": "qualifier-pod",
                        "uid": "44444444-4444-4444-8444-444444444444",
                        "labels": {"job-name": qualification.JOB_NAME},
                    },
                    "status": {
                        "phase": "Running",
                        "containerStatuses": [
                            {"name": "qualifier", "ready": True, "restartCount": 0}
                        ],
                    },
                },
            ]
        },
        "job": {
            "metadata": {"uid": "33333333-3333-4333-8333-333333333333"},
            "status": {"active": 1},
        },
    }

    def runner(args, **_kwargs):
        if "rayjob" in args:
            value = objects["rayjob"]
        elif "pods" in args:
            value = objects["pods"]
        else:
            value = objects["job"]
        return subprocess.CompletedProcess(args, 0, json.dumps(value).encode(), b"")

    identity, server_pod, qualifier_pod = observer.resolve_identity(ROOT, runner=runner)
    assert identity == {
        "server_rayjob_uid": server["rayjob_uid"],
        "server_head_pod_uid": server_pod_uid,
        "qualifier_job_uid": "33333333-3333-4333-8333-333333333333",
        "qualifier_pod_uid": "44444444-4444-4444-8444-444444444444",
    }
    assert (server_pod, qualifier_pod) == ("server-pod", "qualifier-pod")

    objects["rayjob"]["metadata"]["uid"] = "55555555-5555-4555-8555-555555555555"
    with pytest.raises(observer.ObserverError, match="kubernetes_identity_or_health_mismatch"):
        observer.resolve_identity(ROOT, runner=runner)
