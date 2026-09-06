from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import pytest

from evals.fleet import glm53_dedicated_v33_launch_v1 as launch


class FakeKubernetes:
    def __init__(self, *, collision: tuple[str, str] | None = None) -> None:
        self.objects: dict[tuple[str, str], dict[str, Any]] = {}
        self.collision = collision
        self.token_reads = 0
        self.dry_runs = 0
        self.creates = 0
        self.inject_after_dry_run = False

    @staticmethod
    def _object(kind: str, name: str) -> dict[str, Any]:
        return {
            "apiVersion": "v1" if kind == "configmap" else "batch/v1",
            "kind": "ConfigMap" if kind == "configmap" else "Job",
            "metadata": {
                "name": name,
                "namespace": launch.NAMESPACE,
                "uid": str(uuid.uuid4()),
            },
        }

    def load_fleet_token(self) -> str:
        self.token_reads += 1
        return "test-token"

    def get(self, kind: str, name: str) -> dict[str, Any] | None:
        if self.collision == (kind, name):
            return self._object(kind, name)
        return self.objects.get((kind, name))

    def dry_run(self, _objects: dict[str, Any]) -> None:
        self.dry_runs += 1
        if self.inject_after_dry_run:
            kind, name = launch._identities()[0]  # noqa: SLF001
            self.objects[(kind, name)] = self._object(kind, name)

    def create(self, _objects: dict[str, Any]) -> None:
        self.creates += 1
        for kind, name in launch._identities():  # noqa: SLF001
            self.objects[(kind, name)] = self._object(kind, name)


def _patch_live_builders(monkeypatch: pytest.MonkeyPatch) -> None:
    reconciliation = {"receipt_sha256": "sha256:stale"}
    authorization = {"receipt_sha256": "sha256:authorization"}
    monkeypatch.setattr(launch.stale, "SystemBackend", lambda: object())
    monkeypatch.setattr(launch.stale, "build_reconciliation", lambda **_kwargs: reconciliation)
    monkeypatch.setattr(launch.stale, "validate_reconciliation", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        launch.live, "build_live_authorization", lambda **_kwargs: authorization
    )
    monkeypatch.setattr(launch.live, "SystemBackend", lambda: object())
    monkeypatch.setattr(launch.server, "validate_authorization", lambda _value: None)
    monkeypatch.setattr(launch.server, "payload", lambda: {"priority_class": "fleet-infra-quiet"})
    monkeypatch.setattr(launch.server, "request_sha256", lambda: "sha256:request")
    monkeypatch.setattr(
        launch.package,
        "render",
        lambda *_args: {
            "objects": {
                "apiVersion": "v1",
                "kind": "List",
                "items": [
                    {
                        "data": {
                            "package.json": json.dumps(
                                {"package_sha256": launch.FROZEN_PACKAGE_SHA256}
                            )
                        }
                    },
                    {},
                    {},
                ],
            }
        },
    )


def test_held_preview_uses_frozen_reviewed_package_and_no_live_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        launch.SystemKubernetesBackend,
        "__init__",
        lambda _self: pytest.fail("held preview must not construct a live backend"),
    )
    value = launch.build_held(Path(__file__).parents[1])
    assert value["status"] == "PASSED_HELD_NO_LAUNCH"
    assert value["package_commit"] == launch.FROZEN_PACKAGE_COMMIT
    assert value["package_sha256"] == launch.FROZEN_PACKAGE_SHA256
    assert value["kubernetes_mutation_calls"] == 0
    assert value["server_launch_authorized"] is False


def test_execute_builds_fresh_evidence_and_creates_controller_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_live_builders(monkeypatch)

    class StaleBackend:
        def list_runs(self) -> tuple[list[dict[str, Any]], int]:
            return [], 1

    kubernetes = FakeKubernetes()
    prior = os.environ.pop("FLEET_API_KEY", None)
    try:
        result = launch.execute(
            root=tmp_path,
            output=tmp_path / "SUBMITTED.json",
            kubernetes=kubernetes,
            stale_backend_factory=StaleBackend,
            live_backend_factory=lambda: object(),
            now=lambda: 123.0,
        )
    finally:
        if prior is not None:
            os.environ["FLEET_API_KEY"] = prior
    assert result["status"] == "CONTROLLER_SUBMITTED_CREATE_ONCE"
    assert result["stale_reconciliation_receipt_sha256"] == "sha256:stale"
    assert result["authorization_receipt_sha256"] == "sha256:authorization"
    assert kubernetes.token_reads == 1
    assert kubernetes.dry_runs == 1
    assert kubernetes.creates == 1
    assert "FLEET_API_KEY" not in os.environ
    persisted = json.loads((tmp_path / "SUBMITTED.json").read_text())
    assert persisted["receipt_sha256"] == result["receipt_sha256"]


@pytest.mark.parametrize("kind,name", launch._identities())  # noqa: SLF001
def test_any_exact_controller_identity_collision_blocks_before_secret_or_create(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, kind: str, name: str
) -> None:
    _patch_live_builders(monkeypatch)
    kubernetes = FakeKubernetes(collision=(kind, name))
    with pytest.raises(launch.LaunchError, match="v33_controller_identity_already_exists"):
        launch.execute(
            root=tmp_path,
            output=tmp_path / "SUBMITTED.json",
            kubernetes=kubernetes,
        )
    assert kubernetes.token_reads == 0
    assert kubernetes.creates == 0


def test_identity_created_during_server_dry_run_blocks_real_create(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_live_builders(monkeypatch)

    class StaleBackend:
        def list_runs(self) -> tuple[list[dict[str, Any]], int]:
            return [], 1

    kubernetes = FakeKubernetes()
    kubernetes.inject_after_dry_run = True
    with pytest.raises(launch.LaunchError, match="v33_controller_identity_already_exists"):
        launch.execute(
            root=tmp_path,
            output=tmp_path / "SUBMITTED.json",
            kubernetes=kubernetes,
            stale_backend_factory=StaleBackend,
            live_backend_factory=lambda: object(),
        )
    assert kubernetes.dry_runs == 1
    assert kubernetes.creates == 0
    assert not (tmp_path / "SUBMITTED.json").exists()


def test_existing_local_result_blocks_before_any_cluster_read(tmp_path: Path) -> None:
    output = tmp_path / "SUBMITTED.json"
    output.write_text("existing")
    kubernetes = FakeKubernetes()
    with pytest.raises(launch.LaunchError, match="v33_launch_result_already_exists"):
        launch.execute(root=tmp_path, output=output, kubernetes=kubernetes)
    assert kubernetes.token_reads == 0
    assert kubernetes.dry_runs == 0
    assert kubernetes.creates == 0
