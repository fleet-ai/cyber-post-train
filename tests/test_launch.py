"""Boundary tests for the compact 96k launcher; no cluster or private data."""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from functools import lru_cache
from pathlib import Path
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq

from training import launch
from training.runtime import MODEL, TOKENIZER_FILES


CONFIG = Path(__file__).resolve().parents[1] / "configs/runs/qwen38-96k-mechanics-v1.json"
FULL_CONFIG = Path(__file__).resolve().parents[1] / "configs/runs/qwen38-96k-full-v1.json"


@lru_cache
def mechanics_parquet() -> bytes:
    rows = [{"source_session_id": f"session-{i}", "window_id": f"window-{i}",
             "input_ids": [1] * 90_000, "loss_mask": [0] * 89_999 + [1],
             "token_count": 90_000, "target_token_count": 1, "task_key": "train-family"} for i in range(8)]
    sink = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_pylist(rows), sink)
    return sink.getvalue().to_pybytes()


def fake_mechanics_manifest() -> dict:
    digest = lambda char: "sha256:" + char * 64
    value = {"schema": "cyber_dense_sft_corpus_v1", "algorithm": launch.MECHANICS_METHOD,
             "trainer_ready": False, "validation_mode": "task_outcomes_only",
             "split_sha256": digest("a"), "materialization": {"request_sha256": digest("b")},
             "builder_sha256": {"message_aligned_teacher_corpus.py": launch.MECHANICS_BUILDER},
             "tokenizer": {"repo": MODEL[0], "revision": MODEL[1],
                           "files": [{"path": p, "sha256": d} for p, d in TOKENIZER_FILES.items()],
                           "chat_template_sha256": TOKENIZER_FILES["chat_template.jinja"]},
             "diagnostic": {"schema": "qwen38_provisional96_mechanics_v1", "purpose": "one_step_mechanics_only",
                            "source_receipt_file_sha256": launch.PROVISIONAL_SOURCE,
                            "parent_method_sha256": digest("c"), "parent_train_sha256": digest("d")},
             "files": {"train": {"path": "train.parquet", "sha256": "sha256:" + launch.sha(mechanics_parquet()),
                                 "rows": 8, "task_keys": ["train-family"], "format": "pretokenized_assistant_segments_v1",
                                 "supervised_tokens": 8, "source_sessions": 8, "assistant_responses": 8,
                                 "source_total_assistant_responses": 8, "excluded_assistant_responses": 0}}}
    value["sha256"] = "sha256:" + launch.sha(launch.canonical(value))
    return value


def fake_full_manifest(*, tokens: int = 20_000_001) -> dict:
    digest = lambda char: "sha256:" + char * 64
    value = {"schema": "cyber_dense_sft_corpus_v1", "algorithm": "old_original_anchor_not_goal_corrected",
             "validation_mode": "teacher_cross_entropy", "split_sha256": digest("d"),
             "materialization": {"request_sha256": digest("1"), "family_roster_sha256": digest("2")},
             "tokenizer": {"repo": MODEL[0], "revision": MODEL[1], "files": [
                 {"path": p, "sha256": d} for p, d in TOKENIZER_FILES.items()]},
             "composition": {"schema": "qwen38_dense_train_contiguous_dev_v1",
                             "corpus_root": launch.FULL_DATA_ROOT, **{key: digest(char) for key, char in (
                                 ("dense_manifest_sha256", "3"), ("dense_receipt_sha256", "4"),
                                 ("dev_manifest_sha256", "5"), ("dev_source_receipt_sha256", "6"),
                                 ("family_roster_sha256", "2"))}},
             "files": {"train": {"path": "train.parquet", "sha256": digest("a"), "rows": 813,
                                 "task_keys": ["train-family"], "format": "pretokenized_assistant_segments_v1",
                                 "supervised_tokens": tokens, "source_sessions": 400, "assistant_responses": 1600,
                                 "source_total_assistant_responses": 1625, "excluded_assistant_responses": 25},
                       "dev": {"path": "dev.parquet", "sha256": digest("b"), "rows": 1,
                               "task_keys": ["dev-family"], "format": "chat_messages_last_assistant_v2"}}}
    value["sha256"] = "sha256:" + launch.sha(launch.canonical(value))
    return value


def fake_preview(request: dict) -> dict:
    resources = request["resources"]
    c = {"image": request["image"],
         "resources": {kind: {"cpu": resources[f"cpu_{suffix}"],
                              "memory": resources[f"memory_{suffix}"],
                              "nvidia.com/gpu": request["gpus_per_worker"]}
                       for kind, suffix in (("requests", "request"), ("limits", "limit"))},
         "env": [{"name": "RUN_DIR", "value": request["run_dir"]}] +
                [{"name": k, "value": v} for k, v in request["env"].items()],
         "envFrom": [{"secretRef": {"name": name, "optional": False}} for name in request["secrets"]]}
    obj = {"kind": "RayJob", "metadata": {"namespace": "fleet-train-jobs",
           "labels": {"kueue.x-k8s.io/queue-name": "training-lq", "kueue.x-k8s.io/priority-class": "q1",
                      "fleet.ai/requeue-if-preempted": "false"},
           "annotations": {"fleet.ai/run-dir": request["run_dir"], "fleet.ai/failure-alerts": "off"}},
           "spec": {"suspend": True, "shutdownAfterJobFinishes": True, "entrypoint": request["command"],
                    "rayClusterSpec": {"headGroupSpec": {"template": {"spec": {
                        "priorityClassName": "c1", "containers": [c]}}}, "workerGroupSpecs": []}}}
    return {"manifest_yaml": json.dumps(obj), "errors": [], "warnings": []}


def native_preflight(receipt: dict) -> dict:
    return {"status": "passed", "gpus": 0, "request_sha256": receipt["request_sha256"],
            "plan_sha256": receipt["plan_sha256"],
            "checked": ["native_sources", "model_files", "dataset_files", "native_config",
                        "native_forward_backward_signature", "native_train_only_loader",
                        "tokenization", "target_accounting"],
            "counts": {"train": {"rows": 8, "supervised_tokens": 8}}}


class LaunchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = json.loads(CONFIG.read_text())
        self.config["data"]["manifest"] = "qwen38-96k-mechanics-v1.manifest.json"
        (self.root / "train.parquet").write_bytes(mechanics_parquet())
        (self.root / "qwen38-96k-mechanics-v1.manifest.json").write_text(json.dumps(fake_mechanics_manifest()))
        self.config_path = self.root / "run.json"
        self.witness = mock.patch.object(launch, "_require_subset_witness")
        self.witness.start(); self.addCleanup(self.witness.stop)

    def prepare(self) -> tuple[Path, dict]:
        self.config_path.write_text(json.dumps(self.config))
        dest = self.root / "prepared"
        launch.prepare(self.config_path, dest)
        return dest, json.loads((dest / "request.json").read_text())

    def test_prepares_exact_historical_runtime_and_immutable_request(self) -> None:
        dest, request = self.prepare()
        plan, rebuilt, receipt = launch.prepared(dest)
        self.assertEqual(request, rebuilt)
        self.assertEqual(plan["recipe"]["max_steps"], 1)
        self.assertEqual(plan["recipe"]["max_length"], 98304)
        self.assertEqual(plan["validation_mode"], "task_outcomes_only")
        self.assertEqual(set(plan["datasets"]), {"train"})
        self.assertEqual(plan["recipe"]["checkpoint_interval"], 1)
        self.assertEqual(receipt["purpose"], "one_step_mechanics_only")
        self.assertEqual(request["priority_class"], "c1")
        self.assertIs(request["failureAlerts"], False)
        self.assertEqual(receipt["historical_commit"], launch.COMMIT)
        self.assertEqual(launch._legacy(
            "validate_preview", {"request": request, "preview": fake_preview(request)}
        )["gpus"], 8)

    def test_real_prepare_rejects_unproven_parent_membership(self) -> None:
        self.witness.stop()
        self.config_path.write_text(json.dumps(self.config))
        with self.assertRaisesRegex(ValueError, "TRAIN subset membership"):
            launch.prepare(self.config_path, self.root / "unproven")

    def test_rendered_root_alert_and_priority_drift_fail_closed(self) -> None:
        _dest, request = self.prepare()
        for field in ("failure-alerts", "priority-class"):
            with self.subTest(field=field):
                preview = fake_preview(request)
                obj = json.loads(preview["manifest_yaml"])
                if field == "failure-alerts":
                    del obj["metadata"]["annotations"]["fleet.ai/failure-alerts"]
                else:
                    obj["metadata"]["labels"]["kueue.x-k8s.io/priority-class"] = "q0"
                preview["manifest_yaml"] = json.dumps(obj)
                with self.assertRaisesRegex(ValueError, "validate_preview gate rejected"):
                    launch._legacy("validate_preview", {"request": request, "preview": preview})

    def test_cpu_gate_must_bind_train_only_and_exact_request(self) -> None:
        dest, _request = self.prepare()
        _plan, _request, receipt = launch.prepared(dest)
        result = native_preflight(receipt)
        self.assertTrue(launch._preflight_matches(result, receipt))
        result["counts"]["dev"] = {"rows": 1, "supervised_tokens": 1}
        self.assertFalse(launch._preflight_matches(result, receipt))
        result["counts"].pop("dev")
        result["request_sha256"] = "0" * 64
        self.assertFalse(launch._preflight_matches(result, receipt))

    def test_config_and_prepared_tampering_fail_closed(self) -> None:
        self.config["cluster"]["priority"] = "c0"
        self.config_path.write_text(json.dumps(self.config))
        with self.assertRaisesRegex(ValueError, "c1 recipe differs"):
            launch.prepare(self.config_path, self.root / "bad")
        self.config["cluster"]["priority"] = "c1"
        dest, request = self.prepare()
        request["failureAlerts"] = True
        (dest / "request.json").write_text(json.dumps(request))
        with self.assertRaisesRegex(ValueError, "prepared binding changed"):
            launch.prepared(dest)

    def test_mechanics_rejects_extra_rows_and_accepted_science_claim(self) -> None:
        manifest = fake_mechanics_manifest()
        manifest["files"]["train"]["rows"] = 9
        manifest.pop("sha256")
        manifest["sha256"] = "sha256:" + launch.sha(launch.canonical(manifest))
        (self.root / "qwen38-96k-mechanics-v1.manifest.json").write_text(json.dumps(manifest))
        self.config_path.write_text(json.dumps(self.config))
        with self.assertRaisesRegex(ValueError, "eight-row provisional"):
            launch.prepare(self.config_path, self.root / "too-many")
        manifest = fake_mechanics_manifest()
        manifest["trainer_ready"] = True
        manifest.pop("sha256")
        manifest["sha256"] = "sha256:" + launch.sha(launch.canonical(manifest))
        (self.root / "qwen38-96k-mechanics-v1.manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "eight-row provisional"):
            launch.prepare(self.config_path, self.root / "false-science")

    def test_cpu_job_is_exact_zero_gpu_c1_q1_and_root_alerts_off(self) -> None:
        dest, request = self.prepare()
        job = launch.cpu_job(dest)
        self.assertEqual(job["metadata"]["annotations"]["fleet.ai/failure-alerts"], "off")
        self.assertEqual(job["metadata"]["labels"]["kueue.x-k8s.io/priority-class"], "q1")
        spec = job["spec"]["template"]["spec"]
        self.assertEqual((spec["priorityClassName"], spec["priority"]), ("c1", 10000))
        self.assertEqual(spec["containers"][0]["image"], request["image"])
        self.assertNotIn("nvidia.com/gpu", json.dumps(spec))
        self.assertTrue(spec["volumes"][0]["persistentVolumeClaim"]["readOnly"])
        self.assertTrue(spec["containers"][0]["volumeMounts"][0]["readOnly"])
        self.assertLess(max(len(e["value"]) for e in spec["containers"][0]["env"]), 131072)
        launch._check_cpu_render(job, job)
        for field, error in (("alert", "root alert"), ("priority", "c1/q1")):
            bad = json.loads(json.dumps(job))
            if field == "alert":
                del bad["metadata"]["annotations"]["fleet.ai/failure-alerts"]
            else:
                bad["spec"]["template"]["spec"]["priorityClassName"] = "c0"
            with self.assertRaisesRegex(ValueError, error):
                launch._check_cpu_render(job, bad)

    def test_cpu_preview_is_server_checked_and_create_is_not_implicit(self) -> None:
        dest, _ = self.prepare()
        job = launch.cpu_job(dest)
        pvc = {"metadata": {"uid": "34cb6b11-8766-4294-9f9e-332064ea17d5"},
               "status": {"phase": "Bound"}}
        calls = []
        def kubectl(_context, args, payload=None, **_kwargs):
            calls.append(args)
            return pvc if args[2:4] == ["get", "pvc"] else job
        with mock.patch.object(launch, "_kubectl", side_effect=kubectl):
            proof = launch.cpu_preview(dest, launch.PROD_CONTEXT)
        self.assertEqual((proof["status"], len(calls)), ("previewed_not_created", 2))
        self.assertTrue(any("--dry-run=server" in c for c in calls))
        self.assertFalse((dest / "CPU-01-INTENT.json").exists())
        with mock.patch.object(launch, "cpu_preview", return_value=proof):
            with self.assertRaisesRegex(ValueError, "reviewed Job"):
                launch.cpu_create(dest, launch.PROD_CONTEXT, "0" * 64)
        self.assertFalse((dest / "CPU-01-INTENT.json").exists())

    def test_gpu_submit_requires_fresh_uid_bound_cpu_receipt_before_network(self) -> None:
        dest, _ = self.prepare()
        receipt = json.loads((dest / "PREPARED.json").read_text())
        native = native_preflight(receipt)
        for issue, job in (("UID-bound", None), ("fresh UID-bound", {
            "name": "pre", "uid": "u", "pod_uid": "p", "context": launch.PROD_CONTEXT,
            "output_absent": True, "observed_at_unix": 0})):
            if job is not None:
                native["cpu_job"] = job
            (dest / "PREFLIGHT.json").write_text(json.dumps(native))
            with mock.patch.object(launch, "_lease") as lease:
                with self.assertRaisesRegex(ValueError, issue):
                    launch.submit(dest, "a" * 64)
                lease.assert_not_called()

    def test_capacity_counts_allocated_pod_not_its_running_rayjob_as_queued(self) -> None:
        cluster = "chris-q38-run-abcd-cluster"
        pod = {"metadata": {"name": "chris-q38-run-abcd-head-xyz", "labels": {
            "fleet.ai/run-name": "chris-q38-run", "ray.io/cluster": cluster}},
            "status": {"phase": "Running"}, "spec": {"nodeName": "node-1", "containers": [
                {"resources": {"requests": {"nvidia.com/gpu": "8"}}}]}}
        cpu = {"metadata": {"name": "chris-q38-cpu-pre-a01", "labels": {
            "cyber-post-train.fleet.ai/owner": "chris"}}, "status": {"phase": "Running"},
            "spec": {"nodeName": "node-cpu", "containers": [{"resources": {"requests": {"cpu": "4"}}}]}}
        running = {"metadata": {"name": "chris-q38-run-abcd"},
                   "status": {"jobStatus": "RUNNING", "rayClusterName": cluster}}
        queued = {"metadata": {"name": "chris-q38-next-efgh"}, "status": {"jobStatus": "PENDING"}}

        def kubectl(context, args, *_a, **_k):
            if context == launch.DEV_CONTEXT:
                return {"items": []}
            return {"items": {"pods": [pod, cpu], "rayjobs.ray.io": [running, queued]}.get(args[1], [])}

        with mock.patch.object(launch, "_kubectl", side_effect=kubectl):
            census = launch.capacity()
        self.assertEqual(census, {"active_nodes": 1, "active_gpus": 8,
                                  "queued_jobs": 1, "owned_active_gpu_pods": 1})

    def test_lease_release_must_match_original_holder(self) -> None:
        existing = {"metadata": {"uid": "lease-uid", "resourceVersion": "3"},
                    "spec": {"holderIdentity": "another", "renewTime": "2026-09-25T00:00:00Z",
                             "leaseDurationSeconds": 900}}
        with mock.patch.object(launch, "_kubectl", return_value=existing) as kubectl:
            with self.assertRaisesRegex(ValueError, "identity changed"):
                launch._lease(None, expected_uid="lease-uid", expected_holder="mine")
            kubectl.assert_called_once()

    def test_gpu_review_drift_blocks_post_under_lease(self) -> None:
        dest, _ = self.prepare()
        receipt = json.loads((dest / "PREPARED.json").read_text())
        native = native_preflight(receipt)
        native["cpu_job"] = {"uid": "job", "pod_uid": "pod", "output_absent": True,
                             "context": launch.PROD_CONTEXT, "observed_at_unix": time.time()}
        (dest / "PREFLIGHT.json").write_text(json.dumps(native))
        historical = launch._legacy

        def bridge(mode, value, **kwargs):
            if mode == "preview":
                return {"manifest_sha256": "b" * 64}
            if mode == "submit":
                self.fail("POST path reached despite reviewed-preview drift")
            return historical(mode, value, **kwargs)

        with (mock.patch.object(launch, "_lease", return_value={"metadata": {"uid": "lease"}})
              as lease, mock.patch.object(launch, "capacity", return_value={"active_nodes": 1}),
              mock.patch.object(launch, "_cpu_observation", return_value=native),
              mock.patch.object(launch, "_legacy", side_effect=bridge)):
            with self.assertRaisesRegex(ValueError, "reviewed GPU"):
                launch.submit(dest, "a" * 64)
        self.assertEqual(lease.call_count, 2)  # acquire and release, no POST intent
        self.assertFalse((dest / "GPU-POST-INTENT.jsonl").exists())


class FullLaunchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = json.loads(FULL_CONFIG.read_text())
        self.config["data"]["manifest"] = "qwen38-96k-full-v1.manifest.json"
        self.manifest = fake_full_manifest()
        self.manifest_path = self.root / self.config["data"]["manifest"]
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.config_path = self.root / "run.json"

    def prepare(self) -> Path:
        self.config_path.write_text(json.dumps(self.config))
        destination = self.root / "prepared"
        launch.prepare(self.config_path, destination)
        return destination

    def test_full_profile_is_not_submittable_with_old_anchor_method(self) -> None:
        with self.assertRaisesRegex(ValueError, "target-anchor producer/attestation"):
            self.prepare()
        self.assertFalse((self.root / "prepared").exists())

    def test_full_profile_compiles_mixed_data_and_keeps_every_checkpoint_after_gate(self) -> None:
        # Only a synthetic test bypass: production's method gate always rejects
        # until the corrected source/attestation implementation is reviewed.
        with mock.patch.object(launch, "_require_goal_anchor"):
            destination = self.prepare()
            plan, request, receipt = launch.prepared(destination)
        self.assertEqual(receipt["schema"], "qwen38_96k_full_prepared_v1")
        self.assertEqual(plan["schema"], "cyber_sft_runtime_dense_v1")
        self.assertEqual((plan["datasets"]["train"]["format"], plan["datasets"]["dev"]["format"]),
                         ("pretokenized_assistant_segments_v1", "chat_messages_last_assistant_v2"))
        self.assertEqual((plan["recipe"]["max_steps"], plan["recipe"]["keep_checkpoints"],
                          receipt["planned_native_checkpoints"], receipt["checkpoint_retention_capacity"]),
                         (102, 102, 3, 102))
        self.assertEqual((plan["recipe"]["eval_interval"], plan["recipe"]["checkpoint_interval"]), (50, 50))
        self.assertGreaterEqual(receipt["supervised_tokens"], 20_000_000)
        self.assertEqual(request["run_dir"], launch.FULL_OUTPUT)
        self.assertEqual((request["priority_class"], request["failureAlerts"]), ("c1", False))
        self.assertEqual(launch._legacy(
            "validate_preview", {"request": request, "preview": fake_preview(request)}
        )["gpus"], 8)
        with self.assertRaisesRegex(ValueError, "target-anchor producer/attestation"):
            launch.prepared(destination)

    def test_full_profile_rejects_small_corpus_and_unsafe_priority(self) -> None:
        self.manifest = fake_full_manifest(tokens=19_999_999)
        self.manifest_path.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, "20|sealed dense"):
            self.prepare()
        self.config["cluster"]["priority"] = "c0"
        self.config_path.write_text(json.dumps(self.config))
        with self.assertRaisesRegex(ValueError, "c1 recipe"):
            launch.prepare(self.config_path, self.root / "unsafe")

    def test_full_profile_rejects_manifest_drift_and_family_overlap(self) -> None:
        self.manifest["files"]["train"]["rows"] += 1
        self.manifest_path.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, "sealed dense"):
            self.prepare()
        self.manifest = fake_full_manifest()
        self.manifest["files"]["dev"]["task_keys"] = ["train-family"]
        self.manifest.pop("sha256")
        self.manifest["sha256"] = "sha256:" + launch.sha(launch.canonical(self.manifest))
        self.manifest_path.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, "disjoint teacher-CE dev"):
            self.prepare()


if __name__ == "__main__":
    unittest.main()
