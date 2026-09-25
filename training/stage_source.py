"""One-shot CPU staging of a sealed private teacher-source cache to SFS."""

import json
import os
import sys
import time
from pathlib import Path

from training.source import SourceError, digest, verify_hydration_cache

DEST = Path("/mnt/sfs/jobs/chris-q38-goal-teacher-source-v1")
NAME = "chris-q38-goal-teacher-source-stage-v1"
IMAGE = ("661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/skyrl-train@sha256:"
         "ba288751cd227c5be146d28f4a03237545d87d2cbd4c48464945b17fde566ff4")
SELECTION_SHA = "sha256:441f489c11e2f775bca81d98b7f578e0993a459375bf07f9e1828532ae3d91b6"
RECEIPT_FILE_SHA = "sha256:be6620b6dbf5f544b4df5b1017075299108849cc27eee0274708c7e5e572b365"
RECEIPT_SHA = "sha256:9bf8e76340a87ffa936571d8e24fb68e24e1764862e7937766454039a8fb68a2"
SESSIONS_SHA = "sha256:899cebe82457009c0747329b2a2b636ce7e5e5a7d9b449396f1f3ab1c99ccd8a"


def verify(root: Path, *, seal: bool = False) -> dict:
    root = Path(root)
    raw = root / "raw"
    if (root.is_symlink() or not root.is_dir() or root.stat().st_mode & 0o077
            or {p.name for p in root.iterdir()} != {"raw", "source-selection.private.jsonl"}
            or (root / "source-selection.private.jsonl").is_symlink()
            or (root / "source-selection.private.jsonl").stat().st_mode & 0o077
            or {p.name for p in raw.iterdir()} != {"sessions", "HYDRATE_REQUEST.json", "HYDRATED.json"}):
        raise SourceError("private stage directory is unsafe or incomplete")
    found = verify_hydration_cache(raw, root / "source-selection.private.jsonl", RECEIPT_FILE_SHA)
    receipt = json.loads((raw / "HYDRATED.json").read_text())
    if (found != {"selected_sessions": 2886, "selection_sha256": SELECTION_SHA,
                  "hydration_receipt_file_sha256": RECEIPT_FILE_SHA,
                  "sessions_sha256": SESSIONS_SHA} or receipt.get("sha256") != RECEIPT_SHA):
        raise SourceError("staged source does not match reviewed identity")
    result = {"schema": "qwen38_teacher_source_staged_v1", "status": "verified",
              "selection_sha256": SELECTION_SHA, "hydrated_file_sha256": RECEIPT_FILE_SHA,
              "hydration_receipt_sha256": RECEIPT_SHA, "session_count": 2886,
              "sessions_sha256": SESSIONS_SHA}
    if seal:
        if root != DEST:
            raise SourceError("seal destination differs")
        result["verified_at_unix"] = time.time()
        result["sha256"] = digest(result)
        marker = root / "STAGED.json"
        if marker.exists():
            raise SourceError("stage marker already exists")
        temporary = root / ".STAGED.json.tmp"
        with os.fdopen(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as out:
            json.dump(result, out, sort_keys=True, separators=(",", ":"))
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, marker)
    return result


def job() -> dict:
    # Suspended until reviewed. After admission, the Pod creates DEST once and
    # waits for the verifier to publish STAGED.json after full copied-byte check.
    waiter = """import hashlib,json,os,time
from pathlib import Path
os.umask(0o077)
p=Path('/mnt/sfs/jobs/chris-q38-goal-teacher-source-v1'); p.mkdir(mode=0o700)
for _ in range(1650):
 q=p/'STAGED.json'
 if q.exists():
  v=json.loads(q.read_text()); b={k:x for k,x in v.items() if k!='sha256'}
  d='sha256:'+hashlib.sha256(json.dumps(b,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
  if v.get('sha256')==d and (v.get('schema'),v.get('status'),v.get('hydration_receipt_sha256'),v.get('session_count'))==('qwen38_teacher_source_staged_v1','verified','sha256:9bf8e76340a87ffa936571d8e24fb68e24e1764862e7937766454039a8fb68a2',2886): raise SystemExit(0)
  raise SystemExit(2)
 time.sleep(2)
raise SystemExit(2)
"""
    return {"apiVersion": "batch/v1", "kind": "Job", "metadata": {
        "name": NAME, "namespace": "fleet-train-jobs",
        "annotations": {"fleet.ai/failure-alerts": "off",
                        "cyber-post-train.fleet.ai/destination": str(DEST),
                        "cyber-post-train.fleet.ai/receipt-sha256": RECEIPT_SHA},
        "labels": {"kueue.x-k8s.io/queue-name": "training-lq",
                   "kueue.x-k8s.io/priority-class": "q1",
                   "cyber-post-train.fleet.ai/owner": "chris"}},
        "spec": {"suspend": True, "backoffLimit": 0, "activeDeadlineSeconds": 3600,
                 "ttlSecondsAfterFinished": 7200, "template": {"spec": {
                     "restartPolicy": "Never", "automountServiceAccountToken": False,
                     "nodeSelector": {"kubernetes.io/arch": "amd64", "workload": "fleetai-training-ng-cpu"},
                     "priorityClassName": "c1", "priority": 10000,
                     "tolerations": [{"key": "workload", "operator": "Equal",
                                      "value": "fleetai-training-ng-cpu", "effect": "NoSchedule"}],
                     "containers": [{"name": "staging", "image": IMAGE,
                                     "command": ["python", "-u", "-c", waiter],
                                     "resources": {"requests": {"cpu": "2", "memory": "4Gi"},
                                                   "limits": {"cpu": "4", "memory": "8Gi"}},
                                     "volumeMounts": [{"name": "sfs", "mountPath": "/mnt/sfs"}]}],
                     "volumes": [{"name": "sfs", "persistentVolumeClaim": {"claimName": "sfs-shared"}}]}}}}


if __name__ == "__main__":
    try:
        result = job() if sys.argv[1] == "job" else verify(Path(sys.argv[2]), seal="--seal" in sys.argv[3:])
        print(json.dumps(result, sort_keys=True))
    except (SourceError, ValueError, OSError, KeyError, IndexError):
        raise SystemExit(2) from None
