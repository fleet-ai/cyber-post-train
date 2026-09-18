"""Synthetic native-image transaction test; stdin carries code/fixture, never data."""

import importlib.util
import json
import os
import tempfile
from pathlib import Path


def run(payload):
    with tempfile.TemporaryDirectory(prefix="model-stage-native-") as directory:
        root = Path(directory)
        module_path = root / "model_stage.py"
        module_path.write_text(payload["source"])
        spec = importlib.util.spec_from_file_location("native_stage", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        plan, objects, environment = payload["plan"], payload["objects"], payload["environment"]
        (root / "chris-autoresearch").mkdir()
        # Only this fixture's /models open is redirected. Linux atomic rename
        # still uses the actual directory descriptors and no-replace syscall.
        original_open = os.open

        def local_open(path, *args, **kwargs):
            return original_open(root if path == "/models" else path, *args, **kwargs)

        import contextlib
        import io

        @contextlib.contextmanager
        def opener(path):
            yield io.BytesIO(objects[path].encode())

        os.open = local_open
        try:
            first = module.execute_stage(plan, open_source=opener, environment=environment)
            second = module.execute_stage(plan, open_source=opener, environment={})
            assert first == second
            assert first["source"]["optimizer_steps_executed"] == 0
            final = root / "chris-autoresearch/test"
            (final / "payload-00.bin").write_bytes(b"tampered")
            try:
                module.execute_stage(plan, open_source=opener, environment={})
            except ValueError:
                pass
            else:
                raise AssertionError("tamper not detected")
        finally:
            os.open = original_open
        print(json.dumps({"native_stream_and_reopen": True, "tamper_rejected": True, "gpus": 0}))


if __name__ == "__main__":
    import sys

    run(json.load(sys.stdin))
