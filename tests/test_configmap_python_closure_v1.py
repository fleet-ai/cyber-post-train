import pytest

from evals.fleet import configmap_python_closure_v1 as closure


def test_recursive_static_import_must_be_installed():
    data = {
        "run.sh": "a.py:a.py b.py:b.py",
        "a.py": "from evals.fleet import b",
        "b.py": "from evals.fleet import missing",
    }
    with pytest.raises(ValueError, match="missing imported by b"):
        closure.validate(data)


def test_explicit_dynamic_allowlist_is_fail_closed():
    data = {"run.sh": "a.py:a.py", "a.py": "from evals.fleet import generated_at_runtime"}
    assert closure.validate(data, dynamic_import_allowlist={"generated_at_runtime"})
    with pytest.raises(ValueError, match="generated_at_runtime"):
        closure.validate(data)
