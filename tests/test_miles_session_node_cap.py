import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

SOURCE = (
    Path(__file__).parents[1] / "training/images/miles-opencode-long-context/tree_trajectory.py"
)


def _load(monkeypatch, name, cap=None):
    for package in (
        "miles",
        "miles.rollout",
        "miles.rollout.session",
        "miles.utils",
        "miles.utils.chat_template_utils",
        "miles.utils.chat_template_utils.message_matcher_hub",
    ):
        monkeypatch.setitem(sys.modules, package, ModuleType(package))
    types = ModuleType("miles.rollout.session.types")
    types.SessionRecord = object
    monkeypatch.setitem(sys.modules, "miles.rollout.session.types", types)
    matcher = sys.modules["miles.utils.chat_template_utils.message_matcher_hub"]
    matcher.SessionMessageMatcher = object
    matcher.strict_message_matches = lambda left, right: left == right
    if cap is None:
        monkeypatch.delenv("MILES_SESSION_MAX_NODES", raising=False)
    else:
        monkeypatch.setenv("MILES_SESSION_MAX_NODES", cap)
    spec = importlib.util.spec_from_file_location(name, SOURCE)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _append(tree, parent, count):
    for index in range(count):
        parent = tree.create_node(
            parent,
            delta_messages=[{"role": "assistant", "content": str(index)}],
            token_ids=[index],
            completion_span=(0, 1),
            committed_at=float(index),
            response_id=str(index),
            record=object(),
            finish_reason="stop",
        )
    return parent


def test_default_cap_remains_1024(monkeypatch):
    module = _load(monkeypatch, "reviewed_tree_default")
    tree = module.SessionTree()
    parent = _append(tree, None, 1024)
    with pytest.raises(ValueError, match=r"node cap reached \(1024\)"):
        _append(tree, parent, 1)


def test_explicit_4096_cap_accepts_more_than_1000_model_calls(monkeypatch):
    module = _load(monkeypatch, "reviewed_tree_long", "4096")
    tree = module.SessionTree()
    _append(tree, None, 1105)
    assert module.MAX_NODES == 4096
    assert len(tree.nodes) == 1105


@pytest.mark.parametrize("value", ["4096 ", "01024", "1000", "16385", "wrong"])
def test_invalid_node_cap_fails_at_import(monkeypatch, value):
    with pytest.raises(ValueError):
        _load(monkeypatch, "reviewed_tree_invalid_" + value.strip(), value)
