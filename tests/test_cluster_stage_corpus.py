from __future__ import annotations

import pytest

from cluster.stage_corpus import _safe_relative


@pytest.mark.parametrize(
    "path",
    [
        "session-corpus/env_key=x/date=2026-08-28/data.parquet",
        "session-scores/data.parquet",
    ],
)
def test_safe_relative_accepts_only_corpus_namespaces(path):
    assert str(_safe_relative(path)) == path


@pytest.mark.parametrize("path", ["/etc/passwd", "../escape", "session-corpus/../../escape"])
def test_safe_relative_rejects_escape(path):
    with pytest.raises(ValueError):
        _safe_relative(path)
