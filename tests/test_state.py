import pytest
from dex.state import records


def test_json_types_paths_and_key_order():
    a = {"a/b": "é", "z": [True, 1, "1", None, {}, []]}
    assert records(a) == records({"z": a["z"], "a/b": "é"})
    assert records(a)[1].path == "/a~1b"
    assert len(set((r.kind, r.text) for r in records(a))) >= 6


def test_cache_key_changes_with_semantics_and_model():
    r = records("Bonjour")[0]
    assert r.cache_key("a", "t") != r.cache_key("b", "t")
    assert r.cache_key("a", "t") != r.cache_key("a", "u")
    assert records(["ancien", "neuf"])[1] == records(["ancien", "neuf", "suite"])[1]
    with pytest.raises(ValueError):
        records({"x": float("nan")})

