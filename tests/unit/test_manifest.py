"""Unit tests: manifest builder (no network)."""
import pytest

from sgrag.data.manifest import build_manifest, REQUIRED_KEYS


def test_manifest_has_required_keys():
    m = build_manifest(requested=200, retrieved=198, retained=190,
                       pmid_coverage=1.0)
    for k in REQUIRED_KEYS:
        assert k in m
    assert m["requested"] == 200
    assert m["retrieval_date_utc"]  # non-empty ISO timestamp


def test_manifest_rejects_missing_keys():
    import sgrag.data.manifest as mm
    orig = mm.REQUIRED_KEYS
    mm.REQUIRED_KEYS = ["project", "nonexistent_key"]
    try:
        with pytest.raises(ValueError):
            build_manifest()
    finally:
        mm.REQUIRED_KEYS = orig
