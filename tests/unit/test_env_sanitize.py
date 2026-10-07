"""Unit tests: proxy env sanitizer (no network)."""
import os

from sgrag.utils.env import sanitize_proxy_env


def _set(var, val):
    old = os.environ.get(var)
    os.environ[var] = val
    return old


def _restore(var, old):
    if old is None:
        os.environ.pop(var, None)
    else:
        os.environ[var] = old


def test_removes_bracketed_ipv6_entries():
    o1, o2 = _set("NO_PROXY", "localhost,[::1],example.com"), _set("no_proxy", "a.com,[fd8b::1]")
    try:
        removed = sanitize_proxy_env()
        assert os.environ["NO_PROXY"] == "localhost,example.com"
        assert os.environ["no_proxy"] == "a.com"
        assert len(removed) == 2
    finally:
        _restore("NO_PROXY", o1)
        _restore("no_proxy", o2)


def test_wellformed_vars_untouched():
    o1, o2 = _set("NO_PROXY", "localhost,127.0.0.1"), _set("no_proxy", "")
    try:
        removed = sanitize_proxy_env()
        assert os.environ["NO_PROXY"] == "localhost,127.0.0.1"
        assert removed == []
    finally:
        _restore("NO_PROXY", o1)
        _restore("no_proxy", o2)


def test_missing_vars_ok():
    os.environ.pop("NO_PROXY", None)
    os.environ.pop("no_proxy", None)
    assert sanitize_proxy_env() == []
