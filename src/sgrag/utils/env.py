"""Environment hygiene for network-facing scripts.

Some sandboxed VMs export NO_PROXY/no_proxy with bare bracketed IPv6 literals
(e.g. "[::1]"), which crash httpx's proxy-pattern parser (httpx.InvalidURL).
This sanitizer drops only malformed entries and preserves everything else.
It is a no-op on machines with well-formed proxy variables (e.g. a Mac).
"""

import os

_PROXY_VARS = ("NO_PROXY", "no_proxy")


def sanitize_proxy_env() -> list:
    """Remove malformed (bracketed) entries from NO_PROXY/no_proxy.

    Returns the list of removed "VAR:entry" strings for logging.
    """
    removed = []
    for var in _PROXY_VARS:
        raw = os.environ.get(var)
        if not raw:
            continue
        keep = []
        for entry in raw.split(","):
            entry = entry.strip()
            if not entry:
                continue
            if "[" in entry or "]" in entry:
                removed.append(f"{var}:{entry}")
                continue
            keep.append(entry)
        os.environ[var] = ",".join(keep)
    return removed
