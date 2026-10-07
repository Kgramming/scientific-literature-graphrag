"""Test-session hygiene: strip malformed proxy entries (sandbox quirk) so
Hugging Face downloads / cache checks never crash on httpx URL parsing.
Runs before any test module is imported.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sgrag.utils.env import sanitize_proxy_env

sanitize_proxy_env()
