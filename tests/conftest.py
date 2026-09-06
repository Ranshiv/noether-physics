"""Test configuration.

Two rules the suite depends on:

* Cassettes default to ``replay``, so a test that forgets to stub the network
  fails on a cassette miss instead of quietly hitting arXiv from CI.
* Tests marked ``live`` are skipped unless ``NOETHER_LIVE=1``. They exist to
  re-record cassettes and to check that a provider has not changed its schema.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from noether.config import Config

FIXTURES = Path(__file__).parent / "fixtures"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("NOETHER_LIVE") == "1":
        return
    skip = pytest.mark.skip(reason="live test; set NOETHER_LIVE=1 to run")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def tmp_config(tmp_path: Path) -> Config:
    """A config rooted in a temp dir, with the network sealed off."""
    cfg = Config(
        root=tmp_path,
        cassette_mode="replay",
        cassette_dir=tmp_path / "cassettes",
    )
    cfg.ensure_dirs()
    return cfg


@pytest.fixture
def atom_feed() -> str:
    """A minimal but realistic arXiv Atom response."""
    return (FIXTURES / "arxiv_feed.xml").read_text(encoding="utf-8")
