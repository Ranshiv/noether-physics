"""Loader for ``research/sources/source_registry.yaml``.

Connectors read their base URL and rate limit from the registry rather than
hard-coding them, so that changing how politely we treat an archive is a
one-line edit in a reviewed file instead of a grep across the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


class UnknownSource(KeyError):
    """A connector asked for a source that is not in the registry."""


@dataclass(frozen=True)
class SourceSpec:
    key: str
    name: str
    base_url: str
    min_interval_s: float
    auth: str
    provides: tuple[str, ...]
    status: str
    docs: str | None = None
    terms: str | None = None
    verified_on: str | None = None
    extra: dict[str, Any] | None = None

    def url(self, field: str) -> str:
        """Fetch an auxiliary URL such as arXiv's ``eprint_url``."""
        if self.extra and field in self.extra:
            return str(self.extra[field])
        raise UnknownSource(f"{self.key} has no {field!r} in the registry")


def registry_path() -> Path:
    """Locate the registry relative to the installed package."""
    # src/noether/sources/registry.py -> repo root is three parents up.
    return Path(__file__).resolve().parents[3] / "research" / "sources" / "source_registry.yaml"


@lru_cache(maxsize=1)
def load_registry(path: Path | None = None) -> dict[str, SourceSpec]:
    target = path or registry_path()
    raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    known = {
        "name", "base_url", "min_interval_s", "auth", "provides",
        "status", "docs", "terms", "verified_on",
    }
    specs: dict[str, SourceSpec] = {}
    for key, entry in (raw.get("sources") or {}).items():
        specs[key] = SourceSpec(
            key=key,
            name=entry["name"],
            base_url=entry["base_url"],
            min_interval_s=float(entry["min_interval_s"]),
            auth=entry.get("auth", "none"),
            provides=tuple(entry.get("provides", ())),
            status=entry.get("status", "contract-unverified"),
            docs=entry.get("docs"),
            terms=entry.get("terms"),
            verified_on=entry.get("verified_on"),
            extra={k: v for k, v in entry.items() if k not in known},
        )
    return specs


def get_source(key: str) -> SourceSpec:
    registry = load_registry()
    if key not in registry:
        raise UnknownSource(f"{key!r} is not in the source registry; add it before calling it")
    return registry[key]
