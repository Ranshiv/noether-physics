"""Runtime configuration and filesystem layout.

The data root holds everything fetched or derived. It lives outside the repo by
default so that a checkout stays small and a corpus survives a reclone.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from platformdirs import user_data_dir

CassetteMode = Literal["record", "replay", "off"]

#: arXiv asks automated clients for one request every three seconds. This is not
#: a suggestion we tune for throughput -- exceeding it gets the tool blocked and
#: is the kind of thing that gets programmatic access withdrawn for everyone.
ARXIV_MIN_INTERVAL_S = 3.0

DEFAULT_MIN_INTERVAL_S = 1.0

#: Sent on every request so archive operators can identify and contact us.
USER_AGENT = "noether/{version} (+https://github.com/noether-physics/noether; research tool)"


def default_root() -> Path:
    """Resolve the data root: ``$NOETHER_ROOT`` if set, else the OS data dir."""
    env = os.environ.get("NOETHER_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path(user_data_dir("noether", appauthor=False)).resolve()


@dataclass(frozen=True)
class Config:
    """Where things live and how politely we fetch them."""

    root: Path = field(default_factory=default_root)
    cassette_mode: CassetteMode = "off"
    cassette_dir: Path | None = None
    #: Wall-clock seconds a single HTTP request may take before it is abandoned.
    timeout_s: float = 30.0
    max_retries: int = 3
    #: Contact address forwarded to APIs that ask for one (Crossref, OpenAlex)
    #: in exchange for their faster, more reliable "polite" request pools.
    mailto: str | None = None

    # ---- derived paths -------------------------------------------------

    @property
    def cache_dir(self) -> Path:
        """Fetched sources, keyed by content. A paper is downloaded once, ever."""
        return self.root / "cache"

    @property
    def db_path(self) -> Path:
        """SQLite: paper metadata, spans, citation edges, resolution results."""
        return self.root / "noether.sqlite"

    @property
    def figures_dir(self) -> Path:
        """Figures extracted from paper sources, plus plots we generate."""
        return self.root / "figures"

    @property
    def answers_dir(self) -> Path:
        """Persisted AnswerRecords, one directory per answer id."""
        return self.root / "answers"

    def cassettes(self) -> Path:
        """Where recorded HTTP interactions live."""
        if self.cassette_dir is not None:
            return self.cassette_dir
        return self.root / "cassettes"

    def ensure_dirs(self) -> None:
        """Create the directories we own. Safe to call repeatedly."""
        for path in (self.cache_dir, self.figures_dir, self.answers_dir, self.cassettes()):
            path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> Config:
        """Build a config from environment variables.

        ``NOETHER_CASSETTES`` selects record/replay/off. Tests default it to
        ``replay`` in ``conftest.py`` so the default suite never touches the network.
        """
        mode = os.environ.get("NOETHER_CASSETTES", "off")
        if mode not in ("record", "replay", "off"):
            raise ValueError(
                f"NOETHER_CASSETTES must be record, replay or off; got {mode!r}"
            )
        return cls(
            root=default_root(),
            cassette_mode=mode,  # type: ignore[arg-type]
            mailto=os.environ.get("NOETHER_MAILTO"),
        )
