"""Content-addressed record/replay for HTTP interactions.

Connector tests that monkeypatch the HTTP layer prove only that the connector
calls the mock the way the test author imagined. Recording a real response once
and replaying it forever means the test is pinned to what the archive actually
returned, so a provider changing its schema shows up as a failing test rather
than as silently wrong science months later.

Modes:
    record  -- perform the live request, store the response, return it
    replay  -- serve from the store; a miss is an error, never a live fallback
    off     -- pass through, store nothing
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Query/header names whose values are redacted before anything touches disk.
#: Matched case-insensitively against the *whole* name.
_SECRET_NAMES = frozenset(
    {
        "api_key",
        "apikey",
        "key",
        "token",
        "access_token",
        "authorization",
        "password",
        "secret",
        "client_secret",
    }
)

_REDACTED = "<redacted>"


class CassetteMiss(LookupError):
    """Replay was requested for an interaction that was never recorded."""


def redact(params: Mapping[str, Any]) -> dict[str, Any]:
    """Strip credential-shaped values, preserving key order-independence."""
    return {
        k: (_REDACTED if k.lower() in _SECRET_NAMES else v) for k, v in params.items()
    }


def identity(provider: str, method: str, url: str, params: Mapping[str, Any]) -> str:
    """Stable key for one interaction.

    Params are sorted so that call-site ordering cannot produce two cassettes
    for what is really the same request. Credentials are redacted *before*
    hashing, so a rotated key does not invalidate a recorded corpus.
    """
    payload = {
        "provider": provider,
        "method": method.upper(),
        "url": url,
        "params": sorted(redact(params).items()),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Interaction:
    """One recorded request/response pair."""

    key: str
    provider: str
    method: str
    url: str
    params: dict[str, Any]
    status: int
    #: Response body. Stored base64 when not valid UTF-8 (tarballs, PDFs).
    content: bytes
    headers: dict[str, str]

    def content_sha256(self) -> str:
        return hashlib.sha256(self.content).hexdigest()

    def to_json(self) -> dict[str, Any]:
        try:
            body = self.content.decode("utf-8")
            encoding = "utf-8"
        except UnicodeDecodeError:
            body = base64.b64encode(self.content).decode("ascii")
            encoding = "base64"
        return {
            "key": self.key,
            "provider": self.provider,
            "method": self.method,
            "url": self.url,
            "params": redact(self.params),
            "status": self.status,
            "encoding": encoding,
            "body": body,
            "content_sha256": self.content_sha256(),
            "headers": {k: v for k, v in self.headers.items() if k.lower() not in _SECRET_NAMES},
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Interaction:
        raw = data["body"]
        content = (
            base64.b64decode(raw) if data.get("encoding") == "base64" else raw.encode("utf-8")
        )
        recorded = data.get("content_sha256")
        actual = hashlib.sha256(content).hexdigest()
        if recorded and recorded != actual:
            raise ValueError(
                f"cassette {data['key']} is corrupt: body hashes to {actual}, "
                f"manifest says {recorded}"
            )
        return cls(
            key=data["key"],
            provider=data["provider"],
            method=data["method"],
            url=data["url"],
            params=dict(data.get("params", {})),
            status=int(data["status"]),
            content=content,
            headers=dict(data.get("headers", {})),
        )


class CassetteStore:
    """A directory of interactions, one JSON file per key."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    def path_for(self, key: str) -> Path:
        # Shard by first two hex chars; a flat directory of tens of thousands
        # of files is painful on Windows.
        return self.directory / key[:2] / f"{key}.json"

    def has(self, key: str) -> bool:
        return self.path_for(key).exists()

    def load(self, key: str) -> Interaction:
        path = self.path_for(key)
        if not path.exists():
            raise CassetteMiss(
                f"no cassette for {key}. Re-record with NOETHER_CASSETTES=record, "
                f"or mark the test 'live'."
            )
        return Interaction.from_json(json.loads(path.read_text(encoding="utf-8")))

    def save(self, interaction: Interaction) -> Path:
        path = self.path_for(interaction.key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(interaction.to_json(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return path

    def keys(self) -> list[str]:
        return sorted(p.stem for p in self.directory.glob("*/*.json"))
