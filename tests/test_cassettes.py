"""Cassette store: keying, redaction, corruption detection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from noether.cassettes import CassetteMiss, CassetteStore, Interaction, identity, redact


def test_identity_is_order_independent() -> None:
    a = identity("arxiv", "GET", "https://x/api", {"b": 2, "a": 1})
    b = identity("arxiv", "GET", "https://x/api", {"a": 1, "b": 2})
    assert a == b


def test_identity_distinguishes_provider_and_method() -> None:
    base = identity("arxiv", "GET", "https://x/api", {"q": 1})
    assert base != identity("inspire", "GET", "https://x/api", {"q": 1})
    assert base != identity("arxiv", "POST", "https://x/api", {"q": 1})


def test_credentials_do_not_change_the_key() -> None:
    """A rotated API key must not invalidate an entire recorded corpus."""
    with_old = identity("s2", "GET", "https://x", {"q": 1, "api_key": "old"})
    with_new = identity("s2", "GET", "https://x", {"q": 1, "api_key": "new"})
    assert with_old == with_new


def test_redact_masks_secret_names_only() -> None:
    out = redact({"q": "rabi", "API_KEY": "hunter2", "token": "t"})
    assert out["q"] == "rabi"
    assert out["API_KEY"] == "<redacted>"
    assert out["token"] == "<redacted>"


def _interaction(content: bytes = b"hello", **kw: object) -> Interaction:
    params = {"q": "rabi", "api_key": "hunter2"}
    defaults = {
        "key": identity("arxiv", "GET", "https://x/api", params),
        "provider": "arxiv",
        "method": "GET",
        "url": "https://x/api",
        "params": params,
        "status": 200,
        "content": content,
        "headers": {"Content-Type": "text/xml", "Authorization": "Bearer nope"},
    }
    defaults.update(kw)
    return Interaction(**defaults)  # type: ignore[arg-type]


def test_round_trip_utf8(tmp_path: Path) -> None:
    store = CassetteStore(tmp_path)
    original = _interaction(b"<feed>caf\xc3\xa9</feed>")
    store.save(original)
    assert store.load(original.key).content == original.content


def test_round_trip_binary(tmp_path: Path) -> None:
    """Tarballs and PDFs must survive the JSON round trip."""
    store = CassetteStore(tmp_path)
    blob = bytes(range(256))
    original = _interaction(blob)
    store.save(original)
    assert store.load(original.key).content == blob


def test_secrets_never_reach_disk(tmp_path: Path) -> None:
    store = CassetteStore(tmp_path)
    interaction = _interaction()
    path = store.save(interaction)
    raw = path.read_text(encoding="utf-8")
    assert "hunter2" not in raw
    assert "Bearer nope" not in raw


def test_miss_is_an_error_not_a_fallback(tmp_path: Path) -> None:
    with pytest.raises(CassetteMiss):
        CassetteStore(tmp_path).load("0" * 64)


def test_corrupt_body_is_detected(tmp_path: Path) -> None:
    store = CassetteStore(tmp_path)
    interaction = _interaction()
    path = store.save(interaction)

    data = json.loads(path.read_text(encoding="utf-8"))
    data["body"] = "tampered"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="corrupt"):
        store.load(interaction.key)
