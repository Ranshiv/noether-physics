"""Reproducibility bundles -- guarantee 4.

A bundle seals an answer together with everything needed to check it later: the
record, the corpus fingerprint, the environment, and a content hash over all of
it. ``verify`` re-derives the hash and reports precisely what moved, so "this
answer no longer reproduces" becomes a specific finding rather than a suspicion.

Signing is optional and Ed25519 when available. A bundle without a signature is
still verifiable for integrity -- it simply cannot prove *who* built it, and
says so rather than implying more than it can support.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .. import __version__
from ..answer.records import AnswerRecord

BUNDLE_VERSION = 1


@dataclass
class Environment:
    """What produced the bundle. Enough to explain a divergence later."""

    noether_version: str
    python_version: str
    platform: str

    @classmethod
    def capture(cls) -> Environment:
        return cls(
            noether_version=__version__,
            python_version=sys.version.split()[0],
            platform=platform.platform(),
        )


@dataclass
class Bundle:
    """A sealed, checkable answer."""

    bundle_id: str
    record: dict[str, Any]
    corpus: dict[str, Any]
    environment: dict[str, Any]
    created_at: str
    content_hash: str = ""
    signature: str | None = None
    public_key: str | None = None
    bundle_version: int = BUNDLE_VERSION
    #: Files referenced by the answer, with their hashes at seal time.
    artifacts: dict[str, str] = field(default_factory=dict)

    def compute_hash(self) -> str:
        """Hash over everything except the hash and signature themselves."""
        payload = asdict(self)
        payload.pop("content_hash", None)
        payload.pop("signature", None)
        payload.pop("public_key", None)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> Bundle:
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class Verification:
    """The outcome of checking a bundle."""

    ok: bool
    #: Named differences, so a failure says what changed rather than that it did.
    problems: list[str] = field(default_factory=list)
    signature_checked: bool = False

    def summary(self) -> str:
        if self.ok:
            suffix = " (signature verified)" if self.signature_checked else " (unsigned)"
            return "bundle verifies" + suffix
        return "bundle FAILED: " + "; ".join(self.problems)


def build(
    record: AnswerRecord,
    corpus_stats: dict[str, int],
    artifacts: dict[str, Path] | None = None,
) -> Bundle:
    """Seal an answer record into a bundle."""
    bundle = Bundle(
        bundle_id=f"bundle_{record.answer_id}",
        record=record.to_dict(),
        corpus={"hash": record.corpus_hash, "stats": dict(corpus_stats)},
        environment=asdict(Environment.capture()),
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        artifacts={
            name: _hash_file(path)
            for name, path in (artifacts or {}).items()
            if Path(path).exists()
        },
    )
    bundle.content_hash = bundle.compute_hash()
    return bundle


def verify(bundle: Bundle, corpus_hash: str | None = None) -> Verification:
    """Check a bundle's integrity, and optionally against the live corpus."""
    problems: list[str] = []

    recomputed = bundle.compute_hash()
    if recomputed != bundle.content_hash:
        problems.append(
            f"content hash mismatch: bundle says {bundle.content_hash[:12]}, "
            f"contents hash to {recomputed[:12]}"
        )

    if corpus_hash is not None and bundle.corpus.get("hash") != corpus_hash:
        problems.append(
            f"corpus has changed: sealed against {str(bundle.corpus.get('hash'))[:12]}, "
            f"now {corpus_hash[:12]}"
        )

    signature_checked = False
    if bundle.signature and bundle.public_key:
        ok, reason = _verify_signature(bundle)
        signature_checked = ok
        if not ok:
            problems.append(reason)

    return Verification(ok=not problems, problems=problems, signature_checked=signature_checked)


def sign(bundle: Bundle, private_key_hex: str) -> Bundle:
    """Sign a bundle with an Ed25519 key, if cryptography is installed."""
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "signing needs the 'cryptography' package; a bundle without a "
            "signature is still integrity-checkable"
        ) from exc

    key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(private_key_hex))
    bundle.content_hash = bundle.compute_hash()
    bundle.signature = key.sign(bundle.content_hash.encode("utf-8")).hex()
    bundle.public_key = key.public_key().public_bytes_raw().hex()
    return bundle


def _verify_signature(bundle: Bundle) -> tuple[bool, str]:
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError:  # pragma: no cover - optional dependency
        return False, "bundle is signed but 'cryptography' is not installed to check it"

    try:
        public = Ed25519PublicKey.from_public_bytes(bytes.fromhex(bundle.public_key or ""))
        public.verify(bytes.fromhex(bundle.signature or ""), bundle.content_hash.encode("utf-8"))
        return True, ""
    except InvalidSignature:
        return False, "signature does not match the bundle contents"
    except Exception as exc:
        return False, f"signature could not be checked: {type(exc).__name__}"


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return digest.hexdigest()
