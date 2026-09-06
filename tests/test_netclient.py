"""Rate limiting, replay behaviour, and the source registry."""

from __future__ import annotations

import time

import pytest

from noether.cassettes import CassetteMiss, CassetteStore, Interaction, identity
from noether.config import Config
from noether.netclient import NetClient, RateLimiter
from noether.sources.registry import UnknownSource, get_source, load_registry


class TestRateLimiter:
    def test_first_call_does_not_wait(self) -> None:
        assert RateLimiter(0.05).wait("example.org") == 0.0

    def test_second_call_waits(self) -> None:
        limiter = RateLimiter(0.05)
        limiter.wait("example.org")
        start = time.monotonic()
        limiter.wait("example.org")
        assert time.monotonic() - start >= 0.04

    def test_hosts_are_independent(self) -> None:
        """A slow archive must not throttle a fast one."""
        limiter = RateLimiter(0.05)
        limiter.wait("a.org")
        assert limiter.wait("b.org") == 0.0

    def test_per_host_interval_overrides_default(self) -> None:
        limiter = RateLimiter(1.0)
        limiter.configure("export.arxiv.org", 3.0)
        assert limiter.interval_for("export.arxiv.org") == 3.0
        assert limiter.interval_for("api.openalex.org") == 1.0


class TestReplay:
    def test_replay_serves_recorded_response(self, tmp_config: Config) -> None:
        params = {"id_list": "2001.11966"}
        url = "https://export.arxiv.org/api/query"
        key = identity("arxiv", "GET", url, params)

        CassetteStore(tmp_config.cassettes()).save(
            Interaction(
                key=key, provider="arxiv", method="GET", url=url, params=params,
                status=200, content=b"<feed/>", headers={},
            )
        )

        response = NetClient(tmp_config).get("arxiv", url, params=params)
        assert response.status == 200
        assert response.text == "<feed/>"
        assert response.replayed is True

    def test_replay_miss_never_falls_back_to_network(self, tmp_config: Config) -> None:
        """A CI run that quietly starts hitting arXiv is worse than a red test."""
        with pytest.raises(CassetteMiss):
            NetClient(tmp_config).get("arxiv", "https://export.arxiv.org/api/query", params={"q": "x"})


class TestRegistry:
    def test_every_source_declares_its_terms(self) -> None:
        for key, spec in load_registry().items():
            assert spec.base_url.startswith("https://"), f"{key} must use TLS"
            assert spec.min_interval_s > 0, f"{key} must declare a rate limit"
            assert spec.provides, f"{key} must say what it provides"

    def test_arxiv_interval_matches_their_documented_ask(self) -> None:
        """Three seconds is a floor we respect, not a knob to tune."""
        assert get_source("arxiv").min_interval_s == 3.0

    def test_arxiv_exposes_the_eprint_endpoint(self) -> None:
        assert get_source("arxiv").url("eprint_url").endswith("/e-print/")

    def test_unknown_source_is_refused(self) -> None:
        with pytest.raises(UnknownSource):
            get_source("sci-hub")

    def test_nothing_claims_verification_it_has_not_had(self) -> None:
        """status and verified_on must agree; an unverified contract says so."""
        for key, spec in load_registry().items():
            if spec.status == "contract-verified":
                assert spec.verified_on, f"{key} claims verified but names no date"
