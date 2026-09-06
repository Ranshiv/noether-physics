"""The MCP server's tool surface.

The structural test here exists because of a real bug: four tools were appended
below the ``if __name__ == "__main__"`` block, so importing the module registered
14 tools while *running* it registered only 10 — ``main()`` blocks in
``server.run()`` before execution ever reaches the definitions below it.

An import-based check cannot see that difference. Only reading the source, or
driving the server over stdio, can.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "src" / "noether" / "mcp_server.py"

#: Every tool the server is expected to advertise.
EXPECTED_TOOLS = {
    "search_arxiv",
    "ingest_paper",
    "corpus_status",
    "gather_evidence",
    "submit_answer",
    "list_equations",
    "check_equation",
    "run_equation",
    "audit_references",
    "search_corpus",
    "list_concepts",
    "explain_concept",
    "list_simulations",
    "run_simulation",
}


def _source() -> str:
    return SOURCE.read_text(encoding="utf-8")


class TestEntryPointPlacement:
    """A tool defined after the entry point is silently never registered."""

    def test_main_guard_is_the_last_statement(self) -> None:
        text = _source()
        guard = text.index('if __name__ == "__main__":')
        after = text[guard:]
        assert "@server.tool()" not in after, (
            "a @server.tool() is defined after the __main__ guard; it will not be "
            "registered when the server actually runs"
        )

    def test_every_tool_precedes_the_guard(self) -> None:
        text = _source()
        guard = text.index('if __name__ == "__main__":')
        for match in re.finditer(r"@server\.tool\(\)", text):
            assert match.start() < guard, "tool defined below the __main__ guard"


class TestToolRegistration:
    def test_all_expected_tools_are_registered(self) -> None:
        mcp_server = pytest.importorskip("noether.mcp_server")
        tools = asyncio.run(mcp_server.server.list_tools())
        assert {t.name for t in tools} == EXPECTED_TOOLS

    def test_every_tool_documents_itself(self) -> None:
        """The description is how the model knows when to call a tool."""
        mcp_server = pytest.importorskip("noether.mcp_server")
        for tool in asyncio.run(mcp_server.server.list_tools()):
            assert tool.description and len(tool.description) > 40, tool.name

    def test_source_defines_exactly_the_expected_tools(self) -> None:
        """Catches a tool added to the source but not to the expected set."""
        defined = set(
            re.findall(r"@server\.tool\(\)\s*\ndef (\w+)", _source())
        )
        assert defined == EXPECTED_TOOLS


class TestHonestyInstructions:
    """The tool descriptions carry the rules the model is judged by."""

    def test_gather_evidence_warns_against_quoting_from_memory(self) -> None:
        text = _source()
        assert "Do not quote the corpus from memory" in text

    def test_submit_answer_states_that_claims_are_dropped(self) -> None:
        assert "**dropped**" in _source()

    def test_check_equation_states_unknown_is_not_a_pass(self) -> None:
        assert "``unknown`` is not a pass" in _source()

    def test_run_simulation_mentions_convergence(self) -> None:
        assert "convergence check" in _source()
