# Installing the noether skill

Copy `noether/` into `~/.claude/skills/`, then register the MCP server:

```bash
claude mcp add noether -- uv run --directory /path/to/NOETHER python -m noether.mcp_server
```

Requires the MCP extra: `uv pip install -e ".[mcp]"`.
