import json
import os
import re
import subprocess
import sys
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient

_CONFIG_FILE = Path(__file__).parent / "mcp_servers.json"


def _resolve_env_vars(value: str) -> str:
    """Replace ${VAR_NAME} placeholders with actual environment variable values."""
    return re.sub(
        r"\$\{(\w+)\}",
        lambda m: os.environ.get(m.group(1), m.group(0)),
        value,
    )


def _resolve(obj):
    """Recursively resolve env var placeholders in all string values."""
    if isinstance(obj, dict):
        return {k: _resolve(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve(v) for v in obj]
    if isinstance(obj, str):
        return _resolve_env_vars(obj)
    return obj


def load_mcp_config(path: Path = _CONFIG_FILE) -> dict:
    """Load and resolve mcp_servers.json into a MultiServerMCPClient-compatible config."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    servers = raw.get("mcpServers", {})

    config = {}
    for name, entry in servers.items():
        transport = entry.get("transport") or entry.get("type", "stdio")

        if transport in ("streamable-http", "http"):
            # HTTP-based transport — only url and transport are needed
            config[name] = {
                "transport": "http",
                "url": _resolve(entry["url"]),
            }
        else:
            # stdio transport
            config[name] = {
                "command": _resolve(entry.get("command", "")),
                "args": _resolve(entry.get("args", [])),
                "transport": "stdio",
            }
            if "env" in entry:
                config[name]["env"] = _resolve(entry["env"])

    return config


import logging

_log = logging.getLogger(__name__)
_MCP_ERRLOG = Path(__file__).parent / "mcp_stderr.log"


def _patch_mcp_stderr_for_jupyter() -> None:
    """In Jupyter: swap virtual stderr → log file (no fileno).
    In FastAPI/uvicorn: no-op (real stderr works fine).
    """
    try:
        sys.stderr.fileno()
        return  # real stderr — no patch needed
    except Exception:
        pass

    import mcp.client.stdio as _mcp_stdio

    _errlog_fh = open(_MCP_ERRLOG, "ab")  # noqa: SIM115
    _orig = _mcp_stdio._create_platform_compatible_process

    async def _patched(command, args, env, errlog, cwd):
        return await _orig(command, args, env, _errlog_fh, cwd)

    _mcp_stdio._create_platform_compatible_process = _patched
    _log.info("MCP stderr → %s", _MCP_ERRLOG)


def build_mcp_client(
    servers: list[str] | None = None,
    path: Path = _CONFIG_FILE,
) -> MultiServerMCPClient:
    """
    Build a MultiServerMCPClient from mcp_servers.json.

    Args:
        servers: Optional list of server names to include. Loads all if None.
        path:    Path to the JSON config file.
    """
    config = load_mcp_config(path)

    _patch_mcp_stderr_for_jupyter()

    if servers:
        missing = [s for s in servers if s not in config]
        if missing:
            raise ValueError(f"Servers not found in config: {missing}")
        config = {s: config[s] for s in servers}
        # print(config)

    return MultiServerMCPClient(config)
