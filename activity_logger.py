"""Activity logger for the Clinical Genomic Agent.

Writes one JSON-Lines record per agent turn to:
    logs/activity_YYYY-MM-DD.jsonl   — daily rotating log file

Each record contains:
    timestamp       ISO-8601 UTC time of the request
    session_id      thread_id that identifies the conversation
    channel         "web" | "whatsapp" | "introduce"
    user_message    raw user input
    agent_response  final concatenated agent reply
    tools_called    list of tool names invoked during the turn
    seed            random integer seed passed to the LLM for this call
    latency_ms      wall-clock time from request to last token (ms)
    token_usage     {prompt_tokens, completion_tokens, total_tokens} or null
    error           error message string if the turn failed, else null
"""

import json
import logging
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOG_DIR = Path(os.environ.get("LOG_DIR", Path(__file__).parent / "logs"))
_LOG_DIR.mkdir(parents=True, exist_ok=True)

# Standard Python logger for server-level messages (startup, errors).
_console = logging.getLogger("agent.activity")
if not _console.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    _console.addHandler(_h)
    _console.setLevel(logging.INFO)


def generate_seed() -> int:
    """Return a fresh random 32-bit integer seed for one LLM call."""
    return random.getrandbits(32)


def _log_path() -> Path:
    """Return today's JSONL log file path (UTC date)."""
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return _LOG_DIR / f"activity_{date_str}.jsonl"


def write_record(
    *,
    session_id: str,
    channel: str,
    user_message: str,
    agent_response: str,
    tools_called: list[str],
    seed: int,
    latency_ms: float,
    token_usage: dict[str, int] | None = None,
    error: str | None = None,
) -> None:
    """Append one activity record to today's JSONL log file.

    Thread-safe for single-process use; each write is a single line append
    which is atomic on POSIX systems and safe enough for asyncio workloads
    (no concurrent file handles from multiple threads).
    """
    record: dict[str, Any] = {
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "session_id":     session_id,
        "channel":        channel,
        "user_message":   user_message,
        "agent_response": agent_response,
        "tools_called":   tools_called,
        "seed":           seed,
        "latency_ms":     round(latency_ms, 1),
        "token_usage":    token_usage,
        "error":          error,
    }
    line = json.dumps(record, ensure_ascii=False)
    with _log_path().open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")

    status = "ERROR" if error else "OK"
    _console.info(
        "[%s] channel=%s session=%s seed=%d latency=%.0fms tools=%s",
        status, channel, session_id, seed, latency_ms, tools_called,
    )


def extract_tools_called(chunks: list[dict]) -> list[str]:
    """Extract unique tool names from a list of LangGraph astream chunk dicts."""
    from langchain_core.messages import ToolMessage
    seen: list[str] = []
    for chunk in chunks:
        for node_output in chunk.values():
            for msg in node_output.get("messages", []):
                if isinstance(msg, ToolMessage) and msg.name not in seen:
                    seen.append(msg.name)
    return seen


def extract_token_usage(chunks: list[dict]) -> dict[str, int] | None:
    """Pull token usage from the last AIMessage that carries usage_metadata."""
    from langchain_core.messages import AIMessage
    for chunk in reversed(chunks):
        for node_output in chunk.values():
            for msg in reversed(node_output.get("messages", [])):
                if isinstance(msg, AIMessage) and getattr(msg, "usage_metadata", None):
                    u = msg.usage_metadata
                    return {
                        "prompt_tokens":     u.get("input_tokens", 0),
                        "completion_tokens": u.get("output_tokens", 0),
                        "total_tokens":      u.get("total_tokens", 0),
                    }
    return None
