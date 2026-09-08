"""FastAPI server exposing a LangGraph ReAct agent over HTTP and WhatsApp.

Endpoints:
    GET  /           — Serves the static chat UI.
    POST /chat       — Streams agent replies for the web UI.
    GET  /whatsapp   — Meta webhook verification handshake.
    POST /whatsapp   — Receives WhatsApp messages and replies via Graph API.
"""

import asyncio
import os
from typing import Annotated
import numpy as np 
from typing import Any
from pathlib import Path
import subprocess
import pandas as pd

import httpx
from fastapi import FastAPI, Request, Form
from fastapi.responses import StreamingResponse, FileResponse, PlainTextResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.vectorstores import FAISS

from rank_bm25 import BM25Okapi
import bm25s
from sentence_transformers import SentenceTransformer 

from subagents import ClinicalSubAgent, GenomicSubAgent
from mcp_client import build_mcp_client

import time
import torch
from activity_logger import generate_seed, write_record, extract_tools_called, extract_token_usage
Base_Path = Path(__file__).parent
# Maps each tool name to its corresponding skill/instruction markdown file.
# These docs are injected into the system prompt when the agent uses that tool.
TOOL_SKILL_MAP = {
    "geocoding_tools": "skills/geomap_skill.md",
    "github_tools":    "skills/github_tools.md",
    "weather":         "skills/weather_skill.md",
    "google_search":   "skills/google_search_skill.md",
    "pdf_search":      "skills/pdf_search_skill.md",
}
gene_info_map=pd.read_csv(Base_Path / "tools" / "dbNSFP4.0_gene.complete",sep= "\t", header=0,index_col=0).to_dict(orient="index")
def get_skill(tool_name: str) -> str:
    """Return the skill markdown for a tool, or an empty string if not mapped."""
    rel = TOOL_SKILL_MAP.get(tool_name)
    if not rel:
        return ""
    p = Path(__file__).parent / rel
    return p.read_text(encoding="utf-8") if p.exists() else ""

# Base system prompt loaded from agent.md; defines the agent's persona and behaviour.
SYSTEM_PROMPT = (Path(__file__).parent / "agent.md").read_text(encoding="utf-8")

def _sanitize_messages(messages: list) -> list:
    """Remove AIMessages with tool_calls that have no matching ToolMessage.

    An orphaned tool call occurs when a tool errors out before its result
    is written back to state, leaving the history in an invalid state that
    most LLM providers reject.
    """
    from langchain_core.messages import AIMessage, ToolMessage
    answered_ids = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    clean = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            orphaned = [tc for tc in msg.tool_calls if tc["id"] not in answered_ids]
            if orphaned:
                continue  # drop the entire AIMessage with unmatched tool calls
        clean.append(msg)
    return clean


def build_prompt(state) -> list:
    """Build the message list passed to the LLM on each agent step.

    Scans the conversation history in reverse to find the most recent
    ToolMessage and appends the matching skill doc to the system prompt,
    giving the model tool-specific guidance at the right moment.

    Args:
        state: LangGraph agent state containing a ``messages`` list.

    Returns:
        A list starting with a system message followed by all conversation
        messages.
    """
    from langchain_core.messages import ToolMessage
    skill = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage):
            skill = get_skill(msg.name)
            break
    content = f"{SYSTEM_PROMPT}\n\n---\n\n{skill}" if skill else SYSTEM_PROMPT

    # Strip orphaned tool calls then keep last 20 to avoid context overflow
    messages = _sanitize_messages(state["messages"])[-20:]
    return [{"role": "system", "content": content}, *messages]

# Load API keys from a local JSON file and push them into the environment.
# api_key.json must contain keys: GITHUB_PAT, tavilyApiKey, youApiKey, OPENAI_API_KEY.
from langchain_opentutorial import set_env
import json
_APIKEY_FILE = Path(__file__).parent / "api_key.json"
apikey_dict = json.loads(_APIKEY_FILE.read_text(encoding="utf-8"))
set_env(
    {
        "GITHUB_PAT": apikey_dict.get("GITHUB_PAT", {}),
        "tavilyApiKey":  apikey_dict.get("tavilyApiKey", ""),
        "youApiKey":  apikey_dict.get("youApiKey", {}),
        "OPENAI_API_KEY": apikey_dict.get("OPENAI_API_KEY", ""),
        # "LANGCHAIN_API_KEY": "",
        # "LANGCHAIN_TRACING_V2": "true",
        # "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        # "LANGCHAIN_PROJECT": "01-Tools",
    }
)

from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
# Serve frontend assets (HTML/CSS/JS) from the ./static directory.
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")

# Global agent instance, tools list, checkpointer, and LLM reference; populated during startup.
agent = None
_tools = None
_checkpointer = None
_llm = None

_BASE = Path(__file__).parent
# Prefer GPU if available; override with the DEVICE environment variable.
_device = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# Medical embedding model used for semantic search over clinical documents.
retrieve_model = SentenceTransformer(
    str(_BASE / "MedEmbed-large-v0.1"),
    device=_device
)

# Load the pre-built FAISS vector store for medical document retrieval.
vectorstore = FAISS.load_local(
    str(_BASE / "medical-faiss-db"),
    retrieve_model,
    allow_dangerous_deserialization=True
)

# Flatten the FAISS docstore into a plain list for downstream use.
documents = [
doc
for doc in vectorstore.docstore._dict.values()
]
    
@tool
def gene_map_tools(
    gene_id,
) -> str:
    """find gene information for a gene id"""
    gene_info = gene_info_map.get(gene_id, {'gene_name': 'Unknown', 'gene_description': 'No description available'})
    return f"Gene information for {gene_id}: {gene_info}"

# def clinical_tools(query: str) -> str:
#     """Search clinical trials."""
#     return "No clinical trials found."

async def init_agent():
    """Initialise MCP clients, load tools, and create the global ReAct agent."""
    global agent, _tools, _checkpointer, _llm
    
    github_client = build_mcp_client(servers=["github"])
    github_tools = await github_client.get_tools(server_name="github")
    
    clinical_tool = ClinicalSubAgent(documents=documents, model=retrieve_model)
    genomic_tool = GenomicSubAgent()
    
    LLAMA_SERVER_URL = os.environ.get("LLAMA_SERVER_URL", "http://localhost:8080/v1")
    _llm = ChatOpenAI(
        model="gemma-4-E2B-it-Q4_0",
        temperature=0,
        base_url=LLAMA_SERVER_URL,
        api_key="none",
    )

    _tools = [gene_map_tools, clinical_tool, genomic_tool]
    _checkpointer = MemorySaver()
    agent = create_react_agent(
        model=_llm,
        tools=_tools,
        checkpointer=_checkpointer,
        prompt=build_prompt,
    )

@app.on_event("startup")
async def startup():
    """FastAPI startup hook — initialises the agent before the server accepts requests."""
    await init_agent()


@app.get("/")
async def root():
    """Serve the main chat UI."""
    return FileResponse(str(Path(__file__).parent / "static" / "index.html"))


class ChatRequest(BaseModel):
    """Request body for the /chat endpoint."""
    message: str
    thread_id: str = "default"  # Identifies the conversation; defaults to a single shared thread.


WHATSAPP_TOKEN   = apikey_dict.get("WHATSAPP_TOKEN", "")       # Meta permanent system token
WHATSAPP_PHONE_ID = apikey_dict.get("WHATSAPP_PHONE_ID", "")   # Phone number ID from Meta dashboard
WEBHOOK_VERIFY_TOKEN = apikey_dict.get("WEBHOOK_VERIFY_TOKEN", "")  # Any string you choose

async def send_whatsapp_reply(to: str, text: str):
    """Send a text reply to a WhatsApp user via the Meta Graph API.

    Args:
        to: Recipient's WhatsApp phone number (E.164 format, e.g. "15551234567").
        text: Message body to send.
    """
    url = f"https://graph.facebook.com/v19.0/{WHATSAPP_PHONE_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}
    payload = {"messaging_product": "whatsapp", "to": to, "type": "text", "text": {"body": text}}
    async with httpx.AsyncClient() as client:
        await client.post(url, json=payload, headers=headers)


@app.get("/whatsapp")
async def whatsapp_verify(request: Request):
    """Meta webhook verification handshake."""
    params = request.query_params
    if params.get("hub.verify_token") == WEBHOOK_VERIFY_TOKEN:
        return PlainTextResponse(params.get("hub.challenge", ""))
    return PlainTextResponse("Forbidden", status_code=403)


@app.post("/whatsapp")
async def whatsapp_webhook(request: Request):
    """Meta Cloud API webhook — receives inbound WhatsApp messages and replies.

    Only text messages are handled; other types (image, audio, etc.) are
    silently ignored. The sender's phone number is used as the thread_id so
    each user maintains their own conversation history.
    """
    data = await request.json()
    try:
        entry = data["entry"][0]["changes"][0]["value"]
        msg   = entry["messages"][0]
        if msg.get("type") != "text":
            return JSONResponse({"status": "ignored"})
        sender = msg["from"]
        text   = msg["text"]["body"]
    except (KeyError, IndexError):
        # Payload did not contain a user message (e.g. delivery receipt).
        return JSONResponse({"status": "no message"})

    # Use the sender's phone number as a stable conversation thread ID.
    config = {"configurable": {"thread_id": sender}}
    seed = generate_seed()
    # bind(seed=seed) passes the seed in the request body to llama.cpp
    seeded_agent = create_react_agent(
        model=_llm.bind(seed=seed),
        tools=_tools,
        checkpointer=_checkpointer,
        prompt=build_prompt,
    )
    t0 = time.monotonic()
    chunks: list = []
    reply_parts = []
    error_msg = None
    try:
        async for chunk in seeded_agent.astream({"messages": [("user", text)]}, config=config):
            chunks.append(chunk)
            for node_output in chunk.values():
                for m in node_output.get("messages", []):
                    if hasattr(m, "content") and m.content and m.type == "ai":
                        reply_parts.append(m.content)
    except Exception as e:
        error_msg = str(e)
    finally:
        write_record(
            session_id=sender,
            channel="whatsapp",
            user_message=text,
            agent_response="".join(reply_parts),
            tools_called=extract_tools_called(chunks),
            seed=seed,
            latency_ms=(time.monotonic() - t0) * 1000,
            token_usage=extract_token_usage(chunks),
            error=error_msg,
        )

    reply = "".join(reply_parts) or "Sorry, I could not process your request."
    await send_whatsapp_reply(sender, reply)
    return JSONResponse({"status": "ok"})


@app.get("/logs")
async def get_logs(n: int = 50, date: str = ""):
    """Return the last n activity log records as JSON.

    Query params:
        n     number of tail records to return (default 50, max 500)
        date  log date in YYYY-MM-DD format; defaults to today (UTC)
    """
    from activity_logger import _LOG_DIR
    from datetime import datetime, timezone
    date_str = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log_file = _LOG_DIR / f"activity_{date_str}.jsonl"
    if not log_file.exists():
        return JSONResponse({"date": date_str, "records": []})
    lines = log_file.read_text(encoding="utf-8").splitlines()
    tail = lines[-min(n, 500):]
    records = [json.loads(l) for l in tail if l.strip()]
    return JSONResponse({"date": date_str, "count": len(records), "records": records})


@app.get("/introduce")
async def introduce(thread_id: str = "default"):
    """Stream the agent's self-introduction; called by the frontend on page load."""
    config = {"configurable": {"thread_id": thread_id}}
    seed = generate_seed()
    seeded_agent = create_react_agent(
        model=_llm.bind(seed=seed),
        tools=_tools,
        checkpointer=_checkpointer,
        prompt=build_prompt,
    )

    async def stream():
        t0 = time.monotonic()
        chunks: list = []
        full = ""
        error_msg = None
        try:
            async for chunk in seeded_agent.astream(
                {"messages": [("user", "Please briefly introduce yourself and list your capabilities.")]},
                config=config,
            ):
                chunks.append(chunk)
                for node_output in chunk.values():
                    for msg in node_output.get("messages", []):
                        if hasattr(msg, "content") and msg.content and msg.type == "ai":
                            full += msg.content
                            yield msg.content
        except Exception as e:
            error_msg = str(e)
            yield f"[Error: {e}]"
        finally:
            write_record(
                session_id=thread_id,
                channel="introduce",
                user_message="[introduce]",
                agent_response=full,
                tools_called=extract_tools_called(chunks),
                seed=seed,
                latency_ms=(time.monotonic() - t0) * 1000,
                token_usage=extract_token_usage(chunks),
                error=error_msg,
            )

    return StreamingResponse(stream(), media_type="text/plain")


@app.post("/chat")
async def chat(req: ChatRequest):
    """Stream agent responses to the web UI."""
    config = {"configurable": {"thread_id": req.thread_id}}
    seed = generate_seed()
    seeded_agent = create_react_agent(
        model=_llm.bind(seed=seed),
        tools=_tools,
        checkpointer=_checkpointer,
        prompt=build_prompt,
    )

    async def stream():
        t0 = time.monotonic()
        chunks: list = []
        full = ""
        error_msg = None
        try:
            async for chunk in seeded_agent.astream(
                {"messages": [("user", req.message)]}, config=config
            ):
                chunks.append(chunk)
                for node_output in chunk.values():
                    for msg in node_output.get("messages", []):
                        if hasattr(msg, "content") and msg.content and msg.type == "ai":
                            full += msg.content
                            yield msg.content
        except Exception as e:
            error_msg = str(e)
            yield f"[Error: {e}]"
        finally:
            write_record(
                session_id=req.thread_id,
                channel="web",
                user_message=req.message,
                agent_response=full,
                tools_called=extract_tools_called(chunks),
                seed=seed,
                latency_ms=(time.monotonic() - t0) * 1000,
                token_usage=extract_token_usage(chunks),
                error=error_msg,
            )

    return StreamingResponse(stream(), media_type="text/plain")
