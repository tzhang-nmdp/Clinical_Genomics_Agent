"""Genomic catalog subagent: guarded DuckDB SQL + BM25 + FAISS hybrid retrieval.

Offline setup (run once before starting the server):
    python -m subagents.genomic_hybrid_agent ingest --input catalog.csv --table dataset_catalog
    python -m subagents.genomic_hybrid_agent build-index --table dataset_catalog \
      --id-cols dataset_id dataset_name \
      --text-cols dataset_name description data_type host old_location new_location project_id publications
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from click import prompt
import duckdb
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.tools import BaseTool
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import Field
from langchain_openai import ChatOpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_BASE = Path(__file__).parent.parent  # clinical_genomic_agent root
# Path to the DuckDB database file; overridable via environment variable.
DB_PATH    = Path(os.getenv("GENOMIC_DB",   str(_BASE / "tools/genomic_digest_biobert/genomic_knowledge.duckdb")))

# Directory where the FAISS vector index is persisted.
INDEX_PATH = Path(os.getenv("GENOMIC_FAISS", str(_BASE / "tools/genomic_digest_biobert/faiss_index")))

# JSON file storing the BM25 corpus (tokenized document texts + metadata).
BM25_PATH  = Path(os.getenv("GENOMIC_BM25",  str(_BASE / "tools/genomic_digest_biobert/bm25_entities.json")))

# Local path or HuggingFace model ID for the sentence embedding model.
EMBED_MODEL  = os.getenv("EMBED_MODEL", str(_BASE / "Sentence-BioBert-snli"))

# Base URL of the llama.cpp OpenAI-compatible server used as the LLM backend.
LLAMA2_SERVER_URL = os.getenv("LLAMA2_SERVER_URL", "http://localhost:8081/v1")

# Device for embedding inference: "cpu", "cuda", or "mps".
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cpu")

# Number of top results to return from each retrieval stage.
TOP_K      = int(os.getenv("TOP_K",      "8"))

# RRF constant k: higher values reduce the impact of rank differences.
FUSION_K   = int(os.getenv("FUSION_K",   "60"))

# Maximum rows fetched from DuckDB per query to prevent runaway result sets.
MAX_SQL_ROWS = int(os.getenv("MAX_SQL_ROWS", "200"))

# Regex that matches any SQL keyword that could mutate or expose the database.
# Used as a guard in _validate_sql to enforce read-only access.
DENIED_SQL = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|copy|export|import|"
    r"install|load|pragma|call|vacuum|truncate|merge)\b", re.I
)

# Tokenization pattern: alphanumeric runs optionally joined by delimiters
# common in genomic identifiers (e.g. "hg38", "chr1:12345", "rs_123456").
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-_.*:/][A-Za-z0-9]+)*")

LLAMA_MODEL_NAME = os.getenv("LLAMA_MODEL_NAME", "gemma-4-E2B-it-Q4_0")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def quote_ident(name: str) -> str:
    """Wrap a SQL identifier in double-quotes after validating it is safe.

    Raises ValueError if the name contains characters outside [A-Za-z0-9_]
    or does not start with a letter/underscore, preventing SQL injection via
    table or column names.
    """
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"Unsafe identifier: {name!r}")
    return f'"{name}"'


def _text(value: Any) -> str:
    """Coerce any scalar value to a clean string for document construction.

    - None  → "unknown"  (avoids literal "None" in indexed text)
    - float → compact scientific notation (avoids trailing zeros)
    - other → str().strip()
    """
    if value is None:
        return "unknown"
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value).strip()


def tokenize(value: str) -> list[str]:
    """Extract and lowercase tokens from a string using TOKEN_RE.

    Used to build the BM25 corpus and to tokenize queries at search time,
    ensuring consistent vocabulary between indexing and retrieval.
    """
    return [x.casefold() for x in TOKEN_RE.findall(value)]


def _embedding_model() -> HuggingFaceEmbeddings:
    """Instantiate the sentence embedding model.

    Embeddings are L2-normalised so that inner-product == cosine similarity,
    which is required by FAISS IndexFlatIP used internally by LangChain FAISS.
    batch_size=32 balances throughput and memory on CPU.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": EMBED_DEVICE},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )


# def _chat_model() -> ChatOllama:
#     """Instantiate the LLM client pointing at the local llama.cpp server.

#     temperature=0 ensures deterministic JSON output from the planner.
#     openai_api_key="none" satisfies the SDK's required-field check without
#     sending a real credential to the local server.
#     """
#     return ChatOllama(model=OLLAMA_MODEL, temperature=0, num_ctx=16384)

def _chat_model() -> ChatOpenAI:
    """Instantiate the LLM client pointing at the local llama.cpp server.

    temperature=0 ensures deterministic JSON output from the planner.
    openai_api_key="none" satisfies the SDK's required-field check without
    sending a real credential to the local server.
    """
    return ChatOpenAI(
        model=LLAMA_MODEL_NAME,
        temperature=0,
        openai_api_base=LLAMA2_SERVER_URL,
        openai_api_key="none",
    )
    
# def _chat_model():
#     """Instantiate the LLM client pointing at the local llama.cpp server.

#     temperature=0 ensures deterministic JSON output from the planner.
#     openai_api_key="none" satisfies the SDK's required-field check without
#     sending a real credential to the local server.
#     """
#     tokenizer = AutoTokenizer.from_pretrained(
#     model_path,
#     trust_remote_code=True
# )
#     return AutoModelForCausalLM.from_pretrained(
#     model_path,
#     device_map="auto",
#     trust_remote_code=True
# )


# ---------------------------------------------------------------------------
# Retrieval internals
# ---------------------------------------------------------------------------

def _load_bm25() -> tuple[list[dict[str, Any]], BM25Okapi]:
    """Load the BM25 corpus from disk and build an in-memory BM25Okapi index.

    The JSON file contains a list of records, each with "text" and "metadata".
    BM25Okapi is rebuilt on every call; for large corpora consider caching.

    Returns:
        records: raw list of dicts (text + metadata per document)
        index:   fitted BM25Okapi object ready for get_scores()
    """
    records = json.loads(BM25_PATH.read_text(encoding="utf-8"))
    return records, BM25Okapi([tokenize(r["text"]) for r in records])


def _bm25_search(query: str, k: int) -> list[dict[str, Any]]:
    """Run BM25 retrieval and return the top-k positively-scored documents.

    Documents with score <= 0 are excluded because they have no term overlap
    with the query and would only add noise to the RRF fusion step.

    Each result dict contains:
        text, metadata (with doc_id), bm25_score, rank (1-based)
    """
    records, index = _load_bm25()
    scores = index.get_scores(tokenize(query))
    # Sort all document indices by descending BM25 score, take top k.
    order = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)[:k]
    results = []
    for rank, i in enumerate(order, 1):
        if scores[i] <= 0:
            continue
        r = records[i]
        # Ensure metadata dict always contains doc_id for RRF deduplication.
        meta = dict(r.get("metadata") or {})
        meta.setdefault("doc_id", r.get("doc_id", str(i)))
        results.append({"text": r["text"], "metadata": meta,
                        "bm25_score": float(scores[i]), "rank": rank})
    return results


def _vector_search(query: str, k: int) -> list[dict[str, Any]]:
    """Run dense vector retrieval against the FAISS index.

    Loads the persisted FAISS index from INDEX_PATH, embeds the query with
    the same model used at index time, and returns the k nearest neighbours
    with their L2 distances (lower = more similar for normalised vectors).

    Each result dict contains:
        text, metadata (with doc_id), vector_distance, rank (1-based)
    """
    store = FAISS.load_local(str(INDEX_PATH), _embedding_model(), allow_dangerous_deserialization=True)
    hits = store.similarity_search_with_score(query, k=k)
    return [{"text": d.page_content, "metadata": d.metadata,
             "vector_distance": float(score), "rank": rank}
            for rank, (d, score) in enumerate(hits, 1)]


def _rrf(bm25_hits: list[dict[str, Any]], vector_hits: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    """Fuse BM25 and vector results using Reciprocal Rank Fusion (RRF).

    RRF score for document d = sum over retrievers of 1 / (FUSION_K + rank_d).
    Documents appearing in both result lists receive contributions from both
    retrievers, naturally boosting cross-validated hits.

    Deduplication is performed by doc_id so each document appears only once
    in the fused list regardless of how many retrievers returned it.

    Returns the top-k documents sorted by descending rrf_score, each annotated
    with matched_by listing which retrievers contributed ("bm25", "vector").
    """
    fused: dict[str, dict[str, Any]] = {}
    for source, hits in (("bm25", bm25_hits), ("vector", vector_hits)):
        for rank, item in enumerate(hits, 1):
            doc_id = item["metadata"]["doc_id"]
            # Create entry on first encounter; accumulate score on subsequent ones.
            entry = fused.setdefault(doc_id, {
                "text": item["text"], "metadata": item["metadata"],
                "rrf_score": 0.0, "matched_by": []
            })
            entry["rrf_score"] += 1.0 / (FUSION_K + rank)
            entry["matched_by"].append(source)
    return sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)[:k]


def _schema_text(con: duckdb.DuckDBPyConnection) -> str:
    """Render the DuckDB 'main' schema as a compact one-line-per-table string.

    Format: table_name(col1 TYPE, col2 TYPE, ...)
    This compact representation is injected into the planner prompt so the LLM
    can generate valid SQL without hallucinating column names.
    """
    rows = con.execute("""
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema='main'
        ORDER BY table_name, ordinal_position
    """).fetchall()
    grouped: dict[str, list[str]] = {}
    for table, column, dtype in rows:
        grouped.setdefault(table, []).append(f"{column} {dtype}")
    return "\n".join(f"{t}({', '.join(cols)})" for t, cols in grouped.items())


def _validate_sql(sql: str) -> str:
    """Enforce read-only SQL before execution.

    Checks:
    1. No semicolons — prevents multi-statement injection.
    2. Must start with SELECT or WITH — only read queries allowed.
    3. No denied keywords (INSERT, DROP, PRAGMA, etc.) anywhere in the query.

    Raises ValueError with a descriptive message if any check fails.
    Returns the cleaned SQL string (stripped, trailing semicolon removed).
    """
    sql = sql.strip().rstrip(";")
    if ";" in sql or not re.match(r"^(select|with)\b", sql, re.I) or DENIED_SQL.search(sql):
        raise ValueError("Blocked non-read-only or multi-statement SQL")
    return sql


def _run_sql(sql: str) -> dict[str, Any]:
    """Validate and execute a SQL query against the read-only DuckDB database.

    Opens a fresh read-only connection per call to avoid shared-state issues
    in concurrent server environments. Fetches at most MAX_SQL_ROWS + 1 rows
    to detect truncation without loading the full result set into memory.

    Returns a dict with:
        sql:       the validated query string
        columns:   list of column names
        rows:      up to MAX_SQL_ROWS result rows
        truncated: True if the result set exceeded MAX_SQL_ROWS
    """
    sql = _validate_sql(sql)
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        cur = con.execute(sql)
        columns = [d[0] for d in cur.description]
        rows = cur.fetchmany(MAX_SQL_ROWS + 1)
    finally:
        con.close()
    truncated = len(rows) > MAX_SQL_ROWS
    return {"sql": sql, "columns": columns, "rows": rows[:MAX_SQL_ROWS], "truncated": truncated}


def _extract_json(raw: str) -> dict[str, Any]:
    """Extract the first JSON object from a raw LLM response string.

    The LLM may wrap the JSON in markdown fences or add preamble text.
    The regex greedily matches from the first '{' to the last '}' to capture
    the complete object even when the response contains surrounding prose.

    Raises ValueError if no JSON object is found.
    """
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        raise ValueError(f"Planner returned invalid JSON: {raw}")
    return json.loads(match.group())


def _plan(question: str, schema: str) -> dict[str, Any]:
    """Ask the LLM to route the question and generate retrieval parameters.

    The planner returns a JSON object with four fields:
        route:          "sql" | "retrieval" | "hybrid"
        sql:            a read-only DuckDB SELECT/WITH query (or "")
        lexical_query:  query string optimised for BM25 (exact terms preserved)
        semantic_query: query string optimised for vector search (conceptual)

    The schema string is injected so the LLM can reference real table/column
    names when generating SQL, reducing hallucination.
    """
    prompt = f"""Route a question for a genomic catalog assistant.
Return JSON only with: route, sql, lexical_query, semantic_query.
route must be one of sql, retrieval, hybrid.
SQL is for exact counts, locations, dates, filters, joins, and aggregations.
Retrieval means BM25 plus vector retrieval for descriptions, identifiers, synonyms, and narrative summaries.
Hybrid means SQL plus BM25 plus vector retrieval.
If route includes SQL, generate one read-only DuckDB SELECT or WITH query using only the schema.
For non-aggregate detail SQL, use LIMIT {MAX_SQL_ROWS}.
lexical_query should preserve exact terms such as VCF, BAM, WGS, project IDs, genes, alleles, and storage paths.
semantic_query should express the conceptual intent.
Use empty strings for unused fields.

Schema:\n{schema}\n\nQuestion: {question}
"""
    return _extract_json(_chat_model().invoke(prompt).content)

def _synthesize(question: str, route: str, sql_result: dict[str, Any] | None,
                retrieval: list[dict[str, Any]] | None) -> str:
    """Generate a grounded natural-language answer from structured evidence.

    Combines SQL results (exact, structured) and fused retrieval hits
    (descriptive, narrative) into a single evidence bundle passed to the LLM.

    The prompt instructs the LLM to:
    - Cite retrieved statements with [source_table: identifiers].
    - Acknowledge truncated SQL output rather than silently omitting rows.
    - Admit when evidence is insufficient rather than hallucinating.
    - Append an 'SQL used' section only when SQL was actually executed.
    """
    evidence = {"route": route, "sql_result": sql_result, "retrieval": retrieval}
    prompt = f"""Answer a user question about a genomic data catalog using only the evidence below.
Treat SQL output as exact structured evidence. Treat fused retrieval as descriptive evidence.
Do not invent counts, locations, dates, links, clinical meaning, or dataset content.
For each retrieved statement, cite the entity as [source_table: identifiers].
If SQL output was truncated, state that. If evidence is insufficient, state what is missing.
End with an 'SQL used' section only when SQL was executed.

Question: {question}\nEvidence:\n{json.dumps(evidence, default=str, indent=2)}
"""
    return _chat_model().invoke(prompt).content

# ---------------------------------------------------------------------------
# SubAgent (BaseTool) — used by the LangGraph ReAct agent in server.py
# ---------------------------------------------------------------------------

class GenomicSubAgent(BaseTool):
    """LangChain BaseTool wrapping the full hybrid retrieval pipeline.

    Registered as "genomic_search" so the ReAct agent can invoke it by name.
    The agent passes a natural-language query; this tool handles routing,
    retrieval, and synthesis internally and returns a grounded answer string.
    """
    name: str = "genomic_search"
    description: str = (
        "Search the genomic dataset catalog using hybrid SQL + BM25 + vector retrieval. "
        "Use this tool for questions about genomic datasets, data types (VCF, BAM, WGS, methylation, etc.), "
        "storage locations, project IDs, publications, and dataset descriptions."
    )

    # Number of top fused results to pass to the synthesizer.
    top_k: int = Field(default=TOP_K)

    def _run(self, query: str) -> str:
        """Execute the hybrid retrieval pipeline for a natural-language query.

        Pipeline steps:
        1. Read the live DB schema to ground the planner prompt.
        2. Call _plan() to get route + SQL + lexical/semantic queries from LLM.
        3. Run SQL if route is "sql" or "hybrid" and a query was generated.
        4. Run BM25 + vector retrieval and fuse with RRF if route includes retrieval.
        5. Call _synthesize() to produce a grounded natural-language answer.

        Returns an early error string if the catalog has not been ingested yet.
        """
        con = duckdb.connect(str(DB_PATH), read_only=True)
        try:
            schema = _schema_text(con)
        finally:
            con.close()

        # Guard: if the DB is empty or missing, fail fast with a helpful message.
        if not schema:
            return "Genomic catalog not available. Run ingest and build-index first."

        p = _plan(query, schema)
        route = str(p.get("route", "hybrid")).lower()
        # Fall back to hybrid if the LLM returns an unrecognised route value.
        if route not in {"sql", "retrieval", "hybrid"}:
            route = "hybrid"

        # Execute SQL only when the planner produced a non-empty query.
        sql_result = _run_sql(str(p["sql"])) if route in {"sql", "hybrid"} and p.get("sql") else None

        retrieval = None
        if route in {"retrieval", "hybrid"}:
            # Use planner-generated queries; fall back to raw user query if empty.
            lexical  = str(p.get("lexical_query") or query)
            semantic = str(p.get("semantic_query") or query)
            # Fetch 2× top_k from each retriever before fusing to improve recall.
            retrieval = _rrf(
                _bm25_search(lexical,  self.top_k * 2),
                _vector_search(semantic, self.top_k * 2),
                self.top_k,
            )

        return _synthesize(query, route, sql_result, retrieval)


# ---------------------------------------------------------------------------
# Offline CLI — ingest data and build indexes before starting the server
# ---------------------------------------------------------------------------

def _make_doc(table: str, record: dict[str, Any], id_cols: list[str], text_cols: list[str]) -> Document:
    """Build a LangChain Document from a single database row for indexing.

    The page_content is structured as:
        "Genomic catalog entity\n<identity line>\n\n<field: value lines>"

    A stable doc_id is derived from a SHA-256 hash of the table name and
    identity column values, ensuring consistent deduplication across re-indexes.

    Args:
        table:     source table name (stored in metadata as source_table)
        record:    dict mapping column name → value for one row
        id_cols:   columns used to form the human-readable identity line
        text_cols: columns whose values are concatenated into the document body
    """
    identity = " | ".join(f"{c}: {_text(record.get(c))}" for c in id_cols)
    body = "\n".join(f"{c}: {_text(record.get(c))}" for c in text_cols)
    raw_id = f"{table}|" + "|".join(_text(record.get(c)) for c in id_cols)
    # Truncate hash to 24 hex chars (96 bits) — sufficient for collision resistance
    # at catalog scale while keeping metadata compact.
    doc_id = hashlib.sha256(raw_id.encode()).hexdigest()[:24]
    metadata = {c: _text(record.get(c)) for c in id_cols}
    metadata.update({"source_table": table, "doc_id": doc_id})
    return Document(page_content=f"Genomic catalog entity\n{identity}\n\n{body}", metadata=metadata)


def ingest(input_path: Path, table: str) -> None:
    """Load a CSV, TSV, TXT, or Parquet file into a DuckDB table.

    Uses DuckDB's native read_csv_auto / read_parquet functions for efficient
    bulk loading. The table is replaced entirely on each call (CREATE OR REPLACE)
    so re-ingestion is idempotent.

    Args:
        input_path: path to the source data file
        table:      target DuckDB table name (validated by quote_ident)
    """
    table_q = quote_ident(table)
    con = duckdb.connect(str(DB_PATH))
    try:
        suffix = input_path.suffix.lower()
        if suffix == ".parquet":
            con.execute(f"CREATE OR REPLACE TABLE {table_q} AS SELECT * FROM read_parquet(?)", [str(input_path)])
        elif suffix in {".csv", ".tsv", ".txt"}:
            delim = "\t" if suffix in {".tsv", ".txt"} else ","
            con.execute(
                f"CREATE OR REPLACE TABLE {table_q} AS SELECT * FROM read_csv_auto(?, delim=?, header=true)",
                [str(input_path), delim],
            )
        else:
            raise ValueError("Input must be CSV, TSV, TXT, or Parquet")
        count = con.execute(f"SELECT COUNT(*) FROM {table_q}").fetchone()[0]
        print(f"Loaded {count:,} rows into {DB_PATH}:{table}")
    finally:
        con.close()


def build_index(table: str, id_cols: list[str], text_cols: list[str], max_docs: int | None) -> None:
    """Build and persist the FAISS vector index and BM25 JSON corpus.

    Reads the specified columns from DuckDB, converts each row to a Document
    via _make_doc, then:
    - Embeds all documents and saves a FAISS index to INDEX_PATH.
    - Serialises the raw text + metadata to BM25_PATH as JSON for BM25Okapi.

    Both artefacts must exist before the server can handle retrieval queries.

    Args:
        table:    source DuckDB table name
        id_cols:  columns used as document identifiers (appear in metadata)
        text_cols: columns whose text is indexed for search
        max_docs: optional row limit for testing with a subset of the catalog
    """
    # Deduplicate columns while preserving order (id_cols first).
    selected = list(dict.fromkeys(id_cols + text_cols))
    sql = f"SELECT {', '.join(quote_ident(c) for c in selected)} FROM {quote_ident(table)}"
    if max_docs:
        sql += f" LIMIT {int(max_docs)}"
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        cur = con.execute(sql)
        columns = [d[0] for d in cur.description]
        docs = [_make_doc(table, dict(zip(columns, row)), id_cols, text_cols) for row in cur.fetchall()]
    finally:
        con.close()
    if not docs:
        raise RuntimeError("No index documents generated")
    # Persist FAISS index; embedding model must match the one used at query time.
    FAISS.from_documents(docs, _embedding_model()).save_local(str(INDEX_PATH))
    # Persist BM25 corpus as JSON; rebuilt in-memory on each search call.
    BM25_PATH.write_text(json.dumps(
        [{"text": d.page_content, "metadata": d.metadata} for d in docs],
        ensure_ascii=False, indent=2,
    ), encoding="utf-8")
    print(f"Indexed {len(docs):,} summaries in FAISS and BM25")


def main() -> None:
    """CLI entry point for offline data preparation.

    Subcommands:
        ingest       -- load a data file into DuckDB
        build-index  -- embed documents and write FAISS + BM25 artefacts
    """
    parser = argparse.ArgumentParser(description="Genomic catalog: ingest and index data")
    sub = parser.add_subparsers(dest="command", required=True)

    # 'ingest' subcommand: load raw data file into DuckDB.
    a = sub.add_parser("ingest")
    a.add_argument("--input", type=Path, required=True)
    a.add_argument("--table", required=True)

    # 'build-index' subcommand: create FAISS and BM25 search indexes.
    b = sub.add_parser("build-index")
    b.add_argument("--table", required=True)
    b.add_argument("--id-cols", nargs="+", required=True)
    b.add_argument("--text-cols", nargs="+", required=True)
    b.add_argument("--max-docs", type=int)

    args = parser.parse_args()
    if args.command == "ingest":
        ingest(args.input, args.table)
    else:
        build_index(args.table, args.id_cols, args.text_cols, args.max_docs)


if __name__ == "__main__":
    main()
