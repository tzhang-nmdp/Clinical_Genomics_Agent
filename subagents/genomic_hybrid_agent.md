# Genomic Hybrid Agent — `genomic_search`

## Overview

`GenomicSubAgent` is a LangChain `BaseTool` that answers questions about a
genomic data catalog stored in a local DuckDB database.  
It combines three retrieval strategies and fuses their results with
Reciprocal Rank Fusion (RRF):

| Layer | Method | Best for |
|-------|--------|----------|
| SQL | Guarded DuckDB SELECT / WITH | Exact counts, filters, joins, aggregations, locations, dates |
| BM25 | Okapi BM25 over `bm25_entities.json` | Exact terms: VCF, BAM, WGS, gene names, project IDs, storage paths |
| Vector | FAISS + `biobert-v1.1` embeddings | Conceptual / semantic queries, synonyms, narrative descriptions |

The LLM planner (`gemma4:e4b` via Ollama) selects one of three routes —
`sql`, `retrieval`, or `hybrid` — before executing retrieval.

---

## Tool Registration

```python
# subagents/__init__.py
from subagents.genomic_hybrid_agent import GenomicSubAgent

# server.py — init_agent()
genomic_tool = GenomicSubAgent()
agent = create_react_agent(model=llm, tools=[..., genomic_tool], ...)
```

**Tool name exposed to the ReAct agent:** `genomic_search`

---

## Configuration

All paths and tunables are read from environment variables with hardcoded
fallbacks:

| Variable | Default | Purpose |
|----------|---------|---------|
| `GENOMIC_DB` | `tools/genomic_digest/genomic_knowledge.duckdb` | DuckDB catalog file |
| `GENOMIC_FAISS` | `tools/genomic_digest/faiss_index` | FAISS index directory |
| `GENOMIC_BM25` | `tools/genomic_digest/bm25_entities.json` | BM25 entity records |
| `EMBED_MODEL` | `biobert-v1.1` (local path) | HuggingFace embedding model |
| `OLLAMA_MODEL` | `gemma4:e4b` | Planner + synthesizer LLM |
| `EMBED_DEVICE` | `cpu` | `cpu` or `cuda` |
| `TOP_K` | `8` | Final fused hits returned |
| `FUSION_K` | `60` | RRF constant (higher = softer rank penalty) |
| `MAX_SQL_ROWS` | `200` | Row cap on SQL results |

---

## Database Schema (genomic_knowledge.duckdb)

```
annotation_databases(row_id, document_id, db_name, explanation, page_number)
cohorts(cohort_id, document_id, dataset_name, sample_size, clinical_setup,
        sequencing_facility, page_number)
documents(document_id, file_name, source_url, title, extraction_model, page_count)
extraction_audit(audit_id, document_id, page_number, extracted_fact_count,
                 extracted_cohort_count, extracted_result_count,
                 extracted_link_count, table_count, text_char_count)
facts(fact_id, document_id, fact_type, name, abbreviation, description,
      category, value, unit, url, status, page_number, section)
links(link_id, document_id, label, url, link_type, page_number, section)
narrative_sections(section_id, document_id, heading, section_text,
                   page_start, page_end)
study_results(result_id, document_id, primary_outcome, cohort_size,
              sample_source, omics_platform, analytic_subject,
              primary_analysis, entity_name, entity_type,
              genomic_coordinates, page_number)
```

---

## Query Routing Logic

The planner LLM returns a JSON object:

```json
{
  "route": "sql | retrieval | hybrid",
  "sql": "<DuckDB SELECT or WITH query, or empty string>",
  "lexical_query": "<exact-term query for BM25>",
  "semantic_query": "<conceptual intent for vector search>"
}
```

### Route selection guide

| User intent | Route |
|-------------|-------|
| Count rows, filter by value, join tables, aggregate | `sql` |
| Describe a dataset, find synonyms, narrative summaries | `retrieval` |
| Mixed — e.g. "how many WGS samples and what platform?" | `hybrid` |

---

## SQL Guard

Only `SELECT` and `WITH` statements are permitted. The following keywords
are blocked and will raise a `ValueError`:

`INSERT UPDATE DELETE DROP ALTER CREATE ATTACH DETACH COPY EXPORT IMPORT
INSTALL LOAD PRAGMA CALL VACUUM TRUNCATE MERGE`

Multi-statement SQL (containing `;`) is also rejected.

---

## BM25 Record Format

Each record in `bm25_entities.json` must contain at minimum:

```json
{
  "doc_id": "<unique id>",
  "text":   "<searchable text>",
  "metadata": { ... }
}
```

`_bm25_search` normalises the shape so `doc_id` is always available inside
`metadata`, which is required by the RRF fusion step.

---

## Offline Index Setup (run once)

```bash
# 1. Ingest a CSV / TSV / Parquet file into DuckDB
python -m subagents.genomic_hybrid_agent ingest \
    --input catalog.csv \
    --table dataset_catalog

# 2. Build FAISS + BM25 indexes from the ingested table
python -m subagents.genomic_hybrid_agent build-index \
    --table dataset_catalog \
    --id-cols   dataset_id dataset_name \
    --text-cols dataset_name description data_type host \
                old_location new_location project_id publications
```

The current indexes were built by `tools/digest_genomic_pdf.py` from PDF
extraction output and cover the tables listed in the schema above.

---

## When to Use This Tool

Use `genomic_search` when the user asks about:

- Genomic datasets, cohorts, or study results
- Data types: VCF, BAM, WGS, RRBS, methylation, whole-genome sequencing
- Reference genomes (hg38, GRCh38, etc.)
- Sequencing platforms or omics technologies
- Storage locations, project IDs, or publication references
- Specific genes, variants, or genomic coordinates
- Clinical setup, sample sources, or sequencing facilities

---

## Answer Format

- **SQL evidence** — cite as exact structured data; include the SQL used at the end of the response.
- **Retrieval evidence** — cite each statement as `[source_table: identifiers]`.
- **Truncated SQL** — if `truncated: true`, state that the result was capped at `MAX_SQL_ROWS` rows.
- **Insufficient evidence** — state explicitly what information is missing rather than guessing.
- Do **not** invent counts, locations, dates, links, clinical meaning, or dataset content.
