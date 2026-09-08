#!/usr/bin/env python3
"""Digest a genomics PDF into DuckDB tables, BM25 entities, and FAISS vector documents.

Routing targets:
  * structured facts and table rows -> DuckDB (~80%)
  * exact genomic identifiers -> BM25 corpus (~10%)
  * narrative/methodological sections -> vector documents (~10%)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import pdfplumber
import tabula
from pydantic import BaseModel, Field, ValidationError, field_validator
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

PAGE_URL_RE = re.compile(r"https?://\S+")
TOKEN_RE = re.compile(
    r"(?:chr(?:[0-9XYM]+):[0-9,]+(?:-[0-9,]+)?(?:[:][A-Za-z<>*-]+)?)"
    r"|(?:rs\d+)"
    r"|(?:[A-Z][A-Z0-9.-]{1,20}(?:\*[0-9:]+)?)"
    r"|(?:[A-Za-z0-9]+(?:[-_./:*][A-Za-z0-9<>+-]+)+)"
    r"|(?:[A-Za-z0-9]+)",
    re.I,
)


def _coerce_str_fields(data: Any) -> Any:
    if not isinstance(data, dict):
        return data
    return {k: str(v) if isinstance(v, (int, float)) else v for k, v in data.items()}


class Fact(BaseModel):
    model_config = {"populate_by_name": True}
    fact_type: str
    name: str
    abbreviation: str | None = None
    description: str | None = None
    category: str | None = None
    value: str | None = None
    unit: str | None = None
    url: str | None = None
    status: str | None = None
    section: str

    @classmethod
    def model_validate(cls, obj: Any, **kw: Any) -> "Fact":
        return super().model_validate(_coerce_str_fields(obj), **kw)


class Cohort(BaseModel):
    dataset_name: str
    sample_size: str | None = None
    clinical_setup: str | None = None
    sequencing_facility: str | None = None

    @classmethod
    def model_validate(cls, obj: Any, **kw: Any) -> "Cohort":
        return super().model_validate(_coerce_str_fields(obj), **kw)


class StudyResult(BaseModel):
    primary_outcome: str | None = None
    cohort_size: str | None = None
    sample_source: str | None = None
    omics_platform: str | None = None
    analytic_subject: str | None = None
    primary_analysis: str | None = None
    entity_name: str
    entity_type: str
    genomic_coordinates: str | None = None

    @classmethod
    def model_validate(cls, obj: Any, **kw: Any) -> "StudyResult":
        return super().model_validate(_coerce_str_fields(obj), **kw)


class LinkRecord(BaseModel):
    label: str | None = None
    url: str
    link_type: str
    section: str

    @classmethod
    def model_validate(cls, obj: Any, **kw: Any) -> "LinkRecord":
        return super().model_validate(_coerce_str_fields(obj), **kw)


class DocExtraction(BaseModel):
    facts: list[Fact] = Field(default_factory=list)
    cohorts: list[Cohort] = Field(default_factory=list)
    study_results: list[StudyResult] = Field(default_factory=list)
    links: list[LinkRecord] = Field(default_factory=list)
    narrative_sections: list[dict[str, str]] = Field(default_factory=list)

    @field_validator("facts", "cohorts", "study_results", "links", mode="before")
    @classmethod
    def coerce_to_list(cls, v: Any, info: Any) -> Any:
        if isinstance(v, dict):
            v = list(v.values()) if v else []
        if not isinstance(v, list):
            return []
        items = [_coerce_str_fields(i) if isinstance(i, dict) else i for i in v]
        if info.field_name == "facts":
            items = [i for i in items if isinstance(i, dict) and i.get("name") and i.get("section")]
        return items

    @field_validator("narrative_sections", mode="before")
    @classmethod
    def coerce_narrative_sections(cls, v: Any) -> Any:
        if isinstance(v, dict):
            v = list(v.values()) if v else []
        if not isinstance(v, list):
            return []
        return [{k: str(val) for k, val in i.items()} if isinstance(i, dict) else i for i in v]


@dataclass
class PageData:
    page_number: int
    text: str
    tables: list[list[list[str | None]]]



def clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\\_", "_")
    text = re.sub(r"(?m)^\s*\d+/\d+\s*$", "", text)
    text = re.sub(r"(?m)^https://confluence\.nmdp\.org/pages/viewpage\.action\?pageI?d=\d+\S*\s*$", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_header_row(row: list[str | None]) -> bool:
    cells = [c for c in row if c and c.strip()]
    if not cells:
        return False
    header_like = sum(1 for c in cells if c.strip() == c.strip().title() or c.strip().isupper())
    return header_like / len(cells) >= 0.6


def _is_layout_table(table: list[list[str | None]]) -> bool:
    non_empty = [c for row in table for c in row if c and c.strip()]
    if not non_empty:
        return False
    if sum(1 for c in non_empty if len(c) > 500) / len(non_empty) > 0.5:
        return True
    if all(len(row) == 1 for row in table):
        first = (table[0][0] or "").strip()
        if " / " in first or re.search(r"viewpage\.action", first):
            return True
    return False


def _tabula_tables_for_page(pdf_path: Path, page_number: int) -> list[list[list[str | None]]]:
    dfs = tabula.read_pdf(str(pdf_path), pages=page_number, multiple_tables=True, silent=True)
    result = []
    for df in dfs:
        header = [None if str(c) in ("", "nan") else str(c) for c in df.columns.tolist()]
        rows = [[None if str(c) in ("", "nan") else str(c) for c in row] for row in df.values.tolist()]
        table = [header] + rows
        if not all(not (c and c.strip()) for row in table for c in row) and not _is_layout_table(table):
            result.append(table)
    return result


def extract_pdf(pdf_path: Path) -> list[PageData]:
    pages: list[PageData] = []
    with pdfplumber.open(pdf_path) as pdf:
        for number, page in enumerate(pdf.pages, 1):
            text = clean_text(page.extract_text(x_tolerance=2, y_tolerance=3) or "")
            tables = _tabula_tables_for_page(pdf_path, number)
            pages.append(PageData(number, text, tables))
    return _merge_split_tables(pages)


def _merge_split_tables(pages: list[PageData]) -> list[PageData]:
    anchor_idx: int | None = None
    for i in range(len(pages) - 1):
        nxt = pages[i + 1]
        if not nxt.tables:
            anchor_idx = None
            continue
        owner_idx = anchor_idx if anchor_idx is not None else i
        owner = pages[owner_idx]
        if not owner.tables:
            anchor_idx = None
            continue
        tail, head = owner.tables[-1], nxt.tables[0]
        if not tail or not head or len(tail[0]) != len(head[0]):
            anchor_idx = None
            continue
        continuation = head[1:] if _is_header_row(head[0]) else head
        if not continuation:
            anchor_idx = None
            continue
        owner.tables[-1] = tail + continuation
        nxt.tables = nxt.tables[1:]
        anchor_idx = owner_idx
    return pages


def split_sections(pages: list[PageData]) -> list[dict[str, Any]]:
    """Split full-document text into sections using a generic heading detector (ALL-CAPS or Title Case line ending with ':')."""
    heading_re = re.compile(r"(?m)^([A-Z][A-Za-z0-9 /,()-]{2,60}):\s*$")
    full_text = "\n".join(p.text for p in pages)
    sections: list[dict[str, Any]] = []
    current = {"heading": "Introduction", "text": ""}
    for line in full_text.splitlines():
        m = heading_re.match(line.strip())
        if m:
            if current["text"].strip():
                sections.append(current)
            current = {"heading": m.group(1).strip(), "text": ""}
        else:
            current["text"] += line + "\n"
    if current["text"].strip():
        sections.append(current)
    return sections


def table_as_markdown(table: list[list[str | None]]) -> str:
    return "\n".join(" | ".join(clean_text(cell or "") for cell in row) for row in table)


def _repair_json(raw: str) -> str:
    raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw.strip())
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and len(obj) == 1:
            inner = next(iter(obj.values()))
            if isinstance(inner, dict):
                return json.dumps(inner)
        return raw
    except json.JSONDecodeError:
        text = raw.strip()
        text = re.sub(r',?\s*"[^"]*$', "", text)
        text = re.sub(r',?\s*\{[^{}]*$', "", text)
        depth_brace = text.count("{") - text.count("}")
        depth_bracket = text.count("[") - text.count("]")
        text = text.rstrip(", \t\n")
        text += "]" * max(depth_bracket, 0) + "}" * max(depth_brace, 0)
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and len(obj) == 1:
                inner = next(iter(obj.values()))
                if isinstance(inner, dict):
                    return json.dumps(inner)
        except json.JSONDecodeError:
            pass
        return text


def llm_extract(pages: list[PageData], model: str, source_url: str) -> DocExtraction:
    LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://localhost:8080/v1")
    LLAMA_MODEL_NAME = os.getenv("LLAMA_MODEL_NAME", "gemma-4-E2B-it-Q4_0")
    llm = ChatOpenAI(
        model=LLAMA_MODEL_NAME,
        temperature=0,
        openai_api_base=LLAMA_SERVER_URL,
        openai_api_key="none",
    )
    full_text = "\n\n".join(p.text for p in pages)
    all_tables = "\n\n".join(
        f"TABLE {i+1}\n{table_as_markdown(t)}"
        for p in pages for i, t in enumerate(p.tables)
    )
    schema = DocExtraction.model_json_schema()
    prompt = f"""Extract normalized genomics knowledge from a PDF document.
Return JSON only matching the supplied JSON Schema.
Do not infer missing cells. Preserve exact gene symbols, rsIDs, variants, genomic coordinates, acronyms, URLs, cohort sizes, and version strings.
Classify entity_type as gene, variant, pathway, structural_variant, coordinate, or other.
Use facts for definitions, fields, file formats, reference genomes, software tools, annotation databases, and quantitative specifications.
Use cohorts only for cohort inventory rows.
Use study_results only for rows from a study summary table.
Use narrative_sections only for prose about background, methods, workflows, or interpretation.
Source URL: {source_url}
JSON Schema: {json.dumps(schema)}

DOCUMENT TEXT:
{full_text}

EXTRACTED TABLES:
{all_tables}
"""
    raw = llm.invoke(prompt).content
    repaired = _repair_json(raw)
    if not repaired.strip():
        return DocExtraction()
    try:
        return DocExtraction.model_validate_json(repaired)
    except ValidationError as exc:
        raise RuntimeError(f"Invalid extraction JSON: {exc}\n{raw[:1000]}") from exc


def _entity_type_for(entity_name: str) -> str:
    if entity_name.startswith("rs"):
        return "variant"
    if re.match(r"chr[0-9XYM]", entity_name, re.I):
        return "coordinate"
    if "pathway" in entity_name.lower():
        return "pathway"
    return "gene"


def _normalize_header(h: str) -> str:
    return re.sub(r"[\s_-]+", "_", h.strip().lower())


def parse_study_summary(pages: list[PageData]) -> list[StudyResult]:
    SUMMARY_RE = re.compile(r"Related study summary", re.I)
    STOP_RE = re.compile(r"^(?:Related BOP|References)\b", re.I | re.M)
    FIELD_MAP = {
        "primary_outcome", "cohort_size", "sample_source", "omics_platform",
        "analytic_subject", "primary_analysis", "entity_name", "genomic_coordinates",
    }
    ALIASES = {
        "primary-outcome": "primary_outcome", "primary_outcome": "primary_outcome",
        "cohort-size": "cohort_size", "cohort_size": "cohort_size",
        "sample-source": "sample_source", "sample_source": "sample_source",
        "omics-platform": "omics_platform", "omics_platform": "omics_platform",
        "analytic-subject": "analytic_subject", "analytic_subject": "analytic_subject",
        "primary-analysis": "primary_analysis", "primary_analysis": "primary_analysis",
        "gene-variant-pathway": "entity_name", "gene_variant_pathway": "entity_name", "entity_name": "entity_name",
        "genomic-coordinates": "genomic_coordinates", "genomic_coordinates": "genomic_coordinates",
    }

    results: list[StudyResult] = []
    in_section = False
    for page in pages:
        if not in_section and SUMMARY_RE.search(page.text):
            in_section = True
        if not in_section:
            continue
        if STOP_RE.search(page.text) and results:
            break
        for table in page.tables:
            if not table or len(table[0]) < 2:
                continue
            header = [_normalize_header(str(c or "")) for c in table[0]]
            col_map: dict[int, str] = {}
            for idx, h in enumerate(header):
                canonical = ALIASES.get(h) or next((f for f in FIELD_MAP if h == f), None)
                if canonical:
                    col_map[idx] = canonical
            if "entity_name" not in col_map.values():
                continue
            for row in table[1:]:
                def cell(f: str) -> str | None:
                    for i, fn in col_map.items():
                        if fn == f:
                            v = (row[i] or "").strip() if i < len(row) else ""
                            return v if v and v.upper() != "N/A" else None
                    return None
                entity_name = cell("entity_name")
                if not entity_name:
                    continue
                results.append(StudyResult(
                    primary_outcome=cell("primary_outcome"),
                    cohort_size=cell("cohort_size"),
                    sample_source=cell("sample_source"),
                    omics_platform=cell("omics_platform"),
                    analytic_subject=cell("analytic_subject"),
                    primary_analysis=cell("primary_analysis"),
                    entity_name=entity_name,
                    entity_type=_entity_type_for(entity_name),
                    genomic_coordinates=cell("genomic_coordinates"),
                ))
    return results


def deterministic_links(pages: list[PageData]) -> list[LinkRecord]:
    out: list[LinkRecord] = []
    seen: set[str] = set()
    for page in pages:
        for url in PAGE_URL_RE.findall(page.text):
            url = url.rstrip(".,);]")
            if url not in seen:
                seen.add(url)
                kind = "confluence" if "confluence" in url else "github" if "github.com" in url else "external"
                out.append(LinkRecord(url=url, link_type=kind, section="detected_url"))
    return out


def stable_id(prefix: str, *values: Any) -> str:
    raw = "|".join(str(v or "") for v in values)
    return prefix + "_" + hashlib.sha256(raw.encode()).hexdigest()[:20]


def init_db(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
    CREATE TABLE IF NOT EXISTS documents(
      document_id VARCHAR PRIMARY KEY, file_name VARCHAR, source_url VARCHAR,
      title VARCHAR, extraction_model VARCHAR, page_count INTEGER
    );
    CREATE TABLE IF NOT EXISTS facts(
      fact_id VARCHAR PRIMARY KEY, document_id VARCHAR, fact_type VARCHAR, name VARCHAR,
      abbreviation VARCHAR, description VARCHAR, category VARCHAR, value VARCHAR, unit VARCHAR,
      url VARCHAR, status VARCHAR, section VARCHAR
    );
    CREATE TABLE IF NOT EXISTS cohorts(
      cohort_id VARCHAR PRIMARY KEY, document_id VARCHAR, dataset_name VARCHAR,
      sample_size VARCHAR, clinical_setup VARCHAR, sequencing_facility VARCHAR
    );
    CREATE TABLE IF NOT EXISTS study_results(
      result_id VARCHAR PRIMARY KEY, document_id VARCHAR, primary_outcome VARCHAR,
      cohort_size VARCHAR, sample_source VARCHAR, omics_platform VARCHAR,
      analytic_subject VARCHAR, primary_analysis VARCHAR, entity_name VARCHAR,
      entity_type VARCHAR, genomic_coordinates VARCHAR
    );
    CREATE TABLE IF NOT EXISTS links(
      link_id VARCHAR PRIMARY KEY, document_id VARCHAR, label VARCHAR, url VARCHAR,
      link_type VARCHAR, section VARCHAR
    );
    CREATE TABLE IF NOT EXISTS narrative_sections(
      section_id VARCHAR PRIMARY KEY, document_id VARCHAR, heading VARCHAR, section_text VARCHAR
    );
    """)


def insert_rows(con: duckdb.DuckDBPyConnection, table: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns = list(rows[0])
    placeholders = ",".join("?" for _ in columns)
    con.executemany(
        f"INSERT OR REPLACE INTO {table} ({','.join(columns)}) VALUES ({placeholders})",
        [[row.get(c) for c in columns] for row in rows],
    )


def tokenize(value: str) -> list[str]:
    return [x.casefold() for x in TOKEN_RE.findall(value)]


def build_bm25_entities(document_id: str, facts: list[Fact], results: list[StudyResult], links: list[LinkRecord]) -> list[dict[str, Any]]:
    entities: dict[str, dict[str, Any]] = {}

    def add(kind: str, name: str, content: str, metadata: dict[str, Any]) -> None:
        key = f"{kind}:{name.casefold()}"
        item = entities.setdefault(key, {
            "doc_id": stable_id("bm25", document_id, kind, name),
            "entity_type": kind, "entity_name": name, "text": content, "metadata": metadata,
        })
        if content not in item["text"]:
            item["text"] += "\n" + content

    for f in facts:
        content = " | ".join(x for x in [f.fact_type, f.name, f.abbreviation, f.description, f.value, f.url] if x)
        for name in [f.name] + ([f.abbreviation] if f.abbreviation else []):
            add(f.fact_type, name, content, {"section": f.section})
    for r in results:
        content = " | ".join(x for x in [r.entity_name, r.entity_type, r.primary_outcome, r.cohort_size,
                                           r.sample_source, r.omics_platform, r.analytic_subject,
                                           r.primary_analysis, r.genomic_coordinates] if x)
        add(r.entity_type, r.entity_name, content, {"outcome": r.primary_outcome})
    for link in links:
        label = link.label or Path(link.url.split("?")[0]).name or link.url
        add("link", label, f"{label} | {link.url} | {link.link_type}", {"url": link.url})

    for item in entities.values():
        item["tokens"] = tokenize(item["text"])
    return list(entities.values())


def chunk_narratives(document_id: str, sections: list[dict[str, Any]], max_chars: int = 1800, overlap: int = 250) -> list[Document]:
    docs: list[Document] = []
    for section in sections:
        body = clean_text(section.get("text") or section.get("section_text") or "")
        if len(body) < 120:
            continue
        start, part = 0, 0
        while start < len(body):
            end = min(len(body), start + max_chars)
            if end < len(body):
                cut = body.rfind(". ", start, end)
                if cut > start + max_chars // 2:
                    end = cut + 1
            chunk = body[start:end].strip()
            if chunk:
                heading = section.get("heading", "Narrative")
                doc_id = stable_id("vec", document_id, heading, part, chunk[:100])
                docs.append(Document(
                    page_content=f"Section: {heading}\n\n{chunk}",
                    metadata={"doc_id": doc_id, "document_id": document_id, "heading": heading},
                ))
            if end >= len(body):
                break
            start = max(end - overlap, start + 1)
            part += 1
    return docs


def quality_report(pages: list[PageData], facts: list[Fact], cohorts: list[Cohort],
                   results: list[StudyResult], bm25: list[dict[str, Any]], vectors: list[Document]) -> dict[str, Any]:
    return {
        "pages": len(pages),
        "pdf_tables_detected": sum(len(p.tables) for p in pages),
        "sql_rows": {"facts": len(facts), "cohorts": len(cohorts), "study_results": len(results)},
        "bm25_entities": len(bm25),
        "vector_documents": len(vectors),
        "study_results_with_coordinates": sum(1 for r in results if r.genomic_coordinates),
        "warnings": [
            "The 80/10/10 split is semantic routing, not an exact row or byte percentage.",
            "LLM extraction must not be treated as authoritative without source-page validation.",
        ],
    }


def digest(args: argparse.Namespace) -> None:
    pages = extract_pdf(args.pdf)
    document_id = stable_id("pdf", args.pdf.name, args.source_url)
    title = next((line for line in pages[0].text.splitlines() if "OMICS" in line.upper()), args.pdf.stem)

    print(f"Running LLM extraction over {len(pages)} pages...")
    extraction = llm_extract(pages, args.ollama_model, args.source_url)

    facts = extraction.facts
    cohorts = extraction.cohorts
    results = extraction.study_results

    det_results = parse_study_summary(pages)
    llm_keys = {(r.entity_name, r.genomic_coordinates, r.primary_outcome, r.cohort_size,
                 r.analytic_subject, r.primary_analysis) for r in results}
    results.extend(r for r in det_results if (r.entity_name, r.genomic_coordinates, r.primary_outcome,
                                               r.cohort_size, r.analytic_subject, r.primary_analysis) not in llm_keys)

    links = extraction.links
    seen_urls = {x.url for x in links}
    links.extend(x for x in deterministic_links(pages) if x.url not in seen_urls)

    heuristic_sections = split_sections(pages)
    model_sections = [{"heading": s.get("heading", "Narrative"), "text": s.get("text", "")}
                      for s in extraction.narrative_sections]
    vector_docs = chunk_narratives(document_id, heuristic_sections + model_sections)
    bm25_entities = build_bm25_entities(document_id, facts, results, links)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(args.output_dir / "genomic_knowledge.duckdb"))
    try:
        init_db(con)
        insert_rows(con, "documents", [{
            "document_id": document_id, "file_name": args.pdf.name, "source_url": args.source_url,
            "title": title, "extraction_model": args.ollama_model, "page_count": len(pages),
        }])
        insert_rows(con, "facts", [{"fact_id": stable_id("fact", document_id, f.section, f.fact_type, f.name),
                                    "document_id": document_id, **f.model_dump()} for f in facts])
        insert_rows(con, "cohorts", [{"cohort_id": stable_id("cohort", document_id, c.dataset_name),
                                      "document_id": document_id, **c.model_dump()} for c in cohorts])
        insert_rows(con, "study_results", [{"result_id": stable_id("result", document_id, r.entity_name,
                                      r.genomic_coordinates, r.primary_outcome, r.cohort_size,
                                      r.analytic_subject, r.primary_analysis),
                                      "document_id": document_id, **r.model_dump()} for r in results])
        insert_rows(con, "links", [{"link_id": stable_id("link", document_id, x.url),
                                    "document_id": document_id, **x.model_dump()} for x in links])
        insert_rows(con, "narrative_sections", [{
            "section_id": stable_id("section", document_id, s["heading"]),
            "document_id": document_id, "heading": s["heading"], "section_text": s["text"],
        } for s in heuristic_sections])
    finally:
        con.close()

    (args.output_dir / "bm25_entities.json").write_text(
        json.dumps(bm25_entities, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if bm25_entities:
        BM25Okapi([x["tokens"] for x in bm25_entities])

    if vector_docs:
        embedding = HuggingFaceEmbeddings(
            model_name=args.embedding_model,
            model_kwargs={"device": args.device},
            encode_kwargs={"normalize_embeddings": True, "batch_size": args.batch_size},
        )
        FAISS.from_documents(vector_docs, embedding).save_local(str(args.output_dir / "faiss_index"))
    (args.output_dir / "vector_documents.json").write_text(json.dumps([
        {"page_content": d.page_content, "metadata": d.metadata} for d in vector_docs
    ], ensure_ascii=False, indent=2), encoding="utf-8")

    report = quality_report(pages, facts, cohorts, results, bm25_entities, vector_docs)
    (args.output_dir / "quality_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Artifacts written to {args.output_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description="Digest a genomics PDF into SQL, BM25, and vector stores")
    p.add_argument("pdf", type=Path)
    p.add_argument("--source-url", default="")
    p.add_argument("--output-dir", type=Path, default=Path("genomic_digest"))
    p.add_argument("--ollama-model", default=os.getenv("OLLAMA_MODEL", "gemma4:e4b"))
    p.add_argument("--embedding-model", default=os.getenv("EMBED_MODEL", r"C:\Users\tzhang\Desktop\Project\DP-AI\LangChain-OpenTutorial-main\Sentence-BioBert-snli"))
    p.add_argument("--device", default=os.getenv("EMBED_DEVICE", "cpu"))
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()
    if not args.pdf.exists():
        p.error(f"PDF not found: {args.pdf}")
    digest(args)


if __name__ == "__main__":
    main()
