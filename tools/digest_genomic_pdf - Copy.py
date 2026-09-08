#!/usr/bin/env python3
"""Digest a genomics PDF into DuckDB tables, BM25 entities, and FAISS vector documents.

The 80/10/10 values are routing targets, not exact byte quotas:
  * structured facts and table rows -> DuckDB
  * exact genomic identifiers -> BM25 corpus
  * narrative/methodological sections -> vector documents

Requires a local Ollama model capable of JSON output for robust section/table normalization.
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
import pandas as pd
import duckdb
import pdfplumber
import tabula
from pydantic import BaseModel, Field, ValidationError, field_validator
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
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
SECTION_HEADINGS = [
    "Background", "Whole genome sequencing data", "MDS patient cohort of WGS data",
    "Genomic nomenclature", "Genomic variation definition",
    "Genomic sequencing specification", "Genomic file format",
    "Genomic data processing pipelines", "Reference genome",
    "Variant calling tools", "Variant annotation tools",
    "Variant annotation database", "Related study summary", "Related BOP", "References",
]


def _coerce_str_fields(data: Any) -> Any:
    """Coerce any str | None field that the LLM returned as a number back to str."""
    if not isinstance(data, dict):
        return data
    return {k: str(v) if isinstance(v, (int, float)) and k != "page_number" else v
            for k, v in data.items()}


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
    page_number: int
    section: str

    @field_validator("*", mode="before")
    @classmethod
    def _pre(cls, v: Any) -> Any:
        return v

    @classmethod
    def model_validate(cls, obj: Any, **kw: Any) -> "Fact":
        return super().model_validate(_coerce_str_fields(obj), **kw)


class Cohort(BaseModel):
    dataset_name: str
    sample_size: str | None = None
    clinical_setup: str | None = None
    sequencing_facility: str | None = None
    page_number: int

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
    page_number: int

    @classmethod
    def model_validate(cls, obj: Any, **kw: Any) -> "StudyResult":
        return super().model_validate(_coerce_str_fields(obj), **kw)


class SectionTable(BaseModel):
    """A single row parsed from a two-column section table in the PDF text layer."""
    table_name: str          # registry key, e.g. "annotation_databases"
    col1: str                # value of the first (key) column
    col2: str                # value of the second (description) column
    page_number: int
    document_id: str


class LinkRecord(BaseModel):
    label: str | None = None
    url: str
    link_type: str
    page_number: int
    section: str

    @classmethod
    def model_validate(cls, obj: Any, **kw: Any) -> "LinkRecord":
        return super().model_validate(_coerce_str_fields(obj), **kw)


class PageExtraction(BaseModel):
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
        items = [_coerce_str_fields(item) if isinstance(item, dict) else item for item in v]
        if info.field_name == "facts":
            items = [item for item in items if isinstance(item, dict) and item.get("name") and item.get("section")]
        return items

    @field_validator("narrative_sections", mode="before")
    @classmethod
    def coerce_narrative_sections(cls, v: Any) -> Any:
        if isinstance(v, dict):
            v = list(v.values()) if v else []
        if not isinstance(v, list):
            return []
        # Coerce every value in each dict to str so page_number=2 doesn't fail
        return [{k: str(val) for k, val in item.items()} if isinstance(item, dict) else item for item in v]


@dataclass
class PageData:
    page_number: int
    text: str
    tables: list[list[list[str | None]]]


def clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\\_", "_")
    text = re.sub(r"(?m)^\s*\d+/\d+\s*$", "", text)
    # Match both correct spelling (pageId) and OCR artefact (pageld)
    text = re.sub(r"(?m)^https://confluence\.nmdp\.org/pages/viewpage\.action\?pageI?d=\d+\S*\s*$", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_header_row(row: list[str | None]) -> bool:
    """Heuristic: a row is a header if most non-empty cells are title-cased or ALL-CAPS short strings."""
    cells = [c for c in row if c and c.strip()]
    if not cells:
        return False
    header_like = sum(1 for c in cells if c.strip() == c.strip().title() or c.strip().isupper())
    return header_like / len(cells) >= 0.6


def _is_layout_table(table: list[list[str | None]]) -> bool:
    """True if this is a page-layout wrapper (e.g. Confluence chrome), not a real data table.

    Two signals, either is sufficient:
    1. Majority rule: >50% of non-empty cells each exceed 500 chars (page container).
    2. Single-column breadcrumb: 1-column table whose first cell looks like a
       navigation breadcrumb (contains ' / ' or ends with a page-footer pattern).
    """
    non_empty = [c for row in table for c in row if c and c.strip()]
    if not non_empty:
        return False
    # Signal 1 — majority of cells are large blobs
    large = sum(1 for c in non_empty if len(c) > 500)
    if large / len(non_empty) > 0.5:
        return True
    # Signal 2 — single-column navigation/breadcrumb table
    if all(len(row) == 1 for row in table):
        first = (table[0][0] or "").strip()
        if " / " in first or re.search(r"viewpage\.action", first):
            return True
    return False


def _is_empty_table(table: list[list[str | None]]) -> bool:
    """True if every cell is None or blank — no useful content."""
    return all(not (c and c.strip()) for row in table for c in row)


def merge_split_tables(pages: list[PageData]) -> list[PageData]:
    """Stitch tables that continue across 2 or more consecutive pages.

    Tracks the page that owns the growing table (anchor) explicitly so that
    a table spanning pages 1→2→3 is fully merged into page 1 regardless of
    whether page 2 is left with an empty tables list after its fragment is consumed.
    """
    anchor_idx: int | None = None  # index of the page that owns the growing table

    for i in range(len(pages) - 1):
        nxt = pages[i + 1]
        if not nxt.tables:
            anchor_idx = None
            continue

        # Determine which page currently holds the tail to extend.
        owner_idx = anchor_idx if anchor_idx is not None else i
        owner = pages[owner_idx]
        if not owner.tables:
            anchor_idx = None
            continue

        tail = owner.tables[-1]
        head = nxt.tables[0]
        if not tail or not head or len(tail[0]) != len(head[0]):
            anchor_idx = None
            continue

        continuation_rows = head[1:] if _is_header_row(head[0]) else head
        if not continuation_rows:
            anchor_idx = None
            continue

        owner.tables[-1] = tail + continuation_rows
        nxt.tables = nxt.tables[1:]
        anchor_idx = owner_idx  # keep extending from the same anchor page

    return pages


def _tabula_tables_for_page(pdf_path: Path, page_number: int) -> list[list[list[str | None]]]:
    """Extract tables from a single page using tabula and convert to list-of-rows format."""
    dfs = tabula.read_pdf(str(pdf_path), pages=page_number, multiple_tables=True, silent=True)
    result = []
    table_tmp = []
    for i in range(len(dfs)):
        rows = [[None if (str(c) in ("", "nan")) else str(c) for c in row] for row in dfs[i].values.tolist()]
        header = [None if (str(c) in ("", "nan")) else str(c) for c in dfs[i].columns.tolist()]
        table = [header] + rows
        if header!=None:
            if table_tmp != []:
                result.append(table_tmp)
            table_tmp=table
        else:
            table_tmp=pd.concat([table_tmp,table], ignore_index=True)
        if i==len(dfs)-1:
            result.append(table_tmp)
          
    return result


def extract_pdf(pdf_path: Path) -> list[PageData]:
    pages: list[PageData] = []
    with pdfplumber.open(pdf_path) as pdf:
        for number, page in enumerate(pdf.pages, 1):
            text = clean_text(page.extract_text(x_tolerance=2, y_tolerance=3) or "")
            tables = _tabula_tables_for_page(pdf_path, number)
            pages.append(PageData(number, text, tables))
    return merge_split_tables(pages)


def split_sections(pages: list[PageData]) -> list[dict[str, Any]]:
    heading_pattern = re.compile(
        r"(?im)^(" + "|".join(re.escape(x) for x in sorted(SECTION_HEADINGS, key=len, reverse=True)) + r"):\s*$"
    )
    sections: list[dict[str, Any]] = []
    current = {"heading": "Page introduction", "text": "", "pages": []}
    for page in pages:
        pieces = heading_pattern.split(page.text)
        if pieces[0].strip():
            current["text"] += "\n" + pieces[0].strip()
            current["pages"].append(page.page_number)
        for i in range(1, len(pieces), 2):
            if current["text"].strip():
                current["pages"] = sorted(set(current["pages"]))
                sections.append(current)
            current = {
                "heading": pieces[i].strip(),
                "text": pieces[i + 1].strip() if i + 1 < len(pieces) else "",
                "pages": [page.page_number],
            }
    if current["text"].strip():
        current["pages"] = sorted(set(current["pages"]))
        sections.append(current)
    return sections


def table_as_markdown(table: list[list[str | None]]) -> str:
    return "\n".join(" | ".join(clean_text(cell or "") for cell in row) for row in table)


def _repair_json(raw: str) -> str:
    """Strip markdown fences, unwrap a {"PageExtraction": {...}} envelope, and close truncated JSON."""
    # Strip ```json ... ``` or ``` ... ``` fences
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
        # Truncated: strip trailing incomplete token then close all open braces/brackets.
        text = raw.strip()
        # Remove last incomplete string or key-value fragment
        text = re.sub(r',?\s*"[^"]*$', "", text)
        text = re.sub(r',?\s*\{[^{}]*$', "", text)
        # Close open arrays and objects
        depth_brace = text.count("{") - text.count("}")
        depth_bracket = text.count("[") - text.count("]")
        text = text.rstrip(", \t\n")
        text += "]" * max(depth_bracket, 0) + "}" * max(depth_brace, 0)
        # Unwrap envelope after repair
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and len(obj) == 1:
                inner = next(iter(obj.values()))
                if isinstance(inner, dict):
                    return json.dumps(inner)
        except json.JSONDecodeError:
            pass
        return text


# Pages whose tables are fully handled by deterministic parsers — skip table text in LLM prompt
_DETERMINISTIC_TABLE_HEADINGS = re.compile(
    r"Variant annotation database|Related study summary",
    re.I,
)


def llm_extract(page: PageData, model: str, source_url: str) -> PageExtraction:
    # ollama_host= os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    # llm = ChatOllama(model=model, temperature=0, format="json", base_url=ollama_host)
    LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://localhost:8080/v1")    
    LLAMA_MODEL_NAME = os.getenv("LLAMA_MODEL_NAME", "gemma-4-E2B-it-Q4_0")

    llm =ChatOpenAI(
        model=LLAMA_MODEL_NAME,
        temperature=0,
        openai_api_base=LLAMA_SERVER_URL,
        openai_api_key="none",
    )
    # Omit table text for pages already handled deterministically to avoid oversized prompts
    if _DETERMINISTIC_TABLE_HEADINGS.search(page.text):
        table_text = "(tables on this page are extracted deterministically — omitted)"
    else:
        table_text = "\n\n".join(f"TABLE {i+1}\n{table_as_markdown(t)}" for i, t in enumerate(page.tables))
    schema = PageExtraction.model_json_schema()
    prompt = f"""Extract normalized genomics knowledge from one PDF page.
Return JSON only matching the supplied JSON Schema.
Do not infer missing cells. Carry forward merged table cells only when visually/textually evident.
Preserve exact gene symbols, rsIDs, variants, genomic coordinates, acronyms, URLs, cohort sizes, and version strings.
Classify entity_type as gene, variant, pathway, structural_variant, coordinate, or other.
Use facts for definitions, fields, file formats, reference genomes, software tools, annotation databases, and quantitative specifications.
Use cohorts only for cohort inventory rows.
Use study_results only for rows from a study summary table.
Use narrative_sections only for prose about background, methods, workflows, or interpretation that is unsuitable as a row-level fact.
Each extracted item must include page_number={page.page_number}; section should be the nearest visible heading.
Source URL: {source_url}
JSON Schema: {json.dumps(schema)}

PAGE TEXT:
{page.text}

EXTRACTED TABLES:
{table_text}
"""
    raw = llm.invoke(prompt).content
    repaired = _repair_json(raw)
    if not repaired.strip():
        return PageExtraction()
    try:
        return PageExtraction.model_validate_json(repaired)
    except ValidationError as exc:
        raise RuntimeError(f"Invalid extraction JSON on page {page.page_number}: {exc}\n{raw[:1000]}") from exc


# Genomic coordinate: chr1:12345-67890(+) or chrM:1234-5678
_COORD_RE = re.compile(r"chr(?:[0-9]{1,2}|[XYM]):[0-9,]+-[0-9,]+(?:\([+-]\))?")
# Gene symbol or rsID
_GENE_RE = re.compile(r"\b([A-Z][A-Z0-9]{1,19}(?:[._-][A-Z0-9]+)*|rs\d+|MT-[A-Z0-9]+)\b")
# Tokens that are not gene/variant names
_NON_GENE: frozenset[str] = frozenset({
    "WGS", "WES", "SNV", "SNVS", "DMR", "DMRS", "NA", "RRBS", "GSEA", "NCI", "UCSC", "OMICS",
    "SAM", "BAM", "CRAM", "BED", "VCF", "FASTQ", "NGS", "PCR", "DNA", "RNA", "MDS", "HCT",
    "SOP", "BOP", "GDR", "QC", "ID", "OS", "EFS", "DFS", "RFS", "SV", "CNV", "UF", "LT", "DEL",
    "ALL", "AFR", "AMR", "EAS", "FIN", "NFE", "OTH", "SAS", "AC", "PM", "GT", "DP", "VAF",
    "POS", "CADD", "SIFT", "VEP", "FASTA", "GATK4", "MUTECT2", "OCTOPUS", "CLINVAR", "COSMIC",
    "ANNOVAR", "SNPEFF", "NHLBI", "ESP", "EXAC", "GNOMAD", "GERP", "GWAVA", "REVEL",
    "MCAP", "EIGEN", "FATHMM", "ICGC", "INTERVAR", "KAVIAR", "REGSNP", "DBSNP",
    "REFGENE", "KNOWNGENE", "ENSGENE", "REFSEQ", "NCBI", "ENSEMBL", "HGNC",
    "PRIMARY", "COHORT", "SAMPLE", "OMICS", "ANALYTIC", "GENE", "VARIANT", "PATHWAY",
    "GENOMIC", "RELATED", "STUDY", "SUMMARY", "OUTCOME", "SIZE", "SOURCE", "PLATFORM",
    "SUBJECT", "ANALYSIS", "COORDINATES", "CONFLUENCE", "BIOINFORMATICS",
})


# Column header aliases: maps any header variant the PDF/tabula may produce -> canonical field name
_STUDY_COL_ALIASES: dict[str, str] = {
    "primary-outcome": "primary_outcome", "primary outcome": "primary_outcome",
    "cohort-size": "cohort_size", "cohort size": "cohort_size",
    "sample-source": "sample_source", "sample source": "sample_source",
    "omics-platform": "omics_platform", "omics platform": "omics_platform",
    "analytic-subject": "analytic_subject", "analytic subject": "analytic_subject",
    "primary-analysis": "primary_analysis", "primary analysis": "primary_analysis",
    "gene-variant-pathway": "entity_name", "gene variant pathway": "entity_name",
    "genomic-coordinates": "genomic_coordinates", "genomic coordinates": "genomic_coordinates",
}


def _entity_type_for(entity_name: str) -> str:
    if entity_name.startswith("rs"):
        return "variant"
    if re.match(r"chr[0-9XYM]", entity_name, re.I):
        return "coordinate"
    if "pathway" in entity_name.lower():
        return "pathway"
    if "cluster" in entity_name.lower():
        return "other"
    return "gene"


def _parse_study_summary_from_tables(pages: list[PageData]) -> list[StudyResult]:
    """Read study results directly from structured table cells on the summary page(s)."""
    SUMMARY_PAGE_RE = re.compile(r"Related study summary", re.I)
    STOP_RE = re.compile(r"^(?:Related BOP|References)\b", re.I | re.M)

    results: list[StudyResult] = []
    in_section = False
    for page in pages:
        if not in_section and SUMMARY_PAGE_RE.search(page.text):
            in_section = True
        if not in_section:
            continue
        if STOP_RE.search(page.text) and in_section and results:
            break

        for table in page.tables:
            if not table or len(table[0]) < 2:
                continue
            # Map column indices to canonical field names via header row
            header = [str(c or "").strip().lower() for c in table[0]]
            col_map: dict[int, str] = {}
            for idx, h in enumerate(header):
                if h in _STUDY_COL_ALIASES:
                    col_map[idx] = _STUDY_COL_ALIASES[h]
            if "entity_name" not in col_map.values():
                continue  # not the study summary table

            for row in table[1:]:  # skip header
                def cell(field: str, _row: list = row) -> str | None:
                    for i, f in col_map.items():
                        if f == field:
                            v = (_row[i] or "").strip() if i < len(_row) else ""
                            return v if v and v.upper() != "N/A" else None
                    return None

                entity_name = cell("entity_name")
                if not entity_name:
                    continue
                coord = cell("genomic_coordinates")
                results.append(StudyResult(
                    primary_outcome=cell("primary_outcome"),
                    cohort_size=cell("cohort_size"),
                    sample_source=cell("sample_source"),
                    omics_platform=cell("omics_platform"),
                    analytic_subject=cell("analytic_subject"),
                    primary_analysis=cell("primary_analysis"),
                    entity_name=entity_name,
                    entity_type=_entity_type_for(entity_name),
                    genomic_coordinates=coord,
                    page_number=page.page_number,
                ))
    return results


def parse_study_summary(pages: list[PageData]) -> list[StudyResult]:
    """Deterministically extract study result rows from tabula table cells in the 'Related study summary' section."""
    return _parse_study_summary_from_tables(pages)


# ---------------------------------------------------------------------------
# Generic two-column section-table parser
# ---------------------------------------------------------------------------

class TwoColumnSectionSpec:
    """Declarative config for one two-column section table.

    Parameters
    ----------
    table_name      : DuckDB table name (also used as the registry key)
    section_heading : Regex pattern that matches the section heading line
    stop_headings   : Regex pattern that marks the end of the section
    col1_name       : Logical name for the key column (stored in SectionTable.col1)
    col2_name       : Logical name for the description column (stored in SectionTable.col2)
    skip_patterns      : List of regex patterns whose matching lines are silently skipped
    known_solo_keys    : Set of key values that may appear alone on a line (no 2-space separator)
    body_content_hint  : Regex that must match within 15 lines after the heading to confirm
                         this is the body occurrence and not a TOC entry. If None, any
                         non-section-heading line is accepted.
    """
    def __init__(
        self,
        table_name: str,
        section_heading: str,
        stop_headings: str,
        col1_name: str,
        col2_name: str,
        skip_patterns: list[str] | None = None,
        known_solo_keys: frozenset[str] | None = None,
        body_content_hint: str | None = None,
    ) -> None:
        self.table_name = table_name
        self.section_re = re.compile(section_heading, re.I)
        self.stop_re    = re.compile(stop_headings, re.I | re.M)
        self.col1_name  = col1_name
        self.col2_name  = col2_name
        self.skip_res   = [re.compile(p, re.I) for p in (skip_patterns or [])]
        self.known_solo_keys: frozenset[str] = known_solo_keys or frozenset()
        self.body_hint_re = re.compile(body_content_hint, re.I) if body_content_hint else None


# Registry: add one TwoColumnSectionSpec entry per new table — no other code changes needed.
TWO_COLUMN_SECTIONS: list[TwoColumnSectionSpec] = [
    TwoColumnSectionSpec(
        table_name      = "annotation_databases",
        section_heading = r"Variant annotation database:",
        stop_headings   = r"^(?:Related study summary|Variant calling tools|Variant annotation tools)\b",
        col1_name       = "db_name",
        col2_name       = "explanation",
        skip_patterns   = [r"^Database\s+Name\s+Explanation", r"^\(", r"^http"],
        known_solo_keys = frozenset({
            "cadd", "eigen", "gwava", "mcap", "revel", "fathmm", "nci60",
            "icgc28", "hrer1", "snp138", "refgene", "knowngene", "ensgene",
        }),
        body_content_hint = r"(?:Database\s+Name|1000g|avsift|avsnp|clinvar|cosmic|dbnsfp)",
    ),
    # -----------------------------------------------------------------------
    # Add future two-column section tables here, e.g.:
    # TwoColumnSectionSpec(
    #     table_name      = "variant_calling_tools",
    #     section_heading = r"Variant calling tools:",
    #     stop_headings   = r"^(?:Variant annotation tools|Variant annotation database)\b",
    #     col1_name       = "tool_name",
    #     col2_name       = "description",
    # ),
    # -----------------------------------------------------------------------
]


def parse_two_column_section(
    pages: list[PageData],
    document_id: str,
    spec: TwoColumnSectionSpec,
) -> list[SectionTable]:
    """Extract rows from a two-column section table using tabula cells only."""
    # Locate the body page (skip TOC occurrence)
    toc_headings_re = re.compile(
        r"^(?:" + "|".join(re.escape(h) for h in SECTION_HEADINGS) + r")[.:\s]*$", re.I,
    )
    body_page: PageData | None = None
    for page in pages:
        if not spec.section_re.search(page.text):
            continue
        m = spec.section_re.search(page.text)
        after_lines = [l.strip() for l in page.text[m.end():].splitlines() if l.strip()][:15]
        if spec.body_hint_re is not None:
            if any(spec.body_hint_re.search(l) for l in after_lines):
                body_page = page
                break
        else:
            if any(not toc_headings_re.match(l) for l in after_lines):
                body_page = page
                break

    if body_page is None:
        return []

    # Collect section pages until stop heading
    section_pages: list[PageData] = []
    in_section = False
    for page in pages:
        if not in_section and page.page_number == body_page.page_number:
            in_section = True
        if in_section:
            section_pages.append(page)
            if spec.stop_re.search("\n".join(p.text for p in section_pages)) and len(section_pages) > 1:
                break

    rows: list[SectionTable] = []
    for page in section_pages:
        for table in page.tables:
            if not table or len(table[0]) < 2:
                continue
            col_key, col_val = 0, 1
            if _is_header_row(table[0]):
                header_cells = [(idx, c) for idx, c in enumerate(table[0]) if c and c.strip()]
                if len(header_cells) >= 2:
                    col_key, col_val = header_cells[0][0], header_cells[-1][0]
            for row in table[1:]:
                key_cell = (row[col_key] or "").strip() if col_key < len(row) else ""
                val_cell = (row[col_val] or "").strip() if col_val < len(row) else ""
                if not key_cell or not val_cell:
                    continue
                if any(skip.match(key_cell) for skip in spec.skip_res):
                    continue
                rows.append(SectionTable(
                    table_name=spec.table_name, col1=key_cell, col2=val_cell,
                    page_number=page.page_number, document_id=document_id,
                ))
    return rows


def parse_all_section_tables(pages: list[PageData], document_id: str) -> dict[str, list[SectionTable]]:
    """Run parse_two_column_section for every registered spec. Returns {table_name: rows}."""
    return {
        spec.table_name: parse_two_column_section(pages, document_id, spec)
        for spec in TWO_COLUMN_SECTIONS
    }


def deterministic_links(pages: list[PageData]) -> list[LinkRecord]:
    out: list[LinkRecord] = []
    seen: set[tuple[int, str]] = set()
    for page in pages:
        for url in PAGE_URL_RE.findall(page.text):
            url = url.rstrip(".,);]")
            key = (page.page_number, url)
            if key not in seen:
                seen.add(key)
                kind = "confluence" if "confluence" in url else "github" if "github.com" in url else "external"
                out.append(LinkRecord(url=url, link_type=kind, page_number=page.page_number, section="detected_url"))
    return out


def init_db(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
    CREATE TABLE IF NOT EXISTS documents(
      document_id VARCHAR PRIMARY KEY, file_name VARCHAR, source_url VARCHAR,
      title VARCHAR, extraction_model VARCHAR, page_count INTEGER
    );
    CREATE TABLE IF NOT EXISTS facts(
      fact_id VARCHAR PRIMARY KEY, document_id VARCHAR, fact_type VARCHAR, name VARCHAR,
      abbreviation VARCHAR, description VARCHAR, category VARCHAR, value VARCHAR, unit VARCHAR,
      url VARCHAR, status VARCHAR, page_number INTEGER, section VARCHAR
    );
    CREATE TABLE IF NOT EXISTS cohorts(
      cohort_id VARCHAR PRIMARY KEY, document_id VARCHAR, dataset_name VARCHAR,
      sample_size VARCHAR, clinical_setup VARCHAR, sequencing_facility VARCHAR, page_number INTEGER
    );
    CREATE TABLE IF NOT EXISTS study_results(
      result_id VARCHAR PRIMARY KEY, document_id VARCHAR, primary_outcome VARCHAR,
      cohort_size VARCHAR, sample_source VARCHAR, omics_platform VARCHAR,
      analytic_subject VARCHAR, primary_analysis VARCHAR, entity_name VARCHAR,
      entity_type VARCHAR, genomic_coordinates VARCHAR, page_number INTEGER
    );
    CREATE TABLE IF NOT EXISTS links(
      link_id VARCHAR PRIMARY KEY, document_id VARCHAR, label VARCHAR, url VARCHAR,
      link_type VARCHAR, page_number INTEGER, section VARCHAR
    );
    CREATE TABLE IF NOT EXISTS narrative_sections(
      section_id VARCHAR PRIMARY KEY, document_id VARCHAR, heading VARCHAR,
      section_text VARCHAR, page_start INTEGER, page_end INTEGER
    );
    CREATE TABLE IF NOT EXISTS extraction_audit(
      audit_id VARCHAR PRIMARY KEY, document_id VARCHAR, page_number INTEGER,
      extracted_fact_count INTEGER, extracted_cohort_count INTEGER,
      extracted_result_count INTEGER, extracted_link_count INTEGER,
      table_count INTEGER, text_char_count INTEGER
    );
    """)
    # Create one table per registered two-column section spec
    for spec in TWO_COLUMN_SECTIONS:
        con.execute(f"""
        CREATE TABLE IF NOT EXISTS {spec.table_name}(
          row_id VARCHAR PRIMARY KEY, document_id VARCHAR,
          {spec.col1_name} VARCHAR, {spec.col2_name} VARCHAR, page_number INTEGER
        );
        """)


def stable_id(prefix: str, *values: Any) -> str:
    raw = "|".join(str(v or "") for v in values)
    return prefix + "_" + hashlib.sha256(raw.encode()).hexdigest()[:20]


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
    def add(kind: str, name: str, content: str, page: int, metadata: dict[str, Any]) -> None:
        key = f"{kind}:{name.casefold()}"
        item = entities.setdefault(key, {
            "doc_id": stable_id("bm25", document_id, kind, name),
            "entity_type": kind, "entity_name": name,
            "text": content, "pages": [], "metadata": metadata,
        })
        item["pages"].append(page)
        if content not in item["text"]:
            item["text"] += "\n" + content
    for f in facts:
        names = [f.name] + ([f.abbreviation] if f.abbreviation else [])
        content = " | ".join(x for x in [f.fact_type, f.name, f.abbreviation, f.description, f.value, f.url] if x)
        for name in names:
            add(f.fact_type, name, content, f.page_number, {"section": f.section})
    for r in results:
        content = " | ".join(x for x in [r.entity_name, r.entity_type, r.primary_outcome, r.cohort_size,
                                           r.sample_source, r.omics_platform, r.analytic_subject,
                                           r.primary_analysis, r.genomic_coordinates] if x)
        add(r.entity_type, r.entity_name, content, r.page_number, {"outcome": r.primary_outcome})
    for link in links:
        label = link.label or Path(link.url.split("?")[0]).name or link.url
        add("link", label, f"{label} | {link.url} | {link.link_type}", link.page_number, {"url": link.url})
    for item in entities.values():
        item["pages"] = sorted(set(item["pages"]))
        item["tokens"] = tokenize(item["text"])
    return list(entities.values())


def chunk_narratives(document_id: str, sections: list[dict[str, Any]], max_chars: int = 1800, overlap: int = 250) -> list[Document]:
    docs: list[Document] = []
    for section in sections:
        body = clean_text(section.get("text") or section.get("section_text") or "")
        if len(body) < 120:
            continue
        start = 0
        part = 0
        while start < len(body):
            end = min(len(body), start + max_chars)
            if end < len(body):
                cut = body.rfind(". ", start, end)
                if cut > start + max_chars // 2:
                    end = cut + 1
            chunk = body[start:end].strip()
            if chunk:
                heading = section.get("heading", "Narrative")
                pages = section.get("pages") or [section.get("page_number", 0)]
                doc_id = stable_id("vec", document_id, heading, part, chunk[:100])
                docs.append(Document(
                    page_content=f"Section: {heading}\n\n{chunk}",
                    metadata={"doc_id": doc_id, "document_id": document_id,
                              "heading": heading, "pages": json.dumps(pages)},
                ))
            if end >= len(body):
                break
            start = max(end - overlap, start + 1)
            part += 1
    return docs


def quality_report(pages: list[PageData], facts: list[Fact], cohorts: list[Cohort],
                   results: list[StudyResult], section_tables: dict[str, list[SectionTable]],
                   bm25: list[dict[str, Any]], vectors: list[Document]) -> dict[str, Any]:
    coordinates = sum(1 for r in results if r.genomic_coordinates)
    return {
        "pages": len(pages), "pdf_tables_detected": sum(len(p.tables) for p in pages),
        "sql_rows": {"facts": len(facts), "cohorts": len(cohorts), "study_results": len(results),
                     **{k: len(v) for k, v in section_tables.items()}},
        "bm25_entities": len(bm25), "vector_documents": len(vectors),
        "study_results_with_coordinates": coordinates,
        "warnings": [
            "The 80/10/10 split is semantic routing, not an exact row or byte percentage.",
            "Cross-page tables are auto-stitched by column count; verify merged tables in raw_page_extractions.json.",
            "LLM extraction must not be treated as authoritative without source-page validation.",
        ],
    }


def digest(args: argparse.Namespace) -> None:
    pages = extract_pdf(args.pdf)
    document_id = stable_id("pdf", args.pdf.name, args.source_url)
    section_tables = parse_all_section_tables(pages, document_id)
    title = next((line for line in pages[0].text.splitlines() if "OMICS" in line.upper()), args.pdf.stem)

    extracted: list[PageExtraction] = []
    for page in pages:
        print(f"Extracting page {page.page_number}/{len(pages)}")
        extracted.append(llm_extract(page, args.ollama_model, args.source_url))

    facts = [x for page in extracted for x in page.facts]
    cohorts = [x for page in extracted for x in page.cohorts]
    results = [x for page in extracted for x in page.study_results]
    # Supplement with deterministic parse — covers pages where LLM failed or misclassified
    det_results = parse_study_summary(pages)
    llm_keys = {(r.entity_name, r.genomic_coordinates, r.primary_outcome, r.cohort_size, r.analytic_subject, r.primary_analysis) for r in results}
    results.extend(r for r in det_results if (r.entity_name, r.genomic_coordinates, r.primary_outcome, r.cohort_size, r.analytic_subject, r.primary_analysis) not in llm_keys)
    links = [x for page in extracted for x in page.links]
    # Deterministic URL extraction supplements, but never replaces, model extraction.
    seen_urls = {(x.page_number, x.url) for x in links}
    links.extend(x for x in deterministic_links(pages) if (x.page_number, x.url) not in seen_urls)

    heuristic_sections = split_sections(pages)
    model_sections = [
        {"heading": s.get("heading", "Narrative"), "text": s.get("text", ""),
         "pages": [page.page_number]}
        for page, extraction in zip(pages, extracted)
        for s in extraction.narrative_sections
    ]
    vector_docs = chunk_narratives(document_id, heuristic_sections + model_sections)
    bm25_entities = build_bm25_entities(document_id, facts, results, links)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    db_path = args.output_dir / "genomic_knowledge.duckdb"
    con = duckdb.connect(str(db_path))
    try:
        init_db(con)
        insert_rows(con, "documents", [{
            "document_id": document_id, "file_name": args.pdf.name, "source_url": args.source_url,
            "title": title, "extraction_model": args.ollama_model, "page_count": len(pages),
        }])
        insert_rows(con, "facts", [{"fact_id": stable_id("fact", document_id, f.page_number, f.section, f.fact_type, f.name),
                                    "document_id": document_id, **f.model_dump()} for f in facts])
        insert_rows(con, "cohorts", [{"cohort_id": stable_id("cohort", document_id, c.dataset_name, c.page_number),
                                      "document_id": document_id, **c.model_dump()} for c in cohorts])
        insert_rows(con, "study_results", [{"result_id": stable_id("result", document_id, r.page_number,
                                      r.entity_name, r.genomic_coordinates, r.primary_outcome,
                                      r.cohort_size, r.analytic_subject, r.primary_analysis),
                                      "document_id": document_id, **r.model_dump()} for r in results])
        insert_rows(con, "links", [{"link_id": stable_id("link", document_id, x.page_number, x.url),
                                    "document_id": document_id, **x.model_dump()} for x in links])
        insert_rows(con, "narrative_sections", [{
            "section_id": stable_id("section", document_id, s["heading"], s["pages"]),
            "document_id": document_id, "heading": s["heading"], "section_text": s["text"],
            "page_start": min(s["pages"]), "page_end": max(s["pages"]),
        } for s in heuristic_sections])
        # Insert all registered two-column section tables
        for spec in TWO_COLUMN_SECTIONS:
            con.execute(f"DELETE FROM {spec.table_name} WHERE document_id = ?", [document_id])
            insert_rows(con, spec.table_name, [{
                "row_id": stable_id(spec.table_name[:6], document_id, r.col1),
                "document_id": document_id,
                spec.col1_name: r.col1,
                spec.col2_name: r.col2,
                "page_number": r.page_number,
            } for r in section_tables.get(spec.table_name, [])])
        insert_rows(con, "extraction_audit", [{
            "audit_id": stable_id("audit", document_id, p.page_number), "document_id": document_id,
            "page_number": p.page_number, "extracted_fact_count": len(e.facts),
            "extracted_cohort_count": len(e.cohorts), "extracted_result_count": len(e.study_results),
            "extracted_link_count": len(e.links), "table_count": len(p.tables), "text_char_count": len(p.text),
        } for p, e in zip(pages, extracted)])
    finally:
        con.close()

    (args.output_dir / "bm25_entities.json").write_text(
        json.dumps(bm25_entities, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # Instantiate once to validate corpus and make querying cheap to add later.
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

    report = quality_report(pages, facts, cohorts, results, section_tables, bm25_entities, vector_docs)
    (args.output_dir / "quality_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output_dir / "raw_page_extractions.json").write_text(json.dumps([
        {"page_number": p.page_number, "text": p.text, "tables": p.tables,
         "normalized": e.model_dump()} for p, e in zip(pages, extracted)
    ], ensure_ascii=False, indent=2), encoding="utf-8")
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
