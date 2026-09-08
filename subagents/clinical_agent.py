from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.vectorstores import FAISS

import numpy as np
from typing import Any
from rank_bm25 import BM25Okapi
import bm25s
from sentence_transformers import SentenceTransformer

import torch
# Uncomment to verify GPU availability before running on CUDA:
# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name(0))

from langchain_core.tools import BaseTool
from pydantic import Field


class ClinicalSubAgent(BaseTool):
    """LangChain BaseTool implementing a two-stage hybrid retrieval pipeline
    over clinical documents.

    Stage 1 — BM25 (lexical): fast keyword-based pre-filtering that narrows
    the full corpus down to the top bm25_top_k candidates.  BM25 excels at
    exact term matching (drug names, ICD codes, gene symbols, etc.).

    Stage 2 — Dense re-ranking: the BM25 candidates are re-scored using
    cosine similarity between their dense embeddings and the query embedding.
    This captures semantic similarity that BM25 misses (synonyms, paraphrases).

    The two-stage design avoids embedding the entire corpus at query time while
    still benefiting from semantic understanding for the final ranking.
    """

    name: str = "clinical_search"
    description: str = "Search clinical documents using hybrid BM25 + dense retrieval."

    # Full list of LangChain Documents loaded from the FAISS docstore.
    # Pydantic treats this as a required constructor argument.
    documents: list[Document]

    # SentenceTransformer (or compatible) model used for dense encoding.
    # Typed as Any to avoid Pydantic trying to validate the torch model object.
    model: Any

    # Stored query string; set externally before invoke() if needed,
    # but _run() always uses the query passed directly as an argument.
    query: str = ""

    # Number of top BM25 candidates to pass to the dense re-ranking stage.
    # Higher values improve recall at the cost of more embedding computations.
    bm25_top_k: int = 100

    # Final number of documents returned to the caller after dense re-ranking.
    final_k: int = 10

    # Plain-text list of document contents, built from `documents` in post-init.
    # Stored as a field so it is accessible across methods without recomputation.
    texts: list[str] = []

    # Fitted BM25Okapi index, built from `texts` in post-init.
    # Typed as Any because BM25Okapi is not a Pydantic-compatible type.
    bm25: Any = None

    def model_post_init(self, __context):
        """Pydantic v2 post-initialisation hook — runs after __init__.

        Extracts plain text from each Document and builds the BM25 index.
        This is done here rather than in __init__ so that Pydantic field
        validation completes before we access self.documents.

        Tokenisation strategy: simple whitespace split on lowercased text.
        This is intentionally lightweight; clinical abbreviations and codes
        (e.g. "HLA-A", "eGFR") are preserved as single tokens by BM25Okapi.
        """
        self.texts = [doc.page_content for doc in self.documents]
        # BM25Okapi expects a list of token lists (pre-tokenised corpus).
        self.bm25 = BM25Okapi([t.lower().split() for t in self.texts])

    def _hybrid_search(self, query: str) -> list[dict[str, Any]]:
        """Run the two-stage BM25 → dense hybrid search for a given query.

        Stage 1 — BM25 pre-filter:
            Score all documents with BM25Okapi and select the top bm25_top_k
            indices sorted by descending BM25 score.  argsort returns ascending
            order, so [::-1] reverses it to descending.

        Stage 2 — Dense re-ranking:
            Encode only the BM25 candidate texts (not the full corpus) and the
            query with the SentenceTransformer model using L2-normalised
            embeddings.  Cosine similarity then reduces to a dot product:
                similarity(q, d) = q · d  (since ||q|| = ||d|| = 1)
            Matrix multiplication (doc_embs @ query_emb) computes all dot
            products in one vectorised operation, which is faster than a loop.

        Args:
            query: natural-language search string from the user or agent.

        Returns:
            List of up to final_k dicts, each with:
                score: float cosine similarity (higher = more relevant)
                text:  raw page_content of the matched document
            Sorted by descending score.
        """
        # --- Stage 1: BM25 candidate selection ---
        bm25_scores = self.bm25.get_scores(query.lower().split())
        # argsort gives ascending indices; [::-1] reverses to descending score order.
        candidates = np.argsort(bm25_scores)[::-1][:self.bm25_top_k]

        # --- Stage 2: Dense re-ranking over BM25 candidates ---
        # Encode the query as a 1-D normalised vector.
        query_emb = self.model.encode(query, normalize_embeddings=True)

        # Encode only the candidate subset; batch_size=64 balances GPU memory
        # and throughput. show_progress_bar=True helps monitor long batches.
        doc_embs = self.model.encode(
            [self.texts[i] for i in candidates],
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        # Vectorised cosine similarity: shape (bm25_top_k,)
        dense_scores = doc_embs @ query_emb
        # Commented alternative — explicit loop, kept for reference:
        # dense_scores = np.array([np.dot(doc_emb, query_emb) for doc_emb in doc_embs])

        # Select the top final_k indices within the candidate subset.
        best = np.argsort(dense_scores)[::-1][:self.final_k]

        # Map back to original corpus indices via `candidates` and return results.
        return [
            {"score": float(dense_scores[i]), "text": self.texts[candidates[i]]}
            for i in best
        ]

    def _run(self, query: str) -> str:
        """BaseTool entry point called by the LangGraph ReAct agent.

        Runs hybrid search and concatenates the retrieved document texts into
        a single string separated by double newlines, which the agent can read
        as a block of evidence to answer the user's question.

        Args:
            query: the natural-language question forwarded by the agent.

        Returns:
            Newline-separated concatenation of the top final_k document texts.
        """
        results = self._hybrid_search(query)
        return "\n\n".join(r["text"] for r in results)


# ---------------------------------------------------------------------------
# Usage example — runs only when this file is executed directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # Load the domain-specific medical embedding model onto GPU for fast encoding.
    model = SentenceTransformer(
        r"C:\Users\tzhang\Desktop\Project\DP-AI\LangChain-OpenTutorial-main\MedEmbed-large-v0.1",
        device="cuda"
    )

    # Load the pre-built FAISS index from disk.
    # allow_dangerous_deserialization=True is required by LangChain when loading
    # a FAISS index that was pickled; only use with indexes you created yourself.
    vectorstore = FAISS.load_local(
        r"C:\Users\tzhang\Desktop\Project\DP-AI\LangChain-OpenTutorial-main\medical-faiss-db",
        model,
        allow_dangerous_deserialization=True
    )

    # Extract all Document objects from the FAISS docstore dictionary.
    # The docstore maps internal FAISS IDs → Document objects.
    documents = [
        doc
        for doc in vectorstore.docstore._dict.values()
    ]

    query = "impact of HLA antibodies"

    # Instantiate the subagent; BM25 index is built during post-init.
    retriever = ClinicalSubAgent(documents=documents, model=model, query=query)

    # invoke() is the standard LangChain tool call; internally calls _run().
    docs = retriever.invoke(query)
    for doc in docs:
        print(doc.page_content)
