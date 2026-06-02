"""
Citation pipeline for AI-Scientist-v2.

Three-stage author-blind citation system:
  1. vlm_pdf_extractor    — qwen2.5vl:7b extracts structured text+visuals from PDFs
  2. langextract_processor — LangExtract (gemma4:e4b) extracts grounded facts;
                             Qwen3-VL-Embedding-2B indexes them; no LightRAG here
  3. citation_resolver    — replaces [[tags]] with proper LaTeX citations/cross-refs

Retrieval uses two-stage HF pipeline:
  cosine sim (Qwen3-VL-Embedding-2B) → rerank (Qwen3-Reranker-4B) → top-k doc_ids
"""

from .vlm_pdf_extractor import extract_pdf, extract_pdf_batch
from .langextract_processor import (
    build_citation_index,
    query_citations,
    query_citations_hf,
    enrich_registry_from_bibtex,
)
from .citation_resolver import (
    resolve_tags,
    resolve_latex_file,
    load_registry,
    save_registry,
    register_doc,
)

__all__ = [
    "extract_pdf", "extract_pdf_batch",
    "build_citation_index", "query_citations", "query_citations_hf",
    "enrich_registry_from_bibtex",
    "resolve_tags", "resolve_latex_file",
    "load_registry", "save_registry", "register_doc",
]
