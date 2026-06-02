"""
Citation pipeline for AI-Scientist-v2.

Three-stage author-blind citation system:
  1. vlm_pdf_extractor   — qwen2.5vl:7b extracts structured text+visuals from PDFs
  2. langextract_processor — LangExtract + Ollama indexes grounded facts by doc_id
  3. citation_resolver   — replaces [[tags]] with proper LaTeX citations/cross-refs

Usage:
    from ai_scientist.citation_pipeline import full_pipeline
    full_pipeline(pdf_dir="refs/", output_tex="latex/template.tex")
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
