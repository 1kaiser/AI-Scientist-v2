"""
Stage 2: LangExtract + Ollama on VLM-extracted text files.

Processes the structured text produced by vlm_pdf_extractor and creates:
  - A grounded fact store: each fact mapped to its doc_id + char position
  - A LightRAG vector index for semantic retrieval
  - A doc_id registry: doc_id → {bibtex_key, year, first_author, ...}

The LLM never sees author names during fact extraction — only doc_ids.
This prevents author bias in the Stage A composition step.
"""

import asyncio
import json
import os
import os.path as osp
import re
from functools import partial
from typing import Any

import langextract as lx
from langextract.providers.ollama import OllamaLanguageModel
from langextract.core.data import Extraction, ExampleData

OLLAMA_HOST  = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL  = os.environ.get("CITATION_EMBED_MODEL", "qwen3-embedding:0.6b")
EMBED_DIM    = 1024
LLM_MODEL    = os.environ.get("CITATION_LLM_MODEL", "gemma4:e4b")

_PROMPT_DESC = (
    "Extract scientific findings from this document excerpt. "
    "For each distinct claim or metric, extract its exact text span. "
    "Label each as one of: result | method | metric | limitation | background"
)

_EXAMPLES = [
    ExampleData(
        text=(
            "[PAGE 3][TEXT] The model achieves mAP of 0.91 on the synthetic dataset "
            "but drops to 0.11 on MNIST cross-domain, indicating severe domain shift."
        ),
        extractions=[
            Extraction(extraction_class="metric",     extraction_text="mAP of 0.91 on the synthetic dataset"),
            Extraction(extraction_class="result",     extraction_text="drops to 0.11 on MNIST cross-domain"),
            Extraction(extraction_class="limitation", extraction_text="severe domain shift"),
        ],
    ),
    ExampleData(
        text=(
            "[PAGE 5][FIGURE] Caption: Training loss and validation mAP over epochs.\n"
            "[PAGE 5][FIGURE] Visual: Line plot showing loss decreasing from 0.8 to 0.01 "
            "over 30 epochs while mAP rises from 0.1 to 1.0."
        ),
        extractions=[
            Extraction(extraction_class="result", extraction_text="loss decreasing from 0.8 to 0.01 over 30 epochs"),
            Extraction(extraction_class="metric", extraction_text="mAP rises from 0.1 to 1.0"),
        ],
    ),
]


def _make_ollama_model() -> OllamaLanguageModel:
    """Return a LangExtract OllamaLanguageModel instance."""
    return OllamaLanguageModel(
        model_id=LLM_MODEL,
        model_url=OLLAMA_HOST,
    )


def extract_facts(text_file: str, doc_id: str) -> list[dict]:
    """
    Run LangExtract on a VLM-extracted text file.

    Returns list of grounded facts:
    {
      "doc_id":     str,
      "claim":      str,
      "claim_type": str,
      "numeric":    str | None,
      "visual_ref": str | None,
      "char_start": int,
      "char_end":   int,
      "source_text": str,
    }
    """
    with open(text_file, encoding="utf-8") as f:
        text = f.read()

    ollama_model = _make_ollama_model()
    try:
        results = lx.extract(
            text,
            prompt_description=_PROMPT_DESC,
            examples=_EXAMPLES,
            model=ollama_model,
            max_char_buffer=3000,
            show_progress=False,
        )
    except Exception as exc:
        print(f"[langextract] {doc_id}: extraction error: {exc}")
        return []

    if not isinstance(results, list):
        results = [results]

    facts = []
    for result in results:
        for extraction in (result.extractions or []):
            ci = extraction.char_interval
            facts.append({
                "doc_id":      doc_id,
                "claim":       extraction.extraction_text,
                "claim_type":  extraction.extraction_class,
                "numeric":     None,
                "visual_ref":  None,
                "char_start":  ci.start if ci else 0,
                "char_end":    ci.end   if ci else 0,
                "source_text": text[ci.start:ci.end][:300] if ci else "",
            })

    print(f"[langextract] {doc_id}: extracted {len(facts)} facts")
    return facts


# ── LightRAG index ─────────────────────────────────────────────────────── #

def _citation_rag_dir(index_dir: str) -> str:
    return osp.join(index_dir, "citation_rag")


async def _make_citation_rag(index_dir: str):
    from lightrag import LightRAG
    from lightrag.llm.ollama import ollama_model_complete, ollama_embed
    from lightrag.utils import EmbeddingFunc

    rag_dir = _citation_rag_dir(index_dir)
    os.makedirs(rag_dir, exist_ok=True)

    rag = LightRAG(
        working_dir=rag_dir,
        llm_model_func=partial(ollama_model_complete, host=OLLAMA_HOST),
        llm_model_name=LLM_MODEL,
        llm_model_kwargs={"options": {"num_ctx": 4096}},
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBED_DIM,
            max_token_size=2048,
            func=partial(ollama_embed, embed_model=EMBED_MODEL, host=OLLAMA_HOST),
        ),
    )
    await rag.initialize_storages()
    return rag


async def _index_facts_async(facts: list[dict], index_dir: str):
    rag = await _make_citation_rag(index_dir)
    for fact in facts:
        # Index: author-blind — only doc_id, claim, type, numeric
        doc = (
            f"DOC_ID:{fact['doc_id']} "
            f"TYPE:{fact['claim_type']} "
            f"CLAIM:{fact['claim']} "
            f"VALUE:{fact.get('numeric','') or ''} "
            f"VISUAL:{fact.get('visual_ref','') or ''}"
        )
        await rag.ainsert(doc)
    await rag.finalize_storages()


def build_citation_index(
    registry_entries: list[dict],
    index_dir: str,
    force: bool = False,
) -> dict[str, dict]:
    """
    Build the full citation index from VLM registry entries.

    Args:
        registry_entries: Output of extract_pdf_batch().
        index_dir:        Directory for RAG index + facts JSON.
        force:            Re-index even if already done.

    Returns:
        doc_registry: {doc_id → {bibtex_key, text_file, facts, ...}}
    """
    os.makedirs(index_dir, exist_ok=True)
    registry_path = osp.join(index_dir, "doc_registry.json")

    # Load existing registry
    doc_registry: dict[str, dict] = {}
    if osp.exists(registry_path) and not force:
        with open(registry_path) as f:
            doc_registry = json.load(f)

    all_new_facts = []
    for entry in registry_entries:
        doc_id = entry["doc_id"]
        if doc_id in doc_registry and not force:
            continue

        facts = extract_facts(entry["text_file"], doc_id)
        all_new_facts.extend(facts)

        doc_registry[doc_id] = {
            "doc_id":     doc_id,
            "bibtex_key": entry["bibtex_key"],
            "pdf_path":   entry["pdf_path"],
            "text_file":  entry["text_file"],
            "facts":      facts,
            # Metadata to be filled in from BibTeX
            "year":        None,
            "first_author": None,
        }

    if all_new_facts:
        print(f"[langextract] Indexing {len(all_new_facts)} facts into LightRAG ...")
        asyncio.run(_index_facts_async(all_new_facts, index_dir))

    # Save updated registry
    with open(registry_path, "w") as f:
        json.dump(doc_registry, f, indent=2)

    return doc_registry


def enrich_registry_from_bibtex(
    doc_registry: dict[str, dict],
    bib_text: str,
) -> dict[str, dict]:
    """
    Parse BibTeX string and add year + first_author to each registry entry.
    Called after build_citation_index so citation_resolver can sort properly.
    """
    for doc_id, entry in doc_registry.items():
        key = entry["bibtex_key"]
        # Find the @type{key, ... } block
        pattern = rf"@\w+\{{{re.escape(key)}\b(.*?)^\}}"
        m = re.search(pattern, bib_text, re.DOTALL | re.MULTILINE)
        if not m:
            continue
        block = m.group(1)

        year_m   = re.search(r"year\s*=\s*\{?(\d{4})\}?", block)
        author_m = re.search(r"author\s*=\s*\{([^}]+)\}", block)

        if year_m:
            entry["year"] = int(year_m.group(1))
        if author_m:
            # First author surname: "Last, First and ..." or "First Last and ..."
            authors = author_m.group(1).split(" and ")
            first   = authors[0].strip()
            surname = first.split(",")[0].strip() if "," in first else first.split()[-1]
            entry["first_author"] = surname.lower()

    return doc_registry


async def _query_async(index_dir: str, topic: str, top_k_docs: int) -> str:
    from lightrag import QueryParam
    rag = await _make_citation_rag(index_dir)
    result = await rag.aquery(topic, param=QueryParam(mode="naive"))
    await rag.finalize_storages()
    return result or ""


def query_citations(
    index_dir: str,
    topic: str,
    top_k_docs: int = 5,
) -> list[str]:
    """
    Query the citation index for doc_ids relevant to topic.

    Returns list of doc_ids sorted by relevance (most relevant first).
    """
    result = asyncio.run(_query_async(index_dir, topic, top_k_docs))

    # Extract DOC_ID: references from retrieved text
    doc_ids = re.findall(r"DOC_ID:(\S+)", result)
    seen, unique = set(), []
    for d in doc_ids:
        if d not in seen:
            seen.add(d)
            unique.append(d)
    return unique[:top_k_docs]
