"""
Stage 2: LangExtract + HF embedding/reranking on VLM-extracted text files.

Pipeline:
  extract_facts()       — LangExtract + gemma4:e4b extracts grounded claims
  build_citation_index()— embeds facts with Qwen3-VL-Embedding-2B → .npy
  query_citations()     — cosine sim → Qwen3-Reranker-4B → top-k doc_ids

No LightRAG in this stage (LightRAG is used only in latex_rag.py for the
LaTeX syntax knowledge graph). Facts are stored as numpy arrays alongside
doc_registry.json for fast, deterministic retrieval without LLM calls.
"""

import json
import os
import os.path as osp
import re

import langextract as lx
from langextract.providers.ollama import OllamaLanguageModel
from langextract.core.data import Extraction, ExampleData

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL   = os.environ.get("CITATION_LLM_MODEL", "gemma4:e4b")

# HF index artefacts stored alongside doc_registry.json
_EMB_FILE   = "fact_embeddings.npy"
_INDEX_FILE = "fact_index.json"

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


# ── Stage 2a: LangExtract fact extraction ──────────────────────────────── #

def _filter_text_for_facts(text: str) -> str:
    """
    Keep only text/table/equation lines from the rich-tagged .txt file.
    Drops [FIGURE-REGION] lines — VLM figure descriptions are not grounded
    scientific claims suitable for fact extraction.
    Also drops [FALLBACK] lines from pages where VLM failed entirely.
    """
    kept = []
    for line in text.splitlines():
        # Keep untagged lines (header, blank)
        if not line.startswith("[PAGE"):
            kept.append(line)
            continue
        # Drop figure-region and fallback lines
        if "[FIGURE-REGION]" in line or "[FALLBACK]" in line:
            continue
        kept.append(line)
    return "\n".join(kept)


def extract_facts(text_file: str, doc_id: str,
                  out_dir: str | None = None) -> list[dict]:
    """
    Run LangExtract on a VLM-extracted text file.

    Filters out figure-region and fallback lines before sending to the LLM
    so that only grounded text claims are extracted.

    Returns list of grounded facts:
      {doc_id, claim, claim_type, char_start, char_end, source_text}
    """
    with open(text_file, encoding="utf-8") as f:
        raw_text = f.read()

    text = _filter_text_for_facts(raw_text)

    ollama_model = OllamaLanguageModel(model_id=LLM_MODEL, model_url=OLLAMA_HOST)
    try:
        results = lx.extract(
            text,  # filtered text (no figure regions)
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
            # LangExtract 1.5 uses .begin/.end; guard with getattr for compat
            c_start = getattr(ci, "begin", None) or getattr(ci, "start", 0) or 0
            c_end   = getattr(ci, "end", 0) or 0
            facts.append({
                "doc_id":      doc_id,
                "claim":       extraction.extraction_text,
                "claim_type":  extraction.extraction_class,
                "char_start":  c_start if ci else 0,
                "char_end":    c_end   if ci else 0,
                "source_text": text[c_start:c_end][:300] if ci else "",
            })

    print(f"[langextract] {doc_id}: extracted {len(facts)} facts")

    # Update per-doc log if out_dir provided
    if out_dir:
        try:
            from ai_scientist.citation_pipeline.vlm_pdf_extractor import load_doc_log, _save_doc_log
            doc_log = load_doc_log(out_dir, doc_id)
            doc_log.setdefault("stages", {})["langextract"] = {
                "status": "done", "n_facts": len(facts), "llm": LLM_MODEL,
            }
            _save_doc_log(out_dir, doc_id, doc_log)
        except Exception:
            pass

    return facts


# ── Stage 2b: HF embedding index ───────────────────────────────────────── #

def _index_paths(index_dir: str) -> tuple[str, str]:
    return osp.join(index_dir, _EMB_FILE), osp.join(index_dir, _INDEX_FILE)


def _build_hf_index(index_dir: str, doc_registry: dict) -> None:
    """
    Embed all facts with Qwen3-VL-Embedding-2B and save to disk.

    Outputs:
      fact_embeddings.npy  — (N, 2048) float32
      fact_index.json      — [{doc_id, fact_text}, ...]
    """
    import numpy as np
    from ai_scientist.hf_embed import get_embedder

    fact_records = []
    for doc_id, entry in doc_registry.items():
        for fact in entry.get("facts", []):
            fact_records.append({
                "doc_id":    doc_id,
                "fact_text": f"{fact['claim_type']}: {fact['claim']}",
            })

    if not fact_records:
        print("[hf_index] No facts to embed.")
        return

    embedder = get_embedder()
    instr = "Retrieve scientific facts relevant to this research query."
    texts = [r["fact_text"] for r in fact_records]
    print(f"[hf_index] Embedding {len(texts)} facts with Qwen3-VL-Embedding-2B ...")
    embs = embedder.embed_texts(texts, instruction=instr, batch_size=8)

    emb_path, idx_path = _index_paths(index_dir)
    np.save(emb_path, embs)
    with open(idx_path, "w") as f:
        json.dump(fact_records, f)
    print(f"[hf_index] Saved {embs.shape[0]} embeddings → {emb_path}")


def build_citation_index(
    registry_entries: list[dict],
    index_dir: str,
    force: bool = False,
) -> dict[str, dict]:
    """
    Extract facts from VLM registry entries and build the HF embedding index.

    Args:
        registry_entries: Output of extract_pdf() / extract_pdf_batch().
        index_dir:        Directory for index files + doc_registry.json.
        force:            Re-index even if doc already in registry.

    Returns:
        doc_registry: {doc_id → {bibtex_key, text_file, facts, ...}}
    """
    os.makedirs(index_dir, exist_ok=True)
    registry_path = osp.join(index_dir, "doc_registry.json")

    doc_registry: dict[str, dict] = {}
    if osp.exists(registry_path) and not force:
        with open(registry_path) as f:
            doc_registry = json.load(f)

    new_docs = []
    for entry in registry_entries:
        doc_id = entry["doc_id"]
        if doc_id in doc_registry and not force:
            continue

        # Pass out_dir so extract_facts can update the per-doc log
        doc_out_dir = osp.dirname(entry["text_file"])
        facts = extract_facts(entry["text_file"], doc_id, out_dir=doc_out_dir)
        doc_registry[doc_id] = {
            "doc_id":       doc_id,
            "bibtex_key":   entry["bibtex_key"],
            "pdf_path":     entry["pdf_path"],
            "text_file":    entry["text_file"],
            "facts":        facts,
            "year":         None,
            "first_author": None,
        }
        new_docs.append(doc_id)

    if new_docs:
        _build_hf_index(index_dir, doc_registry)

        # Mark embedding stage done in each new doc's log
        for entry in registry_entries:
            doc_id = entry["doc_id"]
            if doc_id not in new_docs:
                continue
            doc_out_dir = osp.dirname(entry["text_file"])
            try:
                from ai_scientist.citation_pipeline.vlm_pdf_extractor import load_doc_log, _save_doc_log
                doc_log = load_doc_log(doc_out_dir, doc_id)
                doc_log.setdefault("stages", {})["hf_embedding"] = {
                    "status": "done",
                    "n_facts": len(doc_registry.get(doc_id, {}).get("facts", [])),
                    "embed_model": "Qwen/Qwen3-VL-Embedding-2B",
                }
                _save_doc_log(doc_out_dir, doc_id, doc_log)
            except Exception:
                pass

    with open(registry_path, "w") as f:
        json.dump(doc_registry, f, indent=2)

    return doc_registry


# ── Stage 2c: two-stage retrieval (embed → rerank) ─────────────────────── #

def query_citations(
    index_dir: str,
    topic: str,
    top_k_docs: int = 5,
    n_candidates: int = 20,
) -> list[str]:
    """
    Retrieve doc_ids relevant to topic using two-stage HF retrieval:
      1. Qwen3-VL-Embedding-2B cosine similarity → top-N candidate facts
      2. Qwen3-Reranker-4B cross-encoder         → top-k final doc_ids

    Returns list of doc_ids sorted by reranker score (best first).
    """
    import numpy as np
    from ai_scientist.hf_embed import get_embedder
    from ai_scientist.hf_rerank import get_reranker

    emb_path, idx_path = _index_paths(index_dir)
    if not osp.exists(emb_path) or not osp.exists(idx_path):
        print(f"[citation] HF index not found at {index_dir} — returning empty.")
        return []

    with open(idx_path) as f:
        fact_records = json.load(f)
    fact_embs = np.load(emb_path)                    # (N, 2048)

    # Stage 1: cosine similarity
    embedder  = get_embedder()
    instr     = "Retrieve scientific facts relevant to this research query."
    query_emb = embedder.embed_texts([topic], instruction=instr)[0]   # (2048,)
    cos_scores = fact_embs @ query_emb                                 # (N,)
    n_cand     = min(n_candidates, len(fact_records))
    top_idx    = cos_scores.argsort()[::-1][:n_cand]
    candidates = [(fact_records[i], float(cos_scores[i])) for i in top_idx]
    print(f"[citation] Stage 1: {n_cand} candidates from {len(fact_records)} facts")

    # Stage 2: reranking
    reranker     = get_reranker()
    cand_texts   = [c[0]["fact_text"] for c in candidates]
    rerank_scores = reranker.rerank(topic, cand_texts)

    ranked = sorted(
        zip(candidates, rerank_scores),
        key=lambda x: x[1],
        reverse=True,
    )

    seen:   set[str]  = set()
    result: list[str] = []
    for (record, _cos), rr in ranked:
        doc_id = record["doc_id"]
        if doc_id not in seen:
            seen.add(doc_id)
            result.append(doc_id)
            print(f"[citation]   {doc_id}  rerank={rr:.4f}")
        if len(result) >= top_k_docs:
            break

    return result


# kept as alias so existing callers of query_citations_hf still work
query_citations_hf = query_citations


# ── BibTeX enrichment ───────────────────────────────────────────────────── #

def enrich_registry_from_bibtex(
    doc_registry: dict[str, dict],
    bib_text: str,
) -> dict[str, dict]:
    """Add year + first_author to each registry entry from a BibTeX string."""
    for doc_id, entry in doc_registry.items():
        key = entry["bibtex_key"]
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
            authors  = author_m.group(1).split(" and ")
            first    = authors[0].strip()
            surname  = first.split(",")[0].strip() if "," in first else first.split()[-1]
            entry["first_author"] = surname.lower()

    return doc_registry
