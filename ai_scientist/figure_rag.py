"""
Figure RAG: retrieves relevant experiment figure filenames for a given section topic.

Two backends:
  1. Ollama text embeddings (qwen3-embedding:0.6b) — default, fast, filename→topic match.
  2. HuggingFace Qwen3-VL-Embedding-2B — pixel-level image retrieval, enabled via
     FIGURE_RAG_BACKEND=hf env var. Auto selects GPU/CPU. Loads ~5-6 GB VRAM on float16.

VL-Embedding note (Ollama):
  MedAIBase/Qwen3-VL-Embedding:2b is pulled but packaged as a *completion* model
  in Ollama (capabilities: completion) — /api/embed is unsupported. Use HF backend
  instead for true pixel-level embeddings.
"""

import asyncio
import os
import os.path as osp
import re
from functools import partial
from typing import Optional

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
_RAG_BASE = osp.join(osp.dirname(__file__), "..", "data")

# Backend selection: "ollama" (default, text-only) or "hf" (Qwen3-VL pixel-level)
FIGURE_RAG_BACKEND = os.environ.get("FIGURE_RAG_BACKEND", "ollama")

# Ollama text embed model
FIGURE_EMBED_MODEL = os.environ.get("FIGURE_RAG_EMBED_MODEL", "qwen3-embedding:0.6b")
FIGURE_EMBED_DIM = 1024


def _fig_rag_dir(exp_figures_dir: str) -> str:
    """Per-experiment RAG index directory alongside figures/."""
    return osp.join(osp.dirname(exp_figures_dir), ".figure_rag")


def _describe_figure(filepath: str) -> str:
    """Build a text description from a figure filename."""
    name = osp.splitext(osp.basename(filepath))[0]
    # Strip leading numbering like "02_" or "appendix_06_"
    name = re.sub(r"^(appendix_)?\d+_", "", name)
    # Snake_case → natural language
    return name.replace("_", " ").lower()


async def _make_fig_rag(rag_dir: str):
    from lightrag import LightRAG
    from lightrag.llm.ollama import ollama_model_complete, ollama_embed
    from lightrag.utils import EmbeddingFunc

    os.makedirs(rag_dir, exist_ok=True)
    rag = LightRAG(
        working_dir=rag_dir,
        llm_model_func=partial(ollama_model_complete, host=OLLAMA_HOST),
        llm_model_name="gemma4:e4b",
        llm_model_kwargs={"options": {"num_ctx": 2048}},
        embedding_func=EmbeddingFunc(
            embedding_dim=FIGURE_EMBED_DIM,
            max_token_size=512,
            func=partial(
                ollama_embed,
                embed_model=FIGURE_EMBED_MODEL,
                host=OLLAMA_HOST,
                keep_alive=300,
            ),
        ),
    )
    await rag.initialize_storages()
    return rag


async def _build_figure_index(rag, figures_dir: str, rag_dir: str) -> list[str]:
    """Index all PNG figures. Returns list of indexed filenames."""
    flag = osp.join(rag_dir, ".indexed")
    png_files = sorted(
        f for f in os.listdir(figures_dir) if f.lower().endswith(".png")
    )
    if not png_files:
        return []

    existing_flag = osp.exists(flag)
    if existing_flag:
        # Index exists — return the filenames from the flag file
        with open(flag) as f:
            return [l.strip() for l in f if l.strip()]

    print(f"[figure_rag] Indexing {len(png_files)} figures ...")
    for fname in png_files:
        description = _describe_figure(fname)
        doc = f"FIGURE_FILE:{fname}\nDESCRIPTION:{description}"
        await rag.ainsert(doc)

    with open(flag, "w") as f:
        f.write("\n".join(png_files))
    print(f"[figure_rag] Figure index built at {rag_dir}")
    return png_files


async def _query_figures(figures_dir: str, topic: str, top_k: int = 4) -> list[str]:
    """Return up to top_k existing figure filenames relevant to topic."""
    from lightrag import QueryParam

    rag_dir = _fig_rag_dir(figures_dir)
    rag = await _make_fig_rag(rag_dir)
    png_files = await _build_figure_index(rag, figures_dir, rag_dir)

    if not png_files:
        await rag.finalize_storages()
        return []

    result = await rag.aquery(topic, param=QueryParam(mode="naive"))
    await rag.finalize_storages()

    # Extract FIGURE_FILE: references from result text
    found = re.findall(r"FIGURE_FILE:(\S+\.png)", result or "")
    # Deduplicate, keep only files that actually exist on disk
    seen, out = set(), []
    for f in found:
        if f not in seen and f in png_files:
            seen.add(f)
            out.append(f)
    # Pad with remaining files ordered by description similarity (simple fallback)
    if len(out) < top_k:
        for f in png_files:
            if f not in seen and len(out) < top_k:
                out.append(f)
                seen.add(f)
    return out[:top_k]


def _hf_get_figures(figures_dir: str, section_topic: str, top_k: int) -> list[str]:
    """
    HF backend: rank figures using Qwen3-VL-Embedding-2B pixel-level similarity.
    Embeds both the query text AND each figure image, returns closest matches.
    """
    from ai_scientist.hf_embed import get_embedder
    import numpy as np

    png_files = sorted(f for f in os.listdir(figures_dir) if f.lower().endswith(".png"))
    if not png_files:
        return []

    embedder = get_embedder()

    # Embed query text
    q_emb = embedder.embed_texts([section_topic])           # (1, D)

    # Embed each figure image
    img_paths = [osp.join(figures_dir, f) for f in png_files]
    img_embs = embedder.embed_images(img_paths)             # (N, D)

    # Cosine similarity (embeddings already L2-normalised)
    scores = (img_embs @ q_emb.T).squeeze()                 # (N,)
    ranked = np.argsort(scores)[::-1]
    return [png_files[i] for i in ranked[:top_k]]


def get_figures_for_section(
    figures_dir: str,
    section_topic: str,
    top_k: int = 4,
) -> list[str]:
    """
    Return up to top_k figure filenames relevant to section_topic.
    All returned filenames are guaranteed to exist in figures_dir.

    Backend selected via FIGURE_RAG_BACKEND env var:
      "ollama"  (default) — text embedding of filenames, fast
      "hf"               — Qwen3-VL-Embedding-2B pixel-level image similarity
    """
    if not osp.isdir(figures_dir):
        return []

    if FIGURE_RAG_BACKEND == "hf":
        try:
            return _hf_get_figures(figures_dir, section_topic, top_k)
        except Exception as exc:
            print(f"[figure_rag] HF backend failed, falling back to Ollama: {exc}")

    try:
        return asyncio.run(_query_figures(figures_dir, section_topic, top_k))
    except RuntimeError:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(asyncio.run, _query_figures(figures_dir, section_topic, top_k))
            return fut.result(timeout=60) or []
    except Exception as exc:
        print(f"[figure_rag] Ollama backend failed: {exc}")
        try:
            return sorted(
                f for f in os.listdir(figures_dir) if f.lower().endswith(".png")
            )[:top_k]
        except Exception:
            return []


def list_valid_figures(figures_dir: str) -> list[str]:
    """Return all PNG filenames that physically exist in figures_dir."""
    if not osp.isdir(figures_dir):
        return []
    return sorted(f for f in os.listdir(figures_dir) if f.lower().endswith(".png"))
