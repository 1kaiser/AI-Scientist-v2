"""
LaTeX syntax RAG using LightRAG + local Ollama embeddings.

Indexes data/latex2e.txt once; subsequent calls hit the cache.
Supports two embedding backends for benchmarking:
  - qwen3-embedding:0.6b  (1024-dim, 32K ctx, best MTEB retrieval 64.65)
  - embeddinggemma:latest (768-dim,  2K ctx, 308M params, retrieval 62.49)

Usage:
    from ai_scientist.latex_rag import query_latex_help
    snippet = query_latex_help("figure environment caption label")
"""

import asyncio
import os
import os.path as osp
from functools import partial

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
LATEX_TXT = osp.join(osp.dirname(__file__), "..", "data", "latex2e.txt")
_RAG_BASE = osp.join(osp.dirname(__file__), "..", "data")

EMBED_CONFIGS = {
    "qwen3-embedding:0.6b": {
        "dim": 1024,
        "max_token_size": 8192,
        "chunk_size": 6000,
    },
    "embeddinggemma:latest": {
        "dim": 768,
        "max_token_size": 1800,
        "chunk_size": 1400,
    },
}

DEFAULT_EMBED_MODEL = os.environ.get("LATEX_RAG_EMBED_MODEL", "qwen3-embedding:0.6b")


def _rag_dir(embed_model: str) -> str:
    safe = embed_model.replace(":", "_").replace("/", "_")
    return osp.join(_RAG_BASE, f"latex_rag_{safe}")


async def _make_rag(embed_model: str):
    from lightrag import LightRAG
    from lightrag.llm.ollama import ollama_model_complete, ollama_embed
    from lightrag.utils import EmbeddingFunc

    cfg = EMBED_CONFIGS[embed_model]
    working_dir = _rag_dir(embed_model)
    os.makedirs(working_dir, exist_ok=True)

    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=partial(ollama_model_complete, host=OLLAMA_HOST),
        llm_model_name="gemma4:e4b",
        llm_model_kwargs={"options": {"num_ctx": 4096}},
        embedding_func=EmbeddingFunc(
            embedding_dim=cfg["dim"],
            max_token_size=cfg["max_token_size"],
            func=partial(ollama_embed, embed_model=embed_model, host=OLLAMA_HOST),
        ),
    )
    await rag.initialize_storages()
    return rag


async def _build_index(rag, embed_model: str) -> bool:
    """Index latex2e.txt. Returns True if already indexed."""
    flag = osp.join(_rag_dir(embed_model), ".indexed")
    if osp.exists(flag):
        return True
    if not osp.exists(LATEX_TXT):
        print(f"[latex_rag] latex2e.txt not found at {LATEX_TXT}")
        return False

    with open(LATEX_TXT, encoding="utf-8", errors="ignore") as f:
        text = f.read()

    chunk_size = EMBED_CONFIGS[embed_model]["chunk_size"]
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    total = len(chunks)
    print(f"[latex_rag] Indexing {total} chunks ({embed_model}) ...")

    for i, chunk in enumerate(chunks):
        await rag.ainsert(chunk)
        if (i + 1) % 10 == 0 or (i + 1) == total:
            print(f"  [{i+1}/{total}]")

    open(flag, "w").close()
    print(f"[latex_rag] Index built → {_rag_dir(embed_model)}")
    return True


async def _async_query(embed_model: str, topic: str, mode: str) -> str:
    from lightrag import QueryParam

    rag = await _make_rag(embed_model)
    await _build_index(rag, embed_model)
    result = await rag.aquery(topic, param=QueryParam(mode=mode))
    await rag.finalize_storages()
    return result or ""


def query_latex_help(
    topic: str,
    embed_model: str = DEFAULT_EMBED_MODEL,
    mode: str = "naive",
) -> str:
    """
    Query LaTeX syntax reference for the given topic.

    Args:
        topic:       Natural language description of what LaTeX help is needed.
        embed_model: One of EMBED_CONFIGS keys (default: qwen3-embedding:0.6b).
        mode:        LightRAG query mode — "naive" | "local" | "hybrid".
                     "naive" = pure vector search (fast, no graph).

    Returns:
        Relevant LaTeX reference text (empty string on failure).
    """
    if embed_model not in EMBED_CONFIGS:
        print(f"[latex_rag] Unknown model {embed_model!r}, using default.")
        embed_model = DEFAULT_EMBED_MODEL
    try:
        return asyncio.run(_async_query(embed_model, topic, mode))
    except RuntimeError:
        # Nested event loop (e.g. Jupyter) — fall back to thread executor
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(asyncio.run, _async_query(embed_model, topic, mode))
            return fut.result(timeout=120) or ""
    except Exception as exc:
        print(f"[latex_rag] Query failed: {exc}")
        return ""


def build_index_for_model(embed_model: str) -> None:
    """Explicitly pre-build index for an embedding model."""
    async def _run():
        rag = await _make_rag(embed_model)
        await _build_index(rag, embed_model)
        await rag.finalize_storages()
    asyncio.run(_run())
