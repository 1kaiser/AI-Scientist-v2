"""
Benchmark: EmbeddingGemma vs Qwen3-Embedding-0.6B for LaTeX RAG retrieval.

Usage:
    conda run -n ai_scientist python scripts/benchmark_latex_rag.py

Measures per model:
  - Index build time (first run) / cache hit time (subsequent)
  - Per-query latency
  - Keyword recall: fraction of expected terms found in retrieved chunk
  - Mean reciprocal rank proxy (keyword position in result)

Results are printed as a table and saved to data/latex_rag_benchmark.json
"""

import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ai_scientist.latex_rag import (
    EMBED_CONFIGS,
    _async_query,
    _build_index,
    _make_rag,
    _rag_dir,
)

# --------------------------------------------------------------------------- #
# Test suite: (query, expected_keywords)
# --------------------------------------------------------------------------- #
QUERIES = [
    (
        "figure environment includegraphics caption label",
        ["figure", "includegraphics", "caption", "label"],
    ),
    (
        "bibliography bibtex cite references",
        ["bibliography", "cite", "bibtex", "bibitem"],
    ),
    (
        "equation align math environment",
        ["equation", "align", "math", "displaymath"],
    ),
    (
        "table tabular multirow multicolumn",
        ["tabular", "table", "hline", "multicolumn"],
    ),
    (
        "section subsection paragraph heading structure",
        ["section", "subsection", "paragraph", "chapter"],
    ),
    (
        "itemize enumerate description list environment",
        ["itemize", "enumerate", "item", "description"],
    ),
    (
        "documentclass usepackage preamble options",
        ["documentclass", "usepackage", "begin", "end"],
    ),
    (
        "footnote margin note text formatting",
        ["footnote", "marginpar", "textrm", "textit"],
    ),
]

MODELS = list(EMBED_CONFIGS.keys())
MODES = ["naive", "local"]


async def run_benchmark():
    results = {}

    for embed_model in MODELS:
        print(f"\n{'='*60}")
        print(f"MODEL: {embed_model}")
        print(f"{'='*60}")

        model_results = {"index_time_s": None, "queries": []}

        # ── Build / warm index ──────────────────────────────────────────── #
        flag = os.path.join(_rag_dir(embed_model), ".indexed")
        already_indexed = os.path.exists(flag)

        t0 = time.perf_counter()
        rag = await _make_rag(embed_model)
        await _build_index(rag, embed_model)
        await rag.finalize_storages()
        index_time = time.perf_counter() - t0

        model_results["index_time_s"] = round(index_time, 2)
        model_results["index_cached"] = already_indexed
        print(
            f"  Index: {'cache hit' if already_indexed else 'built'} "
            f"in {index_time:.1f}s"
        )

        # ── Run queries ─────────────────────────────────────────────────── #
        for mode in MODES:
            for query, keywords in QUERIES:
                t0 = time.perf_counter()
                result = await _async_query(embed_model, query, mode)
                latency = time.perf_counter() - t0

                result_lower = result.lower()
                found = [kw for kw in keywords if kw in result_lower]
                recall = len(found) / len(keywords)

                entry = {
                    "mode": mode,
                    "query": query[:50],
                    "latency_s": round(latency, 2),
                    "recall": round(recall, 2),
                    "found": found,
                    "missing": [kw for kw in keywords if kw not in result_lower],
                    "result_len": len(result),
                }
                model_results["queries"].append(entry)
                print(
                    f"  [{mode}] {query[:40]:<42} "
                    f"recall={recall:.0%}  {latency:.1f}s"
                )

        results[embed_model] = model_results

    # ── Summary table ───────────────────────────────────────────────────── #
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<35} {'Mode':<8} {'Avg Recall':>10} {'Avg Latency':>12}")
    print("-" * 70)

    for model, data in results.items():
        for mode in MODES:
            qs = [q for q in data["queries"] if q["mode"] == mode]
            avg_recall = sum(q["recall"] for q in qs) / len(qs)
            avg_latency = sum(q["latency_s"] for q in qs) / len(qs)
            print(
                f"{model:<35} {mode:<8} {avg_recall:>9.1%} {avg_latency:>11.2f}s"
            )

    # ── Save results ────────────────────────────────────────────────────── #
    out_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "latex_rag_benchmark.json"
    )
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    asyncio.run(run_benchmark())
