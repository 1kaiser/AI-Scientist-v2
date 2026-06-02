"""
End-to-end paper ingestion pipeline.

Usage:
    conda run -n ai_scientist python scripts/process_papers.py \\
        --input  data/papers/raw/          \\
        --output data/papers/              \\
        --index  data/papers/citation_index/ \\
        [--bibtex data/papers/refs.bib]    \\
        [--max-pages 20]                   \\
        [--force]

Folder layout produced:
    data/papers/
    ├── raw/                    ← user drops PDFs here
    ├── extracted/
    │   └── ref_01/
    │       ├── ref_01.json     ← VLM extraction registry
    │       └── ref_01.txt      ← flat text for LangExtract
    ├── citation_index/
    │   ├── doc_registry.json   ← central registry
    │   └── citation_rag/       ← LightRAG vector index
    └── processing_log.json     ← status of every PDF ever seen
"""

import argparse
import json
import os
import os.path as osp
import re
import sys
from datetime import datetime

sys.path.insert(0, osp.dirname(osp.dirname(__file__)))

from ai_scientist.citation_pipeline.vlm_pdf_extractor import extract_pdf
from ai_scientist.citation_pipeline.langextract_processor import (
    build_citation_index,
    enrich_registry_from_bibtex,
)
from ai_scientist.citation_pipeline.citation_resolver import (
    load_registry,
    save_registry,
)


# ── Helpers ─────────────────────────────────────────────────────────────── #

def _load_log(output_dir: str) -> dict:
    path = osp.join(output_dir, "processing_log.json")
    if osp.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_log(output_dir: str, log: dict) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = osp.join(output_dir, "processing_log.json")
    with open(path, "w") as f:
        json.dump(log, f, indent=2)


def _next_doc_id(log: dict) -> str:
    existing = [k for k in log.keys() if re.match(r"ref_\d+", k)]
    n = len(existing) + 1
    return f"ref_{n:02d}"


def _infer_bibtex_key(pdf_path: str) -> str:
    """
    Infer a BibTeX key from the filename.
    e.g. '1-s2.0-S0022169426001757-main.pdf' → 'paper_2024_s0022169426'
         'smith2023_deeplearning.pdf'          → 'smith2023_deeplearning'
    """
    stem = osp.splitext(osp.basename(pdf_path))[0]
    # Try to extract year
    year_match = re.search(r"(20\d{2}|19\d{2})", stem)
    year = year_match.group() if year_match else "unknown"
    # Normalise: lowercase, replace non-alnum with _
    key = re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")
    # Truncate to reasonable length
    if len(key) > 30:
        key = key[:30].rstrip("_")
    return key


def _find_pdfs(input_path: str) -> list[str]:
    if osp.isfile(input_path):
        return [input_path] if input_path.lower().endswith(".pdf") else []
    pdfs = []
    for root, _, files in os.walk(input_path):
        for f in sorted(files):
            if f.lower().endswith(".pdf"):
                pdfs.append(osp.join(root, f))
    return pdfs


# ── Main pipeline ────────────────────────────────────────────────────────── #

def process_papers(
    input_path: str,
    output_dir: str,
    index_dir: str,
    bibtex_path: str | None = None,
    max_pages: int = 20,
    force: bool = False,
) -> dict:
    """
    Process all PDFs in input_path → extracted/ → citation_index/.

    Returns the updated processing log.
    """
    extracted_dir = osp.join(output_dir, "extracted")
    os.makedirs(extracted_dir, exist_ok=True)
    os.makedirs(index_dir, exist_ok=True)

    log = _load_log(output_dir)
    pdfs = _find_pdfs(input_path)

    if not pdfs:
        print(f"No PDFs found in {input_path}")
        return log

    print(f"Found {len(pdfs)} PDF(s). Already processed: "
          f"{sum(1 for e in log.values() if e.get('status') == 'done')}")

    # Load optional BibTeX
    bib_text = ""
    if bibtex_path and osp.exists(bibtex_path):
        with open(bibtex_path) as f:
            bib_text = f.read()
        print(f"Loaded BibTeX from {bibtex_path}")

    new_entries = []

    for pdf_path in pdfs:
        abs_pdf = osp.abspath(pdf_path)

        # Check if already processed
        existing = next(
            (e for e in log.values() if osp.abspath(e.get("pdf_path","")) == abs_pdf),
            None,
        )
        if existing and existing.get("status") == "done" and not force:
            print(f"  SKIP (done): {osp.basename(pdf_path)}")
            continue

        doc_id = existing["doc_id"] if existing else _next_doc_id(log)
        bibtex_key = existing.get("bibtex_key") if existing else _infer_bibtex_key(pdf_path)

        print(f"\n  Processing: {osp.basename(pdf_path)}")
        print(f"    doc_id={doc_id}  bibtex_key={bibtex_key}")

        doc_out_dir = osp.join(extracted_dir, doc_id)

        # Update log: extracting
        log[doc_id] = {
            "pdf_path": abs_pdf,
            "doc_id": doc_id,
            "bibtex_key": bibtex_key,
            "status": "extracting",
            "started_at": datetime.utcnow().isoformat(),
            "processed_at": None,
            "pages_extracted": 0,
            "facts_extracted": 0,
            "error": None,
        }
        _save_log(output_dir, log)

        # Stage 1: VLM extraction
        try:
            entry = extract_pdf(
                pdf_path=abs_pdf,
                out_dir=doc_out_dir,
                doc_id=doc_id,
                bibtex_key=bibtex_key,
                max_pages=max_pages,
                force=force,
            )
            log[doc_id]["pages_extracted"] = len(entry.get("elements", []))
            log[doc_id]["status"] = "indexing"
            _save_log(output_dir, log)
            print(f"    VLM extraction: {log[doc_id]['pages_extracted']} elements")
            new_entries.append(entry)

        except Exception as exc:
            log[doc_id]["status"] = "failed"
            log[doc_id]["error"] = str(exc)
            _save_log(output_dir, log)
            print(f"    FAILED (extraction): {exc}")
            continue

    if not new_entries:
        print("\nNothing new to index.")
        return log

    # Stage 2: Build citation index for all new entries
    print(f"\n  Building citation index for {len(new_entries)} new document(s)...")
    try:
        registry = build_citation_index(
            registry_entries=new_entries,
            index_dir=index_dir,
            force=force,
        )

        # Enrich with BibTeX metadata if available
        if bib_text:
            registry = enrich_registry_from_bibtex(registry, bib_text)
            save_registry(registry, index_dir)
            print(f"  BibTeX enrichment: year/author added to {len(registry)} entries")

        # Mark all new entries as done
        for entry in new_entries:
            doc_id = entry["doc_id"]
            facts = registry.get(doc_id, {}).get("facts", [])
            log[doc_id]["facts_extracted"] = len(facts)
            log[doc_id]["status"] = "done"
            log[doc_id]["processed_at"] = datetime.utcnow().isoformat()
        _save_log(output_dir, log)

    except Exception as exc:
        for entry in new_entries:
            log[entry["doc_id"]]["status"] = "failed"
            log[entry["doc_id"]]["error"] = f"indexing: {exc}"
        _save_log(output_dir, log)
        print(f"  FAILED (indexing): {exc}")

    # Summary table
    print("\n" + "="*65)
    print(f"{'doc_id':<10} {'status':<12} {'pages':>6} {'facts':>6}  bibtex_key")
    print("-"*65)
    for doc_id, entry in sorted(log.items()):
        print(
            f"{entry['doc_id']:<10} {entry['status']:<12} "
            f"{entry['pages_extracted']:>6} {entry['facts_extracted']:>6}  "
            f"{entry['bibtex_key']}"
        )
    print("="*65)
    print(f"Index location: {osp.abspath(index_dir)}")
    print(f"Log location:   {osp.abspath(osp.join(output_dir, 'processing_log.json'))}")

    return log


# ── CLI ──────────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PDFs into citation index.")
    parser.add_argument("--input",     required=True,  help="PDF file or folder of PDFs")
    parser.add_argument("--output",    default="data/papers", help="Base output folder")
    parser.add_argument("--index",     default="data/papers/citation_index", help="Citation index dir")
    parser.add_argument("--bibtex",    default=None,   help="Optional master .bib file")
    parser.add_argument("--max-pages", type=int, default=20, help="Max PDF pages to process")
    parser.add_argument("--force",     action="store_true", help="Re-process even if done")
    args = parser.parse_args()

    process_papers(
        input_path=args.input,
        output_dir=args.output,
        index_dir=args.index,
        bibtex_path=args.bibtex,
        max_pages=args.max_pages,
        force=args.force,
    )
