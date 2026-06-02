"""
Stage 1: PDF → structured text via olmOCR-2 (richardyoung/olmocr2:7b-q8).

Per-page layout metadata is analysed FIRST using PyMuPDF block detection,
persisted as {doc_id}_layout.json, and threaded through every downstream stage:

  {doc_id}_layout.json   — per-page: layout type, column split, figure rects,
                            content bbox, margins (pts and DPI-scaled px)
  {doc_id}.json          — element registry; each element carries:
                              page, col ("left"|"right"|"full"|"figure"),
                              source ("text"|"figure"), clip_idx
  {doc_id}.txt           — flat text with rich tags:
                              [PAGE n][2COL-LEFT][TEXT] ...
                              [PAGE n][FIGURE][VISUAL] ...

Downstream consumers:
  langextract_processor  — filters by source="text" for fact extraction;
                           skips figure-only pages for better signal
  hf_embed (_build_hf_index) — tags each fact record with source_type
  vocabulary_extractor   — samples only from text-dense pages (layout.n_text_blocks > 5)
  process_papers         — logs layout_file path + per-page summary in processing_log
"""

import base64
import io
import json
import os
import os.path as osp
import re
from typing import Any

import fitz

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
VLM_MODEL   = os.environ.get("VLM_PDF_MODEL", "richardyoung/olmocr2:7b-q8")


def _save_doc_log(out_dir: str, doc_id: str, log: dict) -> None:
    """Persist per-doc stage log to {doc_id}_log.json."""
    from datetime import datetime
    log["updated_at"] = datetime.utcnow().isoformat()
    path = osp.join(out_dir, f"{doc_id}_log.json")
    with open(path, "w") as f:
        json.dump(log, f, indent=2)


def load_doc_log(out_dir: str, doc_id: str) -> dict:
    """Load per-doc stage log; returns empty dict if not found."""
    path = osp.join(out_dir, f"{doc_id}_log.json")
    if osp.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}
_OCR_DPI    = int(os.environ.get("VLM_PDF_DPI", "100"))
_FIG_DPI    = 120
_MIN_FIG_W  = 80   # pts — smaller images treated as logos/icons, ignored
_MIN_FIG_H  = 80

_OCR_PROMPT = (
    "Convert this PDF page region to markdown. "
    "Preserve all text, tables (as markdown tables), equations (as LaTeX inside $$), "
    "and figure captions. Return ONLY the markdown, no commentary."
)
_FIG_PROMPT = (
    "Describe this figure from a scientific paper in 2-3 sentences: "
    "what it shows, axes/legend labels if present, and the key visual finding."
)


# ── Step 0: per-page layout analysis ────────────────────────────────────── #

def analyse_page(page: fitz.Page) -> dict:
    """
    Analyse a single PDF page and return a layout descriptor.

    Returned dict (serialisable to JSON):
      page_num       int   1-based
      size_pts       {w, h}
      layout         "1-col" | "2-col"
      n_text_blocks  int
      content_bbox   {x0, y0, x1, y1}  — text bounding box (pts)
      margins        {L, R, T, B}      — whitespace (pts)
      figures        [{x0,y0,x1,y1,w,h}]  significant image regions (pts)
      mid_x          float             — column split x coordinate (pts)
    """
    pw, ph = page.rect.width, page.rect.height
    mid_x  = pw / 2.0

    blocks      = page.get_text("blocks")
    text_blocks = [b for b in blocks if b[6] == 0]

    if text_blocks:
        cx0 = min(b[0] for b in text_blocks)
        cy0 = min(b[1] for b in text_blocks)
        cx1 = max(b[2] for b in text_blocks)
        cy1 = max(b[3] for b in text_blocks)
    else:
        cx0, cy0, cx1, cy1 = 0.0, 0.0, pw, ph

    # Figure regions: only significant embedded images
    figures = []
    for img in page.get_images(full=True):
        try:
            bbox = page.get_image_bbox(img)
            if bbox.width >= _MIN_FIG_W and bbox.height >= _MIN_FIG_H:
                figures.append({
                    "x0": round(bbox.x0), "y0": round(bbox.y0),
                    "x1": round(bbox.x1), "y1": round(bbox.y1),
                    "w":  round(bbox.width), "h": round(bbox.height),
                })
        except Exception:
            pass

    # Two-column detection
    left_b  = [b for b in text_blocks if b[2] < mid_x + 20]
    right_b = [b for b in text_blocks if b[0] > mid_x - 20]
    layout  = "2-col" if (len(left_b) >= 2 and len(right_b) >= 2) else "1-col"

    return {
        "size_pts":      {"w": round(pw), "h": round(ph)},
        "layout":        layout,
        "n_text_blocks": len(text_blocks),
        "content_bbox":  {"x0": round(cx0), "y0": round(cy0),
                          "x1": round(cx1), "y1": round(cy1)},
        "margins":       {"L": round(cx0), "R": round(pw - cx1),
                          "T": round(cy0), "B": round(ph - cy1)},
        "figures":       figures,
        "mid_x":         round(mid_x, 1),
    }


def analyse_pdf(pdf_path: str, max_pages: int = 20) -> list[dict]:
    """
    Run layout analysis on all pages and return a list of page descriptors.
    Each dict has page_num (1-based) prepended.
    """
    doc     = fitz.open(pdf_path)
    n_pages = min(len(doc), max_pages)
    meta    = []
    for i in range(n_pages):
        page = doc[i]
        entry = analyse_page(page)
        entry["page_num"] = i + 1
        meta.append(entry)
    doc.close()
    return meta


# ── Step 1: clip descriptors from layout ────────────────────────────────── #

def _get_clips(layout: dict) -> list[dict]:
    """
    Produce ordered clip descriptors from a page layout dict.

    Each clip: {type, rect (fitz.Rect), col, dpi}
    Order: figure clips first, then left-col / right-col (or full) text.
    """
    pw    = layout["size_pts"]["w"]
    ph    = layout["size_pts"]["h"]
    cb    = layout["content_bbox"]
    mid_x = layout["mid_x"]
    clips = []

    # Figure clips
    for fig in layout["figures"]:
        clips.append({
            "type": "figure",
            "rect": fitz.Rect(fig["x0"], fig["y0"], fig["x1"], fig["y1"]),
            "col":  "figure",
            "dpi":  _FIG_DPI,
        })

    # Text clips
    cr = fitz.Rect(cb["x0"], cb["y0"], cb["x1"], cb["y1"])
    if layout["layout"] == "2-col":
        clips.append({"type": "text", "rect": fitz.Rect(cr.x0, cr.y0, mid_x, cr.y1),
                      "col": "left",  "dpi": _OCR_DPI})
        clips.append({"type": "text", "rect": fitz.Rect(mid_x, cr.y0, cr.x1, cr.y1),
                      "col": "right", "dpi": _OCR_DPI})
    else:
        clips.append({"type": "text", "rect": cr, "col": "full", "dpi": _OCR_DPI})

    return clips


# ── Step 2: rendering + VLM ─────────────────────────────────────────────── #

def _clip_to_b64(page: fitz.Page, rect: fitz.Rect, dpi: int) -> str:
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, clip=rect, colorspace=fitz.csRGB)
    buf = io.BytesIO(pix.tobytes("jpeg", jpg_quality=85))
    return base64.b64encode(buf.getvalue()).decode()


def _call_vlm(b64: str, prompt: str, timeout: int = 90) -> str:
    import requests
    payload = {
        "model":   VLM_MODEL,
        "messages": [{"role": "user", "content": prompt, "images": [b64]}],
        "stream":  False,
        "options": {"num_ctx": 8192, "temperature": 0},
    }
    r = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=timeout)
    r.raise_for_status()
    text = r.json()["message"]["content"].strip()
    text = re.sub(r"^```(?:markdown)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text.strip())
    return text


# ── Step 3: markdown parser ──────────────────────────────────────────────── #

def _parse_markdown(md: str, page_num: int, col: str) -> list[dict]:
    """Parse olmOCR-2 markdown into typed element dicts with col metadata."""
    elements: list[dict] = []
    lines = md.splitlines()
    i = 0

    while i < len(lines):
        line = lines[i]

        # Table
        if re.match(r"^\s*\|", line) or line.strip().lower().startswith("<table"):
            table_lines = []
            while i < len(lines) and (
                re.match(r"^\s*\|", lines[i])
                or lines[i].strip().lower().startswith("<")
                or (table_lines and lines[i].strip() == "")
            ):
                table_lines.append(lines[i]); i += 1
            caption = ""
            if i < len(lines) and re.match(r"(?i)^(table|tab\.?)\s*\d*", lines[i].strip()):
                caption = lines[i].strip(); i += 1
            label_m = re.search(r"\\label\{([^}]+)\}", caption)
            elements.append({
                "type": "table", "page": page_num, "col": col,
                "source": "text", "caption": caption,
                "content": "\n".join(table_lines).strip(),
                "label":   label_m.group(1) if label_m else None,
            })
            continue

        # Equation
        if line.strip().startswith("$$") or line.strip().startswith("\\["):
            closing = "$$" if line.strip().startswith("$$") else "\\]"
            eq_lines = [line]; i += 1
            while i < len(lines) and closing not in lines[i]:
                eq_lines.append(lines[i]); i += 1
            if i < len(lines):
                eq_lines.append(lines[i]); i += 1
            latex   = "\n".join(eq_lines).strip()
            label_m = re.search(r"\\label\{([^}]+)\}", latex)
            elements.append({
                "type": "equation", "page": page_num, "col": col,
                "source": "text", "content": latex, "desc": "",
                "label": label_m.group(1) if label_m else None,
            })
            continue

        # Figure caption
        if re.match(r"(?i)^\s*(figure|fig\.?)\s*\d*[\.:]\s*", line):
            label_m = re.search(r"\\label\{([^}]+)\}", line)
            elements.append({
                "type": "figure", "page": page_num, "col": col,
                "source": "text", "caption": line.strip(), "desc": "",
                "label": label_m.group(1) if label_m else None,
            })
            i += 1; continue

        if not line.strip():
            i += 1; continue

        # Text / heading
        text_lines = []
        while i < len(lines) and lines[i].strip() and not (
            re.match(r"^\s*\|", lines[i])
            or lines[i].strip().startswith("$$")
            or lines[i].strip().startswith("\\[")
            or re.match(r"(?i)^\s*(figure|fig\.?)\s*\d*[\.:]\s*", lines[i])
        ):
            text_lines.append(lines[i]); i += 1
        content = "\n".join(text_lines).strip()
        if content:
            elements.append({
                "type": "text", "page": page_num, "col": col,
                "source": "text", "content": content,
            })

    return elements


# ── Main extraction ─────────────────────────────────────────────────────── #

def extract_pdf(
    pdf_path: str,
    out_dir: str,
    doc_id: str,
    bibtex_key: str,
    max_pages: int = 20,
    dpi: int = _OCR_DPI,
    force: bool = False,
) -> dict[str, Any]:
    """
    Full extraction pipeline for one PDF.

    Outputs (all in out_dir/):
      {doc_id}_layout.json  — per-page layout metadata (persisted for all stages)
      {doc_id}.json         — element registry with col + source metadata
      {doc_id}.txt          — flat text with rich [PAGE n][2COL-LEFT][TEXT] tags

    Returns the registry entry dict.
    """
    os.makedirs(out_dir, exist_ok=True)
    layout_out = osp.join(out_dir, f"{doc_id}_layout.json")
    json_out   = osp.join(out_dir, f"{doc_id}.json")
    text_out   = osp.join(out_dir, f"{doc_id}.txt")

    if not force and osp.exists(json_out):
        with open(json_out) as f:
            return json.load(f)

    # ── Phase A: layout analysis (no VLM needed) ────────────────────────
    print(f"[vlm_extractor] {doc_id}: analysing layout ...")
    page_layouts = analyse_pdf(pdf_path, max_pages)

    # Save layout metadata immediately — downstream can use it even if VLM fails
    with open(layout_out, "w") as f:
        json.dump(page_layouts, f, indent=2)

    n_pages = len(page_layouts)
    n_figs  = sum(len(pl["figures"]) for pl in page_layouts)
    n_2col  = sum(1 for pl in page_layouts if pl["layout"] == "2-col")
    print(f"  {n_pages} pages: {n_2col} two-column, {n_figs} figures detected")
    print(f"  Layout saved → {layout_out}")

    # ── Phase B: VLM extraction per clip ────────────────────────────────
    doc          = fitz.open(pdf_path)
    all_elements: list[dict] = []

    # Folder for cropped figure images — used by downstream VLM / figure RAG
    fig_dir = osp.join(out_dir, f"{doc_id}_figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Per-doc processing log
    doc_log: dict = {
        "doc_id": doc_id, "bibtex_key": bibtex_key,
        "stages": {
            "layout_analysis": {"status": "done", "n_pages": n_pages,
                                 "n_two_col": n_2col, "n_figures": n_figs},
            "vlm_extraction":  {"status": "pending"},
            "langextract":     {"status": "pending"},
            "hf_embedding":    {"status": "pending"},
        }
    }
    _save_doc_log(out_dir, doc_id, doc_log)

    print(f"[vlm_extractor] {doc_id}: extracting with {VLM_MODEL} @ {dpi}DPI ...")

    for pl in page_layouts:
        page_num = pl["page_num"]
        page     = doc[page_num - 1]
        clips    = _get_clips(pl)
        pg_elements: list[dict] = []
        clip_log: list[str]     = []

        for ci, clip in enumerate(clips):
            ctype = clip["type"]
            rect  = clip["rect"]
            col   = clip["col"]
            cdpi  = clip.get("dpi", dpi)

            if ctype == "figure":
                # Save cropped figure image for downstream use
                fig_path = osp.join(fig_dir, f"page{page_num:02d}_fig{ci}.jpg")
                try:
                    mat  = fitz.Matrix(cdpi / 72, cdpi / 72)
                    pix  = page.get_pixmap(matrix=mat, clip=rect, colorspace=fitz.csRGB)
                    pix.save(fig_path)
                except Exception:
                    fig_path = None

                try:
                    b64  = _clip_to_b64(page, rect, dpi=cdpi)
                    desc = _call_vlm(b64, _FIG_PROMPT, timeout=60)
                    pg_elements.append({
                        "type": "figure", "page": page_num,
                        "col": "figure", "source": "figure",
                        "caption": "", "desc": desc, "label": None,
                        "clip_idx": ci, "fig_path": fig_path,
                    })
                    clip_log.append(f"fig✓")
                except Exception as exc:
                    # Even if VLM fails, keep the saved image path
                    if fig_path and osp.exists(fig_path):
                        pg_elements.append({
                            "type": "figure", "page": page_num,
                            "col": "figure", "source": "figure",
                            "caption": "", "desc": "", "label": None,
                            "clip_idx": ci, "fig_path": fig_path,
                        })
                    clip_log.append(f"fig✗({str(exc)[:30]})")

            else:  # text clip
                try:
                    b64      = _clip_to_b64(page, rect, dpi=cdpi)
                    md       = _call_vlm(b64, _OCR_PROMPT, timeout=90)
                    elements = _parse_markdown(md, page_num, col)
                    for el in elements:
                        el["clip_idx"] = ci
                    pg_elements.extend(elements)
                    clip_log.append(f"{col}({len(elements)})")
                except Exception as exc:
                    clip_log.append(f"{col}✗({str(exc)[:30]})")
                    # Fallback: raw PyMuPDF text for this region
                    raw = page.get_textbox(rect).strip()
                    if raw:
                        pg_elements.append({
                            "type": "text", "page": page_num,
                            "col": col, "source": "fallback",
                            "content": raw, "clip_idx": ci,
                        })

        all_elements.extend(pg_elements)
        layout_tag = pl["layout"]
        print(f"  p{page_num:02d} [{layout_tag}] "
              f"clips=[{' '.join(clip_log)}]  {len(pg_elements)}el")

    doc.close()

    # ── Phase C: write flat text with rich tags ──────────────────────────
    lines = [f"DOCUMENT: {doc_id}  BIBTEX: {bibtex_key}", ""]
    for el in all_elements:
        el_type = el.get("type", "text")
        pg      = el.get("page", "?")
        col     = el.get("col",  "full")
        source  = el.get("source", "text")

        # Rich tag: [PAGE n][LAYOUT-COL][TYPE]
        layout_tag = {
            "left":   "2COL-LEFT",
            "right":  "2COL-RIGHT",
            "full":   "1COL",
            "figure": "FIGURE-REGION",
        }.get(col, col.upper())
        prefix = f"[PAGE {pg}][{layout_tag}][{el_type.upper()}]"

        if el_type == "figure":
            if el.get("caption"):
                lines.append(f"{prefix} Caption: {el['caption']}")
            if el.get("desc"):
                lines.append(f"{prefix} Visual:  {el['desc']}")
            if el.get("label"):
                lines.append(f"{prefix} Label:   {el['label']}")
        elif el_type == "table":
            lines.append(f"{prefix} Caption: {el.get('caption','')}")
            lines.append(f"{prefix} Data:    {el.get('content','')[:500]}")
            if el.get("label"):
                lines.append(f"{prefix} Label:   {el['label']}")
        elif el_type == "equation":
            lines.append(f"{prefix} LaTeX:   {el.get('content','')}")
            if el.get("label"):
                lines.append(f"{prefix} Label:   {el['label']}")
        else:
            content = el.get("content", "")
            if source == "fallback":
                lines.append(f"{prefix}[FALLBACK] {content}")
            else:
                lines.append(f"{prefix} {content}")
        lines.append("")

    text_content = "\n".join(lines)
    with open(text_out, "w", encoding="utf-8") as f:
        f.write(text_content)

    # ── Phase D: save registry + update doc log ─────────────────────────
    ls = {
        "n_pages":     n_pages,
        "n_two_col":   n_2col,
        "n_figures":   n_figs,
        "n_elements":  len(all_elements),
        "n_text_el":   sum(1 for e in all_elements if e.get("source") == "text"),
        "n_figure_el": sum(1 for e in all_elements if e.get("source") == "figure"),
        "n_fallback":  sum(1 for e in all_elements if e.get("source") == "fallback"),
    }
    registry_entry = {
        "doc_id":         doc_id,
        "bibtex_key":     bibtex_key,
        "pdf_path":       pdf_path,
        "layout_file":    layout_out,
        "figures_dir":    fig_dir,
        "elements":       all_elements,
        "text_file":      text_out,
        "layout_summary": ls,
    }
    with open(json_out, "w") as f:
        json.dump(registry_entry, f, indent=2)

    # Mark VLM stage done in per-doc log
    doc_log["stages"]["vlm_extraction"] = {
        "status": "done", **ls,
        "model": VLM_MODEL,
    }
    _save_doc_log(out_dir, doc_id, doc_log)

    print(f"[vlm_extractor] done → {ls['n_elements']} elements  "
          f"({ls['n_text_el']} text, {ls['n_figure_el']} figure, "
          f"{ls['n_fallback']} fallback)")
    return registry_entry


def extract_pdf_batch(
    pdf_meta: list[dict],
    out_dir: str,
    max_pages: int = 20,
) -> list[dict]:
    results = []
    for meta in pdf_meta:
        entry = extract_pdf(
            pdf_path   = meta["pdf_path"],
            out_dir    = out_dir,
            doc_id     = meta["doc_id"],
            bibtex_key = meta["bibtex_key"],
            max_pages  = max_pages,
        )
        results.append(entry)
    return results
