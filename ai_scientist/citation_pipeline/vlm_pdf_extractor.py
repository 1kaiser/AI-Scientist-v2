"""
Stage 1: PDF → structured text via qwen2.5vl:7b.

For each PDF page the VLM produces a JSON record describing:
  - main text content
  - figures (caption + visual description)
  - tables  (caption + cell data)
  - equations (LaTeX transcription + description)

The merged output is a single text file per PDF, structured so that
LangExtract can parse it with full visual–textual context.

Output schema per page element:
  {
    "type":    "text" | "figure" | "table" | "equation",
    "page":    int,
    "content": str,        # main text or equation LaTeX
    "caption": str | null, # figure/table caption
    "desc":    str | null, # VLM visual description
    "label":   str | null  # LaTeX \label if detectable
  }
"""

import base64
import io
import json
import os
import os.path as osp
import re
from typing import Any

import fitz  # pymupdf


OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
VLM_MODEL   = os.environ.get("VLM_PDF_MODEL", "qwen2.5vl:7b")

_PAGE_PROMPT = """\
You are a scientific document parser. Analyse this PDF page image and extract ALL content.

Return a JSON array. Each element must be one of:

{"type":"text",     "page":<n>, "content":"<paragraph text>"}
{"type":"figure",   "page":<n>, "caption":"<caption>", "desc":"<visual description of what the figure shows>", "label":"<\\\\label{} if visible or null>"}
{"type":"table",    "page":<n>, "caption":"<caption>", "content":"<rows as CSV>", "label":"<\\\\label{} if visible or null>"}
{"type":"equation", "page":<n>, "content":"<LaTeX expression>", "desc":"<what it represents>", "label":"<\\\\label{} if visible or null>"}

Rules:
- Preserve ALL text verbatim. Do not summarise.
- One JSON element per logical block (paragraph, figure, table, equation).
- If the page has no structured elements, return a single text element with the full page text.
- Return ONLY the JSON array, no other text.
"""


def _page_to_base64(page: fitz.Page, dpi: int = 150) -> str:
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    buf = io.BytesIO(pix.tobytes("jpeg", jpg_quality=85))
    return base64.b64encode(buf.getvalue()).decode()


def _call_vlm(image_b64: str, page_num: int) -> list[dict]:
    """Call qwen2.5vl:7b for one page; returns list of element dicts."""
    import requests

    payload = {
        "model": VLM_MODEL,
        "messages": [{
            "role": "user",
            "content": _PAGE_PROMPT.replace("<n>", str(page_num)),
            "images": [image_b64],
        }],
        "stream": False,
        "options": {"num_ctx": 4096, "temperature": 0.1},
    }
    r = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=120)
    r.raise_for_status()
    raw = r.json()["message"]["content"].strip()

    # Strip markdown fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        elements = json.loads(raw)
        if not isinstance(elements, list):
            elements = [elements]
    except json.JSONDecodeError:
        # Fallback: treat entire response as plain text
        elements = [{"type": "text", "page": page_num, "content": raw}]

    # Ensure page number is set
    for el in elements:
        el.setdefault("page", page_num)
    return elements


def extract_pdf(
    pdf_path: str,
    out_dir: str,
    doc_id: str,
    bibtex_key: str,
    max_pages: int = 20,
    dpi: int = 150,
    force: bool = False,
) -> dict[str, Any]:
    """
    Extract structured content from a PDF using the VLM.

    Args:
        pdf_path:    Path to the PDF file.
        out_dir:     Directory to write extraction JSON and text files.
        doc_id:      Short identifier used in [[doc_id]] tags (e.g. "ref_03").
        bibtex_key:  BibTeX cite key (e.g. "dong2024deeplearning").
        max_pages:   Maximum pages to process (default 20).
        dpi:         Render resolution (higher = more detail, slower).
        force:       Re-extract even if output exists.

    Returns:
        Registry entry dict:
        {
          "doc_id": str, "bibtex_key": str, "pdf_path": str,
          "elements": [...], "text_file": str
        }
    """
    os.makedirs(out_dir, exist_ok=True)
    json_out  = osp.join(out_dir, f"{doc_id}.json")
    text_out  = osp.join(out_dir, f"{doc_id}.txt")

    if not force and osp.exists(json_out):
        with open(json_out) as f:
            return json.load(f)

    doc = fitz.open(pdf_path)
    n_pages = min(len(doc), max_pages)
    all_elements: list[dict] = []

    print(f"[vlm_extractor] {doc_id}: extracting {n_pages}/{len(doc)} pages ...")
    for page_num in range(n_pages):
        page = doc[page_num]
        img_b64 = _page_to_base64(page, dpi=dpi)
        try:
            elements = _call_vlm(img_b64, page_num + 1)
            all_elements.extend(elements)
            print(f"  page {page_num+1}/{n_pages}: {len(elements)} elements")
        except Exception as exc:
            print(f"  page {page_num+1} FAILED: {exc}")
            # Fallback: extract raw text via pymupdf
            text = page.get_text("text")
            all_elements.append({"type": "text", "page": page_num + 1, "content": text})

    doc.close()

    # Build flat text for LangExtract (visual context preserved)
    lines = [f"DOCUMENT: {doc_id}  BIBTEX: {bibtex_key}", ""]
    for el in all_elements:
        el_type = el.get("type", "text")
        page    = el.get("page", "?")
        prefix  = f"[PAGE {page}][{el_type.upper()}]"

        if el_type == "figure":
            lines.append(f"{prefix} Caption: {el.get('caption','')}")
            lines.append(f"{prefix} Visual:  {el.get('desc','')}")
            if el.get("label"):
                lines.append(f"{prefix} Label:   {el['label']}")
        elif el_type == "table":
            lines.append(f"{prefix} Caption: {el.get('caption','')}")
            lines.append(f"{prefix} Data:    {el.get('content','')}")
            if el.get("label"):
                lines.append(f"{prefix} Label:   {el['label']}")
        elif el_type == "equation":
            lines.append(f"{prefix} LaTeX:   {el.get('content','')}")
            lines.append(f"{prefix} Meaning: {el.get('desc','')}")
            if el.get("label"):
                lines.append(f"{prefix} Label:   {el['label']}")
        else:
            lines.append(f"{prefix} {el.get('content','')}")
        lines.append("")

    text_content = "\n".join(lines)
    with open(text_out, "w", encoding="utf-8") as f:
        f.write(text_content)

    registry_entry = {
        "doc_id":     doc_id,
        "bibtex_key": bibtex_key,
        "pdf_path":   pdf_path,
        "elements":   all_elements,
        "text_file":  text_out,
    }
    with open(json_out, "w") as f:
        json.dump(registry_entry, f, indent=2)

    print(f"[vlm_extractor] {doc_id}: done → {text_out}")
    return registry_entry


def extract_pdf_batch(
    pdf_meta: list[dict],
    out_dir: str,
    max_pages: int = 20,
) -> list[dict]:
    """
    Extract multiple PDFs. pdf_meta is a list of:
      {"pdf_path": str, "doc_id": str, "bibtex_key": str}

    Returns list of registry entries.
    """
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
