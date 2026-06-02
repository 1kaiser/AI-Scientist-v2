"""
Vocabulary profile extractor (from research_engine_v17 pattern).

Reads VLM-extracted .txt files from the citation pipeline and builds a
style profile that Stage A composition can use to match the writing style
of published papers in the same domain.

Output: {citation_index_dir}/vocabulary_profile.json
"""

import json
import os
import os.path as osp
import re
from typing import Optional


VOCAB_PROMPT = """Analyse the following excerpts from published scientific papers and extract a writing style profile.

Return a JSON object with exactly these keys:
{
  "dominant_tense": "past" | "present" | "mixed",
  "avg_sentence_length": <integer, estimated words per sentence>,
  "avg_paragraph_words": <integer, estimated words per paragraph>,
  "subsection_pattern": "<description of how sections are typically structured>",
  "key_terms": [<list of up to 20 domain-specific technical terms>],
  "style_notes": "<2-3 sentences about tone, citation density, quantitative precision>"
}

PAPER EXCERPTS:
{excerpts}

Return ONLY the JSON object, no other text."""


def _extract_text_samples(txt_file: str, max_chars: int = 3000,
                          layout_file: str | None = None) -> str:
    """
    Pull a representative style sample from a VLM-extracted text file.

    Uses layout metadata (if available) to prefer text-dense pages:
    - Skips pages where layout has figures dominating (n_text_blocks < 5)
    - Always uses [TEXT] tagged lines; skips [FIGURE-REGION] and [FALLBACK]
    """
    # Build set of preferred page numbers from layout metadata
    good_pages: set[int] = set()
    if layout_file and osp.exists(layout_file):
        try:
            with open(layout_file) as f:
                page_layouts = json.load(f)
            for pl in page_layouts:
                # Prefer pages with substantial text and few/no figures
                if pl.get("n_text_blocks", 0) >= 5 and len(pl.get("figures", [])) == 0:
                    good_pages.add(pl["page_num"])
        except Exception:
            pass

    with open(txt_file, encoding="utf-8", errors="ignore") as f:
        content = f.read()

    text_lines = []
    for line in content.splitlines():
        # Only keep rich-tagged text lines from text columns
        if "[TEXT]" not in line:
            continue
        if "[FIGURE-REGION]" in line or "[FALLBACK]" in line:
            continue
        # If we have page preferences, filter by page
        if good_pages:
            pg_match = re.search(r"\[PAGE (\d+)\]", line)
            if pg_match and int(pg_match.group(1)) not in good_pages:
                continue
        text_lines.append(re.sub(r"\[PAGE \d+\]\[[^\]]+\]\[TEXT\]\s*", "", line).strip())

    sample = " ".join(text_lines)[:max_chars]
    # Fallback: use all text lines if filtered result is too short
    if len(sample) < 500:
        all_text = [re.sub(r"\[PAGE \d+\]\[[^\]]+\]\[TEXT\]\s*", "", l).strip()
                    for l in content.splitlines() if "[TEXT]" in l
                    and "[FIGURE-REGION]" not in l and "[FALLBACK]" not in l]
        sample = " ".join(all_text)[:max_chars]
    return sample


def extract_vocabulary_profile(
    citation_index_dir: str,
    max_docs: int = 5,
    force: bool = False,
) -> dict:
    """
    Build a vocabulary/style profile from processed paper .txt files.

    Reads doc_registry.json to find text files, samples text from each,
    calls gemma4:e4b to extract style profile, saves to vocabulary_profile.json.

    Returns the profile dict (or {} if no papers available).
    """
    out_path = osp.join(citation_index_dir, "vocabulary_profile.json")
    if osp.exists(out_path) and not force:
        with open(out_path) as f:
            return json.load(f)

    registry_path = osp.join(citation_index_dir, "doc_registry.json")
    if not osp.exists(registry_path):
        return {}

    with open(registry_path) as f:
        registry = json.load(f)

    # Collect text samples from up to max_docs papers
    samples = []
    for doc_id, entry in list(registry.items())[:max_docs]:
        txt_file    = entry.get("text_file", "")
        layout_file = entry.get("layout_file", "")
        if txt_file and osp.exists(txt_file):
            sample = _extract_text_samples(txt_file, layout_file=layout_file or None)
            if sample:
                samples.append(f"--- Paper: {doc_id} ---\n{sample}")

    if not samples:
        return {}

    combined = "\n\n".join(samples)[:8000]

    # Call LLM for vocabulary analysis
    try:
        import os as _os
        _os.environ.setdefault("OLLAMA_HOST", "http://localhost:11434")
        from ai_scientist.llm import get_response_from_llm, create_client, route_model

        model = route_model("eval judge feedback")
        client, client_model = create_client(model)

        response, _ = get_response_from_llm(
            prompt=VOCAB_PROMPT.format(excerpts=combined),
            client=client,
            model=client_model,
            system_message="You are an expert scientific writing analyst. Return only valid JSON.",
            print_debug=False,
            max_tokens=512,
        )

        # Extract JSON from response
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            profile = json.loads(json_match.group())
        else:
            profile = json.loads(response)

        os.makedirs(citation_index_dir, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(profile, f, indent=2)
        print(f"[vocab] Profile saved → {out_path}")
        return profile

    except Exception as exc:
        print(f"[vocab] Extraction failed: {exc}")
        return {}


def format_vocab_hint(profile: dict) -> str:
    """Format vocabulary profile as a style hint for Stage A composition."""
    if not profile:
        return ""
    parts = ["WRITING STYLE (match published papers in this domain):"]
    if profile.get("dominant_tense"):
        parts.append(f"  - Tense: {profile['dominant_tense']}")
    if profile.get("avg_sentence_length"):
        parts.append(f"  - Sentence length: ~{profile['avg_sentence_length']} words")
    if profile.get("key_terms"):
        terms = ", ".join(profile["key_terms"][:10])
        parts.append(f"  - Key domain terms: {terms}")
    if profile.get("style_notes"):
        parts.append(f"  - Style: {profile['style_notes']}")
    return "\n".join(parts) + "\n"
