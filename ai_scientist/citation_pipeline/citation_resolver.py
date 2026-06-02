"""
Stage 3: Resolve [[tags]] → proper LaTeX citations and cross-references.

Tag formats supported:
  Citations (multi-doc, author-blind):
    [[ref_03]]                → \cite{dong2024}
    [[ref_03, ref_07]]        → \cite{smith2023, dong2024}   (sorted: year asc → alpha)
    [[ref_03, ref_07, ref_12]] same-year gets letter suffix  (2023a, 2023b)

  Cross-references (figures, tables, equations, sections, supplementary):
    [[fig:label]]   → Figure~\ref{fig:label}
    [[tab:label]]   → Table~\ref{tab:label}
    [[eq:label]]    → \eqref{eq:label}
    [[sec:label]]   → Section~\ref{sec:label}
    [[app:label]]   → Appendix~\ref{app:label}
    [[sfig:label]]  → Supplementary Figure~\ref{sfig:label}
    [[stab:label]]  → Supplementary Table~\ref{stab:label}
    [[seq:label]]   → Supplementary \eqref{seq:label}

Sorting rules (ICLR 2025 / iclr2025.bst):
  Primary:   year ascending
  Secondary: first_author surname alphabetical
  Tertiary:  bibtex_key (tiebreaker)
  Same author+year: append letter suffix a, b, c, ...
"""

import json
import os
import os.path as osp
import re
from typing import Optional


# ── Cross-reference prefix map ─────────────────────────────────────────── #
_XREF_PREFIXES = {
    "fig":  ("Figure",               r"Figure~\ref{{{label}}}"),
    "tab":  ("Table",                r"Table~\ref{{{label}}}"),
    "eq":   ("Equation",             r"\eqref{{{label}}}"),
    "sec":  ("Section",              r"Section~\ref{{{label}}}"),
    "app":  ("Appendix",             r"Appendix~\ref{{{label}}}"),
    "alg":  ("Algorithm",            r"Algorithm~\ref{{{label}}}"),
    "sfig": ("Supplementary Figure", r"Supplementary Figure~\ref{{{label}}}"),
    "stab": ("Supplementary Table",  r"Supplementary Table~\ref{{{label}}}"),
    "seq":  ("Supplementary",        r"Supplementary~\eqref{{{label}}}"),
    "sch":  ("Supplementary Chapter",r"Supplementary Chapter~\ref{{{label}}}"),
}


# ── Citation sorting ───────────────────────────────────────────────────── #

def _sort_key(doc_id: str, registry: dict) -> tuple:
    """Returns (year, first_author, bibtex_key) sort tuple for a doc_id."""
    entry = registry.get(doc_id, {})
    year        = entry.get("year") or 9999
    first_author = entry.get("first_author") or "zzz"
    bibtex_key  = entry.get("bibtex_key") or doc_id
    return (year, first_author, bibtex_key)


def _assign_year_suffixes(sorted_entries: list[dict]) -> dict[str, str]:
    """
    Assign letter suffixes (a, b, c) to bibtex_keys that share same
    first_author AND year — matching iclr2025.bst behaviour.

    Returns: {bibtex_key: final_cite_key}
    """
    from collections import defaultdict
    groups: dict[tuple, list[str]] = defaultdict(list)
    for entry in sorted_entries:
        key_group = (entry.get("first_author") or "", entry.get("year") or 0)
        groups[key_group].append(entry.get("bibtex_key") or "")

    cite_map: dict[str, str] = {}
    for group_key, keys in groups.items():
        if len(keys) == 1:
            cite_map[keys[0]] = keys[0]
        else:
            for i, k in enumerate(keys):
                suffix = chr(ord("a") + i)
                cite_map[k] = k + suffix
    return cite_map


def _resolve_citation_tag(doc_ids_str: str, registry: dict) -> str:
    """
    Convert "ref_03, ref_07, ref_12" → \\cite{key2020, key2023a, key2023b}
    sorted by (year asc, first_author alpha, bibtex_key).
    """
    raw_ids = [d.strip() for d in doc_ids_str.split(",") if d.strip()]

    # Build sorted list of registry entries for these doc_ids
    entries = []
    unknown = []
    for doc_id in raw_ids:
        if doc_id in registry:
            entries.append(registry[doc_id])
        else:
            unknown.append(doc_id)

    entries.sort(key=lambda e: (
        e.get("year") or 9999,
        e.get("first_author") or "zzz",
        e.get("bibtex_key") or "",
    ))

    cite_map = _assign_year_suffixes(entries)

    cite_keys = [cite_map.get(e["bibtex_key"], e["bibtex_key"]) for e in entries]
    cite_keys += unknown  # keep unresolved ids as-is

    if not cite_keys:
        return f"[UNRESOLVED: {doc_ids_str}]"
    return r"\cite{" + ", ".join(cite_keys) + "}"


def _resolve_xref_tag(prefix: str, label: str) -> str:
    """Convert prefix:label → proper LaTeX cross-reference string."""
    if prefix in _XREF_PREFIXES:
        _, template = _XREF_PREFIXES[prefix]
        return template.format(label=f"{prefix}:{label}")
    # Unknown prefix — generic \ref
    return rf"\ref{{{prefix}:{label}}}"


# ── Main resolver ──────────────────────────────────────────────────────── #

_TAG_RE = re.compile(r"\[\[([^\]]+)\]\]")


def resolve_tags(
    text: str,
    doc_registry: dict,
    warn_unresolved: bool = True,
) -> tuple[str, list[str]]:
    """
    Replace all [[tags]] in text with proper LaTeX.

    Args:
        text:            Input text containing [[tags]].
        doc_registry:    {doc_id → entry} from build_citation_index().
        warn_unresolved: Print warnings for unknown doc_ids.

    Returns:
        (resolved_text, list_of_warnings)
    """
    warnings: list[str] = []

    def _replace(m: re.Match) -> str:
        inner = m.group(1).strip()

        # ── Cross-reference tag: prefix:label ──────────────────────────
        xref_m = re.match(r"^([a-z]+):(.+)$", inner)
        if xref_m:
            prefix = xref_m.group(1)
            label  = xref_m.group(2).strip()
            return _resolve_xref_tag(prefix, label)

        # ── Citation tag: ref_id, ref_id, ... ──────────────────────────
        doc_ids = [d.strip() for d in inner.split(",")]
        unresolved = [d for d in doc_ids if d not in doc_registry]
        if unresolved and warn_unresolved:
            w = f"Unresolved citation doc_ids: {unresolved}"
            warnings.append(w)
        return _resolve_citation_tag(inner, doc_registry)

    resolved = _TAG_RE.sub(_replace, text)
    return resolved, warnings


def resolve_latex_file(
    tex_path: str,
    doc_registry: dict,
    out_path: Optional[str] = None,
    warn_unresolved: bool = True,
) -> list[str]:
    """
    Resolve all [[tags]] in a LaTeX file in-place (or to out_path).

    Returns list of warning strings.
    """
    with open(tex_path, encoding="utf-8") as f:
        content = f.read()

    resolved, warnings = resolve_tags(content, doc_registry, warn_unresolved)

    dest = out_path or tex_path
    with open(dest, "w", encoding="utf-8") as f:
        f.write(resolved)

    if warnings:
        print(f"[citation_resolver] {len(warnings)} warnings:")
        for w in warnings:
            print(f"  ⚠ {w}")
    print(f"[citation_resolver] Resolved → {dest}")
    return warnings


# ── Registry I/O helpers ───────────────────────────────────────────────── #

def load_registry(index_dir: str) -> dict:
    """Load doc_registry.json produced by build_citation_index()."""
    path = osp.join(index_dir, "doc_registry.json")
    if not osp.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def save_registry(registry: dict, index_dir: str) -> None:
    os.makedirs(index_dir, exist_ok=True)
    path = osp.join(index_dir, "doc_registry.json")
    with open(path, "w") as f:
        json.dump(registry, f, indent=2)


def register_doc(
    registry: dict,
    doc_id: str,
    bibtex_key: str,
    year: Optional[int] = None,
    first_author: Optional[str] = None,
) -> dict:
    """Add or update a single entry in the registry."""
    registry[doc_id] = {
        "doc_id":       doc_id,
        "bibtex_key":   bibtex_key,
        "year":         year,
        "first_author": first_author.lower() if first_author else None,
    }
    return registry


# ── CLI entry point ────────────────────────────────────────────────────── #

if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(
        description="Resolve [[citation/xref tags]] in a LaTeX file."
    )
    parser.add_argument("tex_file",     help="Input .tex file with [[tags]]")
    parser.add_argument("index_dir",    help="Directory containing doc_registry.json")
    parser.add_argument("--out",        help="Output .tex (default: overwrite input)")
    parser.add_argument("--no-warn",    action="store_true")
    args = parser.parse_args()

    registry = load_registry(args.index_dir)
    if not registry:
        print(f"ERROR: No doc_registry.json in {args.index_dir}", file=sys.stderr)
        sys.exit(1)

    warnings = resolve_latex_file(
        args.tex_file, registry, args.out, not args.no_warn
    )
    sys.exit(0 if not warnings else 1)
