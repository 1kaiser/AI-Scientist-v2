"""
Trajectory memory for Stage A composition (from research_engine_v17 pattern).

Logs per-section failures so compose_section() can learn to avoid repeating
the same mistakes across retries AND across experiment runs.

Storage: {base_folder}/trajectories/{section_name}_failures.jsonl
Each line: {"timestamp", "reason", "draft_excerpt", "attempt"}
"""

import json
import os
import os.path as osp
from datetime import datetime


def _traj_dir(base_folder: str) -> str:
    return osp.join(base_folder, "trajectories")


def _failures_path(base_folder: str, section_name: str) -> str:
    safe = section_name.lower().replace(" ", "_").replace("/", "_").replace("&", "and")
    return osp.join(_traj_dir(base_folder), f"{safe}_failures.jsonl")


def load_failures(base_folder: str, section_name: str, n: int = 3) -> str:
    """
    Return a formatted "AVOID THESE MISTAKES" prefix for section_name.
    Returns empty string if no failures logged yet.
    """
    path = _failures_path(base_folder, section_name)
    if not osp.exists(path):
        return ""

    lines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    lines.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if not lines:
        return ""

    recent = lines[-n:]
    parts = ["AVOID THESE MISTAKES (from previous attempts at this section):"]
    for i, entry in enumerate(recent, 1):
        reason = entry.get("reason", "unknown issue")
        excerpt = entry.get("draft_excerpt", "")[:200]
        parts.append(f"  {i}. {reason}")
        if excerpt:
            parts.append(f"     Bad draft excerpt: \"{excerpt}...\"")
    parts.append("")
    return "\n".join(parts)


def save_failure(
    base_folder: str,
    section_name: str,
    reason: str,
    draft_excerpt: str = "",
    attempt: int = 0,
) -> None:
    """Log a section composition failure for future reference."""
    os.makedirs(_traj_dir(base_folder), exist_ok=True)
    path = _failures_path(base_folder, section_name)
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "section": section_name,
        "reason": reason,
        "draft_excerpt": draft_excerpt[:300],
        "attempt": attempt,
    }
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def load_all_section_stats(base_folder: str) -> dict:
    """Return failure counts per section for diagnostics."""
    traj_dir = _traj_dir(base_folder)
    if not osp.isdir(traj_dir):
        return {}
    stats = {}
    for fname in os.listdir(traj_dir):
        if fname.endswith("_failures.jsonl"):
            section = fname.replace("_failures.jsonl", "").replace("_", " ")
            with open(osp.join(traj_dir, fname)) as f:
                count = sum(1 for line in f if line.strip())
            stats[section] = count
    return stats
