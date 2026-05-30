#!/usr/bin/env python3
"""
AI-Scientist-v2 Pipeline Progress Monitor
==========================================
Run in a separate terminal alongside the pipeline to get a live dashboard.

Usage:
    python monitor.py                          # Auto-detects latest experiment
    python monitor.py --experiment <dir_name>  # Monitor a specific experiment
    python monitor.py --interval 10            # Refresh every 10 seconds (default: 5)
"""

import argparse
import glob
import json
import os
import pickle
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# ─── ANSI Colors ──────────────────────────────────────────────────────────────
RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
RED     = "\033[91m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
CYAN    = "\033[96m"
MAGENTA = "\033[95m"
BLUE    = "\033[94m"
WHITE   = "\033[97m"
BG_DARK = "\033[48;5;235m"

# ─── Configuration ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
EXPERIMENTS_DIR = BASE_DIR / "experiments"


def clear_screen():
    os.system("clear" if os.name != "nt" else "cls")


def find_latest_experiment():
    """Find the most recently modified experiment directory."""
    exp_dirs = sorted(EXPERIMENTS_DIR.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for d in exp_dirs:
        if d.is_dir() and not d.name.startswith("."):
            return d
    return None


def get_ollama_status():
    """Check Ollama server health and loaded models."""
    try:
        import requests
        r = requests.get("http://localhost:11434/", timeout=2)
        server_up = r.status_code == 200
    except Exception:
        server_up = False

    loaded_models = []
    vram_used = "N/A"
    if server_up:
        try:
            import requests
            r = requests.get("http://localhost:11434/api/ps", timeout=2)
            if r.status_code == 200:
                data = r.json()
                for m in data.get("models", []):
                    name = m.get("name", "unknown")
                    size_gb = m.get("size_vram", m.get("size", 0)) / 1e9
                    loaded_models.append(f"{name} ({size_gb:.1f}GB)")
                if not loaded_models:
                    loaded_models = ["(none — idle)"]
        except Exception:
            pass

    return server_up, loaded_models


def get_worker_processes():
    """Count active pipeline worker processes."""
    try:
        import psutil
        workers = []
        for proc in psutil.process_iter(["pid", "name", "cmdline", "cpu_percent", "memory_info"]):
            try:
                cmdline = " ".join(proc.info.get("cmdline") or [])
                if "launch_scientist_bfts" in cmdline and "python" in cmdline:
                    mem_mb = (proc.info.get("memory_info") or type("", (), {"rss": 0})()).rss / 1e6
                    workers.append({
                        "pid": proc.info["pid"],
                        "mem_mb": mem_mb,
                        "cpu": proc.info.get("cpu_percent", 0),
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return workers
    except ImportError:
        return []


def scan_stage_logs(log_dir: Path):
    """Scan stage directories for progress data."""
    stages = {}
    if not log_dir.exists():
        return stages

    for stage_dir in sorted(log_dir.glob("stage_*")):
        if not stage_dir.is_dir():
            continue
        stage_name = stage_dir.name

        # Look for journal pickle
        journal_path = stage_dir / "journal.pkl"
        node_count = 0
        buggy_count = 0
        good_count = 0
        best_metric = "—"

        if journal_path.exists():
            try:
                with open(journal_path, "rb") as f:
                    journal = pickle.load(f)
                node_count = len(journal.nodes) if hasattr(journal, "nodes") else 0
                buggy_count = len(journal.buggy_nodes) if hasattr(journal, "buggy_nodes") else 0
                good_count = len(journal.good_nodes) if hasattr(journal, "good_nodes") else 0
                # Try to get best metric
                if hasattr(journal, "get_best_node"):
                    try:
                        best = journal.get_best_node()
                        if best and hasattr(best, "metric"):
                            best_metric = f"{best.metric.value:.4f}" if hasattr(best.metric, "value") else str(best.metric)
                    except Exception:
                        pass
            except Exception:
                pass

        # Look for progress JSON
        progress_path = stage_dir / "notes" / "stage_progress.json"
        progress_data = {}
        if progress_path.exists():
            try:
                with open(progress_path) as f:
                    progress_data = json.load(f)
                    node_count = progress_data.get("total_nodes", node_count)
                    buggy_count = progress_data.get("buggy_nodes", buggy_count)
                    good_count = progress_data.get("good_nodes", good_count)
                    if progress_data.get("best_metric", "None") != "None":
                        best_metric = progress_data["best_metric"]
            except Exception:
                pass

        stages[stage_name] = {
            "nodes": node_count,
            "buggy": buggy_count,
            "good": good_count,
            "best_metric": best_metric,
            "has_journal": journal_path.exists(),
        }

    return stages


def scan_worker_outputs(exp_dir: Path):
    """Find all worker-generated runfiles and their status."""
    outputs = []
    run_dirs = sorted(exp_dir.glob("*-run"))
    for run_dir in run_dirs:
        for proc_dir in sorted(run_dir.glob("process_*")):
            runfile = proc_dir / "runfile.py"
            if runfile.exists():
                size = runfile.stat().st_size
                mtime = datetime.fromtimestamp(runfile.stat().st_mtime)
                age = datetime.now() - mtime

                # Check for execution results
                result_files = list(proc_dir.glob("*.json")) + list(proc_dir.glob("*.log"))
                has_results = len(result_files) > 0

                outputs.append({
                    "path": str(proc_dir.relative_to(exp_dir)),
                    "size_bytes": size,
                    "modified": mtime.strftime("%H:%M:%S"),
                    "age": str(age).split(".")[0],
                    "has_results": has_results,
                    "n_result_files": len(result_files),
                })
    return outputs


def count_llm_calls(exp_dir: Path):
    """Try to read token tracker for LLM call counts."""
    tracker_path = exp_dir / "token_tracker.json"
    if tracker_path.exists():
        try:
            with open(tracker_path) as f:
                data = json.load(f)
            return data
        except Exception:
            pass

    # Also check interactions
    interactions_path = exp_dir / "token_tracker_interactions.json"
    if interactions_path.exists():
        try:
            with open(interactions_path) as f:
                data = json.load(f)
            total = sum(len(v) if isinstance(v, list) else 0 for v in data.values())
            return {"total_interactions": total}
        except Exception:
            pass

    return None


def read_llm_call_log():
    """Read the live LLM call log written by call_ollama_v1."""
    log_path = BASE_DIR / "llm_calls.jsonl"
    if not log_path.exists():
        return None

    calls = []
    try:
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        calls.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except Exception:
        return None

    if not calls:
        return None

    # Aggregate per model
    model_stats = {}
    total_duration = 0
    total_retries = 0
    for c in calls:
        m = c.get("model", "unknown")
        if m not in model_stats:
            model_stats[m] = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "duration_s": 0}
        model_stats[m]["calls"] += 1
        model_stats[m]["prompt_tokens"] += c.get("prompt_tokens", 0)
        model_stats[m]["completion_tokens"] += c.get("completion_tokens", 0)
        model_stats[m]["duration_s"] += c.get("duration_s", 0)
        total_duration += c.get("duration_s", 0)
        total_retries += c.get("empty_retries", 0)

    return {
        "total_calls": len(calls),
        "total_duration_s": total_duration,
        "total_empty_retries": total_retries,
        "last_call": calls[-1] if calls else None,
        "per_model": model_stats,
    }


def render_dashboard(exp_dir: Path, start_time: datetime, refresh_count: int):
    """Render the full monitoring dashboard."""
    clear_screen()
    
    exp_name = exp_dir.name
    elapsed = datetime.now() - start_time
    elapsed_str = str(elapsed).split(".")[0]

    # ── Header ────────────────────────────────────────────────────────────
    print(f"{BOLD}{CYAN}╔{'═' * 78}╗{RESET}")
    print(f"{BOLD}{CYAN}║  🔬  AI-Scientist-v2 Pipeline Monitor{' ' * 40}║{RESET}")
    print(f"{BOLD}{CYAN}╚{'═' * 78}╝{RESET}")
    print()
    print(f"  {DIM}Experiment:{RESET} {BOLD}{exp_name}{RESET}")
    print(f"  {DIM}Elapsed:{RESET}    {elapsed_str}  {DIM}|{RESET}  {DIM}Refresh #{refresh_count}{RESET}  {DIM}|{RESET}  {DIM}{datetime.now().strftime('%H:%M:%S')}{RESET}")
    print()

    # ── Ollama Status ─────────────────────────────────────────────────────
    server_up, loaded_models = get_ollama_status()
    status_icon = f"{GREEN}●{RESET}" if server_up else f"{RED}●{RESET}"
    status_text = f"{GREEN}Running{RESET}" if server_up else f"{RED}Down{RESET}"
    print(f"  {BOLD}Ollama Server:{RESET} {status_icon} {status_text}")
    if loaded_models:
        for m in loaded_models:
            color = YELLOW if "none" not in m.lower() else DIM
            print(f"    {color}↳ {m}{RESET}")
    print()

    # ── Worker Processes ──────────────────────────────────────────────────
    workers = get_worker_processes()
    print(f"  {BOLD}Pipeline Processes:{RESET} {len(workers)} active")
    if workers:
        for w in workers[:6]:  # Show max 6
            print(f"    {DIM}PID {w['pid']:>7}  |  RAM: {w['mem_mb']:.0f} MB{RESET}")
    print()

    # ── Stage Progress ────────────────────────────────────────────────────
    log_dir = exp_dir / "logs"
    stages = scan_stage_logs(log_dir)

    print(f"  {BOLD}{MAGENTA}{'─' * 78}{RESET}")
    print(f"  {BOLD}Stage Progress:{RESET}")
    if stages:
        print(f"    {DIM}{'Stage':<30} {'Nodes':>6} {'Good':>6} {'Buggy':>6} {'Best Metric':>15}{RESET}")
        print(f"    {DIM}{'─'*30} {'─'*6} {'─'*6} {'─'*6} {'─'*15}{RESET}")
        for name, data in stages.items():
            good_pct = f"({data['good']/data['nodes']*100:.0f}%)" if data["nodes"] > 0 else ""
            node_color = GREEN if data["good"] > 0 else (RED if data["buggy"] > 0 else WHITE)
            print(f"    {CYAN}{name:<30}{RESET} {node_color}{data['nodes']:>6}{RESET} "
                  f"{GREEN}{data['good']:>6}{RESET} {RED}{data['buggy']:>6}{RESET} "
                  f"{YELLOW}{data['best_metric']:>15}{RESET} {DIM}{good_pct}{RESET}")
    else:
        print(f"    {DIM}No stage data yet (still initializing...){RESET}")
    print()

    # ── Worker Outputs ────────────────────────────────────────────────────
    outputs = scan_worker_outputs(exp_dir)
    print(f"  {BOLD}Generated Code:{RESET} {len(outputs)} runfile(s)")
    if outputs:
        print(f"    {DIM}{'Worker':<40} {'Size':>8} {'Modified':>10} {'Results':>8}{RESET}")
        print(f"    {DIM}{'─'*40} {'─'*8} {'─'*10} {'─'*8}{RESET}")
        for o in outputs[-8:]:  # Show latest 8
            result_status = f"{GREEN}✓ {o['n_result_files']}{RESET}" if o["has_results"] else f"{YELLOW}⏳{RESET}"
            print(f"    {o['path']:<40} {o['size_bytes']:>6}B  {o['modified']:>10} {result_status:>8}")
    print()

    # ── LLM Inference Stats ────────────────────────────────────────────────
    llm_log = read_llm_call_log()
    if llm_log:
        total = llm_log["total_calls"]
        dur = llm_log["total_duration_s"]
        retries = llm_log["total_empty_retries"]
        last = llm_log.get("last_call", {})
        
        print(f"  {BOLD}LLM Inference Stats:{RESET}  {total} calls  |  {dur:.0f}s total  |  {retries} empty retries")
        if last:
            print(f"    {DIM}Last call: {last.get('model','?')} at {last.get('ts','?')} ({last.get('duration_s',0):.1f}s, PID {last.get('pid','?')}){RESET}")
        
        print(f"    {DIM}{'Model':<25} {'Calls':>6} {'Prompt':>10} {'Completion':>12} {'Avg Latency':>12}{RESET}")
        print(f"    {DIM}{'─'*25} {'─'*6} {'─'*10} {'─'*12} {'─'*12}{RESET}")
        for m, s in llm_log["per_model"].items():
            avg_lat = s["duration_s"] / s["calls"] if s["calls"] > 0 else 0
            print(f"    {CYAN}{m:<25}{RESET} {s['calls']:>6} {s['prompt_tokens']:>10} {s['completion_tokens']:>12} {avg_lat:>10.1f}s")
        print()
    else:
        # Fallback to token tracker
        token_data = count_llm_calls(exp_dir)
        if token_data:
            print(f"  {BOLD}Token Usage:{RESET}")
            if "total_interactions" in token_data:
                print(f"    Total LLM calls: {token_data['total_interactions']}")
            else:
                for model, info in token_data.items():
                    if isinstance(info, dict) and "tokens" in info:
                        tokens = info["tokens"]
                        total = tokens.get("prompt", 0) + tokens.get("completion", 0)
                        print(f"    {CYAN}{model:<30}{RESET} {total:>10} tokens")
            print()

    # ── Filesystem Activity ───────────────────────────────────────────────
    recent_files = []
    for p in exp_dir.rglob("*"):
        if p.is_file() and not p.name.startswith("."):
            mtime = p.stat().st_mtime
            if time.time() - mtime < 120:  # Modified in last 2 minutes
                recent_files.append((mtime, p))
    recent_files.sort(key=lambda x: x[0], reverse=True)

    if recent_files:
        print(f"  {BOLD}Recent File Activity:{RESET} (last 2 min)")
        for mtime, p in recent_files[:6]:
            rel = p.relative_to(exp_dir)
            age = int(time.time() - mtime)
            print(f"    {DIM}{age:>4}s ago{RESET}  {rel}")
        print()

    # ── Footer ────────────────────────────────────────────────────────────
    print(f"  {DIM}Press Ctrl+C to stop monitoring{RESET}")


def main():
    parser = argparse.ArgumentParser(description="AI-Scientist-v2 Progress Monitor")
    parser.add_argument("--experiment", "-e", type=str, help="Experiment directory name")
    parser.add_argument("--interval", "-i", type=int, default=5, help="Refresh interval in seconds")
    args = parser.parse_args()

    if args.experiment:
        exp_dir = EXPERIMENTS_DIR / args.experiment
        if not exp_dir.exists():
            print(f"Error: {exp_dir} does not exist")
            sys.exit(1)
    else:
        exp_dir = find_latest_experiment()
        if not exp_dir:
            print("No experiments found. Start a pipeline run first.")
            sys.exit(1)

    print(f"Monitoring: {exp_dir.name}")
    print(f"Refresh interval: {args.interval}s")
    print()

    start_time = datetime.now()
    refresh_count = 0

    def handle_sigint(sig, frame):
        clear_screen()
        print(f"\n{BOLD}Monitor stopped.{RESET} Ran for {str(datetime.now() - start_time).split('.')[0]}")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    while True:
        try:
            refresh_count += 1
            render_dashboard(exp_dir, start_time, refresh_count)
            time.sleep(args.interval)
        except KeyboardInterrupt:
            handle_sigint(None, None)


if __name__ == "__main__":
    main()
