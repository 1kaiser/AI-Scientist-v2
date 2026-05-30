import atexit
import logging
import shutil
import json
import os
import pickle
import time
from . import backend
from .journal import Journal, Node
from .journal2report import journal2report
from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    SpinnerColumn,
)
from rich.table import Table
from rich.text import Text
from rich.status import Status
from rich.tree import Tree
from .utils.config import load_task_desc, prep_agent_workspace, save_run, load_cfg
from .agent_manager import AgentManager
from pathlib import Path
from .agent_manager import Stage
from .log_summarization import overall_summarize


logger = logging.getLogger("ai-scientist")
console = Console()


# ─── Progress State ──────────────────────────────────────────────────────────
class PipelineMonitor:
    """Tracks pipeline progress in-process for the Rich live dashboard."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.start_time = time.time()
        self.current_stage_name = "initializing"
        self.completed_stages = []
        self.total_steps = 0
        self.total_nodes = 0
        self.good_nodes = 0
        self.buggy_nodes = 0
        self.best_metric_str = "—"
        self.last_event = "Starting pipeline..."
        self.llm_calls = 0
        self.llm_total_tokens = 0
        self.llm_total_duration = 0.0
        self.llm_empty_retries = 0
        self.log_dir = cfg.log_dir

    @property
    def elapsed(self):
        e = int(time.time() - self.start_time)
        h, m, s = e // 3600, (e % 3600) // 60, e % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    def update_from_step(self, stage, journal):
        """Called by step_callback after each iteration."""
        self.current_stage_name = stage.name
        self.total_steps += 1
        self.total_nodes = len(journal.nodes)
        self.good_nodes = len(journal.good_nodes)
        self.buggy_nodes = len(journal.buggy_nodes)

        best = journal.get_best_node(cfg=self.cfg)
        if best and hasattr(best, "metric") and hasattr(best.metric, "value"):
            self.best_metric_str = f"{best.metric.value:.4f}"
        elif best:
            self.best_metric_str = str(best.metric)

        self.last_event = (
            f"Step {len(journal)}/{stage.max_iterations} "
            f"at {stage.name}"
        )

    def update_from_llm_log(self):
        """Read the llm_calls.jsonl written by call_ollama_v1."""
        log_path = Path(os.environ.get("AI_SCIENTIST_ROOT", ".")) / "llm_calls.jsonl"
        if not log_path.exists():
            return
        try:
            calls = 0
            total_tok = 0
            total_dur = 0.0
            retries = 0
            with open(log_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        c = json.loads(line)
                        calls += 1
                        total_tok += c.get("prompt_tokens", 0) + c.get("completion_tokens", 0)
                        total_dur += c.get("duration_s", 0)
                        retries += c.get("empty_retries", 0)
                    except json.JSONDecodeError:
                        continue
            self.llm_calls = calls
            self.llm_total_tokens = total_tok
            self.llm_total_duration = total_dur
            self.llm_empty_retries = retries
        except Exception:
            pass

    def mark_stage_complete(self, stage_name):
        if stage_name not in self.completed_stages:
            self.completed_stages.append(stage_name)

    def get_ollama_status(self):
        """Quick check: is Ollama up and what model is loaded?"""
        try:
            import requests as req
            r = req.get("http://localhost:11434/api/ps", timeout=1)
            if r.status_code == 200:
                models = r.json().get("models", [])
                if models:
                    m = models[0]
                    name = m.get("name", "?")
                    vram_gb = m.get("size_vram", m.get("size", 0)) / 1e9
                    return f"[green]●[/green] {name} ({vram_gb:.1f}GB)"
                return "[green]●[/green] idle (no model loaded)"
        except Exception:
            pass
        return "[red]●[/red] unreachable"


def journal_to_rich_tree(journal: Journal, cfg):
    best_node = journal.get_best_node(cfg=cfg)

    def append_rec(node: Node, tree):
        if node.is_buggy:
            s = "[red]◍ bug"
        else:
            style = "bold " if node is best_node else ""

            if node is best_node:
                s = f"[{style}green]● {node.metric.value:.3f} (best)"
            else:
                s = f"[{style}green]● {node.metric.value:.3f}"

        subtree = tree.add(s)
        for child in node.children:
            append_rec(child, subtree)

    tree = Tree("[bold blue]Solution tree")
    for n in journal.draft_nodes:
        append_rec(n, tree)
    return tree


def perform_experiments_bfts(config_path: str):
    # turn config path string into a path object
    config_path = Path(config_path)
    cfg = load_cfg(config_path)
    logger.info(f'Starting run "{cfg.exp_name}"')

    task_desc = load_task_desc(cfg)
    print(task_desc)
    task_desc_str = backend.compile_prompt_to_md(task_desc)

    global_step = 0

    with Status("Preparing agent workspace (copying and extracting files) ..."):
        prep_agent_workspace(cfg)

    def cleanup():
        if global_step == 0:
            shutil.rmtree(cfg.workspace_dir)

    atexit.register(cleanup)

    manager = AgentManager(
        task_desc=task_desc,
        cfg=cfg,
        workspace_dir=Path(cfg.workspace_dir),
    )

    # ── Initialize Monitor ────────────────────────────────────────────────
    monitor = PipelineMonitor(cfg)

    # ── Rich Progress Bar ─────────────────────────────────────────────────
    prog = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
    )
    prog_task = prog.add_task("Progress:", total=cfg.agent.steps, completed=global_step)

    # ── Step Callback ─────────────────────────────────────────────────────
    def step_callback(stage, journal):
        nonlocal global_step
        global_step += 1
        prog.update(prog_task, completed=global_step)
        monitor.update_from_step(stage, journal)

        try:
            # Save notes
            notes_dir = cfg.log_dir / f"stage_{stage.name}" / "notes"
            notes_dir.mkdir(parents=True, exist_ok=True)

            # Save latest node summary
            if journal.nodes:
                latest_node = journal.nodes[-1]
                if hasattr(latest_node, "_agent"):
                    summary = latest_node._agent._generate_node_summary(latest_node)
                    with open(
                        notes_dir / f"node_{latest_node.id}_summary.json", "w"
                    ) as f:
                        json.dump(summary, f, indent=2)

            if cfg.agent.get("summary", None) is not None:
                current_findings = journal.generate_summary(
                    include_code=False, 
                    **{
                        "model": cfg.agent.summary.model, 
                        "temp": cfg.agent.summary.temp
                    }
                )
            else:
                current_findings = journal.generate_summary(include_code=False)

            best_metric = journal.get_best_node(cfg=cfg)

            stage_summary = {
                "stage": stage.name,
                "step": global_step,
                "total_nodes": len(journal.nodes),
                "buggy_nodes": len(journal.buggy_nodes),
                "good_nodes": len(journal.good_nodes),
                "best_metric": (
                    str(best_metric.metric)
                    if best_metric
                    else "None"
                ),
                "current_findings": current_findings,
                "elapsed_s": int(time.time() - monitor.start_time),
            }

            with open(notes_dir / "stage_progress.json", "w") as f:
                json.dump(stage_summary, f, indent=2)

            save_run(cfg, journal, stage_name=f"stage_{stage.name}")

        except Exception as e:
            logger.warning(f"Error in step callback: {e}")

        logger.info(
            f"Step {len(journal)}/{stage.max_iterations} at stage_{stage.name} | "
            f"nodes={len(journal.nodes)} good={len(journal.good_nodes)} buggy={len(journal.buggy_nodes)}"
        )

    # ── Exec Callback ─────────────────────────────────────────────────────
    # Note: In the parallel agent architecture, workers create their own
    # Interpreter inside _process_node_wrapper, so exec_callback is received
    # by step() but never actually called. We provide a passthrough that
    # satisfies the type signature (Callable[[str, bool], ExecutionResult]).
    def exec_callback(code: str, reset: bool = False):
        monitor.last_event = "Executing code..."
        from .interpreter import Interpreter
        interp = Interpreter(
            timeout=cfg.exec.timeout,
            working_dir=str(cfg.workspace_dir),
        )
        return interp.run(code, reset=reset)

    # ── Live Dashboard Generator ──────────────────────────────────────────
    def generate_live():
        # Refresh LLM call stats
        monitor.update_from_llm_log()

        current_stage = manager.current_stage
        current_journal = manager.journals.get(
            current_stage.name if current_stage else None, None
        )

        # ─ Solution Tree ─
        if current_journal and current_journal.nodes:
            tree = journal_to_rich_tree(current_journal, cfg)
        else:
            tree = Tree("[bold blue]No results yet")

        # ─ Info Table ─
        info_table = Table(show_header=False, box=None, padding=(0, 1))
        info_table.add_column("Key", style="dim")
        info_table.add_column("Value")

        info_table.add_row("Experiment", f"[bold]{cfg.exp_name}[/bold]")
        info_table.add_row("Elapsed", f"[cyan]{monitor.elapsed}[/cyan]")
        info_table.add_row(
            "Current Stage",
            f"[cyan]{current_stage.name if current_stage else 'done'}[/cyan]",
        )
        info_table.add_row(
            "Completed",
            f"[green]{', '.join(manager.completed_stages) or '—'}[/green]",
        )
        info_table.add_row("Total Steps", f"[bold]{monitor.total_steps}[/bold]")
        info_table.add_row("", "")  # spacer

        # ─ Node Stats ─
        info_table.add_row(
            "Nodes",
            f"[white]{monitor.total_nodes}[/white] total  "
            f"[green]{monitor.good_nodes}[/green] good  "
            f"[red]{monitor.buggy_nodes}[/red] buggy",
        )
        info_table.add_row("Best Metric", f"[yellow]{monitor.best_metric_str}[/yellow]")
        info_table.add_row("", "")

        # ─ Ollama Status ─
        ollama_status = monitor.get_ollama_status()
        info_table.add_row("Ollama", ollama_status)

        # ─ LLM Stats ─
        if monitor.llm_calls > 0:
            avg_lat = monitor.llm_total_duration / monitor.llm_calls
            info_table.add_row(
                "LLM Calls",
                f"{monitor.llm_calls} calls  |  "
                f"{monitor.llm_total_tokens:,} tokens  |  "
                f"avg {avg_lat:.1f}s",
            )
            if monitor.llm_empty_retries > 0:
                info_table.add_row(
                    "Empty Retries",
                    f"[yellow]{monitor.llm_empty_retries}[/yellow]",
                )

        # ─ File Paths ─
        file_paths = Table(show_header=False, box=None, padding=(0, 1))
        file_paths.add_column("Label", style="dim")
        file_paths.add_column("Path", style="yellow")
        file_paths.add_row("Logs", str(cfg.log_dir))
        file_paths.add_row("Workspace", str(cfg.workspace_dir))

        # ─ Layout ─
        left = Group(
            Panel(info_table, title="[bold]Pipeline Status[/bold]", border_style="blue"),
            prog,
        )
        right = Panel(tree, title="[bold]Solution Tree[/bold]", border_style="green")

        return Panel(
            Group(
                Padding(file_paths, (0, 1, 0, 1)),
                Columns(
                    [Padding(left, (0, 1, 0, 0)), Padding(right, (0, 0, 0, 1))],
                    equal=True,
                ),
                Text(f"  {monitor.last_event}", style="dim"),
            ),
            title=f'[b]🔬 AI-Scientist-v2: [bold green]"{cfg.exp_name}"[/b]',
            subtitle="Press [b]Ctrl+C[/b] to stop",
            border_style="bright_blue",
        )

    # ── Run Pipeline with Live Dashboard ──────────────────────────────────
    console.print(f"\n[bold cyan]🚀 Starting experiment: {cfg.exp_name}[/bold cyan]\n")

    with Live(generate_live(), refresh_per_second=2, console=console) as live:
        # Wrap the manager.run to update the live display on each step
        original_step_callback = step_callback

        def live_step_callback(stage, journal):
            original_step_callback(stage, journal)
            live.update(generate_live())

        manager.run(exec_callback=exec_callback, step_callback=live_step_callback)

    # ── Post-Run: Save State ──────────────────────────────────────────────
    console.print("\n[bold green]✓ Experiment complete![/bold green]\n")

    manager_pickle_path = cfg.log_dir / "manager.pkl"
    try:
        with open(manager_pickle_path, "wb") as f:
            pickle.dump(manager, f)
        logger.info(f"Saved manager state to: {manager_pickle_path}")
    except Exception as e:
        logger.warning(f"Failed to save full manager state: {e}")
        try:
            with open(manager_pickle_path, "wb") as f:
                pickle.dump(manager.journals.items(), f)
            logger.info(f"Saved manager journals to: {manager_pickle_path}")
        except Exception as e:
            logger.error(f"Failed to save manager journals: {e}")

    if cfg.generate_report:
        console.print("[bold]Generating final report from all stages...[/bold]")
        (
            draft_summary,
            baseline_summary,
            research_summary,
            ablation_summary,
        ) = overall_summarize(manager.journals.items(), cfg)
        draft_summary_path = cfg.log_dir / "draft_summary.json"
        baseline_summary_path = cfg.log_dir / "baseline_summary.json"
        research_summary_path = cfg.log_dir / "research_summary.json"
        ablation_summary_path = cfg.log_dir / "ablation_summary.json"

        with open(draft_summary_path, "w") as draft_file:
            json.dump(draft_summary, draft_file, indent=2)

        with open(baseline_summary_path, "w") as baseline_file:
            json.dump(baseline_summary, baseline_file, indent=2)

        with open(research_summary_path, "w") as research_file:
            json.dump(research_summary, research_file, indent=2)

        with open(ablation_summary_path, "w") as ablation_file:
            json.dump(ablation_summary, ablation_file, indent=2)

        console.print(f"[green]Summary reports written to:[/green]")
        console.print(f"  - {draft_summary_path}")
        console.print(f"  - {baseline_summary_path}")
        console.print(f"  - {research_summary_path}")
        console.print(f"  - {ablation_summary_path}")

    # ── Final Summary ─────────────────────────────────────────────────────
    console.print(f"\n[bold]Pipeline finished in {monitor.elapsed}[/bold]")
    console.print(
        f"  Stages: {len(manager.completed_stages)}  |  "
        f"Steps: {monitor.total_steps}  |  "
        f"Nodes: {monitor.total_nodes} ({monitor.good_nodes} good, {monitor.buggy_nodes} buggy)  |  "
        f"Best: {monitor.best_metric_str}"
    )
    if monitor.llm_calls > 0:
        console.print(
            f"  LLM: {monitor.llm_calls} calls  |  "
            f"{monitor.llm_total_tokens:,} tokens  |  "
            f"{monitor.llm_total_duration:.0f}s inference time  |  "
            f"{monitor.llm_empty_retries} empty retries"
        )
    console.print()


if __name__ == "__main__":
    cfg_path = "treesearch/utils/config.yaml"
    cfg = load_cfg(cfg_path)
    perform_experiments_bfts(cfg_path)
