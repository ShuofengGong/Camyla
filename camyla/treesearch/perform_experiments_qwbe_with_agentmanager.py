import logging
import json
import pickle
import os
from . import backend
from .journal import Journal, Node
from rich.columns import Columns
from rich.console import Group
from rich.live import Live
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)
from rich.text import Text
from rich.status import Status
from rich.tree import Tree
from .utils.config import load_task_desc, prep_agent_workspace, save_run, load_cfg
from .utils.metric import set_tiebreak_threshold
from .utils.text_manager import set_query_log_dir
from .agent_manager import AgentManager
from pathlib import Path
from .agent_manager import Stage
from .log_summarization import overall_summarize, generate_paper_ready_summary


logger = logging.getLogger("camyla")


def journal_to_rich_tree(journal: Journal):
    best_node = journal.get_best_node()
    def append_rec(node: Node, tree):
        if node.is_buggy:
            s = "[red]◍ bug"
        else:
            style = "bold " if node is best_node else ""

            if node is best_node:
                s = f"[{style}green]● {node.metric.get_mean_value():.3f} (best)"
            else:
                s = f"[{style}green]● {node.metric.get_mean_value():.3f}"

        subtree = tree.add(s)
        for child in node.children:
            append_rec(child, subtree)

    tree = Tree("[bold blue]Solution tree")
    for n in journal.draft_nodes:
        append_rec(n, tree)
    return tree


def perform_experiments_qwbe(config_path: str = None, resume_checkpoint_path: str = None):
    manager = None
    cfg = None

    if resume_checkpoint_path:
        logger.info(f"Resuming experiment from checkpoint: {resume_checkpoint_path}")
        # Set env var so AgentManager.__setstate__ can find the checkpoint path
        os.environ["QWBE_CHECKPOINT_PATH"] = resume_checkpoint_path
        with open(resume_checkpoint_path, "rb") as f:
            try:
                manager = pickle.load(f)
                logger.debug(f"Loaded manager: current_stage={manager.current_stage.name if manager.current_stage else 'None'}, stages={[s.name for s in manager.stages]}")
            except Exception as e:
                logger.error(f"Error loading checkpoint: {e}")
                raise
            
        # --- Key fix: add this guard ---
        if manager.current_stage is None:
            logger.info("Experiment has already completed. Nothing to resume. Exiting.")
            # You could fall through to run remaining cleanup (report generation, etc.)
            # but for safety we return immediately.
            return
        
        # Re-read the YAML config and override iteration limits so that
        # changes the user made to the config file take effect on resume
        # (pickle restores the original cfg, ignoring YAML edits).
        _timeout_recalculated = False
        if config_path:
            try:
                from omegaconf import OmegaConf
                fresh_cfg = OmegaConf.load(config_path)
                override_keys = [
                    "stage1_max_iters", "stage2_max_iters", "stage3_max_iters",
                ]
                if OmegaConf.select(fresh_cfg, "experiment.stages") and OmegaConf.select(manager.cfg, "experiment.stages"):
                    for key in override_keys:
                        new_val = OmegaConf.select(fresh_cfg, f"experiment.stages.{key}")
                        if new_val is not None:
                            old_val = OmegaConf.select(manager.cfg, f"experiment.stages.{key}")
                            if old_val != new_val:
                                OmegaConf.update(manager.cfg, f"experiment.stages.{key}", new_val)
                                logger.info(f"Config override on resume: {key}: {old_val} -> {new_val}")
                if OmegaConf.select(fresh_cfg, "experiment.stage2.max_iterations_per_innovation"):
                    new_val = fresh_cfg.experiment.stage2.max_iterations_per_innovation
                    old_val = OmegaConf.select(manager.cfg, "experiment.stage2.max_iterations_per_innovation")
                    if old_val != new_val:
                        OmegaConf.update(manager.cfg, "experiment.stage2.max_iterations_per_innovation", new_val)
                        logger.info(f"Config override on resume: max_iterations_per_innovation: {old_val} -> {new_val}")

                # exec.timeout: first restore the YAML base value, then recompute dynamically based on num_epochs.
                # The pickled value may be the previously recomputed one and cannot be used as a base.
                fresh_timeout = OmegaConf.select(fresh_cfg, "exec.timeout")
                if fresh_timeout is not None:
                    OmegaConf.update(manager.cfg, "exec.timeout", fresh_timeout)
                    logger.info(f"Config override on resume: exec.timeout reset to base {fresh_timeout}s")
                    manager._recalculate_dynamic_timeout()
                    _timeout_recalculated = True
            except Exception as e:
                logger.warning(f"Failed to apply config overrides from {config_path}: {e}")

        if not _timeout_recalculated:
            # Still attempt dynamic recalculation when config_path is missing or YAML loading failed.
            # cfg.exec.timeout may be the stale (already recomputed) value from the pickle,
            # but _recalculate_dynamic_timeout's env-var fallback can still supply the correct epochs.
            logger.warning("exec.timeout was not reset from YAML; dynamic timeout may use stale base value")
            manager._recalculate_dynamic_timeout()

        cfg = manager.cfg
        task_desc_str = json.dumps(manager.task_desc) # restore task_desc from manager
        logger.info(f'Resuming run "{cfg.exp_name}" from stage: {manager.current_stage.name}')

        set_query_log_dir(str(cfg.log_dir / "query_logs"))
    else:
        # Logic for creating a new experiment stays the same
        cfg = load_cfg(Path(config_path))
        logger.info(f'Starting new run "{cfg.exp_name}"')
        
        task_desc_str = load_task_desc(cfg)
        prep_agent_workspace(cfg)

        # Must be set before AgentManager init, which triggers LLM calls
        # during proposal queue initialization
        set_query_log_dir(str(cfg.log_dir / "query_logs"))
        
        manager = AgentManager(
            task_desc=task_desc_str,
            cfg=cfg,
            workspace_dir=Path(cfg.workspace_dir),
        )

    task_desc_str = backend.compile_prompt_to_md(manager.task_desc)

    # Apply metric tiebreak threshold from config
    threshold = getattr(cfg, 'metric_tiebreak_threshold', 0.005)
    set_tiebreak_threshold(threshold)
    logger.info(f"Metric tiebreak threshold set to {threshold}")

    if not resume_checkpoint_path:
        with Status("Preparing agent workspace (copying and extracting files) ..."):
            prep_agent_workspace(cfg)
    else:
        if not cfg.workspace_dir.exists():
            logger.warning(f"⚠️ Workspace directory not found during checkpoint resume: {cfg.workspace_dir}")
            logger.warning(f"   This may cause issues with proposal files and experiment data.")
        else:
            logger.info(f"✅ Workspace directory verified: {cfg.workspace_dir}")

    prog = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=20),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
    )
    status = Status("[green]Running experiments...")
    prog.add_task("Progress:", total=cfg.experiment.steps, completed=0)

    def step_callback(stage, journal):
        logger.debug("Step complete")
        try:
            # Generate and save notes for this step
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

            # Generate and save stage progress summary
            stage_summary = {
                "stage": stage.name,
                "total_nodes": len(journal.nodes),
                "buggy_nodes": len(journal.buggy_nodes),
                "good_nodes": len(journal.good_nodes),
                "best_metric": (
                    str(journal.get_best_node().metric)
                    if journal.get_best_node()
                    else "None"
                ),
                "current_findings": journal.generate_summary(include_code=False),
            }

            with open(notes_dir / "stage_progress.json", "w") as f:
                json.dump(stage_summary, f, indent=2)

            # Save the run as before
            save_run(cfg, journal, stage_name=f"stage_{stage.name}")

        except Exception as e:
            logger.warning(f"Error in step callback: {e}")

        logger.info(f"Step {len(journal)}/{stage.max_iterations} at stage_{stage.name}")

    def generate_live(manager):
        current_stage = manager.current_stage
        current_journal = manager.journals.get(
            current_stage.name if current_stage else None, None
        )

        if current_journal:
            tree = journal_to_rich_tree(current_journal)
        else:
            tree = Tree("[bold blue]No results yet")

        file_paths = [
            f"Result visualization:\n[yellow]▶ {str((cfg.log_dir / 'tree_plot.html'))}",
            f"Agent workspace directory:\n[yellow]▶ {str(cfg.workspace_dir)}",
            f"Experiment log directory:\n[yellow]▶ {str(cfg.log_dir)}",
        ]

        stage_info = [
            "[bold]Experiment Progress:",
            f"Current Stage: [cyan]{current_stage.name if current_stage else 'None'}[/cyan]",
            f"Completed Stages: [green]{', '.join(manager.completed_stages)}[/green]",
        ]

        left = Group(
            Panel(Text(task_desc_str.strip()), title="Task description"),
            Panel(Text("\n".join(stage_info)), title="Stage Progress"),
            prog,
            status,
        )
        right = tree
        wide = Group(*file_paths)

        return Panel(
            Group(
                Padding(wide, (1, 1, 1, 1)),
                Columns(
                    [Padding(left, (1, 2, 1, 1)), Padding(right, (1, 1, 1, 2))],
                    equal=True,
                ),
            ),
            title=f'[b]AIDE is working on experiment: [bold green]"{cfg.exp_name}[/b]"',
            subtitle="Press [b]Ctrl+C[/b] to stop the run",
        )

    live = Live(
        generate_live(manager),
        refresh_per_second=16,
        screen=True,
    )

    manager.run(step_callback=step_callback)

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
        logger.info("Generating final report from all stages...")
        (
            baseline_summary,
            research_summary,
            ablation_summary,
        ) = overall_summarize(manager.journals.items())
        baseline_summary_path = cfg.log_dir / "baseline_summary.json"
        research_summary_path = cfg.log_dir / "research_summary.json"
        ablation_summary_path = cfg.log_dir / "ablation_summary.json"

        with open(baseline_summary_path, "w") as baseline_file:
            json.dump(baseline_summary, baseline_file, indent=2)

        with open(research_summary_path, "w") as research_file:
            json.dump(research_summary, research_file, indent=2)

        with open(ablation_summary_path, "w") as ablation_file:
            json.dump(ablation_summary, ablation_file, indent=2)

        logger.info(f"Summary reports written: baseline={baseline_summary_path}, research={research_summary_path}, ablation={ablation_summary_path}")
        
        # Compute efficiency metrics (Params/FLOPs/Inference Time) before report
        logger.info("Computing computational efficiency metrics...")
        try:
            from camyla.treesearch.compute_efficiency import compute_all_efficiency_metrics
            efficiency_metrics = compute_all_efficiency_metrics(str(cfg.log_dir))
            logger.info(f"Computed efficiency metrics for {len(efficiency_metrics)} nodes")
        except Exception as e:
            logger.warning(f"Efficiency metrics computation failed (non-fatal): {e}")
            import traceback
            traceback.print_exc()

        logger.info("Generating paper-ready Markdown summary...")
        try:
            # Build journals_dict from merged journals
            # For Stage 2 (creative research), collect ALL proposal journals
            # so the paper summary can find the global best across all proposals.
            journals_dict = {
                '1_baseline_implementation': None,
                '2_creative_research': [],  # list of all Stage 2 proposal journals
                '3_ablation_studies': None,
            }
            for stage_name, journal in manager.journals.items():
                for prefix in journals_dict.keys():
                    if stage_name.startswith(prefix):
                        if prefix == '2_creative_research':
                            journals_dict[prefix].append(journal)
                        elif journals_dict[prefix] is None:
                            journals_dict[prefix] = journal
                        break
            
            paper_summary_md = generate_paper_ready_summary(
                journals_dict=journals_dict,
                log_dir=str(cfg.log_dir),
                use_llm_rewrite=True
            )
            
            paper_summary_path = cfg.log_dir / "experiment_report.md"
            with open(paper_summary_path, "w", encoding="utf-8") as f:
                f.write(paper_summary_md)
            
            logger.info(f"Paper-ready summary written to: {paper_summary_path}")
        except Exception as e:
            logger.warning(f"Failed to generate paper-ready summary: {e}")
            import traceback
            traceback.print_exc()
        
        
def perform_experiments_qwbe_checkpoint(config_path: str, experiment_dir: str):
    """Regenerate stage summaries from a previously-saved manager.pkl checkpoint.

    Args:
        config_path: Path to the run config (currently unused here but kept
            for signature parity with ``perform_experiments_qwbe``).
        experiment_dir: Experiment output directory containing
            ``logs/0-run/manager.pkl``. Summary JSONs are written next to it.
    """
    run_dir = os.path.join(experiment_dir, "logs", "0-run")
    manager_pickle_path = os.path.join(run_dir, "manager.pkl")
    with open(manager_pickle_path, "rb") as f:
        manager = pickle.load(f)

    print("Generating final report from all stages...")

    (
        baseline_summary,
        research_summary,
        ablation_summary,
    ) = overall_summarize(manager.journals.items())
    baseline_summary_path = os.path.join(run_dir, "baseline_summary.json")
    research_summary_path = os.path.join(run_dir, "research_summary.json")
    ablation_summary_path = os.path.join(run_dir, "ablation_summary.json")

    with open(baseline_summary_path, "w") as baseline_file:
        json.dump(baseline_summary, baseline_file, indent=2)

    with open(research_summary_path, "w") as research_file:
        json.dump(research_summary, research_file, indent=2)

    with open(ablation_summary_path, "w") as ablation_file:
        json.dump(ablation_summary, ablation_file, indent=2)

    print(f"Summary reports written to files:")
    print(f"- Baseline summary: {baseline_summary_path}")
    print(f"- Research summary: {research_summary_path}")
    print(f"- Ablation summary: {ablation_summary_path}")


if __name__ == "__main__":
    cfg_path = "treesearch/utils/config.yaml"
    cfg = load_cfg(cfg_path)
    perform_experiments_qwbe(cfg_path)
