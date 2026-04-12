"""
PlotCodeExecutor - runs plotting code in a sandbox via the OpenHands SDK.

Reuses the core pattern from the experiment module's openhands_coder.py
(LLM + Agent + Conversation + Tools), but the task changes from "generating
experiment code" to "executing/debugging plotting scripts and producing image files".
"""

import glob
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from camyla.paper_agent.func.config_resolver import load_qwbe_config, resolve_config_path

logger = logging.getLogger(__name__)

DEFAULT_PYTHON_PATH = "/opt/conda/bin/python"


def _find_qwbe_config() -> Optional[Path]:
    """Locate the active config file for plot execution."""
    return resolve_config_path(search_from=__file__)


def _find_python_path() -> str:
    """Find available Python interpreter."""
    candidates = [
        "/opt/conda/envs/py310/bin/python",
        "/opt/conda/bin/python",
        "python3",
        "python",
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return "python3"


class PlotCodeExecutor:
    """
    Executes matplotlib/seaborn plotting scripts via OpenHands Agent.
    The agent can edit, debug, and re-run scripts until plots are generated.
    """

    def __init__(
        self,
        workspace_dir: Optional[Path] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_iterations: int = 80,
    ):
        self.workspace_dir = Path(workspace_dir) if workspace_dir else None
        self.python_path = _find_python_path()

        from camyla.model_config import get_role
        self._endpoint = get_role("plot_executor", group="paper_writing")

        self._qwbe_cfg = self._load_config()
        code_cfg = self._qwbe_cfg.get("experiment", {}).get("code", {})

        self.model = model or self._endpoint["model"]
        self.temperature = temperature if temperature is not None else self._endpoint.get("temperature", 0.9)
        self.max_iterations = max_iterations
        self.max_output_tokens = code_cfg.get("max_tokens", 16384)

        self.llm = None
        self.agent = None
        self.conversation = None
        self._initialized = False

    @staticmethod
    def _load_config() -> Dict[str, Any]:
        return load_qwbe_config(search_from=__file__)

    def initialize(self, workspace_dir: Path) -> bool:
        """Initialize OpenHands environment for plotting."""
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        try:
            import logging as _logging
            for mod in [
                "openhands", "openhands.sdk", "openhands.sdk.conversation",
                "openhands.sdk.conversation.state", "openhands.agent",
                "openhands.tools", "litellm", "httpx", "httpcore",
            ]:
                _logging.getLogger(mod).setLevel(_logging.WARNING)
            for mod in [
                "openhands.tools.terminal.terminal.terminal_session",
                "openhands.tools.terminal",
            ]:
                _logging.getLogger(mod).setLevel(_logging.ERROR)

            from openhands.sdk import LLM, Agent, Conversation, Tool
            from openhands.sdk import AgentContext
            from openhands.sdk.context import Skill
            from openhands.tools.file_editor import FileEditorTool
            from openhands.tools.terminal import TerminalTool

            try:
                from camyla.treesearch.openhands_coder import ConciseVisualizer
            except ImportError:
                ConciseVisualizer = None

            api_key = self._endpoint["api_key"]
            base_url = self._endpoint["base_url"]
            llm_model_string = self.model
            if not api_key:
                logger.error("API Key not set for endpoint 'plot_execution'")
                return False
            logger.info(f"PlotCodeExecutor: using {self.model} @ {base_url}")

            self.llm = LLM(
                model=llm_model_string,
                api_key=api_key,
                base_url=base_url,
                max_output_tokens=self.max_output_tokens,
                timeout=180,
                temperature=self.temperature,
                num_retries=5,
                retry_multiplier=4.0,
                retry_min_wait=5,
                retry_max_wait=60,
            )

            plot_skill = Skill(
                name="plot_execution",
                content=self._build_plot_skill(),
                trigger=None,
            )

            agent_context = AgentContext(skills=[plot_skill])

            self.agent = Agent(
                llm=self.llm,
                tools=[
                    Tool(name=FileEditorTool.name),
                    Tool(name=TerminalTool.name),
                ],
                agent_context=agent_context,
            )

            conv_kwargs = dict(
                agent=self.agent,
                workspace=str(self.workspace_dir),
                max_iteration_per_run=self.max_iterations,
            )
            if ConciseVisualizer is not None:
                conv_kwargs["visualizer"] = ConciseVisualizer(verbose=False)

            self.conversation = Conversation(**conv_kwargs)

            self._initialized = True
            logger.info(f"PlotCodeExecutor initialized: workspace={self.workspace_dir}")
            return True

        except ImportError as e:
            logger.error(f"Failed to import OpenHands SDK: {e}")
            return False
        except Exception as e:
            logger.error(f"PlotCodeExecutor initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _build_plot_skill(self) -> str:
        return f"""# Plot Execution Skill

## Your Role
You are a scientific figure generation assistant. Your job is to execute Python plotting
scripts and ensure they produce high-quality PDF and PNG figures for academic papers.

## Python Path
Use `{self.python_path}` for all Python commands.

## Workflow
1. Read the plot script files in the workspace
2. Run each script: `{self.python_path} <script_name>.py`
3. If a script fails, debug it:
   - Read the error message carefully
   - Fix the code (missing imports, data issues, path problems, etc.)
   - Re-run until it succeeds
4. Verify that output files (PDF/PNG) were created
5. If data files are provided in the workspace, use them for realistic plots

## Rules
- Always use `{self.python_path}` (NOT `python` or `python3`)
- Fix errors yourself — do not give up after the first failure
- Ensure matplotlib uses the Agg backend (non-interactive): add `matplotlib.use('Agg')` at the top
- Keep the academic style (300 DPI, Arial font, clean axes)
- Save outputs to the designated output directory
"""

    def execute_plot_task(
        self,
        plot_scripts: Dict[str, str],
        output_dir: Path,
        data_files: Optional[Dict[str, Path]] = None,
        repair_feedback: str = "",
    ) -> Tuple[bool, List[Path]]:
        """
        Execute plotting scripts and collect generated figures.

        Args:
            plot_scripts: Dict mapping filename -> Python code content
            output_dir: Where final figures should be collected
            data_files: Optional dict mapping filename -> source path for data files
                        to copy into workspace

        Returns:
            (success, list_of_generated_figure_paths)
        """
        if not self._initialized:
            logger.error("PlotCodeExecutor not initialized. Call initialize() first.")
            return False, []

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write plot scripts to workspace
        for filename, code in plot_scripts.items():
            script_path = self.workspace_dir / filename
            script_path.write_text(code, encoding="utf-8")
            logger.info(f"  Wrote plot script: {script_path}")

        # Copy data files if provided
        if data_files:
            for fname, src_path in data_files.items():
                dest = self.workspace_dir / fname
                if Path(src_path).is_dir():
                    shutil.copytree(src_path, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(src_path, dest)
                logger.info(f"  Copied data file: {fname}")

        # Build task prompt
        script_list = "\n".join(f"- `{fn}`" for fn in plot_scripts.keys())
        data_list = ""
        if data_files:
            data_list = "\nAvailable data files:\n" + "\n".join(
                f"- `{fn}`" for fn in data_files.keys()
            )
        repair_block = ""
        if repair_feedback.strip():
            repair_block = (
                "\nRepair requirements (must be fully resolved before success):\n"
                f"{repair_feedback.strip()}\n"
            )

        task_prompt = f"""Execute the following plot scripts and ensure they generate figures successfully.

Scripts to run:
{script_list}
{data_list}

For each script:
1. Run it with `{self.python_path} <script_name>`
2. If it fails, fix the errors and re-run
3. Verify that the canonical expected files `<figure_id>.pdf` and `<figure_id>.png` exist after execution
4. Remove any placeholder/TODO code before finishing

Output directory for figures: `{output_dir}`
{repair_block}

Important: Make sure `import matplotlib; matplotlib.use('Agg')` is at the top of each script
before any other matplotlib imports, to avoid display issues.
"""

        # Record pre-existing files so we only count NEW outputs as success
        pre_existing: set = set()
        for pattern in ["*.pdf", "*.png"]:
            for p in output_dir.glob(pattern):
                pre_existing.add(p.name)

        try:
            self.conversation.send_message(task_prompt)

            for attempt in range(3):
                try:
                    self.conversation.run()
                    break
                except Exception as e:
                    error_str = str(e).lower()
                    if any(kw in error_str for kw in ["rate limit", "timeout", "connection"]):
                        delay = 5 * (2 ** attempt)
                        logger.warning(f"API error, retrying in {delay}s: {e}")
                        time.sleep(delay)
                    else:
                        logger.error(f"Conversation run failed: {e}")
                        break

        except Exception as e:
            logger.error(f"Failed to execute plot task: {e}")

        expected_stems = set()
        for fn in plot_scripts:
            stem = fn.replace("_plot.py", "")
            expected_stems.add(stem)

        generated = []

        # 1. Collect from workspace (top-level)
        for pattern in ["*.pdf", "*.png"]:
            for fig_path in self.workspace_dir.glob(pattern):
                dest = output_dir / fig_path.name
                if fig_path != dest:
                    shutil.copy2(fig_path, dest)
                generated.append(dest)

        # 2. Collect from workspace subdirectories
        for pattern in ["**/*.pdf", "**/*.png"]:
            for fig_path in self.workspace_dir.glob(pattern):
                if fig_path.parent == self.workspace_dir:
                    continue
                dest = output_dir / fig_path.name
                if not dest.exists():
                    shutil.copy2(fig_path, dest)
                    generated.append(dest)

        # 3. Collect from workspace parent (result_dir) in case scripts saved there
        workspace_parent = self.workspace_dir.parent
        if workspace_parent != output_dir:
            for pattern in ["*.pdf", "*.png"]:
                for fig_path in workspace_parent.glob(pattern):
                    dest = output_dir / fig_path.name
                    if not dest.exists():
                        shutil.copy2(fig_path, dest)
                    generated.append(dest)

        # 4. Check output_dir for expected result_fig files only (ignore pre-existing)
        for pattern in ["*.pdf", "*.png"]:
            for fig_path in output_dir.glob(pattern):
                if fig_path.name not in pre_existing and fig_path not in generated:
                    generated.append(fig_path)

        unique_figures = list(dict.fromkeys(generated))
        expected_outputs = {
            f"{stem}.pdf" for stem in expected_stems
        } | {
            f"{stem}.png" for stem in expected_stems
        }
        present_outputs = {f.name for f in unique_figures}
        missing_outputs = sorted(expected_outputs - present_outputs)
        new_count = sum(1 for f in unique_figures if f.name not in pre_existing)
        logger.info(f"PlotCodeExecutor collected {len(unique_figures)} figure files "
                     f"({new_count} newly generated)")
        if missing_outputs:
            logger.info(
                "PlotCodeExecutor missing expected canonical outputs: %s",
                ", ".join(missing_outputs),
            )

        return not missing_outputs, unique_figures

    def cleanup(self):
        """Release resources."""
        self.conversation = None
        self.agent = None
        self.llm = None
        self._initialized = False
