"""
TexEditorExecutor - edits LaTeX files via the OpenHands SDK.

Reuses the core pattern from PlotCodeExecutor (LLM + Agent + Conversation + Tools),
but switches the task from "executing plotting scripts" to "directly editing
tex files".
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from camyla.paper_agent.func.config_resolver import load_qwbe_config, resolve_config_path

logger = logging.getLogger(__name__)


def _find_qwbe_config() -> Optional[Path]:
    """Locate the active config file for tex editing."""
    return resolve_config_path(search_from=__file__)


class TexEditorExecutor:
    """
    Edits LaTeX files via OpenHands Agent.

    The workspace contains only the tex file to edit; the agent uses
    FileEditorTool to make in-place modifications.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_iterations: int = 80,
    ):
        from camyla.model_config import get_role
        self._endpoint = get_role("latex_editor", group="paper_writing")

        self._qwbe_cfg = self._load_config()
        code_cfg = self._qwbe_cfg.get("experiment", {}).get("code", {})

        self.model = model or self._endpoint["model"]
        self.temperature = (
            temperature if temperature is not None else self._endpoint.get("temperature", 0.7)
        )
        self.max_iterations = max_iterations
        self.max_output_tokens = code_cfg.get("max_tokens", 16384)

        self.workspace_dir: Optional[Path] = None
        self.llm = None
        self.agent = None
        self.conversation = None
        self._initialized = False

    @staticmethod
    def _load_config() -> Dict[str, Any]:
        return load_qwbe_config(search_from=__file__)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self, workspace_dir: Path) -> bool:
        """Initialize OpenHands environment for tex editing."""
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        try:
            import logging as _logging

            for mod in [
                "openhands",
                "openhands.sdk",
                "openhands.sdk.conversation",
                "openhands.sdk.conversation.state",
                "openhands.agent",
                "openhands.tools",
                "litellm",
                "httpx",
                "httpcore",
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
                logger.error("API Key not set for endpoint 'latex_editing'")
                return False
            logger.info(f"TexEditorExecutor: using {self.model} @ {base_url}")

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

            tex_skill = Skill(
                name="tex_editing",
                content=self._build_skill(),
                trigger=None,
            )
            agent_context = AgentContext(skills=[tex_skill])

            self.agent = Agent(
                llm=self.llm,
                tools=[
                    Tool(name=FileEditorTool.name),
                    Tool(name=TerminalTool.name),
                ],
                agent_context=agent_context,
            )

            conv_kwargs: Dict[str, Any] = dict(
                agent=self.agent,
                workspace=str(self.workspace_dir),
                max_iteration_per_run=self.max_iterations,
            )
            if ConciseVisualizer is not None:
                conv_kwargs["visualizer"] = ConciseVisualizer(verbose=False)

            self.conversation = Conversation(**conv_kwargs)

            self._initialized = True
            logger.info(
                f"TexEditorExecutor initialized: workspace={self.workspace_dir}"
            )
            return True

        except ImportError as e:
            logger.error(f"Failed to import OpenHands SDK: {e}")
            return False
        except Exception as e:
            logger.error(f"TexEditorExecutor initialization failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    # ------------------------------------------------------------------
    # Skill description
    # ------------------------------------------------------------------

    @staticmethod
    def _build_skill() -> str:
        return """# LaTeX Editing Skill

## Your Role
You are a professional academic LaTeX editor. Your job is to edit a LaTeX
file in the workspace according to the instructions given in each task.

## Workflow
1. Read the tex file in the workspace using the file_editor tool.
2. Carefully analyse the content according to the task instructions.
3. Make targeted edits using the file_editor tool (str_replace).
4. After editing, verify the file is still valid LaTeX (no broken commands).

## Rules
- Edit the file IN-PLACE. Do NOT create new files.
- Preserve ALL math formulas ($...$, \\[...\\], equation environments).
- Preserve ALL \\cite{}, \\ref{}, \\label{} references.
- Preserve ALL \\begin{figure} ... \\end{figure} blocks unchanged.
- Preserve ALL \\begin{table} ... \\end{table} blocks unchanged.
- Do NOT modify \\documentclass, \\usepackage, or preamble commands.
- Keep the output language as English throughout.
- If no changes are needed, leave the file as-is.
"""

    # ------------------------------------------------------------------
    # Core editing interface
    # ------------------------------------------------------------------

    def run_edit(
        self,
        tex_content: str,
        task_prompt: str,
        tex_filename: str = "main.tex",
    ) -> str:
        """Write *tex_content* to the workspace, let the agent edit it,
        and return the (possibly modified) content.

        Falls back to the original *tex_content* on any failure.
        """
        if not self._initialized:
            logger.error(
                "TexEditorExecutor not initialized. Call initialize() first."
            )
            return tex_content

        tex_path = self.workspace_dir / tex_filename
        tex_path.write_text(tex_content, encoding="utf-8")
        logger.info(
            f"  Wrote {tex_filename} to workspace "
            f"({len(tex_content)} chars)"
        )

        full_prompt = (
            f"Please edit the file `{tex_filename}` in the workspace "
            f"according to the following instructions.\n\n{task_prompt}"
        )

        try:
            self.conversation.send_message(full_prompt)

            for attempt in range(3):
                try:
                    self.conversation.run()
                    break
                except Exception as e:
                    error_str = str(e).lower()
                    if any(
                        kw in error_str
                        for kw in ["rate limit", "timeout", "connection"]
                    ):
                        delay = 5 * (2 ** attempt)
                        logger.warning(
                            f"API error, retrying in {delay}s: {e}"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"Conversation run failed: {e}")
                        break

        except Exception as e:
            logger.error(f"Failed to run tex editing task: {e}")
            return tex_content

        if tex_path.exists():
            edited = tex_path.read_text(encoding="utf-8")
            if edited.strip():
                logger.info(
                    f"  Tex editing complete "
                    f"({len(tex_content)} -> {len(edited)} chars)"
                )
                return edited
            else:
                logger.warning(
                    "  Edited file is empty — falling back to original"
                )
                return tex_content
        else:
            logger.warning(
                f"  {tex_filename} not found after editing — "
                "falling back to original"
            )
            return tex_content

    # ------------------------------------------------------------------
    # Compile-error fix (continues the same conversation)
    # ------------------------------------------------------------------

    def run_fix(
        self,
        compile_log: str,
        tex_filename: str = "main.tex",
    ) -> str:
        """Send compile error log to the agent and ask it to fix the tex.

        Reuses the existing conversation so the agent retains context from
        the previous editing round.  Returns the (possibly fixed) content,
        or the current workspace content unchanged on failure.
        """
        if not self._initialized or self.conversation is None:
            logger.error("TexEditorExecutor not initialized for run_fix.")
            tex_path = self.workspace_dir / tex_filename
            return tex_path.read_text(encoding="utf-8") if tex_path.exists() else ""

        tex_path = self.workspace_dir / tex_filename

        log_excerpt = compile_log[-3000:] if len(compile_log) > 3000 else compile_log

        fix_prompt = (
            "The LaTeX file you edited failed to compile. "
            "Below is the compiler error log (possibly truncated). "
            "Please fix the errors in `{fn}` so that it compiles "
            "successfully.\n\n"
            "**Important rules for fixing**:\n"
            "- Only fix compilation errors (e.g. unmatched braces, "
            "undefined control sequences, missing \\end commands).\n"
            "- Do NOT change the academic content or meaning.\n"
            "- Do NOT remove figures, tables, equations, or references.\n"
            "- If an \\includegraphics path causes the error, keep the "
            "command but make sure the LaTeX syntax around it is correct.\n\n"
            "```\n{log}\n```"
        ).format(fn=tex_filename, log=log_excerpt)

        try:
            self.conversation.send_message(fix_prompt)

            for attempt in range(3):
                try:
                    self.conversation.run()
                    break
                except Exception as e:
                    error_str = str(e).lower()
                    if any(
                        kw in error_str
                        for kw in ["rate limit", "timeout", "connection"]
                    ):
                        delay = 5 * (2 ** attempt)
                        logger.warning(
                            f"API error during fix, retrying in {delay}s: {e}"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"Conversation run (fix) failed: {e}")
                        break

        except Exception as e:
            logger.error(f"Failed to run fix task: {e}")

        if tex_path.exists():
            content = tex_path.read_text(encoding="utf-8")
            if content.strip():
                return content
        return ""

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self):
        """Release resources."""
        self.conversation = None
        self.agent = None
        self.llm = None
        self._initialized = False
