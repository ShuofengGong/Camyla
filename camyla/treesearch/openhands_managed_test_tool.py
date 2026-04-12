#!/usr/bin/env python3
"""
Minimal OpenHands managed test tool.

This tool gives the agent a safe, narrow way to execute `test.py` without using
TerminalTool for Python execution.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import Field

from openhands.sdk.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)

if TYPE_CHECKING:
    from openhands.sdk.conversation import LocalConversation
    from openhands.sdk.conversation.state import ConversationState


DEFAULT_TEST_PYTHON = "/opt/conda/envs/py310/bin/python"


class ManagedTestAction(Action):
    """Run the workspace `test.py` with a controlled timeout."""

    timeout_sec: int = Field(
        default=120,
        ge=1,
        le=3600,
        description="Timeout for running workspace/test.py in seconds.",
    )


class ManagedTestObservation(Observation):
    """Structured observation for a managed test run."""

    passed: bool = Field(description="Whether test.py finished successfully.")
    timed_out: bool = Field(description="Whether the test timed out.")
    return_code: int | None = Field(
        default=None, description="Process return code, if available."
    )
    summary: str = Field(description="Short human-readable test summary.")
    log_path: str | None = Field(
        default=None, description="Absolute path to the saved full log file."
    )
    result_path: str | None = Field(
        default=None, description="Absolute path to the saved JSON result file."
    )


class ManagedTestExecutor(ToolExecutor[ManagedTestAction, ManagedTestObservation]):
    """Executes workspace/test.py in a controlled subprocess group."""

    def __init__(self, workspace_root: str, python_path: str = DEFAULT_TEST_PYTHON):
        self.workspace_root = Path(workspace_root).resolve()
        self.python_path = python_path
        self.results_dir = self.workspace_root / ".managed_test"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._active_pgid: int | None = None

    def __call__(
        self,
        action: ManagedTestAction,
        conversation: "LocalConversation | None" = None,  # noqa: ARG002
    ) -> ManagedTestObservation:
        test_file = self.workspace_root / "test.py"
        if not test_file.exists():
            return self._build_observation(
                passed=False,
                timed_out=False,
                return_code=None,
                summary="test.py not found in workspace; create it before running managed_test.",
                stdout="",
                stderr="",
            )

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("nnUNet_n_proc_DA", "0")
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{self.workspace_root}:{existing_pythonpath}"
            if existing_pythonpath
            else str(self.workspace_root)
        )

        proc = subprocess.Popen(
            [self.python_path, str(test_file)],
            cwd=str(self.workspace_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            preexec_fn=os.setsid,
        )
        self._active_pgid = proc.pid

        timed_out = False
        try:
            stdout, stderr = proc.communicate(timeout=action.timeout_sec)
        except subprocess.TimeoutExpired:
            timed_out = True
            stdout, stderr = self._terminate_process_group(
                proc, grace_sec=3, collect_timeout_sec=5
            )
        finally:
            self._active_pgid = None

        passed = (proc.returncode == 0) and not timed_out
        summary = self._summarize_result(
            passed=passed,
            timed_out=timed_out,
            return_code=proc.returncode,
            stdout=stdout,
            stderr=stderr,
        )
        return self._build_observation(
            passed=passed,
            timed_out=timed_out,
            return_code=proc.returncode,
            summary=summary,
            stdout=stdout,
            stderr=stderr,
        )

    def close(self) -> None:
        if self._active_pgid is None:
            return
        try:
            os.killpg(self._active_pgid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
        finally:
            self._active_pgid = None

    def _terminate_process_group(
        self,
        proc: subprocess.Popen,
        grace_sec: int,
        collect_timeout_sec: int,
    ) -> tuple[str, str]:
        try:
            pgid = os.getpgid(proc.pid)
        except (ProcessLookupError, OSError):
            pgid = None

        if pgid is not None:
            try:
                os.killpg(pgid, signal.SIGTERM)
            except (ProcessLookupError, OSError):
                pass
            time.sleep(grace_sec)
            try:
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass

        try:
            stdout, stderr = proc.communicate(timeout=collect_timeout_sec)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate(timeout=3)

        return stdout, stderr

    def _build_observation(
        self,
        *,
        passed: bool,
        timed_out: bool,
        return_code: int | None,
        summary: str,
        stdout: str,
        stderr: str,
    ) -> ManagedTestObservation:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = self.results_dir / f"managed_test_{timestamp}.log"
        result_path = self.results_dir / "latest_result.json"

        combined = []
        if stdout:
            combined.append("=== STDOUT ===")
            combined.append(stdout.rstrip())
        if stderr:
            combined.append("=== STDERR ===")
            combined.append(stderr.rstrip())
        log_text = "\n".join(part for part in combined if part).strip()
        if log_text:
            log_path.write_text(log_text + "\n", encoding="utf-8")
        else:
            log_path.write_text("(no output)\n", encoding="utf-8")

        result_payload = {
            "passed": passed,
            "timed_out": timed_out,
            "return_code": return_code,
            "summary": summary,
            "log_path": str(log_path),
        }
        result_path.write_text(
            json.dumps(result_payload, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )

        return ManagedTestObservation.from_text(
            text=summary,
            passed=passed,
            timed_out=timed_out,
            return_code=return_code,
            summary=summary,
            log_path=str(log_path),
            result_path=str(result_path),
            is_error=not passed,
        )

    def _summarize_result(
        self,
        *,
        passed: bool,
        timed_out: bool,
        return_code: int | None,
        stdout: str,
        stderr: str,
    ) -> str:
        if passed:
            return "managed_test passed: workspace/test.py exited with code 0."

        if timed_out:
            return "managed_test timed out and the process group was terminated."

        tail_lines: list[str] = []
        combined = "\n".join(part for part in [stdout, stderr] if part).strip()
        if combined:
            tail_lines = combined.splitlines()[-8:]
        tail = "\n".join(tail_lines).strip()
        if tail:
            return (
                f"managed_test failed with return code {return_code}.\n"
                f"Last output lines:\n{tail}"
            )
        return f"managed_test failed with return code {return_code}."


class ManagedTestTool(ToolDefinition[ManagedTestAction, ManagedTestObservation]):
    """OpenHands tool for running the mandatory test safely."""

    @classmethod
    def create(
        cls,
        conv_state: "ConversationState",
    ) -> Sequence["ManagedTestTool"]:
        workspace_root = conv_state.workspace.working_dir
        python_path = os.environ.get("OPENHANDS_MANAGED_TEST_PYTHON", DEFAULT_TEST_PYTHON)
        executor = ManagedTestExecutor(
            workspace_root=workspace_root,
            python_path=python_path,
        )
        description = (
            "Run the workspace `test.py` in a managed subprocess group. "
            "Use this tool instead of TerminalTool for any Python-based verification. "
            "The tool enforces timeout, captures logs, and terminates the whole "
            "process group on timeout."
        )
        return [
            cls(
                description=description,
                action_type=ManagedTestAction,
                observation_type=ManagedTestObservation,
                annotations=ToolAnnotations(
                    title="managed_test",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=False,
                    openWorldHint=False,
                ),
                executor=executor,
            )
        ]


register_tool(ManagedTestTool.name, ManagedTestTool)
