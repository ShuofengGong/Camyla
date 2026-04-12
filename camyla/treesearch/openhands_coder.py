#!/usr/bin/env python3
"""
OpenHands-integrated code generator.

Replaces the legacy AiderCodeGenerator by using the OpenHands SDK for code generation.
"""

import os
import json
import hashlib
import logging
import shutil
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

# Import OpenHands components
from openhands.sdk import LLM, Agent, Conversation, Tool
from openhands.sdk import AgentContext
from openhands.sdk.context import Skill
from openhands.sdk.context.condenser import LLMSummarizingCondenser
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.terminal import TerminalTool
from openhands.tools.task_tracker import TaskTrackerTool

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# from openhands.tools.gemini import GEMINI_FILE_TOOLS

# List of LLM API error types used to decide whether to retry
LLM_API_ERROR_KEYWORDS = [
    "BadRequestError",
    "OpenrouterException",
    "Provider returned error",
    "RateLimitError",
    "APIConnectionError",
    "APITimeoutError",
    "ServiceUnavailable",
    "InternalServerError",
    "AuthenticationError",
    "litellm",
    "openai.error",
    "anthropic.error",
    "Connection reset",
    "Connection refused",
    "timeout",
    "rate limit",
    "quota exceeded",
]


def _is_llm_api_error(error: Exception) -> bool:
    """Return True if the exception is a (transient) LLM API error that is safe to retry.
    
    Args:
        error: caught exception.
        
    Returns:
        bool: True when the error is LLM-API related.
    """
    error_str = str(error).lower()
    error_type = type(error).__name__
    
    # Check the exception type name
    for keyword in LLM_API_ERROR_KEYWORDS:
        if keyword.lower() in error_str or keyword.lower() in error_type.lower():
            return True
    
    # Specifically check for litellm errors
    if hasattr(error, '__module__') and 'litellm' in str(error.__module__):
        return True
    
    return False

logger = logging.getLogger("camyla")


def _get_visualizer_base_class():
    """Dynamically fetch the ConversationVisualizerBase class.
    
    Falls back to `object` if openhands is not installed.
    """
    try:
        from openhands.sdk.conversation.visualizer import ConversationVisualizerBase
        return ConversationVisualizerBase
    except ImportError:
        return object


# Dynamically fetch the base class
_VisualizerBase = _get_visualizer_base_class()


class ConciseVisualizer(_VisualizerBase):
    """Concise visualizer — shows only key operation summaries, never the full prompt.
    
    Shown:
    - Tool calls (file edits, terminal commands, ...)
    - Execution result summaries
    - Error messages
    
    Suppressed:
    - Full LLM prompt/response text
    - Full file contents
    """
    
    def __init__(self, verbose: bool = False):
        """Initialize the concise visualizer.
        
        Args:
            verbose: if True, show additional detail (still without full prompts).
        """
        self.verbose = verbose
        self.iteration_count = 0
        self.tool_call_count = 0
    
    def on_event(self, event) -> None:
        """Process an event and emit a concise summary."""
        event_type = type(event).__name__
        
        # Filter out event types we don't want to show (those containing full prompt/response)
        skip_types = [
            'MessageEvent',  # contains the full prompt
            'LLMResponseEvent',  # contains the full response
            'SystemEvent',
            'UserMessageEvent',  # user message
            'AssistantMessageEvent',  # assistant message
        ]
        
        if event_type in skip_types:
            return
        
        # Iteration-start event
        if event_type == 'IterationStartEvent':
            self.iteration_count += 1
            logger.debug("[OpenHands Iteration %d]", self.iteration_count)
            return
        
        # Tool-call event
        if 'ToolCall' in event_type or event_type == 'ToolCallEvent':
            self.tool_call_count += 1
            tool_name = getattr(event, 'name', getattr(event, 'tool_name', 'unknown'))
            logger.debug("  Tool: %s", tool_name)
            
            if self.verbose and hasattr(event, 'arguments'):
                args = event.arguments
                if isinstance(args, dict):
                    for key, value in args.items():
                        if key in ['command', 'path', 'file_path']:
                            logger.debug("      %s: %s", key, str(value)[:80])
            return
        
        # Tool-result event
        if 'ToolResult' in event_type or event_type == 'ToolResultEvent':
            success = getattr(event, 'success', None)
            if success is True:
                logger.debug("  Tool completed successfully")
            elif success is False:
                error = getattr(event, 'error', '')
                logger.debug("  Tool failed: %s", str(error)[:100])
            return
        
        # File-write event
        if 'FileWrite' in event_type or 'FileEdit' in event_type:
            file_path = getattr(event, 'path', getattr(event, 'file_path', 'unknown'))
            logger.debug("  File modified: %s", file_path)
            return
        
        # Terminal-command event
        if 'Terminal' in event_type or 'Command' in event_type:
            command = getattr(event, 'command', '')
            if command:
                cmd_preview = command[:80] + '...' if len(command) > 80 else command
                logger.debug("  Command: %s", cmd_preview)
            return
        
        # Error event
        if 'Error' in event_type:
            error_msg = getattr(event, 'message', str(event))
            logger.warning("  OpenHands Error: %s", str(error_msg)[:200])
            return
        
        if self.verbose:
            logger.debug("  %s", event_type)


class OpenHandsEventLogger:
    """Records every event in an OpenHands conversation.
    
    Used for debugging and analyzing OpenHands decisions; records the full prompt/response interaction history.
    """
    
    def __init__(self, log_dir: Path, exp_name: str, timestamp: Optional[str] = None):
        """Initialize the event log recorder.
        
        Args:
            log_dir: directory where logs are saved.
            exp_name: experiment name.
        """
        self.log_dir = Path(log_dir)
        self.exp_name = exp_name
        self.events = []
        self.llm_messages = []
        
        # Create the log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Build timestamped log filename
        ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"openhands_events_{ts}.jsonl"
        self.summary_file = self.log_dir / f"openhands_summary_{ts}.md"
        
        # logger.info(f"📊 EventLogger initialized: {self.log_dir}")
    
    def callback(self, event) -> None:
        """Event callback invoked by Conversation.
        
        Args:
            event: OpenHands Event object.
        """
        try:
            # Import Event types lazily so we don't fail when openhands is missing
            try:
                from openhands.sdk import Event, LLMConvertibleEvent
            except ImportError:
                logger.warning("openhands.sdk not available for event logging")
                return
            
            # Record the base info for every event
            event_data = {
                "timestamp": datetime.now().isoformat(),
                "event_type": type(event).__name__,
                "event_str": str(event),
            }
            
            # For LLM-convertible events, record the full message
            if isinstance(event, LLMConvertibleEvent):
                try:
                    llm_msg = event.to_llm_message()
                    self.llm_messages.append(llm_msg)
                    
                    # Extract key fields (supports both dict and Message objects)
                    if isinstance(llm_msg, dict):
                        role = llm_msg.get("role", "unknown")
                        content = str(llm_msg.get("content", ""))
                    else:
                        # Message objects use attribute access
                        role = getattr(llm_msg, "role", "unknown")
                        content = str(getattr(llm_msg, "content", ""))
                    
                    event_data["llm_message"] = {
                        "role": role,
                        "content_preview": content[:500],
                        "content_length": len(content),
                    }
                except Exception as e:
                    logger.debug(f"Could not convert event to LLM message: {e}")
            
            self.events.append(event_data)
            
            # Stream-write JSONL (one event per line)
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event_data, ensure_ascii=False) + "\n")
                
        except Exception as e:
            # Logging failures must not break the main flow
            logger.warning(f"Error logging event: {e}")
    
    def save_summary(self) -> None:
        """Save a human-readable interaction summary to a Markdown file."""
        try:
            with open(self.summary_file, 'w', encoding='utf-8') as f:
                f.write(f"# OpenHands Interaction Log\n\n")
                f.write(f"**Experiment**: `{self.exp_name}`\n\n")
                f.write(f"**Log Directory**: `{self.log_dir}`\n\n")
                f.write(f"**Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"---\n\n")
                f.write(f"## Summary Statistics\n\n")
                f.write(f"- **Total Events**: {len(self.events)}\n")
                f.write(f"- **LLM Messages**: {len(self.llm_messages)}\n\n")
                
                # Tally by event type
                event_types = {}
                for event in self.events:
                    event_type = event.get("event_type", "unknown")
                    event_types[event_type] = event_types.get(event_type, 0) + 1
                
                f.write(f"### Event Type Distribution\n\n")
                for event_type, count in sorted(event_types.items(), key=lambda x: -x[1]):
                    f.write(f"- `{event_type}`: {count}\n")
                f.write("\n---\n\n")
                
                # LLM message history (full prompt/response records)
                f.write("## LLM Message History\n\n")
                f.write("Complete record of all LLM interactions (prompts and responses).\n\n")
                
                for i, msg in enumerate(self.llm_messages):
                    # Supports both dict and Message objects
                    if isinstance(msg, dict):
                        role = msg.get("role", "unknown")
                        content = str(msg.get("content", ""))
                    else:
                        # Message objects use attribute access
                        role = getattr(msg, "role", "unknown")
                        content = str(getattr(msg, "content", ""))
                    
                    f.write(f"### Message {i+1}: {role.upper()}\n\n")
                    
                    # Show full content (capped to avoid huge files)
                    max_length = 100000  # up to 100k characters per message
                    if len(content) <= max_length:
                        f.write("```\n")
                        f.write(content)
                        f.write("\n```\n\n")
                    else:
                        f.write("```\n")
                        f.write(content[:max_length])
                        f.write(f"\n\n... (truncated, total length: {len(content)} chars)\n")
                        f.write("```\n\n")
                    
                    f.write("---\n\n")
            
            # logger.info(f"📄 Saved interaction summary to: {self.summary_file}")
            # print(f"📊 Interaction logs saved to: {self.log_dir}")
            
        except Exception as e:
            logger.warning(f"Error saving summary: {e}")


class OpenHandsFullLogger:
    """Fully record raw OpenHands events and sidecar big-text payloads.

    Design goals:
    1. Do not change the existing openhands_logs format or output.
    2. Write full event objects, full LLM messages, and full observation text into openhands_full.
    3. Keep the main index file at one event per line; real big-text payloads go into sidecar blob files.
    """

    def __init__(self, log_dir: Path, exp_name: str, timestamp: Optional[str] = None):
        self.log_dir = Path(log_dir)
        self.exp_name = exp_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.events_file = self.log_dir / f"openhands_full_events_{ts}.jsonl"
        self.blobs_file = self.log_dir / f"openhands_full_blobs_{ts}.jsonl"
        self.manifest_file = self.log_dir / f"openhands_full_manifest_{ts}.json"

        self._blob_ids_written: set[str] = set()
        self._event_count = 0
        self._blob_count = 0
        self._sdk_persistence_dir: Optional[str] = None
        self._sdk_observations_dir: Optional[str] = None

    @staticmethod
    def _safe_model_dump(obj):
        if obj is None:
            return None

        if hasattr(obj, "model_dump"):
            for kwargs in ({"mode": "json"}, {}):
                try:
                    return obj.model_dump(**kwargs)
                except TypeError:
                    continue
                except Exception:
                    break

        if hasattr(obj, "dict"):
            try:
                return obj.dict()
            except Exception:
                pass

        return str(obj)

    @staticmethod
    def _preview_text(text: Optional[str], max_length: int = 500) -> str:
        if not text:
            return ""
        text = str(text).replace("\n", " \\n ")
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."

    @staticmethod
    def _to_json_text(value: Any) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, sort_keys=True)

    def _write_blob(self, kind: str, value: Any) -> Optional[Dict[str, Any]]:
        if value is None:
            return None

        text = self._to_json_text(value)
        blob_hash = hashlib.sha256(f"{kind}\0{text}".encode("utf-8")).hexdigest()
        blob_id = f"sha256:{blob_hash}"

        if blob_id not in self._blob_ids_written:
            record = {
                "blob_id": blob_id,
                "kind": kind,
                "bytes": len(text.encode("utf-8")),
                "content": text,
            }
            with open(self.blobs_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._blob_ids_written.add(blob_id)
            self._blob_count += 1

        return {
            "blob_id": blob_id,
            "kind": kind,
            "bytes": len(text.encode("utf-8")),
        }

    def set_sdk_persistence_dir(self, persistence_dir: Optional[str]) -> None:
        self._sdk_persistence_dir = persistence_dir
        if persistence_dir:
            self._sdk_observations_dir = str(Path(persistence_dir) / "observations")

    def callback(self, event) -> None:
        try:
            try:
                from openhands.sdk import LLMConvertibleEvent
            except ImportError:
                logger.warning("openhands.sdk not available for full event logging")
                return

            event_payload = self._safe_model_dump(event)

            event_data: Dict[str, Any] = {
                "logged_at": datetime.now().isoformat(),
                "event_type": type(event).__name__,
                "event_str_preview": str(event),
                "event_id": getattr(event, "id", None),
                "event_timestamp": getattr(event, "timestamp", None),
                "source": getattr(event, "source", None),
                "tool_name": getattr(event, "tool_name", None),
                "tool_call_id": getattr(event, "tool_call_id", None),
                "action_id": getattr(event, "action_id", None),
            }

            event_data["event_blob_ref"] = self._write_blob(
                "event_model_dump_json", event_payload
            )

            action = getattr(event, "action", None)
            if action is not None:
                action_payload = self._safe_model_dump(action)
                event_data["action_type"] = type(action).__name__
                event_data["action_inline"] = action_payload
                event_data["action_blob_ref"] = self._write_blob(
                    "action_payload_json", action_payload
                )
                if isinstance(action_payload, dict):
                    command = action_payload.get("command")
                    if command is not None:
                        event_data["terminal_command"] = command
                        event_data["terminal_is_input"] = action_payload.get("is_input")
                        event_data["terminal_timeout"] = action_payload.get("timeout")
                        event_data["terminal_reset"] = action_payload.get("reset")

            tool_call = getattr(event, "tool_call", None)
            if tool_call is not None:
                tool_call_payload = self._safe_model_dump(tool_call)
                event_data["tool_call_inline"] = tool_call_payload
                event_data["tool_call_blob_ref"] = self._write_blob(
                    "tool_call_payload_json", tool_call_payload
                )

            observation = getattr(event, "observation", None)
            if observation is not None:
                observation_payload = self._safe_model_dump(observation)
                event_data["observation_type"] = type(observation).__name__
                event_data["observation_blob_ref"] = self._write_blob(
                    "observation_payload_json", observation_payload
                )

                if isinstance(observation_payload, dict):
                    event_data["observation_meta_inline"] = {
                        "command": observation_payload.get("command"),
                        "exit_code": observation_payload.get("exit_code"),
                        "timeout": observation_payload.get("timeout"),
                        "metadata": observation_payload.get("metadata"),
                    }
                    command = observation_payload.get("command")
                    if command is not None:
                        event_data["terminal_command"] = command
                    exit_code = observation_payload.get("exit_code")
                    if exit_code is not None:
                        event_data["exit_code"] = exit_code
                    metadata = observation_payload.get("metadata")
                    if isinstance(metadata, dict):
                        event_data["working_dir"] = metadata.get("working_dir")
                        event_data["py_interpreter_path"] = metadata.get("py_interpreter_path")

                if hasattr(observation, "text"):
                    observation_text = observation.text
                    event_data["observation_text_ref"] = self._write_blob(
                        "observation_text", observation_text
                    )
                    event_data["observation_text_preview"] = self._preview_text(
                        observation_text
                    )

            if isinstance(event, LLMConvertibleEvent):
                try:
                    llm_msg = event.to_llm_message()
                    llm_payload = self._safe_model_dump(llm_msg)
                    event_data["llm_message_role"] = (
                        llm_payload.get("role")
                        if isinstance(llm_payload, dict)
                        else getattr(llm_msg, "role", None)
                    )
                    event_data["llm_message_blob_ref"] = self._write_blob(
                        "llm_message_payload_json", llm_payload
                    )

                    llm_content = (
                        llm_payload.get("content")
                        if isinstance(llm_payload, dict)
                        else getattr(llm_msg, "content", None)
                    )
                    if llm_content is not None:
                        event_data["llm_content_blob_ref"] = self._write_blob(
                            "llm_message_content_repr", str(llm_content)
                        )
                        event_data["llm_content_preview"] = self._preview_text(
                            str(llm_content)
                        )
                except Exception as e:
                    event_data["llm_message_error"] = str(e)

            with open(self.events_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event_data, ensure_ascii=False) + "\n")
            self._event_count += 1

        except Exception as e:
            logger.warning(f"Error logging full OpenHands event: {e}")

    def finalize(self) -> None:
        manifest = {
            "experiment": self.exp_name,
            "full_log_dir": str(self.log_dir),
            "events_file": str(self.events_file),
            "blobs_file": str(self.blobs_file),
            "event_count": self._event_count,
            "blob_count": self._blob_count,
            "sdk_persistence_dir": self._sdk_persistence_dir,
            "sdk_observations_dir": self._sdk_observations_dir,
            "timestamp": datetime.now().isoformat(),
        }
        with open(self.manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)


class OpenHandsCodeGenerator:
    """OpenHands-integrated code generator that replaces AiderCodeGenerator."""
    
    # Default Python environment (may be overridden by cfg)
    DEFAULT_PYTHON_PATH = "/opt/conda/envs/py310/bin/python"
    DEFAULT_PYTEST_PATH = "/opt/conda/envs/py310/bin/pytest"
    DEFAULT_MAX_ITERATIONS = 30
    DEFAULT_MAX_OUTPUT_TOKENS = 8192  # Cap LLM output tokens to prevent runaway generation
    DEFAULT_LLM_TIMEOUT = 180  # LLM API request timeout (seconds)
    DEFAULT_LLM_TEMPERATURE = 0.5  # LLM temperature

    def __init__(
        self,
        cfg,
        workspace_dir,
        task_desc=None,
        exp_name=None,
        reference_files=None,
    ):
        """Initialize the OpenHands code generator.
        
        Args:
            cfg: config object.
            workspace_dir: working directory.
            task_desc: task description (dataset, baseline, etc.).
            exp_name: unique experiment name.
            reference_files: list of Stage 3 innovation reference files.
        """
        self.cfg = cfg
        self.workspace_dir = Path(workspace_dir)
        self.task_desc = task_desc
        self.exp_name = exp_name
        self.reference_files = reference_files or []
        
        # Read from the experiment.openhands section (nested structure)
        oh = cfg.experiment.openhands
        self.PYTHON_PATH = getattr(oh, 'python_path', self.DEFAULT_PYTHON_PATH)
        self.PYTEST_PATH = getattr(oh, 'pytest_path', self.DEFAULT_PYTEST_PATH)
        self.max_iterations = getattr(oh, 'max_iterations', self.DEFAULT_MAX_ITERATIONS)
        oh_llm = getattr(oh, 'llm', None)
        self.max_output_tokens = getattr(oh_llm, 'max_output_tokens', self.DEFAULT_MAX_OUTPUT_TOKENS) if oh_llm else self.DEFAULT_MAX_OUTPUT_TOKENS
        self.llm_timeout = getattr(oh_llm, 'timeout', self.DEFAULT_LLM_TIMEOUT) if oh_llm else self.DEFAULT_LLM_TIMEOUT
        self.llm_temperature = getattr(oh_llm, 'temperature', self.DEFAULT_LLM_TEMPERATURE) if oh_llm else self.DEFAULT_LLM_TEMPERATURE

        # Context Condenser configuration
        oh_cond = getattr(oh, 'condenser', None)
        self.condenser_enabled = getattr(oh_cond, 'enabled', True) if oh_cond else True
        self.condenser_max_size = getattr(oh_cond, 'max_size', 20) if oh_cond else 20
        self.condenser_keep_first = getattr(oh_cond, 'keep_first', 2) if oh_cond else 2
        
        # OpenHands components
        self.llm = None
        self.agent = None
        self.conversation = None
        self.experiment_file = None
        
        # 🆕 Event logger (lazily initialized)
        self.event_logger = None
        self.full_event_logger = None
        
        # Framework documentation path
        self.framework_doc_path = None
        self.framework_template_path = None
        
        logger.info(f"🎯 OpenHandsCodeGenerator initialized with exp_name: {self.exp_name}")
        # logger.info(f"🐍 Python path: {self.PYTHON_PATH}")
        # logger.info(f"🧪 Pytest path: {self.PYTEST_PATH}")

    def _get_file_hash(self, file_path: Path) -> str:
        """Compute an MD5 hash for a file."""
        if not file_path.exists():
            return ""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def _file_was_modified(self, original_hash: str, current_path: Path) -> bool:
        """Check whether the file has been modified."""
        current_hash = self._get_file_hash(current_path)
        was_modified = original_hash != current_hash
        if was_modified:
            logger.info(f"✅ File was modified (hash changed: {original_hash[:8]}... → {current_hash[:8]}...)")
        else:
            logger.warning(f"⚠️ File was NOT modified (hash unchanged: {original_hash[:8]}...)")
        return was_modified

    def _find_framework_documentation(self) -> Tuple[Optional[Path], Optional[Path]]:
        """Locate the framework documentation and template files."""
        try:
            # Find the Camyla root directory
            camyla_root = os.environ.get('CAMYLA_ROOT')
            if not camyla_root:
                camyla_root = Path(__file__).parent.parent.parent
            
            framework_dir = Path(camyla_root) / "skills" / "frameworks" / "camylanet"
            
            doc_path = framework_dir / "documentation.md"
            template_path = framework_dir / "template.py"
            
            if doc_path.exists():
                self.framework_doc_path = doc_path
                # logger.info(f"📄 Found framework documentation: {doc_path}")
            
            if template_path.exists():
                self.framework_template_path = template_path
                # logger.info(f"📝 Found framework template: {template_path}")
            
            return self.framework_doc_path, self.framework_template_path
        
        except Exception as e:
            logger.warning(f"Error finding framework documentation: {e}")
            return None, None

    def _copy_framework_files_to_workspace(self) -> List[str]:
        """Copy framework documentation into the working directory as read-only references."""
        copied_files = []
        
        # Create the reference-docs directory
        ref_dir = self.workspace_dir / "reference_docs"
        ref_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy documentation files
        if self.framework_doc_path and self.framework_doc_path.exists():
            dest = ref_dir / "camylanet_documentation.md"
            shutil.copy2(self.framework_doc_path, dest)
            copied_files.append(str(dest))
            # logger.info(f"📄 Copied documentation to: {dest}")
        
        # Copy template files
        if self.framework_template_path and self.framework_template_path.exists():
            dest = ref_dir / "camylanet_template.py"
            shutil.copy2(self.framework_template_path, dest)
            copied_files.append(str(dest))
            # logger.info(f"📝 Copied template to: {dest}")
        
        
        return copied_files

    def _build_skill_content(self) -> str:
        """Build framework-skill content (part of the system prompt)."""
        skill_content = []
        
        # Read the framework documentation
        if self.framework_doc_path and self.framework_doc_path.exists():
            doc_content = self.framework_doc_path.read_text()
            skill_content.append("## CamylaNet Framework Documentation\n")
            skill_content.append(doc_content)
            skill_content.append("\n---\n")
        
        # 🔥 Change: do not inject the full template code — use a short API description instead
        skill_content.append("## CamylaNet Code Structure\n")
        skill_content.append("""
**Basic API** (for reference only):
- `camylanet.plan_and_preprocess(dataset_id, configurations)` - Prepare data
- `camylanet.training_network(dataset_id, configuration, plans_identifier, exp_name)` - Train model
- `camylanet.evaluate(dataset_id, result_folder, exp_name)` - Evaluate results

**CRITICAL**: The actual code to modify is in experiment.py, NOT a template.
See framework documentation above for detailed API usage.
""")
        
        return "\n".join(skill_content)

    def _build_execution_control_skill(self) -> str:
        """Build the execution-control skill content — loaded from template via PromptBuilder."""
        try:
            from .prompt_builder import OpenHandsPromptBuilder
            builder = OpenHandsPromptBuilder(self.cfg)
            return builder.build_execution_control_skill()
        except Exception as e:
            logger.warning(f"Failed to load execution control from template: {e}")
            # Minimal fallback
            return f"# Execution Control: Use {self.PYTHON_PATH}, run tests only, NO experiment.py execution"

    def initialize_openhands(self, experiment_file: str = "experiment.py") -> bool:
        """Initialize the OpenHands Agent.
        
        Args:
            experiment_file: name of the experiment file to generate.
            
        Returns:
            bool: whether initialization succeeded.
        """
        try:
            # Suppress noisy OpenHands INFO logs
            import logging as _logging
            for module_name in [
                'openhands',
                'openhands.sdk',
                'openhands.sdk.conversation',
                'openhands.sdk.conversation.state',
                'openhands.agent',
                'openhands.tools',
                'litellm',
                'httpx',
                'httpcore',
            ]:
                _logging.getLogger(module_name).setLevel(_logging.WARNING)
            
            # terminal_session's PS1-metadata WARNING dumps the entire terminal content — extremely noisy
            for module_name in [
                'openhands.tools.terminal.terminal.terminal_session',
                'openhands.tools.terminal',
            ]:
                _logging.getLogger(module_name).setLevel(_logging.ERROR)
            
            # Locate framework documentation
            self._find_framework_documentation()
            
            # Copy files into the working directory
            self._copy_framework_files_to_workspace()
            
            # 1. Configure the LLM — read from experiment.code.candidates[0]
            from camyla.model_config import get_endpoint
            candidates = list(getattr(self.cfg.experiment.code, 'candidates', []) or [])
            if not candidates:
                logger.error("No candidates configured in experiment.code.candidates")
                return False
            endpoint_name = candidates[0]
            ep = get_endpoint(endpoint_name)
            api_key = ep["api_key"]
            base_url = ep["base_url"]
            model_name = ep["model"]
            llm_model_string = model_name

            if not api_key:
                logger.error(f"API Key not found for endpoint '{endpoint_name}'")
                return False

            logger.info(f"⚡ Using endpoint '{endpoint_name}': {model_name} @ {base_url}")
            
            self.llm = LLM(
                model=llm_model_string,
                api_key=api_key,
                base_url=base_url,
                max_output_tokens=self.max_output_tokens,
                timeout=self.llm_timeout,
                temperature=self.llm_temperature,
                num_retries=10,
                retry_multiplier=8.0,
                retry_min_wait=8,
                retry_max_wait=120,
            )
            logger.info(f"🔢 Max output tokens: {self.max_output_tokens:,}")
            logger.info(f"⏱️ LLM timeout: {self.llm_timeout}s")
            logger.info(f"🌡️ LLM temperature: {self.llm_temperature}")
            logger.info(f"🤖 LLM configured: {llm_model_string}")
            
            # 2. Build the framework skill
            framework_skill_content = self._build_skill_content()
            
            skills = []
            if framework_skill_content:
                skills.append(Skill(
                    name="camylanet_framework",
                    content=framework_skill_content,
                    trigger=None,  # always active
                ))
                logger.info("📚 Added CamylaNet framework skill")
            
            # Add the execution-control skill
            execution_control_content = self._build_execution_control_skill()
            skills.append(Skill(
                name="execution_control",
                content=execution_control_content,
                trigger=None,  # always active
            ))
            logger.info("🔒 Added execution control skill")
            
            # Add Code Generation Guidelines Skill (New Optimization)
            # Load the new skill via prompt_builder
            from .prompt_builder import OpenHandsPromptBuilder
            builder = OpenHandsPromptBuilder(self.cfg)
            guidelines_content = builder.build_code_generation_guidelines_skill()
            
            if guidelines_content:
                skills.append(Skill(
                    name="code_generation_guidelines",
                    content=guidelines_content,
                    trigger=None,  # Always active
                ))
                logger.info("📜 Added Code Generation Guidelines skill")
            
            # Create the AgentContext
            agent_context = AgentContext(skills=skills) if skills else None
            
            # 3. Configure the Context Condenser
            condenser = None
            if self.condenser_enabled:
                condenser = LLMSummarizingCondenser(
                    llm=self.llm.model_copy(update={"usage_id": "condenser"}),
                    max_size=self.condenser_max_size,
                    keep_first=self.condenser_keep_first,
                )
                logger.info(f"📦 Context Condenser enabled: max_size={self.condenser_max_size}, keep_first={self.condenser_keep_first}")
            
            # 4. Configure Agent tools
            self.agent = Agent(
                llm=self.llm,
                tools=[
                    # *GEMINI_FILE_TOOLS,
                    Tool(name=FileEditorTool.name),
                    Tool(name=TerminalTool.name),
                    Tool(name=TaskTrackerTool.name),
                ],
                agent_context=agent_context,
                condenser=condenser,  # 🆕 add the context condenser
            )
            logger.info("🔧 Agent configured with FileEditor, Terminal, TaskTracker tools")
            
            # 4. Create the working directory
            self.workspace_dir.mkdir(parents=True, exist_ok=True)
            
            # 🆕 4.5. Create the event logger.
            # 🔧 openhands_logs sits alongside workspace_dir (openhands_workspace) so cleanup does not delete it.
            # Layout: stage_xxx/openhands_workspace/ and stage_xxx/openhands_logs/
            log_dir = self.workspace_dir.parent / "openhands_logs"
            full_log_dir = self.workspace_dir.parent / "openhands_full"
            log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.event_logger = OpenHandsEventLogger(
                log_dir=log_dir,
                exp_name=self.exp_name or "unknown_exp",
                timestamp=log_timestamp,
            )
            logger.info(f"📊 Event logger initialized: {log_dir}")
            self.full_event_logger = OpenHandsFullLogger(
                log_dir=full_log_dir,
                exp_name=self.exp_name or "unknown_exp",
                timestamp=log_timestamp,
            )
            logger.info(f"🗂️ Full event logger initialized: {full_log_dir}")
            
            # 5. Create the conversation (with event callback + concise visualizer)
            persistence_base_dir = full_log_dir / "sdk_persistence"
            self.conversation = Conversation(
                agent=self.agent,
                workspace=str(self.workspace_dir),
                persistence_dir=str(persistence_base_dir),
                max_iteration_per_run=self.max_iterations,
                visualizer=ConciseVisualizer(verbose=False),  # concise mode
                callbacks=[
                    self.event_logger.callback,
                    self.full_event_logger.callback,
                ],
            )
            logger.info(f"💬 Conversation created with event logging enabled")
            if self.full_event_logger:
                try:
                    self.full_event_logger.set_sdk_persistence_dir(
                        getattr(self.conversation.state, "persistence_dir", None)
                    )
                except Exception as e:
                    logger.warning(f"Failed to bind sdk persistence dir for full logger: {e}")
            
            # 6. Set the experiment file path
            self.experiment_file = self.workspace_dir / experiment_file
            
            # logger.info(f"✅ OpenHands initialized successfully")
            # logger.info(f"📁 Workspace: {self.workspace_dir}")
            logger.info(f"📄 Experiment file: {self.experiment_file}")
            
            return True
            
        except ImportError as e:
            logger.error(f"Failed to import OpenHands: {e}")
            logger.error("Please install openhands-sdk: pip install openhands-sdk")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize OpenHands: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _extract_dataset_info(self, prompt_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract dataset info from the prompt dictionary."""
        dataset_info = {}
        
        try:
            import json
            # Handle the various task_desc types
            if isinstance(self.task_desc, dict):
                task_data = self.task_desc
            elif isinstance(self.task_desc, str):
                task_data = json.loads(self.task_desc)
            else:
                return None
            
            if 'dataset' in task_data:
                dataset = task_data['dataset']
                dataset_info['dataset_id'] = dataset.get('dataset_id')
                dataset_info['configuration'] = dataset.get('configuration', '3d_fullres')
                dataset_info['dataset_name'] = dataset.get('name', f"Dataset {dataset_info['dataset_id']}")
                dataset_info['task_description'] = dataset.get('description', 'Medical segmentation task')
                dataset_info['target_structure'] = dataset.get('target_structure', 'unknown structure')
                dataset_info['modality'] = dataset.get('modality', 'unknown modality')
                dataset_info['patch_size'] = dataset.get('patch_size', None)
                
                if dataset_info.get('dataset_id'):
                    return dataset_info
        except Exception as e:
            logger.warning(f"Error extracting dataset info: {e}")
        
        return None

    def _build_openhands_prompt(
        self,
        prompt_dict: Dict[str, Any],
        error_feedback: Optional[str] = None,
        existing_code: Optional[str] = None,
    ) -> str:
        """Build the OpenHands prompt.
        
        Args:
            prompt_dict: structured prompt dict.
            error_feedback: error feedback from the previous run.
            existing_code: existing code (for incremental improvement).
            
        Returns:
            str: the full prompt.
        """
        prompt_parts = []
        
        # Task description
        prompt_parts.append("# Camyla Experiment Code Generation Task")
        prompt_parts.append("")
        
        # 🆕 Workspace constraints (concise version)
        prompt_parts.append(f"**⚠️ File Operation Restriction**: All file edits MUST be within workspace `{self.workspace_dir}`. DO NOT modify files outside workspace. Import libraries normally, never edit their source.")
        prompt_parts.append("")
        
        if "Introduction" in prompt_dict:
            prompt_parts.append("## Task Description")
            prompt_parts.append(prompt_dict["Introduction"])
            prompt_parts.append("")
        
        # Research idea
        if "Research idea" in prompt_dict:
            prompt_parts.append("## Research Idea")
            prompt_parts.append(prompt_dict["Research idea"])
            prompt_parts.append("")
        
        # Stage 2: innovation implementation
        if "Innovation to Implement" in prompt_dict:
            innovation_text = prompt_dict["Innovation to Implement"]
            prompt_parts.append("## 🚀 SPECIFIC INNOVATION TO IMPLEMENT")
            prompt_parts.append("**CRITICAL**: You must implement the following specific innovation:")
            prompt_parts.append("")
            prompt_parts.append(innovation_text)
            prompt_parts.append("")
            prompt_parts.append("### Implementation Guidelines")
            prompt_parts.append("**Core Requirement**: Implement the fundamental concept and mechanism of this innovation.")
            prompt_parts.append("")
            prompt_parts.append("**Implementation Flexibility**: While maintaining the innovation's core idea, you may adapt specific details:")
            prompt_parts.append("- ✅ Adjust layer dimensions and channel sizes for compatibility")
            prompt_parts.append("- ✅ Modify hyperparameters to work with the framework")
            prompt_parts.append("- ✅ Fine-tune architectural details to fix bugs")
            prompt_parts.append("- ❌ Do NOT change the fundamental concept or mechanism")
            prompt_parts.append("- ❌ Do NOT remove or skip core innovation modules")
            prompt_parts.append("")
        
        # Subagent competition notice
        if "Competition Notice" in prompt_dict:
            prompt_parts.append("## 🏆 Competition Notice")
            prompt_parts.append(prompt_dict["Competition Notice"])
            prompt_parts.append("")

        # Dataset configuration
        dataset_info = self._extract_dataset_info(prompt_dict)
        if dataset_info:
            prompt_parts.append("## 🚨 CRITICAL Dataset Configuration")
            prompt_parts.append("")
            prompt_parts.append("**MUST USE THE FOLLOWING DATASET CONFIGURATION:**")
            prompt_parts.append("```python")
            prompt_parts.append(f"dataset_id = {dataset_info['dataset_id']}  # {dataset_info['dataset_name']}")
            prompt_parts.append(f"configuration = '{dataset_info['configuration']}'")
            prompt_parts.append("```")
            prompt_parts.append("")
            prompt_parts.append("**⚠️ CRITICAL REQUIREMENTS:**")
            prompt_parts.append(f"- You MUST use dataset_id = {dataset_info['dataset_id']}")
            prompt_parts.append(f"- You MUST use configuration = '{dataset_info['configuration']}'")
            prompt_parts.append(f"- Target structure: {dataset_info.get('target_structure', 'medical structure')}")
            prompt_parts.append(f"- Modality: {dataset_info.get('modality', 'medical imaging')}")
            if dataset_info.get('patch_size'):
                prompt_parts.append(f"- Patch Size: {dataset_info['patch_size']}")
            prompt_parts.append("")
        
        # Experiment name
        if self.exp_name:
            prompt_parts.append("## Experiment Organization")
            prompt_parts.append("")
            prompt_parts.append(f"**CRITICAL**: Use the following unique experiment name for ALL camylanet operations:")
            prompt_parts.append("```python")
            prompt_parts.append(f"exp_name = '{self.exp_name}'")
            prompt_parts.append("```")
            prompt_parts.append("")
        
        # 📂 Target File note — added for every mode
        prompt_parts.append("## 📂 Target File")
        prompt_parts.append("")
        
        if existing_code:
            # Debug/Improve mode — modify existing code
            prompt_parts.append(f"**Mode**: MODIFY existing `{self.workspace_dir}/experiment.py`")
        else:
            # Draft mode — create from scratch
            prompt_parts.append(f"**Mode**: CREATE new `{self.workspace_dir}/experiment.py`")
        prompt_parts.append("")
        
        # Error feedback
        if error_feedback:
            prompt_parts.append("## 🐛 Previous Execution Error")
            prompt_parts.append("The previous code failed with the following error. Please fix this issue:")
            prompt_parts.append("```")
            prompt_parts.append(error_feedback)
            prompt_parts.append("```")
            prompt_parts.append("")
        
        # Diagnostic information (from proposal diagnostic module)
        if "Diagnostic Information" in prompt_dict:
            diag = prompt_dict["Diagnostic Information"]
            prompt_parts.append("## 🔍 Diagnostic Information")
            prompt_parts.append("")
            if isinstance(diag, dict):
                for dk, dv in diag.items():
                    if isinstance(dv, list):
                        prompt_parts.append(f"**{dk}**:")
                        for item in dv:
                            prompt_parts.append(f"- {item}")
                    else:
                        prompt_parts.append(f"**{dk}**: {dv}")
                prompt_parts.append("")
            else:
                prompt_parts.append(str(diag))
                prompt_parts.append("")

        # Implementation instructions
        if "Instructions" in prompt_dict:
            prompt_parts.append("## Implementation Instructions")
            instructions = prompt_dict["Instructions"]
            if isinstance(instructions, dict):
                for key, value in instructions.items():
                    prompt_parts.append(f"### {key}")
                    if isinstance(value, list):
                        for item in value:
                            prompt_parts.append(f"- {item}")
                    else:
                        prompt_parts.append(str(value))
                    prompt_parts.append("")
            elif isinstance(instructions, list):
                for item in instructions:
                    prompt_parts.append(f"- {item}")
                prompt_parts.append("")
            else:
                prompt_parts.append(str(instructions))
                prompt_parts.append("")
        
        # Performance optimization hints (after beating baseline)
        if "Performance Optimization Hints" in prompt_dict:
            prompt_parts.append("## 📈 Performance Optimization Hints")
            prompt_parts.append(prompt_dict["Performance Optimization Hints"])
            prompt_parts.append("")

        # Experiment history (memory from previous iterations in this stage)
        if "Memory" in prompt_dict and prompt_dict["Memory"]:
            prompt_parts.append("## 📜 Experiment History (Memory)")
            prompt_parts.append(
                "Below is a summary of previous experiments in this stage. "
                "Use this to AVOID repeating failed approaches and build on successful ones."
            )
            prompt_parts.append("")
            prompt_parts.append(prompt_dict["Memory"])
            prompt_parts.append("")

        # Add guidance hints
        prompt_parts.append("## 📝 Guidelines")
        prompt_parts.append("Please refer to the 'Code Generation Guidelines' in the System Prompt for:")
        prompt_parts.append("- File Operation Rules")
        prompt_parts.append("- Main Guard Requirements")
        prompt_parts.append("- **MANDATORY 1-Epoch Integration Test** (you MUST pass `camylanet.training_network_1epoch()` before finishing)")
        prompt_parts.append("- Execution Control Restrictions")
        prompt_parts.append("")
        prompt_parts.append("**⚠️ IMPORTANT**: Both `experiment.py` AND `test.py` MUST have `if __name__ == '__main__':` guard.")
        prompt_parts.append("If the 1-epoch test fails with 'multiprocessing spawn' error, fix the `__main__` guard in `test.py` — do NOT skip the test or replace it with a forward-pass-only test.")
        prompt_parts.append("")
        
        prompt_parts.append("**Your task is to WRITE the code. Full experiment runs separately.**")
        prompt_parts.append("")
        
        # Add task-tracking instructions — simplified to avoid overuse
        prompt_parts.append("## 📋 Task Progress Tracking")
        prompt_parts.append("")
        prompt_parts.append("You should use `task_tracker` to organize your work, but **prioritize writing code over updating tasks**.")
        prompt_parts.append("")
        prompt_parts.append("Do NOT use `task_tracker` too frequently")
        prompt_parts.append("")
        
        return "\n".join(prompt_parts)

    def _build_followup_prompt(self, attempt_num: int) -> str:
        """Build a follow-up prompt to nudge the agent to keep working when no file was modified.
        
        Args:
            attempt_num: current attempt number (starting at 1).
            
        Returns:
            Follow-up message string.
        """
        prompt = f"""
## ⚠️ IMPORTANT: File Was NOT Modified

Your previous operation did not successfully modify the `experiment.py` file. This might be because:
1. A tool call failed or returned an error
2. You chose to give up instead of continuing to try
3. The file editing operation did not execute correctly

**You MUST continue working**. Please:
1. Check the error messages above (if any)
2. Use a different approach or fix the previous error
3. **MUST** call the tool to modify `experiment.py`
4. **DO NOT** just reply with an explanatory message — you must take action

This is attempt {attempt_num + 1}. Please start modifying the file immediately.
"""
        return prompt
    
    def _run_with_verification(self, original_hash: str, max_followups: int = 3, max_llm_retries: int = 4) -> bool:
        """Wrapper around conversation.run() with inline verification.
        
        If run() finishes without modifying the file, send a follow-up message in the same session
        so the agent can self-correct using the in-context error information.
        
        For LLM API errors (e.g. transient OpenRouter/LiteLLM failures), use exponential-backoff retries.
        
        Args:
            original_hash: original file MD5 hash.
            max_followups: max number of follow-ups.
            max_llm_retries: max retries for LLM API errors.
            
        Returns:
            bool: whether the file was successfully modified.
        """
        logger.info("🔄 Running OpenHands conversation with internal verification...")
        
        for attempt in range(max_followups + 1):  # +1 because the first run does not count as a follow-up
            # conversation.run() wrapped with LLM API error retries
            run_succeeded = self._run_conversation_with_llm_retry(max_llm_retries)
            
            if not run_succeeded:
                logger.warning(f"⚠️ Conversation run failed after {max_llm_retries} LLM retries on attempt {attempt + 1}")
                # Even if run failed, check whether the file was modified (may have changed before the exception)
            
            # Check whether the file was modified
            if self._file_was_modified(original_hash, self.experiment_file):
                logger.info(f"✅ File modified successfully on attempt {attempt + 1}")
                return True
            
            # If this is the final attempt, do not follow up again
            if attempt >= max_followups:
                logger.warning(f"❌ File not modified after {max_followups + 1} attempts (1 initial + {max_followups} followups)")
                return False
            
            # Send the follow-up message (staying in the same session)
            followup_message = self._build_followup_prompt(attempt + 1)
            logger.info(f"🔄 File not modified, sending followup (attempt {attempt + 2}/{max_followups + 1})")
            
            # Sending the follow-up also needs LLM retry protection
            try:
                self.conversation.send_message(followup_message)
            except Exception as e:
                if _is_llm_api_error(e):
                    logger.warning(f"⚠️ LLM API error while sending followup: {e}")
                    # Wait and retry
                    time.sleep(5)
                    try:
                        self.conversation.send_message(followup_message)
                    except Exception as retry_e:
                        logger.error(f"❌ Failed to send followup after retry: {retry_e}")
                else:
                    logger.error(f"❌ Non-LLM error while sending followup: {e}")
        
        return False
    
    def _run_conversation_with_llm_retry(self, max_retries: int = 4) -> bool:
        """conversation.run() with LLM API error retries.
        
        Uses exponential backoff for LLM API errors.
        
        Args:
            max_retries: max retry count.
            
        Returns:
            bool: whether conversation.run() completed successfully.
        """
        base_delay = 10  # base delay (seconds)
        
        for retry_num in range(max_retries + 1):
            try:
                self.conversation.run()
                return True  # completed successfully
                
            except Exception as e:
                error_type = type(e).__name__
                
                # Decide whether this is an LLM API error
                if _is_llm_api_error(e):
                    if retry_num < max_retries:
                        # Exponential backoff delays: 5s, 10s, 20s, 40s
                        delay = base_delay * (2 ** retry_num)
                        logger.warning(
                            f"🔄 LLM API error ({error_type}): {str(e)[:200]}... "
                            f"Retrying in {delay}s (attempt {retry_num + 1}/{max_retries})"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"❌ LLM API error ({error_type}) after {max_retries} retries: {e}"
                        )
                        return False
                else:
                    logger.error(f"❌ Non-LLM error in conversation.run(): {error_type}: {e}")
                    return False
        
        return False

    def generate_code_with_openhands(
        self,
        prompt_dict: Dict[str, Any],
        error_feedback: Optional[str] = None,
        existing_code: Optional[str] = None,
        is_metrics_parsing: bool = False,
    ) -> Tuple[Optional[str], Optional[str], bool]:
        """Generate code using OpenHands.
        
        Args:
            prompt_dict: structured prompt dict.
            error_feedback: error feedback from the previous run.
            existing_code: existing code (for incremental improvement).
            is_metrics_parsing: whether running in metric-parsing mode.
            
        Returns:
            Tuple[plan, code, file_was_modified]:
                - plan: generated plan description
                - code: generated code
                - file_was_modified: whether the file was modified
        """
        if not self.conversation:
            logger.error("OpenHands not initialized")
            return None, None, False
        
        try:
            # Save the original file hash
            original_hash = self._get_file_hash(self.experiment_file)
            logger.info(f"📸 Original file hash: {original_hash[:16] if original_hash else 'empty'}...")
            
            # If existing code is provided, write it to disk as the starting point
            if existing_code:
                logger.info(f"📝 Writing existing code to experiment.py ({len(existing_code)} chars)")
                self.experiment_file.write_text(existing_code)
                original_hash = self._get_file_hash(self.experiment_file)
            
            # 🆕 test.py lifecycle management: delete the file so the agent can 'create' it normally.
            # In tree search, nodes are not linearly related; test.py must match the current code.
            # Debug/Improve may come from different parent nodes and cannot share test.py.
            # Use deletion instead of placeholder overwrite to avoid agent 'create' command failures.
            test_file = self.workspace_dir / "test.py"
            if test_file.exists():
                test_file.unlink()
                logger.info(f"🧹 Deleted test.py to ensure fresh creation (tree search)")
            else:
                logger.info(f"📝 test.py does not exist, Agent will create it")
            
            # Build the prompt
            prompt = self._build_openhands_prompt(prompt_dict, error_feedback, existing_code)
            
            # Print the token estimate
            estimated_tokens = len(prompt) // 4
            logger.info(f"🔢 Estimated prompt tokens: {estimated_tokens:,}")
            
            # Send the message
            logger.info("📤 Sending message to OpenHands...")
            self.conversation.send_message(prompt)
            
            # Run (with internal verification follow-ups).
            # If the file is not modified after the first run, send a follow-up in the same session.
            file_was_modified = self._run_with_verification(original_hash)
            
            # 🆕 Save the interaction summary (full prompt/response history)
            if self.event_logger:
                self.event_logger.save_summary()
                logger.debug(f"📊 Interaction logs saved to: {self.event_logger.log_dir}")
            if self.full_event_logger:
                self.full_event_logger.finalize()
                logger.debug(f"🗂️ Full interaction logs saved to: {self.full_event_logger.log_dir}")
            
            # Read the generated code
            if self.experiment_file.exists():
                generated_code = self.experiment_file.read_text()
                logger.info(f"📄 Generated code length: {len(generated_code)} characters")
                
                if not file_was_modified:
                    logger.warning("⚠️ OpenHands completed but did NOT modify the file")
                    return None, generated_code, False
                
                # 🆕 Check whether test.py exists
                test_file = self.workspace_dir / "test.py"
                if not test_file.exists():
                    logger.warning("⚠️ test.py not found - OpenHands should create unit tests")
                    # Optionally return False to force regeneration, or continue
                    # return None, generated_code, False
                else:
                    logger.info(f"✅ test.py found ({test_file.stat().st_size} bytes)")
                
                # Extract plan description
                plan = self._extract_plan()
                
                logger.info("✅ Code generated successfully")
                return plan, generated_code, True
            else:
                logger.error("Experiment file not found after OpenHands execution")
                return None, None, False
                
        except Exception as e:
            logger.error(f"Error generating code with OpenHands: {e}")
            import traceback
            traceback.print_exc()
            return None, None, False

    def _extract_plan(self) -> str:
        """Extract a plan description from the conversation history."""
        try:
            # Try to extract a meaningful plan description from the conversation events
            if self.conversation and hasattr(self.conversation, 'state'):
                events = self.conversation.state.events
                if events:
                    # Look for the agent's first response
                    for event in events:
                        if hasattr(event, 'content') and event.content:
                            content = str(event.content)
                            if len(content) > 50:
                                return content[:200] + "..."
        except Exception as e:
            logger.debug(f"Error extracting plan: {e}")
        
        return "Implemented experiment using OpenHands code generation."

    def generate_code_with_error_feedback(
        self,
        prompt_dict: Dict[str, Any],
        error_message: str,
    ) -> Tuple[Optional[str], Optional[str], bool]:
        """Generate fix code based on error feedback.
        
        Args:
            prompt_dict: structured prompt dict.
            error_message: error message.
            
        Returns:
            Tuple[plan, code, file_was_modified]
        """
        logger.info("🔧 Generating code with error feedback...")
        return self.generate_code_with_openhands(prompt_dict, error_feedback=error_message)

    def cleanup(self):
        """Release resources."""
        try:
            logger.info("🧹 Cleaning up OpenHands resources...")
            
            # 🆕 Clean junk files in the workspace, keeping only experiment.py, test.py, reference_docs/
            if hasattr(self, 'workspace_dir') and self.workspace_dir and self.workspace_dir.exists():
                keep_files = {"experiment.py", "test.py"}
                keep_dirs = {"reference_docs"}
                
                for item in self.workspace_dir.iterdir():
                    try:
                        if item.is_file() and item.name not in keep_files:
                            item.unlink()
                            logger.info(f"🗑️ Removed file: {item.name}")
                        elif item.is_dir() and item.name not in keep_dirs:
                            shutil.rmtree(item)
                            logger.info(f"🗑️ Removed directory: {item.name}")
                    except Exception as e:
                        logger.warning(f"Failed to remove {item}: {e}")
            
            if self.conversation is not None:
                try:
                    self.conversation.close()
                except Exception as e:
                    logger.warning(f"Error closing conversation: {e}")
            if self.full_event_logger is not None:
                try:
                    self.full_event_logger.finalize()
                except Exception as e:
                    logger.warning(f"Error finalizing full event logger: {e}")
            self.conversation = None
            self.agent = None
            self.llm = None
            self.full_event_logger = None
            logger.info("✅ Cleanup complete")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

    def __del__(self):
        """Destructor."""
        self.cleanup()
