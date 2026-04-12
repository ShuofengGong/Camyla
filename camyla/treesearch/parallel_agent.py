from typing import List, Optional, Any, Callable, cast, Dict, Tuple
import random
import subprocess
import os
import logging
import humanize
from .backend import FunctionSpec, compile_prompt_to_md, query
from .interpreter import ExecutionResult
from .journal import Journal, Node
from .utils.config import Config
from .utils.metric import MetricValue, WorstMetricValue
from .utils.response import wrap_code
import copy
import re
from dataclasses import asdict
from omegaconf import OmegaConf


from pathlib import Path
import base64
import sys
import tempfile
import shutil
import time
import tiktoken
from datetime import datetime

from .openhands_coder import OpenHandsCodeGenerator
from .innovation_generator import InnovationGenerator

from camyla.model_config import get_role, get_endpoint

logger = logging.getLogger("camyla")


review_func_spec = FunctionSpec(
    name="submit_review",
    json_schema={
        "type": "object",
        "properties": {
            "is_bug": {
                "type": "boolean",
                "description": "true if the output log shows that the execution failed, crashed, threw exceptions, or has actual bugs. Do NOT flag negative loss values, negative validation losses, or negative metrics as bugs - these can be legitimate depending on the loss function used.",
            },
            "summary": {
                "type": "string",
                "description": "Provide a concise summary of the execution results. Include key metrics, training behavior, and any notable observations. If there were errors or bugs, describe them and propose a fix.",
            },
        },
        "required": [
            "is_bug",
            "summary",
        ],
    },
    description="Submit a review evaluating the output of the training script. Always provide a summary of execution results including metrics and observations. Do not flag negative metric values as bugs.",
)

class AblationPlanItem:
    """A single item in the Stage 3 ablation plan.

    ablation_type:
        "removal"    – remove / disable one or more components (1 training run)
        "comparison" – run up to 5 scientifically meaningful variants in one script
    """

    def __init__(
        self,
        name: str,
        description: str,
        ablation_type: str,
        variants: Optional[List[str]] = None,
    ):
        self.name = name
        self.description = description
        self.ablation_type = ablation_type
        self.variants = variants or []

    def __repr__(self):
        return f"AblationPlanItem(name={self.name!r}, type={self.ablation_type!r}, variants={len(self.variants)})"


class MinimalAgent:
    """A minimal agent class that only contains what's needed for processing nodes"""

    def __init__(
        self,
        task_desc,
        cfg,
        memory_summary=None,
        evaluation_metrics=None,
        stage=None,
        stage_name=None,
        agent_manager=None,
        exp_name=None,
    ):
        self.task_desc = task_desc
        self.memory_summary = memory_summary
        self.cfg = cfg
        self.evaluation_metrics = evaluation_metrics
        self.stage_name = stage_name
        self.current_stage = stage  # Store the Stage object so proposal_idx is accessible
        self.data_preview = None
        self.exp_name = exp_name  # Store exp_name for aider code generation

        # Store reference to agent_manager for accessing innovation queue
        self.agent_manager = agent_manager
        
        logger.info(f"🎯 MinimalAgent initialized with exp_name: {self.exp_name}")

        # Detect if we're using the new structure
        try:
            import json
            logger.debug(f"MinimalAgent Init - task_desc type: {type(task_desc)}")
            logger.debug(f"MinimalAgent Init - task_desc preview: {str(task_desc)[:200]}...")

            task_dict = json.loads(task_desc) if isinstance(task_desc, str) else task_desc
            self._is_new_structure = True

            logger.debug(f"MinimalAgent Init - _is_new_structure: {self._is_new_structure}")

            # Store the original task dict for dataset and baseline info (NOT for innovations)
            if self._is_new_structure:
                self._task_dict = task_dict
                logger.debug("MinimalAgent Init - stored task_dict for dataset/baseline info")
                logger.info(f"🔍 New structure detected, stored task_dict for dataset/baseline info")
            else:
                self._task_dict = None
                logger.debug("MinimalAgent Init - using old structure")

            # Log innovation queue info from agent_manager
            if self.agent_manager and hasattr(self.agent_manager, 'proposals'):
                innovations_count = len(self.agent_manager.proposals)
                logger.debug(f"MinimalAgent Init - agent_manager has {innovations_count} innovations")
                logger.info(f"🔍 Agent manager has {innovations_count} innovations available")
            else:
                logger.debug("MinimalAgent Init - no agent_manager or proposals available")
        except Exception as e:
            logger.debug(f"MinimalAgent Init - Error detecting structure: {e}")
            logger.warning(f"Error detecting structure: {e}")
            self._is_new_structure = False
            self._task_dict = None

        # Initialize InnovationGenerator (lazy initialization)
        self._innovation_generator = None
        logger.info(f"🧠 MinimalAgent ready for InnovationGenerator initialization")

    @property
    def _prompt_environment(self):
        pkgs = [
            "numpy",
            "pandas",
            "scikit-learn",
            "seaborn",
            "SimpleITK",
            "medpy",
            "torch",
            "torchvision",
            "transformers",
            "nibabel",
            "timm",
            "monai",
            "mamba_ssm",
            "einops",
        ]
        random.shuffle(pkgs)
        pkg_str = ", ".join([f"`{p}`" for p in pkgs])

        transformer_usage = (
            "For **transformer blocks** (ViT-style), you can use MONAI: "
            "`from monai.networks.blocks.transformerblock import TransformerBlock` and "
            "`from monai.networks.blocks.selfattention import SABlock`, "
            "`from monai.networks.blocks.mlp import MLPBlock`. "
            "TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate) expects input shape (B, seq_len, hidden_size); "
            "SABlock(hidden_size, num_heads, dropout_rate, qkv_bias) is the same. "
            "Prefer these instead of hand-rolling attention to avoid shape/dtype bugs."
        )
        mamba_usage = (
            "For **Mamba / SSM** modules, use `mamba_ssm`: "
            "`from mamba_ssm.modules.mamba_simple import Mamba`. "
            "Mamba(d_model, d_state=16, d_conv=4, expand=2) expects input (B, L, d_model). "
            "Use this instead of custom selective-scan code when the proposal mentions state-space or Mamba."
        )

        # Reference implementations (MONAI-style) for the agent to adapt in its own code
        sablock_impl = r'''
SABlock (self-attention, ViT-style). Input/output shape: (B, seq_len, hidden_size).
```python
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

class SABlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout_rate: float = 0.0, qkv_bias: bool = False, save_attn: bool = False):
        super().__init__()
        assert 0 <= dropout_rate <= 1 and hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.input_rearrange = Rearrange("b h (qkv l d) -> qkv b l h d", qkv=3, l=num_heads)
        self.out_rearrange = Rearrange("b h l d -> b l (h d)")
        self.drop_output = nn.Dropout(dropout_rate)
        self.drop_weights = nn.Dropout(dropout_rate)
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.save_attn = save_attn
        self.att_mat = torch.tensor([])

    def forward(self, x):
        output = self.input_rearrange(self.qkv(x))
        q, k, v = output[0], output[1], output[2]
        att_mat = (torch.einsum("blxd,blyd->blxy", q, k) * self.scale).softmax(dim=-1)
        if self.save_attn:
            self.att_mat = att_mat.detach()
        att_mat = self.drop_weights(att_mat)
        x = torch.einsum("bhxy,bhyd->bhxd", att_mat, v)
        x = self.out_rearrange(x)
        x = self.out_proj(x)
        x = self.drop_output(x)
        return x
```
'''
        transformerblock_impl = r'''
TransformerBlock (pre-norm ViT block: norm -> attn -> residual -> norm -> mlp -> residual). Input/output: (B, seq_len, hidden_size).
```python
import torch.nn as nn
from monai.networks.blocks.mlp import MLPBlock
from monai.networks.blocks.selfattention import SABlock

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, mlp_dim: int, num_heads: int, dropout_rate: float = 0.0, qkv_bias: bool = False):
        super().__init__()
        assert 0 <= dropout_rate <= 1 and hidden_size % num_heads == 0
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = SABlock(hidden_size, num_heads, dropout_rate, qkv_bias)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = MLPBlock(hidden_size, mlp_dim, dropout_rate)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
```
'''
        transformer_ref = (
            "Reference implementations you can copy and adapt (ensure hidden_size % num_heads == 0):"
            + sablock_impl
            + transformerblock_impl
        )

        env_prompt = {
            "Installed Packages": (
                f"Your solution can use any relevant computer vision, machine learning, and deep learning packages "
                f"such as: {pkg_str}. Feel free to use any other packages too (all packages are already installed!). "
                "For neural networks you should use PyTorch."
            ),
            "Transformer and Mamba building blocks": f"{transformer_usage} {mamba_usage}",
            "TransformerBlock and SABlock reference implementation": transformer_ref,
        }
        return env_prompt

    @property
    def _prompt_impl_guideline(self):
        """Implementation guide — loaded from template via PromptBuilder."""
        try:
            from .prompt_builder import OpenHandsPromptBuilder
            builder = OpenHandsPromptBuilder(self.cfg)
            timeout_duration = humanize.naturaldelta(self.cfg.exec.timeout)
            content = builder.build_impl_guideline(timeout_duration=timeout_duration)
            return {"Implementation guideline": content}
        except Exception as e:
            logger.warning(f"Failed to load impl_guideline from template: {e}")
            # Minimal fallback
            return {"Implementation guideline": [
                "Use camylanet.training_network() and camylanet.evaluate()",
                "Save experiment_data to working_dir using np.save()",
                f"Code should complete within {humanize.naturaldelta(self.cfg.exec.timeout)}"
            ]}

    def _draft(self) -> Node:
        # New structure: focus on baseline implementation
        prompt: Any = {
            "Introduction": (
                "You are an AI researcher implementing a baseline solution for a machine learning task. "
                "Your task is to implement a solid, working baseline using the provided framework and requirements. "
                "Focus on creating a functional implementation that follows the specified requirements exactly. "
                "This baseline will serve as the foundation for future innovations and improvements."
            ),
            "Task Information": self.task_desc,
            "Memory": self.memory_summary if self.memory_summary else "",
            "Instructions": {},
        }
        

        # New structure: baseline-focused instructions
        prompt["Instructions"] |= {
            "Code Requirements": [
                "Use the exact dataset configuration specified in the task information.",
                "Implement proper preprocessing, training, and evaluation steps.",
                "Report all expected metrics clearly.",
                "Include proper error handling and logging.",
            ],
            "Evaluation Metric(s)": self.evaluation_metrics,
        }
        
        prompt["Instructions"] |= self._prompt_impl_guideline # This remains the same
        prompt["Instructions"] |= self._prompt_environment # This remains the same

        # Inject subagent competition hint if competition mode is active
        _candidates = list(getattr(self.cfg.experiment.code, 'candidates', []) or [])
        if (len(_candidates) > 1
                and self.stage_name and self.stage_name.startswith("2_")):
            prompt["Competition Notice"] = (
                "You are one of multiple AI subagents competing on this exact same task. "
                "Only the subagent producing the best-performing solution will be selected and rewarded. "
                "You must push yourself to deliver the highest quality, most innovative, and "
                "best-performing implementation possible to earn your reward."
            )

        logger.info("MinimalAgent: Getting plan and code")

        # Use aider for code generation with retry mechanism
        try:
            plan, code = self._generate_with_openhands(prompt, "stage1")
            # Override plan with structured description (OpenHands _extract_plan often returns generic fallback)
            dataset_name = ""
            if isinstance(self.task_desc, dict):
                dataset_name = self.task_desc.get("dataset", {}).get("name", "")
            if dataset_name:
                plan = f"Baseline implementation for {dataset_name}"
            logger.debug(plan)
            logger.info("MinimalAgent: Draft complete")
            return Node(plan=plan, code=code)
        except RuntimeError as e:
            # Aider failed to modify file after retries - create buggy node
            logger.warning(f"Creating buggy node: {str(e)}")
            logger.error(f"Creating buggy node due to aider failure: {str(e)}")
            buggy_node = Node(plan="BUGGY: Aider failed to modify file", code="# Aider failed")
            buggy_node.is_buggy = True
            return buggy_node


    def _generate_with_openhands(self, prompt, stage_name="stage2", existing_code="", is_debug=False, is_metrics_parsing=False, reference_files=None):
        """Generate code using OpenHands with persistent workspace
        
        Args:
            prompt: Prompt dict or string
            stage_name: Stage name for workspace organization
            existing_code: Code to start with
            is_debug: Whether this is a debug session (used for prompt content, not retry logic)
            is_metrics_parsing: Whether this is metrics parsing
            reference_files: List of Path objects to innovation reference files
            
        Raises:
            RuntimeError: If file is not modified after internal retries (caller should create buggy node)
        """
        mode_desc = "debug" if is_debug else "generation"
        
        # Create persistent workspace in logs directory (not temporary)
        stage_logs_dir = self._get_stage_logs_dir()
        workspace_dir = stage_logs_dir / "openhands_workspace"
        workspace_dir.mkdir(parents=True, exist_ok=True)
        
        # print(f"📁 OpenHands workspace (persistent): {workspace_dir}")
        # logger.info(f"Using persistent workspace for {mode_desc}: {workspace_dir}")

        # Single attempt - internal verification is handled by OpenHandsCodeGenerator._run_with_verification()
        openhands_gen = None
        
        try:
            logger.info(f"Attempting {stage_name} code {mode_desc} with OpenHands...")
            
            # Initialize OpenHands with reference files
            openhands_gen = OpenHandsCodeGenerator(
                self.cfg, 
                workspace_dir, 
                self.task_desc, 
                exp_name=self.exp_name,
                reference_files=reference_files  # Pass innovation reference files
            )

            # Debug: Print prompt being sent to OpenHands
            if not is_metrics_parsing:
                logger.debug("OpenHands Prompt Debug - Sending prompt to OpenHands:")
                if isinstance(prompt, dict) and "Innovation to Implement" in prompt:
                    logger.debug(f"  Innovation: {prompt['Innovation to Implement'][:500]}...")
                    logger.info(f"🔍 OpenHands receiving innovation prompt")
                elif isinstance(prompt, dict) and "Ablation Study" in str(prompt):
                    logger.debug("  Stage 3: Ablation study")
                    logger.info(f"🔍 OpenHands receiving Stage 3 ablation study prompt")
                else:
                    logger.debug(f"  Prompt type: {type(prompt)}")
                    logger.debug(f"  Prompt keys: {list(prompt.keys()) if isinstance(prompt, dict) else 'Not a dict'}")
                    logger.info(f"🔍 OpenHands receiving non-innovation prompt: {type(prompt)}")

            if not openhands_gen.initialize_openhands():
                raise RuntimeError(f"OpenHands {stage_name} initialization failed")

            # Generate code - internal verification handles followup prompts if file not modified
            if is_debug:
                plan, code, file_modified = openhands_gen.generate_code_with_openhands(
                    prompt, 
                    existing_code=existing_code, 
                    is_metrics_parsing=is_metrics_parsing, 
                    error_feedback=prompt.get("Execution output", None) if isinstance(prompt, dict) else None
                )
            else:
                plan, code, file_modified = openhands_gen.generate_code_with_openhands(
                    prompt, 
                    existing_code=existing_code, 
                    is_metrics_parsing=is_metrics_parsing
                )

            # Check file modification - now handled consistently for all modes
            if not file_modified and not is_metrics_parsing:
                # Internal verification already tried followup prompts, still not modified
                logger.error("FAILED: File was not modified after internal verification attempts")
                raise RuntimeError(f"OpenHands failed to modify file after internal verification - marking as buggy node")

            if not plan or not code:
                raise RuntimeError(f"OpenHands {stage_name} generated empty result")

            logger.info(f"OpenHands {stage_name} code {mode_desc} successful!")
            return plan, code
            
        finally:
            # Only cleanup internal resources, not the workspace directory
            # Workspace is now persistent and kept for auditing
            if openhands_gen is not None:
                openhands_gen.cleanup()
            # logger.info(f"✅ Workspace preserved at: {workspace_dir}")


    
    def _get_stage_logs_dir(self):
        """Get the logs directory for the current stage
        
        Returns path like: logs/0-run/stage_X_xxx/
        """
        # Use cfg.log_dir which is already set to logs/0-run/
        # Then append stage_X_xxx to get the final path
        if self.stage_name:
            stage_logs = Path(self.cfg.log_dir) / f"stage_{self.stage_name}"
        else:
            stage_logs = Path(self.cfg.log_dir) / "stage_default"
        
        stage_logs.mkdir(parents=True, exist_ok=True)
        # logger.info(f"📂 Stage logs directory: {stage_logs}")
        return stage_logs

    def _debug(self, parent_node: Node) -> Node:
        # Determine the appropriate introduction based on stage
        if self.stage_name and self.stage_name.startswith("1_"):
            # Stage 1: Baseline implementation - focus on fixing bugs without improvements
            introduction = (
                "You are an experienced AI researcher working on a baseline implementation. "
                "Your previous baseline code had a bug that needs to be fixed. "
                "IMPORTANT: This is Stage 1 (Baseline Implementation) - your goal is to fix the bug "
                "while maintaining the baseline approach. Do NOT add improvements, optimizations, "
                "or advanced features. Simply fix the bug to get a working baseline implementation. "
                "Your response should be a brief outline followed by a single markdown code block "
                "that fixes the bug while keeping the implementation as a basic, working baseline."
            )
        else:
            # Other stages: choose introduction based on whether this is a diagnostic issue or an actual error
            has_diagnostic = hasattr(parent_node, 'diagnostic_info') and parent_node.diagnostic_info
            if has_diagnostic:
                introduction = (
                    "You are an experienced AI researcher. Your previous implementation ran successfully "
                    "but did not meet performance expectations. Based on the diagnostic information below, "
                    "you should revise the implementation to address the identified issue. "
                )
            else:
                introduction = (
                    "You are an experienced AI researcher. Your previous code for research experiment "
                    "had an error. Based on the information below, you should revise it to fix this issue. "
                )

        prompt: Any = {
            "Introduction": introduction,
            "Experiment Context": self._get_experiment_context(),
            "Previous implementation reference": "The previous code has been provided to openhands and is available in experiment.py for analysis and revision.",
            "Execution output": wrap_code(parent_node.term_out, lang=""),
            "Feedback about execution time": parent_node.exec_time_feedback,
            "Instructions": {},
        }

        # Add Stage 2 innovation information for bug fix preservation
        if self.stage_name and self.stage_name.startswith("2_"):
            innovation_description = self._extract_innovation_description_from_stage(from_node=parent_node)
            if innovation_description:
                prompt["Innovation to Implement"] = innovation_description
                prompt["Baseline Information"] = self._get_baseline_info_for_innovation()
                
                # Validate innovation information transmission
                self._validate_innovation_transmission(innovation_description)
                
                # Debug logging
                logger.info(f"🔧 Stage 2 Debug: Added innovation info to debug prompt")
                logger.info(f"🔧 Innovation: {innovation_description[:100]}...")
            else:
                logger.warning(f"⚠️ Stage 2 Debug: Could not extract innovation description")

        if hasattr(parent_node, 'diagnostic_info') and parent_node.diagnostic_info:
            diag_info = parent_node.diagnostic_info
            diag_type = diag_info.get("type", "unknown")
            
            logger.info(f"🔍 Using diagnostic info for debug: {diag_type}")
            
            prompt["Diagnostic Information"] = {
                "Issue Type": diag_type,
                "Diagnosis Reasoning": diag_info.get("reasoning", "")[:1000],
                "Improvement Suggestions": diag_info.get("improvement_suggestions", []),
                "Performance Gap": f"Baseline: {diag_info.get('baseline_metric', 'N/A'):.4f}, Current: {diag_info.get('current_metric', 'N/A'):.4f}"
            }
            
            if diag_type == "code_issue":
                prompt["Instructions"]["Diagnostic-Guided Revision"] = [
                    "Based on the diagnostic analysis, 5 expert improvement suggestions are provided above.",
                    "Select 1-2 most promising suggestions and implement them.",
                    "In medical image segmentation, compact/smaller networks are generally preferred and sometimes "
                    "perform as well as or better than larger ones. Do NOT increase model capacity unless previous "
                    "experiments have shown that reducing capacity led to performance degradation. Prefer parameter-efficient designs.",
                    "Focus on targeted, high-impact changes rather than broad modifications.",
                ]

        # Add stage-specific debugging guidelines
        if self.stage_name and self.stage_name.startswith("1_"):
            # Stage 1: Baseline debugging guidelines
            prompt["Instructions"] |= {
                "Stage 1 Baseline Debugging Guidelines": [
                    "CRITICAL: This is Stage 1 (Baseline Implementation). Your ONLY goal is to fix the bug to get a working baseline.",
                    "DO NOT add improvements, optimizations, custom architectures, or advanced features.",
                    "DO NOT implement novel techniques or research innovations.",
                    "ONLY fix the specific bug or error that caused the failure.",
                    "Keep the implementation simple, standard, and following the baseline requirements exactly.",
                    "Use default/standard configurations and approaches provided by the framework.",
                    "Write a brief description (2-3 sentences) of the specific bug fix, not improvements.",
                ],
            }
        else:
            # Other stages: Allow improvements during revision
            prompt["Instructions"] |= {
                "Revision guideline": [
                    "You should also write a brief natural language description (3-5 sentences) in python file of how the issue in the previous implementation can be resolved.",
                ],
            }
        prompt["Instructions"] |= self._prompt_impl_guideline

        # Inject subagent competition hint if competition mode is active
        _candidates = list(getattr(self.cfg.experiment.code, 'candidates', []) or [])
        if (len(_candidates) > 1
                and self.stage_name and self.stage_name.startswith("2_")):
            prompt["Competition Notice"] = (
                "You are one of multiple AI subagents competing on this exact same task. "
                "Only the subagent producing the best-performing solution will be selected and rewarded. "
                "You must push yourself to deliver the highest quality, most innovative, and "
                "best-performing implementation possible to earn your reward."
            )

        # Determine stage name for aider debug
        stage_name = self._get_current_stage_name()

        # Add debugging information
        if self.stage_name and self.stage_name.startswith("1_"):
            logger.info("🔧 Stage 1 Debugging: Applying baseline-only bug fix constraints")
        else:
            logger.info(f"🔧 {stage_name} Debugging: Allowing improvements during bug fix")

        # Use aider for debug with retry mechanism
        # Pass the buggy code for incremental improvement
        try:
            plan, code = self._generate_with_openhands(
                prompt,
                stage_name=stage_name,
                existing_code=parent_node.code,
                is_debug=True
            )
            # Override plan with structured context
            diag_info = getattr(parent_node, 'diagnostic_info', None)
            if diag_info:
                diag_type = diag_info.get("type", "unknown")
                diag_reason = diag_info.get("reasoning", "")[:80]
                debug_plan = f"Revision ({diag_type}): {diag_reason}"
            else:
                debug_plan = f"Debug: fixing {parent_node.exc_type or 'error'}"
            if self.stage_name and self.stage_name.startswith("2_"):
                innovation_desc = self._extract_innovation_description_from_stage(from_node=parent_node)
                if innovation_desc:
                    title = innovation_desc.split('\n')[0].strip()[:100]
                    if diag_info:
                        debug_plan = f"Revision ({title}): {diag_info.get('type', 'unknown')}"
                    else:
                        debug_plan = f"Debug ({title}): fixing {parent_node.exc_type or 'error'}"
            elif self.stage_name and self.stage_name.startswith("3_"):
                debug_plan = f"Debug (Ablation study): fixing {parent_node.exc_type or 'error'}"
            plan = debug_plan
            inherited_proposal = getattr(parent_node, 'proposal_content', None)
            return Node(plan=plan, code=code, parent=parent_node,
                        proposal_content=inherited_proposal)
        except RuntimeError as e:
            # Aider failed to modify file after retries - create buggy node
            logger.warning(f"Debug failed, creating buggy node: {str(e)}")
            logger.error(f"Debug failed due to aider failure: {str(e)}")
            inherited_proposal = getattr(parent_node, 'proposal_content', None)
            buggy_node = Node(plan="BUGGY: Aider failed to modify file during debug", code=parent_node.code, parent=parent_node,
                              proposal_content=inherited_proposal)
            buggy_node.is_buggy = True
            return buggy_node

    def _get_current_stage_name(self) -> str:
        """Get current stage name for aider debug"""
        if hasattr(self, 'stage_name') and self.stage_name:
            # Extract main stage from stage_name like "2_baseline_tuning_1_first_attempt"
            if self.stage_name.startswith("1_"):
                return "stage1"
            elif self.stage_name.startswith("2_"):
                return "stage2"
            elif self.stage_name.startswith("3_"):
                return "stage3"
        # Default fallback to stage1 for unknown stages
        return "stage1"

    def _extract_innovation_description_from_stage(self, from_node=None) -> str:
        """Extract proposal content for current stage.
        
        If from_node is provided, walks up the parent chain looking for a
        proposal_content snapshot first (avoids stale-global reads after
        proposal refinement).  Falls back to the global ProposalQueueManager.
        
        Args:
            from_node: Optional Node whose ancestor chain is searched first.
        """
        # Must be Stage 2
        if not self.stage_name or not self.stage_name.startswith("2_"):
            logger.debug(f"Not a Stage 2, returning None")
            return None
        
        # --- Prefer reading from the node snapshot chain ---
        if from_node is not None:
            node = from_node
            while node is not None:
                snapshot = getattr(node, 'proposal_content', None)
                if snapshot:
                    logger.info(f"✅ Using proposal snapshot from node {node.id[:8]} ({len(snapshot)} chars)")
                    return snapshot
                node = getattr(node, 'parent', None)
        
        # --- Fallback: read from the global ProposalQueueManager ---
        proposal_idx = None
        if hasattr(self, 'current_stage') and self.current_stage is not None:
            proposal_idx = getattr(self.current_stage, 'proposal_idx', None)
            if proposal_idx is not None:
                logger.info(f"✅ Using explicit proposal_idx={proposal_idx} from Stage object")
        
        if proposal_idx is None:
            logger.warning(f"Stage {self.stage_name} has no proposal_idx, falling back to string parsing")
            proposal_idx = self._parse_proposal_idx_from_stage_name()
        
        if proposal_idx is None:
            logger.error(f"❌ Could not determine proposal_idx for stage {self.stage_name}")
            return None
        
        if not self.agent_manager:
            logger.error("❌ agent_manager not available")
            return None
        
        if not hasattr(self.agent_manager, 'proposal_queue_manager') or not self.agent_manager.proposal_queue_manager:
            logger.error("❌ proposal_queue_manager not available")
            return None
        
        try:
            proposal_content = self.agent_manager.proposal_queue_manager.get_proposal_for_prompt(proposal_idx)
            logger.info(f"✅ Successfully loaded proposal {proposal_idx} from global queue ({len(proposal_content)} chars)")
            return proposal_content
        except Exception as e:
            logger.error(f"❌ Failed to get proposal {proposal_idx}: {e}")
            raise
    
    def _parse_proposal_idx_from_stage_name(self) -> Optional[int]:
        """Backward-compat fallback: parse the proposal index from the stage name (legacy data only)."""
        try:
            parts = self.stage_name.split("_")
            
            # Supports the "2_creative_research_1_proposal_1" format
            if "proposal" in parts:
                proposal_pos = parts.index("proposal")
                if proposal_pos + 1 < len(parts):
                    return int(parts[proposal_pos + 1]) - 1
            
            # Supports the "2_creative_research_1_first_attempt" format
            if "first" in parts and "attempt" in parts:
                sub_stage_num = int(parts[3])
                return sub_stage_num - 1
            
            return None
        except Exception as e:
            logger.warning(f"Failed to parse proposal_idx from stage name: {e}")
            return None

    def _validate_innovation_transmission(self, innovation_description):
        """Validate that innovation information is properly transmitted"""
        if not innovation_description:
            logger.error("CRITICAL: Innovation description is None or empty!")
            return False

        if len(innovation_description.strip()) < 50:
            logger.warning(f"Innovation description seems too short ({len(innovation_description)} chars)")

        # Check for common baseline indicators
        baseline_indicators = [
            "baseline implementation",
            "basic training",
            "standard approach",
            "default configuration",
            "simple implementation"
        ]

        description_lower = innovation_description.lower()
        for indicator in baseline_indicators:
            if indicator in description_lower and "innovation" not in description_lower:
                logger.warning(f"Innovation description contains baseline indicator: '{indicator}'")

        logger.info(f"✅ Innovation validation passed - {len(innovation_description)} characters")
        return True



    def _generate_default_baseline(self, dataset_info: dict) -> dict:
        """Generate a default baseline configuration dynamically from dataset info.
        
        With this helper, the baseline field no longer needs to be filled in manually in the JSON file;
        the system auto-generates a standard nnUNet baseline configuration from the dataset info.
        """
        target_structure = dataset_info.get('target_structure', 'target')
        dataset_id = dataset_info.get('dataset_id', 'unknown')
        configuration = dataset_info.get('configuration', '3d_fullres')
        
        return {
            "name": "nnUNet Baseline",
            "description": f"Standard nnUNet implementation for {target_structure} segmentation using camylanet framework",
            "requirements": [
                "Use camylanet framework for nnUNet implementation",
                "Follow standard preprocessing and training pipeline",
                f"Use dataset_id={dataset_id} and configuration='{configuration}'",
                "Implement basic training, evaluation, and result reporting",
                "Ensure reproducible results with proper metrics reporting"
            ],
            "expected_metrics": [
                "Dice Score",
                "HD95 Score"
            ]
        }

    def _get_baseline_info(self, task_dict: dict = None) -> dict:
        """Fetch baseline info; auto-generate if the JSON does not contain it.
        
        Args:
            task_dict: task dict; when None, read from self.task_desc.
            
        Returns:
            Baseline info dict.
        """
        if task_dict is None:
            import json
            task_dict = json.loads(self.task_desc) if isinstance(self.task_desc, str) else self.task_desc
        
        # If JSON contains a baseline field, use it directly
        if "baseline" in task_dict and task_dict["baseline"]:
            return task_dict["baseline"]
        
        # Otherwise auto-generate from dataset info
        dataset_info = task_dict.get("dataset", {})
        return self._generate_default_baseline(dataset_info)

    def _get_baseline_info_for_innovation(self) -> str:
        """Get baseline information for innovation implementation"""
        try:
            import json
            task_dict = json.loads(self.task_desc) if isinstance(self.task_desc, str) else self.task_desc
            dataset_info = task_dict.get("dataset", {})
            
            # Use the unified helper to fetch baseline info (auto-generate or read from JSON)
            baseline_info = self._get_baseline_info(task_dict)

            info = f"Dataset: {dataset_info.get('name', 'Unknown')}\n"
            info += f"Task: {dataset_info.get('description', 'Unknown')}\n"
            info += f"Baseline: {baseline_info.get('name', 'Unknown')}\n"
            info += f"Requirements: {'; '.join(baseline_info.get('requirements', []))}\n"

            return info
        except:
            return "Please implement the innovation based on the baseline code provided."

    def _get_experiment_context(self) -> str:
        """Get appropriate experiment context"""
        if hasattr(self, '_is_new_structure') and self._is_new_structure:
            try:
                import json
                task_dict = json.loads(self.task_desc) if isinstance(self.task_desc, str) else self.task_desc
                dataset_info = task_dict.get("dataset", {})

                context = f"Dataset: {dataset_info.get('name', 'Unknown')} - {dataset_info.get('description', 'Unknown task')}"

                # Add stage-specific context
                if self.stage_name:
                    if self.stage_name.startswith("1_"):
                        context += " | Stage: Baseline Implementation"
                    elif self.stage_name.startswith("2_"):
                        innovation_desc = self._extract_innovation_description_from_stage()
                        if innovation_desc:
                            context += f" | Stage: Innovation Testing - {innovation_desc.split(':')[1].split('Description:')[0].strip() if ':' in innovation_desc else 'Innovation'}"
                        else:
                            context += " | Stage: Innovation Testing"

                    elif self.stage_name.startswith("4_"):
                        context += " | Stage: Ablation Studies"

                return context
            except:
                pass

        # Fallback for old structure
        return f"Research idea: {self.task_desc}..." if len(self.task_desc) > 200 else self.task_desc

    def _improve(self, parent_node: Node) -> Node:
        # Check if this is Stage 2 (Creative Research) - extract innovation description
        innovation_description = None
        if self.stage_name and self.stage_name.startswith("2_"):
            innovation_description = self._extract_innovation_description_from_stage(from_node=parent_node)

        # Check if parent_node is a precomputed baseline (no code)
        is_precomputed_baseline = (
            getattr(parent_node, 'is_precomputed_baseline', False) or 
            (parent_node.code is None or parent_node.code.strip() == "")
        )
        
        if is_precomputed_baseline:
            logger.info("Parent node is precomputed baseline (no code), generating from scratch")

        # Build prompt based on stage type
        if innovation_description:
            # Stage 2: Use specific innovation description
            logger.info("Stage 2: Building innovation-specific prompt")

            if is_precomputed_baseline:
                # Stage 2 with precomputed baseline: generate from scratch based on proposal
                prompt: Any = {
                    "Introduction": (
                        "You are an experienced AI researcher implementing a research proposal from scratch. "
                        "You need to implement the specific innovation described below using the camylanet framework. "
                        "This is a fresh implementation - focus on correctly implementing the innovation."
                    ),
                    "Innovation to Implement": innovation_description,
                    "Baseline Information": self._get_baseline_info_for_innovation(),
                    "Task Information": self.task_desc,
                    "Memory": self.memory_summary if self.memory_summary else "",
                    "Instructions": {},
                }
            else:
                # Stage 2 with existing code: improve upon it
                prompt: Any = {
                    "Introduction": (
                        "You are an experienced AI researcher. You are provided with a baseline implementation "
                        "and need to implement a specific innovation to improve it. Focus on implementing "
                        "the innovation described below while maintaining the core functionality."
                    ),
                    "Innovation to Implement": innovation_description,
                    "Baseline Information": self._get_baseline_info_for_innovation(),
                    "Memory": self.memory_summary if self.memory_summary else "",
                    "Feedback about execution time": parent_node.exec_time_feedback if parent_node.exec_time_feedback else "",
                    "Instructions": {},
                }

            # Validate innovation information transmission
            self._validate_innovation_transmission(innovation_description)
        else:
            # Default improvement logic for other stages
            prompt: Any = {
                "Introduction": (
                    "You are an experienced AI researcher. You are provided with a previously developed "
                    "implementation. Your task is to improve it based on the current experimental stage."
                ),
                "Research idea": self._get_task_desc_str(),
                "Memory": self.memory_summary if self.memory_summary else "",
                "Feedback about execution time": parent_node.exec_time_feedback if parent_node.exec_time_feedback else "",
                "Instructions": {},
            }
        
        # Add previous solution reference only if not precomputed baseline
        if not is_precomputed_baseline:
            prompt["Previous solution"] = {
                "Code reference": "The previous solution has been provided to openhands and is available in experiment.py for modification.",
            }

        prompt["Instructions"] |= self._prompt_impl_guideline
        prompt["Instructions"] |= self._prompt_environment

        # Add stage-specific instructions
        if innovation_description:
            if is_precomputed_baseline:
                # Stage 2 with precomputed baseline: fresh implementation instructions
                prompt["Instructions"] |= {
                    "Innovation Implementation Guidelines": [
                        "Implement the specific innovation described above from scratch.",
                        "Use the camylanet framework for model training and evaluation.",
                        "Follow the dataset configuration and evaluation metrics specified.",
                        "Ensure proper experiment_data.npy output format.",
                        "Document your implementation approach clearly.",
                        "Every core innovation module in the proposal MUST have a corresponding implementation. "
                        "You may adapt the implementation approach if needed (e.g., use efficient approximation "
                        "for OOM, windowed attention instead of full attention, reduced channel dimensions), "
                        "but do NOT replace novel modules with plain Conv3d/MLP or skip them entirely.",
                    ],
                    "Code Requirements": [
                        "Use the exact dataset configuration specified in the task information.",
                        "Implement proper preprocessing, training, and evaluation steps.",
                        "Report all expected metrics clearly.",
                        "Include proper error handling and logging.",
                    ],
                }
            else:
                # Stage 2 with existing code: modification instructions
                prompt["Instructions"] |= {
                    "Innovation Implementation Guidelines": [
                        "Focus on implementing the specific innovation described above.",
                        "Build upon the baseline code provided, don't start from scratch.",
                        "Ensure the innovation is properly integrated into the existing framework.",
                        "Maintain compatibility with the dataset and evaluation metrics.",
                        "Test that the implementation works correctly before finalizing.",
                        "Document any significant changes or new components added.",
                        "Every core innovation module in the proposal MUST have a corresponding implementation. "
                        "You may adapt the implementation approach if needed (e.g., use efficient approximation "
                        "for OOM, windowed attention instead of full attention, reduced channel dimensions), "
                        "but do NOT replace novel modules with plain Conv3d/MLP or skip them entirely.",
                    ],
                    "Code Modification Strategy": [
                        "Identify the specific parts of the baseline that need modification.",
                        "Implement the innovation incrementally to avoid breaking existing functionality.",
                        "Preserve the overall structure and flow of the baseline implementation.",
                        "Ensure proper error handling and logging for new components.",
                    ]
                }
            
            # 🆕 Inject optimization hints if available (after beating baseline)
            if (hasattr(self, 'agent_manager') and self.agent_manager 
                and hasattr(self.agent_manager, 'proposal_diagnostic') 
                and self.agent_manager.proposal_diagnostic
                and self.stage_name):
                try:
                    hints = self.agent_manager.proposal_diagnostic.get_optimization_hints(self.stage_name)
                    if hints:
                        has_proposal_gaps = "Proposal Implementation Gaps" in hints
                        if has_proposal_gaps:
                            prompt["Implementation & Optimization Guidance"] = (
                                "🎯 Your implementation has exceeded the baseline — great work! "
                                "However, analysis shows that some core innovation modules from the research proposal "
                                "have been **simplified or are missing**. Your TOP PRIORITY is to properly implement "
                                "these missing modules — they represent the core research contribution and may unlock "
                                "significantly better performance.\n\n"
                                "After addressing proposal gaps, you may also fine-tune hyperparameters.\n\n"
                                + hints
                            )
                        else:
                            prompt["Implementation & Optimization Guidance"] = (
                                "🎯 Your implementation has exceeded the baseline and appears to faithfully "
                                "implement the proposal's core innovations. Focus on fine-tuning hyperparameters "
                                "to push performance even higher.\n\n"
                                + hints
                            )
                        logger.info(f"Injected optimization hints into Stage 2 prompt "
                                    f"(has_proposal_gaps={has_proposal_gaps})")
                except Exception as e:
                    logger.debug(f"Could not inject optimization hints: {e}")

        if (self.stage_name and self.stage_name.startswith("2_")
                and hasattr(parent_node, 'diagnostic_info') and parent_node.diagnostic_info):
            diag_info = parent_node.diagnostic_info
            diag_type = diag_info.get("type", "unknown")
            logger.info(f"🔍 Injecting diagnostic info into improve prompt: {diag_type}")

            prompt["Diagnostic Information"] = {
                "Issue Type": diag_type,
                "Diagnosis Reasoning": diag_info.get("reasoning", "")[:1000],
                "Improvement Suggestions": diag_info.get("improvement_suggestions", []),
                "Performance Gap": (
                    f"Baseline: {diag_info.get('baseline_metric', 'N/A'):.4f}, "
                    f"Current: {diag_info.get('current_metric', 'N/A'):.4f}"
                ),
            }

            if diag_type == "code_issue":
                prompt["Instructions"]["Diagnostic-Guided Revision"] = [
                    "Based on the diagnostic analysis, 5 expert improvement suggestions are provided above.",
                    "Select 1-2 most promising suggestions and implement them.",
                    "In medical image segmentation, compact/smaller networks are generally preferred and sometimes "
                    "perform as well as or better than larger ones. Do NOT increase model capacity unless previous "
                    "experiments have shown that reducing capacity led to performance degradation. Prefer parameter-efficient designs.",
                    "Focus on targeted, high-impact changes rather than broad modifications.",
                ]

        # Inject subagent competition hint if competition mode is active
        _candidates = list(getattr(self.cfg.experiment.code, 'candidates', []) or [])
        if (len(_candidates) > 1
                and self.stage_name and self.stage_name.startswith("2_")):
            prompt["Competition Notice"] = (
                "You are one of multiple AI subagents competing on this exact same task. "
                "Only the subagent producing the best-performing solution will be selected and rewarded. "
                "You must push yourself to deliver the highest quality, most innovative, and "
                "best-performing implementation possible to earn your reward."
            )

        # Determine stage name for aider
        if self.stage_name and self.stage_name.startswith("1_"):
            stage_name = "stage1"
        elif self.stage_name and self.stage_name.startswith("2_"):
            stage_name = "stage2"
        elif self.stage_name and self.stage_name.startswith("3_"):
            stage_name = "stage3"
        else:
            stage_name = "stage1"  # default to stage1 for unknown stages

        # Use aider for code generation with retry mechanism
        # For precomputed baseline, start from empty code
        existing_code = "" if is_precomputed_baseline else parent_node.code
        
        try:
            plan, code = self._generate_with_openhands(
                prompt,
                stage_name=stage_name,
                existing_code=existing_code
            )
            # Override plan with proposal/innovation info
            if innovation_description:
                # Extract title from innovation_description (first line or first sentence)
                plan_title = innovation_description.split('\n')[0].strip()
                if len(plan_title) > 200:
                    plan_title = plan_title[:200] + "..."
                plan = plan_title
            return Node(
                plan=plan,
                code=code,
                parent=parent_node,
                proposal_content=innovation_description,
            )
        except RuntimeError as e:
            # Aider failed to modify file after retries - create buggy node
            logger.warning(f"Creating buggy node for draft_solution_from_seed: {str(e)}")
            logger.error(f"Creating buggy node due to aider failure: {str(e)}")
            buggy_node = Node(plan="BUGGY: Aider failed to modify file", code=existing_code, parent=parent_node,
                              proposal_content=innovation_description)
            buggy_node.is_buggy = True
            return buggy_node



    def _get_task_desc_str(self):
        # New structure: focus on dataset and baseline implementation
        dataset_info = self.task_desc["dataset"]
        # Fetch baseline via the unified helper (supports auto-generation)
        baseline_info = self._get_baseline_info(self.task_desc)

        task_desc = f"""You are an AI researcher working on {dataset_info['task_type']} tasks.
Your current task is to implement and experiment with different approaches for {dataset_info['name']}.

Dataset Information:
- Name: {dataset_info['name']}
- Description: {dataset_info['description']}
- Dataset ID: {dataset_info['dataset_id']}
- Configuration: {dataset_info['configuration']}
- Modality: {dataset_info['modality']}
- Target: {dataset_info['target_structure']}

Baseline Implementation:
- Name: {baseline_info['name']}
- Description: {baseline_info['description']}
- Requirements: {'; '.join(baseline_info['requirements'])}

"""
        # Code will be loaded from Python file and added via loaded_code field
        if "loaded_code" in self.task_desc:
            task_desc += f"Code Template:\n{self.task_desc['loaded_code']}\n"

        return task_desc





    def _generate_ablation_node(self, parent_node: Node, plan_item: AblationPlanItem):
        """Generate a node for ablation study (Stage 3).

        Supports two ablation_type values from the plan:
          - "removal": remove / disable one or more components (single training run)
          - "comparison": run up to 5 variants in a single script
        """
        # --- Build prompt based on ablation type ---
        if plan_item.ablation_type == "comparison" and plan_item.variants:
            variants_desc = "\n".join(
                f"  {i+1}. {v}" for i, v in enumerate(plan_item.variants)
            )
            intro = (
                "You are an experienced AI researcher conducting an ABLATION STUDY "
                "(comparison experiment). You are provided with the best model "
                "implementation from Stage 2. Your task: "
                f"{plan_item.description}\n\n"
                f"Variants to test (run each as a SEPARATE training session "
                f"in the SAME script, sequentially):\n{variants_desc}"
            )
            data_instructions = [
                "Data saving requirements for COMPARISON experiments:",
                "- Run each variant as a separate full training session sequentially.",
                f"- You have {len(plan_item.variants)} variants. Each one trains the model from scratch.",
                "- Save ALL variant results into a single experiment_data dict with "
                "keys like 'msd_hippocampus__<variant_short_name>'.",
                "  ```python",
                "  experiment_data = {}",
                "  for variant_name, variant_config in variants:",
                "      # ... train model with this variant ...",
                "      # ... evaluate to get dice_score and hd95_score ...",
                "      experiment_data[f'msd_hippocampus__{variant_name}'] = {",
                "          'metrics': {",
                "              'train': train_losses,   # list of per-epoch loss floats",
                "              'val': [{'dice': dice_score, 'hd95': hd95_score}],  # MUST be list of dicts",
                "          },",
                "          'dice_scores': [dice_score],    # list of float",
                "          'hd95_scores': [hd95_score],    # list of float",
                "          'result_folder': result_folder,  # str path from training_network()",
                "          'epochs': list(range(num_epochs)),",
                "      }",
                "  np.save(os.path.join(working_dir, 'experiment_data.npy'), experiment_data)",
                "  ```",
                "- IMPORTANT: 'metrics.val' MUST be a list of dicts with 'dice' and 'hd95' keys, NOT raw loss values.",
                "- At the end, print a summary table comparing all variants.",
                "- Use filename 'experiment_data.npy'. Do not use any other filename.",
            ]
        else:
            intro = (
                "You are an experienced AI researcher conducting an ABLATION STUDY "
                "(component removal). You are provided with the best model "
                "implementation from Stage 2. Your task: "
                f"{plan_item.description}"
            )
            data_instructions = [
                "Data saving requirements:",
                "- Save all plottable data as numpy arrays using np.save().",
                "  ```python",
                "  experiment_data = {",
                "      'msd_hippocampus': {",
                "          'metrics': {",
                "              'train': train_losses,   # list of per-epoch loss floats",
                "              'val': [{'dice': dice_score, 'hd95': hd95_score}],  # MUST be list of dicts",
                "          },",
                "          'dice_scores': [dice_score],    # list of float",
                "          'hd95_scores': [hd95_score],    # list of float",
                "          'result_folder': result_folder,  # str path from training_network()",
                "          'epochs': list(range(num_epochs)),",
                "      }",
                "  }",
                "  ```",
                "- IMPORTANT: 'metrics.val' MUST be a list of dicts with 'dice' and 'hd95' keys, NOT raw loss values.",
                "- Use filename 'experiment_data.npy'. Do not use any other filename.",
            ]

        # Compute effective timeout for the LLM prompt
        exec_timeout = self.cfg.exec.timeout
        if plan_item.ablation_type == "comparison" and plan_item.variants:
            exec_timeout = exec_timeout * 5
        timeout_str = humanize.naturaldelta(exec_timeout)

        prompt: Any = {
            "Introduction": intro,
            "Base code reference": (
                "The model code has been provided to openhands and is "
                "available in experiment.py for modification."
            ),
            "Ablation Objective": plan_item.description,
            "Instructions": {
                "Implementation guideline": [
                    "The code should be a single-file python program that is "
                    "self-contained and can be executed as-is.",
                    "No parts of the code should be skipped.",
                    "Keep all other components and training settings unchanged "
                    "except for the parts being ablated.",
                    "The model should still be functional after modification.",
                    f"**Target Runtime**: The code you write should complete within {timeout_str}.",
                ] + data_instructions,
            },
        }

        try:
            plan, code = self._generate_with_openhands(
                prompt,
                stage_name="stage3",
                existing_code=parent_node.code,
            )
            return Node(
                plan=f"Ablation: {plan_item.name} ({plan_item.ablation_type}).\n{plan_item.description}",
                code=code,
                parent=parent_node,
                ablation_name=plan_item.name,
                ablation_type=plan_item.ablation_type,
            )
        except RuntimeError as e:
            logger.warning(f"Creating buggy node for ablation study: {e}")
            logger.error(f"Creating buggy node due to openhands failure: {e}")
            buggy_node = Node(
                plan=f"BUGGY: Ablation {plan_item.name} - OpenHands failed",
                code=parent_node.code,
                parent=parent_node,
                ablation_name=plan_item.name,
                ablation_type=plan_item.ablation_type,
            )
            buggy_node.is_buggy = True
            return buggy_node

    def parse_exec_result(
        self, node: Node, exec_result: ExecutionResult, workspace: str
    ):
        logger.info(f"Agent is parsing execution results for node {node.id}")

        node.absorb_exec_result(exec_result)

        prompt = {
            "Introduction": (
                "You are an experienced AI researcher. "
                "You have written code for your research experiment and now need to evaluate the output of the code execution. "
                "Analyze the execution output, determine if there were any bugs, and provide a summary of the findings. "
            ),
            "Important Guidelines": (
                "When evaluating the execution output, please note the following:\n"
                "- Negative loss values are ACCEPTABLE and NOT a bug. Some loss functions (like negative log-likelihood, "
                "negative Dice coefficient, or other metrics used as losses) can legitimately produce negative values.\n"
                "- Validation losses can be negative if the underlying metric is designed that way.\n"
                "- Only report as a bug if there are actual execution errors, exceptions, crashes, or clearly incorrect behavior.\n"
                "- Do NOT flag negative loss values, negative validation losses, or negative metrics as bugs."
            ),
            "Research idea": self._get_task_desc_str(),
            "Implementation": wrap_code(node.code),
            "Execution output": wrap_code(node.term_out, lang=""),
        }

        max_retries = 5
        last_error = None
        
        for attempt in range(max_retries):
            try:
                response = cast(
                    dict,
                    query(
                        system_message=None,
                        user_message=prompt,
                        func_spec=review_func_spec,
                        **{"model": get_role("feedback")["model"], "temperature": get_role("feedback").get("temperature", 0.9)},
                    ),
                )

                node.analysis = response["summary"]
                node.is_buggy = response["is_bug"] or node.exc_type is not None
                logger.info(f"Successfully parsed execution result for node {node.id}")
                return  # Success, exit function
                
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for node {node.id}: {e}")
                logger.warning(f"Retry {attempt + 1}/{max_retries}: LLM parsing failed - {e}")
                
                if attempt < max_retries - 1:
                    import time
                    time.sleep(1)  # Brief delay before retry
        
        # All retries exhausted - fallback to exception-based detection
        logger.error(f"All {max_retries} retries failed for node {node.id}: {last_error}")
        node.analysis = f"Failed to parse after {max_retries} retries: {str(last_error)}"
        node.is_buggy = node.exc_type is not None
        logger.error(f"All retries exhausted for node {node.id}, falling back to exception-based bug detection")

    def _generate_node_summary(self, node: Node) -> dict:
        """Generate a summary of the node's experimental findings"""
        summary_prompt = {
            "Introduction": (
                "You are an AI researcher analyzing experimental results. "
                "Please summarize the findings from this experiment iteration."
            ),
            "Experiment Context": self._get_experiment_context(),
            "Implementation": wrap_code(node.code),
            "Plan": node.plan,
            "Execution output": wrap_code(node.term_out, lang=""),
            "Analysis": node.analysis,
            "Metric": str(node.metric) if node.metric else "Failed",
        }

        return cast(
            dict,
            query(
                system_message=None,
                user_message=summary_prompt,
                func_spec={
                    "name": "summarize_experiment",
                    "description": "Summarize experimental findings",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "findings": {
                                "type": "string",
                                "description": "Key findings and results",
                            },
                            "significance": {
                                "type": "string",
                                "description": "Why these results matter",
                            },
                            "next_steps": {
                                "type": "string",
                                "description": "Suggested improvements or next experiments",
                            },
                        },
                        "required": ["findings", "significance"],
                    },
                },
                **{"model": get_role("feedback")["model"], "temperature": get_role("feedback").get("temperature", 0.9)},
            ),
        )




class ParallelAgent:
    def __init__(
        self,
        task_desc: str,
        cfg: Config,
        journal: Journal,
        stage_name=None,
        stage=None,  # 🆕 Stage object used to access proposal_idx
        best_stage2_node=None,
        best_stage1_node=None,
        agent_manager=None,
    ):
        super().__init__()
        self.task_desc = task_desc
        self.cfg = cfg
        self.journal = journal
        self.stage_name = stage_name
        self.current_stage = stage  # Store the Stage object so proposal_idx is accessible
        self.best_stage1_node = (
            best_stage1_node  # used for Stage 2 creative research baseline reference
        )
        self.best_stage2_node = (
            best_stage2_node  # used for Stage 3 ablation studies
        )
        self.data_preview = None
        self.timeout = self.cfg.exec.timeout

        # Store reference to agent_manager for innovation queue access
        self.agent_manager = agent_manager
        
        # Get exp_name from agent_manager
        self.exp_name = None
        if self.agent_manager and hasattr(self.agent_manager, 'exp_name'):
            self.exp_name = self.agent_manager.exp_name
            logger.info(f"🎯 ParallelAgent initialized with exp_name: {self.exp_name}")

        # Define the metric once at initialization
        self.evaluation_metrics = self._define_global_metrics()

        # Stage 3: generate ablation plan upfront (or restore from journal on checkpoint resume)
        if self.stage_name and self.stage_name.startswith("3_"):
            saved_plan = getattr(self.journal, '_ablation_plan', None)
            saved_idx = getattr(self.journal, '_ablation_plan_idx', None)
            if saved_plan is not None and saved_idx is not None:
                logger.info(
                    f"Restoring ablation plan from journal "
                    f"(progress: {saved_idx}/{len(saved_plan)})"
                )
                self._ablation_plan = saved_plan
                self._ablation_plan_idx = saved_idx
                self._current_item_debug_attempted = getattr(
                    self.journal, '_current_item_debug_attempted', False
                )
            else:
                self._ablation_plan = self._generate_ablation_plan()
                self._ablation_plan_idx = 0
                self._current_item_debug_attempted = False
                self.journal._ablation_plan = self._ablation_plan
                self.journal._ablation_plan_idx = self._ablation_plan_idx
                self.journal._current_item_debug_attempted = self._current_item_debug_attempted
        else:
            self._ablation_plan: List[AblationPlanItem] = []
            self._ablation_plan_idx = 0
            self._current_item_debug_attempted = False

    def _save_code_to_stage_directory(self, code: str, node_id: str, stage_name: str, code_type: str = "experiment"):
        """
        Save generated code to the stage directory with timestamp and ID

        Args:
            code: The code content to save
            node_id: The unique node ID
            stage_name: The current stage name
            code_type: Type of code (experiment, plotting, etc.)
        """
        try:
            # 🔧 After directory merge: use cfg.log_dir directly
            stage_dir = Path(self.cfg.log_dir) / f"stage_{stage_name}" / "generated_codes"
            stage_dir.mkdir(parents=True, exist_ok=True)

            # Create filename with timestamp and ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{node_id}_{code_type}_code.py"
            file_path = stage_dir / filename

            # Save the code
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"# Generated at: {datetime.now().isoformat()}\n")
                f.write(f"# Node ID: {node_id}\n")
                f.write(f"# Stage: {stage_name}\n")
                f.write(f"# Code Type: {code_type}\n")
                f.write(f"# File: {filename}\n")
                f.write("#" + "="*70 + "\n\n")
                f.write(code)

            logger.info(f"Saved {code_type} code to stage directory: {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"Failed to save code to stage directory: {e}")
            return None

    def _define_global_metrics(self) -> str:
        """Define eval metric to be used across all experiments - Fixed for segmentation tasks"""
        # Fixed metrics for segmentation tasks: Dice and HD95
        fixed_metrics = [
            {
                "name": "Dice",
                "maximize": True,
                "description": "Dice Similarity Coefficient (DSC), measures the overlap between predicted segmentation and ground truth. Range [0,1], higher is better."
            },
            {
                "name": "HD95",
                "maximize": False,
                "description": "95th percentile Hausdorff Distance, measures boundary accuracy between predicted and ground truth surfaces. Lower values indicate better segmentation quality."
            }
        ]

        import json
        response = json.dumps(fixed_metrics)

        logger.info("Using fixed segmentation metrics: Dice (↑) and HD95 (↓)")
        return response

    def _generate_ablation_plan(self) -> List[AblationPlanItem]:
        """Generate a complete ablation plan at Stage 3 start via a single LLM call.

        Analyses the best Stage 2 code and returns a structured list of ablation
        experiments (removal and comparison types).
        """
        if not self.best_stage2_node or not getattr(self.best_stage2_node, 'code', None):
            logger.error("No valid Stage 2 node available for ablation plan generation")
            return []

        import json as _json

        plan_prompt = {
            "Introduction": (
                "You are an AI researcher designing a comprehensive ABLATION STUDY plan. "
                "Analyze the provided model implementation and produce a structured list of "
                "ablation experiments that will appear in a scientific paper."
            ),
            "Base model code": wrap_code(self.best_stage2_node.code),
            "Experiment types": {
                "removal": (
                    "Remove / disable one or MORE components at once to measure their "
                    "joint contribution. Produces 1 training run."
                ),
                "comparison": (
                    "Test different VALUES of an internal parameter, or different "
                    "architectural/design choices (e.g. CNN encoder vs ViT encoder, "
                    "self-attention vs cross-attention). Each variant trains separately "
                    "in the same script. Max 5 variants per comparison experiment."
                ),
            },
            "Good comparison examples": [
                "Internal parameter values: attention weighting alpha = 0, 0.25, 0.5, 0.75, 1.0",
                "Design variants: CNN encoder vs ViT encoder vs hybrid",
                "Design variants: multi-scale fusion strategies (early vs late vs pyramid)",
            ],
            "What to EXCLUDE": [
                "Trivial hyper-parameter sweeps with no scientific insight "
                "(e.g. channel count grids, learning rate grids)",
                "Experiments that violate framework constraints "
                "(custom loss, data augmentation changes, multi-output)",
            ],
            "Requirements": [
                "Produce 3-6 experiments depending on model complexity.",
                "Each removal experiment can disable one OR multiple related components.",
                "Each comparison experiment must have at most 5 variants.",
                "Prioritize ablating the CORE INNOVATIONS introduced in Stage 2.",
                "Every experiment must be scientifically meaningful — "
                "something you would include in an ablation table of a published paper.",
            ],
            "Output format": (
                "Return a JSON array. Each element is an object with keys: "
                '"name" (short identifier), '
                '"description" (what to do and why, 2-3 sentences), '
                '"ablation_type" ("removal" or "comparison"), '
                '"variants" (list of variant descriptions for comparison type; '
                "empty list [] for removal type). "
                "Return ONLY the JSON array, no other text."
            ),
        }

        retry_limit = 3
        for attempt in range(retry_limit):
            try:
                raw = query(
                    system_message=None,
                    user_message=plan_prompt,
                    **{"model": get_role("ablation_ideation")["model"], "temperature": get_role("ablation_ideation").get("temperature", 0.9)},
                )
                text = str(raw).strip()
                # Strip markdown code fences if present
                if text.startswith("```"):
                    text = re.sub(r"^```(?:json)?\s*", "", text)
                    text = re.sub(r"\s*```$", "", text)
                items = _json.loads(text)
                if not isinstance(items, list):
                    raise ValueError("Expected a JSON array")

                plan: List[AblationPlanItem] = []
                for item in items:
                    plan.append(AblationPlanItem(
                        name=item.get("name", f"ablation_{len(plan)}"),
                        description=item.get("description", ""),
                        ablation_type=item.get("ablation_type", "removal"),
                        variants=item.get("variants", []),
                    ))
                logger.info(f"Generated ablation plan with {len(plan)} items")
                for i, p in enumerate(plan):
                    logger.info(f"  [{i}] {p}")
                    logger.debug(f"  Ablation plan [{i}] {p.ablation_type}: {p.name}")
                return plan

            except Exception as e:
                logger.warning(f"Ablation plan generation attempt {attempt+1}/{retry_limit} failed: {e}")

        logger.error("Failed to generate ablation plan after retries; returning empty plan")
        return []

    def _get_leaves(self, node: Node) -> List[Node]:
        """Get all leaf nodes in the subtree rooted at node."""
        if not node.children:
            return [node]

        leaves = []
        for child in node.children:
            leaves.extend(self._get_leaves(child))
        return leaves

    # =========================================================================
    # Stage 2 UCB Tree Search (PUCT-based exploration-exploitation)
    # =========================================================================

    def _normalize_q(self, metric_value: float, baseline: float) -> float:
        """Normalize metric to Q in [-1, +1]. Baseline maps to 0.

        - metric >= baseline: linear map to [0, +1].
        - metric < baseline: power-law Q = -gap^q_below_exponent (gap in [0,1]);
          exponent < 1 strengthens penalty; exponent == 1 is linear (legacy).
        - NaN -> -1.0.
        """
        import math
        if math.isnan(metric_value):
            return -1.0
        if metric_value >= baseline:
            denom = max(1.0 - baseline, 1e-8)
            return min((metric_value - baseline) / denom, 1.0)
        else:
            denom = max(baseline, 1e-8)
            gap = (baseline - metric_value) / denom
            ucb_cfg = getattr(self.cfg.experiment.search, "stage2_ucb", None)
            exponent = (
                float(getattr(ucb_cfg, "q_below_exponent", 1.0)) if ucb_cfg else 1.0
            )
            q = -min(1.0, gap ** exponent)
            return max(-1.0, q)

    def _find_ancestor_metric(self, node: Node) -> float:
        """Walk up parent chain to find nearest ancestor with a valid (non-NaN) metric."""
        import math
        current = node.parent
        while current:
            if current.metric:
                val = current.metric.get_mean_value()
                if not math.isnan(val):
                    return val
            current = current.parent
        return float('nan')

    def _node_q(self, node: Node, baseline_metric: float) -> float:
        """Return the normalized Q value for a single node, handling buggy/NaN."""
        import math
        raw = node.metric.get_mean_value() if node.metric else float('nan')
        if math.isnan(raw):
            ancestor_raw = self._find_ancestor_metric(node)
            if not math.isnan(ancestor_raw):
                ucb_cfg = getattr(self.cfg.experiment.search, 'stage2_ucb', None)
                penalty = float(getattr(ucb_cfg, 'buggy_q_penalty', 0.2)) if ucb_cfg else 0.2
                return max(-1.0, self._normalize_q(ancestor_raw, baseline_metric) - penalty)
            return -1.0
        return self._normalize_q(raw, baseline_metric)

    def _get_branch_root(self, node: Node) -> Node:
        """Walk up the parent chain to find the branch root (direct child of baseline)."""
        current = node
        while current.parent and current.parent != self.best_stage1_node:
            current = current.parent
        return current

    def _get_subtree_nodes(self, root: Node) -> List[Node]:
        """Collect all nodes in the subtree rooted at root (inclusive)."""
        nodes = [root]
        for child in root.children:
            nodes.extend(self._get_subtree_nodes(child))
        return nodes

    def _get_branch_nodes(self, target_node: Node) -> List[Node]:
        """Get all nodes belonging to the same branch as target_node."""
        branch_root = self._get_branch_root(target_node)
        return self._get_subtree_nodes(branch_root)

    def _check_any_beat_baseline(self, baseline_metric: float) -> bool:
        """Check if any non-stale good node has beaten the baseline metric."""
        baseline_id = self.best_stage1_node.id if self.best_stage1_node else None
        for node in self.journal.good_nodes:
            if node.id == baseline_id:
                continue
            if getattr(node, 'is_precomputed_baseline', False):
                continue
            if getattr(node, 'is_stale', False):
                continue
            if node.metric and node.metric.get_mean_value() > baseline_metric:
                return True
        return False

    def _select_stage2_phase2_depth_first(self, stage2_max_debug: int) -> Optional[Node]:
        """Phase 2: always improve from the global best good node.

        Buggy nodes are ignored — no debug attempts in Phase 2.
        The best node may be non-leaf (already has children); this is
        intentional so that we always branch from the highest-performing
        code rather than chasing worse descendants.
        """
        best_node = self.journal.get_best_node()
        if not best_node:
            return self.best_stage1_node

        logger.info(
            f"Phase 2: improve global best {best_node.id[:8]} "
            f"(metric={best_node.metric}, is_leaf={best_node.is_leaf})"
        )
        return best_node

    def _get_branch_stats(
        self, baseline_metric: float, stage2_max_debug: int,
    ) -> List[dict]:
        """Collect per-branch statistics for hierarchical PUCT.

        Each branch is rooted at a direct child of baseline.  Returns a list
        of dicts: root, n_expansions, q_mean, best_metric, best_leaf.
        """
        import math
        baseline_node = self.best_stage1_node
        if not baseline_node:
            return []

        baseline_id = baseline_node.id
        branch_map: Dict[str, dict] = {}

        all_nodes = list(self.journal.good_nodes)
        try:
            all_nodes.extend(
                n for n in self.journal.buggy_nodes if isinstance(n, Node)
            )
        except Exception:
            pass

        for node in all_nodes:
            if node.id == baseline_id:
                continue
            if getattr(node, 'is_precomputed_baseline', False):
                continue
            if getattr(node, 'is_stale', False):
                continue
            branch_root = self._get_branch_root(node)
            if branch_root.id == baseline_id:
                continue
            rid = branch_root.id

            if rid not in branch_map:
                branch_map[rid] = {
                    "root": branch_root,
                    "n_expansions": branch_root.visit_count,
                    "q_sum": 0.0,
                    "q_count": 0,
                    "best_metric": float('-inf'),
                    "best_leaf": None,
                }
            info = branch_map[rid]
            q = self._node_q(node, baseline_metric)
            info["q_sum"] += q
            info["q_count"] += 1
            m = node.metric.get_mean_value() if node.metric else float('nan')
            if not math.isnan(m) and m > info["best_metric"]:
                info["best_metric"] = m
            if node.is_leaf:
                cur_best = info.get("best_leaf")
                if node.is_buggy and node.debug_depth > stage2_max_debug:
                    continue
                node_m = m if not math.isnan(m) else float('-inf')
                cur_m = (
                    cur_best.metric.get_mean_value()
                    if cur_best and cur_best.metric else float('-inf')
                )
                if cur_best is None or node_m > cur_m:
                    info["best_leaf"] = node

        result = []
        for info in branch_map.values():
            q_mean = info["q_sum"] / max(info["q_count"], 1)
            result.append({
                "root": info["root"],
                "n_expansions": info["n_expansions"],
                "q_mean": q_mean,
                "best_metric": info["best_metric"],
                "best_leaf": info["best_leaf"],
            })
        return result

    def _select_stage2_phase1_ucb(self, stage2_max_debug: int) -> Optional[Node]:
        """Phase 1: branch-level (hierarchical) PUCT selection.

        Each baseline child-branch is one arm.  A special "open new branch"
        action competes with deepening existing branches.  The cost of opening
        new branches grows with the number already opened (N_new = K), so the
        algorithm naturally limits branching as alternatives accumulate.

        Within a chosen branch the best available leaf is selected for
        expansion (debug if buggy, otherwise improve).

        Score_i   = Q_i + c_puct * P(Q_i) * sqrt(N_total) / (1 + N_i)
        Score_new = 0   + c_puct * 1       * sqrt(N_total) / (1 + K)

        P(Q) = max(0, 1+Q)^prior_power   (risk-averse prior, default power=3)
        """
        import math

        ucb_cfg = getattr(self.cfg.experiment.search, 'stage2_ucb', None)
        c_puct = float(getattr(ucb_cfg, 'c_puct', 1.5)) if ucb_cfg else 1.5
        prior_power = int(getattr(ucb_cfg, 'prior_power', 3)) if ucb_cfg else 3

        baseline_metric = (
            self.best_stage1_node.metric.get_mean_value()
            if self.best_stage1_node and self.best_stage1_node.metric
            else 0.0
        )

        branches = self._get_branch_stats(baseline_metric, stage2_max_debug)
        k_branches = len(branches)
        n_total = k_branches + sum(b["n_expansions"] for b in branches)

        n_new = k_branches
        expl_new = c_puct * 1.0 * math.sqrt(max(n_total, 1)) / (1.0 + n_new)
        score_new = 0.0 + expl_new

        logger.debug(
            f"Phase 1 branch-PUCT: K={k_branches} N_total={n_total} "
            f"score_new={score_new:.3f} (N_new={n_new})"
        )

        best_choice = "NEW"
        best_score = score_new
        best_node: Optional[Node] = self.best_stage1_node

        for b in branches:
            q_mean = b["q_mean"]
            n_i = b["n_expansions"]
            prior = max(0.0, 1.0 + q_mean) ** prior_power
            expl_i = c_puct * prior * math.sqrt(max(n_total, 1)) / (1.0 + n_i)
            score_i = q_mean + expl_i
            root_id = b["root"].id[:8]
            logger.debug(
                f"  Branch {root_id}: Q_mean={q_mean:.3f} N={n_i} "
                f"P={prior:.3f} expl={expl_i:.3f} score={score_i:.3f} "
                f"best_metric={b['best_metric']:.4f}"
            )
            if score_i > best_score and b["best_leaf"] is not None:
                best_score = score_i
                best_node = b["best_leaf"]
                best_choice = f"BR-{root_id}"

        if best_node is None:
            logger.warning("Phase 1 branch-PUCT: no candidates, falling back to drafting")
            return None

        if best_choice == "NEW":
            logger.info(
                f"Phase 1 branch-PUCT: open new branch from baseline, "
                f"score={best_score:.3f}, K={k_branches}"
            )
        else:
            node_type = "debug" if best_node.is_buggy else "improve"
            logger.info(
                f"Phase 1 branch-PUCT: deepen {best_node.id[:8]} in {best_choice} "
                f"(type={node_type}, score={best_score:.3f})"
            )
        return best_node

    def _select_next_node(self) -> Optional[Node]:
        """Select the next node to process in serial mode,
        balancing between tree exploration and exploitation.
        Note:
        - This function runs in the main process.
        Some design considerations:
        - For Stage 3 (ablation studies), we generate nodes in the main process.
        This is to make sure we don't run duplicate ablation experiments.
        - For Stage 1 and 2, we also generate nodes in the main process.
        """
        search_cfg = self.cfg.experiment.search
        logger.debug("Serial processing: selecting 1 node")

        logger.debug(
            f"Checking draft nodes... num of journal.draft_nodes: {len(self.journal.draft_nodes)}, search_cfg.num_drafts: {search_cfg.num_drafts}"
        )
        if len(self.journal.draft_nodes) < search_cfg.num_drafts:
            logger.debug("Need more draft nodes, returning None for drafting")
            return None

        # --- Stage 2: Two-phase UCB strategy ---
        if self.stage_name and self.stage_name.startswith("2_"):
            stage2_max_debug = getattr(
                self.cfg.experiment.stage2, 'max_iterations_per_innovation', 10
            )

            # Check if UCB is enabled (default: True)
            ucb_cfg = getattr(search_cfg, 'stage2_ucb', None)
            ucb_enabled = bool(getattr(ucb_cfg, 'enabled', True)) if ucb_cfg else True

            if not ucb_enabled:
                # Fallback to legacy debug-first behavior
                return self._select_stage2_legacy(stage2_max_debug)

            # Determine baseline metric for beat-baseline check
            baseline_metric = (
                self.best_stage1_node.metric.get_mean_value()
                if self.best_stage1_node and self.best_stage1_node.metric
                else 0.0
            )

            # Phase 2: any node has beaten baseline -> lock onto winning branch
            if self._check_any_beat_baseline(baseline_metric):
                logger.debug("Stage 2: Phase 2 (depth-first on winning branch)")
                return self._select_stage2_phase2_depth_first(stage2_max_debug)

            # Phase 1: branch-level PUCT (branch vs deepen)
            logger.debug("Stage 2: Phase 1 (branch-level PUCT)")
            return self._select_stage2_phase1_ucb(stage2_max_debug)

        # --- General debug_prob check (Stage 1 / Stage 3 only) ---
        if random.random() < search_cfg.debug_prob:
            logger.debug("Checking debuggable nodes")
            try:
                debuggable_nodes = [
                    n
                    for n in self.journal.buggy_nodes
                    if (
                        isinstance(n, Node)
                        and n.is_leaf
                        and n.debug_depth <= search_cfg.max_debug_depth
                        and not getattr(n, 'is_stale', False)
                    )
                ]
            except Exception as e:
                logger.error(f"Error getting debuggable nodes: {e}")
                debuggable_nodes = []

            if debuggable_nodes:
                logger.debug("Found debuggable nodes, selecting one for debugging")
                return random.choice(debuggable_nodes)

        # --- Stage 3 (Plan-driven Ablation Studies) ---
        if self.stage_name and self.stage_name.startswith("3_"):
            if not self._current_item_debug_attempted and self.journal.nodes:
                latest = self.journal.nodes[-1]
                if (latest.is_buggy
                        and latest.parent is not None
                        and getattr(latest, 'ablation_name', None)):
                    logger.info(f"Stage 3: returning buggy ablation node {latest.id} for debug")
                    return latest
            return self.best_stage2_node

        # --- Stage 1 (normal best-first search) ---
        good_nodes = self.journal.good_nodes
        if not good_nodes:
            logger.debug("No good nodes found, falling back to drafting")
            return None

        best_node = self.journal.get_best_node()
        logger.debug(f"Selected best node: {best_node.id} (metric: {best_node.metric})")
        return best_node

    def _select_stage2_legacy(self, stage2_max_debug: int) -> Optional[Node]:
        """Legacy Stage 2 selection (used when UCB is disabled).

        1. With debug_prob (default 0.5) probability, debug a random buggy node.
        2. Otherwise, best-first among all good leaves + baseline.
           If nothing beats baseline, baseline is naturally selected every time.
        """
        search_cfg = self.cfg.experiment.search
        debug_prob = getattr(search_cfg, 'debug_prob', 0.5)

        # Collect debuggable buggy leaf nodes
        try:
            debuggable_nodes = [
                n
                for n in self.journal.buggy_nodes
                if (
                    isinstance(n, Node)
                    and n.is_leaf
                    and n.debug_depth <= stage2_max_debug
                    and not getattr(n, 'is_stale', False)
                )
            ]
        except Exception as e:
            logger.error(f"Error getting debuggable nodes for Stage 2: {e}")
            debuggable_nodes = []

        if debuggable_nodes and random.random() < debug_prob:
            selected = random.choice(debuggable_nodes)
            logger.info(
                f"Legacy Stage 2: debug node {selected.id[:8]} "
                f"(prob={debug_prob}, pool={len(debuggable_nodes)})"
            )
            return selected

        # Best-first: all good leaves + baseline
        candidates: List[Node] = []
        baseline_id = self.best_stage1_node.id if self.best_stage1_node else None
        for n in self.journal.good_nodes:
            if (n.id != baseline_id
                    and not getattr(n, 'is_precomputed_baseline', False)
                    and not getattr(n, 'is_stale', False)
                    and n.is_leaf):
                candidates.append(n)

        if self.best_stage1_node:
            candidates.append(self.best_stage1_node)

        if candidates:
            selected = max(candidates, key=lambda n: n.metric)
            node_type = "baseline" if selected == self.best_stage1_node else "improve"
            logger.info(
                f"Legacy Stage 2: best-first {node_type} {selected.id[:8]} "
                f"(metric={selected.metric}, candidates={len(candidates)})"
            )
            return selected

        return None

    def step(self):
        logger.debug("Selecting node to process (serial mode)")
        node = self._select_next_node()  # This now returns a single node
        logger.debug(f"Selected node: {node.id if node else None}")

        # Increment N_i on the branch root for hierarchical PUCT.
        # For deepen (node is an existing branch leaf): increment branch root.
        # For NEW (node == baseline): do nothing here; the new branch root's
        # visit_count will be set to 1 after processing (see below).
        _is_new_branch_action = False
        if node is not None:
            if (self.stage_name and self.stage_name.startswith("2_")
                    and self.best_stage1_node
                    and node != self.best_stage1_node):
                branch_root = self._get_branch_root(node)
                branch_root.visit_count = getattr(branch_root, 'visit_count', 0) + 1
            elif (self.stage_name and self.stage_name.startswith("2_")
                    and self.best_stage1_node
                    and node == self.best_stage1_node):
                _is_new_branch_action = True

        # Process single node (serial processing)
        if node is None:
            logger.info("No node selected, will draft new node")
        else:
            logger.debug(f"Selected node {node.id} for processing")
        memory_summary = self.journal.generate_summary(include_code=False)

        logger.debug("Processing node in serial mode")
        result_node = None
        try:
            # Determine what type of processing is needed
            current_plan_item = None
            if self.stage_name and self.stage_name.startswith("3_") and node is not None:
                # --- Stage 3 plan-driven ablation logic ---
                if node.is_buggy:
                    # _select_next_node returned a buggy ablation node → debug attempt
                    self._current_item_debug_attempted = True
                    self.journal._current_item_debug_attempted = True
                    current_plan_item = None  # will enter debug branch
                    logger.info("Stage 3: ablation buggy, attempting debug once")
                else:
                    # node is best_stage2_node (not buggy)
                    if self._current_item_debug_attempted:
                        # Previous debug attempt was done (success or fail already handled).
                        # Reset flag and move on to next plan item.
                        self._current_item_debug_attempted = False
                        self.journal._current_item_debug_attempted = False
                    if self._ablation_plan_idx < len(self._ablation_plan):
                        current_plan_item = self._ablation_plan[self._ablation_plan_idx]
                        self._ablation_plan_idx += 1
                        self.journal._ablation_plan_idx = self._ablation_plan_idx
                        logger.info(f"Stage 3: executing plan item "
                                    f"[{self._ablation_plan_idx}/{len(self._ablation_plan)}] "
                                    f"{current_plan_item.name}")
                    else:
                        self.journal.ablation_plan_complete = True
                        self.journal._ablation_plan_idx = self._ablation_plan_idx
                        current_plan_item = None
                        logger.info("Stage 3: ablation plan complete")

            # Process the node directly (serial processing)
            result_node = self._process_node_serial(
                node, memory_summary, current_plan_item
            )

            if result_node:
                # For NEW branch action, set the new branch root's N_i = 1
                # (creation counts as the first expansion allocated to this branch).
                if _is_new_branch_action:
                    result_node.visit_count = 1

                # Check if the result_node is already marked as buggy from _process_node_serial
                if result_node.is_buggy:
                    logger.debug(f"Processing returned buggy node {result_node.id} with actual executed code")
                    logger.info(f"Buggy node {result_node.id} returned from _process_node_serial with actual code")
                    
                    # Save the actual executed code that failed (not parent node code!)
                    if hasattr(result_node, 'code') and result_node.code:
                        self._save_code_to_stage_directory(
                            code=result_node.code,
                            node_id=result_node.id,
                            stage_name=self.stage_name or "unknown",
                            code_type="buggy_experiment"
                        )
                        logger.debug(f"Saved actual executed code for buggy node {result_node.id}")
                    
                    # Add buggy node to journal
                    self.journal.append(result_node)
                    logger.debug(f"Added buggy node {result_node.id} to journal")
                else:
                    # Normal successful processing
                    # Save the generated code to stage directory
                    if hasattr(result_node, 'code') and result_node.code:
                        self._save_code_to_stage_directory(
                            code=result_node.code,
                            node_id=result_node.id,
                            stage_name=self.stage_name or "unknown",
                            code_type="experiment"
                        )

                    # Update states
                    self._update_ablation_state(result_node)

                    # Add node to journal
                    self.journal.append(result_node)
                    logger.debug("Added result node to journal")
            else:
                # _process_node_serial returned None (failure before child_node creation)
                logger.warning("_process_node_serial returned None, creating fallback buggy node")
                
                # Create fallback buggy node with parent code (best we can do)
                result_node = Node()
                result_node.parent = node
                if node:
                    result_node.parent_id = node.id
                    result_node.code = getattr(node, 'code', '')
                    result_node.plan = f"Failed to process node {node.id} before code generation"
                else:
                    result_node.code = ''
                    result_node.plan = "Failed to draft new node before code generation"
                
                result_node.is_buggy = True
                result_node.exc_type = "ProcessingFailure"
                result_node.exc_info = {"error": "Processing failed before code generation", "type": "ProcessingFailure"}
                result_node.metric = WorstMetricValue()
                result_node.analysis = "Processing failed before code generation"
                
                # Save fallback code if available
                if hasattr(result_node, 'code') and result_node.code:
                    self._save_code_to_stage_directory(
                        code=result_node.code,
                        node_id=result_node.id,
                        stage_name=self.stage_name or "unknown",
                        code_type="buggy_experiment"
                    )
                
                self.journal.append(result_node)
                logger.debug(f"Added fallback buggy node {result_node.id} to journal")

        except Exception as e:
            logger.error(f"Unexpected error in step processing: {str(e)}")
            import traceback
            error_traceback = traceback.format_exc()
            traceback.print_exc()
            
            # This should rarely happen now since _process_node_serial handles its own exceptions
            logger.error("Creating emergency buggy node for unexpected error")
            
            result_node = Node()
            result_node.parent = node
            if node:
                result_node.parent_id = node.id
                result_node.code = getattr(node, 'code', '')
                result_node.plan = f"Emergency fallback for node {node.id}"
            else:
                result_node.code = ''
                result_node.plan = "Emergency fallback for new node"
            
            result_node.is_buggy = True
            result_node.exc_type = type(e).__name__
            result_node.exc_info = {"error": str(e), "type": type(e).__name__}
            result_node.exc_stack = error_traceback
            result_node._term_out = [f"Error: {str(e)}", error_traceback]
            result_node.metric = WorstMetricValue()
            result_node.analysis = f"Emergency error handling: {type(e).__name__}: {str(e)}"
            
            # Save emergency fallback code
            if hasattr(result_node, 'code') and result_node.code:
                self._save_code_to_stage_directory(
                    code=result_node.code,
                    node_id=result_node.id,
                    stage_name=self.stage_name or "unknown",
                    code_type="buggy_experiment"
                )
            
            self.journal.append(result_node)
            logger.error(f"Emergency buggy node {result_node.id} added to journal")

    def _generate_modification_summary(self, child_node: Node, parent_node: Optional[Node]) -> str:
        """Generate a 2-3 sentence LLM summary of what changed relative to the parent node.

        Uses difflib to compute a concise code diff, then asks the feedback LLM
        to produce a human-readable modification summary.  Falls back gracefully
        on any error.
        """
        import difflib

        try:
            parent_code = (parent_node.code or "") if parent_node else ""
            child_code = child_node.code or ""

            if not child_code.strip():
                return "No code generated."

            if not parent_code.strip():
                # Draft mode – summarise the whole implementation
                code_snippet = child_code[:3000]
                user_msg = (
                    "A new experiment script was created from scratch. "
                    "Summarise the implementation approach in 2-3 sentences. "
                    "Focus on: model architecture, key modules, training strategy.\n\n"
                    f"```python\n{code_snippet}\n```"
                )
            else:
                diff_lines = list(difflib.unified_diff(
                    parent_code.splitlines(keepends=True),
                    child_code.splitlines(keepends=True),
                    lineterm="",
                    n=1,
                ))
                if not diff_lines:
                    return "No code changes detected."

                diff_text = "\n".join(diff_lines[:150])
                if len(diff_lines) > 150:
                    diff_text += f"\n... ({len(diff_lines) - 150} more diff lines omitted)"

                parent_context = ""
                if parent_node:
                    prev_mod = getattr(parent_node, 'modification_summary', '') or parent_node.plan or ""
                    if prev_mod:
                        parent_context = f"Parent approach: {prev_mod[:300]}\n\n"

                user_msg = (
                    f"{parent_context}"
                    "Below is the unified diff of this code modification. "
                    "Write a 2-3 sentence summary of what was changed and why. "
                    "Focus on: architectural changes, new/removed modules, hyperparameter changes, bug fixes.\n\n"
                    f"```diff\n{diff_text}\n```"
                )

            mod_summary_spec = FunctionSpec(
                name="modification_summary",
                description="Summarise code changes in 2-3 sentences",
                json_schema={
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "2-3 sentence modification summary",
                        }
                    },
                    "required": ["summary"],
                },
            )

            response = cast(dict, query(
                system_message=None,
                user_message=user_msg,
                func_spec=mod_summary_spec,
                **{"model": get_role("feedback")["model"]},
                temperature=0.3,
            ))
            summary = response.get("summary", "")
            if summary:
                return summary[:500]
            return "Summary generation returned empty result."

        except Exception as e:
            logger.warning(f"Failed to generate modification summary: {e}")
            return f"Summary generation failed: {type(e).__name__}"

    def _process_node_serial(self, node, memory_summary, current_plan_item=None):
        """Process a single node in serial mode (no parallel processing)"""
        from .interpreter import Interpreter

        # Check if subagent competition is enabled for Stage 2
        candidates = list(getattr(self.cfg.experiment.code, 'candidates', []) or [])
        use_competition = (
            len(candidates) > 1
            and self.stage_name
            and self.stage_name.startswith("2_")
            and current_plan_item is None
        )

        if use_competition:
            return self._process_node_with_subagent_competition(
                node, memory_summary, candidates
            )

        logger.info(f"Processing node in serial mode: {node.id if node else 'new draft'}")

        # Create minimal agent for processing
        processing_agent = MinimalAgent(
            task_desc=self.task_desc,
            cfg=self.cfg,
            memory_summary=memory_summary,
            evaluation_metrics=self.evaluation_metrics,
            stage=self.current_stage,  # 🔧 Pass the Stage object so proposal_idx is accessible
            stage_name=self.stage_name,
            agent_manager=self.agent_manager,  # Pass agent_manager for innovation queue access
            exp_name=self.exp_name,  # Pass exp_name for aider code generation
        )

        # Stage 3 comparison experiments run up to 5 variants sequentially
        exec_timeout = self.cfg.exec.timeout
        if (current_plan_item is not None
                and getattr(current_plan_item, 'ablation_type', '') == "comparison"):
            exec_timeout = self.cfg.exec.timeout * 5
            logger.info(f"Stage 3 comparison: timeout x5 -> {exec_timeout}s ({exec_timeout / 3600:.1f}h)")

        # Create interpreter instance for this node processing
        process_interpreter = Interpreter(
            working_dir=self.cfg.workspace_dir,
            timeout=exec_timeout,
            format_tb_ipython=self.cfg.exec.format_tb_ipython,
            agent_file_name=self.cfg.exec.agent_file_name,
            use_conda=getattr(self.cfg.exec, 'use_conda', False),
            conda_env=getattr(self.cfg.exec, 'conda_env', 'py310'),
        )

        child_node = None  # Initialize child_node to track actual execution
        try:
            # Process the node using processing agent
            logger.debug("Starting node processing")
            if node is None:
                logger.info("Drafting new node")
                child_node = processing_agent._draft()
            elif node.is_buggy:
                logger.info(f"Debugging node with id: {node.id}")
                child_node = processing_agent._debug(node)
                child_node.parent = node
                child_node.visit_count = getattr(node, 'visit_count', 0)
                if child_node.parent:
                    child_node.parent_id = child_node.parent.id
            else:
                if current_plan_item is not None:
                    # Stage 3 (plan-driven ablation study)
                    child_node = processing_agent._generate_ablation_node(
                        node, current_plan_item
                    )
                    child_node.parent = node
                    child_node.visit_count = getattr(node, 'visit_count', 0)
                    logger.info(f"Processing ablation study: {child_node.ablation_name}")
                    logger.info(f"Running ablation study: {child_node.ablation_name}")
                else:
                    # Regular improvement
                    logger.info(f"Improving node with id: {node.id}")
                    child_node = processing_agent._improve(node)
                    child_node.parent = node
                    child_node.visit_count = getattr(node, 'visit_count', 0)
                    if child_node.parent:
                        child_node.parent_id = child_node.parent.id

            # Generate modification summary (LLM-based diff analysis)
            child_node.modification_summary = self._generate_modification_summary(child_node, node)
            logger.info(f"Modification summary for node {child_node.id}: {child_node.modification_summary[:120]}")

            # Execute and parse results (like original implementation)
            
            # Skip execution if node is already marked as buggy (e.g., OpenHands failed to generate code)
            if child_node.is_buggy:
                logger.info(f"⏭️ Skipping execution for pre-marked buggy node {child_node.id}")
                logger.warning(f"Skipping execution: Node {child_node.id} is already marked buggy (code generation failed)")
                # Ensure the node has worst metric (should already be set, but be safe)
                if child_node.metric is None:
                    child_node.metric = WorstMetricValue()
                return child_node
            
            # CLEANUP: Remove experiment_data.npy before starting new execution
            # This prevents failed runs from picking up metrics from previous successful runs
            working_dir = os.path.join(self.cfg.workspace_dir, "working")
            exp_data_path = os.path.join(working_dir, "experiment_data.npy")
            if os.path.exists(exp_data_path):
                try:
                    os.remove(exp_data_path)
                    logger.debug("Cleaned up stale experiment_data.npy from previous run")
                except Exception as e:
                    logger.warning(f"Failed to clean up experiment_data.npy: {e}")
            
            logger.info("Running code")
            exec_result = process_interpreter.run_with_escape_hatch(child_node.code, True)

            logger.info("Parsing execution results")
            processing_agent.parse_exec_result(child_node, exec_result, self.cfg.workspace_dir)
            
            # Add check for saved data files
            data_files = [f for f in os.listdir(os.path.join(self.cfg.workspace_dir, "working")) if f.endswith(".npy")]
            logger.debug(f"data_files: {data_files}")
            if data_files:
                logger.debug(f"Found data files: {data_files}")

                # Use fixed template for metrics extraction (replacing LLM-generated parsing code)
                logger.debug(f"Using fixed template for metrics extraction for node {child_node.id}")
                logger.debug(f"Extracting metrics using fixed template for node {child_node.id}")
                
                try:
                    # Use the fixed template method to extract metrics directly
                    metrics_response = self._extract_metrics_with_fixed_template(child_node)
                    
                    logger.debug(f"Fixed Template Metrics: {metrics_response}")
                    
                    if metrics_response["valid_metrics_received"]:
                        child_node.metric = MetricValue(
                            value={"metric_names": metrics_response["metric_names"]}
                        )
                        logger.info(f"Successfully extracted metrics using fixed template for node {child_node.id}")
                        
                        # Post-experiment file handling: move result_folder and copy experiment_data.npy
                        try:
                            self._handle_post_experiment_files(child_node)
                        except Exception as post_exp_e:
                            logger.warning(f"Failed to handle post-experiment files for node {child_node.id}: {post_exp_e}")
                    else:
                        child_node.metric = WorstMetricValue()
                        child_node.is_buggy = True
                        logger.warning(f"No valid metrics received using fixed template for node {child_node.id}")

                except Exception as metrics_e:
                    logger.error(f"Error extracting metrics using fixed template for node {child_node.id}: {str(metrics_e)}")
                    child_node.metric = WorstMetricValue()
                    child_node.is_buggy = True
            else:
                # No data files found, set worst metric
                logger.warning(f"No data files found for node {child_node.id}")
                child_node.metric = WorstMetricValue()

            return child_node

        except Exception as e:
            import traceback
            # If child_node was created but execution failed, mark it as buggy and return it
            # This preserves the actual executed code for debugging
            if child_node is not None:
                logger.error(f"Execution failed for child_node {child_node.id}, marking as buggy")
                logger.error(f"Error in serial node processing for child_node {child_node.id}: {e}")
                
                # Mark as buggy and preserve error information
                child_node.is_buggy = True
                child_node.exc_type = type(e).__name__
                child_node.exc_info = {"error": str(e), "type": type(e).__name__}
                child_node.exc_stack = traceback.format_exc()
                child_node._term_out = [f"Error: {str(e)}", traceback.format_exc()]
                child_node.metric = WorstMetricValue()
                child_node.analysis = f"Experiment failed with error: {type(e).__name__}: {str(e)}"
                
                # This child_node contains the actual executed code that failed
                return child_node
            else:
                # If child_node creation failed, we can't preserve the code
                logger.error(f"Error in serial node processing before child_node creation: {e}")
                logger.error(f"Error in serial node processing before child_node creation: {e}")
                return None

        finally:
            if 'process_interpreter' in locals():
                process_interpreter.cleanup_session()
        

    def _process_node_with_subagent_competition(self, node, memory_summary, candidate_endpoints):
        """Run multiple LLM subagents on the same prompt, execute each, pick best result.

        Each subagent sequentially: generates code -> executes -> extracts metrics.
        Workspace is restored to pre-competition state before each subagent to ensure
        truly independent code generation.

        candidate_endpoints: List[str] — endpoint names from experiment.code.candidates.
        """
        import numpy as np
        from .interpreter import Interpreter

        num_subagents = len(candidate_endpoints)
        logger.info(
            f"{'='*60}\n"
            f"  SUBAGENT COMPETITION: {num_subagents} subagents competing\n"
            f"  Node: {node.id if node else 'new draft'}\n"
            f"{'='*60}"
        )

        candidates = []
        all_candidate_info = []  # Track all subagent attempts for memory
        all_child_nodes = []  # Track all created child nodes for ghost cleanup

        # Derive workspace state from node's actual code, NOT from filesystem
        # leftovers.  Previous steps may leave stale experiment.py that does not
        # belong to the current node (e.g. PUCT chose NEW branch from baseline
        # but workspace still has the previous step's winner code).
        if self.stage_name:
            _stage_logs = Path(self.cfg.log_dir) / f"stage_{self.stage_name}"
        else:
            _stage_logs = Path(self.cfg.log_dir) / "stage_default"
        oh_workspace_dir = _stage_logs / "openhands_workspace"
        oh_workspace_dir.mkdir(parents=True, exist_ok=True)
        _exp_file = oh_workspace_dir / "experiment.py"
        _test_file = oh_workspace_dir / "test.py"

        _is_precomputed = (
            node is not None
            and (getattr(node, 'is_precomputed_baseline', False)
                 or node.code is None
                 or node.code.strip() == "")
        )
        if node is None or _is_precomputed:
            # Draft / NEW-branch-from-baseline: clean workspace
            _snapshot_exp_exists = False
            _snapshot_exp_content = None
            if _exp_file.exists():
                _exp_file.unlink()
        else:
            # Improve / Debug: workspace should contain the node's code
            _snapshot_exp_exists = True
            _snapshot_exp_content = node.code
            _exp_file.write_text(node.code)

        logger.info(
            f"Workspace snapshot: experiment.py "
            f"{'exists (from node code)' if _snapshot_exp_exists else 'absent (clean for draft/NEW)'}"
        )

        original_candidates = list(self.cfg.experiment.code.candidates)
        for idx, endpoint_name in enumerate(candidate_endpoints):
            label = endpoint_name  # label = endpoint name in new schema
            ep_info = get_endpoint(endpoint_name)
            model_name = ep_info.get("model", "")

            logger.info(
                f"--- Subagent [{idx+1}/{num_subagents}] {label} "
                f"(model: {model_name}) ---"
            )

            # Restore workspace to pre-competition snapshot so each subagent
            # starts from an identical, clean state.
            try:
                if _snapshot_exp_exists:
                    _exp_file.write_text(_snapshot_exp_content)
                    logger.info(f"{label}: restored experiment.py to parent code")
                elif _exp_file.exists():
                    _exp_file.unlink()
                    logger.info(f"{label}: deleted stale experiment.py (draft mode)")
                if _test_file.exists():
                    _test_file.unlink()
                    logger.info(f"{label}: deleted stale test.py")
            except Exception as restore_err:
                logger.warning(f"{label}: workspace restore failed: {restore_err}")

            # Temporarily force candidates to single element so downstream readers
            # (openhands_coder) pick the current subagent's endpoint.
            self.cfg.experiment.code.candidates = [endpoint_name]

            child_node = None
            process_interpreter = None
            try:
                # 1. Code generation (reuse existing _improve/_debug/_draft)
                processing_agent = MinimalAgent(
                    task_desc=self.task_desc,
                    cfg=self.cfg,
                    memory_summary=memory_summary,
                    evaluation_metrics=self.evaluation_metrics,
                    stage=self.current_stage,
                    stage_name=self.stage_name,
                    agent_manager=self.agent_manager,
                    exp_name=self.exp_name,
                )

                if node is None:
                    child_node = processing_agent._draft()
                elif node.is_buggy:
                    child_node = processing_agent._debug(node)
                    child_node.parent = node
                    child_node.visit_count = 0
                    if child_node.parent:
                        child_node.parent_id = child_node.parent.id
                else:
                    child_node = processing_agent._improve(node)
                    child_node.parent = node
                    child_node.visit_count = 0
                    if child_node.parent:
                        child_node.parent_id = child_node.parent.id

                # Generate modification summary before execution
                child_node.modification_summary = self._generate_modification_summary(child_node, node)
                logger.info(f"{label} modification: {child_node.modification_summary[:120]}")

                # Skip if code generation already failed
                if child_node.is_buggy:
                    logger.warning(f"{label}: code generation failed, skipping execution")
                    if child_node.metric is None:
                        child_node.metric = WorstMetricValue()
                    all_candidate_info.append({
                        "label": label,
                        "modification_summary": getattr(child_node, 'modification_summary', ''),
                        "metric": str(child_node.metric) if child_node.metric else "N/A",
                        "is_buggy": True,
                        "exc_type": child_node.exc_type,
                    })
                    continue

                # 2. Clean up stale experiment_data.npy
                working_dir = os.path.join(self.cfg.workspace_dir, "working")
                exp_data_path = os.path.join(working_dir, "experiment_data.npy")
                if os.path.exists(exp_data_path):
                    try:
                        os.remove(exp_data_path)
                    except Exception:
                        pass

                # 3. Execute code
                process_interpreter = Interpreter(
                    working_dir=self.cfg.workspace_dir,
                    timeout=self.cfg.exec.timeout,
                    format_tb_ipython=self.cfg.exec.format_tb_ipython,
                    agent_file_name=self.cfg.exec.agent_file_name,
                    use_conda=getattr(self.cfg.exec, 'use_conda', False),
                    conda_env=getattr(self.cfg.exec, 'conda_env', 'py310'),
                )

                logger.info(f"{label}: Running code...")
                exec_result = process_interpreter.run_with_escape_hatch(
                    child_node.code, True
                )
                processing_agent.parse_exec_result(
                    child_node, exec_result, self.cfg.workspace_dir
                )

                # 4. Extract metrics
                data_files = [
                    f for f in os.listdir(working_dir)
                    if f.endswith(".npy")
                ] if os.path.isdir(working_dir) else []

                if data_files:
                    metrics_response = self._extract_metrics_with_fixed_template(
                        child_node
                    )
                    if metrics_response["valid_metrics_received"]:
                        child_node.metric = MetricValue(
                            value={"metric_names": metrics_response["metric_names"]}
                        )
                        child_node.subagent_label = label
                        candidates.append(child_node)
                        logger.info(
                            f"{label}: metric = {child_node.metric}"
                        )
                        try:
                            self._handle_post_experiment_files(child_node)
                        except Exception as post_e:
                            logger.warning(
                                f"{label}: post-experiment file handling failed: {post_e}"
                            )
                    else:
                        child_node.metric = WorstMetricValue()
                        child_node.is_buggy = True
                        logger.warning(f"{label}: no valid metrics received")
                else:
                    child_node.metric = WorstMetricValue()
                    logger.warning(f"{label}: no data files found")

            except Exception as e:
                import traceback
                logger.error(f"{label} failed: {e}")
                logger.error(traceback.format_exc())
                if child_node is not None:
                    child_node.is_buggy = True
                    child_node.metric = WorstMetricValue()
                    child_node.exc_type = type(e).__name__
                    child_node.exc_info = {"error": str(e), "type": type(e).__name__}
            finally:
                self.cfg.experiment.code.candidates = original_candidates
                if process_interpreter is not None:
                    process_interpreter.cleanup_session()
                if child_node is not None:
                    all_child_nodes.append(child_node)
                # Collect candidate info for memory (after try/except resolves)
                if child_node is not None:
                    info = {
                        "label": label,
                        "modification_summary": getattr(child_node, 'modification_summary', ''),
                        "metric": str(child_node.metric) if child_node.metric else "N/A",
                        "is_buggy": child_node.is_buggy if child_node.is_buggy is not None else True,
                        "exc_type": child_node.exc_type,
                    }
                    # Avoid duplicates from the continue branch
                    if not any(c["label"] == label for c in all_candidate_info):
                        all_candidate_info.append(info)

        # --- Select the best candidate ---
        valid_candidates = [c for c in candidates if c.metric and not c.is_buggy]

        if valid_candidates:
            best = max(valid_candidates, key=lambda n: n.metric)
            winner_label = getattr(best, 'subagent_label', '?')

            # Attach non-winner candidate summaries for memory
            best.competition_siblings = [
                c for c in all_candidate_info if c["label"] != winner_label
            ]

            logger.info(
                f"{'='*60}\n"
                f"  COMPETITION WINNER: {winner_label}\n"
                f"  Metric: {best.metric}\n"
                f"  Candidates evaluated: {len(valid_candidates)}/{num_subagents}\n"
                f"  Siblings recorded: {len(best.competition_siblings)}\n"
                f"{'='*60}"
            )
            # Remove non-winner child nodes from parent's children set.
            # Node.__post_init__ auto-registers every child into parent.children,
            # but only the winner should remain to avoid ghost nodes polluting
            # _get_subtree_nodes and the tree search.
            if node is not None:
                for child in all_child_nodes:
                    if child is not best:
                        node.children.discard(child)
            return best
        else:
            # Pick one actual failed node to preserve real code + error info
            # for subsequent debug attempts, instead of a synthetic empty fallback.
            import random
            chosen = random.choice(all_child_nodes) if all_child_nodes else None

            if chosen is not None:
                logger.warning(
                    f"All subagents failed — returning actual failed node "
                    f"{chosen.id[:8]} for debugging"
                )
                if chosen.is_buggy is None:
                    chosen.is_buggy = True
                if chosen.metric is None:
                    chosen.metric = WorstMetricValue()
                chosen.competition_siblings = [
                    c for c in all_candidate_info
                    if c.get("label") != getattr(chosen, 'subagent_label', None)
                ]
                if node is not None:
                    for child in all_child_nodes:
                        if child is not chosen:
                            node.children.discard(child)
                return chosen
            else:
                logger.warning("All subagents failed — no child nodes created, returning synthetic fallback")
                fallback = Node()
                fallback.parent = node
                if node:
                    fallback.parent_id = node.id
                    fallback.code = getattr(node, 'code', '')
                else:
                    fallback.code = ''
                fallback.plan = "BUGGY: All subagent competition candidates failed"
                fallback.is_buggy = True
                fallback._term_out = ["All subagent competition candidates failed.\n"]
                fallback.metric = WorstMetricValue()
                return fallback

    def _update_ablation_state(self, result_node: Node):
        """Log ablation study progress (plan-driven tracking is in step())."""
        if not self.stage_name or not self.stage_name.startswith("3_"):
            return
        ablation_name = getattr(result_node, 'ablation_name', None)
        if ablation_name and not result_node.is_buggy:
            logger.info(f"Ablation study '{ablation_name}' completed successfully "
                        f"(plan progress: {self._ablation_plan_idx}/{len(self._ablation_plan)})")

    def _extract_from_dataset_data(self, dataset_data, dice_score, hd95_score):
        """
        Extract dice_score and hd95_score from a single dataset dict.
        
        Args:
            dataset_data: dataset dict containing metrics/dice_scores/hd95_scores.
            dice_score: dice_score already extracted so far (may be None).
            hd95_score: hd95_score already extracted so far (may be None).
            
        Returns:
            tuple: (dice_score, hd95_score)
        """
        # Method 1: extract the latest metric from metrics['val'] (expects val to be a list like [{'dice': x, 'hd95': y}])
        if 'metrics' in dataset_data and 'val' in dataset_data['metrics']:
            val_metrics = dataset_data['metrics']['val']
            if val_metrics:
                latest_metrics = val_metrics[-1]
                if isinstance(latest_metrics, dict):
                    if dice_score is None:
                        dice_score = next((latest_metrics[k] for k in ('dice', 'Dice', 'DICE') if latest_metrics.get(k) is not None), None)
                    if hd95_score is None:
                        hd95_score = next((latest_metrics[k] for k in ('hd95', 'HD95', 'Hausdorff Distance 95') if latest_metrics.get(k) is not None), None)
        
        if dice_score is None:
            scores = dataset_data.get('dice_scores')
            if scores is not None and len(scores) > 0:
                dice_score = scores[-1]
                
        if hd95_score is None:
            scores = dataset_data.get('hd95_scores')
            if scores is not None and len(scores) > 0:
                hd95_score = scores[-1]
        
        return dice_score, hd95_score

    def _extract_metrics_with_fixed_template(self, node):
        """
        Read saved metrics directly from experiment_data.npy to avoid calling camylanet.evaluate() again.
        
        Args:
            node: current experiment node.
            
        Returns:
            dict: extracted metric info with the same shape as the LLM parse result.
        """
        import numpy as np
        import os
        
        # Read experiment_data.npy for the saved metrics
        working_dir = os.path.join(self.cfg.workspace_dir, 'working')
        experiment_data_path = os.path.join(working_dir, 'experiment_data.npy')
        
        if not os.path.exists(experiment_data_path):
            logger.warning(f"experiment_data.npy not found for node {node.id} at {experiment_data_path}")
            return {"valid_metrics_received": False, "metric_names": []}
            
        try:
            experiment_data = np.load(experiment_data_path, allow_pickle=True).item()
            logger.debug(f"Extracting metrics directly from experiment_data.npy for node {node.id}")
            
            # Extract metrics from the saved data
            is_comparison = getattr(node, 'ablation_type', '') == 'comparison'

            if is_comparison:
                # comparison ablation: collect metrics from all variants
                all_dice_data = []
                all_hd95_data = []
                for dataset_key in experiment_data:
                    if not isinstance(experiment_data[dataset_key], dict):
                        continue
                    dataset_data = experiment_data[dataset_key]
                    v_dice, v_hd95 = None, None
                    is_flat = 'metrics' in dataset_data or 'dice_scores' in dataset_data
                    if is_flat:
                        v_dice, v_hd95 = self._extract_from_dataset_data(dataset_data, None, None)
                    else:
                        for nested_key in dataset_data:
                            if isinstance(dataset_data[nested_key], dict):
                                v_dice, v_hd95 = self._extract_from_dataset_data(
                                    dataset_data[nested_key], v_dice, v_hd95)
                                if v_dice is not None and v_hd95 is not None:
                                    break
                    variant_name = dataset_key.split('__', 1)[1] if '__' in dataset_key else dataset_key
                    if v_dice is not None and v_hd95 is not None:
                        if isinstance(v_dice, (int, float)) and isinstance(v_hd95, (int, float)):
                            all_dice_data.append({
                                "dataset_name": variant_name,
                                "final_value": float(v_dice),
                                "best_value": float(v_dice),
                            })
                            all_hd95_data.append({
                                "dataset_name": variant_name,
                                "final_value": float(v_hd95),
                                "best_value": float(v_hd95),
                            })

                if all_dice_data:
                    avg_dice = np.mean([d["final_value"] for d in all_dice_data])
                    avg_hd95 = np.mean([d["final_value"] for d in all_hd95_data])
                    metrics_response = {
                        "valid_metrics_received": True,
                        "metric_names": [
                            {
                                "metric_name": "Dice Score",
                                "lower_is_better": False,
                                "description": f"Dice Similarity Coefficient (avg: {avg_dice:.4f})",
                                "data": all_dice_data,
                            },
                            {
                                "metric_name": "HD95 Score",
                                "lower_is_better": True,
                                "description": f"95% Hausdorff Distance (avg: {avg_hd95:.4f})",
                                "data": all_hd95_data,
                            },
                        ],
                    }
                    variant_names = [d["dataset_name"] for d in all_dice_data]
                    logger.info(
                        f"Comparison ablation: extracted {len(all_dice_data)} variants "
                        f"({variant_names}) avg Dice={avg_dice:.4f}, HD95={avg_hd95:.4f}"
                    )
                    return metrics_response
                else:
                    logger.warning(f"No valid variant metrics found for comparison node {node.id}")
                    logger.debug(f"Available keys in experiment_data: {list(experiment_data.keys())}")
                    return {"valid_metrics_received": False, "metric_names": []}

            # Non-comparison (removal / normal node): pick the first valid metric
            dice_score = None
            hd95_score = None

            for dataset_key in experiment_data:
                if isinstance(experiment_data[dataset_key], dict):
                    dataset_data = experiment_data[dataset_key]

                    # Check whether this is the flat layout (Stage 1-3),
                    # i.e. top-level keys directly include metrics/dice_scores
                    is_flat_structure = 'metrics' in dataset_data or 'dice_scores' in dataset_data

                    if is_flat_structure:
                        # Flat layout: extract directly from dataset_data
                        dice_score, hd95_score = self._extract_from_dataset_data(dataset_data, dice_score, hd95_score)
                    else:
                        # Nested layout (legacy format): descend one level
                        for nested_key in dataset_data:
                            if isinstance(dataset_data[nested_key], dict):
                                nested_data = dataset_data[nested_key]
                                dice_score, hd95_score = self._extract_from_dataset_data(nested_data, dice_score, hd95_score)
                                if dice_score is not None and hd95_score is not None:
                                    break

                    # Break once metrics are found
                    if dice_score is not None and hd95_score is not None:
                        break

            # Check whether metrics were successfully extracted
            if dice_score is not None and hd95_score is not None:
                # Ensure metric values are valid numbers
                if not isinstance(dice_score, (int, float)) or not isinstance(hd95_score, (int, float)):
                    logger.warning(f"Invalid metric types for node {node.id}: dice={type(dice_score)}, hd95={type(hd95_score)}")
                    return {"valid_metrics_received": False, "metric_names": []}

                metrics_response = {
                    "valid_metrics_received": True,
                    "metric_names": [
                        {
                            "metric_name": "Dice Score",
                            "lower_is_better": False,
                            "description": f"Dice Similarity Coefficient (avg: {dice_score:.4f})",
                            "data": [
                                {
                                    "dataset_name": "msd_segmentation",
                                    "final_value": float(dice_score),
                                    "best_value": float(dice_score)
                                }
                            ]
                        },
                        {
                            "metric_name": "HD95 Score",
                            "lower_is_better": True,
                            "description": f"95% Hausdorff Distance (avg: {hd95_score:.4f})",
                            "data": [
                                {
                                    "dataset_name": "msd_segmentation",
                                    "final_value": float(hd95_score),
                                    "best_value": float(hd95_score)
                                }
                            ]
                        }
                    ]
                }

                logger.info(f"Successfully extracted metrics from experiment_data: Dice={dice_score:.4f}, HD95={hd95_score:.4f}")
                return metrics_response
            else:
                logger.warning(f"No valid metrics found in experiment_data.npy for node {node.id}")
                logger.debug(f"Available keys in experiment_data: {list(experiment_data.keys())}")
                return {"valid_metrics_received": False, "metric_names": []}
                
        except Exception as e:
            logger.error(f"Error extracting metrics from experiment_data.npy for node {node.id}: {e}")
            return {"valid_metrics_received": False, "metric_names": []}

    def _handle_post_experiment_files(self, node):
        """
        Handle the file moves/copies after a successful experiment.
        
        Args:
            node: current experiment node.
        """
        import numpy as np
        
        # Read experiment_data.npy to get the result_folder path
        working_dir = os.path.join(self.cfg.workspace_dir, 'working')
        experiment_data_path = os.path.join(working_dir, 'experiment_data.npy')
        
      
        if not os.path.exists(experiment_data_path):
            logger.warning(f"experiment_data.npy not found for node {node.id} at {experiment_data_path}")
            return
            
        try:
            experiment_data = np.load(experiment_data_path, allow_pickle=True).item()
            
            # Extract the result_folder path
            result_folder = None
            for dataset_key in experiment_data:
                if isinstance(experiment_data[dataset_key], dict) and 'result_folder' in experiment_data[dataset_key]:
                    result_folder = experiment_data[dataset_key]['result_folder']
                    break
            
            if not result_folder or not os.path.exists(result_folder):
                logger.warning(f"result_folder not found or doesn't exist for node {node.id}: {result_folder}")
                return
            
            # Extract exp_name from the result_folder path
            exp_name = self._extract_exp_name_from_result_folder(result_folder)
            
            # Build the target directory: experiments/{exp_name}/logs/0-run/results/{node.id}/
            target_base = os.path.join(self.cfg.log_dir, "results", node.id)
            os.makedirs(target_base, exist_ok=True)
            
            # Move result_folder to the target directory and rename to model_results
            target_model_results = os.path.join(target_base, "model_results")
            if os.path.exists(target_model_results):
                shutil.rmtree(target_model_results)  # Remove the target if it already exists
            shutil.move(result_folder, target_model_results)
            
            # Update experiment_data.npy with the move info
            for dataset_key in experiment_data:
                if isinstance(experiment_data[dataset_key], dict) and 'result_folder' in experiment_data[dataset_key]:
                    experiment_data[dataset_key]['result_folder_info'] = {
                        'original_path': result_folder,
                        'moved_path': target_model_results,
                        'experiment_id': node.id,
                        'exp_name': exp_name,
                        'timestamp': datetime.now().isoformat()
                    }
            
            # Save the updated experiment_data.npy back in place
            np.save(experiment_data_path, experiment_data)
            
            # Copy experiment_data.npy into the target directory
            target_experiment_data = os.path.join(target_base, "experiment_data.npy")
            shutil.copy2(experiment_data_path, target_experiment_data)
            
            # Record the result directory on the node so get_node_log() can serialize it later
            node.exp_results_dir = target_base

            logger.info(f"Successfully moved result_folder to {target_model_results} for node {node.id}")

        except Exception as e:
            logger.error(f"Error handling post-experiment files for node {node.id}: {e}")
            raise
    
    def _extract_exp_name_from_result_folder(self, result_folder: str) -> str:
        """
        Extract exp_name from a result_folder path.
        Example: /path/to/camylanetdata/2025-10-19_16-12-38_msd_segmentation_attempt_0/...
        Returns: 2025-10-19_16-12-38_msd_segmentation_attempt_0
        """
        import re
        
        # Look for the date-pattern segment in the path
        path_parts = result_folder.split(os.sep)
        for part in path_parts:
            if re.match(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_.*_attempt_\d+', part):
                return part
        
        # If no standard pattern is found, look for a segment containing "attempt"
        for part in path_parts:
            if "attempt" in part.lower():
                return part
        
        # If nothing matches, fall back to a default
        logger.warning(f"Could not extract exp_name from result_folder: {result_folder}")
        return "unknown_exp"

    def __enter__(self):
        return self

    def cleanup(self):
        """Cleanup resources"""
        logger.debug("Cleanup complete")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
