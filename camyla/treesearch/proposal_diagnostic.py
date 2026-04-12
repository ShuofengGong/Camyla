"""
Proposal Diagnostic Module - in-substage diagnostics and improvement suggestions.

During substage execution, diagnose the cause of performance issues from
experiment feedback:
- code_issue: problem in code implementation, hyperparameters, or architecture
  → brainstorm 5 improvement suggestions and inject them into the subsequent
    _improve() prompt.
- proposal_infeasible: the proposal's technical plan itself is infeasible
  → directly refine the proposal.

After surpassing the baseline:
- Generate hyperparameter optimization hints and inject them into subsequent
  iteration prompts.
- Do not mark the node as needing revision and do not modify the proposal.
"""

import os
import json
import logging
import math
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field

from .backend import query, FunctionSpec
from .backend.utils import compile_prompt_to_md

logger = logging.getLogger(__name__)


def _get_node_status_label(node) -> str:
    """Return a human-readable status label for a node."""
    if not node.is_buggy:
        return "Success"
    if node.exc_type is not None:
        return "Error"
    return "Underperforming"


# ============================================================================
# Function Spec for LLM Diagnosis
# ============================================================================

diagnosis_func_spec = FunctionSpec(
    name="diagnose_performance_issue",
    description="Diagnose why the experiment performance is below expectations",
    json_schema={
        "type": "object",
        "properties": {
            "diagnosis_type": {
                "type": "string",
                "enum": ["code_issue", "proposal_infeasible"],
                "description": (
                    "code_issue: Implementation or hyperparameter problems fixable in code. "
                    "proposal_infeasible: Fundamental design flaws or hard technical barriers "
                    "requiring proposal revision."
                )
            },
            "reasoning": {
                "type": "string",
                "description": "Overall analysis of why performance is below expectations"
            },
            "improvement_suggestions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "enum": ["architecture", "hyperparameter", "code_fix", "proposal_gap"],
                            "description": (
                                "architecture: structural changes to the network design. "
                                "hyperparameter: tuning dimensions, learning rate, etc. "
                                "code_fix: fixing bugs or incorrect implementations. "
                                "proposal_gap: modules described in proposal but missing/simplified in code."
                            )
                        },
                        "suggestion": {
                            "type": "string",
                            "description": "Specific, actionable improvement suggestion"
                        },
                        "expected_impact": {
                            "type": "string",
                            "description": "Expected impact: high / medium / low with brief justification"
                        },
                    },
                    "required": ["category", "suggestion", "expected_impact"]
                },
                "minItems": 5,
                "maxItems": 5,
                "description": "Exactly 5 concrete improvement suggestions covering diverse categories"
            }
        },
        "required": ["diagnosis_type", "reasoning", "improvement_suggestions"]
    }
)

diagnosis_no_refinement_func_spec = FunctionSpec(
    name="diagnose_performance_issue",
    description="Diagnose why the experiment performance is below expectations (proposal modification unavailable)",
    json_schema={
        "type": "object",
        "properties": {
            "diagnosis_type": {
                "type": "string",
                "enum": ["code_issue"],
                "description": (
                    "code_issue: Implementation or hyperparameter problems fixable in code. "
                    "(Proposal modification budget exhausted — only this option is available.)"
                )
            },
            "reasoning": {
                "type": "string",
                "description": "Overall analysis of why performance is below expectations"
            },
            "improvement_suggestions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "enum": ["architecture", "hyperparameter", "code_fix", "proposal_gap"],
                            "description": (
                                "architecture: structural changes to the network design. "
                                "hyperparameter: tuning dimensions, learning rate, etc. "
                                "code_fix: fixing bugs or incorrect implementations. "
                                "proposal_gap: modules described in proposal but missing/simplified in code."
                            )
                        },
                        "suggestion": {
                            "type": "string",
                            "description": "Specific, actionable improvement suggestion"
                        },
                        "expected_impact": {
                            "type": "string",
                            "description": "Expected impact: high / medium / low with brief justification"
                        },
                    },
                    "required": ["category", "suggestion", "expected_impact"]
                },
                "minItems": 5,
                "maxItems": 5,
                "description": "Exactly 5 concrete improvement suggestions covering diverse categories"
            }
        },
        "required": ["diagnosis_type", "reasoning", "improvement_suggestions"]
    }
)


refinement_func_spec = FunctionSpec(
    name="refine_proposal_content",
    description="Refine proposal content",
    json_schema={
        "type": "object",
        "properties": {
            "refined_sections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "section_name": {"type": "string", "description": "Name of the section to modify"},
                        "original_text": {"type": "string", "description": "Original text (for locating)"},
                        "refined_text": {"type": "string", "description": "Refined text"}
                    },
                    "required": ["section_name", "original_text", "refined_text"]
                },
                "description": "List of sections to modify"
            },
            "changes_summary": {
                "type": "string",
                "description": "Summary of changes made"
            }
        },
        "required": ["refined_sections", "changes_summary"]
    }
)


optimization_func_spec = FunctionSpec(
    name="generate_optimization_hints",
    description="Analyze proposal completeness and generate optimization hints for an implementation that has already beaten the baseline",
    json_schema={
        "type": "object",
        "properties": {
            "proposal_gaps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "module_name": {"type": "string", "description": "Name of the proposal module/component"},
                        "status": {
                            "type": "string",
                            "enum": ["fully_implemented", "simplified", "missing"],
                            "description": "Implementation status: fully_implemented = faithfully implemented; simplified = core concept replaced with trivial operations (identity/plain conv); missing = not present in code at all"
                        },
                        "description": {"type": "string", "description": "Brief description of what was simplified/missing and how to properly implement it"}
                    },
                    "required": ["module_name", "status", "description"]
                },
                "description": "Analysis of each core innovation module from the proposal — whether it is fully implemented, simplified to trivial ops, or missing entirely"
            },
            "implementation_hints": {
                "type": "string",
                "description": "Specific, actionable suggestions for implementing the missing/simplified proposal modules. Focus on the most impactful missing innovations first. Include concrete implementation guidance (e.g., which PyTorch ops to use, how to handle dimensions)."
            },
            "hyperparameter_hints": {
                "type": "string",
                "description": "Concise hyperparameter optimization suggestions (learning rate, architecture hyperparameters like layer dimensions, channel numbers, feature sizes, attention heads, etc.)"
            },
            "priority_changes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Top 3-5 most impactful changes to try, ordered by expected impact. Implementing missing proposal modules should rank higher than hyperparameter tuning."
            },
            "reasoning": {
                "type": "string",
                "description": "Brief reasoning for the analysis and suggestions"
            }
        },
        "required": ["proposal_gaps", "implementation_hints", "hyperparameter_hints", "priority_changes", "reasoning"]
    }
)


# ============================================================================
# Refinement Rules
# ============================================================================

REFINEMENT_RULES = """
## Refinement Rules (Must be strictly followed)

### Core Principles:
- **Preserve the main idea**: The core innovation concept and method names must NOT change
- **Preserve innovation**: When removing or simplifying infeasible parts, you MUST provide concrete alternative(s) that achieve similar goals within framework constraints
- **Keep academic depth**: The proposal should remain technically substantive and publishable

### ✅ Allowed Modifications:
- Implementation details (e.g., layer configurations, feature dimensions, connection patterns, activation functions)
- Incorrect or unreasonable technical descriptions (e.g., wrong algorithm steps, infeasible operations)
- Ambiguous or vague descriptions (make them clearer, more specific, easier to implement)
- **Add necessary implementation details** (e.g., missing module explanations, unclear data flow)
- **Fix logical issues in technical descriptions**
- **Optimize inter-module interactions**
- **When an approach is proven infeasible** (e.g. framework-incompatible, not implementable as-is): you MAY **remove or simplify** that module/description; you MUST **replace it with or add a concrete alternative** that preserves the core innovation and is feasible (e.g. nnUNet-compatible, standard operations only)
- **Adjust component configurations** while preserving the overall architecture

### ❌ Prohibited Modifications:
- Core method names and research motivation (keep the research direction unchanged)
- The essence of core innovation points
- The main architectural concept
- **Removing content without offering an alternative**: Do not delete or simplify an infeasible part and leave a gap; always provide a feasible alternative that achieves similar goals

### 📝 Refinement Strategy:
1. If a module is difficult to implement but fixable, **ADD** implementation details or **FIX** the description
2. If a module/approach is **proven infeasible** (e.g. diagnosis states framework-incompatible): **REMOVE or SIMPLIFY** the infeasible part, and **PROVIDE a concrete alternative** (e.g. equivalent mechanism using standard convolutions/attention, or a simplified variant) that preserves the core contribution
3. Output a **single, coherent proposal**: prefer replacing the infeasible section with the alternative in place, rather than appending long "Implementation Refinement" blocks; if you add an alternative, integrate it clearly (e.g. "Alternative (framework-compatible): ...") and keep the document readable
4. Maintain the proposal's academic depth and ensure it remains innovative and publishable
"""


@dataclass
class DiagnosisResult:
    """Diagnosis result"""
    diagnosis_type: str  # "code_issue" or "proposal_infeasible"
    reasoning: str
    improvement_suggestions: List[Dict[str, str]]
    trigger_node_id: str
    baseline_metric: float
    current_metric: float
    metric_gap: float


@dataclass 
class RefinementResult:
    """Refinement result"""
    success: bool
    new_file_path: Optional[str]
    version: int
    changes_summary: str
    error_message: Optional[str] = None


class ProposalDiagnostic:
    """
    Proposal Diagnostic Module
    
    Responsibilities:
    1. Diagnose the cause of performance issues (code vs hyperparameter vs proposal)
    2. Generate fix suggestions
    3. Execute proposal refinement when needed
    4. Manage refinement history and versions
    """
    
    def __init__(self, cfg, proposal_queue_manager):
        """
        Initialize the diagnostic module
        
        Args:
            cfg: Configuration object
            proposal_queue_manager: ProposalQueueManager instance
        """
        self.cfg = cfg
        self.proposal_queue_manager = proposal_queue_manager
        
        # Read parameters from config
        refinement_cfg = getattr(cfg.experiment, 'proposal_refinement', None)
        if refinement_cfg:
            self.enabled = getattr(refinement_cfg, 'enabled', True)
            self.metric_threshold = getattr(refinement_cfg, 'metric_threshold', 0.05)
            self.max_refinements_per_substage = getattr(refinement_cfg, 'max_refinements_per_substage', 2)
            self.term_out_tail_chars = getattr(refinement_cfg, 'term_out_tail_chars', 1000)
            self.diagnostic_model = getattr(refinement_cfg, 'diagnostic_model', None)
            self.diagnostic_temperature = getattr(refinement_cfg, 'diagnostic_temperature', 0.3)
        else:
            self.enabled = True
            self.metric_threshold = 0.05
            self.max_refinements_per_substage = 2
            self.term_out_tail_chars = 1000
            self.diagnostic_model = None
            self.diagnostic_temperature = 0.3
        
        # Use feedback role's model if diagnostic model not specified
        if self.diagnostic_model is None:
            from camyla.model_config import get_role
            self.diagnostic_model = get_role("feedback")["model"]
        
        # Load CamylaNet framework documentation for feasibility assessment
        self.framework_doc = self._load_framework_documentation()
        
        # Refinement state per substage
        # key: substage_name, value: {"refinement_count": int, "last_refinement_node_idx": int, "has_beat_baseline": bool, "optimization_count": int, "optimization_hints": str}
        self.substage_refinement_state: Dict[str, Dict[str, Any]] = {}
    
    def should_diagnose(
        self,
        substage_name: str,
        latest_node,
        baseline_metric: float,
        journal,
        baseline_metric_value=None
    ) -> bool:
        """
        Determine if diagnosis is needed.
        
        Uses MetricValue comparison (Dice primary, HD95 direct tiebreak when
        Dice is within threshold) for has_beat_baseline check, consistent
        with journal.get_best_node().
        Falls back to float comparison if baseline_metric_value is not provided.
        
        Args:
            substage_name: Current substage name
            latest_node: Latest experiment node
            baseline_metric: Baseline Dice value (float, used for metric_gap in LLM prompts)
            journal: Current journal
            baseline_metric_value: Baseline MetricValue object for has_beat_baseline comparison
        
        Returns:
            bool: Whether diagnosis is needed
        """
        if not self.enabled:
            return False
        
        if substage_name not in self.substage_refinement_state:
            self.substage_refinement_state[substage_name] = {
                "refinement_count": 0,
                "last_refinement_node_idx": 0,
                "has_beat_baseline": False,
                "optimization_count": 0,
                "optimization_hints": None
            }
        
        state = self.substage_refinement_state[substage_name]
        
        if state["has_beat_baseline"]:
            logger.debug(f"Substage {substage_name} has already beaten baseline, disabling refinement")
            return False
        
        for node in journal.good_nodes:
            if getattr(node, 'is_stale', False):
                continue
            if node.metric is None:
                continue
            # Use MetricValue comparison when available, otherwise fall back to float
            if baseline_metric_value is not None:
                beats = node.metric > baseline_metric_value
            else:
                beats = node.metric.get_mean_value() > baseline_metric
            if beats:
                state["has_beat_baseline"] = True
                logger.info(f"✅ Substage {substage_name} has produced results exceeding baseline (MetricValue comparison), disabling further refinement")
                return False
        
        if latest_node is None or latest_node.metric is None:
            logger.debug("Latest node has no metric, skipping diagnosis")
            return False
        
        current_metric = latest_node.metric.get_mean_value()
        
        if math.isnan(current_metric):
            logger.debug(f"Latest node metric is nan, skipping diagnosis")
            return False
        
        metric_gap = baseline_metric - current_metric
        
        if metric_gap <= 0:
            logger.debug(f"Metric gap ({metric_gap:.4f}) <= 0, performance meets or exceeds baseline, skipping diagnosis")
            return False
        
        logger.info(f"⚠️ Performance below baseline by {metric_gap:.4f}, triggering diagnosis")
        return True

    def _format_budget_section(self, budget_info: Optional[Dict]) -> str:
        """Format iteration budget info for the diagnostic prompt."""
        if not budget_info:
            return ""
        used = budget_info.get("proposal_steps_used", "?")
        total = budget_info.get("proposal_steps_total", "?")
        remaining = (total - used) if isinstance(total, int) and isinstance(used, int) else "?"
        idx = budget_info.get("proposal_idx", "?")
        num_proposals = budget_info.get("num_proposals", "?")
        g_used = budget_info.get("global_substages_used", "?")
        g_total = budget_info.get("global_substages_total", "?")
        ref_used = budget_info.get("refinement_used", 0)
        ref_total = budget_info.get("refinement_total", self.max_refinements_per_substage)
        ref_remaining = ref_total - ref_used if isinstance(ref_total, int) and isinstance(ref_used, int) else "?"
        refinement_exhausted = (isinstance(ref_remaining, int) and ref_remaining <= 0)
        
        base = (
            f"## Iteration Budget Status\n"
            f"- Current proposal: used {used}/{total} iterations "
            f"(proposal {idx}/{num_proposals})\n"
            f"- Global Stage 2 budget: used {g_used}/{g_total} substages\n"
            f"- Remaining iterations for this proposal: {remaining}\n"
            f"- Proposal modification (refinement) chances: total {ref_total}, "
            f"used {ref_used}, remaining {ref_remaining}\n"
        )
        
        if refinement_exhausted:
            base += (
                f"\nProposal modification budget EXHAUSTED. "
                f"Diagnosis is limited to `code_issue` only. Do NOT suggest proposal changes."
            )
        else:
            base += (
                f"\nIMPORTANT: If the current proposal has used most of its budget "
                f"with no improving trend and is unlikely to beat baseline, strongly "
                f"consider diagnosing as `proposal_infeasible` to allow switching to "
                f"a different proposal while budget remains.\n\n"
                f"NOTE: You have {ref_remaining} proposal modification chance(s) left "
                f"(total {ref_total}, used {ref_used}). If you diagnose as "
                f"`proposal_infeasible`, the proposal will be refined using one of "
                f"these chances. Once all chances are exhausted, no further proposal "
                f"modifications are possible for this substage."
            )
        
        return base

    def diagnose(
        self,
        substage_name: str,
        proposal_content: str,
        journal,
        baseline_metric: float,
        latest_node,
        budget_info: Optional[Dict] = None
    ) -> Optional[DiagnosisResult]:
        """
        Diagnose the cause of performance issues
        
        Args:
            substage_name: Current substage name
            proposal_content: Full proposal content
            journal: Current journal
            baseline_metric: Baseline metric value
            latest_node: Latest node
            budget_info: Optional dict with iteration budget context
            
        Returns:
            DiagnosisResult or None (if diagnosis fails)
        """
        # Collect feedback from the latest node, including OpenHands action log
        feedback_summary = self._collect_latest_node_feedback(latest_node, substage_name=substage_name)
        
        # Collect metric panorama from all nodes under this proposal
        all_nodes_metrics = self._collect_all_nodes_metrics(journal, baseline_metric)
        
        current_metric = latest_node.metric.get_mean_value() if latest_node.metric else 0
        metric_gap = baseline_metric - current_metric
        
        # Build framework constraints section
        framework_section = ""
        if self.framework_doc:
            framework_section = f"""
## CamylaNet Framework Documentation (reference):
{self.framework_doc}
"""
        
        # Check if refinement budget is exhausted
        refinement_exhausted = False
        if budget_info:
            ref_used = budget_info.get("refinement_used", 0)
            ref_total = budget_info.get("refinement_total", self.max_refinements_per_substage)
            if isinstance(ref_used, int) and isinstance(ref_total, int) and ref_used >= ref_total:
                refinement_exhausted = True
                logger.info(f"Proposal refinement budget exhausted ({ref_used}/{ref_total}), "
                            f"excluding proposal_infeasible from diagnosis options")
        
        # Build instructions based on whether refinement is available
        if refinement_exhausted:
            instructions_text = """
Analyze why performance is below expectations.

NOTE: Proposal modification budget is EXHAUSTED. Diagnose as `code_issue`.

CONTEXT: We are innovating WITHIN the nnUNet framework (a well-tuned medical image
segmentation baseline). In medical image segmentation, compact/smaller networks are generally
preferred and sometimes perform as well as or better than larger ones. Do NOT suggest
increasing model capacity (e.g., adding more layers, widening channels, increasing attention
heads) unless previous experiments have shown that reducing capacity led to performance
degradation. Prefer parameter-efficient designs.

TREND ANALYSIS:
Results may come from MULTIPLE independent branches. Analyse each branch's trend SEPARATELY.
For each branch, check for: improving trend, plateau, stagnation, or degradation.
If the most recent experiment achieved the best metric so far, the trend is improving.

DIAGNOSIS:
After analyzing the trend and comparing the code against the proposal, brainstorm exactly
5 concrete, actionable improvement suggestions. Each suggestion must be:
- Specific enough for a developer to implement directly
- Feasible within the nnUNet framework constraints
- Likely to improve performance (high success probability)

Cover diverse categories:
- **architecture**: structural changes to the network design
- **hyperparameter**: tuning channel dimensions, learning rate (try 1e-3 or 1e-4), etc.
  NOTE: Our nnUNet-like framework is already well-tuned, so large gains from hyperparameter
  tuning alone are rare. Only include hyperparameter suggestions when the metric gap to
  baseline is within 1% (0.01 Dice).
- **code_fix**: fixing bugs or incorrect implementations in the current code
- **proposal_gap**: modules described in the research proposal but missing or simplified
  in the code (e.g., replaced with plain Conv3d/MLP/Identity)

At least one suggestion MUST check whether core innovation modules from the proposal are
properly implemented in the code. If any are missing or simplified to trivial operations,
include a `proposal_gap` suggestion with specific guidance on what to implement.

Do NOT suggest: regularization, training tricks, data augmentation, optimizer changes,
training duration, batch size, or ablation experiments.
"""
        else:
            instructions_text = """
Analyze why performance is below expectations.

CONTEXT: We are innovating WITHIN the nnUNet framework (a well-tuned medical image
segmentation baseline). In medical image segmentation, compact/smaller networks are generally
preferred and sometimes perform as well as or better than larger ones. Do NOT suggest
increasing model capacity (e.g., adding more layers, widening channels, increasing attention
heads) unless previous experiments have shown that reducing capacity led to performance
degradation. Prefer parameter-efficient designs.

TREND ANALYSIS:
Results may come from MULTIPLE independent branches. Analyse each branch's trend SEPARATELY.
For each branch, check for: improving trend, plateau, stagnation, or degradation.
Key signals (per branch):
- If >= 5 experiments AND best metric has NOT improved in 3+ runs → ceiling reached.
- Monotonic degradation across 3+ experiments → search is diverging.
- If the most recent experiment achieved the best metric so far → improving, give more iterations.

DIAGNOSIS — Choose ONE type:

1. **code_issue**: The problem is fixable in code (implementation, hyperparameters, architecture).
   You MUST provide exactly 5 improvement suggestions (see below).

2. **proposal_infeasible**: Hard technical barriers (repeated OOM, numerical instability,
   framework constraint violations), OR approach plateaued/degraded over 2+ experiments
   with a faithful implementation, OR metric gap > 1% despite faithful implementation.
   The proposal itself needs revision. Improvement suggestions can be empty.
   HOWEVER: Do NOT choose this if the most recent experiment achieved a new best metric
   AND the gap to baseline is small.

For `code_issue`, brainstorm exactly 5 concrete, actionable improvement suggestions.
Each suggestion must be:
- Specific enough for a developer to implement directly
- Feasible within the nnUNet framework constraints
- Likely to improve performance (high success probability)

Cover diverse categories:
- **architecture**: structural changes to the network design
- **hyperparameter**: tuning channel dimensions, learning rate (try 1e-3 or 1e-4), etc.
  NOTE: Our nnUNet-like framework is already well-tuned, so large gains from hyperparameter
  tuning alone are rare. Only include hyperparameter suggestions when the metric gap to
  baseline is within 1% (0.01 Dice).
- **code_fix**: fixing bugs or incorrect implementations in the current code
- **proposal_gap**: modules described in the research proposal but missing or simplified
  in the code (e.g., replaced with plain Conv3d/MLP/Identity)

At least one suggestion MUST check whether core innovation modules from the proposal are
properly implemented in the code. If any are missing or simplified to trivial operations,
include a `proposal_gap` suggestion with specific guidance on what to implement.

Do NOT suggest: regularization, training tricks, data augmentation, optimizer changes,
training duration, batch size, or ablation experiments.
"""

        # Select the appropriate function spec
        active_func_spec = diagnosis_no_refinement_func_spec if refinement_exhausted else diagnosis_func_spec

        # Build diagnosis prompt
        prompt = {
            "Task": "Diagnose why experiment performance is below expectations",
            "Context": f"""
The current experiment is testing a Research Proposal, but performance is below baseline.

## Framework Constraints:
We use the **CamylaNet framework** (nnUNet v2 wrapper) for medical image segmentation.
The ONLY thing we can customize is the **network architecture** (via `build_network_architecture` in a custom Trainer).

What we CAN do:
- Define a custom `nn.Module` network architecture of any complexity
- Use ALL standard PyTorch nn modules and operations (convolutions, attention, normalization, pooling, etc.)
- Adjust learning rate via `initial_lr` parameter
- Adjust architecture hyperparameters (channel dims, layer depth, kernel sizes, attention heads, etc.)

What we CANNOT do (hard constraints):
- Modify the loss function, training loop, optimizer type, or learning rate schedule
- Modify data loading, preprocessing, or augmentation pipeline
- Use multi-input pipelines or return multiple outputs (tuple/dict)
- Use torch.linalg.qr / torch.linalg.svd / torch.linalg.eigh (fail with float16)
- Change the training paradigm (no self-supervised, contrastive, GAN, etc.)
{framework_section}
## Baseline Metric (best baseline): {baseline_metric:.4f}
## Current Metric (latest node): {current_metric:.4f}
## Metric Gap: {metric_gap:.4f} (below baseline)

## All Experiment Results Under This Proposal:
{all_nodes_metrics}

## Research Proposal Content:
{proposal_content}

## Latest Experiment Feedback:
{feedback_summary}

{self._format_budget_section(budget_info)}
""",
            "Instructions": instructions_text
        }
        
        try:
            response = query(
                system_message=None,
                user_message=prompt,
                func_spec=active_func_spec,
                model=self.diagnostic_model,
                temperature=self.diagnostic_temperature,
            )
            
            self._save_llm_interaction(
                substage_name=substage_name,
                interaction_type="diagnosis",
                system_message=None,
                user_message=prompt,
                func_spec=active_func_spec,
                response=response,
                model=self.diagnostic_model,
                temperature=self.diagnostic_temperature,
            )
            
            diagnosis = DiagnosisResult(
                diagnosis_type=response["diagnosis_type"],
                reasoning=response["reasoning"],
                improvement_suggestions=response.get("improvement_suggestions", []),
                trigger_node_id=latest_node.id,
                baseline_metric=baseline_metric,
                current_metric=current_metric,
                metric_gap=metric_gap,
            )
            
            logger.info(f"📋 Diagnosis result: {diagnosis.diagnosis_type} "
                        f"({len(diagnosis.improvement_suggestions)} suggestions)")
            
            return diagnosis
            
        except Exception as e:
            logger.error(f"Diagnosis failed: {e}")
            return None
    
    def refine_proposal(
        self,
        substage_name: str,
        proposal_idx: int,
        diagnosis: DiagnosisResult
    ) -> RefinementResult:
        """
        Refine proposal based on diagnosis result
        
        Args:
            substage_name: Current substage name
            proposal_idx: Proposal index
            diagnosis: Diagnosis result
            
        Returns:
            RefinementResult
        """
        if diagnosis.diagnosis_type != "proposal_infeasible":
            return RefinementResult(
                success=False,
                new_file_path=None,
                version=0,
                changes_summary="",
                error_message=f"Diagnosis type is {diagnosis.diagnosis_type}, not proposal_infeasible"
            )
        
        # Get current proposal info
        proposal = self.proposal_queue_manager.get_proposal(proposal_idx)
        if not proposal:
            return RefinementResult(
                success=False,
                new_file_path=None,
                version=0,
                changes_summary="",
                error_message=f"Proposal {proposal_idx} not found"
            )
        
        md_file = proposal.get("md_file", "")
        if not md_file or not os.path.exists(md_file):
            return RefinementResult(
                success=False,
                new_file_path=None,
                version=0,
                changes_summary="",
                error_message=f"Proposal MD file not found: {md_file}"
            )
        
        # Read current proposal content
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                original_content = f.read()
        except Exception as e:
            return RefinementResult(
                success=False,
                new_file_path=None,
                version=0,
                changes_summary="",
                error_message=f"Failed to read proposal file: {e}"
            )
        
        # Calculate new version number
        version = self._get_next_version(md_file)
        
        # Build framework documentation section for refinement
        framework_ref = ""
        if self.framework_doc:
            framework_ref = f"""
## CamylaNet Framework Documentation (the refined proposal MUST be compatible):
{self.framework_doc}
"""
        
        # Build refinement prompt
        prompt = {
            "Task": "Refine Research Proposal",
            "Original Proposal": original_content,
            "Diagnosis": f"""
## Diagnosis Result: proposal_infeasible
## Diagnosis Reasoning: {diagnosis.reasoning}

## Improvement Suggestions:
{chr(10).join(f"- [{s.get('category', 'unknown')}] {s.get('suggestion', '')}" for s in diagnosis.improvement_suggestions)}

## Performance Information:
- Baseline Metric: {diagnosis.baseline_metric:.4f}
- Current Metric: {diagnosis.current_metric:.4f}
- Gap: {diagnosis.metric_gap:.4f}
""",
            "Framework Constraints": f"""
The refined proposal MUST be implementable within the CamylaNet framework:
- ONLY the network architecture can be customized (via build_network_architecture)
- Model receives a single input tensor → returns a single output tensor
- Loss function, training loop, optimizer, data loading are ALL FIXED
- Standard supervised learning only

What IS allowed (do NOT unnecessarily simplify these):
- Complex attention mechanisms (self-attention, cross-attention, window attention)
- All standard PyTorch operations (Linear, Conv3d, Softmax, matmul, einsum)
- Gating, multi-scale, dilated convolutions, etc.

What is NOT allowed (only simplify if diagnosis specifically identifies these):
- Custom loss functions, training loop changes, multi-input pipelines
- Models returning tuples/dicts, torch.linalg.qr/svd/eigh
{framework_ref}""",
            "Rules": REFINEMENT_RULES,
            "Instructions": """
Please refine the proposal based on the diagnosis. Output the complete refined Markdown content.

IMPORTANT:
1. Strictly follow the refinement rules
2. ONLY modify what the diagnosis specifically identifies as problematic. Do NOT over-simplify.
3. If the diagnosis identifies a specific hard constraint violation (e.g., custom loss, multi-output):
   replace ONLY that part with a framework-compatible alternative
4. Do NOT replace attention mechanisms with simple convolutions unless the diagnosis
   specifically says attention itself is the problem (attention is fully supported in CamylaNet)
5. Preserve the proposal's innovation level — the refined version should be equally ambitious
6. For parts not diagnosed as infeasible, keep them unchanged
7. Maintain the academic depth and technical sophistication of the proposal
"""
        }
        
        try:
            # Use LLM to generate refined content
            refinement_system_msg = "You are an AI research assistant that refines Research Proposals. When an approach is proven infeasible (e.g. framework-incompatible), you may remove or simplify it but must provide a concrete alternative that preserves the core innovation and is feasible. Output a single, coherent proposal; prefer replacing infeasible sections with the alternative in place rather than only appending long refinement blocks."
            refined_content = query(
                system_message=refinement_system_msg,
                user_message=prompt,
                model=self.diagnostic_model,
                temperature=self.diagnostic_temperature,
            )
            
            self._save_llm_interaction(
                substage_name=substage_name,
                interaction_type="refinement",
                system_message=refinement_system_msg,
                user_message=prompt,
                func_spec=None,
                response=refined_content,
                model=self.diagnostic_model,
                temperature=self.diagnostic_temperature,
            )
            
            # Clean response
            if isinstance(refined_content, str):
                refined_content = refined_content.strip()
                # Remove possible markdown code block markers
                if refined_content.startswith("```"):
                    import re
                    refined_content = re.sub(r'^```\w*\n?', '', refined_content)
                    refined_content = re.sub(r'\n?```$', '', refined_content)
            else:
                return RefinementResult(
                    success=False,
                    new_file_path=None,
                    version=version,
                    changes_summary="",
                    error_message="LLM response is not a string"
                )
            
            # Validate content length
            if len(refined_content) < 500:
                return RefinementResult(
                    success=False,
                    new_file_path=None,
                    version=version,
                    changes_summary="",
                    error_message=f"Refined content too short: {len(refined_content)} chars"
                )
            
            # Generate new file path
            base_name = os.path.splitext(md_file)[0]
            # Remove possible old version suffix
            import re
            base_name = re.sub(r'_v\d+$', '', base_name)
            new_file_path = f"{base_name}_v{version}.md"
            
            # Add refinement info to content end
            refinement_note = f"""

---
## Refinement History (v{version})

- **Timestamp**: {datetime.now().isoformat()}
- **Trigger Node**: {diagnosis.trigger_node_id}
- **Baseline Metric**: {diagnosis.baseline_metric:.4f}
- **Trigger Metric**: {diagnosis.current_metric:.4f}
- **Diagnosis**: {diagnosis.reasoning[:500]}...
"""
            refined_content += refinement_note
            
            # Save new version
            with open(new_file_path, 'w', encoding='utf-8') as f:
                f.write(refined_content)
            
            # Update proposal_queue_manager file path
            self.proposal_queue_manager.proposals[proposal_idx]["md_file"] = new_file_path
            
            # Update substage state
            state = self.substage_refinement_state[substage_name]
            state["refinement_count"] += 1
            
            # Save refinement log
            self._save_refinement_log(
                md_file, 
                new_file_path, 
                version, 
                diagnosis,
                f"Refined based on diagnosis: {diagnosis.reasoning[:200]}..."
            )
            
            logger.info(f"✅ Proposal refinement successful: {new_file_path}")
            
            return RefinementResult(
                success=True,
                new_file_path=new_file_path,
                version=version,
                changes_summary=f"Refined based on diagnosis: {diagnosis.reasoning[:200]}..."
            )
            
        except Exception as e:
            logger.error(f"Refinement failed: {e}")
            import traceback
            traceback.print_exc()
            return RefinementResult(
                success=False,
                new_file_path=None,
                version=version,
                changes_summary="",
                error_message=str(e)
            )
    
    def _update_node_proposal_content(
        self,
        substage_name: str,
        original_proposal: str,
        latest_node,
        diagnosis: DiagnosisResult,
    ) -> Optional[str]:
        """Generate updated proposal content reflecting the actual implementation.
        
        NOTE: This method is currently unused (proposal_diverged logic removed).
        Kept for potential future use.
        
        This only updates the node-level snapshot and does NOT affect the global proposal
        file or other nodes.
        
        Args:
            substage_name: Current substage name
            original_proposal: Original proposal content (from node snapshot)
            latest_node: The node whose implementation diverged
            diagnosis: Diagnosis result with reasoning about the divergence
            
        Returns:
            Updated proposal content string, or None on failure
        """
        # Collect rich implementation details: code, plan, analysis, and action log
        impl_details = ""
        
        # Actual code (most important for understanding what was built)
        if latest_node.code:
            code_text = latest_node.code
            if len(code_text) > 15000:
                code_text = code_text[:15000] + "\n... (truncated)"
            impl_details += f"## Actual Code Implementation:\n```python\n{code_text}\n```\n\n"
        
        if latest_node.plan:
            impl_details += f"## Agent's Implementation Plan:\n{latest_node.plan[:2000]}\n\n"
        
        if latest_node.analysis:
            impl_details += f"## Agent's Analysis of Results:\n{latest_node.analysis[:1000]}\n\n"
        
        # OpenHands action log (shows agent's decision-making, simplifications, etc.)
        action_log = ""
        jsonl_path = self._find_latest_openhands_log(substage_name)
        if jsonl_path:
            action_log = self._extract_action_log(jsonl_path, max_chars=20000)
        if action_log:
            impl_details += (
                f"## Agent Decision Log (shows what was attempted and what was simplified):\n"
                f"```\n{action_log}\n```\n\n"
            )
        
        if not impl_details.strip():
            logger.warning("No implementation details available for proposal update")
            return None
        
        prompt = {
            "Task": "Update research proposal to reflect actual implementation",
            "Original Proposal": original_proposal,
            "Divergence Analysis": f"""
The diagnostic system found that the actual implementation has DIVERGED from the original proposal.

## Diagnosis Reasoning:
{diagnosis.reasoning}

{impl_details}
## Improvement Suggestions (from diagnosis):
{chr(10).join(f"- [{s.get('category', '')}] {s.get('suggestion', '')}" for s in diagnosis.improvement_suggestions) if diagnosis.improvement_suggestions else "None"}
""",
            "Instructions": """
Rewrite the proposal to accurately describe what was ACTUALLY implemented (see the code above),
not what was originally planned. This updated proposal will be used to guide hyperparameter tuning.

Rules:
1. Keep the same document structure and section headings as the original
2. Replace descriptions of modules/components with what the code actually builds
3. If the agent simplified a module, describe the simplified version accurately
4. Preserve any parts of the original proposal that were faithfully implemented
5. Maintain academic tone and technical depth
6. Do NOT add new innovations — just accurately describe the current implementation
7. Output the complete updated proposal in Markdown format
"""
        }
        
        try:
            system_msg = (
                "You are an AI research assistant. Your task is to update a research proposal "
                "to accurately reflect the actual code implementation, which has diverged from "
                "the original proposal. The updated proposal will guide hyperparameter tuning."
            )
            updated_content = query(
                system_message=system_msg,
                user_message=prompt,
                model=self.diagnostic_model,
                temperature=self.diagnostic_temperature,
            )
            
            self._save_llm_interaction(
                substage_name=substage_name,
                interaction_type="node_proposal_update",
                system_message=system_msg,
                user_message=prompt,
                func_spec=None,
                response=updated_content,
                model=self.diagnostic_model,
                temperature=self.diagnostic_temperature,
            )
            
            if isinstance(updated_content, str) and len(updated_content.strip()) > 500:
                import re
                updated_content = updated_content.strip()
                if updated_content.startswith("```"):
                    updated_content = re.sub(r'^```\w*\n?', '', updated_content)
                    updated_content = re.sub(r'\n?```$', '', updated_content)
                return updated_content
            else:
                logger.warning(f"Node proposal update returned invalid content (len={len(str(updated_content))})")
                return None
                
        except Exception as e:
            logger.error(f"Failed to update node proposal content: {e}")
            return None
    
    def update_refinement_node_idx(self, substage_name: str, node_idx: int):
        """Update the last refinement node index"""
        if substage_name in self.substage_refinement_state:
            self.substage_refinement_state[substage_name]["last_refinement_node_idx"] = node_idx
    
    def _find_latest_openhands_log(self, substage_name: str) -> Optional[Path]:
        """Find the latest OpenHands JSONL event log for the given substage
        
        Path structure: {cfg.log_dir}/stage_{substage_name}/openhands_logs/openhands_events_*.jsonl
        Files are named with YYYYMMDD_HHMMSS timestamps, so sorting by name gives chronological order.
        
        Args:
            substage_name: Current substage name (e.g., '2_creative_research_1_proposal_1')
            
        Returns:
            Path to the latest JSONL file, or None if not found
        """
        try:
            log_dir = Path(self.cfg.log_dir) / f"stage_{substage_name}" / "openhands_logs"
            if not log_dir.exists():
                logger.debug(f"OpenHands logs directory not found: {log_dir}")
                return None
            jsonl_files = sorted(log_dir.glob("openhands_events_*.jsonl"))
            if not jsonl_files:
                logger.debug(f"No JSONL log files found in: {log_dir}")
                return None
            latest = jsonl_files[-1]
            logger.info(f"📄 Found latest OpenHands log: {latest.name}")
            return latest
        except Exception as e:
            logger.warning(f"Error finding OpenHands log: {e}")
            return None
    
    def _extract_action_log(self, jsonl_path: Path, max_chars: int = 80000) -> str:
        """Extract concise action log from JSONL event log (event_str fields only)
        
        The event_str field contains a compact summary of each event including:
        - Agent thoughts and reasoning
        - Tool calls (file edits, terminal commands)
        - Tool results (success/failure, output excerpts)
        - Error messages
        
        This gives the diagnostic LLM visibility into the agent's decision-making
        process without the full prompt/response overhead.
        
        Args:
            jsonl_path: Path to the JSONL event log file
            max_chars: Maximum characters to include (keeps tail if exceeded)
            
        Returns:
            Formatted action log string
        """
        try:
            lines = []
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                        event_str = event.get("event_str", "")
                        if event_str:
                            lines.append(event_str)
                    except json.JSONDecodeError:
                        continue
            
            if not lines:
                return ""
            
            full_log = "\n---\n".join(lines)
            
            # If too long, keep the tail (most recent actions are most relevant)
            if len(full_log) > max_chars:
                full_log = "...(earlier actions truncated)...\n" + full_log[-max_chars:]
            
            return full_log
        except Exception as e:
            logger.warning(f"Error extracting action log from {jsonl_path}: {e}")
            return ""
    
    def _collect_latest_node_feedback(self, latest_node, substage_name: str = None) -> str:
        """
        Collect feedback from the latest node, including OpenHands agent action log
        
        Args:
            latest_node: The latest node
            substage_name: Current substage name (used to locate OpenHands logs)
            
        Returns:
            Formatted feedback summary
        """
        if latest_node is None:
            return "No experiment conducted yet."
        
        node_info = f"### Latest Experiment (Node: {latest_node.id[:8]}...)\n"
        node_info += f"- **Status**: {_get_node_status_label(latest_node)}\n"
        
        if latest_node.metric:
            metric_val = latest_node.metric.get_mean_value()
            if not math.isnan(metric_val):
                node_info += f"- **Metric**: {metric_val:.4f}\n"
            else:
                node_info += f"- **Metric**: nan (invalid)\n"
        
        if latest_node.analysis:
            node_info += f"- **Analysis**: {latest_node.analysis[:500]}...\n"
        
        # Get terminal output tail
        if hasattr(latest_node, '_term_out') and latest_node._term_out:
            term_out_str = ''.join(latest_node._term_out)
            if len(term_out_str) > self.term_out_tail_chars:
                term_out_excerpt = term_out_str[-self.term_out_tail_chars:]
            else:
                term_out_excerpt = term_out_str
            node_info += f"- **Terminal Output (last {self.term_out_tail_chars} chars)**:\n```\n{term_out_excerpt}\n```\n"
        
        # Append OpenHands agent action log (reveals agent's decision-making process)
        if substage_name:
            jsonl_path = self._find_latest_openhands_log(substage_name)
            if jsonl_path:
                action_log = self._extract_action_log(jsonl_path)
                if action_log:
                    node_info += (
                        f"\n### OpenHands Agent Action Log\n"
                        f"The following is the complete decision chain of the code generation agent. "
                        f"Pay close attention to:\n"
                        f"- Whether the agent encountered repeated errors implementing specific proposal modules\n"
                        f"- Whether the agent explicitly simplified or removed any proposal components\n"
                        f"- Whether the agent's 'Thought' statements mention infeasibility or incompatibility\n\n"
                        f"```\n{action_log}\n```\n"
                    )
                    logger.info(f"📄 Appended OpenHands action log ({len(action_log)} chars) to diagnostic feedback")
        
        return node_info
    
    def _collect_feedback(self, journal, since_idx: int) -> str:
        """
        Collect node feedback from specified index (legacy method, kept for compatibility)
        
        Args:
            journal: Journal object
            since_idx: Starting index
            
        Returns:
            Formatted feedback summary
        """
        nodes = journal.nodes[since_idx:]
        if not nodes:
            return "No experiments conducted yet."
        
        feedback_parts = []
        for i, node in enumerate(nodes):
            node_info = f"### Experiment {since_idx + i + 1} (Node: {node.id[:8]}...)\n"
            node_info += f"- **Status**: {_get_node_status_label(node)}\n"
            
            if node.metric:
                node_info += f"- **Metric**: {node.metric.get_mean_value():.4f}\n"
            
            if node.analysis:
                node_info += f"- **Analysis**: {node.analysis[:300]}...\n"
            
            # Get terminal output tail
            if hasattr(node, '_term_out') and node._term_out:
                term_out_str = ''.join(node._term_out)
                if len(term_out_str) > self.term_out_tail_chars:
                    term_out_excerpt = term_out_str[-self.term_out_tail_chars:]
                else:
                    term_out_excerpt = term_out_str
                node_info += f"- **Terminal Output (last {self.term_out_tail_chars} chars)**:\n```\n{term_out_excerpt}\n```\n"
            
            feedback_parts.append(node_info)
        
        return "\n".join(feedback_parts)
    
    def _get_next_version(self, md_file: str) -> int:
        """Get next version number"""
        import re
        
        base_name = os.path.splitext(md_file)[0]
        # Remove possible old version suffix
        base_name_clean = re.sub(r'_v\d+$', '', base_name)
        
        # Check existing versions
        directory = os.path.dirname(md_file)
        existing_versions = []
        
        for f in os.listdir(directory):
            if f.startswith(os.path.basename(base_name_clean)) and f.endswith('.md'):
                match = re.search(r'_v(\d+)\.md$', f)
                if match:
                    existing_versions.append(int(match.group(1)))
        
        if existing_versions:
            return max(existing_versions) + 1
        else:
            return 2  # First refinement starts from v2
    
    def _save_refinement_log(
        self,
        original_file: str,
        new_file: str,
        version: int,
        diagnosis: DiagnosisResult,
        changes_summary: str
    ):
        """Save refinement log"""
        log_file = original_file.replace('.md', '_refinement_log.json')
        
        # Load existing log
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
        else:
            log_data = {
                "original_file": original_file,
                "refinements": []
            }
        
        # Add new entry
        log_data["refinements"].append({
            "version": version,
            "file": new_file,
            "timestamp": datetime.now().isoformat(),
            "trigger_node_id": diagnosis.trigger_node_id,
            "diagnosis_type": diagnosis.diagnosis_type,
            "reasoning": diagnosis.reasoning,
            "improvement_suggestions": diagnosis.improvement_suggestions,
            "baseline_metric": diagnosis.baseline_metric,
            "current_metric": diagnosis.current_metric,
            "metric_gap": diagnosis.metric_gap,
            "changes_summary": changes_summary
        })
        
        # Save
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📁 Refinement log saved: {log_file}")
    
    def generate_optimization_hints(
        self,
        substage_name: str,
        journal,
        baseline_metric: float,
        latest_node,
        proposal_content: Optional[str] = None,
    ) -> Optional[str]:
        """
        Generate optimization hints after beating baseline.
        
        When proposal_content is provided, performs a two-phase analysis:
        1. Proposal completeness check — identifies missing/simplified innovation modules
        2. Hyperparameter optimization suggestions
        
        Implementation hints (restoring missing proposal modules) take priority over
        pure hyperparameter tuning, because the innovation modules are the core
        contribution and may unlock further performance gains.
        
        Args:
            substage_name: Current substage name
            journal: Current journal
            baseline_metric: Baseline metric value
            latest_node: Latest node
            proposal_content: Full proposal markdown content (enables completeness analysis)
            
        Returns:
            Combined hints string or None
        """
        if substage_name not in self.substage_refinement_state:
            return None
        
        state = self.substage_refinement_state[substage_name]
        
        if not state.get("has_beat_baseline", False):
            return None
        
        optimization_count = state.get("optimization_count", 0)
        
        best_node = journal.get_best_node()
        if not best_node or not best_node.metric:
            logger.debug("No best node available for optimization hints")
            return None
        
        best_metric = best_node.metric.get_mean_value()
        
        # Build code section from the best node
        code_section = ""
        if best_node.code:
            code_text = best_node.code
            if len(code_text) > 15000:
                code_text = code_text[:15000] + "\n... (truncated)"
            code_section = f"## Best Node's Actual Code:\n```python\n{code_text}\n```\n"
        
        proposal_section = ""
        if proposal_content:
            proposal_section = f"## Research Proposal:\n{proposal_content}\n"
        
        has_proposal_and_code = bool(proposal_content and best_node.code)
        
        if has_proposal_and_code:
            instructions_text = """
Perform TWO analyses in order:

## PHASE 1 — Proposal Completeness Analysis
Compare the actual code implementation against the research proposal. For EACH core innovation module described in the proposal:
1. Identify the module by name (e.g., "3D DCT Transform", "Cross-Scale Gating", "Frequency-Domain Attention")
2. Check if it is **fully_implemented** (faithfully implements the core concept), **simplified** (replaced with identity/plain convolution/trivial operation), or **missing** (not present at all)
3. For simplified/missing modules, describe what should be implemented and how

A module is "simplified" if the proposal describes a non-trivial operation (e.g., frequency transform, spectral decomposition, attention mechanism) but the code replaces it with a trivial operation (e.g., `return x`, plain Conv3d, identity mapping).
Acceptable adaptations (e.g., efficient approximation, windowed attention, reduced dimensions) should be marked as "fully_implemented".

Fill the `proposal_gaps` field with your analysis.

## PHASE 2 — Optimization Suggestions
Based on Phase 1:
- If there are simplified/missing modules: `implementation_hints` should provide specific guidance on how to properly implement them. This is the HIGHEST PRIORITY improvement path.
- For hyperparameter tuning: `hyperparameter_hints` may suggest learning rate and architecture hyperparameter adjustments.

For `priority_changes`: implementing missing proposal innovations should rank HIGHER than hyperparameter tuning, because:
- The proposal's core innovations are the research contribution
- Properly implementing them may unlock significantly better performance than parameter tweaks
- A "simplified to identity" module means the architecture is missing a key component

NETWORK SIZE: In medical image segmentation, compact/smaller networks are generally preferred
and sometimes perform as well as or better than larger ones. Do NOT suggest increasing model
capacity (more layers, wider channels, more attention heads) unless previous experiments have
shown that reducing capacity led to performance degradation. Prefer parameter-efficient designs.

Do NOT suggest: training duration, batch size, regularization, data augmentation, optimizer settings (momentum, beta, epsilon, weight decay).
"""
        else:
            instructions_text = """
Suggest specific optimizations to further improve performance.

Since proposal content or code is not available for completeness analysis, focus on hyperparameter optimization.
Set `proposal_gaps` to an empty array and `implementation_hints` to "Proposal content not available for analysis."

You may suggest ONLY the following (all other hyperparameters are FIXED and must NOT be suggested):
- **Learning rate**.
- **Architecture hyperparameters** - like layer dimensions, channel numbers, feature sizes, number of attention heads, hidden dimensions, kernel sizes, depth of blocks, etc.

NETWORK SIZE: In medical image segmentation, compact/smaller networks are generally preferred
and sometimes perform as well as or better than larger ones. Do NOT suggest increasing model
capacity (more layers, wider channels, more attention heads) unless previous experiments have
shown that reducing capacity led to performance degradation. Prefer parameter-efficient designs.

Do NOT suggest: training duration, batch size, regularization, data augmentation, optimizer settings.

Provide 3-5 concrete, actionable suggestions. Be specific with values and ranges.
"""
        
        prompt = {
            "Task": "Analyze proposal implementation completeness and generate optimization suggestions",
            "Context": f"""
The experiment has exceeded the baseline. However, exceeding baseline does NOT mean the implementation
is complete — the agent may have simplified or skipped core innovation modules from the proposal.
Your task is to identify what is missing and suggest both implementation improvements and hyperparameter tuning.

## Performance Status:
- Baseline Metric: {baseline_metric:.4f}
- Best Achieved Metric: {best_metric:.4f}
- Improvement over Baseline: {(best_metric - baseline_metric):.4f}

{proposal_section}
{code_section}
## Best Experiment Details:
{self._collect_node_summary(best_node)}

## Latest Experiment:
{self._collect_latest_node_feedback(latest_node) if latest_node else "No recent experiment"}
""",
            "Instructions": instructions_text
        }
        
        try:
            optimization_system_msg = (
                "You are an AI research assistant. Your primary goal is to ensure the research proposal's "
                "core innovations are fully and faithfully implemented, and secondarily to optimize hyperparameters. "
                "A simplified or missing innovation module is a bigger problem than suboptimal hyperparameters."
            )
            response = query(
                system_message=optimization_system_msg,
                user_message=prompt,
                func_spec=optimization_func_spec,
                model=self.diagnostic_model,
                temperature=self.diagnostic_temperature,
            )
            
            self._save_llm_interaction(
                substage_name=substage_name,
                interaction_type="optimization",
                system_message=optimization_system_msg,
                user_message=prompt,
                func_spec=optimization_func_spec,
                response=response,
                model=self.diagnostic_model,
                temperature=self.diagnostic_temperature,
            )
            
            if not isinstance(response, dict):
                logger.warning("Optimization hints generation returned non-dict response")
                return None
            
            # Build combined hints string
            parts = []
            
            # Phase 1: Proposal gaps
            proposal_gaps = response.get("proposal_gaps", [])
            gaps_with_issues = [g for g in proposal_gaps if g.get("status") in ("simplified", "missing")]
            if gaps_with_issues:
                parts.append("## Proposal Implementation Gaps (HIGH PRIORITY)")
                parts.append("The following core innovation modules from the proposal are NOT fully implemented:")
                for gap in gaps_with_issues:
                    status_icon = "⚠️" if gap["status"] == "simplified" else "❌"
                    parts.append(f"- {status_icon} **{gap['module_name']}** [{gap['status']}]: {gap['description']}")
                parts.append("")
            
            # Phase 1.5: Implementation hints
            impl_hints = response.get("implementation_hints", "")
            if impl_hints and gaps_with_issues:
                parts.append("## How to Implement Missing Modules")
                parts.append(impl_hints)
                parts.append("")
            
            # Phase 2: Hyperparameter hints
            hp_hints = response.get("hyperparameter_hints", "")
            if hp_hints:
                parts.append("## Hyperparameter Optimization Suggestions")
                parts.append(hp_hints)
                parts.append("")
            
            # Priority changes
            priority = response.get("priority_changes", [])
            if priority:
                parts.append("## Priority Changes (ordered by expected impact)")
                for i, change in enumerate(priority, 1):
                    parts.append(f"{i}. {change}")
                parts.append("")
            
            combined_hints = "\n".join(parts).strip()
            
            if len(combined_hints) > 20:
                state["optimization_hints"] = combined_hints
                state["optimization_count"] = optimization_count + 1
                
                logger.info(f"📈 Generated optimization hints (total count: {state['optimization_count']}, "
                            f"proposal gaps: {len(gaps_with_issues)}/{len(proposal_gaps)})")
                logger.info(f"📈 Hints preview: {combined_hints[:200]}...")
                
                return combined_hints
            else:
                logger.warning("Optimization hints generation returned empty combined hints")
                return None
                
        except Exception as e:
            logger.error(f"Failed to generate optimization hints: {e}")
            return None
    
    def get_optimization_hints(self, substage_name: str) -> Optional[str]:
        """
        Get current optimization hints for a substage
        
        Args:
            substage_name: Substage name
            
        Returns:
            Optimization hints string or None
        """
        if substage_name not in self.substage_refinement_state:
            return None
        
        state = self.substage_refinement_state[substage_name]
        return state.get("optimization_hints", None)
    
    def _collect_node_summary(self, node) -> str:
        """
        Collect summary information from a node
        
        Args:
            node: Node to summarize
            
        Returns:
            Formatted node summary
        """
        if node is None:
            return "No node available"
        
        summary = f"Node ID: {node.id[:8]}...\n"
        summary += f"Metric: {node.metric.get_mean_value():.4f}\n" if node.metric else "Metric: N/A\n"
        
        if node.plan:
            summary += f"Design: {node.plan[:300]}...\n"
        
        if node.analysis:
            summary += f"Analysis: {node.analysis[:300]}...\n"
        
        return summary
    
    @staticmethod
    def _find_branch_root(node) -> "Node":
        """Walk up the parent chain to the first node whose parent is the
        baseline (parent.parent is None) or whose parent is a precomputed
        baseline.  Returns the branch root (direct child of baseline)."""
        current = node
        while current.parent is not None:
            parent = current.parent
            if parent.parent is None or getattr(parent, 'is_precomputed_baseline', False):
                return current
            current = parent
        return current

    def _collect_all_nodes_metrics(self, journal, baseline_metric: float) -> str:
        """Collect metric summary from all nodes, grouped by branch.

        Each branch is identified by its root node (direct child of the
        baseline).  Nodes within a branch are listed in chronological order.
        This allows the diagnostic LLM to analyse per-branch trends rather
        than being confused by interleaved multi-branch metrics.

        Excludes the Stage 1 baseline node (parent is None) and stale nodes.

        Args:
            journal: Journal containing all nodes for this substage/proposal
            baseline_metric: Best baseline metric value

        Returns:
            Formatted markdown with per-branch tables and global statistics
        """
        if not journal.nodes:
            return "No previous experiments."

        proposal_nodes = [
            n for n in journal.nodes
            if n.parent is not None and not getattr(n, 'is_stale', False)
        ]

        if not proposal_nodes:
            return "No proposal implementation experiments yet (only baseline exists)."

        # Group nodes by branch root id
        from collections import OrderedDict
        branches: OrderedDict[str, list] = OrderedDict()
        for node in proposal_nodes:
            branch_root = self._find_branch_root(node)
            root_id = branch_root.id
            if root_id not in branches:
                branches[root_id] = []
            branches[root_id].append(node)

        lines: list[str] = []

        if len(branches) == 1:
            # Single branch — use flat table (backward-compatible)
            lines.append("| Node | Status | Metric | Gap to Baseline |")
            lines.append("|------|--------|--------|-----------------|")
            for i, node in enumerate(proposal_nodes):
                lines.append(self._format_node_row(i + 1, node, baseline_metric))
        else:
            # Multiple branches — group by branch
            for branch_idx, (root_id, nodes) in enumerate(branches.items(), 1):
                lines.append(f"### Branch {branch_idx} (root: {root_id[:8]}, {len(nodes)} experiments)")
                lines.append("")
                lines.append("| Node | Status | Metric | Gap to Baseline |")
                lines.append("|------|--------|--------|-----------------|")
                for i, node in enumerate(nodes):
                    lines.append(self._format_node_row(i + 1, node, baseline_metric))
                lines.append("")

        # Global statistics across all branches
        valid_metrics = [
            n.metric.get_mean_value()
            for n in proposal_nodes
            if n.metric and not math.isnan(n.metric.get_mean_value())
        ]
        if valid_metrics:
            best = max(valid_metrics)
            worst = min(valid_metrics)
            lines.append(f"\n**Global statistics ({len(branches)} branch(es))**:")
            lines.append(f"Best result: {best:.4f}")
            lines.append(f"Worst result: {worst:.4f}")
            lines.append(f"Range (max - min): {best - worst:.4f}")
            lines.append(f"Best gap to baseline: {baseline_metric - best:+.4f}")
            lines.append(f"Total proposal experiments: {len(proposal_nodes)}")

        return "\n".join(lines)

    @staticmethod
    def _format_node_row(index: int, node, baseline_metric: float) -> str:
        """Format a single node as a markdown table row."""
        status = _get_node_status_label(node)
        if node.metric:
            val = node.metric.get_mean_value()
            if not math.isnan(val):
                gap = baseline_metric - val
                return f"| {index} ({node.id[:8]}) | {status} | {val:.4f} | {gap:+.4f} |"
            else:
                return f"| {index} ({node.id[:8]}) | {status} | nan | N/A |"
        else:
            return f"| {index} ({node.id[:8]}) | {status} | N/A | N/A |"
    
    def reset_substage_state(self, substage_name: str):
        """Reset substage state (for when new substage starts)"""
        self.substage_refinement_state[substage_name] = {
            "refinement_count": 0,
            "last_refinement_node_idx": 0,
            "has_beat_baseline": False,
            "optimization_count": 0,
            "optimization_hints": None
        }
    
    def _load_framework_documentation(self) -> str:
        """Load CamylaNet framework documentation for feasibility assessment.
        
        Returns:
            Framework documentation content, or empty string if not found
        """
        try:
            doc_path = Path(__file__).parent.parent.parent / "skills" / "frameworks" / "camylanet" / "documentation.md"
            if doc_path.exists():
                content = doc_path.read_text(encoding='utf-8')
                logger.info(f"Loaded CamylaNet framework documentation ({len(content)} chars)")
                return content
            else:
                logger.warning(f"Framework documentation not found at: {doc_path}")
                return ""
        except Exception as e:
            logger.warning(f"Failed to load framework documentation: {e}")
            return ""

    def _save_llm_interaction(
        self,
        substage_name: str,
        interaction_type: str,
        system_message,
        user_message,
        func_spec: Optional[FunctionSpec],
        response,
        model: str,
        temperature: float,
    ):
        """Save complete LLM input/output to a markdown file.
        
        Args:
            substage_name: Current substage name (e.g. '2_creative_research_1_proposal_1')
            interaction_type: Type of interaction ('diagnosis', 'refinement', 'optimization')
            system_message: Raw system message passed to query()
            user_message: Raw user message passed to query()
            func_spec: Function spec if used
            response: LLM response (str or dict)
            model: Model identifier
            temperature: Temperature used
        """
        try:
            save_dir = Path(self.cfg.log_dir) / f"stage_{substage_name}" / "proposal_diagnostic"
            save_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{interaction_type}_{timestamp}.md"
            filepath = save_dir / filename

            compiled_system = compile_prompt_to_md(system_message) if system_message else None
            compiled_user = compile_prompt_to_md(user_message) if user_message else None

            lines = [
                f"# Proposal Diagnostic LLM Interaction: {interaction_type}",
                f"",
                f"- **Timestamp**: {datetime.now().isoformat()}",
                f"- **Model**: {model}",
                f"- **Temperature**: {temperature}",
                f"- **Substage**: {substage_name}",
                f"- **Interaction Type**: {interaction_type}",
            ]
            if func_spec:
                lines.append(f"- **Function Spec**: {func_spec.name}")

            lines.append("")
            lines.append("---")
            lines.append("")

            # System message
            lines.append("## System Message")
            lines.append("")
            if compiled_system:
                lines.append(compiled_system if isinstance(compiled_system, str) else json.dumps(compiled_system, ensure_ascii=False, indent=2))
            else:
                lines.append("*(None)*")
            lines.append("")

            # User message
            lines.append("---")
            lines.append("")
            lines.append("## User Message (LLM Input)")
            lines.append("")
            if compiled_user:
                lines.append(compiled_user if isinstance(compiled_user, str) else json.dumps(compiled_user, ensure_ascii=False, indent=2))
            else:
                lines.append("*(None)*")
            lines.append("")

            # Function spec details
            if func_spec:
                lines.append("---")
                lines.append("")
                lines.append("## Function Spec")
                lines.append("")
                lines.append(f"**Name**: {func_spec.name}")
                lines.append(f"**Description**: {func_spec.description}")
                lines.append("")
                lines.append("**JSON Schema**:")
                lines.append("```json")
                lines.append(json.dumps(func_spec.json_schema, ensure_ascii=False, indent=2))
                lines.append("```")
                lines.append("")

            # LLM response
            lines.append("---")
            lines.append("")
            lines.append("## LLM Response (Output)")
            lines.append("")
            if isinstance(response, dict):
                lines.append("```json")
                lines.append(json.dumps(response, ensure_ascii=False, indent=2))
                lines.append("```")
            elif isinstance(response, str):
                lines.append(response)
            else:
                lines.append(str(response))

            filepath.write_text("\n".join(lines), encoding="utf-8")
            logger.info(f"💾 Saved LLM interaction to: {filepath}")

        except Exception as e:
            logger.warning(f"Failed to save LLM interaction ({interaction_type}): {e}")
