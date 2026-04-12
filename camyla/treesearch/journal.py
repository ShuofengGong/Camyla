from __future__ import annotations
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Literal, Optional, Union, Tuple
import copy
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Set
from .interpreter import ExecutionResult

from dataclasses_json import DataClassJsonMixin
from .utils.metric import MetricValue, WorstMetricValue
from .utils.response import trim_long_string
from .utils.stage_constants import MAIN_STAGE_GOALS, MAIN_STAGE_DICT

from rich import print

import logging

logger = logging.getLogger(__name__)


@dataclass(eq=False)
class Node(DataClassJsonMixin):
    """A single node in the solution tree. Contains code, execution results, and evaluation information."""

    # ---- code & plan ----
    plan: str = field(default="")  # type: ignore
    overall_plan: str = field(default="")  # type: ignore
    code: str = field(default="")  # type: ignore
    plot_code: str = field(default=None)  # type: ignore
    plot_plan: str = field(default=None)  # type: ignore

    # ---- general attrs ----
    step: int = field(default=None)  # type: ignore
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    ctime: float = field(default_factory=lambda: time.time())
    parent: Optional["Node"] = field(default=None)
    parent_id: Optional[str] = field(default=None)  # Store parent ID for serialization
    children: set["Node"] = field(default_factory=set)
    exp_results_dir: str = field(default=None)  # type: ignore
    origin_stage: str = field(default=None)  # origin_stage field

    # ---- execution info ----
    _term_out: List[str] = field(default=None)  # type: ignore
    exec_time: float = field(default=None)  # type: ignore
    exc_type: Optional[str] = field(default=None)
    exc_info: Optional[dict] = field(default=None)
    exc_stack: Optional[List[Tuple]] = field(default=None)

    # ---- parsing info ----
    parse_metrics_plan: str = field(default="")
    parse_metrics_code: str = field(default="")
    # parse_exec_result: ExecutionResult = field(default=None)
    parse_term_out: List[str] = field(default=None)
    parse_exc_type: Optional[str] = field(default=None)
    parse_exc_info: Optional[dict] = field(default=None)
    parse_exc_stack: Optional[List[Tuple]] = field(default=None)

    # ---- plot execution info ----
    plot_term_out: List[str] = field(default=None)  # type: ignore
    plot_exec_time: float = field(default=None)  # type: ignore
    plot_exc_type: Optional[str] = field(default=None)
    plot_exc_info: Optional[dict] = field(default=None)
    plot_exc_stack: Optional[List[Tuple]] = field(default=None)

    # ---- evaluation ----
    # post-execution result analysis (findings/feedback)
    analysis: str = field(default=None)  # type: ignore
    metric: MetricValue = field(default=None)  # type: ignore
    # whether the node needs revision (execution error or underperforming)
    # True if: exc_type is not None, no valid metric, or diagnostic module flagged performance issue
    is_buggy: bool = field(default=None)  # type: ignore
    is_buggy_plots: bool = field(default=None)

    # ---- plotting ----
    plot_data: dict = field(default_factory=dict)
    plots_generated: bool = field(default=False)
    plots: List[str] = field(default_factory=list)  # Relative paths for visualization
    plot_paths: List[str] = field(
        default_factory=list
    )  # Absolute paths for programmatic access

    # ---- Plot feedback ----
    datasets_successfully_tested: List[str] = field(default_factory=list)

    # ---- execution time feedback ----
    exec_time_feedback: str = field(default="")

    # ---- ablation study ----
    ablation_name: str = field(default=None)
    ablation_type: str = field(default=None)  # "removal" or "comparison"

    # ---- innovation combination ----
    combination_name: str = field(default=None)

    # ---- hyperparam tuning (legacy) ----
    hyperparam_name: str = field(default=None)

    # ---- param tuning (legacy, kept for backward compatibility) ----
    param_tuning_name: str = field(default=None)

    # ---- precomputed baseline info ----
    baseline_model: str = field(default=None)  # Model name (e.g., nnUNet, SegResNet)
    is_precomputed_baseline: bool = field(default=False)  # Whether this node is from precomputed baseline

    # ---- diagnostic info (from proposal diagnostic) ----
    # Stores diagnosis information for underperforming nodes (NOT marked buggy)
    # Format: {"type": str, "reasoning": str, "improvement_suggestions": List[Dict]}
    diagnostic_info: Optional[Dict[str, Any]] = field(default=None)

    # ---- proposal snapshot ----
    # Stores the proposal content at the time this node was created (Stage 2 only)
    # Ensures debug/improve uses the same proposal version the node was created with
    proposal_content: Optional[str] = field(default=None)

    # ---- stale flag (proposal refinement isolation) ----
    # When proposal_infeasible triggers a proposal refinement, all pre-refinement
    # Stage 2 nodes are marked stale so QWBE and diagnostic skip them.
    is_stale: bool = field(default=False)

    # ---- UCB tree search visit count ----
    # Number of times this node has been selected as a parent to produce a child node
    visit_count: int = field(default=0)

    # ---- modification summary (LLM-generated) ----
    # 2-3 sentence summary of what changed compared to the parent node
    modification_summary: str = field(default="")

    # ---- competition siblings (subagent competition metadata) ----
    # Non-winner candidates from subagent competition, stored on the winner node
    competition_siblings: list = field(default_factory=list)

    def __post_init__(self) -> None:
        # Ensure children is a set even if initialized with a list
        if isinstance(self.children, list):
            self.children = set(self.children)
        # Only try to add to parent's children if parent is a Node object
        if self.parent is not None and not isinstance(self.parent, str):
            self.parent.children.add(self)
            if self.parent_id is None:
                self.parent_id = self.parent.id

    def __deepcopy__(self, memo):
        # Create a new instance with copied attributes
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        # Copy all attributes except parent and children to avoid circular references
        for k, v in self.__dict__.items():
            if k not in ("parent", "children"):
                setattr(result, k, copy.deepcopy(v, memo))

        # Handle parent and children separately
        result.parent = self.parent  # Keep the same parent reference
        result.children = set()  # Start with empty children set

        return result

    def __getstate__(self):
        """Return state for pickling"""
        state = self.__dict__.copy()
        if hasattr(self, "id"):
            state["id"] = self.id
        # Always sync parent_id from parent.id as the basis for rebuilding relationships
        if state.get("parent_id") is None and self.parent is not None and not isinstance(self.parent, str):
            state["parent_id"] = self.parent.id
        return state

    def __setstate__(self, state):
        """Set state during unpickling - restore the ID field first"""

        # 🔥 Restore the id field first to avoid AttributeError in subsequent operations
        if 'id' in state and state['id'] is not None:
            self.id = state['id']
        else:
            # Back-compat: generate a new ID for old checkpoints
            import uuid
            self.id = uuid.uuid4().hex
            logger.debug("Generated new ID for node missing an ID: %s", self.id)

        # Restore all other attributes
        self.__dict__.update(state)

        # 🔥 Ensure critical fields exist
        if not hasattr(self, 'children'):
            self.children = set()

    @property
    def stage_name(self) -> Literal["draft", "debug", "improve"]:
        """
        Return the stage of the node:
        - "stage" if the node is an initial solution draft
        - "debug" if the node is the result of a debugging step
        - "improve" if the node is the result of an improvement step
        """
        if self.parent is None:
            return "draft"
        return "debug" if self.parent.is_buggy else "improve"

    def absorb_exec_result(self, exec_result: ExecutionResult):
        """Absorb the result of executing the code from this node."""
        self._term_out = exec_result.term_out
        self.exec_time = exec_result.exec_time
        self.exc_type = exec_result.exc_type
        self.exc_info = exec_result.exc_info
        self.exc_stack = exec_result.exc_stack

    def absorb_plot_exec_result(self, plot_exec_result: ExecutionResult):
        """Absorb the result of executing the plotting code from this node."""
        self.plot_term_out = plot_exec_result.term_out
        self.plot_exec_time = plot_exec_result.exec_time
        self.plot_exc_type = plot_exec_result.exc_type
        self.plot_exc_info = plot_exec_result.exc_info
        self.plot_exc_stack = plot_exec_result.exc_stack

    @property
    def term_out(self) -> str:
        """Get the terminal output of the code execution (after truncating it)."""
        if not self._term_out:
            return ""
        return trim_long_string("".join(self._term_out))

    @property
    def is_leaf(self) -> bool:
        """Check if the node is a leaf node in the solution tree."""
        return not self.children

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        
        # 🔥 Pickle unpickling ordering fix: handle cases where id is temporarily unavailable
        self_has_id = hasattr(self, 'id') and self.id is not None
        other_has_id = hasattr(other, 'id') and other.id is not None

        if self_has_id and other_has_id:
            return self.id == other.id
        elif not self_has_id and not other_has_id:
            # Neither object has an id: fall back to identity comparison
            return self is other
        else:
            # One has an id and the other does not: definitely not equal
            return False

    def __hash__(self):
        # 🔥 Pickle unpickling ordering fix: provide a fallback when id is unavailable
        if hasattr(self, 'id') and self.id is not None:
            return hash(self.id)
        else:
            # Temporary hash during unpickling: use the object's memory address
            # This keeps the object set-insertable before the id attribute is assigned
            return hash(id(self))

    @property
    def debug_depth(self) -> int:
        """
        Length of the current debug path
        - 0 if the node is not a debug node (parent is not buggy)
        - 1 if the parent is buggy but the skip parent isn't
        - n if there were n consecutive debugging steps
        """
        if self.stage_name != "debug":
            return 0
        return self.parent.debug_depth + 1  # type: ignore

    def _safe_relative_path(self, path: str) -> str:
        """
        Convert path to relative path if possible, otherwise return absolute path.
        This handles cases where the path is not under the current working directory.
        """
        try:
            return str(Path(path).resolve().relative_to(os.getcwd()))
        except ValueError:
            # Path is not under cwd, return absolute path instead
            return str(Path(path).resolve())

    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for serialization"""
        node_dict = {
            "code": self.code,
            "plan": self.plan,
            "overall_plan": (
                self.overall_plan if hasattr(self, "overall_plan") else None
            ),
            "plot_code": self.plot_code,
            "plot_plan": self.plot_plan,
            "step": self.step,
            "id": self.id,
            "ctime": self.ctime,
            "_term_out": self._term_out,
            "parse_metrics_plan": self.parse_metrics_plan,
            "parse_metrics_code": self.parse_metrics_code,
            "parse_term_out": self.parse_term_out,
            "parse_exc_type": self.parse_exc_type,
            "parse_exc_info": self.parse_exc_info,
            "parse_exc_stack": self.parse_exc_stack,
            "exec_time": self.exec_time,
            "exc_type": self.exc_type,
            "exc_info": self.exc_info,
            "exc_stack": self.exc_stack,
            "analysis": self.analysis,
            "exp_results_dir": (
                self._safe_relative_path(self.exp_results_dir)
                if self.exp_results_dir
                else None
            ),
            "metric": {
                "value": self.metric.value if self.metric else None,
                "maximize": self.metric.maximize if self.metric else None,
                "name": self.metric.name if hasattr(self.metric, "name") else None,
                "description": (
                    self.metric.description
                    if hasattr(self.metric, "description")
                    else None
                ),
            },
            "is_buggy": self.is_buggy,
            "is_buggy_plots": self.is_buggy_plots,
            "parent_id": None if self.parent is None else self.parent.id,
            "children": [child.id for child in self.children] if self.children else [],
            "plot_data": self.plot_data,
            "plots_generated": self.plots_generated,
            "plots": self.plots,
            "plot_paths": (
                [
                    self._safe_relative_path(p)
                    for p in self.plot_paths
                ]
                if self.plot_paths
                else []
            ),
            "datasets_successfully_tested": self.datasets_successfully_tested,
            "ablation_name": self.ablation_name,
            "ablation_type": getattr(self, "ablation_type", None),
            "combination_name": self.combination_name,
            "hyperparam_name": self.hyperparam_name,
            "param_tuning_name": getattr(self, "param_tuning_name", None),
            "exec_time_feedback": self.exec_time_feedback,
            # Ensure origin_stage is included
            "origin_stage": getattr(self, "origin_stage", None),
            # Precomputed baseline info
            "baseline_model": getattr(self, "baseline_model", None),
            "is_precomputed_baseline": getattr(self, "is_precomputed_baseline", False),
            # Diagnostic info (from proposal diagnostic)
            "diagnostic_info": getattr(self, "diagnostic_info", None),
            # Proposal snapshot (only record presence and length; full content is retained via pickle)
            "has_proposal_snapshot": getattr(self, "proposal_content", None) is not None,
            # modification summary and competition metadata
            "modification_summary": getattr(self, "modification_summary", ""),
            "competition_siblings": getattr(self, "competition_siblings", []),
            # stale flag and UCB visit count
            "is_stale": self.is_stale,
            "visit_count": self.visit_count,
        }
        return node_dict

    @classmethod
    def from_dict(cls, data: Dict, journal: Optional["Journal"] = None) -> "Node":
        """Create a Node from a dictionary, optionally linking to journal for relationships"""
        # Remove relationship IDs from constructor data
        parent_id = data.pop("parent_id", None)
        children = data.pop("children", [])

        # Handle metric conversion
        metric_data = data.pop("metric", None)
        if metric_data:
            if isinstance(metric_data, dict):
                data["metric"] = MetricValue(
                    value=metric_data["value"],
                    maximize=metric_data["maximize"],
                    name=metric_data["name"],
                    description=metric_data["description"],
                )
            else:
                # Handle legacy format or None
                data["metric"] = (
                    WorstMetricValue()
                    if data.get("is_buggy")
                    else MetricValue(metric_data)
                )

        # Remove non-constructor fields that may exist in serialized data
        data.pop("has_proposal_snapshot", None)

        # Create node instance
        node = cls(**data)

        # Always store parent_id for reference, even if we can't restore the relationship
        node.parent_id = parent_id

        # If journal is provided, restore relationships
        if journal is not None and parent_id:
            parent = journal.get_node_by_id(parent_id)
            if parent:
                node.parent = parent
                parent.children.add(node)

        return node


@dataclass
class InteractiveSession(DataClassJsonMixin):
    """
    A collection of nodes for an interaction session
    (when the agent interacts with a Jupyter notebook-like interface).
    """

    nodes: List[Node] = field(default_factory=list)
    completed: bool = False

    def append(self, node: Node) -> None:
        node.step = len(self.nodes)
        self.nodes.append(node)

    def generate_nb_trace(self, include_prompt, comment_headers=True) -> str:
        """Generate a trace of the interactive session in IPython format."""
        trace = []
        header_prefix = "## " if comment_headers else ""
        for n in self.nodes:
            trace.append(f"\n{header_prefix}In [{n.step+1}]:\n")
            trace.append(n.code)
            trace.append(f"\n{header_prefix}Out [{n.step+1}]:\n")
            trace.append(n.term_out)

        if include_prompt and self.nodes:
            trace.append(f"\n{header_prefix}In [{self.nodes[-1].step+2}]:\n")

        return "\n".join(trace).strip()


@dataclass
class Journal:
    """A collection of nodes representing the solution tree."""

    nodes: List[Node] = field(default_factory=list)
    stage_name: Optional[str] = None  # populated by AgentManager
    _cached_best_node: Optional[Node] = field(default=None, init=False)
    _cache_invalidated: bool = field(default=True, init=False)

    def __getitem__(self, idx: int) -> Node:
        return self.nodes[idx]

    def __len__(self) -> int:
        """Return the number of nodes in the journal."""
        return len(self.nodes)

    def append(self, node: Node) -> None:
        """Append a new node to the journal. Also tag node with origin stage for later reports."""
        if self.stage_name and not hasattr(node, "origin_stage"):
            node.origin_stage = self.stage_name
        node.step = len(self.nodes)
        self.nodes.append(node)
        # Invalidate cache when new node is added
        self._cache_invalidated = True

    @property
    def draft_nodes(self) -> List[Node]:
        """Return a list of nodes representing intial coding drafts"""
        return [n for n in self.nodes if n.parent is None]

    @property
    def buggy_nodes(self) -> List[Node]:
        """Return a list of nodes that are considered buggy by the agent."""
        return [n for n in self.nodes if n.is_buggy]

    @property
    def good_nodes(self) -> List[Node]:
        """Return a list of nodes that are not considered buggy by the agent."""
        if logger.isEnabledFor(logging.DEBUG):
            list_of_nodes = [
                [n.step, n.parent.step if n.parent else None, n.id, n.is_buggy, n.is_buggy_plots]
                for n in self.nodes
            ]
            logger.debug("all nodes ID and is_buggy/is_buggy_plots flags: %s", list_of_nodes)
        return [
            n for n in self.nodes if n.is_buggy is False and (n.is_buggy_plots is False or n.is_buggy_plots is None)
        ]

    def get_node_by_id(self, node_id: str) -> Optional[Node]:
        """Get a node by its ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def get_metric_history(self) -> List[MetricValue]:
        """Return a list of all metric values in the journal."""
        return [n.metric for n in self.nodes]

    def get_best_node(self, only_good=True, use_val_metric_only=False, exclude_baselines=False) -> None | Node:
        """Return the best solution found so far using deterministic MetricValue comparison.
        
        Uses MetricValue.__gt__() as the single source of truth:
        primary metric (Dice) decides unless the Dice difference is below
        TIEBREAK_THRESHOLD, then secondary metric (HD95) breaks the tie
        directly with lower-is-better comparison.
        
        Args:
            only_good: If True, only consider non-buggy nodes.
            use_val_metric_only: If True, use validation metric for comparison.
            exclude_baselines: If True, exclude precomputed baseline nodes (is_precomputed_baseline=True).
        """
        if only_good and not use_val_metric_only and not exclude_baselines and not self._cache_invalidated and self._cached_best_node:
            return self._cached_best_node

        if only_good:
            nodes = self.good_nodes
            if not nodes:
                return None
        else:
            nodes = self.nodes

        if exclude_baselines:
            nodes = [n for n in nodes if not getattr(n, 'is_precomputed_baseline', False)]
            if not nodes:
                return None

        result = max(nodes, key=lambda n: n.metric)
        logger.info(f"Selected best node {result.id} with metric: {result.metric}")

        if only_good and not use_val_metric_only and not exclude_baselines:
            self._cached_best_node = result
            self._cache_invalidated = False
        return result

    def get_nodes_by_stage(self, stage_name: str) -> List[Node]:
        """Return nodes that belong to a specific stage."""
        return [n for n in self.nodes if getattr(n, 'origin_stage', None) == stage_name]

    def get_good_nodes_by_stage(self, stage_name: str) -> List[Node]:
        """Return good nodes that belong to a specific stage."""
        stage_nodes = self.get_nodes_by_stage(stage_name)
        return [n for n in stage_nodes if n.is_buggy is False and (n.is_buggy_plots is False or n.is_buggy_plots is None)]

    def get_buggy_nodes_by_stage(self, stage_name: str) -> List[Node]:
        """Return buggy nodes that belong to a specific stage."""
        stage_nodes = self.get_nodes_by_stage(stage_name)
        return [n for n in stage_nodes if n.is_buggy]

    def generate_summary(self, include_code: bool = False, max_analysis_chars: int = 2000) -> str:
        """Generate a per-experiment structured summary of the research progress.

        Each experiment gets its own brief entry listing what was attempted,
        the outcome, and why it failed if applicable.  No LLM call is used;
        the output is deterministic and based on stored node data.

        Args:
            include_code: Whether to include code in the summary
            max_analysis_chars: Maximum characters to include from each node's analysis
        """
        stage_name = self.stage_name
        all_nodes = [n for n in self.nodes if n.parent is not None or n.is_buggy is not None]
        stage_info = f" for stage '{stage_name}'" if stage_name else ""

        if not all_nodes:
            return f"No experiments conducted yet{stage_info}."

        lines = [f"## Experiment History{stage_info}", ""]

        for idx, node in enumerate(all_nodes, 1):
            # Determine status label
            is_baseline = getattr(node, 'is_precomputed_baseline', False) or node.parent is None
            if is_baseline:
                status = "Baseline"
            elif not node.is_buggy and node.is_buggy is not None:
                status = "Success"
            elif node.is_buggy and node.exc_type is not None:
                status = "Error"
            elif node.is_buggy:
                status = "Underperforming"
            else:
                status = "Pending"

            # Metric string
            metric_str = ""
            if node.metric is not None:
                try:
                    val = node.metric.get_mean_value()
                    if not math.isnan(val):
                        metric_str = f", metric: {val:.4f}"
                except Exception:
                    metric_str = f", metric: {node.metric}"

            lines.append(f"### Experiment {idx} [{status}{metric_str}]")
            lines.append(f"- Design: {node.plan}")

            if node.exc_type:
                lines.append(f"- Error: {node.exc_type}")

            # Modification summary
            mod_summary = getattr(node, 'modification_summary', '')
            if mod_summary:
                lines.append(f"- Modification: {mod_summary}")

            # Analysis (truncated)
            analysis = node.analysis or ""
            if analysis:
                if len(analysis) > max_analysis_chars:
                    analysis = analysis[:max_analysis_chars] + "..."
                lines.append(f"- Analysis: {analysis}")

            if include_code and node.code:
                lines.append(f"- Code: {node.code}")

            # Competition siblings
            siblings = getattr(node, 'competition_siblings', [])
            if siblings:
                lines.append(f"- Competition ({len(siblings)+1} subagents, this was the winner):")
                for sib in siblings:
                    sib_status = "Error" if sib.get("is_buggy") else "OK"
                    sib_mod = sib.get("modification_summary", "N/A")
                    lines.append(f"  - {sib['label']} [{sib_status}, {sib['metric']}]: {sib_mod[:300]}")

            lines.append("")

        return "\n".join(lines)

    def rebuild_relationships(self):
        """Rebuild parent/children references from parent_id, removing ghost parents created by unpickling.

        After pickle deserialization, a child's parent may point to a "ghost"
        object not in journal.nodes (same UUID as the seed, but a different
        Python object). This method redirects all parent references to the
        actual nodes in journal.nodes and rebuilds the children sets.
        """
        uuid_to_node = {n.id: n for n in self.nodes}

        # First sync parent_id from parent references (in case old checkpoints lack parent_id)
        for n in self.nodes:
            if getattr(n, 'parent_id', None) is None:
                if n.parent is not None and not isinstance(n.parent, str):
                    n.parent_id = n.parent.id

        # Clear all children sets in preparation for rebuilding
        for n in self.nodes:
            n.children = set()

        # Rebuild parent references and children sets from parent_id
        rebuilt = 0
        for n in self.nodes:
            pid = getattr(n, 'parent_id', None)
            if pid is not None:
                parent_node = uuid_to_node.get(pid)
                if parent_node is not None:
                    n.parent = parent_node
                    parent_node.children.add(n)
                    rebuilt += 1
                else:
                    n.parent = None
                    logger.warning(
                        "rebuild_relationships: node %s parent_id %s not found in journal",
                        n.id[:8], pid[:8],
                    )
            else:
                n.parent = None

        self._cache_invalidated = True
        logger.debug(
            "rebuild_relationships: %d nodes, %d parent links rebuilt",
            len(self.nodes), rebuilt,
        )

    def to_dict(self):
        """Convert journal to a JSON-serializable dictionary"""
        return {"nodes": [node.to_dict() for node in self.nodes]}
