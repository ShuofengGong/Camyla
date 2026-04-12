from typing import List, Optional, Dict, Callable, Any, Tuple
import pickle
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
import logging
from datetime import datetime
from .parallel_agent import ParallelAgent
from .journal import Journal, Node
import copy
import re
from .backend import query, FunctionSpec
import json
from .utils.serialize import parse_markdown_to_dict
from .utils.metric import MetricValue, WorstMetricValue
from .utils.stage_constants import MAIN_STAGE_DICT, MAIN_STAGE_GOALS
import os
import shutil
import itertools
import traceback
import inspect
import numpy as np
from omegaconf import OmegaConf

from camyla.model_config import get_role

# 🆕 Import Proposal diagnostic module
from .proposal_diagnostic import ProposalDiagnostic


logger = logging.getLogger(__name__)


stage_config_spec = FunctionSpec(
    name="generate_stage_config",
    description="Generate configuration for the next experimental stage",
    json_schema={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Brief, descriptive name for the stage",
            },
            "description": {
                "type": "string",
                "description": "Detailed description of the stage's purpose",
            },
            "goals": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of specific, measurable goals for this stage",
            },
            "max_iterations": {
                "type": "integer",
                "description": "Maximum number of iterations to run in this stage",
            },
        },
        "required": ["name", "description", "goals", "max_iterations"],
    },
)

stage_progress_eval_spec = FunctionSpec(
    name="evaluate_stage_progression",
    description="Evaluate readiness to progress to next experimental stage",
    json_schema={
        "type": "object",
        "properties": {
            "ready_for_next_stage": {
                "type": "boolean",
                "description": "Whether the experiment is ready to progress to next stage",
            },
            "reasoning": {
                "type": "string",
                "description": "Detailed reasoning for the progression decision",
            },
            "recommendations": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific recommendations for current or next stage",
            },
            "suggested_focus": {
                "type": "string",
                "description": "Key areas to focus on in the next iterations",
            },
        },
        "required": ["ready_for_next_stage", "reasoning", "recommendations"],
    },
)


stage_completion_eval_spec = FunctionSpec(
    name="evaluate_stage_completion",
    description="Evaluate if the current stage is complete",
    json_schema={
        "type": "object",
        "properties": {
            "is_complete": {
                "type": "boolean",
                "description": "Whether the current stage is complete",
            },
            "reasoning": {
                "type": "string",
                "description": "Detailed reasoning for the decision",
            },
            "missing_criteria": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of criteria still needed",
            },
        },
        "required": ["is_complete", "reasoning", "missing_criteria"],
    },
)


@dataclass
class Stage:
    name: str
    description: str
    goals: List[str]
    max_iterations: int
    num_drafts: int
    stage_number: int
    proposal_idx: Optional[int] = None  # Explicit proposal index (0-based); used by Stage 2


@dataclass
class StageTransition:
    """Records transition between stages and the reasoning"""

    from_stage: str
    to_stage: str
    reason: str
    config_adjustments: Dict[str, Any]


class AgentManager:
    def __init__(self, task_desc: str, cfg: Any, workspace_dir: Path):
        self.task_desc = json.loads(task_desc)

        # Only support new structure - validate required keys
        # The baseline field is now optional; it will be generated from the dataset automatically
        required_keys = ["dataset"]
        for k in required_keys:
            if k not in self.task_desc.keys():
                raise ValueError(f"Key {k} not found in task_desc structure")

        self.use_new_structure = True

        self.cfg = cfg
        self.workspace_dir = workspace_dir
        
        # 🐛 Debug baseline mode: dynamically override config parameters to speed up tests (without writing to files)
        # Use OmegaConf.select for safe access to possibly missing keys
        self._debug_baseline_mode = OmegaConf.select(self.cfg, 'debug_baseline', default=False)
        if self._debug_baseline_mode:
            logger.info("🐛 DEBUG MODE: Applying debug parameter overrides")
            
            # target_papers.phase2: 6 -> 3
            if OmegaConf.select(self.cfg, 'idea_generation.literature_search') is not None:
                self.cfg.idea_generation.literature_search.target_papers.phase2 = 3
                logger.debug("   - target_papers.phase2: 3")

            # stage2_max_iters -> 2
            if OmegaConf.select(self.cfg, 'experiment.stages') is not None:
                self.cfg.experiment.stages.stage2_max_iters = 2
                logger.debug("   - stage2_max_iters: 2")

            # max_iterations_per_innovation -> 5
            if OmegaConf.select(self.cfg, 'experiment.stage2') is not None:
                self.cfg.experiment.stage2.max_iterations_per_innovation = 5
                logger.debug("   - max_iterations_per_innovation: 5")

            # num_proposals -> 2
            if OmegaConf.select(self.cfg, 'idea_generation.research_proposal') is not None:
                self.cfg.idea_generation.research_proposal.num_proposals = 2
                logger.debug("   - num_proposals: 2")
        
        # Extract the unique experiment name (exp_name) for camylanet
        self.exp_name = self._extract_exp_name()
        logger.info(f"🎯 AgentManager initialized with exp_name: {self.exp_name}")

        # Whether to use the custom innovation-queue workflow for research proposals
        # Can be set in the YAML config as `agent.use_innovation_queue: true` (default: false)
        self.use_innovation_queue: bool = True  # proposals queue flow is now the default

        # Literature search enabled flag — driven by idea_generation.enabled
        self.use_literature_search: bool = getattr(self.cfg.idea_generation, "enabled", True)

        # literature_search_only mode removed in refactor (unused path)
        self.literature_search_only: bool = False

        # 🔥 Research proposal management: declare as an empty list first; filled by _initialize_proposal_queue_system.
        # Must be declared before calling _initialize_proposal_queue_system, otherwise its values would be overwritten.
        self.proposals: List[Dict[str, Any]] = []
        self.successful_proposals: List[Dict[str, Any]] = []
        self.failed_proposals: List[Dict[str, Any]] = []

        # Initialize proposal queue with literature search integration
        self._initialize_proposal_queue_system()
        
        # 🆕 Initialize the proposal diagnostic (for in-substage on-the-fly refinement)
        self._initialize_proposal_diagnostic()
        
        # Global count of executed sub-stages to enforce budget
        self.executed_substages_count = 0  # For Stage 2 (creative_research)
        self.current_stage_number = 0
        self.stages: List[Stage] = []
        self.current_stage: Optional[Stage] = None
        self.journals: Dict[str, Journal] = {}
        self.stage_history: List[StageTransition] = []
        self.completed_stages: List[str] = []

        # Create initial stage
        self._create_initial_stage()
    
    def get_simple_innovation_list(self) -> List[Dict[str, str]]:
        """Return a simple innovation list (only name and description)."""
        return [{"name": inn["name"], "description": inn["description"]} for inn in self.successful_proposals]
    
    def _get_max_iterations(self, stage_number: int) -> int:
        """Get max iterations for a stage from config or default"""
        return getattr(
            self.cfg.experiment.stages,
            f"stage{stage_number}_max_iters",
            self.cfg.experiment.steps,
        )

    def _extract_exp_name(self):
        """Extract the unique experiment name from task_desc or workspace_dir."""
        # Method 1: read directly from task_desc (preferred)
        if "exp_name" in self.task_desc:
            exp_name = self.task_desc["exp_name"]
            logger.info(f"📝 Extracted exp_name from task_desc: {exp_name}")
            return exp_name
        
        # Method 2: extract from the workspace_dir path (fallback)
        if self.workspace_dir:
            # Match pattern: experiments/YYYY-MM-DD_HH-MM-SS_idea_name_attempt_N
            path_str = str(self.workspace_dir)
            match = re.search(r'experiments[/\\](.+?)(?:[/\\]|$)', path_str)
            if match:
                exp_name = match.group(1)
                logger.info(f"📁 Extracted exp_name from workspace path: {exp_name}")
                return exp_name
        
        # Method 3: use the default value
        exp_name = "default_experiment"
        logger.warning(f"⚠️ Could not extract exp_name, using default: {exp_name}")
        return exp_name

    def _generate_default_baseline(self, dataset_info: dict) -> dict:
        """Dynamically generate a default baseline configuration based on dataset info."""
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
        """Return baseline info; auto-generate from dataset if JSON does not contain it."""
        if task_dict is None:
            task_dict = self.task_desc

        # If the JSON has a baseline field, use it directly
        if "baseline" in task_dict and task_dict["baseline"]:
            return task_dict["baseline"]

        # Otherwise, auto-generate from dataset info
        dataset_info = task_dict.get("dataset", {})
        return self._generate_default_baseline(dataset_info)

    def _get_task_desc_str(self):
        # New structure: focus on dataset and baseline implementation
        dataset_info = self.task_desc["dataset"]
        # Use the unified method to fetch baseline (supports auto-generation)
        baseline_info = self._get_baseline_info()

        task_desc = f"""You are an AI researcher working on {dataset_info['task_type']} tasks.
Your current task is to implement and experiment with different approaches for {dataset_info['name']}.

Dataset Information:
- Name: {dataset_info['name']}
- Description: {dataset_info['description']}
- Dataset ID: {dataset_info['dataset_id']}
- Configuration: {dataset_info['configuration']}
- Modality: {dataset_info['modality']}
- Target: {dataset_info['target_structure']}
- Patch Size (Network Input Shape): {dataset_info.get('patch_size', 'Not specified')}

Baseline Implementation:
- Name: {baseline_info['name']}
- Description: {baseline_info['description']}
- Requirements: {'; '.join(baseline_info['requirements'])}

"""
        # Code will be loaded from Python file and added via loaded_code field
        if "loaded_code" in self.task_desc:
            task_desc += f"Code Template:\n{self.task_desc['loaded_code']}\n"

        return task_desc

    # ============== Precomputed Baseline Loading ==============
    
    def _get_baseline_results_path(self) -> Optional[Path]:
        """Return the baseline results path from environment variables."""
        baseline_path = os.environ.get("camylanet_results")
        if baseline_path and Path(baseline_path).exists():
            return Path(baseline_path)
        return None
    
    def _extract_model_name(self, trainer_name: str) -> str:
        """Extract the model name from a trainer name (stripping the ``Trainer`` suffix).

        Examples:
            nnUNetTrainer -> nnUNet
            SegResNetTrainer -> SegResNet
            UNETRTrainer -> UNETR
        """
        if trainer_name.endswith("Trainer"):
            return trainer_name[:-7]  # strip "Trainer" (7 chars)
        return trainer_name
    
    def _scan_baseline_results_dir(self, base_path: Path, dataset_id: str) -> List[Dict[str, Any]]:
        """Scan the baseline results directory and find matching dataset results.

        Directory structure:
        - base_path/Dataset{ID}_{Name}*/{ID}_{Trainer}/experiment_data.npy
        - or base_path/{ID}_{Trainer}/experiment_data.npy

        Args:
            base_path: Baseline results root directory.
            dataset_id: Dataset ID (e.g. "909").

        Returns:
            List of dicts with keys: trainer, model_name, exp_data_path, result_folder.
        """
        found_baselines = []

        # Pattern 1: subdirectories under base_path/Dataset{ID}_*/
        for dataset_dir in base_path.iterdir():
            if not dataset_dir.is_dir():
                continue

            # Check whether the directory name matches Dataset{ID}
            dir_name = dataset_dir.name
            if dir_name.startswith(f"Dataset{dataset_id}") or dir_name.startswith(f"dataset{dataset_id}"):
                # Scan the trainer subdirectories
                for trainer_dir in dataset_dir.iterdir():
                    if not trainer_dir.is_dir():
                        continue

                    # Check whether the directory name matches {ID}_{Trainer}
                    trainer_name = trainer_dir.name
                    if trainer_name.startswith(f"{dataset_id}_"):
                        trainer = trainer_name[len(f"{dataset_id}_"):]
                        exp_data_path = trainer_dir / "experiment_data.npy"
                        
                        if exp_data_path.exists():
                            found_baselines.append({
                                "trainer": trainer,
                                "model_name": self._extract_model_name(trainer),
                                "exp_data_path": exp_data_path,
                                "result_folder": trainer_dir / "model_results"
                            })
                            logger.info(f"📁 Found baseline: {trainer} at {trainer_dir}")
        
        # Pattern 2: {ID}_{Trainer} directories directly under base_path
        for trainer_dir in base_path.iterdir():
            if not trainer_dir.is_dir():
                continue
            
            trainer_name = trainer_dir.name
            if trainer_name.startswith(f"{dataset_id}_"):
                trainer = trainer_name[len(f"{dataset_id}_"):]
                exp_data_path = trainer_dir / "experiment_data.npy"
                
                # Avoid duplicate entries
                if exp_data_path.exists() and not any(
                    b["exp_data_path"] == exp_data_path for b in found_baselines
                ):
                    found_baselines.append({
                        "trainer": trainer,
                        "model_name": self._extract_model_name(trainer),
                        "exp_data_path": exp_data_path,
                        "result_folder": trainer_dir / "model_results"
                    })
                    logger.info(f"📁 Found baseline: {trainer} at {trainer_dir}")
        
        return found_baselines
    
    def _create_baseline_node_from_file(
        self, 
        exp_data_path: Path, 
        model_name: str, 
        trainer_name: str,
        result_folder: Path
    ) -> Optional[Node]:
        """Create a Node from a precomputed experiment_data.npy.

        Args:
            exp_data_path: Path to experiment_data.npy.
            model_name: Model name (e.g. nnUNet, SegResNet).
            trainer_name: Full trainer name.
            result_folder: Model results folder.

        Returns:
            Node object, or None if loading fails.
        """
        try:
            # Check debug baseline mode (use the instance variable to avoid environment-variable cross-talk across parallel runs)
            debug_baseline_mode = getattr(self, '_debug_baseline_mode', False)

            if debug_baseline_mode:
                # Debug mode: use fake metrics (dice=0, hd95=200)
                dice_score = 0.0
                hd95_score = 200.0
                logger.info(f"🐛 DEBUG MODE: Using fake baseline metrics for {model_name}: dice={dice_score}, hd95={hd95_score}")
            else:
                exp_data = np.load(exp_data_path, allow_pickle=True).item()
                
                # Extract metrics
                dice_score = None
                hd95_score = None

                # Handle flat format (contains metrics directly)
                if 'metrics' in exp_data:
                    val_metrics = exp_data.get('metrics', {}).get('val', [])
                    if val_metrics:
                        latest = val_metrics[-1] if isinstance(val_metrics, list) else val_metrics
                        dice_score = latest.get('dice')
                        hd95_score = latest.get('hd95')
                
                # Fallback: extract from dice_scores/hd95_scores
                if dice_score is None and 'dice_scores' in exp_data:
                    scores = exp_data['dice_scores']
                    dice_score = scores[-1] if scores else None
                if hd95_score is None and 'hd95_scores' in exp_data:
                    scores = exp_data['hd95_scores']
                    hd95_score = scores[-1] if scores else None
                
                if dice_score is None or hd95_score is None:
                    logger.warning(f"Could not extract metrics from {exp_data_path}")
                    return None
            
            # Build the MetricValue
            metric = MetricValue(
                value={
                    "metric_names": [
                        {
                            "metric_name": "Dice Score",
                            "lower_is_better": False,
                            "description": f"Dice from {model_name} baseline",
                            "data": [{
                                "dataset_name": "baseline",
                                "final_value": float(dice_score),
                                "best_value": float(dice_score)
                            }]
                        },
                        {
                            "metric_name": "HD95 Score",
                            "lower_is_better": True,
                            "description": f"HD95 from {model_name} baseline",
                            "data": [{
                                "dataset_name": "baseline",
                                "final_value": float(hd95_score),
                                "best_value": float(hd95_score)
                            }]
                        }
                    ]
                }
            )
            
            # Create the Node
            debug_tag = "[DEBUG] " if debug_baseline_mode else ""
            node = Node(
                plan=f"{debug_tag}Precomputed baseline: {model_name}",
                code="",  # Preloaded baselines have no code
                metric=metric,
                is_buggy=False,
                analysis=f"{debug_tag}Loaded precomputed {model_name} baseline: Dice={dice_score:.4f}, HD95={hd95_score:.4f}",
                exp_results_dir=str(result_folder) if result_folder.exists() else None,
                origin_stage="1_baseline_implementation_1_preliminary"
            )
            
            # Add baseline-specific attributes
            node.baseline_model = model_name
            node.is_precomputed_baseline = True
            
            # Read num_epochs (used to dynamically compute exec.timeout; not exposed to the agent)
            try:
                debug_json_path = result_folder / "debug.json"
                if debug_json_path.exists():
                    with open(debug_json_path, "r") as f:
                        debug_data = json.load(f)
                    node._num_epochs = int(debug_data.get("num_epochs", 0)) or None
                else:
                    node._num_epochs = None
            except Exception as e:
                logger.warning(f"Failed to read num_epochs from debug.json: {e}")
                node._num_epochs = None
            
            logger.info(f"✅ Created baseline node: {model_name} (Dice={dice_score:.4f}, HD95={hd95_score:.4f}, epochs={node._num_epochs}){' [DEBUG MODE]' if debug_baseline_mode else ''}")
            return node
            
        except Exception as e:
            logger.error(f"Error loading baseline from {exp_data_path}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _load_precomputed_baselines(self) -> List[Node]:
        """Load all precomputed baseline model results.

        Returns:
            List of Node objects representing precomputed baselines.
        """
        base_path = self._get_baseline_results_path()
        if not base_path:
            logger.info("No camylanet_results environment variable set or path does not exist")
            return []
        
        dataset_id = self.task_desc.get("dataset", {}).get("dataset_id")
        if not dataset_id:
            logger.warning("No dataset_id found in task_desc, cannot load precomputed baselines")
            return []
        
        logger.info(f"🔍 Scanning for precomputed baselines in {base_path} for dataset {dataset_id}")
        
        # Scan the directory
        found_baselines = self._scan_baseline_results_dir(base_path, str(dataset_id))
        
        if not found_baselines:
            logger.info(f"No precomputed baselines found for dataset {dataset_id}")
            return []
        
        # Build Node objects
        baseline_nodes = []
        for baseline_info in found_baselines:
            node = self._create_baseline_node_from_file(
                baseline_info["exp_data_path"],
                baseline_info["model_name"],
                baseline_info["trainer"],
                baseline_info["result_folder"]
            )
            if node:
                baseline_nodes.append(node)
        
        logger.info(f"✅ Loaded {len(baseline_nodes)} precomputed baselines")
        return baseline_nodes
    
    def _save_all_baselines_json(self, baseline_nodes: List[Node]):
        """Save all baseline info to a JSON file for use by the writing module.

        Args:
            baseline_nodes: List of baseline Node objects.
        """
        try:
            # 🔧 After directory merge: workspace_dir == log_dir, use it directly
            logs_dir = Path(self.cfg.log_dir)
            logs_dir.mkdir(parents=True, exist_ok=True)
            
            baselines_data = {
                "timestamp": datetime.now().isoformat(),
                "dataset_id": self.task_desc.get("dataset", {}).get("dataset_id"),
                "dataset_name": self.task_desc.get("dataset", {}).get("name"),
                "num_baselines": len(baseline_nodes),
                "baselines": []
            }
            
            for node in baseline_nodes:
                baseline_info = {
                    "model_name": getattr(node, 'baseline_model', 'Unknown'),
                    "node_id": node.id,
                    "dice_score": None,
                    "hd95_score": None,
                    "exp_results_dir": node.exp_results_dir
                }
                
                # Extract metric values
                if node.metric and hasattr(node.metric, 'value') and isinstance(node.metric.value, dict):
                    metric_names = node.metric.value.get("metric_names", [])
                    for m in metric_names:
                        if m.get("metric_name") == "Dice Score" and m.get("data"):
                            baseline_info["dice_score"] = m["data"][0].get("final_value")
                        elif m.get("metric_name") == "HD95 Score" and m.get("data"):
                            baseline_info["hd95_score"] = m["data"][0].get("final_value")
                
                baselines_data["baselines"].append(baseline_info)
            
            # Sort by Dice score (descending)
            baselines_data["baselines"].sort(
                key=lambda x: x.get("dice_score", 0) or 0,
                reverse=True
            )
            
            # Flag the best baseline
            if baselines_data["baselines"]:
                baselines_data["best_baseline"] = baselines_data["baselines"][0]["model_name"]
            
            # Save the file
            output_path = logs_dir / "all_baselines.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(baselines_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📁 Saved all baselines info to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save all_baselines.json: {e}")
    
    def _recalculate_dynamic_timeout(self):
        """Dynamically compute exec.timeout based on num_epochs.

        Formula: timeout = max(base, num_epochs / 100 * base),
        where base = cfg.exec.timeout (the per-100-epoch base time configured in YAML).

        num_epochs resolution order:
          1. _num_epochs attribute / debug.json of a precomputed baseline node.
          2. debug.json of any Stage 1 node (from-scratch training scenario).
          3. The default_num_epochs environment variable (specified by the user at launch).

        This method is invoked at:
          - _create_initial_stage() (after precomputed baselines are loaded).
          - Before Stage 2 begins after Stage 1 finishes (normal training flow).
          - After checkpoint restoration.
        """
        from omegaconf import OmegaConf

        base_time = self.cfg.exec.timeout

        # Locate the Stage 1 journal
        stage1_journal = None
        if hasattr(self, "journals"):
            for name, journal in self.journals.items():
                if name.startswith("1_"):
                    stage1_journal = journal
                    break

        num_epochs = None
        source = None

        # --- Strategy 1: obtain from precomputed baseline nodes ---
        if stage1_journal is not None:
            best_precomputed = None
            best_dice = -1.0
            for node in stage1_journal.nodes:
                if not getattr(node, "is_precomputed_baseline", False):
                    continue
                if node.metric is None:
                    continue
                dice = node.metric.get_mean_value()
                if dice > best_dice:
                    best_dice = dice
                    best_precomputed = node

            if best_precomputed is not None:
                num_epochs = getattr(best_precomputed, "_num_epochs", None)
                if num_epochs:
                    source = "precomputed baseline node attribute"
                else:
                    num_epochs = self._read_num_epochs_from_node(best_precomputed)
                    if num_epochs:
                        source = "precomputed baseline debug.json"

        # --- Strategy 2: obtain from debug.json of any Stage 1 best node ---
        if num_epochs is None and stage1_journal is not None:
            best_node = stage1_journal.get_best_node()
            if best_node is not None:
                num_epochs = self._read_num_epochs_from_node(best_node)
                if num_epochs:
                    source = f"Stage 1 best node ({best_node.id[:8]}) debug.json"

        # --- Strategy 3: obtain from the default_num_epochs environment variable ---
        if num_epochs is None:
            env_epochs = os.environ.get("default_num_epochs")
            if env_epochs:
                try:
                    num_epochs = int(env_epochs)
                    source = "environment variable default_num_epochs"
                except ValueError:
                    logger.warning(f"Invalid default_num_epochs env var: {env_epochs!r}")

        if num_epochs is None or num_epochs <= 0:
            logger.debug("Could not determine num_epochs from any source, skipping dynamic timeout")
            return

        new_timeout = max(base_time, int(num_epochs / 100 * base_time))
        if new_timeout == base_time:
            logger.info(f"Dynamic timeout: {num_epochs} epochs, base={base_time}s (no change, source: {source})")
            return

        OmegaConf.update(self.cfg, "exec.timeout", new_timeout)
        logger.info(
            f"Dynamic timeout: {num_epochs} epochs, base={base_time}s -> "
            f"{new_timeout}s ({new_timeout / 3600:.1f}h) [source: {source}]"
        )

    def _read_num_epochs_from_node(self, node) -> Optional[int]:
        """Read num_epochs from the node's exp_results_dir/debug.json."""
        exp_results_dir = getattr(node, "exp_results_dir", None)
        if not exp_results_dir:
            return None
        try:
            debug_json_path = Path(exp_results_dir) / "debug.json"
            if debug_json_path.exists():
                with open(debug_json_path, "r") as f:
                    debug_data = json.load(f)
                val = int(debug_data.get("num_epochs", 0))
                return val if val > 0 else None
        except Exception as e:
            logger.warning(f"Failed to read num_epochs from {exp_results_dir}/debug.json: {e}")
        return None

    # ============== End Precomputed Baseline Loading ==============

    def _create_initial_stage(self):
        """Create the initial stage configuration.

        Supports two modes:
        1. Preload mode: load precomputed baseline results from the camylanet_results environment variable.
        2. Normal mode: create an empty Stage 1 awaiting experiment execution.
        """
        # Try loading precomputed baselines
        baseline_nodes = self._load_precomputed_baselines()

        self.current_stage_number += 1

        if baseline_nodes:
            # Preload mode: Stage 1 uses the precomputed results directly
            logger.info(f"🚀 Precomputed baseline mode: Loading {len(baseline_nodes)} baselines")

            initial_stage = Stage(
                name="1_baseline_implementation_1_preliminary",
                description="Precomputed baselines loaded",
                goals=["Load precomputed baseline results for comparison"],
                max_iterations=1,  # No iteration needed
                num_drafts=0,
                stage_number=self.current_stage_number,
            )
            
            self.stages.append(initial_stage)
            self.current_stage = initial_stage
            
            # Create the Journal and add all baseline nodes
            j0 = Journal()
            j0.stage_name = initial_stage.name
            for node in baseline_nodes:
                j0.append(node)
            
            self.journals[initial_stage.name] = j0
            
            # Save baseline info for writing
            self._save_all_baselines_json(baseline_nodes)
            
            # Dynamically compute exec.timeout from the best baseline's num_epochs
            self._recalculate_dynamic_timeout()
            
            # Set the preloaded flag; run() checks it to skip Stage 1 execution
            self._precomputed_baseline_loaded = True
            
            logger.info("Loaded baselines:")
            for node in baseline_nodes:
                model_name = getattr(node, 'baseline_model', 'Unknown')
                metric_val = node.metric.get_mean_value() if node.metric else 'N/A'
                logger.info(f"   - {model_name}: Dice={metric_val:.4f}" if isinstance(metric_val, float) else f"   - {model_name}: {metric_val}")
        else:
            # Normal mode: create an empty Stage 1
            initial_stage = Stage(
                name="1_baseline_implementation_1_preliminary",
                description="preliminary",
                goals=MAIN_STAGE_GOALS[1],
                max_iterations=self._get_max_iterations(self.current_stage_number),
                num_drafts=self.cfg.experiment.search.num_drafts,
                stage_number=self.current_stage_number,
            )

            self.stages.append(initial_stage)
            self.current_stage = initial_stage
            j0 = Journal()
            j0.stage_name = initial_stage.name
            self.journals[initial_stage.name] = j0
            
            self._precomputed_baseline_loaded = False



    def _save_checkpoint(self, stage: Optional[Stage] = None):
        """Save the current state of the experiment by pickling the entire instance.

        Args:
            stage: The stage object whose name should be used for the checkpoint directory. If None, fall back to self.current_stage.
        """
        stage_to_use = stage or self.current_stage
        if stage_to_use is None:
            logger.warning("Cannot save checkpoint: stage is None")
            return
        stage_name = "stage_" + stage_to_use.name
        # Determine the save path
        # 🔧 After directory merge: workspace_dir == log_dir, use directly
        logs_dir = Path(self.cfg.log_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        stage_dir = logs_dir / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)

        save_path = stage_dir / "checkpoint.pkl"
        tmp_path = stage_dir / "checkpoint.pkl.tmp"
        bak_path = stage_dir / "checkpoint.pkl.bak"

        logger.info(f"Saving checkpoint to {save_path}")
        try:
            with open(tmp_path, "wb") as f:
                pickle.dump(self, f)
                f.flush()
                os.fsync(f.fileno())

            if save_path.exists():
                shutil.copy2(save_path, bak_path)

            os.replace(tmp_path, save_path)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    def _save_stage_summary_markdown(self, stage: Stage, journal: Journal):
        """Generate and save a natural-language substage summary as an MD file.
        
        Args:
            stage: the current stage object.
            journal: experiment journal for the current stage.
        """
        try:
            # Reuse the existing generate_summary method for AI analysis
            summary_text = journal.generate_summary(include_code=False)
            
            # Build the save path (consistent with _save_checkpoint)
            # 🔧 After directory merge: use cfg.log_dir directly
            logs_dir = Path(self.cfg.log_dir)
            stage_dir = logs_dir / f"stage_{stage.name}"
            stage_dir.mkdir(parents=True, exist_ok=True)
            
            # Fetch best node info
            best_node = journal.get_best_node()
            if best_node and best_node.metric:
                best_metric = f"{best_node.metric.get_mean_value():.4f}"
            else:
                best_metric = "N/A"
            
            # Format stage objective
            if isinstance(stage.goals, list):
                goals_text = "\n".join(f"- {g}" for g in stage.goals)
            else:
                goals_text = stage.goals
            
            # Build the Markdown content
            md_content = f"""# Stage summary: {stage.name}

## Overview
- **Stage description**: {stage.description}
- **Total nodes**: {len(journal.nodes)}
- **Successful nodes**: {len(journal.good_nodes)}
- **Buggy nodes**: {len(journal.buggy_nodes)}
- **Best metric**: {best_metric}

## Stage objective
{goals_text}

## AI analysis summary
{summary_text}
"""
            
            # Save the file
            summary_path = stage_dir / "summary.md"
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(md_content)
            
            logger.info(f"📝 Stage summary saved: {summary_path}")
            
            # # 🆕 Step 1: Analyze OpenHands logs and summarize each interaction
            # self._analyze_openhands_logs(stage)
            #
            # # 🆕 Step 2: Merge journal and OpenHands summaries into a complete stage report
            # self._generate_complete_stage_report(stage)
            
        except Exception as e:
            logger.warning(f"Failed to generate stage summary: {e}")

    def _analyze_openhands_logs(self, stage: Stage):
        """Analyze OpenHands logs and generate per-interaction summaries (Step 1-4)."""
        try:
            from .openhands_log_analyzer import OpenHandsLogAnalyzer
            
            # Locate the stage's openhands_logs directory (sibling of openhands_workspace)
            # 🔧 After directory merge: use cfg.log_dir directly
            logs_dir = Path(self.cfg.log_dir)
            stage_logs_dir = logs_dir / f"stage_{stage.name}"
            openhands_logs_dir = stage_logs_dir / "openhands_logs"
            
            if not openhands_logs_dir.exists():
                logger.info(f"No OpenHands logs found for stage {stage.name}")
                return
            
            # Create the analyzer and summarize each interaction
            analyzer = OpenHandsLogAnalyzer(self.cfg)
            analyzer.analyze_openhands_interactions(openhands_logs_dir)
            
            logger.info(f"📊 OpenHands interaction summaries complete")
        except Exception as e:
            logger.warning(f"Failed to analyze OpenHands logs: {e}")

    def _generate_complete_stage_report(self, stage: Stage):
        """Generate the complete stage report (Step 5)."""
        try:
            # Skip log analysis for precomputed-baseline stages
            if getattr(self, '_precomputed_baseline_loaded', False):
                logger.info(f"📊 Skipping complete stage report for precomputed baseline stage: {stage.name}")
                return
            
            # Check whether every node in the journal is a precomputed baseline
            journal = self.journals.get(stage.name)
            if journal and all(getattr(node, 'is_precomputed_baseline', False) for node in journal.nodes):
                logger.info(f"📊 Skipping complete stage report (all nodes are precomputed baselines): {stage.name}")
                return
            
            from .openhands_log_analyzer import OpenHandsLogAnalyzer
            
            # Get the stage log directory
            # 🔧 After directory merge: use cfg.log_dir directly
            logs_dir = Path(self.cfg.log_dir)
            stage_logs_dir = logs_dir / f"stage_{stage.name}"
            
            # Check that journal.json exists (avoids noisy error logs)
            journal_path = stage_logs_dir / "journal.json"
            if not journal_path.exists():
                logger.info(f"📊 Skipping complete stage report (no journal.json found): {stage.name}")
                return
            
            # Create the analyzer and generate the full report
            analyzer = OpenHandsLogAnalyzer(self.cfg)
            report_path = analyzer.generate_stage_summary_report(
                stage_logs_dir=stage_logs_dir,
                stage_name=stage.name
            )
            
            if report_path:
                logger.info(f"📊 Full stage report generated: {report_path}")
        except Exception as e:
            logger.warning(f"Failed to generate full stage report: {e}")

    def _create_agent_for_stage(self, stage: Stage) -> ParallelAgent:
        """Create a ParallelAgent configured for the given stage"""
        logger.debug(f"_create_agent_for_stage called with stage: {stage.name}")
        logger.debug(f"self.current_stage: {self.current_stage.name if self.current_stage else 'None'}")
        logger.debug(f"Are they the same object? {stage is self.current_stage}")

        stage_cfg = self.cfg.copy()
        stage_cfg.experiment.search.num_drafts = stage.num_drafts

        # Parse stage information for both structures
        (
            main_stage,
            main_stage_name,
            sub_stage_num,
            sub_stage_name,
        ) = self.parse_stage_names(stage.name)

        # Pass the original task_desc JSON object for structure detection
        task_desc = self.task_desc.copy()
        # Add stage-specific information as metadata
        task_desc["_stage_info"] = {
            "main_stage_name": main_stage_name,
            "sub_stage_num": sub_stage_num,
            "sub_stage_name": sub_stage_name,
            "goals": stage.goals
        }

        logger.debug("Checking task_desc inside _create_agent_for_stage")
        # print(task_desc)

        if main_stage == 2:
            # Stage 2: Creative Research - use Stage 1 baseline for single innovations
            stage1_substages = [s for s in self.stages if s.name.startswith("1_")]
            if stage1_substages:
                best_stage1_node = self._get_best_implementation(stage1_substages[-1].name)
                logger.debug(f"Stage 2: Found Stage 1 baseline node: {best_stage1_node.id if best_stage1_node else 'None'}")
            else:
                best_stage1_node = None
                logger.debug("Stage 2: No Stage 1 substages found")

            # Stage 2 does not require global best-node tracking
            best_stage2_node = None
        elif main_stage == 3:
            # Stage 3: Ablation Studies - use best from Stage 2
            best_stage2_node, beats_baseline = self._get_best_stage2_proposal()
            if best_stage2_node:
                logger.debug(f"Stage 3: Using best Stage 2 proposal (beats baseline: {beats_baseline})")
                logger.info(f"Stage 3: Using best Stage 2 proposal for ablation studies")
            else:
                raise ValueError("No valid Stage 2 proposals available for Stage 3")
            best_stage1_node = None
        else:
            best_stage2_node = None
            best_stage1_node = None

        return ParallelAgent(
            task_desc=task_desc,
            cfg=stage_cfg,
            journal=self.journals[stage.name],
            stage_name=stage.name,
            stage=stage,  # 🆕 pass the Stage object so proposal_idx is accessible
            best_stage2_node=best_stage2_node,
            best_stage1_node=best_stage1_node,
            agent_manager=self,  # Pass self as agent_manager for innovation queue access
        )


    def _check_substage_completion(
        self, current_substage: Stage, journal: Journal
    ) -> bool:
        """Check if the current sub-stage is complete"""
        main_stage_num, _, _, _ = self.parse_stage_names(current_substage.name)

        # Stage 2: count only actual experiment steps (exclude seeded baseline)
        # The seeded baseline node has parent=None; real experiment nodes always have a parent.
        if main_stage_num == 2:
            actual_steps = sum(1 for n in journal.nodes if n.parent is not None)
            if actual_steps >= current_substage.max_iterations:
                logger.info(
                    f"Stage {current_substage.name} completed: reached max iterations "
                    f"({actual_steps} actual steps)"
                )
                return True, "Reached max iterations"
            return False, "Using full budget for thorough exploration"

        # Stage 3: plan-driven ablation -- early exit when plan is done
        if main_stage_num == 3:
            if getattr(journal, 'ablation_plan_complete', False):
                logger.info(f"Stage {current_substage.name} completed: ablation plan finished")
                return True, "All ablation plan items completed"
            if len(journal.nodes) >= current_substage.max_iterations:
                logger.info(f"Stage {current_substage.name} completed: reached max iterations (hard cap)")
                return True, "Reached max iterations"
            return False, "Ablation plan still in progress"

        # Generic max-iterations check for non-Stage-2/3 substages
        if len(journal.nodes) >= current_substage.max_iterations:
            logger.info(
                f"Stage {current_substage.name} completed: reached max iterations"
            )
            return True, "Reached max iterations"

        # For stage 1, check if we have at least one working implementation
        if main_stage_num == 1:
            if len(journal.good_nodes) > 0:
                logger.info(
                    f"Stage {current_substage.name} completed: found working implementation"
                )
                return True, "Found working implementation"
            else:
                return False, "No working implementation found yet"

        return False, "Using full budget for thorough exploration"

    def _check_stage_completion(self, stage: Stage) -> bool:
        """Check if current stage is complete based on criteria"""
        # Skip main-stage completion logic during creative_research; sub-stages control exit.
        main_stage_num, _, _, _ = self.parse_stage_names(stage.name)
        # For creative_research (stage 2), we bypass main-stage completion only when
        # the innovation-queue workflow is enabled. Otherwise, fall back to default logic.
        if main_stage_num == 2 and self.use_innovation_queue:
            return False, ""
        journal = self.journals[stage.name]
        # Terminate if max iterations reached
        if len(journal.nodes) >= stage.max_iterations:
            logger.info(f"Stage {stage.name} completed: reached max iterations")
            if stage.stage_number == 1:
                # For initial stage, if it didn't even find a working implementation until max iterations,
                # end gracefully and stop the experiment.
                logger.error(
                    f"Initial stage {stage.name} did not find a working implementation after {stage.max_iterations} iterations. Consider increasing the max iterations or reducing the complexity of the research idea."
                )
                
                self.current_stage = None  # This will cause the run loop to exit
                return True, "Failed to find working implementation"
            else:
                return True, "Reached max iterations"

        # For initial stage, complete when we have at least one working implementation
        if stage.stage_number == 1:
            if len(journal.good_nodes) > 0:
                logger.info(
                    f"Stage {stage.name} completed: found working implementation"
                )
                return True, "Found working implementation"

        # For stages 2, 3, 4: just let the agent run until max iterations is reached
            # These stages prioritize using the full budget (max_iterations)
            # Stage 2: Creative research - use all budget to explore innovations
            # Stage 3: Ablation studies - use all budget for comprehensive testing
            pass

        logger.debug(f"Stage {stage.name} not completed")
        return False, "stage not completed"

    def _get_best_implementation(self, stage_name: str) -> Optional[Node]:
        """Get the best implementation from a completed stage"""
        if stage_name not in self.journals:
            return None
        best_node = self.journals[stage_name].get_best_node()
        if best_node:
            # Create a clean copy of the node for the next stage
            copied_node = copy.deepcopy(best_node)
            # Reset parent relationship and children
            copied_node.parent = None
            copied_node.children = set()
            return copied_node
        return None

    def _get_best_stage2_proposal(self, exclude_stage: Optional[str] = None) -> Tuple[Optional[Node], bool]:
        """
        Get the best proposal from all Stage 2 substages, with baseline comparison.
        
        Uses MetricValue comparison (Dice primary, HD95 direct tiebreak when
        Dice is within threshold) for both cross-substage selection and
        beats_baseline check, consistent with journal.get_best_node().

        Args:
            exclude_stage: If provided, skip this substage name when scanning.
                Used on checkpoint resume to avoid counting the current
                in-progress substage as an already-completed winner.
        
        Returns:
            Tuple[Optional[Node], bool]: (best_node, beats_baseline)
            - best_node: The best performing node from Stage 2, or None if no valid nodes
            - beats_baseline: True if best_node's MetricValue exceeds Stage 1 baseline
        """
        stage2_substages = [s for s in self.stages if s.name.startswith("2_")]
        stage1_substages = [s for s in self.stages if s.name.startswith("1_")]
        
        baseline_node = None
        if stage1_substages:
            baseline_node = self._get_best_implementation(stage1_substages[-1].name)
            if baseline_node and baseline_node.metric:
                logger.info(f"📊 Stage 1 baseline metric: {baseline_node.metric}")
        
        best_node = None
        best_stage_name = None
        
        for substage in stage2_substages:
            stage_name = substage.name
            if stage_name == exclude_stage:
                continue
            if stage_name not in self.journals:
                continue
            
            current_best = self.journals[stage_name].get_best_node()
            if current_best and current_best.metric:
                if best_node is None or current_best.metric > best_node.metric:
                    best_node = current_best
                    best_stage_name = stage_name
        
        if best_node:
            beats_baseline = (baseline_node is not None and
                              baseline_node.metric is not None and
                              best_node.metric > baseline_node.metric)
            
            best_dice = best_node.metric.get_mean_value()
            baseline_str = f"{baseline_node.metric.get_mean_value():.4f}" if baseline_node and baseline_node.metric else "N/A"
            logger.info(f"🏆 Selected best Stage 2 proposal from {best_stage_name} with metric: {best_node.metric}")
            logger.info(f"   Baseline comparison: Dice={best_dice:.4f} vs {baseline_str} -> {'BEATS' if beats_baseline else 'DOES NOT BEAT'} baseline")
            
            copied_node = copy.deepcopy(best_node)
            copied_node.parent = None
            copied_node.children = set()
            return copied_node, beats_baseline
        else:
            logger.warning("No valid nodes found in any Stage 2 substages")
            return None, False
    

    def _generate_substage_goal(self, main_stage_goal: str, journal: Journal) -> str:
        """Generate the next sub-stage goal based on what has been done so far.

        Args:
            main_stage_goal: The overall goal for the current main stage
            journal: Journal containing the results and progress so far

        Returns:
            str: Specific goals for the next sub-stage
        """
        # Gather current progress metrics
        metrics = self._gather_stage_metrics(journal)
        issues = self._identify_issues(journal)
        progress = self._analyze_progress(journal)

        # Create prompt for the LLM
        prompt = f"""
        Based on the current experimental progress, generate focused goals for the next sub-stage.

        Main Stage Goals:
        {main_stage_goal}

        Current Progress:
        - Total attempts: {metrics['total_nodes']}
        - Successful implementations: {metrics['good_nodes']}
        - Best performance: {metrics['best_metric']['value'] if metrics['best_metric'] else 'N/A'}
        - Convergence status: {progress['convergence_status']}

        Current Issues:
        {json.dumps(issues, indent=2)}

        Recent Changes:
        {json.dumps(progress['recent_changes'], indent=2)}

        Generate specific, actionable sub-stage goals that:
        1. Address current issues and limitations
        2. Build on recent progress
        3. Move towards main stage goals
        4. Are concrete and measurable
        """

        # Define the function specification for the LLM
        substage_goal_spec = FunctionSpec(
            name="generate_substage_goals",
            description="Generate specific goals for the next experimental sub-stage",
            json_schema={
                "type": "object",
                "properties": {
                    "goals": {
                        "type": "string",
                        "description": "Detailed, specific goals for the next sub-stage",
                    },
                    "sub_stage_name": {
                        "type": "string",
                        "description": "The name of the next sub-stage",
                    },
                },
                "required": ["goals", "sub_stage_name"],
            },
        )

        try:
            # Get response from LLM
            response = query(
                system_message=None,
                user_message=prompt,
                func_spec=substage_goal_spec,
                **{"model": get_role("feedback")["model"], "temperature": get_role("feedback").get("temperature", 0.9)},
            )

            # Format the response into a structured goal string
            goal_str = f"""
            {response['goals']}
            """

            return goal_str.strip(), response["sub_stage_name"]

        except Exception as e:
            logger.error(f"Error generating sub-stage goals: {e}")
            # Provide fallback goals if LLM fails
            return f"""
            Sub-stage Goals:
            Continue progress on main stage objectives while addressing current issues.
            """

    def _create_next_substage(
        self, current_substage: Stage, journal: Journal, substage_feedback: str
    ) -> Optional[Stage]:
        """Create the next sub-stage. Ask LLM to come up with the next sub-stage name and goals
        based on what has been done so far.
        """
        main_stage_num, main_stage_name, sub_stage_num, _ = self.parse_stage_names(current_substage.name)

        # Stage 2: Creative Research - reuses the original innovation-queue logic
        if main_stage_num == 2 and self.use_innovation_queue:
            # 🔥 Defensive check: ensure the proposals list has enough elements
            if not self.proposals or sub_stage_num - 1 >= len(self.proposals):
                logger.warning(f"⚠️ No proposal available for sub_stage_num={sub_stage_num}, proposals count={len(self.proposals)}. Ending Stage 2.")
                return None
            
            # Enforce global budget for creative_research sub-stages
            if self.executed_substages_count >= self._get_max_iterations(main_stage_num):
                max_iters = self._get_max_iterations(main_stage_num)
                logger.info(f"💡 Reached max_iterations ({max_iters}) for Stage {main_stage_num}. No more substages will be created.")
                return None
            self.executed_substages_count += 1
            # evaluate innovation result
            best = journal.get_best_node()
            metric = best.metric.get_mean_value() if best and best.metric else None
            # baseline from last baseline_implementation stage
            stage1_names = [n for n in self.journals.keys() if n.startswith("1_")]
            baseline_node = self._get_best_implementation(stage1_names[-1]) if stage1_names else None
            baseline_metric = (
                baseline_node.metric.get_mean_value()
                if baseline_node and baseline_node.metric
                else None
            )

            # Use baseline as the reference metric
            reference_metric = baseline_metric

            proposal = self.proposals[sub_stage_num-1]
            
            # 🔥 Detect failure via simple node-ID comparison
            is_innovation_failed = (best and baseline_node and best.id == baseline_node.id)
            
            if not is_innovation_failed and best:
                # 🔥 Record innovation info using the unified full structure
                improvement = metric - reference_metric if reference_metric else 0
                
                # 🆕 Extract detailed per-metric improvement info
                detailed_metrics = self._extract_detailed_metric_improvements(best, baseline_node)
                
                proposal_info = {
                    "title": proposal.get('title', 'Unknown'),
                    "md_file": proposal.get('md_file', ''),  # MD file path
                    "node": copy.deepcopy(best),
                    "metric_improvement": improvement,
                    "detailed_metrics": detailed_metrics,
                    "substage_name": current_substage.name
                }
                self.successful_proposals.append(proposal_info)
                
                logger.info(f"✅ Stage 2 proposal succeeded: {proposal.get('title', 'Unknown')} - generated new solution (node: {best.id})")
                logger.info(f"📊 Recorded for Stage 3: improvement={improvement:.4f}, total_successful={len(self.successful_proposals)}")
                if detailed_metrics:
                    logger.info(f"📊 Detailed metrics: {detailed_metrics}")
            else:
                self.failed_proposals.append(proposal)
                logger.info(f"🔴 Stage 2 proposal failed: {proposal.get('title', 'Unknown')} - best solution is still baseline (node: {best.id if best else 'None'} == baseline: {baseline_node.id if baseline_node else 'None'})")
                
                # 🔥 Proposal refinement: refine the proposal based on failure info (remove/modify modules)
                if self.proposal_queue_manager:
                    try:
                        # Collect failure info
                        failure_info = {
                            "error_type": "performance_below_baseline",
                            "error_message": f"Proposal did not exceed baseline metric. Best: {metric}, Baseline: {baseline_metric}",
                            "execution_time": str(best.exec_time_feedback) if best and hasattr(best, 'exec_time_feedback') else "N/A",
                            "performance_issues": f"Metric improvement: {metric - baseline_metric if metric and baseline_metric else 'unknown'}"
                        }
                        
                        # Get the proposal index
                        proposal_idx = sub_stage_num - 1
                        
                        # 🔥 Proposal refinement: refine the proposal based on failure info (remove/modify modules)
                        # Refined proposals are automatically appended to the queue (innovation_integration.py)
                        if self.proposal_queue_manager:
                            try:
                                # Collect failure info
                                failure_info = {
                                    "error_type": "performance_below_baseline",
                                    "error_message": f"Proposal did not exceed baseline metric. Best: {metric}, Baseline: {baseline_metric}",
                                    "execution_time": str(best.exec_time_feedback) if best and hasattr(best, 'exec_time_feedback') else "N/A",
                                    "performance_issues": f"Metric improvement: {metric - baseline_metric if metric and baseline_metric else 'unknown'}"
                                }
                                
                                # Get the proposal index
                                proposal_idx = sub_stage_num - 1
                                
                                # Refine the proposal — it is now appended to the queue automatically
                                refined = self.proposal_queue_manager.refine_proposal_on_failure(
                                    proposal_idx=proposal_idx,
                                    failure_info=failure_info
                                )
                                
                                if refined:
                                    logger.info(f"✅ Proposal refined successfully: {refined.get('title', 'Unknown')}")
                                else:
                                    logger.warning(f"⚠️ Proposal refinement failed; moving on to the next proposal")
                                    
                            except Exception as e:
                                logger.error(f"❌ Proposal refinement failed: {proposal.get('title', 'Unknown')}, error: {e}")
                                logger.warning(f"   Continuing with the existing proposal queue...")
                            
                    except Exception as e:
                        logger.error(f"❌ Proposal refinement failed: {proposal.get('title', 'Unknown')}, error: {e}")
                        logger.warning(f"   Continuing with the existing proposal queue...")

            # Early exit: if any Stage 2 proposal already beats baseline, skip remaining
            _, beats_baseline = self._get_best_stage2_proposal()
            if beats_baseline:
                logger.info(
                    f"💡 Stage 2 early exit: best proposal beats baseline after "
                    f"{sub_stage_num}/{len(self.proposals)} proposals. "
                    f"Skipping remaining proposals."
                )
                return None

            # Determine next proposal index (1-based)
            next_proposal_idx = sub_stage_num + 1

            # If there is another proposal to try and we still have budget, create next sub-stage
            if next_proposal_idx <= len(self.proposals) and self.executed_substages_count < self._get_max_iterations(main_stage_num):
                next_proposal = self.proposals[next_proposal_idx - 1]
                stage2_config = self.cfg.experiment.stage2
                return Stage(
                    name=f"2_creative_research_{next_proposal_idx}_proposal_{next_proposal_idx}",
                    description=next_proposal.get('title', f'Proposal {next_proposal_idx}'),
                    goals=[next_proposal.get('title', '')],
                    max_iterations=stage2_config.max_iterations_per_innovation,
                    num_drafts=0,
                    stage_number=current_substage.stage_number + 1,
                    proposal_idx=next_proposal_idx - 1,  # 0-based index
                )

            # If no more innovations or budget exhausted, end creative_research main stage
            if next_proposal_idx > len(self.proposals):
                logger.info(f"💡 All {len(self.proposals)} proposals have been tried. Ending Stage 2 (Creative Research).")
            else:
                max_iters = self._get_max_iterations(main_stage_num)
                logger.info(f"💡 Budget exhausted: executed {self.executed_substages_count}/{max_iters} substages. Ending Stage 2.")
            return None

        # Default behavior for other stages
        main_stage_goal = MAIN_STAGE_GOALS[main_stage_num]
        sub_stage_goal, sub_stage_name = self._generate_substage_goal(
            main_stage_goal, journal
        )

        return Stage(
            name=f"{main_stage_num}_{main_stage_name}_{sub_stage_num + 1}_{sub_stage_name}",
            description=sub_stage_name,
            goals="Main stage goals:\n"
            + main_stage_goal
            + "\n\nSub-stage goals:\n"
            + sub_stage_goal,
            max_iterations=self._get_max_iterations(main_stage_num),
            num_drafts=0,
            stage_number=current_substage.stage_number + 1,
        )

    def _create_next_main_stage(
        self, current_substage: Stage, journal: Journal
    ) -> Optional[Stage]:
        (
            main_stage_num,
            main_stage_name,
            sub_stage_num,
            sub_stage_name,
        ) = self.parse_stage_names(current_substage.name)
        logger.debug(f"current_substage={current_substage.name}")
        logger.debug(f"parsed main_stage_num={main_stage_num}")
        if main_stage_num == 3:
            # Stage 3 (ablation_studies) is the final stage
            return None
        # Skip directly from Stage 2 to Stage 3, but only if best proposal beats baseline
        if main_stage_num + 1 == 3:
            # Check if best Stage 2 proposal beats baseline
            best_node, beats_baseline = self._get_best_stage2_proposal()
            
            if not beats_baseline:
                logger.warning("❌ No Stage 2 proposal beats baseline. Ending experiment.")
                self.current_stage = None
                return None
            
            # Best proposal beats baseline - proceed to Stage 3 (ablation studies)
            logger.info(f"✅ Best Stage 2 proposal beats baseline. Proceeding to Stage 3 (Ablation Studies)")
            
            return Stage(
                name="3_ablation_studies_1_first_attempt",
                description="first_attempt",
                goals=MAIN_STAGE_GOALS[3],
                max_iterations=self._get_max_iterations(3),
                num_drafts=0,
                stage_number=current_substage.stage_number + 1,
            )

        # Custom initial creative_research setup when innovation queue mode is ON
        elif self.use_innovation_queue and main_stage_num + 1 == 2 and self.proposals:
            proposal = self.proposals[0]
            stage2_config = self.cfg.experiment.stage2
            return Stage(
                name="2_creative_research_1_proposal_1",
                description=proposal.get('title', 'Proposal 1'),
                goals=[proposal.get('title', '')],
                max_iterations=stage2_config.max_iterations_per_innovation,
                num_drafts=0,
                stage_number=current_substage.stage_number + 1,
                proposal_idx=0,  # First proposal (0-based)
            )
        next_main_stage_name = MAIN_STAGE_DICT[main_stage_num + 1]
        sub_stage_num = 1
        sub_stage_name = "first_attempt"
        num_drafts = 0
        stage_number = current_substage.stage_number + 1
        description = f"first_attempt"
        main_stage_goal = MAIN_STAGE_GOALS[main_stage_num + 1]

        # 🆕 For Stage 2, set proposal_idx even on the default path
        proposal_idx = 0 if main_stage_num + 1 == 2 and self.proposals else None

        return Stage(
            name=f"{main_stage_num + 1}_{next_main_stage_name}_{sub_stage_num}_{sub_stage_name}",
            description=description,
            goals=main_stage_goal,
            max_iterations=self._get_max_iterations(main_stage_num + 1),
            num_drafts=num_drafts,
            stage_number=stage_number,
            proposal_idx=proposal_idx,  # 🆕 Stage 2 uses the first proposal
        )

    def run(self, step_callback=None):
        """Run the experiment through generated stages"""
        # Check whether we only run literature search + innovation queue generation
        if self.literature_search_only:
            logger.info(f"Literature search only mode: Generated {len(self.proposals)} proposals, skipping experiments")

            # Set current_stage to None to mark the experiment as complete
            self.current_stage = None
            return

        # If a precomputed baseline is loaded, skip Stage 1 execution
        if getattr(self, '_precomputed_baseline_loaded', False):
            logger.info("🚀 Precomputed baselines loaded, skipping Stage 1 execution")
            
            # Generate the Stage 1 summary (even when preloaded)
            if self.current_stage and self.current_stage.name in self.journals:
                self._save_stage_summary_markdown(
                    self.current_stage, 
                    self.journals[self.current_stage.name]
                )
            
            # Create Stage 2 directly
            next_main_stage = self._create_next_main_stage(
                self.stages[-1], self.journals[self.stages[-1].name]
            )
            
            if next_main_stage:
                self.stage_history.append(
                    StageTransition(
                        from_stage=self.stages[-1].name,
                        to_stage=next_main_stage.name,
                        reason="Precomputed baselines loaded, moving to Stage 2",
                        config_adjustments={},
                    )
                )
                self.stages.append(next_main_stage)
                jmain = Journal()
                jmain.stage_name = next_main_stage.name
                self.journals[next_main_stage.name] = jmain
                self.current_stage = next_main_stage
                
                logger.info(f"Advancing to Stage 2: {next_main_stage.name}")
            else:
                logger.warning("Failed to create Stage 2 after precomputed baselines")
                self.current_stage = None
                return
            
            # Reset the flag
            self._precomputed_baseline_loaded = False

        # Print the current state
        logger.debug(f"Starting run() with current_stage: {self.current_stage.name if self.current_stage else 'None'}")
        logger.debug(f"Available stages: {[s.name for s in self.stages]}")
        logger.debug(f"Last stage in list: {self.stages[-1].name if self.stages else 'None'}")

        while self.current_stage:  # Main stage loop
            main_stage = self.parse_stage_names(self.current_stage.name)[0]
            logger.info(f"Starting main stage: {main_stage}")
            logger.info(f"Goals: {self.current_stage.goals}")

            current_substage = self.current_stage
            _pre_check_exit = False
            while current_substage:  # Sub-stage loop
                # Stage 2 early exit: if a *different* completed substage already
                # beats baseline, skip remaining substages.  Exclude the current
                # substage so that an in-progress substage resumed from checkpoint
                # is not mistakenly treated as an already-completed winner.
                if current_substage.name.startswith("2_") and self.use_innovation_queue:
                    _, beats_baseline = self._get_best_stage2_proposal(
                        exclude_stage=current_substage.name
                    )
                    if beats_baseline:
                        logger.info(
                            "💡 Stage 2 early exit: a completed proposal already "
                            "beats baseline. Skipping remaining substages."
                        )
                        current_substage = None
                        break

                logger.info(f"Starting sub-stage: {current_substage.name}")
                
                # 🆕 Stage 2: reset diagnostic state
                if current_substage.name.startswith("2_") and self.proposal_diagnostic:
                    self.proposal_diagnostic.reset_substage_state(current_substage.name)
                    logger.info(f"🔄 Resetting substage diagnostic state: {current_substage.name}")
                
                with self._create_agent_for_stage(current_substage) as agent:
                    # Initialize with best result from previous sub-stage if available
                    if self.stage_history:
                        prev_stage = self.stage_history[-1].from_stage
                        logger.debug(f"prev_stage: {prev_stage}")
                        logger.debug(f"self.stage_history: {self.stage_history}")
                        prev_best = self._get_best_implementation(prev_stage)
                        if prev_best is not None:
                            current_journal = self.journals[self.current_stage.name]
                            if len(current_journal.nodes) == 0:
                                current_journal.append(prev_best)
                            else:
                                logger.info(
                                    f"Journal for {self.current_stage.name} already has "
                                    f"{len(current_journal.nodes)} nodes, "
                                    f"skipping seed injection (likely checkpoint resume)"
                                )
                        else:
                            logger.info(
                                f"No previous best implementation found for {prev_stage}. Continuing without seeding."
                            )

                    # Run until sub-stage completion
                    _first_iter = True
                    _pre_check_exit = False
                    while True:
                        # On checkpoint resume the journal may already satisfy
                        # completion criteria.  Check once before the first step
                        # to avoid running an unnecessary extra iteration.
                        if _first_iter:
                            _first_iter = False
                            _pre_main_done, _pre_main_fb = self._check_stage_completion(current_substage)
                            if _pre_main_done:
                                logger.info("Pre-check: stage already complete on resume")
                                _pre_check_exit = True
                                self._save_stage_summary_markdown(current_substage, self.journals[current_substage.name])
                                if current_substage.stage_number in [1, 2, 3, 4]:
                                    best_node = self._get_best_implementation(current_substage.name)
                                    if best_node:
                                        logger.info(f"Stage {current_substage.name} completed successfully with best node: {best_node.id}")
                                    else:
                                        logger.error(f"No best node found for {current_substage.name}, finishing experiment...")
                                        self.current_stage = None
                                        current_substage = None
                                        break
                                current_substage = None
                                break
                            _pre_sub_done, _pre_sub_fb = self._check_substage_completion(
                                current_substage, self.journals[current_substage.name]
                            )
                            if _pre_sub_done:
                                logger.info("Pre-check: substage already complete on resume")
                                _pre_check_exit = True
                                self._save_stage_summary_markdown(current_substage, self.journals[current_substage.name])
                                next_substage = self._create_next_substage(
                                    current_substage,
                                    self.journals[current_substage.name],
                                    _pre_sub_fb,
                                )
                                if next_substage is None:
                                    current_substage = None
                                    break
                                self.stages.append(next_substage)
                                jsub = Journal()
                                jsub.stage_name = next_substage.name
                                self.journals[next_substage.name] = jsub
                                current_substage = next_substage
                                self.current_stage = next_substage
                                break

                        agent.step()
                        # After each mini-experiment, immediately save a checkpoint named after the current substage
                        self._save_checkpoint(current_substage)
                        if step_callback:
                            step_callback(
                                current_substage, self.journals[current_substage.name]
                            )
                        
                        # 🆕 Stage 2: check whether the proposal needs inline refinement
                        if current_substage.name.startswith("2_"):
                            # Fetch the baseline node
                            stage1_substages = [s for s in self.stages if s.name.startswith("1_")]
                            baseline_node = self._get_best_implementation(stage1_substages[-1].name) if stage1_substages else None
                            
                            # Run diagnosis and refinement checks
                            self._check_and_refine_proposal_inline(
                                current_substage=current_substage,
                                journal=self.journals[current_substage.name],
                                baseline_node=baseline_node
                            )
                            # Note: Detailed logging is handled inside the method

                        # First check if main stage is complete
                        (
                            main_stage_complete,
                            main_stage_feedback,
                        ) = self._check_stage_completion(current_substage)
                        logger.debug(
                            f"Feedback from _check_stage_completion: {main_stage_feedback}"
                        )
                        if main_stage_complete:
                            # Generate and save the current substage summary
                            self._save_stage_summary_markdown(current_substage, self.journals[current_substage.name])
                            
                            # After main stage completion, verify we have a best node
                            if current_substage.stage_number in [1, 2, 3, 4]:
                                best_node = self._get_best_implementation(
                                    current_substage.name
                                )
                                if best_node:
                                    logger.info(
                                        f"Stage {current_substage.name} completed successfully with best node: {best_node.id}"
                                    )
                                    if step_callback:
                                        step_callback(
                                            current_substage,
                                            self.journals[current_substage.name],
                                        )
                                else:
                                    logger.error(
                                        f"No best node found for {current_substage.name}, something went wrong so finishing the experiment..."
                                    )
                                    self.current_stage = None
                                    current_substage = None
                                    break

                            # Exit the loop to move to next main stage
                            current_substage = None
                            break

                        (
                            substage_complete,
                            substage_feedback,
                        ) = self._check_substage_completion(
                            current_substage, self.journals[current_substage.name]
                        )

                        if substage_complete:
                            # Generate and save the current substage summary
                            self._save_stage_summary_markdown(current_substage, self.journals[current_substage.name])
                            
                            # Derive the next substage from the current substage result (may return None)
                            next_substage = self._create_next_substage(
                                current_substage,
                                self.journals[current_substage.name],
                                substage_feedback,
                            )

                            # If there are no more substages, break and let the outer loop end the main stage
                            if next_substage is None:
                                current_substage = None
                                break

                            # Otherwise add the new substage to the manager
                            self.stages.append(next_substage)
                            jsub = Journal()
                            jsub.stage_name = next_substage.name
                            self.journals[next_substage.name] = jsub
                            current_substage = next_substage
                            # Keep self.current_stage in sync for accurate checkpointing
                            self.current_stage = next_substage
                            break
            # Save a main-stage-level checkpoint (skipped on pre-check exit to avoid overwriting the original checkpoint)
            if not _pre_check_exit:
                self._save_checkpoint(self.current_stage)
            # Main stage complete - create next main stage
            if self.current_stage:
                # Recompute exec.timeout during the Stage 1 -> Stage 2 transition
                if main_stage == 1:
                    self._recalculate_dynamic_timeout()

                next_main_stage = self._create_next_main_stage(
                    self.stages[-1], self.journals[self.stages[-1].name]
                )
                if next_main_stage:
                    # Record main stage transition
                    self.stage_history.append(
                        StageTransition(
                            from_stage=self.stages[-1].name,
                            to_stage=next_main_stage.name,
                            reason=f"Moving to {next_main_stage.description}",
                            config_adjustments={},
                        )
                    )

                    self.stages.append(next_main_stage)
                    jmain = Journal()
                    jmain.stage_name = next_main_stage.name
                    self.journals[next_main_stage.name] = jmain
                    self.current_stage = next_main_stage
                else:
                    # Exit the outer loop if no more main stages
                    if self.current_stage:
                        logger.info(f"Completed stage: {self.current_stage.name}")
                    logger.info("No more stages to run -- exiting the loop...")
                    self.current_stage = None

    def _create_stage_analysis_prompt(
        self,
        previous_stages: List[Stage],
        previous_results: Optional[Dict[str, Any]],
        is_initial_stage: bool,
    ) -> str:
        """Create detailed prompt to determine next stage configuration"""
        prompt_parts = [
            f"Task Description: {self._get_task_desc_str()}",
            f"Current Stage Number: {previous_stages[-1].stage_number}",
        ]

        if previous_stages:
            stage_history = "\n".join(
                f"Stage {i+1}: {stage.name} - {stage.description}"
                for i, stage in enumerate(previous_stages)
            )
            prompt_parts.append(f"Previous Stages:\n{stage_history}")

        if previous_results:
            # Format node summaries
            if "node_summaries" in previous_results["metrics"]:
                summaries = "\n".join(
                    f"Node {i}: {summary}"
                    for i, summary in enumerate(
                        previous_results["metrics"]["node_summaries"]
                    )
                )
                prompt_parts.append(f"Node Analysis:\n{summaries}")

            # Format VLM feedback and plot analysis
            if "plot_insights" in previous_results:
                plot_insights = previous_results["plot_insights"]
                prompt_parts.append("Visual Analysis Findings:")
                for analysis in plot_insights["analyses"]:
                    prompt_parts.append(f"- {analysis['analysis']}")

            # Format other metrics and findings
            metrics_summary = (
                f"Progress Summary:\n"
                f"- Total attempts: {previous_results['metrics']['total_nodes']}\n"
                f"- Successful implementations: {previous_results['metrics']['good_nodes']}\n"
                f"- Failed attempts: {previous_results['metrics']['buggy_nodes']}\n"
                f"- Best performance: {previous_results['metrics']['best_metric']['value'] if previous_results['metrics']['best_metric'] else 'N/A'}\n"
                f"- Issues identified: {', '.join(previous_results['issues'])}\n"
                f"- Progress status: {previous_results['progress']['convergence_status']}"
            )
            prompt_parts.append(metrics_summary)

            # Save stage transition analysis to notes directory
            base_dir = Path(self.workspace_dir).parent.parent
            run_name = Path(self.workspace_dir).name
            notes_dir = (
                base_dir
                / "logs"
                / run_name
                / "notes"
                / f"stage_{previous_stages[-1].stage_number-1}_to_{previous_stages[-1].stage_number}"
            )
            notes_dir.mkdir(parents=True, exist_ok=True)

            analysis_data = {
                "stage_transition": {
                    "from_stage": previous_stages[-1].stage_number - 1,
                    "to_stage": previous_stages[-1].stage_number,
                    "is_initial_stage": is_initial_stage,  # Add flag for initial stage
                    "metrics_summary": metrics_summary,
                    "node_summaries": previous_results["metrics"].get(
                        "node_summaries", []
                    ),
                    "plot_insights": previous_results.get("plot_insights", {}),
                    "issues": previous_results["issues"],
                    "progress": previous_results["progress"],
                }
            }

            with open(notes_dir / "stage_transition_analysis.json", "w") as f:
                json.dump(analysis_data, f, indent=2)

        prompt_parts.append(
            "Based on the above comprehensive analysis, determine the appropriate "
            "configuration for the next experimental stage. Consider:\n"
            "1. Visual analysis insights from plots\n"
            "2. Individual node performance and patterns\n"
            "3. Overall progress and convergence status\n"
            "4. Identified issues and challenges\n\n"
            "Include:\n"
            "1. Stage name (brief, descriptive)\n"
            "2. Detailed description of the stage's purpose\n"
            "3. Specific, measurable goals\n"
            "4. Maximum iterations needed\n"
            "5. Success metric threshold (if applicable)"
        )

        return "\n\n".join(prompt_parts)

    def parse_stage_names(self, stage_name: str) -> Tuple[int, str, int, str]:
        """Parse stage name into main stage number, main stage name,
        sub-stage number, and sub-stage name"""
        # Find the two numbers in the current stage name
        numbers = [int(n) for n in re.findall(r"\d+", stage_name)]

        main_stage = numbers[0]
        sub_stage_num = numbers[1]
        # Extract main_stage_name (everything between the two numbers)
        parts = re.split(r"\d+", stage_name)[1:-1]
        main_stage_name = "_".join(p.strip("_") for p in parts if p.strip("_"))
        # Extract sub_stage_name (everything after the second number)
        sub_stage_name = re.split(r"\d+", stage_name)[-1].strip("_")

        return main_stage, main_stage_name, sub_stage_num, sub_stage_name

    def _save_stage_summary(
        self, current_results: Dict[str, Any], evaluation: Dict[str, Any]
    ):
        """Save comprehensive stage completion summary"""
        base_dir = Path(self.workspace_dir).parent.parent
        run_name = Path(self.workspace_dir).name
        notes_dir = (
            base_dir
            / "logs"
            / run_name
            / "notes"
            / f"stage_{self.current_stage.stage_number}_complete"
        )
        notes_dir.mkdir(parents=True, exist_ok=True)

        completion_data = {
            "stage_completion": {
                "stage_number": self.current_stage.stage_number,
                "stage_name": self.current_stage.name,
                "final_metrics": current_results["metrics"],
                "identified_issues": current_results["issues"],
                "progress_analysis": current_results["progress"],
                "plot_insights": current_results.get("plot_insights", {}),
                "progression_evaluation": {
                    "ready_for_next_stage": evaluation["ready_for_next_stage"],
                    "reasoning": evaluation["reasoning"],
                    "recommendations": evaluation["recommendations"],
                    "suggested_focus": evaluation["suggested_focus"],
                },
            }
        }

        with open(notes_dir / "stage_completion_summary.json", "w") as f:
            json.dump(completion_data, f, indent=2)

    def _get_response(self, prompt: str) -> Dict[str, Any]:
        """Get structured response from LLM for stage configuration.

        Args:
            prompt: The analysis prompt to send to the LLM

        Returns:
            Dictionary containing stage configuration with keys:
            - name: str
            - description: str
            - goals: List[str]
            - max_iterations: int
            - success_metric_threshold: Optional[float]
        """
        stage_config_spec = {
            "name": "generate_stage_config",
            "json_schema": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Brief, descriptive name for the stage",
                    },
                    "description": {
                        "type": "string",
                        "description": "Detailed description of the stage's purpose",
                    },
                    "goals": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of specific, measurable goals for this stage",
                    },
                    "max_iterations": {
                        "type": "integer",
                        "description": "Maximum number of iterations to run in this stage",
                    },
                },
                "required": ["name", "description", "goals", "max_iterations"],
            },
            "description": "Generate configuration for the next experimental stage",
        }

        try:
            response = query(
                system_message=None,
                user_message=prompt,
                func_spec=stage_config_spec,
                **{"model": get_role("feedback")["model"], "temperature": get_role("feedback").get("temperature", 0.9)},
            )
            return response

        except Exception as e:
            logger.error(f"Error getting LLM response: {e}")
            # Provide a fallback configuration in case of errors
            return {
                "name": "fallback_stage",
                "description": "Fallback stage due to LLM error",
                "goals": ["Recover from error and continue execution"],
                "max_iterations": 3,
                "success_metric_threshold": None,
            }

    def _gather_stage_metrics(self, journal: Journal) -> Dict[str, Any]:
        """Gather detailed metrics and analysis from the stage's nodes"""
        metrics = {
            "total_nodes": len(journal.nodes),
            "good_nodes": len(journal.good_nodes),
            "buggy_nodes": len(journal.buggy_nodes),
            "best_metric": None,
            "node_summaries": [],
        }

        # Gather individual node summaries
        for node in journal.nodes:
            if hasattr(node, "_agent"):
                node_summary = node._agent._generate_node_summary(node)
                metrics["node_summaries"].append(node_summary)

        # VLM feedback functionality has been removed

        best_node = journal.get_best_node()
        if best_node:
            metrics["best_metric"] = {
                "value": best_node.metric.value,
                "name": (
                    best_node.metric.name
                    if hasattr(best_node.metric, "name")
                    else "validation_metric"
                ),
                "maximize": (
                    best_node.metric.maximize
                    if hasattr(best_node.metric, "maximize")
                    else False
                ),
                "analysis": (
                    best_node.analysis if hasattr(best_node, "analysis") else None
                ),
            }

        return metrics

    def _identify_issues(self, journal: Journal) -> List[str]:
        """Identify systemic issues and challenges from the current stage's results"""
        issues = []

        # Look for patterns in leaf nodes (endpoints of improvement attempts)
        leaf_nodes = [n for n in journal.nodes if n.is_leaf]
        problem_leaves = [n for n in leaf_nodes if n.is_buggy]

        if problem_leaves:
            error_leaves = [n for n in problem_leaves if n.exc_type is not None]
            underperf_leaves = [n for n in problem_leaves if n.exc_type is None]

            if error_leaves:
                issues.append(f"Found {len(error_leaves)} leaf nodes with execution errors")
                error_patterns = {}
                for node in error_leaves:
                    if hasattr(node, "analysis"):
                        error_patterns.setdefault(node.analysis, []).append(node.id)
                for error_msg, node_ids in error_patterns.items():
                    if len(node_ids) >= 2:
                        issues.append(f"Persistent error in nodes {node_ids}: {error_msg}")

            if underperf_leaves:
                issues.append(f"Found {len(underperf_leaves)} leaf nodes underperforming vs baseline")

        # Include VLM-identified systemic issues
        # VLM analysis functionality has been removed
        vlm_issues = set()  # Placeholder for removed functionality

        issues.extend(list(vlm_issues))

        return issues

    def _analyze_progress(self, journal: Journal) -> Dict[str, Any]:
        """Analyze progress and convergence in the current stage"""
        progress = {
            "iterations_completed": len(journal.nodes),
            "improvements_found": 0,
            "convergence_status": "not_converged",
            "improvement_trend": [],
            "recent_changes": [],
        }

        # Analyze recent changes
        recent_nodes = journal.nodes[-3:] if len(journal.nodes) >= 3 else journal.nodes
        for node in recent_nodes:
            if not node.is_buggy:
                change = {
                    "node_id": node.id,
                    "metric": node.metric.value,
                    "parent_id": node.parent.id if node.parent else None,
                    "analysis": node.analysis if hasattr(node, "analysis") else None,
                }
                progress["recent_changes"].append(change)

        return progress

    def _evaluate_stage_progression(
        self, current_stage: Stage, previous_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate whether experiment is ready for next stage"""

        eval_prompt = f"""
        Evaluate whether the current experimental stage should progress to the next stage.
        Consider all available evidence holistically:

        Current Stage Information:
        - Name: {current_stage.name}
        - Description: {current_stage.description}
        - Goals: {', '.join(current_stage.goals) if isinstance(current_stage.goals, list) else current_stage.goals}

        Performance Metrics:
        {json.dumps(previous_results.get('metrics', {}), indent=2)}

        Identified Issues:
        {json.dumps(previous_results.get('issues', []), indent=2)}

        Progress Analysis:
        {json.dumps(previous_results.get('progress', {}), indent=2)}

        Expected Stage Progression:
        1. Initial Implementation: Focus on basic working implementation
        2. Baseline Tuning: Systematic optimization of core parameters
        3. Creative Research: Novel improvements and approaches
        4. Ablation Studies: Systematic component analysis

        Consider factors like:
        - Progress toward stage goals
        - Performance trends and stability
        - Quality and reliability of results
        - Understanding of the problem
        - Presence of systematic issues
        - Convergence indicators
        - Readiness for next stage challenges

        Provide a holistic evaluation of whether the experiment should:
        1. Progress to next stage
        2. Continue current stage with specific focus
        3. Extend current stage with modifications
        """

        try:
            evaluation = query(
                system_message=None,
                user_message=eval_prompt,
                func_spec=stage_progress_eval_spec,
                **{"model": get_role("feedback")["model"], "temperature": get_role("feedback").get("temperature", 0.9)},
            )

            # Log the evaluation for transparency
            logger.info(
                f"Stage progression evaluation:\n{json.dumps(evaluation, indent=2)}"
            )

            return evaluation

        except Exception as e:
            logger.error(f"Error in stage progression evaluation: {e}")
            return {
                "ready_for_next_stage": False,
                "reasoning": "Error in evaluation process - continuing current stage",
                "recommendations": [
                    "Address evaluation error",
                    "Continue current approach",
                ],
                "suggested_focus": "Maintain current direction while resolving evaluation issues",
            }

    def _parse_innovation_response(self, innovation_response: str) -> Optional[Dict[str, str]]:
        """Parse the innovation response returned by _generate_dynamic_innovation."""
        try:
            # Parse the "Innovation: xxx\nDescription: xxx" format
            lines = innovation_response.strip().split('\n')
            innovation_name = None
            description = None
            
            for line in lines:
                line = line.strip()
                if "Innovation:" in line:
                    innovation_name = line.split("Innovation:", 1)[1].strip()
                elif "Description:" in line:
                    description = line.split("Description:", 1)[1].strip()
            
            if innovation_name and description:
                return {
                    "name": innovation_name,
                    "description": description
                }
            else:
                logger.warning(f"Failed to parse innovation response: {innovation_response}")
                return None
        except Exception as e:
            logger.error(f"Error parsing innovation response: {e}")
            return None

    # ------------------------------------------------------------------
    # Pickle compatibility helpers (allow loading checkpoints created
    # before the introduction of some attributes)
    # ------------------------------------------------------------------

    def __getstate__(self):
        """Return state for pickling."""
        return self.__dict__

    def __setstate__(self, state):
        logger.debug("__setstate__ called")
        """Restore state from pickle while ensuring new attributes exist."""
        self.__dict__.update(state)
        
        # Print the resumed state
        if hasattr(self, "current_stage") and self.current_stage:
            logger.debug(f"Restored current_stage: {self.current_stage.name}")
        else:
            logger.debug("Restored current_stage is None")
            
        if hasattr(self, "stages") and self.stages:
            logger.debug(f"Available stages: {[s.name for s in self.stages]}")
            logger.debug(f"Last stage in list: {self.stages[-1].name}")
        else:
            logger.debug("No stages available")
            
        # Compatibility: ensure the preloaded-baseline flag exists
        if "_precomputed_baseline_loaded" not in self.__dict__:
            self._precomputed_baseline_loaded = False

        # Recalculate executed_substages_count from the actual stages list.
        # Formula: (number of Stage 2 substages) - 1, because the first
        # substage (P1) is created by _create_next_main_stage (not counted),
        # while _create_next_substage increments the counter for each
        # subsequent substage (P2, P3, ...).
        if hasattr(self, "stages") and self.stages:
            stage2_count = sum(
                1 for s in self.stages if s.name.startswith("2_creative_research_")
            )
            recalculated = max(0, stage2_count - 1)
            old_val = getattr(self, "executed_substages_count", 0)
            if recalculated != old_val:
                logger.info(
                    f"Recalculated executed_substages_count on resume: "
                    f"{old_val} -> {recalculated}"
                )
            self.executed_substages_count = recalculated

        # 🔧 Compatibility handling for legacy checkpoint directory layouts
        # Detect the legacy layout (separate workspace_dir and log_dir) and migrate automatically
        self._migrate_legacy_directory_structure()

        # Retroactively label journals and nodes with stage names for older checkpoints
        if hasattr(self, "journals"):
            for stage_name, journal in self.journals.items():
                if journal is None:
                    continue
                if not hasattr(journal, "stage_name") or journal.stage_name is None:
                    journal.stage_name = stage_name
                for node in getattr(journal, "nodes", []):
                    if not hasattr(node, "origin_stage") or node.origin_stage is None:
                        node.origin_stage = stage_name

        # Rebuild parent/children references in every journal to drop ghost parents
        if hasattr(self, "journals"):
            for stage_name, journal in self.journals.items():
                if journal is not None and hasattr(journal, 'rebuild_relationships'):
                    journal.rebuild_relationships()
                    logger.debug("Rebuilt node relationships for journal: %s", stage_name)

        # If current_stage is missing or not in stages, fall back to the latest stage
        if (
            ("current_stage" not in self.__dict__ or self.current_stage is None)
            and hasattr(self, "stages")
            and self.stages
        ):
            self.current_stage = self.stages[-1]
            logger.debug(f"current_stage was None, set to: {self.current_stage.name}")
        elif hasattr(self, "stages") and self.stages and self.current_stage not in self.stages:
            # If the pointer lags behind, also update to the last stage
            old_stage_name = self.current_stage.name if self.current_stage else "None"
            self.current_stage = self.stages[-1]
            logger.debug(f"current_stage {old_stage_name} not in stages list, updated to: {self.current_stage.name}")
        
        ## Smart detection: read the checkpoint path from env vars
        if hasattr(self, "stages") and self.stages:
            checkpoint_path = os.environ.get("QWBE_CHECKPOINT_PATH")
            if checkpoint_path:
                logger.debug(f"Found checkpoint path from env: {checkpoint_path}")
                try:
                    import re
                    stage_match = re.search(r'stage_([^/]+)', checkpoint_path)
                    if stage_match:
                        target_stage_name = stage_match.group(1)
                        logger.debug(f"Detected target stage: {target_stage_name}")
                        
                        matching_stages = [s for s in self.stages if s.name == target_stage_name]
                        if matching_stages:
                            logger.debug(f"Found matching stage: {matching_stages[0].name}")
                            self.current_stage = matching_stages[0]
                            logger.debug(f"Set current_stage to: {self.current_stage.name}")
                except Exception as e:
                    logger.debug(f"Error during intelligent stage detection: {e}")
                    logger.debug("Falling back to default stage recovery logic")

        # 🔧 Compatibility fix: backfill proposal_idx on Stage objects in legacy checkpoints
        if hasattr(self, "stages") and self.stages:
            for stage in self.stages:
                if stage.name.startswith("2_") and not hasattr(stage, 'proposal_idx'):
                    # Older Stage objects may not have the proposal_idx attribute
                    stage.proposal_idx = None
                if stage.name.startswith("2_") and stage.proposal_idx is None:
                    # Try to infer proposal_idx from the stage name
                    try:
                        parts = stage.name.split("_")
                        if "proposal" in parts:
                            proposal_pos = parts.index("proposal")
                            if proposal_pos + 1 < len(parts):
                                stage.proposal_idx = int(parts[proposal_pos + 1]) - 1
                                logger.debug(f"Fixed proposal_idx for {stage.name}: {stage.proposal_idx}")
                        elif len(parts) >= 4:
                            # "2_creative_research_1_first_attempt" format
                            sub_stage_num = int(parts[3])
                            stage.proposal_idx = sub_stage_num - 1
                            logger.debug(f"Fixed proposal_idx for {stage.name}: {stage.proposal_idx}")
                    except Exception as e:
                        logger.debug(f"Could not fix proposal_idx for {stage.name}: {e}")
        
        # 🆕 Verify and regenerate any missing proposal files
        self._validate_and_regenerate_proposals()

        # Note: do NOT call _recalculate_dynamic_timeout() here.
        # The pickled cfg.exec.timeout may already be a dynamically computed value (e.g. 72000);
        # using it as the base here would cause compound amplification (72000 -> 720000).
        # The correct call site is perform_experiments_qwbe(), which first restores the
        # original base value from YAML and only then calls _recalculate_dynamic_timeout().

    def _migrate_legacy_directory_structure(self):
        """Detect the legacy directory layout (separate workspace_dir and log_dir) and migrate files automatically.
        
        Legacy layout:
            experiments/{exp}/0-run/          <- workspace_dir (input/, working/, research_proposals/)
            experiments/{exp}/logs/0-run/     <- log_dir (checkpoints, stage logs)
        
        New layout:
            experiments/{exp}/logs/0-run/     <- workspace_dir == log_dir (all files unified)
        """
        try:
            if not hasattr(self, 'workspace_dir') or not self.workspace_dir:
                logger.debug("No workspace_dir found, skipping migration check")
                return
            
            workspace_path = Path(self.workspace_dir)
            
            # Detect legacy-layout marker: a sibling "logs" directory next to workspace_dir
            # Legacy layout: experiments/{exp}/0-run/ (workspace) and experiments/{exp}/logs/0-run/ (log)
            potential_logs_dir = workspace_path.parent / "logs" / workspace_path.name
            
            # Detect whether this is the legacy layout
            is_legacy = (
                potential_logs_dir.exists() and  # logs/0-run/ directory exists
                workspace_path.parent.name != "logs" and  # workspace is not under logs
                workspace_path.name == potential_logs_dir.name  # names match (both 0-run)
            )
            
            if not is_legacy:
                logger.debug("New unified directory structure detected, no migration needed")
                return
            
            logger.debug(f"Detected LEGACY directory structure, starting migration...")
            logger.debug(f"Old workspace_dir: {workspace_path}")
            logger.debug(f"New workspace_dir (log_dir): {potential_logs_dir}")
            
            # Migrate key files
            migrated_count = 0
            
            # List of directories to migrate
            dirs_to_migrate = ["input", "working", "research_proposals"]
            
            for dir_name in dirs_to_migrate:
                old_dir = workspace_path / dir_name
                new_dir = potential_logs_dir / dir_name
                
                if old_dir.exists() and not new_dir.exists():
                    logger.debug(f"Migrating {dir_name}/ ...")
                    import shutil
                    shutil.copytree(old_dir, new_dir)
                    migrated_count += 1
                    logger.debug(f"Migrated: {old_dir} -> {new_dir}")
                elif old_dir.exists() and new_dir.exists():
                    logger.debug(f"Both old and new {dir_name}/ exist, skipping (using new)")
                elif not old_dir.exists():
                    logger.debug(f"{dir_name}/ not in old workspace (will be created if needed)")
            
            # Point paths at the new unified directory
            old_workspace_str = str(workspace_path)
            self.workspace_dir = potential_logs_dir
            if hasattr(self, 'cfg') and hasattr(self.cfg, 'workspace_dir'):
                self.cfg.workspace_dir = potential_logs_dir
            if hasattr(self, 'cfg') and hasattr(self.cfg, 'log_dir'):
                self.cfg.log_dir = potential_logs_dir
            
            # 🔧 Update md_file paths inside proposals (replace old workspace paths with new ones)
            new_workspace_str = str(potential_logs_dir)
            if hasattr(self, 'proposals') and self.proposals:
                path_updated = 0
                for proposal in self.proposals:
                    md_file = proposal.get('md_file', '')
                    if md_file and old_workspace_str in md_file:
                        proposal['md_file'] = md_file.replace(old_workspace_str, new_workspace_str)
                        path_updated += 1
                if path_updated > 0:
                    logger.debug(f"Updated {path_updated} proposal md_file paths to new workspace")
            
            logger.debug(f"Migration complete: {migrated_count} directories migrated")
            logger.debug(f"Updated workspace_dir to: {self.workspace_dir}")
            
        except Exception as e:
            logger.debug(f"Migration check failed (non-fatal): {e}")
            import traceback
            traceback.print_exc()

    def _validate_and_regenerate_proposals(self):
        """Verify proposal MD files exist; attempt to relocate or regenerate missing ones.
        
        Called when resuming from a checkpoint.
        Priority order:
          1. Original path exists -> use directly.
          2. Match by filename under current workspace_dir/research_proposals/ -> update the path.
          3. Not found in either -> regenerate from metadata (fallback).
        """
        logger.debug("Validating proposal MD files...")
        
        if not hasattr(self, 'proposals') or not self.proposals:
            logger.debug("No proposals to validate")
            return
        
        # Resolve the current research_proposals directory
        proposal_config = getattr(self.cfg.idea_generation, "research_proposal", None) if hasattr(self, 'cfg') else None
        if proposal_config:
            output_dir_name = getattr(proposal_config, "output_dir", "research_proposals")
        else:
            output_dir_name = "research_proposals"
        
        current_proposals_dir = None
        if hasattr(self, 'workspace_dir') and self.workspace_dir:
            current_proposals_dir = Path(self.workspace_dir) / output_dir_name
        elif hasattr(self, 'cfg') and hasattr(self.cfg, 'workspace_dir'):
            current_proposals_dir = Path(self.cfg.workspace_dir) / output_dir_name
        
        # Gather all .md files under the current research_proposals directory (for fuzzy matching)
        existing_md_files = []
        if current_proposals_dir and current_proposals_dir.exists():
            existing_md_files = list(current_proposals_dir.glob("*.md"))
            logger.debug(f"Found {len(existing_md_files)} .md files in {current_proposals_dir}")
        
        # Check whether each proposal's MD file exists
        missing_proposals = []
        updated_count = 0
        for i, proposal in enumerate(self.proposals):
            md_file = proposal.get('md_file')
            if md_file:
                md_path = Path(md_file)
                if md_path.exists():
                    logger.debug(f"Found proposal {i+1}: {md_path.name}")
                    continue
                
                # Path missing -> try to find by filename under the current workspace_dir
                old_filename = md_path.name
                relocated = False
                
                if current_proposals_dir and current_proposals_dir.exists():
                    # Strategy 1: exact filename match
                    candidate = current_proposals_dir / old_filename
                    if candidate.exists():
                        self.proposals[i]['md_file'] = str(candidate)
                        logger.debug(f"Relocated proposal {i+1} (exact match): {candidate.name}")
                        relocated = True
                        updated_count += 1
                    else:
                        # Strategy 2: fuzzy match by proposal-number prefix (e.g. proposal_01_...)
                        prefix = f"proposal_{i+1:02d}_"
                        candidates = [f for f in existing_md_files 
                                      if f.name.startswith(prefix) 
                                      and not f.name.endswith('_refinement_log.json')]
                        
                        if candidates:
                            # Prefer the largest file (typically the most complete version)
                            best_candidate = max(candidates, key=lambda f: f.stat().st_size)
                            self.proposals[i]['md_file'] = str(best_candidate)
                            logger.debug(f"Relocated proposal {i+1} (fuzzy match): {best_candidate.name} ({best_candidate.stat().st_size} bytes)")
                            relocated = True
                            updated_count += 1
                
                if not relocated:
                    logger.debug(f"Missing proposal MD file: {md_file}")
                    missing_proposals.append((i, proposal))
        
        if not missing_proposals and updated_count > 0:
            logger.debug(f"All proposals validated: {updated_count} paths relocated to current workspace")
            if hasattr(self, 'cfg') and hasattr(self.cfg, 'log_dir'):
                self._save_initial_proposal_queue()
                logger.debug("Updated initial_proposal_queue.json with relocated paths")
            return
        elif not missing_proposals:
            logger.debug("All proposal MD files validated successfully")
            return
        
        logger.debug(f"{len(missing_proposals)} proposal files still missing after relocation, attempting regeneration...")
        
        try:
            if not current_proposals_dir:
                logger.debug("Cannot determine workspace_dir, skipping proposal regeneration")
                return
            
            # Create the directory if it does not exist
            current_proposals_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created/verified output directory: {current_proposals_dir}")
            
            regenerated_count = 0
            for i, proposal in missing_proposals:
                title = proposal.get('title', f'Proposal_{i+1}')
                safe_title = self._generate_safe_filename(title)
                new_md_path = current_proposals_dir / f"proposal_{i+1:02d}_{safe_title}.md"
                
                if self._regenerate_proposal_md(proposal, new_md_path):
                    self.proposals[i]['md_file'] = str(new_md_path)
                    regenerated_count += 1
                    logger.debug(f"Regenerated proposal {i+1}: {new_md_path.name}")
                else:
                    logger.debug(f"Could not regenerate proposal {i+1}")
            
            logger.debug(f"Recovery complete: {updated_count} relocated, {regenerated_count} regenerated")
            
            if (updated_count > 0 or regenerated_count > 0) and hasattr(self, 'cfg') and hasattr(self.cfg, 'log_dir'):
                self._save_initial_proposal_queue()
                logger.debug("Updated initial_proposal_queue.json with new paths")
                
        except Exception as e:
            logger.debug(f"Error during proposal recovery: {e}")
            import traceback
            traceback.print_exc()

    def _generate_safe_filename(self, title: str, max_length: int = 50) -> str:
        """Generate a safe filename."""
        import re
        # Remove or replace unsafe characters
        safe = re.sub(r'[^\w\s-]', '', title)
        safe = re.sub(r'[-\s]+', '_', safe)
        # Truncate overly long filenames
        if len(safe) > max_length:
            safe = safe[:max_length]
        return safe

    def _regenerate_proposal_md(self, proposal: dict, output_path: Path) -> bool:
        """Regenerate an MD file from proposal metadata.

        🆕 Improvement: now supports restoring full content (modules, motivation, etc.) from full_data.

        Args:
            proposal: proposal dict containing title, challenge_theme, full_data, etc.
            output_path: destination file path.

        Returns:
            bool: whether generation succeeded.
        """
        try:
            title = proposal.get('title', 'Unknown Proposal')
            challenge_theme = proposal.get('challenge_theme', 'Unknown Theme')
            generator = proposal.get('generator', 'unknown')
            
            # Check whether the full proposal data is available
            full_data = proposal.get('full_data', {})
            
            # Check that full_data has meaningful content
            has_full_content = full_data and (
                full_data.get('modules') or 
                full_data.get('motivation') or
                full_data.get('integration')
            )
            
            if has_full_content:
                # 🆕 Restore the MD file from full data (matching the original format)
                motivation = full_data.get('motivation', {})
                modules = full_data.get('modules', [])
                
                md_content = f"""# {title}

**Core Theme**: {full_data.get('core_theme', challenge_theme)}  
**Source Challenge**: {full_data.get('source_challenge', 'N/A')}  
**Generated by**: {generator}

---

## 1. Motivation

### 1.1 Background and Problem
{motivation.get('background', 'N/A') if isinstance(motivation, dict) else 'N/A'}

### 1.2 Existing Limitations
{motivation.get('limitations', 'N/A') if isinstance(motivation, dict) else 'N/A'}

### 1.3 Our Insight
{motivation.get('insight', 'N/A') if isinstance(motivation, dict) else 'N/A'}

---

## 2. Proposed Method

"""
                
                # Add module information
                for i, module in enumerate(modules, 1):
                    if isinstance(module, dict):
                        md_content += f"""### 2.{i} {module.get('name', 'Module')}

**Technical Description**:  
{module.get('description', 'N/A')}

**Mathematical Formulation**:  
{module.get('formulation', 'N/A')}

**Role in Architecture**:  
{module.get('role', 'N/A')}

"""
                
                md_content += f"""---

## 3. Integration and Data Flow

{full_data.get('integration', 'N/A')}

---

## 4. Expected Contributions

"""
                
                for i, contrib in enumerate(full_data.get('contributions', []), 1):
                    md_content += f"{i}. {contrib}\n"
                
                md_content += """
---

> ⚠️ **Note**: This file was regenerated from checkpoint data.
"""
                
            else:
                # Fallback: simplified version when only basic metadata is available
                md_content = f"""# {title}

## 1. Title
{title}

## 2. Challenge Theme
{challenge_theme}

## 3. Generator
{generator}

## Note
⚠️ This proposal file was regenerated from checkpoint metadata. 
The original detailed content was lost and only basic information is available.

Please refer to the experiment logs and OpenHands summaries for more details about this proposal's implementation.
"""
            
            # Write the file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(md_content)
            
            return True
            
        except Exception as e:
            logger.debug(f"Error regenerating proposal MD: {e}")
            return False

    def _initialize_proposal_queue_system(self):
        """Initialize the Research Proposal queue system (literature-search driven).
        
        If initialization fails, exit the process immediately.
        """
        try:
            # Import the Proposal integration module
            from .innovation_integration import integrate_proposal_queue_with_agent_manager

            # Integrate the Proposal queue manager
            self.proposal_queue_manager = integrate_proposal_queue_with_agent_manager(
                self,
                enable_literature_search=self.use_literature_search
            )

            # Fetch proposal-count configuration
            proposal_config = getattr(self.cfg.idea_generation, "research_proposal", None)
            if proposal_config:
                num_proposals = getattr(proposal_config, "num_proposals", 5)
                output_dir_name = getattr(proposal_config, "output_dir", "research_proposals")
            else:
                num_proposals = 5
                output_dir_name = "research_proposals"
            
            # Set up the output directory
            output_dir = Path(self.workspace_dir) / output_dir_name
            
            # Initialize the Proposal queue
            self.proposals = self.proposal_queue_manager.initialize_proposal_queue(
                task_desc=self.task_desc,
                num_proposals=num_proposals,
                output_dir=output_dir
            )

            # Save the initial Proposal queue
            self._save_initial_proposal_queue()

            logger.info(f"✅ Proposal queue system initialized: {len(self.proposals)} Research Proposals")

        except ImportError as e:
            logger.error(f"Fatal error: cannot import Proposal integration module: {e}")
            import traceback
            traceback.print_exc()
            import sys
            sys.exit(1)
            
        except Exception as e:
            logger.error(f"Fatal error: failed to initialize the Proposal queue system: {e}")
            import traceback
            traceback.print_exc()
            import sys
            sys.exit(1)

    def _initialize_proposal_diagnostic(self):
        """Initialize the Proposal diagnostic (used for inline substage refinement)."""
        try:
            if hasattr(self, 'proposal_queue_manager') and self.proposal_queue_manager:
                self.proposal_diagnostic = ProposalDiagnostic(
                    cfg=self.cfg,
                    proposal_queue_manager=self.proposal_queue_manager
                )
                logger.info("✅ Proposal diagnostic initialized")
            else:
                self.proposal_diagnostic = None
                logger.warning("⚠️ Proposal diagnostic not initialized: proposal_queue_manager unavailable")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize Proposal diagnostic: {e}")
            self.proposal_diagnostic = None

    def _check_and_refine_proposal_inline(
        self,
        current_substage: "Stage",
        journal: Journal,
        baseline_node: Optional[Node]
    ) -> bool:
        """
        Check and execute inline proposal diagnosis and refinement
        
        Called after each step() completes, determines if diagnosis/refinement is needed.
        
        Diagnosis types:
        - code_issue: Store diagnostic info (5 improvement suggestions) on node (NOT marked buggy),
          next _improve() injects them into the coding agent's prompt
        - proposal_infeasible: Execute proposal refinement
        
        Args:
            current_substage: Current substage
            journal: Current journal
            baseline_node: Baseline node
            
        Returns:
            bool: Whether action was taken (refinement or buggy marking)
        """
        # Check if diagnostic module is available
        if not self.proposal_diagnostic:
            return False
        
        # Only use in Stage 2
        if not current_substage.name.startswith("2_"):
            return False
        
        # Get baseline metric
        if not baseline_node or not baseline_node.metric:
            logger.debug("No baseline metric available for proposal diagnostic")
            return False
        
        baseline_metric = baseline_node.metric.get_mean_value()
        baseline_metric_value = baseline_node.metric
        
        # Get latest node
        if not journal.nodes:
            return False
        
        latest_node = journal.nodes[-1]
        
        # Check if diagnosis is needed
        if not self.proposal_diagnostic.should_diagnose(
            substage_name=current_substage.name,
            latest_node=latest_node,
            baseline_metric=baseline_metric,
            journal=journal,
            baseline_metric_value=baseline_metric_value
        ):
            # 🆕 If diagnosis is not needed because baseline has been beaten,
            # generate optimization hints with proposal completeness analysis
            if (self.proposal_diagnostic and 
                current_substage.name in self.proposal_diagnostic.substage_refinement_state):
                state = self.proposal_diagnostic.substage_refinement_state[current_substage.name]
                if state.get("has_beat_baseline", False):
                    # Retrieve proposal content for completeness analysis
                    proposal_content = None
                    proposal_idx = getattr(current_substage, 'proposal_idx', None)
                    if proposal_idx is not None:
                        best_node = journal.get_best_node()
                        proposal_content = getattr(best_node, 'proposal_content', None) if best_node else None
                        if not proposal_content:
                            try:
                                proposal_content = self.proposal_queue_manager.get_proposal_for_prompt(proposal_idx)
                            except Exception as e:
                                logger.debug(f"Could not retrieve proposal content for optimization hints: {e}")
                    
                    hints = self.proposal_diagnostic.generate_optimization_hints(
                        substage_name=current_substage.name,
                        journal=journal,
                        baseline_metric=baseline_metric,
                        latest_node=latest_node,
                        proposal_content=proposal_content,
                    )
                    if hints:
                        logger.info(f"📈 Generated optimization hints for {current_substage.name}")
                        logger.debug(f"Hints preview: {hints[:150]}...")
            return False
        
        # Get current proposal content — prefer node snapshot over global queue
        proposal_idx = getattr(current_substage, 'proposal_idx', None)
        if proposal_idx is None:
            logger.warning("Cannot diagnose: proposal_idx not available")
            return False
        
        proposal_content = getattr(latest_node, 'proposal_content', None)
        if proposal_content:
            logger.info(f"Using proposal snapshot from node {latest_node.id[:8]}")
        else:
            try:
                proposal_content = self.proposal_queue_manager.get_proposal_for_prompt(proposal_idx)
            except Exception as e:
                logger.error(f"Cannot get proposal content: {e}")
                return False
        
        # Build budget info for the diagnostic prompt
        refinement_state = self.proposal_diagnostic.substage_refinement_state.get(
            current_substage.name, {}
        )
        budget_info = {
            "proposal_steps_used": sum(
                1 for n in journal.nodes
                if not getattr(n, 'is_precomputed_baseline', False)
                and n.parent is not None
            ),
            "proposal_steps_total": current_substage.max_iterations,
            "proposal_idx": (getattr(current_substage, 'proposal_idx', 0) or 0) + 1,
            "num_proposals": len(self.proposals),
            "global_substages_used": self.executed_substages_count,
            "global_substages_total": self._get_max_iterations(2),
            "refinement_used": refinement_state.get("refinement_count", 0),
            "refinement_total": self.proposal_diagnostic.max_refinements_per_substage,
        }
        
        # Execute diagnosis
        logger.info(f"🔍 Executing Proposal Diagnosis (substage: {current_substage.name})")
        
        diagnosis = self.proposal_diagnostic.diagnose(
            substage_name=current_substage.name,
            proposal_content=proposal_content,
            journal=journal,
            baseline_metric=baseline_metric,
            latest_node=latest_node,
            budget_info=budget_info
        )
        
        if not diagnosis:
            logger.warning("Diagnosis failed")
            return False
        
        logger.info(f"📋 Diagnosis result: {diagnosis.diagnosis_type}")
        logger.debug(f"Diagnosis reasoning: {diagnosis.reasoning}...")
        
        # Handle different diagnosis types
        if diagnosis.diagnosis_type == "code_issue":
            latest_node.diagnostic_info = {
                "type": diagnosis.diagnosis_type,
                "reasoning": diagnosis.reasoning,
                "improvement_suggestions": diagnosis.improvement_suggestions,
                "baseline_metric": diagnosis.baseline_metric,
                "current_metric": diagnosis.current_metric,
                "metric_gap": diagnosis.metric_gap,
            }
            
            diag_summary = (
                f"\n[Diagnostic: {diagnosis.diagnosis_type}] "
                f"{diagnosis.reasoning}"
            )
            if latest_node.analysis:
                latest_node.analysis = f"{latest_node.analysis}{diag_summary}"
            else:
                latest_node.analysis = diag_summary.strip()
            
            logger.info(f"📋 Node {latest_node.id[:8]} annotated with diagnostic: {diagnosis.diagnosis_type} "
                        f"({len(diagnosis.improvement_suggestions)} suggestions, NOT marked buggy)")
            
            self.proposal_diagnostic.update_refinement_node_idx(
                current_substage.name, 
                len(journal.nodes)
            )
            
            return True
            
        elif diagnosis.diagnosis_type == "proposal_infeasible":
            refinement_state = self.proposal_diagnostic.substage_refinement_state.get(
                current_substage.name, {}
            )
            ref_count = refinement_state.get("refinement_count", 0)
            if ref_count >= self.proposal_diagnostic.max_refinements_per_substage:
                logger.warning(
                    f"⚠️ LLM diagnosed proposal_infeasible but refinement budget "
                    f"exhausted ({ref_count}/{self.proposal_diagnostic.max_refinements_per_substage}). "
                    f"Downgrading to code_issue (not marking buggy)."
                )
                latest_node.diagnostic_info = {
                    "type": "code_issue",
                    "reasoning": (
                        f"[Downgraded from proposal_infeasible — refinement budget exhausted] "
                        f"{diagnosis.reasoning}"
                    ),
                    "improvement_suggestions": diagnosis.improvement_suggestions,
                    "baseline_metric": diagnosis.baseline_metric,
                    "current_metric": diagnosis.current_metric,
                    "metric_gap": diagnosis.metric_gap,
                }
                if latest_node.analysis:
                    latest_node.analysis += (
                        f"\n[Diagnostic: code_issue (downgraded from proposal_infeasible)] "
                        f"{diagnosis.reasoning}"
                    )
                else:
                    latest_node.analysis = (
                        f"[Diagnostic: code_issue (downgraded from proposal_infeasible)] "
                        f"{diagnosis.reasoning}"
                    )
                self.proposal_diagnostic.update_refinement_node_idx(
                    current_substage.name,
                    len(journal.nodes)
                )
                return True
            
            # Execute proposal refinement
            logger.info(f"🔧 Executing Proposal Refinement")
            
            result = self.proposal_diagnostic.refine_proposal(
                substage_name=current_substage.name,
                proposal_idx=proposal_idx,
                diagnosis=diagnosis
            )
            
            if result.success:
                logger.info(f"✅ Proposal refinement successful: {result.new_file_path}")
                logger.debug(f"Changes summary: {result.changes_summary}")
                
                # Mark all old-proposal nodes as stale so QWBE and diagnostic skip them
                baseline_id = None
                stage1_substages = [s for s in self.stages if s.name.startswith("1_")]
                if stage1_substages:
                    bl_node = self._get_best_implementation(stage1_substages[-1].name)
                    baseline_id = bl_node.id if bl_node else None
                
                stale_count = 0
                for node in journal.nodes:
                    if (node.id != baseline_id
                            and not getattr(node, 'is_precomputed_baseline', False)
                            and not getattr(node, 'is_stale', False)):
                        node.is_stale = True
                        stale_count += 1
                
                logger.info(f"🔒 Marked {stale_count} old-proposal nodes as stale after refinement")
                
                # Update refinement node index
                self.proposal_diagnostic.update_refinement_node_idx(
                    current_substage.name, 
                    len(journal.nodes)
                )
                
                return True
            else:
                logger.warning(f"❌ Proposal refinement failed: {result.error_message}")
                return False
        
        return False

    def _extract_detailed_metric_improvements(self, innovation_node, baseline_node):
        """Extract detailed per-metric improvement info.
        
        Args:
            innovation_node: innovation node.
            baseline_node: baseline node.
            
        Returns:
            List[Dict]: detailed improvement info per metric.
        """
        if not innovation_node or not baseline_node:
            return []
        
        if not hasattr(innovation_node, 'metric') or not hasattr(baseline_node, 'metric'):
            return []
        
        innov_metric = innovation_node.metric
        baseline_metric = baseline_node.metric
        
        if not innov_metric or not baseline_metric:
            return []
        
        detailed_metrics = []
        
        # Check whether this is the new multi-metric format
        if isinstance(innov_metric.value, dict) and "metric_names" in innov_metric.value:
            innov_metrics = innov_metric.value["metric_names"]
            baseline_metrics = baseline_metric.value["metric_names"]
            
            # Compute improvement for each metric
            for innov_m, baseline_m in zip(innov_metrics, baseline_metrics):
                metric_name = innov_m.get("metric_name", "unknown")
                lower_is_better = innov_m.get("lower_is_better", False)
                
                # Fetch values
                innov_val = innov_m["data"][0]["final_value"] if innov_m["data"] else None
                baseline_val = baseline_m["data"][0]["final_value"] if baseline_m["data"] else None
                
                if innov_val is not None and baseline_val is not None:
                    # Compute absolute improvement
                    if lower_is_better:
                        # For lower-is-better metrics, improvement = baseline - innovation (positive = better)
                        improvement = baseline_val - innov_val
                        improvement_pct = (improvement / baseline_val * 100) if baseline_val != 0 else 0
                    else:
                        # For higher-is-better metrics, improvement = innovation - baseline
                        improvement = innov_val - baseline_val
                        improvement_pct = (improvement / baseline_val * 100) if baseline_val != 0 else 0
                    
                    detailed_metrics.append({
                        "metric_name": metric_name,
                        "baseline_value": baseline_val,
                        "innovation_value": innov_val,
                        "improvement": improvement,
                        "improvement_pct": improvement_pct,
                        "lower_is_better": lower_is_better
                    })
        
        return detailed_metrics

    def _save_initial_proposal_queue(self):
        """Save the initial Proposal queue to the experiment path."""
        try:
            # Determine the save path
            # 🔧 After directory merge: workspace_dir == log_dir, use directly
            logs_dir = Path(self.cfg.log_dir)
            logs_dir.mkdir(parents=True, exist_ok=True)

            # Save path
            proposal_queue_path = logs_dir / "initial_proposal_queue.json"

            # Prepare the data to save
            queue_data = {
                "timestamp": datetime.now().isoformat(),
                "experiment_name": Path(self.workspace_dir).name,
                "task_description": self.task_desc.get("Title", "Unknown Task"),
                "num_proposals": len(self.proposals),
                "generation_method": "literature_search" if (
                    self.use_literature_search and 
                    hasattr(self, 'proposal_queue_manager') and 
                    self.proposal_queue_manager and 
                    self.proposal_queue_manager.innovation_generator
                ) else "traditional",
                "proposals": self.proposals
            }

            # Save the JSON file
            with open(proposal_queue_path, "w", encoding="utf-8") as f:
                json.dump(queue_data, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"✅ Initial Proposal queue saved to: {proposal_queue_path}")

        except Exception as e:
            logger.error(f"Failed to save the initial Proposal queue: {e}")
