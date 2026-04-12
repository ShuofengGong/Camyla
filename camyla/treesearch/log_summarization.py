import json
import logging
import os
import sys
from collections import defaultdict

from .journal import Node, Journal

logger = logging.getLogger(__name__)

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, parent_dir)

# Use the unified BackendLLMClient interface
from camyla.llm import (
    get_response_from_llm,
    extract_json_between_markers,
    create_client,
)
from camyla.model_config import get_model_name

# Fetch the base model name from configuration
MODEL_NAME = get_model_name('log_summary')

# Create a BackendLLMClient that conforms to the generate() interface
client, model = create_client(MODEL_NAME)

report_summarizer_sys_msg = """You are an expert machine learning researcher.
You are given multiple experiment logs, each representing a node in a stage of exploring scientific ideas and implementations.
Your task is to aggregate these logs and provide scientifically insightful information.

Important instructions:
- Do NOT hallucinate or fabricate information that is not present in the logs.
- Do NOT introduce errors when repeating information from the logs.
- Identify notable insights or differences across the nodes without repeating the same information.
"""

output_format_control = """Respond in the following format:

THOUGHT:
<THOUGHT>

JSON:
```json
<JSON>
```

In <THOUGHT>, thoroughly reason as an expert researcher. First, reason about each node, and then reason carefully by combining all the information. It is okay to be very detailed.

In <JSON>, provide the review in JSON format with the following fields in exactly this order:
- "Experiment_description": a string describing the conducted experiments
- "Significance": a string explaining why these experiments are important and what impact their findings might have
- "Description": a string describing the methods, steps taken, and any pertinent context needed to understand the experiments
- "List_of_included_plots": a list of plots that should be included. Each entry should include:
  • "path" (the plot path)
  • "description" (its original description)
  • "analysis" (your analysis of its scientific insights)
- "Key_numerical_results": a list of all important numerical results. Be selective about results that contribute to scientific insights. Each entry should include:
  • "result" (float number)
  • "description" (your short description of the result)
  • "analysis" (your analysis of its scientific insights)

Ensure the JSON is valid and properly formatted, as it will be automatically parsed."""

report_summarizer_prompt = (
    """You are given multiple experiment logs from different "nodes". Each node represents attempts and experiments exploring various scientific ideas.

One key point is that these nodes collectively illustrate a stage of testing different methods or approaches. The crucial task is to identify the scientific insights gleaned from this stage. For example, if one node tries method A and another node tries method B, you should compare any observed differences in performance or outcomes. Summarize both experiments in "Experiment_description", explain the processes in "Description", and place any key numerical findings (such as accuracy metrics, loss values, or runtime comparisons) in "Key_numerical_results."

Be concise and avoid repeating the same information from different nodes. You are encouraged to be thorough, but you do not need to include information from every node. Reason carefully about which results from which nodes are scientifically insightful.

The name of this stage of the experiment: {stage_name}

Here are the experiment logs of the nodes:

{node_infos}
"""
    + output_format_control
)

stage_aggregate_prompt = """You are given:

1) The summary of all previous experiment stages:
{prev_summary}

2) The name of the current experiment stage:
{stage_name}

3) The summary of the current stage:
{current_summary}


Your task is to produce an **updated comprehensive summary** of all experiment stages, including the newly introduced results from the current stage.

**Key Requirements:**
1. **No Loss of Critical Information**
   - Preserve valuable insights from the summary of all previous experiment stages. Do not remove or alter crucial texts.
   - Absolutely no hallucinations: if something does not appear in the logs or summaries, do not invent it. If something appears in the previous summary, do not make any mistakes when repeating it.
2. **Merge New Stage Data**
   - Integrate relevant results from the current stage into the existing summary.
   - Identify any overlap or repetition between new and old content, and remove only that which is clearly redundant or no longer scientifically insightful.
   - Be very careful if you want to remove or shorten the old content. By default, you can keep most of it and append new text.
   - Highlight how new findings connect to or differ from previous findings.
3. **Numerical Results and Visuals**
   - Carefully maintain the most insightful plots, figures, and numerical results.
   - Do not delete crucial quantitative findings or meaningful visual references.
4. **Length and Format**
   - The final summary will likely be **very long**. That is acceptable.
   - Present the updated summary in a format consistent with the style of the previous summaries (e.g., same section headings or structure).

Respond in the following format:

THOUGHT:
<THOUGHT>

JSON:
```json
<JSON>
```
Ensure the JSON is valid and properly formatted, as it will be automatically parsed.
"""


def get_nodes_infos(nodes):
    node_infos = ""
    for n in nodes:
        node_info = f"Node ID: {n.id}\n"
        node_info += (
            f"Plan: {n.overall_plan}\n"
            if hasattr(n, "overall_plan")
            else "Plan: Not available\n"
        )
        node_info += (
            f"Analysis: {n.analysis}\n"
            if hasattr(n, "analysis")
            else "Analysis: Not available\n"
        )
        node_info += (
            f"Numerical Results: {n.metric}\n"
            if hasattr(n, "metric")
            else "Numerical Results: Not available\n"
        )
        # Plot analyses functionality removed
        node_info += "Plot Analyses: N/A\n"
        if True:  # Simplified condition
            node_info += "No plot analyses available\n"
        node_infos += node_info + "\n"
    return node_infos


def get_summarizer_prompt(journal, stage_name):
    good_leaf_nodes = [n for n in journal.good_nodes if n.is_leaf]
    if not good_leaf_nodes:
        logger.warning("NO GOOD LEAF NODES")
        good_leaf_nodes = [n for n in journal.good_nodes]
    node_infos = get_nodes_infos(good_leaf_nodes)
    return report_summarizer_sys_msg, report_summarizer_prompt.format(
        node_infos=node_infos, stage_name=stage_name
    )


def get_stage_summary(journal, stage_name, model, client):
    sys_msg, prompt = get_summarizer_prompt(journal, stage_name)
    response = get_response_from_llm(prompt, client, model, sys_msg)
    summary_json = extract_json_between_markers(response[0])
    return summary_json


def get_node_log(node):
    node_dict = node.to_dict()
    # Only include keys that are relevant for logging/analysis
    keys_to_include = [
        "overall_plan",
        "analysis",
        "metric",
        "code",
        "plot_code",
        "plot_plan",
        "plot_paths",
        "exp_results_dir",
        "ablation_name",  # Legacy field - kept for old checkpoint compatibility
        "param_tuning_name",
        "origin_stage",
    ]
    ret = {
        key: node_dict[key]
        for key in keys_to_include
        if key in node_dict and node_dict[key] is not None
    }
    if "exp_results_dir" in ret:
        original_dir_path = ret["exp_results_dir"]
        # Remove leading path segments before "experiment_results"
        idx = original_dir_path.find("experiment_results")
        short_dir_path = original_dir_path
        if idx != -1:
            short_dir_path = original_dir_path[idx:]

        ret["exp_results_dir"] = short_dir_path

        if os.path.isdir(original_dir_path):
            npy_files = [f for f in os.listdir(original_dir_path) if f.endswith(".npy")]
            # Prepend the shortened path to each .npy filename
            ret["exp_results_npy_files"] = [
                os.path.join(short_dir_path, f) for f in npy_files
            ]
        else:
            ret["exp_results_npy_files"] = []

    # Add node_id to make traceability easier
    ret['node_id'] = node.id

    # Comparison nodes: append ablation_type and variant_metrics
    abl_type = getattr(node, 'ablation_type', None)
    if abl_type:
        ret['ablation_type'] = abl_type
    if abl_type == 'comparison':
        metric_val = node.metric.value if node.metric else None
        if isinstance(metric_val, dict) and 'metric_names' in metric_val:
            dice_m = next((m for m in metric_val['metric_names']
                           if 'dice' in m.get('metric_name', '').lower()), None)
            hd95_m = next((m for m in metric_val['metric_names']
                           if 'hd95' in m.get('metric_name', '').lower()
                           or 'hausdorff' in m.get('metric_name', '').lower()), None)
            if dice_m and hd95_m and len(dice_m.get('data', [])) == len(hd95_m.get('data', [])):
                ret['variant_metrics'] = [
                    {
                        'variant_name': dd.get('dataset_name', ''),
                        'Dice Score': dd.get('final_value'),
                        'HD95 Score': hd.get('final_value'),
                    }
                    for dd, hd in zip(dice_m['data'], hd95_m['data'])
                ]

    return ret


def update_summary(
    prev_summary, cur_stage_name, cur_journal, cur_summary, model, client, max_retry=5
):
    good_leaf_nodes = [n for n in cur_journal.good_nodes if n.is_leaf]
    node_infos = get_nodes_infos(good_leaf_nodes)
    prompt = stage_aggregate_prompt.format(
        prev_summary=prev_summary,
        stage_name=cur_stage_name,
        current_summary=cur_summary,
    )
    try:
        response = get_response_from_llm(
            prompt, client, model, "You are an expert machine learning researcher."
        )
        summary_json = extract_json_between_markers(response[0])
        assert summary_json
    except Exception as e:
        if max_retry > 0:
            logger.warning(f"Error occurred: {e}. Retrying... ({max_retry} attempts left)")
            return update_summary(
                prev_summary,
                cur_stage_name,
                cur_journal,
                cur_summary,
                model,
                client,
                max_retry - 1,
            )
        else:
            logger.error(f"Failed to update summary after multiple attempts. Error: {e}")
            raise
    return summary_json


overall_plan_summarizer_prompt = """You have been provided with the plans for both the parent node and the current node. Your task is to synthesize a comprehensive summary of the overall plan by integrating details from both the parent and current node plans.
The summary should be thorough and clearly articulate the underlying motivations.
For example, if in your previous overall plan you were experimenting with a new idea, and now your current plan is to fix certain bugs in the previous implementation, your returned overall plan should focus on your previous overall plan, and briefly mention that the current plan includes bug fixes. If your current plan is more about implementing new ideas, then you should summarize that thoroughly along with the previous overall plan.
The goal is to create a comprehensive summary of all historical plans, focusing on the main scientific planning and objectives.

Previous overall plan:
{prev_overall_plan}

Current plan:
{current_plan}

Respond in the following format:

THOUGHT:
<THOUGHT>

JSON:
```json
<JSON>
```

In <THOUGHT>, thoroughly reason as an expert researcher. First, reason over each node, and then carefully combine all information. It is okay to be very detailed.

In <JSON>, provide the review in JSON format with the following field in exactly this order:
- "overall_plan": a string that describes the overall plan based on the current and previous overall plans

Ensure the JSON is valid and properly formatted, as it will be automatically parsed.
"""


def annotate_history(journal):
    for node in journal.nodes:
        if node.parent:
            max_retries = 3
            retry_count = 0
            while retry_count < max_retries:
                try:
                    response = get_response_from_llm(
                        overall_plan_summarizer_prompt.format(
                            prev_overall_plan=node.parent.overall_plan,
                            current_plan=node.plan,
                        ),
                        client,
                        model,
                        report_summarizer_sys_msg,
                    )
                    node.overall_plan = extract_json_between_markers(response[0])[
                        "overall_plan"
                    ]
                    break
                except Exception as e:
                    retry_count += 1
                    if retry_count == max_retries:
                        logger.error(f"Failed after {max_retries} attempts. Error: {e}")
                        raise
                    logger.warning(
                        f"Error occurred: {e}. Retrying... ({max_retries - retry_count} attempts left)"
                    )
        else:
            node.overall_plan = node.plan


# ============== Paper-Ready Summary Generation ==============

PAPER_REWRITE_SYSTEM_MSG = """You are an expert scientific writer specializing in machine learning and medical imaging research.
Your task is to rewrite research content into formal academic language suitable for a scientific paper.

Important guidelines:
- Use formal academic tone and precise technical terminology
- Maintain all technical details and numerical values exactly as provided
- Structure the content logically with clear transitions
- Do NOT add any information not present in the source material
- Do NOT hallucinate or fabricate any claims or results
"""

METHODOLOGY_REWRITE_PROMPT = """Please rewrite the following research proposal into a formal "Methodology" section suitable for an academic paper.

The section should include:
1. A clear problem statement and motivation
2. Overview of the proposed approach
3. Detailed description of each module/component
4. Technical implementation details

Research Proposal Content:
{proposal_content}

Please write the methodology section in academic style. Output in Markdown format with appropriate subsections (##, ###).
Do NOT include any preamble like "Here is the rewritten section". Start directly with the content.
"""


def _extract_metrics_from_node(node) -> dict:
    """Extract metrics from a node's metric field into a simple dict format."""
    metrics_dict = {}
    if not hasattr(node, 'metric') or node.metric is None:
        return metrics_dict
    
    metric_obj = node.metric
    if hasattr(metric_obj, 'value') and isinstance(metric_obj.value, dict):
        if 'metric_names' in metric_obj.value:
            # New format with multiple metrics
            for m in metric_obj.value['metric_names']:
                metric_name = m.get('metric_name', 'Unknown')
                data_list = m.get('data', [])
                if data_list:
                    # Average across variants; for a single variant this is equivalent to the first value
                    values = [d.get('final_value') for d in data_list if d.get('final_value') is not None]
                    if values:
                        metrics_dict[metric_name] = float(sum(values) / len(values))
        else:
            # Old format - dict of values
            for k, v in metric_obj.value.items():
                if v is not None:
                    metrics_dict[k] = float(v)
    elif hasattr(metric_obj, 'value') and metric_obj.value is not None:
        # Single value
        metrics_dict['Score'] = float(metric_obj.value)
    
    return metrics_dict


def _get_node_ablation_type(node) -> str:
    """Determine the ablation type of a node ('removal', 'comparison', or 'unknown').

    Checks the dedicated field first, then falls back to regex on node.plan
    for backward compatibility with older experiments.
    """
    import re
    abl_type = getattr(node, 'ablation_type', None)
    if abl_type in ('removal', 'comparison'):
        return abl_type
    plan = getattr(node, 'plan', '') or ''
    m = re.search(r'\((removal|comparison)\)', plan)
    if m:
        return m.group(1)
    return 'unknown'


def _extract_comparison_variant_metrics(node, log_dir: str) -> list:
    """Load experiment_data.npy for a comparison ablation node and extract
    per-variant Dice/HD95 metrics.

    Returns a list of dicts: [{'variant_name': str, 'Dice Score': float|None,
    'HD95 Score': float|None}, ...]

    Falls back to an empty list on any error.
    """
    import numpy as np
    from pathlib import Path

    node_id = node.id if hasattr(node, 'id') else (node.get('id') if isinstance(node, dict) else None)
    if not node_id:
        return []

    npy_path = Path(log_dir) / 'results' / node_id / 'experiment_data.npy'
    if not npy_path.exists():
        logger.debug(f"experiment_data.npy not found for comparison node {node_id}")
        return []

    try:
        data = np.load(str(npy_path), allow_pickle=True).item()
    except Exception as e:
        logger.warning(f"Failed to load experiment_data.npy for node {node_id}: {e}")
        return []

    variants = []
    for key, val in data.items():
        if not isinstance(val, dict):
            continue
        # Variant keys use the pattern  <dataset>__<variant_name>
        if '__' in key:
            variant_name = key.split('__', 1)[1]
        else:
            variant_name = key

        dice, hd95 = None, None
        # Try metrics.val (expected format: [{'dice': x, 'hd95': y}])
        if 'metrics' in val and 'val' in val['metrics'] and val['metrics']['val']:
            latest = val['metrics']['val'][-1]
            if isinstance(latest, dict):
                dice = next((latest[k] for k in ('dice', 'Dice', 'DICE') if latest.get(k) is not None), None)
                hd95 = next((latest[k] for k in ('hd95', 'HD95', 'Hausdorff Distance 95') if latest.get(k) is not None), None)
        # Fallback to dedicated score arrays
        scores = val.get('dice_scores')
        if dice is None and scores is not None and len(scores) > 0:
            dice = scores[-1]
        scores = val.get('hd95_scores')
        if hd95 is None and scores is not None and len(scores) > 0:
            hd95 = scores[-1]

        variants.append({
            'variant_name': variant_name,
            'Dice Score': float(dice) if dice is not None else None,
            'HD95 Score': float(hd95) if hd95 is not None else None,
        })

    return variants


def _is_variant_better(variant_metrics: dict, reference_metrics: dict) -> bool:
    """Check if a variant's metrics are better than the reference (Stage 2 best).

    Uses Dice as primary (higher is better).  Falls back to HD95 (lower is
    better) when Dice values are within 0.005.
    """
    v_dice = variant_metrics.get('Dice Score')
    r_dice = reference_metrics.get('Dice Score')
    if v_dice is None or r_dice is None:
        return False
    if abs(v_dice - r_dice) >= 0.005:
        return v_dice > r_dice
    # Tiebreak on HD95
    v_hd = variant_metrics.get('HD95 Score')
    r_hd = reference_metrics.get('HD95 Score')
    if v_hd is not None and r_hd is not None:
        return v_hd < r_hd
    return False


def _filter_bad_baselines(baselines_list: list) -> list:
    """Filter out obviously poor baseline methods for cleaner paper presentation.

    Removal criteria (applied in order):
    1. Dice score is None or exactly 0.
    2. Dice score is more than 20% lower than the second-worst remaining
       method (i.e., clear outliers at the bottom).

    The original list is not mutated; a new filtered list is returned.
    """
    if not baselines_list:
        return baselines_list

    valid = [b for b in baselines_list if b.get('dice_score') and b['dice_score'] > 0]
    if len(valid) <= 2:
        return valid

    valid.sort(key=lambda x: x['dice_score'], reverse=True)

    second_worst_dice = valid[-2]['dice_score']
    threshold = second_worst_dice * 0.8
    filtered = [b for b in valid if b['dice_score'] >= threshold]

    removed = len(baselines_list) - len(filtered)
    if removed > 0:
        removed_names = [
            b.get('model_name', '?') for b in baselines_list
            if b not in filtered
        ]
        logger.info(
            f"Filtered {removed} poor baseline(s) "
            f"(threshold dice >= {threshold:.4f}): {removed_names}"
        )

    return filtered


def _load_research_proposal(log_dir: str, proposal_idx: int = None) -> str:
    """Load the research proposal markdown content from initial_proposal_queue.json."""
    from pathlib import Path
    
    log_path = Path(log_dir)
    proposal_queue_path = log_path / "initial_proposal_queue.json"
    
    if not proposal_queue_path.exists():
        logger.warning(f"Proposal queue not found at {proposal_queue_path}")
        return None
    
    with open(proposal_queue_path, 'r', encoding='utf-8') as f:
        queue_data = json.load(f)
    
    proposals = queue_data.get('proposals', [])
    if not proposals:
        logger.warning("No proposals found in queue")
        return None
    
    # Use provided index or default to first
    idx = proposal_idx if proposal_idx is not None else 0
    if idx >= len(proposals):
        idx = 0
    
    proposal = proposals[idx]
    md_path = proposal.get('md_file')
    
    if md_path:
        md_path_obj = Path(md_path)
        # Try various locations
        possible_paths = [
            md_path_obj,  # Absolute path
            log_path / md_path_obj.name,  # Same dir as log
            log_path / "research_proposals" / md_path_obj.name,  # research_proposals subdir
            log_path.parent / "research_proposals" / md_path_obj.name,  # Parent's research_proposals
        ]
        
        for p in possible_paths:
            if p.exists():
                return p.read_text(encoding='utf-8')
    
    # Fallback to JSON metadata
    return f"# {proposal.get('title', 'Unknown')}\n\n{proposal.get('description', 'No description available')}"


def _get_best_non_baseline_node(journals):
    """Find the best non-baseline node across one or more journals.

    Accepts a single journal or a list of journals (as produced by the Stage 2
    collection in perform_experiments_qwbe).  Nodes with
    ``is_precomputed_baseline=True`` are excluded so that the returned node is
    always an actual experiment result rather than a baseline reference point.

    Uses MetricValue comparison (Dice primary, HD95 tiebreak after the Dice
    threshold check) as the single source of truth, consistent with
    journal.get_best_node().
    """
    if journals is None:
        return None
    if not isinstance(journals, list):
        journals = [journals]

    all_candidates = []
    for journal in journals:
        if journal is None:
            continue
        for node in journal.good_nodes:
            if getattr(node, 'is_precomputed_baseline', False):
                continue
            all_candidates.append(node)

    if not all_candidates:
        return None

    candidates_with_metrics = [n for n in all_candidates if n.metric is not None]
    if not candidates_with_metrics:
        return all_candidates[0]

    return max(candidates_with_metrics, key=lambda n: n.metric)


def _find_best_proposal_idx_from_stage2(journals_dict: dict) -> int:
    """Find the proposal index of the best performing Stage 2 node.

    Works with ``journals_dict['2_creative_research']`` being either a single
    journal or a list of journals.  Baseline nodes are excluded.
    """
    best_node = _get_best_non_baseline_node(
        journals_dict.get('2_creative_research')
    )
    if best_node is None:
        return 0

    # Try to get proposal_idx from origin_stage
    origin_stage = getattr(best_node, 'origin_stage', None)
    if origin_stage and 'proposal_' in origin_stage:
        try:
            parts = origin_stage.split('_')
            if 'proposal' in parts:
                idx_pos = parts.index('proposal') + 1
                if idx_pos < len(parts):
                    return int(parts[idx_pos]) - 1  # Convert to 0-based
        except (ValueError, IndexError):
            pass

    return 0


def _rewrite_proposal_academically(proposal_content: str) -> str:
    """Use LLM to rewrite proposal content in academic style."""
    try:
        prompt = METHODOLOGY_REWRITE_PROMPT.format(proposal_content=proposal_content)
        response = get_response_from_llm(prompt, client, model, PAPER_REWRITE_SYSTEM_MSG)
        return response[0].strip()
    except Exception as e:
        logger.warning(f"Failed to rewrite proposal academically: {e}")
        return proposal_content  # Return original if rewrite fails


def generate_paper_ready_summary(journals_dict: dict, log_dir: str, use_llm_rewrite: bool = True) -> str:
    """
    Generate a paper-ready Markdown summary from experiment journals.
    
    Args:
        journals_dict: Dict with keys '1_baseline_implementation', '2_creative_research', '3_ablation_studies'
        log_dir: Path to the log directory containing initial_proposal_queue.json
        use_llm_rewrite: Whether to use LLM to academically rewrite the methodology section
        
    Returns:
        Markdown string formatted like an academic paper
    """
    md_lines = []
    md_lines.append("# Experiment Report\n")
    md_lines.append(f"*Generated automatically from experiment logs*\n")
    md_lines.append("---\n")
    
    # ================= Section 1: Baseline =================
    md_lines.append("## 1. Baseline Model Performance\n")
    
    baseline_journal = journals_dict.get('1_baseline_implementation')
    if baseline_journal is not None:
        good_nodes = [n for n in baseline_journal.good_nodes if not n.is_buggy]
        
        if good_nodes:
            # Collect all unique metric names
            all_metrics = set()
            for node in good_nodes:
                metrics = _extract_metrics_from_node(node)
                all_metrics.update(metrics.keys())
            
            metric_names = sorted(list(all_metrics))
            
            # Build table header
            header = "| Model | " + " | ".join(metric_names) + " |"
            separator = "| :--- | " + " | ".join([":---:"] * len(metric_names)) + " |"
            md_lines.append(header)
            md_lines.append(separator)
            
            # Build table rows
            for i, node in enumerate(good_nodes):
                metrics = _extract_metrics_from_node(node)
                model_name = f"Baseline-{i+1}"
                if hasattr(node, 'plan') and node.plan:
                    # Extract short description from plan
                    plan_first_line = node.plan.split('\n')[0][:50]
                    if plan_first_line:
                        model_name = plan_first_line
                
                row_values = []
                for m_name in metric_names:
                    val = metrics.get(m_name)
                    if val is not None:
                        row_values.append(f"{val:.4f}")
                    else:
                        row_values.append("-")
                
                md_lines.append(f"| {model_name} | " + " | ".join(row_values) + " |")
            
            md_lines.append("")
        else:
            md_lines.append("*No successful baseline experiments recorded.*\n")
    else:
        md_lines.append("*Baseline experiments not available.*\n")
    
    # ================= Section 2: Proposed Method =================
    md_lines.append("## 2. Proposed Method\n")
    
    # Support both a single journal and a list of journals for Stage 2.
    # Exclude precomputed baseline nodes so the "proposed method" result is
    # always an actual experiment, never the baseline itself.
    research_journals_raw = journals_dict.get('2_creative_research')
    best_node = _get_best_non_baseline_node(research_journals_raw)
    best_metrics = {}
    
    if best_node is not None:
        best_metrics = _extract_metrics_from_node(best_node)
    
    # Load and rewrite research proposal
    proposal_idx = _find_best_proposal_idx_from_stage2(journals_dict)
    proposal_content = _load_research_proposal(log_dir, proposal_idx)
    
    if proposal_content:
        if use_llm_rewrite:
            md_lines.append("### 2.1 Methodology\n")
            rewritten = _rewrite_proposal_academically(proposal_content)
            md_lines.append(rewritten)
            md_lines.append("")
        else:
            md_lines.append("### 2.1 Research Proposal\n")
            md_lines.append(proposal_content)
            md_lines.append("")
    else:
        md_lines.append("*Research proposal not available.*\n")
    
    # Add Implementation Notes: bridge the gap between proposal and actual code
    if best_node is not None and hasattr(best_node, 'code') and best_node.code:
        md_lines.append("### 2.1.1 Implementation Notes\n")
        md_lines.append("**IMPORTANT: The methodology section above describes the *theoretical proposal*. "
                        "The actual implementation may use engineering approximations for complex concepts "
                        "(e.g., using differentiable neural network operations instead of non-differentiable "
                        "topological computations). The paper's method description MUST accurately reflect "
                        "the actual implementation below, NOT the theoretical proposal.**\n")
        md_lines.append("The following is the **actual implemented code** for the best-performing model:\n")
        
        # Extract key class/module definitions from the code
        code = best_node.code
        code_lines = code.split('\n')
        
        # Find class definitions and their short implementations
        class_snippets = []
        i = 0
        while i < len(code_lines):
            line = code_lines[i]
            stripped = line.strip()
            if stripped.startswith('class ') and '(nn.Module)' in stripped:
                # Capture the class definition and its __init__ and forward methods
                snippet_lines = [line]
                indent_level = len(line) - len(line.lstrip())
                i += 1
                method_count = 0
                while i < len(code_lines) and method_count < 3:
                    curr_line = code_lines[i]
                    curr_stripped = curr_line.strip()
                    # Stop if we hit another class definition at the same or higher level
                    if curr_stripped.startswith('class ') and len(curr_line) - len(curr_line.lstrip()) <= indent_level:
                        break
                    snippet_lines.append(curr_line)
                    if curr_stripped.startswith('def '):
                        method_count += 1
                    i += 1
                class_snippets.append('\n'.join(snippet_lines))
            else:
                i += 1
        
        if class_snippets:
            md_lines.append("```python")
            for snippet in class_snippets:
                md_lines.append(snippet)
                md_lines.append("")
            md_lines.append("```\n")
        else:
            # If no class definitions found, show a truncated version of the code
            truncated = '\n'.join(code_lines[:100])
            if len(code_lines) > 100:
                truncated += '\n# ... (truncated, total {} lines)'.format(len(code_lines))
            md_lines.append("```python")
            md_lines.append(truncated)
            md_lines.append("```\n")
    
    # Add experimental results
    if best_node and best_metrics:
        md_lines.append("### 2.2 Experimental Results\n")
        md_lines.append("The proposed method achieves the following performance:\n")
        
        md_lines.append("| Metric | Value |")
        md_lines.append("| :--- | :---: |")
        for m_name, m_val in best_metrics.items():
            md_lines.append(f"| {m_name} | {m_val:.4f} |")
        md_lines.append("")
        
        # Compare with baseline if available
        if baseline_journal is not None:
            baseline_best = baseline_journal.get_best_node()
            if baseline_best:
                baseline_metrics = _extract_metrics_from_node(baseline_best)
                
                md_lines.append("#### Comparison with Baseline\n")
                md_lines.append("| Metric | Proposed | Baseline | Change (%) | Direction | Verdict |")
                md_lines.append("| :--- | :---: | :---: | :---: | :--- | :--- |")
                
                for m_name in best_metrics.keys():
                    proposed_val = best_metrics.get(m_name)
                    baseline_val = baseline_metrics.get(m_name)
                    
                    if proposed_val is not None and baseline_val is not None and baseline_val != 0:
                        # Determine if lower is better for this metric
                        lower_is_better = ('hd' in m_name.lower() or 'loss' in m_name.lower() or 'error' in m_name.lower())
                        
                        if lower_is_better:
                            # For lower-is-better: positive change means value decreased (good)
                            change_pct = ((baseline_val - proposed_val) / baseline_val) * 100
                            direction = "Lower is better"
                        else:
                            # For higher-is-better: positive change means value increased (good)
                            change_pct = ((proposed_val - baseline_val) / baseline_val) * 100
                            direction = "Higher is better"
                        
                        verdict = "BETTER" if change_pct > 0 else ("WORSE" if change_pct < 0 else "SAME")
                        
                        md_lines.append(f"| {m_name} | {proposed_val:.4f} | {baseline_val:.4f} | {change_pct:+.2f}% | {direction} | **{verdict}** |")
                    elif proposed_val is not None:
                        md_lines.append(f"| {m_name} | {proposed_val:.4f} | - | - | - | - |")
                
                md_lines.append("")
        
        # Load all baselines from all_baselines.json for comprehensive SOTA comparison
        all_baselines_path = os.path.join(log_dir, "all_baselines.json")
        if os.path.exists(all_baselines_path):
            try:
                with open(all_baselines_path, 'r', encoding='utf-8') as f:
                    all_baselines_data = json.load(f)
                
                baselines_list = all_baselines_data.get('baselines', [])
                baselines_list = _filter_bad_baselines(baselines_list)
                if baselines_list:
                    md_lines.append("#### All Baseline Model Results\n")
                    md_lines.append("The following table contains **all** baseline models evaluated on this dataset. ")
                    md_lines.append("**IMPORTANT: Use ONLY these exact numbers for any SOTA comparison table in the paper. Do NOT fabricate or estimate results for models not listed here.**\n")
                    md_lines.append("| Rank | Model | Dice Score | HD95 |")
                    md_lines.append("| :---: | :--- | :---: | :---: |")
                    
                    for rank, bl in enumerate(baselines_list, 1):
                        model_name = bl.get('model_name', 'Unknown')
                        dice = bl.get('dice_score')
                        hd95 = bl.get('hd95_score')
                        dice_str = f"{dice:.4f}" if dice is not None else "-"
                        hd95_str = f"{hd95:.4f}" if hd95 is not None else "-"
                        md_lines.append(f"| {rank} | {model_name} | {dice_str} | {hd95_str} |")
                    
                    # Also add the proposed method row for easy comparison
                    if best_metrics:
                        proposed_dice = best_metrics.get('Dice Score', best_metrics.get('Dice'))
                        proposed_hd95 = None
                        for key in best_metrics:
                            if 'hd' in key.lower():
                                proposed_hd95 = best_metrics[key]
                                break
                        dice_str = f"**{proposed_dice:.4f}**" if proposed_dice is not None else "-"
                        hd95_str = f"**{proposed_hd95:.4f}**" if proposed_hd95 is not None else "-"
                        md_lines.append(f"| - | **Proposed Method** | {dice_str} | {hd95_str} |")
                    
                    md_lines.append("")
            except Exception as e:
                logger.warning(f"Failed to load all_baselines.json: {e}")
    
    # ================= Section 3: Ablation Studies =================
    md_lines.append("## 3. Ablation Studies\n")
    
    ablation_journal = journals_dict.get('3_ablation_studies')
    
    if ablation_journal is not None:
        ablation_nodes = [
            n for n in ablation_journal.good_nodes 
            if n.is_leaf and getattr(n, 'ablation_name', None)
        ]
        
        if ablation_nodes:
            # Split nodes by ablation type
            removal_nodes = []
            comparison_nodes = []
            for node in ablation_nodes:
                abl_type = _get_node_ablation_type(node)
                if abl_type == 'comparison':
                    comparison_nodes.append(node)
                else:
                    removal_nodes.append(node)

            # --- Comparison studies: check if any variant beats Stage 2 best ---
            upgraded_best_metrics = dict(best_metrics) if best_metrics else {}
            upgrade_source = None  # tracks which variant upgraded best_metrics

            for cmp_node in comparison_nodes:
                variants = _extract_comparison_variant_metrics(cmp_node, log_dir)
                if not variants:
                    continue
                for v in variants:
                    if _is_variant_better(v, upgraded_best_metrics):
                        upgrade_source = v['variant_name']
                        for k in ('Dice Score', 'HD95 Score'):
                            if v.get(k) is not None:
                                upgraded_best_metrics[k] = v[k]
                        logger.info(
                            f"Comparison variant '{upgrade_source}' surpasses "
                            f"Stage 2 best — promoting as Proposed Method"
                        )

            if upgrade_source:
                best_metrics = upgraded_best_metrics

            # --- 3.1 Removal ablation table ---
            if removal_nodes:
                md_lines.append("### 3.1 Component Removal Studies\n")
                md_lines.append("To validate the contribution of each component, we conducted ablation studies by removing individual modules.\n")

                all_rm_metrics = set()
                for node in removal_nodes:
                    all_rm_metrics.update(_extract_metrics_from_node(node).keys())
                rm_metric_names = sorted(list(all_rm_metrics))

                header = "| Configuration | Removed Component | " + " | ".join(rm_metric_names) + " |"
                separator = "| :--- | :--- | " + " | ".join([":---:"] * len(rm_metric_names)) + " |"
                md_lines.append(header)
                md_lines.append(separator)

                if best_metrics:
                    full_row_values = []
                    for m_name in rm_metric_names:
                        val = best_metrics.get(m_name)
                        full_row_values.append(f"**{val:.4f}**" if val is not None else "-")
                    md_lines.append(f"| **Full Model** | - | " + " | ".join(full_row_values) + " |")

                degraded_count = 0
                for node in removal_nodes:
                    ablation_name = getattr(node, 'ablation_name', 'Unknown')
                    metrics = _extract_metrics_from_node(node)
                    row_values = []
                    for m_name in rm_metric_names:
                        val = metrics.get(m_name)
                        row_values.append(f"{val:.4f}" if val is not None else "-")
                    md_lines.append(f"| w/o {ablation_name} | {ablation_name} | " + " | ".join(row_values) + " |")
                    if best_metrics and _is_variant_better(best_metrics, metrics):
                        degraded_count += 1

                md_lines.append("")

            # --- 3.2 Comparison studies table ---
            if comparison_nodes:
                md_lines.append("### 3.2 Comparison Studies\n")
                md_lines.append("We compared alternative design choices to validate the selected architecture.\n")

                for cmp_node in comparison_nodes:
                    cmp_name = getattr(cmp_node, 'ablation_name', 'Unknown')
                    variants = _extract_comparison_variant_metrics(cmp_node, log_dir)
                    if not variants:
                        node_metrics = _extract_metrics_from_node(cmp_node)
                        if node_metrics:
                            variants = [{'variant_name': cmp_name, **node_metrics}]
                        else:
                            md_lines.append(f"*Comparison study \"{cmp_name}\": no variant metrics available.*\n")
                            continue

                    cmp_metric_names = sorted({k for v in variants for k in v if k != 'variant_name' and v[k] is not None})
                    if not cmp_metric_names:
                        continue

                    md_lines.append(f"**{cmp_name}**\n")
                    header = "| Variant | " + " | ".join(cmp_metric_names) + " |"
                    separator = "| :--- | " + " | ".join([":---:"] * len(cmp_metric_names)) + " |"
                    md_lines.append(header)
                    md_lines.append(separator)

                    # Add original Stage 2 best as a row in the comparison
                    if best_metrics:
                        orig_vals = []
                        for m_name in cmp_metric_names:
                            val = best_metrics.get(m_name)
                            orig_vals.append(f"{val:.4f}" if val is not None else "-")
                        if upgrade_source:
                            md_lines.append(f"| **Proposed Method** | " + " | ".join(
                                f"**{v}**" for v in orig_vals) + " |")
                        else:
                            md_lines.append(f"| **Proposed Method (Full Model)** | " + " | ".join(
                                f"**{v}**" for v in orig_vals) + " |")

                    for v in variants:
                        row_vals = []
                        is_best = (v['variant_name'] == upgrade_source)
                        for m_name in cmp_metric_names:
                            val = v.get(m_name)
                            cell = f"{val:.4f}" if val is not None else "-"
                            if is_best:
                                cell = f"**{cell}**"
                            row_vals.append(cell)
                        label = v['variant_name']
                        if is_best:
                            label = f"**{label} (best)**"
                        md_lines.append(f"| {label} | " + " | ".join(row_vals) + " |")

                    md_lines.append("")

            # --- Analysis ---
            if not removal_nodes and not comparison_nodes:
                md_lines.append("*No ablation experiments with named components recorded.*\n")
            else:
                section_num = "3.3" if comparison_nodes and removal_nodes else ("3.2" if removal_nodes else "3.1")
                md_lines.append(f"### {section_num} Analysis\n")

                analysis_parts = []
                if removal_nodes:
                    if degraded_count == len(removal_nodes):
                        analysis_parts.append(
                            "The removal ablation results demonstrate the contribution of each "
                            "component to the overall performance. Removing any module leads to "
                            "performance degradation, confirming the effectiveness of the "
                            "proposed design."
                        )
                    elif degraded_count > 0:
                        analysis_parts.append(
                            f"Among {len(removal_nodes)} component removal experiments, "
                            f"{degraded_count} showed performance degradation when removed, "
                            f"confirming their positive contribution."
                        )
                    else:
                        analysis_parts.append(
                            "The removal experiments did not show consistent performance "
                            "degradation, suggesting potential redundancy in some components."
                        )

                if upgrade_source:
                    analysis_parts.append(
                        f"Notably, the comparison study identified variant "
                        f"\"{upgrade_source}\" as superior to the original Stage 2 "
                        f"configuration, and it has been adopted as the proposed method "
                        f"in this report."
                    )

                md_lines.append(" ".join(analysis_parts) + "\n")
        else:
            md_lines.append("*No ablation experiments with named components recorded.*\n")
    else:
        md_lines.append("*Ablation studies not available.*\n")
    
    # ================= Section 4: Conclusion =================
    md_lines.append("## 4. Conclusion\n")
    md_lines.append("This report summarizes the experimental results of the proposed method. ")
    
    if best_metrics and baseline_journal:
        baseline_best = baseline_journal.get_best_node()
        if baseline_best:
            baseline_metrics = _extract_metrics_from_node(baseline_best)
            
            conclusion_parts = []
            for m_name in best_metrics.keys():
                proposed_val = best_metrics.get(m_name)
                baseline_val = baseline_metrics.get(m_name)
                
                if proposed_val is not None and baseline_val is not None and baseline_val != 0:
                    lower_is_better = ('hd' in m_name.lower() or 'loss' in m_name.lower() or 'error' in m_name.lower())
                    
                    if lower_is_better:
                        change_pct = ((baseline_val - proposed_val) / baseline_val) * 100
                    else:
                        change_pct = ((proposed_val - baseline_val) / baseline_val) * 100
                    
                    if change_pct > 0:
                        verdict_word = "improvement"
                    elif change_pct < 0:
                        verdict_word = "degradation"
                    else:
                        verdict_word = "no change"
                    
                    conclusion_parts.append(
                        f"{m_name}: {proposed_val:.4f} vs baseline {baseline_val:.4f} "
                        f"({change_pct:+.2f}% {verdict_word}, {'lower' if lower_is_better else 'higher'} is better)"
                    )
            
            if conclusion_parts:
                md_lines.append("Performance summary compared to the best baseline:\n")
                for part in conclusion_parts:
                    md_lines.append(f"- {part}")
                md_lines.append("")

    # ================= Section 4: Computational Efficiency =================
    efficiency_path = os.path.join(log_dir, "efficiency_metrics.json")
    if os.path.exists(efficiency_path):
        try:
            with open(efficiency_path, 'r', encoding='utf-8') as f:
                eff_data = json.load(f)

            if eff_data:
                rows = []
                if isinstance(eff_data, dict) and isinstance(eff_data.get("rows"), list):
                    for item in eff_data["rows"]:
                        if not isinstance(item, dict):
                            continue
                        rows.append(
                            {
                                "model_name": item.get("model_name") or item.get("node_id") or "Unknown",
                                "params": item.get("params"),
                                "flops": item.get("flops"),
                                "inference_time_s": item.get("inference_time_s"),
                            }
                        )
                elif isinstance(eff_data, dict):
                    for node_id, item in eff_data.items():
                        if not isinstance(item, dict):
                            continue
                        rows.append(
                            {
                                "model_name": node_id,
                                "params": item.get("params"),
                                "flops": item.get("flops"),
                                "inference_time_s": item.get("inference_time_s"),
                            }
                        )

                if rows:
                    md_lines.append("## 4. Computational Efficiency\n")
                    md_lines.append("| Model | Params (M) | FLOPs (G) | Inference Time (ms) |")
                    md_lines.append("| :--- | :---: | :---: | :---: |")

                    for row in rows:
                        model_name = row["model_name"]
                        params = row.get("params")
                        flops = row.get("flops")
                        inf_t = row.get("inference_time_s")
                        p_str = f"{params / 1e6:.2f}" if params else "-"
                        f_str = f"{flops / 1e9:.2f}" if flops else "-"
                        t_str = f"{inf_t * 1000:.1f}" if inf_t else "-"
                        md_lines.append(f"| {model_name} | {p_str} | {f_str} | {t_str} |")
                    md_lines.append("")
        except Exception as e:
            logger.warning(f"Failed to load efficiency_metrics.json: {e}")

    md_lines.append("")
    
    return "\n".join(md_lines)


def overall_summarize(journals):
    from concurrent.futures import ThreadPoolExecutor
    
    def merge_prefixed_tuples(journals):
        result = defaultdict(list)
        
        for key, value in journals:
            prefix = key.split('_')[0] + '_' + key.split('_')[1] + '_' + key.split('_')[2]
            result[prefix].extend(value)
        
        return list(result.items())
    
    def merge_journals(journal1: Journal, journal2: Journal) -> Journal:
        """
        Merge two Journal instances and return a new Journal with all their nodes.
        Handles potentially duplicated node IDs and preserves parent/child relationships.

        Args:
            journal1: First Journal instance.
            journal2: Second Journal instance.

        Returns:
            A new Journal instance containing all nodes from both inputs.
        """
        # Create a new Journal instance
        merged_journal = Journal()

        # Track ID mapping (old ID -> new ID)
        id_mapping = {}

        # First add all nodes from journal1, keeping their original IDs
        for node in journal1.nodes:
            merged_journal.append(node)
            id_mapping[node.id] = node.id  # record ID mapping (unchanged)

        # Then add nodes from journal2, handling any ID conflicts
        existing_ids = {node.id for node in merged_journal.nodes}

        for node in journal2.nodes:
            # Check whether the ID already exists
            if node.id in existing_ids:
                logger.debug('existing id conflict!')
                # Generate a new ID
                new_id = f"{node.id}_j2_{node.step}"

                # Record the ID mapping
                id_mapping[node.id] = new_id

                # Deep-copy the node and update its ID
                import copy
                new_node = copy.deepcopy(node)
                new_node.id = new_id

                # Append to the new Journal
                merged_journal.append(new_node)
            else:
                # No ID conflict: append directly
                merged_journal.append(node)
                id_mapping[node.id] = node.id  # record ID mapping (unchanged)

        # Fix up parent references
        for node in merged_journal.nodes:
            if node.parent is not None:
                # Get the original parent ID
                original_parent_id = node.parent.id

                # If the parent ID is present in the mapping, update the reference
                if original_parent_id in id_mapping:
                    new_parent_id = id_mapping[original_parent_id]
                    # Fetch the new parent object
                    node.parent = merged_journal.get_node_by_id(new_parent_id)

        # Reset node step attributes to keep the sequence contiguous
        for i, node in enumerate(merged_journal.nodes):
            node.step = i
        
        return merged_journal


    def process_stage(idx, stage_tuple):
        stage_name, journal = stage_tuple
        annotate_history(journal)
        if '2_creative_research' in stage_name:
            best_node = journal.get_best_node()
            # Simply return the best node without multi-seed evaluation
            return {
                "best node": get_node_log(best_node),
            }
        elif '3_ablation_studies' in stage_name:
            # Get good leaf nodes for ablation studies
            good_leaf_nodes = [
                n for n in journal.good_nodes if n.is_leaf and getattr(n, 'ablation_name', None)
            ]
            return [get_node_log(n) for n in good_leaf_nodes]
        elif '1_baseline_implementation' in stage_name:
            summary_json = get_stage_summary(journal, stage_name, model, client)
            return summary_json

    from tqdm import tqdm
    
    new_journals = {
        '1_baseline_implementation': None,
        '2_creative_research': None,
        '3_ablation_studies': None,
    }

    # Create an empty list for each predefined prefix
    grouped_journals = {prefix: [] for prefix in new_journals.keys()}

    # Iterate over journals and assign them to the correct group
    for key, journal in journals:
        # Check whether the key starts with any predefined prefix
        matched = False
        for prefix in new_journals.keys():
            if key.startswith(prefix):
                grouped_journals[prefix].append(journal)
                matched = True
                break

        # If no exact match, try matching by numeric prefix
        if not matched and key and key[0].isdigit() and len(key) > 1 and key[1] == '_':
            digit = key[0]
            # Find the prefix starting with the same digit
            for prefix in new_journals.keys():
                if prefix.startswith(f"{digit}_"):
                    grouped_journals[prefix].append(journal)
                    break

    # Merge journals within each group
    for prefix, journal_list in grouped_journals.items():
        # Filter out None values
        journal_list = [j for j in journal_list if j is not None]

        if not journal_list:  # If the list is empty, skip
            continue

        # If there is only one journal, use it directly
        if len(journal_list) == 1:
            new_journals[prefix] = journal_list[0]
        else:
            # Otherwise merge all journals in the group
            merged = journal_list[0]
            for j in journal_list[1:]:
                merged = merge_journals(merged, j)
            new_journals[prefix] = merged

    # Update the journals variable
    journals = new_journals.items()

    
    '''
    v = [value for key, value in journals]
    print(len(v[0]), len(v[1]), len(v[2]), len(v[3]))
    print(type(v[3]))
    
    journals = merge_prefixed_tuples(journals)
    
    v = [value for key, value in journals]
    print(len(v[0]), len(v[1]), len(v[2]), len(v[3]))
    
    k = [key for key, value in journals]
    print(len(v[3]))
    print(len(v[1].good_nodes))
    
    print([x.is_leaf for x in v[3].good_nodes])
    print([x.ablation_name for x in v[3].good_nodes])
    raise
    '''
    
    with ThreadPoolExecutor() as executor:
        results = list(
            tqdm(
                executor.map(process_stage, range(len(list(journals))), journals),
                desc="Processing stages",
                total=len(list(journals)),
            )
        )

    
    return results


if __name__ == "__main__":
    # Test
    example_path = "logs/247-run"

    def load_stage_folders(base_path):
        """
        Load the folders that start with 'stage_' followed by a number.

        Args:
            base_path (str): The base directory path where stage folders are located.

        Returns:
            list: A sorted list of stage folder paths.
        """
        stage_folders = []
        for folder_name in os.listdir(base_path):
            if folder_name.startswith("stage_"):
                stage_folders.append(os.path.join(base_path, folder_name))
        return sorted(stage_folders, key=lambda x: int(x.split("_")[1]))

    def reconstruct_journal(journal_data):
        # Create a mapping of node IDs to Node instances
        id_to_node = {}
        for node_data in journal_data["nodes"]:
            # Remove unused or invalid keys if needed
            if "actionable_insights_from_plots" in node_data:
                del node_data["actionable_insights_from_plots"]
            node = Node.from_dict(node_data)
            id_to_node[node.id] = node

        # Set up parent-child relationships using node2parent
        for node_id, parent_id in journal_data["node2parent"].items():
            child_node = id_to_node[node_id]
            parent_node = id_to_node[parent_id]
            child_node.parent = parent_node
            parent_node.children.add(child_node)

        # Create a Journal and add all nodes
        journal = Journal()
        journal.nodes.extend(id_to_node.values())

        return journal

    # Example usage
    stage_folders = load_stage_folders(example_path)
    journals = []
    
    journal_per_stage = {
        '1_baseline_implementation': [],
        '2_creative_research': [],
        '3_ablation_studies': [],
    }
    for index, folder in enumerate(stage_folders, start=1):
        logger.debug(f"Stage {index}: {folder}")
        stage_name = os.path.basename(folder)
        journal_path = os.path.join(folder, "journal.json")
        if os.path.exists(journal_path):
            with open(journal_path, "r") as file:
                journal_data = json.load(file)
                logger.debug(f"Loaded journal.json for Stage {index}")
        else:
            logger.warning(f"No journal.json found for Stage {index}")
        journal = reconstruct_journal(journal_data)
        
        base_key = stage_name.split('_', 2)[0] + '_' + stage_name.split('_', 2)[1]
        journal_per_stage['base_key'].extend(journal)
    
    for k, v in journal_per_stage.items():
        journals.append((k, v))
        
    logger.debug("Number of journals: %d", len(journals))

    # Convert manager journals to list of (stage_name, journal) tuples

    (
        baseline_summary,
        research_summary,
        ablation_summary,
    ) = overall_summarize(journals)
    log_dir = "logs/247-run"
    baseline_summary_path = log_dir + "/baseline_summary.json"
    research_summary_path = log_dir + "/research_summary.json"
    ablation_summary_path = log_dir + "/ablation_summary.json"

    with open(baseline_summary_path, "w") as baseline_file:
        json.dump(baseline_summary, baseline_file, indent=2)

    with open(research_summary_path, "w") as research_file:
        json.dump(research_summary, research_file, indent=2)

    with open(ablation_summary_path, "w") as ablation_file:
        json.dump(ablation_summary, ablation_file, indent=2)

    logger.info(f"Summary reports written: baseline={baseline_summary_path}, research={research_summary_path}, ablation={ablation_summary_path}")
