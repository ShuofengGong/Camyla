"""Export journal to HTML visualization of tree + code."""

import json
import textwrap
from pathlib import Path

import numpy as np
from igraph import Graph
from ..journal import Journal

from rich import print


def get_edges(journal: Journal):
    """Yield edges as (parent_index, child_index) in journal order.
    Uses node UUID (.id) for matching instead of Python object identity,
    which is robust against deep copies and pickle deserialization."""
    node_uuid_to_idx = {n.id: i for i, n in enumerate(journal)}
    for i, node in enumerate(journal):
        parent_uuid = None
        if node.parent is not None and not isinstance(node.parent, str):
            parent_uuid = node.parent.id
        elif node.parent_id is not None:
            parent_uuid = node.parent_id
        if parent_uuid is not None:
            j = node_uuid_to_idx.get(parent_uuid)
            if j is not None:
                yield (j, i)


def generate_layout(n_nodes, edges, layout_type="rt"):
    """Generate visual layout of graph"""
    layout = Graph(
        n_nodes,
        edges=edges,
        directed=True,
    ).layout(layout_type)
    y_max = max(layout[k][1] for k in range(n_nodes))
    layout_coords = []
    for n in range(n_nodes):
        layout_coords.append((layout[n][0], 2 * y_max - layout[n][1]))
    return np.array(layout_coords)


def normalize_layout(layout: np.ndarray):
    """Normalize layout to [0, 1]"""
    layout = (layout - layout.min(axis=0)) / (layout.max(axis=0) - layout.min(axis=0))
    layout[:, 1] = 1 - layout[:, 1]
    layout[:, 1] = np.nan_to_num(layout[:, 1], nan=0)
    layout[:, 0] = np.nan_to_num(layout[:, 0], nan=0.5)
    return layout


def get_completed_stages(log_dir):
    """
    Determine completed stages by checking for the existence of stage directories
    that contain evidence of completion (tree_data.json, tree_plot.html, or journal.json).

    Returns:
        list: A list of stage names (e.g., ["Stage_1", "Stage_2"])
    """
    completed_stages = []

    # Check for each stage (1-4)
    for stage_num in range(1, 5):
        prefix = f"stage_{stage_num}"

        # Find all directories that match this stage number
        matching_dirs = [
            d for d in log_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)
        ]

        # Check if any of these directories have completion evidence
        for stage_dir in matching_dirs:
            has_tree_data = (stage_dir / "tree_data.json").exists()
            has_tree_plot = (stage_dir / "tree_plot.html").exists()
            has_journal = (stage_dir / "journal.json").exists()

            if has_tree_data or has_tree_plot or has_journal:
                # Found evidence this stage was completed
                completed_stages.append(f"Stage_{stage_num}")
                break  # No need to check other directories for this stage

    return completed_stages


def get_stage2_proposals(log_dir):
    """
    Scan log directory and find all Stage 2 proposal directories.
    
    Returns:
        list: A list of dicts containing proposal metadata:
            - dir_name: Directory name (e.g., 'stage_2_creative_research_1_proposal_1')
            - display_name: Human-readable name (e.g., 'Proposal 1')
            - has_data: Whether tree_data.json exists
            - best_dice: Best Dice score if available
            - status: 'completed', 'in_progress', or 'not_started'
    """
    proposals = []
    
    # Find all stage_2 directories
    stage2_dirs = sorted([
        d for d in log_dir.iterdir()
        if d.is_dir() and d.name.startswith("stage_2_")
    ])
    
    for stage_dir in stage2_dirs:
        # Extract proposal number from directory name
        dir_name = stage_dir.name
        proposal_num = None
        if "proposal_" in dir_name:
            try:
                proposal_num = int(dir_name.split("proposal_")[-1])
            except ValueError:
                proposal_num = None
        
        # Check for tree_data.json
        tree_data_path = stage_dir / "tree_data.json"
        journal_path = stage_dir / "journal.json"
        has_data = tree_data_path.exists()
        has_journal = journal_path.exists()
        
        # Determine status
        if has_data:
            status = "completed"
        elif has_journal:
            status = "in_progress"
        else:
            status = "not_started"
        
        # Try to extract best Dice score from tree_data.json
        best_dice = None
        if has_data:
            try:
                with open(tree_data_path, "r") as f:
                    data = json.load(f)
                    # Look for best node metrics
                    if "metrics" in data and "is_best_node" in data:
                        for i, is_best in enumerate(data.get("is_best_node", [])):
                            if is_best and data["metrics"][i]:
                                metric_data = data["metrics"][i]
                                if "metric_names" in metric_data:
                                    for m in metric_data["metric_names"]:
                                        if "dice" in m.get("metric_name", "").lower():
                                            if m.get("data"):
                                                best_dice = m["data"][0].get("final_value")
                                                break
            except Exception:
                pass
        
        proposals.append({
            "dir_name": dir_name,
            "display_name": f"Proposal {proposal_num}" if proposal_num else dir_name,
            "has_data": has_data,
            "best_dice": best_dice,
            "status": status,
        })
    
    return proposals


def cfg_to_tree_struct(cfg, jou: Journal, out_path: Path = None):
    edges = list(get_edges(jou))
    print(f"[red]Edges: {edges}[/red]")
    try:
        gen_layout = generate_layout(len(jou), edges)
    except Exception as e:
        print(f"Error in generate_layout: {e}")
        raise
    try:
        layout = normalize_layout(gen_layout)
    except Exception as e:
        print(f"Error in normalize_layout: {e}")
        raise

    best_node = jou.get_best_node()
    metrics = []
    is_best_node = []

    for n in jou:
        # print(f"Node {n.id} exc_stack: {type(n.exc_stack)} = {n.exc_stack}")
        if n.metric:
            # Pass the entire metric structure for the new format
            if isinstance(n.metric.value, dict) and "metric_names" in n.metric.value:
                metrics.append(n.metric.value)
            else:
                # Handle legacy format by wrapping it in the new structure
                metrics.append(
                    {
                        "metric_names": [
                            {
                                "metric_name": n.metric.name or "value",
                                "lower_is_better": not n.metric.maximize,
                                "description": n.metric.description or "",
                                "data": [
                                    {
                                        "dataset_name": "default",
                                        "final_value": n.metric.value,
                                        "best_value": n.metric.value,
                                    }
                                ],
                            }
                        ]
                    }
                )
        else:
            metrics.append(None)

        # Track whether this is the best node
        is_best_node.append(n is best_node)

    tmp = {}

    # Add each item individually with error handling
    try:
        tmp["edges"] = edges
    except Exception as e:
        print(f"Error setting edges: {e}")
        raise

    try:
        tmp["layout"] = layout.tolist()
    except Exception as e:
        print(f"Error setting layout: {e}")
        raise

    try:
        tmp["plan"] = [
            textwrap.fill(str(n.plan) if n.plan is not None else "", width=80)
            for n in jou.nodes
        ]
    except Exception as e:
        print(f"Error setting plan: {e}")
        raise

    try:
        tmp["code"] = [n.code for n in jou]
    except Exception as e:
        print(f"Error setting code: {e}")
        raise

    try:
        tmp["term_out"] = [
            textwrap.fill(str(n._term_out) if n._term_out is not None else "", width=80)
            for n in jou
        ]
    except Exception as e:
        print(f"Error setting term_out: {e}")
        print(f"n.term_out: {n._term_out}")
        raise

    try:
        tmp["analysis"] = [
            textwrap.fill(str(n.analysis) if n.analysis is not None else "", width=80)
            for n in jou
        ]
    except Exception as e:
        print(f"Error setting analysis: {e}")
        raise

    try:
        tmp["exc_type"] = [n.exc_type for n in jou]
    except Exception as e:
        print(f"Error setting exc_type: {e}")
        raise

    try:
        tmp["exc_info"] = [n.exc_info for n in jou]
    except Exception as e:
        print(f"Error setting exc_info: {e}")
        raise

    try:
        tmp["exc_stack"] = [n.exc_stack for n in jou]
    except Exception as e:
        print(f"Error setting exc_stack: {e}")
        raise

    try:
        tmp["exp_name"] = cfg.exp_name
    except Exception as e:
        print(f"Error setting exp_name: {e}")
        raise

    try:
        tmp["metrics"] = metrics
    except Exception as e:
        print(f"Error setting metrics: {e}")
        raise

    try:
        tmp["is_best_node"] = is_best_node
    except Exception as e:
        print(f"Error setting is_best_node: {e}")
        raise

    try:
        tmp["plots"] = [n.plots for n in jou]
    except Exception as e:
        print(f"Error setting plots: {e}")
        raise

    try:
        tmp["plot_paths"] = [n.plot_paths for n in jou]
    except Exception as e:
        print(f"Error setting plot_paths: {e}")
        raise



    try:
        tmp["exec_time"] = [n.exec_time for n in jou]
    except Exception as e:
        print(f"Error setting exec_time: {e}")
        raise

    try:
        tmp["exec_time_feedback"] = [
            textwrap.fill(
                str(n.exec_time_feedback) if n.exec_time_feedback is not None else "",
                width=80,
            )
            for n in jou
        ]
    except Exception as e:
        print(f"Error setting exec_time_feedback: {e}")
        raise

    try:
        tmp["datasets_successfully_tested"] = [
            n.datasets_successfully_tested for n in jou
        ]
    except Exception as e:
        print(f"Error setting datasets_successfully_tested: {e}")
        raise

    try:
        tmp["plot_code"] = [n.plot_code for n in jou]
    except Exception as e:
        print(f"Error setting plot_code: {e}")
        raise

    try:
        tmp["plot_plan"] = [n.plot_plan for n in jou]
    except Exception as e:
        print(f"Error setting plot_plan: {e}")
        raise

    try:
        tmp["ablation_name"] = [n.ablation_name for n in jou]
    except Exception as e:
        print(f"Error setting ablation_name: {e}")
        raise

    try:
        tmp["param_tuning_name"] = [getattr(n, 'param_tuning_name', None) for n in jou]
    except Exception as e:
        print(f"Error setting param_tuning_name: {e}")
        raise

    try:
        tmp["hyperparam_name"] = [n.hyperparam_name for n in jou]
    except Exception as e:
        print(f"Error setting hyperparam_name: {e}")
        raise

    '''
    try:
        tmp["is_seed_node"] = [n.is_seed_node for n in jou]
    except Exception as e:
        print(f"Error setting is_seed_node: {e}")
        raise
    

    try:
        tmp["is_seed_agg_node"] = [n.is_seed_agg_node for n in jou]
    except Exception as e:
        print(f"Error setting is_seed_agg_node: {e}")
        raise
    '''

    try:
        tmp["parse_metrics_plan"] = [
            textwrap.fill(
                str(n.parse_metrics_plan) if n.parse_metrics_plan is not None else "",
                width=80,
            )
            for n in jou
        ]
    except Exception as e:
        print(f"Error setting parse_metrics_plan: {e}")
        raise

    try:
        tmp["parse_metrics_code"] = [n.parse_metrics_code for n in jou]
    except Exception as e:
        print(f"Error setting parse_metrics_code: {e}")
        raise

    try:
        tmp["parse_term_out"] = [
            textwrap.fill(
                str(n.parse_term_out) if n.parse_term_out is not None else "", width=80
            )
            for n in jou
        ]
    except Exception as e:
        print(f"Error setting parse_term_out: {e}")
        raise

    try:
        tmp["parse_exc_type"] = [n.parse_exc_type for n in jou]
    except Exception as e:
        print(f"Error setting parse_exc_type: {e}")
        raise

    try:
        tmp["parse_exc_info"] = [n.parse_exc_info for n in jou]
    except Exception as e:
        print(f"Error setting parse_exc_info: {e}")
        raise

    try:
        tmp["parse_exc_stack"] = [n.parse_exc_stack for n in jou]
    except Exception as e:
        print(f"Error setting parse_exc_stack: {e}")
        raise

    # Node UUIDs, parent indices, and modification summaries for list view
    node_uuid_to_idx = {n.id: i for i, n in enumerate(jou)}
    tmp["node_ids"] = [n.id for n in jou]
    parent_indices = []
    for n in jou:
        parent_uuid = None
        if n.parent is not None and not isinstance(n.parent, str):
            parent_uuid = n.parent.id
        elif n.parent_id is not None:
            parent_uuid = n.parent_id
        if parent_uuid is not None and parent_uuid in node_uuid_to_idx:
            parent_indices.append(node_uuid_to_idx[parent_uuid])
        else:
            parent_indices.append(-1)
    tmp["parent_indices"] = parent_indices
    tmp["modification_summaries"] = [
        getattr(n, 'modification_summary', '') or '' for n in jou
    ]

    # Add the list of completed stages by checking directories
    if out_path:
        log_dir = out_path.parent.parent
        tmp["completed_stages"] = get_completed_stages(log_dir)

    return tmp


def generate_html(tree_graph_str: str):
    template_dir = Path(__file__).parent / "viz_templates"

    with open(template_dir / "template.js") as f:
        js = f.read()
        js = js.replace('"PLACEHOLDER_TREE_DATA"', tree_graph_str)

    with open(template_dir / "template.html") as f:
        html = f.read()
        html = html.replace("<!-- placeholder -->", js)

        return html


def generate(cfg, jou: Journal, out_path: Path):
    print("[red]Checking Journal[/red]")
    try:
        tree_struct = cfg_to_tree_struct(cfg, jou, out_path)
    except Exception as e:
        print(f"Error in cfg_to_tree_struct: {e}")
        raise

    # Save tree data as JSON for loading by the tabbed visualization
    try:
        # Save the tree data as a JSON file in the same directory
        data_path = out_path.parent / "tree_data.json"
        with open(data_path, "w") as f:
            json.dump(tree_struct, f)
    except Exception as e:
        print(f"Error saving tree data JSON: {e}")

    try:
        tree_graph_str = json.dumps(tree_struct)
    except Exception as e:
        print(f"Error in json.dumps: {e}")
        raise
    try:
        html = generate_html(tree_graph_str)
    except Exception as e:
        print(f"Error in generate_html: {e}")
        raise
    with open(out_path, "w") as f:
        f.write(html)

    # Create a unified tree visualization that shows all stages
    try:
        create_unified_viz(cfg, out_path)
    except Exception as e:
        print(f"Error creating unified visualization: {e}")
        # Continue even if unified viz creation fails


def create_unified_viz(cfg, current_stage_viz_path):
    """
    Create a unified visualization that shows all completed stages in a tabbed interface.
    All data is embedded directly in the HTML for offline viewing (no fetch required).
    """
    # The main log directory is two levels up from the stage-specific visualization
    log_dir = current_stage_viz_path.parent.parent

    # Get the current stage name from the path
    current_stage = current_stage_viz_path.parent.name
    if current_stage.startswith("stage_"):
        parts = current_stage.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            stage_num = parts[1]
            current_stage = f"Stage_{stage_num}"

    # Create a combined visualization at the top level
    unified_viz_path = log_dir / "unified_tree_viz.html"
    template_dir = Path(__file__).parent / "viz_templates"

    with open(template_dir / "template.html") as f:
        html = f.read()

    with open(template_dir / "template.js") as f:
        js = f.read()

    # Get completed stages and Stage 2 proposals
    completed_stages = get_completed_stages(log_dir)
    stage2_proposals = get_stage2_proposals(log_dir)

    # Build a complete data structure with all stage data embedded
    all_stages_data = {
        "current_stage": current_stage,
        "completed_stages": completed_stages,
        "stage2_proposals": stage2_proposals,
        "stages": {},
    }

    # Stage name to directory prefix mapping
    stage_dir_prefixes = {
        "Stage_1": "stage_1_",
        "Stage_2": "stage_2_",
        "Stage_3": "stage_3_",
        "Stage_4": "stage_4_",
    }

    # Load data for each completed stage
    for stage in completed_stages:
        prefix = stage_dir_prefixes.get(stage, f"stage_{stage.split('_')[1]}_")
        
        # Find matching directory
        matching_dirs = [
            d for d in log_dir.iterdir()
            if d.is_dir() and d.name.startswith(prefix)
        ]
        
        if stage == "Stage_2":
            # For Stage 2, load the first proposal as default
            if stage2_proposals:
                first_proposal_dir = log_dir / stage2_proposals[0]["dir_name"]
                tree_data_path = first_proposal_dir / "tree_data.json"
                if tree_data_path.exists():
                    try:
                        with open(tree_data_path, "r") as f:
                            all_stages_data["stages"][stage] = json.load(f)
                    except Exception as e:
                        print(f"Error loading Stage 2 data: {e}")
                        all_stages_data["stages"][stage] = {"layout": [], "edges": []}
        else:
            # For other stages, load the first matching directory
            for stage_dir in matching_dirs:
                tree_data_path = stage_dir / "tree_data.json"
                if tree_data_path.exists():
                    try:
                        with open(tree_data_path, "r") as f:
                            all_stages_data["stages"][stage] = json.load(f)
                        break
                    except Exception as e:
                        print(f"Error loading {stage} data: {e}")

    # Load all Stage 2 proposal data for embedding
    all_stages_data["stage2_data"] = {}
    for proposal in stage2_proposals:
        if proposal["has_data"]:
            proposal_dir = log_dir / proposal["dir_name"]
            tree_data_path = proposal_dir / "tree_data.json"
            try:
                with open(tree_data_path, "r") as f:
                    all_stages_data["stage2_data"][proposal["dir_name"]] = json.load(f)
            except Exception as e:
                print(f"Error loading proposal {proposal['dir_name']}: {e}")

    # Generate unique experiment ID for namespace isolation
    exp_id = cfg.exp_name if hasattr(cfg, 'exp_name') else str(log_dir.name)
    all_stages_data["experiment_id"] = exp_id

    # Replace the placeholder in the JS with our complete data
    js = js.replace('"PLACEHOLDER_TREE_DATA"', json.dumps(all_stages_data))

    # Replace the placeholder in the HTML with our JS
    html = html.replace("<!-- placeholder -->", js)

    # Write the unified visualization
    with open(unified_viz_path, "w") as f:
        f.write(html)

    print(f"[green]Created unified visualization at {unified_viz_path}[/green]")

