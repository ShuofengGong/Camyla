"""Compute computational-efficiency metrics (Params, FLOPs, Inference Time).

Called after all experiments finish but before ``experiment_report.md`` is
generated.  Results are written to ``efficiency_metrics.json`` in the log
directory and later rendered into the paper-ready Markdown.
"""

import json
import importlib.util
import inspect
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_params_from_state_dict(state_dict: dict) -> int:
    """Count total trainable parameters from a state_dict."""
    return sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))


def _load_plans_and_dataset_json(
    checkpoint: dict,
    model_results_dir: Path,
) -> Tuple[dict, dict]:
    """Load plans and dataset metadata directly from the checkpoint.

    Efficiency computation must use the exact training metadata embedded in the
    checkpoint. If that metadata is missing or malformed, fail fast instead of
    falling back to another dataset's plans.
    """
    init_args = checkpoint.get("init_args", {})
    plans = init_args.get("plans")
    dataset_json = init_args.get("dataset_json")

    if not isinstance(plans, dict) or not plans:
        raise RuntimeError(
            f"Missing or invalid checkpoint init_args['plans'] in {model_results_dir}"
        )
    if not isinstance(dataset_json, dict) or not dataset_json:
        raise RuntimeError(
            f"Missing or invalid checkpoint init_args['dataset_json'] in {model_results_dir}"
        )

    return plans, dataset_json


def _get_run_dir_from_model_results(model_results_dir: Path) -> Path:
    """Return the run directory for ``.../results/<node_id>/model_results``."""
    try:
        return model_results_dir.parent.parent.parent
    except IndexError as e:
        raise RuntimeError(
            f"Could not derive run directory from model results path: {model_results_dir}"
        ) from e


def _write_tree_code_to_cache(run_dir: Path, node_id: str, code: str) -> Path:
    """Persist code extracted from tree_data.json so it can be imported."""
    cache_dir = run_dir / ".efficiency_import_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{node_id}.py"
    cache_path.write_text(code, encoding="utf-8")
    return cache_path


def _find_custom_code_path(run_dir: Path, node_id: str) -> Path:
    """Locate the saved source file for a custom node by node_id."""
    direct_candidates = sorted(
        run_dir.glob(f"stage_*/generated_codes/*_{node_id}_experiment_code.py")
    )
    if direct_candidates:
        return direct_candidates[-1]

    broad_candidates = sorted(
        run_dir.glob(f"stage_*/generated_codes/*_{node_id}_*_code.py")
    )
    if broad_candidates:
        return broad_candidates[-1]

    for tree_path in sorted(run_dir.glob("stage_*/tree_data.json")):
        try:
            tree_data = json.loads(tree_path.read_text(encoding="utf-8"))
        except Exception as e:
            raise RuntimeError(f"Failed to read {tree_path}: {e}") from e

        node_ids = tree_data.get("node_ids", [])
        codes = tree_data.get("code", [])
        if node_id not in node_ids:
            continue

        idx = node_ids.index(node_id)
        code = codes[idx] if idx < len(codes) else ""
        if not isinstance(code, str) or not code.strip():
            raise RuntimeError(
                f"Node {node_id} was found in {tree_path} but has empty code content"
            )
        return _write_tree_code_to_cache(run_dir, node_id, code)

    raise RuntimeError(
        f"No saved source file found for custom node {node_id} under {run_dir}"
    )


def _load_module_from_source(source_path: Path, node_id: str):
    """Dynamically import a node-specific source file as a unique module."""
    repo_root = Path(__file__).resolve().parents[2]
    module_name = f"_efficiency_node_{node_id}"

    if module_name in sys.modules:
        del sys.modules[module_name]

    added_paths: List[str] = []
    for candidate in [str(source_path.parent), str(repo_root)]:
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
            added_paths.append(candidate)

    try:
        spec = importlib.util.spec_from_file_location(module_name, source_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not create import spec for {source_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        raise RuntimeError(
            f"Failed to import custom source for node {node_id} from {source_path}: {e}"
        ) from e
    finally:
        for candidate in reversed(added_paths):
            if candidate in sys.path:
                sys.path.remove(candidate)


def _resolve_trainer_class(
    checkpoint: dict,
    model_results_dir: Path,
    node_id: str,
):
    """Resolve a trainer class from built-ins first, then node-specific code."""
    import camylanet
    from camylanet.utilities.find_class_by_name import recursive_find_python_class

    trainer_name = checkpoint.get("trainer_name", "nnUNetTrainer")
    trainer_class = recursive_find_python_class(
        os.path.join(camylanet.__path__[0], "training", "nnUNetTrainer"),
        trainer_name,
        "camylanet.training.nnUNetTrainer",
    )
    if trainer_class is not None:
        return trainer_class

    run_dir = _get_run_dir_from_model_results(model_results_dir)
    source_path = _find_custom_code_path(run_dir, node_id)
    module = _load_module_from_source(source_path, node_id)
    trainer_class = getattr(module, trainer_name, None)
    if trainer_class is None:
        raise RuntimeError(
            f"Trainer class '{trainer_name}' not found in source file {source_path} "
            f"for node {node_id}"
        )
    return trainer_class


def _rebuild_network(
    checkpoint: dict,
    plans: dict,
    dataset_json: dict,
    model_results_dir: Path,
    node_id: str,
) -> torch.nn.Module:
    """Reconstruct the network from a checkpoint + plans + dataset metadata."""
    try:
        from camylanet.experiment_planning.plan_and_preprocess_api import PlansManager
        from camylanet.utilities.label_handling.label_handling  import (
            determine_num_input_channels,
        )

        configuration_name = checkpoint.get("init_args", {}).get("configuration", "2d")

        plans_manager = PlansManager(plans)
        configuration_manager = plans_manager.get_configuration(configuration_name)
        num_input_channels = determine_num_input_channels(
            plans_manager, configuration_manager, dataset_json
        )
        label_manager = plans_manager.get_label_manager(dataset_json)

        trainer_class = _resolve_trainer_class(checkpoint, model_results_dir, node_id)

        signature = inspect.signature(trainer_class.build_network_architecture)
        param_names = list(signature.parameters.keys())

        if param_names[:4] == [
            "architecture_class_name",
            "arch_init_kwargs",
            "arch_init_kwargs_req_import",
            "num_input_channels",
        ]:
            network = trainer_class.build_network_architecture(
                configuration_manager.network_arch_class_name,
                configuration_manager.network_arch_init_kwargs,
                configuration_manager.network_arch_init_kwargs_req_import,
                num_input_channels,
                label_manager.num_segmentation_heads,
                enable_deep_supervision=False,
            )
        elif param_names[:4] == [
            "plans_manager",
            "dataset_json",
            "configuration_manager",
            "num_input_channels",
        ]:
            network = trainer_class.build_network_architecture(
                plans_manager,
                dataset_json,
                configuration_manager,
                num_input_channels,
                enable_deep_supervision=False,
            )
        elif param_names[:3] == [
            "configuration_manager",
            "num_input_channels",
            "num_output_channels",
        ]:
            if "enable_deep_supervision" in signature.parameters:
                network = trainer_class.build_network_architecture(
                    configuration_manager,
                    num_input_channels,
                    label_manager.num_segmentation_heads,
                    enable_deep_supervision=False,
                )
            else:
                network = trainer_class.build_network_architecture(
                    configuration_manager,
                    num_input_channels,
                    label_manager.num_segmentation_heads,
                )
        elif param_names[:2] == ["num_input_channels", "num_output_channels"]:
            network = trainer_class.build_network_architecture(
                num_input_channels,
                label_manager.num_segmentation_heads,
            )
        else:
            raise RuntimeError(
                f"Unsupported build_network_architecture signature for "
                f"{trainer_class.__name__}: {signature}"
            )

        network.load_state_dict(checkpoint["network_weights"])
        return network
    except Exception as e:
        raise RuntimeError(f"Failed to rebuild network: {e}") from e


def _get_input_shape(
    plans: dict,
    dataset_json: dict,
    configuration_name: str,
) -> List[int]:
    """Derive a representative input tensor shape from the plans."""
    try:
        from camylanet.experiment_planning.plan_and_preprocess_api import PlansManager
        from camylanet.utilities.label_handling.label_handling  import (
            determine_num_input_channels,
        )

        pm = PlansManager(plans)
        cm = pm.get_configuration(configuration_name)
        num_input_channels = determine_num_input_channels(pm, cm, dataset_json)
        patch_size = cm.patch_size
        return [1, num_input_channels] + list(patch_size)
    except Exception as e:
        raise RuntimeError(f"Could not determine input shape: {e}") from e


def _compute_flops(network: torch.nn.Module, input_shape: List[int]) -> Optional[float]:
    """Compute FLOPs using fvcore (preferred) or ptflops."""
    device = next(network.parameters()).device

    try:
        from fvcore.nn import FlopCountAnalysis
        dummy = torch.randn(*input_shape, device=device)
        flops = FlopCountAnalysis(network, dummy).total()
        return float(flops)
    except ImportError:
        pass

    try:
        from thop import profile

        dummy = torch.randn(*input_shape, device=device)
        with torch.no_grad():
            flops, _ = profile(network, inputs=(dummy,), verbose=False)
        if flops is not None:
            return float(flops)
    except ImportError:
        pass

    try:
        from ptflops import get_model_complexity_info

        spatial = tuple(input_shape[2:])
        flops, _ = get_model_complexity_info(
            network,
            (input_shape[1], *spatial),
            as_strings=False,
            print_per_layer_stat=False,
        )
        if flops is not None:
            return float(flops)
    except ImportError:
        pass

    raise RuntimeError(
        "FLOPs computation failed with all available backends. "
        "Tried fvcore, thop, and ptflops."
    )


def _measure_inference_time(
    network: torch.nn.Module,
    input_shape: List[int],
    warmup: int = 10,
    repeats: int = 50,
) -> float:
    """Measure average inference time (seconds) on the current device."""
    device = next(network.parameters()).device
    network.eval()
    dummy = torch.randn(*input_shape, device=device)

    try:
        with torch.no_grad():
            for _ in range(warmup):
                _ = network(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            for _ in range(repeats):
                _ = network(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

        return elapsed / repeats
    except Exception as e:
        raise RuntimeError(f"Inference timing failed: {e}") from e


# ---------------------------------------------------------------------------
# Per-node metric computation
# ---------------------------------------------------------------------------

def compute_metrics_for_node(
    model_results_dir: str,
    node_id: str,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Compute efficiency metrics for a single experiment node.

    Returns a dict with keys: node_id, params, flops, inference_time_s.
    Raises an exception immediately if checkpoint metadata is incomplete or if
    the network/FLOPs/timing pipeline cannot be reconstructed.
    """
    result: Dict[str, Any] = {
        "node_id": node_id,
        "params": None,
        "flops": None,
        "inference_time_s": None,
    }

    mr = Path(model_results_dir)
    ckpt_path = mr / "checkpoint_final.pth"
    if not ckpt_path.exists():
        ckpt_path = mr / "checkpoint_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint found in {mr}")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("network_weights", {})
    result["params"] = _count_params_from_state_dict(state_dict)

    plans, dataset_json = _load_plans_and_dataset_json(checkpoint, mr)

    configuration_name = checkpoint.get("init_args", {}).get("configuration", "2d")
    input_shape = _get_input_shape(plans, dataset_json, configuration_name)

    network = _rebuild_network(checkpoint, plans, dataset_json, mr, node_id)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    network = network.to(dev)

    result["inference_time_s"] = _measure_inference_time(network, input_shape)
    result["flops"] = _compute_flops(network, input_shape)

    del network
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------------
# Main entry: iterate over all relevant nodes
# ---------------------------------------------------------------------------

def _load_tree_data(tree_path: Path) -> Dict[str, Any]:
    """Read a tree_data.json payload and validate its basic structure."""
    try:
        tree_data = json.loads(tree_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to read tree file {tree_path}: {e}") from e

    if not isinstance(tree_data, dict):
        raise RuntimeError(f"Invalid tree payload in {tree_path}: expected dict")
    return tree_data


def _get_stage2_best_node_id(log_dir: Path) -> str:
    """Return the single Stage 2 best node kept for paper writing."""
    results_dir = log_dir / "results"
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    best_candidates: List[Tuple[str, Path]] = []
    seen_ids = set()

    for tree_path in sorted(log_dir.glob("stage_2*/tree_data.json")):
        tree_data = _load_tree_data(tree_path)
        node_ids = tree_data.get("node_ids", [])
        best_flags = tree_data.get("is_best_node", [])

        if not isinstance(best_flags, list):
            raise RuntimeError(f"Invalid Stage 2 tree payload in {tree_path}")

        # Old experiments may lack node_ids; skip those trees
        if not isinstance(node_ids, list) or len(node_ids) == 0:
            continue
        if len(node_ids) != len(best_flags):
            continue

        for idx, node_id in enumerate(node_ids):
            if not best_flags[idx]:
                continue
            if not isinstance(node_id, str) or not node_id:
                continue
            if node_id in seen_ids:
                continue
            best_candidates.append((node_id, tree_path))
            seen_ids.add(node_id)

    # Fallback: old experiments may lack node_ids; extract from best_solution_*.py filenames
    if not best_candidates:
        import re
        for sol_path in sorted(log_dir.glob("stage_2*/best_solution_*.py")):
            m = re.search(r'best_solution_([0-9a-f]{32})\.py$', sol_path.name)
            if m:
                node_id = m.group(1)
                if node_id not in seen_ids:
                    best_candidates.append((node_id, sol_path.parent / "tree_data.json"))
                    seen_ids.add(node_id)

    if not best_candidates:
        raise RuntimeError(f"No Stage 2 best node found under {log_dir}")

    # Pick the first candidate whose model_results directory actually exists.
    for node_id, source in best_candidates:
        if (results_dir / node_id / "model_results").exists():
            return node_id

    # No candidate has model_results
    best_node_id = best_candidates[0][0]
    raise FileNotFoundError(
        f"Stage 2 best node {best_node_id} from {best_candidates[0][1]} has no model_results at "
        f"{results_dir / best_node_id / 'model_results'}"
    )


def _load_baseline_efficiency_targets(log_dir: Path) -> List[Dict[str, str]]:
    """Load baseline model targets from all_baselines.json in display order."""
    baselines_path = log_dir / "all_baselines.json"
    if not baselines_path.exists():
        raise FileNotFoundError(f"all_baselines.json not found at {baselines_path}")

    try:
        baselines_data = json.loads(baselines_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to read {baselines_path}: {e}") from e

    baselines = baselines_data.get("baselines", [])
    if not isinstance(baselines, list) or not baselines:
        raise RuntimeError(f"No baseline entries found in {baselines_path}")

    targets: List[Dict[str, str]] = []
    for idx, item in enumerate(baselines):
        if not isinstance(item, dict):
            raise RuntimeError(f"Invalid baseline entry at index {idx} in {baselines_path}")

        model_name = item.get("model_name")
        node_id = item.get("node_id")
        exp_results_dir = item.get("exp_results_dir")

        if not isinstance(model_name, str) or not model_name.strip():
            raise RuntimeError(f"Baseline entry {idx} is missing model_name in {baselines_path}")
        if not isinstance(node_id, str) or not node_id.strip():
            raise RuntimeError(f"Baseline entry {idx} is missing node_id in {baselines_path}")
        if not isinstance(exp_results_dir, str) or not exp_results_dir.strip():
            raise RuntimeError(
                f"Baseline entry {idx} is missing exp_results_dir in {baselines_path}"
            )

        model_results_dir = Path(exp_results_dir)
        if not model_results_dir.exists():
            logger.warning(
                f"Baseline '{model_name}' results directory does not exist: "
                f"{model_results_dir}, skipping"
            )
            continue

        targets.append(
            {
                "model_name": model_name,
                "source": "baseline",
                "node_id": node_id,
                "model_results_dir": str(model_results_dir),
            }
        )

    return targets


def _collect_paper_efficiency_targets(log_dir: Path) -> List[Dict[str, str]]:
    """Collect exactly the paper-facing models: baselines + Stage 2 best."""
    targets: List[Dict[str, str]] = []

    try:
        targets.extend(_load_baseline_efficiency_targets(log_dir))
    except Exception as e:
        logger.warning(f"Failed to collect baseline efficiency targets: {e}")

    try:
        best_node_id = _get_stage2_best_node_id(log_dir)
    except Exception as e:
        logger.warning(f"Failed to collect Stage 2 best efficiency target: {e}")
    else:
        best_model_results_dir = log_dir / "results" / best_node_id / "model_results"
        targets.append(
            {
                "model_name": "Proposed Method",
                "source": "stage2_best",
                "node_id": best_node_id,
                "model_results_dir": str(best_model_results_dir),
            }
        )

    return targets

def _collect_authoritative_result_node_ids(log_dir: Path) -> List[str]:
    """Collect node ids that are both in the saved stage trees and in results/.

    The results directory may contain subagent-competition losers because each
    successful candidate is persisted before the winner is chosen. The stage
    tree files are the authoritative record of nodes that were actually kept in
    the search tree, so efficiency computation should be limited to their
    intersection with ``results/``.
    """
    results_dir = log_dir / "results"
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    authoritative_ids: List[str] = []
    seen_ids = set()

    for tree_path in sorted(log_dir.glob("stage_*/tree_data.json")):
        try:
            tree_data = json.loads(tree_path.read_text(encoding="utf-8"))
        except Exception as e:
            raise RuntimeError(f"Failed to read authoritative tree file {tree_path}: {e}") from e

        node_ids = tree_data.get("node_ids", [])
        if not isinstance(node_ids, list):
            raise RuntimeError(f"Invalid node_ids payload in {tree_path}")

        for node_id in node_ids:
            if not isinstance(node_id, str) or not node_id or node_id in seen_ids:
                continue

            model_results_dir = results_dir / node_id / "model_results"
            if not model_results_dir.exists():
                continue

            authoritative_ids.append(node_id)
            seen_ids.add(node_id)

    return authoritative_ids


def compute_all_efficiency_metrics(log_dir: str) -> Dict[str, Any]:
    """Compute metrics only for authoritative nodes retained in the stage trees.

    Saves the output to ``{log_dir}/efficiency_metrics.json`` and returns it.
    Raises immediately if any node fails.
    """
    log_dir_path = Path(log_dir)
    results_dir = log_dir_path / "results"
    authoritative_node_ids = _collect_authoritative_result_node_ids(log_dir_path)
    if not authoritative_node_ids:
        raise RuntimeError(
            f"No authoritative result nodes found from stage trees under {log_dir_path}"
        )

    all_metrics: Dict[str, Any] = {}
    for node_id in authoritative_node_ids:
        mr = results_dir / node_id / "model_results"
        logger.info(f"Computing efficiency metrics for node {node_id}...")
        try:
            metrics = compute_metrics_for_node(str(mr), node_id)
        except Exception as e:
            raise RuntimeError(f"Efficiency metrics failed for node {node_id}: {e}") from e

        all_metrics[node_id] = metrics
        params = metrics.get("params")
        flops = metrics.get("flops")
        inft = metrics.get("inference_time_s")
        logger.info(
            f"  -> params={params}, flops={flops}, inference_time={inft:.4f}s"
        )

    out_path = log_dir_path / "efficiency_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"Efficiency metrics saved to {out_path}")
    return all_metrics


def compute_paper_efficiency_metrics(log_dir: str) -> Dict[str, Any]:
    """Compute paper-facing efficiency metrics for baselines and Stage 2 best node.

    The output is intentionally limited to the models that appear in the paper's
    main comparison table: all baselines from ``all_baselines.json`` and the
    single Stage 2 best node rendered as ``Proposed Method``.
    """
    log_dir_path = Path(log_dir)
    targets = _collect_paper_efficiency_targets(log_dir_path)
    if not targets:
        logger.warning(f"No paper-facing efficiency targets found under {log_dir_path}")
        payload = {"rows": []}
        out_path = log_dir_path / "efficiency_metrics.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"Paper-facing efficiency metrics saved to {out_path}")
        return payload

    rows: List[Dict[str, Any]] = []
    for target in targets:
        model_name = target["model_name"]
        node_id = target["node_id"]
        model_results_dir = target["model_results_dir"]

        logger.info(f"Computing efficiency metrics for {model_name} ({node_id})...")
        try:
            metrics = compute_metrics_for_node(model_results_dir, node_id)
        except Exception as e:
            logger.warning(
                f"Skipping efficiency metrics for {model_name} ({node_id}): {e}"
            )
            continue

        row = {
            "model_name": model_name,
            "source": target["source"],
            "node_id": node_id,
            "params": metrics.get("params"),
            "flops": metrics.get("flops"),
            "inference_time_s": metrics.get("inference_time_s"),
        }
        rows.append(row)

        params = row["params"]
        flops = row["flops"]
        inft = row["inference_time_s"]
        logger.info(
            f"  -> params={params}, flops={flops}, inference_time={inft:.4f}s"
        )

    payload = {"rows": rows}
    out_path = log_dir_path / "efficiency_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Paper-facing efficiency metrics saved to {out_path}")
    return payload
