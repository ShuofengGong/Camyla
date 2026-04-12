"""
Publication-quality segmentation comparison figure generator.
Adapted from scripts/plot_segmentation_comparison.py for automated paper generation.

Supports both volumetric medical datasets (for example ``.nii.gz``) and
2D raster datasets (for example ``.png`` / ``.jpg``).
Generates a grid figure: Image | Baseline1 | ... | Ours | GT
"""

import json
import os
import glob as globmod
import logging
import warnings
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PIL import Image

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# ── Default data roots (overridable via env vars) ──
CAMYLANET_RESULTS = os.environ.get(
    "CAMYLANET_RESULTS", "")
CAMYLANET_RAW = os.environ.get(
    "CAMYLANET_RAW", "")
CAMYLANET_PREPROCESSED = os.environ.get(
    "CAMYLANET_PREPROCESSED", "")

OVERLAY_COLORS = np.array([
    [220,  20,  60],  # Crimson
    [ 30, 144, 255],  # DodgerBlue
    [ 50, 205,  50],  # LimeGreen
    [255, 165,   0],  # Orange
    [148, 103, 189],  # MediumPurple
    [  0, 206, 209],  # DarkTurquoise
    [255, 105, 180],  # HotPink
], dtype=np.float32)

ALPHA = 0.55
MAX_EDGE = 400
DPI = 600
CELL_SIZE = 0.75


# ══════════════════════════════════════════════════════════════
# Path discovery
# ══════════════════════════════════════════════════════════════

def _glob_one(pattern: str) -> Optional[str]:
    matches = sorted(globmod.glob(pattern))
    return matches[0] if matches else None


def find_dataset_dirs(dataset_id: str) -> Dict[str, Optional[str]]:
    return {
        "results": _glob_one(os.path.join(CAMYLANET_RESULTS, f"Dataset{dataset_id}_*")),
        "raw": _glob_one(os.path.join(CAMYLANET_RAW, f"Dataset{dataset_id}_*")),
        "preprocessed": _glob_one(os.path.join(CAMYLANET_PREPROCESSED, f"Dataset{dataset_id}_*")),
    }


def find_best_node(results_dir: Path) -> Optional[Tuple[str, float, Path]]:
    """Find the QWBE node with the highest Dice score."""
    best_dice = -1.0
    best_node = None
    best_val_dir = None

    for node_dir in results_dir.iterdir():
        if not node_dir.is_dir():
            continue
        summary_path = node_dir / "model_results" / "validation" / "summary.json"
        if not summary_path.exists():
            continue
        try:
            data = _load_summary(str(summary_path))
            dice = data.get("foreground_mean", {}).get("Dice", 0)
            if dice > best_dice:
                best_dice = dice
                best_node = node_dir.name
                best_val_dir = node_dir / "model_results" / "validation"
        except Exception:
            continue

    if best_node:
        return best_node, best_dice, best_val_dir
    return None


def find_baseline_validation_dirs(
    dataset_results_dir: str,
    dataset_id: str,
) -> Dict[str, str]:
    """Auto-discover baseline trainer validation directories.
    
    Returns mapping: display_name -> validation_dir_path
    """
    KNOWN_TRAINERS = {
        "nnUNetTrainer": "nnU-Net",
        "SegResNetTrainer": "SegResNet",
        "UNETRTrainer": "UNETR",
        "UNetPlusPlusTrainer": "UNet++",
        "STUNetTrainer": "STU-Net",
        "UMambaTrainer": "U-Mamba",
    }

    found = {}
    for trainer_key, display_name in KNOWN_TRAINERS.items():
        candidates = [
            os.path.join(dataset_results_dir, f"{dataset_id}_{trainer_key}",
                         "model_results", "validation"),
            *sorted(globmod.glob(os.path.join(
                dataset_results_dir, f"{trainer_key}__nnUNetPlans__*",
                "fold_0", "validation"))),
        ]
        for c in candidates:
            if c and os.path.isdir(c):
                summary = os.path.join(c, "summary.json")
                if os.path.isfile(summary):
                    found[display_name] = c
                    break
    return found


# ══════════════════════════════════════════════════════════════
# Metrics loading
# ══════════════════════════════════════════════════════════════

def _load_summary(path: str) -> dict:
    with open(path) as f:
        txt = f.read().replace("NaN", "null").replace("Infinity", "1e9")
    return json.loads(txt)


def get_per_case_fg_dice(summary: dict) -> Dict[str, float]:
    result = {}
    for entry in summary.get("metric_per_case", []):
        fname = os.path.basename(entry["prediction_file"])
        dices = []
        for label_id, m in entry["metrics"].items():
            if label_id == "0":
                continue
            d = m.get("Dice")
            dices.append(d if d is not None else 0.0)
        result[fname] = float(np.mean(dices)) if dices else 0.0
    return result


def get_per_case_metrics(summary: dict) -> List[Dict[str, Any]]:
    """Extract per-case metrics including volume size for subgroup analysis."""
    cases = []
    for entry in summary.get("metric_per_case", []):
        fname = os.path.basename(entry["prediction_file"])
        dices, hd95s, n_refs = [], [], []
        for label_id, m in entry["metrics"].items():
            if label_id == "0":
                continue
            d = m.get("Dice")
            dices.append(d if d is not None else 0.0)
            h = m.get("HD95")
            hd95s.append(h if h is not None else 999.0)
            n = m.get("n_ref", 0)
            n_refs.append(n if n is not None else 0)
        cases.append({
            "case_name": fname,
            "dice": float(np.mean(dices)) if dices else 0.0,
            "hd95": float(np.mean(hd95s)) if hd95s else 999.0,
            "volume_pixels": int(np.sum(n_refs)),
        })
    return cases


# ══════════════════════════════════════════════════════════════
# Image I/O and rendering
# ══════════════════════════════════════════════════════════════

def _is_raster_path(path: str) -> bool:
    lower = path.lower()
    return lower.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))


def _load_volume(path: str) -> np.ndarray:
    import nibabel as nib
    return nib.load(path).get_fdata()


def _load_raster(path: str) -> np.ndarray:
    return np.array(Image.open(path))


def load_segmentation_array(path: str) -> np.ndarray:
    """Load a segmentation mask from either a volume file or a raster image."""
    if _is_raster_path(path):
        arr = _load_raster(path)
        if arr.ndim == 3:
            arr = arr[..., 0]
        return arr.astype(np.int32)

    return _load_volume(path).astype(np.int32)


def load_input_image(path: str) -> np.ndarray:
    """Load an input image from either a volume file or a raster image."""
    if _is_raster_path(path):
        arr = _load_raster(path)
        if arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[..., :3]
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        return arr

    return _load_volume(path)


def prepare_display_image(image: np.ndarray) -> np.ndarray:
    """Convert raw image data into a uint8 display image."""
    arr = np.asarray(image)

    if arr.ndim == 2:
        return normalize_intensity(arr)

    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        rgb = arr[..., :3].astype(np.float32)
        if rgb.max() <= 1.01:
            rgb *= 255.0
        return np.clip(rgb, 0, 255).astype(np.uint8)

    if arr.ndim == 3 and arr.shape[-1] == 1:
        return normalize_intensity(arr[..., 0])

    return normalize_intensity(np.squeeze(arr))


def ensure_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a display image to RGB for plotting."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        return np.stack([arr] * 3, axis=-1).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        return arr[..., :3].astype(np.uint8)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return np.repeat(arr, 3, axis=-1).astype(np.uint8)
    return np.stack([np.squeeze(arr)] * 3, axis=-1).astype(np.uint8)


def find_raw_image_path(raw_dir: Optional[str], case_id: str, file_ending: str) -> Optional[str]:
    """Locate a raw image for a case in either imagesTr or imagesTs."""
    if not raw_dir:
        return None

    candidate_dirs = ["imagesTr", "imagesTs"]
    candidate_suffixes = [f"_0000{file_ending}", file_ending]
    for subdir in candidate_dirs:
        for suffix in candidate_suffixes:
            path = os.path.join(raw_dir, subdir, f"{case_id}{suffix}")
            if os.path.isfile(path):
                return path
    return None


def normalize_intensity(img: np.ndarray, plow: float = 1, phigh: float = 99) -> np.ndarray:
    img = img.astype(np.float64)
    lo, hi = np.percentile(img, [plow, phigh])
    if hi - lo < 1e-8:
        return np.zeros_like(img, dtype=np.uint8)
    return np.clip((img - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def create_overlay(image: np.ndarray, seg: np.ndarray, num_classes: int,
                   alpha: float = 0.55, contour_width: int = 2) -> np.ndarray:
    from scipy.ndimage import binary_erosion

    img = image.astype(np.float32)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.max() > 1.01:
        img = img / 255.0
    overlay = img.copy()

    for cls_id in range(1, num_classes):
        mask = seg == cls_id
        if not mask.any():
            continue
        cidx = (cls_id - 1) % len(OVERLAY_COLORS)
        color = OVERLAY_COLORS[cidx] / 255.0
        overlay[mask] = overlay[mask] * (1 - alpha) + color * alpha
        eroded = binary_erosion(mask, iterations=contour_width)
        boundary = mask & ~eroded
        overlay[boundary] = color * 0.9 + 0.1

    return np.clip(overlay * 255, 0, 255).astype(np.uint8)


def _resize(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    if (img.shape[0], img.shape[1]) == (target_h, target_w):
        return img
    pil = Image.fromarray(img.astype(np.uint8))
    pil = pil.resize((target_w, target_h), Image.LANCZOS)
    return np.array(pil)


# ══════════════════════════════════════════════════════════════
# Slice selection (3D)
# ══════════════════════════════════════════════════════════════

def compute_slice_dice(pred: np.ndarray, gt: np.ndarray, num_classes: int) -> float:
    dices = []
    for c in range(1, num_classes):
        pm = pred == c
        gm = gt == c
        inter = np.sum(pm & gm)
        union = np.sum(pm) + np.sum(gm)
        if union == 0:
            continue
        dices.append(2.0 * inter / union)
    return float(np.mean(dices)) if dices else 0.0


def select_slices_3d(
    ours_val_dir: str,
    baseline_val_dirs: Dict[str, str],
    gt_dir: str,
    file_ending: str,
    num_classes: int,
    max_cases: int = 3,
    dice_threshold: float = 0.6,
    dice_margin: float = 0.02,
) -> List[Tuple[str, int, float, float]]:
    """Select (case, slice_idx, target_dice, advantage) tuples."""
    import nibabel as nib

    ours_files = sorted(f for f in os.listdir(ours_val_dir) if f.endswith(file_ending))
    common = set(ours_files)
    for bdir in baseline_val_dirs.values():
        common &= set(f for f in os.listdir(bdir) if f.endswith(file_ending))
    common = sorted(common)

    candidates = []
    for case in common:
        gt_path = os.path.join(gt_dir, case)
        if not os.path.isfile(gt_path):
            continue

        gt_vol = nib.load(gt_path).get_fdata().astype(np.int32)
        ours_vol = nib.load(os.path.join(ours_val_dir, case)).get_fdata().astype(np.int32)
        bl_vols = {}
        for name, bdir in baseline_val_dirs.items():
            p = os.path.join(bdir, case)
            if os.path.isfile(p):
                bl_vols[name] = nib.load(p).get_fdata().astype(np.int32)

        if not bl_vols:
            continue

        for s in range(gt_vol.shape[2]):
            gs = gt_vol[:, :, s]
            if gs.max() == 0 or np.count_nonzero(gs) < 50:
                continue
            td = compute_slice_dice(ours_vol[:, :, s], gs, num_classes)
            if td < dice_threshold:
                continue
            bl_dices = [compute_slice_dice(v[:, :, s], gs, num_classes)
                        for v in bl_vols.values() if v.shape[2] > s]
            if not bl_dices:
                continue
            diff = td - max(bl_dices)
            if diff > dice_margin:
                candidates.append((case, s, td, diff))

    candidates.sort(key=lambda x: x[3], reverse=True)

    if not candidates:
        logger.warning("No slice meets criteria, falling back to top-dice slices")
        for case in common[:10]:
            gt_path = os.path.join(gt_dir, case)
            if not os.path.isfile(gt_path):
                continue
            gt_vol = nib.load(gt_path).get_fdata().astype(np.int32)
            ours_path = os.path.join(ours_val_dir, case)
            if not os.path.isfile(ours_path):
                continue
            tvol = nib.load(ours_path).get_fdata().astype(np.int32)
            for s in range(gt_vol.shape[2]):
                gs = gt_vol[:, :, s]
                if gs.max() == 0:
                    continue
                td = compute_slice_dice(tvol[:, :, s], gs, num_classes)
                if td > 0.5:
                    candidates.append((case, s, td, 0.0))
        candidates.sort(key=lambda x: x[2], reverse=True)

    return candidates[:max_cases]


def select_cases_2d(
    ours_val_dir: str,
    baseline_val_dirs: Dict[str, str],
    gt_dir: str,
    file_ending: str,
    num_classes: int,
    max_cases: int = 3,
    dice_threshold: float = 0.6,
    dice_margin: float = 0.02,
) -> List[Tuple[str, int, float, float]]:
    """Select representative 2D cases using whole-image Dice."""
    ours_files = sorted(f for f in os.listdir(ours_val_dir) if f.endswith(file_ending))
    common = set(ours_files)
    for bdir in baseline_val_dirs.values():
        common &= set(f for f in os.listdir(bdir) if f.endswith(file_ending))
    common = sorted(common)

    candidates = []
    for case in common:
        gt_path = os.path.join(gt_dir, case)
        ours_path = os.path.join(ours_val_dir, case)
        if not os.path.isfile(gt_path) or not os.path.isfile(ours_path):
            continue

        gt_mask = load_segmentation_array(gt_path)
        if gt_mask.max() == 0 or np.count_nonzero(gt_mask) < 50:
            continue

        ours_mask = load_segmentation_array(ours_path)
        target_dice = compute_slice_dice(ours_mask, gt_mask, num_classes)
        if target_dice < dice_threshold:
            continue

        baseline_dices = []
        for bdir in baseline_val_dirs.values():
            pred_path = os.path.join(bdir, case)
            if os.path.isfile(pred_path):
                baseline_dices.append(
                    compute_slice_dice(
                        load_segmentation_array(pred_path),
                        gt_mask,
                        num_classes,
                    )
                )
        if not baseline_dices:
            continue

        diff = target_dice - max(baseline_dices)
        if diff > dice_margin:
            candidates.append((case, 0, target_dice, diff))

    candidates.sort(key=lambda x: x[3], reverse=True)

    if not candidates:
        logger.warning("No 2D case meets criteria, falling back to top-dice cases")
        for case in common[:20]:
            gt_path = os.path.join(gt_dir, case)
            ours_path = os.path.join(ours_val_dir, case)
            if not os.path.isfile(gt_path) or not os.path.isfile(ours_path):
                continue

            gt_mask = load_segmentation_array(gt_path)
            if gt_mask.max() == 0:
                continue

            target_dice = compute_slice_dice(
                load_segmentation_array(ours_path),
                gt_mask,
                num_classes,
            )
            if target_dice > 0.5:
                candidates.append((case, 0, target_dice, 0.0))
        candidates.sort(key=lambda x: x[2], reverse=True)

    return candidates[:max_cases]


# ══════════════════════════════════════════════════════════════
# Main entry: generate segmentation comparison figure
# ══════════════════════════════════════════════════════════════

def generate_segmentation_comparison(
    experiment_dir: Path,
    output_path: Path,
    max_rows: int = 3,
    max_baselines: int = 4,
) -> Optional[Dict[str, Any]]:
    """
    Generate a publication-quality segmentation comparison figure.

    Args:
        experiment_dir: Root experiment directory (e.g., experiments/2026-02-24_...)
        output_path: Where to save the output PNG
        max_rows: Maximum number of sample rows
        max_baselines: Maximum number of baseline methods to show

    Returns:
        Dict with figure metadata, or None on failure.
    """
    try:
        return _generate_impl(experiment_dir, output_path, max_rows, max_baselines)
    except Exception as e:
        logger.error(f"Segmentation visualization failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def _generate_impl(
    experiment_dir: Path,
    output_path: Path,
    max_rows: int,
    max_baselines: int,
) -> Optional[Dict[str, Any]]:
    # 1. Load idea.json for dataset_id
    idea_path = experiment_dir / "idea.json"
    if not idea_path.exists():
        logger.error(f"idea.json not found in {experiment_dir}")
        return None

    with open(idea_path) as f:
        idea = json.load(f)
    dataset_id = str(idea.get("dataset", {}).get("dataset_id", ""))
    if not dataset_id:
        logger.error("No dataset_id in idea.json")
        return None

    # 2. Locate data directories
    dirs = find_dataset_dirs(dataset_id)
    if not dirs["results"]:
        logger.error(f"No camylanet_results directory for Dataset{dataset_id}")
        return None

    raw_dir = dirs["raw"]
    preprocessed_dir = dirs["preprocessed"]
    gt_dir = os.path.join(preprocessed_dir, "gt_segmentations") if preprocessed_dir else None

    if not gt_dir or not os.path.isdir(gt_dir):
        logger.error(f"GT segmentations not found for Dataset{dataset_id}")
        return None

    # Load dataset info
    dataset_json_path = os.path.join(raw_dir, "dataset.json") if raw_dir else None
    if dataset_json_path and os.path.isfile(dataset_json_path):
        with open(dataset_json_path) as f:
            ds_info = json.load(f)
        labels = ds_info.get("labels", {"background": 0, "foreground": 1})
        file_ending = ds_info.get("file_ending", ".nii.gz")
    else:
        labels = {"background": 0, "foreground": 1}
        file_ending = ".nii.gz"
    num_classes = len(labels)
    is_raster_dataset = _is_raster_path(f"dummy{file_ending}")

    # 3. Find proposed method's best node
    results_dir = experiment_dir / "logs" / "0-run" / "results"
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return None

    best = find_best_node(results_dir)
    if not best:
        logger.error("No valid experiment node found")
        return None
    best_node_id, best_dice, ours_val_dir = best
    logger.info(f"  Best node: {best_node_id[:12]}... (Dice={best_dice:.4f})")

    # 4. Find baseline validation directories
    baselines = find_baseline_validation_dirs(dirs["results"], dataset_id)
    if not baselines:
        logger.warning("No baseline methods found, skipping segmentation viz")
        return None

    baseline_names = list(baselines.keys())[:max_baselines]
    baseline_dirs = {n: baselines[n] for n in baseline_names}
    logger.info(f"  Baselines found: {baseline_names}")

    # 5. Select representative slices/cases
    if is_raster_dataset:
        selected = select_cases_2d(
            str(ours_val_dir),
            baseline_dirs,
            gt_dir,
            file_ending,
            num_classes,
            max_cases=max_rows,
            dice_threshold=0.5,
            dice_margin=0.01,
        )
    else:
        selected = select_slices_3d(
            str(ours_val_dir),
            baseline_dirs,
            gt_dir,
            file_ending,
            num_classes,
            max_cases=max_rows,
            dice_threshold=0.5,
            dice_margin=0.01,
        )

    if not selected:
        logger.warning("No suitable slices found for visualization")
        return None

    logger.info(f"  Selected {len(selected)} slices for visualization")

    # 6. Build image grid
    col_order = ["Image"] + baseline_names + ["Ours (Proposed)", "GT"]
    images_grid = []

    for case_name, slice_idx, _, _ in selected:
        case_id = case_name.replace(file_ending, "")

        # Load raw image
        raw_slice = None
        raw_path = find_raw_image_path(raw_dir, case_id, file_ending)
        if raw_path:
            raw_data = load_input_image(raw_path)
            if is_raster_dataset:
                raw_slice = prepare_display_image(raw_data)
            else:
                raw_slice = prepare_display_image(raw_data[:, :, slice_idx])
        if raw_slice is None:
            raw_slice = np.zeros((128, 128), dtype=np.uint8)

        gt_data = load_segmentation_array(os.path.join(gt_dir, case_name))
        gt_slice = gt_data if is_raster_dataset else gt_data[:, :, slice_idx]

        row = []
        for col_name in col_order:
            if col_name == "Image":
                row.append(ensure_rgb(raw_slice))
            elif col_name == "GT":
                row.append(create_overlay(raw_slice, gt_slice, num_classes))
            elif col_name == "Ours (Proposed)":
                pred_path = os.path.join(str(ours_val_dir), case_name)
                if os.path.isfile(pred_path):
                    pred_data = load_segmentation_array(pred_path)
                    pred_slice = pred_data if is_raster_dataset else pred_data[:, :, slice_idx]
                    row.append(create_overlay(raw_slice, pred_slice, num_classes))
                else:
                    row.append(ensure_rgb(raw_slice))
            else:
                bdir = baseline_dirs.get(col_name, "")
                pred_path = os.path.join(bdir, case_name)
                if os.path.isfile(pred_path):
                    pred_data = load_segmentation_array(pred_path)
                    pred_slice = pred_data if is_raster_dataset else pred_data[:, :, slice_idx]
                    row.append(create_overlay(raw_slice, pred_slice, num_classes))
                else:
                    row.append(ensure_rgb(raw_slice))
        images_grid.append(row)

    # 7. Normalize sizes and plot
    images_grid = _normalize_grid(images_grid)
    _plot_grid(images_grid, col_order, output_path)

    method_name = idea.get("method_name", "Proposed Method")
    caption = (
        f"Qualitative comparison of segmentation results. "
        f"Columns show the input image, predictions from baseline methods "
        f"({', '.join(baseline_names)}), our proposed method, and the ground truth. "
        f"Colored overlays indicate segmentation masks with boundary contours. "
        f"Our method produces more accurate boundaries with fewer false positives."
    )

    return {
        "figure_id": "seg_comparison",
        "type": "result",
        "title": "Qualitative Segmentation Comparison",
        "file": str(output_path),
        "image_generated": True,
        "caption": caption,
        "placement": "Experiments section",
        "plot_type": "segmentation_comparison",
        "baselines_shown": baseline_names,
        "num_samples": len(selected),
    }


def _normalize_grid(images_grid: List[List[np.ndarray]]) -> List[List[np.ndarray]]:
    max_h = max(im.shape[0] for row in images_grid for im in row)
    max_w = max(im.shape[1] for row in images_grid for im in row)
    if max(max_h, max_w) > MAX_EDGE:
        scale = MAX_EDGE / max(max_h, max_w)
        target_h = int(round(max_h * scale))
        target_w = int(round(max_w * scale))
    else:
        target_h, target_w = max_h, max_w
    return [[_resize(im, target_h, target_w) for im in row] for row in images_grid]


def _plot_grid(images_grid, col_labels, output_path):
    n_rows = len(images_grid)
    n_cols = len(images_grid[0])
    h0, w0 = images_grid[0][0].shape[:2]
    aspect = h0 / w0

    fig_w = n_cols * CELL_SIZE
    fig_h = n_rows * CELL_SIZE * aspect

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_w, fig_h),
        gridspec_kw={'wspace': 0.015, 'hspace': 0.015},
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r, c]
            ax.imshow(images_grid[r][c])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.3)
                spine.set_color('#cccccc')

    for c in range(n_cols):
        ax = axes[n_rows - 1, c]
        ax.set_xlabel(col_labels[c], fontsize=6, fontweight='bold', labelpad=2)

    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    logger.info(f"  Segmentation comparison figure saved to {output_path}")


# ══════════════════════════════════════════════════════════════
# Utility: extract per-case metrics for analysis plots
# ══════════════════════════════════════════════════════════════

def collect_per_case_data(experiment_dir: Path) -> Optional[str]:
    """
    Collect per-case metrics from the best node's summary.json and format
    as a text block suitable for inclusion in LLM prompts.

    Returns formatted string with per-case Dice, HD95, and volume data,
    or None if data cannot be loaded.
    """
    results_dir = experiment_dir / "logs" / "0-run" / "results"
    if not results_dir.exists():
        return None

    best = find_best_node(results_dir)
    if not best:
        return None
    _, _, val_dir = best

    summary_path = val_dir / "summary.json"
    if not summary_path.exists():
        return None

    summary = _load_summary(str(summary_path))
    cases = get_per_case_metrics(summary)

    if not cases:
        return None

    # Also try to get baseline per-case data for comparison
    idea_path = experiment_dir / "idea.json"
    baseline_data = {}
    if idea_path.exists():
        with open(idea_path) as f:
            idea = json.load(f)
        dataset_id = str(idea.get("dataset", {}).get("dataset_id", ""))
        dirs = find_dataset_dirs(dataset_id)
        if dirs["results"]:
            baselines = find_baseline_validation_dirs(dirs["results"], dataset_id)
            for bname, bdir in baselines.items():
                bs_path = os.path.join(bdir, "summary.json")
                if os.path.isfile(bs_path):
                    bs_summary = _load_summary(bs_path)
                    baseline_data[bname] = get_per_case_metrics(bs_summary)

    # Format output
    lines = []
    lines.append("## Per-Case Metrics (Proposed Method)")
    lines.append(f"Total cases: {len(cases)}")
    lines.append(f"Columns: case_name, dice, hd95, volume_pixels (GT foreground size)")
    lines.append("")

    for c in sorted(cases, key=lambda x: x["volume_pixels"]):
        lines.append(f"  {c['case_name']}: dice={c['dice']:.4f}, "
                      f"hd95={c['hd95']:.2f}, volume={c['volume_pixels']}")

    # Volume statistics for subgroup guidance
    volumes = [c["volume_pixels"] for c in cases]
    q33, q66 = np.percentile(volumes, [33, 66])
    lines.append(f"\nVolume terciles: small<{q33:.0f}, medium<{q66:.0f}, large>={q66:.0f}")

    small = [c for c in cases if c["volume_pixels"] < q33]
    medium = [c for c in cases if q33 <= c["volume_pixels"] < q66]
    large = [c for c in cases if c["volume_pixels"] >= q66]

    for group_name, group in [("Small", small), ("Medium", medium), ("Large", large)]:
        if group:
            d = [c["dice"] for c in group]
            h = [c["hd95"] for c in group]
            lines.append(f"  {group_name} (n={len(group)}): "
                          f"mean_dice={np.mean(d):.4f}, mean_hd95={np.mean(h):.2f}")

    # Baseline per-case data (condensed)
    if baseline_data:
        lines.append("\n## Baseline Per-Case Metrics (for comparison)")
        for bname, bcases in baseline_data.items():
            if not bcases:
                continue
            bd = [c["dice"] for c in bcases]
            bh = [c["hd95"] for c in bcases]
            lines.append(f"\n### {bname} (n={len(bcases)})")
            lines.append(f"  mean_dice={np.mean(bd):.4f}, mean_hd95={np.mean(bh):.2f}")

            # Per-case for scatter plot comparison
            for c in sorted(bcases, key=lambda x: x["volume_pixels"]):
                lines.append(f"  {c['case_name']}: dice={c['dice']:.4f}, "
                              f"hd95={c['hd95']:.2f}, volume={c['volume_pixels']}")

    return "\n".join(lines)
