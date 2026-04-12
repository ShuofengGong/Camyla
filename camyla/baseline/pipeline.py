"""Idempotent baseline orchestration.

`ensure_baseline(dataset_id)` is the single entry point called by
`launch_camyla.py`. It checks whether baseline artifacts already exist under
`$camylanet_results` and runs screening + full training only when missing.
"""

import logging
import os
from pathlib import Path
from typing import List, Tuple

from camyla.baseline.run_best import run_best_trainers
from camyla.baseline.screen_trainers import screen_dataset, successful_trainers

logger = logging.getLogger(__name__)


def _results_root() -> Path:
    path = os.environ.get("camylanet_results")
    if not path:
        raise RuntimeError(
            "camylanet_results environment variable is not set. "
            "Point it at your baseline output directory (see CamylaNet README)."
        )
    return Path(path)


def _has_existing_baselines(dataset_id: int) -> bool:
    """True if any `Dataset{id}_*/{id}_*Trainer/experiment_data.npy` exists."""
    root = _results_root()
    if not root.exists():
        return False
    for matched in root.glob(f"Dataset{dataset_id}_*/{dataset_id}_*Trainer/experiment_data.npy"):
        if matched.exists():
            return True
    # Flat-layout fallback matching agent_manager's Pattern 2
    for matched in root.glob(f"{dataset_id}_*Trainer/experiment_data.npy"):
        if matched.exists():
            return True
    return False


def _screening_csv_path(dataset_id: int) -> Path:
    return _results_root() / "_screening" / f"dataset_{dataset_id}.csv"


def ensure_baseline(
    dataset_id: int,
    screening_epochs: int = 2,
) -> List[Tuple[str, str]]:
    """Guarantee baseline artifacts exist for `dataset_id`.

    Returns the list of (trainer_name, configuration) that have baselines.
    """
    if _has_existing_baselines(dataset_id):
        logger.info("Baseline artifacts already present for dataset %s; skipping", dataset_id)
        return []

    csv_path = _screening_csv_path(dataset_id)
    logger.info("Screening trainers for dataset %s -> %s", dataset_id, csv_path)
    screen_dataset(dataset_id, str(csv_path), num_epochs=screening_epochs)

    trainers = successful_trainers(dataset_id, str(csv_path))
    if not trainers:
        logger.warning("No trainers succeeded in screening for dataset %s", dataset_id)
        return []

    logger.info("Running full baseline training for %d trainers", len(trainers))
    run_best_trainers(dataset_id, trainers)
    return trainers
