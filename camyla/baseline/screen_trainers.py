"""Trainer compatibility screening.

Runs each applicable CamylaNet trainer on the target dataset for a small number
of epochs and records success/failure in a CSV file. Used as a prerequisite for
`run_best` so only trainers that actually run on the dataset get trained to
completion.
"""

import csv
import logging
import os
import time
import traceback
from datetime import datetime
from typing import Dict, List, Tuple

import camylanet
from camylanet import dataset_exists, get_available_configurations, plan_and_preprocess

logger = logging.getLogger(__name__)

TRAINERS_2D_AND_3D = [
    "SwinUNETRTrainer",
    "SegResNetTrainer",
    "UNetPlusPlusTrainer",
    "UMambaTrainer",
]

TRAINERS_3D_ONLY = [
    "MedNeXtTrainer",
    "ThreeDUXNetTrainer",
    "UNETRTrainer",
    "nnFormerTrainer",
    "STUNetTrainer",
]

TRAINERS_2D_ONLY = [
    "TransUNetTrainer",
    "UTNetTrainer",
    "SwinUMambaTrainer",
    "UKANTrainer",
]

FIELDNAMES = [
    "timestamp",
    "dataset_id",
    "dataset_type",
    "trainer_name",
    "configuration",
    "status",
    "error_message",
    "duration_seconds",
]


class ResultRecorder:
    def __init__(self, results_file: str):
        self.results_file = results_file
        os.makedirs(os.path.dirname(os.path.abspath(results_file)), exist_ok=True)
        if not os.path.exists(results_file):
            with open(results_file, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()
        self.completed = self._load_completed()

    def _load_completed(self) -> set:
        done = set()
        with open(self.results_file, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done.add((row["dataset_id"], row["trainer_name"], row["configuration"]))
        return done

    def is_completed(self, dataset_id: int, trainer_name: str, configuration: str) -> bool:
        return (str(dataset_id), trainer_name, configuration) in self.completed

    def record(
        self,
        dataset_id: int,
        dataset_type: str,
        trainer_name: str,
        configuration: str,
        status: str,
        error_message: str = "",
        duration_seconds: float = 0.0,
    ):
        row = {
            "timestamp": datetime.now().isoformat(),
            "dataset_id": dataset_id,
            "dataset_type": dataset_type,
            "trainer_name": trainer_name,
            "configuration": configuration,
            "status": status,
            "error_message": (error_message or "")[:500],
            "duration_seconds": f"{duration_seconds:.2f}",
        }
        with open(self.results_file, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)
        self.completed.add((str(dataset_id), trainer_name, configuration))


def _dataset_type(configurations: List[str]) -> str:
    if "3d_fullres" in configurations:
        return "3d"
    if "2d" in configurations:
        return "2d"
    return "unknown"


def _trainers_for(dataset_type: str) -> List[Tuple[str, str]]:
    if dataset_type == "3d":
        return [(t, "3d_fullres") for t in TRAINERS_3D_ONLY + TRAINERS_2D_AND_3D]
    if dataset_type == "2d":
        return [(t, "2d") for t in TRAINERS_2D_ONLY + TRAINERS_2D_AND_3D]
    return []


def _run_one(dataset_id: int, trainer_name: str, configuration: str,
             plans_identifier: str, num_epochs: int) -> Tuple[bool, str]:
    try:
        _, training_log = camylanet.training_network(
            dataset_id=dataset_id,
            configuration=configuration,
            trainer_class=trainer_name,
            plans_identifier=plans_identifier,
            num_epochs=num_epochs,
            exp_name=f"trainer_test_{trainer_name}",
        )
        if len(training_log.get("epochs", [])) >= 1:
            return True, ""
        return False, "empty training log"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def screen_dataset(dataset_id: int, results_file: str, num_epochs: int = 2) -> Dict:
    """Screen all applicable trainers on a dataset. Results appended to CSV."""
    if not dataset_exists(dataset_id):
        return {"skipped": True, "reason": "dataset_not_exists"}

    try:
        plans_identifier = plan_and_preprocess(dataset_id=dataset_id)
        configurations = get_available_configurations(dataset_id)
    except Exception as e:
        logger.error("Preprocess failed for %s: %s", dataset_id, e)
        return {"skipped": True, "reason": f"preprocess_failed: {e}"}

    dtype = _dataset_type(configurations)
    if dtype == "unknown":
        return {"skipped": True, "reason": "unknown_dataset_type"}

    recorder = ResultRecorder(results_file)
    trainers = _trainers_for(dtype)
    counts = {"success": 0, "failed": 0, "skipped": 0}

    for trainer_name, configuration in trainers:
        if recorder.is_completed(dataset_id, trainer_name, configuration):
            counts["skipped"] += 1
            continue

        t0 = time.time()
        ok, err = _run_one(dataset_id, trainer_name, configuration,
                           plans_identifier, num_epochs)
        duration = time.time() - t0
        recorder.record(
            dataset_id=dataset_id,
            dataset_type=dtype,
            trainer_name=trainer_name,
            configuration=configuration,
            status="success" if ok else "failed",
            error_message=err,
            duration_seconds=duration,
        )
        counts["success" if ok else "failed"] += 1
        logger.info("[%s] %s (%s) -> %s", dataset_id, trainer_name,
                    configuration, "ok" if ok else "fail")

    return counts


def successful_trainers(dataset_id: int, results_file: str) -> List[Tuple[str, str]]:
    """Return [(trainer_name, configuration), ...] of successful screenings,
    with nnUNetTrainer always prepended as default.
    """
    found: List[Tuple[str, str]] = []
    if os.path.exists(results_file):
        with open(results_file, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if int(row["dataset_id"]) == dataset_id and row["status"] == "success":
                    found.append((row["trainer_name"], row["configuration"]))

    default_config = found[0][1] if found else "3d_fullres"
    if not any(t == "nnUNetTrainer" for t, _ in found):
        found.insert(0, ("nnUNetTrainer", default_config))
    return found
