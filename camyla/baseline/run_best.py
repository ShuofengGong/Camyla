"""Full baseline training for trainers that passed screening.

For each successful trainer, runs preprocessing + full training + evaluation
under CamylaNet and writes `experiment_data.npy` to the exact layout
`agent_manager._scan_baseline_results_dir` expects:

    $camylanet_results/Dataset{ID}_{Abbr}/{ID}_{TrainerName}/experiment_data.npy
    $camylanet_results/Dataset{ID}_{Abbr}/{ID}_{TrainerName}/model_results/
"""

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import camylanet

logger = logging.getLogger(__name__)


def run_best_trainers(
    dataset_id: int,
    trainers: List[Tuple[str, str]],
) -> Dict[str, Dict]:
    """Run full training for the given (trainer_name, configuration) list.

    Returns a dict keyed by `f"dataset_{id}_{trainer}"` with metrics + paths.
    Trainers that error out are logged and skipped; the rest still proceed.
    """
    experiment_data: Dict[str, Dict] = {}

    for trainer_name, configuration in trainers:
        exp_key = f"dataset_{dataset_id}_{trainer_name}"
        exp_name = f"{dataset_id}_{trainer_name}"
        experiment_data[exp_key] = {
            "trainer": trainer_name,
            "configuration": configuration,
            "metrics": {"train": [], "val": []},
            "result_folder": None,
            "epochs": [],
            "dice_scores": [],
            "hd95_scores": [],
        }

        try:
            logger.info("[%s] plan_and_preprocess", trainer_name)
            plans_identifier = camylanet.plan_and_preprocess(
                dataset_id=dataset_id,
                configurations=[configuration],
            )

            logger.info("[%s] training_network", trainer_name)
            result_folder, training_log = camylanet.training_network(
                dataset_id=dataset_id,
                configuration=configuration,
                trainer_class=trainer_name,
                plans_identifier=plans_identifier,
                exp_name=exp_name,
            )

            experiment_data[exp_key]["result_folder"] = result_folder
            experiment_data[exp_key]["epochs"] = training_log.get("epochs", [])
            experiment_data[exp_key]["metrics"]["train"] = training_log.get("train_losses", [])

            logger.info("[%s] evaluate", trainer_name)
            results = camylanet.evaluate(
                dataset_id=dataset_id,
                result_folder=result_folder,
                exp_name=exp_name,
            )

            # Restructure: .../{exp_name}/Trainer__Plans__Config/fold_0 -> .../{exp_name}/model_results
            fold_0_path = Path(result_folder)
            trainer_config_path = fold_0_path.parent
            base_exp_path = trainer_config_path.parent
            target = base_exp_path / "model_results"
            if fold_0_path.exists():
                if target.exists():
                    shutil.rmtree(target)
                shutil.move(str(fold_0_path), str(target))
                experiment_data[exp_key]["result_folder"] = str(target)
            try:
                if trainer_config_path.exists() and not any(trainer_config_path.iterdir()):
                    trainer_config_path.rmdir()
            except OSError as e:
                logger.warning("cleanup empty dir failed: %s", e)

            if "foreground_mean" in results:
                dice = results["foreground_mean"]["Dice"]
                hd95 = results["foreground_mean"]["HD95"]
                experiment_data[exp_key]["dice_scores"].append(dice)
                experiment_data[exp_key]["hd95_scores"].append(hd95)
                experiment_data[exp_key]["metrics"]["val"].append(
                    {"dice": dice, "hd95": hd95}
                )
                logger.info("[%s] Dice=%.4f HD95=%.4f", trainer_name, dice, hd95)
            else:
                logger.warning("[%s] foreground_mean missing in evaluate()", trainer_name)

            # Write artifact that agent_manager._scan_baseline_results_dir reads
            np.save(str(base_exp_path / "experiment_data.npy"), experiment_data[exp_key])
            logger.info("[%s] wrote %s", trainer_name, base_exp_path / "experiment_data.npy")

        except Exception as e:
            logger.error("[%s] failed: %s", trainer_name, e, exc_info=True)
            continue

    return experiment_data
