#!/usr/bin/env python3
"""
DatasetConfig - dataset configuration management class.

Handles reading and managing the config.yaml file in a dataset directory.
"""

import yaml
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)


class DatasetConfig:
    """
    Dataset configuration management class.

    Reads configuration from dataset/{DatasetName}/config.yaml, including
    dataset name, aliases, task modes, and related metadata.
    """

    def __init__(self, dataset_dir):
        """
        Initialize the dataset configuration.

        Args:
            dataset_dir: Dataset directory path (e.g., Path("dataset/TN3K")).
        """
        self.dataset_dir = Path(dataset_dir)
        self.config_path = self.dataset_dir / "config.yaml"
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """
        Load the config.yaml file.

        Returns:
            Configuration dict.

        Raises:
            FileNotFoundError: If config.yaml does not exist.
            yaml.YAMLError: If the YAML is malformed.
        """
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}\n"
                f"Please create config.yaml in {self.dataset_dir}"
            )

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            if not isinstance(config, dict):
                raise ValueError("config.yaml must contain a dictionary")

            logger.info(f"Loaded config from {self.config_path}")
            return config

        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML config: {e}")
            raise

    def get_dataset_name(self) -> str:
        """Return the dataset name."""
        return self.config.get("dataset_name", self.dataset_dir.name)

    def get_full_name(self) -> str:
        """Return the full dataset name."""
        return self.config.get("full_name", self.get_dataset_name())

    def get_task_description(self) -> str:
        """Return the task description."""
        return self.config.get("task_description", "Unknown task")

    def get_aliases(self) -> List[str]:
        """
        Return all dataset aliases.

        Returns:
            Alias list; the primary name is always included.
        """
        aliases = self.config.get("aliases", [])

        # Ensure the primary name is in the alias list
        main_name = self.get_dataset_name()
        if main_name not in aliases:
            aliases.insert(0, main_name)

        return aliases

    def get_image_dimension(self) -> str:
        """Return the image dimension (2D/3D)."""
        return self.config.get("image_dimension", "Unknown")

    def get_task_modes(self) -> List[str]:
        """
        Return the list of supported task modes.

        Returns:
            Task mode list (e.g., ["fully_supervised", "semi_supervised"]).
        """
        return self.config.get("task_modes", ["fully_supervised"])

    def get_keywords(self) -> List[str]:
        """Return the dataset keywords."""
        return self.config.get("keywords", [])

    def get_baseline_dir(self) -> Path:
        """
        Return the baseline directory path.

        Returns:
            Path object for the baseline directory.
        """
        baseline_dir = self.dataset_dir / "baseline"

        # Create the directory if it does not exist
        if not baseline_dir.exists():
            logger.warning(f"Baseline directory does not exist, creating: {baseline_dir}")
            baseline_dir.mkdir(parents=True, exist_ok=True)

        return baseline_dir

    def get_dataset_pdf_path(self) -> Optional[Path]:
        """
        Return the dataset documentation PDF path.

        Returns:
            PDF path, or None if it does not exist.
        """
        pdf_path = self.dataset_dir / "dataset.pdf"

        if pdf_path.exists():
            return pdf_path
        else:
            logger.warning(f"Dataset PDF not found: {pdf_path}")
            return None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration to a dict.

        Returns:
            Configuration info dict.
        """
        return {
            "dataset_name": self.get_dataset_name(),
            "full_name": self.get_full_name(),
            "task_description": self.get_task_description(),
            "aliases": self.get_aliases(),
            "image_dimension": self.get_image_dimension(),
            "task_modes": self.get_task_modes(),
            "keywords": self.get_keywords(),
            "baseline_dir": str(self.get_baseline_dir()),
            "dataset_pdf": str(self.get_dataset_pdf_path()) if self.get_dataset_pdf_path() else None
        }

    def __repr__(self) -> str:
        return f"DatasetConfig(dataset={self.get_dataset_name()}, dir={self.dataset_dir})"


def load_dataset_config(dataset_name: str, base_dir: str = "dataset") -> DatasetConfig:
    """
    Convenience function: load configuration by dataset name.

    Args:
        dataset_name: Dataset name (e.g., "TN3K").
        base_dir: Base dataset directory (default: "dataset").

    Returns:
        A DatasetConfig instance.

    Example:
        >>> config = load_dataset_config("TN3K")
        >>> print(config.get_aliases())
        ['TN3K', 'TN-3K', 'Thyroid Nodule 3K']
    """
    dataset_dir = Path(base_dir) / dataset_name
    return DatasetConfig(dataset_dir)
