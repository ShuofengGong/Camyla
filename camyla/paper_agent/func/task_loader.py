"""
Task configuration loader.

Provides unified task configuration loading and management.
Supports a uniform List[Dict] format for both single- and multi-dataset setups.
"""

import json
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class TaskConfig:
    """Task configuration data class."""

    def __init__(self, task_dir: Path):
        """
        Initialize the task configuration.

        Args:
            task_dir: Task directory path.
        """
        self.task_dir = Path(task_dir)

        if not self.task_dir.exists():
            raise FileNotFoundError(f"Task directory not found: {self.task_dir}")

        self.config = self._load_config()
        self.dataset_info = self._load_dataset_info()

        # Validate required files
        self._validate()

    def _load_config(self) -> Dict[str, Any]:
        """Load the task configuration file."""
        config_path = self.task_dir / "task_config.json"

        if not config_path.exists():
            raise FileNotFoundError(f"Task config not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        logger.info(f"Loaded task config: {config.get('task_name', 'unknown')}")
        return config

    def _load_dataset_info(self) -> List[Dict[str, Any]]:
        """
        Load dataset information.

        Supports two modes:
        1. Legacy mode: config["datasets"] is a list of filenames
           (e.g. ["tn3k_dataset.json"]) under the task directory.
        2. New mode: config["datasets"] is a list of objects
           (e.g. [{"name": "TN3K"}]) resolved to metadata.json under dataset/.
        """
        if "datasets" not in self.config:
            return []

        datasets = []

        for ds_item in self.config["datasets"]:
            # Case 1: legacy mode (string filename)
            if isinstance(ds_item, str):
                dataset_path = self.task_dir / ds_item
                if not dataset_path.exists():
                     # Try loading from the dataset directory (compatibility)
                     dataset_name = ds_item.replace("_dataset.json", "").replace(".json", "")
                     dataset_dir = Path("dataset") / dataset_name
                     metadata_path = dataset_dir / "metadata.json"

                     if metadata_path.exists():
                         datasets.append(self._load_metadata_as_dataset_info(metadata_path))
                         continue

                     raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

                with open(dataset_path, 'r', encoding='utf-8') as f:
                    datasets.append(json.load(f))

            # Case 2: new mode (config object)
            elif isinstance(ds_item, dict):
                dataset_name = ds_item.get("name")
                if not dataset_name:
                    logger.warning(f"Skipping invalid dataset config: {ds_item}")
                    continue

                dataset_dir = Path("dataset") / dataset_name
                metadata_path = dataset_dir / "metadata.json"

                if metadata_path.exists():
                    # Pass dataset_name so the name from config takes precedence over metadata
                    datasets.append(self._load_metadata_as_dataset_info(metadata_path, dataset_name))
                else:
                    # Fallback: try to read config.yaml
                    config_path = dataset_dir / "config.yaml"
                    if config_path.exists():
                         with open(config_path, 'r') as f:
                             config = yaml.safe_load(f)
                             datasets.append({
                                 "name": dataset_name,  # use the name from config
                                 "full_name": config.get("full_name", dataset_name),
                                 "task": config.get("task_description", "Unknown"),
                                 "task_modes": config.get("task_modes", [])
                             })
                    else:
                        logger.warning(f"No metadata found for dataset: {dataset_name}")

        logger.info(f"Loaded {len(datasets)} dataset(s)")
        return datasets

    def _load_metadata_as_dataset_info(
        self,
        metadata_path: Path,
        override_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Convert metadata.json to a dataset_info dict.

        Args:
            metadata_path: metadata.json file path.
            override_name: Dataset name that takes precedence (from config, not metadata).
        """
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        basic = metadata.get("basic_info", {})

        # Prefer override_name (from task_config.json); otherwise use dataset_name from metadata
        dataset_name = override_name if override_name else metadata.get("dataset_name")

        info = {
            "name": dataset_name,
            "full_name": metadata.get("full_name"),
            "task": basic.get("task_type", "Unknown"),
            "modalities": ["Ultrasound"],  # default; should be read from metadata in practice
            "image_dimension": basic.get("image_dimension", "2D"),
            "classes": ["nodule", "background"], # default
            "metrics": ["Dice", "IoU", "HD95"],
        }

        # Add baseline_results info (new format)
        baseline_results = metadata.get("baseline_results", {})
        if baseline_results:
            info["baseline_results_files"] = baseline_results.get("result_files", [])
            info["baseline_results_dir"] = str(metadata_path.parent / "baseline_results")
        else:
            # Legacy format fallback: if task_specific_baselines exists
            info["baselines"] = metadata.get("task_specific_baselines", {})

        # Populate additional fields where available
        if "keywords" in basic:
             info["keywords"] = basic["keywords"]

        return info

    def _validate(self):
        """Validate required files and configuration fields."""
        # Validate required fields
        required_fields = ["task_name", "topic", "pdfs", "datasets"]
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required field in task_config.json: {field}")

        # Validate that PDF files exist
        for pdf_path in self.config["pdfs"]:
            full_path = self.task_dir / pdf_path
            if not full_path.exists():
                raise FileNotFoundError(f"PDF file not found: {full_path}")

    def get_pdf_paths(self) -> List[Path]:
        """
        Return absolute paths of all PDF files.

        Returns:
            List of PDF file paths.
        """
        pdf_paths = []
        for pdf_path in self.config["pdfs"]:
            full_path = self.task_dir / pdf_path
            pdf_paths.append(full_path)

        return pdf_paths

    def get_dataset_pdf_paths(self) -> List[Path]:
        """
        Return absolute paths of dataset documentation PDFs (optional).

        Returns:
            List of dataset PDF file paths; empty list if none configured.
        """
        dataset_pdfs = self.config.get("dataset_pdfs", [])
        pdf_paths = []

        for pdf_path in dataset_pdfs:
            full_path = self.task_dir / pdf_path
            if full_path.exists():
                pdf_paths.append(full_path)
            else:
                logger.warning(f"Dataset PDF not found: {full_path}")

        return pdf_paths

    def get_topic(self) -> str:
        """Return the research topic."""
        return self.config["topic"]

    def get_image_dimension(self) -> str:
        """Return the image dimension (2D/3D)."""
        return self.config.get("image_dimension", "2D")

    def get_task_mode(self) -> str:
        """
        Return the task mode (fully_supervised, semi_supervised, domain_adaptation, ...).

        Returns:
            Task mode string; defaults to "fully_supervised".
        """
        return self.config.get("task_mode", "fully_supervised")

    def get_task_name(self) -> str:
        """Return the task name."""
        return self.config["task_name"]

    def get_style_path(self) -> Optional[Path]:
        """
        Return the style reference file path.

        Returns:
            Style file path, or None if it does not exist.
        """
        style_ref = self.config.get("style_reference")
        if not style_ref:
            return None

        style_path = self.task_dir / style_ref
        if not style_path.exists():
            logger.warning(f"Style reference not found: {style_path}")
            return None

        return style_path

    def get_output_dir(self) -> Path:
        """
        Return the output directory path.

        Returns:
            Output directory path.
        """
        output_dir = self.config.get("output_dir", "outputs")
        return self.task_dir / output_dir

    def get_dataset_info(self) -> List[Dict[str, Any]]:
        """Return the list of dataset info."""
        return self.dataset_info

    def get_dataset_count(self) -> int:
        """Return the number of datasets."""
        return len(self.dataset_info)

    def get_custom_instructions(self) -> Optional[str]:
        """
        Load the custom instructions file (optional).

        Users may create a custom_instructions.txt file in the task directory to
        provide research-direction guidance that controls the direction of idea
        generation.

        Returns:
            Custom instruction content, or None if the file is absent or empty.
        """
        instruction_path = self.task_dir / "custom_instructions.txt"

        if not instruction_path.exists():
            return None

        try:
            content = instruction_path.read_text(encoding='utf-8').strip()
            if content:
                logger.info(f"Loaded custom instructions from {instruction_path} ({len(content)} chars)")
                return content
            else:
                logger.info("Custom instructions file is empty")
                return None
        except Exception as e:
            logger.warning(f"Failed to read custom instructions from {instruction_path}: {e}")
            return None

    def get_baseline_results_content(self, dataset_name: str, task_mode: str = "fully_supervised") -> Optional[str]:
        """
        Return the baseline-results MD file content for the given dataset and task mode (new format).

        Args:
            dataset_name: Dataset name.
            task_mode: Task mode (fully_supervised, semi_supervised, ...).

        Returns:
            MD file text content, or None if absent.
        """
        # Find the matching dataset info
        target_ds = None
        for ds in self.dataset_info:
            if ds.get("name") == dataset_name:
                target_ds = ds
                break

        if not target_ds:
            logger.warning(f"Dataset {dataset_name} not found")
            return None

        # Try reading the new-format MD file
        baseline_results_dir = target_ds.get("baseline_results_dir")
        if not baseline_results_dir:
            logger.warning(f"No baseline_results_dir found for {dataset_name}, using old format")
            return None

        md_path = Path(baseline_results_dir) / f"{task_mode}.md"

        if not md_path.exists():
            logger.warning(f"Baseline MD file not found: {md_path}")
            return None

        try:
            content = md_path.read_text(encoding='utf-8')
            logger.info(f"Loaded baseline results from {md_path} ({len(content)} chars)")
            return content
        except Exception as e:
            logger.error(f"Error reading {md_path}: {e}")
            return None

    def get_baseline_methods(self, dataset_name: str, task_mode: str = "fully_supervised") -> List[Dict[str, Any]]:
        """
        Return baseline methods for the given dataset and task mode (legacy format, kept for back-compat).
        """
        # Find the matching dataset info
        target_ds = None
        for ds in self.dataset_info:
            if ds.get("name") == dataset_name:
                target_ds = ds
                break

        if not target_ds:
            return []

        # Get baselines from dataset_info (legacy format)
        baselines = target_ds.get("baselines", {})
        mode_data = baselines.get(task_mode, {})

        return mode_data.get("methods", [])

    def get_challenges_content(self, dataset_name: str, task_mode: Optional[str] = None) -> Optional[str]:
        """
        Return the research-challenges MD file content for the given dataset and task mode.

        Args:
            dataset_name: Dataset name.
            task_mode: Task mode; if None, use the task_mode from config.

        Returns:
            MD file text content, or None if absent.
        """
        if task_mode is None:
            task_mode = self.get_task_mode()

        dataset_dir = Path('dataset') / dataset_name
        challenge_file = dataset_dir / 'challenges' / f'{task_mode}.md'

        if not challenge_file.exists():
            logger.warning(f"Challenge file not found: {challenge_file}")
            return None

        try:
            content = challenge_file.read_text(encoding='utf-8')
            logger.info(f"Loaded challenges from {challenge_file} ({len(content)} chars)")
            return content
        except Exception as e:
            logger.error(f"Error reading {challenge_file}: {e}")
            return None

    def get_dataset_prompt(self) -> str:
        """
        Generate a dataset description for prompts (concise version, used for paper summaries).

        Returns:
            Formatted dataset description text.
        """
        if len(self.dataset_info) == 1:
            # Single dataset
            info = self.dataset_info[0]

            prompt_parts = [
                f"Dataset Context: {info.get('name', 'Unknown')}",
                f"Task: {info.get('task', 'N/A')}",
            ]

            if "modalities" in info:
                modalities = ", ".join(info["modalities"])
                prompt_parts.append(f"Modalities: {modalities}")

            if "classes" in info:
                classes = ", ".join(info["classes"])
                prompt_parts.append(f"Classes: {classes}")

            return "\n".join(prompt_parts)
        else:
            # Multiple datasets
            prompt_parts = [
                f"Multi-Dataset Context ({len(self.dataset_info)} datasets):"
            ]

            for i, info in enumerate(self.dataset_info, 1):
                prompt_parts.append(f"\nDataset {i}: {info.get('name', 'Unknown')}")
                prompt_parts.append(f"  Task: {info.get('task', 'N/A')}")

                if "modalities" in info:
                    modalities = ", ".join(info["modalities"])
                    prompt_parts.append(f"  Modalities: {modalities}")

            return "\n".join(prompt_parts)

    def get_dataset_context_for_verification(self) -> str:
        """
        Generate dataset context for research-proposal verification (academic tone).

        Emphasizes that the dataset serves as a validation benchmark rather than
        a design constraint.

        Returns:
            Academic-style dataset context description.
        """
        if len(self.dataset_info) == 1:
            # Single-dataset case
            return self._get_single_dataset_verification_context(self.dataset_info[0])
        else:
            # Multi-dataset case
            return self._get_multi_dataset_verification_context()

    def _get_single_dataset_verification_context(self, info: Dict[str, Any]) -> str:
        """Single-dataset verification context."""
        context_parts = [
            f"We will validate the proposed method on the following benchmark:",
            "",
            f"**Benchmark Dataset**: {info.get('name', 'Unknown')}",
        ]

        if "full_name" in info:
            context_parts.append(f"- **Full Name**: {info['full_name']}")

        if "task" in info:
            context_parts.append(
                f"- **Task**: {info['task']} (representative of Medical Image Segmentation challenges)"
            )

        context_parts.append("- **Key Characteristics**:")

        if "modalities" in info:
            modalities = ", ".join(info["modalities"])
            context_parts.append(f"  - Imaging modalities: {modalities}")

        if "classes" in info:
            classes = ", ".join(info["classes"])
            context_parts.append(f"  - Target structures: {classes}")

        if "metrics" in info:
            metrics = ", ".join(info["metrics"])
            context_parts.append(f"  - Standard evaluation: {metrics}")

        if "train_samples" in info or "val_samples" in info:
            context_parts.append("- **Dataset Scale**:")
            if "train_samples" in info:
                context_parts.append(f"  - Training: {info['train_samples']} cases")
            if "val_samples" in info:
                context_parts.append(f"  - Validation: {info['val_samples']} cases")

        if "background" in info:
            context_parts.extend([
                "",
                f"**Context**: {info['background']}"
            ])

        context_parts.extend([
            "",
            "**How to use this information**:",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "✓ Treat as a VALIDATION SCENARIO to demonstrate method effectiveness",
            "✓ Methods should be GENERAL SOLUTIONS applicable to similar segmentation tasks",
            "✓ Dataset-specific details belong in 'Experimental Validation' section",
            "✓ Frame your approach as solving BROAD CHALLENGES this dataset exemplifies",
            "✓ Avoid over-constraining the method design to this specific benchmark",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        ])

        return "\n".join(context_parts)

    def _get_multi_dataset_verification_context(self) -> str:
        """Multi-dataset verification context."""
        context_parts = [
            f"We will validate the proposed method on {len(self.dataset_info)} benchmark datasets:",
            ""
        ]

        for i, info in enumerate(self.dataset_info, 1):
            context_parts.extend([
                f"**Dataset {i}: {info.get('name', 'Unknown')}**",
                f"- Task: {info.get('task', 'N/A')}",
            ])

            if "modalities" in info:
                modalities = ", ".join(info["modalities"])
                context_parts.append(f"- Modalities: {modalities}")

            if "classes" in info:
                classes = ", ".join(info["classes"][:3])  # only show the first 3
                if len(info["classes"]) > 3:
                    classes += ", etc."
                context_parts.append(f"- Target classes: {classes}")

            context_parts.append("")  # blank-line separator

        context_parts.extend([
            "**Multi-Dataset Validation Strategy**:",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "✓ Demonstrate CROSS-DOMAIN GENERALIZATION across multiple benchmarks",
            "✓ Your method should be UNIVERSALLY APPLICABLE, not dataset-specific",
            "✓ Main experiments will validate on ALL datasets to prove robustness",
            "✓ Ablation studies will focus on the PRIMARY (first) dataset for detailed analysis",
            "✓ Emphasize method's ability to handle DIVERSE imaging conditions",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        ])

        return "\n".join(context_parts)

    def get_dataset_context_for_summary(self) -> str:
        """
        Generate dataset context for paper summaries (emphasizes generality).

        Returns:
            Dataset context emphasizing the general applicability of innovations.
        """
        if len(self.dataset_info) == 1:
            info = self.dataset_info[0]

            context_parts = [
                "Research Application Context:",
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                f"You are analyzing papers with potential applicability to {info.get('task', 'segmentation')} tasks.",
                "",
                "Example target scenario:",
            ]

            if "name" in info:
                context_parts.append(f"- Dataset: {info['name']}")

            if "modalities" in info:
                modalities = ", ".join(info["modalities"])
                context_parts.append(f"- Modalities: {modalities}")

            if "classes" in info:
                classes_brief = ", ".join(info["classes"][:3])
                if len(info["classes"]) > 3:
                    classes_brief += ", etc."
                context_parts.append(f"- Typical targets: {classes_brief}")
        else:
            context_parts = [
                "Research Application Context:",
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                f"You are analyzing papers with potential applicability to multi-domain segmentation.",
                "",
                f"Target scenarios ({len(self.dataset_info)} datasets):",
            ]

            for i, info in enumerate(self.dataset_info, 1):
                context_parts.append(f"  {i}. {info.get('name', 'Unknown')} - {info.get('task', 'N/A')}")

        context_parts.extend([
            "",
            "Your extraction focus:",
            "✓ Innovations applicable to SIMILAR tasks (not just these datasets)",
            "✓ General principles with TRANSFER POTENTIAL",
            "✓ Methodological contributions beyond dataset-specific optimizations",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        ])

        return "\n".join(context_parts)

    def get_dataset_full_description(self) -> str:
        """
        Generate a full detailed description of the dataset (used for experiment generation).

        Returns:
            Detailed dataset description text.
        """
        if len(self.dataset_info) == 1:
            return self._get_single_dataset_description(self.dataset_info[0])
        else:
            return self._get_multi_dataset_description()

    def _get_single_dataset_description(self, info: Dict[str, Any]) -> str:
        """Detailed description for a single dataset."""
        desc_parts = [
            f"# Dataset: {info.get('name', 'Unknown')}",
            f"**Full Name**: {info.get('full_name', 'N/A')}",
            f"**Task Type**: {info.get('task', 'N/A')}",
        ]

        if "modalities" in info:
            modalities = ", ".join(info["modalities"])
            desc_parts.append(f"**Imaging Modalities**: {modalities}")

        if "classes" in info:
            classes = "\n  - " + "\n  - ".join(info["classes"])
            desc_parts.append(f"**Segmentation Classes**:{classes}")

        if "metrics" in info:
            metrics = "\n  - " + "\n  - ".join(info["metrics"])
            desc_parts.append(f"**Evaluation Metrics**:{metrics}")

        desc_parts.append("\n**Dataset Statistics**:")
        if "train_samples" in info:
            desc_parts.append(f"  - Training: {info['train_samples']} samples")
        if "val_samples" in info:
            desc_parts.append(f"  - Validation: {info['val_samples']} samples")
        if "test_samples" in info:
            desc_parts.append(f"  - Test: {info['test_samples']} samples")
        if "resolution" in info:
            desc_parts.append(f"  - Resolution: {info['resolution']}")

        if "background" in info:
            desc_parts.append(f"\n**Background**: {info['background']}")

        if "references" in info:
            refs = "\n  - " + "\n  - ".join(info["references"])
            desc_parts.append(f"\n**References**:{refs}")

        return "\n".join(desc_parts)

    def _get_multi_dataset_description(self) -> str:
        """Detailed description for multiple datasets."""
        desc_parts = [
            f"# Multi-Dataset Validation ({len(self.dataset_info)} Datasets)",
            ""
        ]

        for i, info in enumerate(self.dataset_info, 1):
            desc_parts.extend([
                f"## Dataset {i}: {info.get('name', 'Unknown')}",
                f"- **Full Name**: {info.get('full_name', 'N/A')}",
                f"- **Task**: {info.get('task', 'N/A')}",
            ])

            if "modalities" in info:
                modalities = ", ".join(info["modalities"])
                desc_parts.append(f"- **Modalities**: {modalities}")

            if "metrics" in info:
                metrics = ", ".join(info["metrics"])
                desc_parts.append(f"- **Metrics**: {metrics}")

            if "train_samples" in info and "test_samples" in info:
                desc_parts.append(
                    f"- **Size**: {info['train_samples']} train / {info['test_samples']} test"
                )

            desc_parts.append("")  # blank-line separator

        return "\n".join(desc_parts)
