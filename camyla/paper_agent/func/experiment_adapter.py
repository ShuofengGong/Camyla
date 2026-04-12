import copy
import json
import logging
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

_EFFICIENCY_SENTINEL = "## Computational Efficiency"
_EFFICIENCY_WORSE_MARGIN = 1.25
_EFFICIENCY_BETTER_MARGIN = 0.90


def formalize_ablation_names(ablation_results: str) -> str:
    """Replace code-style ablation names with formal academic abbreviations.

    Uses an LLM call to identify informal names (snake_case, code identifiers)
    in the ablation results text and replace them with concise uppercase
    abbreviations.  The full meaning is inserted on first occurrence.

    The markdown table structure and all numerical values are preserved.
    """
    if not ablation_results or not ablation_results.strip():
        return ablation_results

    # Quick heuristic: skip if there are no obvious code-style names
    if not re.search(r'\b[a-z]+_[a-z]+', ablation_results):
        logger.info("No code-style ablation names detected, skipping formalization")
        return ablation_results

    try:
        from func.openrouter_client import OpenRouterClient
        client = OpenRouterClient()
    except Exception as e:
        logger.warning(f"Could not initialize LLM client for name formalization: {e}")
        return ablation_results

    prompt = (
        "Below is an ablation study section from an experiment report. "
        "Some experiment/component names use informal code-style naming "
        "(e.g. snake_case like \"direct_mult\", \"self_attn\", \"no_skip_conn\").\n\n"
        "Please:\n"
        "1. Identify all such informal names in tables and text.\n"
        "2. Replace each with a formal academic abbreviation (e.g. \"DM\" for "
        "\"direct_mult\", \"SA\" for \"self_attn\").\n"
        "3. On the FIRST occurrence of each abbreviation, include the full "
        "meaning in parentheses, e.g. \"DM (Direct Multiplication)\".\n"
        "4. Keep ALL numerical values, table structure, and markdown "
        "formatting EXACTLY unchanged.\n"
        "5. Only modify the experiment/component names, nothing else.\n"
        "6. The \"w/o\" prefix in table rows should be kept as-is, just "
        "replace the name after it.\n\n"
        "Input:\n"
        f"{ablation_results}\n\n"
        "Output the modified text directly with no explanation or code fences:"
    )

    try:
        response = client.chat(
            messages=[{"role": "user", "content": prompt}],
            model="deepseek/deepseek-v3.2",
            temperature=0.3,
        )
        result = response.content.strip() if response.content else ""
        if not result:
            logger.warning("LLM returned empty response for ablation name formalization")
            return ablation_results

        logger.info(f"Formalized ablation names ({len(ablation_results)} -> {len(result)} chars)")
        return result
    except Exception as e:
        logger.warning(f"Ablation name formalization failed: {e}")
        return ablation_results


def filter_baselines_in_report(run_dir: str, experimental_results_md: str) -> str:
    """Filter obviously poor baselines from the markdown results text.

    Loads ``{run_dir}/all_baselines.json``, removes entries with Dice == 0 or
    more than 20 % below the second-worst method, then rebuilds the
    ``#### All Baseline Model Results`` table in *experimental_results_md*.

    If ``all_baselines.json`` does not exist or no filtering is needed the
    original text is returned unchanged.
    """
    baselines_path = Path(run_dir) / "all_baselines.json"
    if not baselines_path.exists():
        return experimental_results_md

    try:
        with open(baselines_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        baselines_list = data.get("baselines", [])
    except Exception as e:
        logger.warning(f"Could not load all_baselines.json: {e}")
        return experimental_results_md

    if not baselines_list:
        return experimental_results_md

    # --- filter logic (same rules as _filter_bad_baselines) ---
    valid = [b for b in baselines_list if b.get("dice_score") and b["dice_score"] > 0]
    if len(valid) > 2:
        valid.sort(key=lambda x: x["dice_score"], reverse=True)
        second_worst_dice = valid[-2]["dice_score"]
        threshold = second_worst_dice * 0.8
        valid = [b for b in valid if b["dice_score"] >= threshold]

    removed = len(baselines_list) - len(valid)
    if removed == 0:
        return experimental_results_md

    logger.info(f"Filtered {removed} poor baseline(s) from report text")

    # --- rebuild the markdown table ---
    new_rows = ["#### All Baseline Model Results\n"]
    new_rows.append(
        "The following table contains **all** baseline models evaluated on "
        "this dataset. "
    )
    new_rows.append(
        "**IMPORTANT: Use ONLY these exact numbers for any SOTA comparison "
        "table in the paper. Do NOT fabricate or estimate results for models "
        "not listed here.**\n"
    )
    new_rows.append("| Rank | Model | Dice Score | HD95 |")
    new_rows.append("| :---: | :--- | :---: | :---: |")
    for rank, bl in enumerate(valid, 1):
        model_name = bl.get("model_name", "Unknown")
        dice = bl.get("dice_score")
        hd95 = bl.get("hd95_score")
        dice_str = f"{dice:.4f}" if dice is not None else "-"
        hd95_str = f"{hd95:.4f}" if hd95 is not None else "-"
        new_rows.append(f"| {rank} | {model_name} | {dice_str} | {hd95_str} |")

    # Try to preserve the proposed-method row that may follow the table
    proposed_match = re.search(
        r"(\| - \| \*\*Proposed Method\*\* \|[^\n]+)", experimental_results_md
    )
    if proposed_match:
        new_rows.append(proposed_match.group(1))

    new_table = "\n".join(new_rows) + "\n"

    # Replace the old table section
    pattern = re.compile(
        r"#### All Baseline Model Results\n.*?"
        r"(?=\n##|\n####|\Z)",
        re.DOTALL,
    )
    replaced = pattern.sub(new_table, experimental_results_md)
    if replaced == experimental_results_md:
        logger.warning("Could not locate baseline table in report text for replacement")
    return replaced


class ExperimentResultsAdapter:
    """
    Adapter to convert Camyla experiment results into formats expected by Paper Agent.
    
    Reads from unified experiment report files:
    - experiment_report.md: Contains methodology, results, ablation studies
    - research_summary.json: Contains best node code and metrics
    - ablation_summary.json: Contains ablation experiment details
    """
    
    def __init__(self, experiment_dir: str):
        self.experiment_dir = Path(experiment_dir)
        self.run_dir = self.experiment_dir / "logs" / "0-run"
        
        if not self.run_dir.exists():
            # Check if experiment_dir provided IS the 0-run dir
            if (Path(experiment_dir) / "experiment_report.md").exists():
                self.run_dir = Path(experiment_dir)
                self.experiment_dir = self.run_dir.parent.parent
            else:
                raise ValueError(f"Could not find run directory in {experiment_dir}")
        
        # Unified report paths
        self.report_path = self.run_dir / "experiment_report.md"
        self.research_summary_path = self.run_dir / "research_summary.json"
        self.ablation_summary_path = self.run_dir / "ablation_summary.json"
        
        # Validate required files exist
        if not self.report_path.exists():
            raise FileNotFoundError(f"experiment_report.md not found at {self.report_path}")
        
        # Cache for loaded data
        self._report_cache: Optional[str] = None
        self._research_summary_cache: Optional[Dict] = None
        self._ablation_summary_cache: Optional[List] = None
        
        # LLM client for section extraction
        self._llm_client = None
        
    def _get_llm_client(self):
        """Lazy initialization of LLM client."""
        if self._llm_client is None:
            from func.openrouter_client import OpenRouterClient
            self._llm_client = OpenRouterClient()
        return self._llm_client
    
    def _load_report(self) -> str:
        """Load and cache the experiment report."""
        if self._report_cache is None:
            self._report_cache = self.report_path.read_text(encoding='utf-8')
        return self._report_cache
    
    def _load_research_summary(self) -> Dict:
        """Load and cache research summary JSON."""
        if self._research_summary_cache is None:
            if self.research_summary_path.exists():
                with open(self.research_summary_path, 'r', encoding='utf-8') as f:
                    self._research_summary_cache = json.load(f)
            else:
                logger.warning(f"research_summary.json not found at {self.research_summary_path}")
                self._research_summary_cache = {}
        return self._research_summary_cache
    
    def _load_ablation_summary(self) -> List:
        """Load and cache ablation summary JSON."""
        if self._ablation_summary_cache is None:
            if self.ablation_summary_path.exists():
                with open(self.ablation_summary_path, 'r', encoding='utf-8') as f:
                    self._ablation_summary_cache = json.load(f)
            else:
                logger.warning(f"ablation_summary.json not found at {self.ablation_summary_path}")
                self._ablation_summary_cache = []
        return self._ablation_summary_cache
    
    def _extract_section_direct(self, start_marker: str, end_markers: list) -> str:
        """
        Extract a section from the report by finding start/end text markers.

        Uses direct string search instead of LLM extraction to avoid
        truncation issues with long sections (e.g. methodology with inline code).

        Args:
            start_marker: Text that marks the beginning of the section
                          (e.g. ``"### 2.1 Methodology"``).
            end_markers:  List of text strings, any of which marks the start of
                          the *next* section.  The first match wins.

        Returns:
            Extracted section content (including the start heading).
        """
        report = self._load_report()

        start_idx = report.find(start_marker)
        if start_idx == -1:
            return ""

        search_from = start_idx + len(start_marker)
        end_idx = len(report)
        for marker in end_markers:
            pos = report.find(marker, search_from)
            if pos != -1 and pos < end_idx:
                end_idx = pos

        return report[start_idx:end_idx].strip()
    
    def get_dataset_info(self) -> List[Dict]:
        """Load dataset info from idea.json (enhanced version with detailed descriptions)"""
        idea_path = self.experiment_dir / "idea.json"
        if not idea_path.exists():
            logger.warning(f"idea.json not found at {idea_path}. Constructing minimal dataset info.")
            return [{"name": "unknown_dataset", "task": "unknown_task"}]

        try:
            with open(idea_path, 'r', encoding='utf-8') as f:
                idea = json.load(f)
            
            dataset = idea.get("dataset", {})
            
            # Adapt to Paper Agent's dataset_info format with enhanced fields
            return [{
                # Basic info
                "name": dataset.get("name", "Unknown Dataset"),
                "full_name": dataset.get("full_name") or dataset.get("name", "Unknown Dataset"),
                "abbreviation": dataset.get("abbreviation", ""),
                "task": dataset.get("description", "Unknown Task"),
                "dataset_id": dataset.get("dataset_id", 0),
                "configuration": dataset.get("configuration", "default"),
                
                # Imaging info
                "modality": dataset.get("modality", "Unknown"),
                "image_dimension": dataset.get("image_dimension", "3D"),
                "anatomical_region": dataset.get("anatomical_region", "Unknown"),
                
                # Target structure
                "target_structure": dataset.get("target_structure", "Unknown"),
                "labels": dataset.get("labels", {}),
                "num_classes": dataset.get("num_classes", 2),
                
                # Data scale
                "num_samples": dataset.get("num_samples"),
                "num_images": dataset.get("num_images"),
                
                # Technical parameters
                "patch_size": dataset.get("patch_size"),
                "spacing": dataset.get("spacing"),
                "median_spacing": dataset.get("median_spacing"),
                "file_format": dataset.get("file_format", ".nii.gz"),
                
                # Detailed description (used for paper writing)
                "acquisition_info": dataset.get("acquisition_info", ""),
                "annotation_info": dataset.get("annotation_info", ""),
                "data_source": dataset.get("data_source", ""),
                "detailed_description": dataset.get("detailed_description", ""),
                "clinical_relevance": dataset.get("clinical_relevance", ""),
                
                # Metadata
                "reference": dataset.get("reference"),
                "license": dataset.get("license"),
            }]
        except Exception as e:
            logger.error(f"Error loading dataset info: {e}")
            return [{"name": "error_loading_dataset"}]
    
    def get_dataset_context(self) -> str:
        """
        Generate a formatted dataset context string for paper writing prompts.
        Provides factual information about the dataset.
        """
        dataset_info_list = self.get_dataset_info()
        if not dataset_info_list:
            return "Dataset information not available."
        
        dataset = dataset_info_list[0]
        
        # Build concise context
        lines = []
        
        # Basic info
        lines.append(f"Name: {dataset.get('full_name', 'Unknown')}")
        if dataset.get('abbreviation'):
            lines.append(f"Abbreviation: {dataset.get('abbreviation')}")
        lines.append(f"Task: {dataset.get('task', 'Unknown')}")
        lines.append(f"Modality: {dataset.get('modality', 'Unknown')}")
        lines.append(f"Anatomical Region: {dataset.get('anatomical_region', 'Unknown')}")
        lines.append(f"Image Dimension: {dataset.get('image_dimension', 'Unknown')}")
        
        # Target structure
        lines.append(f"Target Structure: {dataset.get('target_structure', 'Unknown')}")
        labels = dataset.get('labels', {})
        if labels:
            label_str = ", ".join([str(k) for k in labels.keys() if str(k).lower() != 'background'][:5])
            lines.append(f"Classes: {label_str} ({dataset.get('num_classes', len(labels))} classes)")
        
        # Data scale
        if dataset.get('num_samples'):
            lines.append(f"Number of Cases: {dataset.get('num_samples')}")
        if dataset.get('num_images'):
            lines.append(f"Number of Images: {dataset.get('num_images')}")
        
        # Detailed description (most important)
        if dataset.get('detailed_description'):
            lines.append(f"\nDescription: {dataset.get('detailed_description')}")
        
        # Additional info if available
        if dataset.get('acquisition_info'):
            lines.append(f"Acquisition: {dataset.get('acquisition_info')}")
        if dataset.get('annotation_info'):
            lines.append(f"Annotation: {dataset.get('annotation_info')}")
        if dataset.get('data_source'):
            lines.append(f"Source: {dataset.get('data_source')}")
        if dataset.get('clinical_relevance'):
            lines.append(f"Clinical Relevance: {dataset.get('clinical_relevance')}")
        
        return "\n".join(lines)

    def _sanitize_dataset_text_for_paper(self, text: str) -> str:
        """Remove challenge/synthetic wording from paper-facing dataset text."""
        if not text:
            return ""

        sanitized = text.strip()

        targeted_replacements = [
            (
                r"Synthetic dataset generated from the public Fetal Tissue Annotations \(FeTA\) dataset using the FaBiAN v2\.0 simulation framework\.",
                "Publicly available dataset based on the public Fetal Tissue Annotations (FeTA) resource.",
            ),
            (
                r"This dataset consists of 70 synthetic T2-weighted MRI scans of the developing fetal brain, generated using the FaBiAN v2\.0 numerical phantom\.",
                "This dataset consists of 70 T2-weighted MRI scans of the developing fetal brain.",
            ),
            (
                r"The images simulate realistic clinical low-resolution acquisitions from 1\.5T and 3T GE scanners\.\s*",
                "",
            ),
            (
                r"The synthetic data is derived from high-resolution annotations of real clinical subjects from the FeTA dataset, with gestational ages ranging from ([0-9.]+) to ([0-9.]+) weeks\.",
                r"The cohort spans gestational ages from \1 to \2 weeks.",
            ),
            (
                r"Synthetic images simulating 1\.5T/3T GE scanners using T2-weighted Single-Shot Fast Spin Echo \(SS-FSE\) sequences\.",
                "T2-weighted Single-Shot Fast Spin Echo (SS-FSE) MRI with 1.5T/3T GE scanner settings.",
            ),
            (
                r"simulated low-resolution series",
                "corresponding low-resolution series",
            ),
        ]
        for pattern, repl in targeted_replacements:
            sanitized = re.sub(pattern, repl, sanitized, flags=re.IGNORECASE)

        replacements = [
            (r"\bbenchmark challenge\b", "publicly available dataset"),
            (r"\bchallenge\b", "publicly available dataset"),
            (r"\bcompetition\b", "publicly available dataset"),
        ]
        for pattern, repl in replacements:
            sanitized = re.sub(pattern, repl, sanitized, flags=re.IGNORECASE)

        # Remove synthetic/simulation-heavy clauses from paper-facing summaries.
        sanitized = re.sub(r"\bsynthetic\b[, ]*", "", sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(
            r",?\s*generated using the FaBiAN v2\.0 numerical phantom",
            "",
            sanitized,
            flags=re.IGNORECASE,
        )
        sanitized = re.sub(
            r",?\s*using the FaBiAN v2\.0 simulation framework",
            "",
            sanitized,
            flags=re.IGNORECASE,
        )
        sanitized = re.sub(
            r",?\s*derived from high-resolution annotations of real clinical subjects from the FeTA dataset",
            "",
            sanitized,
            flags=re.IGNORECASE,
        )
        sanitized = re.sub(
            r"Part of the dataset was used in[^.]*\.\s*",
            "This is a publicly available dataset. ",
            sanitized,
            flags=re.IGNORECASE,
        )
        sanitized = re.sub(
            r"^This dataset consists of\s+an?\s+dataset of\s+",
            "This dataset consists of ",
            sanitized,
            flags=re.IGNORECASE,
        )
        sanitized = re.sub(r"\s+\.", ".", sanitized)
        sanitized = re.sub(r"\s{2,}", " ", sanitized).strip(" ,;")
        return sanitized

    def _sanitize_dataset_for_paper(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Create a paper-facing dataset dict with sanitized wording."""
        sanitized = copy.deepcopy(dataset)

        full_name = str(dataset.get("full_name") or "").strip()
        abbreviation = str(dataset.get("abbreviation") or "").strip()
        raw_name = str(dataset.get("name") or "").strip()

        if full_name and "synthetic" not in full_name.lower():
            display_name = full_name
        elif abbreviation:
            display_name = abbreviation
        elif full_name:
            display_name = full_name
        elif raw_name and not raw_name.isdigit():
            display_name = raw_name
        else:
            display_name = "Unknown Dataset"

        sanitized["full_name"] = self._sanitize_dataset_text_for_paper(display_name)
        sanitized["task"] = self._sanitize_dataset_text_for_paper(dataset.get("task", ""))
        sanitized["data_source"] = self._sanitize_dataset_text_for_paper(dataset.get("data_source", ""))
        sanitized["detailed_description"] = self._sanitize_dataset_text_for_paper(
            dataset.get("detailed_description", "")
        )
        sanitized["acquisition_info"] = self._sanitize_dataset_text_for_paper(
            dataset.get("acquisition_info", "")
        )
        sanitized["annotation_info"] = self._sanitize_dataset_text_for_paper(
            dataset.get("annotation_info", "")
        )
        sanitized["clinical_relevance"] = self._sanitize_dataset_text_for_paper(
            dataset.get("clinical_relevance", "")
        )

        if sanitized.get("data_source"):
            source_lower = sanitized["data_source"].lower()
            if "publicly available dataset" not in source_lower:
                sanitized["data_source"] = f"Publicly available dataset. {sanitized['data_source']}".strip()
        else:
            sanitized["data_source"] = "Publicly available dataset."

        return sanitized

    def get_dataset_context_for_paper(self) -> str:
        """Generate a paper-facing dataset context with sanitized wording."""
        dataset_info_list = self.get_dataset_info()
        if not dataset_info_list:
            return "Dataset information not available."

        dataset = self._sanitize_dataset_for_paper(dataset_info_list[0])

        lines = []
        lines.append(f"Name: {dataset.get('full_name', 'Unknown')}")
        if dataset.get('abbreviation'):
            lines.append(f"Abbreviation: {dataset.get('abbreviation')}")
        lines.append(f"Task: {dataset.get('task', 'Unknown')}")
        lines.append(f"Modality: {dataset.get('modality', 'Unknown')}")
        lines.append(f"Anatomical Region: {dataset.get('anatomical_region', 'Unknown')}")
        lines.append(f"Image Dimension: {dataset.get('image_dimension', 'Unknown')}")
        lines.append("Availability: Publicly available dataset")

        lines.append(f"Target Structure: {dataset.get('target_structure', 'Unknown')}")
        labels = dataset.get('labels', {})
        if labels:
            label_str = ", ".join([str(k) for k in labels.keys() if str(k).lower() != 'background'][:5])
            lines.append(f"Classes: {label_str} ({dataset.get('num_classes', len(labels))} classes)")

        if dataset.get('num_samples'):
            lines.append(f"Number of Cases: {dataset.get('num_samples')}")
        if dataset.get('num_images'):
            lines.append(f"Number of Images: {dataset.get('num_images')}")

        if dataset.get('detailed_description'):
            lines.append(f"\nDescription: {dataset.get('detailed_description')}")
        if dataset.get('acquisition_info'):
            lines.append(f"Acquisition: {dataset.get('acquisition_info')}")
        if dataset.get('annotation_info'):
            lines.append(f"Annotation: {dataset.get('annotation_info')}")
        if dataset.get('data_source'):
            lines.append(f"Source: {dataset.get('data_source')}")
        if dataset.get('clinical_relevance'):
            lines.append(f"Clinical Relevance: {dataset.get('clinical_relevance')}")

        return "\n".join(lines)

    def load_proposal(self) -> str:
        """
        Load the research proposal/methodology from experiment_report.md.

        Extracts everything from ``### 2.1 Methodology`` up to
        ``### 2.2 Experimental Results`` using direct text parsing
        (no LLM call, so no truncation risk).

        Returns:
            Methodology content as markdown string
        """
        logger.info("Loading methodology from experiment_report.md (direct extraction)...")

        methodology = self._extract_section_direct(
            "### 2.1 Methodology",
            ["### 2.2 Experimental Results", "## 3. Ablation"],
        )

        if not methodology:
            methodology = self._extract_section_direct(
                "## 2. Proposed Method",
                ["## 3. Ablation", "## 4."],
            )

        if not methodology:
            raise ValueError("Could not extract methodology section from experiment report")

        logger.info(f"Loaded methodology ({len(methodology)} chars)")
        return methodology

    def load_experimental_results(self) -> str:
        """
        Load experimental results from experiment_report.md.

        Returns:
            Experimental results as markdown string
        """
        logger.info("Loading experimental results from experiment_report.md (direct extraction)...")

        results = self._extract_section_direct(
            "### 2.2 Experimental Results",
            ["## 3. Ablation", "## 4."],
        )

        if not results:
            raise ValueError("Could not extract experimental results section from experiment report")

        logger.info(f"Loaded experimental results ({len(results)} chars)")
        return results

    def load_ablation_results(self) -> str:
        """
        Load ablation study results from experiment_report.md.

        Returns:
            Ablation results as markdown string
        """
        logger.info("Loading ablation results from experiment_report.md (direct extraction)...")

        ablation = self._extract_section_direct(
            "## 3. Ablation Studies",
            ["## 4. Conclusion", "## 5."],
        )

        if not ablation:
            logger.warning("Ablation studies section not found, returning empty string")
            return ""

        logger.info(f"Loaded ablation results ({len(ablation)} chars)")
        return ablation
    
    def load_best_node_code(self) -> str:
        """
        Load the best node's implementation code from research_summary.json.
        This code can be used as method content assistance for paper writing.
        
        Returns:
            Python code string of the best performing experiment
        """
        logger.info("Loading best node code from research_summary.json...")
        
        summary = self._load_research_summary()
        
        best_node = summary.get("best node", {})
        code = best_node.get("code", "")
        
        if not code:
            logger.warning("No code found in research_summary.json")
            return ""
        
        logger.info(f"Loaded best node code ({len(code)} characters)")
        return code
    
    def load_ablation_codes(self) -> List[Dict[str, str]]:
        """
        Load ablation experiment codes from ablation_summary.json.
        
        Returns:
            List of dicts with 'ablation_name' and 'code' keys
        """
        logger.info("Loading ablation codes from ablation_summary.json...")
        
        summary = self._load_ablation_summary()
        
        ablation_codes = []
        for item in summary:
            ablation_name = item.get("ablation_name", "Unknown")
            code = item.get("code", "")
            if code:
                ablation_codes.append({
                    "ablation_name": ablation_name,
                    "code": code
                })
        
        logger.info(f"Loaded {len(ablation_codes)} ablation experiment codes")
        return ablation_codes
    
    def load_full_report(self) -> str:
        """
        Load the complete experiment report.
        
        Returns:
            Full content of experiment_report.md
        """
        return self._load_report()
    
    def load_training_config(self) -> str:
        """
        Extract paper-facing training configuration from an available debug.json.

        Returns:
            Formatted training configuration string for paper writing prompts.
        """
        logger.info("Loading training config from debug.json...")
        
        results_dir = self.run_dir / "results"
        if not results_dir.exists():
            logger.warning(f"Results directory not found: {results_dir}")
            return "Training configuration not available."
        
        # Find the first result node with debug.json
        debug_data = None
        for node_dir in sorted(results_dir.iterdir()):
            debug_path = node_dir / "model_results" / "debug.json"
            if debug_path.exists():
                try:
                    with open(debug_path, 'r', encoding='utf-8') as f:
                        debug_data = json.load(f)
                    break
                except Exception as e:
                    logger.warning(f"Failed to parse {debug_path}: {e}")
                    continue
        
        if not debug_data:
            logger.warning("No debug.json found in any result node")
            return "Training configuration not available."
        
        # Extract key training parameters
        lines = []
        lines.append(f"Framework: nnU-Net")
        lines.append(f"Configuration: {debug_data.get('configuration_name', 'Unknown')}")
        lines.append(f"Number of Epochs: {debug_data.get('num_epochs', 'Unknown')}")
        lines.append(f"Iterations per Epoch: {debug_data.get('num_iterations_per_epoch', 'Unknown')}")
        lines.append(f"Batch Size: {debug_data.get('batch_size', 'Unknown')}")
        
        # Loss function
        loss_str = debug_data.get('loss', '')
        if 'DC_and_CE_loss' in loss_str:
            lines.append("Loss Function: Dice loss + Cross-Entropy loss (DC_and_CE_loss)")
        elif loss_str:
            loss_name = loss_str.split('(')[0].strip()
            lines.append(f"Loss Function: {loss_name}")
        
        # Optimizer
        opt_str = debug_data.get('optimizer', '')
        if opt_str:
            opt_name = opt_str.split('(')[0].strip()
            lines.append(f"Optimizer: {opt_name}")
        lines.append(f"Initial Learning Rate: {debug_data.get('initial_lr', 'Unknown')}")
        lines.append(f"Weight Decay: {debug_data.get('weight_decay', 'Unknown')}")
        
        # Extract momentum/nesterov from optimizer string
        if 'momentum' in opt_str:
            import re
            mom_match = re.search(r'momentum:\s*([\d.]+)', opt_str)
            nest_match = re.search(r'nesterov:\s*(True|False)', opt_str)
            if mom_match:
                lines.append(f"Momentum: {mom_match.group(1)}")
            if nest_match:
                lines.append(f"Nesterov: {nest_match.group(1)}")
        
        # Learning rate scheduler
        lr_sched = debug_data.get('lr_scheduler', '')
        if 'PolyLR' in lr_sched:
            lines.append("LR Scheduler: Polynomial decay (PolyLR)")
        
        # Patch size from configuration_manager
        config_mgr = debug_data.get('configuration_manager', '')
        if 'patch_size' in config_mgr:
            import re
            patch_match = re.search(r"'patch_size':\s*\[([^\]]+)\]", config_mgr)
            if patch_match:
                lines.append(f"Patch Size: [{patch_match.group(1)}]")
        
        lines.append(f"GPU: {debug_data.get('gpu_name', 'Unknown')}")
        lines.append(f"Deep Supervision: {debug_data.get('enable_deep_supervision', 'Unknown')}")
        lines.append(f"Foreground Oversampling: {debug_data.get('oversample_foreground_percent', 'Unknown')}")
        
        lines.append(f"Train/Test Split: 8:2")
        lines.append("Preprocessing: Standard nnU-Net preprocessing pipeline")
        lines.append("Postprocessing: Standard nnU-Net postprocessing")
        
        config_text = "\n".join(lines)
        logger.info(f"Loaded training config ({len(config_text)} chars)")
        return config_text

    def load_baseline_training_policy(self) -> str:
        """Return the paper-facing baseline-training policy description."""
        return (
            "All methods use the nnU-Net preprocessing and postprocessing pipeline. "
            "For non-nnU-Net baselines, method-specific architectural and "
            "hyperparameter settings follow the original implementation or the "
            "authors' recommended configuration when available. When an original "
            "implementation does not expose a specific setting, the official "
            "default configuration is used."
        )

    def get_best_metrics(self) -> Dict[str, float]:
        """
        Get the best node's metrics from research_summary.json.
        
        Returns:
            Dict mapping metric names to values
        """
        summary = self._load_research_summary()
        best_node = summary.get("best node", {})
        metric_data = best_node.get("metric", {}).get("value", {}).get("metric_names", [])
        
        metrics = {}
        for m in metric_data:
            name = m.get("metric_name", "Unknown")
            data = m.get("data", [{}])
            if data:
                metrics[name] = data[0].get("final_value", 0.0)
        
        return metrics

    def load_efficiency_metrics(self) -> str:
        """Load computational efficiency metrics from efficiency_metrics.json.

        Returns a formatted markdown table string suitable for inclusion in
        paper-writing prompts.  Returns an empty string if the file does not
        exist or contains no data.
        """
        rows = self._load_efficiency_rows()
        if not rows:
            return ""

        result = self._format_efficiency_rows(rows)
        logger.info(f"Loaded efficiency metrics ({len(result)} chars)")
        return result

    def get_efficiency_section_for_paper(self) -> Dict[str, Any]:
        """Return the paper-facing efficiency section and inclusion decision.

        The section is omitted from the paper prompt if the proposed method is
        clearly inefficient overall relative to the baselines.
        """
        rows = self._load_efficiency_rows()
        if not rows:
            return {
                "rows": [],
                "markdown": "",
                "include_in_paper": False,
                "decision_reason": "No efficiency rows available.",
            }

        include_in_paper, decision_reason = self._should_include_efficiency_in_paper(rows)
        markdown = self._format_efficiency_rows(rows)
        return {
            "rows": rows,
            "markdown": markdown if include_in_paper else "",
            "include_in_paper": include_in_paper,
            "decision_reason": decision_reason,
        }

    def _load_efficiency_rows(self) -> List[Dict[str, Any]]:
        """Load and normalize efficiency rows from efficiency_metrics.json."""
        eff_path = self.run_dir / "efficiency_metrics.json"
        if not eff_path.exists():
            logger.info("efficiency_metrics.json not found, skipping")
            return []

        try:
            with open(eff_path, "r", encoding="utf-8") as f:
                eff_data = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load efficiency_metrics.json: {e}")
            return []

        if not eff_data:
            return []

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

        if not rows:
            return []

        return rows

    def _format_efficiency_rows(self, rows: List[Dict[str, Any]]) -> str:
        """Format normalized efficiency rows into markdown."""
        if not rows:
            return ""

        lines = [
            f"{_EFFICIENCY_SENTINEL}\n",
            "| Model | Params (M) | FLOPs (G) | Inference Time (ms) |",
            "| :--- | :---: | :---: | :---: |",
        ]
        for row in rows:
            model_name = row["model_name"]
            params = row.get("params")
            flops = row.get("flops")
            inf_t = row.get("inference_time_s")
            p_str = f"{params / 1e6:.2f}" if params else "-"
            f_str = f"{flops / 1e9:.2f}" if flops else "-"
            t_str = f"{inf_t * 1000:.1f}" if inf_t else "-"
            lines.append(f"| {model_name} | {p_str} | {f_str} | {t_str} |")

        lines.append("")
        return "\n".join(lines)

    def _should_include_efficiency_in_paper(
        self, rows: List[Dict[str, Any]]
    ) -> tuple[bool, str]:
        """Decide whether the efficiency subsection should appear in the paper.

        Current policy is intentionally conservative: only hide the efficiency
        subsection when the proposed method is clearly worse overall, i.e. it is
        substantially worse than the baseline median on at least two metrics and
        does not show a clear efficiency advantage on any remaining metric.
        """
        proposed = next(
            (row for row in rows if row.get("model_name") == "Proposed Method"),
            None,
        )
        baselines = [
            row for row in rows
            if row.get("model_name") != "Proposed Method"
        ]

        if proposed is None:
            return True, "Proposed Method row not found; keeping efficiency section."
        if len(baselines) < 2:
            return True, "Not enough baselines for a reliable efficiency comparison."

        metric_specs = [
            ("params", "parameter count"),
            ("flops", "FLOPs"),
            ("inference_time_s", "inference time"),
        ]

        worse_metrics: List[str] = []
        better_metrics: List[str] = []
        compared_metrics: List[str] = []

        for metric_key, metric_name in metric_specs:
            proposed_value = proposed.get(metric_key)
            if not self._is_valid_efficiency_value(proposed_value):
                continue

            baseline_values = [
                row.get(metric_key) for row in baselines
                if self._is_valid_efficiency_value(row.get(metric_key))
            ]
            if len(baseline_values) < 2:
                continue

            baseline_median = self._median(baseline_values)
            compared_metrics.append(metric_name)

            if proposed_value > baseline_median * _EFFICIENCY_WORSE_MARGIN:
                worse_metrics.append(metric_name)
            elif proposed_value < baseline_median * _EFFICIENCY_BETTER_MARGIN:
                better_metrics.append(metric_name)

        if len(compared_metrics) < 2:
            return True, "Too few valid efficiency metrics for a reliable decision."

        if len(worse_metrics) >= 2 and not better_metrics:
            return (
                False,
                "Efficiency section omitted because Proposed Method is substantially "
                f"worse than the baseline median on {', '.join(worse_metrics)}.",
            )

        return (
            True,
            "Efficiency section kept because Proposed Method is not clearly "
            "inefficient overall.",
        )

    @staticmethod
    def _is_valid_efficiency_value(value: Any) -> bool:
        """Return True for numeric, finite, positive efficiency values."""
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value > 0
        )

    @staticmethod
    def _median(values: List[float]) -> float:
        """Compute a simple median without importing statistics."""
        sorted_values = sorted(values)
        mid = len(sorted_values) // 2
        if len(sorted_values) % 2 == 1:
            return float(sorted_values[mid])
        return float(sorted_values[mid - 1] + sorted_values[mid]) / 2.0
