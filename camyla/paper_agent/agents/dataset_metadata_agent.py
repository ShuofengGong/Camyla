#!/usr/bin/env python3
"""
DatasetMetadataAgent - extracts metadata from manually collected baseline PDFs.

Version 2.0: based on manual PDF collection rather than automated search.
"""

import json
import json_repair
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from pypdf import PdfReader

from camyla.paper_agent.agents.base_agent import BaseAgent
from camyla.paper_agent.func.dataset_config import DatasetConfig

logger = logging.getLogger(__name__)


class DatasetMetadataAgent(BaseAgent):
    """
    Extracts dataset metadata from manually collected baseline paper PDFs.

    Workflow:
    1. Read config.yaml for dataset info and aliases.
    2. Scan the baseline directory and identify task-mode suffixes in PDF filenames.
    3. Read each PDF and extract methods and metrics (including efficiency metrics).
    4. Aggregate methods by task mode.
    5. Generate metadata.json.
    """

    # Mapping from filename suffix to task mode
    SUFFIX_MAP = {
        "_fs": "fully_supervised",
        "_ss": "semi_supervised",
        "_ws": "weakly_supervised",
        "_da": "domain_adaptation",
        "_ssl": "self_supervised",
        "_is": "interactive",
        "_few": "few_shot",
        "_zero": "zero_shot"
    }

    def __init__(self):
        super().__init__()

    def run(
        self,
        dataset_name: str,
        dataset_dir: str,
        task_modes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Extract metadata from manually collected baseline PDFs and generate Markdown files.

        Args:
            dataset_name: Dataset name.
            dataset_dir: Dataset directory path (e.g., "dataset/TN3K").
            task_modes: Task mode list (optional; read from config.yaml).

        Returns:
            Simplified metadata dict (basic info plus MD file paths).
        """
        logger.info(f"Starting metadata extraction for dataset: {dataset_name}")
        logger.info(f"Dataset directory: {dataset_dir}")

        # Step 1: load configuration
        logger.info("Step 1/5: Loading dataset configuration...")
        config = DatasetConfig(dataset_dir)
        aliases = config.get_aliases()
        task_description = config.get_task_description()

        if not task_modes:
            task_modes = config.get_task_modes()

        logger.info(f"  Dataset aliases: {aliases}")
        logger.info(f"  Task modes: {task_modes}")

        # Step 2: scan the baseline directory and identify task modes
        logger.info("Step 2/5: Scanning baseline PDFs...")
        baseline_pdfs = self._scan_baseline_pdfs(config.get_baseline_dir())
        logger.info(f"  Found {len(baseline_pdfs)} PDF files")

        if len(baseline_pdfs) == 0:
            logger.warning("No baseline PDFs found! Returning minimal metadata.")
            return self._generate_fallback_metadata(config, task_modes)

        # Step 2.5: detect incremental changes
        logger.info("Step 2.5/6: Detecting incremental changes...")
        from func.pdf_tracker import PDFTracker
        import json

        tracker = PDFTracker()
        metadata_path = Path(dataset_dir) / "metadata.json"

        # Load previously processed PDF records
        processed_pdfs = {}
        existing_metadata = None
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    existing_metadata = json.load(f)
                processed_pdfs = tracker.load_processed_pdfs(existing_metadata, "baseline_results")
                logger.info(f"  Loaded {len(processed_pdfs)} previously processed PDFs")
            except Exception as e:
                logger.warning(f"  Failed to load existing metadata: {e}")
                processed_pdfs = {}

        # Detect changes
        new_pdfs, deleted_pdfs, unchanged_pdfs = tracker.detect_changes(
            config.get_baseline_dir(),
            processed_pdfs
        )

        # Log change summary
        logger.info(tracker.format_change_summary(new_pdfs, deleted_pdfs, unchanged_pdfs))

        # Handle deleted PDFs (warning only)
        if deleted_pdfs:
            logger.warning("  " + "=" * 60)
            logger.warning(f"  ⚠️  Detected {len(deleted_pdfs)} PDF file(s) removed from the baseline directory")
            logger.warning("  Results from these papers remain in the MD files")
            logger.warning("  Suggestion: for a full reset, delete the following files and re-run:")
            logger.warning(f"    - {metadata_path}")
            logger.warning(f"    - {Path(dataset_dir) / 'baseline_results'}/")
            logger.warning("  " + "=" * 60)

        # If there are no new PDFs, return the existing metadata
        if not new_pdfs:
            logger.info("  ✓ No new PDFs; skipping processing")
            if existing_metadata:
                return existing_metadata
            else:
                # Metadata does not exist but there are no new PDFs (an unlikely edge case)
                logger.warning("  Metadata missing but no new PDFs; generating basic metadata")

        # Step 3: extract basic dataset info
        logger.info("Step 3/6: Extracting basic dataset info...")
        basic_info = self._extract_basic_info(config)

        # Step 4: create the baseline_results directory
        baseline_results_dir = Path(dataset_dir) / "baseline_results"
        baseline_results_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"  Baseline results directory: {baseline_results_dir}")

        # Step 5: process baseline PDFs one by one and generate MD files
        logger.info("Step 4/6: Processing PDFs and generating MD files...")

        from func.baseline_md_generator import BaselineMDGenerator
        md_generator = BaselineMDGenerator()

        # Initialize MD files (if they do not exist)
        md_files_created = set()
        for task_mode in task_modes:
            md_path = baseline_results_dir / f"{task_mode}.md"
            if not md_path.exists():
                md_generator.create_baseline_md(task_mode, dataset_name, baseline_results_dir)
                md_files_created.add(task_mode)

        # Only process new PDFs
        pdfs_to_process = tracker.convert_to_list_format(new_pdfs) if new_pdfs else baseline_pdfs
        papers_processed = {mode: 0 for mode in task_modes}
        new_pdf_records = {}  # Records of newly processed PDFs

        if new_pdfs:
            logger.info(f"  Processing {len(new_pdfs)} new PDF(s)...")
        else:
            logger.info(f"  Initial run: processing all {len(baseline_pdfs)} PDF(s)...")

        for pdf_path, pdf_task_mode in pdfs_to_process:
            tag = "[NEW]" if new_pdfs else "[INITIAL]"
            logger.info(f"  Processing: {pdf_path.name} {tag} [Task: {pdf_task_mode}]")

            try:
                paper_markdown = self._extract_paper_experiment(
                    pdf_path=pdf_path,
                    dataset_name=dataset_name,
                    aliases=aliases,
                    task_mode=pdf_task_mode,
                    target_task_type=task_description
                )

                # Check whether to skip
                if paper_markdown.strip().startswith("[SKIP"):
                    logger.info(f"    Skipped: Dataset not used in this paper")
                    continue

                # Append to the corresponding MD file
                md_path = baseline_results_dir / f"{pdf_task_mode}.md"
                md_generator.append_paper_to_md(md_path, paper_markdown)
                papers_processed[pdf_task_mode] += 1
                logger.info(f"    ✓ Added to {md_path.name}")

                # Record the processed PDF (keyed by file hash)
                file_hash = tracker.calculate_file_hash(pdf_path)
                new_pdf_records[file_hash] = {
                    "filename": pdf_path.name,
                    "task_mode": pdf_task_mode,
                    "processed_at": datetime.now().isoformat(),
                    "file_size": pdf_path.stat().st_size,
                    "status": "completed"
                }

            except Exception as e:
                logger.error(f"  Error processing {pdf_path.name}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Merge processed-PDF records
        all_processed_pdfs = tracker.update_processed_pdfs(processed_pdfs, new_pdf_records)

        # Step 6: generate/update metadata.json
        logger.info("Step 5/6: Generating/updating metadata.json...")
        metadata = self._generate_simplified_metadata(
            config=config,
            basic_info=basic_info,
            task_modes=task_modes,
            papers_processed=papers_processed,
            baseline_results_dir=baseline_results_dir,
            processed_pdfs=all_processed_pdfs  # pass the merged processed_pdfs
        )

        logger.info(f"✓ Metadata extraction complete for {dataset_name}")
        logger.info(f"  Total papers processed in this run: {sum(papers_processed.values())}")
        logger.info(f"  Total papers in metadata: {len(all_processed_pdfs)}")
        for mode, count in papers_processed.items():
            if count > 0:
                logger.info(f"    - {mode}: {count} papers")

        return metadata


    def _scan_baseline_pdfs(
        self,
        baseline_dir: Path
    ) -> List[Tuple[Path, str]]:
        """
        Scan the baseline directory and return PDF paths with their task modes.

        Returns:
            List of (pdf_path, task_mode).
        """
        results = []

        pdf_files = sorted(baseline_dir.glob("*.pdf"))

        for pdf_file in pdf_files:
            task_mode = self._identify_task_mode_from_filename(pdf_file.name)
            results.append((pdf_file, task_mode))
            logger.info(f"    {pdf_file.name} -> {task_mode}")

        return results

    def _identify_task_mode_from_filename(self, pdf_filename: str) -> str:
        """
        Identify the task mode from a PDF filename.

        Args:
            pdf_filename: PDF filename (e.g., "01_Chen2018_UNet_fs.pdf").

        Returns:
            Task mode (e.g., "fully_supervised").
        """
        # Remove the .pdf suffix
        name_without_ext = pdf_filename.replace(".pdf", "")

        # Check each suffix
        for suffix, mode in self.SUFFIX_MAP.items():
            if name_without_ext.endswith(suffix):
                return mode

        # Default to fully_supervised
        logger.warning(
            f"No task mode suffix found in '{pdf_filename}', "
            f"defaulting to 'fully_supervised'"
        )
        return "fully_supervised"

    def _extract_basic_info(self, config: DatasetConfig) -> Dict[str, Any]:
        """Extract basic info from config or the dataset PDF."""
        basic_info = {
            "task_type": config.get_task_description(),
            "image_dimension": config.get_image_dimension(),
            "keywords": config.get_keywords()
        }

        # Try to extract more details from dataset.pdf
        dataset_pdf = config.get_dataset_pdf_path()
        if dataset_pdf:
            try:
                reader = PdfReader(dataset_pdf)
                text = ""
                for page in reader.pages[:3]:  # Read the first 3 pages
                    text += page.extract_text() + "\n"

                # An LLM call could be used here to extract more info.
                # For now, we rely on the config.
            except Exception as e:
                logger.warning(f"Could not read dataset PDF: {e}")

        return basic_info


    def _extract_paper_experiment(
        self,
        pdf_path: Path,
        dataset_name: str,
        aliases: List[str],
        task_mode: str,
        target_task_type: str = "segmentation"
    ) -> str:
        """
        Extract the full experiment information from a single PDF and return Markdown.

        Args:
            pdf_path: PDF file path.
            dataset_name: Dataset name.
            aliases: Dataset alias list.
            task_mode: Target task mode.
            target_task_type: Target task type (e.g., "segmentation", "classification").

        Returns:
            Markdown-formatted summary of the paper's experiments, ready to append to the MD file.
            If the paper does not use this dataset, returns "[SKIP: ...]".
        """
        try:
            # Read PDF content
            pdf_content = self._read_pdf_content(pdf_path)
            logger.info(f"    PDF content length: {len(pdf_content)} chars")

            # Prepare prompt
            aliases_str = ", ".join([f'"{a}"' for a in aliases])

            prompt = self.load_skill(
                "medical_segmentation/paper_experiment_extraction.md",
                dataset_name=dataset_name,
                dataset_aliases=aliases_str,
                task_mode=task_mode,
                target_task_type=target_task_type,
                pdf_content=pdf_content[:120000],  # 120k chars limit
                pdf_filename=pdf_path.name
            )

            response = self.chat(messages=[{"role": "user", "content": prompt}])

            # Clean the response (remove potential code-block wrappers)
            response = response.strip()
            if response.startswith("```markdown"):
                response = response[len("```markdown"):].strip()
            if response.startswith("```"):
                response = response[3:].strip()
            if response.endswith("```"):
                response = response[:-3].strip()

            logger.info(f"    Response length: {len(response)} chars")

            return response

        except Exception as e:
            logger.error(f"Error extracting experiment from {pdf_path.name}: {e}")
            raise


    def _read_pdf_content(self, pdf_path: Path, max_pages: int = 30) -> str:
        """
        Read the full PDF content.

        Args:
            pdf_path: PDF file path.
            max_pages: Maximum number of pages to read.

        Returns:
            PDF text content.
        """
        try:
            reader = PdfReader(pdf_path)
            text = ""

            num_pages = min(len(reader.pages), max_pages)

            for i in range(num_pages):
                page_text = reader.pages[i].extract_text()
                if page_text:
                    text += page_text + "\n"

            return text

        except Exception as e:
            logger.error(f"Error reading PDF {pdf_path}: {e}")
            return ""

    def _parse_json_response(
        self,
        response: str,
        task_mode: str
    ) -> List[Dict[str, Any]]:
        """
        Parse JSON returned by the LLM, handling markdown code blocks.

        Returns:
            Method list.
        """
        # First strip markdown code blocks (if any)
        cleaned = response.strip()
        if cleaned.startswith("```"):
            # Remove leading ```json or ```
            cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned, flags=re.MULTILINE)
            # Remove trailing ```
            cleaned = re.sub(r'\s*```\s*$', '', cleaned, flags=re.MULTILINE)
            cleaned = cleaned.strip()

        try:
            # Parse using json_repair
            result = json_repair.loads(cleaned)

            # If it's a list, return directly
            if isinstance(result, list):
                return result

            # If it's a dict
            if isinstance(result, dict):
                # Check whether it contains the task_mode key
                if task_mode in result:
                    return result[task_mode]
                # Otherwise check whether it's a single method object (unlikely)
                if "method_name" in result:
                    return [result]

            # Otherwise return an empty list
            logger.warning(f"Unexpected JSON structure: {type(result)}")
            return []

        except Exception as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.error(f"Raw response (first 1000 chars): {response[:1000]}")
            return []


    def _generate_simplified_metadata(
        self,
        config: DatasetConfig,
        basic_info: Dict[str, Any],
        task_modes: List[str],
        papers_processed: Dict[str, int],
        baseline_results_dir: Path,
        processed_pdfs: Dict[str, Dict[str, Any]]  # new parameter
    ) -> Dict[str, Any]:
        """Generate a simplified metadata.json (no longer includes aggregated method data)."""

        # Collect the MD files actually generated
        result_files = []
        task_modes_available = []

        for task_mode in task_modes:
            md_file = baseline_results_dir / f"{task_mode}.md"
            if md_file.exists():
                # Use a relative path
                relative_path = f"baseline_results/{task_mode}.md"
                result_files.append(relative_path)
                task_modes_available.append(task_mode)

        total_papers = len(processed_pdfs)  # use the length of processed_pdfs as the total

        metadata = {
            "dataset_name": config.get_dataset_name(),
            "full_name": config.get_full_name(),
            "analyzed_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),

            "basic_info": basic_info,

            "baseline_results": {
                "description": "Baseline results organized by task mode in Markdown files",
                "result_files": result_files,
                "total_papers_processed": total_papers,
                "task_modes_available": task_modes_available,
                "papers_by_task_mode": papers_processed,
                "processed_pdfs": processed_pdfs  # add PDF tracking info
            },

            "extraction_metadata": {
                "source": "manual_pdf_collection",
                "extraction_date": datetime.now().strftime("%Y-%m-%d"),
                "agent_version": "2.0_markdown_incremental"  # update version number
            },

            "update_policy": {
                "mode": "incremental",  # changed from manual_only to incremental
                "last_manual_update": datetime.now().isoformat(),
                "next_suggested_update": None
            }
        }

        return metadata

    def _generate_fallback_metadata(
        self,
        config: DatasetConfig,
        task_modes: List[str]
    ) -> Dict[str, Any]:
        """Generate minimal metadata when no PDFs are available."""
        return {
            "dataset_name": config.get_dataset_name(),
            "full_name": config.get_full_name(),
            "analyzed_at": datetime.now().isoformat(),
            "basic_info": {
                "task_type": config.get_task_description(),
                "image_dimension": config.get_image_dimension()
            },
            "task_specific_baselines": {
                mode: {"methods": []} for mode in task_modes
            },
            "_task_mode_note": "Task modes are extensible.",
            "extraction_metadata": {
                "source": "manual_pdf_collection",
                "pdfs_processed": 0,
                "note": "No baseline PDFs found"
            },
            "update_policy": {
                "mode": "manual_only",
                "last_manual_update": datetime.now().isoformat()
            }
        }
