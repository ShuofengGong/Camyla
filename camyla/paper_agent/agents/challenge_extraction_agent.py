#!/usr/bin/env python3
"""
ChallengeExtractionAgent - extracts research challenges from baseline papers.

Extracts challenges and limitations from research paper PDFs and generates
Markdown files organized by task mode. Parallel to the baseline_results
structure, stored under the dataset/{NAME}/challenges/ directory.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from pypdf import PdfReader
from datetime import datetime

from camyla.paper_agent.agents.base_agent import BaseAgent
from camyla.paper_agent.func.challenge_md_generator import ChallengeMDGenerator

logger = logging.getLogger(__name__)


class ChallengeExtractionAgent(BaseAgent):
    """
    Agent responsible for extracting research challenges from baseline papers.
    
    Unlike DatasetAnalysisAgent which reads dataset PDFs,
    this agent extracts challenges from research papers (Introduction, Related Work, Discussion).
    """
    
    # Task mode suffix mapping (same as DatasetMetadataAgent)
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
        self.md_generator = ChallengeMDGenerator()
    
    def run(
        self,
        dataset_name: str,
        dataset_dir: str,
        task_modes: Optional[List[str]] = None,
        custom_instructions: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract research challenges from baseline PDFs and generate MD files under challenges/.

        Args:
            dataset_name: Dataset name.
            dataset_dir: Dataset directory path (e.g., "dataset/TN3K").
            task_modes: Task mode list (optional; inferred from existing PDFs).
            custom_instructions: User-provided instructions (optional; used to filter
                or emphasize specific challenge types).

        Returns:
            Dict with extraction statistics.
        """
        logger.info(f"Starting challenge extraction for dataset: {dataset_name}")
        logger.info(f"Dataset directory: {dataset_dir}")
        
        if custom_instructions:
            logger.info("  ✓ Custom instructions loaded for challenge extraction")
        
        dataset_path = Path(dataset_dir)
        baseline_dir = dataset_path / "baseline"
        challenges_dir = dataset_path / "challenges"
        
        # Step 1: scan baseline PDFs
        logger.info("Step 1/5: Scanning baseline PDFs...")
        baseline_pdfs = self._scan_baseline_pdfs(baseline_dir)
        logger.info(f"  Found {len(baseline_pdfs)} PDF files")
        
        if len(baseline_pdfs) == 0:
            logger.warning("No baseline PDFs found! Skipping challenge extraction.")
            return {"status": "skipped", "reason": "no PDFs found"}
        
        # Step 1.5: detect incremental changes
        logger.info("Step 1.5/5: Detecting incremental changes...")
        from func.pdf_tracker import PDFTracker
        import json
        
        tracker = PDFTracker()
        metadata_path = dataset_path / "metadata.json"
        
        # Load records of previously processed PDFs
        processed_pdfs = {}
        existing_metadata = None
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    existing_metadata = json.load(f)
                processed_pdfs = tracker.load_processed_pdfs(existing_metadata, "challenges")
                logger.info(f"  Loaded {len(processed_pdfs)} previously processed PDFs")
            except Exception as e:
                logger.warning(f"  Failed to load existing metadata: {e}")
                processed_pdfs = {}
        
        # Detect changes
        new_pdfs, deleted_pdfs, unchanged_pdfs = tracker.detect_changes(
            baseline_dir,
            processed_pdfs
        )
        
        # Log change summary
        logger.info(tracker.format_change_summary(new_pdfs, deleted_pdfs, unchanged_pdfs))

        # Handle deleted PDFs (warning only)
        if deleted_pdfs:
            logger.warning("  " + "=" * 60)
            logger.warning(f"  ⚠️  Detected {len(deleted_pdfs)} PDF file(s) removed from the baseline directory")
            logger.warning("  Challenges from these papers remain in the MD files")
            logger.warning("  " + "=" * 60)

        # If there are no new PDFs, return the existing state
        if not new_pdfs:
            logger.info("  ✓ No new PDFs; skipping processing")
            return {
                "status": "skipped",
                "reason": "no new PDFs",
                "dataset_name": dataset_name,
                "total_papers": len(processed_pdfs)
            }
        
        # Step 2: create the challenges directory
        challenges_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"  Challenges directory: {challenges_dir}")
        
        # Step 3: identify all task modes
        if not task_modes:
            task_modes = list(set(mode for _, mode in baseline_pdfs))
        logger.info(f"  Task modes found: {task_modes}")
        
        # Initialize MD files (if they do not exist)
        for task_mode in task_modes:
            md_path = challenges_dir / f"{task_mode}.md"
            if not md_path.exists():
                self.md_generator.create_challenge_md(task_mode, dataset_name, challenges_dir)
        
        # Step 4: process new PDFs
        logger.info("Step 2/5: Extracting challenges from PDFs...")

        # Only process newly added PDFs
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
                challenge_markdown = self._extract_challenges_from_pdf(
                    pdf_path=pdf_path,
                    dataset_name=dataset_name,
                    task_mode=pdf_task_mode,
                    custom_instructions=custom_instructions
                )
                
                # Check whether to skip
                if challenge_markdown.strip().startswith("[SKIP"):
                    logger.info(f"    Skipped: Dataset not used in this paper")
                    continue
                
                # Append to the corresponding MD file
                md_path = challenges_dir / f"{pdf_task_mode}.md"
                self.md_generator.append_paper_to_md(md_path, challenge_markdown)
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

        # Step 5: regenerate cross-paper analysis (if there are new PDFs)
        logger.info("Step 3/5: Generating/updating cross-paper analysis...")
        for task_mode in task_modes:
            md_path = challenges_dir / f"{task_mode}.md"
            if not md_path.exists():
                continue
            
            # Count total papers for this task mode
            total_papers_in_mode = sum(1 for h, info in all_processed_pdfs.items()
                                       if info.get('task_mode') == task_mode)

            # If there are multiple papers, regenerate the analysis (because new PDFs were added)
            if total_papers_in_mode > 1:
                try:
                    # Remove the old cross-paper analysis
                    content = md_path.read_text(encoding='utf-8')
                    if "## Cross-Paper Challenge Analysis" in content:
                        # Remove old analysis
                        lines = content.split('\n')
                        new_lines = []
                        skip = False
                        for line in lines:
                            if line.startswith("## Cross-Paper Challenge Analysis"):
                                skip = True
                            elif skip and line.startswith("---"):
                                skip = False
                                continue
                            if not skip:
                                new_lines.append(line)
                        content = '\n'.join(new_lines)
                        md_path.write_text(content, encoding='utf-8')
                        logger.info(f"    Removed old cross-paper analysis from {md_path.name}")
                    
                    # Generate new analysis
                    self._generate_cross_paper_analysis(md_path, dataset_name, task_mode)
                    logger.info(f"    ✓ Cross-paper analysis regenerated for {md_path.name}")
                except Exception as e:
                    logger.warning(f"    Failed to regenerate cross-paper analysis: {e}")
        
        # Step 6: update MD file metadata
        logger.info("Step 4/5: Updating MD file metadata...")
        for task_mode in task_modes:
            md_path = challenges_dir / f"{task_mode}.md"
            if md_path.exists():
                self.md_generator.update_md_metadata(md_path)
        
        # Step 7: update the challenges section of metadata.json
        logger.info("Step 5/5: Updating metadata.json...")
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}, creating new")
                metadata = {}
        else:
            metadata = {}
        
        # Update the challenges section
        if "challenges" not in metadata:
            metadata["challenges"] = {}
        
        metadata["challenges"]["processed_pdfs"] = all_processed_pdfs
        metadata["challenges"]["last_updated"] = datetime.now().isoformat()
        metadata["challenges"]["total_papers_processed"] = len(all_processed_pdfs)
        
        # Save metadata
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        total_papers = len(all_processed_pdfs)
        logger.info(f"✓ Challenge extraction complete for {dataset_name}")
        logger.info(f"  Total papers processed in this run: {sum(papers_processed.values())}")
        logger.info(f"  Total papers in metadata: {total_papers}")
        for mode, count in papers_processed.items():
            if count > 0:
                logger.info(f"    - {mode}: {count} papers")
        
        return {
            "status": "success",
            "dataset_name": dataset_name,
            "papers_processed": papers_processed,
            "total_papers": total_papers,
            "challenges_dir": str(challenges_dir)
        }
    
    def _scan_baseline_pdfs(self, baseline_dir: Path) -> List[Tuple[Path, str]]:
        """
        Scan the baseline directory for PDF files and identify their task mode.

        Returns:
            List of (pdf_path, task_mode) tuples.
        """
        if not baseline_dir.exists():
            logger.warning(f"Baseline directory not found: {baseline_dir}")
            return []
        
        pdfs = []
        for pdf_path in baseline_dir.glob("*.pdf"):
            task_mode = self._identify_task_mode_from_filename(pdf_path.name)
            pdfs.append((pdf_path, task_mode))
        
        return pdfs
    
    def _identify_task_mode_from_filename(self, pdf_filename: str) -> str:
        """Identify the task mode from a PDF filename."""
        name_without_ext = pdf_filename.replace(".pdf", "")
        
        for suffix, mode in self.SUFFIX_MAP.items():
            if name_without_ext.endswith(suffix):
                return mode
        
        # Default to fully_supervised
        logger.warning(
            f"No task mode suffix found in '{pdf_filename}', "
            f"defaulting to 'fully_supervised'"
        )
        return "fully_supervised"
    
    def _extract_challenges_from_pdf(
        self,
        pdf_path: Path,
        dataset_name: str,
        task_mode: str,
        custom_instructions: Optional[str] = None
    ) -> str:
        """
        Extract research challenges from a single PDF and return them in Markdown format.

        Args:
            pdf_path: PDF file path.
            dataset_name: Dataset name.
            task_mode: Target task mode.
            custom_instructions: User-provided instructions (optional).

        Returns:
            Challenge summary in Markdown format.
        """
        try:
            # Read PDF content
            pdf_content = self._read_pdf_content(pdf_path)
            logger.info(f"    PDF content length: {len(pdf_content)} chars")
            
            # Process custom instructions
            instruction_context = ""
            if custom_instructions:
                instruction_context = f"""
## Custom Research Guidance
When extracting research challenges, pay special attention to the following user-specified directions:
{custom_instructions}

Based on the instructions above, you should:
- Prioritize challenges relevant to the user's research direction
- Filter or de-emphasize challenge types unrelated to the user's instructions
- Take the user's research preferences and constraints into account when describing challenges
"""
            
            # Prepare prompt
            prompt = self.load_skill(
                "medical_segmentation/challenge_extraction.md",
                dataset_name=dataset_name,
                task_mode=task_mode,
                pdf_content=pdf_content[:100000],  # 100k chars limit
                pdf_filename=pdf_path.name,
                custom_instructions=instruction_context
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
            logger.error(f"Error extracting challenges from {pdf_path.name}: {e}")
            raise
    
    def _read_pdf_content(self, pdf_path: Path, max_pages: int = 30) -> str:
        """Read PDF content."""
        try:
            reader = PdfReader(pdf_path)
            text_parts = []
            
            for i, page in enumerate(reader.pages[:max_pages]):
                try:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
                except Exception as e:
                    logger.warning(f"Error extracting text from page {i+1}: {e}")
                    continue
            
            return "\n\n".join(text_parts)
            
        except Exception as e:
            logger.error(f"Error reading PDF {pdf_path}: {e}")
            return ""
    
    def _generate_cross_paper_analysis(
        self,
        md_path: Path,
        dataset_name: str,
        task_mode: str
    ) -> None:
        """
        Generate the Cross-Paper Challenge Analysis section after analyzing multiple papers.
        """
        # Read existing content
        content = md_path.read_text(encoding='utf-8')

        # Check whether a cross-paper analysis already exists
        if "## Cross-Paper Challenge Analysis" in content:
            logger.info("    Cross-paper analysis already exists, skipping")
            return
        
        # Use LLM to generate the analysis
        prompt = self.load_skill(
            "common/cross_paper_analysis.md",
            document_content=content[:50000]
        )
        
        response = self.chat(messages=[{"role": "user", "content": prompt}])
        
        # Clean the response
        response = response.strip()
        if response.startswith("```"):
            response = response.split("\n", 1)[1] if "\n" in response else response[3:]
        if response.endswith("```"):
            response = response[:-3].strip()
        
        # Remove the footer, add the analysis, and re-append the footer
        content = self.md_generator._remove_footer(content)
        content = content.strip() + "\n\n" + response + "\n\n"
        content += self.md_generator._generate_footer(
            self.md_generator.parse_paper_count(content)
        )
        
        md_path.write_text(content, encoding='utf-8')
