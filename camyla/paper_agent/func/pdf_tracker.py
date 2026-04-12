#!/usr/bin/env python3
"""
PDF Tracker - utility for tracking PDF file changes.

Detects additions and deletions of PDF files under the baseline directory to
support incremental updates. Uses the file-content hash as a unique identifier
to avoid reprocessing when files are renamed.
"""

import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class PDFTracker:
    """
    PDF file change tracker.

    Features:
    - Scan PDF files in a directory and compute their hashes.
    - Compare the current PDF list against processed records.
    - Identify new, deleted, and unchanged PDFs.
    - Manage processed_pdfs records.
    """

    # Task mode suffix mapping (kept in sync with DatasetMetadataAgent)
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
        pass

    def calculate_file_hash(self, file_path: Path, chunk_size: int = 8192) -> str:
        """
        Compute the MD5 hash of a PDF file.

        Args:
            file_path: PDF file path.
            chunk_size: Read chunk size in bytes.

        Returns:
            The file's MD5 hash string.
        """
        md5_hash = hashlib.md5()

        try:
            with open(file_path, 'rb') as f:
                while chunk := f.read(chunk_size):
                    md5_hash.update(chunk)
            return md5_hash.hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate hash for {file_path}: {e}")
            return ""

    def identify_task_mode(self, pdf_filename: str) -> str:
        """
        Identify the task mode from a PDF filename.

        Args:
            pdf_filename: PDF filename (e.g., "01_Chen2018_UNet_fs.pdf").

        Returns:
            Task mode string (e.g., "fully_supervised").
        """
        name_without_ext = pdf_filename.replace(".pdf", "")

        for suffix, mode in self.SUFFIX_MAP.items():
            if name_without_ext.endswith(suffix):
                return mode

        logger.warning(
            f"No task mode suffix found in '{pdf_filename}', "
            f"defaulting to 'fully_supervised'"
        )
        return "fully_supervised"

    def scan_pdfs(self, baseline_dir: Path) -> Dict[str, Dict[str, Any]]:
        """
        Scan all PDF files in the baseline directory.

        Args:
            baseline_dir: Baseline directory path.

        Returns:
            Dict of {file_hash: {filename, task_mode, file_path, file_size}}.
        """
        if not baseline_dir.exists():
            logger.warning(f"Baseline directory not found: {baseline_dir}")
            return {}

        pdf_records = {}
        pdf_files = sorted(baseline_dir.glob("*.pdf"))

        logger.info(f"Scanning {len(pdf_files)} PDF files in {baseline_dir.name}/")

        for pdf_path in pdf_files:
            try:
                file_hash = self.calculate_file_hash(pdf_path)

                if not file_hash:
                    logger.warning(f"Skipping {pdf_path.name} (hash calculation failed)")
                    continue

                task_mode = self.identify_task_mode(pdf_path.name)

                pdf_records[file_hash] = {
                    "filename": pdf_path.name,
                    "task_mode": task_mode,
                    "file_path": str(pdf_path),
                    "file_size": pdf_path.stat().st_size
                }

            except Exception as e:
                logger.error(f"Error scanning {pdf_path.name}: {e}")
                continue

        return pdf_records

    def detect_changes(
        self,
        baseline_dir: Path,
        processed_pdfs: Dict[str, Dict[str, Any]]
    ) -> Tuple[Dict[str, Dict], Dict[str, Dict], Dict[str, Dict]]:
        """
        Detect changes among PDF files.

        Args:
            baseline_dir: Baseline directory path.
            processed_pdfs: Already-processed PDF records (from metadata.json).
                Format: {file_hash: {filename, task_mode, processed_at, ...}}.

        Returns:
            Triple (new_pdfs, deleted_pdfs, unchanged_pdfs):
            - new_pdfs: Newly added PDFs {file_hash: {filename, task_mode, file_path, file_size}}.
            - deleted_pdfs: Removed PDFs {file_hash: {filename, task_mode, ...}}.
            - unchanged_pdfs: Unchanged PDFs {file_hash: {filename, task_mode, ...}}.
        """
        # Scan current PDFs
        current_pdfs = self.scan_pdfs(baseline_dir)

        # Extract hash sets
        current_hashes = set(current_pdfs.keys())
        processed_hashes = set(processed_pdfs.keys())

        # Compute differences
        new_hashes = current_hashes - processed_hashes
        deleted_hashes = processed_hashes - current_hashes
        unchanged_hashes = current_hashes & processed_hashes

        # Build the result
        new_pdfs = {h: current_pdfs[h] for h in new_hashes}
        deleted_pdfs = {h: processed_pdfs[h] for h in deleted_hashes}
        unchanged_pdfs = {h: current_pdfs[h] for h in unchanged_hashes}

        return new_pdfs, deleted_pdfs, unchanged_pdfs

    def create_pdf_record(
        self,
        pdf_path: Path,
        task_mode: str,
        status: str = "completed"
    ) -> Dict[str, Any]:
        """
        Create a PDF processing record.

        Args:
            pdf_path: PDF file path.
            task_mode: Task mode.
            status: Processing status (default: "completed").

        Returns:
            PDF record dict.
        """
        file_hash = self.calculate_file_hash(pdf_path)

        return {
            "filename": pdf_path.name,
            "task_mode": task_mode,
            "file_hash": file_hash,
            "processed_at": datetime.now().isoformat(),
            "file_size": pdf_path.stat().st_size,
            "status": status
        }

    def load_processed_pdfs(
        self,
        metadata: Dict[str, Any],
        section: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        Load processed-PDF records from metadata.

        Args:
            metadata: metadata.json content.
            section: Which section to read ("baseline_results" or "challenges").

        Returns:
            Processed-PDF dict {file_hash: {filename, task_mode, processed_at, ...}}.
        """
        if section not in metadata:
            return {}

        processed_pdfs = metadata[section].get("processed_pdfs", {})

        if not processed_pdfs:
            return {}

        # Detect format: the new format's keys are 32-char hex strings (MD5 hash).
        # The old format may key by filename (e.g., "paper.pdf").
        first_key = next(iter(processed_pdfs.keys()))

        # An MD5 hash is a 32-character hex string
        is_hash_format = len(first_key) == 32 and all(c in '0123456789abcdef' for c in first_key.lower())

        if not is_hash_format:
            logger.warning(f"Detected old format processed_pdfs in {section} (filename as key)")
            logger.warning(f"  First key: {first_key}")
            logger.warning(f"  Migration needed: Please delete metadata.json and regenerate")
            # Return an empty dict to force reprocessing and generate the new format
            return {}

        return processed_pdfs

    def format_change_summary(
        self,
        new_pdfs: Dict[str, Dict],
        deleted_pdfs: Dict[str, Dict],
        unchanged_pdfs: Dict[str, Dict]
    ) -> str:
        """
        Format a change summary string.

        Args:
            new_pdfs: Newly added PDFs.
            deleted_pdfs: Deleted PDFs.
            unchanged_pdfs: Unchanged PDFs.

        Returns:
            Formatted summary string.
        """
        lines = []
        lines.append(f"  PDF file change statistics:")
        lines.append(f"    Added: {len(new_pdfs)}")
        lines.append(f"    Deleted: {len(deleted_pdfs)}")
        lines.append(f"    Unchanged: {len(unchanged_pdfs)}")

        if new_pdfs:
            lines.append(f"  New PDFs:")
            for file_hash, info in new_pdfs.items():
                lines.append(f"    + {info['filename']} [{info['task_mode']}]")

        if deleted_pdfs:
            lines.append(f"  Deleted PDFs:")
            for file_hash, info in deleted_pdfs.items():
                lines.append(f"    - {info.get('filename', 'unknown')} [{info.get('task_mode', 'unknown')}]")

        return "\n".join(lines)

    def convert_to_list_format(
        self,
        pdf_dict: Dict[str, Dict]
    ) -> List[Tuple[Path, str]]:
        """
        Convert a PDF dict to list format (compatible with existing code).

        Args:
            pdf_dict: PDF dict {file_hash: {filename, task_mode, file_path, ...}}.

        Returns:
            List format [(pdf_path, task_mode), ...].
        """
        result = []

        for file_hash, info in pdf_dict.items():
            pdf_path = Path(info["file_path"])
            task_mode = info["task_mode"]
            result.append((pdf_path, task_mode))

        return result

    def update_processed_pdfs(
        self,
        existing_pdfs: Dict[str, Dict],
        new_records: Dict[str, Dict]
    ) -> Dict[str, Dict]:
        """
        Update the processed-PDF records.

        Args:
            existing_pdfs: Existing processed_pdfs.
            new_records: Newly processed PDF records.

        Returns:
            Merged processed_pdfs.
        """
        updated = existing_pdfs.copy()
        updated.update(new_records)
        return updated
