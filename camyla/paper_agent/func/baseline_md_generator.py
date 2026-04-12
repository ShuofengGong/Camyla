#!/usr/bin/env python3
"""
Baseline Markdown Generator

Utility for creating and managing baseline results markdown files.
Each MD file contains experiment summaries from multiple papers for a specific task mode.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class BaselineMDGenerator:
    """Generator for baseline results markdown files"""
    
    def __init__(self):
        pass
    
    def create_baseline_md(
        self,
        task_mode: str,
        dataset_name: str,
        output_dir: Path
    ) -> Path:
        """
        Create a new baseline results markdown file
        
        Args:
            task_mode: Task mode (e.g., "fully_supervised", "domain_adaptation")
            dataset_name: Name of the dataset
            output_dir: Directory to create the file in
            
        Returns:
            Path to the created file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        md_path = output_dir / f"{task_mode}.md"
        
        # Create initial content
        content = self._generate_header(task_mode, dataset_name)
        
        # Add metadata footer
        content += self._generate_footer(paper_count=0)
        
        md_path.write_text(content, encoding='utf-8')
        logger.info(f"Created baseline MD file: {md_path}")
        
        return md_path
    
    def append_paper_to_md(
        self,
        md_path: Path,
        paper_markdown: str
    ) -> None:
        """
        Append a paper's experiment summary to existing MD file
        
        Args:
            md_path: Path to the markdown file
            paper_markdown: Complete markdown section for the paper
                           (should start with "## Paper X:")
        """
        md_path = Path(md_path)
        
        if not md_path.exists():
            raise FileNotFoundError(f"MD file not found: {md_path}")
        
        # Read existing content
        existing_content = md_path.read_text(encoding='utf-8')
        
        # Remove old footer
        existing_content = self._remove_footer(existing_content)
        
        # Append new paper content
        # Ensure paper_markdown starts with newline and ends with newline
        paper_markdown = paper_markdown.strip()
        if not paper_markdown.endswith('---'):
            paper_markdown += '\n\n---'
        
        updated_content = existing_content + '\n\n' + paper_markdown + '\n\n'
        
        # Add updated footer
        new_paper_count = self.parse_paper_count(existing_content) + 1
        updated_content += self._generate_footer(paper_count=new_paper_count)
        
        # Write back
        md_path.write_text(updated_content, encoding='utf-8')
        logger.info(f"Appended paper to {md_path} (now {new_paper_count} papers)")
    
    def update_md_metadata(
        self,
        md_path: Path
    ) -> None:
        """
        Update the metadata footer of an MD file
        (last updated time, paper count)
        
        Args:
            md_path: Path to the markdown file
        """
        md_path = Path(md_path)
        
        if not md_path.exists():
            raise FileNotFoundError(f"MD file not found: {md_path}")
        
        content = md_path.read_text(encoding='utf-8')
        
        # Remove old footer
        content = self._remove_footer(content)
        
        # Count papers
        paper_count = self.parse_paper_count(content)
        
        # Add new footer
        content += self._generate_footer(paper_count=paper_count)
        
        md_path.write_text(content, encoding='utf-8')
        logger.info(f"Updated metadata for {md_path}")
    
    def parse_paper_count(self, content: str) -> int:
        """
        Parse the number of papers in the markdown content
        
        Args:
            content: Markdown file content
            
        Returns:
            Number of papers found
        """
        # Count occurrences of "## Paper"
        pattern = r'^## Paper\s+\d+:'
        matches = re.findall(pattern, content, re.MULTILINE)
        return len(matches)
    
    def _generate_header(self, task_mode: str, dataset_name: str) -> str:
        """Generate markdown file header"""
        task_mode_display = task_mode.replace('_', ' ').title()
        
        header = f"""# Baseline Results: {task_mode_display} on {dataset_name}

## Experiment Setting Summary
- **Task Mode**: {task_mode_display}
- **Dataset**: {dataset_name}
- **Purpose**: This file aggregates experiment results from multiple papers for the {task_mode} setting

---

"""
        return header
    
    def _generate_footer(self, paper_count: int) -> str:
        """Generate markdown file footer with metadata"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        footer = f"""
*Last Updated: {now}*  
*Total Papers: {paper_count}*
"""
        return footer
    
    def _remove_footer(self, content: str) -> str:
        """Remove existing footer from content"""
        # Remove lines starting with "*Last Updated:" and "*Total Papers:"
        lines = content.split('\n')
        
        # Find the footer start
        footer_start = None
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].startswith('*Last Updated:') or lines[i].startswith('*Total Papers:'):
                footer_start = i
        
        if footer_start is not None:
            # Remove footer and trailing empty lines
            lines = lines[:footer_start]
            # Remove trailing empty lines
            while lines and not lines[-1].strip():
                lines.pop()
        
        return '\n'.join(lines)
    
    def get_next_paper_number(self, md_path: Path) -> int:
        """
        Get the next paper number for a new entry
        
        Args:
            md_path: Path to the markdown file
            
        Returns:
            Next paper number (e.g., if 3 papers exist, returns 4)
        """
        if not md_path.exists():
            return 1
        
        content = md_path.read_text(encoding='utf-8')
        current_count = self.parse_paper_count(content)
        return current_count + 1
