#!/usr/bin/env python3
"""
Challenge Markdown Generator

Utility for creating and managing challenge markdown files.
Each MD file contains research challenges from multiple papers for a specific task mode.
Similar to BaselineMDGenerator but for challenges.
"""

import re
from pathlib import Path
from typing import Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ChallengeMDGenerator:
    """Generator for challenge markdown files"""
    
    def __init__(self):
        pass
    
    def create_challenge_md(
        self,
        task_mode: str,
        dataset_name: str,
        output_dir: Path
    ) -> Path:
        """
        Create a new challenge markdown file
        
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
        logger.info(f"Created challenge MD file: {md_path}")
        
        return md_path
    
    def append_paper_to_md(
        self,
        md_path: Path,
        paper_markdown: str
    ) -> None:
        """
        Append a paper's challenge summary to existing MD file
        
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
        
        # Remove cross-paper analysis if exists (will be regenerated)
        if "## Cross-Paper Challenge Analysis" in existing_content:
            existing_content = existing_content.split("## Cross-Paper Challenge Analysis")[0].strip()
        
        # Append new paper content
        paper_markdown = paper_markdown.strip()
        if not paper_markdown.endswith('---'):
            paper_markdown += '\n\n---'
        
        updated_content = existing_content + '\n\n' + paper_markdown + '\n\n'
        
        # Add updated footer
        new_paper_count = self.parse_paper_count(updated_content)
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
        pattern = r'^## Paper\s+\d*:?'
        matches = re.findall(pattern, content, re.MULTILINE)
        return len(matches)
    
    def _generate_header(self, task_mode: str, dataset_name: str) -> str:
        """Generate markdown file header"""
        task_mode_display = task_mode.replace('_', ' ').title()
        
        header = f"""# Research Challenges: {task_mode_display} on {dataset_name}

## Challenge Summary

This document aggregates research challenges and limitations identified from papers targeting **{task_mode_display}** tasks on the **{dataset_name}** dataset.

**Purpose**: 
- Understand current limitations in the field
- Identify research gaps and opportunities
- Inform novel research directions

---

"""
        return header
    
    def _generate_footer(self, paper_count: int) -> str:
        """Generate markdown file footer with metadata"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        footer = f"""
*Last Updated: {now}*  
*Total Papers Analyzed: {paper_count}*
"""
        return footer
    
    def _remove_footer(self, content: str) -> str:
        """Remove existing footer from content"""
        lines = content.split('\n')
        
        # Find the footer start
        footer_start = None
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].startswith('*Last Updated:') or lines[i].startswith('*Total Papers'):
                footer_start = i
        
        if footer_start is not None:
            lines = lines[:footer_start]
            while lines and not lines[-1].strip():
                lines.pop()
        
        return '\n'.join(lines)
    
    def get_next_paper_number(self, md_path: Path) -> int:
        """
        Get the next paper number for a new entry
        
        Args:
            md_path: Path to the markdown file
            
        Returns:
            Next paper number
        """
        if not md_path.exists():
            return 1
        
        content = md_path.read_text(encoding='utf-8')
        current_count = self.parse_paper_count(content)
        return current_count + 1
