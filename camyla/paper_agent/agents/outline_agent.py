import logging
import json_repair
import re
from typing import Dict, Any, List

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class OutlineAgent(BaseAgent):
    """
    Agent responsible for generating subsection outlines for paper sections.
    """
    
    def run(
        self,
        section_name: str,
        section_description: str,
        research_idea: str,
        previous_sections_summary: str,
        subsection_config: Dict[str, Any]
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        Generate a structured outline of subsections for a paper section.
        
        Args:
            section_name: Name of the section (e.g., "Method")
            section_description: Description of the section's purpose
            research_idea: The research context
            previous_sections_summary: Summary of previously generated sections
            subsection_config: Configuration with min/max subsections and target words
            
        Returns:
            Dictionary with:
                - subsections: List of subsection definitions
                  Each subsection has: name, description, focus
        """
        logger.info(f"Generating outline for section: {section_name}...")
        
        # Extract config parameters
        min_subsections = subsection_config.get("min_subsections", 3)
        max_subsections = subsection_config.get("max_subsections", 6)
        target_words = subsection_config.get("target_words_per_subsection", 350)
        
        # Load and populate prompt template
        prompt = self.load_skill(
            "common/generate_outline.md",
            section_name=section_name,
            section_description=section_description,
            research_idea=research_idea,
            previous_sections_summary=previous_sections_summary,
            min_subsections=str(min_subsections),
            max_subsections=str(max_subsections),
            target_words_per_subsection=str(target_words)
        )
        
        # Get LLM response
        response = self.chat(messages=[{"role": "user", "content": prompt}])
        
        # Parse JSON response
        # Try to find JSON block
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Assume entire response is JSON
            json_str = response.strip()
        
        outline = json_repair.loads(json_str)
        assert isinstance(outline, dict), "Parsed JSON is not a dictionary!"
        
        # Validate structure
        if "subsections" not in outline:
            logger.error("Outline missing 'subsections' field")
            raise ValueError("Invalid outline structure")
        
        logger.info(f"✓ Generated outline with {len(outline['subsections'])} subsections")
        for i, subsec in enumerate(outline['subsections'], 1):
            logger.info(f"  {i}. {subsec.get('name', 'Unnamed')}")
        
        return outline
