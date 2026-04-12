from typing import Dict, Any, List
import logging
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class SectionAgent(BaseAgent):
    """
    Agent responsible for generating a single section of the paper.
    """
    
    def run(self, 
            section_name: str, 
            section_description: str, 
            research_idea: str,
            experimental_results: str,
            ablation_results: str,
            previous_sections_summary: str = "",
            figures_description: str = "",
            dataset_context: str = "") -> str:
        """
        Generate content for a specific section.
        
        Args:
            section_name: Name of the section (e.g., "Introduction")
            section_description: Description of what this section should contain
            research_idea: The research context
            experimental_results: Results to include
            ablation_results: Ablation studies to include
            previous_sections_summary: Context from previously generated sections
            figures_description: Available figures information (LaTeX code + metadata)
            dataset_context: Detailed dataset information to prevent hallucination
            
        Returns:
            LaTeX content for the section
        """
        logger.info(f"Generating section: {section_name}...")
        
        prompt = self.load_skill(
            "medical_segmentation/section.md",
            section_name=section_name,
            section_description=section_description,
            research_idea=research_idea,
            experimental_results=experimental_results,
            ablation_results=ablation_results,
            previous_context=previous_sections_summary,
            figures_description=figures_description,
            dataset_context=dataset_context,
        )
        
        response = self.chat(messages=[{"role": "user", "content": prompt}])
        
        # Clean up output (remove markdown blocks if present)
        import re
        cleaned_response = re.sub(r'```latex\s*', '', response)
        cleaned_response = re.sub(r'```\s*$', '', cleaned_response)
        
        return cleaned_response.strip()
