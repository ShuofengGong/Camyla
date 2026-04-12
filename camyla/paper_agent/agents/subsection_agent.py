import logging
import re
from typing import Dict, Any

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class SubsectionAgent(BaseAgent):
    """
    Agent responsible for generating detailed content for a single subsection.
    """
    
    def run(
        self,
        section_name: str,
        subsection_name: str,
        subsection_description: str,
        subsection_focus: str,
        research_idea: str,
        experimental_results: str = "",
        ablation_results: str = "",
        sibling_subsections_summary: str = "",
        previous_sections_summary: str = "",
        figures_description: str = "",
        dataset_context: str = "",
    ) -> str:
        """
        Generate LaTeX content for a specific subsection.
        
        Args:
            section_name: Parent section name (e.g., "Method")
            subsection_name: Name of this subsection
            subsection_description: Description of what this subsection should contain
            subsection_focus: Specific focus points for writing
            research_idea: The research context
            experimental_results: Experimental results (if applicable)
            ablation_results: Ablation study results (if applicable)
            sibling_subsections_summary: Summary of other subsections in same section
            previous_sections_summary: Summary of previous sections
            figures_description: Available figures information (LaTeX code + metadata)
            dataset_context: Detailed dataset information to prevent hallucination
            
        Returns:
            LaTeX content for the subsection (without \\subsection{} header)
        """
        logger.info(f"Generating subsection: {section_name} > {subsection_name}...")
        
        # Load and populate prompt template
        prompt = self.load_skill(
            "common/generate_subsection.md",
            section_name=section_name,
            subsection_name=subsection_name,
            subsection_description=subsection_description,
            subsection_focus=subsection_focus,
            research_idea=research_idea,
            experimental_results=experimental_results,
            ablation_results=ablation_results,
            sibling_subsections_summary=sibling_subsections_summary,
            previous_sections_summary=previous_sections_summary,
            figures_description=figures_description,
            dataset_context=dataset_context,
        )
        
        # Get LLM response
        response = self.chat(messages=[{"role": "user", "content": prompt}])
        
        # Clean up output (remove markdown blocks if present)
        cleaned_response = re.sub(r'```latex\s*', '', response)
        cleaned_response = re.sub(r'```\s*$', '', cleaned_response)
        cleaned_response = cleaned_response.strip()
        
        # Remove subsection header if LLM added it despite instructions
        pattern = rf'\\subsection\{{{re.escape(subsection_name)}\}}'
        cleaned_response = re.sub(pattern, '', cleaned_response, flags=re.IGNORECASE)
        cleaned_response = cleaned_response.strip()
        
        logger.info(f"✓ Generated {len(cleaned_response)} characters for {subsection_name}")
        
        return cleaned_response
