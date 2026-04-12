import logging
from pathlib import Path
from typing import Dict, Any

from camyla.paper_agent.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class ResultAnalysisAgent(BaseAgent):
    """
    Agent responsible for analyzing experimental results and planning ablation studies.
    """
    def run(self, research_idea: str, experimental_results: str) -> str:
        """
        Analyze experimental results and generate ablation study plan.
        
        Args:
            research_idea: The original research proposal/idea
            experimental_results: The core experimental results provided by the human researcher
            
        Returns:
            Analysis report with ablation study plan and visualization recommendations
        """
        logger.info("Analyzing experimental results...")
        
        # Step 1: generate the analysis report and the ablation study plan
        prompt = self.load_skill(
            "medical_segmentation/analysis.md",
            research_idea=research_idea,
            experimental_results=experimental_results
        )
        
        analysis_report = self.chat(messages=[{"role": "user", "content": prompt}])
        
        # Step 2: generate the figure plan
        logger.info("Generating plot planning...")
        plot_planning_prompt = self.load_skill(
            "medical_segmentation/plot_planning.md",
            research_idea=research_idea,
            experimental_results=experimental_results
        )
        
        plot_plan = self.chat(messages=[{"role": "user", "content": plot_planning_prompt}])
        
        # Step 3: append the figure plan to the analysis report
        combined_report = f"""{analysis_report}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Figure Planning

{plot_plan}
"""
        
        return combined_report
