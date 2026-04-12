from camyla.paper_agent.agents.base_agent import BaseAgent
from camyla.paper_agent.func.experiment_adapter import ExperimentResultsAdapter
from pathlib import Path
from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)

class RealExperimentAgent(BaseAgent):
    """
    Agent that loads REAL experimental results from Camyla execution 
    instead of generating mock data.
    """
    
    def __init__(self, experiment_dir: Path):
        super().__init__()
        self.adapter = ExperimentResultsAdapter(str(experiment_dir))
        
    def run(self, research_idea: str, dataset_info: Optional[List[Dict]] = None) -> str:
        """
        Load experimental results.
        Ignores research_idea input as results are already pre-computed.
        """
        logger.info("Loading REAL experimental results from Camyla logs...")
        return self.adapter.load_experimental_results()

class RealAblationAgent(BaseAgent):
    """
    Agent that loads REAL ablation study results from Camyla execution.
    """
    
    def __init__(self, experiment_dir: Path):
        super().__init__()
        self.adapter = ExperimentResultsAdapter(str(experiment_dir))
        
    def run(self, research_idea: str, ablation_plan: str, main_results: str,
            dataset_info: Optional[List[Dict]] = None) -> str:
        """
        Load ablation results.
        Ignores inputs as results are already pre-computed.
        """
        logger.info("Loading REAL ablation results from Camyla logs...")
        return self.adapter.load_ablation_results()
