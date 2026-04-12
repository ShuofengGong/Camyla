import logging
from typing import Dict, Any, Optional, List
from camyla.paper_agent.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class MockExperimentAgent(BaseAgent):
    """
    Mock agent that simulates experimental results based on a research idea.
    This bridges Part 1 (Idea Generation) and Part 2 (Result Analysis).
    """
    
    def run(self, research_idea: str, dataset_info: Optional[List[Dict]] = None) -> str:
        """
        Generate simulated experimental results.
        
        Args:
            research_idea: The research proposal from Part 1
            dataset_info: List of dataset information dictionaries
            
        Returns:
            Formatted experimental results as text
        """
        logger.info("Generating mock experimental results...")
        
        if dataset_info is None or len(dataset_info) == 0:
            raise ValueError("At least one dataset is required")
        
        # Generate dataset description based on count
        if len(dataset_info) == 1:
            # Single dataset
            ds = dataset_info[0]
            dataset_desc = f"""Dataset: {ds.get('name', 'Unknown')}
- Full Name: {ds.get('full_name', 'N/A')}
- Task: {ds.get('task', 'N/A')}
- Modalities: {', '.join(ds.get('modalities', []))}
- Classes: {', '.join(ds.get('classes', []))}
- Evaluation Metrics: {', '.join(ds.get('metrics', []))}
- Training Samples: {ds.get('train_samples', 'N/A')}
- Validation Samples: {ds.get('val_samples', 'N/A')}"""
        else:
            # Multiple datasets
            dataset_parts = [f"Multiple Datasets (Total: {len(dataset_info)}):\n"]
            for i, ds in enumerate(dataset_info, 1):
                dataset_parts.append(f"""Dataset {i}: {ds.get('name', 'Unknown')}
- Task: {ds.get('task', 'N/A')}
- Modalities: {', '.join(ds.get('modalities', []))}
- Metrics: {', '.join(ds.get('metrics', []))}
- Samples: {ds.get('train_samples', 'N/A')} train / {ds.get('test_samples', 'N/A')} test
""")
            dataset_desc = "\n".join(dataset_parts)
        
        # Try to load baseline results from MD files (new format)
        baseline_context = ""
        try:
            from func.task_loader import TaskConfig
            # Get the first dataset's baseline results
            dataset_name = dataset_info[0].get('name')
            baseline_results_dir = dataset_info[0].get('baseline_results_dir')
            
            if baseline_results_dir:
                # Try to read fully_supervised MD file
                from pathlib import Path
                md_path = Path(baseline_results_dir) / "fully_supervised.md"
                if md_path.exists():
                    baseline_md_content = md_path.read_text(encoding='utf-8')
                    baseline_context = f"""
## Reference: Existing Baseline Results on {dataset_name}

{baseline_md_content}

**IMPORTANT: The above baseline results are for your reference.**
"""
                    logger.info(f"Loaded baseline context from {md_path} ({len(baseline_md_content)} chars)")
        except Exception as e:
            logger.warning(f"Could not load baseline MD files: {e}")
            baseline_context = ""
        
        prompt = self.load_skill(
            "common/mock_experiment.md",
            research_idea=research_idea,
            dataset_desc=dataset_desc,
            baseline_context=baseline_context
        )
        
        response = self.chat(messages=[{"role": "user", "content": prompt}])
        return response


class MockAblationAgent(BaseAgent):
    """
    Mock agent that simulates ablation study results based on experimental results.
    This bridges Part 2 (Analysis) and Part 3 (Paper Writing).
    """
    
    def run(self, research_idea: str, ablation_plan: str, main_results: str,
            dataset_info: Optional[List[Dict]] = None) -> str:
        """
        Generate simulated ablation study results.
        
        NOTE: Ablation studies are conducted ONLY on the primary (first) dataset,
        even if multiple datasets are provided.
        
        Args:
            research_idea: The original research idea
            ablation_plan: The ablation study plan from Part 2
            main_results: Main experimental results
            dataset_info: List of datasets (only first one will be used)
            
        Returns:
            Formatted ablation study results
        """
        logger.info("Generating mock ablation study results...")
        
        # Ablation studies focus on primary dataset only
        primary_dataset_name = "the primary dataset"
        if dataset_info and len(dataset_info) > 0:
            primary_dataset_name = dataset_info[0].get('name', 'the primary dataset')
            logger.info(f"Ablation study will focus on: {primary_dataset_name}")
            
            if len(dataset_info) > 1:
                logger.info(f"Note: {len(dataset_info)} datasets available, but ablation focuses on first one")
        
        prompt = self.load_skill(
            "common/mock_ablation.md",
            research_idea=research_idea,
            main_results=main_results,
            ablation_plan=ablation_plan,
            primary_dataset_name=primary_dataset_name
        )
        
        response = self.chat(messages=[{"role": "user", "content": prompt}])
        return response
