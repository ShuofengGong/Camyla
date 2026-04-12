"""
Stage constants for the agent manager.

This module contains the main stage dictionary and goals that define
the different stages of the experimental workflow.
"""

from typing import Dict

# Main stage dictionary mapping stage numbers to stage names
# Stage 1: Baseline implementation
# Stage 2: Creative research (test research proposals)
# Stage 3: Ablation studies (verify component contributions)
MAIN_STAGE_DICT: Dict[int, str] = {
    1: "baseline_implementation",
    2: "creative_research",
    3: "ablation_studies",
}

# Main stage goals dictionary mapping stage numbers to their goals
MAIN_STAGE_GOALS: Dict[int, str] = {
    1: """
                - Focus on getting basic working baseline implementation
                - Use provided dataset
                - Aim for basic functional correctness
                - If you are given \"Code To Use\", you can directly use it as a starting point.""",
    2: """
                - Test complete Research Proposals (each containing multiple coherent modules)
                - Implement and evaluate the full proposal architecture as a whole
                - Compare against baseline to determine success
                - Be creative and think outside the box""",
    3: """
                - Perform ablation studies on the best proposal from Stage 2
                - Remove/disable ONE component at a time to verify its contribution
                - Keep all other parts unchanged during each ablation
                - Use the same datasets from the previous stage
                - Document the performance drop for each ablated component""",
}
