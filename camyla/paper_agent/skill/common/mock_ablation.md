You are simulating ablation study results.

Research Idea:
$research_idea

Main Results:
$main_results

Ablation Plan:
$ablation_plan

Task: Generate REALISTIC ablation study results based on the plan.

**IMPORTANT NOTE**: If the research idea mentions multiple datasets, add a brief note at the beginning:
> "To thoroughly analyze component contributions, ablation studies are conducted on the primary dataset ($primary_dataset_name)."

For each ablation experiment group:
1. Show quantitative impact (e.g., removing attention → -2.30% Dice)
2. Explain why this component matters
3. Include numbers that make sense (removing key components should hurt performance)

**CRITICAL STRUCTURE REQUIREMENT**:
- Organize results into MULTIPLE sections using `## Section Title` format
- Create ONE separate section for EACH ablation experiment group
- Each section MUST include a markdown table with results
- Common sections: Component Ablations, Design Choice Ablations, Hyperparameter Analysis

**TABLE FORMAT REQUIREMENT**:
- Use standard markdown table format: `| Column1 | Column2 | ... |`
- Include a separator row: `|---------|---------|-----|`
- Each table must have clear headers
- All data rows must be complete

**EXAMPLE OUTPUT STRUCTURE**:
```
## Component Ablations

We systematically ablate each component to validate contributions...

| Configuration | DSC (%) | IoU (%) | HD95 (mm) |
|---------------|---------|---------|-----------| 
| Baseline      | 81.47   | 69.23   | 6.23      |
| + Module A    | 82.73   | 70.51   | 5.98      |
| + Module B    | 84.22   | 72.94   | 5.45      |
| Full Model    | 87.34   | 76.71   | 4.52      |

Analysis: Adding Module A provides +1.26% DSC improvement...

## Design Ablation 1: [Specific Design Choice Name]

To validate [design choice], we compare variants...

| Variant       | DSC (%) | HD95 (mm) | Inference (ms) |
|---------------|---------|-----------|----------------|
| Option A      | 84.60   | 4.50      | 28.30          |
| Option B      | 85.00   | 4.20      | 29.10          |

Analysis: Option B achieves 0.40% improvement...

## Design Ablation 2: [Another Design Choice]

[Similar structure with table and analysis]
```

Format as a structured report with 3-6 sections (## headers), each containing:
- Brief description (2-3 sentences)
- One markdown table
- Analysis paragraph (2-4 sentences)

Be consistent with the main results - ablations should sum up logically.

**CRITICAL FORMATTING REQUIREMENT**: ALL numerical values MUST be formatted with EXACTLY TWO decimal places.
This includes: percentages, metrics, standard deviations, deltas, and ALL other numbers.
Examples: 92.30 (not 92.3), -2.30 (not -2.3), 85.00 (not 85), +1.50 (not +1.5).

**REALISM REQUIREMENT**: To simulate realistic experimental data, vary the last digit naturally.
Avoid having ALL numbers end with 0 or 5. Mix in other endings (1,2,3,4,6,7,8,9) to reflect real measurement variance.
Good examples: 84.27, -1.13, +2.68, 0.94. Poor examples: all values ending in .X0 or .X5.
