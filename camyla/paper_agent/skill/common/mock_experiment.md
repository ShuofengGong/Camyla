You are simulating experimental results for the following research idea:

$research_idea

$dataset_desc

$baseline_context

Task: Generate REALISTIC experimental results that this method might achieve.

**IMPORTANT**: For MULTIPLE datasets, create SEPARATE sections for each dataset with individual tables.
For SINGLE dataset, use a unified format.

For each dataset, include:
1. Main quantitative results using the dataset's evaluation metrics, comparing your method to baselines
2. Performance on different classes/subsets (based on dataset classes)
3. Training details (time, convergence)
4. Brief observations about what works well

**BASELINE METHOD SELECTION**:
If baseline results are provided above:
1. **Select 10-12 comparison methods intelligently:**
   - Prioritize HIGH-IMPACT methods (published in top venues like CVPR, ICCV, MICCAI, IEEE TMI)
   - Include WELL-KNOWN foundational methods (e.g., U-Net, nnU-Net, TransUNet, U-Mamba)
   - Include RECENT methods (published in last 2-3 years)
   - Ensure diversity in method types (CNN-based, Transformer-based, hybrid)
2. **Use realistic performance ranges** based on the baseline results shown
3. **Maintain similar experimental settings** (data split, metrics) for comparability
4. **Your proposed method should improve by 2-5%** over the best baseline
5. **For multi-dataset experiments**: Use the SAME set of baseline methods across all datasets (exclude challenge-specific baselines); if a baseline is missing on some dataset, reasonably estimate its performance.

If no baseline results are provided:
- Generate reasonable baseline methods (U-Net, nnU-Net, TransUNet, U-Mamba, Swin-Unet, etc.)
- Use realistic performance ranges for medical image segmentation

**MULTI-DATASET FORMAT** (if applicable):
```
# Experimental Results

## Dataset 1: [Name]

### Quantitative Results
[Table with method vs baselines]

### Analysis
[Brief discussion]

## Dataset 2: [Name]

### Quantitative Results
[Table with method vs baselines]

### Analysis
[Brief discussion]

## Cross-Dataset Analysis
[Summary of performance trends across datasets]
```

Format as a structured report with tables. Be specific with numbers (e.g., Dice: 92.30±1.20%).
Make the results believable - your method should improve over baselines by 2-5%, not unrealistic margins.
Ensure the metrics align with each dataset's standard evaluation metrics.

**CRITICAL FORMATTING REQUIREMENT**: ALL numerical values MUST be formatted with EXACTLY TWO decimal places.
This includes: percentages, metrics, standard deviations, time measurements, and ALL other numbers.
Examples: 92.30 (not 92.3), 1.20 (not 1.2), 85.00 (not 85), 0.50 (not 0.5).

**REALISM REQUIREMENT**: To simulate realistic experimental data, vary the last digit naturally.
Avoid having ALL numbers end with 0 or 5. Mix in other endings (1,2,3,4,6,7,8,9) to reflect real measurement variance.
Good examples: 92.37, 84.23, 1.18, 5.64. Poor examples: all values ending in .X0 or .X5.
