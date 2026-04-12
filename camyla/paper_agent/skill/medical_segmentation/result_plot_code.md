You are an expert in data visualization and academic publication standards, proficient in Python plotting libraries such as matplotlib and seaborn.

Research Content:
${research_idea}

Experimental Results:
${experimental_results}

Ablation Study Results:
${ablation_results}

Plot Plan:
${plot_plan}

Per-Case Data (IMPORTANT — use this for scatter plots, subgroup analysis, distribution plots):
${per_case_data}

Your Task:
Generate Python code (matplotlib/seaborn) for **result analysis figures** that go BEYOND simple bar charts. A segmentation comparison figure is already generated separately — you must NOT generate a simple SOTA bar chart. Instead, focus on deeper analysis that reveals insights not captured by tables.

================================================================

## ⚠️ CRITICAL: What NOT to Generate

**DO NOT generate these types of figures (they duplicate the results table):**
- Simple bar charts comparing method Dice/HD95 scores
- Grouped bar charts showing method-vs-method comparisons
- Performance ranking line plots

These add NO value beyond the existing results table. The reviewer will reject figures that merely restate tabular data.

================================================================

## ✅ What TO Generate (Pick 2-3 from below)

### 1. Per-Case Performance Distribution (HIGHLY RECOMMENDED)
**Use Case**: Show that our method is consistently better, not just on average
**Data Source**: Per-case metrics from `per_case_data` section above
**Chart Types**:
- **Paired scatter plot**: X-axis = baseline Dice per case, Y-axis = our Dice per case, diagonal line showing where methods are equal. Points above the line show our method is better.
- **Box/Violin plot**: Side-by-side distribution of Dice scores for each method, showing median, IQR, and outliers.
- **Raincloud plot**: Combine violin + strip + box for full distribution insight.

**Example code structure (paired scatter)**:
```python
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(baseline_dice, ours_dice, alpha=0.6, s=30, c='#2E86AB', edgecolors='white', linewidth=0.5)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Equal performance')
ax.set_xlabel('Baseline Dice Score')
ax.set_ylabel('Our Method Dice Score')
# Count points above/below diagonal
above = sum(1 for o, b in zip(ours_dice, baseline_dice) if o > b)
ax.set_title(f'Per-Case Comparison (Ours better in {above}/{len(ours_dice)} cases)')
```

### 2. Subgroup Analysis by Lesion/Organ Volume (HIGHLY RECOMMENDED)
**Use Case**: Prove method works across different difficulty levels (small lesions are harder)
**Data Source**: Per-case `volume_pixels` from `per_case_data` (ground truth foreground size)
**Chart Types**:
- **Grouped box plot**: Group cases by volume tercile (Small/Medium/Large), show Dice distribution per group per method
- **Scatter with regression**: X-axis = log(volume), Y-axis = Dice, separate regression lines per method

**Example code structure (grouped box)**:
```python
# Classify cases into volume terciles based on GT foreground pixel count
volumes = [...]  # from per_case_data volume_pixels
q33, q66 = np.percentile(volumes, [33, 66])
groups = ['Small' if v < q33 else 'Medium' if v < q66 else 'Large' for v in volumes]

# Build DataFrame for seaborn
df = pd.DataFrame({'Dice': all_dices, 'Method': all_methods, 'Volume Group': all_groups})
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df, x='Volume Group', y='Dice', hue='Method', ax=ax)
ax.set_title('Performance by Lesion Volume')
```

### 3. Ablation Component Heatmap (RECOMMENDED if ablation data available)
**Use Case**: Visualize contribution of each module as a matrix
**Data Source**: Ablation results from `ablation_results`
**Chart Type**: Annotated heatmap with performance drop annotations

**Example code structure**:
```python
fig, ax = plt.subplots(figsize=(7, 4))
im = ax.imshow(values, cmap='RdYlGn', aspect='auto')
for i in range(n_rows):
    for j in range(n_cols):
        val = values[i, j]
        drop = val - full_model_values[j]
        text = f'{val:.2f}\n({drop:+.2f})'
        ax.text(j, i, text, ha='center', va='center', fontsize=9)
```

### 4. Error Distribution Analysis (RECOMMENDED)
**Use Case**: Show that our method reduces extreme errors (long tail of HD95)
**Data Source**: Per-case HD95 from `per_case_data`
**Chart Types**:
- **ECDF (Empirical CDF)**: Show cumulative distribution of HD95 — our curve should be more left-shifted
- **Violin + swarm**: Compare HD95 distributions between methods

**Example code structure (ECDF)**:
```python
from matplotlib.ticker import PercentFormatter
fig, ax = plt.subplots(figsize=(7, 4))
for method, hd95_values in methods_hd95.items():
    sorted_vals = np.sort(hd95_values)
    cdf = np.arange(1, len(sorted_vals)+1) / len(sorted_vals)
    ax.step(sorted_vals, cdf, label=method, linewidth=2)
ax.set_xlabel('HD95 (mm)')
ax.set_ylabel('Cumulative Proportion')
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
ax.legend()
ax.set_title('Cumulative Distribution of Boundary Errors')
```

================================================================

## ⚠️ Code Executability Requirements (Highest Priority)

1. **Runnable without any modification**: `python xxx.py` must work
2. **Complete data extraction**: Precisely extract numerical values from per_case_data and experimental_results
3. **No placeholders**: Strictly forbidden: `TODO`, `FIXME`, `placeholder`, `...`
4. **Complete imports**: Include ALL necessary imports
5. **Self-contained**: All data hardcoded directly in the code, no external file reads

================================================================

## matplotlib Configuration Standards

```python
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2
```

### Color Scheme
- Our method: orange `#F77F00`
- Best baseline: blue `#2E86AB`
- Other baselines: gray `#6C757D`
- Lower-is-better best: green `#06A77D`

================================================================

## Output Format (JSON)

```json
{
  "plots": [
    {
      "title": "Per-Case Performance Comparison",
      "plot_type": "paired_scatter",
      "description": "Paired scatter plot showing per-case Dice comparison between our method and strongest baseline",
      "data_preparation": "# Data extracted from per_case_data\nours_dice = [0.81, 0.85, ...]\nbaseline_dice = [0.78, 0.82, ...]\ncase_names = ['cirr_112', ...]",
      "plot_code": "fig, ax = plt.subplots(figsize=(6, 6))\n...",
      "caption": "Per-case Dice score comparison between our method and nnU-Net baseline. Each point represents one test case. Points above the diagonal indicate our method outperforms the baseline.",
      "placement": "Experiments section"
    }
  ]
}
```

## Important Notes

1. **Extract REAL per-case data** from the `per_case_data` section above — it contains actual case-level Dice, HD95, and volume data
2. **Use volume_pixels for subgroup analysis** — group by GT foreground size (small/medium/large terciles)
3. **Do NOT simply re-plot the aggregated table numbers** as a bar chart
4. **plots array**: 2-3 elements, each providing genuine analytical insight
5. **Code must be complete and runnable** — no TODOs, no external files, hardcoded data only

================================================================

## Code Generation Validation Checklist

- [ ] Does the code avoid simple bar charts of average metrics? (Must NOT be a table copy)
- [ ] Does the code use per-case data for scatter/distribution/subgroup analysis?
- [ ] Is all data precisely extracted (not example/placeholder data)?
- [ ] Does the code include all import statements?
- [ ] Can the code run directly (`python result_fig*.py`)?
- [ ] Are both PDF and PNG generated?

**Fix any issues before outputting JSON.**

================================================================

Start generating analysis figures!
