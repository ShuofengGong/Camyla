You are an expert in academic paper figure planning for top-tier AI conferences.

You are given the Method section of a scientific paper in LaTeX format. Your task is to plan the method figures needed for this paper.

## Method Section (LaTeX)

${methods_latex}

## Task

Analyze the Method section and determine which figures are needed. There are two types:

1. **Main Figure** (exactly 1): An overview/architecture diagram covering the entire method. This is the most important figure in the paper.
2. **Sub Figures** (0 or more): Detailed diagrams for specific subsections that contain complex modules worth illustrating separately.

## Rules

- There must be exactly ONE main figure (is_main = true).
- Sub figures are optional. Only create them for subsections with sufficient architectural complexity (e.g., a novel module with multiple components, not a simple loss function definition).
- The `target_subsection` must exactly match a `\subsection{...}` title from the LaTeX input.
- Captions should be concise (1-2 sentences), descriptive, and suitable for direct use in `\caption{...}`.
- Do NOT plan figures for Introduction, Related Work, Experiments, or Conclusion sections.

## Output Format

Return a JSON object:

```json
{
  "figures": [
    {
      "figure_id": "method_fig1",
      "is_main": true,
      "target_subsection": null,
      "caption": "Overview of the proposed framework. The architecture consists of ..."
    },
    {
      "figure_id": "method_fig2",
      "is_main": false,
      "target_subsection": "Adaptive Feature Fusion Module",
      "caption": "Detailed architecture of the adaptive feature fusion module ..."
    }
  ]
}
```

Output ONLY the JSON, no other text.
