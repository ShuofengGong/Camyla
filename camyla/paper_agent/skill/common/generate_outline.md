You are an academic paper writing expert specializing in structuring well-organized section outlines.

Current Section: **${section_name}**
Section Goal: ${section_description}

Research Context:
${research_idea}

Previous Sections Summary:
${previous_sections_summary}

Subsection Configuration Requirements:
- Minimum subsections: ${min_subsections}
- Maximum subsections: ${max_subsections}
- Target words per subsection: ${target_words_per_subsection}

Your Task:
Plan a well-structured subsection outline for the **${section_name}** section.

Requirements:
1. Subsection names should be concise and professional (2-6 words)
2. Descriptions should clearly explain the core content of each subsection
3. Focus field should specify key writing points and technical aspects to cover
4. Logical progression: start with overview, then dive into details
5. Avoid content overlap between subsections
6. For Method sections:
   - Each subsection should correspond to a **novel contribution** described in the research idea
   - Do NOT create a separate subsection for standard/well-known components (e.g., standard loss functions like Dice+CE, common optimizers, basic encoder-decoder structure) unless they are explicitly an innovation of this work
   - Standard training details (loss function, optimizer, learning rate) should be briefly mentioned in the Overview or the last component subsection, not given their own subsection
   - Only create a dedicated "Loss Function" or "Training Objective" subsection if the loss function itself is a key contribution of the paper
7. For Experiments sections: include datasets, setup, results, ablations, analysis
8. For Related Work sections:
   - Organize the review into no more than two broad thematic or chronological subsections
   - Merge minor themes into the closest major theme instead of creating extra subsections
   - Prefer two balanced subsections when the literature scope allows it

Output Format (JSON):
```json
{
  "subsections": [
    {
      "name": "Subsection Title",
      "description": "2-3 sentences describing the content scope of this subsection",
      "focus": "Key writing points, such as important citations, technical details to emphasize"
    }
  ]
}
```

Example (Method Section):
```json
{
  "subsections": [
    {
      "name": "Overview",
      "description": "Introduce the overall framework design philosophy and main components. Explain how different modules work together to achieve the research goal.",
      "focus": "Provide architecture diagram, articulate design motivation and innovation points"
    },
    {
      "name": "Adaptive Preprocessing Module",
      "description": "Detail the mathematical formulation and algorithm flow of the preprocessing module. Explain how it handles cross-scanner variations.",
      "focus": "CLAHE enhancement, ensemble strategy, mathematical derivations, complexity analysis"
    },
    {
      "name": "Multi-Scale Encoder",
      "description": "Introduce the encoder's multi-scale feature extraction mechanism and attention mechanisms.",
      "focus": "ResNet backbone, feature pyramid, spatial and channel attention formulations"
    },
    {
      "name": "Adaptive Feature Fusion Decoder",
      "description": "Describe the proposed decoder with the novel adaptive fusion mechanism. Explain how it integrates multi-scale features from the encoder.",
      "focus": "Adaptive gating, cross-level feature fusion, skip connection enhancement, mathematical formulation"
    }
  ]
}
```

Example (Experiments Section):
```json
{
  "subsections": [
    {
      "name": "Datasets and Evaluation Metrics",
      "description": "Describe the datasets used for evaluation and the metrics employed to measure performance.",
      "focus": "Dataset statistics, train/val/test splits, Dice, IoU, Hausdorff distance definitions"
    },
    {
      "name": "Implementation Details",
      "description": "Provide implementation details including network hyperparameters, training procedures, and computational resources.",
      "focus": "Optimizer settings, learning rate schedule, data augmentation, hardware specifications"
    },
    {
      "name": "Comparison with State-of-the-Art",
      "description": "Present quantitative and qualitative comparisons with existing methods on benchmark datasets.",
      "focus": "Main results table, baseline methods citations, performance analysis"
    },
    {
      "name": "Ablation Studies",
      "description": "Analyze the contribution of each component through systematic ablation experiments.",
      "focus": "Component-wise ablation results, statistical significance tests"
    },
    {
      "name": "Qualitative Analysis",
      "description": "Provide visual examples and failure case analysis to demonstrate model behavior.",
      "focus": "Segmentation visualizations, attention maps, challenging cases discussion"
    }
  ]
}
```

Example (Related Work Section):
```json
{
  "subsections": [
    {
      "name": "Segmentation Backbones",
      "description": "Review the evolution of medical image segmentation models from early fully convolutional architectures to stronger encoder-decoder and transformer-based backbones.",
      "focus": "FCN, U-Net, V-Net, UNETR, Swin-based models, strengths and limitations, cite 5-7 seminal papers"
    },
    {
      "name": "Generalization and Enhancement Strategies",
      "description": "Discuss approaches that improve robustness across domains, scanners, and acquisition settings, including normalization, augmentation, adaptation, and preprocessing design.",
      "focus": "Domain generalization, domain adaptation, normalization, data augmentation, intensity enhancement, comparative positioning"
    }
  ]
}
```

Critical Notes:
- Ensure subsection count is within the specified range [${min_subsections}, ${max_subsections}]
- For Related Work, never exceed two subsections even if several themes are available
- Each subsection should have sufficient content to meet the target word count
- Consider the logical flow and dependencies between subsections
- For Method sections, ensure all **novel** components from research idea are covered
- For Method sections, do NOT add subsections for standard components (e.g., standard loss functions, common optimizers, basic training procedures) that are not part of the paper's contribution. Mention them briefly within other subsections instead

Output:
Output ONLY the JSON structure, without markdown code block markers (no ```json or ```).
