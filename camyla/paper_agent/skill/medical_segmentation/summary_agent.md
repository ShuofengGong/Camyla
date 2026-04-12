You are an expert researcher in medical image segmentation with deep knowledge of transferable methodologies.
Your task is to read a research paper and extract innovations that have BROAD APPLICABILITY beyond the specific dataset used.

Research Application Context:
${dataset_context}

Paper Text:
${paper_text}

Your Goal: Extract innovations with GENERALIZATION POTENTIAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Focus on:
✓ General architectural principles that could transfer to related domains
✓ Novel training strategies applicable beyond this specific dataset  
✓ Algorithmic components with broad utility in medical imaging
✓ Methodological innovations addressing common segmentation challenges
✓ Theory-driven design choices with wide applicability

Avoid:
✗ Dataset-specific hyperparameter tuning tricks
✗ Narrow optimizations tied to one benchmark
✗ Implementation details without conceptual novelty
✗ Engineering hacks without generalizable insights
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Extraction Instructions:
1. Identify the FUNDAMENTAL problem, not just dataset-specific challenges
2. Extract the core METHOD as a generalizable framework
3. Highlight innovations that could apply to similar segmentation tasks
4. Note results as evidence of effectiveness, not the primary contribution

Format your output as a structured summary:
─────────────────────────────────────────────────────────────
Title: [Paper Title]

Core Problem: 
[Describe the general challenge, e.g., "domain shift in medical imaging" not "BraTS scanner variations"]

Proposed Method: 
[High-level framework description applicable to similar tasks]

Key Innovations (with generalization potential):
For each innovation, provide:

  **Innovation 1: [Name]**
  - **Conceptual Idea**: [What problem it solves and why it's generalizable]
  - **Technical Approach**: [Extract the algorithm steps or key equations FROM THE PAPER - do not invent, quote the paper's actual method]
  - **Applicability**: [Which other tasks/datasets could benefit from this]

  **Innovation 2: [Name]**
  - **Conceptual Idea**: [...]
  - **Technical Approach**: [Extract actual algorithm/equations from paper]
  - **Applicability**: [...]

  **Innovation 3: [Name]**
  - **Conceptual Idea**: [...]
  - **Technical Approach**: [Extract actual algorithm/equations from paper]
  - **Applicability**: [...]

CRITICAL: For "Technical Approach", EXTRACT the actual algorithm steps, formulas, or architectural details described in the paper. This will be used by subsequent stages to generate detailed technical specifications. Do NOT summarize too abstractly - preserve concrete method details.

Validation Evidence: 
[Results on original dataset, as proof of concept]

Transferability Assessment:
[Your analysis: Could this work on other segmentation tasks? Why?]
─────────────────────────────────────────────────────────────

