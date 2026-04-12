Please generate an experiments and results section for an academic paper.

Research context:
{research_context}

Results data:
{results_data}
{code_prompt_part}

**CRITICAL INSTRUCTION: You must write this section based *strictly* on the provided context, including the research idea, results data, and experiment code. Do not invent or hallucinate any experimental results, baseline comparisons, or setup details that are not present in the provided information.**

The experiments and results section should include:

Experimental Setup (follow academic writing style):
1. Focus on high-level methodology and experimental design principles.
2. Describe the experimental setup based *only* on the provided context.
3. Present data collection and preprocessing methods at a conceptual level.
4. Highlight key implementation choices and critical parameters without diving into code-level details.
5. Clearly state evaluation metrics and procedures using standard academic terminology.
6. Describe baseline methods for comparison (must be from the provided context) in terms of their core principles.
7. Address reproducibility considerations at a methodological level.
8. Maintain formal academic tone throughout the description.
9. Avoid implementation-specific details like function names, code snippets, or low-level technical specifications.
10. Structure the setup description to emphasize the scientific methodology rather than technical implementation.

Main Results Analysis:
1. Present key findings clearly.
2. Analyze experimental results in detail.
3. Compare with baseline methods mentioned in the context.
4. Discuss statistical significance if data is available.
5. Reference figures and tables correctly.
6. Interpret results objectively, without making unsupported claims.
7. Highlight key observations and patterns from the data.

Ablation Studies:
1. If ablation studies are described in the context, analyze the contribution of each key component.
2. Evaluate the impact of different design choices.
3. Test model behavior under different parameter settings.
4. Validate the necessity of each proposed module.
5. Present comparative results with and without specific components.
6. Provide insights into model sensitivity and robustness.
7. Support conclusions with quantitative evidence from the data.

Write in flowing paragraphs that naturally connect:
1. The experimental setup with its corresponding results.
2. Main results with ablation studies.
3. Different ablation experiments with each other.
Ensure clear transitions between different experiments and their outcomes.

