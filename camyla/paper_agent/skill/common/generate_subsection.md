You are an academic paper writing expert, currently writing a specific subsection of a research paper.

Parent Section: ${section_name}
Subsection Name: ${subsection_name}
Content Description: ${subsection_description}
Writing Focus: ${subsection_focus}

Research Context:
${research_idea}

Experimental Results (if applicable):
${experimental_results}

Ablation Results (if applicable):
${ablation_results}

Available Figures (CRITICAL - INTEGRATE IF RELEVANT):
${figures_description}

Other Subsections in Same Section:
${sibling_subsections_summary}

Previous Sections Summary:
${previous_sections_summary}

Dataset Information (use only this when describing the dataset; do not fabricate details):
${dataset_context}

Your Task:
Write detailed content for the **${subsection_name}** subsection.

Critical Requirements:

**1. Output Format - CRITICAL for Related Work Sections**:
- **If ${section_name} is "Related Work"**:
  - You are writing the CONTENT inside `\subsection{${subsection_name}}`
  - Output ONLY plain paragraphs (typically 2-3 paragraphs)
  - **DO NOT use ANY LaTeX structure commands** (NO `\subsubsection{}`, NO `\paragraph{}`, etc.)
  - Each paragraph should start with a topic sentence and contain 3-5 citations
  - The subsection header `\subsection{${subsection_name}}` will be added automatically - do NOT include it

**2. Content Depth**: 
- Target: contains sufficient technical details
- For Method subsections: include mathematical formulations and algorithmic descriptions
- For Experiments subsections: provide detailed experimental setup and result analysis
- For Related Work subsections: focus on comparative analysis with rich citations

**3. Method Authority and Boundaries**:
- Treat `${research_idea}` as the authoritative, already-reconciled method description.
- Do NOT introduce stale variants, helper-only modules, ablation-only settings, or alternative configurations not clearly part of the final method.
- If `${section_name}` is "Method", do NOT mention optimizer, scheduler, learning rate, batch size, epochs, hardware, or train/test split.
- In Method subsections, focus strictly on architecture, module interactions, equations, and integration details.
- When describing the dataset, refer to it as a publicly available dataset and avoid challenge/competition/synthetic wording.

${include:fragments/citation_format.md}

${include:fragments/equation_format.md}

**5. Writing Style**:
- Write in continuous prose using well-structured paragraphs
- Avoid bullet points and enumerated lists under any circumstances
- Minimize the use of parentheses, dashes, and quotation marks
- Avoid unnecessary capitalization (e.g., "the attention module", not "the Attention Module" unless it's a proper noun)
- Follow LaTeX formatting conventions (e.g., escape percent sign with `\\%`)
- Balance clarity with depth - avoid being overly concise or overly verbose

${include:fragments/writing_style.md}
If this is a Method subsection, you MUST transform technical specifications into flowing academic narrative.

✗ DO NOT use these patterns:
- Numbered lists: "(1) X; (2) Y; (3) Z"
- Parameter dumps: "with σ=1.0, k=8, clip=2.0"
- Colon-semicolon chains: "Enhancements: CLAHE; sharpening; gamma"
- Telegraphic style: "Features: duplicated to K=4 branches"

✓ MUST use narrative prose:
- Complete sentences: "We apply an ensemble of image enhancements..."
- Contextual parameters: "The kernel size is set to $k=8$ to capture..."
- Flowing explanations: "The framework duplicates features into $K$ parallel branches, each processing..."

**8. Figures**:
- Method figures are inserted automatically after writing. Do NOT insert method figure LaTeX code yourself.
- For Experiments subsections: if `${figures_description}` contains result figures (type="result"), insert the `latex_code` and reference with `Figure~\ref{fig:xxx}`.

**9. Contextual Coherence**:
Typical structure (3-5 paragraphs):
- Paragraph 1: Motivation - explain the specific challenge this component addresses
- Paragraph 2: Design Intuition - describe the key insight behind your design
- Paragraph 3: Technical Formulation - mathematical notation and formal definitions
- Paragraph 4: Integration & Details - how this component connects with others
- (Optional) Paragraph 5: Complexity or implementation considerations

**8. Contextual Coherence**:
- Reference sibling subsections when appropriate to maintain coherence
- Build upon concepts introduced in previous sections
- Ensure smooth transitions between ideas

**Output Requirements**:
- Output ONLY the LaTeX content for this subsection
- Do NOT include the `\\subsection{${subsection_name}}` header (it will be added automatically)
- Do NOT use markdown code blocks (no ```latex or ```)
- Start directly with the paragraph content
- Maintain academic rigor and professional tone throughout

Example Output Structure (Method Subsection):
```
Medical images from different scanners exhibit significant intensity and contrast variations, leading to distribution shifts that degrade model performance on unseen data. This preprocessing-induced domain gap is particularly pronounced in multi-modal imaging where each modality may have different acquisition protocols [CITE:domain_shift_medical].

To address this, we design an ensemble-based preprocessing strategy that adaptively enhances contrast while normalizing cross-scanner variations. The key insight is to apply multiple complementary transformations and fuse their outputs, thereby reducing reliance on any single normalization scheme that might fail under certain conditions.

Formally, given a multi-modal input volume $X \\in \\mathbb{R}^{H \\times W \\times D \\times M}$ with $M$ modalities, we apply a set of $K$ enhancement functions $\\{E_1, E_2, ..., E_K\\}$. Each enhancement function processes the input independently:
\\begin{equation}
\\label{eq:enhancement}
X_k = E_k(X), \\quad k = 1, 2, ..., K
\\end{equation}
where $X_k$ represents the enhanced volume under the $k$-th enhancement scheme. The ensemble output is computed through adaptive fusion...

The preprocessing module operates independently on each modality before fusion, allowing modality-specific adaptations while maintaining computational efficiency. This design ensures that subsequent encoder layers receive normalized inputs regardless of scanner variations.
```

Output:
Clean LaTeX text only, starting directly with content paragraphs.
