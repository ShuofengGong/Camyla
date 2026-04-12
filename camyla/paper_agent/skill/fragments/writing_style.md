**Academic Writing Style Requirements**:

## 1. Transform Specification Style into Academic Narrative

**CRITICAL**: You MUST write in academic prose, NOT technical specification style.

### ✗ WRONG (Technical Report/Specification Style):
```
Apply M=6 deterministic enhancements: (1) identity; (2) CLAHE 
with clip limit 2.0, tile size 8×8; (3) unsharp sharpening with 
σ=1.0, strength 1.5; (4) gamma correction γ=0.8; (5) gamma 
correction γ=1.2; (6) local contrast stretch kernel 51;
```

### ✓ CORRECT (Academic Narrative):
```
The enhancement module generates multiple complementary representations 
of the input image to capture diverse contrast and intensity patterns. 
We employ contrast-limited adaptive histogram equalization (CLAHE) to 
enhance local regions where tumor boundaries are subtle, combined with 
sharpening operations for edge preservation and gamma transformations 
for dynamic range adjustment.

Formally, given an input image $x \in \mathbb{R}^{H \times W}$, we 
apply $M$ enhancement functions $\{\mathcal{E}_m\}_{m=1}^M$ to generate 
enhanced versions $\{e_m = \mathcal{E}_m(x)\}$...
```

## 2. Banned Patterns (NEVER Use These)

- ✗ Numbered lists in parentheses: "(1) X; (2) Y; (3) Z; ..."
- ✗ Parameter dumps: "with σ=1.0, k=8, clip=2.0, tile=8×8"
- ✗ Colon-semicolon chains: "Enhancements: CLAHE; sharpening; gamma;"
- ✗ Telegraphic style: "Features: duplicated to K=4 branches"
- ✗ Spec-sheet format: "Input: x ∈ R^H×W; Output: y ∈ R^H×W×C"

## 3. Required Patterns

- ✓ Complete sentences with proper grammar
- ✓ Narrative flow: explain WHY, then WHAT, then HOW
- ✓ Parameters introduced contextually: "We set the kernel size to $k=51$ to capture..."
- ✓ Use prose to connect ideas smoothly

## 4. Method Section Paragraph Structure

Each method subsection should follow this narrative structure:

**Paragraph 1 - Motivation (40-60 words)**:
Explain the specific challenge this component addresses.

Example: "Medical images from different scanners exhibit significant variations in contrast and intensity ranges. Standard preprocessing pipelines fail to generalize as they apply fixed parameters that suit only specific acquisition protocols."

**Paragraph 2 - Design Intuition (60-80 words)**:
Describe the key insight behind your design.

Example: "We address this by learning to combine multiple enhancement strategies, allowing the model to dynamically emphasize transformations that reveal anatomical structures relevant to the segmentation task. This adaptive fusion enables robust feature extraction across domains without explicit domain labels."

**Paragraph 3 - Mathematical Formulation (120-200 words)**:
Provide formal definitions with equations.

Example: "Formally, given input $x$, we define..."

**Paragraph 4 - Integration & Details (60-100 words)**:
Non-engineering details like parameter count, architecture choices.

Example: "The fusion module comprises lightweight depthwise-separable convolutions, adding approximately 0.1M parameters to the overall framework..."

## 5. General Writing Style Rules

**Word Choice**:
- Avoid overused academic words: "inherent", "fundamental", "critical", "systematically", "subtle"
- Minimize overly complex vocabulary
- Write in the style of a highly accomplished non-native English-speaking researcher

**Formatting**:
- Never use bullet points under any circumstances
- Minimize the use of parentheses, dashes, and quotation marks
- Avoid unnecessary capitalization (e.g., "the attention module", NOT "the Attention Module")
- Follow LaTeX formatting conventions (e.g., escape percent sign with `\%`)
- Write in continuous prose using well-structured paragraphs

**Prose Flow**:
- Use well-structured paragraphs with coherent flow
- Provide narrative transitions between ideas
- Balance clarity with depth - avoid being overly concise or overly verbose

## 6. Output Format Rules

**Plain Text Only**:
- Use plain text - no quotation marks (" "), no em-dashes (—), no markdown bold (**) or italics (*)
- Do NOT use markdown code blocks (no ```latex or ```)
- Start directly with LaTeX commands or paragraph content
