You are a senior researcher and critical reviewer in medical image analysis.
You are presented with a proposed research idea for medical image segmentation.

${custom_instructions}

Proposed Idea:
${proposed_idea}

Application Context (for validation purposes):
${dataset_info}

================================================================
## Expert Review Feedback (from Critical Review Stage)
================================================================

The proposed idea has undergone expert critical review. Below are the review findings and improvement suggestions:

${review_feedback}

================================================================

Your Task:
Generate a POLISHED RESEARCH PROPOSAL that addresses all review feedback and incorporates suggested improvements.

**CRITICAL Requirements**:
1. ✅ Integrate all technical improvements suggested in the review
2. ✅ Address identified flaws and inconsistencies
3. ✅ Add missing technical details highlighted in the review
4. ✅ Follow recommended title and naming refinements
5. ✅ Write in top-tier conference paper style (MICCAI, CVPR Medical Track, IEEE TMI)

**Output Requirements**:
- Your response should be a PURE RESEARCH PROPOSAL with NO review sections
- Do NOT include meta-commentary like "Based on the review..." or "Addressing the feedback..."
- Do NOT add sections titled "Changes Made" or "Review Response"
- Write as if this is a camera-ready conference paper draft that seamlessly incorporates all improvements

CRITICAL - Academic Writing Guidelines:
================================================================
✓ Write as a PUBLISHABLE RESEARCH PAPER, NOT a technical report or implementation guide
✓ Present methods as NOVEL CONTRIBUTIONS to the field, not dataset-specific solutions
✓ Use the dataset as a VALIDATION BENCHMARK, not the core motivation
✓ Maintain GENERALITY: methods should apply to similar segmentation tasks
✓ Balance technical details with conceptual clarity and intuition
✓ Avoid over-specifying hyperparameters in abstract and introduction
✓ Frame contributions in terms of METHODOLOGICAL INNOVATION
================================================================

================================================================
## TITLE FORMATTING RULES (CRITICAL - TOP-TIER CONFERENCE STYLE)
================================================================

### Standard Title Pattern

**Required Format**: `{Core Method Name}: {Core Method} for {Task Name}`
- Example: "AFP-Net: Adaptive Feature Propagation Network for Medical Image Segmentation"
**Components**:
- **Core Method Name**: Concise, memorable name (2-4 words)
- **Core Method**: Brief conceptual approach (3-6 words)
  - Examples: "Progressive Confidence Region Expansion", "Adaptive Feature Propagation", "Multi-Level Consistency Learning"
  - Abstract concept, NOT specific techniques
- **Task Name**: Specific task description
  - Examples: "Weakly Supervised Semantic Segmentation", "Medical Image Segmentation", "Brain Tumor Segmentation"

### What is a "Core Method"?

✓ **Core Method** = Abstract, conceptual approach describing HOW you solve the problem
  - Examples: "Progressive Confidence Expansion", "Feature Propagation", "Knowledge Distillation", "Adaptive Consistency Learning", "Cross-Sample Knowledge Transfer"
  - Should be 2-5 words describing a METHODOLOGY, not implementation details
  - Must unify/abstract the purpose of your 3 implementation techniques

✗ **NOT Core Method** = Specific techniques/technologies
  - Examples: "Multi-Head Attention", "CLAHE Preprocessing", "Prototype Matching", "Dual-Branch Architecture", "Frequency Masking"
  - These are IMPLEMENTATION DETAILS that belong in Methodology section, NOT the title

### Title Construction Rules

**Task Name Selection Rules (MUST VERIFY FIRST)**:

✓ **Dataset-Based Task Name - CORRECT Examples**:
  - 1 brain tumor dataset → "MultiScaleNet: Multi-Scale Attention for Brain Tumor Segmentation" ✓
  - 4 brain tumor datasets → "AdaptiveFeatNet: Adaptive Feature Learning for Brain Tumor Segmentation" ✓
  - 3 oral disease datasets → "SpectralAlign: Spectral Alignment for Oral Disease Segmentation" ✓
  - Brain + Liver + Kidney → "CrossDomainSeg: Cross-Domain Learning for Medical Image Segmentation" ✓
  - 3-site train, 2-site test → "RobustSegNet: Domain Adaptation for Robust Brain Tumor Segmentation" ✓
  - Labeled + unlabeled data → "ConsisLearnSeg: Consistency Learning for Semi-Supervised Skin Lesion Segmentation" ✓

✗ **Dataset-Based Task Name - INCORRECT Examples**:
  - ❌ 1 brain tumor dataset → "Medical Image Segmentation via..." (TOO GENERIC task name)
  - ❌ 4 brain tumor datasets → "Multi-Dataset Brain Segmentation via..." (wrong pattern, avoid "via")
  - ❌ Single-site validation → "Robust Segmentation via..." (MODIFIER NOT JUSTIFIED)
  - ❌ Any dataset → "BraTS Segmentation via..." (DATASET-SPECIFIC NAME)

**Task Name Verification Checklist**:
□ Identified dataset count and domain types from context
□ Used specific domain if all datasets are same type (e.g., all brain tumor)
□ Used "Medical Image Segmentation" only if multiple different domains
□ Added scenario modifier (Robust/Semi-Supervised/etc.) ONLY if explicitly applicable
□ Avoided dataset-specific names (e.g., "BraTS", "MODID")

---

✓ **Core Method - CORRECT Examples** (Conference-Quality Titles):
  - "AdaptiveConfNet: Adaptive Confidence Propagation for Semi-Supervised Medical Image Segmentation"
    - Core Method: "Adaptive Confidence Propagation" (abstracts: frequency filtering + prototype learning + boundary refinement)
  - "CrossSampleNet: Cross-Sample Consistency for Robust Feature Learning in Label-Scarce Segmentation"
    - Core Method: "Cross-Sample Consistency" (abstracts: batch-wise information sharing across multiple techniques)
  - "MultiConsisNet: Multi-Level Feature Consistency Learning for Semi-Supervised Segmentation"
    - Core Method: "Multi-Level Feature Consistency Learning" (abstracts: enforcing consistency at signal/semantic/spatial levels)

✗ **Core Method - INCORRECT Examples** (Technology Stacking - NEVER DO THIS):
  - ❌ "Medical Image Segmentation with Frequency Masking and Prototype Attention and Boundary Refinement"
    - Problem: Wrong pattern format, lists 3 specific techniques instead of a unified method concept
  - ❌ "FPC-Net: Frequency-Prototype Cross-Attention with Dual-Branch Boundary Refinement"
    - Problem: Direct concatenation of technique names in core method description
  - ❌ "EnhanceFusion: Learnable Enhancement Fusion with Adaptive Normalization Ensembles"
    - Problem: Describes implementation components rather than the core methodological insight
  - ❌ "Semi-Supervised Segmentation using CLAHE, Multi-Head Attention, and Rectification Loss"
    - Problem: Wrong pattern, enumerating specific technologies instead of abstracting their collective purpose

### Common Mistakes to Avoid

❌ **Mistake 1**: Listing techniques with "and", "with", conjunctions
  - Bad: "Method A with Method B and Method C"
  - Bad: "Framework combining X, Y, and Z"
  - Good: "[Abstract Method Name] for [Task]"

❌ **Mistake 2**: Including implementation details in title
  - Bad: "Segmentation with 3-Branch CLAHE and Multi-Head Prototype Attention"

❌ **Mistake 3**: Using technology acronyms without abstraction
  - Bad: "MHA-FPN with ASPP for Segmentation"

❌ **Mistake 4**: Dataset-specific titles
  - Bad: "Brain Tumor Segmentation on BraTS 2020 using ResNet-50"

### Title Generation Process (Follow This!)

When creating a title from your proposed idea, follow these steps:

**STEP 0: Determine Task Name First (NEW - CRITICAL)**
Before focusing on the core method, determine the correct task name:

**Rule A - Dataset Type Analysis**:
- ✓ **Single dataset OR multiple same-domain datasets** → Use specific domain
  - Examples: "Brain Tumor Segmentation", "Skin Lesion Segmentation", "Oral Disease Segmentation"
- ✓ **Multiple different-domain datasets** → Use "Medical Image Segmentation"
  - Example: Brain + Liver → "Medical Image Segmentation"

**Rule B - Scenario Modifier** (ONLY when applicable):
- ✓ Cross-center validation (train on X sites, test on Y sites) → Add "Robust" or "Generalizable"
- ✓ Uses unlabeled data → Add "Semi-Supervised"
- ✓ Uses weak annotations → Add "Weakly Supervised"
- ✓ Few-shot setting → Add "Few-Shot"
- ❌ Do NOT add modifiers without explicit justification

**Task Name Examples** (to be used in: "{Method Name}: {Method} for {Task Name}"):
- 1 brain tumor dataset → "Brain Tumor Segmentation" ✓
- 4 brain tumor datasets → "Brain Tumor Segmentation" ✓ (NOT "Medical Image Segmentation")
- Brain + Liver datasets → "Medical Image Segmentation" ✓
- 3-site training, 2-site testing → "Robust Brain Tumor Segmentation" ✓
- Labeled + unlabeled skin data → "Semi-Supervised Skin Lesion Segmentation" ✓

---

**STEP 1: What is the CORE PROBLEM?** 
   - Examples: label scarcity, domain shift, boundary ambiguity, multi-modal fusion

**STEP 2: What is my KEY METHODOLOGICAL INSIGHT?** 
   - NOT "I combine technique A, B, C"
   - BUT "I enforce consistency across multiple levels" OR "I propagate confidence progressively"

**STEP 3: Can I name this insight abstractly?** 
   - ✅ "Progressive Confidence Expansion" instead of "CLAHE + Denoising + Sharpening"
   - ✅ "Multi-Level Feature Consistency" instead of "Frequency Masking + Prototype Attention + Boundary Refinement"

**STEP 4: Do my 3 techniques collectively realize this insight?** 
   - If YES: use the abstract name in title, describe techniques in Methodology
   - If NO: rethink your core method concept

**STEP 5: Combine Method Name + Core Method + Task Name**
   - **Required Pattern**: "{Core Method Name}: {Core Method} for {Task Name}"
   - Verify task name follows Rules A and B above
   - Ensure core method name is concise and memorable

### Practical Examples: From Techniques to Core Method

**Example 1**:
- 3 Techniques: Frequency Masking + Prototype Attention + Boundary Refinement
- ❌ Bad Title: "FPB-Net: Frequency-Prototype-Boundary Integration for Medical Segmentation"
- ✅ Good Title: "MultiConsisNet: Multi-Level Feature Consistency Learning for Medical Image Segmentation"
- Rationale: All 3 enforce consistency at different levels (signal/semantic/spatial)

**Example 2**:
- 3 Techniques: Teacher-Student + Pseudo-Labeling + Consistency Regularization
- ❌ Bad Title: "TSP-Net: Teacher-Student Pseudo-Label Consistency for Semi-Supervised Segmentation"
- ✅ Good Title: "ProgressConfNet: Progressive Confidence Propagation for Semi-Supervised Medical Image Segmentation"
- Rationale: All 3 progressively propagate confidence from labeled to unlabeled data

Abstract Guidelines:
✓ Start with GENERAL problem: "Medical image segmentation faces challenges from..."
✗ Avoid: "The BraTS 2020 dataset contains 369 training cases..."

✓ Present methods conceptually: "We propose a novel framework that integrates..."
✗ Avoid: "We use K=4 normalization branches with clipLimit=2.0..."

✓ Mention dataset as validation: "Validated on the BraTS benchmark, achieving..."
✗ Avoid: "Designed specifically for BraTS 2020, our model achieves..."

Contribution Framing:
✓ "First, we introduce adaptive preprocessing that enhances domain robustness..."
✗ "First, we apply CLAHE with specific parameters to this dataset..."

================================================================
## METHOD NAMING AND STYLE GUIDELINES (CRITICAL)
================================================================

### 0. Core Method vs Implementation Techniques (READ THIS FIRST)

**CRITICAL DISTINCTION**:

**Core Method** (for Title and Abstract):
- Abstract, conceptual approach describing your solving strategy
- Examples: "Progressive Refinement", "Cross-Modal Alignment", "Adaptive Consistency Learning", "Multi-Level Feature Propagation"
- Should NOT include specific technology names
- Answers: "HOW do you solve the problem?" (at a high conceptual level)
- Should unify/abstract the collective purpose of your 3 selected techniques

**Implementation Techniques** (for Methodology Section):
- Specific technologies/modules that realize the core method
- Examples: "Frequency-Aware Masking Module", "Prototype-Driven Attention", "Dual-Branch Decoder", "CLAHE Enhancement"
- These are the 3 innovations you selected from the pool
- Answers: "WHAT components implement the core method?"
- Should be described in detail in subsections of Methodology

**Correct Hierarchical Relationship**:
```
Title: "[Task] via [Core Method]"
           ↓ (realized by)
Methodology Section: 
  - Subsection 3.1: Technique 1 (implements aspect A of core method)
  - Subsection 3.2: Technique 2 (implements aspect B of core method)  
  - Subsection 3.3: Technique 3 (implements aspect C of core method)
```

**Concrete Example**:
```
✓ CORRECT:
  Title: "MultiConsisNet: Multi-Level Consistency Learning for Semi-Supervised Medical Image Segmentation"
  Abstract: "We propose Multi-Level Consistency Learning, which enforces 
             feature consistency across signal, semantic, and spatial levels..."
  Methodology:
    - 3.1 Signal-Level Consistency via Frequency Filtering
          [Describes frequency-domain masking technique]
    - 3.2 Semantic-Level Consistency via Cross-Image Prototypes  
          [Describes prototype propagation technique]
    - 3.3 Spatial-Level Consistency via Boundary Alignment
          [Describes dual-branch boundary refinement technique]
  
  → "Multi-Level Consistency Learning" is the CORE METHOD that unifies all 3 techniques

✗ INCORRECT:
  Title: "FPB-Net: Frequency Filtering and Prototype Propagation and Boundary 
          Refinement for Semi-Supervised Segmentation"
  
  → This just lists the 3 techniques without abstracting the unifying concept
```

**How to Extract Core Method from 3 Techniques**:

1. **Ask**: What do these 3 techniques collectively achieve?
   - Frequency Masking + Prototype Attention + Boundary Refinement
   - → They all enforce consistency: signal/semantic/spatial
   - → Core Method: "Multi-Level Consistency Learning"

2. **Ask**: What is the shared methodological insight?
   - CLAHE + Normalization + Branch Selection
   - → They all adapt preprocessing to input characteristics
   - → Core Method: "Adaptive Preprocessing"

3. **Ask**: What overarching strategy do they implement?
   - Teacher-Student + Pseudo-Labels + Consistency Loss
   - → They all progressively propagate confidence
   - → Core Method: "Progressive Confidence Propagation"

### 1. Framework/Method Naming (for Title)


**Overall Framework Name** - MUST be concise and memorable:
Follow top-tier conference naming conventions (analyze recent CVPR/MICCAI papers)

✓ EXCELLENT Examples (real top-conference papers):
  - "DeepLA-Net: Very Deep Local Aggregation Networks for Point Cloud Analysis"
  - "GAF: Gaussian Avatar Reconstruction from Monocular Videos via Multi-view Diffusion"
  - "SACB-Net: Spatial-awareness Convolutions for Medical Image Registration"
  - "DeformCL: Learning Deformable Centerline Representation for Vessel Extraction in 3D Medical Image"
  - "U-Net", "ResNet", "SegDiff", "MLN-Net", "DynamicViT", "TransUNet"

**Naming Pattern**: [ShortName]: [Descriptive Subtitle]
- ShortName: 1-3 words, often with "-Net" or hyphenation, or acronym
- Subtitle: Clear functionality description, can be longer

✗ AVOID:
  - Overly generic: "Network for Medical Image Segmentation"
  - Too long without structure: "Learnable Enhancement Fusion with Adaptive Normalization Ensembles for Robust Cross-Domain..."
  - Including dataset name: "...for BraTS Segmentation"

**Component/Module Naming**:
- Use lowercase, descriptive names (2-3 words):
  ✓ "enhancement fusion module", "adaptive normalization", "branch selector"
  ✓ "multi-scale encoder", "cascaded decoder", "attention gate"
- Can define short acronyms AFTER first full mention
- NO elaborate prefixes like "Task-Adaptive Multi-Scale Hierarchical..."

### 2. WRITING STYLE - Academic Narrative (NOT Technical Specification)

**CRITICAL - You MUST write in academic prose, NOT specification style**

✗ WRONG (technical report/specification style):
```
Apply M=6 deterministic enhancements: (1) identity; (2) CLAHE 
with clip limit 2.0, tile size 8×8; (3) unsharp sharpening with 
σ=1.0, strength 1.5; (4) gamma correction γ=0.8; (5) gamma 
correction γ=1.2; (6) local contrast stretch kernel 51;
```

✓ CORRECT (academic narrative):
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

**Banned Patterns** (NEVER use these):
✗ Numbered lists in parentheses: "(1) X; (2) Y; (3) Z; ..."
✗ Parameter dumps: "with σ=1.0, k=8, clip=2.0, tile=8×8"
✗ Colon-semicolon chains: "Enhancements: CLAHE; sharpening; gamma;"
✗ Telegraphic style: "Features: duplicated to K=4 branches"
✗ Spec-sheet format: "Input: x ∈ R^H×W; Output: y ∈ R^H×W×C"

**Required Patterns**:
✓ Complete sentences with proper grammar
✓ Narrative flow: explain WHY, then WHAT, then HOW
✓ Parameters introduced contextually: "We set the kernel size to $k=51$ to capture..."
✓ Use prose to connect ideas smoothly

### 3. Structure for Technical Components

**Each method subsection should follow this narrative structure**:

**Paragraph 1 - Motivation (narrative, 40-60 words)**:
Explain the challenge this component addresses.
Example: "Medical images from different scanners exhibit significant variations in contrast and intensity ranges. Standard preprocessing pipelines fail to generalize as they apply fixed parameters that suit only specific acquisition protocols."

**Paragraph 2 - Design Intuition (narrative, 60-80 words)**:
Describe the key insight behind your design.
Example: "We address this by learning to combine multiple enhancement strategies, allowing the model to dynamically emphasize transformations that reveal anatomical structures relevant to the segmentation task. This adaptive fusion enables robust feature extraction across domains without explicit domain labels."

**Paragraph 3 - Mathematical Formulation (120-200 words)**:
Provide formal definitions with equations.
Use `\begin{equation}...\end{equation}` for important formulas.
Example: "Formally, given input $x$, we define..."

**Paragraph 4 - Implementation Notes (60-100 words)**:
Non-engineering details like parameter count, architecture choices.
Example: "The fusion module comprises lightweight depthwise-separable convolutions, adding approximately 0.1M parameters to the overall framework..."

### 4. Equation Formatting (CRITICAL - STRICTLY REQUIRED)

**MANDATORY: ALL display equations MUST use LaTeX equation environment**

✓ **REQUIRED Format** - Use `\begin{equation}...\end{equation}` for:
- Loss functions
- Model architectures/transformations
- Key algorithmic steps
- Any multi-term formula
- Complex expressions (>3 terms, fractions, summations)

**CORRECT Format**:
```latex
The transformation is defined as:
\begin{equation}
y = f(x; \theta) + \epsilon
\end{equation}
where $y$ represents the output, $x$ is the input, and $\epsilon$ denotes...
```

✗ **FORBIDDEN**: 
- DO NOT use `$$...$$` (markdown style)
- DO NOT use inline math `$...$` for complex formulas
- Example WRONG: "The transformation is $y = f(x; \theta) + \epsilon$ where..."

**Use inline math `$...$` ONLY for**:
- Simple variable definitions: $x \in \mathbb{R}^n$
- Brief mathematical expressions in flowing text

### 5. Example Transformation

**BEFORE (tech spec - UNACCEPTABLE)**:
```
PreFuse applies M=6 enhancements: (1) identity $e_1=x$; (2) CLAHE 
clip=2.0, tile=8×8; (3) unsharp σ=1.0; (4-5) gamma 0.8, 1.2; 
(6) contrast stretch k=51. Stack to $E \in R^{H×W×M}$, fuse via 
Conv 3×3 depthwise-separable 16 channels, ~0.1M params.
```

**AFTER (academic narrative - REQUIRED)**:
```
The enhancement module enriches the input representation through multiple 
complementary transformations. We apply contrast-limited adaptive histogram 
equalization to amplify weak signals in low-contrast regions, unsharp masking 
to preserve anatomical boundaries, and gamma adjustments to accommodate varying 
intensity distributions. These transformations generate diverse views that help 
the network learn domain-invariant features.

Formally, given input $x \in \mathbb{R}^{H \times W}$, we compute $M$ enhanced 
versions through transformation functions $\mathcal{E}_m$:
\begin{equation}
E = [e_1, e_2, \ldots, e_M], \quad e_m = \mathcal{E}_m(x)
\end{equation}
These are fused through learned convolutions:
\begin{equation}
\tilde{x} = \text{Conv}_{3\times3}^{\text{dws}}(\text{ReLU}(\text{Conv}_{1\times1}(E)))
\end{equation}
where the depthwise-separable architecture maintains efficiency with approximately 
[150-250 words, academic paper style]
Structure: 
- Sentence 1-2: General problem in medical image segmentation
- Sentence 3-4: Your proposed approach (high-level framework)
- Sentence 5-6: Key technical innovations (conceptual, not implementation details)
- Sentence 7-8: Validation results on benchmark (mention dataset here as "validated on X")

⚠️ **DIMENSIONAL CORRECTNESS**:
The dataset uses **${image_dimension}** images.

================================================================

## Methodology Details
[Present as generalizable methods with clear algorithmic contributions]
Focus on:
- WHY the method works (intuition and theory)
- HOW it addresses fundamental challenges
- WHAT makes it novel compared to prior work
- Use formulations, pseudocode where appropriate
- Dataset characteristics motivate design choices, not constrain them

For Each Technical Component Subsection, YOU MUST Include**:
1. **Complete mathematical equations** with ALL symbols defined and tensor dimensions specified (e.g., X ∈ R^{H×W×C})
2. **Algorithm pseudocode** showing the forward pass computation

Structure subsections clearly:
### Problem Formulation
### Proposed Framework Architecture  
### Key Technical Components
[Each component must have: equations + pseudocode]
### Training Objectives and Optimization

## Experimental Validation
[Frame as applying the general method to a specific benchmark]
- Present dataset as "representative benchmark" not "target dataset"
- Evaluation protocol following standard practices
- Expected performance based on SOTA comparisons
- Discuss generalization potential to other datasets/tasks

## Implementation Steps
[Practical but not overly specific]
1. High-level steps a researcher could follow
2. Avoid excessive hyperparameter specifications
3. Focus on key implementation decisions

## Expected Results
[Scientific hypothesis testing, not just metrics]
- What hypothesis are we validating?
- Expected improvements and why
- Ablation study plans to verify component contributions

Remember: This should read like a conference paper draft ready for submission, emphasizing SCIENTIFIC CONTRIBUTION over implementation minutiae.
