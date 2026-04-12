# Research Proposal Generation Prompt

Generate a COMPLETE Research Proposal as a coherent paper-like document.

## Research Context
- **Task Type**: {task_type}
- **Modality**: {modality}
- **Target Structure**: {target_structure}
- **Core Theme**: {core_theme}
- **Challenge Being Addressed**: {challenge_name} - {challenge_description}

## Technical Methods from Literature
{paper_methods}

## Dataset Constraints
{dataset_constraints}

## Patch Size Constraint
**Input Shape**: {patch_size}
⚠️ All proposed modules MUST be compatible with this input shape.

---

## 🎯 Your Task

Generate a COMPLETE Research Proposal with the following structure:

### 1. MOTIVATION
- **Problem Background**: What is the fundamental problem? (Keep it concise and factual)
- **Existing Limitations**: What are the limitations of current approaches? (Be specific)
- **Our Insight**: What is the technical insight driven by the chosen method?

### 2. PROPOSED METHOD (2-3 Modules)
For each module:
- **Module Name**: A descriptive, functional name (e.g., "Sparse Attention Block"). 
  - ❌ AVOID: "Novel [Theme] Module", "Adaptive [Theme] Layer" 
  - ✅ USE: Functional names describing WHAT it does
- **Technical Description**: Clear explanation of the mechanism using standard academic terminology.
- **Mathematical Formulation**: Key equations (use LaTeX format).
- **Role in Overall Architecture**: Specific function within the network.

### 3. INTEGRATION
- Description of the overall architecture flow.
- How modules interact to solve the specific challenge.

### 4. EXPECTED CONTRIBUTIONS
- 3 clear, technical contributions (Algorithm, Architecture, or Performance).

---

## ⚠️ CRITICAL REQUIREMENTS

1. **Coherence**: All modules must work together as a unified method
2. **Novelty**: The combination should be innovative, not just stacking existing techniques
3. **Feasibility**: Must be implementable within dataset constraints AND the framework constraints below
4. **Conference Quality**: Suitable for top-tier venues (CVPR, ICCV, NeurIPS, ICML)
5. **No Code**: Focus on mathematical formulations and conceptual descriptions
6. **Theme Connection**: Technical connection to {core_theme} should be in the description, NOT forced into module names

## 🔧 IMPLEMENTATION FRAMEWORK CONSTRAINTS

Your proposal will be implemented using the **CamylaNet framework** (nnUNet v2 wrapper).
The **ONLY** thing that can be customized is the **network architecture** (via `build_network_architecture`).

**What you CAN freely propose (all fully supported — be creative!):**
- Custom nn.Module network architectures of any complexity
- Attention mechanisms: self-attention, cross-attention, window attention, multi-head attention
- Convolutions: standard, dilated, depthwise separable, grouped, deformable
- Gating mechanisms, feature fusion, multi-scale architectures
- All standard PyTorch operations (Linear, Softmax, matmul, einsum, Conv3d, etc.)
- Novel module designs that process a single input tensor → single output tensor
- Architecture hyperparameters (channel sizes, layer depths, kernel sizes, number of heads, etc.)

**Hard constraints — CANNOT propose (these are fixed by the framework):**
- Custom loss functions or any loss modifications
- Changes to the training loop, optimizer type, or learning rate schedule
- Multi-input pipelines (model receives only ONE input tensor)
- Models returning tuples or dicts (must return a single tensor)
- `torch.linalg.qr`, `torch.linalg.svd`, `torch.linalg.eigh` (incompatible with float16)
- Self-supervised, contrastive, GAN-based, or non-standard supervised training paradigms
- Changes to data loading, preprocessing, or augmentation

**Practical guidance:**
- Prefer 1-2 well-designed modules over 3+ loosely connected modules
- Ensure tensor dimensions can be consistently traced through encoder → novel modules → decoder
- Use window/local attention instead of full global attention for 3D volumes to manage memory

## ⛔ NEGATIVE CONSTRAINTS (CRITICAL)
{negative_constraints}

## 📝 TITLE REQUIREMENTS

**Required Title Format**:
`{Core Method Name}: {Core Method Concept} for {Task Name}`

**Components**:
- **Core Method Name**: Concise, memorable name (2-4 words)
- **Core Method Concept**: Your abstract methodological approach (3-6 words)
  - From your Core Method Concept section above
- **Task Name**: Determined in Step 1 (with modifier if applicable)

**Examples with Correct Task Names**:
✅ "MFC-Net: Multi-Level Feature Consistency Network for Brain Tumor Segmentation"
✅ "SSA: Sparse Spectral Alignment for Oral Disease Segmentation"

Your title should follow these patterns seen in real top-tier papers:

**Naming Convention for Core Method Name**:
Use concise abbreviations (2-4 letters, optionally with "-Net" suffix) that are easy to pronounce and remember.
- ✅ **Good**: `MFC-Net`, `SSA`, `CDFL`, `DoIR`, `ConReg` — short, professional, follows top-conference naming style
- ❌ **Avoid**: `MultiConsisNet` (coined word too long, hard to pronounce), `SpectralAlign` (two full words lack abbreviation feel), `CrossDomainSeg` (three-word camelCase is verbose), `RobustSegNet` (redundant with task name modifier), `ConsisReg` (truncation feels informal)

❌ Single brain dataset: "MedSegNet: ... for Medical Image Segmentation" (TOO GENERIC task name)
❌ Old pattern: "Brain Tumor Segmentation via Multi-Level Feature Consistency" (wrong pattern format)
❌ No cross-site validation: "RobustNet: ... for Robust Segmentation" (MODIFIER NOT JUSTIFIED)
❌ Listing techniques: "FPB-Net: Frequency-Prototype-Boundary Integration for..." (lists techniques, not concept)
❌ Abstract math terms (Manifold, Topology, et. al)

## 📝 OUTPUT FORMAT

Respond with ONLY a valid JSON object:

```json
{{
    "title": "Proposal Title (should read like a paper title)",
    "motivation": {{
        "background": "Problem background (2-3 sentences)",
        "limitations": "Existing limitations (2-3 sentences)",
        "insight": "Key insight (1-2 sentences)"
    }},
    "modules": [
        {{
            "name": "Module 1 Name",
            "description": "Technical description",
            "formulation": "Key equations in LaTeX",
            "role": "Role in architecture"
        }},
        {{
            "name": "Module 2 Name",
            "description": "Technical description",
            "formulation": "Key equations in LaTeX",
            "role": "Role in architecture"
        }}
    ],
    "integration": "How modules work together (3-5 sentences)",
    "contributions": ["Contribution 1", "Contribution 2", "Contribution 3"]
}}
```

Do not include any text before or after the JSON object.
