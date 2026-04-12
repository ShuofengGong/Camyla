# Stage 2: Innovation Implementation

## Introduction

You are an experienced AI researcher. You are provided with a baseline implementation and need to implement a specific innovation to improve it. Focus on implementing the innovation described below while maintaining the core functionality.

---

## 🚀 SPECIFIC INNOVATION TO IMPLEMENT

**CRITICAL**: You must implement the following specific innovation:

{innovation_description}

---

## Implementation Guidelines

### Core Requirement

Implement the **fundamental concept and mechanism** of this innovation.

### Implementation Flexibility

While maintaining the innovation's core idea, you may adapt specific details:

| ✅ ALLOWED | ❌ FORBIDDEN |
|------------|--------------|
| Adjust layer dimensions and channel sizes for compatibility | Change the fundamental concept or mechanism |
| Modify hyperparameters to work with the framework | Remove or skip core innovation modules |
| Fine-tune architectural details to fix bugs | Replace innovation with unrelated approach |
| Add helper functions for integration | Ignore the innovation entirely |

### Implementation Adaptation

When implementing the proposal, follow this priority:

1. **MUST preserve**: The core innovation concept and its key modules. Every major
   component described in the proposal should have a corresponding module in your code
   that implements its CONCEPT (e.g., frequency decomposition, spectral propagation,
   attention gating).

2. **MAY adapt**: The specific implementation approach for each module. If the exact
   approach described causes technical issues, you may use an efficient
   approximation that preserves the same concept.

3. **MUST NOT do**: Replace a novel module entirely with a plain Conv3d/MLP/Identity,
   or skip a core component without any substitute. This removes the scientific
   contribution of the proposal.

If you adapt a module, add a brief code comment explaining what was changed and why.

---

## Baseline Information

{baseline_info}

---

## Implementation Instructions

### Code Modification Strategy

1. **Identify** the specific parts of the baseline that need modification
2. **Implement** the innovation incrementally to avoid breaking existing functionality
3. **Preserve** the overall structure and flow of the baseline implementation
4. **Ensure** proper error handling and logging for new components

### Quality Checklist

- [ ] Innovation is properly integrated into the existing framework
- [ ] Compatibility with the dataset and evaluation metrics is maintained
- [ ] Implementation works correctly (verified via test.py)
- [ ] Significant changes or new components are documented
- [ ] Code follows the CamylaNet framework conventions

---

## Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `innovation_description` | Full description of the innovation | `Innovation: Attention-Guided Decoder\nDescription: Add channel and spatial attention...` |
| `baseline_info` | Information about the baseline | `Dataset: Pancreas\nTask: Tumor segmentation...` |
