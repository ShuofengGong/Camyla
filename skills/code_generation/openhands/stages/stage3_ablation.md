# Stage 3: Ablation Studies

## Introduction

You are an experienced AI researcher conducting an ABLATION STUDY. Your task is to systematically remove or disable specific components from the best model to measure their individual contributions to overall performance.

---

## 🔬 Ablation Study Task

### Component to Ablate

**{component_name}**

### Removal Strategy

{removal_description}

---

## What is Ablation Study?

Ablation study is a scientific method to verify the contribution of each component in a system:

1. **Remove ONE component** from the complete model
2. **Train/evaluate** the ablated model
3. **Compare performance** with the full model
4. **Quantify contribution** based on performance drop

If performance drops significantly → the component is important.
If performance stays similar → the component may be redundant.

---

## Ablation Guidelines

### Common Ablation Strategies

| Component Type | Ablation Approach |
|---------------|-------------------|
| **Attention Module** | Replace with identity function or uniform weights |
| **Skip Connection** | Remove the residual path, use only main path |
| **Custom Loss** | Replace with standard loss (CE, Dice) |
| **Normalization** | Replace custom norm with standard BatchNorm |
| **Multi-scale Features** | Use single-scale instead |
| **Auxiliary Outputs** | Remove auxiliary branches, keep main output |

### Implementation Steps

1. **Identify** the exact component to remove in the code
2. **Comment out or modify** the component implementation
3. **Replace** with minimal/identity alternative if needed
4. **Keep all other parts unchanged**
5. **Document** what was removed and expected impact
6. **Verify** the model still runs correctly

### Code Structure

```python
# experiment.py - Ablation Study: {component_name}

import os
import camylanet

# Configuration
dataset_id = ...
configuration = '3d_fullres'
exp_name = '..._ablation_{component_name}'

# ==========================================
# ABLATION: {component_name}
# ==========================================
# What was removed: [describe the component]
# How it was removed: [describe the removal strategy]
# Expected impact: [describe expected performance change]
# ==========================================

def main():
    print("🔬 Running ablation study: {component_name}")
    # ... training code with ablated component ...

if __name__ == "__main__":
    main()
```

---

## Data Saving Requirements

Save results using the **FLAT structure** (same as Stage 1-2):

```python
experiment_data = {
    'dataset_name': {  # Use dataset name directly as top-level key
        'metrics': {'train': [], 'val': []},
        'dice_scores': [],
        'hd95_scores': [],
        'result_folder': None,
        'epochs': []
    }
}

# IMPORTANT: Do NOT nest ablation name as a key. Use flat structure.
np.save(os.path.join(working_dir, 'experiment_data.npy'), experiment_data)
```

---

## Quality Checklist

- [ ] Only the specified component is removed/disabled
- [ ] All other components and settings are unchanged
- [ ] The model still trains and evaluates correctly
- [ ] Clear comments document what was ablated
- [ ] Metrics are properly saved for comparison

---

## Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `component_name` | Name of the component being ablated | `"attention_mechanism"` |
| `removal_description` | How to remove the component | `"Replace attention with identity function"` |
