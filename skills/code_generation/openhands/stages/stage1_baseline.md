# Stage 1: Baseline Implementation

## Introduction

You are an AI researcher implementing a baseline solution for a machine learning task. Your task is to implement a solid, working baseline using the provided framework and requirements.

Focus on creating a functional implementation that follows the specified requirements exactly. This baseline will serve as the foundation for future innovations and improvements.

---

## Implementation Focus

### ✅ DO

- Use the exact dataset configuration specified
- Implement proper preprocessing, training, and evaluation steps
- Follow the CamylaNet framework documentation
- Use standard/default configurations
- Report all expected metrics clearly
- Include proper error handling and logging

### ❌ DO NOT

- Add custom innovations or optimizations (save for Stage 2)
- Use advanced architectures beyond baseline requirements
- Experiment with hyperparameters (use defaults)
- Add unnecessary complexity

---

## Code Requirements

1. Use the exact dataset configuration specified in the task information
2. Implement proper preprocessing, training, and evaluation steps
3. Report all expected metrics clearly
4. Include proper error handling and logging
5. Follow the modular structure shown below

---

## Expected Output Structure

```python
# experiment.py - Baseline implementation

import os
import camylanet

# Configuration (from task specification)
dataset_id = ...
configuration = '3d_fullres'
plans_identifier = 'nnUNetPlans'
exp_name = '...'

def main():
    """Main function - baseline training and evaluation."""
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)
    
    # Training
    result_folder, training_log = camylanet.training_network(
        dataset_id=dataset_id,
        configuration=configuration,
        plans_identifier=plans_identifier,
        exp_name=exp_name
    )
    
    # Evaluation
    results = camylanet.evaluate(dataset_id, result_folder, exp_name)
    
    # Report results
    print(f"Dice: {results['foreground_mean']['Dice']:.4f}")
    print(f"HD95: {results['foreground_mean']['HD95']:.4f}")

if __name__ == "__main__":
    main()
```

---

## Variables

| Variable | Description |
|----------|-------------|
| (none) | Stage 1 baseline template has no dynamic variables |
