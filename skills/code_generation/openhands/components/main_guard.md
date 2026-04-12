# Main Guard Requirement

## 🚨 CRITICAL: Main Guard Requirement

**YOU MUST WRAP ALL EXECUTABLE CODE IN A MAIN FUNCTION WITH A GUARD!**

### ✅ CORRECT STRUCTURE (REQUIRED)

```python
import os
import numpy as np
import camylanet

# Configuration variables at module level (OK)
dataset_id = ...  # From experiment config
configuration = ...  # From experiment config
exp_name = ...  # From experiment config
plans_identifier = ...  # Plans identifier

def main():
    """Main function containing all experiment logic."""
    # All experiment code goes here
    working_dir = os.path.join(os.getcwd(), 'working')
    os.makedirs(working_dir, exist_ok=True)
    
    # Training code
    result_folder, training_log = camylanet.training_network(
        dataset_id=dataset_id,
        configuration=configuration,
        plans_identifier=plans_identifier,
        exp_name=exp_name
    )
    
    # Evaluation code
    results = camylanet.evaluate(dataset_id, result_folder, exp_name)
    
    # Save results...

if __name__ == "__main__":
    main()
```

### ❌ INCORRECT STRUCTURE (FORBIDDEN)

```python
# DON'T DO THIS - code runs on import!
import camylanet

result_folder, training_log = camylanet.training_network(...)  # Runs immediately!
```

### Why This Is Required

Without the main guard, `import experiment` will execute all code, causing training to run during syntax checks and corrupting the experiment.

### Quick Checklist

- [ ] Use `if __name__ == "__main__":` guard
- [ ] Wrap executable code in `main()` function
- [ ] Keep only imports and config variables at module level
