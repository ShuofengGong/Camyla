# Implementation Guideline

## Code Structure Requirements

The code should start with:
```python
import os
working_dir = os.path.join(os.getcwd(), 'working')
os.makedirs(working_dir, exist_ok=True)
```

The code should be a single-file python program that is self-contained and can be executed as-is.

No parts of the code should be skipped, don't terminate the code execution before finishing the script.

Your response should only contain a single code block.

**Target Runtime**: The code you write should complete within {timeout_duration} when run in the actual experiment.

**Validation Limit**: During code development, you are LIMITED to quick tests only (≤10 minutes). Use `camylanet.training_network_1epoch()` for validation, NOT full training.

You can also use the "./working" directory to store any temporary files that your code needs to create.

---

## Data Saving Requirements

Save all plottable data as numpy arrays using np.save().

Use the following naming convention for saved files:

```python
# At the start of your code
experiment_data = {
    'dataset_name1': {
        'metrics': {'train': [], 'val': []},
        'result_folder': None,  # Will store camylanet training result path
        'epochs': [],
        'dice_scores': [],  # Final dice scores per evaluation
        'hd95_scores': []   # Final HD95 scores per evaluation
    },
    # Add additional datasets as needed for multi-dataset experiments:
    # 'dataset_name2': {
    #     'metrics': {'train': [], 'val': []},
    #     'result_folder': None,
    #     'dice_scores': [],
    #     'hd95_scores': []
    # },
}

# After camylanet.training_network():
experiment_data['dataset_name1']['result_folder'] = result_folder

# After camylanet.evaluate():
dice_score = results['foreground_mean']['Dice']
hd95_score = results['foreground_mean']['HD95']
experiment_data['dataset_name1']['dice_scores'].append(dice_score)
experiment_data['dataset_name1']['hd95_scores'].append(hd95_score)
experiment_data['dataset_name1']['metrics']['val'].append({
    'dice': dice_score, 'hd95': hd95_score
})
```

- Include evaluation timestamps with the saved metrics
- For large datasets, consider saving in chunks or using np.savez_compressed()

---

## CRITICAL Evaluation Requirements

Your code MUST include ALL of these:

1. **Use camylanet framework for training and evaluation:**
```python
result_folder, training_log = camylanet.training_network(
    dataset_id, configuration, plans_identifier, exp_name
)
results = camylanet.evaluate(dataset_id, result_folder, exp_name)
```

2. **Track and report ONLY these specific metrics:**
   - Dice (higher is better)
   - HD95 (lower is better)

3. **Extract metrics from camylanet.evaluate() results:**
```python
dice_score = results['foreground_mean']['Dice']
hd95_score = results['foreground_mean']['HD95']
```

4. **Report FINAL metrics** after evaluation completion (not per-epoch metrics)

5. **Save ALL experimental data** at the end:
```python
np.save(os.path.join(working_dir, 'experiment_data.npy'), experiment_data)
```

---

## CRITICAL camylanet Usage Requirements

When calling camylanet.training_network(), ALWAYS save the result_folder path:

```python
result_folder, training_log = camylanet.training_network(
    dataset_id=dataset_id,
    configuration=configuration,
    plans_identifier=plans_identifier,
    exp_name=exp_name
)
# CRITICAL: Save result_folder path to experiment_data
experiment_data['dataset_name1']['result_folder'] = result_folder
```

This path is required for post-experiment file management operations.

---

## Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `timeout_duration` | Maximum execution time | `"6 hours"` |
