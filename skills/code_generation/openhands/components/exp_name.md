# Experiment Name Configuration

## Experiment Organization

**CRITICAL**: Use the following unique experiment name for ALL camylanet operations:

```python
exp_name = '{exp_name}'
```

This experiment name ensures:
- Results are saved to a unique directory
- No conflicts with other experiments
- Easy tracking and comparison

**Always use this exact `exp_name` in:**
- `camylanet.training_network(..., exp_name=exp_name)`
- `camylanet.training_network_1epoch(..., exp_name=f'{exp_name}_test')`
- `camylanet.evaluate(..., exp_name=exp_name)`

---

## Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `exp_name` | Unique experiment identifier | `2025-01-15_10-30-00_msd_pancreas_attempt_0` |
