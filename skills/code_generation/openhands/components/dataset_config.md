# Dataset Configuration

## 🚨 CRITICAL Dataset Configuration

**MUST USE THE FOLLOWING DATASET CONFIGURATION:**

```python
dataset_id = {dataset_id}  # {dataset_name}
configuration = '{configuration}'
```

**⚠️ CRITICAL REQUIREMENTS:**
- You MUST use `dataset_id = {dataset_id}`
- You MUST use `configuration = '{configuration}'`
- Target structure: {target_structure}
- Modality: {modality}
- Patch Size: {patch_size}

**FAILURE TO USE THE CORRECT DATASET CONFIGURATION WILL RESULT IN EXPERIMENT FAILURE.**

---

## Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `dataset_id` | Dataset ID number | `27` |
| `dataset_name` | Human-readable name | `Pancreas` |
| `configuration` | nnUNet configuration | `3d_fullres` |
| `target_structure` | Segmentation target | `pancreas and tumors` |
| `modality` | Imaging modality | `CT` |
| `patch_size` | Network input shape | `[64, 128, 128]` |
