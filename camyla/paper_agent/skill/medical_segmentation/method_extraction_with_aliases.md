You are an expert in medical image segmentation research.

Your task is to extract baseline methods and their performance metrics from a research paper on dataset "${dataset_name}".

## Dataset Information

**Primary Name**: ${dataset_name}

**Aliases** (the paper may refer to this dataset using ANY of these names):
${dataset_aliases}

**Task**: ${task_description}

**Target Task Mode**: ${target_task_mode}

## Paper Content (Partial)

${pdf_content}

## CRITICAL Instructions

### Dataset Identification
1. Check if this paper uses the dataset by looking for ANY of the aliases above
2. If the paper does NOT mention any of these aliases, return an empty list `[]`
3. Be careful with papers that test on multiple datasets - ONLY extract results for THIS specific dataset

### What to Extract

For THIS dataset (${dataset_name}) ONLY, extract:

1. **All baseline methods** mentioned in comparisons or experiments
2. **Performance metrics**: Dice, IoU, Sensitivity, Specificity, Precision, Recall, F1, AUC, etc.
3. **Efficiency metrics** (if available):
   - Model Parameters (Params, in millions M or thousands K)
   - FLOPs (floating point operations, in G or M)
   - Inference Time (in ms or seconds)
   - GPU Memory (in MB or GB)
4. **Citation key**: Generate in format "FirstAuthorYear" (e.g., "Ronneberger2015")
5. **Venue**: The publication venue (Conference/Journal) and Year of the method's original paper (e.g., "MICCAI 2021", "CVPR 2020", "IEEE TMI 2022"). Check the References section if needed.

### Task Mode Filtering

Since this paper is classified as "${target_task_mode}", focus on extracting methods for this specific task mode. If the paper includes multiple task settings, only extract the methods relevant to "${target_task_mode}".

## Example Scenarios

### Example 1: Multiple Datasets

If the paper has:
```
Method      | TN3K (Dice) | DDTI (Dice) | BUSI (Dice)
------------|-------------|-------------|-------------
U-Net       | 0.847       | 0.792       | 0.831
DeepLabV3+  | 0.863       | 0.805       | 0.849
```

You should ONLY extract:
- U-Net: Dice=0.847 (from TN3K column)
- DeepLabV3+: Dice=0.863 (from TN3K column)

DO NOT extract DDTI or BUSI results!

### Example 2: With Efficiency Metrics

If the paper shows:
```
Method      | Dice  | IoU   | Params (M) | FLOPs (G) | Time (ms)
------------|-------|-------|------------|-----------|----------
U-Net       | 0.847 | 0.735 | 31.0       | 55.8      | 45
ResUNet     | 0.863 | 0.758 | 43.5       | 78.2      | 62
```

Extract:
- U-Net: {metrics: {Dice: 0.847, IoU: 0.735}, efficiency_metrics: {Params: "31.0M", FLOPs: "55.8G", Time: "45ms"}, venue: "MICCAI 2015"}
- ResUNet: {metrics: {Dice: 0.863, IoU: 0.758}, efficiency_metrics: {Params: "43.5M", FLOPs: "78.2G", Time: "62ms"}, venue: "IEEE TMI 2019"}

### Example 3: Method Not Using This Dataset

If the paper title is "Advanced Segmentation on ACDC Dataset" and only uses ACDC, return:
```json
[]
```

Because it doesn't use ${dataset_name}.

## Output Format

Return ONLY a valid JSON array (no markdown, no code blocks, no explanations).

Format:
```json
[
  {
    "method_name": "U-Net",
    "metrics": {
      "Dice": "0.847",
      "IoU": "0.735",
      "Sensitivity": "0.862"
    },
    "efficiency_metrics": {
      "Params": "31.0M",
      "FLOPs": "55.8G"
    },
    "citation_key": "Ronneberger2015",
    "venue": "MICCAI 2015"
  },
  {
    "method_name": "DeepLabV3+",
    "metrics": {
      "Dice": "0.863",
      "IoU": "0.758"
    },
    "citation_key": "Chen2018",
    "venue": "ECCV 2018"
  }
]
```

### Field Rules:
- `method_name`: Required
- `metrics`: Performance metrics dict. **Prefer "Dice" over "DSC"**.
- `efficiency_metrics`: Optional dict, only if paper provides these
- `citation_key`: Optional, format "FirstAuthorYear"
- `venue`: Optional but recommended, e.g. "CVPR 2020"

### If No Methods Found:
Return empty array: `[]`

## CRITICAL REMINDERS

1. ✅ Look for ALL aliases of ${dataset_name}
2. ✅ ONLY extract results for ${dataset_name}, not other datasets
3. ✅ Include efficiency metrics and VENUE info (from References)
4. ✅ Extract ALL methods (proposed + ALL baselines)
5. ❌ NO markdown code blocks
6. ❌ NO explanations or notes
7. ❌ NO mixing results from different datasets

Your response must be ONLY the JSON array, starting with `[` and ending with `]`.
