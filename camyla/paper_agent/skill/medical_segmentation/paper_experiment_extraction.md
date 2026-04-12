You are tasked with extracting experiment information from a research paper and formatting it as a Markdown section that will be appended to a baseline results file.

## Input Information

- **Dataset Name**: ${dataset_name}
- **Dataset Aliases**: ${dataset_aliases}
- **Target Task Mode**: ${task_mode}
- **Target Task Type**: ${target_task_type}
- **Paper PDF Content**: ${pdf_content}
- **PDF Filename**: ${pdf_filename}

## Your Task

Generate a complete Markdown section for this paper following the template below. Your output should be ready to append directly to the `baseline_results/${task_mode}.md` file.

## CRITICAL: Pre-Extraction Verification

**Before extracting anything, complete these verification steps:**

### Step 1: Dataset Verification
1. Search for ANY of the dataset aliases in the paper
2. **IMPORTANT**: If this is a **dataset introduction paper** (a paper introducing/presenting the dataset itself), **DO NOT SKIP IT**. Dataset papers typically provide baseline experiments and should be extracted.
3. If the paper does NOT mention this dataset AND is NOT a dataset introduction paper, output ONLY: `[SKIP: Dataset not used in this paper]`

**How to identify dataset introduction papers:**
- Paper title contains the dataset name or related keywords
- Paper describes dataset collection, annotation, or characteristics
- Paper presents baseline experiments on a newly introduced dataset
- Paper is from a journal focused on datasets (e.g., "Scientific Data", "Data in Brief")
- Examples: "SegRap2023: A benchmark of organs-at-risk and gross tumor volume Segmentation...", "A dataset of primary nasopharyngeal carcinoma MRI..."

### Step 2: Task Type Verification (Multi-Task Filtering)
Some datasets support multiple tasks (e.g., segmentation AND classification/diagnosis). We ONLY need results for: **${target_task_type}**

1. If the paper reports results for MULTIPLE tasks on this dataset, ONLY extract the **${target_task_type}** results
2. SKIP tables that are ONLY for other tasks (e.g., skip classification/diagnosis tables if target is segmentation)
3. If the paper does NOT report ${target_task_type} results for this dataset, output: `[SKIP: No ${target_task_type} results for this dataset]`

### Step 3: Table Type Identification (Main Results vs Ablation)
Papers often contain multiple result tables. You must identify and extract ONLY the **MAIN SOTA comparison table**.

**EXTRACT this table type:**
- ✅ **Main SOTA Comparison Table**: Compares the proposed method against diverse prior methods (e.g., U-Net, TransUNet, DeepLabV3+, etc.)
- ✅ Identified by: Multiple different baseline methods from different papers/authors

**DO NOT extract these table types:**
- ❌ **Ablation Study Table**: Compares variants of the SAME proposed method (e.g., "Ours w/o Module A", "Ours + Module B", "Baseline", "Full Model")
- ❌ **Cross-Validation Fold Results**: Shows results per fold rather than overall comparison
- ❌ **Hyperparameter Sensitivity Table**: Shows effect of different hyperparameter values

### Step 4: Multi-Dataset Handling
If the paper uses MULTIPLE datasets:
1. ONLY extract the table/results for **${dataset_name}**
2. If there's a combined table with multiple datasets, extract ONLY the rows/columns for ${dataset_name}
3. DO NOT mix results from different datasets

## Output Template

```markdown
## Paper X: [Paper Title or First Author Name]

**Source:** `baseline/${pdf_filename}`  
**Publication:** [Venue Year]  
**Citation Key:** [FirstAuthorYear]

### Experiment Setting

[Write 2-4 natural paragraphs describing the experimental setup. Include:
- How the data was split (train/val/test, percentages or sample counts)
- What portion of the dataset was used (full dataset or subset)
- Any special protocols:
  * For domain adaptation: source domain, target domain, and how they're used
  * For semi-supervised: how labeled and unlabeled data are used
  * For few-shot: number of shots per class
  * For weakly-supervised: type of weak labels
- Data augmentation techniques if mentioned
- Any other relevant experimental choices (optimizer, learning rate schedule, training epochs, etc.)

IMPORTANT: Write in natural, flowing paragraphs as you would find in a paper's methods section. 
DO NOT use bullet points or structured lists.]

### Results

[Create a Markdown table with ALL methods from the paper's comparison table:
- Include ALL baseline methods mentioned in the paper
- Include the proposed method and mark it with **bold** (e.g., **Proposed**, **MethodName**)
- Include ALL metrics reported (Dice, IoU, HD95, Params, FLOPs, etc.)
- Keep the original table structure from the paper
- Use the EXACT metric values from the paper (with appropriate decimal places)

Example table format:
| Method       | Dice (%) | IoU (%) | HD95 (mm) | Params (M) |
|--------------|----------|---------|-----------|------------|
| U-Net        | 78.45    | 65.32   | 12.45     | 31.0       |
| TransUNet    | 81.23    | 68.91   | 10.32     | 105.1      |
| **Proposed** | 84.67    | 72.58   | 8.91      | 45.3       |
]

### Key Observations
- [List 2-4 key findings from this paper]
- [Focus on: main performance improvements, what works well, limitations]
- [Mention if results may not be directly comparable to other papers due to different experimental settings]

---
```

## Critical Instructions

1. **Output ONLY the Markdown section** - no JSON, no code blocks wrapping the markdown, no explanations before or after
2. **Start directly with `## Paper X:`** - do not add any preamble
3. **Use natural paragraphs** for Experiment Setting - write like a methods section, not a form
4. **Extract the COMPLETE results table** - don't summarize, include all methods and metrics
5. **Bold the proposed method** in the table (use ** ** around the method name)
6. **End with `---`** separator line (three hyphens)
7. **If dataset not used**: Output only `[SKIP: Dataset not used in this paper]`

## Tips for Finding Information

- **Experiment Setting**: Usually in "Methods", "Experimental Setup", or "Implementation Details" sections
- **Data Split**: Look for phrases like "70-10-20 split", "5-fold cross-validation", "train/val/test"
- **Results Table**: Usually in "Results", "Experiments", or "Quantitative Evaluation" sections
- **Citation Key**: From the first author's last name + year in references (e.g., "Ronneberger et al. 2015" → "Ronneberger2015")
- **Venue**: Check the paper header, footer, or references section

## Example Output

## Paper 1: Multi-scale Attention Network for Thyroid Segmentation

**Source:** `baseline/wang2023_attention_fs.pdf`  
**Publication:** IEEE TMI 2023  
**Citation Key:** Wang2023

### Experiment Setting

The experiments were conducted on the complete TN3K dataset consisting of 3000 ultrasound images. The dataset was randomly partitioned into training (70%, 2100 images), validation (10%, 300 images), and testing (20%, 600 images) subsets following the standard protocol. All images were resized to 256×256 pixels and normalized to zero mean and unit variance.

During training, extensive data augmentation was applied to improve model robustness, including random horizontal flipping (50% probability), rotation within ±15 degrees, elastic deformation, and color jittering (brightness and contrast adjustment). The models were trained for 200 epochs using the Adam optimizer with an initial learning rate of 1e-4, which was reduced by a factor of 0.1 when validation loss plateaued for 10 consecutive epochs. A batch size of 16 was used across all experiments.

### Results

| Method         | Dice (%) | IoU (%) | HD95 (mm) | Params (M) | FLOPs (G) |
|----------------|----------|---------|-----------|------------|-----------|
| U-Net          | 78.45    | 65.32   | 12.45     | 31.0       | 55.8      |
| FCN            | 76.23    | 63.11   | 14.23     | 18.6       | 42.3      |
| DeepLabV3+     | 79.67    | 66.82   | 11.34     | 59.3       | 87.2      |
| TransUNet      | 81.23    | 68.91   | 10.32     | 105.1      | 125.4     |
| Swin-Unet      | 80.67    | 67.82   | 11.01     | 41.4       | 68.9      |
| **MS-AttNet**  | 84.67    | 72.58   | 8.91      | 45.3       | 71.2      |

### Key Observations
- MS-AttNet achieves +3.44% Dice improvement over the best baseline (TransUNet)
- Significant reduction in boundary error (HD95: 8.91 vs 10.32 mm) demonstrates better edge localization
- The method maintains reasonable computational efficiency (45.3M parameters) compared to transformer-based methods
- Performance gains are particularly pronounced on small nodules (<10mm diameter) as discussed in ablation studies

---
