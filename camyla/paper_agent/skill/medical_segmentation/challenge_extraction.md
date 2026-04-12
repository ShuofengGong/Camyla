You are tasked with extracting research challenges and limitations from a medical image segmentation paper.

${custom_instructions}

## Input Information

- **Dataset Name**: ${dataset_name}
- **Task Mode**: ${task_mode}
- **Paper PDF Content**: ${pdf_content}
- **PDF Filename**: ${pdf_filename}

## Your Task

Extract and summarize the research challenges mentioned in this paper. Output a Markdown section that can be directly appended to `challenges/${task_mode}.md`.

## Where to Find Challenges

Look for challenges in these sections:
- **Introduction**: Limitations of existing methods, unsolved problems
- **Related Work**: Technical challenges and method deficiencies in the field
- **Discussion/Conclusion**: Remaining challenges, future directions
- **Motivation**: Why the authors propose a new method (usually implies existing problems)

## Challenge Categories

When extracting, try to categorize challenges:
- **Data-related**: Annotation noise, data scarcity, class imbalance, data quality
- **Model-related**: Generalization ability, computational cost, interpretability, model complexity
- **Task-related**: Boundary ambiguity, small objects, multi-scale, complex morphology
- **Evaluation-related**: Metric limitations, clinical applicability, robustness

## Output Template

```markdown
## Paper X: [Paper Title or First Author Name]

**Source:** `baseline/${pdf_filename}`  
**Publication:** [Venue Year]  
**Citation Key:** [FirstAuthorYear]

### Identified Challenges

1. **[Challenge Name]**
   - [Detailed description of what this challenge is]
   - [Why this is a problem / What is the impact]
   - [Possible solution approaches (if mentioned in the paper)]

2. **[Another Challenge]**
   - [Description]
   - [Impact]
   - [Possible solutions]

3. **[Third Challenge]**
   - ...

### Proposed Solutions in This Paper
- [How this paper attempts to address the above challenges]
- [Key technical innovations]
- [Achieved results]

---
```

## Critical Instructions

1. **Focus on CHALLENGES, not contributions** - We want problems and limitations, not achievements
2. **Write challenge names descriptively** - Make them easy to understand
3. **Be SPECIFIC** - Avoid generalities, extract concrete technical problems
4. **Extract 3-5 challenges per paper** - Not too many, not too few
5. **Include context** - Explain why it's a challenge and what the impact is
6. **Output pure Markdown** - Do not wrap in code blocks
7. **End with `---`** separator line
8. **If the paper doesn't discuss challenges clearly**: Extract implied challenges from the motivation section

## Example Output

## Paper 1: Boundary-aware Progressive Attention UNet for Thyroid Segmentation

**Source:** `baseline/BPAT-UNet_fs.pdf`  
**Publication:** CMPB 2023  
**Citation Key:** Bi2023

### Identified Challenges

1. **Boundary Ambiguity**
   - Thyroid nodule boundaries are often unclear in ultrasound images and have low contrast with surrounding tissues
   - This causes automatic segmentation algorithms to produce large errors in boundary regions, hurting boundary-sensitive metrics such as HD95
   - The model's boundary awareness must be enhanced, e.g. through boundary supervision or edge attention

2. **Multi-scale Object Segmentation**
   - Thyroid nodules vary substantially in size, with diameters ranging from 5 mm to 50 mm
   - Convolutional networks with fixed receptive fields struggle to simultaneously capture the global semantics of large targets and the local details of small ones
   - Multi-scale feature-fusion strategies such as FPN, ASPP, or pyramid pooling are required

3. **Sensitivity to Annotation Noise**
   - Medical image annotations depend on expert experience, and different experts may disagree on the same case
   - Noisy labels mislead model training and yield unstable performance
   - Uncertainty modeling or noise-robust loss functions can be considered

4. **Limited Computational Resources**
   - Deployment environments in hospitals typically have limited compute
   - Large Transformer models are hard to run in real time on edge devices
   - Lightweight designs such as knowledge distillation, pruning, or efficient architectures are needed

### Proposed Solutions in This Paper
- Propose a Bilateral Path Attention module that jointly attends to boundary and region features through a two-path mechanism
- Apply Progressive Feature Aggregation for progressive multi-scale feature fusion
- Use Focal Loss to address the class-imbalance issue caused by small objects and hard samples

---
