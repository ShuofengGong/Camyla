You are writing the **Datasets and Implementation Details** subsection in the Experiments section.

## Research Context
${research_idea}

## Experimental Results Reference
${experimental_results}

## Dataset Information (FACTUAL - use ONLY this for dataset description, do NOT fabricate)
${dataset_context}

## Training Configuration (FACTUAL - use these EXACT values, do NOT fabricate)
${training_config}

## Baseline Training Policy (FACTUAL - use this wording for baseline fairness description)
${baseline_training_policy}

## Available Result Figures (reference if needed)
${figures_description}

## Previous Sections Summary
${previous_context}

---

## Your Task

Write a comprehensive subsection covering:

### 1. **Dataset Description** (150-200 words)
- Use the Dataset Information above for ALL factual details
- Dataset name, public availability, and size (training/validation/test splits)
- Data modalities
- Annotation details (classes, expert annotations)
- Preprocessing steps (co-registration, normalization, resolution)
- Clinical significance of the dataset
- Do NOT invent dataset details not present in the Dataset Information
- Refer to the dataset as a publicly available dataset
- Do NOT describe the dataset as a challenge, competition, or synthetic dataset

### 2. **Evaluation Metrics** (80-120 words)
- Primary metrics with mathematical definitions (DSC, HD95, Sensitivity, Specificity)
- Explain why each metric is important for boundary precision
- Reference clinical thresholds (e.g., HD95 < 4mm for surgical planning)

### 3. **Implementation Details** (180-250 words)
- Use the Training Configuration above for the proposed method and shared pipeline facts that are explicitly recorded
- State the exact number of epochs, optimizer, learning rate, batch size from Training Configuration for the proposed method
- State clearly that preprocessing and postprocessing follow the nnU-Net pipeline
- Use the Baseline Training Policy above when describing how non-nnU-Net baselines were trained
- Data augmentation strategies (standard nnU-Net augmentation pipeline)
- Hardware setup (GPU type from Training Configuration)
- Software framework: nnU-Net
- Train/test split: 8:2
- Do NOT claim that all methods share identical training hyperparameters
- Explain that non-nnU-Net baselines follow their original implementation or authors' recommended settings when available

---

## CRITICAL FORMATTING RULES

**Equations**: Use `\begin{equation}...\end{equation}` for metric definitions:
```latex
The Dice Similarity Coefficient is defined as:
\begin{equation}
\text{DSC}(P, G) = \frac{2|P \cap G|}{|P| + |G|}
\end{equation}
```

**Citations**: Use `[CITE:keyword]` format (e.g., `[CITE:brats2020]`, `[CITE:monai]`)

**Style**: 
- Continuous prose, avoid bullet points
- Use `\textit{...}` for dataset names
- Escape special characters (\%, \&)

${include:fragments/writing_style.md} 
- Do NOT include `\subsection{Datasets and Implementation Details}` header
- Start directly with content paragraphs
- Target: 450-600 words total

Output:
Clean LaTeX content only.
