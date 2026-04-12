You are writing the **State-of-the-Art Comparisons** subsection in the Experiments section.

## Research Context
${research_idea}

## Experimental Results
${experimental_results}

## Dataset Information (FACTUAL - do NOT fabricate dataset details)
${dataset_context}

## Available Result Figures (MUST USE)
${figures_description}

## Previous Sections Summary
${previous_context}

---

---

## ⚠️ CRITICAL: Multi-Dataset Handling (READ THIS FIRST)

**STEP 0: Analyze Dataset Count in ${experimental_results}**

Before writing anything, you MUST:
1. Count how many `## Dataset N:` sections exist in `${experimental_results}`
2. Extract the name of each dataset from these sections
3. Plan to create ONE subsection and ONE table for EACH dataset

**If ${experimental_results} contains MULTIPLE datasets:**

You MUST create SEPARATE `\subsubsection{}` for EACH dataset with its own complete SOTA comparison table.

**Required Structure for Multi-Dataset (Example with 3 datasets):**
```latex
% Brief intro paragraph (1-2 sentences)

\subsubsection{Performance on xxx}
[Introductory text about xxx results]

\begin{table}[h]
\centering
\caption{State-of-the-art comparison on xxx dataset.}
\label{tab:sota_xxx}
\begin{tabular}
...
\end{tabular}
\end{table}

[Analysis text for xxx]

\subsubsection{Performance on yyy}
[Introductory text about yyy results]

\begin{table}[h]
\centering
\caption{State-of-the-art comparison on yyy dataset.}
\label{tab:sota_yyy}
\begin{tabular}
...
\end{tabular}
\end{table}

[Analysis text for yyy]

\subsubsection{Performance on zzz}
[Introductory text about zzz results]

\begin{table}[h]
\centering
\caption{State-of-the-art comparison on zzz dataset.}
\label{tab:sota_zzz}
\begin{tabular}
...
\end{tabular}
\end{table}

[Analysis text for zzz]
```

**VALIDATION REQUIREMENTS:**
- ✅ Count of `\subsubsection{Performance on ...}` = Number of datasets in experimental_results
- ✅ Each dataset from experimental_results has its own table
- ✅ Each table caption mentions the specific dataset name
- ✅ NO dataset is omitted
- ✅ Each table uses values from the corresponding dataset section in experimental_results

**If ${experimental_results} contains SINGLE dataset:**

Use the standard single-table format with caption: "State-of-the-art comparison on [DatasetName] dataset."

---

## ⚠️ CRITICAL: Efficiency subsection is conditional

Only include a computational-efficiency paragraph if `${experimental_results}` explicitly contains a
`## Computational Efficiency` section.

If that section is absent:
- Do NOT mention computational efficiency
- Do NOT speculate about parameters, FLOPs, memory, or inference time
- Do NOT add an efficiency trade-off discussion just because it is common in papers

If that section is present:
- Use ONLY the values provided there
- Keep the discussion brief and objective
- If the proposed method is not the most efficient, describe the trade-off factually without overstating it

---

## Your Task

Write a comprehensive comparison subsection covering:

### 1. **Main Results Table** (100-150 words + table)
- Compare against 6-8 SOTA baselines (nnU-Net, UNETR, nnFormer, Swin-UNETR, U-Mamba etc.)
- Include per-class metrics (ET/TC/WT or task-specific) and averages
- Do NOT use `\resizebox` for tables; let tables use their natural width
- **Table Formatting (CRITICAL)**:
  - Highlight your method's **numerical values only** in bold using `\textbf{}`
  - Example: `\textbf{89.2} & \textbf{85.8}` (numbers bold)
  - **DO NOT** bold the method name in the row label
  - Example CORRECT: `ESSA (Ours) & \textbf{89.2} & \textbf{85.8} \\`
  - Example WRONG: `\textbf{ESSA (Ours)} & 89.2 & 85.8 \\` ❌
- Reference exact values from `${experimental_results}`

### 2. **Result Visualization Figures** (MANDATORY - MUST INSERT)
- **CRITICAL**: Extract ALL figures from `${figures_description}` where:
  * `type` == "result"
  * `placement` mentions "Comparisons" or "Experiments"
- **Insert complete LaTeX figure code** from the `latex_code` field
- **Reference figures** in your qualitative analysis using `Figure~\ref{fig:xxx}`
- **Minimum requirement**: Insert at least 1 figure, recommended 2-3 figures
- **Placement**: Insert figures AFTER discussing the main results table, BEFORE or WITHIN qualitative analysis
- **Example reference**: "As evident in Figure~\ref{fig:result_fig1}, our method outperforms..."

### 3. **Qualitative Analysis** (150-200 words + figure references)
- Discuss where your method excels (e.g., specific tumor regions, boundary precision)
- Explain performance gaps vs baselines (mechanism-based analysis)
- Clinical implications (e.g., HD95 reduction enables surgical precision)
- Mention statistical significance if available (p-values, confidence intervals)

### 3. **Generalization** (100-150 words)
- Results on additional datasets if available (LiTS, ACDC, etc.)
- Cross-modality or cross-domain robustness
- Adaptation mechanisms (e.g., how frequency cutoff adapts)
- Performance consistency across different data distributions

### 4. **Computational Efficiency** (100-150 words, only when provided)
- Training time, inference speed (fps or ms/volume)
- Memory usage (GPU RAM requirements)
- Parameter count and model size overhead
- Trade-off analysis (accuracy improvement vs computational cost)

---

## FORMATTING REQUIREMENTS

**Table Example**:
```latex
\begin{table}[t]
\centering
\caption{State-of-the-art comparison on BraTS 2020 validation set.}
\label{tab:sota}
\begin{tabular}{l|ccc|c|ccc|c}
\hline
Method & DSC ET & DSC TC & DSC WT & DSC Avg & HD95 ET & HD95 TC & HD95 WT & HD95 Avg \\
\hline
nnU-Net [CITE:nnunet] & 76.8 & 83.7 & 88.7 & 83.1 & 5.2 & 6.8 & 4.9 & 5.6 \\
UNETR [CITE:unetr] & 77.5 & 84.2 & 89.1 & 83.6 & 4.9 & 6.5 & 4.6 & 5.3 \\
Swin-UNETR [CITE:swin_unetr] & 78.1 & 84.5 & 89.3 & 84.0 & 4.7 & 6.2 & 4.4 & 5.1 \\
HBAR (Ours) & \textbf{79.2} & \textbf{85.8} & \textbf{90.1} & \textbf{85.0} & \textbf{3.8} & \textbf{5.0} & \textbf{3.8} & \textbf{4.2} \\
\hline
\end{tabular}
\end{table}
```

**Citations**: Use `[CITE:keyword]` format (e.g., `[CITE:nnunet]`, `[CITE:unetr]`)

**Equations**: Use `\begin{equation}...\end{equation}` if defining metrics or formulas

**Style**: 
- Continuous prose, avoid bullet points
- **Bold Formatting Rules (CRITICAL)**:
  - Use `\textbf{...}` ONLY for numerical values in tables (to highlight best results)
  - NEVER use `\textbf{...}` in paragraph text for method names, terms, or any other text
  - Example CORRECT in table: `ESSA & \textbf{89.2} \\` 
  - Example WRONG in table: `\textbf{ESSA (Ours)} & 89.2 \\` ❌
  - Example WRONG in text: "The proposed \textbf{ESSA} achieves..." ❌
- Escape special characters (\%, \&)
- **CRITICAL**: Protect square brackets in table cells with curly braces `{[...]}` to avoid compilation freeze
  - Example CORRECT: `{[3,5,7]} & 87.23 \\`
  - Example WRONG: `[3,5,7] & 87.23 \\` (causes LaTeX freeze)
- Be objective about limitations or close competitors

${include:fragments/writing_style.md}

**Output**: 
- Do NOT include `\subsection{State-of-the-Art Comparisons}` header
- Start with intro paragraph leading to main results table
- Target: 450-650 words + 1-2 tables

---

## CRITICAL REQUIREMENTS

1. **Extract exact numbers** from `${experimental_results}` - do not invent or modify values
2. **Include at least 4 baseline methods** for meaningful comparison
3. **Explain WHY your method performs better** (mechanism-based, not just numbers)
4. **Be honest about close competitors** or cases where performance is similar
5. **Reference clinical relevance** (e.g., "HD95 below 4mm threshold for surgical planning")

---

## 🚫 FORBIDDEN PATTERNS

❌ **DO NOT** claim superiority without evidence from experimental_results
❌ **DO NOT** use vague comparisons ("significantly better" without quantification)
❌ **DO NOT** fabricate efficiency analysis when `${experimental_results}` does not provide it
❌ **DO NOT** use bullet points (write in prose)

---

Output:
Clean LaTeX content only, starting with intro paragraph.
