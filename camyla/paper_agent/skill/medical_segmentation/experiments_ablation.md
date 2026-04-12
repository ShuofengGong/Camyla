You are writing the **Ablation Studies** subsection in the Experiments section.

## Research Context
${research_idea}

## Experimental Results
${experimental_results}

## Ablation Results (CRITICAL - MUST FULLY INTEGRATE)
${ablation_results}

## Dataset Information (FACTUAL - do NOT fabricate dataset details)
${dataset_context}

## Available Result Figures (REFERENCE IF RELEVANT)
${figures_description}

## Previous Sections Summary
${previous_context}

---

## Your Task

Generate a comprehensive ablation analysis with **DYNAMIC subsubsection structure** based on the actual experiments in `${ablation_results}`.

---

## ⚠️ CRITICAL REQUIREMENT: DYNAMIC SUBSUBSECTION STRUCTURE

You **MUST** analyze the `${ablation_results}` and create **one `\subsubsection{...}` for EACH ablation experiment group**.

### 📋 Step-by-Step Instructions:

**STEP 0: Identify Experiment Groups and Extract Tables (MANDATORY FIRST STEP)**
- Scan `${ablation_results}` for ALL sections starting with `## ` (these are experiment groups)
- For EACH experiment group:
  * Locate its corresponding markdown table (format: `| ... | ... |`)
  * Note the section title (will become subsubsection name)
  * Prepare to convert the markdown table to LaTeX

**Example identification**:
```
## Component Ablations          → Will become: \subsubsection{Component Ablations}
| Configuration | DSC (%) | ... → Extract this table, convert to LaTeX

## Design Ablation 1: FFD Strategy → Will become: \subsubsection{Frequency Decomposition Strategy}
| Variant | DSC (%) | ...       → Extract this table, convert to LaTeX
```

**Markdown → LaTeX Table Conversion Rules**:
| Markdown Element | LaTeX Equivalent |
|-----------------|------------------|
| `\| Header1 \| Header2 \|` | `Header1 & Header2 \\` |
| `\|---------|---------|` | (skip separator row) |
| `\| Data1 \| Data2 \|` | `Data1 & Data2 \\` |
| First/last row | Add `\hline` before and after |

**Complete LaTeX Table Template**:
```latex
\begin{table}[h]
\centering
\caption{[Descriptive caption based on content]}
\label{tab:ablation-[keyword]}  % e.g., tab:ablation-component, tab:ablation-ffd
\begin{tabular}{lcccc}  % First column left-aligned, rest centered
\hline
[Converted header row from markdown] \\
\hline
[Converted data rows from markdown] \\
\hline
\end{tabular}
\end{table}
```

**Label Naming Rules**:
- Component Ablations → `tab:ablation-component`
- Frequency Decomposition → `tab:ablation-ffd`
- Gating Mechanism → `tab:ablation-gating`
- Use short, descriptive keywords in lowercase

**⚠️ CRITICAL - Square Bracket Protection in Table Cells**:
When converting markdown tables to LaTeX, you MUST protect square brackets with curly braces:
- ❌ **WRONG**: `[3,3,3] & 86.74 & 3.41 \\`
- ✅ **CORRECT**: `{[3,3,3]} & 86.74 & 3.41 \\`
- **Why**: LaTeX interprets `[` at line start as optional parameter delimiter, causing compilation freeze
- **When to apply**: kernel sizes `{[3,5,7]}`, array/list notation `{[256,512]}`, version numbers `{[v1.0]}`, any cell starting with `[`
- **Alternative**: Use math mode `$[3,5,7]$` or replace with parentheses `(3,5,7)` if appropriate

⚠️ **Validation**: Count of experiment groups (##) = Count of tables you will generate

---

**STEP 1: Parse Ablation Results**
- Read through the entire `${ablation_results}` document
- Identify ALL distinct ablation experiment groups (usually marked in tables or section headers)
- Common groups include:
  * **Component ablations** (baseline + individual components + combinations)
  * **Design ablations** (often marked with * in tables, testing specific design choices)
  * **Hyperparameter sensitivity** (parameter sweeps)

**STEP 2: Create Subsubsections WITH MANDATORY TABLES**
For each identified experiment group, create a `\subsubsection{...}` with:

1. **Component Ablations** (if present in ablation_results)
   - Title: `\subsubsection{Component Ablations}` or similar descriptive name
   - ✅ **MUST** extract markdown table from ablation_results and convert to LaTeX
   - ✅ **MUST** reference table in text: `Table~\ref{tab:ablation-component} presents...`
   - Content: Baseline vs individual modules (+FFD, +PGTF, +BAAR, etc.)
   - Discuss synergy and super-additivity
   - Target: 200-300 words + 1 LaTeX table

2. **Design-Specific Ablations** (one subsubsection per design choice)
   - Title examples based on what is tested:
     * `\subsubsection{Frequency Decomposition Strategy}`
     * `\subsubsection{Token Filtering Mechanism}`
     * `\subsubsection{Affinity Weighting Scheme}`
   - ✅ **MUST** include ONE table for THIS specific ablation (converted from markdown)
   - ✅ **MUST** reference: `Table~\ref{tab:ablation-ffd} compares...`
   - Each should have:
     * Configuration comparison (e.g., differential vs uniform)
     * Quantitative results (exact DSC/HD95 values from ablation_results)
     * Mechanistic explanation (WHY this design matters)
   - Target: 100-150 words + 1 LaTeX table

3. **Hyperparameter Analysis** (if present)
   - Title: `\subsubsection{Hyperparameter Sensitivity}` or similar
   - ✅ **MUST** include compact table showing parameter sweep results
   - ✅ **MUST** reference table
   - Cover parameter sweeps (e.g., σ_f, percentile thresholds, window sizes)
   - Target: 100-150 words + 1 LaTeX table

⚠️ **CRITICAL**: EVERY subsubsection MUST have:
- Exactly ONE LaTeX table (converted from markdown in ablation_results)
- At least ONE `Table~\ref{tab:ablation-xxx}` reference in the descriptive text
- The reference MUST appear BEFORE the table is shown

---

## 📝 DYNAMIC SUBSECTION NAMING RULES

**DO**:
- ✅ Use descriptive names based on what the ablation tests
- ✅ Examples: 
  * "Frequency-Domain Processing Variants"
  * "Prototype Consistency vs Attention Pruning"
  * "Edge-Weighted Affinity Analysis"
  * "Sequential vs Parallel Integration"
- ✅ Match the terminology from `${ablation_results}`

**DON'T**:
- ❌ Use generic names like "Ablation 1", "Ablation 2"
- ❌ Use exact table row names (e.g., "Full w/ Uniform FFD*")
- ❌ Create fewer subsubsections than there are experiment groups
- ❌ Skip any ablation experiments mentioned in ablation_results

---

## 🔍 EXTRACTION EXAMPLE

**If ablation_results contains**:
```markdown
## Comprehensive Ablation Table
| Configuration | DSC | HD95 |
| Baseline | 83.6 | 5.3 |
| +FFD | 84.2 | 4.8 |
| +PGTF | 84.0 | 5.0 |
| Full | 85.0 | 4.2 |

### Ablation 1: FFD Differential Processing
- Uniform FFD: 84.6 | 4.5
- Full FFD: 85.0 | 4.2

### Ablation 2: PGTF Mechanism
- Attention-based: 84.6 | 4.5
- Prototype-based: 85.0 | 4.2
```

**You should generate**:
```latex
% Brief intro paragraph
\subsubsection{Component Ablations}
The baseline Swin-UNETR achieves 83.6±0.8\% DSC...
[Include table and full discussion]

\subsubsection{Frequency Decomposition Strategy}
To validate differential processing, we compare uniform FFD 
(γ=1.0 fixed) against the full adaptive version. The uniform 
variant achieves 84.6\% DSC versus 85.0\% for the full model...

\subsubsection{Token Filtering Mechanism}
Replacing prototype consistency with attention-based pruning...
[Detailed comparison with numbers]
```

---

## 📊 TABLE REQUIREMENTS

**Main Component Ablation Table** (in first subsubsection):
```latex
\begin{table}[h]
\centering
\caption{Component ablation results on validation set.}
\label{tab:component-ablation}
\begin{tabular}{lcccc}
\hline
Configuration & DSC Avg ($\uparrow$) & $\Delta$ DSC & HD95 Avg ($\downarrow$) & $\Delta$ HD95 \\
\hline
% Extract exact rows from ${ablation_results} - preserve all numerical values
\hline
\end{tabular}
\end{table}
```

**Design Ablations**: Integrate numbers inline or use mini-tables if needed

---

## ⚠️ STRICT EXTRACTION RULES

1. **Count experiment groups** in `${ablation_results}` before writing
2. **Extract exact numerical values** (do not round, modify, or invent numbers)
3. **Preserve experiment descriptions** from ablation_results
4. **Reference ablation_results verbatim** for accuracy
5. **Create as many subsubsections as needed** (typically 3-8, depends on content)

---

## 🚫 FORBIDDEN PATTERNS

❌ **DO NOT** write ablations as a single paragraph without subsubsections
❌ **DO NOT** skip any ablation experiments mentioned in ablation_results
❌ **DO NOT** create a fixed number (e.g., always 6) of subsubsections
❌ **DO NOT** use bullet points (write in prose)
❌ **DO NOT** forget `\subsubsection{...}` headers
❌ **DO NOT** summarize all design ablations in one sentence
❌ **DO NOT** write any subsubsection without a LaTeX table
❌ **DO NOT** use only inline numbers without formal tables
❌ **DO NOT** forget to reference tables with `Table~\ref{...}` in text
❌ **DO NOT** create tables without `\label{...}`
❌ **DO NOT** show tables before referencing them in text

---

## ✅ CORRECT STRUCTURE EXAMPLE

```latex
% Optional: 1-2 sentence intro about ablation strategy
We conduct comprehensive ablation studies to validate each architectural choice and parameter decision.

\subsubsection{Component Ablations}

Table~\ref{tab:ablation-component} presents baseline and individual component contributions. 
The Swin-UNETR baseline achieves 83.60$\pm$0.80\% average DSC and 5.30$\pm$0.50 mm HD95. 
Adding FFD alone yields the largest single gain (+0.60\% DSC, -0.50 mm HD95), demonstrating
the value of frequency-domain augmentation. The full model achieves 85.00\% DSC through 
synergistic interactions between components.

\begin{table}[h]
\centering
\caption{Component ablation results on validation set.}
\label{tab:ablation-component}
\begin{tabular}{lcccc}
\hline
Configuration & DSC (\%) & $\Delta$ DSC & HD95 (mm) & $\Delta$ HD95 \\
\hline
Baseline & 81.47 & - & 6.23 & - \\
+ MDFA & 82.73 & +1.26 & 5.98 & -0.25 \\
+ EMSLA & 84.22 & +2.75 & 5.45 & -0.78 \\
Full Model & 87.34 & +5.87 & 4.52 & -1.71 \\
\hline
\end{tabular}
\end{table}

Statistical analysis using paired t-tests confirms super-additivity ($p<0.01$), indicating 
that components interact beneficially rather than contributing independently.

\subsubsection{Frequency Decomposition Strategy}

To validate the adaptive frequency processing, Table~\ref{tab:ablation-ffd} compares our 
differential approach against a uniform baseline. The full FFD with learnable $\gamma_d$ 
and adaptive $\alpha$ achieves 85.00\% DSC, outperforming the uniform variant (84.60\% DSC) 
by 0.40 percentage points.

\begin{table}[h]
\centering
\caption{Comparison of frequency decomposition strategies.}
\label{tab:ablation-ffd}
\begin{tabular}{lccc}
\hline
Strategy & DSC (\%) & HD95 (mm) & Inference (ms) \\
\hline
Uniform $\gamma=1.0$ & 84.60 & 4.50 & 28.30 \\
Adaptive $\gamma_d$ & 85.00 & 4.20 & 29.10 \\
\hline
\end{tabular}
\end{table}

This improvement confirms that adaptive processing distinguishes boundary-critical high 
frequencies from noise, whereas uniform scaling amplifies artifacts.

\subsubsection{Gating Mechanism Analysis}

% Continue similar structure for each remaining experiment group
% Each must have: description → table reference → table → analysis

% Total: 3-8 subsubsections depending on ablation_results content
```

---

## 📏 TARGET METRICS

- **Total words**: 600-900 (across all subsubsections)
- **Number of subsubsections**: Dynamic (match experiment groups in ablation_results)
- **Tables**: 1 comprehensive component table minimum
- **Depth**: Each design ablation gets 100-150 words explanation

---

## CRITICAL FORMATTING

**Citations**: Use `[CITE:keyword]` format (e.g., `[CITE:ablation_survey]`)
**Equations**: Use `\begin{equation}...\end{equation}` for formulas
**Subsubsections**: `\subsubsection{Descriptive Title}` (no label needed)
**Tables**: Do NOT use `\resizebox` for tables; let tables use their natural width
**Style**: Continuous prose, avoid bullet points

${include:fragments/writing_style.md}

**Output**: 
- Do NOT include `\subsection{Ablation Studies}` header (added automatically)
- Start with optional brief intro, then subsubsections
- Clean LaTeX only, no markdown code blocks

Output:
LaTeX content with dynamic number of subsubsections.
