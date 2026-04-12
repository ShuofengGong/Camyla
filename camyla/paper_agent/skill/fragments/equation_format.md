**Mathematical Equations (CRITICAL - STRICTLY REQUIRED)**:

**MANDATORY: Use `\begin{equation}...\end{equation}` for**:
- ✓ Loss functions and objective formulations
- ✓ Model predictions or key outputs  
- ✓ Important algorithmic steps
- ✓ Complex expressions (>3 terms, fractions, summations)

**FORBIDDEN - DO NOT USE**:
- ✗ Markdown style equations: `$$...$$` 
- ✗ LaTeX displaymath: `\[...\]`
- ✗ Double dollar signs in any form

**Correct Format Pattern**:
```latex
Leading description in text:
\begin{equation}
    \mathcal{L} = \mathcal{L}_{\text{seg}} + \lambda \mathcal{L}_{\text{reg}}
\end{equation}
where $\mathcal{L}_{\text{seg}}$ is the segmentation loss, $\mathcal{L}_{\text{reg}}$ is regularization, and $\lambda$ controls the trade-off.
```

**For equation labeling**:
```latex
\begin{equation}
\label{eq:total_loss}
    \mathcal{L} = \mathcal{L}_{\text{seg}} + \lambda \mathcal{L}_{\text{reg}}
\end{equation}
```
Reference as: "As shown in Eq.~\eqref{eq:total_loss}, ..."

**Use inline math `$...$` ONLY for**:
- ✓ Simple variable definitions: $x \in \mathbb{R}^n$
- ✓ Brief mathematical expressions in flowing text
- ✗ Complex formulas - use equation environment instead

**Examples**:
```latex
The output prediction is computed as:
\begin{equation}
    y = \sigma(f_\theta(x))
\end{equation}
where $f_\theta$ is the network parameterized by $\theta$, and $\sigma$ is the sigmoid activation.

For the Dice loss, we use:
\begin{equation}
    \mathcal{L}_{\text{Dice}} = 1 - \frac{2\sum_{i} p_i g_i + \epsilon}{\sum_{i} p_i + \sum_{i} g_i + \epsilon}
\end{equation}
```
