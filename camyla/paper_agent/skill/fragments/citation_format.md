**Citation Format (CRITICAL - DO NOT VIOLATE THIS RULE)**:

- You MUST use the placeholder format `[CITE:keyword]` for ALL citations.
- **NEVER** use standard LaTeX citation formats like `\cite{Author2023}` or `~\cite{...}`.
- The system will automatically convert placeholders to proper BibTeX citations later.

**Correct Examples**:
- ✓ "U-Net [CITE:unet] is widely used for medical image segmentation."
- ✓ "Recent work [CITE:swin_transformer] shows promising results."
- ✓ "We employ ResNet-50 [CITE:resnet] as the backbone architecture."

**Incorrect Examples** (NEVER USE):
- ✗ "U-Net~\cite{Ronneberger2015}" 
- ✗ "\cite{Author2023}"
- ✗ "recent work \cite{CITE:unet}"

**Keyword Guidelines**:
- Use short, descriptive keywords that include author/year/topic information
- Examples: `[CITE:ronneberger_unet_2015]`, `[CITE:attention_mech_medical_2019]`, `[CITE:domain_generalization_mri]`
- Avoid overly generic keywords like `[CITE:deep_learning]`
- Include enough context for automatic citation resolution via semantic search

**Citation Frequency Guidelines**:
- **Introduction Section**: 10-15 citations total
- **Related Work Section**: 3-5 citations per paragraph minimum
- **Method Section**: Cite when referencing specific techniques, architectures, or loss functions
- **Experiments Section**: Cite datasets and baseline methods
