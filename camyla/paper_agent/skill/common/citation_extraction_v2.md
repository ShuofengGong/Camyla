You are a citation management expert skilled at inferring specific literature citations based on academic paper context.

Task Description:
Below are citation placeholders and their context snippets from the paper. For each citation, you must **infer the intended paper purely from the surrounding sentences**, then generate a short search query.

**Critical**: The placeholder keyword (e.g., "unet", "attention_mechanism") is AI-generated and may be inaccurate or hallucinated. **Do NOT trust or rely on the keyword itself.** Instead, read the sentences around the placeholder carefully — look for method names, architectural details, or task descriptions — and deduce what paper the author actually intended to cite.

**Warning about author names**: Author names appearing in the context (e.g., "Zhang et al.", "Wang et al.") are also AI-generated and **very likely hallucinated**. **Do NOT include author surnames in search queries.** Only use method/model names and technical terms for searching.

**Important**: The context may contain multiple citation placeholders. **[TARGET]**...**[/TARGET]** markers wrap the citation currently being processed; ignore other citations.

Citation List (with context):
${citations_with_context}

Output Format Example:
```
Citations Needed:

1. Keyword: "unet"
   Context: "...medical image segmentation. Ronneberger et al. proposed U-Net **[TARGET]**[CITE:unet]**[/TARGET]** which utilizes skip connections..."
   Analysis: Context describes U-Net with skip connections for biomedical segmentation — the original U-Net paper. Ignore author name "Ronneberger" (may be hallucinated).
   Search Query: "U-Net biomedical image segmentation"
   
2. Keyword: "attention_mechanism"
   Context: "...previous work [CITE:unet] showed promise. The attention mechanism **[TARGET]**[CITE:attention_mechanism]**[/TARGET]** helps the model focus on salient regions in medical images..."
   Analysis: Context describes spatial attention for medical imaging — likely the Attention U-Net paper
   Search Query: "Attention U-Net medical image"

3. Keyword: "swin_transformer"
   Context: "...hierarchical representation. Swin Transformer **[TARGET]**\cite{CITE:swin_transformer}**[/TARGET]** introduces shifted windows for efficient self-attention..."
   Analysis: Context mentions "Swin Transformer" and "shifted windows" — the original Swin Transformer paper
   Search Query: "Swin Transformer shifted windows"
```

Search Query Rules (MUST follow):
1. **Maximum 6 words** — keep queries concise for better Semantic Scholar recall
2. **NO year numbers** — never include years like "2015", "2021" in the query
3. **NO author names** — never include surnames like "Ronneberger", "Chen", "Wang" in the query (they are likely hallucinated)
4. **Use ONLY: method/model name + technical terms (optional)** (e.g., "U-Net biomedical image segmentation", "nnU-Net")
5. **Avoid generic terms alone** like "deep learning" or "medical imaging" without a specific method name

Output Requirements:
1. **Read the surrounding sentences** to understand what paper is being cited — do NOT just reformulate the keyword
2. Look for clues: model/method names, architectural features, dataset names, unique terminology
3. If the context says "X et al. proposed Y", the search query should focus on Y (the method name), NOT X (the author name)
4. If context is vague with no identifiable paper, use the most specific technical term visible in context
5. Prioritize finding the original/seminal paper, not surveys or follow-up works

Notes:
- Analysis field should briefly explain the inference reasoning in English
- Search Query must be in English for Semantic Scholar search
- Each citation must strictly follow the above format
