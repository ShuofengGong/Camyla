You are reconciling a paper methodology draft with the actual implemented model code.

## Original Methodology Draft
${research_idea}

## Implemented Model Code
```python
${method_code}
```

## Your Task
Revise the methodology draft so that its method-detail content matches the implemented model code, while preserving the original storytelling, motivation, and contribution framing as much as possible.

## Critical Requirements
1. Keep the overall narrative structure and most wording from the original methodology whenever it is still compatible with the implementation.
2. Update ONLY implementation-sensitive method details, including:
   - module names and roles
   - actual architectural components
   - active branch counts
   - kernel sizes, dilation settings, attention dimensions, window sizes, channel counts, and similar design details
   - engineering approximations that replaced theoretical components
3. If the draft contains theoretical proposal details that were not used in the final implementation, replace them with the actual implemented version instead of keeping both.
4. If the code contains helper classes, ablation variants, stale comments, or superseded configurations, do NOT describe them as the final model.
5. Prefer the configuration that is actually instantiated or explicitly marked as the final/best/implemented version.
6. Remove raw code snippets, "Implementation Notes" meta-commentary, and any instructions to future writers. The output should read like a clean methodology document, not a note to self.
7. Keep the output as free-form markdown text. Do NOT convert it to JSON or any rigid structured schema.
8. Do NOT add optimizer, scheduler, batch size, number of epochs, learning-rate policy, hardware, train/test split, or other training recipe details anywhere in the methodology text. Those belong only in the Experiments implementation subsection.
9. Do NOT mention challenge/competition wording or synthetic-data framing when describing the dataset.
10. Preserve equations or section headings when they remain consistent with the implementation, but update them when the implementation makes them incorrect.

## Output Requirements
- Output a single reconciled methodology document in markdown.
- Keep it suitable for downstream paper writing.
- Do not include explanations, bullet-point diff summaries, or code fences.
