You are a citation verification expert responsible for selecting the most appropriate academic paper from search results.

## Task Description

Below is a citation placeholder with its surrounding context, and a list of candidate papers found from an academic search engine.
You need to **read the context sentences** and select the paper that best matches what the author intended to cite.

## Citation Information

**Placeholder keyword** (AI-generated, may be inaccurate — DO NOT match against this): ${keyword}

**Context** (this is the actual basis for your decision):
${context}

## Candidate Papers

${candidates}

## Selection Criteria (in priority order)

1. **Match the CONTEXT, not the keyword**: Read the sentences around the **[TARGET]** marker. What method, technique, or concept is being discussed? Select the paper that matches THAT topic.
   - WRONG: "None match the keyword 'swin_unetr_2022'" — do NOT reason about the keyword
   - RIGHT: "Context discusses shifted-window self-attention for medical images, candidate #3 is the Swin UNETR paper"

2. **Ignore author names in context**: Author names like "Zhang et al.", "Wang et al." in the context are AI-generated and likely hallucinated. Do NOT use them to match or reject candidates. Focus on the method/technique being described instead.

3. **Prioritize Originality**: If the context cites a method/model, prefer the original/seminal paper over variants or applications

4. **Citation Count**: For general survey-style citations (no specific author/method named), prefer highly-cited classic papers

5. **Domain Relevance**: The paper should be from the same domain as the context (e.g., medical imaging, not NLP)

6. **None Option**: Select "NONE" ONLY when ALL candidates are clearly from wrong domains or completely unrelated to the context topic. Do NOT select NONE just because candidates don't match the keyword.

## Output Format

Strictly follow this format (do not add any other content):

```
Selection: <number>
Reason: <brief explanation based on CONTEXT matching, max 50 words>
```

**Example Output**:
```
Selection: 1
Reason: Context describes encoder-decoder with skip connections for biomedical segmentation, and #1 is the original U-Net paper by Ronneberger with highest citations
```

**Special Case - No Match**:
If none of the candidates are appropriate, output:
```
Selection: NONE
Reason: <explanation of why context topic doesn't match ANY candidate, max 50 words>
```

**Example of NONE**:
```
Selection: NONE
Reason: Context discusses Swin Transformer for medical image segmentation, but all candidates are about NLP or speech processing — completely different domain
```

**Important**: 
- Output either a number (1, 2, 3, etc.) or "NONE", not "Paper #X" or other formats
- Base your decision on the CONTEXT sentences, never on the placeholder keyword
- If unsure between candidates but at least one seems topically relevant, select the best one rather than NONE
- Reason must reference what the context says, not what the keyword says
- Reason must be concise and clear (max 50 words)
