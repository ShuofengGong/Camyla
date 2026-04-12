# Research Proposal Duplicate Check Prompt

Determine if a NEW research proposal is essentially THE SAME as any existing proposals.

## New Proposal
**Title**: {new_title}
**Core Insight**: {new_insight}
**Modules**: {new_modules}

## Existing Proposals
{existing_proposals}

---

## Your Task

Analyze whether the NEW proposal is **duplicate** with any existing proposal.

### Definition of Duplicate:
A proposal is considered duplicate if:
1. **Same Core Insight**: The fundamental technical idea is the same
2. **Same Key Modules**: Uses essentially the same module types/techniques
3. **Same Problem Framing**: Addresses the problem in the same way

### NOT Duplicate If:
- Uses the same core theme but with DIFFERENT modules
- Addresses the same challenge but with DIFFERENT insights
- Has similar module names but DIFFERENT technical approaches
- Shares ONE module but other modules are substantially different

---

## Output Format

Respond with ONLY a valid JSON object:

```json
{{
    "is_duplicate": true/false,
    "reason": "Brief explanation (1-2 sentences)",
    "most_similar_to": "Title of most similar existing proposal (or null if not duplicate)"
}}
```

Do not include any text before or after the JSON object.
