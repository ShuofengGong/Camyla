You are consolidating multiple research challenges into {target_count} distinct, representative challenges.

## Context
Task Type: {task_type}
Modality: {modality}

## Collected Challenges
{all_challenges}

## Instructions
Consolidate the above challenges into exactly {target_count} distinct challenges:

1. **Cluster similar challenges** - Group challenges that address the same underlying problem
2. **Select representative ones** - Pick the most specific and actionable challenge from each cluster
3. **Ensure diversity** - The {target_count} final challenges should cover DIFFERENT aspects of the problem
4. **Focus on architecture** - Only keep challenges related to network architecture under supervised learning

## STRICTLY AVOID challenges related to:
- Data augmentation, preprocessing, input enhancement
- Unsupervised/self-supervised/semi-supervised learning
- Pre-training, transfer learning, domain adaptation
- Training techniques, optimization, loss functions
- Dataset construction, labeling strategies
- Multi-modal/cross-modality fusion IF the dataset is single-modality (check Modality field above)

## Output
Return exactly {target_count} challenges as a JSON array.

- The output MUST be a **valid JSON array** `[...]`.
- Each element MUST be a **JSON object** with exactly two string fields: `"name"` and `"description"`.
- Do NOT include any extra keys, comments, natural language explanation, or Markdown code fences in the output.

Example (structure only, values are placeholders):
[
  {"name": "2-5 words", "description": "50-80 words explaining the challenge"},
  {"name": "Another challenge", "description": "50-80 words explaining the challenge"}
]
