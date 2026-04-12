You are extracting research challenges from a paper's Introduction/Related Work section.

TASK CONTEXT:
- Task: {task_type} (medical image segmentation)
- Modality: {modality}

{% if '--- Paper' in paper_content %}
> **MULTIPLE ABSTRACTS MODE**: You are analyzing abstracts from multiple papers.

> 🚨 **CRITICAL STEP - RELEVANCE FILTERING**:
> Before extracting challenges, you MUST ignore any abstract that does not match the **Task** ({task_type}) or **Modality** ({modality}).
> - If an abstract is about a different task (e.g., classification vs segmentation), IGNORE IT.
> - If an abstract is generic deep learning without medical application, IGNORE IT.

> Synthesize challenges ONLY from the remaining **RELEVANT** abstracts.
> **FALLBACK**: If NO abstracts strictly match the criteria, select the 3-5 most similar/relevant abstracts available and extract challenges from them.
{% endif %}

PAPER CONTENT:
{paper_content}

Extract up to {max_challenges} data-specific challenges that this paper aims to solve.

STRICTLY AVOID these topics (DO NOT extract):
- Data augmentation, input enhancement, preprocessing techniques
- Unsupervised learning, self-supervised learning, semi-supervised learning
- Pre-training strategies, transfer learning, domain adaptation
- Training techniques, optimization methods, loss functions
- Simple attention mechanisms (unless very novel architecture)
- dataset construction, labeling strategies

ONLY extract challenges related to NETWORK ARCHITECTURE under SUPERVISED LEARNING.

Output JSON:
[
  {"name": "2-5 words", "description": "50-80 words", "source": "Introduction/Related Work"}
]
