Your goal is to analyze the full text of a research paper and perform citation network analysis. You need to complete THREE tasks in a single response:

🎯 TASK 1: Extract ALL citations from the paper
- Find all referenced papers in the bibliography/references section
- Extract paper titles, authors, publication years, and venues

🎯 TASK 2: Filter high-quality recent citations
- Keep only papers from 2023 or later
- Focus on good quality conferences and journals (not limited to top-tier venues)
- Prioritize papers about practical application-oriented architecture innovations
- Avoid overly famous or mainstream works, look for emerging and specialized research
- Include domain-specific conferences and journals relevant to the application area

🎯 TASK 3: Generate SPECIFIC search keywords for technical innovations
- Extract 5 DISTINCT technical terms from THIS PAPER'S citations in RECENT top-tier papers
- Use standard academic terminology. Avoid invented jargon or overly "marketing-style" names.
- ⚠️ DO NOT copy examples directly - extract real terms from the paper
- AVOID outdated but common methods: 'Shifted Windows', 'Cross-Attention', 'Standard Vision Transformer', 'Dilated Convolution'
- DO NOT use generic words: 'architecture', 'network', 'design', 'model', 'framework'
- DO NOT include year numbers (2023, 2024, 2025)
- Focus on specific architectural mechanisms (e.g., 'Linear Attention', 'State Space Models', 'Sparse Mixture of Experts')

📋 OUTPUT REQUIREMENT:
You MUST output a valid JSON object with the following structure:
{
  "all_citations": [
    {
      "title": "Paper Title",
      "authors": "Author names",
      "year": "2024",
      "venue": "Conference/Journal name"
    }
  ],
  "filtered_citations": [
    {
      "title": "Application-Focused Architecture Paper",
      "authors": "Author names",
      "year": "2023",
      "venue": "Domain-Specific Conference",
      "relevance_reason": "Why this paper is relevant to practical architecture innovations"
    }
  ],
  "search_keywords": [
    "keyword1",
    "keyword2",
    "keyword3",
    "keyword4",
    "keyword5"
  ]
}

⚠️ CRITICAL: Output ONLY valid JSON. No additional text, explanations, or markdown formatting.

