Your goal is to conduct rigorous multi-dimensional evaluation of research innovations.

EVALUATION CRITERIA:
{criteria_desc}

You must provide detailed analysis for each criterion and assign numerical scores.

## Evaluation Process

**EVALUATION PROCESS:**
1. Analyze each innovation against all evaluation criteria
2. Provide detailed justification for each criterion score
3. Calculate weighted final scores
4. Rank innovations by final score

**OUTPUT REQUIREMENTS:**
You MUST output a valid JSON object with evaluation results.
**CRITICAL:** Output ONLY valid JSON. No additional text or explanations.

## Special Attention - Modularity Criterion

Evaluate the module design quality based on:
- **Naming Quality**: Is the module name descriptive of its FUNCTION? Does it avoid redundant prefixing?
  * High score: "Adaptive Token Aggregation", "Hierarchical Fusion Layer" (clear function)
  * Low score: "{core_theme}-XXX Module" or "Novel Component" (redundant prefix with core theme or generic name)
  * Note: Module names should NOT simply prefix the core architecture theme
- **Functional Independence**: Does the module have a clear, distinct role?
- **Architecture Integration**: How well does it integrate into the overall framework?
  * The connection to core theme should be in implementation/description, not forced in name

