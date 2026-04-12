## LaTeX Formatting Rules

Ensure you are always writing good compilable LaTeX code. Common mistakes that should be fixed include:

### Syntax Requirements
- LaTeX syntax errors (unenclosed math, unmatched braces, etc.)
- Duplicate figure labels or references
- Unescaped special characters: & % $ # _ { } ~ ^ \
- Proper table/figure closure
- Do not hallucinate new citations or any results not in the logs

### Code Formatting
- Use proper LaTeX syntax (no markdown)
- Use `\cite{}` for citations
- For tables, use proper LaTeX tabular environment
- For figures:
  - Use complete figure environment with proper captions
  - Use simple filenames without paths (e.g., 'figure1.png' instead of 'path/to/figure1.png')
  - Always include `\label{}` for cross-referencing
- No placeholders or TODOs

### Text Emphasis
- Use `\textbf{}` for bold text (never use **, * or other markdown syntax)
- Use `\textit{}` for italics
- Use `\emph{}` for emphasis within italicized text

### LaTeX-specific Formatting
- Use proper LaTeX quotation marks ('') instead of straight quotes
- Avoid non-LaTeX punctuation marks
- Use proper LaTeX dashes (-- for en-dash, --- for em-dash)
- Use `\dots` for ellipsis instead of ...

### Subsections
- Use proper LaTeX `\subsection{}` syntax
- Each subsection must contain at least 2 well-developed paragraphs
- For Related Work section, limit to maximum 3 subsections
- Ensure smooth transitions between subsections

### Citations
- All citations MUST reference actual papers from the provided bibtex file
- Do not make up or reference non-existent papers

When returning final code, place it in fenced triple backticks with 'latex' syntax highlighting.

