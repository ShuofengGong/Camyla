# Extract Section from Experiment Report

You are given a complete experiment report in Markdown format and asked to extract a specific section.

## Input

**Target Section:** ${section_name}

**Full Report:**
```markdown
${full_report}
```

## Task

Extract ONLY the content belonging to the "${section_name}" section from the report above.

## Rules

1. Include the section header and all its content
2. Include any subsections that belong to this section
3. Stop when you reach the next major section (same heading level as the target)
4. Do NOT modify the content - extract it exactly as written
5. If the section is not found, respond with: "SECTION_NOT_FOUND"

## Output

Respond with ONLY the extracted section content, nothing else.
