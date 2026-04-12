# Skills System

## 📚 Overview

The Skills system stores all prompts as Markdown files, giving you a simple and intuitive way to manage them.

## 🚀 Quick Start

### Basic usage

```python
from skills.prompt_loader import load_skill

# Load a prompt
prompt = load_skill("agents/phd_student/literature_review.md")

# Load with variables
prompt = load_skill(
    "agents/phd_student/literature_review.md",
    reviewed_papers="paper1, paper2"
)
```

### List all skills

```python
from skills.prompt_loader import list_all_skills

# List every skill
all_skills = list_all_skills()

# List a specific category
agent_skills = list_all_skills("agents")
```

### Reload (hot reload during development)

```python
from skills.prompt_loader import reload_skill

# After editing a prompt file, reload it
prompt = reload_skill("agents/phd_student/literature_review.md")
```

## 📁 Directory Layout

```
skills/
├── README.md                          # This file
├── prompt_loader.py                   # Prompt loader
├── agents/                            # Agent-related prompts
│   ├── phd_student/                   # PhD student agent
│   │   ├── role_description.md
│   │   ├── literature_review.md
│   │   ├── plan_formulation.md
│   │   └── ...
│   ├── citation_network_agent.md
│   └── ...
├── code_generation/                   # Code-generation prompts
│   └── aider/
│       ├── stage1_base_generation.md
│       ├── stage2_creative_research.md
│       └── stage3_innovation_integration.md
├── paper_writing/                     # Paper-writing prompts
│   ├── style_guides/
│   │   └── academic_writing_style.md
│   └── sections/
│       ├── abstract.md
│       ├── introduction.md
│       └── ...
└── review/                            # Review prompts
    ├── reviewer_system_prompts.md
    └── review_template.md
```

## 🔧 Variable Formats

The Skills system supports two variable formats:

### 1. Python format string `{variable}`

```markdown
Your task is to analyze {dataset_name} using {method_name}.
```

Usage:
```python
prompt = load_skill(
    "path/to/skill.md",
    dataset_name="MNIST",
    method_name="CNN"
)
```

### 2. Double-brace format `{{variable}}`

Used to avoid conflicts with LaTeX syntax:

```markdown
Papers in your review: {{reviewed_papers}}
```

Usage:
```python
prompt = load_skill(
    "agents/phd_student/literature_review.md",
    reviewed_papers="paper1, paper2"
)
```

## ✍️ Writing a New Prompt

### 1. Create a Markdown file

Create a `.md` file in the appropriate category directory.

### 2. Write the content

```markdown
# Prompt Title

Briefly describe what this prompt is for.

## Main Content

The actual prompt content goes here...

You can use variables: {variable_name}

## Variable Reference

- `{variable_name}`: description of the variable
```

### 3. Use it from code

```python
from skills.prompt_loader import load_skill

prompt = load_skill("category/your_prompt.md", variable_name="value")
```

## 📝 Best Practices

1. **Clear title**: every prompt file should have a clear title.
2. **Variable documentation**: when using variables, document each variable's purpose.
3. **Keep it concise**: prompts should be concise and direct — avoid rambling.
4. **Include examples**: you can embed usage examples inside the prompt.
5. **Versioning notes**: for major edits, add a version note at the top of the file.

## 🔄 Migrating Existing Code

### Step 1: extract the prompt

Find prompt strings in the code:

```python
# Old code
def phase_prompt(self, phase):
    if phase == "literature review":
        return """
        Your goal is to perform literature review...
        """
```

### Step 2: create a markdown file

Save the prompt content under `skills/agents/phd_student/literature_review.md`.

### Step 3: update the code

```python
# New code
from skills.prompt_loader import load_skill

def phase_prompt(self, phase):
    if phase == "literature review":
        return load_skill("agents/phd_student/literature_review.md")
```

## 🎯 Benefits

1. ✅ **Simple and intuitive**: Markdown format is easy to edit.
2. ✅ **Version control**: track every change with Git.
3. ✅ **Team collaboration**: easy to review and discuss.
4. ✅ **No learning curve**: Markdown is a universal format.
5. ✅ **Flexibility**: supports rich text, code blocks, lists, etc.
6. ✅ **Easy maintenance**: no complex class hierarchies required.

## 📞 Getting Help

If you have questions, see:
- `SKILLS_DESIGN.md` — detailed design document
- `prompt_loader.py` — source code with inline comments

