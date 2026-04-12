# OpenHands Code Generation Prompts

This directory contains prompt templates for OpenHands-based code generation in the AI Scientist agent.

## Directory Structure

```
openhands/
├── skills/          # OpenHands Skills (system-level, always injected)
├── stages/          # Stage-specific prompts (1-4)
├── modes/           # Operation mode prompts (draft, improve, debug)
├── components/      # Reusable prompt components
└── templates/       # Full prompt templates
```

## Usage

```python
from skills.prompt_loader import load_skill

# Load a component
dataset_config = load_skill(
    "code_generation/openhands/components/dataset_config.md",
    dataset_id=27,
    dataset_name="Pancreas",
    configuration="3d_fullres"
)

# Load a stage prompt
stage2_prompt = load_skill(
    "code_generation/openhands/stages/stage2_innovation.md",
    innovation_description="...",
    baseline_info="..."
)
```

## Variable Format

- Use `{variable_name}` for variable placeholders
- Use `{{variable_name}}` to avoid LaTeX conflicts
