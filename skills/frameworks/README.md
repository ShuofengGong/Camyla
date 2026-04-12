# Frameworks

This directory contains documentation and code templates for the deep-learning frameworks supported by the project.

## Available Frameworks

- **camylanet**: Medical image segmentation framework based on nnUNet

## Usage

```python
from skills import FrameworkLoader

# Create a loader instance
framework_loader = FrameworkLoader()

# List available frameworks
frameworks = framework_loader.get_available_frameworks()
print(frameworks)  # ['camylanet']

# Load framework documentation
doc_path, source = framework_loader.find_documentation("camylanet")
print(f"Documentation: {doc_path}")

# Load the code template
code_path, source = framework_loader.find_code_template("camylanet")
print(f"Template: {code_path}")
```

## File Layout

```
frameworks/
├── __init__.py
├── loader.py           # FrameworkLoader class
├── README.md           # This file
└── camylanet/
    ├── documentation.md  # Framework documentation
    └── template.py       # Code template
```

## Adding a New Framework

1. Create a new framework directory under `frameworks/`.
2. Add `documentation.md` (framework description).
3. Add `template.py` (base code template).
4. FrameworkLoader will automatically discover the new framework.
